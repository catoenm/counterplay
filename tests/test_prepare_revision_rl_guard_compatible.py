"""CPU-only administrative first-write tests; no provider or budget calls."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import uuid

from scripts import prepare_revision_rl_guard_compatible as helper
from scripts.cloud.cardinal_guard import Guard, GuardError


class GuardCompatiblePreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.prior=self.root/'prior';self.prior.mkdir()
        self.target=self.root/'next';self.clock=1800000000.25
        self.intent=uuid.UUID('12345678-1234-1234-1234-123456789abc')
        self.guard=self.root/'scripts/cloud/cardinal_guard.py';self.guard.parent.mkdir(parents=True)
        shutil.copyfile(helper.ROOT/'scripts/cloud/cardinal_guard.py',self.guard)
        original=json.loads((helper.ROOT/'runs/cardinal-revision-rl-deployment-v1/plan.json').read_text())
        original.update(deadline_unix=self.clock+25*3600,intent='abcdefababcdefababcdefababcdefab')
        self.write('plan.json',original)
        self.write('state.json',{'plan_sha256':helper.base.digest(self.prior/'plan.json'),
            'roles':{'guard':{'state':'terminated_confirmed','pod_id':'old-guard'}}})
        self.write('guard-startup-failure.json',{'kind':'definite_guard_startup_configuration_failure',
            'guard_404_confirmed':True,'gpu_created':False,'training_wrapper_launched':False})
        self.write('terminated-liability-review.json',{'passed':True,'intent':original['intent'],
            'cleanup':{'fresh_exact_guard_http_status':404,'matching_current_intent_pods':0}})
        self.key=self.root/'project.pub';self.key.write_text('ssh-ed25519 fixture-only-public-key test\n')
        self.patches=[patch.object(helper,'ROOT',self.root),patch.object(helper.time,'time',return_value=self.clock),
            patch.object(helper.uuid,'uuid4',return_value=self.intent)]
        for p in self.patches:p.start();self.addCleanup(p.stop)

    def write(self,name,value):
        (self.prior/name).write_text(json.dumps(value,sort_keys=True)+'\n')

    def hashes(self):
        return {str(p.relative_to(self.root)):helper.base.digest(p) for p in self.root.rglob('*') if p.is_file()}

    def test_actual_guard_accepts_first_written_plan_and_old_evidence_is_unchanged(self):
        before=self.hashes();writes=[];original=helper.base.atomic_json
        def recording(path,value):
            writes.append((Path(path),deepcopy(value)));return original(path,value)
        with patch.object(helper.base,'atomic_json',side_effect=recording):
            result=helper.prepare(self.target,self.prior,self.key)
        plan=json.loads((self.target/'plan.json').read_text());state=json.loads((self.target/'state.json').read_text())
        self.assertEqual(plan['deadline_unix'],float(int(self.clock+23*3600+55*60)))
        self.assertEqual(plan['maximum_hours'],25);self.assertEqual(plan['reservation_usd'],145)
        self.assertLess(plan['deadline_unix'],json.loads((self.prior/'plan.json').read_text())['deadline_unix'])
        self.assertEqual([p.name for p,_ in writes],['plan.json','state.json'])
        self.assertEqual(writes[0][1],plan) # Never write a rejected 25h deadline then repair it.
        self.assertEqual(plan['intent'],self.intent.hex)
        self.assertEqual(state['roles'],{});self.assertFalse(state['reservation_applied'])
        self.assertEqual(state['plan_sha256'],helper.base.digest(self.target/'plan.json'))
        self.assertEqual(plan['guard_source_sha256'],helper.base.digest(self.guard))
        self.assertTrue(result['actual_guard_constructor_passed'])
        helper.controller.validate(plan)
        guard=Guard(None,plan['intent'],'fixture-guard',plan['deadline_unix'],'fixture-token-of-at-least-32-characters',
            plan['guard_source_sha256'],clock=lambda:self.clock)
        self.assertEqual(guard.deadline,plan['deadline_unix'])
        for p,digest in before.items():self.assertEqual(helper.base.digest(self.root/p),digest)
        secret=Path(plan['guard_secret_path'])
        self.assertEqual(secret.stat().st_mode & 0o777,0o600)
        self.assertEqual(secret.parent.stat().st_mode & 0o777,0o700)
        with self.assertRaises(FileExistsError):helper.prepare(self.target,self.prior,self.key)
        self.assertEqual(json.loads((self.target/'plan.json').read_text()),plan)

    def test_validation_failures_make_no_files_and_preserve_prior(self):
        originals={p.name:json.loads(p.read_text()) for p in self.prior.iterdir()}
        cases=[('state.json',lambda x:x['roles'].update(gpu={'state':'create_unknown'})),
            ('state.json',lambda x:x['roles']['guard'].update(state='identified')),
            ('state.json',lambda x:x.update(plan_sha256='0'*64)),
            ('guard-startup-failure.json',lambda x:x.update(gpu_created=True)),
            ('guard-startup-failure.json',lambda x:x.update(guard_404_confirmed=False)),
            ('terminated-liability-review.json',lambda x:x.update(passed=False)),
            ('terminated-liability-review.json',lambda x:x.update(intent='other')),
            ('terminated-liability-review.json',lambda x:x['cleanup'].update(matching_current_intent_pods=1))]
        for name,mutate in cases:
            with self.subTest(name=name):
                for key,value in originals.items():self.write(key,value)
                value=deepcopy(originals[name]);mutate(value);self.write(name,value);before=self.hashes()
                with self.assertRaises(ValueError):helper.prepare(self.target,self.prior,self.key)
                self.assertEqual(self.hashes(),before);self.assertFalse(self.target.exists())
                self.assertFalse((self.root/'.secrets').exists())

    def test_expired_or_non_earlier_deadline_and_bad_key_fail_before_write(self):
        for deadline in (self.clock-1,float(int(self.clock+100))):
            plan=json.loads((self.prior/'plan.json').read_text());plan['deadline_unix']=deadline;self.write('plan.json',plan)
            state=json.loads((self.prior/'state.json').read_text());state['plan_sha256']=helper.base.digest(self.prior/'plan.json');self.write('state.json',state)
            before=self.hashes()
            with self.assertRaisesRegex(ValueError,'earlier future'):helper.prepare(self.target,self.prior,self.key)
            self.assertEqual(self.hashes(),before)
        self.key.write_text('ssh-rsa unsupported')
        before=self.hashes()
        with self.assertRaisesRegex(ValueError,'Ed25519'):helper.prepare(self.target,self.prior,self.key)
        self.assertEqual(self.hashes(),before)

    def test_actual_constructor_rejection_occurs_before_any_mutation(self):
        before=self.hashes()
        with patch.object(helper,'Guard',side_effect=GuardError('fixture constructor refusal')):
            with self.assertRaises(GuardError):helper.prepare(self.target,self.prior,self.key)
        self.assertEqual(self.hashes(),before);self.assertFalse(self.target.exists())
        self.assertFalse((self.root/'.secrets').exists())
        with self.assertRaisesRegex(GuardError,'invalid_deadline'):
            Guard(None,self.intent.hex,'fixture-guard',self.clock+25*3600,'fixture-token-of-at-least-32-characters',
                helper.base.digest(self.guard),clock=lambda:self.clock)


if __name__=='__main__':unittest.main()
