"""Two-lease correction fixtures: no provider requests or real preparation."""
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts import runpod_arena_mixed_scalar as controller
from scripts import runpod_cardinal as base
from tests.test_runpod_cardinal import FakeProvider


class ScalarDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);(self.root/'runs').mkdir();(self.root/'scripts/cloud').mkdir(parents=True)
        shutil.copy2(controller.ROOT/'scripts/cloud/cardinal_guard.py',self.root/'scripts/cloud/cardinal_guard.py')
        for module in (controller,base):
            p=patch.object(module,'ROOT',self.root);p.start();self.addCleanup(p.stop)
        self.key=self.root/'ssh.pub';self.key.write_text('ssh-ed25519 fixture-public-only')
        self.previous=self.root/'failed';self.previous.mkdir()
        self.old_eval=self.root/'unused-evaluation';self.coding=self.root/'active-coding';self.coding.mkdir()
        self.parent_run=self.root/'failed-run';self.parent_run.mkdir()
        base.atomic_json(self.parent_run/'manifest.json',{'fixture':'failed original'})
        base.atomic_json(self.parent_run/'status.json',{'state':'failed'})
        base.atomic_json(self.previous/'plan.json',{'intent':'a'*32,'phase':'mixed_training',
            'evaluation_intent':'b'*32,'evaluation_deployment':str(self.old_eval),
            'coding_evaluation_intent':'c'*32,'coding_evaluation_deployment':str(self.coding)})
        self.terminal(self.previous,'failed',self.parent_run,exit_code=1)
        self.audit=self.parent_run/'final-audit.json'
        base.atomic_json(self.audit,{'kind':'mixed_arena_independent_audit_v1','integrity_passed':True,
            'state':'failed','phase':'training','complete':False,
            'manifest_sha256':base.digest(self.parent_run/'manifest.json'),
            'status_sha256':base.digest(self.parent_run/'status.json'),
            'optimizer_outcome_unknown':True})
        self.review=self.previous/'independent-release.json'
        base.atomic_json(self.review,{'kind':'independent_scoped_proposed_accounting_review',
            'scope':{'training_intent':'a'*32,'unused_mixed_evaluation_intent':'b'*32},
            'proposal':{'combined_release_usd':169,'training_pending_liability_usd':6,
                'unused_mixed_evaluation_release_usd':50},'budget_and_ledgers_unchanged':True})
        self.budget={'approved_pilot_cap':350,'cloud_spend_to_date':0,'cloud_reserved_usd':145,
            'cloud_reservations':{'old':{'amount_usd':74,'state':'terminated_pending_billing'},
                'a'*32:{'amount_usd':6,'state':'terminated_pending_billing','original_reservation_usd':125,
                    'released_unused_reservation_usd':119},
                'b'*32:{'amount_usd':0,'state':'released_unused','original_reservation_usd':50,
                    'released_unused_reservation_usd':50},
                'c'*32:{'amount_usd':65,'state':'reserved','deployment':str(self.coding)}}}
        self.accounting=self.previous/'applied-release.json';self.write_budget()
        base.atomic_json(self.root/'runs/api-ledger.json',{'requests':{'unknown':{
            'state':'interrupted_unknown','reserved_usd':.576147678}}})
        self.folder=self.root/'corrected-training';self.evaluation=self.root/'corrected-evaluation'
        self.prepared=self.prepare();self.provider=FakeProvider()
        self.deployment=controller.Deployment(self.folder,self.provider)

    def write_budget(self):
        base.atomic_json(self.root/'budget.json',self.budget)
        base.atomic_json(self.accounting,{'independent_review':str(self.review),
            'independent_review_sha256':base.digest(self.review),
            'new_budget_sha256':base.digest(self.root/'budget.json')})

    def prepare(self,folder=None,evaluation=None):
        return controller.prepare(folder or self.folder,'mixed_training',self.key,self.previous,
            evaluation or self.evaluation,predecessor_audit=self.audit,accounting_review=self.accounting)

    def terminal(self,folder,state,run=None,exit_code=0):
        plan=base.load(folder/'plan.json');run=run or self.parent_run;manifest=base.digest(run/'manifest.json')
        base.atomic_json(folder/'state.json',{'plan_sha256':base.digest(folder/'plan.json'),
            'creation_closed':True,'roles':{r:{'pod_id':r,'known_ids':[r],'matching_ids':[],
            'state':'terminated_confirmed','confirmed_404_ids':[r]} for r in ('gpu','guard')}})
        base.atomic_json(folder/'supervisor-origin.json',{'manifest_sha256':manifest,
            'run':str(run),'deployment':{'intent':plan['intent']}})
        base.atomic_json(folder/'supervisor-status.json',{'scientific_artifacts_verified':True,
            'gpu_404_confirmed':True,'guard_404_confirmed':True,'pull_returncode':0,
            'expected_manifest_sha256':manifest,'study_state':state,
            'remote_exit':{'exit_code':exit_code,'verification_completed':True,'manifest_sha256':manifest}})

    def test_real_guard_deadline_first_write_and_no_prepare_mutations(self):
        self.assertTrue(self.prepared['actual_guard_constructor_passed'])
        self.assertEqual(self.provider.calls,[]);self.assertFalse(self.deployment.state['reservation_applied'])
        self.assertEqual((self.deployment.plan['maximum_hours'],self.deployment.plan['reservation_usd']),(20.,125.))
        self.assertFalse(self.evaluation.exists());self.assertTrue(self.coding.exists())
        self.assertEqual(base.load(self.root/'budget.json'),self.budget)
        original=(self.folder/'plan.json').read_bytes()
        with self.assertRaises(FileExistsError):self.prepare()
        self.assertEqual(original,(self.folder/'plan.json').read_bytes())

    def test_only_two_phase_175_funding_preserves_existing_closure(self):
        closure=deepcopy(self.budget['cloud_reservations']['c'*32])
        self.deployment.reserve();self.deployment.reserve()
        budget=base.load(self.root/'budget.json')
        self.assertEqual(budget['cloud_reserved_usd'],320)
        self.assertAlmostEqual(controller.budget_total(budget),320.576147678)
        self.assertEqual(budget['cloud_reservations']['c'*32],closure)
        new=set(budget['cloud_reservations'])-set(self.budget['cloud_reservations'])
        self.assertEqual(new,set(self.deployment.expected_pair()));self.assertEqual(len(new),2)
        for key in new:
            row=budget['cloud_reservations'][key]
            self.assertEqual(row['funded_together_usd'],175)
            self.assertNotIn('deadline_unix',row);self.assertNotIn('pod_id',row)
        self.assertEqual(self.provider.calls,[])

    def test_failed_audit_and_applied_release_required_before_writes(self):
        audit=base.load(self.audit);audit['integrity_passed']=False;base.atomic_json(self.audit,audit)
        with self.assertRaisesRegex(ValueError,'audit'):self.prepare(self.root/'bad',self.root/'e2')
        self.assertFalse((self.root/'bad').exists())
        audit['integrity_passed']=True;base.atomic_json(self.audit,audit)
        self.budget['cloud_reservations']['a'*32]['amount_usd']=125;self.write_budget()
        with self.assertRaisesRegex(ValueError,'release'):self.prepare(self.root/'bad',self.root/'e2')
        self.assertFalse((self.root/'bad').exists())

    def test_unknown_temporary_optimizer_is_retained_not_reused(self):
        self.assertTrue(base.load(self.audit)['optimizer_outcome_unknown'])
        self.assertIn(str(self.audit.resolve()),self.deployment.plan['predecessor_evidence_sha256'])
        self.assertEqual(self.deployment.state['roles'],{})

    def test_stale_accounting_or_prepared_old_eval_rejected(self):
        self.budget['note']='changed';base.atomic_json(self.root/'budget.json',self.budget)
        with self.assertRaisesRegex(ValueError,'receipt'):self.prepare(self.root/'bad',self.root/'e2')
        self.write_budget();self.old_eval.mkdir()
        with self.assertRaisesRegex(ValueError,'unprepared'):self.prepare(self.root/'bad',self.root/'e2')

    def test_no_early_evaluation_and_unknown_deletion(self):
        with self.assertRaises((ValueError,FileNotFoundError)):
            controller.prepare(self.evaluation,'mixed_evaluation',self.key,self.folder)
        state=base.load(self.previous/'state.json');state['roles']['gpu']['known_ids'].append('unknown')
        base.atomic_json(self.previous/'state.json',state)
        with self.assertRaisesRegex(ValueError,'deletion'):self.prepare(self.root/'bad',self.root/'e2')

    def test_future_eval_fresh_clock_original_funding_and_context_only(self):
        self.deployment.reserve();self.terminal(self.folder,'training_complete')
        result=controller.prepare(self.evaluation,'mixed_evaluation',self.key,self.folder)
        evaluation=controller.Deployment(self.evaluation,self.provider);evaluation.reserve()
        self.assertEqual((result['reservation_usd'],evaluation.plan['maximum_hours']),(50.,8.))
        self.assertEqual(evaluation.plan['intent'],self.deployment.plan['evaluation_intent'])
        self.assertEqual(evaluation.plan['coding_evaluation_intent'],'c'*32)
        self.assertEqual(base.load(self.root/'budget.json')['cloud_reserved_usd'],320)
        self.assertNotIn('coding_evaluation',controller.PHASES)
        with self.assertRaisesRegex(ValueError,'phase'):
            controller.prepare(self.root/'cannot','coding_evaluation',self.key,self.folder)

    def test_guard_failure_precedes_any_new_plan_secret(self):
        old=set((self.root/'.secrets/cardinal-gpu').iterdir())
        with patch.object(controller,'Guard',side_effect=ValueError('guard bound')):
            with self.assertRaisesRegex(ValueError,'guard bound'):self.prepare(self.root/'bad',self.root/'e2')
        self.assertFalse((self.root/'bad').exists());self.assertEqual(set((self.root/'.secrets/cardinal-gpu').iterdir()),old)

    def test_one_post_unknown_create_and_guard_requirement(self):
        self.deployment.reserve();self.provider.lose_create_response=True
        with self.assertRaisesRegex(RuntimeError,'read-only'):self.deployment.create('guard')
        with self.assertRaisesRegex(ValueError,'already attempted'):self.deployment.create('guard')
        self.assertEqual(self.provider.calls.count(('POST','/pods')),1)
        with patch.object(self.deployment,'guard_health',return_value={'ready':False}):
            with self.assertRaisesRegex(ValueError,'armed'):self.deployment.create('gpu')
        self.assertEqual(self.provider.calls.count(('POST','/pods')),1)

    def test_cap_unknown_calls_partial_funding_fail_closed(self):
        for cap in (200,320):
            self.budget['approved_pilot_cap']=cap;self.write_budget()
            with self.assertRaisesRegex(ValueError,'cap'):self.deployment.reserve()
        self.budget['approved_pilot_cap']=350;self.write_budget();self.deployment.reserve()
        b=base.load(self.root/'budget.json');b['cloud_reservations'].pop(self.deployment.plan['evaluation_intent']);b['cloud_reserved_usd']-=50
        base.atomic_json(self.root/'budget.json',b)
        with self.assertRaisesRegex(ValueError,'Incomplete'):self.deployment.reserve()

    def test_350_remains_hard_cap_even_if_budget_field_is_larger(self):
        self.budget['approved_pilot_cap']=1000
        self.budget['cloud_reservations']['other']={'amount_usd':30,'state':'reserved'}
        self.budget['cloud_reserved_usd']+=30;self.write_budget()
        before=(self.root/'budget.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'cap'):self.deployment.reserve()
        self.assertEqual((self.root/'budget.json').read_bytes(),before)
        self.assertEqual(self.provider.calls,[])

    def test_deadline_boolean_nan_and_context_id_tamper(self):
        for key,value in [('deadline_unix',True),('estimate_usd',float('nan')),
                ('maximum_gpu_rate_usd_hour',True),('created_at','2026-10-23T08:00:00'),
                ('coding_evaluation_intent',self.deployment.plan['intent']),('coding_evaluation_context_only',False)]:
            plan=deepcopy(self.deployment.plan);plan[key]=value
            with self.assertRaises(ValueError):controller.validate(plan)


if __name__=='__main__':unittest.main()
