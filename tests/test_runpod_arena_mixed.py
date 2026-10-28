"""All leases are fake; even guard admission uses its real CPU constructor only."""
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts import runpod_arena_mixed as controller
from scripts import runpod_cardinal as base
from tests.test_runpod_cardinal import FakeProvider


class MixedDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);(self.root/'runs').mkdir();(self.root/'scripts/cloud').mkdir(parents=True)
        shutil.copy2(controller.ROOT/'scripts/cloud/cardinal_guard.py',self.root/'scripts/cloud/cardinal_guard.py')
        for module in (controller,base):
            p=patch.object(module,'ROOT',self.root);p.start();self.addCleanup(p.stop)
        self.key=self.root/'ssh.pub';self.key.write_text('ssh-ed25519 fixture-public-only')
        self.previous=self.root/'previous';self.previous.mkdir()
        base.atomic_json(self.previous/'plan.json',{'intent':'a'*32})
        self.terminal(self.previous,'throughput_failed')
        self.folder=self.root/'training';self.evaluation=self.root/'evaluation';self.coding=self.root/'coding-evaluation'
        self.prepared=controller.prepare(self.folder,'mixed_training',self.key,self.previous,self.evaluation,self.coding)
        self.provider=FakeProvider();self.deployment=controller.Deployment(self.folder,self.provider)
        self.budget(350,74)

    def terminal(self,folder,state):
        plan=base.load(folder/'plan.json');manifest='b'*64
        base.atomic_json(folder/'state.json',{'plan_sha256':base.digest(folder/'plan.json'),
            'creation_closed':True,'roles':{r:{'pod_id':r,'known_ids':[r],'matching_ids':[],
            'state':'terminated_confirmed','confirmed_404_ids':[r]} for r in ('gpu','guard')}})
        base.atomic_json(folder/'supervisor-origin.json',{'manifest_sha256':manifest,'deployment':{'intent':plan['intent']}})
        base.atomic_json(folder/'supervisor-status.json',{'scientific_artifacts_verified':True,
            'gpu_404_confirmed':True,'guard_404_confirmed':True,'pull_returncode':0,
            'expected_manifest_sha256':manifest,'study_state':state,
            'remote_exit':{'exit_code':0,'verification_completed':True,'manifest_sha256':manifest}})

    def budget(self,cap,held):
        base.atomic_json(self.root/'budget.json',{'approved_pilot_cap':cap,'cloud_spend_to_date':0,
            'cloud_reserved_usd':held,'cloud_reservations':{'old':{'amount_usd':held,'state':'terminated_pending_billing'}}})

    def test_unpaid_first_write_deadline_and_real_guard_admission(self):
        self.assertTrue(self.prepared['actual_guard_constructor_passed'])
        self.assertEqual(self.provider.calls,[]);self.assertFalse(self.deployment.state['reservation_applied'])
        self.assertEqual((self.deployment.plan['maximum_hours'],self.deployment.plan['reservation_usd']),(20.,125.))
        self.assertFalse(self.evaluation.exists());self.assertFalse(self.coding.exists())
        self.assertEqual(base.load(self.root/'budget.json')['cloud_reserved_usd'],74)
        original=(self.folder/'plan.json').read_bytes()
        with self.assertRaises(FileExistsError):controller.prepare(self.folder,'mixed_training',self.key,self.previous,self.evaluation,self.coding)
        self.assertEqual(original,(self.folder/'plan.json').read_bytes())

    def test_atomic_upfront_three_phase_funding_idempotent_no_future_deadlines(self):
        base.atomic_json(self.root/'runs/api-ledger.json',{'requests':{'unknown':{'state':'interrupted_unknown','reserved_usd':.6}}})
        self.deployment.reserve();self.deployment.reserve()
        budget=base.load(self.root/'budget.json');self.assertEqual(budget['cloud_reserved_usd'],314)
        self.assertAlmostEqual(controller.budget_total(budget),314.6)
        records=budget['cloud_reservations'];self.assertEqual(len(records),4)
        for phase,key in controller.INTENT_KEYS.items():
            row=records[self.deployment.plan[key]]
            self.assertEqual(row['amount_usd'],controller.PHASES[phase]['reservation'])
            self.assertNotIn('deadline_unix',row);self.assertNotIn('pod_id',row)
        self.assertEqual(self.provider.calls,[])

    def test_lower_cap_hard350_cap_unknown_call_and_inconsistent_aggregate(self):
        for cap,held in ((200,0),(1000,111),(350,110)):
            self.budget(cap,held)
            base.atomic_json(self.root/'runs/api-ledger.json',{'requests':{'u':{'state':'interrupted_unknown','reserved_usd':.6}}})
            before=(self.root/'budget.json').read_bytes()
            with self.assertRaisesRegex(ValueError,'cap'):self.deployment.reserve()
            self.assertEqual(before,(self.root/'budget.json').read_bytes())
        self.budget(350,74);budget=base.load(self.root/'budget.json');budget['cloud_reserved_usd']=12
        base.atomic_json(self.root/'budget.json',budget)
        with self.assertRaisesRegex(ValueError,'liability'):self.deployment.reserve()

    def test_future_phase_has_fresh_deadline_consumes_original_funding_once(self):
        self.deployment.reserve();self.terminal(self.folder,'training_complete')
        before=base.load(self.root/'budget.json')['cloud_reserved_usd']
        result=controller.prepare(self.evaluation,'mixed_evaluation',self.key,self.folder)
        evaluation=controller.Deployment(self.evaluation,self.provider);evaluation.reserve();evaluation.reserve()
        self.assertEqual(evaluation.plan['intent'],self.deployment.plan['evaluation_intent'])
        self.assertEqual((evaluation.plan['maximum_hours'],result['reservation_usd']),(8.,50.))
        self.assertTrue(result['actual_guard_constructor_passed'])
        coding=controller.prepare(self.coding,'coding_evaluation',self.key,self.previous,funding_deployment=self.folder)
        self.assertEqual((coding['intent'],coding['reservation_usd']),
            (self.deployment.plan['coding_evaluation_intent'],65.))
        controller.Deployment(self.coding,self.provider).reserve()
        self.assertEqual(base.load(self.root/'budget.json')['cloud_reserved_usd'],before)

    def test_no_early_evaluation_unknown_deletion_or_mutation_on_failure(self):
        with self.assertRaises((ValueError,FileNotFoundError)):
            controller.prepare(self.evaluation,'mixed_evaluation',self.key,self.folder)
        self.assertFalse(self.evaluation.exists())
        state=base.load(self.previous/'state.json');state['roles']['gpu']['known_ids'].append('unknown-duplicate')
        base.atomic_json(self.previous/'state.json',state)
        with self.assertRaisesRegex(ValueError,'deletion'):
            controller.prepare(self.root/'new','mixed_training',self.key,self.previous,self.root/'e2',self.root/'c2')
        self.assertFalse((self.root/'new').exists())

    def test_actual_guard_rejection_precedes_all_new_plan_secret_writes(self):
        before=set((self.root/'.secrets/cardinal-gpu').iterdir())
        with patch.object(controller,'Guard',side_effect=ValueError('guard bound')):
            with self.assertRaisesRegex(ValueError,'guard bound'):
                controller.prepare(self.root/'failed','mixed_training',self.key,self.previous,self.root/'e2',self.root/'c2')
        self.assertFalse((self.root/'failed').exists())
        self.assertEqual(set((self.root/'.secrets/cardinal-gpu').iterdir()),before)

    def test_lost_post_never_retries_and_gpu_requires_armed_guard(self):
        self.deployment.reserve();self.provider.lose_create_response=True
        with self.assertRaisesRegex(RuntimeError,'read-only'):self.deployment.create('guard')
        with self.assertRaisesRegex(ValueError,'already attempted'):self.deployment.create('guard')
        self.assertEqual(self.provider.calls.count(('POST','/pods')),1)
        self.assertNotIn(self.provider.key,(self.folder/'state.json').read_text())
        with patch.object(self.deployment,'guard_health',return_value={'ready':False}):
            with self.assertRaisesRegex(ValueError,'armed'):self.deployment.create('gpu')
        self.assertEqual(self.provider.calls.count(('POST','/pods')),1)

    def test_partial_funding_not_recreated_and_reviewed_prior_liability_retained(self):
        self.deployment.reserve();budget=base.load(self.root/'budget.json')
        missing=self.deployment.plan['coding_evaluation_intent'];saved=budget['cloud_reservations'].pop(missing)
        budget['cloud_reserved_usd']-=65;base.atomic_json(self.root/'budget.json',budget)
        with self.assertRaisesRegex(ValueError,'Incomplete'):self.deployment.reserve()
        budget['cloud_reservations'][missing]=saved;budget['cloud_reserved_usd']+=65
        training=budget['cloud_reservations'][self.deployment.plan['intent']]
        training.update(amount_usd=100.,state='terminated_pending_billing');budget['cloud_reserved_usd']-=25
        base.atomic_json(self.root/'budget.json',budget);self.terminal(self.folder,'training_complete')
        controller.prepare(self.evaluation,'mixed_evaluation',self.key,self.folder)
        controller.Deployment(self.evaluation,self.provider).reserve()
        self.assertEqual(base.load(self.root/'budget.json')['cloud_reserved_usd'],289.)


if __name__=='__main__':unittest.main()
