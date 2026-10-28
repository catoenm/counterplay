"""Fake-provider tests only; no paid resources or real budget mutation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import runpod_debug as debug
from scripts import runpod_cardinal as base
from tests.test_runpod_cardinal import FakeProvider


class DebugDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);(self.root/'runs').mkdir()
        (self.root/'scripts/cloud').mkdir(parents=True)
        (self.root/'scripts/cloud/cardinal_guard.py').write_text('fixture only')
        self.key=self.root/'ssh.pub';self.key.write_text('ssh-ed25519 fixture-only')
        for module in (debug,base):
            p=patch.object(module,'ROOT',self.root);p.start();self.addCleanup(p.stop)
        self.folder=self.root/'deployment'
        debug.prepare(self.folder,'NVIDIA H200',5.29,self.key,11.5)
        self.budget={'approved_pilot_cap':100,'cloud_spend_to_date':0,'cloud_reserved_usd':18,
            'cloud_reservations':{'old1':{'state':'terminated_pending_billing','amount_usd':6},
                                  'old2':{'state':'terminated_pending_billing','amount_usd':12}}}
        base.atomic_json(self.root/'budget.json',self.budget)
        self.provider=FakeProvider();self.d=debug.Deployment(self.folder,self.provider)

    def test_preparation_is_integer_deadline_with_no_reservation_or_provider_calls(self):
        self.assertEqual(self.d.plan['reservation_usd'],75)
        self.assertEqual(self.d.plan['cpu_id'],'cpu5c');self.assertEqual(self.d.plan['cpu_vcpu'],4)
        self.assertEqual(self.d.plan['deadline_unix'],int(self.d.plan['deadline_unix']))
        self.assertLess(self.d.plan['estimate_usd']+10,75)
        self.assertFalse(self.d.state['reservation_applied']);self.assertEqual(self.provider.calls,[])
        self.assertEqual(base.load(self.root/'budget.json'),self.budget)
        with self.assertRaisesRegex(ValueError,'contingency'):
            debug.prepare(self.root/'too-long','NVIDIA H200',5.29,self.key,12)

    def test_reservation_counts_pending_liability_and_api_unknown_and_is_idempotent(self):
        base.atomic_json(self.root/'runs/api-ledger.json',{'requests':{'unknown':{'state':'interrupted_unknown','reserved_usd':.6}}})
        self.d.reserve();self.d.reserve()
        budget=base.load(self.root/'budget.json')
        self.assertEqual(budget['cloud_reserved_usd'],93)
        self.assertEqual(budget['cloud_reservations'][self.d.plan['intent']]['amount_usd'],75)
        self.assertEqual(budget['cloud_reservations']['old2'],self.budget['cloud_reservations']['old2'])
        self.assertEqual(self.provider.calls,[])

    def test_over_cap_and_inconsistent_cloud_liability_fail_before_mutation(self):
        base.atomic_json(self.root/'runs/api-ledger.json',{'requests':{'unknown':{'state':'interrupted_unknown','reserved_usd':8}}})
        before=(self.root/'budget.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'cap'):self.d.reserve()
        self.assertEqual(before,(self.root/'budget.json').read_bytes())
        self.budget['cloud_reserved_usd']=12;base.atomic_json(self.root/'budget.json',self.budget)
        with self.assertRaisesRegex(ValueError,'liability'):self.d.reserve()

    def test_scope_cannot_silently_fall_back_to_old_eighty_dollar_controller(self):
        plan=base.load(self.folder/'plan.json');plan['reservation_usd']=80
        base.atomic_json(self.folder/'plan.json',plan)
        state=base.load(self.folder/'state.json');state['plan_sha256']=base.digest(self.folder/'plan.json')
        base.atomic_json(self.folder/'state.json',state)
        with self.assertRaisesRegex(ValueError,'envelope'):debug.Deployment(self.folder,self.provider)

    def test_inherited_unknown_create_has_one_post_and_no_secret_journal(self):
        self.d.reserve();self.provider.lose_create_response=True
        with self.assertRaisesRegex(RuntimeError,'read-only'):self.d.create('guard')
        with self.assertRaisesRegex(ValueError,'already attempted'):self.d.create('guard')
        self.assertEqual(self.provider.calls.count(('POST','/pods')),1)
        self.assertNotIn(self.provider.key,(self.folder/'state.json').read_text())
        self.assertEqual(base.load(self.root/'budget.json')['cloud_reserved_usd'],93)

    def test_inherited_guard_admission_prevents_gpu_post(self):
        self.d.reserve();self.d.state['roles']['guard']={'pod_id':'guard'}
        with patch.object(self.d,'guard_health',return_value={'ready':False}):
            with self.assertRaisesRegex(ValueError,'armed'):self.d.create('gpu')
        self.assertEqual(self.provider.calls,[])


if __name__=='__main__':unittest.main()
