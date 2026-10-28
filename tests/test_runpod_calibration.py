"""Fake-provider coverage for the separately authorized calibration envelope."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import runpod_calibration as calibration
from scripts import runpod_cardinal as base
from tests.test_runpod_cardinal import FakeProvider


class CalibrationDeploymentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / 'runs').mkdir()
        (self.root / 'scripts/cloud').mkdir(parents=True)
        (self.root / 'scripts/cloud/cardinal_guard.py').write_text('fixture')
        self.key = self.root / 'ssh.pub'
        self.key.write_text('ssh-ed25519 fixture-only')
        for module in (calibration, base):
            patched = patch.object(module, 'ROOT', self.root)
            patched.start()
            self.addCleanup(patched.stop)
        self.folder = self.root / 'deployment'
        calibration.prepare(self.folder, 'NVIDIA H200', 5.29, self.key)
        self.provider = FakeProvider()
        self.deployment = calibration.Deployment(self.folder, self.provider)
        self.set_budget(200, 36)

    def set_budget(self, cap, held):
        base.atomic_json(self.root / 'budget.json', {
            'approved_pilot_cap': cap, 'cloud_spend_to_date': 0,
            'cloud_reserved_usd': held,
            'cloud_reservations': {'prior': {
                'amount_usd': held, 'state': 'terminated_pending_billing'}}})

    def test_prepare_is_unpaid_and_three_hour_bounded(self):
        self.assertEqual(self.deployment.plan['reservation_usd'], 25)
        self.assertEqual(self.deployment.plan['maximum_hours'], 3)
        self.assertLess(self.deployment.plan['estimate_usd'] + 8, 25)
        self.assertEqual(self.provider.calls, [])
        self.assertFalse(self.deployment.state['reservation_applied'])
        with self.assertRaisesRegex(ValueError, 'envelope'):
            calibration.prepare(self.root / 'too-long', 'NVIDIA H200', 5.29, self.key, 4)

    def test_reservation_preserves_unknown_calls_and_pending_liabilities(self):
        base.atomic_json(self.root / 'runs/api-ledger.json', {'requests': {
            'unknown': {'state': 'interrupted_unknown', 'reserved_usd': .6}}})
        self.deployment.reserve()
        self.deployment.reserve()
        budget = base.load(self.root / 'budget.json')
        self.assertEqual(budget['cloud_reserved_usd'], 61)
        self.assertEqual(budget['cloud_reservations']['prior']['amount_usd'], 36)
        self.assertEqual(self.provider.calls, [])

    def test_ceiling_honors_lower_user_cap_and_never_exceeds_two_hundred(self):
        for cap, held in [(100, 85), (1000, 180)]:
            self.set_budget(cap, held)
            before = (self.root / 'budget.json').read_bytes()
            with self.assertRaisesRegex(ValueError, 'cap'):
                self.deployment.reserve()
            self.assertEqual((self.root / 'budget.json').read_bytes(), before)
        self.set_budget(200, 150)
        self.deployment.reserve()
        self.assertEqual(base.load(self.root / 'budget.json')['cloud_reserved_usd'], 175)

    def test_inconsistent_cloud_accounting_rejected(self):
        budget = base.load(self.root / 'budget.json')
        budget['cloud_reserved_usd'] = 20
        base.atomic_json(self.root / 'budget.json', budget)
        with self.assertRaisesRegex(ValueError, 'liability'):
            self.deployment.reserve()

    def test_lost_create_response_is_never_posted_twice(self):
        self.deployment.reserve()
        self.provider.lose_create_response = True
        with self.assertRaisesRegex(RuntimeError, 'read-only'):
            self.deployment.create('guard')
        with self.assertRaisesRegex(ValueError, 'already attempted'):
            self.deployment.create('guard')
        self.assertEqual(self.provider.calls.count(('POST', '/pods')), 1)
        self.assertNotIn(self.provider.key, (self.folder / 'state.json').read_text())

    def test_gpu_requires_armed_guard(self):
        self.deployment.reserve()
        self.deployment.state['roles']['guard'] = {'pod_id': 'guard'}
        with patch.object(self.deployment, 'guard_health', return_value={'ready': False}):
            with self.assertRaisesRegex(ValueError, 'armed'):
                self.deployment.create('gpu')
        self.assertEqual(self.provider.calls, [])


if __name__ == '__main__':
    unittest.main()
