"""CPU-only accounting and fail-closed dry-run tests. No network or GPU use."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts import runpod_pilot as pilot


class RunpodPlanTests(unittest.TestCase):
    def test_ledger_counts_uncertain_calls_conservatively(self):
        rows={'a':{'state':'completed','charged_upper_usd':.1,'reserved_usd':.4},
              'b':{'state':'unknown','charged_upper_usd':.2,'reserved_usd':.7}}
        self.assertAlmostEqual(pilot.ledger_commitment({'requests':rows}),.8)
        for row in [{'state':'completed'}, {'state':'unknown','reserved_usd':0},
                    {'state':'completed','charged_upper_usd':float('nan')}]:
            with self.assertRaises(ValueError):pilot.ledger_commitment({'requests':{'x':row}})

    def test_all_ledgers_and_cloud_reservations_are_read_without_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'runs').mkdir()
            (root/'runs/external-budget.lock').touch()
            budget={'approved_pilot_cap':100,'cloud_spend_to_date':3,'cloud_reserved_usd':5}
            path=root/'budget.json';path.write_text(json.dumps(budget));before=path.read_bytes()
            for name,amount in [('first-ledger.json',2),('new-client-ledger.json',4)]:
                (root/'runs'/name).write_text(json.dumps({'requests':{'x':{'state':'completed','charged_upper_usd':amount}}}))
            result=pilot.budget_snapshot(root)
            self.assertEqual(result['cumulative_commitment_usd'],14)
            self.assertEqual(path.read_bytes(),before);self.assertFalse(result['reservation_applied'])
            (root/'runs/unknown-ledger.json').write_text('{}')
            with self.assertRaises(ValueError):pilot.budget_snapshot(root)

    def test_costs_include_disk_rounding_and_reject_invalid_numbers(self):
        result=pilot.estimate(.00076*3600,23405,True)
        self.assertLess(result['total_usd'],20)
        self.assertEqual(result['disk_billed_seconds'],23700)
        self.assertGreater(result['total_usd'],result['compute_usd'])
        self.assertFalse(result['hard_upper_bound_proven'])
        for rate in [float('inf'),-1,True]:
            with self.assertRaises(ValueError):pilot.estimate(rate,3600)

    def test_plan_uses_official_fields_and_never_claims_launch_readiness(self):
        evidence=Path('runs/runpod-deployment-plan')
        spec=json.loads((evidence/'openapi.json').read_text())
        quote=json.loads((evidence/'a100-quote.json').read_text())
        now=pilot.DEADLINE-8*3600;quote['checked_at_unix']=now
        plan=pilot.make_plan(spec,quote,{'reservation_fits':True},now)
        self.assertFalse(plan['can_launch']);self.assertEqual(plan['external_actions'],0)
        request=plan['serverless_request_example_NOT_SUBMITTED']
        self.assertEqual(request['workers'],{'min':0,'max':1,'idleTimeout':5})
        self.assertNotIn('retryLimit',request)
        self.assertNotIn('terminateAfter',pilot.schema_properties(spec,'CreatePodRequest'))
        policy=plan['serverless_job_policy']
        self.assertEqual(policy['executionTimeout'],6*3600*1000)
        self.assertLess(now+policy['ttl']/1000,pilot.DEADLINE)
        self.assertTrue(any('Strategic capability gate failed' in s for s in plan['blockers']))
        with self.assertRaises(ValueError):pilot.make_plan(spec,quote,{'reservation_fits':True},pilot.DEADLINE)
        broken=deepcopy(quote);broken['gpu']['memory']=48
        with self.assertRaises(ValueError):pilot.make_plan(spec,broken,{'reservation_fits':True},now)


if __name__=='__main__':unittest.main()
