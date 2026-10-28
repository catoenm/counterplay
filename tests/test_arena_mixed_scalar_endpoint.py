"""Fixed endpoint admission uses full update timings, never scores."""
import math
import json
import unittest
from selfplay.arena_mixed_scalar.experiment import select_endpoint

class FixedEndpoint(unittest.TestCase):
    def test_twenty_full_costs_and_export_are_required(self):
        self.assertIsNone(select_endpoint([100, 200], 7799)['endpoint'])
        selected=select_endpoint([100, 200], 7800)
        self.assertEqual(selected['endpoint'],20)
        self.assertEqual(selected['projected_seconds'],{'20':7800})
        self.assertEqual(selected['full_cohort_upper_seconds'],300)
        self.assertEqual(json.loads(json.dumps(selected)),selected)
    def test_extra_time_does_not_select_more_or_less_samples(self):
        self.assertEqual(select_endpoint([10],10**8)['endpoint'],20)
        self.assertIsNone(select_endpoint([10],2000)['endpoint'])
    def test_no_unmeasured_or_nonfinite_profile_can_admit(self):
        for costs in ([],[0],[-1],[math.nan],[math.inf]):
            with self.subTest(costs=costs),self.assertRaises(ValueError):
                select_endpoint(costs,10**8)
if __name__=='__main__':unittest.main()
