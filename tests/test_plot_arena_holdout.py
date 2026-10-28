"""Reporting-statistic tests: preserve shared clusters, reject partial evidence."""
import copy
import unittest

from scripts.plot_arena_holdout import extract_complete, macro_summary, GAMES, OPPONENTS


def evidence():
    rows = []
    for index, game in enumerate(GAMES):
        for opponent in OPPONENTS:
            sign = 1 if opponent == "base" else -1
            pairs = [{"seed": 3900000+index*1000+i*2, "mean_return_change": sign*(i-5.5)/20}
                     for i in range(12)]
            rows.append({"before": "base", "after": "cycle3", "environment": game,
                         "opponent": opponent, "paired_scenario_changes": pairs,
                         "scenario_clusters": 12, "mean_return_change": 0,
                         "descriptive_bootstrap_95_interval": [-.1, .1]})
    return {"complete": True, "all_completed_episodes_verified": True,
            "episodes_verified": 288, "scheduled_episodes": 288, "missing_episode_ids": [],
            "paired_checkpoint_changes": rows}


class PlotHoldoutTests(unittest.TestCase):
    def test_shared_opponent_scenarios_cancel_before_resampling(self):
        cells = extract_complete(evidence())
        result = macro_summary(cells, draws=500)
        self.assertEqual(result["mean"], 0)
        self.assertEqual(result["interval_95"], [0, 0])
        self.assertEqual(result["scenario_clusters"], 36)
        # Independently resampling opponent cells would fabricate uncertainty
        # here, despite an exactly zero mean in every actual shared scenario.

    def test_partial_and_mismatched_opponent_clusters_are_rejected(self):
        partial = evidence(); partial["complete"] = False
        with self.assertRaisesRegex(ValueError, "complete independent"):
            extract_complete(partial)
        cells = extract_complete(evidence())
        cells[("market", "base")]["paired_scenario_changes"][0]["seed"] += 200
        with self.assertRaisesRegex(ValueError, "same scenario clusters"):
            macro_summary(cells, draws=10)

    def test_actual_chess_opening_families_cannot_be_double_counted(self):
        cells = extract_complete(evidence())
        for opponent in OPPONENTS:
            for i, row in enumerate(cells[("chess", opponent)]["paired_scenario_changes"]):
                row["seed"] = 3902000+i
        with self.assertRaisesRegex(ValueError, "actual scenario families"):
            macro_summary(cells, draws=10)


if __name__ == "__main__":
    unittest.main()
