"""Pure CPU fixed-screen and cost-gate fixtures; no model/provider calls."""
from copy import deepcopy
import unittest

from selfplay.planning_curriculum import admission as a
from selfplay.planning_curriculum.cuda_policy import digest


def screen(success_indices=(0, 2, 4, 6), hard_successes=0):
    specs, episodes = [], []
    base = {"checkpoint_id": "base", "version": 0, "adapter_sha256": "a"*64, "base_identity": {}}
    for group, depths, count in (("hard", range(3, 7), 8), ("easy", (1, 2), 16)):
        group_index = 0
        for depth in depths:
            for i in range(count):
                name = f"{group}-{depth}-{i}"
                spec = {"episode_id": name, "environment": "planning", "seed": len(specs), "regime": "train",
                    "depth": depth, "scenario_sha256": digest(name), "scenario_family_id": digest([name]),
                    "learner_seats": [0], "opponent": "none", "split": "screen", "paired_seed_key": name,
                    "screen_group": group, "pair_id": f"easy-depth{depth}-pair{i//2:02d}" if group == "easy" else None}
                success = group_index in success_indices if group == "easy" else group_index < hard_successes
                row = {"episode_id": name, "turn": 0, "input_ids": list(range(100)), "environment": "planning", "player": 0}
                episode = {"episode_id": name, "environment": "planning", "depth": depth,
                    "schedule": spec, "scenario_sha256": spec["scenario_sha256"],
                    "scenario_family_id": spec["scenario_family_id"], "learner_seats": [0],
                    "behavior_binding": base, "returns": [float(success), 0.], "transitions": [row]}
                specs.append(spec); episodes.append(episode); group_index += 1
    return episodes, specs


def cost_fixture():
    episodes, specs = screen()
    timing = []
    for e in episodes:
        row = e["transitions"][0]
        timing.append({"rows": [row], "timing": {"kind": "planning_screen_sampling_timing_v1",
            "elapsed_seconds": .11, "nonforward_seconds": .01, "row_sha256": [digest(row)],
            "forward_timings": [{"row_sha256": digest(row), "input_tokens": 100, "training": False,
                "grad_enabled": False, "seconds": .1}]}})
    forward = [{"input_tokens": 100, "training": mode, "grad_enabled": mode, "seconds": .1}
        for mode in (False, True) for _ in range(2)]
    profile = {"kind": "planning_curriculum_restored_full_step_probe_v1", "passed": True,
        "training_use": False, "retained_optimizer_steps": 0, "temporary_optimizer_steps": 1,
        "state_restored": True, "state_before": {"adapter": "a", "adam": 0},
        "state_after": {"adapter": "a", "adam": 0}, "decisions": 2,
        "mature_state_roundtrip": {"passed": True, "state_before": {"optimizer": {"step": 1}},
            "state_after": {"optimizer": {"step": 1}}, "optimizer_steps": 0, "model_forwards": 0, "seconds": .2},
        "temporary_checkpoint": {"optimizer_step": 1}, "gate": {"passed": True},
        "gate_forward_timings": forward, "gate_seconds": .5, "checkpoint_seconds": .3,
        "temporary_update": {"microbatch_timings": [{"row_indices": [0, 1], "input_tokens": [100, 100],
            "forward_backward_seconds": .4}], "optimizer_verification_seconds": .2, "elapsed_seconds": .7}}
    context = {"method": "quadratic_context_upper_envelope", "maximum_input_tokens": 100,
        "justification": "Synthetic test renderer has a fixed 100-token maximum, no inference evidence.",
        "renderer_sha256": "a"*64, "tokenizer_sha256": "b"*64, "review_sha256": "c"*64}
    budget = {"aggregate_started_at": "2026-10-25T00:00:00Z", "checked_at": "2026-10-25T01:00:00Z",
        "hard_stop": "2026-10-25T12:00:00Z", "fixed_main_overhead_seconds": 600,
        "overhead_review_sha256": "d"*64, "gpu_hourly_rate": 5.29, "guard_hourly_rate": .14,
        "storage_hourly_rate": .02, "admission_spent_usd": 5.45, "other_reserved_cost_usd": 1.}
    return profile, timing, context, budget


class SignalTests(unittest.TestCase):
    def test_fixed_pairs_signal_and_hard_descriptive_only(self):
        episodes, specs = screen()
        result = a.signal_gate(episodes, specs)
        self.assertTrue(result["passed"])
        self.assertEqual(result["easy_successes"], 4)
        self.assertEqual(result["mixed_easy_pairs"], 4)
        self.assertEqual(result["timing_probe_episode_ids"], ["hard-6-0", "hard-6-1"])
        other, other_specs = screen(hard_successes=32)
        self.assertTrue(a.signal_gate(other, other_specs)["passed"])
        self.assertFalse(result["training_authorized"])
    def test_all_fail_all_succeed_and_no_contrast_fail_without_reselection(self):
        for successes in ((), tuple(range(32)), (0, 1, 2, 3)):
            episodes, specs = screen(successes)
            result = a.signal_gate(episodes, specs)
            self.assertFalse(result["passed"])
            self.assertEqual(len(result["easy_pairs"]), 16)
    def test_incomplete_wrong_base_repair_and_cross_depth_pairs_rejected(self):
        episodes, specs = screen()
        with self.assertRaises(ValueError): a.signal_gate(episodes[:-1], specs[:-1])
        for change in (
            lambda e, s: e[0]["behavior_binding"].update(version=1),
            lambda e, s: e[0].update(returns=[.5, 0.]),
            lambda e, s: s[32].update(pair_id="easy-depth2-pair00"),
            lambda e, s: s[0].update(scenario_family_id=s[1]["scenario_family_id"]),
        ):
            e, s = deepcopy(episodes), deepcopy(specs); change(e, s)
            with self.assertRaises(ValueError): a.signal_gate(e, s)


class TimingTests(unittest.TestCase):
    def test_caps_full_step_once_margin_and_protected_hour(self):
        args = cost_fixture(); result = a.timing_gate(*args)
        self.assertTrue(result["passed"], result)
        # sample=.1+.01; gate=.1+.1; learner=.4+.2; fixed=.2+.3+.1+.1
        expected_cohort = 24*(.11+.6)+.7
        self.assertAlmostEqual(result["full_cohort_seconds_before_margin"], expected_cohort)
        expected_main = 1.5*(80*expected_cohort+320*12*.11+600)
        self.assertAlmostEqual(result["main_seconds_with_margin"], expected_main)
        self.assertAlmostEqual(result["required_remaining_seconds"], expected_main+3600)
        self.assertEqual(result["sampled_screen_decisions"], 64)
        self.assertFalse(result["training_authorized"])
    def test_longest_context_and_short_remaining_time_cannot_be_waived(self):
        profile, timings, context, budget = cost_fixture()
        baseline = a.timing_gate(profile, timings, context, budget)
        context["maximum_input_tokens"] = 300
        longer = a.timing_gate(profile, timings, context, budget)
        self.assertGreater(longer["required_remaining_seconds"], baseline["required_remaining_seconds"])
        budget["checked_at"] = "2026-10-25T11:00:00Z"
        failed = a.timing_gate(profile, timings, context, budget)
        self.assertFalse(failed["passed"])
        self.assertIn("immutable_aggregate_deadline", failed["failure_reasons"])
    def test_frozen_cost_evidence_all_64_rows_and_restoration_required(self):
        for edit in (
            lambda p, s, c, b: p.update(state_after={"adapter": "changed"}),
            lambda p, s, c, b: p.update(temporary_optimizer_steps=0),
            lambda p, s, c, b: p["mature_state_roundtrip"].update(model_forwards=1),
            lambda p, s, c, b: s.pop(),
            lambda p, s, c, b: s[0]["rows"][0].update(turn=7),
            lambda p, s, c, b: c.update(review_sha256="unreviewed"),
            lambda p, s, c, b: b.update(hard_stop="2026-10-26T00:00:00Z"),
            lambda p, s, c, b: b.update(gpu_hourly_rate=float("nan")),
            lambda p, s, c, b: b.update(fixed_main_overhead_seconds=0),
        ):
            args = cost_fixture(); edit(*args)
            with self.assertRaises(ValueError): a.timing_gate(*args)
    def test_separate_stage_caps_bind_even_when_total_is_low(self):
        profile, timings, context, budget = cost_fixture()
        budget["admission_spent_usd"] = 15.01
        result = a.timing_gate(profile, timings, context, budget)
        self.assertFalse(result["passed"])
        self.assertIn("admission_dollar_cap", result["failure_reasons"])
        budget["admission_spent_usd"] = 1; budget["other_reserved_cost_usd"] = 6
        result = a.timing_gate(profile, timings, context, budget)
        self.assertFalse(result["passed"])
        self.assertIn("protected_dollar_cap", result["failure_reasons"])


if __name__ == "__main__": unittest.main()
