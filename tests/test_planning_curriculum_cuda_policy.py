"""Tiny real CPU Torch tensors only; no pretrained weights or CUDA work."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from selfplay.arena_mixed import cuda_policy as original
from selfplay.planning_curriculum import cuda_policy as policy
from tests.test_arena_mixed_cuda_policy import fixture as original_fixture, request


def fixture(role="learner"):
    obj = original_fixture(role)
    obj.__class__ = policy.CudaActionPolicy
    obj.config = policy.default_config(model_index_sha256="a"*64, tokenizer_files_sha256="b"*64)
    return obj


def episodes(obj, returns=(1., 0.), lengths=(2, 1)):
    result = []
    for index, (reward, length) in enumerate(zip(returns, lengths, strict=True)):
        name = f"hard-depth6-{index}"
        transitions = []
        for turn in range(length):
            row = obj.choose([request("planning", name, turn=turn, seed=100+index*20+turn)], obj.binding)[0]
            row["terminal_return"] = reward
            transitions.append(row)
        result.append({"episode_id": name, "environment": "planning", "scenario_family_id": name,
            "depth": 6, "schedule": {"depth": 6, "screen_group": "hard"},
            "learner_seats": [0], "returns": [reward, 0.], "transitions": transitions})
    return result


class PlanningPolicyTests(unittest.TestCase):
    def setUp(self):
        self.cuda_rng = [torch.get_rng_state().clone()]
        self.patches = [patch.object(torch.cuda, "get_rng_state_all", side_effect=lambda: [x.clone() for x in self.cuda_rng]),
            patch.object(torch.cuda, "set_rng_state_all", side_effect=lambda x: self.cuda_rng.__setitem__(slice(None), [y.clone() for y in x]))]
        for p in self.patches: p.start()
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
    def tearDown(self):
        for p in reversed(self.patches): p.stop()
    def base(self, obj):
        folder = Path(self.tmp.name)/f"base-{id(obj)}"
        return folder, obj.save_checkpoint(folder)

    def test_fixed_objective_config_and_single_task_validation(self):
        cfg = policy.default_config()
        self.assertEqual(cfg["environment_weights"], {"planning": 1.})
        self.assertEqual(cfg["global_loss_multiplier"], .24)
        for key, value in (("global_loss_multiplier", 1), ("learning_rate", 1e-5),
                ("entropy_coefficient", 0), ("environment_weights", {"planning": .24}),
                ("max_context_tokens", 8192), ("seed", True)):
            bad = deepcopy(cfg); bad[key] = value
            with self.assertRaises(ValueError): policy.validate_config(bad)
        obj = fixture(); self.base(obj)
        with self.assertRaisesRegex(ValueError, "planning"): obj.choose([request("coding")], obj.binding)

    def test_all_three_objective_values_and_gradients_scale_together(self):
        cfg = policy.default_config()
        logits = torch.tensor([[.3, -.2, .8], [-.1, .2, .6]], requires_grad=True)
        old = torch.tensor([[.1, .2, .4], [.2, .1, -.1]]).log_softmax(-1)
        args = (old, torch.tensor([3, 3]), torch.tensor([0, 1]), torch.tensor([1., -1.]), torch.tensor([.5, .5]), cfg)
        full, terms = original.objective(logits.log_softmax(-1), *args)
        full_grad = torch.autograd.grad(full, logits)[0]
        scaled, scaled_terms = policy.objective(logits.log_softmax(-1), *args)
        scaled_grad = torch.autograd.grad(scaled, logits)[0]
        self.assertTrue(torch.allclose(scaled, .24*full))
        self.assertTrue(torch.allclose(scaled_grad, .24*full_grad, atol=1e-7))
        for key in ("policy_loss", "entropy_loss", "kl_loss"):
            self.assertTrue(torch.allclose(scaled_terms[key], .24*terms[key]))

    def test_loo_weights_and_real_persistent_step_then_exact_reload(self):
        obj = fixture(); folder, base = self.base(obj); data = episodes(obj)
        rows = policy.cohort_rows(data, obj.config["environment_weights"])
        self.assertEqual([r["weight"] for r in rows], [.5]*3)
        self.assertEqual([r["advantage"] for r in rows], [1., 1., -1.])
        raw = [r for e in data for r in e["transitions"]]; binding = deepcopy(obj.binding)
        gate = obj.full_likelihood_gate(raw, binding)
        report = obj.train_cohort(data, binding, gate)
        self.assertEqual(report["optimizer_steps"], 1)
        self.assertGreater(report["parameter_delta_l2"], 0)
        self.assertEqual(report["global_loss_multiplier"], .24)
        self.assertEqual(len(report["microbatch_timings"]), 2)
        # Reconstruct all three reported terms from saved full vectors.
        total = 0.
        for saved in report["learner_records"]:
            row = raw[saved["index"]]
            lp = torch.tensor([saved["learner_log_probs"]])
            old = torch.tensor([row["old_log_probs"]])
            loss, _ = policy.objective(lp, old, torch.tensor([len(row["actions"])]),
                torch.tensor([row["action_index"]]), torch.tensor([saved["advantage"]]),
                torch.tensor([saved["weight"]]), obj.config)
            total += float(loss)
        self.assertAlmostEqual(total, report["loss"], places=6)
        target = Path(self.tmp.name)/"cycle1"; saved = obj.save_checkpoint(target)
        self.assertEqual(saved["step"], 1)
        before = obj._state_identity()
        obj.gate_at_checkpoint(raw, folder, base["files"])
        self.assertEqual(before, obj._state_identity())
        actor = fixture("actor"); actor.load_checkpoint(target, saved["files"])
        self.assertEqual(actor._optimizer_identity(), obj._optimizer_identity())
        report2 = obj.train_cohort(data, binding, gate)
        self.assertEqual(report2["policy_lag"], 1)
        self.assertEqual(report2["adam_step_after"], 2)

    def test_profiled_sampling_is_same_draw_and_preserves_rng(self):
        obj = fixture(); self.base(obj)
        requests = [request("planning", seed=10), request("planning", seed=11, extra="longer "*50)]
        plain = obj.choose(requests, obj.binding); before = obj._state_identity()
        profiled = obj.choose_profiled(requests, obj.binding)
        self.assertEqual(profiled["rows"], plain)
        self.assertEqual(before, obj._state_identity())
        self.assertEqual(len(profiled["timing"]["forward_timings"]), 2)
        self.assertTrue(all(x["seconds"] > 0 for x in profiled["timing"]["forward_timings"]))

    def test_once_only_real_probe_restores_all_state_and_saves_actual_step(self):
        obj = fixture(); self.base(obj); data = episodes(obj); before = obj._state_identity()
        forward_count = len(obj.model.model.calls)
        with patch.object(obj.optimizer, "step", wraps=obj.optimizer.step) as steps:
            profile = obj.timing_probe(data, [e["episode_id"] for e in data], Path(self.tmp.name)/"probe")
        self.assertEqual(steps.call_count, 1)
        self.assertEqual(len(obj.model.model.calls)-forward_count, 3*len([r for e in data for r in e["transitions"]]))
        self.assertTrue(profile["passed"], profile)
        self.assertEqual(profile["temporary_optimizer_steps"], 1)
        self.assertEqual(profile["retained_optimizer_steps"], 0)
        self.assertEqual(before, obj._state_identity())
        self.assertEqual(policy.checkpoint_state(Path(self.tmp.name)/"probe")["step"], 1)
        self.assertEqual(len(profile["gate_forward_timings"]), 6)
        self.assertEqual(len(profile["update_forward_timings"]), 3)
        mature = profile["mature_state_roundtrip"]
        self.assertTrue(mature["passed"])
        self.assertEqual(mature["state_before"], mature["state_after"])
        self.assertEqual(mature["state_after"]["optimizer"]["step"], 1)
        self.assertEqual(mature["model_forwards"], 0)
        self.assertEqual(mature["optimizer_steps"], 0)
        self.assertGreater(mature["seconds"], 0)
        with self.assertRaisesRegex(ValueError, "retried"):
            obj.timing_probe(data, [e["episode_id"] for e in data], Path(self.tmp.name)/"retry")

    def test_ambiguous_probe_restores_but_poisoned_never_retried(self):
        obj = fixture(); self.base(obj); data = episodes(obj); before = obj._state_identity()
        original_step = obj.optimizer.step
        def broken():
            original_step(); raise RuntimeError("synthetic step-then-throw")
        with patch.object(obj.optimizer, "step", side_effect=broken):
            profile = obj.timing_probe(data, [e["episode_id"] for e in data], Path(self.tmp.name)/"bad")
        self.assertFalse(profile["passed"])
        self.assertEqual(profile["state_before"], profile["state_after"])
        self.assertEqual(before, obj._state_identity())
        self.assertIsNone(profile["temporary_optimizer_steps"])
        self.assertTrue(obj.poisoned)
        with self.assertRaises(ValueError): obj.save_checkpoint(Path(self.tmp.name)/"forbidden")

    def test_probe_cannot_use_easy_or_reselected_pair(self):
        obj = fixture(); self.base(obj); data = episodes(obj)
        with self.assertRaises(ValueError):
            obj.timing_probe(data, ["wrong", "wrong2"], Path(self.tmp.name)/"probe")
        data[0]["depth"] = 1
        with self.assertRaises(ValueError):
            obj.timing_probe(data, [e["episode_id"] for e in data], Path(self.tmp.name)/"probe")
        self.assertFalse(getattr(obj, "_timing_probe_attempted", False))


if __name__ == "__main__": unittest.main()
