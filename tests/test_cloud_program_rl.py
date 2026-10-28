"""CPU contract/replay/resume tests. These do not validate CUDA kernels or fit."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from safetensors.numpy import save_file, load_file
from scripts.cloud import train_program_rl as rl


COLLECTOR = '''def act():
    if energy == 0:
        return {'action':'rest'}
    if self_x == home_x and self_y == home_y and carrying > 0:
        return {'action':'deposit'}
    tx = home_x
    ty = home_y
    target = -1
    best = 10000
    if carrying < capacity and horizon - turn > abs(self_x-home_x)+abs(self_y-home_y)+5:
        for i in range(6):
            d = abs(site(i,'x')-self_x)+abs(site(i,'y')-self_y)
            if site(i,'units') > 0 and d < best:
                best = d
                tx = site(i,'x')
                ty = site(i,'y')
                target = i
    if self_x == tx and self_y == ty:
        if target >= 0:
            return {'action':'harvest'}
        return {'action':'rest'}
    if self_x < tx:
        return {'action':'move','dx':1,'dy':0}
    if self_x > tx:
        return {'action':'move','dx':-1,'dy':0}
    if self_y < ty:
        return {'action':'move','dx':0,'dy':1}
    return {'action':'move','dx':0,'dy':-1}
'''
IDLE = "def act():\n    return {'action':'rest'}\n"


class FakePolicy:
    """Never imports torch; deterministic fixture adapter and fresh responses."""
    def __init__(self):
        self.value = 0.0
        self.samples = 0
        self.updates = 0
        self.fail_update = False
        self.fail_gate = False
        self.broken_opponents = False

    def sample(self, prompt, seed):
        self.samples += 1
        # Candidate group: one invalid, one idle, two useful; all opponents useful.
        candidate = 144401 <= seed <= 144504
        text = COLLECTOR
        if candidate and seed % 4 == 1:
            text = "import os"
        if candidate and seed % 4 == 2:
            text = IDLE
        if self.broken_opponents:
            text = IDLE
        return {"text": text, "raw_text": text, "prompt_template": prompt, "prompt_ids": [1, 2],
                "completion_ids": [4, 5, 9], "sampled_token_logprobs": [-1.0, -2.0, -3.0],
                "seed": seed, "finish_reason": "eos", "eos_token_ids": [9]}

    def likelihood_gate(self, records):
        reports = [rl.check_likelihoods(r["sampled_token_logprobs"], r["sampled_token_logprobs"], .002, .02)
                   for r in records]
        return {"passed": not self.fail_gate, "records": reports}

    def update(self, records, advantages):
        if self.fail_update:
            self.value += 123  # Simulate a crash after in-memory mutation, before durable commit.
            raise RuntimeError("simulated interrupted update")
        self.updates += 1
        self.value += 1
        return {"adapter_l2_delta": 1.0, "optimizer_steps": 1,
                "gradient_likelihood_gate": self.likelihood_gate(records)}

    def save_checkpoint(self, folder):
        folder.mkdir(parents=True)
        save_file({"lora": np.array([self.value], dtype=np.float32)}, str(folder / "adapter.safetensors"))
        (folder / "optimizer.pt").write_text(str(self.value))
        return {name: rl.digest(folder / name) for name in ("adapter.safetensors", "optimizer.pt")}

    def load_checkpoint(self, folder, hashes):
        for name, expected in hashes.items():
            if rl.digest(folder / name) != expected:
                raise ValueError("checkpoint hash mismatch")
        self.value = float(load_file(str(folder / "adapter.safetensors"))["lora"][0])


class CloudProgramRLTests(unittest.TestCase):
    def prepare(self, parent, **kwargs):
        prompt = Path(parent) / "prompt.txt"
        prompt.write_text("Fixture specification, never used with a real model")
        run = Path(parent) / "run"
        rl.prepare(run, prompt, "fixture/repo", "a" * 40,
                   overrides={"deadline": "2099-01-01T00:00:00+00:00", "map_seeds_per_group": 1,
                              "eval_samples": 1, "opponent_count": 1, **kwargs})
        return run

    def test_completion_mask_includes_eos_without_prompt_or_invented_token(self):
        self.assertEqual(rl.completion_positions([10, 11, 12], [20, 21, 99]), [2, 3, 4])
        self.assertEqual(rl.completion_positions([10], [20]), [0])
        with self.assertRaises(ValueError):
            rl.completion_positions([10], [])

    def test_baseline_excludes_own_reward_and_zero_variance_aborts(self):
        a = rl.rloo_advantages([1., 0., -1., .5])
        changed = rl.rloo_advantages([10., 0., -1., .5])
        self.assertAlmostEqual(changed[0] - a[0], 9.)
        self.assertAlmostEqual(sum(a), 0.)
        with self.assertRaisesRegex(ValueError, "Zero reward variance"):
            rl.rloo_advantages([0., 0., 0., 0.])

    def test_reinforce_sign_for_full_sequence_sum(self):
        # Two independent sampled tokens, both token0. Positive advantage must
        # push their scalar logit up; negative advantage must push it down.
        def loss(theta, advantage):
            logp = -math.log1p(math.exp(-theta))
            return -advantage * (logp + logp) / 4
        derivative = (loss(.00001, 1) - loss(-.00001, 1)) / .00002
        self.assertLess(derivative, 0)
        self.assertAlmostEqual(derivative, -.25, places=6)
        self.assertGreater((loss(.00001, -1) - loss(-.00001, -1)) / .00002, 0)

    def test_likelihood_gate_checks_sequence_accumulation_not_only_max_token(self):
        report = rl.check_likelihoods([-1.] * 100, [-1.001] * 100, .002, .02)
        self.assertFalse(report["passed"])
        self.assertGreater(report["sequence_error"], .09)
        self.assertTrue(rl.check_likelihoods([-1., -2.], [-1.00001, -2.00001], .002, .02)["passed"])
        with self.assertRaises(ValueError):
            rl.check_likelihoods([-1.], [], .002, .02)

    def test_invalid_or_truncated_authorship_has_no_fabricated_game(self):
        config = dict(rl.DEFAULTS)
        for text, finish in ((IDLE, "length"), ("import os", "eos"), ("<think>" + IDLE, "eos")):
            score = rl.score_program({"text": text, "finish_reason": finish}, [], [1], config)
            self.assertEqual(score["reward"], -1)
            self.assertEqual(score["reward_kind"], "invalid_authorship")
            self.assertEqual(score["matches"], [])
            self.assertIsNone(score["game_return"])
        classified, program = rl.classify("Here is code\n```python\n" + IDLE + "```\n", "eos")
        self.assertTrue(classified["valid"])
        self.assertIsNotNone(program)

    def test_full_two_update_fixture_replays_and_resume_does_not_repeat(self):
        with tempfile.TemporaryDirectory() as folder:
            run = self.prepare(folder)
            backend = FakePolicy()
            rl.Runner(run, backend).run_all()
            self.assertEqual((backend.updates, backend.value), (2, 2))
            result = rl.audit(run)
            self.assertTrue(result["complete"])
            self.assertGreater(result["replayed_matches"], 0)
            resumed = FakePolicy()
            rl.Runner(run, resumed).run_all()
            self.assertEqual((resumed.samples, resumed.updates, resumed.value), (0, 0, 2))

    def test_interrupted_update_reloads_source_and_reuses_same_rollouts(self):
        with tempfile.TemporaryDirectory() as folder:
            run = self.prepare(folder, iterations=1)
            first = FakePolicy()
            first.fail_update = True
            with self.assertRaisesRegex(RuntimeError, "interrupted update"):
                rl.Runner(run, first).run_all()
            self.assertEqual(first.value, 123)
            original = {p.name: rl.digest(p) for p in (run / "generations").glob("candidate-*.json")}
            resumed = FakePolicy()
            rl.Runner(run, resumed).run_all()
            self.assertEqual((resumed.updates, resumed.value), (1, 1))
            self.assertEqual(original, {p.name: rl.digest(p) for p in (run / "generations").glob("candidate-*.json")})
            self.assertTrue(rl.audit(run)["complete"])

    def test_zero_banking_opponents_and_failed_likelihood_stop_before_update(self):
        for kind in ("broken_opponents", "fail_gate"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                run = self.prepare(folder, iterations=1)
                backend = FakePolicy()
                setattr(backend, kind, True)
                with self.assertRaises(ValueError):
                    rl.Runner(run, backend).run_all()
                self.assertEqual(backend.updates, 0)
                self.assertFalse((run / "checkpoints/step1/metadata.json").exists())

    def test_committed_generation_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            run = self.prepare(folder, iterations=1)
            rl.Runner(run, FakePolicy()).run_all()
            path = next((run / "generations").glob("candidate-*.json"))
            row = json.loads(path.read_text())
            row["completion_ids"][0] = 12345
            rl.atomic_json(path, row)
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                rl.audit(run)

    def test_audit_rejects_failed_gradient_gate_even_with_consistent_file_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            run = self.prepare(folder, iterations=1)
            rl.Runner(run, FakePolicy()).run_all()
            relative = "checkpoints/step1/metadata.json"
            metadata = rl.load(run / relative)
            metadata["report"]["gradient_likelihood_gate"]["passed"] = False
            rl.atomic_json(run / relative, metadata)
            status = rl.load(run / "status.json")
            status["artifacts"][relative] = rl.digest(run / relative)
            rl.atomic_json(run / "status.json", status)
            with self.assertRaisesRegex(ValueError, "gate coverage"):
                rl.audit(run)

    def test_config_rejects_hidden_sampler_changes_and_overlong_generation(self):
        base = {**rl.DEFAULTS, "model_repo": "x", "model_revision": "a" * 40, "loader": "causal_lm"}
        for change in ({"top_p": .8}, {"temperature": .6}, {"max_new_tokens": 9999}, {"thinking": True},
                       {"learning_rate": float("nan")}, {"gradient_clip": -1},
                       {"sequence_logprob_tolerance": .2}, {"train_seed_start": 3899000}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                rl.validate_config({**base, **change})

    def test_model_index_requires_exact_revision_and_all_staged_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            for name in ("config.json", "tokenizer_config.json", "model.safetensors"):
                (directory / name).write_bytes(b"fixture")
            config = {"model_repo": "fixture/repo", "model_revision": "a" * 40}
            index = {"repo": "fixture/repo", "revision": "a" * 40,
                     "files": {p.name: rl.digest(p) for p in directory.iterdir()}}
            rl.atomic_json(directory / "model-index.json", index)
            rl.verify_model_index(directory, config)
            (directory / "unindexed.txt").write_text("unexpected")
            with self.assertRaisesRegex(ValueError, "Unindexed"):
                rl.verify_model_index(directory, config)


if __name__ == "__main__":
    unittest.main()
