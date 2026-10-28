"""CPU-only tests for the matched-study state machine; never imports MLX."""
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from selfplay import program_study as study


class FakePolicy:
    def __init__(self):
        self.weight, self.step = 0, 0
        self.updates, self.samples, self.loads = 0, [], []
        self.fail_save = False

    def load_checkpoint(self, folder, hashes):
        for name, sha in hashes.items():
            assert study.common.digest(folder / name) == sha
        self.weight = json.loads((folder / "adapter.safetensors").read_text())
        self.step = json.loads((folder / "optimizer.safetensors").read_text())
        self.loads.append((folder.name, self.weight, self.step))

    def save_checkpoint(self, folder):
        folder.mkdir(parents=True)
        (folder / "adapter.safetensors").write_text(json.dumps(self.weight))
        (folder / "optimizer.safetensors").write_text(json.dumps(self.step))
        if self.fail_save:
            self.fail_save = False
            raise RuntimeError("simulated interrupted checkpoint")
        return {f: study.common.digest(folder / f) for f in ("adapter.safetensors", "optimizer.safetensors")}

    def sample(self, prompt, seed):
        self.samples.append((prompt, seed, self.weight, self.step))
        return {"text": "fake", "raw_text": "fake", "seed": seed, "finish_reason": "eos",
                "prompt_ids": [1], "completion_ids": [2], "sampled_token_logprobs": [-.5],
                "input_tokens": 1, "output_tokens": 1, "eos_token_ids": [2]}

    def likelihood_gate(self, records):
        return {"passed": True, "records": [{"tokens": 1, "passed": True, "max_token_error": 0,
                "sequence_error": 0, "recorded_sequence_logprob": -.5} for _ in records]}

    def update(self, records, advantages):
        assert len(records) == len(advantages) == 8
        assert any(advantages)
        self.weight += 1
        self.step += 1
        self.updates += 1
        return {"optimizer_steps": 1, "adapter_l2_delta": 1.0,
                "gradient_likelihood_gate": self.likelihood_gate(records), "loss": 1.0,
                "gradient_norm_before_clip": 1.0, "microbatch_losses": [.125] * 8,
                "peak_memory_bytes": 1}


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as directory, patch.object(study, "SOURCES", []):
        run = Path(directory)
        cfg = study.config()
        timetable = study.schedule()
        for relative, data in {"plan.json": {}, "schedule.json": timetable,
                               "map-fingerprints.json": {}, "evaluation-opponents.json": []}.items():
            study.common.atomic_json(run / relative, data)
        (run / "prompts").mkdir()
        for name in ("compact", "plain"):
            (run / f"prompts/{name}.txt").write_text(name)
        policy = FakePolicy()
        files = policy.save_checkpoint(run / "inputs/base")
        original = {"complete": True, "label": "base", "parent": None, "report": None,
                    "files": files, "model_repo": study.local.MODEL_REPO,
                    "model_revision": study.local.MODEL_REVISION}
        study.common.atomic_json(run / "inputs/base/metadata.json", original)
        inputs = ["evaluation-opponents.json", "prompts/compact.txt", "prompts/plain.txt",
                  "inputs/base/metadata.json", "inputs/base/adapter.safetensors", "inputs/base/optimizer.safetensors"]
        manifest = {"kind": "matched_program_study", "config": cfg, "source_sha256": {},
                    "plan_sha256": study.common.digest(run / "plan.json"),
                    "schedule_sha256": study.common.digest(run / "schedule.json"),
                    "map_fingerprints_sha256": study.common.digest(run / "map-fingerprints.json"),
                    "input_sha256": {p: study.common.digest(run / p) for p in inputs},
                    "prompt_sha256": {p: study.common.digest(run / f"prompts/{p}.txt") for p in ("compact", "plain")}}
        study.common.atomic_json(run / "manifest.json", manifest)
        study.common.atomic_json(run / "status.json", {"manifest_sha256": study.common.digest(run / "manifest.json"),
                               "artifacts": {}, "state": "prepared", "claims_allowed": False})
        engine = study.Study(run, policy)
        engine.initialize()
        pool = {"selected": [], "passed": True}
        engine.commit("opponents.json", pool)
        yield run, engine, policy, pool


def varying_score(record, opponents, seeds, config, arm):
    reward = .2 if record["seed"] % 2 else -1.
    return {"reward": reward, "runtime_valid": reward != -1, "authorship_valid": reward != -1,
            "matches": [], "classification": {"valid": reward != -1},
            "candidate_program_faults": 0, "opponent_program_faults": 0,
            "banked_point_yield": 0, "game_return": reward if reward != -1 else None,
            "reward_kind": "fixture"}


class StudyTests(unittest.TestCase):
    def test_fault_penalty_only_attributes_candidate_seat(self):
        good = {"text": "def act():\n    return {'action': 'rest'}", "finish_reason": "eos"}
        bad = {"text": "def act():\n    return {'action': 'move', 'dx': 0, 'dy': 0}", "finish_reason": "eos"}
        bad_ir = study.common.classify(bad["text"], "eos")[0]["program"]["compiled_ir"]
        good_ir = study.common.classify(good["text"], "eos")[0]["program"]["compiled_ir"]
        row = study.score(good, [{"id": "bad", "origin": "model", "compiled_ir": bad_ir}], [100], study.config(), "validity_rl")
        self.assertEqual(row["candidate_program_faults"], 0)
        self.assertGreater(row["opponent_program_faults"], 0)
        self.assertEqual(row["reward"], 0)
        row = study.score(bad, [{"id": "good", "origin": "model", "compiled_ir": good_ir}], [100], study.config(), "game_rl")
        self.assertGreater(row["candidate_program_faults"], 0)
        self.assertEqual(row["reward"], -1)

    def test_failed_legal_action_is_not_runtime_invalid(self):
        candidate = {"text": "def act():\n    return {'action': 'harvest'}", "finish_reason": "eos"}
        row = study.score(candidate, [{"id": "nearest", "origin": "scripted",
                         "compiled_ir": study.common.reference_programs()["nearest"]}], [101], study.config(), "validity_rl")
        self.assertTrue(row["runtime_valid"])
        self.assertEqual(row["reward"], 0)
        self.assertGreater(sum(m["replay"]["metrics"]["failed_actions"][m["candidate_seat"]] for m in row["matches"]), 0)

    def test_rloo_does_not_use_own_reward_in_baseline(self):
        rewards = [-1, 0, .2, .4, .1, .3, .5, .6]
        first = study.advantages(rewards)
        rewards[0] = -.5
        second = study.advantages(rewards)
        self.assertAlmostEqual(first[0] - (-1), second[0] - (-.5))
        self.assertEqual(study.advantages([-1] * 8), [0] * 8)

    def test_arm_switch_restores_adam_and_committed_group_never_updates_twice(self):
        with fixture() as (run, engine, policy, pool), patch.object(study, "score", side_effect=varying_score):
            engine.group("game_rl", 1, pool)
            engine.group("validity_rl", 1, pool)
            engine.group("game_rl", 2, pool)
            self.assertEqual((policy.weight, policy.step), (2, 2))
            self.assertIn(("base", 0, 0), policy.loads)
            self.assertIn(("game_rl-round1", 1, 1), policy.loads)
            self.assertEqual(policy.updates, 3)
            samples = len(policy.samples)
            resumed = study.Study(run, policy)
            resumed.group("game_rl", 2, pool)
            self.assertEqual(policy.updates, 3)
            self.assertEqual(len(policy.samples), samples)

    def test_interrupted_uncommitted_update_restarts_from_source(self):
        with fixture() as (run, engine, policy, pool), patch.object(study, "score", side_effect=varying_score):
            policy.fail_save = True
            with self.assertRaisesRegex(RuntimeError, "interrupted checkpoint"):
                engine.group("game_rl", 1, pool)
            self.assertEqual((policy.weight, policy.step), (1, 1))
            self.assertIsNone(study.Study(run, policy).checkpoint("game_rl-round1"))
            resumed = study.Study(run, policy)
            resumed.group("game_rl", 1, pool)
            self.assertEqual((policy.weight, policy.step), (1, 1))
            self.assertEqual(len(policy.samples), 8)
            self.assertEqual(policy.updates, 2)
            self.assertTrue(list((run / "checkpoints").glob(".game_rl-round1.pending-*")))

    def test_zero_variance_retains_eight_draws_and_exactly_copies_checkpoint(self):
        def zero(*args):
            return {**varying_score(*args), "reward": -1., "runtime_valid": False}
        with fixture() as (run, engine, policy, pool), patch.object(study, "score", side_effect=zero):
            report = engine.group("validity_rl", 1, pool)
            self.assertEqual(policy.updates, 0)
            self.assertEqual(len(policy.samples), 8)
            self.assertEqual(report["optimizer_steps"], 0)
            self.assertEqual(report["outcome"], "skipped_zero_variance")
            self.assertEqual(engine.checkpoint("base")["files"], engine.checkpoint("validity_rl-round1")["files"])
            self.assertEqual(len(study.common.load(run / "groups/validity_rl/1/scores.json")["rows"]), 8)

    def test_no_evaluation_until_both_round3_endpoints_and_all_draws_sealed(self):
        with fixture() as (run, engine, policy, pool):
            with self.assertRaisesRegex(ValueError, "round3 endpoints"):
                engine.evaluate()
            for arm in study.ARMS:
                engine.save_checkpoint(arm + "-round3", "base", {"fixture": True}, copy_from=run / "checkpoints/base")
            calls = []
            def checked(*args):
                self.assertEqual(len([k for k in engine.status["artifacts"] if k.startswith("generations/eval-")]), 36)
                self.assertIn("evaluation/sealed.json", engine.status["artifacts"])
                calls.append(1)
                return varying_score(*args)
            with patch.object(study, "score", side_effect=checked):
                engine.evaluate()
            self.assertEqual(len(policy.samples), 36)
            self.assertEqual(len(calls), 72)
            self.assertTrue(engine.status["claims_allowed"])
            self.assertEqual(engine.status["state"], "complete")
            self.assertEqual(len(study.common.load(run / "evaluation/base/compact/holdout.json")["scores"]), 8)
            self.assertEqual(len(study.common.load(run / "evaluation/base/plain/holdout.json")["scores"]), 4)

    def test_modified_committed_artifact_blocks_resume(self):
        with fixture() as (run, engine, policy, pool):
            (run / "opponents.json").write_text('{}')
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                study.Study(run, policy)

    def test_whole_study_wall_budget_does_not_reset_on_resume(self):
        start = datetime(2026, 10, 22, 10, 0, tzinfo=timezone.utc)
        budget, remaining = study.execution_budget(study.config(), at=start)
        self.assertEqual(remaining, 13500)
        _, resumed = study.execution_budget(study.config(), budget, start + timedelta(seconds=7200))
        self.assertEqual(resumed, 6300)
        with self.assertRaises(TimeoutError):
            study.execution_budget(study.config(), budget, start + timedelta(seconds=13500))
        late = datetime(2026, 10, 22, 14, 29, tzinfo=timezone.utc)
        _, remaining = study.execution_budget(study.config(), at=late)
        self.assertEqual(remaining, 60)

    def test_worker_checks_lock_inode_and_exclusive_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(study, "LOCK", root / "global.lock"):
                with study.common.process_lock(study.LOCK) as first, study.common.process_lock(root / "run.lock") as second:
                    study.verify_lock_descriptors(root, f"{first},{second}")
                    with self.assertRaisesRegex(ValueError, "expected lock file"):
                        study.verify_lock_descriptors(root, f"{second},{first}")
                    with study.LOCK.open('a+') as other:
                        with self.assertRaises(BlockingIOError):
                            study.verify_lock_descriptors(root, f"{other.fileno()},{second}")

    def test_global_lock_and_hard_memory_stop_preserve_incomplete_status(self):
        class Child:
            pid, returncode = 999999, None
            def poll(self): return None
            def wait(self, timeout): self.returncode = -15; return -15
        with fixture() as (run, engine, policy, pool):
            child = Child()
            usage = type("Usage", (), {"returncode": 0, "stdout": str(30 * 1024**2)})()
            with patch.object(study, "LOCK", run / "global.lock"), \
                 patch.object(study.local, "duration_limit", return_value=13500), \
                 patch.object(study.subprocess, "Popen", return_value=child) as spawn, \
                 patch.object(study.subprocess, "run", return_value=usage), \
                 patch.object(study.os, "killpg") as kill:
                with self.assertRaises(MemoryError):
                    study.launch(run, run / "model")
                kill.assert_called_once_with(child.pid, study.signal.SIGTERM)
                self.assertEqual(len(spawn.call_args.kwargs["pass_fds"]), 2)
            status = study.common.load(run / "status.json")
            self.assertEqual(status["state"], "stopped")
            self.assertFalse(status["claims_allowed"])
            lock = run / "exclusive.lock"
            with study.common.process_lock(lock):
                with self.assertRaises(RuntimeError):
                    with study.common.process_lock(lock):
                        self.fail("Duplicate lock accepted")


if __name__ == "__main__":
    unittest.main()
