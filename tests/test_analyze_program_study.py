"""Independent CPU audit fixtures. No model inference, downloads or provider calls."""
import copy
from datetime import datetime, timezone, timedelta
import importlib.util
import math
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from safetensors.numpy import save_file
from safetensors.numpy import load_file

from selfplay.program_arena import compile_program, play, verify_replay
from selfplay.program_python import compile_python_program, PythonProgramValidationError
from scripts.program_response_normalization import normalize_final

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("independent_program_study_audit", ROOT / "scripts/analyze_program_study.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)

REST = "def act():\n    return {'action': 'rest'}\n"
FAULT = "def act():\n    return {'action': 'move', 'dx': 0, 'dy': 0}\n"
FAILED_INTENT = "def act():\n    return {'action': 'harvest'}\n"


def match(candidate=REST, opponent=REST, seat=0, seed=6100100):
    programs = [compile_python_program(candidate).compiled, compile_python_program(opponent).compiled]
    if seat:
        programs.reverse()
    return {"opponent_id": "fixture", "candidate_seat": seat,
            "replay": play(programs, seed, 48, origins=["model", "model"])}


def row(name, matches, valid=True, arm="game_rl"):
    return {"generation": name, "matches": matches, **a.score_from_matches(matches, valid, arm)}


def generation():
    return {"name": "fixture", "seed": 7, "checkpoint": "base", "model_repo": "fixture",
            "model_revision": "0"*40, "prompt_sha256": "prompt", "prompt_ids": [1, 2],
            "completion_ids": [3, 4], "sampled_token_logprobs": [-1., -2.],
            "input_tokens": 2, "output_tokens": 2, "finish_reason": "eos", "eos_token_ids": [4]}


CONFIG = {"max_new_tokens": 8, "max_context_tokens": 16, "model_repo": "fixture", "model_revision": "0"*40,
          "token_logprob_tolerance": .002, "sequence_logprob_tolerance": .02}


class ProgramStudyAuditTests(unittest.TestCase):
    def test_durable_execution_budget_and_incomplete_claims(self):
        cfg = {"deadline": "2026-10-22T07:30:00-07:00"}
        budget = {"first_started": "2026-10-22T10:00:00+00:00", "hard_stop": "2026-10-22T13:45:00+00:00",
                  "maximum_seconds": 13500, "resume_counts_wall_time": True}
        status = {"state": "running", "execution_budget": budget}
        a.validate_execution_budget(cfg, status, {"execution_budget": budget})
        bad = copy.deepcopy(status); bad["execution_budget"]["hard_stop"] = cfg["deadline"]
        with self.assertRaisesRegex(ValueError, "renewed"):
            a.validate_execution_budget(cfg, bad)
        status.update(state="complete", updated="2026-10-22T13:46:00+00:00")
        with self.assertRaisesRegex(ValueError, "after hard deadline"):
            a.validate_execution_budget(cfg, status)

    def test_actual_saved_backend_runtime_schema(self):
        parent = ROOT/"runs/program-local-27b-compact-v1"
        runtime = a.read(parent/"runtime.json")
        manifest = a.read(parent/"manifest.json")
        index = a.read(ROOT/"models/qwen38-27b-4bit/model-index.json")
        manifest["model_index_sha256"] = a.sha(ROOT/"models/qwen38-27b-4bit/model-index.json")
        a.validate_runtime(runtime, index, manifest)
        bad = copy.deepcopy(runtime); bad["model_index"].pop("index_sha256")
        with self.assertRaisesRegex(ValueError, "model index"):
            a.validate_runtime(bad, index, manifest)

    def test_reward_only_candidate_faults_and_no_penalty_for_failed_intents(self):
        for seat in (0, 1):
            rival_fault = row("healthy", [match(REST, FAULT, seat)])
            self.assertTrue(rival_fault["runtime_valid"])
            self.assertEqual(rival_fault["candidate_program_faults"], 0)
            self.assertEqual(rival_fault["opponent_program_faults"], 48)
            self.assertEqual(rival_fault["reward"], 0.)
            own_fault = row("fault", [match(FAULT, REST, seat)])
            self.assertFalse(own_fault["runtime_valid"])
            self.assertEqual(own_fault["reward"], -1.)
            failed = row("legal_failed", [match(FAILED_INTENT, REST, seat)])
            self.assertTrue(failed["runtime_valid"])
            self.assertEqual(failed["reward"], 0.)
            self.assertEqual(a.row_metrics(failed)["failed_actions"], 48)

    def test_actual_game_reward_is_distinct_from_validity_only(self):
        from selfplay.program_arena.examples import reference_programs
        controller = compile_program(reference_programs()["nearest"])
        idle = compile_python_program(REST).compiled
        m = {"opponent_id": "rest", "candidate_seat": 0,
             "replay": play([controller, idle], 6100100, 48, origins=["model", "model"])}
        game = a.score_from_matches([m], True, "game_rl")
        validity = a.score_from_matches([m], True, "validity_rl")
        self.assertGreater(game["reward"], 0)
        self.assertEqual(validity["reward"], 0)
        self.assertEqual(game["game_return"], validity["game_return"])
        self.assertEqual(game["banked_point_yield"], validity["banked_point_yield"])
        self.assertGreater(game["banked_point_yield"], 0)

    def test_invalid_draws_stay_in_cluster_denominator_missing_draws_are_not_zero_games(self):
        rows = [row("invalid"+str(i), [], False) for i in range(7)]
        good = row("rest", [match()])
        good["banked_point_yield"] = 8.0  # Explicit metric fixture, not a reported game.
        summary = a.summarize_programs(rows+[good], 8)
        self.assertEqual(summary["primary_mean_banked_point_yield"], 1.)
        self.assertEqual(summary["prescribed_program_clusters"], 8)
        self.assertEqual(summary["invalid_no_game_programs"], 7)
        self.assertEqual(summary["played_games"], 1)
        self.assertEqual(summary["all_rest_programs"], 1)
        partial = a.summarize_programs(rows, 8)
        self.assertFalse(partial["complete"])
        self.assertIsNone(partial["primary_mean_banked_point_yield"])
        self.assertEqual(partial["missing_program_clusters"], 1)
        with self.assertRaisesRegex(ValueError, "Duplicate authoring"):
            a.summarize_programs([good, good], 8)
        with self.assertRaisesRegex(ValueError, "fabricated"):
            a.score_from_matches([match()], False, "game_rl")

    def test_rloo_baseline_excludes_own_reward_and_zero_variance(self):
        self.assertEqual(a.advantages([-1.]*8), [0.]*8)
        rewards = [-1., -.8, -.6, -.4, -.2, 0., .2, .4]
        adv = a.advantages(rewards)
        self.assertAlmostEqual(sum(adv), 0.)
        for i in range(8):
            self.assertAlmostEqual(adv[i], rewards[i]-math.fsum(rewards[:i]+rewards[i+1:])/7)
        changed = list(rewards); changed[0] += .1
        self.assertAlmostEqual(a.advantages(changed)[0]-adv[0], .1)
        self.assertAlmostEqual(a.advantages(changed)[1]-adv[1], -.1/7)
        with self.assertRaises(ValueError): a.advantages([0.]*7)

    def test_generation_actual_eos_and_gate_identity_coverage(self):
        record = generation()
        kwargs = dict(name="fixture", seed=7, checkpoint="base", prompt_sha256="prompt", config=CONFIG)
        a.validate_generation(record, **kwargs)
        bad = copy.deepcopy(record); bad["completion_ids"][-1] = 5
        with self.assertRaisesRegex(ValueError, "EOS"): a.validate_generation(bad, **kwargs)
        bad = copy.deepcopy(record); bad["sampled_token_logprobs"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "probability"): a.validate_generation(bad, **kwargs)
        gate = {"passed": True, "generation_names": ["fixture"], "records": [{"passed": True, "tokens": 2,
                "max_token_error": .001, "sequence_error": .002, "recorded_sequence_logprob": -3.}]}
        a.validate_gate(gate, [record], ["fixture"], CONFIG)
        for field, value in (("tokens", 1), ("max_token_error", .003), ("recorded_sequence_logprob", -2.)):
            bad = copy.deepcopy(gate); bad["records"][0][field] = value
            with self.assertRaises(ValueError): a.validate_gate(bad, [record], ["fixture"], CONFIG)

    def test_map_and_model_ir_overlap_guards(self):
        a.check_fingerprint_overlap({"train": ["a"], "holdout": ["b"]}, ["prior"])
        with self.assertRaisesRegex(ValueError, "overlap"):
            a.check_fingerprint_overlap({"train": ["same"], "holdout": ["same"]}, [])
        with self.assertRaisesRegex(ValueError, "overlap"):
            a.check_fingerprint_overlap({"train": ["prior"]}, ["prior"])
        anchor = {"origin": "scripted", "ir_sha256": "shared"}
        a.validate_ir_isolation([anchor], [anchor])
        controller = {"origin": "model", "ir_sha256": "same"}
        with self.assertRaisesRegex(ValueError, "leaked"):
            a.validate_ir_isolation([controller], [controller])

    def test_safetensor_optimizer_lineage_and_zero_variance_skip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent, child = root/"base", root/"step"
            parent.mkdir(); child.mkdir()
            save_file({"weight": np.zeros((2, 2), np.float32)}, str(parent/"adapter.safetensors"))
            save_file({"step": np.array(0, np.uint64)}, str(parent/"optimizer.safetensors"))
            for name in ("adapter.safetensors", "optimizer.safetensors"):
                shutil.copyfile(parent/name, child/name)
            metadata = {"files": {p.name: a.sha(p) for p in child.iterdir()},
                        "report": {"outcome": "skipped_zero_variance"}}
            result = a.validate_checkpoint_tensors(child, metadata, parent, 0)
            self.assertEqual(result["adapter_l2_delta"], 0.)
            save_file({"step": np.array(1, np.uint64)}, str(child/"optimizer.safetensors"))
            metadata["files"]["optimizer.safetensors"] = a.sha(child/"optimizer.safetensors")
            with self.assertRaisesRegex(ValueError, "Skipped group changed"):
                a.validate_checkpoint_tensors(child, metadata, parent, 1)
            save_file({"weight": np.ones((2, 2), np.float32)}, str(child/"adapter.safetensors"))
            metadata["files"]["adapter.safetensors"] = a.sha(child/"adapter.safetensors")
            metadata["report"] = {"outcome": "updated", "adapter_l2_delta": 2.}
            self.assertEqual(a.validate_checkpoint_tensors(child, metadata, parent, 1)["changed_parameters"], 4)
            with self.assertRaisesRegex(ValueError, "step lineage"):
                a.validate_checkpoint_tensors(child, metadata, parent, 2)

    def test_archived_runtime_rejects_source_changes_and_replays_game(self):
        sources = ["selfplay/program_arena/"+n+".py" for n in ("__init__", "dsl", "game", "runner", "spec", "examples")]
        sources += ["selfplay/program_python.py", "scripts/program_response_normalization.py"]
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            hashes = {}
            for relative in sources:
                target = run/"source"/relative; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT/relative, target); hashes[relative] = a.sha(target)
            with a.archived_runtime(run, {"source_sha256": hashes}) as runtime:
                expected = match()["replay"]
                self.assertTrue(runtime.verify_replay(expected))
                self.assertIsNotNone(a.classify({"text": REST, "finish_reason": "eos"}, runtime)[1])
                altered = copy.deepcopy(expected); altered["metrics"]["scores"][0] = 100
                with self.assertRaises(ValueError): runtime.verify_replay(altered)
            (run/"source/selfplay/program_arena/game.py").write_text("changed")
            with self.assertRaisesRegex(ValueError, "source changed"):
                with a.archived_runtime(run, {"source_sha256": hashes}): pass

    def test_full_cpu_fixture_exact_skips_and_invalid_evaluation_denominators(self):
        """Real archived game/tensors, synthetic authoring tokens: never a model result."""
        from selfplay import program_study as study
        from unittest.mock import patch
        class FixturePolicy:
            def load_checkpoint(self, folder, hashes):
                for name, expected in hashes.items():
                    self_hash = a.sha(folder/name)
                    if self_hash != expected: raise ValueError("fixture checkpoint hash")
                self.arrays = {name: load_file(folder/name) for name in hashes}
            def save_checkpoint(self, folder):
                folder.mkdir(parents=True)
                for name, arrays in self.arrays.items():
                    save_file(arrays, str(folder/name))
                return {name: a.sha(folder/name) for name in self.arrays}
            def likelihood_gate(self, records):
                return {"passed": True, "records": [{"tokens": 1, "passed": True, "max_token_error": 0.,
                        "sequence_error": 0., "recorded_sequence_logprob": -.5} for _ in records]}
            def update(self, records, advantages):
                adapter = self.arrays["adapter.safetensors"]
                key = sorted(adapter)[0]
                before = adapter[key].copy()
                adapter[key].flat[0] += .001
                delta = float(abs(float(adapter[key].flat[0])-float(before.flat[0])))
                self.arrays["optimizer.safetensors"]["step"] += np.array(1, np.uint64)
                return {"optimizer_steps": 1, "adapter_l2_delta": delta, "loss": 1.,
                        "gradient_norm_before_clip": 1., "microbatch_losses": [.125]*8,
                        "gradient_likelihood_gate": self.likelihood_gate(records), "peak_memory_bytes": 1}
            def sample(self, prompt, seed):
                if 8200000 <= seed < 8200008:
                    previous = study.common.load(study.PARENT/"generations/opponent-1-1.json")["text"]
                    text = previous.replace("def act():", f"def act():\n    m15 = {seed-8199999}", 1)
                elif seed >= 8300000:
                    text = "invalid fixture authoring"
                elif 8100100 <= seed < 8100108 and seed % 2:
                    text = FAULT
                else:
                    text = REST
                return {"text": text, "raw_text": text, "prompt_ids": [1], "completion_ids": [248044],
                        "sampled_token_logprobs": [-.5], "input_tokens": 1, "output_tokens": 1,
                        "finish_reason": "eos", "eos_token_ids": [248044, 248046], "elapsed_seconds": 0.}
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)/"cpu-fixture"
            study.prepare(run, ROOT/"models/qwen38-27b-4bit")
            prepared = a.audit(run)
            self.assertFalse(prepared["planned_study_complete"])
            self.assertEqual(prepared["committed_fresh_authoring_draws"], 0)
            engine = study.Study(run, FixturePolicy())
            at = datetime(2026, 10, 22, 10, 0, tzinfo=timezone.utc)
            # CPU fixture status follows the real protocol, but no neural worker exists.
            budget, _ = study.execution_budget(engine.config, at=at)
            with patch.object(study.common, "admission"), patch.object(study.common, "now", return_value=at.isoformat()):
                engine.state(execution_budget=budget)
                fixture_runtime = study.common.load(study.PARENT/"runtime.json")
                fixture_runtime["execution_budget"] = budget
                engine.commit("runtime.json", fixture_runtime)
                engine.run_all()
            result = a.audit(run)
            self.assertTrue(result["planned_study_complete"])
            self.assertEqual(result["evaluation_draws"], 36)
            self.assertEqual(result["actual_primary_compact_holdout_games"], 0)
            self.assertEqual(result["actual_secondary_plain_holdout_games"], 0)
            self.assertEqual(len(result["training_groups"]), 6)
            self.assertEqual(result["checkpoint_tensors"]["base"]["optimizer_steps"], 0)
            self.assertTrue(all(result["checkpoint_tensors"][endpoint]["optimizer_steps"] == 1 for endpoint in a.ENDPOINTS[1:]))
            self.assertEqual(sum(r["gradient_signal"] == "no_update" for r in result["training_groups"]), 4)
            for condition, n in (("compact", 8), ("plain", 4)):
                for endpoint in a.ENDPOINTS:
                    r = result["evaluation"][f"{endpoint}/{condition}/holdout"]
                    self.assertEqual(r["prescribed_program_clusters"], n)
                    self.assertEqual(r["invalid_no_game_programs"], n)
                    self.assertEqual(r["primary_mean_banked_point_yield"], 0.)
            # Hash-journal mismatch catches even a text-only mutation of a sampled output.
            path = run/"generations/game_rl-round1-0.json"
            value = a.read(path); value["text"] = FAULT
            path.write_text(a.canonical(value))
            with self.assertRaisesRegex(ValueError, "Artifact hash differs"):
                a.audit(run)


if __name__ == "__main__":
    unittest.main()
