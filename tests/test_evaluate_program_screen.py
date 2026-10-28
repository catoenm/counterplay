"""CPU-only evaluator fixtures; these are explicitly not model screen results."""
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts import evaluate_program_screen as screen
from scripts import program_response_normalization as packaging


IDLE = "def act():\n    return {'action':'rest'}\n"
FAULT = "def act():\n    m0 = 1 // 0\n    return {'action':'rest'}\n"


class ScreenEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.run = Path(self.temp.name) / "explicit-test-fixture"
        self.run.mkdir()
        (self.run / "source").mkdir()
        hashes = {}
        for relative in screen.SOURCE_FILES:
            data = (screen.ROOT / relative).read_bytes()
            (self.run / "source" / Path(relative).name).write_bytes(data)
            hashes[relative] = screen.digest(data)
        prompt = "Explicit synthetic unit-test fixture; not a generated model policy."
        (self.run / "prompt.txt").write_text(prompt)
        self.plan = {"samples": 2, "validation_seeds": list(range(6)), "fixed_opponents": ["nearest", "value"],
                     "paired_seats": True, "peer_crossplay": True, "max_rounds": 2, "training": False,
                     "model": "explicit-test-fixture", "reasoning": "off", "source_sha256": hashes,
                     "prompt_sha256": screen.digest(prompt.encode())}
        self.write_plan()
        for number, source in enumerate((IDLE, FAULT), 1):
            self.write_response(number, source)

    def tearDown(self):
        self.temp.cleanup()

    def write_plan(self):
        (self.run / "plan.json").write_bytes(screen.encoded(self.plan))

    def write_response(self, number, text, finish="stop"):
        value = {"sample_id": f"explicit-fixture-{number}", "text": text, "finish_reason": finish,
                 "reasoning": "def act():\n    return {'action':'harvest'}\n",  # Never consumed.
                 "prompt": (self.run / "prompt.txt").read_text(), "prompt_sha256": self.plan["prompt_sha256"],
                 "client_source_sha256": self.plan["source_sha256"]["selfplay/program_api.py"],
                 "returned_model": self.plan["model"], "returned_provider": "fixture", "reasoning_mode": "off",
                 "cost_usd": 0.01, "cost_upper_usd": 0.02, "usage": {"completion_tokens": 10}}
        (self.run / f"generation-{number}.json").write_bytes(screen.encoded(value))

    def test_final_only_failure_classes_and_fence_normalization(self):
        with screen.archived_runtime(self.run, self.plan) as runtime:
            for text, finish, expected in (("", "stop", "missingfinal"), ("", "length", "truncated"),
                                           (IDLE, "length", "truncated"), ("def act(:", "stop", "invalidsyntax"),
                                           ("Here is code:\n```python\n" + IDLE + "```", "stop", "invalidsyntax")):
                entry, program = screen.classify_generation({"sample_id": "fixture", "text": text,
                    "finish_reason": finish, "reasoning": IDLE}, runtime)
                self.assertEqual(entry["failure"], expected)
                self.assertIsNone(program)
                self.assertEqual(entry["original_text"], text)
            wrapped = "\n```python\n" + IDLE + "```\n"
            entry, program = screen.classify_generation({"sample_id": "fixture", "text": wrapped, "finish_reason": "stop"}, runtime)
            self.assertTrue(entry["valid"])
            self.assertEqual(entry["original_text"], wrapped)
            self.assertEqual(program.source, IDLE)
            self.assertEqual(entry["normalization"], "stripped_single_wrapping_python_fence")
            self.assertEqual(screen.normalize_final(IDLE), (IDLE, "none"))

    def test_archive_ignores_live_frontend_and_rejects_hash_changes(self):
        with patch("selfplay.program_python.compile_python_program", side_effect=AssertionError("live frontend used")):
            with screen.archived_runtime(self.run, self.plan) as runtime:
                self.assertEqual(runtime.compile_python_program(IDLE).source, IDLE)
                self.assertTrue(runtime.model_spec())
        path = self.run / "source/program_python.py"
        path.write_text(path.read_text() + "\n# changed\n")
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            screen.evaluate(self.run)
        self.assertFalse((self.run / "evaluation.json").exists())

    def test_schedule_provenance_faults_and_idempotent_completion(self):
        result = screen.evaluate(self.run)
        self.assertEqual((result["valid_programs"], result["model_matches"]), (2, 60))
        self.assertEqual((result["fixed_reference_matches"], result["peer_crossplay_matches"]), (48, 12))
        wrappers = [json.loads(line) for line in (self.run / "matches.jsonl").read_text().splitlines()]
        fault = result["players"]["explicit-fixture-2"]["overall"]
        self.assertEqual(fault["games"], 36)
        self.assertEqual(fault["program_faults"], 72)
        self.assertEqual(sum(fault["fault_types"].values()), 72)
        self.assertEqual(fault["draws"], 36)
        for record in wrappers:
            self.assertEqual(len(record["python_programs"]), 2 if record["match_kind"] == "peer_crossplay" else 1)
            for name, provenance in record["python_programs"].items():
                seat = record["players"].index(name)
                self.assertEqual(provenance["ir_sha256"], record["record"]["programs"][seat]["digest"])
                self.assertEqual(provenance["python_sha256"], screen.digest(provenance["python_source"].encode()))
        before = {p.name: p.read_bytes() for p in (self.run / "matches.jsonl", self.run / "evaluation.json", self.run / "summary.json")}
        self.assertEqual(result, screen.evaluate(self.run))
        self.assertEqual(before, {self.run.joinpath(name).name: self.run.joinpath(name).read_bytes() for name in before})

    def test_resume_complete_prefix_without_duplicating_matches(self):
        original = screen.archived_runtime
        calls = 0

        @contextmanager
        def fail_after_two(run, plan):
            with original(run, plan) as runtime:
                play = runtime.play

                def intermittent(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 3:
                        raise RuntimeError("simulated interruption")
                    return play(*args, **kwargs)
                runtime.play = intermittent
                yield runtime

        with patch.object(screen, "archived_runtime", fail_after_two), self.assertRaisesRegex(RuntimeError, "interruption"):
            screen.evaluate(self.run)
        first = (self.run / "matches.jsonl").read_bytes()
        self.assertEqual(len(first.splitlines()), 2)
        self.assertFalse((self.run / "summary.json").exists())
        self.assertEqual(screen.evaluate(self.run)["model_matches"], 60)
        self.assertTrue((self.run / "matches.jsonl").read_bytes().startswith(first))

    def test_invalid_programs_have_no_fabricated_matches(self):
        self.write_response(1, "", "length")
        self.write_response(2, "prose\n```python\n" + IDLE + "```")
        result = screen.evaluate(self.run)
        self.assertEqual((result["valid_programs"], result["model_matches"]), (0, 0))
        self.assertEqual(result["failures"], {"truncated": 1, "invalidsyntax": 1})
        self.assertEqual((self.run / "matches.jsonl").read_bytes(), b"")
        self.assertTrue(result["complete"])

    def test_partial_or_changed_artifacts_are_retained_and_rejected(self):
        screen.evaluate(self.run)
        matches = self.run / "matches.jsonl"
        complete = matches.read_bytes()
        matches.write_bytes(complete + b'{"partial":')
        with self.assertRaisesRegex(ValueError, "partial final"):
            screen.evaluate(self.run)
        self.assertEqual(matches.read_bytes(), complete + b'{"partial":')
        matches.write_bytes(complete)
        self.write_response(1, IDLE + "\n")
        with self.assertRaisesRegex(ValueError, "incompatible existing artifact"):
            screen.evaluate(self.run)
        self.assertEqual(matches.read_bytes(), complete)

    def test_both_generation_files_required_and_prompt_integrity(self):
        (self.run / "generation-2.json").unlink()
        with self.assertRaises(FileNotFoundError):
            screen.evaluate(self.run)
        self.assertFalse((self.run / "evaluation.json").exists())
        self.write_response(2, IDLE)
        path = self.run / "generation-1.json"
        data = screen.read_json(path)
        data["prompt"] += "mutated"
        path.write_bytes(screen.encoded(data))
        with self.assertRaisesRegex(ValueError, "prompt contents"):
            screen.evaluate(self.run)

    def test_local_generation_requires_exact_client_revision_and_completed_final(self):
        client = "selfplay/program_local.py"
        data = (screen.ROOT / client).read_bytes()
        (self.run / "source/program_local.py").write_bytes(data)
        del self.plan["source_sha256"]["selfplay/program_api.py"]
        self.plan["source_sha256"][client] = screen.digest(data)
        self.plan.update(generation_backend="local_mlx", revision="a" * 40,
                         sampling={"temperature": .6, "top_p": .95, "top_k": 20})
        normalizer = Path(packaging.__file__).read_bytes()
        (self.run / "source/response_normalization.py").write_bytes(normalizer)
        self.plan["response_normalization"] = {
            "version": 2, "mode": "single_python_fence_with_optional_prose",
            "source_archive": "source/response_normalization.py",
            "source_sha256": screen.digest(normalizer)}
        self.write_plan()
        for number in (1, 2):
            path = self.run / f"generation-{number}.json"
            value = screen.read_json(path)
            value.update(client_source_sha256=screen.digest(data), model=self.plan["model"],
                         revision=self.plan["revision"], backend="mlx", adapter_path=None,
                         training=False, thinking=False, external_spend_usd=0,
                         sampling=self.plan["sampling"], usable_final=True, extraction_status="complete")
            value.pop("cost_usd"); value.pop("cost_upper_usd")
            path.write_bytes(screen.encoded(value))
        result = screen.evaluate(self.run)
        self.assertEqual(result["valid_programs"], 2)
        self.assertEqual(result["costs"]["new_generation_cost_usd"], 0)
        self.assertFalse(result["posthoc_diagnostic"])
        self.assertEqual(result["generations"][0]["returned_model"], self.plan["model"])
        path = self.run / "generation-1.json"
        value = screen.read_json(path)
        value["revision"] = "b" * 40
        path.write_bytes(screen.encoded(value))
        with self.assertRaisesRegex(ValueError, "local generation provenance"):
            screen.load_generations(self.run, self.plan)
        value["revision"] = self.plan["revision"]
        value["usable_final"] = False
        path.write_bytes(screen.encoded(value))
        with self.assertRaisesRegex(ValueError, "final answer is incomplete"):
            screen.load_generations(self.run, self.plan)

    def test_reuse_keeps_original_costs_without_new_api_spend(self):
        original = Path(self.temp.name) / "original-fixture"
        shutil.copytree(self.run, original)
        hashes = {f"generation-{i}.json": screen.digest((original / f"generation-{i}.json").read_bytes()) for i in (1, 2)}
        self.plan["generation_reuse"] = {"source_run": str(original), "source_generation_sha256": hashes, "reason": "explicit test fixture"}
        self.write_plan()
        result = screen.evaluate(self.run)
        self.assertEqual(result["costs"]["new_generation_cost_usd"], 0)
        self.assertEqual(result["costs"]["original_response_cost_usd"], .02)
        self.assertTrue(result["costs"]["reused_generation_responses"])
        (original / "generation-1.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "differs from its original"):
            screen.evaluate(self.run)

    def test_model_gateway_profile_and_provider_are_bound_to_archive(self):
        client = "selfplay/program_model_screen.py"
        data = (screen.ROOT / client).read_bytes()
        (self.run / "source/program_model_screen.py").write_bytes(data)
        del self.plan["source_sha256"]["selfplay/program_api.py"]
        self.plan["source_sha256"][client] = screen.digest(data)
        profile = {"name": "fixture", "model": self.plan["model"], "returned_provider": "fixture"}
        config = {"profiles": {"fixture": {k: v for k, v in profile.items() if k != "name"}}}
        (self.run / "source/model-screen-profiles.json").write_bytes(screen.encoded(config))
        config_hash = screen.digest(json.dumps(config, sort_keys=True, separators=(",", ":")).encode())
        self.plan.update(generation_backend="openrouter_model_screen", generation_client_profile=profile,
                         generation_client_configuration_sha256=config_hash)
        self.write_plan()
        for number in (1, 2):
            path = self.run / f"generation-{number}.json"
            value = screen.read_json(path)
            value.update(client_source_sha256=screen.digest(data), generation_client="program_model_screen",
                         generation_client_profile=profile, generation_client_configuration_sha256=config_hash)
            path.write_bytes(screen.encoded(value))
        self.assertEqual(screen.evaluate(self.run)["valid_programs"], 2)
        path = self.run / "generation-1.json"
        value = screen.read_json(path); value["returned_provider"] = "wrong-provider"
        path.write_bytes(screen.encoded(value))
        with self.assertRaisesRegex(ValueError, "model-screen generation provenance"):
            screen.load_generations(self.run, self.plan)
        (self.run / "source/model-screen-profiles.json").write_text('{}')
        with self.assertRaisesRegex(ValueError, "profile configuration mismatch"):
            with screen.archived_runtime(self.run, self.plan):
                pass

    def test_actual_bank_points_determine_wins_and_losses(self):
        with screen.archived_runtime(self.run, self.plan) as runtime:
            idle = runtime.compile_python_program(IDLE).compiled
            nearest = runtime.compile_program(runtime.reference_programs()["nearest"])
            record = runtime.play([idle, nearest], seed=720000, max_rounds=48,
                                  origins=["model", "scripted"])
            self.assertTrue(runtime.verify_replay(record))
        self.assertEqual(record["metrics"]["scores"][0], 0)
        self.assertGreater(record["metrics"]["scores"][1], 0)
        self.assertEqual(record["returns"][0], -record["metrics"]["scores"][1] / record["metrics"]["total_value"])
        self.assertEqual(screen.aggregate([(record, 0)])["losses"], 1)
        self.assertEqual(screen.aggregate([(record, 1)])["wins"], 1)
        self.assertEqual(screen.aggregate([(record, 0)])["program_faults"], 0)

    def test_packaging_diagnostic_is_versioned_and_keeps_source_unchanged(self):
        content = Path(packaging.__file__).read_bytes()
        (self.run / "source/response_normalization.py").write_bytes(content)
        self.plan["response_normalization"] = {
            "version": 2, "mode": packaging.NORMALIZATION_MODE,
            "source_archive": "source/response_normalization.py", "source_sha256": screen.digest(content)}
        self.plan["interpretation"] = "Post-hoc packaging test fixture, not fresh model evidence."
        self.write_plan()
        response = "A synthetic fixture, not model output.\n```python\n" + IDLE + "```\nFixture explanation.\n"
        self.write_response(1, response)
        original = Path(self.temp.name) / "original-packaging-fixture"
        shutil.copytree(self.run, original)
        self.plan["generation_reuse"] = {
            "source_run": str(original), "reason": "Explicit post-hoc test fixture",
            "source_generation_sha256": {
                f"generation-{i}.json": screen.digest((original / f"generation-{i}.json").read_bytes())
                for i in (1, 2)}}
        self.write_plan()
        result = screen.evaluate(self.run)
        self.assertEqual(result["valid_programs"], 2)
        self.assertEqual(result["model_matches"], 60)
        self.assertTrue(result["posthoc_diagnostic"])
        generation = result["generations"][0]
        self.assertEqual(generation["normalization"], "extracted_single_python_fence_with_prose")
        self.assertEqual(generation["normalization_version"], 2)
        self.assertEqual(generation["original_text"], response)
        self.assertEqual(generation["original_text_sha256"], screen.digest(response.encode()))
        self.assertEqual(generation["compiled_program"]["python_source"], IDLE)
        self.assertEqual(generation["response_sha256"], screen.digest((self.run / "generation-1.json").read_bytes()))
        self.assertEqual(screen.evaluate(self.run), result)
        (self.run / "source/response_normalization.py").write_bytes(content + b"\n# altered")
        with self.assertRaisesRegex(ValueError, "normalization hash mismatch"):
            screen.evaluate(self.run)


class PackagingNormalizationTests(unittest.TestCase):
    def test_extracts_one_python_fence_and_preserves_exact_body(self):
        cases = [
            ("Introduction.\n```python\n" + IDLE + "```\nExplanation.", IDLE, "extracted_single_python_fence_with_prose"),
            ("```python\n" + IDLE + "```", IDLE, "stripped_single_wrapping_python_fence"),
            ("Before\r\n```python\r\ndef act():\r\n    pass\r\n```\r\nAfter", "def act():\r\n    pass\r\n", "extracted_single_python_fence_with_prose"),
            ("Before\n  ```python  \n\n" + IDLE + "\n  ```  \nAfter", "\n" + IDLE + "\n", "extracted_single_python_fence_with_prose"),
            ("```python\n```", "", "stripped_single_wrapping_python_fence"),
        ]
        for text, source, kind in cases:
            with self.subTest(text=text):
                self.assertEqual(packaging.normalize_final(text), (source, kind))

    def test_rejects_ambiguous_or_other_fences_without_extracting(self):
        one = "```python\n" + IDLE + "```"
        cases = [IDLE, "Explanation without code", one + "\n" + one,
                 one + "\n```json\n{}\n```", "~~~python\npass\n~~~\n" + one,
                 "```\n" + IDLE + "```", "```Python\n" + IDLE + "```",
                 "```py\n" + IDLE + "```", "```python\n" + IDLE,
                 "Inline: ```python\n" + IDLE + "```", "````python\n" + IDLE + "````",
                 "```python\n" + IDLE + "``` trailing text", "```python\n# ```\n" + IDLE + "```",
                 "```python\n" + IDLE + "```python\n"]
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(packaging.normalize_final(text), (text, "none"))

    def test_does_not_repair_semantic_or_syntax_errors(self):
        invalid = "def act(:\n    return missing_helper()\n"
        self.assertEqual(packaging.normalize_final("Prose.\n```python\n" + invalid + "```\nProse."),
                         (invalid, "extracted_single_python_fence_with_prose"))


if __name__ == "__main__":
    unittest.main()
