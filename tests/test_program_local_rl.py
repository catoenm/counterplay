"""CPU-only local controller smoke admission, provenance and watchdog tests."""
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from selfplay import program_local_rl as local
from scripts.cloud import train_program_rl as common


class Tokenizer:
    def __init__(self, count):
        self.count = count

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "exact template:" + messages[0]["content"]

    def encode(self, text, **kwargs):
        assert kwargs["add_special_tokens"] is False
        return list(range(self.count))


class Child:
    pid = 999999
    returncode = None

    def poll(self):
        return None

    def wait(self, timeout):
        self.returncode = -15
        return -15


class LocalProgramRLTests(unittest.TestCase):
    def prepare(self, folder):
        prompt = Path(folder) / "prompt.txt"
        prompt.write_bytes(local.PLAIN_PROMPT.read_bytes())
        run = Path(folder) / "run"
        local.prepare(run, prompt)
        return run

    def test_prompt_preflight_rejects_oversize_without_truncation(self):
        config = {**local.OVERRIDES, **local.CONDITIONS["feedback-exact-r1-v1"]}
        oversized = local.prompt_measurement(Tokenizer(11988), "unchanged", config)
        self.assertFalse(oversized["admitted"])
        self.assertEqual(oversized["prompt_tokens"], 11988)
        self.assertFalse(oversized["truncation"])
        self.assertTrue(local.prompt_measurement(Tokenizer(7168), "unchanged", config)["admitted"])
        self.assertFalse(local.prompt_measurement(Tokenizer(7169), "unchanged", config)["admitted"])

    def test_local_manifest_freezes_runner_and_installed_source(self):
        with tempfile.TemporaryDirectory() as folder:
            run = self.prepare(folder)
            manifest, _ = common.verify_sources(run)
            self.assertIn("selfplay/program_local_rl.py", manifest["source_sha256"])
            self.assertIn("mlx_lm/models/gated_delta.py", manifest["library_source_sha256"])
            self.assertEqual(manifest["model_weight_sha256"], local.WEIGHT_HASHES)
            self.assertEqual(manifest["config"]["lora_last_layers"], 1)
            self.assertEqual(manifest["config"]["condition"], "plain-v1")

    def test_plain_condition_cannot_hide_changed_specification(self):
        with tempfile.TemporaryDirectory() as folder:
            prompt = Path(folder) / "prompt.txt"
            prompt.write_text("A different task")
            with self.assertRaisesRegex(ValueError, "original specification"):
                local.prepare(Path(folder) / "run", prompt)

    def test_authoritative_metadata_git_blob_hash_includes_length_prefix(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_bytes(b'{"version":1}')
            data = path.read_bytes()
            identity = ("git-sha1", len(data), common.hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest())
            local.verify_identity_bytes(path, identity)
            path.write_bytes(b'{"version":2}')
            with self.assertRaisesRegex(ValueError, "Authoritative metadata hash"):
                local.verify_identity_bytes(path, identity)

    def test_deadline_and_wall_time_bound_are_separate(self):
        config = {"deadline": "2099-01-01T00:00:00+00:00", "worker_seconds": 1800}
        self.assertEqual(local.duration_limit(config, datetime(2098, 1, 1, tzinfo=timezone.utc)), 1800)
        self.assertEqual(local.duration_limit(config, datetime(2098, 12, 31, 23, 59, tzinfo=timezone.utc)), 60)
        with self.assertRaises(TimeoutError):
            local.duration_limit(config, datetime(2099, 1, 1, tzinfo=timezone.utc))

    def test_compact_context_has_all_fixed_practice_rows_and_no_successful_answer(self):
        text, context = local.compact_context()
        self.assertEqual(len(context["facts"]["practice_games"]), 24)
        self.assertEqual([t["seat"] for t in context["facts"]["traces"]], [0, 1])
        for trace in context["facts"]["traces"]:
            self.assertEqual(trace["seed"], 730000)
            self.assertEqual(trace["opponent"], "nearest")
            self.assertEqual([row[0] for row in trace["rows"]], list(range(8)) + list(range(40, 48)))
        self.assertNotIn("failed_turn_groups", text)
        self.assertFalse(context["successful_response_included"])
        self.assertFalse(context["algorithm_hints_added"])
        self.assertTrue(all("generations/" not in name for name in context["input_sha256"]))
        self.assertEqual(context["prompt_sha256"], common.hashlib.sha256(text.encode()).hexdigest())

    def test_compact_prompt_cannot_be_silently_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "compact.txt"
            local.write_compact_prompt(path)
            before = path.read_bytes()
            local.write_compact_prompt(path)
            self.assertEqual(before, path.read_bytes())
            path.write_text("altered")
            with self.assertRaisesRegex(ValueError, "replace"):
                local.write_compact_prompt(path)

    def test_parent_memory_watchdog_terminates_child_and_persists_stop(self):
        with tempfile.TemporaryDirectory() as folder:
            run = self.prepare(folder)
            child = Child()
            usage = type("Usage", (), {"returncode": 0, "stdout": str(30 * 1024 * 1024)})()
            with patch.object(local, "LOCK", Path(folder) / "global.lock"), \
                 patch.object(local, "duration_limit", return_value=1800), \
                 patch.object(local.subprocess, "Popen", return_value=child) as spawn, \
                 patch.object(local.subprocess, "run", return_value=usage), \
                 patch.object(local.os, "killpg") as kill:
                with self.assertRaisesRegex(MemoryError, "Resident"):
                    local.launch(run, Path(folder) / "staged-model")
                kill.assert_called_once_with(child.pid, local.signal.SIGTERM)
                self.assertEqual(len(spawn.call_args.kwargs["pass_fds"]), 2)
            status = common.load(run / "status.json")
            self.assertEqual(status["state"], "stopped")
            self.assertEqual(status["error_type"], "MemoryError")

    def test_global_neural_lock_rejects_duplicate_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "global.lock"
            with common.process_lock(path):
                with self.assertRaisesRegex(RuntimeError, "process lock"):
                    with common.process_lock(path):
                        self.fail("Duplicate lock should never be acquired")


if __name__ == "__main__":
    unittest.main()
