import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from selfplay.feedback import main, feedback_prompt


class FeedbackTests(unittest.TestCase):
    def test_feedback_control_withholds_outcomes(self):
        history = [{"successes": 1, "attempts": 2, "frontier_reward": 1.0}]
        control = feedback_prompt(1, [], history, "none")
        treatment = feedback_prompt(1, [], history, "outcomes")
        self.assertNotIn('"successes"', control)
        self.assertIn('"successes": 1', treatment)
        self.assertEqual(control.split("Past feedback:")[0], treatment.split("Past feedback:")[0])

    def test_verified_feedback_reaches_next_proposal(self):
        prompts = []

        class ScriptedBackend:
            metadata = {"model": "test", "backend": "mlx", "thinking": True}

            def __init__(self, *args, **kwargs):
                self.proposal_count = 0

            def complete(self, prompt, seed, max_tokens, temperature):
                if prompt.startswith("Create a planning puzzle"):
                    prompts.append(prompt)
                    value = {"modulus": 7, "start": [self.proposal_count, 0],
                             "actions": ["inc_x", "inc_y", "sum_y", "swap"],
                             "max_steps": 2, "witness": ["inc_x", "inc_x"]}
                    self.proposal_count += 1
                else:
                    value = {"plan": ["tool_0", "tool_0"]}
                return {"text": "<think>test</think>" + json.dumps(value),
                        "finish_reason": "stop", "prompt_tokens": 1, "completion_tokens": 1}

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            argv = ["feedback", "--out", str(out), "--revision", "test", "--proposals", "2"]
            with patch("selfplay.feedback.MLXBackend", ScriptedBackend), patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                main()
            result = json.loads((out / "summary.json").read_text())
            self.assertTrue(result["complete"])
            self.assertEqual(result["accepted_tasks"], 2)
            self.assertEqual(result["solver_successes"], 4)
            self.assertEqual(result["model_calls"], 6)
            self.assertIn('"successes": 2', prompts[1])
            self.assertNotIn('"successes": 2', prompts[0])
            self.assertFalse(result["training"])


if __name__ == "__main__":
    unittest.main()
