import json
import math
import importlib.util
from pathlib import Path
import tempfile
import unittest

from selfplay.curriculum import load_run
from selfplay.train import DirectDataset, completion_loss


class TokenizerStub:
    eos_token_id = 256

    def apply_chat_template(self, messages, **kwargs):
        if kwargs.get("enable_thinking") is not False:
            raise AssertionError("Thinking mode changed")
        return messages[0]["content"] + "<think></think>"

    def encode(self, text, **kwargs):
        return list(text.encode())


class TrainingProtocolTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("mlx") is not None, "Optional MLX runtime")
    def test_loss_masks_prompt_and_padding_but_keeps_eos(self):
        import mlx.core as mx
        # Last two zeros are padding; first two positions are prompt tokens.
        batch = mx.array([[1, 2, 3, 4, 0, 0]])
        lengths = mx.array([[2, 4]])
        def model(inputs):
            return mx.zeros((*inputs.shape, 5))
        loss, count = completion_loss(model, batch, lengths)
        self.assertEqual(count.item(), 2)
        self.assertAlmostEqual(loss.item(), math.log(5), places=5)

    def test_supervised_tokens_exclude_prompt_and_keep_end_token(self):
        row = {"prompt": "puzzle", "answer": '{"plan":["tool_2"]}'}
        ds = DirectDataset([row], TokenizerStub())
        tokens, offset = ds.process(row)
        self.assertEqual(bytes(tokens[:offset]).decode(), "puzzle<think></think>")
        self.assertEqual(bytes(tokens[offset:-1]).decode(), row["answer"])
        self.assertEqual(tokens[-1], 256)

    def test_oracle_rollouts_cannot_be_self_training_data(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            (path / "manifest.json").write_text(json.dumps({"model": {"backend": "oracle"}}))
            with self.assertRaisesRegex(ValueError, "base-model"):
                load_run(path)


if __name__ == "__main__":
    unittest.main()
