"""CPU-only actual-autograd/Adam fixtures; no pretrained model or inference."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from selfplay import program_cardinal_sft as sft
from selfplay.program_cardinal_cuda import checkpoint_state

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("safetensors") is not None
if HAS_TORCH:
    import torch
    from tests.test_program_cardinal_cuda import fixture


class Tokenizer:
    eos_token, eos_token_id = "<eos>", 0

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "user:" + messages[0]["content"] + "\nassistant:\n"

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        result = []
        while text:
            if text.startswith(self.eos_token):
                result.append(self.eos_token_id); text = text[len(self.eos_token):]
            else:
                result.append(ord(text[0]) + 1); text = text[1:]
        return result


def targets(tokenizer=None):
    tokenizer = tokenizer or Tokenizer()
    return [sft.make_teacher_record(tokenizer, "fixture", "x" * size, [0]) for size in (1, 2, 4, 7)]


class TargetTests(unittest.TestCase):
    def test_actual_tokenizer_eos_and_supervised_provenance(self):
        records = targets()
        self.assertEqual([r["output_tokens"] for r in records], [2, 3, 5, 8])
        for row in records:
            self.assertEqual(row["completion_ids"][-1], Tokenizer.eos_token_id)
            self.assertEqual(row["completion_ids"].count(0), 1)
            self.assertFalse(row["sampled"]); self.assertFalse(row["on_policy"])
            self.assertNotIn("seed", row); self.assertNotIn("sampled_token_logprobs", row)
            self.assertEqual(row["teacher_record_sha256"], sft._json_hash({k: v for k, v in row.items() if k != "teacher_record_sha256"}))

    def test_rejects_cap_context_premature_eos_and_wrong_model_eos(self):
        for text, eos, context in [("x"*700,[0],4096),("x<eos>x",[0],4096),("x",[1],4096),("x",[0],2)]:
            with self.assertRaises(ValueError): sft.make_teacher_record(Tokenizer(), "fixture", text, eos, context)
        self.assertEqual(sft.make_teacher_record(Tokenizer(), "fixture", "x"*699, [0])["output_tokens"], 700)

    def test_tokenization_boundary_change_is_rejected(self):
        class Bad(Tokenizer):
            def encode(self, text, add_special_tokens):
                values = super().encode(text, add_special_tokens)
                return values[:-1] if "assistant:\nx" in text else values
        with self.assertRaisesRegex(ValueError, "prefix"):
            sft.make_teacher_record(Bad(), "fixture", "x", [0])


def policy():
    instance = fixture()
    instance.__class__ = sft.CardinalSftPolicy
    instance.config = {"learning_rate": sft.SFT_LEARNING_RATE, "gradient_clip": .5,
        "max_context_tokens": 4096, "token_logprob_tolerance": .002, "sequence_logprob_tolerance": .02}
    instance.tokenizer, instance.eos = Tokenizer(), [0]
    instance.optimizer = torch.optim.AdamW(instance.parameters.values(), lr=sft.SFT_LEARNING_RATE, weight_decay=0)
    instance._optimizer_groups = instance.optimizer.state_dict()["param_groups"]

    def logps(row):
        # Four tiny q/v LoRA matrices all contribute to a genuine full-softmax
        # objective. Targets of different lengths distinguish the requested
        # per-program token mean from a token-pooled or sequence-sum loss.
        total = sum((i+1)*p.sum() for i,p in enumerate(instance.parameters.values()))
        positions = torch.arange(len(row["completion_ids"]), dtype=torch.float32)
        base = torch.stack([total*.03, -total*.02, total*.01, -total*.04])
        logits = base[None,:] + positions[:,None]*torch.tensor([[.1,-.04,.03,0.]])
        labels = torch.tensor([i % 4 for i in row["completion_ids"]])
        return -torch.nn.functional.cross_entropy(logits, labels, reduction="none")

    instance.token_logps = logps
    return instance


@unittest.skipUnless(HAS_TORCH, "Requires optional CPU Torch; no CUDA required")
class SftTensorTests(unittest.TestCase):
    def test_exact_mean_of_program_means_matches_independent_adam(self):
        actual, expected = policy(), policy()
        batch = targets()
        losses = [-expected.token_logps(row).mean() for row in batch]
        objective = torch.stack(losses).mean()
        self.assertNotAlmostEqual(float(objective.detach()), float(-torch.cat([expected.token_logps(r) for r in batch]).mean().detach()), places=5)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(list(expected.parameters.values()), .5, error_if_nonfinite=True)
        expected.optimizer.step()
        report = actual.sft_update(batch)
        self.assertAlmostEqual(report["loss"], float(objective.detach()), places=6)
        self.assertEqual((report["adam_step_before"], report["adam_step_after"]), (0, 1))
        self.assertEqual(report["total_completion_tokens"], 18)
        self.assertGreater(report["adapter_l2_delta"], 0)
        self.assertLessEqual(report["gradient_norm_after_clip"], .500001)
        self.assertTrue(report["teacher_forced_eval_train_gate"]["passed"])
        for name, parameter in actual.parameters.items():
            self.assertTrue(torch.equal(parameter, expected.parameters[name]))
            self.assertIsNone(parameter.grad)
        self.assertFalse(actual.model.training)

    def test_tampered_record_or_onpolicy_fields_rejected_before_step(self):
        instance = policy()
        for mutate in (
            lambda b: b[0]["completion_ids"].__setitem__(0, 999),
            lambda b: b[0].update(on_policy=True),
            lambda b: b[0].update(on_policy=0),
            lambda b: b[0].update(sampled_token_logprobs=[]),
            lambda b: b.__setitem__(1, deepcopy(b[0])),
        ):
            batch = targets(); mutate(batch)
            with self.assertRaises(ValueError): instance.sft_update(batch)
            self.assertEqual(instance._optimizer_identity()["step"], 0)

    def test_train_mode_likelihood_mismatch_aborts_without_adam(self):
        instance = policy(); original = instance.token_logps
        instance.token_logps = lambda row: original(row) + (.1 if instance.model.training else 0)
        before = instance._adapter_identity()
        with self.assertRaisesRegex(ValueError, "consistency gate"):
            instance.sft_update(targets())
        self.assertEqual(instance._optimizer_identity()["step"], 0)
        self.assertEqual(instance._adapter_identity(), before)
        self.assertFalse(instance.model.training)
        self.assertTrue(all(p.grad is None for p in instance.parameters.values()))

    def test_nonfinite_and_missing_gradients_rejected(self):
        instance = policy()
        parameter = next(iter(instance.parameters.values()))
        hook = parameter.register_hook(lambda gradient: gradient*float("nan"))
        with self.assertRaisesRegex(ValueError, "Nonfinite supervised gradient"):
            instance.sft_update(targets())
        hook.remove()
        self.assertEqual(instance._optimizer_identity()["step"], 0)
        first = next(iter(instance.parameters.values()))
        instance.token_logps = lambda row: first.sum()*torch.ones(len(row["completion_ids"]))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            instance.sft_update(targets())
        self.assertEqual(instance._optimizer_identity()["step"], 0)

    def test_large_gradient_is_clipped_and_all_zero_gradient_aborts(self):
        instance = policy(); original = instance.token_logps
        instance.token_logps = lambda row: original(row) * 100
        report = instance.sft_update(targets())
        self.assertGreater(report["gradient_norm_before_clip"], .5)
        self.assertAlmostEqual(report["gradient_norm_after_clip"], .5, places=5)
        zero = policy(); original_zero = zero.token_logps
        zero.token_logps = lambda row: original_zero(row) * 0
        with self.assertRaisesRegex(ValueError, "zero"):
            zero.sft_update(targets())
        self.assertEqual(zero._optimizer_identity()["step"], 0)

    def test_32_steps_and_explicit_rl_reset_preserve_adapter_files_and_rng(self):
        instance = policy()
        with self.assertRaisesRegex(ValueError, "exactly32"):
            instance.reset_optimizer()
        for _ in range(32): instance.sft_update(targets())
        with self.assertRaisesRegex(ValueError, "budget"):
            instance.sft_update(targets())
        gpu_rng = [torch.get_rng_state().clone()]
        with tempfile.TemporaryDirectory() as temp, patch.object(torch.cuda, "get_rng_state_all", return_value=gpu_rng):
            final, base = Path(temp)/"sft-step32", Path(temp)/"rl-base"
            files_before = instance.save_checkpoint(final)
            previous = checkpoint_state(final, 5e-5)
            report = instance.reset_optimizer()
            files_after = instance.save_checkpoint(base)
            after = checkpoint_state(base, 1e-5)
            self.assertEqual(files_before["adapter.safetensors"], files_after["adapter.safetensors"])
            self.assertEqual(previous["step"], 32); self.assertEqual(after["step"], 0)
            self.assertEqual(previous["adapter_tensors"], after["adapter_tensors"])
            self.assertEqual(previous["rng"], after["rng"])
            self.assertEqual(after["moments"], {})
            self.assertTrue(report["adapter_tensors_unchanged"])
            self.assertTrue(report["rl_moments_empty"])
            self.assertNotEqual(files_before["optimizer.pt"], files_after["optimizer.pt"])
        with self.assertRaisesRegex(ValueError, "SFT5e-5"):
            instance.reset_optimizer()

    def test_nonadvancing_adam_and_changed_frozen_state_fail(self):
        instance = policy()
        with patch.object(instance.optimizer, "step", return_value=None):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                instance.sft_update(targets())
        instance.model.base.packed[0] += 1
        with self.assertRaisesRegex(ValueError, "Frozen"):
            instance.sft_update(targets())


if __name__ == "__main__":
    unittest.main()
