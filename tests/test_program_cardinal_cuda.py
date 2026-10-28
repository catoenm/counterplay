"""Tiny CPU tensor fixtures; no pretrained model, CUDA, API, or game rollout."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from selfplay import program_cardinal_cuda as cuda

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("safetensors") is not None
if HAS_TORCH:
    import torch


def fixture():
    torch.manual_seed(77)
    class Pair(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 2, bias=False)})
            self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 2, bias=False)})
    model = torch.nn.Module()
    model.self_attn = torch.nn.Module()
    model.self_attn.q_proj, model.self_attn.v_proj = Pair(), Pair()
    model.base = torch.nn.Module()
    model.base.weight = torch.nn.Parameter(torch.tensor([1., 2.]), requires_grad=False)
    model.base.register_buffer("quant_map", torch.tensor([0., 1.]))
    model.base.register_buffer("packed", torch.tensor([1, 15, 255], dtype=torch.uint8))
    policy = object.__new__(cuda.CardinalCudaPolicy)
    policy.torch, policy.model, policy.runtime = torch, model, {}
    policy.config = {"learning_rate": 1e-5}
    policy.parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    policy.optimizer = torch.optim.AdamW(policy.parameters.values(), lr=1e-5, weight_decay=0)
    policy._initialize_identity()
    return policy


def one_step(policy):
    policy.optimizer.zero_grad(set_to_none=True)
    sum(value.square().sum() for value in policy.parameters.values()).backward()
    policy.optimizer.step()


def sampler():
    class FixedTokenTable:
        def __init__(self): self.batches = []
        def eval(self): return self
        def __call__(self, input_ids, past_key_values, use_cache, logits_to_keep):
            self.batches.append(len(input_ids))
            # Pure fixed transition table; no trainable or pretrained language model.
            table = torch.tensor([[0.2, 0.1, 0.4, -0.2], [0.4, -0.1, 0.2, 0.1],
                                  [0.5, 0.0, 0.1, -0.2], [0.1, 0.3, -0.2, 0.1]])
            return SimpleNamespace(logits=table[input_ids[:, -1]][:, None, :],
                                   past_key_values=(past_key_values or 0) + 1)
    policy = object.__new__(cuda.CardinalCudaPolicy)
    policy.torch, policy.model, policy.eos = torch, FixedTokenTable(), [0]
    policy.config = {"max_new_tokens": 15, "max_context_tokens": 20, "sample_seconds": 30}
    policy._sample_device = "cpu"
    policy.template = lambda _: ("fixture template", [1, 2])
    policy.tokenizer = SimpleNamespace(decode=lambda ids, skip_special_tokens: ",".join(
        str(i) for i in ids if not skip_special_tokens or i != 0))
    return policy


@unittest.skipUnless(HAS_TORCH, "Requires optional CPU Torch; no CUDA required")
class CardinalCudaTests(unittest.TestCase):
    def test_batched_rng_eos_and_exact_token_logps_equal_individual_rows(self):
        policy = sampler()
        batch = policy.sample_batch("fixture", list(range(8)))
        self.assertEqual(len(batch["records"]), 8)
        self.assertTrue(all(size == 8 for size in policy.model.batches))
        self.assertGreater(batch["timing"]["finished_eos"], 0)
        for seed, row in enumerate(batch["records"]):
            single = sampler().sample_batch("fixture", [seed])["records"][0]
            for key in ("seed", "completion_ids", "sampled_token_logprobs", "finish_reason", "text"):
                self.assertEqual(row[key], single[key])
            self.assertEqual(row["output_tokens"], len(row["sampled_token_logprobs"]))
            if row["finish_reason"] == "eos":
                self.assertEqual(row["completion_ids"][-1], 0)
                self.assertNotIn(0, row["completion_ids"][:-1])
        self.assertEqual(batch["timing"]["total_output_tokens"],
                         sum(r["output_tokens"] for r in batch["records"]))

    def test_sampler_bounds_time_and_independent_duplicate_seed(self):
        policy = sampler()
        for seeds in ([], list(range(9)), [True], [-1], [2**63]):
            with self.assertRaises(ValueError): policy.sample_batch("fixture", seeds)
        rows = policy.sample_batch("fixture", [42, 42])["records"]
        self.assertEqual(rows[0]["completion_ids"], rows[1]["completion_ids"])
        policy.config["sample_seconds"] = 0
        row = policy.sample_batch("fixture", [1])["records"][0]
        self.assertEqual((row["finish_reason"], row["completion_ids"]), ("time", []))
        policy.config["max_context_tokens"] = 2
        with self.assertRaisesRegex(ValueError, "context"): policy.sample_batch("fixture", [1])

    def test_checkpoint_restores_adapter_adam_lr_step_and_rng_exactly(self):
        policy = fixture(); one_step(policy)
        gpu_rng = [torch.get_rng_state().clone()]
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(torch.cuda, "get_rng_state_all", side_effect=lambda: [x.clone() for x in gpu_rng]), \
             patch.object(torch.cuda, "set_rng_state_all", side_effect=lambda values: gpu_rng.__setitem__(slice(None), values)):
            folder = Path(temp) / "checkpoint"
            hashes = policy.save_checkpoint(folder)
            identity = cuda.checkpoint_state(folder, 1e-5)
            self.assertEqual(identity, json.loads(json.dumps(identity)))
            self.assertEqual(identity["step"], 1)
            self.assertEqual(identity["learning_rate"], 1e-5)
            self.assertEqual(identity["trainable_names"], list(policy.parameters))
            original = {n: x.detach().clone() for n, x in policy.parameters.items()}
            one_step(policy); torch.manual_seed(999); gpu_rng[0] = torch.get_rng_state()
            self.assertEqual(policy._optimizer_identity()["step"], 2)
            policy.load_checkpoint(folder, hashes)
            self.assertEqual(policy._optimizer_identity()["state_sha256"], identity["state_sha256"])
            self.assertEqual(cuda.tensor_identity(torch.get_rng_state()), identity["rng"]["cpu"])
            for name, value in policy.parameters.items(): self.assertTrue(torch.equal(value, original[name]))
            with self.assertRaisesRegex(ValueError, "learning rate"):
                cuda.checkpoint_state(folder, 1e-6)
            with self.assertRaisesRegex(ValueError, "hashes"):
                policy.load_checkpoint(folder, {"adapter.safetensors": hashes["adapter.safetensors"]})

    def test_fresh_adam_zero_state_and_exact_schema(self):
        policy = fixture()
        identity = policy._optimizer_identity()
        self.assertEqual(identity["step"], 0); self.assertEqual(identity["moments"], {})
        self.assertEqual(policy.runtime["full_attention_lora_module_counts"], {"q_proj": 1, "v_proj": 1})
        state = deepcopy(policy.optimizer.state_dict())
        state["param_groups"][0]["params"].reverse()
        with self.assertRaisesRegex(ValueError, "order"):
            cuda.optimizer_identity(state, policy._schema, 1e-5)

    def test_moment_coverage_finite_shape_and_step_tampering_rejected(self):
        policy = fixture(); one_step(policy)
        original = policy.optimizer.state_dict()
        mutations = [
            lambda s: s["state"].pop(1),
            lambda s: s["state"][0].__setitem__("step", torch.tensor(1.5)),
            lambda s: s["state"][0].__setitem__("step", torch.tensor(2.)),
            lambda s: s["state"][0].__setitem__("exp_avg", torch.tensor([float("nan")])),
            lambda s: s["state"][0].__setitem__("exp_avg_sq", -torch.ones((2, 2))),
            lambda s: s["param_groups"][0].__setitem__("lr", 1e-6),
            lambda s: s["state"][0].__setitem__("unexpected", torch.tensor(1)),
        ]
        for mutate in mutations:
            state = deepcopy(original); mutate(state)
            with self.assertRaises(ValueError): cuda.optimizer_identity(state, policy._schema, 1e-5)

    def test_frozen_packed_weights_and_quantization_buffers_bound(self):
        policy = fixture()
        original = policy.verify_frozen()
        one_step(policy)
        self.assertEqual(policy.verify_frozen(), original)
        self.assertIn("base.quant_map", policy._frozen["quantization_state_tensors"])
        policy.model.base.packed[0] += 1
        with self.assertRaisesRegex(ValueError, "Frozen"): policy.verify_frozen()
        policy.model.base.packed[0] -= 1
        policy.model.base.quant_map[0] = .25
        with self.assertRaisesRegex(ValueError, "Frozen"): policy.verify_frozen()

    def test_saved_response_bytes_and_optimizer_names_bound_before_restore(self):
        policy = fixture()
        with tempfile.TemporaryDirectory() as temp, patch.object(torch.cuda, "get_rng_state_all", return_value=[torch.get_rng_state()]):
            folder = Path(temp) / "checkpoint"; hashes = policy.save_checkpoint(folder)
            state = torch.load(folder / "optimizer.pt", weights_only=True)
            state["cardinal_state"]["parameter_schema"].reverse()
            torch.save(state, folder / "optimizer.pt")
            with self.assertRaisesRegex(ValueError, "bytes"):
                policy.load_checkpoint(folder, hashes)
            hashes["optimizer.pt"] = cuda.common.digest(folder / "optimizer.pt")
            with self.assertRaisesRegex(ValueError, "identity"):
                policy.load_checkpoint(folder, hashes)

    def test_update_one_real_adam_step_and_mismatched_advantages_rejected(self):
        policy = fixture()
        with self.assertRaisesRegex(ValueError, "advantage"): policy.update([{}], [])
        with self.assertRaisesRegex(ValueError, "advantage"): policy.update([{}], [float("nan")])
        def update(instance, records, advantages):
            one_step(instance)
            return {"optimizer_steps": 1}
        with patch.object(cuda.common.CudaPolicy, "update", update):
            report = policy.update([{}], [1.])
        self.assertEqual((report["adam_step_before"], report["adam_step_after"]), (0, 1))
        with patch.object(cuda.common.CudaPolicy, "update", return_value={"optimizer_steps": 1}):
            with self.assertRaisesRegex(ValueError, "exactly one"): policy.update([{}], [1.])

    def test_smoke_derivative_preserves_optimizer_config_rng_and_prior_gradients(self):
        policy = fixture()
        policy.config.update(seed=124301, max_new_tokens=2048, token_logprob_tolerance=.002,
                             sequence_logprob_tolerance=.02)
        original_config = policy.config
        for value in policy.parameters.values(): value.grad = torch.ones_like(value) * .25
        before_grads = {n: p.grad.clone() for n, p in policy.parameters.items()}
        sample_calls = []
        def sample_batch(prompt, seeds):
            sample_calls.append((policy.config["max_new_tokens"], seeds))
            torch.rand(3)  # Prove global RNG is also restored on success/failure.
            return {"records": [{"completion_ids": [1, 2], "sampled_token_logprobs": [-.5, -.7]}
                                for _ in seeds], "timing": {"batch_elapsed_seconds": .1}}
        policy.sample_batch = sample_batch
        def logps(record):
            total = sum(p.sum() for p in policy.parameters.values())
            # Value-identical differentiable scalar fixture, without a neural model.
            return torch.tensor(record["sampled_token_logprobs"]) + (total-total.detach())
        policy.token_logps = logps
        gpu_rng = [torch.get_rng_state().clone()]
        before_rng = torch.get_rng_state().clone()
        with patch.object(torch.cuda, "get_rng_state_all", side_effect=lambda: [x.clone() for x in gpu_rng]), \
             patch.object(torch.cuda, "set_rng_state_all", side_effect=lambda values: gpu_rng.__setitem__(slice(None), values)):
            report = policy.mechanics_smoke("fixture")
            self.assertTrue(report["passed"], report)
            self.assertTrue(report["derivative"]["finite_nonzero"])
            self.assertEqual(report["optimizer_steps"], 0)
            self.assertEqual(sample_calls[0][0], 32)
            self.assertEqual(len(sample_calls[0][1]), 8)
            self.assertIs(policy.config, original_config)
            self.assertEqual(policy._optimizer_identity()["step"], 0)
            self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))
            for name, value in policy.parameters.items(): self.assertTrue(torch.equal(value.grad, before_grads[name]))
            policy.token_logps = lambda row: logps(row) + .1
            failed = policy.mechanics_smoke("fixture")
            self.assertFalse(failed["passed"])
            self.assertIn("likelihood gate failed", failed["reason"])
            self.assertTrue(failed["optimizer_unchanged"])
            self.assertTrue(failed["adapter_unchanged"])
            self.assertTrue(failed["rng_restored"])
            self.assertFalse(policy.model.training)

    def test_actual_torch_checkpoint_survives_gpu_journal_json_reload(self):
        from selfplay import program_cardinal_gpu as gpu
        from tests.test_program_cardinal_gpu import fixture as journal_fixture
        policy = fixture()
        gpu_rng = [torch.get_rng_state().clone()]
        with journal_fixture() as (run, _, _), \
             patch.object(gpu, "checkpoint_state", cuda.checkpoint_state), \
             patch.object(torch.cuda, "get_rng_state_all", side_effect=lambda: [x.clone() for x in gpu_rng]), \
             patch.object(torch.cuda, "set_rng_state_all", side_effect=lambda values: gpu_rng.__setitem__(slice(None), values)):
            # Replace the explicitly fake checkpoint in this temporary fixture
            # with real CPU Torch Adam/adapter files; retain its frozen run plan.
            shutil.rmtree(run / "checkpoints")
            status = gpu.common.load(run / "status.json")
            status["artifacts"] = {}
            gpu.common.atomic_json(run / "status.json", status)
            engine = gpu.GPUCycle(run, policy)
            engine.initialize()
            first = engine.checkpoint("base")
            resumed = gpu.GPUCycle(run, policy)
            self.assertEqual(resumed.checkpoint("base"), first)
            resumed.activate("base")
            self.assertEqual(policy._optimizer_identity()["step"], 0)


if __name__ == "__main__":
    unittest.main()
