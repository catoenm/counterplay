"""Batched CUDA sampling and strict checkpoint identity for the cardinal pilot.

The numerical objective and teacher-forcing gates remain those of the archived
single-GPU backend. Importing this module does not import Torch or load a model.
Only ``CardinalCudaPolicy`` construction requires CUDA. ``checkpoint_state`` is
a CPU-only, weights-only deserializer suitable for an independent auditor.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

from scripts.cloud import train_program_rl as common

CHECKPOINT_VERSION = 1
CHECKPOINT_FILES = ("adapter.safetensors", "optimizer.pt")


def _json_hash(value):
    return hashlib.sha256(common.canonical(value).encode()).hexdigest()


def tensor_identity(tensor, chunk_bytes=8 << 20):
    """Hash exact storage values without dequantization or a full CPU copy."""
    import torch
    if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
        raise ValueError("Expected a dense tensor for identity")
    flat = tensor.detach().reshape(-1)
    chunk_elements = max(1, chunk_bytes // tensor.element_size())
    h = hashlib.sha256()
    for start in range(0, flat.numel(), chunk_elements):
        part = flat[start:start + chunk_elements].to("cpu").contiguous()
        h.update(part.view(torch.uint8).numpy().tobytes())
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "numel": tensor.numel(), "sha256": h.hexdigest()}


def frozen_fingerprints(model, trainable_names):
    """Include packed base weights, buffers, and serialized BNB quantization state."""
    names = set(trainable_names)
    actual = {name for name, value in model.named_parameters() if value.requires_grad}
    if names != actual:
        raise ValueError("Trainable parameter identity changed")
    state = model.state_dict()
    if not names <= set(state):
        raise ValueError("Trainable parameters missing from state_dict")
    tensors = {name: tensor_identity(value) for name, value in sorted(state.items()) if name not in names}
    quantized = {name: value for name, value in tensors.items()
                 if any(part in name for part in ("quant_state", ".absmax", ".quant_map", ".nested_"))}
    return {"sha256": _json_hash(tensors), "tensors": tensors,
            "quantization_state_sha256": _json_hash(quantized),
            "quantization_state_tensors": list(quantized)}


def parameter_schema(parameters):
    return [{"name": name, "shape": list(p.shape), "dtype": str(p.dtype), "numel": p.numel()}
            for name, p in parameters.items()]


def _validate_schema(schema):
    if not isinstance(schema, list) or not schema:
        raise ValueError("Missing ordered trainable schema")
    names = []
    for row in schema:
        if (set(row) != {"name", "shape", "dtype", "numel"} or not isinstance(row["name"], str)
                or not row["name"] or row["dtype"] != "torch.float32"
                or not isinstance(row["shape"], list) or any(type(n) is not int or n < 1 for n in row["shape"])
                or type(row["numel"]) is not int or row["numel"] != math.prod(row["shape"])):
            raise ValueError("Invalid ordered trainable schema")
        names.append(row["name"])
    if len(set(names)) != len(names):
        raise ValueError("Duplicate trainable names")
    return names


def optimizer_identity(state, schema, expected_learning_rate, expected_groups=None):
    """Validate one AdamW group and bind each state slot to its ordered parameter."""
    import torch
    names = _validate_schema(schema)
    if not isinstance(state, dict) or set(state) != {"state", "param_groups"}:
        raise ValueError("Unexpected optimizer state schema")
    groups = state["param_groups"]
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError("Expected exactly one AdamW parameter group")
    group = groups[0]
    ids = group.get("params")
    if (not isinstance(ids, list) or any(type(x) is not int for x in ids)
            or ids != list(range(len(names)))):
        raise ValueError("Optimizer parameter IDs/order differs")
    lr = group.get("lr")
    if (type(lr) not in (int, float) or not math.isfinite(lr) or lr <= 0
            or lr != expected_learning_rate):
        raise ValueError("Optimizer learning rate differs")
    hyperparameters = {k: v for k, v in group.items() if k != "params"}
    if (group.get("weight_decay") != 0 or group.get("amsgrad") is not False
            or group.get("maximize") is not False):
        raise ValueError("Unexpected AdamW algorithm settings")
    if expected_groups is not None and groups != expected_groups:
        raise ValueError("Optimizer parameter-group settings differ")
    slots = state["state"]
    if (not isinstance(slots, dict) or any(type(k) is not int for k in slots)
            or (slots and set(slots) != set(ids))):
        raise ValueError("Partial or unexpected Adam state coverage")
    steps, moments = [], {}
    for index, row in enumerate(schema):
        if not slots:
            continue
        slot = slots[index]
        if not isinstance(slot, dict) or set(slot) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Unexpected Adam moment schema")
        step = slot["step"]
        if (not isinstance(step, torch.Tensor) or step.numel() != 1 or step.ndim != 0
                or step.dtype != torch.float32 or not bool(torch.isfinite(step).all())):
            raise ValueError("Invalid Adam step tensor")
        number = float(step.item())
        if not number.is_integer() or number < 1:
            raise ValueError("Invalid Adam step value")
        steps.append(int(number))
        moments[row["name"]] = {"step": tensor_identity(step)}
        for key in ("exp_avg", "exp_avg_sq"):
            value = slot[key]
            if (not isinstance(value, torch.Tensor) or list(value.shape) != row["shape"]
                    or str(value.dtype) != row["dtype"] or not bool(torch.isfinite(value).all())):
                raise ValueError("Adam moment shape/dtype/finiteness differs")
            if key == "exp_avg_sq" and bool((value < 0).any()):
                raise ValueError("Negative Adam second moment")
            moments[row["name"]][key] = tensor_identity(value)
    if len(set(steps)) > 1:
        raise ValueError("Inconsistent Adam parameter steps")
    identity = {"learning_rate": float(lr), "step": steps[0] if steps else 0,
                "trainable_names": names, "parameter_schema": schema,
                "parameter_groups": [{**hyperparameters, "params": ids}], "moments": moments}
    # Checkpoint summaries are also journaled as JSON; Adam betas are tuples in
    # Torch but lists after JSON reload. Return one canonical representation.
    identity = json.loads(common.canonical(identity))
    return {**identity, "state_sha256": _json_hash(identity)}


def _rng_identity(state):
    import torch
    rng, cuda_rng = state["rng"], state["cuda_rng"]
    if not isinstance(cuda_rng, list) or len(cuda_rng) != 1:
        raise ValueError("Expected one CUDA RNG state")
    for value in [rng, *cuda_rng]:
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.uint8
                or value.ndim != 1 or not value.numel()):
            raise ValueError("Invalid RNG state")
    return {"cpu": tensor_identity(rng), "cuda": [tensor_identity(x) for x in cuda_rng]}


def _inspect_checkpoint(folder, expected_learning_rate=None):
    import torch
    from safetensors.torch import load_file
    folder = Path(folder)
    hashes = {name: common.digest(folder / name) for name in CHECKPOINT_FILES}
    state = torch.load(folder / "optimizer.pt", map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or set(state) != {"optimizer", "rng", "cuda_rng", "cardinal_state"}:
        raise ValueError("Checkpoint is not a versioned cardinal optimizer")
    meta = state["cardinal_state"]
    if (not isinstance(meta, dict) or set(meta) != {"format_version", "parameter_schema", "learning_rate",
            "parameter_groups", "frozen_state_sha256", "quantization_state_sha256"}
            or meta["format_version"] != CHECKPOINT_VERSION):
        raise ValueError("Checkpoint backend schema differs")
    for key in ("frozen_state_sha256", "quantization_state_sha256"):
        if (not isinstance(meta[key], str) or len(meta[key]) != 64
                or any(c not in "0123456789abcdef" for c in meta[key])):
            raise ValueError("Missing frozen-state fingerprint")
    lr = meta["learning_rate"] if expected_learning_rate is None else expected_learning_rate
    if meta["learning_rate"] != lr:
        raise ValueError("Checkpoint declared learning rate differs")
    identity = optimizer_identity(state["optimizer"], meta["parameter_schema"], lr, meta["parameter_groups"])
    tensors = load_file(str(folder / "adapter.safetensors"), device="cpu")
    if set(tensors) != set(identity["trainable_names"]):
        raise ValueError("Checkpoint adapter names differ from optimizer")
    adapter = {}
    for row in meta["parameter_schema"]:
        value = tensors[row["name"]]
        if (list(value.shape) != row["shape"] or str(value.dtype) != row["dtype"]
                or not bool(torch.isfinite(value).all())):
            raise ValueError("Checkpoint adapter shape/dtype/finiteness differs")
        adapter[row["name"]] = tensor_identity(value)
    summary = {**identity, "files": hashes, "adapter_tensors": adapter, "rng": _rng_identity(state),
               "frozen_state_sha256": meta["frozen_state_sha256"],
               "quantization_state_sha256": meta["quantization_state_sha256"],
               "format_version": CHECKPOINT_VERSION}
    return summary, state, tensors


def checkpoint_state(folder, expected_learning_rate=None):
    """Read and validate exact CPU optimizer/adapter values, without a neural model."""
    return _inspect_checkpoint(folder, expected_learning_rate)[0]


class CardinalCudaPolicy(common.CudaPolicy):
    def __init__(self, model_dir, config):
        if (config["temperature"], config["top_p"], config["top_k"]) != (1.0, 1.0, 0):
            raise ValueError("Only the unchanged full-softmax sampler is implemented")
        super().__init__(model_dir, config)
        self._initialize_identity()

    def _initialize_identity(self):
        self._schema = parameter_schema(self.parameters)
        names = _validate_schema(self._schema)
        modules = {name.split(".lora_")[0] for name in names}
        if (any(".lora_A." not in n and ".lora_B." not in n for n in names)
                or any(not n.endswith((".q_proj", ".v_proj")) for n in modules)
                or any(sum(n.startswith(m + ".lora_") for n in names) != 2 for m in modules)):
            raise ValueError("Expected paired q/v LoRA matrices only")
        counts = {name: sum(m.endswith("." + name) for m in modules) for name in ("q_proj", "v_proj")}
        if counts["q_proj"] != counts["v_proj"] or not counts["q_proj"]:
            raise ValueError("Expected matched full-attention q/v adapters")
        self._optimizer_groups = self.optimizer.state_dict()["param_groups"]
        self._frozen = frozen_fingerprints(self.model, names)
        self.runtime.update(trainable_names=names, parameter_schema=self._schema,
                            full_attention_lora_module_counts=counts,
                            full_attention_lora_modules=sorted(modules),
                            frozen_state=self._frozen, checkpoint_format_version=CHECKPOINT_VERSION,
                            sampling_batch="constant batch <=8, shared prompt, independent row RNG; inactive rows ignored")
        self._optimizer_identity()

    def _optimizer_identity(self):
        return optimizer_identity(self.optimizer.state_dict(), self._schema,
                                  self.config["learning_rate"], self._optimizer_groups)

    def verify_frozen(self):
        current = frozen_fingerprints(self.model, self.parameters)
        if current != self._frozen:
            raise ValueError("Frozen base or quantization-state tensor changed")
        return current["sha256"]

    def sample_batch(self, prompt, seeds):
        torch, config = self.torch, self.config
        if (not isinstance(seeds, (list, tuple)) or not 1 <= len(seeds) <= 8
                or any(type(s) is not int or not 0 <= s < 2**63 for s in seeds)):
            raise ValueError("A batch requires one to eight nonnegative integer seeds")
        rendered, prompt_ids = self.template(prompt)
        if not prompt_ids or len(prompt_ids) + config["max_new_tokens"] > config["max_context_tokens"]:
            raise ValueError("Prompt exceeds frozen context admission bound")
        # The CPU value is used only by tiny deterministic unit fixtures; actual
        # construction inherits the one-visible-CUDA-device admission check.
        device = getattr(self, "_sample_device", "cuda")
        generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
        ids = torch.tensor([prompt_ids] * len(seeds), dtype=torch.long, device=device)
        sampled, logps = [[] for _ in seeds], [[] for _ in seeds]
        finish, elapsed = ["length"] * len(seeds), [None] * len(seeds)
        active, active_counts, cache = list(range(len(seeds))), [], None
        started = time.monotonic()
        self.model.eval()
        with torch.no_grad():
            for _ in range(config["max_new_tokens"]):
                if time.monotonic() - started >= config["sample_seconds"]:
                    for i in active:
                        finish[i] = "time"
                    break
                active_counts.append(len(active))
                output = self.model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
                lp = torch.log_softmax(output.logits[:, -1].float(), dim=-1)
                if not bool(torch.isfinite(lp[active]).all()):
                    raise ValueError("Nonfinite sampled-token distribution")
                # Finished rows keep their cache slots; these filler tokens are
                # never sampled, scored, decoded, or returned as trajectory data.
                chosen = torch.full((len(seeds),), self.eos[0], dtype=torch.long, device=device)
                for i in active:
                    chosen[i] = torch.multinomial(lp[i].exp(), 1, generator=generators[i])[0]
                values = lp.gather(1, chosen[:, None]).squeeze(1).cpu().tolist()
                tokens = chosen.cpu().tolist()
                cache = output.past_key_values
                del output, lp
                remaining = []
                for i in active:
                    sampled[i].append(tokens[i]); logps[i].append(float(values[i]))
                    if tokens[i] in self.eos:
                        finish[i], elapsed[i] = "eos", time.monotonic() - started
                    else:
                        remaining.append(i)
                active = remaining
                if not active:
                    break
                ids = chosen[:, None]
        del cache
        duration = time.monotonic() - started
        records = [{"text": self.tokenizer.decode(sampled[i], skip_special_tokens=True),
                    "raw_text": self.tokenizer.decode(sampled[i], skip_special_tokens=False),
                    "prompt_template": rendered, "prompt_ids": list(prompt_ids),
                    "completion_ids": sampled[i], "sampled_token_logprobs": logps[i],
                    "seed": seed, "finish_reason": finish[i], "input_tokens": len(prompt_ids),
                    "output_tokens": len(sampled[i]), "elapsed_seconds": elapsed[i] or duration,
                    "eos_token_ids": list(self.eos), "sampling_batch_size": len(seeds), "sampling_batch_row": i}
                   for i, seed in enumerate(seeds)]
        return {"records": records, "timing": {"batch_elapsed_seconds": duration,
                "total_output_tokens": sum(map(len, sampled)), "active_sequences": len(seeds),
                "active_sequences_by_step": active_counts, "batch_steps": len(active_counts),
                "finished_eos": finish.count("eos"), "constant_cache_batch_size": len(seeds)}}

    def update(self, records, advantages):
        if (not records or len(records) != len(advantages) or len(records) > 8
                or not all(type(a) in (int, float) and math.isfinite(a) for a in advantages)):
            raise ValueError("One finite advantage per rollout required")
        before = self._optimizer_identity()
        self.verify_frozen()
        report = super().update(records, advantages)
        after = self._optimizer_identity()
        if after["step"] != before["step"] + 1:
            raise ValueError("Optimizer did not advance exactly one step")
        frozen_sha = self.verify_frozen()
        return {**report, "adam_step_before": before["step"], "adam_step_after": after["step"],
                "optimizer_state_sha256": after["state_sha256"], "frozen_state_sha256": frozen_sha}

    def mechanics_smoke(self, prompt):
        """Eight <=32-token draws and a derivative check, never an optimizer step.

        These seeds are outside every registered experiment generation family.
        Sampled records are returned for provenance, but are never training data.
        Failure is a report so the runner can preserve the failed admission gate.
        """
        torch, original_config = self.torch, self.config
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = [value.clone() for value in torch.cuda.get_rng_state_all()]
        previous_grads = {name: None if p.grad is None else p.grad.detach().clone()
                          for name, p in self.parameters.items()}
        adapter_before = {name: tensor_identity(p) for name, p in self.parameters.items()}
        optimizer_before = self._optimizer_identity()
        report = {"passed": False, "kind": "unscored_32_token_batch_and_derivative_admission",
                  "training_use": False, "optimizer_steps": 0, "batch_size": 8,
                  "max_new_tokens": min(32, original_config["max_new_tokens"]),
                  "seeds": [50_000_000 + original_config["seed"] + i for i in range(8)],
                  "optimizer_state_before_sha256": optimizer_before["state_sha256"]}
        try:
            self.verify_frozen()
            self.config = {**original_config, "max_new_tokens": report["max_new_tokens"]}
            batch = self.sample_batch(prompt, report["seeds"])
            report["batch"] = batch
            records = batch["records"]
            if len(records) != 8 or any(not row["completion_ids"] for row in records):
                raise ValueError("Smoke did not produce all eight nonempty sampled records")
            report["eval_likelihood_gate"] = self.likelihood_gate(records)
            if not report["eval_likelihood_gate"]["passed"]:
                raise ValueError("Smoke cached-batch versus teacher-forced eval likelihood gate failed")
            self.optimizer.zero_grad(set_to_none=True)
            self.model.train()
            checks = []
            for record in records:
                logps = self.token_logps(record)
                check = common.check_likelihoods(record["sampled_token_logprobs"],
                    logps.detach().cpu().tolist(), original_config["token_logprob_tolerance"],
                    original_config["sequence_logprob_tolerance"])
                checks.append(check)
                report["train_likelihood_gate"] = {"passed": all(r["passed"] for r in checks), "records": checks}
                if not check["passed"]:
                    raise ValueError("Smoke cached-batch versus train-mode likelihood gate failed")
                (-logps.sum() / len(records)).backward()
                del logps
            squared, missing = 0.0, []
            for name, value in self.parameters.items():
                if value.grad is None:
                    missing.append(name)
                elif not bool(torch.isfinite(value.grad).all()):
                    raise ValueError("Nonfinite smoke derivative")
                else:
                    squared += float(value.grad.detach().double().square().sum().cpu())
            report["derivative"] = {"gradient_l2_norm": math.sqrt(squared),
                                    "missing_gradient_names": missing,
                                    "finite_nonzero": math.isfinite(squared) and squared > 0 and not missing}
            if not report["derivative"]["finite_nonzero"]:
                raise ValueError("Smoke derivative is zero, incomplete, or nonfinite")
            report["passed"] = True
        except Exception as error:
            report.update(passed=False, error_type=type(error).__name__, reason=str(error)[:2000])
        finally:
            self.config = original_config
            for name, value in self.parameters.items():
                value.grad = previous_grads[name]
            self.model.eval()
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(cuda_rng)
        adapter_after = {name: tensor_identity(p) for name, p in self.parameters.items()}
        optimizer_after = self._optimizer_identity()
        report["optimizer_state_after_sha256"] = optimizer_after["state_sha256"]
        report["adapter_unchanged"] = adapter_before == adapter_after
        report["optimizer_unchanged"] = optimizer_before == optimizer_after
        report["rng_restored"] = (torch.equal(torch.get_rng_state(), cpu_rng)
            and all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), cuda_rng)))
        report["config_restored"] = self.config is original_config
        try:
            report["frozen_state_sha256"] = self.verify_frozen()
        except ValueError as error:
            report.update(passed=False, error_type=type(error).__name__, reason=str(error))
        if not all(report[k] for k in ("adapter_unchanged", "optimizer_unchanged", "rng_restored", "config_restored")):
            report.update(passed=False, reason="Smoke mutated adapter, optimizer, RNG, or configuration")
        return report

    numerical_smoke = mechanics_smoke

    def save_checkpoint(self, folder):
        from safetensors.torch import save_file
        folder = Path(folder)
        self.verify_frozen(); self._optimizer_identity()
        folder.mkdir(parents=True, exist_ok=False)
        save_file({n: p.detach().cpu().contiguous() for n, p in self.parameters.items()},
                  str(folder / "adapter.safetensors"))
        self.torch.save({"optimizer": self.optimizer.state_dict(), "rng": self.torch.get_rng_state(),
            "cuda_rng": self.torch.cuda.get_rng_state_all(), "cardinal_state": {
                "format_version": CHECKPOINT_VERSION, "parameter_schema": self._schema,
                "learning_rate": self.config["learning_rate"], "parameter_groups": self._optimizer_groups,
                "frozen_state_sha256": self._frozen["sha256"],
                "quantization_state_sha256": self._frozen["quantization_state_sha256"]}}, folder / "optimizer.pt")
        return self.checkpoint_state(folder)["files"]

    def checkpoint_state(self, folder):
        return checkpoint_state(folder, self.config["learning_rate"])

    def load_checkpoint(self, folder, hashes):
        if set(hashes) != set(CHECKPOINT_FILES):
            raise ValueError("Exact adapter and optimizer checkpoint hashes required")
        folder = Path(folder)
        for name, expected in hashes.items():
            if common.digest(folder / name) != expected:
                raise ValueError("Checkpoint bytes changed")
        summary, state, tensors = _inspect_checkpoint(folder, self.config["learning_rate"])
        if (summary["parameter_schema"] != self._schema
                or _json_hash(summary["parameter_groups"]) != _json_hash(self._optimizer_groups)
                or summary["frozen_state_sha256"] != self._frozen["sha256"]
                or summary["quantization_state_sha256"] != self._frozen["quantization_state_sha256"]):
            raise ValueError("Checkpoint model/optimizer identity differs")
        self.verify_frozen()
        with self.torch.no_grad():
            for name, p in self.parameters.items():
                p.copy_(tensors[name].to(p.device))
                if not self.torch.equal(p.detach().cpu(), tensors[name]):
                    raise ValueError("Adapter restoration is not exact")
        self.optimizer.load_state_dict(state["optimizer"])
        if self._optimizer_identity() != {k: summary[k] for k in self._optimizer_identity()}:
            raise ValueError("Adam restoration is not exact")
        self.torch.set_rng_state(state["rng"])
        self.torch.cuda.set_rng_state_all(state["cuda_rng"])
        restored_rng = _rng_identity({"rng": self.torch.get_rng_state(),
                                     "cuda_rng": self.torch.cuda.get_rng_state_all()})
        if restored_rng != summary["rng"]:
            raise ValueError("RNG restoration is not exact")
        self.model.eval()
        self.verify_frozen()
