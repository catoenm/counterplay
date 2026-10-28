"""Prepared MLX 27B controller-RL smoke; no model download or GPU on import.

Only the final full-attention block's q/v LoRA weights are differentiable. The
63-block prefix stays frozen in eval mode, including its gated-delta kernels.
This is intentionally narrower than the CUDA pilot and must be reported as such.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.cloud import train_program_rl as common

MODEL_REPO = "mlx-community/Qwen3.8-27B-4bit"
MODEL_REVISION = "10c35caafbb80f7dc6a7a432cdd11af10a6d4818"
# HF LFS SHA-256 metadata read 2026-10-22; no shard was downloaded in preparation.
WEIGHT_HASHES = {
    "model-00001-of-00003.safetensors": "6cc1508e96fb5d0865dfd5753a79f4ec60651bf3e2a82844a7e8ae9c60528c0d",
    "model-00002-of-00003.safetensors": "83f2a20ca8058f486a3634a27faf99587f4cd3c156a83dee34fb99e6ac178670",
    "model-00003-of-00003.safetensors": "31b8c91ef899f79efaaa69e3d2c096f6e2ebeb2ff20e29222abbd9ebc79e560a",
}
# Authoritative pinned-revision tree metadata, fetched without any weight bytes:
# https://huggingface.co/api/models/mlx-community/Qwen3.8-27B-4bit/revision/10c35caafbb80f7dc6a7a432cdd11af10a6d4818?blobs=true
METADATA_IDENTITIES = {
    ".gitattributes": ("git-sha1", 1570, "52373fe24473b1aa44333d318f578ae6bf04b49b"),
    "README.md": ("git-sha1", 632, "43eb816caadebce9a7fac44466053d32e1c0acd9"),
    "chat_template.jinja": ("git-sha1", 8952, "c0c686f9c38d70d179fb7b5f5aa7530bc913dda3"),
    "config.json": ("git-sha1", 4932, "3fb03e916f494e6745aa23df751da26f79e06470"),
    "generation_config.json": ("git-sha1", 202, "023756cfadf88e5bf69eefeee3e172f38c448d64"),
    "model.safetensors.index.json": ("git-sha1", 218281, "05a3a3a17368d54d2887783e96b3a8191e9655f0"),
    "preprocessor_config.json": ("git-sha1", 390, "2ea84a437d448ff71b08df68fdd949d5cc4ebb64"),
    "processor_config.json": ("git-sha1", 991, "4bbac788e2cf449e619d7fee64f206c793a4695e"),
    "tokenizer.json": ("sha256", 19989325, "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"),
    "tokenizer_config.json": ("git-sha1", 1165, "1d134cd298be1e3be25db393d93a1cefe80e3214"),
    "video_preprocessor_config.json": ("git-sha1", 385, "3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe"),
    "vocab.json": ("git-sha1", 6722759, "0aa0ce0658d60ac4a5d609f4eadb0e8e43514176"),
}
OVERRIDES = {
    "backend": "mlx", "loader": "mlx_lm", "iterations": 1, "opponent_count": 1,
    "eval_samples": 1, "map_seeds_per_group": 1, "max_new_tokens": 1024,
    "max_context_tokens": 3072, "sample_seconds": 300, "worker_seconds": 1800,
    "memory_limit_bytes": 28 * 1024**3, "cache_limit_bytes": 256 * 1024**2,
    "lora_last_layers": 1, "lora_scale": 16.0,
    "prefix_mode": "eval-only, detached; final block eval for generation and gradients",
    "scope": "single last-attention-block local mechanism smoke; not the earlier action-policy pilot",
}
CONDITIONS = {
    "plain-v1": {"condition": "plain-v1", "max_context_tokens": 3072},
    "feedback-exact-r1-v1": {"condition": "feedback-exact-r1-v1", "max_context_tokens": 8192},
    "feedback-compact-r1-v1": {"condition": "feedback-compact-r1-v1", "max_context_tokens": 8192},
}
FEEDBACK_PROMPT = ROOT / "runs/program-feedback-v1/prompts/qwen38-27b-1-r1.txt"
PLAIN_PROMPT = ROOT / "runs/program-feedback-v1/original-prompt.txt"
LOCK = ROOT / "runs/local-neural.lock"


def condition_config(condition):
    if condition not in CONDITIONS:
        raise ValueError("Unknown named local condition")
    return {**OVERRIDES, **CONDITIONS[condition]}


def compact_context():
    """Fixed rows, no successful response or outcome-dependent trace selection."""
    parent = ROOT / "runs/program-feedback-v1"
    specification = parent / "original-prompt.txt"
    original = parent / "originals/generation-1.json"
    matches = parent / "practice/qwen38-27b-1/matches.jsonl"
    summary = parent / "practice/qwen38-27b-1/summary.json"
    evaluation = parent / "practice/qwen38-27b-1/evaluation.json"
    plan = common.load(parent / "plan.json")
    for path in (specification, original):
        if common.digest(path) != plan["input_sha256"][str(path.relative_to(parent))]:
            raise ValueError("Original feedback source changed")
    summary_row = common.load(summary)
    if not summary_row["complete"] or summary_row["matches_sha256"] != common.digest(matches):
        raise ValueError("Practice replay file not bound to completed summary")
    if summary_row["evaluation_sha256"] != common.digest(evaluation):
        raise ValueError("Practice evaluation manifest changed")
    evaluation_row = common.load(evaluation)
    if evaluation_row["plan_sha256"] != common.digest(parent / "plan.json"):
        raise ValueError("Practice evaluation is not bound to the original diagnostic plan")
    original_row = common.load(original)
    normalized, _ = common.normalize_final(original_row["text"])
    original_program = common.compile_python_program(normalized).compiled
    if evaluation_row["classification"]["compiled_program"]["ir_sha256"] != original_program.digest:
        raise ValueError("Evaluated program differs from supplied original failed response")
    prescribed = {row["episode_id"]: row for row in evaluation_row["schedule"]}
    rows = [json.loads(line) for line in matches.read_text().splitlines()]
    schedule = {(opponent, seed, seat) for opponent in ("nearest", "value")
                for seed in range(730000, 730006) for seat in (0, 1)}
    stats, traces, seen = [], [], set()
    for match in rows:
        replay = match["record"]
        common.verify_replay(replay)
        spec = prescribed.get(match["episode_id"])
        if spec is None or any(match[k] != spec[k] for k in ("players", "match_kind")):
            raise ValueError("Practice episode differs from frozen evaluation schedule")
        if replay["seed"] != spec["seed"] or replay["max_rounds"] != spec["max_rounds"]:
            raise ValueError("Practice scenario differs from frozen schedule")
        seat = match["players"].index("qwen38-27b-1")
        opponent = match["players"][1-seat]
        if replay["programs"][seat]["digest"] != original_program.digest:
            raise ValueError("Replay did not execute the supplied failed original program")
        reference = common.compile_program(common.reference_programs()[opponent])
        if replay["programs"][1-seat]["digest"] != reference.digest:
            raise ValueError("Replay used a different scripted reference program")
        identity = (opponent, replay["seed"], seat)
        if identity not in schedule or identity in seen:
            raise ValueError("Unexpected/duplicate practice game in compact context")
        seen.add(identity)
        metrics = replay["metrics"]
        stats.append([opponent, replay["seed"], seat, metrics["scores"][seat], metrics["scores"][1-seat],
                      metrics["program_faults"][seat], metrics["failed_actions"][seat]])
        if opponent == "nearest" and replay["seed"] == 730000:
            selected = []
            for index in list(range(8)) + list(range(40, 48)):
                event = replay["events"][index]
                obs, decision, state = event["observations"][seat], event["decisions"][seat], event["state"]
                selected.append([event["turn"], obs["self_x"], obs["self_y"], obs["energy"], obs["carrying"],
                    decision["action"], state["history"][-1]["outcomes"][str(seat)], decision["fault"],
                    state["robots"][seat]["score"], state["robots"][1-seat]["score"]])
            traces.append({"opponent": opponent, "seed": replay["seed"], "seat": seat, "rows": selected})
    if seen != schedule or len(traces) != 2:
        raise ValueError("Compact context must contain all24 practice games and both predefined traces")
    stats.sort(key=lambda r: (r[0], r[1], r[2]))
    traces.sort(key=lambda r: r["seat"])
    factual = {
        "practice_columns": ["opponent", "seed", "seat", "banked_points", "opponent_banked_points", "program_faults", "failed_actions"],
        "practice_games": stats,
        "trace_columns": ["turn", "x", "y", "energy", "cargo_units", "action", "outcome", "program_fault", "banked_points", "opponent_banked_points"],
        "trace_selection": "First8 and last8 turns from nearest/seed730000, both seats; fixed before local sampling",
        "traces": traces,
    }
    text = (specification.read_text() + "\n\nPrevious controller from a separate model call:\n" +
            common.load(original)["text"] + "\n\nFactual practice results for that controller:\n" +
            common.canonical(factual) + "\n\nWrite a revised complete controller using the same restricted syntax. "
            "Return only the complete act function. It will be evaluated on unseen maps and opponents.\n")
    context = {"condition": "feedback-compact-r1-v1", "builder_version": 2,
               "input_sha256": {str(p.relative_to(ROOT)): common.digest(p) for p in
                                (specification, original, matches, summary, evaluation, parent / "plan.json")},
               "facts": factual, "successful_response_included": False, "algorithm_hints_added": False,
               "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
               "prompt_bytes": len(text.encode()), "gradient_targets": "Fresh learner completion tokens only"}
    return text, context


def write_compact_prompt(path):
    text, context = compact_context()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != text:
        raise ValueError("Refusing to replace different compact prompt bytes")
    if not path.exists():
        path.write_text(text)
    metadata = path.with_suffix(".context.json")
    if metadata.exists() and common.load(metadata) != context:
        raise ValueError("Refusing to replace different context provenance")
    if not metadata.exists():
        common.atomic_json(metadata, context)
    return context


def prepare(run, prompt, condition="plain-v1"):
    if condition == "plain-v1" and Path(prompt).read_bytes() != PLAIN_PROMPT.read_bytes():
        raise ValueError("Plain condition must preserve the original specification bytes")
    if condition == "feedback-exact-r1-v1" and Path(prompt).read_bytes() != FEEDBACK_PROMPT.read_bytes():
        raise ValueError("Exact feedback condition must preserve the original r1 prompt bytes")
    compact = None
    if condition == "feedback-compact-r1-v1":
        text, compact = compact_context()
        if Path(prompt).read_text() != text:
            raise ValueError("Compact condition must preserve the prescribed fixed selection")
    manifest = common.prepare(run, prompt, MODEL_REPO, MODEL_REVISION, overrides=condition_config(condition))
    source = "selfplay/program_local_rl.py"
    target = Path(run) / "source" / source
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((ROOT / source).read_bytes())
    manifest["source_sha256"][source] = common.digest(target)
    manifest["cuda_validation"] = "Not applicable; MLX numerical, memory and gradient smoke must pass before rollouts"
    manifest["model_weight_sha256"] = WEIGHT_HASHES
    manifest["model_metadata_identities"] = METADATA_IDENTITIES
    manifest["model_weight_bytes"] = 16054541349
    if condition == "feedback-exact-r1-v1":
        generation = common.load(ROOT / "runs/program-feedback-v1/generations/qwen38-27b-1-r1.json")
        manifest["teacher_assisted_context"] = {
            "source_prompt_sha256": common.digest(FEEDBACK_PROMPT),
            "source": str(FEEDBACK_PROMPT.relative_to(ROOT)),
            "gateway_reported_prompt_tokens": generation["usage"]["prompt_tokens"],
            "provisional_admission": "Gateway reports 11,988 input tokens; likely exceeds 8,192 bound. Local CPU tokenizer decides.",
            "gradient_targets": "Fresh learner completions only; neither API output nor prompt tokens become targets",
        }
    elif compact:
        manifest["teacher_assisted_context"] = compact
    manifest["expected_installed_versions"] = {name: importlib.metadata.version(name)
                                             for name in ("mlx", "mlx-lm", "transformers", "safetensors")}
    distribution = importlib.metadata.distribution("mlx-lm")
    files = ["mlx_lm/models/qwen3_5.py", "mlx_lm/models/qwen3_next.py", "mlx_lm/models/gated_delta.py",
             "mlx_lm/models/base.py", "mlx_lm/models/cache.py", "mlx_lm/tuner/utils.py", "mlx_lm/tuner/lora.py"]
    manifest["library_source_sha256"] = {name: common.digest(distribution.locate_file(name)) for name in files}
    common.atomic_json(Path(run) / "manifest.json", manifest)
    status = common.load(Path(run) / "status.json")
    status["manifest_sha256"] = common.digest(Path(run) / "manifest.json")
    common.atomic_json(Path(run) / "status.json", status)
    return manifest


def index_model(model_dir):
    """Hash already-present files; reject anything except the pinned shard bytes."""
    directory = Path(model_dir)
    verify_metadata(directory)
    files = {str(path.relative_to(directory)): common.digest(path)
             for path in directory.rglob("*") if path.is_file() and path.name != "model-index.json"
             and not str(path.relative_to(directory)).startswith(".cache/")}
    if any(files.get(name) != expected for name, expected in WEIGHT_HASHES.items()):
        raise ValueError("The three staged weight shards do not match the pinned model")
    index = {"repo": MODEL_REPO, "revision": MODEL_REVISION, "files": files}
    path = directory / "model-index.json"
    if path.exists() and common.load(path) != index:
        raise ValueError("Refusing to replace an incompatible model index")
    if not path.exists():
        common.atomic_json(path, index)
    return index


def verify_identity_bytes(path, identity):
    algorithm, size, expected = identity
    data = Path(path).read_bytes()
    if len(data) != size:
        raise ValueError("Authoritative metadata size differs: " + Path(path).name)
    if algorithm == "git-sha1":
        actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    elif algorithm == "sha256":
        actual = hashlib.sha256(data).hexdigest()
    else:
        raise ValueError("Unsupported metadata hash algorithm")
    if actual != expected:
        raise ValueError("Authoritative metadata hash differs: " + Path(path).name)


def verify_metadata(directory):
    directory = Path(directory)
    required = {"config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                "generation_config.json", "model.safetensors.index.json"}
    if not all((directory / name).is_file() for name in required):
        raise ValueError("Pinned model is missing required authoritative metadata")
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file()
              and not str(path.relative_to(directory)).startswith(".cache/")}
    if actual - (set(METADATA_IDENTITIES) | set(WEIGHT_HASHES) | {"model-index.json"}):
        raise ValueError("Model directory contains files outside the authoritative revision")
    for name, identity in METADATA_IDENTITIES.items():
        if (directory / name).exists():
            verify_identity_bytes(directory / name, identity)


def duration_limit(config, at=None):
    at = at or datetime.now(timezone.utc)
    remaining = (datetime.fromisoformat(config["deadline"]) - at).total_seconds()
    if remaining <= 0:
        raise TimeoutError("Deadline reached before local worker admission")
    return min(config["worker_seconds"], remaining)


def prompt_measurement(tokenizer, prompt, config):
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                      tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    total = len(ids) + config["max_new_tokens"]
    return {"condition": config["condition"], "prompt_template": rendered, "prompt_ids": ids,
            "prompt_tokens": len(ids), "reserved_completion_tokens": config["max_new_tokens"],
            "maximum_total_tokens": total, "context_cap": config["max_context_tokens"],
            "admitted": bool(ids) and total <= config["max_context_tokens"],
            "truncation": False}


class MlxFinalBlockPolicy:
    def __init__(self, model_dir, config):
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten
        from mlx_lm import load
        from mlx_lm.tuner.utils import linear_to_lora_layers
        self.mx, self.nn, self.config = mx, nn, config
        self.started = time.monotonic()
        if not mx.metal.is_available():
            raise RuntimeError("Apple Metal is required for this smoke")
        recommended = mx.metal.device_info()["max_recommended_working_set_size"]
        self.limit = min(config["memory_limit_bytes"], int(recommended))
        mx.set_memory_limit(self.limit)
        mx.set_cache_limit(config["cache_limit_bytes"])
        mx.random.seed(config["seed"])
        self.model, self.tokenizer = load(str(model_dir), lazy=True)
        if self.model.model_type != "qwen3_5" or len(self.model.layers) != 64:
            raise ValueError("Expected the pinned 64-layer qwen3_5 model")
        if self.model.layers[-1].is_linear:
            raise ValueError("The final layer must be full attention")
        if self.model.model.pipeline_size != 1:
            raise ValueError("This runner implements one-device text inference only")
        before = self.integer_fingerprints()
        self.model.set_dtype(mx.float32)
        mx.eval(self.model.parameters())
        if before != self.integer_fingerprints():
            raise ValueError("Packed quantized weights changed during FP32 conversion")
        if any(mx.issubdtype(value.dtype, mx.floating) and value.dtype != mx.float32
               for _, value in tree_flatten(self.model.parameters())):
            raise ValueError("Not all floating tensors became FP32")
        self.model.freeze()
        linear_to_lora_layers(self.model, 1, {"rank": config["lora_rank"], "scale": config["lora_scale"],
                             "dropout": 0.0, "keys": ["self_attn.q_proj", "self_attn.v_proj"]})
        self.last = self.model.layers[-1]
        self.model.eval()
        trained = dict(tree_flatten(self.last.trainable_parameters()))
        expected = {f"self_attn.{projection}.lora_{side}" for projection in ("q_proj", "v_proj") for side in ("a", "b")}
        if set(trained) != expected:
            raise ValueError("Trainable tensor set differs from final-block q/v LoRA")
        all_trained = dict(tree_flatten(self.model.trainable_parameters()))
        if len(all_trained) != 4 or any("layers.63." not in name for name in all_trained):
            raise ValueError("An earlier block unexpectedly has trainable parameters")
        self.optimizer = optim.Adam(learning_rate=config["learning_rate"])
        self.optimizer.init(self.last.trainable_parameters())
        mx.eval(self.optimizer.state)
        self.eos = sorted(self.tokenizer.eos_token_ids)
        if not self.eos:
            raise ValueError("Tokenizer has no explicit EOS identifiers")
        self.runtime = {"versions": {name: importlib.metadata.version(name) for name in
                                    ("mlx", "mlx-lm", "transformers", "safetensors")},
                        "arithmetic": "MLX affine 4bit packed base unchanged; floating tensors/LoRA FP32",
                        "prefix_mode": config["prefix_mode"], "memory_limit_bytes": self.limit,
                        "eos_token_ids": self.eos, "trainable_tensor_names": sorted(trained),
                        "trainable_parameters": sum(v.size for v in trained.values()),
                        "packed_integer_tensors": before, "external_spend_usd": 0}
        self.guard()

    def guard(self):
        common.admission(self.config)
        if time.monotonic() - self.started > self.config["worker_seconds"]:
            raise TimeoutError("Local worker time limit")
        if self.mx.get_active_memory() > self.limit or self.mx.get_peak_memory() > self.limit:
            raise MemoryError("MLX memory ceiling exceeded")

    def integer_fingerprints(self):
        import numpy as np
        from mlx.utils import tree_flatten
        result = {}
        for name, value in tree_flatten(self.model.parameters()):
            if self.mx.issubdtype(value.dtype, self.mx.integer):
                self.mx.eval(value)
                array = np.array(value)
                result[name] = {"dtype": str(value.dtype), "shape": list(value.shape),
                                "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest()}
                del array
        return result

    def template(self, prompt):
        rendered = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                             tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return rendered, self.tokenizer.encode(rendered, add_special_tokens=False)

    def sample(self, prompt, seed, token_cap=None):
        from mlx_lm.models.cache import make_prompt_cache
        mx = self.mx
        rendered, prompt_ids = self.template(prompt)
        maximum = token_cap or self.config["max_new_tokens"]
        if not prompt_ids or len(prompt_ids) + maximum > self.config["max_context_tokens"]:
            raise ValueError("Formatted prompt + generation exceeds context admission cap")
        self.model.eval()
        mx.random.seed(seed)
        cache = make_prompt_cache(self.model)
        ids, sampled, probabilities = prompt_ids, [], []
        started = time.monotonic()
        finish = "length"
        for _ in range(maximum):
            self.guard()
            if time.monotonic() - started >= self.config["sample_seconds"]:
                finish = "time"
                break
            # Project only the last hidden state to avoid prompt*vocabulary logits.
            hidden = self.model.model(mx.array([ids]), cache=cache)
            logits = self.model.language_model.lm_head(hidden[:, -1, :]).astype(mx.float32)[0]
            logps = logits - mx.logsumexp(logits)
            selected = mx.random.categorical(logps)
            mx.eval(selected, logps, [c.state for c in cache])
            token = int(selected.item())
            sampled.append(token)
            probabilities.append(float(logps[token].item()))
            del hidden, logits, logps, selected
            if token in self.eos:
                finish = "eos"
                break
            ids = [token]
        del cache
        mx.clear_cache()
        return {"text": self.tokenizer.decode(sampled, skip_special_tokens=True),
                "raw_text": self.tokenizer.decode(sampled, skip_special_tokens=False),
                "prompt_template": rendered, "prompt_ids": prompt_ids, "completion_ids": sampled,
                "sampled_token_logprobs": probabilities, "seed": seed, "finish_reason": finish,
                "eos_token_ids": self.eos, "input_tokens": len(prompt_ids), "output_tokens": len(sampled),
                "elapsed_seconds": time.monotonic() - started}

    def frozen_prefix(self, record):
        """Evaluate the prefix outside autograd; never backpropagate through it."""
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask
        mx = self.mx
        self.model.eval()
        ids = record["prompt_ids"] + record["completion_ids"][:-1]
        self.guard()
        if len(ids) > self.config["max_context_tokens"]:
            raise ValueError("Training sequence exceeds cap")
        hidden = self.model.model.embed_tokens(mx.array([ids]))
        attention_mask = create_attention_mask(hidden, None)
        state_mask = create_ssm_mask(hidden, None)
        for layer in self.model.layers[:-1]:
            hidden = layer(hidden, mask=state_mask if layer.is_linear else attention_mask, cache=None)
            # Materialize each frozen layer and release its graph. This avoids
            # retaining the 63-layer prefix for the one-block gradient operation.
            mx.eval(hidden)
            hidden = mx.stop_gradient(hidden)
            self.guard()
        return hidden, attention_mask

    def completion_logps(self, record, prefix, attention_mask):
        mx = self.mx
        positions = common.completion_positions(record["prompt_ids"], record["completion_ids"])
        hidden = self.last(prefix, mask=attention_mask, cache=None)
        hidden = self.model.model.norm(hidden[:, mx.array(positions), :])
        logits = self.model.language_model.lm_head(hidden).astype(mx.float32)[0]
        labels = mx.array(record["completion_ids"])
        return mx.take_along_axis(logits, labels[:, None], axis=-1).squeeze(-1) - mx.logsumexp(logits, axis=-1)

    def likelihood_gate(self, records):
        reports = []
        for row in records:
            prefix, mask = self.frozen_prefix(row)
            lp = self.completion_logps(row, prefix, mask)
            self.mx.eval(lp)
            report = common.check_likelihoods(row["sampled_token_logprobs"], lp.tolist(),
                       self.config["token_logprob_tolerance"], self.config["sequence_logprob_tolerance"])
            reports.append(report)
            del prefix, mask, lp
            self.mx.clear_cache()
            self.guard()
        return {"passed": all(r["passed"] for r in reports), "records": reports,
                "path": "cached full model vs detached frozen prefix + final full-attention block; both eval"}

    def sample_gradient(self, record, advantage, divisor):
        prefix, mask = self.frozen_prefix(record)
        def loss_fn():
            lp = self.completion_logps(record, prefix, mask)
            return -advantage * self.mx.sum(lp) / divisor, lp
        (loss, lp), gradient = self.nn.value_and_grad(self.last, loss_fn)()
        self.mx.eval(loss, lp, gradient)
        check = common.check_likelihoods(record["sampled_token_logprobs"], lp.tolist(),
                       self.config["token_logprob_tolerance"], self.config["sequence_logprob_tolerance"])
        if not check["passed"]:
            raise ValueError("Gradient forward likelihood differs from sampled policy")
        self.guard()
        return loss, gradient, check

    def mechanics_smoke(self, prompt):
        from mlx.utils import tree_flatten
        row = self.sample(prompt, self.config["seed"] + 77777, token_cap=32)
        gate = self.likelihood_gate([row])
        if not gate["passed"]:
            raise ValueError("32-token numerical smoke failed before gameplay collection")
        loss, gradient, gradient_check = self.sample_gradient(row, 1.0, 1)
        norm = math.sqrt(sum(float(self.mx.sum(g * g).item()) for _, g in tree_flatten(gradient)))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("Gradient smoke produced zero/nonfinite derivative")
        self.mx.clear_cache()
        self.guard()
        return {"passed": True, "sample": row, "gate": gate, "gradient_norm": norm,
                "gradient_likelihood_check": gradient_check,
                "loss": float(loss.item()), "optimizer_steps": 0,
                "peak_memory_bytes": self.mx.get_peak_memory(),
                "note": "Derivative diagnostic only; no adapter update or game reward was invented"}

    def update(self, records, advantages):
        import numpy as np
        from mlx.utils import tree_flatten, tree_map
        mx = self.mx
        before = {name: np.array(value) for name, value in tree_flatten(self.last.trainable_parameters())}
        accumulated, losses, checks = None, [], []
        for row, advantage in zip(records, advantages):
            loss, gradient, check = self.sample_gradient(row, float(advantage), len(records))
            checks.append(check)
            accumulated = gradient if accumulated is None else tree_map(lambda a, b: a + b, accumulated, gradient)
            losses.append(float(loss.item()))
            mx.eval(accumulated)
            del loss, gradient
            mx.clear_cache()
        norm = math.sqrt(sum(float(mx.sum(g * g).item()) for _, g in tree_flatten(accumulated)))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("Zero/nonfinite policy gradient; no update")
        clipped = tree_map(lambda g: g * min(1.0, self.config["gradient_clip"] / norm), accumulated)
        self.guard()
        self.optimizer.update(self.last, clipped)
        mx.eval(self.last.trainable_parameters(), self.optimizer.state)
        after = {name: np.array(value) for name, value in tree_flatten(self.last.trainable_parameters())}
        if any(not np.isfinite(value).all() for value in after.values()):
            raise ValueError("Nonfinite updated adapters")
        delta = math.sqrt(sum(float(np.square(after[k].astype("float64") - v.astype("float64")).sum())
                              for k, v in before.items()))
        if delta <= 0:
            raise ValueError("Adapter weights did not change")
        self.guard()
        return {"loss": sum(losses), "microbatch_losses": losses, "adapter_l2_delta": delta,
                "optimizer_steps": 1, "gradient_norm_before_clip": norm,
                "gradient_likelihood_gate": {"passed": True, "records": checks},
                "peak_memory_bytes": mx.get_peak_memory()}

    def save_checkpoint(self, folder):
        from mlx.utils import tree_flatten
        folder.mkdir(parents=True, exist_ok=False)
        self.mx.eval(self.last.trainable_parameters(), self.optimizer.state)
        self.mx.save_safetensors(str(folder / "adapter.safetensors"), dict(tree_flatten(self.last.trainable_parameters())))
        self.mx.save_safetensors(str(folder / "optimizer.safetensors"), dict(tree_flatten(self.optimizer.state)))
        # Every sample reseeds independently. The optimizer's full step/moments
        # are saved; no stochastic dropout or optimizer sampling is involved.
        return {name: common.digest(folder / name) for name in ("adapter.safetensors", "optimizer.safetensors")}

    def load_checkpoint(self, folder, hashes):
        import numpy as np
        from mlx.utils import tree_flatten, tree_unflatten
        for name, expected in hashes.items():
            if common.digest(folder / name) != expected:
                raise ValueError("MLX checkpoint bytes changed")
        weights = self.mx.load(str(folder / "adapter.safetensors"))
        expected = dict(tree_flatten(self.last.trainable_parameters()))
        if set(weights) != set(expected):
            raise ValueError("MLX adapter tensor names differ")
        for name, value in weights.items():
            if value.shape != expected[name].shape or value.dtype != self.mx.float32 or not np.isfinite(np.array(value)).all():
                raise ValueError("MLX adapter shape/dtype/finiteness mismatch")
        self.last.load_weights(list(weights.items()), strict=False)
        self.mx.eval(self.last.trainable_parameters())
        actual = dict(tree_flatten(self.last.trainable_parameters()))
        if any(not np.array_equal(np.array(weights[k]), np.array(v)) for k, v in actual.items()):
            raise ValueError("MLX adapter restoration was not exact")
        self.optimizer.state = tree_unflatten(list(self.mx.load(str(folder / "optimizer.safetensors")).items()))
        self.model.eval()


def worker(run, model_dir):
    manifest, status = common.verify_sources(run)
    config = manifest["config"]
    if config["backend"] != "mlx" or any(config.get(k) != v for k, v in condition_config(config["condition"]).items()):
        raise ValueError("Local smoke configuration differs from the bounded plan")
    if status["state"] == "complete":
        return common.audit(run)
    versions = {name: importlib.metadata.version(name) for name in manifest["expected_installed_versions"]}
    if versions != manifest["expected_installed_versions"]:
        raise ValueError("Installed MLX/runtime versions differ from preparation")
    distribution = importlib.metadata.distribution("mlx-lm")
    for name, expected in manifest["library_source_sha256"].items():
        if common.digest(distribution.locate_file(name)) != expected:
            raise ValueError("Installed MLX source differs from frozen preparation")
    index = common.verify_model_index(model_dir, config)
    verify_metadata(model_dir)
    if any(index["files"].get(name) != expected for name, expected in WEIGHT_HASHES.items()):
        raise ValueError("Pinned model weight identity mismatch")
    common.admission(config)
    # Tokenizer-only preflight is performed before importing MLX or loading any
    # neural weights. An oversized feedback context is rejected, never truncated.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    measurement = prompt_measurement(tokenizer, (Path(run) / "prompt.txt").read_text(), config)
    preflight = common.Runner(run, None)
    existing = preflight.existing("prompt-measurement.json")
    if existing is None:
        preflight.commit("prompt-measurement.json", measurement)
    elif existing != measurement:
        raise ValueError("CPU tokenizer/prompt measurement changed")
    del tokenizer
    if not measurement["admitted"]:
        raise ValueError("CPU tokenizer rejects context cap before neural load; no truncation permitted")
    backend = MlxFinalBlockPolicy(model_dir, config)
    runner = common.Runner(run, backend)
    existing = runner.existing("runtime.json")
    runtime = {"model_index": index, **backend.runtime}
    if existing is None:
        runner.commit("runtime.json", runtime)
    elif existing != runtime:
        raise ValueError("MLX runtime or model identity changed on resume")
    # Run a fresh numerical/derivative smoke on every new process, before any
    # optimizer stage. It does not update the adapter or become training data.
    smoke = backend.mechanics_smoke(runner.prompt)
    runner.commit("smoke-" + str(time.time_ns()) + ".json", smoke)
    runner.run_all()
    return common.audit(run)


def launch(run, model_dir):
    run, model_dir = Path(run).resolve(), Path(model_dir).resolve()
    manifest, status = common.verify_sources(run)
    if status["state"] == "complete":
        return common.audit(run)
    timeout = duration_limit(manifest["config"])
    # Inherited flock file descriptors keep the global lock held if the parent
    # disappears. The child owns no cloud resources and starts no subprocesses.
    with common.process_lock(LOCK) as global_fd, common.process_lock(run / "run.lock") as run_fd:
        command = [sys.executable, "-m", "selfplay.program_local_rl", "_worker", "--run", str(run),
                   "--model-dir", str(model_dir), "--lock-fds", f"{global_fd},{run_fd}"]
        child = subprocess.Popen(command, cwd=ROOT, pass_fds=(global_fd, run_fd), start_new_session=True)
        started = time.monotonic()
        try:
            while child.poll() is None:
                if time.monotonic() - started >= timeout:
                    raise TimeoutError("Hard subprocess deadline")
                usage = subprocess.run(["ps", "-o", "rss=", "-p", str(child.pid)],
                                       capture_output=True, text=True, timeout=5)
                if usage.returncode == 0 and usage.stdout.strip():
                    rss_bytes = int(usage.stdout.strip()) * 1024
                    if rss_bytes > manifest["config"]["memory_limit_bytes"]:
                        raise MemoryError("Resident memory exceeded frozen worker ceiling")
                time.sleep(1)
            code = child.returncode
        except BaseException as error:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
            status = common.load(run / "status.json")
            status.update(state="stopped", error_type=type(error).__name__, updated=common.now(),
                          reason=str(error) + "; incomplete optimizer work is not committed")
            common.atomic_json(run / "status.json", status)
            raise
        if code:
            raise RuntimeError(f"Local smoke worker exited {code}; inspect preserved artifacts/log")
    return common.audit(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--prompt", type=Path, required=True)
    p.add_argument("--condition", choices=sorted(CONDITIONS), default="plain-v1")
    p = sub.add_parser("build-compact")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("index-model")
    p.add_argument("--model-dir", type=Path, required=True)
    for name in ("run", "_worker"):
        p = sub.add_parser(name)
        p.add_argument("--run", type=Path, required=True)
        p.add_argument("--model-dir", type=Path, required=True)
        if name == "_worker":
            p.add_argument("--lock-fds", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.run, args.prompt, args.condition)
    elif args.command == "build-compact":
        print(json.dumps(write_compact_prompt(args.out), indent=2))
    elif args.command == "index-model":
        index_model(args.model_dir)
    elif args.command == "run":
        print(json.dumps(launch(args.run, args.model_dir), indent=2))
    else:
        try:
            global_fd, run_fd = (int(x) for x in args.lock_fds.split(","))
            for fd, path in ((global_fd, LOCK), (run_fd, args.run / "run.lock")):
                if os.fstat(fd).st_ino != os.stat(path).st_ino:
                    raise ValueError("Worker did not inherit its parent neural locks")
            print(json.dumps(worker(args.run, args.model_dir), indent=2))
        except Exception as error:
            status = common.load(args.run / "status.json")
            status.update(state="stopped", error_type=type(error).__name__, reason=str(error), updated=common.now())
            common.atomic_json(args.run / "status.json", status)
            raise


if __name__ == "__main__":
    main()
