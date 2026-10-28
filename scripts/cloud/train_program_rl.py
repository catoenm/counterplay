"""Bounded, single-GPU on-policy controller REINFORCE pilot.

``prepare`` and ``audit`` are CPU-only. ``run`` requires an explicitly staged,
hash-indexed model and CUDA; it never downloads models or calls a provider API.
No generated Python is executed: the existing bounded interpreter scores it.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from selfplay.program_python import compile_python_program, PythonProgramValidationError
from selfplay.program_arena import compile_program, play, verify_replay
from selfplay.program_arena.examples import reference_programs
from program_response_normalization import normalize_final

SOURCES = ["scripts/cloud/train_program_rl.py", "scripts/program_response_normalization.py",
           "selfplay/program_python.py", "selfplay/program_arena/__init__.py",
           "selfplay/program_arena/dsl.py", "selfplay/program_arena/game.py",
           "selfplay/program_arena/runner.py", "selfplay/program_arena/spec.py",
           "selfplay/program_arena/examples.py"]
DEFAULTS = {
    "format_version": 1, "backend": "cuda", "algorithm": "one_step_group_RLOO_REINFORCE",
    "iterations": 2, "group_size": 4, "opponent_count": 2,
    "opponent_attempt_cap": 4, "eval_samples": 2, "map_seeds_per_group": 3,
    "max_rounds": 48, "max_new_tokens": 4096, "sample_seconds": 600,
    "max_context_tokens": 8192, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
    "thinking": False, "learning_rate": 1e-6, "lora_rank": 8, "lora_alpha": 16,
    "lora_dropout": 0.0, "lora_targets": ["q_proj", "v_proj"],
    "gradient_clip": 0.5, "compute_dtype": "float32", "attention": "eager",
    "token_logprob_tolerance": 0.002, "sequence_logprob_tolerance": 0.02,
    "invalid_authorship_reward": -1.0, "seed": 124301,
    "deadline": "2026-10-22T08:00:00-07:00",
    "train_seed_start": 3100000, "holdout_seed_start": 3900000,
    "calibration_seed_start": 3700000,
    "reward": "paired-seat relative game return; invalid authorship=-1 without a game",
    "sampling": "full softmax temperature one; all sampled completion tokens incl actual EOS",
    "scope": "mechanism pilot; current-model frozen program opponents, not population training",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    with temporary.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def admission(config):
    if datetime.now(timezone.utc) >= datetime.fromisoformat(config["deadline"]):
        raise TimeoutError("Deadline reached; no new stage admitted")


@contextmanager
def process_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another controller job holds the process lock") from None
        yield stream.fileno()


def validate_config(config):
    for key in ("iterations", "group_size", "opponent_count", "opponent_attempt_cap",
                "eval_samples", "map_seeds_per_group", "max_new_tokens", "sample_seconds"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError("Invalid bounded count: " + key)
    if config["iterations"] > 2 or config["group_size"] != 4:
        raise ValueError("This pilot permits two updates with groups of four only")
    if config["max_new_tokens"] > 4096 or config["sample_seconds"] > 600:
        raise ValueError("Generation cap exceeds the pilot")
    for key, ceiling in (("opponent_count", 2), ("opponent_attempt_cap", 4),
                         ("eval_samples", 2), ("map_seeds_per_group", 3)):
        if config[key] > ceiling:
            raise ValueError("Pilot count exceeds cap: " + key)
    if config["opponent_count"] > config["opponent_attempt_cap"] or config["max_rounds"] != 48:
        raise ValueError("Opponent admission or game horizon differs")
    if (config["temperature"], config["top_p"], config["top_k"]) != (1.0, 1.0, 0):
        raise ValueError("Only full-softmax sampling is implemented")
    if config["thinking"] or config["compute_dtype"] != "float32" or config["lora_dropout"] != 0:
        raise ValueError("Frozen pilot requires thinking off, FP32 arithmetic and zero dropout")
    if config["loader"] not in {"causal_lm", "image_text_to_text", "mlx_lm"}:
        raise ValueError("Unsupported Transformers model loader")
    if (config["backend"] == "mlx") != (config["loader"] == "mlx_lm"):
        raise ValueError("Backend and loader disagree")
    if config["backend"] not in {"cuda", "mlx"}:
        raise ValueError("Unknown backend")
    for key, maximum in (("learning_rate", 1e-5), ("gradient_clip", 1.0),
                         ("token_logprob_tolerance", .002), ("sequence_logprob_tolerance", .02)):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError("Invalid finite positive pilot bound: " + key)
    for key in ("seed", "train_seed_start", "holdout_seed_start", "calibration_seed_start"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError("Invalid nonnegative integer seed: " + key)
    training = {config["train_seed_start"] + step * 1000 + i
                for step in range(1, config["iterations"] + 1) for i in range(config["map_seeds_per_group"])}
    calibration = {config["calibration_seed_start"] + step * 100 + i
                   for step in range(1, config["iterations"] + 1) for i in range(config["map_seeds_per_group"])}
    heldout = {config["holdout_seed_start"] + i for i in range(config["map_seeds_per_group"])}
    if training & calibration or training & heldout or calibration & heldout:
        raise ValueError("Training/calibration/holdout seed families overlap")
    if len(config["model_revision"]) != 40 or any(c not in "0123456789abcdef" for c in config["model_revision"]):
        raise ValueError("Pin an exact lowercase 40-character model revision")
    if datetime.fromisoformat(config["deadline"]).tzinfo is None:
        raise ValueError("Timezone-aware deadline required")


def prepare(run, prompt, repo, revision, loader="causal_lm", overrides=None):
    run = Path(run)
    if run.exists():
        raise FileExistsError("Run directory already exists; use run to resume")
    config = {**DEFAULTS, "model_repo": repo, "model_revision": revision, "loader": loader,
              **(overrides or {})}
    validate_config(config)
    text = Path(prompt).read_text()
    if not text.strip():
        raise ValueError("A nonempty problem/specification prompt is required")
    run.mkdir(parents=True)
    (run / "prompt.txt").write_text(text)
    hashes = {}
    for relative in SOURCES:
        source, target = ROOT / relative, run / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[relative] = digest(target)
    manifest = {"created": now(), "config": config, "source_sha256": hashes,
                "prompt_sha256": digest(run / "prompt.txt"),
                "external_spend_authorization": "Deployment controller enforces cumulative $100; runner allocates nothing",
                "cuda_validation": "Not executed by prepare; run must pass runtime and fresh capability gates"}
    atomic_json(run / "manifest.json", manifest)
    atomic_json(run / "status.json", {"manifest_sha256": digest(run / "manifest.json"), "artifacts": {},
                                      "state": "prepared", "updated": now()})
    return manifest


def verify_sources(run, require_active=True):
    run = Path(run)
    manifest = load(run / "manifest.json")
    status = load(run / "status.json")
    if status["manifest_sha256"] != digest(run / "manifest.json"):
        raise ValueError("Manifest changed after preparation")
    validate_config(manifest["config"])
    if digest(run / "prompt.txt") != manifest["prompt_sha256"]:
        raise ValueError("Frozen prompt changed")
    for relative, expected in manifest["source_sha256"].items():
        if digest(run / "source" / relative) != expected:
            raise ValueError("Archived source changed: " + relative)
        if require_active and digest(ROOT / relative) != expected:
            raise ValueError("Active source differs; run the verified source bundle")
    for relative, expected in status["artifacts"].items():
        if digest(run / relative) != expected:
            raise ValueError("Committed artifact changed: " + relative)
    return manifest, status


def rloo_advantages(rewards):
    if len(rewards) < 2 or not all(math.isfinite(x) for x in rewards):
        raise ValueError("At least two finite rewards required")
    if max(rewards) - min(rewards) <= 1e-12:
        raise ValueError("Zero reward variance: abort without an update")
    total = sum(rewards)
    return [r - (total - r) / (len(rewards) - 1) for r in rewards]


def completion_positions(prompt_ids, completion_ids):
    """Logit positions predicting completion only; includes a sampled EOS token."""
    if not prompt_ids or not completion_ids:
        raise ValueError("Nonempty prompt and completion required")
    return list(range(len(prompt_ids) - 1, len(prompt_ids) + len(completion_ids) - 1))


def check_likelihoods(recorded, recomputed, token_tolerance, sequence_tolerance):
    if not recorded or len(recorded) != len(recomputed):
        raise ValueError("Completion likelihood coverage differs")
    if not all(math.isfinite(x) and x <= 1e-6 for x in recorded + recomputed):
        raise ValueError("Invalid sampled-token log probability")
    maximum = max(abs(a - b) for a, b in zip(recorded, recomputed))
    sequence = abs(math.fsum(recorded) - math.fsum(recomputed))
    report = {"tokens": len(recorded), "max_token_error": maximum, "sequence_error": sequence,
              "recorded_sequence_logprob": math.fsum(recorded), "passed": False}
    report["passed"] = maximum <= token_tolerance and sequence <= sequence_tolerance
    return report


def classify(text, finish):
    if finish != "eos":
        return {"valid": False, "reason": "generation_" + finish}, None
    # The disabled-thinking template normally supplies its empty think block in
    # the PROMPT. Unexpected thinking in a completion is not treated as code.
    if "<think>" in text or "</think>" in text:
        return {"valid": False, "reason": "unexpected_thinking"}, None
    source, packaging = normalize_final(text)
    try:
        program = compile_python_program(source)
    except PythonProgramValidationError as error:
        return {"valid": False, "reason": "invalid_bounded_python", "detail": str(error),
                "normalization": packaging}, None
    return {"valid": True, "normalization": packaging, "program": program.to_dict()}, program.compiled


def score_program(candidate, opponents, seeds, config):
    classification, program = classify(candidate["text"], candidate["finish_reason"])
    if program is None:
        return {"classification": classification, "reward": config["invalid_authorship_reward"],
                "reward_kind": "invalid_authorship", "matches": [], "game_return": None}
    matches, rewards = [], []
    for opponent in opponents:
        rival = compile_program(opponent["compiled_ir"])
        for seed in seeds:
            for seat in (0, 1):
                programs = [program, rival] if seat == 0 else [rival, program]
                origins = ["model", opponent["origin"]] if seat == 0 else [opponent["origin"], "model"]
                replay = play(programs, seed, config["max_rounds"], origins=origins)
                verify_replay(replay)
                rewards.append(replay["returns"][seat])
                matches.append({"opponent_id": opponent["id"], "candidate_seat": seat, "replay": replay})
    reward = math.fsum(rewards) / len(rewards)
    return {"classification": classification, "reward": reward, "reward_kind": "game_return",
            "game_return": reward, "matches": matches}


def verify_model_index(model_dir, config):
    """Deployment must stage this index; no network or trust_remote_code here."""
    model_dir = Path(model_dir)
    index = load(model_dir / "model-index.json")
    if (index["repo"], index["revision"]) != (config["model_repo"], config["model_revision"]):
        raise ValueError("Staged model identity differs from frozen configuration")
    required = {"config.json", "tokenizer_config.json"}
    if not required <= set(index["files"]) or not any(p.endswith(".safetensors") for p in index["files"]):
        raise ValueError("Model index lacks required files")
    for relative, expected in index["files"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or digest(model_dir / path) != expected:
            raise ValueError("Staged model byte identity failed")
    actual = {str(p.relative_to(model_dir)) for p in model_dir.rglob("*") if p.is_file()
              and not str(p.relative_to(model_dir)).startswith(".cache/") and p.name != "model-index.json"}
    if actual != set(index["files"]):
        raise ValueError("Unindexed or missing model files")
    return {"index_sha256": digest(model_dir / "model-index.json"), **index}


class CudaPolicy:
    """Torch is imported only by the explicit CUDA execution path."""
    def __init__(self, model_dir, config):
        import torch
        import transformers
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        self.torch, self.config = torch, config
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one visible CUDA GPU required")
        if torch.cuda.get_device_properties(0).total_memory < 70 * 1024**3:
            raise RuntimeError("This unvalidated pilot targets an 80 GB GPU")
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir, local_files_only=True,
                                                                      trust_remote_code=False)
        quantization = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float32)
        loader = (transformers.AutoModelForCausalLM if config["loader"] == "causal_lm"
                  else transformers.AutoModelForImageTextToText)
        model = loader.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False,
                   quantization_config=quantization, torch_dtype=torch.float32,
                   attn_implementation=config["attention"], device_map={"": 0})
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model = get_peft_model(model, LoraConfig(r=config["lora_rank"], lora_alpha=config["lora_alpha"],
                    target_modules=config["lora_targets"], lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
        self.parameters = {name: p for name, p in self.model.named_parameters() if p.requires_grad}
        if not self.parameters or any(p.dtype != torch.float32 for p in self.parameters.values()):
            raise ValueError("Expected nonempty FP32 LoRA parameters")
        self.optimizer = torch.optim.AdamW(self.parameters.values(), lr=config["learning_rate"], weight_decay=0)
        self.model.eval()  # Gradients still flow in eval; all dropout remains disabled.
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout) and module.p:
                raise ValueError("Nonzero dropout would change the rollout policy")
        ids = self.model.generation_config.eos_token_id
        self.eos = sorted(set(ids if isinstance(ids, list) else [ids]))
        if not self.eos or any(type(x) is not int for x in self.eos):
            raise ValueError("Model has no explicit EOS ids")
        self.runtime = {"versions": {p: importlib.metadata.version(p) for p in
                                     ("torch", "transformers", "peft", "bitsandbytes", "accelerate")},
                        "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                        "eos_token_ids": self.eos, "trainable_parameters": sum(p.numel() for p in self.parameters.values()),
                        "arithmetic": "NF4 frozen base; FP32 compute/LoRA; TF32 off; eager attention",
                        "loader": config["loader"]}

    def template(self, prompt):
        rendered = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                             tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return rendered, self.tokenizer.encode(rendered, add_special_tokens=False)

    def sample(self, prompt, seed):
        torch, config = self.torch, self.config
        rendered, prompt_ids = self.template(prompt)
        if len(prompt_ids) + config["max_new_tokens"] > config["max_context_tokens"]:
            raise ValueError("Prompt exceeds frozen context admission bound")
        generator = torch.Generator(device="cuda").manual_seed(seed)
        ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
        sampled, logps, cache = [], [], None
        started = time.monotonic()
        finish = "length"
        self.model.eval()
        with torch.no_grad():
            for _ in range(config["max_new_tokens"]):
                if time.monotonic() - started >= config["sample_seconds"]:
                    finish = "time"
                    break
                output = self.model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
                logits = output.logits[0, -1].float()
                lp = torch.log_softmax(logits, dim=-1)
                token = int(torch.multinomial(lp.exp(), 1, generator=generator).item())
                sampled.append(token)
                logps.append(float(lp[token].item()))
                cache = output.past_key_values
                del output, logits, lp
                if token in self.eos:
                    finish = "eos"
                    break
                ids = torch.tensor([[token]], dtype=torch.long, device="cuda")
        del cache
        return {"text": self.tokenizer.decode(sampled, skip_special_tokens=True),
                "raw_text": self.tokenizer.decode(sampled, skip_special_tokens=False),
                "prompt_template": rendered, "prompt_ids": prompt_ids, "completion_ids": sampled,
                "sampled_token_logprobs": logps, "seed": seed, "finish_reason": finish,
                "input_tokens": len(prompt_ids), "output_tokens": len(sampled),
                "elapsed_seconds": time.monotonic() - started, "eos_token_ids": self.eos}

    def token_logps(self, record):
        torch = self.torch
        prompt, completion = record["prompt_ids"], record["completion_ids"]
        positions = completion_positions(prompt, completion)
        ids = torch.tensor([prompt + completion[:-1]], dtype=torch.long, device="cuda")
        # No padding and no prompt labels. All completion tokens contribute,
        # including actual EOS, reasoning/packaging tokens and invalid programs.
        output = self.model(input_ids=ids, use_cache=False)
        selected = output.logits[0, positions].float()
        labels = torch.tensor(completion, dtype=torch.long, device="cuda")
        return -torch.nn.functional.cross_entropy(selected, labels, reduction="none")

    def likelihood_gate(self, records):
        reports = []
        self.model.eval()
        with self.torch.no_grad():
            for record in records:
                recomputed = self.token_logps(record).cpu().tolist()
                reports.append(check_likelihoods(record["sampled_token_logprobs"], recomputed,
                            self.config["token_logprob_tolerance"], self.config["sequence_logprob_tolerance"]))
        return {"passed": all(r["passed"] for r in reports), "records": reports}

    def update(self, records, advantages):
        torch = self.torch
        before = {name: p.detach().cpu().clone() for name, p in self.parameters.items()}
        self.optimizer.zero_grad(set_to_none=True)
        losses, checks = [], []
        # Gradient checkpointing is activated in train mode. Zero dropout and the
        # second gate below ensure this has not changed the sampled policy.
        self.model.train()
        for record, advantage in zip(records, advantages):
            token_logps = self.token_logps(record)
            check = check_likelihoods(record["sampled_token_logprobs"], token_logps.detach().cpu().tolist(),
                                      self.config["token_logprob_tolerance"], self.config["sequence_logprob_tolerance"])
            if not check["passed"]:
                self.optimizer.zero_grad(set_to_none=True)
                self.model.eval()
                raise ValueError("Train-mode likelihood gate failed before optimizer step")
            checks.append(check)
            loss = -float(advantage) * token_logps.sum() / len(records)
            loss.backward()
            losses.append(float(loss.detach().cpu()))
            del loss, token_logps
        norm = torch.nn.utils.clip_grad_norm_(list(self.parameters.values()), self.config["gradient_clip"],
                                              error_if_nonfinite=True)
        if not math.isfinite(float(norm)) or float(norm) <= 0:
            raise ValueError("Zero or invalid policy gradient; no update")
        self.optimizer.step()
        self.model.eval()
        squared = 0.0
        for name, p in self.parameters.items():
            if not bool(torch.isfinite(p).all()):
                raise ValueError("Nonfinite updated adapter")
            squared += float((p.detach().cpu().double() - before[name].double()).square().sum())
        delta = math.sqrt(squared)
        if delta <= 0:
            raise ValueError("Optimizer did not change any adapter weights")
        return {"loss": sum(losses), "microbatch_losses": losses, "gradient_norm_before_clip": float(norm),
                "adapter_l2_delta": delta, "optimizer_steps": 1,
                "gradient_likelihood_gate": {"passed": True, "records": checks},
                "peak_cuda_bytes": torch.cuda.max_memory_allocated()}

    def save_checkpoint(self, folder):
        from safetensors.torch import save_file
        folder.mkdir(parents=True, exist_ok=False)
        save_file({name: p.detach().cpu().contiguous() for name, p in self.parameters.items()},
                  str(folder / "adapter.safetensors"))
        self.torch.save({"optimizer": self.optimizer.state_dict(), "rng": self.torch.get_rng_state(),
                         "cuda_rng": self.torch.cuda.get_rng_state_all()}, folder / "optimizer.pt")
        return {name: digest(folder / name) for name in ("adapter.safetensors", "optimizer.pt")}

    def load_checkpoint(self, folder, hashes):
        from safetensors.torch import load_file
        for name, expected in hashes.items():
            if digest(folder / name) != expected:
                raise ValueError("Checkpoint bytes changed")
        tensors = load_file(str(folder / "adapter.safetensors"), device="cpu")
        if set(tensors) != set(self.parameters):
            raise ValueError("Adapter tensor names differ")
        with self.torch.no_grad():
            for name, p in self.parameters.items():
                value = tensors[name]
                if value.shape != p.shape or value.dtype != p.dtype or not bool(self.torch.isfinite(value).all()):
                    raise ValueError("Adapter tensor shape/dtype/finiteness differs")
                p.copy_(value.to(p.device))
                if not self.torch.equal(p.detach().cpu(), value):
                    raise ValueError("Adapter restoration is not exact")
        state = self.torch.load(folder / "optimizer.pt", map_location="cpu", weights_only=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.torch.set_rng_state(state["rng"])
        self.torch.cuda.set_rng_state_all(state["cuda_rng"])
        self.model.eval()


class Runner:
    def __init__(self, run, backend):
        self.run = Path(run)
        self.manifest, self.status = verify_sources(run)
        self.config, self.backend = self.manifest["config"], backend
        self.prompt = (self.run / "prompt.txt").read_text()

    def existing(self, relative):
        path = self.run / relative
        if relative in self.status["artifacts"]:
            if digest(path) != self.status["artifacts"][relative]:
                raise ValueError("Committed artifact changed")
            return load(path)
        if path.exists():
            path.rename(path.with_name(path.name + ".uncommitted-" + uuid.uuid4().hex))
        return None

    def commit(self, relative, value):
        if relative in self.status["artifacts"]:
            raise ValueError("Refusing to overwrite a committed artifact")
        path = self.run / relative
        atomic_json(path, value)
        self.status["artifacts"][relative] = digest(path)
        self.status.update(updated=now(), state="running")
        atomic_json(self.run / "status.json", self.status)
        return value

    def checkpoint(self, label, parent=None, report=None):
        relative = "checkpoints/" + label + "/metadata.json"
        existing = self.existing(relative)
        if existing:
            self.backend.load_checkpoint(self.run / "checkpoints" / label, existing["files"])
            return existing
        folder = self.run / "checkpoints" / label
        if folder.exists():
            folder.rename(folder.with_name(label + ".incomplete-" + uuid.uuid4().hex))
        files = self.backend.save_checkpoint(folder)
        for name, expected in files.items():
            self.status["artifacts"][str((folder / name).relative_to(self.run))] = expected
        return self.commit(relative, {"label": label, "parent": parent, "files": files, "report": report,
                          "model_repo": self.config["model_repo"], "model_revision": self.config["model_revision"],
                          "manifest_sha256": self.status["manifest_sha256"], "complete": True})

    def generation(self, name, seed, checkpoint):
        relative = "generations/" + name + ".json"
        record = self.existing(relative)
        if record is None:
            admission(self.config)
            record = {**self.backend.sample(self.prompt, seed), "name": name, "checkpoint": checkpoint,
                      "model_repo": self.config["model_repo"], "model_revision": self.config["model_revision"],
                      "authorship": "fresh_learner_sample", "api_training_data": False}
            self.commit(relative, record)
        if record["seed"] != seed or record["checkpoint"] != checkpoint:
            raise ValueError("Generation source checkpoint/seed mismatch")
        return record

    def opponents(self, step, checkpoint):
        relative = f"groups/{step}/opponents.json"
        existing = self.existing(relative)
        if existing is not None:
            return existing
        selected, attempts = [], []
        for i in range(self.config["opponent_attempt_cap"]):
            row = self.generation(f"opponent-{step}-{i}", self.config["seed"] + step * 100 + i, checkpoint)
            classification, program = classify(row["text"], row["finish_reason"])
            behavior = None
            useful = False
            if program:
                references = [{"id": "nearest", "origin": "scripted",
                               "compiled_ir": reference_programs()["nearest"]}]
                seeds = [self.config["calibration_seed_start"] + step * 100 + j
                         for j in range(self.config["map_seeds_per_group"])]
                behavior = score_program(row, references, seeds, self.config)
                banked = [m["replay"]["metrics"]["scores"][m["candidate_seat"]] for m in behavior["matches"]]
                # This is a deliberately modest usability gate, not evidence of
                # beating the heuristic or transferable model intelligence.
                useful = sum(x > 0 for x in banked) >= len(banked) // 2
                if useful:
                    selected.append({"id": row["name"], "origin": "model", "compiled_ir": json.loads(program.source),
                                     "source_checkpoint": checkpoint, "generation_sha256":
                                     self.status["artifacts"]["generations/" + row["name"] + ".json"]})
            attempts.append({"name": row["name"], "classification": classification,
                             "useful": useful, "behavior": behavior})
            if len(selected) == self.config["opponent_count"]:
                break
        result = self.commit(relative, {"selected": selected, "attempts": attempts, "source_checkpoint": checkpoint,
                              "passed": len(selected) == self.config["opponent_count"]})
        return result

    def evaluate(self, checkpoint, initial_opponents):
        relative = f"evaluation/{checkpoint['label']}.json"
        if self.existing(relative) is not None:
            return
        references = [{"id": name, "origin": "scripted", "compiled_ir": source}
                      for name, source in reference_programs().items() if name in {"nearest", "value"}]
        opponents = initial_opponents + references
        seeds = [self.config["holdout_seed_start"] + i for i in range(self.config["map_seeds_per_group"])]
        rows = []
        for i in range(self.config["eval_samples"]):
            # Common sampling seeds across checkpoints are held out from training.
            record = self.generation(f"eval-{checkpoint['label']}-{i}", self.config["seed"] + 90000 + i,
                                     checkpoint)
            rows.append({"generation": record["name"], **score_program(record, opponents, seeds, self.config)})
        self.commit(relative, {"checkpoint": checkpoint, "opponents": opponents, "seeds": seeds,
                              "scores": rows, "training_use": False,
                              "caveat": "Two program draws and three paired maps; not a general improvement claim"})

    def run_all(self):
        base = self.checkpoint("base")
        first = self.opponents(1, base)
        if not first["passed"]:
            raise ValueError("Fresh CUDA capability gate failed: insufficient valid opponent programs")
        frozen = first["selected"]
        self.evaluate(base, frozen)
        source = base
        for step in range(1, self.config["iterations"] + 1):
            label = f"step{step}"
            target = self.existing("checkpoints/" + label + "/metadata.json")
            if target is not None:
                if target["parent"] != source:
                    raise ValueError("Checkpoint lineage differs")
                self.backend.load_checkpoint(self.run / "checkpoints" / label, target["files"])
            else:
                self.backend.load_checkpoint(self.run / "checkpoints" / source["label"], source["files"])
                rivals = self.opponents(step, source)
                if not rivals["passed"]:
                    raise ValueError("Current learner failed fresh opponent capability gate")
                records = [self.generation(f"candidate-{step}-{i}", self.config["seed"] + 20000 + step * 100 + i, source)
                           for i in range(self.config["group_size"])]
                relative = f"groups/{step}/scores.json"
                scores = self.existing(relative)
                if scores is None:
                    seeds = [self.config["train_seed_start"] + step * 1000 + i
                             for i in range(self.config["map_seeds_per_group"])]
                    scores = self.commit(relative, {"source": source, "opponents": rivals["selected"], "seeds": seeds,
                          "rows": [{"generation": row["name"], **score_program(row, rivals["selected"], seeds, self.config)}
                                   for row in records]})
                advantages = rloo_advantages([row["reward"] for row in scores["rows"]])
                admission(self.config)
                gate = self.backend.likelihood_gate(records)
                gate["generation_names"] = [r["name"] for r in records]
                # Gate attempts are durable even on failure/restart, never replaced.
                gate_path = f"groups/{step}/gate-{uuid.uuid4().hex}.json"
                self.commit(gate_path, gate)
                if not gate["passed"]:
                    raise ValueError("Cached-sampling / teacher-forced likelihood mismatch; no update")
                admission(self.config)
                report = self.backend.update(records, advantages)
                report["gradient_likelihood_gate"]["generation_names"] = [r["name"] for r in records]
                report.update(advantages=advantages, rewards=[r["reward"] for r in scores["rows"]],
                              training_tokens=sum(len(r["completion_ids"]) for r in records),
                              scores_sha256=self.status["artifacts"][relative],
                              likelihood_gate_path=gate_path,
                              likelihood_gate_sha256=self.status["artifacts"][gate_path],
                              generation_sha256={r["name"]: self.status["artifacts"]["generations/" + r["name"] + ".json"]
                                                 for r in records})
                target = self.checkpoint(label, source, report)
            self.evaluate(target, frozen)
            source = target
        self.status.update(state="complete", updated=now(), final_checkpoint=source["label"])
        atomic_json(self.run / "status.json", self.status)


def audit(run):
    # Replay uses the local interpreter, so require it to match the archive.
    manifest, status = verify_sources(run, require_active=True)
    matches, generations, checkpoints = 0, 0, []
    for relative in status["artifacts"]:
        if not relative.endswith(".json"):
            continue
        row = load(Path(run) / relative)
        if relative.startswith("generations/"):
            generations += 1
            if len(row["completion_ids"]) != len(row["sampled_token_logprobs"]):
                raise ValueError("Missing sampled token likelihoods")
            if row["finish_reason"] == "eos" and row["completion_ids"][-1] not in row["eos_token_ids"]:
                raise ValueError("EOS finish lacks sampled EOS")
        if relative.endswith("/scores.json") or relative.startswith("evaluation/"):
            for scored in row.get("rows", row.get("scores", [])):
                generation = load(Path(run) / "generations" / (scored["generation"] + ".json"))
                expected = {"generation": scored["generation"],
                            **score_program(generation, row["opponents"], row["seeds"], manifest["config"])}
                if scored != expected:
                    raise ValueError("Reward/classification/matches differ from independent scoring")
                if scored["reward_kind"] == "invalid_authorship" and scored["matches"]:
                    raise ValueError("Invalid program has fabricated game outcomes")
                for match in scored["matches"]:
                    verify_replay(match["replay"])
                    matches += 1
        if relative.endswith("/metadata.json"):
            checkpoints.append(row["label"])
            if row["report"] and (row["report"]["adapter_l2_delta"] <= 0 or row["report"]["optimizer_steps"] != 1):
                raise ValueError("Checkpoint lacks a real single policy-gradient update")
            if row["report"]:
                from safetensors.numpy import load_file
                import numpy as np
                report = row["report"]
                step = int(row["label"].removeprefix("step"))
                if report["scores_sha256"] != digest(Path(run) / f"groups/{step}/scores.json"):
                    raise ValueError("Update not bound to recorded rewards")
                for name, expected in report["generation_sha256"].items():
                    if digest(Path(run) / "generations" / (name + ".json")) != expected:
                        raise ValueError("Update not bound to sampled completions")
                scores = load(Path(run) / f"groups/{step}/scores.json")
                names = [r["generation"] for r in scores["rows"]]
                if len(names) != manifest["config"]["group_size"] or set(names) != set(report["generation_sha256"]):
                    raise ValueError("Update group coverage differs")
                rewards = [r["reward"] for r in scores["rows"]]
                if report["rewards"] != rewards or report["advantages"] != rloo_advantages(rewards):
                    raise ValueError("Update reward/advantage provenance differs")
                gate_path = report["likelihood_gate_path"]
                if (status["artifacts"].get(gate_path) != report["likelihood_gate_sha256"]
                    or not gate_path.startswith(f"groups/{step}/gate-")):
                    raise ValueError("Update lacks its committed successful likelihood gate")
                gates = [load(Path(run) / gate_path), report["gradient_likelihood_gate"]]
                for gate in gates:
                    if not gate["passed"] or gate["generation_names"] != names or len(gate["records"]) != len(names):
                        raise ValueError("Successful gate coverage differs from training group")
                    for name, check in zip(names, gate["records"]):
                        generation = load(Path(run) / "generations" / (name + ".json"))
                        if generation["checkpoint"] != row["parent"]:
                            raise ValueError("Sampled policy differs from update source")
                        if check["tokens"] != len(generation["completion_ids"]) or not check["passed"]:
                            raise ValueError("Gate omitted sampled completion tokens")
                        for key, tolerance in (("max_token_error", "token_logprob_tolerance"),
                                               ("sequence_error", "sequence_logprob_tolerance")):
                            if not math.isfinite(check[key]) or not 0 <= check[key] <= manifest["config"][tolerance]:
                                raise ValueError("Recorded numerical gate exceeds frozen tolerance")
                        if check["recorded_sequence_logprob"] != math.fsum(generation["sampled_token_logprobs"]):
                            raise ValueError("Gate is not bound to sampled sequence likelihood")
                before = load_file(str(Path(run) / "checkpoints" / row["parent"]["label"] / "adapter.safetensors"))
                after = load_file(str(Path(run) / "checkpoints" / row["label"] / "adapter.safetensors"))
                if set(before) != set(after) or not all(np.isfinite(v).all() and v.dtype == np.float32
                    and v.shape == before[k].shape for k, v in after.items()):
                    raise ValueError("Updated adapter tensor set/finiteness differs")
                delta = math.sqrt(sum(float(np.square(after[k].astype("float64") - v.astype("float64")).sum())
                                      for k, v in before.items()))
                if delta <= 0 or not math.isclose(delta, report["adapter_l2_delta"], rel_tol=1e-6, abs_tol=1e-10):
                    raise ValueError("Actual adapter delta differs from report")
    return {"complete": status["state"] == "complete", "generations": generations,
            "replayed_matches": matches, "checkpoints": checkpoints,
            "source_archive_verified": True, "artifact_hashes_verified": True,
            "CUDA_likelihood_checks": "Recorded gates; CPU audit cannot rerun the neural model",
            "config": manifest["config"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--run", type=Path, required=True)
    preparation.add_argument("--prompt", type=Path, required=True)
    preparation.add_argument("--repo", required=True)
    preparation.add_argument("--revision", required=True)
    preparation.add_argument("--loader", choices=["causal_lm", "image_text_to_text"], default="causal_lm")
    execution = commands.add_parser("run")
    execution.add_argument("--run", type=Path, required=True)
    execution.add_argument("--model-dir", type=Path, required=True)
    execution.add_argument("--lock", type=Path, default=Path("/workspace/program-rl/cuda.lock"))
    checking = commands.add_parser("audit")
    checking.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.run, args.prompt, args.repo, args.revision, args.loader)
    elif args.command == "audit":
        print(json.dumps(audit(args.run), indent=2))
    else:
        with process_lock(args.lock), process_lock(args.run / "run.lock"):
            manifest, status = verify_sources(args.run)
            if manifest["config"]["backend"] != "cuda":
                raise ValueError("Use the dedicated local runner for MLX")
            if status["state"] == "complete":
                print(json.dumps(audit(args.run), indent=2))
                return
            admission(manifest["config"])
            model_index = verify_model_index(args.model_dir, manifest["config"])
            backend = CudaPolicy(args.model_dir, manifest["config"])
            runner = Runner(args.run, backend)
            old = runner.existing("runtime.json")
            runtime = {"model_index": model_index, **backend.runtime}
            if old is None:
                runner.commit("runtime.json", runtime)
            elif old != runtime:
                raise ValueError("Runtime/model identity differs from original execution")
            try:
                runner.run_all()
            except Exception as error:
                runner.status.update(state="stopped", error_type=type(error).__name__, reason=str(error), updated=now())
                atomic_json(args.run / "status.json", runner.status)
                raise


if __name__ == "__main__":
    main()
