"""Shared, bounded-lag legal-action policy. Importing this file loads no model.

An actor and learner must be separate instances in separate processes. A sealed
cohort has one behavior checkpoint; its numerical gate replays that checkpoint,
whereas the learner's PPO ratio intentionally may differ at policy lag one.
One accumulated Adam step covers the complete weighted cohort, never one step
per microbatch. The checkpoint version counts cohorts; Adam's step counts actual
optimizer updates. No free-form text or reasoning tokens are optimized.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import string
import time

from selfplay import program_cardinal_cuda as identity

MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
FILES = (*identity.CHECKPOINT_FILES, "policy.json")


def canonical(value):
    # Match durable JSON: integer object keys become strings on disk. Sorting
    # only before that conversion changes prompts/hashes after a round trip.
    value = json.loads(json.dumps(value, allow_nan=False))
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_input_identity(folder):
    folder = Path(folder)
    return {"model_index_sha256": file_sha(folder/"model.safetensors.index.json"),
        "tokenizer_files_sha256": digest({n:file_sha(folder/n) for n in
            ("tokenizer.json", "tokenizer_config.json")})}


def action_prompt(observation, actions, labels):
    """The public observation alone determines single-player/competitive wording."""
    roles = observation.get("role_count", 2)
    if type(roles) is not int or roles not in (1,2):
        raise ValueError("Observation role count must be one or two")
    introduction = ("You are solving a single-player task. " if roles==1 else
                    "You are playing a competitive game. The opponent chooses its own actions. ")
    return (introduction+"Maximize the final reward described in the rules. Use only your observation. "
        "Choose one action from the legal menu. Reply with only its exact label.\nObservation: "+
        canonical(observation)+"\nLegal actions:\n"+
        "\n".join(f"{labels[i]}: "+canonical(a) for i,a in enumerate(actions)))


def select_labels(tokenizer, maximum=256):
    """The original arena's stable label enumeration and exact round-trip rule."""
    candidates = list(string.ascii_uppercase + string.ascii_lowercase)
    candidates += sorted((s for s in tokenizer.get_vocab() if s.isascii() and s.isalpha()
                          and 2 <= len(s) <= 6), key=lambda s: (len(s), s))
    labels, ids = [], []
    for label in candidates:
        tokens = tokenizer.encode(label, add_special_tokens=False)
        if len(tokens) == 1 and tokens[0] not in ids and tokenizer.decode(tokens) == label:
            labels.append(label); ids.append(tokens[0])
        if len(labels) == maximum:
            return labels, ids
    raise ValueError("Insufficient distinct round-tripping one-token action labels")


def validate_config(config):
    if config.get("model") != MODEL or config.get("revision") != REVISION:
        raise ValueError("Expected the pinned original Qwen3-8B model")
    weights = config.get("environment_weights", {})
    if (not isinstance(weights, dict) or not weights
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in weights.values())
            or not math.isclose(sum(weights.values()), 1., abs_tol=1e-12)):
        raise ValueError("Environment weights must be positive and sum to one")
    for key in ("learning_rate", "gradient_clip", "likelihood_tolerance"):
        value = config.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid positive configuration: " + key)
    for key in ("entropy_coefficient", "kl_coefficient"):
        value = config.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid nonnegative configuration: " + key)
    if config.get("clip_ratio") != .2 or config.get("maximum_policy_lag") != 1:
        raise ValueError("Expected clip .2 and maximum policy lag one")
    for key in ("max_context_tokens", "inference_batch_size", "microbatch_size"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError("Invalid integer configuration: " + key)
    if config.get("thinking", False) is not False:
        raise ValueError("This is a thinking-disabled one-token action policy")
    return deepcopy(config)


def default_config(environment_weights, **overrides):
    return validate_config({"model": MODEL, "revision": REVISION, "seed": 731001,
        "learning_rate": 2e-5, "gradient_clip": 1., "likelihood_tolerance": .002,
        "clip_ratio": .2, "entropy_coefficient": .01, "kl_coefficient": .02,
        "maximum_policy_lag": 1, "max_context_tokens": 4096,
        "inference_batch_size": 8, "microbatch_size": 2, "thinking": False,
        "environment_weights": environment_weights, **overrides})


def validate_binding(binding):
    if (not isinstance(binding, dict)
            or set(binding) != {"checkpoint_id", "version", "adapter_sha256", "base_identity"}
            or type(binding["version"]) is not int or binding["version"] < 0
            or binding["checkpoint_id"] != ("base" if binding["version"] == 0 else f"cycle{binding['version']}")
            or not isinstance(binding["base_identity"], dict)):
        raise ValueError("Invalid immutable checkpoint binding")
    h = binding["adapter_sha256"]
    if not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
        raise ValueError("Invalid adapter byte hash")
    return binding


def policy_lag(current, behavior):
    validate_binding(current); validate_binding(behavior)
    if current["base_identity"] != behavior["base_identity"]:
        raise ValueError("Actor and learner base identities differ")
    lag = current["version"] - behavior["version"]
    if lag not in (0, 1):
        raise ValueError("Behavior policy lag must be zero or one")
    if lag == 0 and current != behavior:
        raise ValueError("Same-version checkpoint bytes differ")
    return lag


def objective(logps, old_logps, counts, picks, advantages, weights, config):
    """Sum weighted action terms; caller accumulates microbatches, no length mean."""
    import torch
    mask = torch.arange(logps.shape[1], device=logps.device)[None, :] < counts[:, None]
    selected = logps.gather(1, picks[:, None]).squeeze(1)
    old_selected = old_logps.gather(1, picks[:, None]).squeeze(1)
    ratio = (selected - old_selected).exp()
    surrogate = torch.minimum(ratio * advantages, ratio.clamp(1-config["clip_ratio"], 1+config["clip_ratio"]) * advantages)
    entropy = -torch.where(mask, logps.exp() * logps, 0).sum(-1)
    kl = torch.where(mask, old_logps.exp() * (old_logps-logps), 0).sum(-1)
    policy_loss = -(weights * surrogate).sum()
    entropy_loss = -(weights * config["entropy_coefficient"] * entropy).sum()
    kl_loss = (weights * config["kl_coefficient"] * kl).sum()
    return policy_loss + entropy_loss + kl_loss, {"policy_loss": policy_loss,
        "entropy_loss": entropy_loss, "kl_loss": kl_loss,
        "weighted_entropy": (weights*entropy).sum(), "weighted_kl": (weights*kl).sum(),
        "weighted_clip_fraction": (weights*((ratio-1).abs()>.2)).sum()}


def cohort_rows(episodes, environment_weights):
    """Independent episode/role LOO; explicit learner roles support solo tasks."""
    if not episodes or len({e["episode_id"] for e in episodes}) != len(episodes):
        raise ValueError("Missing or duplicate cohort episodes")
    if set(e["environment"] for e in episodes) != set(environment_weights):
        raise ValueError("Every configured environment must occur in a cohort")
    counts, returns, families = {}, {}, set()
    for e in episodes:
        env, seats = e["environment"], e["learner_seats"]
        family = e.get("scenario_family_id")
        if not isinstance(family, str) or not family or (env,family) in families:
            raise ValueError("Independent scenario families must be unique within each environment")
        families.add((env,family))
        if (not isinstance(seats, list) or not seats or len(set(seats)) != len(seats)
                or any(type(p) is not int or p not in (0, 1) for p in seats)):
            raise ValueError("Explicit learner seats must be [0], [1] or [0,1]")
        counts[env] = counts.get(env, 0) + 1
        if not isinstance(e["returns"], list) or len(e["returns"]) != 2:
            raise ValueError("Expected two return slots including inactive zero slot")
        for p in seats:
            reward = e["returns"][p]
            if type(reward) not in (int, float) or not math.isfinite(reward) or not -1 <= reward <= 1:
                raise ValueError("Invalid bounded terminal reward")
            returns.setdefault((env, p), []).append((e["episode_id"], reward))
    if any(n < 2 for n in counts.values()) or any(len(v) < 2 for v in returns.values()):
        raise ValueError("LOO requires at least two independent episodes per environment/learner role")
    rows = []
    for e in episodes:
        env, seats = e["environment"], e["learner_seats"]
        weight = environment_weights[env] / counts[env] / len(seats)
        seen = set()
        for row in e["transitions"]:
            p = row["player"]
            if p not in seats:
                raise ValueError("Opponent transitions cannot be trained as learner data")
            if (row["episode_id"] != e["episode_id"] or row["environment"] != env
                    or row["scenario_family_id"] != e["scenario_family_id"]):
                raise ValueError("Transition episode/environment differs")
            key = (p, row["turn"])
            if key in seen:
                raise ValueError("Duplicate player-turn transition")
            seen.add(key)
            if row["terminal_return"] != e["returns"][p]:
                raise ValueError("Transition terminal reward differs")
            others = [r for name, r in returns[(env, p)] if name != e["episode_id"]]
            rows.append({**deepcopy(row), "advantage": e["returns"][p]-sum(others)/len(others), "weight": weight})
        if not e["transitions"]:
            raise ValueError("Unplayed episode cannot become training data")
    return rows


def checkpoint_state(folder, expected_learning_rate=None):
    folder = Path(folder)
    state = identity.checkpoint_state(folder, expected_learning_rate)
    meta = json.loads((folder/"policy.json").read_text())
    validate_binding(meta["binding"])
    if (meta.get("kind") != "mixed_arena_action_policy_v1"
            or meta["binding"]["adapter_sha256"] != state["files"]["adapter.safetensors"]
            or meta["binding"]["base_identity"]["frozen_state_sha256"] != state["frozen_state_sha256"]
            or meta["optimizer_sha256"] != state["files"]["optimizer.pt"]
            or meta["config_sha256"] != digest(meta["config"])
            or meta["labels_sha256"] != digest([meta["labels"], meta["label_token_ids"]])):
        raise ValueError("Checkpoint policy metadata binding differs")
    validate_config(meta["config"])
    return {**state, "files": {**state["files"], "policy.json": file_sha(folder/"policy.json")}, "policy": meta}


def compare_actor_reports(actor, learner, tolerance=.002):
    """CPU comparison of independent processes at the same immutable checkpoint."""
    if (actor.get("passed") is not True or learner.get("passed") is not True
            or actor["binding"] != learner["binding"] or not actor.get("rows")
            or len(actor["rows"]) != len(learner.get("rows",[]))):
        raise ValueError("Two successful same-checkpoint admission reports required")
    checked=[]
    for a,b in zip(actor["rows"],learner["rows"],strict=True):
        for key in ("checkpoint","input_ids","menu_sha256","sample_seed","action_index","action_token_id","prompt_sha256"):
            if a[key]!=b[key]: raise ValueError("Cross-process actor identity differs: "+key)
        error=max(abs(x-y) for x,y in zip(a["old_log_probs"],b["old_log_probs"],strict=True))
        checked.append({"actor_row_sha256":digest(a),"learner_row_sha256":digest(b),"max_error":error,
            "passed":math.isfinite(error) and error<=tolerance})
    return {"kind":"mixed_arena_cross_process_admission", "actor_sha256":digest(actor),
        "learner_sha256":digest(learner), "binding":actor["binding"], "tolerance":tolerance,
        "rows":checked,"passed":all(r["passed"] for r in checked)}


class CudaActionPolicy(identity.CardinalCudaPolicy):
    def __init__(self, model_dir, config, role="learner"):
        import torch
        import transformers
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if role not in {"actor", "learner"}:
            raise ValueError("Role must be actor or learner")
        self.torch, self.config, self.role = torch, validate_config(config), role
        if any(config.get(k)!=v for k,v in local_input_identity(model_dir).items()):
            raise ValueError("Local model index/tokenizer files differ from frozen configuration")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one visible CUDA device required")
        torch.manual_seed(config["seed"]); torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        self.device = "cuda"
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        model_config = transformers.AutoConfig.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        if model_config.model_type != "qwen3" or model_config.num_hidden_layers < 8:
            raise ValueError("Expected original Qwen3 causal architecture")
        quant = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float32)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True,
            trust_remote_code=False, quantization_config=quant, torch_dtype=torch.float32,
            attn_implementation="eager", device_map={"": 0})
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
        layers = list(range(model_config.num_hidden_layers-8, model_config.num_hidden_layers))
        self.model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.,
            target_modules=["q_proj", "v_proj"], layers_to_transform=layers,
            layers_pattern="layers", bias="none", task_type="CAUSAL_LM"))
        self.parameters = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        if not self.parameters or any(p.dtype != torch.float32 for p in self.parameters.values()):
            raise ValueError("Expected nonempty FP32 LoRA parameters")
        self.optimizer = torch.optim.Adam(self.parameters.values(), lr=config["learning_rate"])
        self.model.eval()
        if any(isinstance(m, torch.nn.Dropout) and m.p for m in self.model.modules()):
            raise ValueError("Nonzero dropout changes the behavior policy")
        self.labels, self.label_ids = select_labels(self.tokenizer)
        self.runtime = {"versions": {p: importlib.metadata.version(p) for p in
            ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "safetensors", "tokenizers")},
            "role": role, "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
            "arithmetic": "NF4 frozen packed base; FP32 compute/adapters; eager attention; TF32 off",
            "thinking": False, "lora_last_layers": layers, "lora_rank": 8, "lora_alpha": 16,
            "trainable_parameters": sum(p.numel() for p in self.parameters.values())}
        self._initialize_identity()
        self.runtime["sampling_batch"] = "Independent legal categorical draws; variable-length public prompts; one token per action"
        actual_layers = sorted({int(n.split(".layers.")[1].split(".")[0]) for n in self.parameters})
        if actual_layers != layers or self.runtime["full_attention_lora_module_counts"] != {"q_proj": 8, "v_proj": 8}:
            raise ValueError("LoRA did not target exactly the final eight q/v layers")
        self.base_identity = {"model": MODEL, "revision": REVISION,
            "model_index_sha256": config["model_index_sha256"],
            "tokenizer_files_sha256": config["tokenizer_files_sha256"],
            "quantization": "nf4-double-fp32", "frozen_state_sha256": self._frozen["sha256"]}
        self.binding, self.poisoned, self.pending_version, self.bound_adapter = None, False, None, None

    def adapter_identity(self):
        return {n:identity.tensor_identity(p) for n,p in self.parameters.items()}

    def _assert_bound(self, binding):
        validate_binding(binding)
        if self.poisoned or self.binding != binding:
            raise ValueError("Policy is poisoned or exact requested checkpoint is not loaded")
        if self.bound_adapter != self.adapter_identity():
            raise ValueError("Live adapter differs from its loaded checkpoint binding")

    def tokens(self, prompt):
        rendered = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = self.tokenizer.encode(rendered, add_special_tokens=False)
        if not ids or len(ids) > self.config["max_context_tokens"]:
            raise ValueError("Prompt exceeds context cap")
        return rendered, ids

    def _request(self, request):
        row = deepcopy(request)
        actions = row["actions"]
        if not isinstance(actions, list) or not 1 <= len(actions) <= len(self.labels):
            raise ValueError("Invalid legal action menu")
        if len({canonical(a) for a in actions}) != len(actions):
            raise ValueError("Duplicate legal actions")
        if type(row["sample_seed"]) is not int or not 0 <= row["sample_seed"] < 2**63:
            raise ValueError("Invalid action RNG seed")
        prompt = action_prompt(row["observation"], actions, self.labels)
        if "prompt" in row and row["prompt"] != prompt:
            raise ValueError("Prompt contains undeclared information")
        rendered, ids = self.tokens(prompt)
        expected = {"prompt": prompt, "rendered_prompt": rendered, "input_ids": ids,
            "input_ids_sha256": digest(ids), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "menu_sha256": digest(actions), "labels_sha256": digest([self.labels, self.label_ids]),
            "actor_config_sha256": digest(self.config), "base_identity": self.base_identity}
        for key, value in expected.items():
            if key in row and row[key] != value:
                raise ValueError("Request provenance differs: " + key)
        row.update(expected)
        return row

    def arrays(self, rows):
        torch = self.torch
        inputs = [r["input_ids"] for r in rows]
        width = max(map(len, inputs))
        pad = self.tokenizer.pad_token_id
        if pad is None: pad = self.tokenizer.eos_token_id
        tokens = torch.tensor([x+[pad]*(width-len(x)) for x in inputs], device=self.device, dtype=torch.long)
        lengths = torch.tensor(list(map(len, inputs)), device=self.device, dtype=torch.long)
        mask = torch.arange(width, device=self.device)[None, :] < lengths[:, None]
        positions = (mask.long().cumsum(-1)-1).clamp(min=0)
        counts = torch.tensor([len(r["actions"]) for r in rows], device=self.device)
        return tokens, mask, positions, lengths, counts

    def log_probs(self, rows):
        """Only B final hidden states reach the vocabulary head, never B×T."""
        torch = self.torch
        tokens, mask, positions, lengths, counts = self.arrays(rows)
        base = self.model.get_base_model()
        hidden = base.model(input_ids=tokens, attention_mask=mask, position_ids=positions,
                            use_cache=False, return_dict=True).last_hidden_state
        last = hidden[torch.arange(len(rows), device=self.device), lengths-1]
        logits = base.get_output_embeddings()(last).float()
        logits = logits[:, torch.tensor(self.label_ids, device=self.device)]
        legal = torch.arange(len(self.labels), device=self.device)[None, :] < counts[:, None]
        return torch.log_softmax(logits.masked_fill(~legal, -1e9), dim=-1)

    def choose(self, requests, checkpointbinding):
        self._assert_bound(checkpointbinding)
        if not requests or len(requests) > self.config["inference_batch_size"]:
            raise ValueError("Invalid inference batch size")
        rows = [self._request(r) for r in requests]
        self.model.eval()
        with self.torch.no_grad(): logps = self.log_probs(rows).cpu()
        result = []
        for row, vector in zip(rows, logps, strict=True):
            old = vector[:len(row["actions"])]
            if not bool(self.torch.isfinite(old).all()): raise FloatingPointError("Nonfinite actor vector")
            generator = self.torch.Generator(device="cpu").manual_seed(row["sample_seed"])
            probabilities = self.torch.softmax(old.double(), dim=-1)
            pick = int(self.torch.multinomial(probabilities, 1, generator=generator).item())
            result.append({**row, "action_index": pick, "action": row["actions"][pick],
                "action_label": self.labels[pick], "action_token_id": self.label_ids[pick],
                "old_log_prob": float(old[pick]), "old_log_probs": old.tolist(),
                "entropy": float(-(probabilities*probabilities.log()).sum()),
                "checkpoint": deepcopy(checkpointbinding), "policy_version": checkpointbinding["version"],
                "actor_snapshot_sha256": checkpointbinding["adapter_sha256"], "thinking": False})
        return result

    def _validate_row(self, row, binding):
        self._request(row)
        if (row["checkpoint"] != binding or row["policy_version"] != binding["version"]
                or row["actor_snapshot_sha256"] != binding["adapter_sha256"] or row["thinking"] is not False):
            raise ValueError("Behavior checkpoint provenance differs")
        pick, values = row["action_index"], row["old_log_probs"]
        if (type(pick) is not int or not 0 <= pick < len(row["actions"])
                or len(values) != len(row["actions"]) or any(not math.isfinite(v) for v in values)
                or abs(sum(math.exp(v) for v in values)-1) > 1e-5
                or row["old_log_prob"] != values[pick] or row["action"] != row["actions"][pick]
                or row["action_label"] != self.labels[pick] or row["action_token_id"] != self.label_ids[pick]):
            raise ValueError("Malformed sampled action/vector")
        generator = self.torch.Generator(device="cpu").manual_seed(row["sample_seed"])
        probs = self.torch.softmax(self.torch.tensor(values, dtype=self.torch.float64), -1)
        if int(self.torch.multinomial(probs, 1, generator=generator)) != pick:
            raise ValueError("Seed does not reproduce sampled action")

    def full_likelihood_gate(self, rows, binding):
        self._assert_bound(binding)
        if not rows: raise ValueError("Cannot gate an empty cohort")
        for row in rows: self._validate_row(row, binding)
        mode, rng = self.model.training, self.torch.get_rng_state().clone()
        cuda_rng = [s.clone() for s in self.torch.cuda.get_rng_state_all()]
        before = self._optimizer_identity()["state_sha256"]
        records = []
        try:
            for training in (False, True):
                self.model.train(training)
                for offset in range(0, len(rows), self.config["microbatch_size"]):
                    batch = rows[offset:offset+self.config["microbatch_size"]]
                    with self.torch.set_grad_enabled(training): actual = self.log_probs(batch)
                    for index, row in enumerate(batch):
                        vector = actual[index, :len(row["actions"])].detach().cpu().tolist()
                        finite = all(math.isfinite(x) for x in vector)
                        error = max(abs(a-b) for a,b in zip(vector, row["old_log_probs"], strict=True)) if finite else None
                        records.append({"index": offset+index, "mode": "train" if training else "eval",
                            "grad_enabled": training, "row_sha256": digest(row),
                            "actual_log_probs": vector if finite else None, "finite": finite,
                            "max_error": error, "passed": finite and error <= self.config["likelihood_tolerance"]})
                    del actual
        finally:
            self.model.train(mode); self.torch.set_rng_state(rng); self.torch.cuda.set_rng_state_all(cuda_rng)
        if before != self._optimizer_identity()["state_sha256"]:
            raise ValueError("Numerical gate changed Adam")
        return {"kind": "mixed_arena_behavior_likelihood_gate", "binding": deepcopy(binding),
            "row_sha256": [digest(r) for r in rows], "records": records,
            "tolerance": self.config["likelihood_tolerance"], "optimizer_steps": 0,
            "passed": len(records)==2*len(rows) and all(r["passed"] for r in records)}

    def gate_at_checkpoint(self, rows, folder, hashes):
        """Replay stale behavior weights then restore learner/Adam/RNG exactly."""
        torch = self.torch
        saved, before, report = self._capture(), self._state_identity(), None
        try:
            self.load_checkpoint(folder, hashes)
            report = self.full_likelihood_gate(rows, self.binding)
        finally:
            self._restore(saved)
            after = self._state_identity()
            if before != after: raise ValueError("Behavior gate failed exact full learner state restoration")
        report.update(learner_state_before=before, learner_state_after=after, learner_restored=True)
        return report

    def _capture(self):
        torch = self.torch
        return {"adapter": {n:p.detach().clone() for n,p in self.parameters.items()},
            "grads": {n:None if p.grad is None else p.grad.detach().clone() for n,p in self.parameters.items()},
            "optimizer": deepcopy(self.optimizer.state_dict()), "rng": torch.get_rng_state().clone(),
            "cuda_rng": [s.clone() for s in torch.cuda.get_rng_state_all()], "config": deepcopy(self.config),
            "binding": deepcopy(self.binding), "mode": self.model.training,
            "bound_adapter": deepcopy(self.bound_adapter), "pending_version": self.pending_version}

    def _state_identity(self):
        return {"adapter":self.adapter_identity(), "optimizer":self._optimizer_identity(),
            "rng":identity._rng_identity({"rng":self.torch.get_rng_state(), "cuda_rng":self.torch.cuda.get_rng_state_all()}),
            "gradients":{n:None if p.grad is None else identity.tensor_identity(p.grad) for n,p in self.parameters.items()},
            "config_sha256":digest(self.config), "model_training":self.model.training,
            "binding":deepcopy(self.binding), "pending_version":self.pending_version,
            "frozen_state_sha256":self._frozen["sha256"]}

    def _restore(self, saved):
        torch = self.torch
        with torch.no_grad():
            for n,p in self.parameters.items():
                p.copy_(saved["adapter"][n]); p.grad = saved["grads"][n]
        self.optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng"]); torch.cuda.set_rng_state_all(saved["cuda_rng"])
        self.binding, self.bound_adapter = saved["binding"], saved["bound_adapter"]
        self.pending_version, self.config = saved["pending_version"], saved["config"]
        self.model.train(saved["mode"])
        if (any(not torch.equal(p, saved["adapter"][n]) for n,p in self.parameters.items())
                or not torch.equal(torch.get_rng_state(),saved["rng"])):
            raise ValueError("Admission did not restore learner adapter/RNG exactly")

    def numerical_smoke(self, requests):
        """Fresh short action decisions plus derivative; no optimizer or lasting state changes."""
        self._assert_bound(self.binding)
        saved, before = self._capture(), self._state_identity()
        report = {"kind":"mixed_arena_numerical_derivative_smoke", "binding":deepcopy(self.binding),
            "optimizer_steps":0, "training_use":False, "state_before":before, "passed":False}
        try:
            rows = self.choose(requests,self.binding)
            report["rows"] = rows
            report["gate"] = self.full_likelihood_gate(rows,self.binding)
            self.optimizer.zero_grad(set_to_none=True); self.model.train()
            lp = self.log_probs(rows)
            picks = self.torch.tensor([r["action_index"] for r in rows],device=self.device)
            loss = -lp.gather(1,picks[:,None]).sum()/len(rows)
            loss.backward()
            missing = [n for n,p in self.parameters.items() if p.grad is None]
            finite = not missing and all(bool(self.torch.isfinite(p.grad).all()) for p in self.parameters.values())
            norm = math.sqrt(sum(float(p.grad.double().square().sum()) for p in self.parameters.values())) if finite else None
            report["derivative"] = {"loss":float(loss.detach()),"missing_gradients":missing,
                "all_finite":finite,"gradient_norm":norm,"finite_nonzero":finite and norm>0}
            report["passed"] = report["gate"]["passed"] and report["derivative"]["finite_nonzero"]
        except Exception as error:
            report.update(error_type=type(error).__name__, reason=str(error))
        finally:
            self._restore(saved)
        report["state_after"] = self._state_identity()
        report["state_restored"] = report["state_before"]==report["state_after"]
        try: self.verify_frozen()
        except ValueError as error: report.update(passed=False, reason=str(error))
        report["passed"] = report["passed"] and report["state_restored"]
        return report

    def profile_cohort(self, episodes, behavior_binding, gate_report, checkpoint_folder=None):
        """Measure a genuine full update on calibration data, then restore base.

        The temporary Adam update is explicitly distinct from retained training.
        Any error poisons this process after restoration; it must not be retried.
        """
        if self.role != "learner" or self.binding is None or self.binding["version"] != 0:
            raise ValueError("Profiling requires the unchanged base learner")
        self._assert_bound(self.binding)
        saved, before, started = self._capture(), self._state_identity(), time.monotonic()
        report = {"kind":"mixed_arena_full_cohort_profile", "training_use":False,
            "retained_optimizer_steps":0, "state_before":before, "cohort_sha256":digest(episodes),
            "gate_sha256":digest(gate_report), "passed":False}
        try:
            temporary = self.train_cohort(episodes, behavior_binding, gate_report)
            report.update(temporary_update=temporary, temporary_optimizer_steps=temporary["optimizer_steps"],
                          passed=temporary["passed"])
            if checkpoint_folder is not None:
                # Save and independently read back the actual temporary Adam
                # checkpoint before restoring base; include this in full timing.
                checkpoint_started = time.monotonic()
                saved_checkpoint = self.save_checkpoint(checkpoint_folder, 'cycle1', 1)
                inspected = checkpoint_state(checkpoint_folder, self.config['learning_rate'])
                if inspected != saved_checkpoint:
                    raise ValueError('Temporary profile checkpoint readback differs')
                report['temporary_checkpoint'] = {'folder':str(Path(checkpoint_folder)),
                    'files':inspected['files'],'binding':inspected['policy']['binding'],
                    'optimizer_step':inspected['step']}
                report['checkpoint_seconds'] = time.monotonic()-checkpoint_started
        except Exception as error:
            self.poisoned = True
            report.update(error_type=type(error).__name__,reason=str(error),
                temporary_update=getattr(self,"last_failure",None),temporary_optimizer_steps=None)
        finally:
            self._restore(saved)
        report["state_after"] = self._state_identity()
        report["state_restored"] = report["state_before"]==report["state_after"]
        report["passed"] = report["passed"] and report["state_restored"] and not self.poisoned
        report["elapsed_seconds"] = time.monotonic()-started
        return report

    def save_checkpoint(self, folder, checkpoint_id=None, version=None):
        if self.poisoned: raise ValueError("Cannot publish an ambiguous update")
        expected_version = self.pending_version if self.pending_version is not None else (0 if self.binding is None else self.binding["version"])
        version = expected_version if version is None else version
        if version != expected_version:
            raise ValueError("Checkpoint version must match the completed cohort")
        checkpoint_id = ("base" if version==0 else f"cycle{version}") if checkpoint_id is None else checkpoint_id
        validate_binding({"checkpoint_id":checkpoint_id,"version":version,"adapter_sha256":"0"*64,
                          "base_identity":self.base_identity})
        files = super().save_checkpoint(folder)
        binding = {"checkpoint_id": checkpoint_id, "version": version,
            "adapter_sha256": files["adapter.safetensors"], "base_identity": self.base_identity}
        validate_binding(binding)
        meta = {"kind": "mixed_arena_action_policy_v1", "binding": binding,
            "optimizer_sha256": files["optimizer.pt"], "config": self.config,
            "config_sha256": digest(self.config), "labels": self.labels, "label_token_ids": self.label_ids,
            "labels_sha256": digest([self.labels, self.label_ids])}
        with (Path(folder)/"policy.json").open("x") as f: f.write(canonical(meta)+"\n")
        self.binding = binding
        self.bound_adapter = self.adapter_identity()
        self.pending_version = None
        return self.checkpoint_state(folder)

    def checkpoint_state(self, folder):
        # Parent save_checkpoint invokes this before policy.json exists.
        if not (Path(folder)/"policy.json").exists():
            return identity.checkpoint_state(folder, self.config["learning_rate"])
        return checkpoint_state(folder, self.config["learning_rate"])

    def load_checkpoint(self, folder, hashes):
        if self.poisoned: raise ValueError("Cannot resume an ambiguous in-process update")
        if set(hashes) != set(FILES) or any(file_sha(Path(folder)/name)!=h for name,h in hashes.items()):
            raise ValueError("Exact three-file checkpoint hashes required")
        inspected = checkpoint_state(folder, self.config["learning_rate"])
        meta = inspected["policy"]
        if (meta["config"] != self.config or meta["labels"] != self.labels or meta["label_token_ids"] != self.label_ids
                or meta["binding"]["base_identity"] != self.base_identity):
            raise ValueError("Checkpoint policy/config/tokenizer identity differs")
        super().load_checkpoint(folder, {n:hashes[n] for n in identity.CHECKPOINT_FILES})
        self.binding = deepcopy(meta["binding"])
        self.bound_adapter = inspected["adapter_tensors"]
        self.pending_version = None
        return inspected

    def train_cohort(self, episodes, behavior_binding, gate_report):
        if self.role != "learner": raise ValueError("Actor instances cannot update weights")
        if self.poisoned or self.binding is None: raise ValueError("Unbound/poisoned learner")
        self._assert_bound(self.binding)
        lag = policy_lag(self.binding, behavior_binding)
        raw = [r for e in episodes for r in e["transitions"]]
        for row in raw: self._validate_row(row, behavior_binding)
        if (gate_report.get("kind") != "mixed_arena_behavior_likelihood_gate"
                or gate_report.get("optimizer_steps") != 0
                or gate_report.get("passed") is not True or gate_report.get("binding") != behavior_binding
                or gate_report.get("row_sha256") != [digest(r) for r in raw]
                or gate_report.get("tolerance") != self.config["likelihood_tolerance"]
                or len(gate_report.get("records", [])) != 2*len(raw)):
            raise ValueError("Missing exact behavior-checkpoint full-vector gate")
        for i, row in enumerate(raw):
            for mode in ("eval", "train"):
                found = [r for r in gate_report["records"] if r["index"]==i and r["mode"]==mode]
                if (len(found)!=1 or found[0]["row_sha256"]!=digest(row) or found[0]["passed"] is not True
                        or found[0]["grad_enabled"] != (mode=="train")
                        or found[0]["actual_log_probs"] is None
                        or len(found[0]["actual_log_probs"]) != len(row["old_log_probs"])
                        or any(not math.isfinite(a) or abs(a-b)>self.config["likelihood_tolerance"]
                            for a,b in zip(found[0]["actual_log_probs"], row["old_log_probs"], strict=True))):
                    raise ValueError("Behavior gate vector coverage/error differs")
        rows = cohort_rows(episodes, self.config["environment_weights"])
        torch, source = self.torch, deepcopy(self.binding)
        before = self._optimizer_identity()
        original = {n:p.detach().clone() for n,p in self.parameters.items()}
        rng, cuda_rng = torch.get_rng_state().clone(), [s.clone() for s in torch.cuda.get_rng_state_all()]
        self.optimizer.zero_grad(set_to_none=True); self.model.train()
        statistics, losses, learner_records, microbatches = {}, [], [], []
        started = time.monotonic(); attempted = False
        try:
            for offset in range(0, len(rows), self.config["microbatch_size"]):
                batch = rows[offset:offset+self.config["microbatch_size"]]
                logps = self.log_probs(batch)
                for i,row in enumerate(batch):
                    vector=logps[i,:len(row["actions"])].detach().cpu().tolist()
                    finite=all(math.isfinite(x) for x in vector)
                    learner_records.append({"index":offset+i,"row_sha256":digest(raw[offset+i]),
                        "source_binding":source,"behavior_binding":behavior_binding,
                        "action_index":row["action_index"],"advantage":row["advantage"],"weight":row["weight"],
                        "old_vector_sha256":digest(row["old_log_probs"]),
                        "learner_log_probs":vector if finite else None,"all_finite":finite})
                    if not finite:raise FloatingPointError("Nonfinite full learner action vector")
                old = torch.tensor([r["old_log_probs"]+[-1e9]*(len(self.labels)-len(r["old_log_probs"])) for r in batch], device=self.device)
                tensor = lambda key: torch.tensor([r[key] for r in batch], device=self.device)
                counts = torch.tensor([len(r["actions"]) for r in batch], device=self.device)
                loss, stats = objective(logps, old, counts, tensor("action_index"), tensor("advantage"), tensor("weight"), self.config)
                if not bool(torch.isfinite(loss)): raise FloatingPointError("Nonfinite cohort loss")
                loss.backward(); losses.append(float(loss.detach()))
                values={key:float(value.detach()) for key,value in stats.items()}
                microbatches.append({"row_indices":list(range(offset,offset+len(batch))),
                    "loss":float(loss.detach()),**values})
                for key,value in values.items(): statistics[key] = statistics.get(key,0.)+value
            missing = [n for n,p in self.parameters.items() if p.grad is None]
            if missing or any(not bool(torch.isfinite(p.grad).all()) for p in self.parameters.values()):
                raise FloatingPointError("Missing or nonfinite learner gradients: " + str(missing))
            norm = torch.nn.utils.clip_grad_norm_(list(self.parameters.values()), self.config["gradient_clip"], error_if_nonfinite=True)
            if float(norm) > 0:
                attempted = True; self.optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in self.parameters.values()):
                raise FloatingPointError("Nonfinite updated adapter")
            after = self._optimizer_identity()
            if after["step"] != before["step"]+int(attempted): raise ValueError("Adam step increment differs")
            delta = math.sqrt(sum(float((p.detach()-original[n]).double().square().sum()) for n,p in self.parameters.items()))
            self.verify_frozen()
            # Never let an actor request use pre-update metadata for new weights.
            self.pending_version = source["version"]+1
            self.binding = None
            self.bound_adapter = None
            return {"passed": True, "source": source, "behavior": deepcopy(behavior_binding), "policy_lag": lag,
                "cohort_sha256": digest(episodes), "gate_sha256": digest(gate_report),
                "episodes": len(episodes), "transitions": len(rows), "optimizer_steps": int(attempted),
                "adam_step_before": before["step"], "adam_step_after": after["step"],
                "optimizer_before": before, "optimizer_after": after,
                "parameter_delta_l2": delta, "gradient_norm_before_clip": float(norm),
                "gradient_clip": self.config["gradient_clip"], "loss": sum(losses), "microbatch_losses": losses,
                "learner_records":learner_records,"microbatches":microbatches,
                **statistics, "nonzero_advantage_actions": sum(abs(r["advantage"])>1e-12 for r in rows),
                "reward_learning_signal": any(abs(r["advantage"])>1e-12 for r in rows),
                "outcome": "updated" if attempted else "zero_gradient_skipped", "elapsed_seconds": time.monotonic()-started}
        except Exception as error:
            self.poisoned = True
            self.last_failure = {"passed": False, "source": source, "behavior": behavior_binding,
                "optimizer_step_attempted": attempted, "optimizer_steps": None if attempted else 0,
                "error_type": type(error).__name__, "reason": str(error), "microbatch_losses": losses,
                "learner_records":learner_records,"microbatches":microbatches}
            raise
        finally:
            self.optimizer.zero_grad(set_to_none=True); self.model.eval()
            torch.set_rng_state(rng); torch.cuda.set_rng_state_all(cuda_rng)

    def update(self, *args, **kwargs):
        raise ValueError("Only train_cohort may perform a shared action-policy update")
