"""Explicit teacher distillation before the unchanged cardinal CUDA RL backend.

No model/Torch import or inference occurs on import. Teacher targets are supplied
externally, never labeled sampled or on-policy. The orchestrator owns the frozen
16-example schedule and atomic checkpoint commits; this backend enforces four
examples per update, 32 updates, and the separate zero-step RL Adam transition.
"""
from __future__ import annotations

import hashlib
import math

from .program_cardinal_cuda import CardinalCudaPolicy, tensor_identity, _json_hash
from scripts.cloud import train_program_rl as common

SFT_LEARNING_RATE = 5e-5
RL_LEARNING_RATE = 1e-5
SFT_STEPS = 32
SFT_BATCH_SIZE = 4
TEACHER_TOKEN_CAP = 700
GRADIENT_CLIP = .5


def _text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def make_teacher_record(tokenizer, prompt, text, eos_token_ids, max_context_tokens=4096):
    """Encode final code plus actual tokenizer EOS, masking the whole prompt.

    A target must preserve the existing no-thinking generation prefix exactly.
    No stripping, code extraction, repair, truncation, or guessed token IDs occur.
    Compiler/game verification is a separate mandatory orchestrator admission.
    """
    if not isinstance(prompt, str) or not prompt or not isinstance(text, str) or not text:
        raise ValueError("Nonempty prompt and exact teacher text required")
    eos_text, eos_id = tokenizer.eos_token, tokenizer.eos_token_id
    if (not isinstance(eos_text, str) or not eos_text or type(eos_id) is not int
            or not isinstance(eos_token_ids, (list, tuple)) or not eos_token_ids
            or any(type(value) is not int for value in eos_token_ids) or eos_id not in eos_token_ids
            or tokenizer.encode(eos_text, add_special_tokens=False) != [eos_id]):
        raise ValueError("Tokenizer EOS does not match the actual model stop token")
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
    completion = tokenizer.encode(text + eos_text, add_special_tokens=False)
    if (not prompt_ids or not completion or completion[-1] != eos_id
            or any(value in eos_token_ids for value in completion[:-1])
            or tokenizer.encode(rendered + text + eos_text, add_special_tokens=False) != prompt_ids + completion):
        raise ValueError("Teacher tokenization changes prompt prefix or contains premature EOS")
    if not 2 <= len(completion) <= TEACHER_TOKEN_CAP:
        raise ValueError("Teacher completion including EOS exceeds the 700-token bound or is empty")
    if type(max_context_tokens) is not int or len(prompt_ids) + len(completion) > max_context_tokens:
        raise ValueError("Teacher target exceeds context admission bound")
    if any(type(value) is not int or value < 0 for value in prompt_ids + completion):
        raise ValueError("Tokenizer returned invalid token IDs")
    # These fields intentionally differ from sampled rollout provenance.
    record = {"kind": "supervised_teacher_demo", "authorship": "external_teacher_distillation",
        "on_policy": False, "sampled": False, "supervised": True,
        "prompt": prompt, "text": text, "prompt_template": rendered,
        "prompt_ids": prompt_ids, "completion_ids": completion,
        "input_tokens": len(prompt_ids), "output_tokens": len(completion),
        "eos_token_ids": sorted(set(eos_token_ids)), "target_eos_token_id": eos_id,
        "finish_reason": "teacher_target_eos", "teacher_text_sha256": _text_hash(text),
        "prompt_sha256": _text_hash(prompt), "completion_token_cap": TEACHER_TOKEN_CAP,
        "loss_mask": "completion tokens only, including actual tokenizer EOS"}
    return {**record, "teacher_record_sha256": _json_hash(record)}


class CardinalSftPolicy(CardinalCudaPolicy):
    """Same frozen NF4 base/FP32 adapters; separate teacher CE update method."""

    def teacher_record(self, prompt, text):
        return make_teacher_record(self.tokenizer, prompt, text, self.eos, self.config["max_context_tokens"])

    def _validate_teacher(self, record):
        if any(key in record for key in ("sampled_token_logprobs", "seed", "advantages", "reward")):
            raise ValueError("Supervised teacher data must not masquerade as sampled RL data")
        expected = self.teacher_record(record["prompt"], record["text"])
        if common.canonical({key: record.get(key) for key in expected}) != common.canonical(expected):
            raise ValueError("Teacher target text/token/provenance binding differs")
        return expected

    def _adapter_identity(self):
        return {name: tensor_identity(value) for name, value in self.parameters.items()}

    def sft_update(self, batch):
        torch = self.torch
        if (not isinstance(batch, (list, tuple)) or len(batch) != SFT_BATCH_SIZE
                or self.config["learning_rate"] != SFT_LEARNING_RATE
                or self.config["gradient_clip"] != GRADIENT_CLIP):
            raise ValueError("SFT requires batch4, learning rate5e-5, and gradient clip0.5")
        records = [self._validate_teacher(row) for row in batch]
        if len({row["teacher_record_sha256"] for row in records}) != SFT_BATCH_SIZE:
            raise ValueError("SFT batch must contain four distinct verified teacher targets")
        before_optimizer = self._optimizer_identity()
        if before_optimizer["step"] >= SFT_STEPS:
            raise ValueError("The fixed 32-step SFT budget is exhausted")
        self.verify_frozen()
        before = {name: value.detach().cpu().clone() for name, value in self.parameters.items()}
        before_identity = self._adapter_identity()
        self.optimizer.zero_grad(set_to_none=True)
        eval_logps, train_logps, checks, losses = [], [], [], []
        try:
            self.model.eval()
            with torch.no_grad():
                for row in records:
                    values = self.token_logps(row)
                    if values.ndim != 1 or len(values) != row["output_tokens"] or not bool(torch.isfinite(values).all()):
                        raise ValueError("Invalid teacher-forced evaluation likelihood vector")
                    eval_logps.append(values.cpu().tolist())
            self.model.train()
            for row, reference in zip(records, eval_logps):
                values = self.token_logps(row)
                actual = values.detach().cpu().tolist()
                check = common.check_likelihoods(reference, actual,
                    self.config["token_logprob_tolerance"], self.config["sequence_logprob_tolerance"])
                if not check["passed"]:
                    raise ValueError("Teacher-forced eval/train likelihood consistency gate failed")
                checks.append(check); train_logps.append(actual)
                # Equal weight for each program regardless of its token length.
                # token_logps already selects completion positions, including EOS.
                loss = -values.mean() / SFT_BATCH_SIZE
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Nonfinite supervised loss")
                loss.backward()
                losses.append(float(loss.detach().cpu()))
                del values, loss
            missing, squared = [], 0.
            for name, value in self.parameters.items():
                if value.grad is None:
                    missing.append(name)
                elif not bool(torch.isfinite(value.grad).all()):
                    raise ValueError("Nonfinite supervised gradient")
                else:
                    squared += float(value.grad.detach().double().square().sum().cpu())
            if missing or not math.isfinite(squared) or squared <= 0:
                raise ValueError("Supervised gradient is incomplete, zero, or nonfinite")
            norm = torch.nn.utils.clip_grad_norm_(list(self.parameters.values()), GRADIENT_CLIP, error_if_nonfinite=True)
            if not math.isfinite(float(norm)) or float(norm) <= 0:
                raise ValueError("Invalid supervised gradient norm")
            after_clip = math.sqrt(sum(float(p.grad.detach().double().square().sum().cpu()) for p in self.parameters.values()))
            if not math.isfinite(after_clip) or after_clip > GRADIENT_CLIP + 1e-6:
                raise ValueError("Supervised gradient clipping failed")
            self.optimizer.step()
            after_optimizer = self._optimizer_identity()
            if after_optimizer["step"] != before_optimizer["step"] + 1:
                raise ValueError("Supervised Adam did not advance exactly one step")
            delta_squared = 0.
            for name, value in self.parameters.items():
                if not bool(torch.isfinite(value).all()):
                    raise ValueError("Nonfinite supervised adapter")
                delta_squared += float((value.detach().cpu().double() - before[name].double()).square().sum())
            if not math.isfinite(delta_squared) or delta_squared <= 0:
                raise ValueError("Supervised Adam made no finite adapter change")
            frozen_sha = self.verify_frozen()
            return {"kind": "external_teacher_supervised_update", "on_policy": False,
                "objective": "mean_over_programs(mean_completion_token_negative_log_likelihood_including_eos)",
                "batch_size": SFT_BATCH_SIZE, "teacher_record_sha256": [r["teacher_record_sha256"] for r in records],
                "completion_tokens": [r["output_tokens"] for r in records],
                "total_completion_tokens": sum(r["output_tokens"] for r in records),
                "loss": math.fsum(losses), "microbatch_losses": losses,
                "teacher_forced_eval_logprobs": eval_logps, "teacher_forced_train_logprobs": train_logps,
                "teacher_forced_eval_train_gate": {"passed": True, "records": checks},
                "gradient_norm_before_clip": float(norm), "gradient_norm_after_clip": after_clip,
                "missing_gradient_names": [], "adapter_l2_delta": math.sqrt(delta_squared),
                "adapter_before_sha256": _json_hash(before_identity), "adapter_after_sha256": _json_hash(self._adapter_identity()),
                "adam_step_before": before_optimizer["step"], "adam_step_after": after_optimizer["step"],
                "learning_rate": SFT_LEARNING_RATE, "optimizer_steps": 1,
                "optimizer_state_before_sha256": before_optimizer["state_sha256"],
                "optimizer_state_sha256": after_optimizer["state_sha256"], "frozen_state_sha256": frozen_sha}
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.model.eval()

    def reset_optimizer(self, learning_rate=RL_LEARNING_RATE):
        """Start RL Adam at zero after32SFTsteps; preserve exact trained tensors.

        The runner must save the RL-base checkpoint and independently compare
        its adapter.safetensors file hash with the committed SFT-step32 file.
        This method proves tensor equality and RNG equality before that write.
        """
        if self.config["learning_rate"] != SFT_LEARNING_RATE or learning_rate != RL_LEARNING_RATE:
            raise ValueError("Only the explicit SFT5e-5 to RL1e-5 transition is allowed")
        torch = self.torch
        before = self._optimizer_identity()
        if before["step"] != SFT_STEPS:
            raise ValueError("RL transition requires exactly32 completed SFT updates")
        self.verify_frozen()
        adapters = self._adapter_identity()
        cpu_rng = tensor_identity(torch.get_rng_state())
        cuda_rng = [tensor_identity(x) for x in torch.cuda.get_rng_state_all()]
        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer = torch.optim.AdamW(self.parameters.values(), lr=learning_rate, weight_decay=0)
        self.config = {**self.config, "learning_rate": learning_rate}
        self._optimizer_groups = self.optimizer.state_dict()["param_groups"]
        after = self._optimizer_identity()
        if after["step"] != 0 or after["moments"] or self.optimizer.state_dict()["state"]:
            raise ValueError("RL Adam has nonzero step or inherited moments")
        if adapters != self._adapter_identity():
            raise ValueError("Optimizer reset changed adapter tensor bytes")
        if cpu_rng != tensor_identity(torch.get_rng_state()) or cuda_rng != [tensor_identity(x) for x in torch.cuda.get_rng_state_all()]:
            raise ValueError("Optimizer reset changed RNG state")
        frozen_sha = self.verify_frozen()
        return {"kind": "supervised_to_rl_optimizer_reset", "learning_rate_before": SFT_LEARNING_RATE,
            "learning_rate_after": RL_LEARNING_RATE, "sft_adam_step": before["step"], "rl_adam_step": 0,
            "optimizer_state_before_sha256": before["state_sha256"], "optimizer_state_after_sha256": after["state_sha256"],
            "adapter_tensor_sha256": _json_hash(adapters), "adapter_tensors_unchanged": True,
            "rl_moments_empty": True, "rng_unchanged": True, "frozen_state_sha256": frozen_sha,
            "adapter_file_equality_still_required": True}
