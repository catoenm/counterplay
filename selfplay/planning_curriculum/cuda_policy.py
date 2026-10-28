"""Planning-only controlled objective, with unchanged scalar CUDA forwards.

Importing this module never loads a model. The inherited checkpoint, action
sampling, exact-context and lag-one contracts remain in force. Only this new
module's objective has the declared global .24 multiplier.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import math
import time

from selfplay.arena_mixed import cuda_policy as original
from selfplay.arena_mixed_scalar import cuda_policy as scalar

MODEL, REVISION, FILES = scalar.MODEL, scalar.REVISION, scalar.FILES
canonical, digest, file_sha = scalar.canonical, scalar.digest, scalar.file_sha
local_input_identity = scalar.local_input_identity
compare_actor_reports = scalar.compare_actor_reports
policy_lag = scalar.policy_lag
PHYSICAL_EXECUTION_CONTRACT = deepcopy(scalar.PHYSICAL_EXECUTION_CONTRACT)
OBJECTIVE_CONTRACT = {
    "version": 1, "environment_weights": {"planning": 1.0},
    "global_loss_multiplier": .24, "terms": ["policy", "entropy", "kl"],
    "episode_baseline": "other independent episode terminal return",
    "action_reduction": "sum, no realized-length divisor",
}


def validate_config(config):
    result = scalar.validate_config(config)
    fixed = {"environment_weights": {"planning": 1.0},
        "global_loss_multiplier": .24, "objective_contract": OBJECTIVE_CONTRACT,
        "learning_rate": 2e-5, "gradient_clip": 1.,
        "entropy_coefficient": .01, "kl_coefficient": .02,
        "max_context_tokens": 4096}
    if any(result.get(k) != v for k, v in fixed.items()):
        raise ValueError("Planning-only objective or frozen hyperparameters differ")
    if type(result.get("seed")) is not int or not 0 <= result["seed"] < 2**63:
        raise ValueError("Invalid common base initialization seed")
    return result


def default_config(**overrides):
    options = {"global_loss_multiplier": .24, "objective_contract": deepcopy(OBJECTIVE_CONTRACT), **overrides}
    return validate_config(scalar.default_config({"planning": 1.0}, **options))


def objective(logps, old_logps, counts, picks, advantages, weights, config):
    validate_config(config)
    # Multiplying weights scales every policy, entropy and KL contribution,
    # including its derivative; it does not rescale advantages or Adam's LR.
    return original.objective(logps, old_logps, counts, picks, advantages,
        weights * config["global_loss_multiplier"], config)


def cohort_rows(episodes, environment_weights):
    if environment_weights != {"planning": 1.0} or len(episodes) != 2:
        raise ValueError("Exactly two independent planning episodes required")
    for episode in episodes:
        if (episode.get("learner_seats") != [0]
                or episode.get("returns") not in ([0., 0.], [1., 0.])
                or not 1 <= len(episode.get("transitions", [])) <= 12):
            raise ValueError("Expected bounded single-player verified-goal episodes")
    return original.cohort_rows(episodes, environment_weights)


def checkpoint_state(folder, expected_learning_rate=None):
    state = scalar.checkpoint_state(folder, expected_learning_rate)
    validate_config(state["policy"]["config"])
    return state


class CudaActionPolicy(scalar.CudaActionPolicy):
    def __init__(self, model_dir, config, role="learner"):
        super().__init__(model_dir, validate_config(config), role=role)
        self.runtime["objective_contract"] = deepcopy(OBJECTIVE_CONTRACT)
        self.runtime["objective_contract_sha256"] = digest(OBJECTIVE_CONTRACT)

    def _request(self, request):
        validate_config(self.config)
        if request.get("environment") != "planning" or request.get("player") != 0:
            raise ValueError("This policy accepts only planning player zero")
        return super()._request(request)

    def checkpoint_state(self, folder):
        if not (Path(folder) / "policy.json").exists():
            return super().checkpoint_state(folder)
        return checkpoint_state(folder, self.config["learning_rate"])

    def load_checkpoint(self, folder, hashes):
        checkpoint_state(folder, self.config["learning_rate"])
        return super().load_checkpoint(folder, hashes)

    def _clock(self):
        if str(self.device).startswith("cuda"):
            self.torch.cuda.synchronize()
        return time.monotonic()

    def log_probs(self, rows):
        validate_config(self.config)
        capture = getattr(self, "_probe_forward_timings", None)
        if capture is None:
            return super().log_probs(rows)
        if not rows:
            raise ValueError("At least one real request required")
        vectors = []
        for row in rows:
            started = self._clock()
            vector = original.CudaActionPolicy.log_probs(self, [row])
            elapsed = self._clock() - started
            capture.append({"row_sha256": digest(row), "input_tokens": len(row["input_ids"]),
                "training": self.model.training, "grad_enabled": self.torch.is_grad_enabled(),
                "seconds": elapsed})
            vectors.append(vector)
        return self.torch.cat(vectors, dim=0)

    def choose_profiled(self, requests, binding):
        """The normal draw once, plus scalar-forward and whole-call timing.

        Use for the fixed screen only. No extra draw is generated. The worker
        saves the untouched rows and this timing receipt in one raw-batch file.
        """
        if getattr(self, "_probe_forward_timings", None) is not None:
            raise ValueError("Nested timing collection is forbidden")
        self._probe_forward_timings = []
        started = self._clock()
        try:
            rows = self.choose(requests, binding)
            forwards = deepcopy(self._probe_forward_timings)
            elapsed = self._clock()-started
            if len(forwards) != len(rows):
                raise ValueError("Scalar sampling timing coverage differs")
            return {"rows": rows, "timing": {"kind": "planning_screen_sampling_timing_v1",
                "elapsed_seconds": elapsed, "forward_timings": forwards,
                "row_sha256": [digest(r) for r in rows],
                "nonforward_seconds": max(0., elapsed-sum(x["seconds"] for x in forwards))}}
        finally:
            self._probe_forward_timings = None

    def timing_probe(self, episodes, expected_episode_ids, checkpoint_folder):
        """One actual temporary update, durably journalled by the caller.

        The caller must create an exclusive intent before invocation. This
        in-process flag survives checkpoint restoration and prevents local
        retries; a fresh process must refuse an unresolved durable intent.
        """
        validate_config(self.config)
        if (self.role != "learner" or self.binding is None or self.binding["version"] != 0
                or self._optimizer_identity()["step"] != 0
                or getattr(self, "_timing_probe_attempted", False)):
            raise ValueError("Timing probe is zero-step base-only and cannot be retried")
        if (len(expected_episode_ids) != 2 or len(set(expected_episode_ids)) != 2
                or [e["episode_id"] for e in episodes] != list(expected_episode_ids)):
            raise ValueError("Probe must use the fixed first two depth-six screen IDs")
        if any(e.get("depth") != 6 or e.get("schedule", {}).get("screen_group") != "hard"
                or e.get("schedule", {}).get("depth") != 6 for e in episodes):
            raise ValueError("Probe episodes must be the original-depth-six screen")
        cohort_rows(episodes, self.config["environment_weights"])
        if Path(checkpoint_folder).exists():
            raise FileExistsError("Probe checkpoint must be new")
        self._assert_bound(self.binding)
        self._timing_probe_attempted = True
        started = self._clock()
        saved, before = self._capture(), self._state_identity()
        capture_seconds = self._clock()-started
        self._probe_forward_timings = []
        report = {"kind": "planning_curriculum_restored_full_step_probe_v1",
            "training_use": False, "retained_optimizer_steps": 0, "passed": False,
            "state_before": before, "episode_ids": list(expected_episode_ids),
            "cohort_sha256": digest(episodes), "decisions": sum(len(e["transitions"]) for e in episodes),
            "objective_contract": deepcopy(OBJECTIVE_CONTRACT), "initial_capture_seconds": capture_seconds}
        try:
            raw = [row for e in episodes for row in e["transitions"]]
            gate_started = self._clock()
            gate = self.full_likelihood_gate(raw, self.binding)
            report["gate_seconds"] = self._clock() - gate_started
            report["gate"] = gate
            report["gate_forward_timings"] = deepcopy(self._probe_forward_timings)
            self._probe_forward_timings.clear()
            if not gate["passed"]:
                raise ValueError("Full eval/train behavior gate failed")
            update = self.train_cohort(episodes, self.binding, gate)
            report["temporary_update"] = update
            report["update_forward_timings"] = deepcopy(self._probe_forward_timings)
            if update["optimizer_steps"] != 1 or update["parameter_delta_l2"] <= 0:
                raise ValueError("Probe did not exercise one genuine nonzero Adam update")
            # Price copying mature Adam moments, rather than extrapolating from
            # the empty zero-step state. This has no model forward or update.
            roundtrip_started = self._clock()
            mature_before = self._state_identity()
            mature = self._capture()
            self._restore(deepcopy(mature))
            mature_after = self._state_identity()
            report["mature_state_roundtrip"] = {"state_before": mature_before, "state_after": mature_after,
                "passed": mature_before == mature_after, "optimizer_steps": 0, "model_forwards": 0,
                "seconds": self._clock()-roundtrip_started}
            del mature
            if mature_before != mature_after or mature_after["optimizer"]["step"] != 1:
                raise ValueError("Temporary mature Adam state roundtrip differs")
            checkpoint_started = self._clock()
            written = self.save_checkpoint(checkpoint_folder, "cycle1", 1)
            inspected = checkpoint_state(checkpoint_folder, self.config["learning_rate"])
            if written != inspected or inspected["step"] != 1:
                raise ValueError("Temporary checkpoint readback differs")
            report["checkpoint_seconds"] = self._clock() - checkpoint_started
            report["temporary_checkpoint"] = {"files": inspected["files"],
                "binding": inspected["policy"]["binding"], "optimizer_step": inspected["step"]}
            report.update(passed=True, temporary_optimizer_steps=1)
        except Exception as error:
            self.poisoned = True
            report.update(error_type=type(error).__name__, reason=str(error),
                temporary_optimizer_steps=None, last_failure=getattr(self, "last_failure", None))
        finally:
            self._probe_forward_timings = None
            self._restore(saved)
        report["state_after"] = self._state_identity()
        report["state_restored"] = before == report["state_after"]
        try:
            self.verify_frozen()
        except Exception as error:
            self.poisoned = True
            report.update(error_type=type(error).__name__, reason=str(error), passed=False)
        report["passed"] = report["passed"] and report["state_restored"] and not self.poisoned
        if not report["passed"]:
            self.poisoned = True
        report["elapsed_seconds"] = self._clock() - started
        return report

    def profile_cohort(self, *args, **kwargs):
        raise ValueError("Use the once-only fixed-depth timing_probe with durable intent")

    def train_cohort(self, episodes, behavior_binding, gate_report):
        validate_config(self.config)
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
        timings = []
        started = time.monotonic(); attempted = False
        try:
            for offset in range(0, len(rows), self.config["microbatch_size"]):
                batch = rows[offset:offset+self.config["microbatch_size"]]
                batch_started = self._clock()
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
                timings.append({"row_indices": list(range(offset, offset+len(batch))),
                    "input_tokens": [len(r["input_ids"]) for r in batch],
                    "forward_backward_seconds": self._clock()-batch_started})
                values={key:float(value.detach()) for key,value in stats.items()}
                microbatches.append({"row_indices":list(range(offset,offset+len(batch))),
                    "loss":float(loss.detach()),**values})
                for key,value in values.items(): statistics[key] = statistics.get(key,0.)+value
            optimizer_started = self._clock()
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
                "objective_contract": deepcopy(OBJECTIVE_CONTRACT),
                "global_loss_multiplier": self.config["global_loss_multiplier"],
                "microbatch_timings": timings,
                "optimizer_verification_seconds": self._clock()-optimizer_started,
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

