"""Versioned scalar physical execution for the unchanged mixed-arena objective.

Logical queues remain batched. Every model forward receives exactly one real
request at its own token length, in sampling, replay and gradient computation.
This removes another request's shape from the physical forward contract. It is
not a claim that CUDA floating-point arithmetic is universally deterministic.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import math

from selfplay.arena_mixed import cuda_policy as previous

MODEL, REVISION, FILES = previous.MODEL, previous.REVISION, previous.FILES
canonical, digest, file_sha = previous.canonical, previous.digest, previous.file_sha
action_prompt, select_labels = previous.action_prompt, previous.select_labels
local_input_identity, validate_binding = previous.local_input_identity, previous.validate_binding
policy_lag, objective, cohort_rows = previous.policy_lag, previous.objective, previous.cohort_rows
compare_actor_reports = previous.compare_actor_reports

PHYSICAL_EXECUTION_CONTRACT = {
    "version": 1, "physical_batch_size": 1, "input_padding": "none_per_request",
    "logical_inference_batch_size": 8, "logical_training_microbatch_size": 2,
    "likelihood_tolerance": .002,
}


def validate_config(config):
    result = previous.validate_config(config)
    if (result.get("physical_execution_contract") != PHYSICAL_EXECUTION_CONTRACT
            or result["inference_batch_size"] != 8 or result["microbatch_size"] != 2
            or result["likelihood_tolerance"] != .002):
        raise ValueError("Exact scalar physical execution and unchanged .002 gate required")
    return result


def default_config(environment_weights, **overrides):
    return validate_config(previous.default_config(environment_weights, **{
        "seed": 742001, "physical_execution_contract": deepcopy(PHYSICAL_EXECUTION_CONTRACT),
        **overrides}))


def checkpoint_state(folder, expected_learning_rate=None):
    state = previous.checkpoint_state(folder, expected_learning_rate)
    validate_config(state["policy"]["config"])
    return state


class CudaActionPolicy(previous.CudaActionPolicy):
    def __init__(self, model_dir, config, role="learner"):
        super().__init__(model_dir, validate_config(config), role=role)
        self.runtime["physical_execution_contract"] = deepcopy(PHYSICAL_EXECUTION_CONTRACT)
        self.runtime["physical_execution_contract_sha256"] = digest(PHYSICAL_EXECUTION_CONTRACT)
        self.runtime["sampling_batch"] = "Logical batches; one physical request at its own context length"

    def _request(self, request):
        validate_config(self.config)
        row = super()._request(request)
        expected = digest(PHYSICAL_EXECUTION_CONTRACT)
        if ("physical_execution_contract_sha256" in row
                and row["physical_execution_contract_sha256"] != expected):
            raise ValueError("Action row physical execution contract differs")
        row["physical_execution_contract_sha256"] = expected
        return row

    def _validate_row(self, row, binding):
        if row.get("physical_execution_contract_sha256") != digest(PHYSICAL_EXECUTION_CONTRACT):
            raise ValueError("Missing exact action physical execution contract")
        return super()._validate_row(row, binding)

    def log_probs(self, rows):
        validate_config(self.config)
        if not rows:
            raise ValueError("Scalar forward requires at least one real request")
        # Do not detach: cat preserves each independent forward's gradient graph.
        return self.torch.cat([super(CudaActionPolicy, self).log_probs([row]) for row in rows], dim=0)

    def checkpoint_state(self, folder):
        # The inherited base checkpoint writer inspects before policy.json exists.
        if not (Path(folder)/"policy.json").exists():
            return super().checkpoint_state(folder)
        return checkpoint_state(folder, self.config["learning_rate"])

    def load_checkpoint(self, folder, hashes):
        checkpoint_state(folder, self.config["learning_rate"])
        return super().load_checkpoint(folder, hashes)

    def execution_diagnostic(self, fixture):
        """One precommitted geometry comparison, then the selected scalar gate.

        Historical request content is replayed at this run's exact fresh base.
        Legacy B8/B2 results are descriptive; they never select a fallback or
        relax the scalar full-vector/derivative admission. No Adam step occurs.
        """
        self._assert_bound(self.binding)
        if self.binding["version"] != 0 or getattr(self, "_execution_diagnostic_attempted", False):
            raise ValueError("Execution diagnostic is base-only and cannot be retried")
        if (fixture.get("kind") != "mixed_scalar_execution_fixtures_v1"
                or fixture.get("physical_execution_contract") != PHYSICAL_EXECUTION_CONTRACT
                or fixture.get("outlier_actor_index") != 6 or fixture.get("outlier_gate_index") != 0):
            raise ValueError("Frozen scalar diagnostic fixture contract differs")
        groups = {key: fixture[key] for key in ("actor_requests", "gate_requests", "smoke_requests")}
        if (len(groups["actor_requests"]) != 8 or len(groups["gate_requests"]) != 2
                or len(groups["smoke_requests"]) != 5
                or {x["environment"] for x in groups["smoke_requests"]} != set(self.config["environment_weights"])
                or groups["actor_requests"][6] != groups["gate_requests"][0]
                or next(x for x in groups["smoke_requests"] if x["environment"] == "chess") != groups["gate_requests"][0]):
            raise ValueError("Diagnostic coverage or exact historical outlier differs")
        rows = {key: [self._request(r) for r in requests] for key, requests in groups.items()}
        self._execution_diagnostic_attempted = True
        saved, before = self._capture(), self._state_identity()
        report = {"kind": "mixed_scalar_physical_execution_diagnostic_v1", "passed": False,
            "fixture_sha256": digest(fixture), "binding": deepcopy(self.binding),
            "physical_execution_contract": deepcopy(PHYSICAL_EXECUTION_CONTRACT),
            "optimizer_steps": 0, "training_use": False, "state_before": before,
            "comparison_scope": "Historical v1 requests at the current exact fresh-base checkpoint; legacy geometry is descriptive, not an adaptive selection.",
            "groups": {}}
        try:
            self.model.eval()
            with self.torch.no_grad():
                for name in ("actor_requests", "gate_requests"):
                    batch = rows[name]
                    legacy = super().log_probs(batch).detach().cpu()
                    scalar = self.log_probs(batch).detach().cpu()
                    records = []
                    for i, row in enumerate(batch):
                        width = len(row["actions"])
                        a, b = legacy[i, :width].tolist(), scalar[i, :width].tolist()
                        finite = all(math.isfinite(v) for v in a+b)
                        if not finite:
                            raise FloatingPointError("Nonfinite diagnostic full legal vector")
                        records.append({"index": i, "request_sha256": digest(groups[name][i]),
                            "input_ids_sha256": row["input_ids_sha256"], "input_tokens": len(row["input_ids"]),
                            "legal_actions": width, "legacy_log_probs": a, "scalar_log_probs": b,
                            "max_error": max(abs(x-y) for x,y in zip(a,b,strict=True)), "all_finite": True})
                    report["groups"][name] = {"legacy_physical_batch_size": len(batch),
                        "legacy_padded_tokens": max(len(r["input_ids"]) for r in batch),
                        "scalar_physical_batch_size": 1, "records": records}
            a = report["groups"]["actor_requests"]["records"][6]
            b = report["groups"]["gate_requests"]["records"][0]
            report["outlier_comparison"] = {
                "legacy_b8_b2_max_error": max(abs(x-y) for x,y in zip(a["legacy_log_probs"],b["legacy_log_probs"],strict=True)),
                "scalar_group_max_error": max(abs(x-y) for x,y in zip(a["scalar_log_probs"],b["scalar_log_probs"],strict=True)),
                "tolerance": .002, "legacy_condition_selected": False}
            report["scalar_smoke"] = self.numerical_smoke(groups["smoke_requests"])
            report["passed"] = (report["scalar_smoke"]["passed"]
                and report["outlier_comparison"]["scalar_group_max_error"] <= .002)
        except Exception as error:
            report.update(error_type=type(error).__name__, reason=str(error))
        finally:
            self._restore(saved)
        report["state_after"] = self._state_identity()
        report["state_restored"] = before == report["state_after"]
        try:
            self.verify_frozen()
        except ValueError as error:
            report.update(passed=False, error_type=type(error).__name__, reason=str(error))
        report["passed"] = report["passed"] and report["state_restored"]
        if not report["passed"]:
            self.poisoned = True
        return report
