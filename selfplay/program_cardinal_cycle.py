"""Bounded cardinal-interface controller RL cycle; preparation is CPU-only.

The frozen numerical backend is reused without changing its token likelihoods.
Only an explicit run command starts a worker. Every run keeps a durable journal,
an absolute deadline, a nonrenewable wall budget and the shared neural lock.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

from selfplay import program_local_rl as local
from selfplay import program_study as journal
from selfplay.program_cardinal import cardinal_model_spec, classify_cardinal
from scripts.cloud import train_program_rl as common

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "runs/program-local-27b-compact-v1"
PRIOR = ROOT / "runs/program-study-v1"
LOCK = ROOT / "runs/local-neural.lock"
ENDPOINTS = ("base", "cycle3")
SOURCES = list(dict.fromkeys(journal.SOURCES + ["selfplay/program_cardinal.py",
                "selfplay/program_cardinal_cycle.py", "scripts/analyze_program_cardinal_cycle.py"]))


def config():
    return {**common.DEFAULTS, **local.OVERRIDES,
            "kind": "cardinal_program_rl_cycle", "condition": "clean-cardinal-v1",
            "model_repo": local.MODEL_REPO, "model_revision": local.MODEL_REVISION,
            "algorithm": "full_completion_RLOO", "iterations": 3, "group_size": 8,
            "learning_rate": 1e-5, "max_new_tokens": 2048, "max_context_tokens": 4096,
            "sample_seconds": 300, "worker_seconds": 21600,
            "deadline": "2026-10-22T16:30:00-07:00", "seed": 124301,
            "capability_samples": 8, "minimum_productive_programs": 4,
            "minimum_productive_distinct_ir": 2, "minimum_positive_calibration_games": 12,
            "opponent_count": 2, "opponent_attempt_cap": 8, "eval_samples": 12,
            "map_seeds_per_group": 2, "max_rounds": 48,
            "train_seed_start": 7200000, "calibration_seed_start": 7100000,
            "holdout_seed_start": 7300000,
            "reward": "mean relative return; invalid authorship or candidate interpreter faults=-1",
            "scope": "One clean cardinal interface, fixed productive base pool, three fresh on-policy groups; no population refresh"}


def schedule():
    return {"calibration_maps": list(range(7100000, 7100006)),
            "train_maps": {str(i): [7200000 + i*100 + j for j in range(2)] for i in range(1, 4)},
            "holdout_maps": list(range(7300000, 7300006)),
            "capability_generation_seeds": list(range(9100000, 9100008)),
            "train_generation_seeds": {str(i): [9200000 + i*100 + j for j in range(8)] for i in range(1, 4)},
            "eval_generation_seeds": list(range(9300000, 9300012))}


def plan():
    return {"kind": "cardinal_program_rl_cycle_plan", "config": config(), "schedule": schedule(),
            "prompt": "Clean complete cardinal syntax/game specification only; no failed controller, successful controller or practice feedback.",
            "capability_gate": {"draws": 8, "games_per_compiled_program": 24,
                "productive": "Authorship valid, zero candidate interpreter faults across all24 games, positive banked points in at least12/24 games.",
                "required_productive": 4, "required_distinct_productive_ir": 2,
                "stop_on_failure": True, "resampling_or_repairs": False},
            "opponents": "First two productive distinct IRs in the fixed capability draw order, plus nearest/value. Frozen for training and evaluation; model opponents are seen controller families on new maps.",
            "training": {"groups": 3, "draws_per_group": 8, "maps_per_group": 2,
                "algorithm": "Leave-one-out advantages; summed completion-token log likelihood including actual EOS, one accumulated optimizer step per nonzero group.",
                "learning_rate": 1e-5, "gradient_clip": .5,
                "initial_optimizer": "Preserve verified zero-step parent adapter and all zero Adam moments; change only serialized scalar learning_rate to float32(1e-5).",
                "zero_signal": "Retain all samples and scores; copy exact checkpoint/optimizer with zero updates; never resample.",
                "invalid_streak_stop": "Two consecutive groups with no runtime-valid program stop incomplete.",
                "numerics": "Full-token cached/teacher-forced pre-update and gradient-forward gates at0.002 token/0.02 sequence; no relaxation. A32-token smoke does not guarantee2048-token memory fit."},
            "evaluation": {"endpoints": list(ENDPOINTS), "draws_per_endpoint": 12,
                "paired_generation_seeds": True, "holdout_maps": 6, "opponents": 4, "seat_swaps": 2,
                "games_per_compiled_program": 48, "seal_all24_draws_before_scoring": True,
                "primary": "Unpenalized banked-point yield over all12 prescribed program draws; invalid authorship contributes0 yield without a fabricated game.",
                "secondary": "Relative returns among compiled programs; authorship/runtime validity, ordinary failed actions, all-rest rate, terminal unbanked cargo.",
                "incomplete": "Missing draws are not zeros. No comparison unless both fixed endpoints and all24 draws and both evaluation cells are complete.",
                "limits": "One training seed; no significance or general transfer claim; no outcome-based checkpoint selection."},
            "resources": {"external_spend_usd": 0, "cumulative_user_cap_usd": 100,
                "maximum_authoring_draws": 56, "extra_smoke": "One unscored32-token derivative smoke per worker process; no update.",
                "absolute_deadline": config()["deadline"], "maximum_wall_seconds": 21600,
                "memory_limit_bytes": 28*1024**3,
                "guard": "Global neural lock and run lock; parent process-group timeout/RSS watchdog plus worker guards. Parent death does not independently guarantee termination."}}


def validate_configuration(cfg, timetable):
    if cfg != config() or timetable != schedule():
        raise ValueError("Cardinal cycle config/schedule differs from the preregistered values")
    datetime.fromisoformat(cfg["deadline"])


def optimizer_identity(folder, learning_rate, require_zero=False):
    """Read small adapter/Adam files on CPU; never loads the quantized model."""
    import numpy as np
    from safetensors.numpy import load_file
    folder = Path(folder)
    adapter = load_file(str(folder / "adapter.safetensors"))
    values = load_file(str(folder / "optimizer.safetensors"))
    names = {f"self_attn.{p}.lora_{s}" for p in ("q_proj", "v_proj") for s in ("a", "b")}
    if set(adapter) != names or set(values) != {"step", "learning_rate"} | {n+suffix for n in names for suffix in (".m", ".v")}:
        raise ValueError("Adapter/optimizer tensor schema differs")
    for name, value in adapter.items():
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("Nonfinite/non-FP32 adapter")
        for suffix in (".m", ".v"):
            moment = values[name+suffix]
            if moment.shape != value.shape or moment.dtype != np.float32 or not np.isfinite(moment).all():
                raise ValueError("Adam moment shape/dtype/finiteness differs")
            if suffix == ".v" and np.any(moment < 0):
                raise ValueError("Negative Adam second moment")
            if require_zero and np.any(moment != 0):
                raise ValueError("Initial Adam moments are not zero")
    lr, step = values["learning_rate"], values["step"]
    if lr.shape != () or lr.dtype != np.float32 or lr.item() != np.float32(learning_rate).item():
        raise ValueError("Actual serialized optimizer learning rate differs")
    if step.shape != () or step.dtype != np.uint64 or (require_zero and step.item() != 0):
        raise ValueError("Adam step identity differs")
    return {"learning_rate": float(lr.item()), "step": int(step.item()),
            "tensor_shapes": {k: list(v.shape) for k, v in values.items()},
            "adapter_sha256": common.digest(folder / "adapter.safetensors"),
            "optimizer_sha256": common.digest(folder / "optimizer.safetensors")}


def initialize_optimizer_copy(source, target, learning_rate):
    import numpy as np
    from safetensors.numpy import load_file, save_file
    source, target = Path(source), Path(target)
    original = optimizer_identity(source, 1e-6, require_zero=True)
    target.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source / "adapter.safetensors", target / "adapter.safetensors")
    before = load_file(str(source / "optimizer.safetensors"))
    after = {k: v.copy() for k, v in before.items()}
    after["learning_rate"] = np.array(learning_rate, dtype=np.float32)
    save_file(after, str(target / "optimizer.safetensors"))
    actual = load_file(str(target / "optimizer.safetensors"))
    if set(actual) != set(before) or any(not np.array_equal(v, actual[k]) for k, v in before.items() if k != "learning_rate"):
        raise ValueError("Optimizer initialization changed more than learning_rate")
    revised = optimizer_identity(target, learning_rate, require_zero=True)
    if original["adapter_sha256"] != revised["adapter_sha256"]:
        raise ValueError("Initial adapter bytes changed")
    return {"operation": "Only float32 learning_rate changed; adapter bytes, zero step and all zero moments retained",
            "before": original, "after": revised, "changed_tensor_names": ["learning_rate"]}


def map_fingerprints(timetable):
    previous = common.load(PRIOR / "map-fingerprints.json")
    manifest = common.load(PRIOR / "manifest.json")
    if common.digest(PRIOR / "map-fingerprints.json") != manifest["map_fingerprints_sha256"]:
        raise ValueError("Prior map provenance differs")
    prior = dict(previous["prior_seed_fingerprints"])
    for rows in previous["families"].values():
        prior.update({str(r["seed"]): r["sha256"] for r in rows})
    families = {"calibration": timetable["calibration_maps"],
                "train": [s for row in timetable["train_maps"].values() for s in row],
                "holdout": timetable["holdout_maps"]}
    used, result = set(prior.values()), {}
    for family, seeds in families.items():
        result[family] = []
        for seed in seeds:
            layout = journal.map_layout(seed)
            fingerprint = journal.object_hash(layout)
            if str(seed) in prior or fingerprint in used:
                raise ValueError("New observable map overlaps a prior/new family")
            used.add(fingerprint)
            result[family].append({"seed": seed, "sha256": fingerprint, "layout": layout})
    return {"families": result, "prior_seed_fingerprints": prior,
            "prior_evidence_sha256": {"inputs/prior-map-fingerprints.json": common.digest(PRIOR / "map-fingerprints.json")},
            "definition": previous["definition"]}


def prepare(run, model_dir):
    run, model_dir = Path(run), Path(model_dir)
    if run.exists():
        raise FileExistsError("New cardinal run directory required")
    parent_manifest, parent_status = common.verify_sources(PARENT)
    if parent_status["state"] != "complete":
        raise ValueError("Parent mechanism run must be complete")
    initial = journal.checkpoint_identity(PARENT / "checkpoints/base")
    if initial["report"] is not None:
        raise ValueError("Expected zero-update base, not trained adapters")
    for relative in SOURCES:
        if not (ROOT / relative).is_file():
            raise ValueError("Source not ready for freeze: " + relative)
    cfg, timetable = config(), schedule()
    fingerprints = map_fingerprints(timetable)
    prompt = cardinal_model_spec()
    if not prompt.strip():
        raise ValueError("Empty cardinal specification")
    run.mkdir(parents=True)
    copied = {}
    def copy_input(source, relative):
        target = run / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied[relative] = common.digest(target)
    for name in ("manifest.json", "status.json", "runtime.json"):
        if name == "runtime.json" and common.digest(PARENT / name) != parent_status["artifacts"][name]:
            raise ValueError("Parent runtime is not committed")
        copy_input(PARENT / name, "inputs/parent-" + name)
    for name in ("metadata.json", *initial["files"]):
        copy_input(PARENT / "checkpoints/base" / name, "inputs/parent-base/" + name)
    copy_input(PRIOR / "manifest.json", "inputs/prior-study-manifest.json")
    copy_input(PRIOR / "map-fingerprints.json", "inputs/prior-map-fingerprints.json")
    copy_input(model_dir / "model-index.json", "inputs/model-index.json")
    initialization = initialize_optimizer_copy(run / "inputs/parent-base", run / "inputs/base", cfg["learning_rate"])
    common.atomic_json(run / "optimizer-initialization.json", initialization)
    files = {name: common.digest(run / "inputs/base" / name) for name in initial["files"]}
    common.atomic_json(run / "inputs/base/metadata.json", {**initial, "files": files,
        "optimizer_initialization_sha256": common.digest(run / "optimizer-initialization.json")})
    for name in ("metadata.json", *files):
        copied["inputs/base/" + name] = common.digest(run / "inputs/base" / name)
    (run / "prompts").mkdir()
    (run / "prompts/cardinal.txt").write_text(prompt)
    copied["prompts/cardinal.txt"] = common.digest(run / "prompts/cardinal.txt")
    common.atomic_json(run / "plan.json", plan())
    common.atomic_json(run / "schedule.json", timetable)
    common.atomic_json(run / "map-fingerprints.json", fingerprints)
    copied["optimizer-initialization.json"] = common.digest(run / "optimizer-initialization.json")
    source_hashes = {}
    for relative in SOURCES:
        target = run / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
        source_hashes[relative] = common.digest(target)
    manifest = {"format_version": 1, "kind": cfg["kind"], "created": common.now(), "config": cfg,
        "plan_sha256": common.digest(run / "plan.json"), "schedule_sha256": common.digest(run / "schedule.json"),
        "map_fingerprints_sha256": common.digest(run / "map-fingerprints.json"),
        "source_sha256": source_hashes, "input_sha256": copied,
        "prompt_sha256": {"cardinal": copied["prompts/cardinal.txt"]},
        "model_index_sha256": copied["inputs/model-index.json"],
        "library_source_sha256": parent_manifest["library_source_sha256"],
        "expected_installed_versions": parent_manifest["expected_installed_versions"],
        "external_spend_usd": 0, "claims_require_complete": True}
    common.atomic_json(run / "manifest.json", manifest)
    common.atomic_json(run / "status.json", {"manifest_sha256": common.digest(run / "manifest.json"),
        "artifacts": {}, "state": "prepared", "updated": common.now(), "stage": "prepared", "claims_allowed": False})
    verify_run(run)
    return manifest


def verify_run(run, require_active=True):
    run = Path(run)
    manifest, status = common.load(run / "manifest.json"), common.load(run / "status.json")
    if manifest["kind"] != config()["kind"] or status["manifest_sha256"] != common.digest(run / "manifest.json"):
        raise ValueError("Cardinal run manifest differs")
    validate_configuration(manifest["config"], common.load(run / "schedule.json"))
    if common.load(run / "plan.json") != plan() or set(manifest["source_sha256"]) != set(SOURCES):
        raise ValueError("Plan/source coverage differs")
    expected = {"plan.json": manifest["plan_sha256"], "schedule.json": manifest["schedule_sha256"],
                "map-fingerprints.json": manifest["map_fingerprints_sha256"],
                **manifest["input_sha256"], **status["artifacts"]}
    for relative, digest in expected.items():
        if common.digest(run / relative) != digest:
            raise ValueError("Cardinal artifact changed: " + relative)
    for relative, digest in manifest["source_sha256"].items():
        if common.digest(run / "source" / relative) != digest or (require_active and common.digest(ROOT / relative) != digest):
            raise ValueError("Archived/active source changed: " + relative)
    return manifest, status


def score(candidate, opponents, seeds, cfg):
    classification, program = classify_cardinal(candidate["text"], candidate["finish_reason"])
    matches = []
    if program is not None:
        for opponent in opponents:
            rival = common.compile_program(opponent["compiled_ir"])
            for seed in seeds:
                for seat in (0, 1):
                    players = [program, rival] if seat == 0 else [rival, program]
                    origins = ["model", opponent["origin"]] if seat == 0 else [opponent["origin"], "model"]
                    replay = common.play(players, seed, cfg["max_rounds"], origins=origins)
                    common.verify_replay(replay)
                    matches.append({"opponent_id": opponent["id"], "candidate_seat": seat, "replay": replay})
    own = sum(m["replay"]["metrics"]["program_faults"][m["candidate_seat"]] for m in matches)
    other = sum(m["replay"]["metrics"]["program_faults"][1-m["candidate_seat"]] for m in matches)
    authored = bool(classification["valid"])
    if authored != bool(matches):
        raise ValueError("Authorship/game coverage differs")
    valid = authored and own == 0
    game_return = math.fsum(m["replay"]["returns"][m["candidate_seat"]] for m in matches)/len(matches) if matches else None
    points = [m["replay"]["metrics"]["scores"][m["candidate_seat"]] for m in matches]
    return {"classification": classification, "matches": matches, "authorship_valid": authored,
            "runtime_valid": valid, "candidate_program_faults": own, "opponent_program_faults": other,
            "game_return": game_return, "banked_point_yield": math.fsum(points)/len(points) if points else 0.0,
            "positive_banking_games": sum(p > 0 for p in points), "reward": game_return if valid else -1.0,
            "reward_kind": "game_return" if valid else "invalid_authorship_or_candidate_runtime"}


def capability_decision(rows, cfg):
    if len(rows) != 8:
        raise ValueError("Gate requires all eight prescribed draws")
    productive, distinct, selected = [], set(), []
    for i, row in enumerate(rows):
        qualifies = row["runtime_valid"] and len(row["matches"]) == 24 and row["positive_banking_games"] >= 12
        if qualifies:
            productive.append(i)
            ir = row["classification"]["program"]["compiled_ir"]
            digest = common.compile_program(ir).digest
            if digest not in distinct and len(selected) < 2:
                selected.append(i)
            distinct.add(digest)
    passed = len(productive) >= cfg["minimum_productive_programs"] and len(distinct) >= cfg["minimum_productive_distinct_ir"]
    return {"passed": passed,
            "productive_indices": productive, "distinct_productive_ir": len(distinct),
            "selected_indices": selected if passed else [], "rule": "At least4/8 fault-free programs, each banking on at least12/24 calibration games, with at least2 distinct IRs"}


class Cycle(journal.Study):
    def __init__(self, run, backend):
        self.run, self.backend = Path(run), backend
        self.manifest, self.status = verify_run(self.run)
        self.config = self.manifest["config"]
        self.schedule = common.load(self.run / "schedule.json")
        self.prompts = {"cardinal": (self.run / "prompts/cardinal.txt").read_text()}

    def checkpoint(self, label):
        metadata = super().checkpoint(label)
        if metadata is not None:
            optimizer_identity(self.run / "checkpoints" / label, self.config["learning_rate"])
        return metadata

    def save_checkpoint(self, label, parent, report, copy_from=None, update_relative=None):
        if self.checkpoint(label):
            raise ValueError("Checkpoint already committed")
        final = self.run / "checkpoints" / label
        if final.exists():
            final.rename(final.with_name(label + ".incomplete-" + uuid.uuid4().hex))
        temporary = final.with_name("." + label + ".pending-" + uuid.uuid4().hex)
        temporary.parent.mkdir(parents=True, exist_ok=True)
        if copy_from is None:
            files = self.backend.save_checkpoint(temporary)
        else:
            temporary.mkdir()
            for name in ("adapter.safetensors", "optimizer.safetensors"):
                shutil.copyfile(copy_from / name, temporary / name)
            files = {name: common.digest(temporary / name) for name in ("adapter.safetensors", "optimizer.safetensors")}
        identity = optimizer_identity(temporary, self.config["learning_rate"], require_zero=parent is None)
        metadata = {"label": label, "parent": parent, "files": files, "report": report,
                    "optimizer_identity": identity, "complete": True,
                    "manifest_sha256": self.status["manifest_sha256"],
                    "model_repo": self.config["model_repo"], "model_revision": self.config["model_revision"]}
        common.atomic_json(temporary / "metadata.json", metadata)
        temporary.rename(final)
        transaction = {f"checkpoints/{label}/{name}": digest for name, digest in files.items()}
        transaction[f"checkpoints/{label}/metadata.json"] = common.digest(final / "metadata.json")
        if update_relative:
            self.existing(update_relative)
            common.atomic_json(self.run / update_relative, report)
            transaction[update_relative] = common.digest(self.run / update_relative)
        self.status["artifacts"].update(transaction)
        self.state()
        return metadata

    def generation(self, name, seed, checkpoint, prompt="cardinal"):
        return super().generation(name, seed, checkpoint, prompt)

    def admit(self):
        common.admission(self.config)
        if "execution_budget" in self.status:
            journal.execution_budget(self.config, self.status["execution_budget"])

    def capability(self):
        previous = self.existing("capability.json")
        if previous is not None:
            return previous
        self.activate("base")
        opponents = [{"id": n, "origin": "scripted", "compiled_ir": common.reference_programs()[n]}
                     for n in ("nearest", "value")]
        records = [self.generation(f"capability-{i}", seed, "base")
                   for i, seed in enumerate(self.schedule["capability_generation_seeds"])]
        rows = [{"generation": r["name"], "generation_sha256": self.status["artifacts"][f"generations/{r['name']}.json"],
                 **score(r, opponents, self.schedule["calibration_maps"], self.config)} for r in records]
        result = {"source_checkpoint": "base", "source_metadata_sha256": self.status["artifacts"]["checkpoints/base/metadata.json"],
                  "seeds": self.schedule["calibration_maps"], "opponents": opponents, "rows": rows,
                  **capability_decision(rows, self.config)}
        return self.commit("capability.json", result)

    def opponents(self, capability):
        if not capability["passed"]:
            raise ValueError("Productive-play gate has not passed")
        previous = self.existing("opponents.json")
        if previous is not None:
            return previous
        selected = list(capability["opponents"])
        for i in capability["selected_indices"]:
            row = capability["rows"][i]
            ir = row["classification"]["program"]["compiled_ir"]
            selected.append({"id": row["generation"], "origin": "model", "compiled_ir": ir,
                "ir_sha256": common.compile_program(ir).digest, "generation_sha256": row["generation_sha256"],
                "source_path": f"generations/{row['generation']}.json", "source_checkpoint": "base",
                "source_metadata_sha256": capability["source_metadata_sha256"]})
        return self.commit("opponents.json", {"selected": selected,
            "capability_sha256": self.status["artifacts"]["capability.json"], "frozen": True})

    def group(self, number, opponents):
        label = f"cycle{number}"
        if self.checkpoint(label):
            return self.checkpoint(label)["report"]
        source = "base" if number == 1 else f"cycle{number-1}"
        self.activate(source)
        folder = f"groups/{number}"
        records = [self.generation(f"train-cycle{number}-{i}", seed, source)
                   for i, seed in enumerate(self.schedule["train_generation_seeds"][str(number)])]
        hashes = {r["name"]: self.status["artifacts"][f"generations/{r['name']}.json"] for r in records}
        scores = self.existing(folder + "/scores.json")
        if scores is None:
            rows = [{"generation": r["name"], "generation_sha256": hashes[r["name"]],
                     **score(r, opponents["selected"], self.schedule["train_maps"][str(number)], self.config)} for r in records]
            rewards = [r["reward"] for r in rows]
            advantages = journal.advantages(rewards)
            scores = self.commit(folder + "/scores.json", {"cycle": number, "source_checkpoint": source,
                "source_metadata_sha256": self.status["artifacts"][f"checkpoints/{source}/metadata.json"],
                "opponents_sha256": self.status["artifacts"]["opponents.json"],
                "seeds": self.schedule["train_maps"][str(number)], "rows": rows, "rewards": rewards,
                "advantages": advantages, "zero_signal": not any(advantages)})
        if (scores["source_checkpoint"] != source or {r["generation"]: r["generation_sha256"] for r in scores["rows"]} != hashes
                or journal.advantages(scores["rewards"]) != scores["advantages"]
                or scores["opponents_sha256"] != self.status["artifacts"]["opponents.json"]):
            raise ValueError("Saved on-policy group provenance differs")
        report = {"cycle": number, "source_checkpoint": source,
            "source_metadata_sha256": self.status["artifacts"][f"checkpoints/{source}/metadata.json"],
            "scores_sha256": self.status["artifacts"][folder + "/scores.json"], "generation_sha256": hashes,
            "rewards": scores["rewards"], "advantages": scores["advantages"],
            "training_tokens": sum(len(r["completion_ids"]) for r in records),
            "all_runtime_invalid": not any(r["runtime_valid"] for r in scores["rows"])}
        if scores["zero_signal"]:
            report.update(outcome="skipped_zero_variance", optimizer_steps=0, adapter_l2_delta=0.0,
                          likelihood_gate_path=None, likelihood_gate_sha256=None)
            self.save_checkpoint(label, source, report, copy_from=self.run / "checkpoints" / source,
                                 update_relative=folder + "/update.json")
        else:
            common.admission(self.config)
            self.state(stage=folder + "/likelihood")
            gate = self.backend.likelihood_gate(records)
            gate.update(generation_sha256=hashes, source_checkpoint=source,
                        source_metadata_sha256=report["source_metadata_sha256"], generation_names=list(hashes))
            gate_path = folder + "/gate-" + uuid.uuid4().hex + ".json"
            self.commit(gate_path, gate)
            if not gate["passed"]:
                raise ValueError("Full completion likelihood gate failed")
            self.state(stage=folder + "/optimizer")
            update = self.backend.update(records, scores["advantages"])
            if update["optimizer_steps"] != 1 or update["adapter_l2_delta"] <= 0:
                raise ValueError("No genuine optimizer update")
            update["gradient_likelihood_gate"]["generation_names"] = list(hashes)
            report.update(update, outcome="updated", likelihood_gate_path=gate_path,
                          likelihood_gate_sha256=self.status["artifacts"][gate_path])
            self.save_checkpoint(label, source, report, update_relative=folder + "/update.json")
        return report

    def evaluate(self, opponents):
        if any(self.checkpoint(label) is None for label in ENDPOINTS):
            raise ValueError("Both fixed evaluation endpoints required")
        paths = []
        for endpoint in ENDPOINTS:
            self.activate(endpoint)
            for i, seed in enumerate(self.schedule["eval_generation_seeds"]):
                name = f"eval-{endpoint}-{i}"
                self.generation(name, seed, endpoint)
                paths.append(f"generations/{name}.json")
        seal = {"checkpoints": {e: self.status["artifacts"][f"checkpoints/{e}/metadata.json"] for e in ENDPOINTS},
                "generations": {p: self.status["artifacts"][p] for p in paths}, "draws": 24, "scoring_allowed": True}
        previous = self.existing("evaluation/sealed.json")
        if previous is None:
            self.commit("evaluation/sealed.json", seal)
        elif previous != seal:
            raise ValueError("Evaluation seal differs")
        for endpoint in ENDPOINTS:
            relative = f"evaluation/{endpoint}/holdout.json"
            if self.existing(relative) is not None:
                continue
            self.admit()
            rows = []
            for i in range(12):
                name = f"eval-{endpoint}-{i}"
                record = self.existing(f"generations/{name}.json")
                rows.append({"generation": name, "generation_sha256": self.status["artifacts"][f"generations/{name}.json"],
                             **score(record, opponents["selected"], self.schedule["holdout_maps"], self.config)})
            self.commit(relative, {"checkpoint": endpoint, "prompt_condition": "cardinal", "split": "holdout",
                "training_use": False, "seeds": self.schedule["holdout_maps"], "opponents": opponents["selected"],
                "opponents_sha256": self.status["artifacts"]["opponents.json"],
                "seal_sha256": self.status["artifacts"]["evaluation/sealed.json"], "prescribed_program_draws": 12, "scores": rows})
        self.admit()
        self.state(state="complete", stage="complete", claims_allowed=True)

    def run_all(self):
        self.initialize()
        capability = self.capability()
        if not capability["passed"]:
            self.state(state="capability_failed", stage="productive_play_gate", claims_allowed=False)
            return
        opponents = self.opponents(capability)
        streak = 0
        for number in range(1, 4):
            report = self.group(number, opponents)
            streak = streak + 1 if report["all_runtime_invalid"] else 0
            if streak == 2:
                self.state(state="capability_failed", stage="two_all_invalid_groups", claims_allowed=False)
                return
        self.evaluate(opponents)


def worker(run, model_dir, lock_fds):
    journal.verify_lock_descriptors(run, lock_fds)
    manifest, status = verify_run(run)
    cfg = manifest["config"]
    if "execution_budget" not in status:
        raise ValueError("Parent must commit durable execution budget before model work")
    budget, remaining = journal.execution_budget(cfg, status["execution_budget"])
    if {n: importlib.metadata.version(n) for n in manifest["expected_installed_versions"]} != manifest["expected_installed_versions"]:
        raise ValueError("Installed versions differ")
    distribution = importlib.metadata.distribution("mlx-lm")
    for name, digest in manifest["library_source_sha256"].items():
        if common.digest(distribution.locate_file(name)) != digest:
            raise ValueError("Installed MLX source differs")
    if common.digest(Path(model_dir) / "model-index.json") != manifest["model_index_sha256"]:
        raise ValueError("Model index differs")
    index = common.verify_model_index(model_dir, cfg)
    local.verify_metadata(model_dir)
    if any(index["files"].get(name) != digest for name, digest in local.WEIGHT_HASHES.items()):
        raise ValueError("Pinned model shards differ")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    engine = Cycle(run, None)
    measurement = local.prompt_measurement(tokenizer, engine.prompts["cardinal"], cfg)
    old = engine.existing("prompt-measurement.json")
    if old is None:
        engine.commit("prompt-measurement.json", measurement)
    elif old != measurement:
        raise ValueError("Prompt/tokenizer changed")
    if not measurement["admitted"]:
        raise ValueError("Clean prompt plus2048 completion tokens exceed context cap")
    del tokenizer
    budget, remaining = journal.execution_budget(cfg, budget)
    backend = local.MlxFinalBlockPolicy(model_dir, {**cfg, "deadline": budget["hard_stop"], "worker_seconds": remaining})
    engine = Cycle(run, backend)
    runtime = {"model_index": index, "execution_budget": budget, **backend.runtime}
    parent_runtime = common.load(Path(run) / "inputs/parent-runtime.json")
    for key in ("model_index", "versions", "arithmetic", "prefix_mode", "memory_limit_bytes", "eos_token_ids",
                "trainable_tensor_names", "trainable_parameters", "packed_integer_tensors"):
        if runtime[key] != parent_runtime[key]:
            raise ValueError("Runtime differs from verified parent: " + key)
    old = engine.existing("runtime.json")
    if old is None:
        engine.commit("runtime.json", runtime)
    elif old != runtime:
        raise ValueError("Runtime identity changed")
    engine.initialize()
    engine.commit("smoke-" + str(time.time_ns()) + ".json", backend.mechanics_smoke(engine.prompts["cardinal"]))
    engine.run_all()
    return engine.status


def launch(run, model_dir):
    run, model_dir = Path(run).resolve(), Path(model_dir).resolve()
    manifest, status = verify_run(run)
    if status["state"] in {"complete", "capability_failed"}:
        return status
    with common.process_lock(LOCK) as global_fd, common.process_lock(run / "run.lock") as run_fd:
        status = common.load(run / "status.json")
        budget, timeout = journal.execution_budget(manifest["config"], status.get("execution_budget"))
        status.update(execution_budget=budget, updated=common.now())
        common.atomic_json(run / "status.json", status)
        child = subprocess.Popen([sys.executable, "-m", "selfplay.program_cardinal_cycle", "_worker",
            "--run", str(run), "--model-dir", str(model_dir), "--lock-fds", f"{global_fd},{run_fd}"],
            cwd=ROOT, pass_fds=(global_fd, run_fd), start_new_session=True)
        started = time.monotonic()
        try:
            while child.poll() is None:
                if time.monotonic()-started >= timeout:
                    raise TimeoutError("Absolute cardinal deadline or nonrenewable six-hour wall budget reached")
                usage = subprocess.run(["ps", "-o", "rss=", "-p", str(child.pid)], capture_output=True, text=True, timeout=5)
                if usage.returncode == 0 and usage.stdout.strip() and int(usage.stdout.strip())*1024 > manifest["config"]["memory_limit_bytes"]:
                    raise MemoryError("Resident memory exceeds28GiB ceiling")
                time.sleep(1)
            if child.returncode:
                raise RuntimeError(f"Cardinal worker exited{child.returncode}; committed artifacts remain resumable")
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
            status.update(state="stopped", updated=common.now(), claims_allowed=False,
                          error_type=type(error).__name__, reason=str(error))
            common.atomic_json(run / "status.json", status)
            raise
    return common.load(run / "status.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "prepare", "run", "_worker"))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/qwen38-27b-4bit")
    parser.add_argument("--lock-fds")
    args = parser.parse_args()
    if args.command == "plan":
        result = plan()
    elif args.run is None:
        parser.error("--run is required")
    elif args.command == "prepare":
        result = prepare(args.run, args.model_dir)
    elif args.command == "run":
        result = launch(args.run, args.model_dir)
    else:
        result = worker(args.run, args.model_dir, args.lock_fds)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
