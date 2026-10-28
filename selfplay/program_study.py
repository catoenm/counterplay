"""Matched MLX coding-RL study. Preparation/tests are CPU-only; run is explicit.

The numerical backend and environment are imported unchanged. This module owns
the two-arm state machine, frozen opponent pool, reward attribution and journal.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import time
import uuid

from selfplay import program_local_rl as local
from selfplay.program_arena.game import RobotArena
from scripts.cloud import train_program_rl as common

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "runs/local-neural.lock"
ARMS = ("game_rl", "validity_rl")
ENDPOINTS = ("base", "game_rl-round3", "validity_rl-round3")
SOURCES = common.SOURCES + ["selfplay/program_local_rl.py", "selfplay/program_study.py",
                            "scripts/analyze_program_study.py"]
PARENT = ROOT / "runs/program-local-27b-compact-v1"
PLAN = ROOT / "runs/program-next-study-plan.json"


def object_hash(value):
    return hashlib.sha256(common.canonical(value).encode()).hexdigest()


def config():
    return {**common.DEFAULTS, **local.condition_config("feedback-compact-r1-v1"),
            "kind": "matched_program_study", "model_repo": local.MODEL_REPO,
            "model_revision": local.MODEL_REVISION, "iterations": 3, "group_size": 8,
            "worker_seconds": 13500, "deadline": "2026-10-22T07:30:00-07:00",
            "algorithm": "matched_game_and_validity_RLOO", "eval_samples": 8,
            "opponent_attempt_cap": 8, "opponent_count": 2,
            "scope": "Fixed self-generated opponent pool, matched reward objectives; best-effort deadline"}


def schedule():
    return {"round_order": [["game_rl", 1], ["validity_rl", 1], ["validity_rl", 2],
                            ["game_rl", 2], ["game_rl", 3], ["validity_rl", 3]],
            "train_maps": {str(i): [6100000 + i * 100 + j for j in range(2)] for i in range(1, 4)},
            "calibration_maps": list(range(6200000, 6200003)),
            "validation_maps": list(range(6300000, 6300004)),
            "holdout_maps": list(range(6400000, 6400006)),
            "train_generation_seeds": {str(i): [8100000 + i * 100 + j for j in range(8)] for i in range(1, 4)},
            "opponent_generation_seeds": list(range(8200000, 8200008)),
            "eval_generation_seeds": {"compact": list(range(8300000, 8300008)),
                                      "plain": list(range(8400000, 8400004))}}


def validate_configuration(value, timetable):
    if value != config() or timetable != schedule():
        raise ValueError("This preregistered study does not accept config/schedule overrides")


def map_layout(seed):
    game = RobotArena(seed, 48)
    # Site list order is observable through site(i,...), so preserve it.
    return {"grid_size": game.size, "capacity": game.capacity, "max_energy": game.max_energy,
            "max_rounds": game.max_rounds, "sites": game.sites,
            "homes": [[r["home_x"], r["home_y"]] for r in game.robots]}


def find_replay_seeds(value, seeds):
    if isinstance(value, dict):
        if value.get("kind") == "program_match" and "initial_state" in value:
            seeds.add(value["seed"])
            return
        for child in value.values():
            find_replay_seeds(child, seeds)
    elif isinstance(value, list):
        for child in value:
            find_replay_seeds(child, seeds)


def map_fingerprints(timetable):
    prior, evidence = set(), {}
    # Prior completed program experiments only; skip sources, checkpoints and
    # raw generations, none of which contain played maps.
    for directory in sorted((ROOT / "runs").glob("program-*")):
        if not directory.is_dir() or directory.name.startswith("program-study"):
            continue
        for path in sorted(directory.rglob("*")):
            if path.suffix not in {".json", ".jsonl"} or any(
                    part in {"source", "checkpoints", "generations", "originals", "inputs"}
                    for part in path.relative_to(directory).parts):
                continue
            before = len(prior)
            if path.suffix == ".jsonl":
                for line in path.read_text().splitlines():
                    if line.strip():
                        find_replay_seeds(json.loads(line), prior)
            else:
                find_replay_seeds(common.load(path), prior)
            if len(prior) != before:
                evidence[str(path.relative_to(ROOT))] = common.digest(path)
    prior_hashes = {str(seed): object_hash(map_layout(seed)) for seed in sorted(prior)}
    families = {"train": [s for row in timetable["train_maps"].values() for s in row],
                **{k: timetable[k + "_maps"] for k in ("calibration", "validation", "holdout")}}
    used = set(prior_hashes.values())
    result = {}
    for family, seeds in families.items():
        result[family] = []
        for seed in seeds:
            layout = map_layout(seed)
            fingerprint = object_hash(layout)
            if fingerprint in used or seed in prior:
                raise ValueError("Observable map overlap; revise the plan before any draw")
            used.add(fingerprint)
            result[family].append({"seed": seed, "sha256": fingerprint, "layout": layout})
    return {"families": result, "prior_seed_fingerprints": prior_hashes,
            "prior_evidence_sha256": evidence,
            "definition": "Initial observable layout including site order; seed and hidden harvest priority excluded"}


def checkpoint_identity(path, expected_metadata=None):
    metadata = common.load(path / "metadata.json")
    if expected_metadata and common.digest(path / "metadata.json") != expected_metadata:
        raise ValueError("Checkpoint metadata hash differs")
    if not metadata["complete"] or set(metadata["files"]) != {"adapter.safetensors", "optimizer.safetensors"}:
        raise ValueError("Incomplete/unexpected checkpoint files")
    for filename, expected in metadata["files"].items():
        if common.digest(path / filename) != expected:
            raise ValueError("Checkpoint tensor file hash differs")
    if (metadata["model_repo"], metadata["model_revision"]) != (local.MODEL_REPO, local.MODEL_REVISION):
        raise ValueError("Checkpoint model identity differs")
    return metadata


def prepare(run, model_dir, plan=PLAN):
    run, model_dir = Path(run), Path(model_dir)
    if run.exists():
        raise FileExistsError("Study directory exists; explicit run resumes compatible artifacts")
    proposal = common.load(plan)
    if proposal["status"] != "IMPLEMENTATION_AUTHORIZED_NOT_PREPARED_NOT_LAUNCHED":
        raise ValueError("Expected the explicitly authorized best-effort study plan")
    parent_manifest, parent_status = common.verify_sources(PARENT)
    if parent_status["state"] != "complete":
        raise ValueError("Parent mechanism study has not completed")
    initial = checkpoint_identity(PARENT / "checkpoints/base")
    if initial["report"] is not None or initial["files"]["adapter.safetensors"] != proposal[
            "controlled_comparison"]["initial_adapter_sha256"]:
        raise ValueError("Initial adapter is not the preregistered zero-update checkpoint")
    if initial["files"]["optimizer.safetensors"] != proposal["controlled_comparison"]["initial_optimizer_sha256"]:
        raise ValueError("Initial optimizer identity differs")
    cfg, timetable = config(), schedule()
    fingerprints = map_fingerprints(timetable)
    run.mkdir(parents=True)
    common.atomic_json(run / "plan.json", proposal)
    common.atomic_json(run / "schedule.json", timetable)
    common.atomic_json(run / "map-fingerprints.json", fingerprints)
    copied = {}

    def copy_input(source, destination):
        target = run / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied[destination] = common.digest(target)
        return target

    copy_input(PARENT / "prompt.txt", "prompts/compact.txt")
    copy_input(local.PLAIN_PROMPT, "prompts/plain.txt")
    if copied["prompts/compact.txt"] != proposal["controlled_comparison"]["training_prompt"]["sha256"]:
        raise ValueError("Compact prompt differs from plan")
    if copied["prompts/plain.txt"] != proposal["evaluation"]["plain_transfer_prompt"]["sha256"]:
        raise ValueError("Plain prompt differs from plan")
    copy_input(PARENT / "manifest.json", "inputs/parent-manifest.json")
    copy_input(PARENT / "status.json", "inputs/parent-status.json")
    if common.digest(PARENT / "runtime.json") != parent_status["artifacts"]["runtime.json"]:
        raise ValueError("Parent runtime identity is not bound to its artifact journal")
    copy_input(PARENT / "runtime.json", "inputs/parent-runtime.json")
    for filename in ["metadata.json", *initial["files"]]:
        copy_input(PARENT / "checkpoints/base" / filename, "inputs/base/" + filename)
    for name in ("opponent-1-1", "candidate-1-1"):
        source = PARENT / "generations" / (name + ".json")
        if common.digest(source) != parent_status["artifacts"]["generations/" + name + ".json"]:
            raise ValueError("Historical opponent is not bound to its source run")
        copy_input(source, "inputs/historical/" + name + ".json")
    tournament = ROOT / "runs/program-tournament-v1"
    tournament_plan = common.load(tournament / "plan.json")
    copy_input(tournament / "plan.json", "inputs/tournament-plan.json")
    eval_opponents = []
    for name in ("nearest", "value"):
        eval_opponents.append({"id": name, "origin": "scripted", "compiled_ir": common.reference_programs()[name]})
    for name in ("controller-python-1", "qwen38-27b-1-r2"):
        relative = f"inputs/{name}/generation.json"
        source = tournament / relative
        if common.digest(source) != tournament_plan["input_sha256"][relative]:
            raise ValueError("Tournament generation identity differs")
        copy_input(source, "inputs/eval/" + name + ".json")
        row = tournament_plan["roster"][name]
        ir = row["python_program"]["compiled_ir"]
        if common.compile_program(ir).digest != row["ir_sha256"]:
            raise ValueError("Frozen tournament IR differs")
        eval_opponents.append({"id": name, "origin": "model", "compiled_ir": ir,
                               "ir_sha256": row["ir_sha256"], "generation_sha256": common.digest(source)})
    common.atomic_json(run / "evaluation-opponents.json", eval_opponents)
    copied["evaluation-opponents.json"] = common.digest(run / "evaluation-opponents.json")
    copy_input(model_dir / "model-index.json", "inputs/model-index.json")
    hashes = {}
    for relative in SOURCES:
        target = run / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
        hashes[relative] = common.digest(target)
    manifest = {"format_version": 1, "kind": "matched_program_study", "created": common.now(),
                "config": cfg, "plan_sha256": common.digest(run / "plan.json"),
                "schedule_sha256": common.digest(run / "schedule.json"),
                "map_fingerprints_sha256": common.digest(run / "map-fingerprints.json"),
                "source_sha256": hashes, "input_sha256": copied,
                "prompt_sha256": {p: copied[f"prompts/{p}.txt"] for p in ("compact", "plain")},
                "model_index_sha256": copied["inputs/model-index.json"],
                "library_source_sha256": parent_manifest["library_source_sha256"],
                "expected_installed_versions": parent_manifest["expected_installed_versions"],
                "external_spend_usd": 0, "claims_require_complete": True}
    common.atomic_json(run / "manifest.json", manifest)
    common.atomic_json(run / "status.json", {"manifest_sha256": common.digest(run / "manifest.json"),
                       "artifacts": {}, "state": "prepared", "updated": common.now(),
                       "stage": "prepared", "claims_allowed": False})
    verify_run(run)
    return manifest


def verify_run(run, require_active=True):
    run = Path(run)
    manifest, status = common.load(run / "manifest.json"), common.load(run / "status.json")
    if manifest["kind"] != "matched_program_study" or status["manifest_sha256"] != common.digest(run / "manifest.json"):
        raise ValueError("Study manifest changed")
    if set(manifest["source_sha256"]) != set(SOURCES):
        raise ValueError("Required source archive coverage differs")
    validate_configuration(manifest["config"], common.load(run / "schedule.json"))
    expected = {"plan.json": manifest["plan_sha256"], "schedule.json": manifest["schedule_sha256"],
                "map-fingerprints.json": manifest["map_fingerprints_sha256"],
                **manifest["input_sha256"], **status["artifacts"]}
    for relative, sha in expected.items():
        if common.digest(run / relative) != sha:
            raise ValueError("Study artifact changed: " + relative)
    for relative, sha in manifest["source_sha256"].items():
        if common.digest(run / "source" / relative) != sha:
            raise ValueError("Archived source changed: " + relative)
        if require_active and common.digest(ROOT / relative) != sha:
            raise ValueError("Active source changed: " + relative)
    return manifest, status


def score(candidate, opponents, seeds, cfg, arm):
    if arm not in ARMS:
        raise ValueError("Unknown objective")
    row = common.score_program(candidate, opponents, seeds, cfg)
    own_faults = sum(m["replay"]["metrics"]["program_faults"][m["candidate_seat"]] for m in row["matches"])
    other_faults = sum(m["replay"]["metrics"]["program_faults"][1-m["candidate_seat"]] for m in row["matches"])
    compiled = row["classification"]["valid"]
    valid = compiled and own_faults == 0
    reward = (-1.0 if not valid else (row["game_return"] if arm == "game_rl" else 0.0))
    points = [m["replay"]["metrics"]["scores"][m["candidate_seat"]] for m in row["matches"]]
    row.update(reward=reward, reward_kind="game_return" if valid and arm == "game_rl" else
               ("validity_only" if valid else "invalid_authorship_or_candidate_runtime"),
               candidate_program_faults=own_faults, opponent_program_faults=other_faults,
               authorship_valid=compiled, runtime_valid=valid,
               banked_point_yield=math.fsum(points)/len(points) if points else 0.0)
    return row


def advantages(rewards):
    if len(rewards) != 8 or any(not math.isfinite(r) or not -1 <= r <= 1 for r in rewards):
        raise ValueError("Expected eight bounded rewards")
    if max(rewards) - min(rewards) <= 1e-12:
        return [0.0] * 8
    return common.rloo_advantages(rewards)


def load_exact(backend, folder, files):
    """Frozen backend already verifies adapter bytes; additionally check optimizer schema/state."""
    optimizer_arrays = None
    if hasattr(backend, "optimizer"):
        import numpy as np
        from mlx.utils import tree_flatten
        from safetensors.numpy import load_file
        optimizer_arrays = load_file(str(folder / "optimizer.safetensors"))
        expected = dict(tree_flatten(backend.optimizer.state))
        if set(optimizer_arrays) != set(expected):
            raise ValueError("Optimizer tensor names differ")
        for name, value in optimizer_arrays.items():
            old = np.array(expected[name])
            if value.shape != old.shape or value.dtype != old.dtype or not np.isfinite(value).all():
                raise ValueError("Optimizer shape/dtype/finiteness mismatch")
    backend.load_checkpoint(folder, files)
    if optimizer_arrays is not None:
        actual = dict(tree_flatten(backend.optimizer.state))
        if any(not np.array_equal(value, np.array(actual[name])) for name, value in optimizer_arrays.items()):
            raise ValueError("Optimizer restore was not exact")


def execution_budget(cfg, previous=None, at=None):
    """The wall-clock allowance starts once and is never renewed by resumption."""
    at = at or datetime.now(timezone.utc)
    first = datetime.fromisoformat(previous["first_started"]) if previous else at
    stop = min(datetime.fromisoformat(cfg["deadline"]), first + timedelta(seconds=cfg["worker_seconds"]))
    budget = {"first_started": first.isoformat(), "hard_stop": stop.isoformat(),
              "maximum_seconds": cfg["worker_seconds"], "resume_counts_wall_time": True}
    if previous is not None and previous != budget:
        raise ValueError("Saved whole-study execution budget changed")
    remaining = (stop - at).total_seconds()
    if remaining <= 0:
        raise TimeoutError("Whole-study wall budget or absolute deadline exhausted")
    return budget, remaining


def verify_lock_descriptors(run, descriptors):
    if not descriptors:
        raise ValueError("Worker requires inherited global/run lock descriptors")
    values = [int(value) for value in descriptors.split(",")]
    if len(values) != 2 or len(set(values)) != 2:
        raise ValueError("Two distinct inherited lock descriptors required")
    for fd, path in zip(values, (LOCK, Path(run) / "run.lock")):
        inherited, named = os.fstat(fd), os.stat(path)
        if (inherited.st_dev, inherited.st_ino) != (named.st_dev, named.st_ino) or not stat.S_ISREG(inherited.st_mode):
            raise ValueError("Inherited descriptor does not identify the expected lock file")
        # Inherited descriptors share the parent's open-file description. This
        # confirms/reacquires its existing lock without releasing either lock.
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


class Study:
    def __init__(self, run, backend):
        self.run, self.backend = Path(run), backend
        self.manifest, self.status = verify_run(self.run)
        self.config = self.manifest["config"]
        self.schedule = common.load(self.run / "schedule.json")
        self.prompts = {p: (self.run / f"prompts/{p}.txt").read_text() for p in ("compact", "plain")}

    def state(self, state="running", stage=None, **fields):
        self.status.update(state=state, updated=common.now(), **fields)
        if stage is not None:
            self.status["stage"] = stage
        common.atomic_json(self.run / "status.json", self.status)

    def existing(self, relative):
        path = self.run / relative
        if relative in self.status["artifacts"]:
            if common.digest(path) != self.status["artifacts"][relative]:
                raise ValueError("Committed artifact changed")
            return common.load(path)
        if path.exists():
            path.rename(path.with_name(path.name + ".uncommitted-" + uuid.uuid4().hex))
        return None

    def commit(self, relative, value):
        if relative in self.status["artifacts"]:
            raise ValueError("Refusing to replace a committed artifact")
        self.existing(relative)
        common.atomic_json(self.run / relative, value)
        self.status["artifacts"][relative] = common.digest(self.run / relative)
        self.state()
        return value

    def checkpoint(self, label):
        relative = f"checkpoints/{label}/metadata.json"
        if relative not in self.status["artifacts"]:
            return None
        return checkpoint_identity(self.run / "checkpoints" / label, self.status["artifacts"][relative])

    def activate(self, label):
        metadata = self.checkpoint(label)
        if metadata is None:
            raise ValueError("Missing committed source checkpoint")
        load_exact(self.backend, self.run / "checkpoints" / label, metadata["files"])
        return metadata

    def save_checkpoint(self, label, parent, report, copy_from=None, update_relative=None):
        # A single status replacement commits all tensor, metadata and update
        # paths. A crash before that replacement leaves only preserved orphans.
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
            for filename in ("adapter.safetensors", "optimizer.safetensors"):
                shutil.copyfile(copy_from / filename, temporary / filename)
            files = {name: common.digest(temporary / name) for name in ("adapter.safetensors", "optimizer.safetensors")}
        metadata = {"label": label, "parent": parent, "files": files, "report": report,
                    "complete": True, "manifest_sha256": self.status["manifest_sha256"],
                    "model_repo": self.config["model_repo"], "model_revision": self.config["model_revision"]}
        common.atomic_json(temporary / "metadata.json", metadata)
        temporary.rename(final)
        transaction = {f"checkpoints/{label}/{name}": sha for name, sha in files.items()}
        transaction[f"checkpoints/{label}/metadata.json"] = common.digest(final / "metadata.json")
        if update_relative:
            self.existing(update_relative)
            common.atomic_json(self.run / update_relative, report)
            transaction[update_relative] = common.digest(self.run / update_relative)
        self.status["artifacts"].update(transaction)
        self.state()
        return metadata

    def initialize(self):
        if not self.checkpoint("base"):
            source = self.run / "inputs/base"
            original = checkpoint_identity(source)
            load_exact(self.backend, source, original["files"])
            self.save_checkpoint("base", None, None, copy_from=source)
        self.activate("base")

    def generation(self, name, seed, checkpoint, prompt="compact"):
        relative = f"generations/{name}.json"
        record = self.existing(relative)
        expected = {"name": name, "seed": seed, "checkpoint": checkpoint, "prompt_condition": prompt,
                    "prompt_sha256": self.manifest["prompt_sha256"][prompt],
                    "source_metadata_sha256": self.status["artifacts"][f"checkpoints/{checkpoint}/metadata.json"]}
        if record is None:
            common.admission(self.config)
            self.state(stage=name)
            record = {**self.backend.sample(self.prompts[prompt], seed), **expected,
                      "model_repo": self.config["model_repo"], "model_revision": self.config["model_revision"],
                      "authorship": "fresh_learner_sample", "api_training_data": False}
            if len(record["completion_ids"]) != len(record["sampled_token_logprobs"]) or not record["completion_ids"]:
                raise ValueError("Empty/incomplete sampled token provenance")
            self.commit(relative, record)
        if any(record.get(k) != value for k, value in expected.items()):
            raise ValueError("Generation identity mismatch")
        return record

    def opponents(self):
        existing = self.existing("opponents.json")
        if existing:
            return existing
        self.activate("base")
        selected = [{"id": name, "origin": "scripted", "compiled_ir": common.reference_programs()[name]}
                    for name in ("nearest", "value")]
        heldout_ir = {common.compile_program(r["compiled_ir"]).digest for r in
                      common.load(self.run / "evaluation-opponents.json") if r["origin"] == "model"}
        attempts, hashes, fresh_count = [], set(), 0
        candidates = [("historical/" + name, common.load(self.run / f"inputs/historical/{name}.json"),
                       f"inputs/historical/{name}.json") for name in ("opponent-1-1", "candidate-1-1")]
        candidates += [(f"pool-{i}", None, f"generations/pool-{i}.json") for i in range(8)]
        for name, row, relative in candidates:
            fresh = row is None
            if fresh and fresh_count == 2:
                break
            if fresh:
                i = int(name.split("-")[1])
                row = self.generation(name, self.schedule["opponent_generation_seeds"][i], "base")
            check = score(row, selected[:1], self.schedule["calibration_maps"], self.config, "game_rl")
            points = [m["replay"]["metrics"]["scores"][m["candidate_seat"]] for m in check["matches"]]
            ir = check["classification"].get("program", {}).get("compiled_ir")
            fingerprint = common.compile_program(ir).digest if ir else None
            useful = (check["runtime_valid"] and sum(p > 0 for p in points) >= 3
                      and fingerprint not in hashes and fingerprint not in heldout_ir)
            attempts.append({"id": name, "source_path": relative, "source_sha256": common.digest(self.run / relative),
                             "fresh": fresh, "useful": useful, "ir_sha256": fingerprint,
                             "duplicates_heldout_model": fingerprint in heldout_ir, "behavior": check})
            if useful:
                hashes.add(fingerprint)
                selected.append({"id": name, "origin": "model", "compiled_ir": ir,
                                 "ir_sha256": fingerprint, "generation_sha256": common.digest(self.run / relative),
                                 "source_path": relative, "source_checkpoint": row.get("checkpoint"),
                                 "source_metadata_sha256": row.get("source_metadata_sha256")})
                fresh_count += int(fresh)
        result = {"selected": selected, "attempts": attempts, "source_checkpoint": "base",
                  "calibration_seeds": self.schedule["calibration_maps"],
                  "passed": len(selected) >= 4, "frozen_for_both_arms": True}
        return self.commit("opponents.json", result)

    def group(self, arm, number, opponents):
        label = f"{arm}-round{number}"
        if self.checkpoint(label):
            return self.checkpoint(label)["report"]
        source = "base" if number == 1 else f"{arm}-round{number-1}"
        self.activate(source)  # Also resets Adam step/moments after another arm.
        folder = f"groups/{arm}/{number}"
        records = [self.generation(f"{arm}-round{number}-{i}", seed, source)
                   for i, seed in enumerate(self.schedule["train_generation_seeds"][str(number)])]
        scores = self.existing(folder + "/scores.json")
        hashes = {r["name"]: self.status["artifacts"]["generations/" + r["name"] + ".json"] for r in records}
        if scores is None:
            rows = [{"generation": r["name"], "generation_sha256": hashes[r["name"]],
                     **score(r, opponents["selected"], self.schedule["train_maps"][str(number)], self.config, arm)}
                    for r in records]
            rewards = [r["reward"] for r in rows]
            adv = advantages(rewards)
            scores = self.commit(folder + "/scores.json", {"arm": arm, "round": number,
                "source_checkpoint": source, "source_metadata_sha256": self.status["artifacts"][f"checkpoints/{source}/metadata.json"],
                "opponents_sha256": self.status["artifacts"]["opponents.json"],
                "seeds": self.schedule["train_maps"][str(number)], "rows": rows,
                "rewards": rewards, "advantages": adv, "zero_signal": not any(adv)})
        expected = (source, hashes, self.status["artifacts"]["opponents.json"])
        actual = (scores["source_checkpoint"], {r["generation"]: r["generation_sha256"] for r in scores["rows"]}, scores["opponents_sha256"])
        if actual != expected or advantages(scores["rewards"]) != scores["advantages"]:
            raise ValueError("Saved group source/advantage identity differs")
        report = {"arm": arm, "round": number, "source_checkpoint": source,
                  "source_metadata_sha256": self.status["artifacts"][f"checkpoints/{source}/metadata.json"],
                  "scores_sha256": self.status["artifacts"][folder + "/scores.json"],
                  "generation_sha256": hashes, "rewards": scores["rewards"], "advantages": scores["advantages"],
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
                        source_metadata_sha256=report["source_metadata_sha256"],
                        generation_names=[r["name"] for r in records])
            gate_path = folder + "/gate-" + uuid.uuid4().hex + ".json"
            self.commit(gate_path, gate)
            if not gate["passed"]:
                raise ValueError("Full completion likelihood gate failed")
            self.state(stage=folder + "/optimizer")
            update = self.backend.update(records, scores["advantages"])
            if update["optimizer_steps"] != 1 or update["adapter_l2_delta"] <= 0:
                raise ValueError("Unexpected optimizer update result")
            update["gradient_likelihood_gate"]["generation_names"] = [r["name"] for r in records]
            report.update(update, outcome="updated", likelihood_gate_path=gate_path,
                          likelihood_gate_sha256=self.status["artifacts"][gate_path])
            self.save_checkpoint(label, source, report, update_relative=folder + "/update.json")
        return report

    def evaluate(self):
        for endpoint in ENDPOINTS:
            if not self.checkpoint(endpoint):
                raise ValueError("Both predetermined round3 endpoints required")
        generation_paths = []
        for endpoint in ENDPOINTS:
            self.activate(endpoint)
            for prompt, seeds in self.schedule["eval_generation_seeds"].items():
                for i, seed in enumerate(seeds):
                    name = f"eval-{endpoint}-{prompt}-{i}"
                    self.generation(name, seed, endpoint, prompt)
                    generation_paths.append("generations/" + name + ".json")
        seal = {"checkpoints": {x: self.status["artifacts"][f"checkpoints/{x}/metadata.json"] for x in ENDPOINTS},
                "generations": {p: self.status["artifacts"][p] for p in generation_paths},
                "draws": 36, "scoring_allowed": True}
        old = self.existing("evaluation/sealed.json")
        if old is None:
            self.commit("evaluation/sealed.json", seal)
        elif old != seal:
            raise ValueError("Sealed evaluation identities differ")
        opponents = common.load(self.run / "evaluation-opponents.json")
        for endpoint in ENDPOINTS:
            for prompt, seeds in self.schedule["eval_generation_seeds"].items():
                for split in ("validation", "holdout"):
                    relative = f"evaluation/{endpoint}/{prompt}/{split}.json"
                    if self.existing(relative) is not None:
                        continue
                    common.admission(self.config)
                    self.state(stage=relative)
                    rows = []
                    for i in range(len(seeds)):
                        name = f"eval-{endpoint}-{prompt}-{i}"
                        record = common.load(self.run / f"generations/{name}.json")
                        rows.append({"generation": name,
                                     "generation_sha256": self.status["artifacts"][f"generations/{name}.json"],
                                     **score(record, opponents, self.schedule[split + "_maps"], self.config, "game_rl")})
                    self.commit(relative, {"checkpoint": endpoint, "prompt_condition": prompt, "split": split,
                        "seeds": self.schedule[split + "_maps"], "opponents": opponents,
                        "opponents_sha256": self.manifest["input_sha256"]["evaluation-opponents.json"],
                        "seal_sha256": self.status["artifacts"]["evaluation/sealed.json"],
                        "prescribed_program_draws": len(seeds), "scores": rows, "training_use": False})
        required = [f"evaluation/{e}/{p}/{s}.json" for e in ENDPOINTS for p in ("compact", "plain")
                    for s in ("validation", "holdout")]
        if not all(p in self.status["artifacts"] for p in required):
            raise ValueError("Incomplete prescribed evaluation")
        self.state(state="complete", stage="complete", claims_allowed=True)

    def run_all(self):
        self.initialize()
        pool = self.opponents()
        if not pool["passed"]:
            self.state(state="capability_failed", stage="opponent_pool", claims_allowed=False)
            return
        invalid_streak = {arm: 0 for arm in ARMS}
        for arm, number in self.schedule["round_order"]:
            report = self.group(arm, number, pool)
            invalid_streak[arm] = invalid_streak[arm] + 1 if report["all_runtime_invalid"] else 0
            if invalid_streak[arm] == 2:
                self.state(state="capability_failed", stage=f"{arm}/two_all_invalid_groups", claims_allowed=False)
                return
        self.evaluate()


def worker(run, model_dir, lock_fds):
    verify_lock_descriptors(run, lock_fds)
    manifest, status = verify_run(run)
    cfg = manifest["config"]
    if "execution_budget" not in status:
        raise ValueError("Parent must commit the whole-study execution budget before model work")
    budget, remaining = execution_budget(cfg, status["execution_budget"])
    versions = {name: importlib.metadata.version(name) for name in manifest["expected_installed_versions"]}
    if versions != manifest["expected_installed_versions"]:
        raise ValueError("Installed versions differ from frozen study")
    distribution = importlib.metadata.distribution("mlx-lm")
    for name, sha in manifest["library_source_sha256"].items():
        if common.digest(distribution.locate_file(name)) != sha:
            raise ValueError("Installed MLX source changed")
    if common.digest(Path(model_dir) / "model-index.json") != manifest["model_index_sha256"]:
        raise ValueError("Model index changed")
    index = common.verify_model_index(model_dir, cfg)
    local.verify_metadata(model_dir)
    if any(index["files"].get(name) != sha for name, sha in local.WEIGHT_HASHES.items()):
        raise ValueError("Pinned model weight identity differs")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    preflight = Study(run, None)
    for prompt in ("compact", "plain"):
        value = local.prompt_measurement(tokenizer, preflight.prompts[prompt], {**cfg, "condition": prompt})
        previous = preflight.existing(f"prompt-measurement-{prompt}.json")
        if previous is None:
            preflight.commit(f"prompt-measurement-{prompt}.json", value)
        elif previous != value:
            raise ValueError("Prompt/tokenizer changed")
        if not value["admitted"]:
            raise ValueError("Prompt exceeds context limit")
    del tokenizer
    # The immutable manifest declares maximum bounds. Effective worker guards
    # enforce the remaining part of the same durable whole-study budget.
    budget, remaining = execution_budget(cfg, budget)
    backend = local.MlxFinalBlockPolicy(model_dir, {**cfg, "deadline": budget["hard_stop"],
                                                  "worker_seconds": remaining})
    study = Study(run, backend)
    runtime = {"model_index": index, "execution_budget": budget, **backend.runtime}
    parent_runtime = common.load(Path(run) / "inputs/parent-runtime.json")
    for key in ("model_index", "versions", "arithmetic", "prefix_mode", "memory_limit_bytes", "eos_token_ids",
                "trainable_tensor_names", "trainable_parameters", "packed_integer_tensors"):
        if runtime[key] != parent_runtime[key]:
            raise ValueError("Numerical runtime differs from the verified parent: " + key)
    previous = study.existing("runtime.json")
    if previous is None:
        study.commit("runtime.json", runtime)
    elif previous != runtime:
        raise ValueError("Runtime identity changed")
    study.initialize()
    smoke = backend.mechanics_smoke(study.prompts["compact"])
    study.commit("smoke-" + str(time.time_ns()) + ".json", smoke)
    study.run_all()
    return study.status


def launch(run, model_dir):
    run, model_dir = Path(run).resolve(), Path(model_dir).resolve()
    manifest, status = verify_run(run)
    if status["state"] in {"complete", "capability_failed"}:
        return status
    with common.process_lock(LOCK) as global_fd, common.process_lock(run / "run.lock") as run_fd:
        status = common.load(run / "status.json")
        budget, timeout = execution_budget(manifest["config"], status.get("execution_budget"))
        status.update(execution_budget=budget, updated=common.now())
        common.atomic_json(run / "status.json", status)
        command = [sys.executable, "-m", "selfplay.program_study", "_worker", "--run", str(run),
                   "--model-dir", str(model_dir), "--lock-fds", f"{global_fd},{run_fd}"]
        child = subprocess.Popen(command, cwd=ROOT, pass_fds=(global_fd, run_fd), start_new_session=True)
        started = time.monotonic()
        try:
            while child.poll() is None:
                if time.monotonic() - started >= timeout:
                    raise TimeoutError("Hard study deadline07:30 or13,500-second limit")
                usage = subprocess.run(["ps", "-o", "rss=", "-p", str(child.pid)],
                                       capture_output=True, text=True, timeout=5)
                if usage.returncode == 0 and usage.stdout.strip() and int(usage.stdout.strip()) * 1024 > manifest["config"]["memory_limit_bytes"]:
                    raise MemoryError("Resident memory exceeds study ceiling")
                time.sleep(1)
            if child.returncode:
                raise RuntimeError(f"Study worker exited{child.returncode}; committed artifacts remain resumable")
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
    parser.add_argument("command", choices=("prepare", "run", "_worker"))
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--model-dir", default=ROOT / "models/qwen38-27b-4bit", type=Path)
    parser.add_argument("--lock-fds")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.run, args.model_dir)
    elif args.command == "run":
        result = launch(args.run, args.model_dir)
    else:
        result = worker(args.run, args.model_dir, args.lock_fds)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
