"""Independent CPU audit of the matched controller-authoring study.

This module never loads a neural model or executes generated Python. It checks
saved tokens and numerical gate reports, but cannot rerun neural likelihoods on
CPU. Game results are recomputed with the hash-verified archived interpreter.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("game_rl", "validity_rl")
ENDPOINTS = ("base", "game_rl-round3", "validity_rl-round3")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def value_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, where):
    if isinstance(expected, float):
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, expected, abs_tol=1e-10, rel_tol=1e-10), where)
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), where)
        for key in expected:
            same(actual[key], expected[key], where + "/" + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), where)
        for i, value in enumerate(expected):
            same(actual[i], value, where + "/" + str(i))
    else:
        require(type(actual) is type(expected) and actual == expected, where)


def safe_path(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), "Artifact path escapes run: " + str(relative))
    return path


@contextmanager
def archived_runtime(run, manifest):
    """Load only trusted, hash-verified archived game/frontend sources."""
    prefix = "_program_study_audit_" + uuid.uuid4().hex
    parent = ModuleType(prefix)
    parent.__path__ = [str(run / "source/selfplay")]
    sys.modules[prefix] = parent
    try:
        required = ["selfplay/program_arena/" + x + ".py" for x in
                    ("__init__", "dsl", "game", "runner", "examples", "spec")]
        required += ["selfplay/program_python.py", "scripts/program_response_normalization.py"]
        for relative in required:
            require(relative in manifest["source_sha256"], "Missing frozen runtime source: " + relative)
            require(sha(run / "source" / relative) == manifest["source_sha256"][relative],
                    "Archived runtime source changed: " + relative)

        def module(name, relative, package=False):
            path = run / "source" / relative
            spec = importlib.util.spec_from_file_location(prefix + "." + name, path,
                submodule_search_locations=[str(path.parent)] if package else None)
            obj = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = obj
            spec.loader.exec_module(obj)
            return obj

        arena = module("program_arena", "selfplay/program_arena/__init__.py", True)
        frontend = module("program_python", "selfplay/program_python.py")
        normalizer = module("normalizer", "scripts/program_response_normalization.py")
        game = sys.modules[prefix + ".program_arena.game"]
        examples = module("program_arena.examples", "selfplay/program_arena/examples.py")
        yield SimpleNamespace(compile_python=frontend.compile_python_program,
            validation_error=frontend.PythonProgramValidationError,
            compile_ir=arena.compile_program, verify_replay=arena.verify_replay,
            normalize=normalizer.normalize_final, game=game.RobotArena, reference_programs=examples.reference_programs)
    finally:
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                del sys.modules[name]


def classify(generation, runtime):
    if generation["finish_reason"] != "eos":
        return {"valid": False, "reason": "generation_" + generation["finish_reason"]}, None
    if "<think>" in generation["text"] or "</think>" in generation["text"]:
        return {"valid": False, "reason": "unexpected_thinking"}, None
    source, packaging = runtime.normalize(generation["text"])
    try:
        program = runtime.compile_python(source)
    except runtime.validation_error as error:
        return {"valid": False, "reason": "invalid_bounded_python", "detail": str(error),
                "normalization": packaging}, None
    return {"valid": True, "normalization": packaging, "program": program.to_dict()}, program


def advantages(rewards):
    require(len(rewards) == 8 and all(type(x) in (float, int) and math.isfinite(x) and -1 <= x <= 1
                                    for x in rewards), "Eight finite bounded rewards required")
    total = math.fsum(rewards)
    return [x - (total - x) / 7 for x in rewards]


def score_from_matches(matches, valid, arm):
    """Candidate faults override its reward; rival faults never do."""
    require(arm in ARMS, "Unknown reward arm")
    require(bool(matches) == bool(valid), "Invalid authorship has no fabricated game; valid has games")
    if not valid:
        return {"authorship_valid": False, "runtime_valid": False,
                "candidate_program_faults": 0, "opponent_program_faults": 0,
                "game_return": None, "banked_point_yield": 0.0,
                "reward": -1.0, "reward_kind": "invalid_authorship_or_candidate_runtime"}
    candidate_faults = sum(m["replay"]["metrics"]["program_faults"][m["candidate_seat"]] for m in matches)
    opponent_faults = sum(m["replay"]["metrics"]["program_faults"][1-m["candidate_seat"]] for m in matches)
    game_return = math.fsum(m["replay"]["returns"][m["candidate_seat"]] for m in matches) / len(matches)
    banked = math.fsum(m["replay"]["metrics"]["scores"][m["candidate_seat"]] for m in matches) / len(matches)
    reward = -1.0 if candidate_faults else game_return if arm == "game_rl" else 0.0
    return {"authorship_valid": True, "runtime_valid": candidate_faults == 0,
            "candidate_program_faults": candidate_faults, "opponent_program_faults": opponent_faults,
            "game_return": game_return, "banked_point_yield": banked, "reward": reward,
            "reward_kind": "invalid_authorship_or_candidate_runtime" if candidate_faults else
                           "game_return" if arm == "game_rl" else "validity_only"}


def row_metrics(row):
    """One independent authoring draw; replay games are correlated measurements."""
    matches = row["matches"]
    result = {"generation": row["generation"], "banked_point_yield": row["banked_point_yield"],
              "authorship_valid": row["authorship_valid"], "runtime_valid": row["runtime_valid"],
              "game_return": row["game_return"], "reward": row["reward"], "games": len(matches),
              "candidate_program_faults": row["candidate_program_faults"],
              "opponent_program_faults": row["opponent_program_faults"],
              "wins": 0, "draws": 0, "losses": 0, "zero_zero_draws": 0,
              "turns": 0, "rest_actions": 0, "failed_actions": 0, "by_opponent": {}, "by_map": {}}
    for match in matches:
        r, seat = match["replay"], match["candidate_seat"]
        ret = r["returns"][seat]
        result["wins" if ret > 0 else "losses" if ret < 0 else "draws"] += 1
        result["zero_zero_draws"] += r["metrics"]["scores"] == [0, 0]
        result["failed_actions"] += r["metrics"]["failed_actions"][seat]
        for event in r["events"]:
            result["turns"] += 1
            result["rest_actions"] += event["decisions"][seat]["action"] == {"action": "rest"}
        for field, key in (("by_opponent", match["opponent_id"]), ("by_map", str(r["seed"]))):
            group = result[field].setdefault(key, {"games": 0, "banked_points": 0, "return_sum": 0.0})
            group["games"] += 1
            group["banked_points"] += r["metrics"]["scores"][seat]
            group["return_sum"] += ret
    result["all_rest"] = bool(matches) and result["rest_actions"] == result["turns"]
    result["rest_action_fraction"] = result["rest_actions"] / result["turns"] if result["turns"] else None
    return result


def summarize_programs(rows, expected):
    require(len(rows) <= expected, "More authoring draws than prescribed")
    require(len({r["generation"] for r in rows}) == len(rows), "Duplicate authoring draw")
    metrics = [row_metrics(row) for row in rows]
    complete = len(rows) == expected
    valid = [r for r in metrics if r["authorship_valid"]]
    fault_free = [r for r in metrics if r["runtime_valid"]]
    def mean(values):
        return math.fsum(values) / len(values) if values else None
    return {"prescribed_program_clusters": expected, "observed_program_clusters": len(rows),
            "missing_program_clusters": expected-len(rows), "complete": complete,
            "primary_mean_banked_point_yield": mean([r["banked_point_yield"] for r in metrics]) if complete else None,
            "observed_mean_banked_point_yield": mean([r["banked_point_yield"] for r in metrics]),
            "compiled_programs": len(valid), "invalid_no_game_programs": len(rows)-len(valid),
            "fault_free_programs": len(fault_free), "played_games": sum(r["games"] for r in metrics),
            "positive_banking_programs": sum(r["banked_point_yield"] > 0 for r in metrics),
            "zero_banking_programs": sum(r["banked_point_yield"] == 0 for r in metrics),
            "all_rest_programs": sum(r["all_rest"] for r in metrics),
            "mean_raw_game_return_among_compiled": mean([r["game_return"] for r in valid]),
            "conditional_fault_free_mean_banked": mean([r["banked_point_yield"] for r in fault_free]),
            "mean_penalized_task_reward": mean([r["reward"] for r in metrics]),
            "programs": metrics, "sampling_unit": "generated controller, not individual replay game"}


def validate_generation(record, *, name, seed, checkpoint, prompt_sha256, config):
    require(record["name"] == name and record["seed"] == seed, "Generation name/seed differs")
    same(record["checkpoint"], checkpoint, "Generation source checkpoint differs")
    require((record["model_repo"], record["model_revision"]) ==
            (config["model_repo"], config["model_revision"]), "Generation model identity differs")
    require(record["prompt_sha256"] == prompt_sha256, "Generation prompt differs")
    ids, logps = record["completion_ids"], record["sampled_token_logprobs"]
    require(len(ids) == len(logps) == record["output_tokens"] <= config["max_new_tokens"],
            "Completion token/logprob coverage differs")
    require(len(record["prompt_ids"]) == record["input_tokens"] > 0, "Prompt token coverage differs")
    require(record["input_tokens"] + len(ids) <= config["max_context_tokens"], "Context cap exceeded")
    require(all(type(x) is int and x >= 0 for x in ids + record["prompt_ids"]), "Invalid token id")
    require(all(type(x) in (float, int) and math.isfinite(x) and x <= 1e-6 for x in logps),
            "Invalid sampled token log probability")
    require(record["finish_reason"] in {"eos", "length", "time"}, "Unknown finish condition")
    eos = set(record["eos_token_ids"])
    if record["finish_reason"] == "eos":
        require(bool(ids) and ids[-1] in eos and not set(ids[:-1]) & eos,
                "Claimed EOS was not the actual first sampled EOS")
    else:
        require(not set(ids) & eos, "Sampled EOS mislabeled as truncation")
    if record["finish_reason"] == "length":
        require(len(ids) == config["max_new_tokens"], "Length stop before token cap")


def validate_gate(gate, generations, names, config):
    require(gate["passed"] is True and gate.get("generation_names", list(gate.get("generation_sha256", {}))) == names,
            "Gate generation identity differs")
    require(len(gate["records"]) == len(generations), "Gate sample coverage differs")
    for check, generation in zip(gate["records"], generations):
        require(check["passed"] is True and check["tokens"] == len(generation["completion_ids"]),
                "Gate token coverage or outcome differs")
        for field, cap in (("max_token_error", config["token_logprob_tolerance"]),
                           ("sequence_error", config["sequence_logprob_tolerance"])):
            value = check[field]
            require(type(value) in (float, int) and math.isfinite(value) and 0 <= value <= cap,
                    "Numerical likelihood gate exceeded: " + field)
        same(check["recorded_sequence_logprob"], math.fsum(generation["sampled_token_logprobs"]),
             "Gate does not cover saved sampling log probabilities")


def validate_seed_families(schedule):
    families = {"train": [x for values in schedule["train_maps"].values() for x in values],
                **{name: schedule[name + "_maps"] for name in ("calibration", "validation", "holdout")}}
    all_seeds = []
    for name, seeds in families.items():
        require(all(type(x) is int for x in seeds) and len(seeds) == len(set(seeds)),
                "Duplicate/non-integer map seed in " + name)
        all_seeds += seeds
    require(len(all_seeds) == len(set(all_seeds)), "Map seed families overlap")
    generation_seeds = [x for seeds in schedule["train_generation_seeds"].values() for x in seeds]
    generation_seeds += schedule["opponent_generation_seeds"]
    generation_seeds += [x for seeds in schedule["eval_generation_seeds"].values() for x in seeds]
    require(len(generation_seeds) == len(set(generation_seeds)), "Generation seed families overlap")
    return families


def check_fingerprint_overlap(families, prior):
    """Different integer seeds alone do not prove distinct public game layouts."""
    seen = set(prior)
    for family, values in families.items():
        require(len(values) == len(set(values)), "Duplicate map fingerprint within " + family)
        require(not seen.intersection(values), "Map layout fingerprint overlap in " + family)
        seen.update(values)


def validate_ir_isolation(training, heldout):
    train_model = [r["ir_sha256"] for r in training if r["origin"] == "model"]
    test_model = [r["ir_sha256"] for r in heldout if r["origin"] == "model"]
    require(len(train_model) == len(set(train_model)), "Duplicate training-model opponent IR")
    require(not set(train_model) & set(test_model), "Heldout model opponent IR leaked into training")


def validate_checkpoint_tensors(folder, metadata, parent_folder=None, expected_steps=None):
    """Safetensors arrays on CPU only; no torch/MLX model is instantiated."""
    import numpy as np
    from safetensors.numpy import load_file
    arrays = {}
    for name in ("adapter.safetensors", "optimizer.safetensors"):
        require(sha(folder / name) == metadata["files"][name], "Checkpoint tensor bytes changed: " + name)
        arrays[name] = load_file(folder / name)
        require(all(np.isfinite(v).all() for v in arrays[name].values()), "Nonfinite checkpoint tensors")
    adapter, optimizer = arrays["adapter.safetensors"], arrays["optimizer.safetensors"]
    require(adapter and all(v.dtype == np.float32 for v in adapter.values()), "Adapter is not FP32")
    require("step" in optimizer and optimizer["step"].size == 1, "Optimizer step counter missing")
    if expected_steps is not None:
        require(int(optimizer["step"].item()) == expected_steps, "Optimizer step lineage differs")
    result = {"optimizer_steps": int(optimizer["step"].item()),
              "trainable_parameters": sum(v.size for v in adapter.values()), "adapter_l2_delta": None}
    if parent_folder is not None:
        before = load_file(parent_folder / "adapter.safetensors")
        old_optimizer = load_file(parent_folder / "optimizer.safetensors")
        require(optimizer.keys() == old_optimizer.keys() and all(optimizer[k].shape == old_optimizer[k].shape
                and optimizer[k].dtype == old_optimizer[k].dtype for k in optimizer), "Optimizer state schema changed")
        require(adapter.keys() == before.keys() and all(adapter[k].shape == before[k].shape for k in adapter),
                "Adapter names/shapes differ from parent")
        delta2 = math.fsum(float(np.square(adapter[k].astype(np.float64)-before[k].astype(np.float64)).sum())
                          for k in adapter)
        result["adapter_l2_delta"] = math.sqrt(delta2)
        result["changed_parameters"] = sum(int(np.count_nonzero(adapter[k] != before[k])) for k in adapter)
        outcome = metadata["report"]["outcome"]
        if outcome == "skipped_zero_variance":
            require(delta2 == 0 and all(sha(folder / n) == sha(parent_folder / n) for n in metadata["files"]),
                    "Skipped group changed adapter or optimizer")
        else:
            require(outcome == "updated" and delta2 > 0, "Updated checkpoint has no weight change")
            same(result["adapter_l2_delta"], metadata["report"]["adapter_l2_delta"], "Recorded adapter delta differs")
    return result


def validate_execution_budget(config, status, runtime=None):
    budget = status.get("execution_budget")
    if budget is None:
        require(runtime is None and status["state"] == "prepared", "Neural execution missing durable time budget")
        return
    start = datetime.fromisoformat(budget["first_started"])
    deadline = datetime.fromisoformat(config["deadline"])
    stop = datetime.fromisoformat(budget["hard_stop"])
    require(start.tzinfo is not None and stop.tzinfo is not None and deadline.tzinfo is not None,
            "Execution timestamps need timezones")
    require(stop == min(start + timedelta(seconds=13500), deadline) and stop > start
            and budget["maximum_seconds"] == 13500 and budget["resume_counts_wall_time"] is True,
            "Whole-study budget was renewed or deadline changed")
    if runtime is not None:
        same(runtime["execution_budget"], budget, "Worker runtime and durable deadline differ")
    if status["state"] == "complete":
        require(datetime.fromisoformat(status["updated"]) <= stop, "Run claimed completion after hard deadline")


def validate_runtime(runtime, model_index, manifest):
    expected_index = {"index_sha256": manifest["model_index_sha256"], **model_index}
    same(runtime["model_index"], expected_index, "Executed model index differs from preparation")
    same(runtime["versions"], manifest["expected_installed_versions"], "Execution runtime versions differ")
    require(runtime["arithmetic"] == "MLX affine 4bit packed base unchanged; floating tensors/LoRA FP32",
            "Execution arithmetic condition differs")
    require(runtime["prefix_mode"] == manifest["config"]["prefix_mode"] and runtime["memory_limit_bytes"] ==
            manifest["config"]["memory_limit_bytes"] and runtime["external_spend_usd"] == 0,
            "Execution resource or prefix configuration differs")
    names = sorted(f"self_attn.{projection}.lora_{side}" for projection in ("q_proj", "v_proj") for side in ("a", "b"))
    require(runtime["trainable_tensor_names"] == names and runtime["trainable_parameters"] == 188416,
            "Trainable parameter identity differs")
    require(runtime["packed_integer_tensors"] and runtime["eos_token_ids"], "Missing packed-weight or EOS identity")


class Audit:
    def __init__(self, run):
        self.run = Path(run).resolve()
        self.manifest = read(self.run / "manifest.json")
        self.status = read(self.run / "status.json")
        self.config = self.manifest["config"]
        self.artifacts = self.status["artifacts"]
        self.schedule = read(self.run / "schedule.json")
        self.plan = read(self.run / "plan.json")
        self.checkpoints, self.generations, self.verified_replays = {}, {}, set()
        self.tensor_reports = {}
        self.training_rows, self.evaluations, self.group_reports = [], {}, []

    def artifact(self, relative, required=True):
        if relative not in self.artifacts:
            require(not required, "Artifact not committed: " + relative)
            return None
        path = safe_path(self.run, relative)
        require(sha(path) == self.artifacts[relative], "Committed artifact hash differs: " + relative)
        return read(path)

    def frozen(self):
        m = self.manifest
        require(m["format_version"] == 1 and m["kind"] == "matched_program_study", "Unknown study format")
        require(sha(self.run / "manifest.json") == self.status["manifest_sha256"], "Manifest hash differs")
        files = {"plan.json": m["plan_sha256"], "schedule.json": m["schedule_sha256"],
                 "map-fingerprints.json": m["map_fingerprints_sha256"], **m["input_sha256"], **self.artifacts}
        for relative, expected in files.items():
            require(sha(safe_path(self.run, relative)) == expected, "Artifact hash differs: " + relative)
        for relative, expected in m["source_sha256"].items():
            require(sha(safe_path(self.run / "source", relative)) == expected, "Archived source differs: " + relative)
        parent = read(self.run / "inputs/parent-manifest.json")
        parent_status = read(self.run / "inputs/parent-status.json")
        require(sha(self.run / "inputs/parent-manifest.json") == parent_status["manifest_sha256"],
                "Parent run manifest provenance differs")
        for relative, expected in parent["source_sha256"].items():
            require(m["source_sha256"].get(relative) == expected, "Previously frozen source was altered: " + relative)
        same(m["library_source_sha256"], parent["library_source_sha256"], "Library source identities changed")
        same(m["expected_installed_versions"], parent["expected_installed_versions"], "Runtime versions changed")
        controlled = self.plan["controlled_comparison"]
        require(sha(self.run / "inputs/base/metadata.json") == controlled["initial_checkpoint"]["sha256"],
                "Common initial checkpoint metadata changed")
        for name, key in (("adapter", "initial_adapter_sha256"), ("optimizer", "initial_optimizer_sha256")):
            require(sha(self.run / f"inputs/base/{name}.safetensors") == controlled[key], "Common initial state differs")
        for condition in ("compact", "plain"):
            require(sha(self.run / f"prompts/{condition}.txt") == m["prompt_sha256"][condition], "Prompt hash differs")
        require(m["prompt_sha256"]["compact"] == controlled["training_prompt"]["sha256"], "Training prompt changed")
        require(m["prompt_sha256"]["plain"] == self.plan["evaluation"]["plain_transfer_prompt"]["sha256"],
                "Plain evaluation prompt changed")
        require(sha(self.run / "inputs/model-index.json") == m["model_index_sha256"], "Model index changed")
        self.model_index = read(self.run / "inputs/model-index.json")
        require((self.model_index["repo"], self.model_index["revision"]) ==
                (self.config["model_repo"], self.config["model_revision"]), "Pinned model index identity differs")
        families = validate_seed_families(self.schedule)
        seed_plan = self.plan["seed_schedule"]
        same(self.schedule["train_maps"], seed_plan["fresh_training_maps"], "Training seed plan changed")
        for family in ("calibration", "validation", "holdout"):
            key = "primary_holdout_maps" if family == "holdout" else family + "_maps"
            same(families[family], seed_plan[key], "Map seed plan changed: " + family)
        require(self.schedule["round_order"] == [[x.split("/")[0], int(x.split("/")[1])]
                for x in controlled["round_order"]], "Arm order differs from plan")
        for number in range(1, 4):
            same(self.schedule["train_generation_seeds"][str(number)],
                 [8100000 + number*100 + i for i in range(8)], "Training RNG schedule changed")
        same(self.schedule["opponent_generation_seeds"], seed_plan["opponent_generation_seeds"], "Opponent RNG changed")
        for condition, key in (("compact", "primary_evaluation_generation_seeds"),
                               ("plain", "plain_probe_generation_seeds")):
            same(self.schedule["eval_generation_seeds"][condition], seed_plan[key], "Evaluation RNG changed")
        require((self.config["group_size"], self.config["iterations"], self.config["max_rounds"],
                 self.config["max_new_tokens"], self.config["thinking"], self.config["worker_seconds"]) ==
                (8, 3, 48, 1024, False, 13500), "Registered study bounds changed")
        require((self.config["temperature"], self.config["top_p"], self.config["top_k"],
                 self.config["learning_rate"], self.config["lora_rank"], self.config["lora_dropout"],
                 self.config["gradient_clip"], self.config["compute_dtype"], self.config["sample_seconds"]) ==
                (1.0, 1.0, 0, 1e-6, 8, 0.0, 0.5, "float32", 300), "Registered training/sampling condition changed")
        require(self.config["deadline"] == self.plan["resource_and_time_plan"]["absolute_process_stop"],
                "Hard deadline differs from plan")
        return families

    def maps(self, runtime, families):
        document = read(self.run / "map-fingerprints.json")
        def layout(seed):
            game = runtime.game(seed, 48)
            return {"grid_size": game.size, "capacity": game.capacity, "max_energy": game.max_energy,
                    "max_rounds": game.max_rounds, "sites": game.sites,
                    "homes": [[r["home_x"], r["home_y"]] for r in game.robots]}
        fingerprints = {}
        for family, seeds in families.items():
            expected = [{"seed": seed, "layout": layout(seed), "sha256": value_sha(layout(seed))} for seed in seeds]
            same(document["families"][family], expected, "Public map fingerprint differs")
            fingerprints[family] = [x["sha256"] for x in expected]
        prior = document["prior_seed_fingerprints"]
        for seed, fingerprint in prior.items():
            require(value_sha(layout(int(seed))) == fingerprint, "Prior map fingerprint differs")
        evidence_seeds = set()
        def collect(value):
            if isinstance(value, dict):
                if value.get("kind") == "program_match" and "initial_state" in value:
                    evidence_seeds.add(value["seed"])
                else:
                    for child in value.values(): collect(child)
            elif isinstance(value, list):
                for child in value: collect(child)
        for relative, expected in document["prior_evidence_sha256"].items():
            path = safe_path(ROOT, relative)
            require(sha(path) == expected, "Historical map evidence changed: " + relative)
            if path.suffix == ".jsonl":
                for line in path.read_text().splitlines():
                    if line.strip(): collect(json.loads(line))
            else:
                collect(read(path))
        require(evidence_seeds == {int(x) for x in prior}, "Historical seed provenance incomplete")
        require(not set(x for seeds in families.values() for x in seeds) & evidence_seeds, "Prior map seed reused")
        check_fingerprint_overlap(fingerprints, list(prior.values()))

    def checkpoint(self, label):
        if label in self.checkpoints:
            return self.checkpoints[label]
        metadata = self.artifact(f"checkpoints/{label}/metadata.json", False)
        if metadata is None:
            return None
        require(metadata["label"] == label and metadata["complete"] is True, "Checkpoint identity incomplete")
        require(metadata["manifest_sha256"] == self.status["manifest_sha256"], "Checkpoint manifest differs")
        require((metadata["model_repo"], metadata["model_revision"]) ==
                (self.config["model_repo"], self.config["model_revision"]), "Checkpoint model differs")
        require(set(metadata["files"]) == {"adapter.safetensors", "optimizer.safetensors"}, "Checkpoint files differ")
        for filename, expected in metadata["files"].items():
            require(self.artifacts.get(f"checkpoints/{label}/{filename}") == expected, "Checkpoint file not committed")
        if label == "base":
            require(metadata["parent"] is None and metadata["report"] is None, "Initial checkpoint has an update")
            same(metadata["files"], read(self.run / "inputs/base/metadata.json")["files"], "Base reset differs")
            tensors = validate_checkpoint_tensors(self.run / "checkpoints/base", metadata, expected_steps=0)
        else:
            arm, round_text = label.rsplit("-round", 1)
            require(arm in ARMS and round_text in {"1", "2", "3"}, "Unregistered checkpoint label")
            number = int(round_text)
            source = "base" if number == 1 else f"{arm}-round{number-1}"
            parent = self.checkpoint(source)
            require(parent is not None and metadata["parent"] == source, "Arm checkpoint lineage differs")
            report = metadata["report"]
            same(self.artifact(f"groups/{arm}/{number}/update.json"), report, "Checkpoint/update transaction differs")
            require(report["source_checkpoint"] == source and report["source_metadata_sha256"] ==
                    self.artifacts[f"checkpoints/{source}/metadata.json"], "Update source checkpoint hash differs")
            require(type(report["optimizer_steps"]) is int and report["optimizer_steps"] in (0, 1), "Invalid optimizer count")
            parent_steps = self.tensor_reports[source]["optimizer_steps"]
            tensors = validate_checkpoint_tensors(self.run / "checkpoints" / label, metadata,
                       self.run / "checkpoints" / source, parent_steps + report["optimizer_steps"])
        self.checkpoints[label] = metadata
        self.tensor_reports[label] = tensors
        return metadata

    def generation(self, name, seed, label, condition):
        record = self.artifact("generations/" + name + ".json")
        require(self.checkpoint(label) is not None, "Generation references missing checkpoint")
        validate_generation(record, name=name, seed=seed, checkpoint=label,
            prompt_sha256=self.manifest["prompt_sha256"][condition], config=self.config)
        require(record["prompt_condition"] == condition and record["source_metadata_sha256"] ==
                self.artifacts[f"checkpoints/{label}/metadata.json"], "Generation checkpoint/prompt provenance differs")
        require(record["authorship"] == "fresh_learner_sample" and record["api_training_data"] is False,
                "Fresh learner provenance differs")
        runtime = self.artifact("runtime.json")
        same(record["eos_token_ids"], runtime["eos_token_ids"], "Generation tokenizer EOS identity differs")
        self.generations[name] = record
        return record

    def scored_row(self, row, generation, opponents, seeds, arm, runtime):
        classification, program = classify(generation, runtime)
        same(row["classification"], classification, "Compilation provenance differs")
        expected = [(opponent, seed, seat) for opponent in opponents for seed in seeds for seat in (0, 1)] if program else []
        require(len(row["matches"]) == len(expected), "Prescribed played-match denominator differs")
        for match, (opponent, seed, seat) in zip(row["matches"], expected):
            record = match["replay"]
            require(match["opponent_id"] == opponent["id"] and type(match["candidate_seat"]) is int
                    and match["candidate_seat"] == seat and record["seed"] == seed and record["max_rounds"] == 48,
                    "Match seat/map/opponent schedule differs")
            rival = runtime.compile_ir(opponent["compiled_ir"])
            require(record["programs"][seat]["digest"] == program.compiled.digest and
                    record["programs"][1-seat]["digest"] == rival.digest, "Replay program identity differs")
            require(record["origins"][seat] == "model" and record["origins"][1-seat] == opponent["origin"],
                    "Replay model/scripted origins differ")
            fingerprint = value_sha(record)
            if fingerprint not in self.verified_replays:
                runtime.verify_replay(record)
                self.verified_replays.add(fingerprint)
        expected_score = score_from_matches(row["matches"], program is not None, arm)
        for key, value in expected_score.items():
            same(row[key], value, "Recomputed reward/metric differs: " + key)
        return expected_score

    def opponent_pool(self, runtime):
        pool = self.artifact("opponents.json", False)
        if pool is None:
            return None
        require(pool["source_checkpoint"] == "base" and pool["frozen_for_both_arms"] is True,
                "Opponent pool is not shared/frozen")
        same(pool["calibration_seeds"], self.schedule["calibration_maps"], "Opponent calibration maps differ")
        evaluation = read(self.run / "evaluation-opponents.json")
        selected = pool["selected"]
        require([r["id"] for r in selected[:2]] == ["nearest", "value"], "Reference anchors differ")
        for entry in selected + evaluation:
            if entry["origin"] == "model":
                require(runtime.compile_ir(entry["compiled_ir"]).digest == entry["ir_sha256"], "Opponent IR hash differs")
        validate_ir_isolation(selected, evaluation)
        heldout_ir = {r["ir_sha256"] for r in evaluation if r["origin"] == "model"}
        admitted, fingerprints, fresh_count = [], set(), 0
        expected_ids = ["historical/opponent-1-1", "historical/candidate-1-1"] + [f"pool-{i}" for i in range(8)]
        require(len(pool["attempts"]) <= 10, "Opponent attempt budget exceeded")
        for index, attempt in enumerate(pool["attempts"]):
            require(attempt["id"] == expected_ids[index] and fresh_count < 2, "Opponent attempt selection changed")
            path = safe_path(self.run, attempt["source_path"])
            require(sha(path) == attempt["source_sha256"], "Opponent generation source changed")
            if attempt["fresh"]:
                j = index-2
                generation = self.generation(f"pool-{j}", self.schedule["opponent_generation_seeds"][j], "base", "compact")
            else:
                generation = read(path)
                relative = "generations/" + path.name
                require(read(self.run / "inputs/parent-status.json")["artifacts"][relative] == sha(path),
                        "Historical model opponent unbound to parent generation")
            row = attempt["behavior"]
            recomputed = self.scored_row(row, generation, selected[:1], self.schedule["calibration_maps"], "game_rl", runtime)
            fingerprint = row["classification"].get("program", {}).get("ir_sha256")
            require(attempt["ir_sha256"] == fingerprint, "Calibration IR fingerprint differs")
            useful = recomputed["runtime_valid"] and sum(m["replay"]["metrics"]["scores"][m["candidate_seat"]] > 0
                         for m in row["matches"]) >= 3 and fingerprint not in fingerprints and fingerprint not in heldout_ir
            require(attempt["duplicates_heldout_model"] == (fingerprint in heldout_ir), "Opponent IR overlap label differs")
            require(attempt["useful"] == useful, "Opponent admission differs from fixed capability rule")
            if useful:
                admitted.append(attempt["id"])
                fingerprints.add(fingerprint)
                fresh_count += attempt["fresh"]
                entry = selected[len(admitted)+1]
                require(entry["id"] == attempt["id"] and entry["source_path"] == attempt["source_path"]
                        and entry["generation_sha256"] == attempt["source_sha256"] and entry["ir_sha256"] == fingerprint,
                        "Admitted opponent provenance differs")
        require([r["id"] for r in selected[2:]] == admitted and pool["passed"] == (len(admitted) >= 2),
                "Frozen roster completeness differs")
        require(fresh_count == 2 or len(pool["attempts"]) == 10, "Opponent search ended before fixed admission rule")
        return pool

    def groups(self, runtime, pool):
        seen_missing, invalid_streak = False, dict.fromkeys(ARMS, 0)
        for arm, number in self.schedule["round_order"]:
            label = f"{arm}-round{number}"
            folder = f"groups/{arm}/{number}"
            scores = self.artifact(folder + "/scores.json", False)
            metadata = self.checkpoint(label)
            if scores is None:
                require(metadata is None, "Checkpoint committed without all eight scores")
                seen_missing = True
                continue
            require(not seen_missing and pool is not None and pool["passed"], "Training skipped prescribed group/pool")
            require(invalid_streak[arm] < 2, "Training continued after capability stop")
            source = "base" if number == 1 else f"{arm}-round{number-1}"
            require(scores["arm"] == arm and scores["round"] == number and scores["source_checkpoint"] == source,
                    "Group source/identity differs")
            require(scores["source_metadata_sha256"] == self.artifacts[f"checkpoints/{source}/metadata.json"] and
                    scores["opponents_sha256"] == self.artifacts["opponents.json"], "Group source hashes differ")
            same(scores["seeds"], self.schedule["train_maps"][str(number)], "Training maps differ")
            require(len(scores["rows"]) == 8, "Training group must retain all eight outputs")
            records, hashes = [], {}
            for i, row in enumerate(scores["rows"]):
                name = f"{arm}-round{number}-{i}"
                record = self.generation(name, self.schedule["train_generation_seeds"][str(number)][i], source, "compact")
                require(row["generation"] == name and row["generation_sha256"] == self.artifacts[f"generations/{name}.json"],
                        "Scored generation provenance differs")
                self.scored_row(row, record, pool["selected"], scores["seeds"], arm, runtime)
                records.append(record)
                hashes[name] = row["generation_sha256"]
            rewards = [r["reward"] for r in scores["rows"]]
            adv = advantages(rewards)
            zero = max(rewards)-min(rewards) <= 1e-12
            if zero: adv = [0.0]*8
            same(scores["rewards"], rewards, "Group rewards differ")
            same(scores["advantages"], adv, "Leave-one-out advantages differ")
            require(scores["zero_signal"] == zero, "Zero-variance classification differs")
            valid_returns = [r["game_return"] for r in scores["rows"] if r["runtime_valid"]]
            signal = "no_update" if zero else "validity_only_signal" if arm == "validity_rl" or len(set(valid_returns)) <= 1 else "gameplay_and_validity_signal"
            group_summary = {"arm": arm, "round": number, "rewards": rewards, "advantages": adv,
                             "gradient_signal": signal, "checkpoint_committed": metadata is not None,
                             "collected_completion_tokens": sum(len(r["completion_ids"]) for r in records),
                             "committed_gradient_tokens": sum(len(r["completion_ids"]) for r in records)
                                                           if metadata is not None and not zero else 0,
                             **summarize_programs(scores["rows"], 8)}
            self.group_reports.append(group_summary)
            self.training_rows += scores["rows"]
            invalid = not any(r["runtime_valid"] for r in scores["rows"])
            invalid_streak[arm] = invalid_streak[arm]+1 if invalid else 0
            if metadata is None:
                seen_missing = True
                continue
            report = metadata["report"]
            for key, value in {"arm": arm, "round": number, "rewards": rewards, "advantages": adv,
                "generation_sha256": hashes, "training_tokens": sum(len(r["completion_ids"]) for r in records),
                "scores_sha256": self.artifacts[folder+"/scores.json"], "all_runtime_invalid": invalid}.items():
                same(report[key], value, "Update report differs: " + key)
            require(report["outcome"] == ("skipped_zero_variance" if zero else "updated") and
                    report["optimizer_steps"] == (0 if zero else 1), "Zero-signal update/skip differs")
            if zero:
                require(report["likelihood_gate_path"] is None and report["likelihood_gate_sha256"] is None,
                        "Skip claims neural update gate")
            else:
                path = report["likelihood_gate_path"]
                gate = self.artifact(path)
                require(report["likelihood_gate_sha256"] == self.artifacts[path], "Update likelihood gate hash differs")
                same(gate["generation_sha256"], hashes, "Likelihood gate generation hashes differ")
                require(gate["source_checkpoint"] == source and gate["source_metadata_sha256"] ==
                        report["source_metadata_sha256"], "Gate source weights differ")
                validate_gate(gate, records, list(hashes), self.config)
                validate_gate(report["gradient_likelihood_gate"], records, list(hashes), self.config)
                require(math.isfinite(report["loss"]) and report["gradient_norm_before_clip"] > 0,
                        "Invalid numerical update summary")

    def evaluation_opponents(self, runtime):
        opponents = read(self.run / "evaluation-opponents.json")
        require([r["id"] for r in opponents] == ["nearest", "value", "controller-python-1", "qwen38-27b-1-r2"],
                "Evaluation opponent schedule differs")
        original = read(self.run / "inputs/tournament-plan.json")
        planned = {r["id"]: r for r in self.plan["evaluation"]["frozen_opponents"]}
        for entry in opponents:
            if entry["origin"] == "scripted":
                same(entry["compiled_ir"], runtime.reference_programs()[entry["id"]], "Scripted anchor changed")
            else:
                prior = original["roster"][entry["id"]]
                same(entry["compiled_ir"], prior["python_program"]["compiled_ir"], "Heldout controller IR changed")
                require(runtime.compile_ir(entry["compiled_ir"]).digest == entry["ir_sha256"] == prior["ir_sha256"],
                        "Heldout controller fingerprint changed")
                path = self.run / "inputs/eval" / (entry["id"] + ".json")
                digest = sha(path)
                require(digest == entry["generation_sha256"] == planned[entry["id"]]["source"]["sha256"]
                        == original["input_sha256"][f"inputs/{entry['id']}/generation.json"],
                        "Heldout controller original model output changed")
        return opponents

    def all_generations(self):
        """Check even incomplete groups/pools, without turning them into scored rows."""
        expected = {}
        for i, seed in enumerate(self.schedule["opponent_generation_seeds"]):
            expected[f"pool-{i}"] = (seed, "base", "compact")
        for arm, number in self.schedule["round_order"]:
            source = "base" if number == 1 else f"{arm}-round{number-1}"
            for i, seed in enumerate(self.schedule["train_generation_seeds"][str(number)]):
                expected[f"{arm}-round{number}-{i}"] = (seed, source, "compact")
        for endpoint in ENDPOINTS:
            for condition, seeds in self.schedule["eval_generation_seeds"].items():
                for i, seed in enumerate(seeds):
                    expected[f"eval-{endpoint}-{condition}-{i}"] = (seed, endpoint, condition)
        for relative in self.artifacts:
            if relative.startswith("generations/") and relative.endswith(".json"):
                name = Path(relative).stem
                require(name in expected, "Unexpected extra authoring draw: " + name)
                seed, label, condition = expected[name]
                self.generation(name, seed, label, condition)

    def evaluation(self, runtime, opponents):
        seal = self.artifact("evaluation/sealed.json", False)
        paths = [f"generations/eval-{endpoint}-{condition}-{i}.json" for endpoint in ENDPOINTS
                 for condition, seeds in self.schedule["eval_generation_seeds"].items() for i in range(len(seeds))]
        if seal is not None:
            require(all(self.checkpoint(e) is not None for e in ENDPOINTS), "Sealed before both final endpoints")
            expected = {"checkpoints": {e: self.artifacts[f"checkpoints/{e}/metadata.json"] for e in ENDPOINTS},
                        "generations": {p: self.artifacts.get(p) for p in paths}, "draws": 36, "scoring_allowed": True}
            require(all(expected["generations"].values()), "Evaluation sealed with missing prescribed draws")
            same(seal, expected, "Evaluation seal changed or incomplete")
        for endpoint in ENDPOINTS:
            for condition, seeds in self.schedule["eval_generation_seeds"].items():
                for split in ("validation", "holdout"):
                    relative = f"evaluation/{endpoint}/{condition}/{split}.json"
                    saved = self.artifact(relative, False)
                    if saved is None:
                        continue
                    require(seal is not None, "Evaluation scored before all 36 authoring outputs were sealed")
                    require(saved["checkpoint"] == endpoint and saved["prompt_condition"] == condition and
                            saved["split"] == split and saved["training_use"] is False,
                            "Evaluation condition or training boundary differs")
                    same(saved["seeds"], self.schedule[split+"_maps"], "Evaluation map schedule differs")
                    same(saved["opponents"], opponents, "Evaluation opponent pool changed")
                    require(saved["opponents_sha256"] == self.manifest["input_sha256"]["evaluation-opponents.json"] and
                            saved["seal_sha256"] == self.artifacts["evaluation/sealed.json"], "Evaluation provenance changed")
                    require(saved["prescribed_program_draws"] == len(seeds) == len(saved["scores"]),
                            "Evaluation lost invalid/no-game authoring draws")
                    for i, row in enumerate(saved["scores"]):
                        name = f"eval-{endpoint}-{condition}-{i}"
                        require(row["generation"] == name and row["generation_sha256"] ==
                                self.artifacts["generations/"+name+".json"], "Evaluated output identity changed")
                        self.scored_row(row, self.generations[name], opponents, saved["seeds"], "game_rl", runtime)
                    result = summarize_programs(saved["scores"], len(seeds))
                    fingerprints = [r["classification"].get("program", {}).get("ir_sha256") for r in saved["scores"]]
                    result["duplicate_compiled_outputs"] = len([x for x in fingerprints if x])-len({x for x in fingerprints if x})
                    completions = [value_sha(self.generations[r["generation"]]["completion_ids"]) for r in saved["scores"]]
                    result["duplicate_token_completions"] = len(completions)-len(set(completions))
                    result["actual_eos_programs"] = sum(self.generations[r["generation"]]["finish_reason"] == "eos"
                                                         for r in saved["scores"])
                    self.evaluations[f"{endpoint}/{condition}/{split}"] = result

    def finish(self):
        complete = (all(e in self.checkpoints for e in ENDPOINTS) and len(self.evaluations) == 12
                    and all(x["complete"] for x in self.evaluations.values()))
        require(self.status["claims_allowed"] is False or complete, "Incomplete study improperly allows final claims")
        require(self.status["state"] != "complete" or complete, "Run claims completion before all prescribed results")
        contrasts = {}
        if complete:
            for condition, n in (("compact", 8), ("plain", 4)):
                for control in ("base", "validity_rl-round3"):
                    candidate = self.evaluations[f"game_rl-round3/{condition}/holdout"]["programs"]
                    reference = self.evaluations[f"{control}/{condition}/holdout"]["programs"]
                    differences = [a["banked_point_yield"]-b["banked_point_yield"] for a,b in zip(candidate, reference)]
                    contrasts[f"game_rl-minus-{control}/{condition}"] = {
                        "program_cluster_n": n, "paired_generation_seeds": self.schedule["eval_generation_seeds"][condition],
                        "raw_paired_banked_yield_differences": differences, "mean_difference": math.fsum(differences)/n,
                        "training_seeds_per_arm": 1}
        orphaned = []
        for path in self.run.rglob("*"):
            relative = str(path.relative_to(self.run))
            if path.is_file() and (".uncommitted-" in path.name or ".incomplete-" in relative or ".pending-" in relative):
                orphaned.append(relative)
        counts = {condition: sum(v["played_games"] for k,v in self.evaluations.items()
                                if f"/{condition}/holdout" in k) for condition in ("compact", "plain")}
        return {"format_version": 1, "kind": "independent_cpu_matched_program_study_audit",
                "run": str(self.run), "manifest_sha256": self.status["manifest_sha256"],
                "status_sha256": sha(self.run / "status.json"), "integrity_passed": True,
                "planned_study_complete": complete, "planned_comparisons_available": complete,
                "recorded_state": self.status["state"], "committed_fresh_authoring_draws": len(self.generations),
                "evaluation_draws": sum(name.startswith("eval-") for name in self.generations),
                "required_evaluation_draws": 36, "verified_distinct_replays": len(self.verified_replays),
                "actual_primary_compact_holdout_games": counts["compact"],
                "actual_secondary_plain_holdout_games": counts["plain"],
                "checkpoint_tensors": self.tensor_reports, "training_groups": self.group_reports,
                "evaluation": self.evaluations, "contrasts": contrasts, "retained_uncommitted_artifacts": orphaned,
                "limits": ["CPU audit verifies recorded numerical gate summaries; it does not independently rerun neural likelihoods.",
                           "Adapter, optimizer, all committed artifacts and archived sources are hash checked; model identity is bound to the frozen model-file index.",
                           "Eight compact and four plain generated controllers are the sampling units, not individual games. Only one training seed per arm.",
                           "Missing authoring draws are not zero-score games. Invalid completed authoring draws have zero banked yield and no fabricated replay.",
                           "The frozen opponent pool is not refreshed. Even a complete positive comparison does not establish continual self-play progress."]}

    def execute(self):
        families = self.frozen()
        runtime_record = self.artifact("runtime.json", False)
        validate_execution_budget(self.config, self.status, runtime_record)
        if runtime_record is not None:
            validate_runtime(runtime_record, self.model_index, self.manifest)
            parent_path = self.run / "inputs/parent-runtime.json"
            require(parent_path.exists() and "inputs/parent-runtime.json" in self.manifest["input_sha256"],
                    "Parent packed-weight execution provenance missing")
            require(sha(parent_path) == read(self.run/"inputs/parent-status.json")["artifacts"]["runtime.json"],
                    "Parent execution provenance changed")
            parent_runtime = read(parent_path)
            for key in ("packed_integer_tensors", "eos_token_ids", "trainable_tensor_names", "trainable_parameters", "arithmetic"):
                same(runtime_record[key], parent_runtime[key], "Parent/new execution identity differs: " + key)
        with archived_runtime(self.run, self.manifest) as runtime:
            self.maps(runtime, families)
            self.checkpoint("base")
            opponents = self.evaluation_opponents(runtime)
            pool = self.opponent_pool(runtime)
            self.groups(runtime, pool)
            self.all_generations()
            self.evaluation(runtime, opponents)
        return self.finish()


def audit(run):
    return Audit(run).execute()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional new immutable CPU audit artifact")
    args = parser.parse_args()
    result = audit(args.run)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        if args.output.exists():
            require(args.output.read_text() == text, "Refusing to replace an earlier audit artifact")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as stream:
                stream.write(text)
    print(text)


if __name__ == "__main__":
    main()
