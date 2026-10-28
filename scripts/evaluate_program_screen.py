"""Evaluate frozen model-authored Python controllers on CPU, without API calls.

Only a complete final response is compiled. Model Python is parsed by the
archived bounded frontend, never executed by Python. Run with --run RUN_DIR.
The flat source archive is verified and imported in an isolated namespace;
changing the live implementation cannot silently change this evaluation.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
import uuid


ROOT = Path(__file__).resolve().parents[1]
FORMAT_VERSION = 1
SOURCE_FILES = (
    "selfplay/program_arena/__init__.py", "selfplay/program_arena/demo.py",
    "selfplay/program_arena/dsl.py", "selfplay/program_arena/examples.py",
    "selfplay/program_arena/game.py", "selfplay/program_arena/runner.py",
    "selfplay/program_arena/spec.py", "selfplay/program_python.py", "selfplay/program_api.py",
)


def generation_client(plan):
    backend = plan.get("generation_backend", "openrouter")
    if backend == "openrouter":
        return "selfplay/program_api.py"
    if backend == "local_mlx":
        return "selfplay/program_local.py"
    if backend == "openrouter_model_screen":
        return "selfplay/program_model_screen.py"
    raise ValueError("unsupported generation backend")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def read_json(path):
    return json.loads(path.read_bytes())


def write_once(path, value):
    """Publish atomically without replacing an existing (possibly failed) file."""
    payload = encoded(value)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"incompatible existing artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Exclusive atomic publication; no replacement.
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def archived_runtime(run, plan):
    hashes = plan.get("source_sha256", {})
    expected_sources = set(SOURCE_FILES) - {"selfplay/program_api.py"}
    expected_sources.add(generation_client(plan))
    if set(hashes) != expected_sources:
        raise ValueError("plan must freeze exactly the nine program sources")
    sources = {}
    for original, expected in hashes.items():
        path = run / "source" / Path(original).name
        content = path.read_bytes()
        if digest(content) != expected:
            raise ValueError(f"frozen source hash mismatch: {path}")
        sources[original] = content

    if plan.get("generation_backend") == "openrouter_model_screen":
        config = read_json(run / "source/model-screen-profiles.json")
        fingerprint = digest(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        if fingerprint != plan.get("generation_client_configuration_sha256"):
            raise ValueError("frozen model-screen profile configuration mismatch")
        profile = plan["generation_client_profile"]
        if config["profiles"].get(profile["name"]) != {k: v for k, v in profile.items() if k != "name"}:
            raise ValueError("model-screen active profile differs from frozen configuration")

    normalization = plan.get("response_normalization")
    if normalization is not None:
        if (normalization.get("source_archive") != "source/response_normalization.py"
                or normalization.get("mode") != "single_python_fence_with_optional_prose"
                or normalization.get("version") != 2):
            raise ValueError("unsupported archived response normalization policy")
        content = (run / normalization["source_archive"]).read_bytes()
        if digest(content) != normalization.get("source_sha256"):
            raise ValueError("frozen response normalization hash mismatch")
        sources["response_normalization.py"] = content

    # Preload only the verified source bytes, bypassing both live modules and
    # Python's timestamp-based .pyc cache. The API client is hashed, never run.
    namespace = "_program_screen_" + uuid.uuid4().hex
    parent = ModuleType(namespace)
    parent.__path__ = []
    sys.modules[namespace] = parent
    package_name = namespace + ".program_arena"
    package = ModuleType(package_name)
    package.__path__ = []
    package.__package__ = package_name
    sys.modules[package_name] = package
    parent.program_arena = package

    def load(suffix, original, module=None):
        name = namespace + "." + suffix
        module = module or ModuleType(name)
        module.__file__ = str(run / "source" / Path(original).name)
        module.__package__ = package_name if suffix.startswith("program_arena.") else namespace
        if suffix == "program_arena":
            module.__package__ = package_name
        sys.modules[name] = module
        exec(compile(sources[original], module.__file__, "exec"), module.__dict__)
        return module

    try:
        for name in ("dsl", "game", "spec", "runner"):
            load("program_arena." + name, "selfplay/program_arena/" + name + ".py")
        load("program_arena", "selfplay/program_arena/__init__.py", package)
        examples = load("program_arena.examples", "selfplay/program_arena/examples.py")
        frontend = load("program_python", "selfplay/program_python.py")
        normalizer = None
        if normalization is not None:
            normalizer = load("response_normalization", "response_normalization.py")
            if (normalizer.NORMALIZATION_VERSION != normalization["version"]
                    or normalizer.NORMALIZATION_MODE != normalization["mode"]):
                raise ValueError("archived normalization version/mode differs from plan")
        yield SimpleNamespace(
            compile_python_program=frontend.compile_python_program,
            validation_error=frontend.PythonProgramValidationError,
            compile_program=package.compile_program,
            model_spec=package.model_spec,
            python_model_spec=frontend.python_model_spec,
            reference_programs=examples.reference_programs,
            play=package.play,
            verify_replay=package.verify_replay,
            frontend_version=frontend.FRONTEND_VERSION,
            python_ast_node_limit=frontend.MAX_PYTHON_AST_NODES,
            normalize_final=normalizer.normalize_final if normalizer else normalize_final,
            response_normalization=normalization,
        )
    finally:
        for name in list(sys.modules):
            if name == namespace or name.startswith(namespace + "."):
                del sys.modules[name]


def normalize_final(text):
    """Allow one fully wrapping Python code fence, and no other code repair."""
    # Whitespace outside the fence is formatting; whitespace inside is retained.
    match = re.fullmatch(r"[ \t\r\n]*```python[ \t]*\r?\n(.*?)\r?\n```[ \t\r\n]*", text, re.S)
    if match and "```" not in match[1]:
        return match[1] + "\n", "stripped_single_wrapping_python_fence"
    return text, "none"


def classify_generation(generation, runtime):
    text = generation.get("text")
    final_missing = not isinstance(text, str) or not text.strip()
    entry = {
        "sample_id": generation["sample_id"], "valid": False,
        "finish_reason": generation.get("finish_reason"), "missing_final": final_missing,
        "original_text": text, "normalization": "none", "failure": None,
        "compiled_program": None,
    }
    if generation.get("finish_reason") in {"length", "max_tokens", "max_output_tokens"}:
        entry["failure"] = "truncated"
        return entry, None
    if final_missing:
        entry["failure"] = "missingfinal"
        return entry, None
    source, normalization = runtime.normalize_final(text)
    entry.update(normalization=normalization, normalized_source=source,
                 normalized_source_sha256=digest(source.encode()), source_bytes=len(source.encode()))
    if runtime.response_normalization is not None:
        entry.update(original_text_sha256=digest(text.encode()),
                     normalization_version=runtime.response_normalization["version"],
                     normalization_source_sha256=runtime.response_normalization["source_sha256"])
    try:
        entry["python_ast_nodes"] = len(list(ast.walk(ast.parse(source))))
    except (SyntaxError, ValueError, RecursionError):
        entry["python_ast_nodes"] = None
    try:
        program = runtime.compile_python_program(source)
    except runtime.validation_error as error:
        entry.update(failure="invalidsyntax", error=str(error))
        # Keep the requested broad classification while exposing whether the
        # rejection was a bounded-language limit instead of Python syntax.
        entry["rejection_kind"] = "bound_limit" if any(
            phrase in str(error) for phrase in ("exceeds", "at most", "too many", "limit")) else "language_contract"
        return entry, None
    entry.update(valid=True, compiled_program=program.to_dict())
    return entry, program


def validate_plan(plan):
    if type(plan.get("samples")) is not int or not 1 <= plan["samples"] <= 100:
        raise ValueError("plan samples must be an integer 1..100")
    seeds = plan.get("validation_seeds")
    if not isinstance(seeds, list) or len(seeds) != 6 or any(type(s) is not int for s in seeds) or len(set(seeds)) != 6:
        raise ValueError("exactly six distinct integer validation seeds required")
    if plan.get("fixed_opponents") != ["nearest", "value"] or plan.get("paired_seats") is not True or plan.get("peer_crossplay") is not True:
        raise ValueError("screen requires nearest/value, paired seats, and peer crossplay")
    if type(plan.get("max_rounds")) is not int or not 1 <= plan["max_rounds"] <= 256:
        raise ValueError("plan max_rounds must be 1..256")
    if plan.get("training") is not False:
        raise ValueError("this CPU screen cannot perform training")


def load_generations(run, plan):
    generations, hashes = [], {}
    for number in range(1, plan["samples"] + 1):
        filename = f"generation-{number}.json"
        raw = (run / filename).read_bytes()  # Both files are required before any match.
        generation = json.loads(raw)
        if (not isinstance(generation.get("sample_id"), str) or not generation["sample_id"]
                or generation["sample_id"] in {"nearest", "value"}):
            raise ValueError(f"missing or reserved sample identity: {filename}")
        if generation.get("prompt_sha256") != plan.get("prompt_sha256"):
            raise ValueError(f"generation prompt hash differs from plan: {filename}")
        if digest(generation.get("prompt", "").encode()) != plan["prompt_sha256"]:
            raise ValueError(f"generation prompt contents differ from recorded hash: {filename}")
        if generation.get("client_source_sha256") != plan["source_sha256"][generation_client(plan)]:
            raise ValueError(f"generation client source mismatch: {filename}")
        local = plan.get("generation_backend") == "local_mlx"
        identity = generation.get("model") if local else generation.get("returned_model")
        if identity != plan.get("model") or generation.get("reasoning_mode") != plan.get("reasoning"):
            raise ValueError(f"generation model/reasoning differs from plan: {filename}")
        if plan.get("generation_backend") == "openrouter_model_screen":
            if (generation.get("generation_client") != "program_model_screen"
                    or generation.get("generation_client_profile") != plan.get("generation_client_profile")
                    or generation.get("generation_client_configuration_sha256") != plan.get("generation_client_configuration_sha256")
                    or generation.get("returned_provider") != plan["generation_client_profile"]["returned_provider"]):
                raise ValueError(f"model-screen generation provenance mismatch: {filename}")
        if local:
            if (generation.get("backend") != "mlx" or generation.get("revision") != plan.get("revision")
                    or generation.get("adapter_path") is not None or generation.get("training") is not False
                    or generation.get("external_spend_usd") != 0
                    or generation.get("thinking") is not (plan.get("reasoning") == "on")):
                raise ValueError(f"local generation provenance mismatch: {filename}")
            if generation.get("sampling") != plan.get("sampling"):
                raise ValueError(f"local generation sampling differs from plan: {filename}")
            if generation.get("text") and (generation.get("finish_reason") != "stop"
                    or generation.get("usable_final") is not True
                    or generation.get("extraction_status") != "complete"):
                raise ValueError(f"local generation final answer is incomplete: {filename}")
        for key in ("cost_usd", "cost_upper_usd"):
            cost = generation.get(key)
            if cost is not None and (type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0):
                raise ValueError(f"invalid generation {key}: {filename}")
        hashes[filename] = digest(raw)
        generations.append((filename, generation))
    if len({g["sample_id"] for _, g in generations}) != len(generations):
        raise ValueError("duplicate generated policy identity")
    if (run / "prompt.txt").exists() and digest((run / "prompt.txt").read_bytes()) != plan["prompt_sha256"]:
        raise ValueError("archived prompt.txt hash mismatch")
    return generations, hashes


def verified_reuse(run, plan, hashes):
    reuse = plan.get("generation_reuse")
    if reuse is None:
        return False
    if not isinstance(reuse, dict) or reuse.get("source_generation_sha256") != hashes:
        raise ValueError("generation_reuse must contain the exact source_generation_sha256 mapping")
    source = Path(reuse["source_run"])
    if not source.is_absolute():
        source = ROOT / source
    if source.resolve() == run.resolve():
        raise ValueError("generation reuse cannot reference itself")
    for filename, expected in hashes.items():
        if digest((source / filename).read_bytes()) != expected:
            raise ValueError(f"reused generation differs from its original: {filename}")
    return True


def schedule_for(identities, plan):
    schedule = []
    pairs = [(model, reference, "fixed_reference") for model in identities for reference in plan["fixed_opponents"]]
    pairs += [(a, b, "peer_crossplay") for a, b in itertools.combinations(identities, 2)]
    for a, b, kind in pairs:
        for seed in plan["validation_seeds"]:
            for seat in (0, 1):
                names = [a, b] if seat == 0 else [b, a]
                schedule.append({"episode_id": f"program-{len(schedule) + 1:04d}", "players": names,
                                 "seed": seed, "max_rounds": plan["max_rounds"], "match_kind": kind})
    return schedule


def verify_wrapper(wrapper, spec, runtime, programs, python_programs):
    expected_python = {name: python_programs[name].to_dict() for name in spec["players"] if name in python_programs}
    origins = ["model" if name in python_programs else "scripted" for name in spec["players"]]
    record = wrapper["record"]
    if (wrapper.get("episode_id") != spec["episode_id"] or wrapper.get("players") != spec["players"]
            or wrapper.get("match_kind") != spec["match_kind"] or wrapper.get("python_programs") != expected_python
            or record.get("seed") != spec["seed"] or record.get("max_rounds") != spec["max_rounds"]
            or record.get("origins") != origins
            or [p["digest"] for p in record["programs"]] != [programs[n].digest for n in spec["players"]]):
        raise ValueError(f"existing match provenance/schedule mismatch: {spec['episode_id']}")
    runtime.verify_replay(record)


def aggregate(rows):
    counts = {"games": len(rows), "wins": 0, "draws": 0, "losses": 0, "banked_points": 0,
              "opponent_banked_points": 0, "return_sum": 0.0, "program_faults": 0,
              "failed_game_actions": 0, "fault_types": {}, "decision_turns": 0}
    faults = Counter()
    for record, seat in rows:
        value = record["returns"][seat]
        counts["wins" if value > 0 else "losses" if value < 0 else "draws"] += 1
        counts["banked_points"] += record["metrics"]["scores"][seat]
        counts["opponent_banked_points"] += record["metrics"]["scores"][1 - seat]
        counts["return_sum"] += value
        counts["program_faults"] += record["metrics"]["program_faults"][seat]
        counts["failed_game_actions"] += record["metrics"]["failed_actions"][seat]
        counts["decision_turns"] += len(record["events"])
        faults.update(e["decisions"][seat]["fault"] for e in record["events"] if e["decisions"][seat]["fault"] is not None)
    counts["fault_types"] = dict(sorted(faults.items()))
    counts["mean_return"] = counts["return_sum"] / len(rows) if rows else None
    counts["mean_banked_points"] = counts["banked_points"] / len(rows) if rows else None
    return counts


def summarize(evaluation, wrappers):
    identities = [entry["sample_id"] for entry in evaluation["generations"]]
    players = {}
    for identity in identities:
        groups = {}
        rows = []
        for wrapper in wrappers:
            if identity in wrapper["players"]:
                seat = wrapper["players"].index(identity)
                opponent = wrapper["players"][1 - seat]
                row = (wrapper["record"], seat)
                groups.setdefault(opponent, []).append(row)
                rows.append(row)
        players[identity] = {"overall": aggregate(rows), "by_opponent": {name: aggregate(group) for name, group in groups.items()}}
    valid = sum(entry["valid"] for entry in evaluation["generations"])
    summary = {
        "format_version": FORMAT_VERSION, "complete": True, "screen": "bounded Python controller authoring",
        "training": False, "evaluation_api_calls": 0, "attempts": len(identities), "valid_programs": valid,
        "model_matches": len(wrappers), "all_replays_verified": True,
        "fixed_reference_matches": sum(w["match_kind"] == "fixed_reference" for w in wrappers),
        "peer_crossplay_matches": sum(w["match_kind"] == "peer_crossplay" for w in wrappers),
        "generations": evaluation["generations"], "failures": dict(Counter(e["failure"] for e in evaluation["generations"] if not e["valid"])),
        "players": players, "costs": evaluation["costs"], "plan_sha256": evaluation["plan_sha256"],
        "source_sha256": evaluation["source_sha256"], "generation_sha256": evaluation["generation_sha256"],
        "evaluation_sha256": digest(encoded(evaluation)),
        "scoring": "W/D/L is the sign of final banked-point difference at the game's fixed time limit. This is the game's actual objective, not distance/material adjudication. Return is that difference divided by initial resource value; unbanked cargo scores zero.",
        "interpretation": "Small adaptive authoring-capability screen against two fixed scripted references and any valid peer. No gradient training, continual improvement, or general-strength claim. Invalid programs receive no fabricated games or losses.",
        "generation_reuse": evaluation.get("generation_reuse"),
    }
    if "response_normalization" in evaluation and evaluation.get("generation_reuse"):
        summary.update(screen="post-hoc Python packaging revalidation", posthoc_diagnostic=True,
                       response_normalization=evaluation["response_normalization"],
                       diagnostic_amendment=evaluation["diagnostic_amendment"],
                       interpretation="Post-hoc packaging revalidation of the same two existing responses. The accepted-program count is not a fresh model capability rate. No semantic code repairs, new API generation, or gradient training.")
    elif "response_normalization" in evaluation:
        summary.update(response_normalization=evaluation["response_normalization"],
                       posthoc_diagnostic=False)
    return summary


def evaluate(run):
    run = Path(run).resolve()
    plan_bytes = (run / "plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    validate_plan(plan)
    # Read all responses first: an incomplete generation stage is not a screen.
    generations, generation_hashes = load_generations(run, plan)
    reused = verified_reuse(run, plan, generation_hashes)
    with (run / ".evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with archived_runtime(run, plan) as runtime:
            entries, python_programs = [], {}
            for filename, response in generations:
                entry, program = classify_generation(response, runtime)
                entry.update(response_file=filename, response_sha256=generation_hashes[filename],
                             returned_model=response.get("returned_model"), returned_provider=response.get("returned_provider"),
                             reasoning_mode=response.get("reasoning_mode"), cost_usd=response.get("cost_usd"),
                             cost_upper_usd=response.get("cost_upper_usd"), usage=response.get("usage"))
                if plan.get("generation_backend") == "local_mlx":
                    entry.update(returned_model=response["model"], returned_provider="local_mlx",
                                 revision=response["revision"], cost_usd=0, cost_upper_usd=0,
                                 extraction_status=response["extraction_status"])
                entries.append(entry)
                if program is not None:
                    python_programs[response["sample_id"]] = program
            actuals = [g.get("cost_usd") for _, g in generations]
            uppers = [g.get("cost_upper_usd") for _, g in generations]
            if plan.get("generation_backend") == "local_mlx":
                # Local responses explicitly assert zero external spend above.
                actuals = [0 for _ in generations]
                uppers = [0 for _ in generations]
            costs = {"original_response_cost_usd": sum(actuals) if all(v is not None for v in actuals) else None,
                     "original_response_cost_upper_usd": sum(uppers) if all(v is not None for v in uppers) else None,
                     "reused_generation_responses": reused, "evaluation_api_cost_usd": 0,
                     "new_generation_cost_usd": 0 if reused else (sum(actuals) if all(v is not None for v in actuals) else None),
                     "scope": "This run's generation arm only. Prior JSON screen and other pilot costs are separate; copied responses are not charged again."}
            programs = {name: runtime.compile_program(source) for name, source in runtime.reference_programs().items() if name in plan["fixed_opponents"]}
            programs.update({name: p.compiled for name, p in python_programs.items()})
            schedule = schedule_for(list(python_programs), plan)
            evaluation = {"format_version": FORMAT_VERSION, "plan_sha256": digest(plan_bytes),
                          "source_sha256": plan["source_sha256"], "generation_sha256": generation_hashes,
                          "generations": entries, "schedule": schedule, "costs": costs,
                          "frontend_version": runtime.frontend_version, "python_ast_node_limit": runtime.python_ast_node_limit,
                          "model_spec_sha256": digest(runtime.model_spec().encode()),
                          "python_model_spec_sha256": digest(runtime.python_model_spec().encode()),
                          "generation_reuse": plan.get("generation_reuse")}
            if plan.get("response_normalization") is not None:
                evaluation.update(response_normalization=plan["response_normalization"],
                                  diagnostic_amendment=plan.get("interpretation"))
            evaluation_path, matches_path, summary_path = (run / name for name in ("evaluation.json", "matches.jsonl", "summary.json"))
            # A match file or result without its provenance manifest cannot be
            # safely adopted. Never silently overwrite an older experiment.
            if not evaluation_path.exists() and any(p.exists() for p in (matches_path, summary_path)):
                raise ValueError("existing matches/results have no compatible evaluation manifest")
            write_once(evaluation_path, evaluation)
            wrappers = []
            if matches_path.exists():
                content = matches_path.read_bytes()
                if content and not content.endswith(b"\n"):
                    raise ValueError("partial final matches.jsonl line retained; manual artifact recovery required")
                wrappers = [json.loads(line) for line in content.splitlines()]
            if len(wrappers) > len(schedule):
                raise ValueError("more existing matches than the frozen schedule")
            for index, wrapper in enumerate(wrappers):
                verify_wrapper(wrapper, schedule[index], runtime, programs, python_programs)
            if summary_path.exists():
                if len(wrappers) != len(schedule) or read_json(summary_path) != summarize(evaluation, wrappers):
                    raise ValueError("existing summary is incompatible or falsely complete")
                return read_json(summary_path)
            with matches_path.open("ab") as stream:
                for spec in schedule[len(wrappers):]:
                    names = spec["players"]
                    record = runtime.play([programs[n] for n in names], spec["seed"], spec["max_rounds"],
                                          origins=["model" if n in python_programs else "scripted" for n in names])
                    wrapper = {"episode_id": spec["episode_id"], "players": names, "match_kind": spec["match_kind"],
                               "record": record, "python_programs": {n: python_programs[n].to_dict() for n in names if n in python_programs}}
                    verify_wrapper(wrapper, spec, runtime, programs, python_programs)
                    stream.write((json.dumps(wrapper, separators=(",", ":"), allow_nan=False) + "\n").encode())
                    stream.flush()
                    os.fsync(stream.fileno())
                    wrappers.append(wrapper)
            summary = summarize(evaluation, wrappers)
            summary["complete"] = len(wrappers) == len(schedule)
            write_once(summary_path, summary)
            return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = evaluate(args.run)
    except Exception as error:
        # No existing artifacts are removed, patched, or replaced on failure.
        if args.run.is_dir():
            directory = args.run / "evaluation-failures"
            directory.mkdir(exist_ok=True)
            write_once(directory / (uuid.uuid4().hex + ".json"),
                       {"complete": False, "error_type": type(error).__name__, "error": str(error)})
        raise
    print(json.dumps({key: summary[key] for key in (
        "complete", "attempts", "valid_programs", "model_matches", "failures", "costs", "players")}, indent=2))


if __name__ == "__main__":
    main()
