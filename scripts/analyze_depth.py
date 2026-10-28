"""Paired held-out evaluation of a frozen base and two trained adapters."""
import argparse
from collections import Counter, defaultdict
from itertools import product
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selfplay.depth import read_tasks, render_depth
from selfplay.pilot import grade_response


def read(path):
    manifest = json.loads((path / "manifest.json").read_text())
    if not json.loads((path / "summary.json").read_text())["complete"]:
        raise ValueError("Incomplete evaluation: " + str(path))
    tasks = {r["task_id"]: (r, task) for r, task in read_tasks(path / "tasks.jsonl")}
    records = {}
    for row in map(json.loads, (path / "rollouts.jsonl").read_text().splitlines()):
        spec, task = tasks[row["task_id"]]
        prompt, mapping = render_depth(spec, task, row["variant"])
        if prompt != row["prompt"] or grade_response(task, row, mapping)["success"] != row["success"]:
            raise ValueError("Evaluation failed independent regrading")
        key = (row["task_id"], row["variant"], row["attempt"])
        if key in records or row["format"] != "strict":
            raise ValueError("Unexpected evaluation protocol")
        records[key] = row
    expected = {(tid, variant, attempt) for tid in tasks for variant in manifest["config"]["variants"].split(",")
                for attempt in range(manifest["config"]["attempts"])}
    if records.keys() != expected or manifest["config"]["attempts"] != 4:
        raise ValueError("Expected complete four-sample evaluation")
    return manifest, tasks, records


def summarize(rows):
    tasks = defaultdict(list)
    for row in rows:
        tasks[row["task_id"]].append(row)
    return {"attempts": len(rows), "successes": sum(r["success"] for r in rows),
            "pass_at_1": sum(r["success"] for r in rows) / len(rows),
            "task_pass_at_4": sum(any(r["success"] for r in rs) for rs in tasks.values()) / len(tasks),
            "unique_tasks": len(tasks), "statuses": dict(Counter(r["status"] for r in rows)),
            "completion_tokens": sum(r["completion_tokens"] for r in rows),
            "mean_distinct_final_plans_per_task": sum(len({r["text"] for r in rs}) for rs in tasks.values()) / len(tasks)}


def compare(base_path, uniform_path, frontier_path, arm_names=("uniform", "frontier"), expected_updates=80):
    first, second = arm_names
    runs = {name: read(path) for name, path in (("base", base_path), (first, uniform_path), (second, frontier_path))}
    reference, tasks, base = runs["base"]
    for name, (manifest, _, rows) in runs.items():
        adapter = manifest["model"].get("adapter_path")
        if (name == "base") != (adapter is None):
            raise ValueError("Expected unadapted base and trained comparison arms")
        if adapter:
            adapter = Path(adapter)
            training = json.loads((adapter / "summary.json").read_text())
            digest = hashlib.sha256((adapter / "adapters.safetensors").read_bytes()).hexdigest()
            if (not training["complete"] or training["updates"] != expected_updates
                    or digest != training["adapter_sha256"]
                    or digest != manifest["model"]["adapter_sha256"]):
                raise ValueError("Evaluated checkpoint differs from completed training")
        for module in ("strict.py", "pilot.py", "environment.py", "depth.py"):
            if manifest["source_sha256"][module] != reference["source_sha256"][module]:
                raise ValueError("Inference or verification implementation differs")
        if rows.keys() != base.keys() or manifest["task_manifest_sha256"] != reference["task_manifest_sha256"]:
            raise ValueError("Task/attempt identities differ")
        for field in ("seed", "attempts", "formats", "variants", "batch_size"):
            if manifest["config"][field] != reference["config"][field]:
                raise ValueError("Evaluation configurations differ")
        for field in ("model", "snapshot", "thinking", "top_k", "protocol"):
            if manifest["model"][field] != reference["model"][field]:
                raise ValueError("Base model or generation settings differ")
        for key, row in rows.items():
            if row["prompt"] != base[key]["prompt"] or row["seed"] != base[key]["seed"]:
                raise ValueError("Paired prompt or seed mismatch")
    result = {"models": {}, "paired_differences": {}, "exact_uniform_random_plan": {},
              "source_runs": {"base": str(base_path), first: str(uniform_path), second: str(frontier_path)},
              "checkpoint_sha256": {name: run[0]["model"].get("adapter_sha256") for name, run in runs.items()}}
    variants = sorted({key[1] for key in base})
    for name, (_, _, rows) in runs.items():
        result["models"][name] = {}
        for variant in variants:
            selected = [r for r in rows.values() if r["variant"] == variant]
            result["models"][name][variant] = summarize(selected)
            result["models"][name][variant]["by_depth"] = {
                d: summarize([r for r in selected if r["depth"] == d]) for d in sorted({r["depth"] for r in selected})}
    for variant in variants:
        for a, b in ((first, "base"), (second, "base"), (second, first)):
            grouped = defaultdict(list)
            for key in base:
                if key[1] == variant:
                    grouped[key[0]].append(int(runs[a][2][key]["success"]) - int(runs[b][2][key]["success"]))
            deltas = [sum(xs)/len(xs) for xs in grouped.values()]
            rng = random.Random(991)
            bootstrap = sorted(sum(rng.choice(deltas) for _ in deltas)/len(deltas) for _ in range(5000))
            result["paired_differences"][variant + ":" + a + "-" + b] = {
                "mean": sum(deltas)/len(deltas), "task_bootstrap_95_interval": [bootstrap[125], bootstrap[4874]]}
    rates = defaultdict(list)
    for spec, task in tasks.values():
        rate = sum(task.accepts(plan) for plan in product(task.actions, repeat=task.max_steps)) / len(task.actions)**task.max_steps
        rates[spec["depth"]].append(rate)
    result["exact_uniform_random_plan"] = {str(d): sum(xs)/len(xs) for d, xs in rates.items()}
    result["exact_uniform_random_plan"]["overall"] = sum(sum(xs) for xs in rates.values())/len(tasks)
    result["limitations"] = "Single training seed, small selected dataset, toy planning domain. Task-cluster intervals reflect sampled tasks, not training-seed uncertainty. Test results are not checkpoint selection criteria."
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "uniform", "frontier", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--arm-names", nargs=2, default=("uniform", "frontier"))
    ap.add_argument("--expected-updates", type=int, default=80)
    args = ap.parse_args()
    result = compare(args.base, args.uniform, args.frontier, args.arm_names, args.expected_updates)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
