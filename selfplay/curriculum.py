"""Select independently verified supervision and matched-depth curricula."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from .depth import MODEL, REVISION, read_tasks, render_depth
from .environment import parse_object
from .pilot import frontier, grade_response


def check_splits(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, info in manifest["split_files"].items():
        if hashlib.sha256((directory / (name + ".jsonl")).read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError("Frozen task split changed")


def load_run(path):
    model = json.loads((path / "manifest.json").read_text())["model"]
    if (model.get("backend") != "mlx" or model.get("model") != MODEL or model.get("snapshot") != REVISION
            or model.get("thinking") is not False or model.get("adapter_path") is not None):
        raise ValueError("Expected frozen base-model direct-plan rollouts")
    if not json.loads((path / "summary.json").read_text())["complete"]:
        raise ValueError("Incomplete collection run")
    tasks = {row["task_id"]: (row, task) for row, task in read_tasks(path / "tasks.jsonl")}
    grouped, identities = defaultdict(list), set()
    for row in map(json.loads, (path / "rollouts.jsonl").read_text().splitlines()):
        key = (row["task_id"], row["attempt"])
        if key in identities or row["format"] != "strict" or row["variant"] != "canonical":
            raise ValueError("Expected unique strict canonical attempts")
        identities.add(key)
        spec, task = tasks[row["task_id"]]
        prompt, mapping = render_depth(spec, task)
        if row["prompt"] != prompt or grade_response(task, row, mapping)["success"] != row["success"]:
            raise ValueError("Independent regrading failed")
        grouped[row["task_id"]].append(row)
    return tasks, grouped


def prepare(args):
    check_splits(args.splits)
    for run, split in ((args.candidates, "candidates"), (args.validation, "validation")):
        if (run / "tasks.jsonl").read_bytes() != (args.splits / (split + ".jsonl")).read_bytes():
            raise ValueError("Collection tasks differ from frozen split")
    tasks, grouped = load_run(args.candidates)
    valid_tasks, valid_grouped = load_run(args.validation)
    train_families = {t[0]["family_id"] for t in tasks.values()}
    valid_families = {t[0]["family_id"] for t in valid_tasks.values()}
    if train_families & valid_families:
        raise ValueError("Training/validation family overlap")
    for filename in ("test.jsonl", "test_ood.jsonl"):
        test_families = {r["family_id"] for r, _ in read_tasks(args.splits / filename)}
        if test_families & (train_families | valid_families):
            raise ValueError("Evaluation leakage")
    eligible, scores, supervision = defaultdict(list), {}, {}
    for tid, rows in grouped.items():
        if {r["attempt"] for r in rows} != set(range(16)):
            raise ValueError("Need eight scoring and eight supervision attempts")
        scoring = [r for r in rows if r["attempt"] < 8]
        answers = [r for r in rows if r["attempt"] >= 8 and r["success"]]
        p = sum(r["success"] for r in scoring) / 8
        scores[tid] = {"score_rate": p, "frontier": frontier(p), "independent_rate": len(answers) / 8}
        if answers:
            eligible[tasks[tid][0]["depth"]].append(tid)
            supervision[tid] = min(answers, key=lambda x: x["attempt"])
    ranked = {d: [tid for tid in ids if scores[tid]["frontier"] > 0] for d, ids in eligible.items()}
    capacity = sum(len(ids) for ids in ranked.values())
    limit = next((n for n in (32, 20, 16) if n <= capacity), 0)
    if not limit or len([d for d, ids in ranked.items() if ids]) < 2:
        raise ValueError(f"Insufficient curriculum: { {d:len(v) for d,v in ranked.items()} }")
    counts = Counter()
    while sum(counts.values()) < limit:
        for depth in sorted(ranked):
            if counts[depth] < len(ranked[depth]) and sum(counts.values()) < limit:
                counts[depth] += 1
    rng = random.Random(71)
    selections = {"uniform": [], "frontier": []}
    for depth, count in sorted(counts.items()):
        uniform = sorted(eligible[depth]); rng.shuffle(uniform)
        ranked_ids = sorted(ranked[depth]); rng.shuffle(ranked_ids)
        ranked_ids.sort(key=lambda tid: scores[tid]["frontier"], reverse=True)
        selections["uniform"].extend(uniform[:count])
        selections["frontier"].extend(ranked_ids[:count])

    def example(tid, row, spec):
        return {"task_id": tid, "family_id": spec["family_id"], "depth": spec["depth"],
                "prompt": row["prompt"], "answer": json.dumps(parse_object(row["text"]), separators=(",", ":"))}

    valid_examples = []
    for tid, rows in sorted(valid_grouped.items()):
        successes = [r for r in rows if r["success"]]
        if successes:
            valid_examples.append(example(tid, min(successes, key=lambda r: r["attempt"]), valid_tasks[tid][0]))
    if len(valid_examples) < 4:
        raise ValueError("Need four independently verified validation examples")
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {"selection_seed": 71, "per_arm_tasks": limit, "depth_counts": dict(counts),
                "eligible_by_depth": {d: len(v) for d, v in eligible.items()},
                "positive_frontier_by_depth": {d: len(v) for d, v in ranked.items()},
                "scoring_attempts": list(range(8)), "supervision_attempts": list(range(8, 16)),
                "selections": selections, "validation_tasks": [r["task_id"] for r in valid_examples],
                "overlap": sorted(set(selections["uniform"]) & set(selections["frontier"])),
                "source_runs": [str(args.candidates), str(args.validation)], "files_sha256": {},
                "split_dir": str(args.splits),
                "split_manifest_sha256": hashlib.sha256((args.splits / "manifest.json").read_bytes()).hexdigest(),
                "selection_statistics": {arm: {"selection_success_rate": sum(scores[t]["score_rate"] for t in ids)/len(ids),
                                                "independent_success_rate": sum(scores[t]["independent_rate"] for t in ids)/len(ids)}
                                         for arm, ids in selections.items()}}
    for arm, ids in selections.items():
        directory = args.out / arm; directory.mkdir()
        examples = [example(tid, supervision[tid], tasks[tid][0]) for tid in ids]
        for name, rows in (("train", examples), ("valid", valid_examples)):
            path = directory / (name + ".jsonl")
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            manifest["files_sha256"][str(path.relative_to(args.out))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.out / "task-scores.json").write_text(json.dumps(scores, indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k not in ("selections", "overlap", "validation_tasks")}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("candidates", "validation", "splits", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    prepare(ap.parse_args())
