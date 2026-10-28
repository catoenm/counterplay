"""Export verified final-answer supervision, grouped and split by task identity.

This supports a future filtered-self-training baseline. It does not train and
does not turn inference-time reasoning into an RL result.
"""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from .environment import Task, parse_object, render
from .pilot import frontier, grade_response


def load_verified(run):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["model"]["backend"] not in ("mlx", "chat_completions"):
        raise ValueError("Only model rollouts may supply self-training examples")
    config = manifest["config"]
    task_rows = [json.loads(x) for x in (run / "tasks.jsonl").read_text().splitlines()]
    tasks = {}
    for i, row in enumerate(task_rows):
        task = Task.from_dict({k: row[k] for k in Task.__dataclass_fields__})
        if row["task_id"] != task.identity or task.identity in tasks:
            raise ValueError("Invalid or duplicate task identity")
        tasks[task.identity] = (i, task)
    outcomes, successes = defaultdict(list), defaultdict(list)
    for line in (run / "rollouts.jsonl").read_text().splitlines():
        row = json.loads(line)
        i, task = tasks[row["task_id"]]
        prompt, mapping = render(task, row["variant"], config["seed"] + i, config.get("prompt_version", "v1"))
        if prompt != row["prompt"]:
            raise ValueError("Prompt differs from recorded task/render protocol")
        check = grade_response(task, row, mapping, manifest["model"].get("thinking") is True)
        if check["success"] != row["success"]:
            raise ValueError("Stored grade does not match independent regrading")
        outcomes[task.identity].append(check["success"])
        if check["success"]:
            answer = json.dumps(parse_object(row["text"]), separators=(",", ":"))
            successes[task.identity].append({"messages": [{"role": "user", "content": prompt},
                                                         {"role": "assistant", "content": answer}]})
    scores = {k: frontier(sum(v) / len(v)) for k, v in outcomes.items()}
    return successes, scores


def split_ids(ids, seed, valid_fraction=0.2):
    ids = sorted(set(ids))
    if len(ids) < 5:
        raise ValueError("Need at least five independently verified tasks to make a development split")
    random.Random(seed).shuffle(ids)
    nvalid = max(1, round(len(ids) * valid_fraction))
    return ids[nvalid:], ids[:nvalid]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--selection", choices=["uniform", "frontier"], default="uniform")
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--exclude-tasks", type=Path, action="append", default=[])
    args = ap.parse_args()
    examples, scores = load_verified(args.run)
    excluded = {json.loads(line)["task_id"] for p in args.exclude_tasks for line in p.read_text().splitlines()}
    eligible = set(examples) - excluded
    train_ids, valid_ids = split_ids(eligible, args.seed)
    if args.limit < 1:
        ap.error("--limit must be positive")
    rng = random.Random(args.seed)
    if args.selection == "frontier":
        rng.shuffle(train_ids)
        train_ids.sort(key=lambda x: scores[x], reverse=True)
        train_ids = [x for x in train_ids if scores[x] > 0]
    selected = train_ids[:args.limit]
    if not selected:
        ap.error("No eligible training tasks; do not fabricate supervision")
    args.out.mkdir(parents=True, exist_ok=False)
    # One example per task avoids giving duplicated attempts disproportionate weight.
    for name, ids in (("train", selected), ("valid", valid_ids)):
        rows = [rng.choice(examples[k]) for k in ids]
        (args.out / (name + ".jsonl")).write_text("".join(json.dumps(x) + "\n" for x in rows))
    meta = {"source_run": str(args.run), "selection": args.selection, "seed": args.seed,
            "train_task_ids": selected, "valid_task_ids": valid_ids,
            "excluded_task_ids": sorted(excluded), "supervision": "verified final answers only; no reasoning trace or oracle solutions",
            "caution": "Task split prevents exact identity overlap, not semantic similarity or family leakage. Curriculum scores reuse rollout samples; an unbiased selection experiment needs separate scoring samples.",
            "files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.glob("*.jsonl")}}
    (args.out / "manifest.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"train_tasks": len(selected), "valid_tasks": len(valid_ids), "eligible_tasks": len(eligible)}))


if __name__ == "__main__":
    main()
