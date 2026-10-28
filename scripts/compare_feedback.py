"""Summarize a frozen outcome-feedback diagnostic and its control."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selfplay.environment import Task, grade, parse_object, render


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def read(path):
    manifest = json.loads((path / "manifest.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    if not summary["complete"]:
        raise ValueError("Expected complete run: " + str(path))
    proposals = records(path / "proposals.jsonl")
    rows = records(path / "rollouts.jsonl")
    raw_tasks = records(path / "tasks.jsonl")
    tasks = [Task.from_dict({k: t[k] for k in Task.__dataclass_fields__}) for t in raw_tasks]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["task_id"]].append(row)
    totals = [sum(r["success"] for r in rs) for rs in grouped.values()]
    task_map = {t.identity: t for t in tasks}
    label_diagnostics = []
    for row in rows:
        if row["status"] != "illegal_action" or row["variant"] != "canonical":
            continue
        original = parse_object(row["text"])
        if not all(isinstance(x, str) for x in original["plan"]):
            continue
        edited = {"plan": [re.sub(r"^tool([0-9]+)$", r"tool_\1", x) for x in original["plan"]]}
        if edited == original:
            continue
        task = task_map[row["task_id"]]
        _, mapping = render(task, prompt_version="v2")
        label_diagnostics.append({"task_id": task.identity, "attempt": row["attempt"],
                                  "original_final": original, "underscore_edit": edited,
                                  "edited_plan_verifies": grade(task, json.dumps(edited), mapping)["success"],
                                  "note": "Post-hoc diagnostic only; primary grade remains a failure."})
    attempts = manifest["config"]["attempts"]
    if len(tasks) != len(grouped) or any(len(rs) != attempts for rs in grouped.values()):
        raise ValueError("Task and rollout counts disagree")
    return manifest, proposals, {
        "proposals": len(proposals), "accepted_tasks": len(tasks),
        "rejections": dict(Counter(p["rejection"] for p in proposals if not p["valid"])),
        "solver_successes": sum(totals), "solver_attempts": len(rows),
        "solver_statuses": dict(Counter(r["status"] for r in rows)),
        "posthoc_label_diagnostics": label_diagnostics,
        "mixed_outcome_tasks": sum(0 < n < attempts for n in totals),
        "all_solved_tasks": sum(n == attempts for n in totals),
        "all_failed_tasks": sum(n == 0 for n in totals),
        "shortest_depth_counts": dict(Counter(len(t.shortest_plan()) for t in tasks)),
        "extra_allowed_steps_above_shortest": dict(Counter(t.max_steps - len(t.shortest_plan()) for t in tasks)),
        "proposed_witness_length_counts": dict(Counter(len(t.witness) for t in tasks)),
        "distinct_action_sets": len({tuple(sorted(t.actions)) for t in tasks}),
        "task_ids": [t.identity for t in tasks],
        "completion_tokens": sum(r["completion_tokens"] for r in proposals + rows),
        "prompt_tokens": sum(r["prompt_tokens"] for r in proposals + rows),
        "elapsed_seconds": summary["elapsed_seconds"],
    }


def compare(on, off):
    mon, pon, son = read(on)
    moff, poff, soff = read(off)
    for key in ("seed", "proposals", "attempts", "max_tokens", "temperature", "prompt_version"):
        if mon["config"][key] != moff["config"][key]:
            raise ValueError("Configuration mismatch: " + key)
    if mon["model"] != moff["model"]:
        raise ValueError("Model configuration mismatch")
    if mon["config"].get("feedback_mode", "outcomes") != "outcomes" or moff["config"]["feedback_mode"] != "none":
        raise ValueError("Expected outcome-feedback arm and no-feedback control")
    if pon[0]["prompt"] != poff[0]["prompt"]:
        raise ValueError("Initial proposer prompts differ")
    return {
        "feedback": son, "control": soff,
        "identical_first_proposal": pon[0]["text"] == poff[0]["text"],
        "overlapping_task_ids": sorted(set(son["task_ids"]) & set(soff["task_ids"])),
        "training": False,
        "warning": "One adaptive context trajectory per arm; rounds are dependent. Two solver samples per task cannot establish true difficulty or a feedback benefit. Actual token budgets differ.",
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("feedback", type=Path)
    ap.add_argument("control", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = compare(args.feedback, args.control)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
