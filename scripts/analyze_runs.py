"""Recompute comparable procedural-task statistics from raw local pilot logs."""
import itertools
import json
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selfplay.environment import Task
from selfplay.pilot import summarize


def analyze(root):
    result = {}
    for directory in sorted(root.iterdir()):
        if not (directory / "summary.json").exists():
            continue
        manifest = json.loads((directory / "manifest.json").read_text())
        # Depth collections have a separate schema and matched-adapter analysis.
        if manifest["config"].get("command") == "collect":
            continue
        rows = [json.loads(s) for s in (directory / "rollouts.jsonl").read_text().splitlines()]
        procedural = [r for r in rows if r["origin"] == "procedural"]
        summary = summarize(procedural)
        by_depth = defaultdict(list)
        for row in procedural:
            by_depth[row["depth"]].append(row["success"])
        task_rows = [json.loads(s) for s in (directory / "tasks.jsonl").read_text().splitlines()]
        expected = []
        for row in task_rows:
            if row["origin"] != "procedural":
                continue
            task = Task.from_dict({k: row[k] for k in Task.__dataclass_fields__})
            if task.max_steps > 6:
                raise ValueError("Exhaustive random baseline is intended only for short pilot tasks")
            p = sum(sum(task.accepts(plan) for plan in itertools.product(task.actions, repeat=k)) /
                    len(task.actions)**k for k in range(1, task.max_steps + 1)) / task.max_steps
            expected.append(p)
        proposals = [json.loads(s) for s in (directory / "proposals.jsonl").read_text().splitlines()]
        result[directory.name] = {
            "complete": json.loads((directory / "summary.json").read_text())["complete"],
            "procedural_tasks": summary["unique_tasks"], "procedural_attempts": len(procedural),
            "procedural_solved": sum(r["success"] for r in procedural),
            "by_variant": summary["by_variant"],
            "paired_gap": summary["paired_renamed_reordered_minus_canonical"],
            "by_depth": {d: {"successes": sum(v), "attempts": len(v)} for d, v in sorted(by_depth.items())},
            "exact_uniform_legal_random_expectation": sum(expected) / len(expected) if expected else None,
            "tasks_with_positive_frontier_score": sum(s["frontier_reward"] > 0 for s in summary["task_scores"]),
            "tasks_with_positive_robust_frontier_score": sum(s["robust_frontier_reward"] > 0 for s in summary["task_scores"]),
            "proposals_attempted": len(proposals), "proposals_accepted": sum(p["valid"] for p in proposals),
            "llm_task_attempts": sum(r["origin"] == "llm" for r in rows),
            "llm_task_solved": sum(r["success"] for r in rows if r["origin"] == "llm"),
            "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in rows + proposals),
            "completion_tokens": sum(r["completion_tokens"] or 0 for r in rows + proposals),
        }
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "runs"
    output = analyze(root)
    (root / "comparison.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
