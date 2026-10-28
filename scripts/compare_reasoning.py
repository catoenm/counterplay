"""Compare complete paired reasoning/non-reasoning runs without mixing tasks."""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def read(run):
    manifest = json.loads((run / "manifest.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    if not summary["complete"]:
        raise ValueError("Cannot report a complete paired comparison from an interrupted run")
    rows = [json.loads(s) for s in (run / "rollouts.jsonl").read_text().splitlines()]
    keys = [(r["task_id"], r["variant"], r["attempt"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate rollout identities")
    return manifest, dict(zip(keys, rows))


def compare(on, off):
    mon, ron = read(on)
    moff, roff = read(off)
    for key in ("model", "snapshot", "backend", "top_k"):
        if mon["model"][key] != moff["model"][key]:
            raise ValueError("Model configuration mismatch: " + key)
    for key in ("seed", "max_tokens", "temperature", "prompt_version", "attempts", "variants", "depths"):
        if mon["config"][key] != moff["config"][key]:
            raise ValueError("Generation configuration mismatch: " + key)
    if not mon["model"]["thinking"] or moff["model"]["thinking"]:
        raise ValueError("Expected thinking-on and thinking-off, respectively")
    if ron.keys() != roff.keys():
        raise ValueError("Task/attempt/presentation sets differ")
    differences = defaultdict(list)
    for key in ron:
        if ron[key]["prompt"] != roff[key]["prompt"]:
            raise ValueError("Prompts differ")
        differences[key[0]].append(int(ron[key]["success"]) - int(roff[key]["success"]))
    deltas = [sum(xs) / len(xs) for xs in differences.values()]
    rng = random.Random(79)
    samples = sorted(sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(10000))
    output = {"unique_tasks": len(deltas), "paired_responses": len(ron),
              "success_difference": sum(deltas) / len(deltas),
              "exploratory_task_bootstrap_95_interval": [samples[250], samples[9749]],
              "warning": "Very small adaptive development study; inference improvement is not weight learning or generalization."}
    for name, rows in (("thinking_on", ron.values()), ("thinking_off", roff.values())):
        rows = list(rows)
        tokens = sum(r["completion_tokens"] for r in rows)
        successes = sum(r["success"] for r in rows)
        by_variant = defaultdict(list)
        for r in rows:
            by_variant[r["variant"]].append(r["success"])
        output[name] = {"successes": successes, "attempts": len(rows), "success_rate": successes / len(rows),
                        "successful_tasks": len({r["task_id"] for r in rows if r["success"]}),
                        "statuses": dict(Counter(r["status"] for r in rows)),
                        "completion_tokens": tokens,
                        "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
                        "successes_per_1000_completion_tokens": successes / tokens * 1000 if tokens else None,
                        "by_variant": {v: {"successes": sum(xs), "attempts": len(xs)} for v, xs in by_variant.items()}}
    return output


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("on", type=Path)
    ap.add_argument("off", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = compare(args.on, args.off)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
