"""Frozen proposer/solver loop with verified tasks and feedback between rounds.

Adaptation here is through proposer context. No model weights are updated.
"""
import argparse
import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .environment import construct_task, parse_object, render
from .pilot import MLXBackend, frontier, grade_response, proposer_prompt


def feedback_prompt(index, tasks, history, mode="outcomes"):
    prompt = proposer_prompt(index, [asdict(t) for t in tasks], "construct")
    return prompt + (
        "\nYou are proposing challenges for a solver using the same model as you. "
        "Aim for solvable problems on which some solver attempts succeed and some fail. "
        "Your score is 4*p*(1-p), where p is its empirical success rate. "
        "Invalid or duplicate tasks score zero. Use the previous feedback to vary the next task. "
        "Past feedback: " + json.dumps(history[-6:] if mode == "outcomes" else [])
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="mlx-community/Qwen3-4B-4bit")
    ap.add_argument("--revision", required=True)
    ap.add_argument("--proposals", type=int, default=8)
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-minutes", type=float, default=20)
    ap.add_argument("--max-calls", type=int, default=40)
    ap.add_argument("--seed", type=int, default=3059)
    ap.add_argument("--feedback-mode", choices=("outcomes", "none"), default="outcomes")
    args = ap.parse_args()
    if min(args.proposals, args.attempts, args.max_tokens, args.max_calls) < 1 or args.max_minutes <= 0:
        ap.error("Counts and limits must be positive")
    if args.proposals * (1 + args.attempts) > args.max_calls:
        ap.error("Worst-case calls exceed cap")
    args.out.mkdir(parents=True, exist_ok=False)
    backend = MLXBackend(args.model, args.revision, thinking=True, top_k=20)
    started = time.monotonic()
    config = vars(args).copy(); config["out"] = str(args.out)
    config.update(prompt_version="v2", temperature=0.6, variants="canonical", thinking=True)
    source = args.out / "source"; source.mkdir()
    hashes = {}
    for path in Path(__file__).parent.glob("*.py"):
        (source / path.name).write_bytes(path.read_bytes())
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"config": config, "model": backend.metadata, "training": False,
                "adaptation": "proposer context only", "source_sha256": hashes}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    tasks, history, proposals, rows = [], [], [], []
    seen, calls = set(), 0

    def complete(prompt, seed):
        nonlocal calls
        if calls >= args.max_calls or time.monotonic() - started > args.max_minutes * 60:
            raise RuntimeError("Run limit reached; partial data retained")
        calls += 1
        return backend.complete(prompt, seed, args.max_tokens, 0.6)

    try:
        with (args.out / "proposals.jsonl").open("w") as pf, (args.out / "rollouts.jsonl").open("w") as rf, (args.out / "tasks.jsonl").open("w") as tf:
            for i in range(args.proposals):
                prompt = feedback_prompt(i, tasks, history, args.feedback_mode)
                response = complete(prompt, args.seed + 100000 + i)
                proposal = {"index": i, "prompt": prompt, **response, "valid": False}
                try:
                    if response.get("finish_reason") == "length" or "</think>" not in response["text"]:
                        raise ValueError("Unfinished proposal")
                    task = construct_task(parse_object(response["text"]))
                    if task.modulus != 7 or len(task.actions) != 4 or not 2 <= task.max_steps <= 4:
                        raise ValueError("Proposal violates requested grammar")
                    if len(task.shortest_plan()) < 2:
                        raise ValueError("Shortest solution is too easy")
                    if task.identity in seen:
                        raise ValueError("Duplicate task")
                    proposal.update(valid=True, task_id=task.identity)
                except (ValueError, TypeError, KeyError) as exc:
                    proposal["rejection"] = str(exc)
                proposals.append(proposal)
                pf.write(json.dumps(proposal) + "\n"); pf.flush()
                if not proposal["valid"]:
                    history.append({"proposal": i, "reward": 0, "rejection": proposal["rejection"]})
                    print(f"round {i+1}: rejected: {proposal['rejection']}", flush=True)
                    continue
                seen.add(task.identity); tasks.append(task)
                tf.write(json.dumps({**asdict(task), "task_id": task.identity, "origin": "llm"}) + "\n"); tf.flush()
                task_index = len(tasks) - 1
                solver_prompt, mapping = render(task, seed=args.seed + task_index, prompt_version="v2")
                successes = 0
                for attempt in range(args.attempts):
                    seed = args.seed + task_index * 100 + attempt
                    t0 = time.monotonic()
                    response = complete(solver_prompt, seed)
                    row = {"task_id": task.identity, "origin": "llm", "depth": len(task.shortest_plan()),
                           "variant": "canonical", "attempt": attempt, "seed": seed, "prompt": solver_prompt,
                           **response, **grade_response(task, response, mapping, True), "seconds": time.monotonic() - t0}
                    rows.append(row); successes += row["success"]
                    rf.write(json.dumps(row) + "\n"); rf.flush()
                p = successes / args.attempts
                history.append({"task": asdict(task), "successes": successes, "attempts": args.attempts,
                                "frontier_reward": frontier(p)})
                (args.out / "feedback.json").write_text(json.dumps(history, indent=2))
                print(f"round {i+1}: valid, solved {successes}/{args.attempts}, reward {frontier(p):.2f}", flush=True)
    finally:
        summary = {"complete": len(proposals) == args.proposals and len(rows) == len(tasks) * args.attempts,
                   "proposals_attempted": len(proposals), "accepted_tasks": len(tasks),
                   "solver_attempts": len(rows), "solver_successes": sum(r["success"] for r in rows),
                   "statuses": dict(Counter(r["status"] for r in rows)),
                   "mixed_outcome_tasks": sum(0 < h.get("successes", 0) < args.attempts for h in history),
                   "model_calls": calls, "elapsed_seconds": time.monotonic() - started,
                   "training": False, "adaptation": "proposer context only"}
        summary["feedback_mode"] = args.feedback_mode
        (args.out / "feedback.json").write_text(json.dumps(history, indent=2))
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
