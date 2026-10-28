"""Run a frozen-model curriculum audit; this command does not train weights."""
import argparse
import hashlib
import json
import os
import platform
import random
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .environment import RULES, Task, construct_task, generate_tasks, grade, parse_object, render


class MLXBackend:
    def __init__(self, model, revision=None, thinking=False, top_k=0):
        import mlx.core as mx
        from huggingface_hub import snapshot_download
        from mlx_lm import load
        self.mx = mx
        self.thinking, self.top_k = thinking, top_k
        path = snapshot_download(model, revision=revision)
        self.model, self.tokenizer = load(path)
        self.metadata = {"model": model, "snapshot": Path(path).name, "backend": "mlx", "thinking": thinking, "top_k": top_k}

    def complete(self, prompt, seed, max_tokens, temperature):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        self.mx.random.seed(seed)
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=self.thinking,
        )
        chunks, last = [], None
        for last in stream_generate(self.model, self.tokenizer, formatted,
                                   max_tokens=max_tokens, sampler=make_sampler(temp=temperature, top_p=0.95, top_k=self.top_k)):
            chunks.append(last.text)
        return {"text": "".join(chunks), "prompt_tokens": last.prompt_tokens if last else 0,
                "completion_tokens": last.generation_tokens if last else 0,
                "finish_reason": last.finish_reason if last else None}


class EndpointBackend:
    def __init__(self, model, base_url):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.metadata = {"model": model, "backend": "chat_completions", "thinking": "server default"}

    def complete(self, prompt, seed, max_tokens, temperature):
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                   "temperature": temperature, "max_tokens": max_tokens, "seed": seed}
        headers = {"Content-Type": "application/json"}
        if os.environ.get("MODEL_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["MODEL_API_KEY"]
        request = urllib.request.Request(self.base_url + "/chat/completions",
                                         json.dumps(payload).encode(), headers)
        # No automatic retries: failed requests may still incur provider cost.
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
        usage = result.get("usage", {})
        return {"text": result["choices"][0]["message"]["content"] or "",
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens")}


def frontier(p):
    return 4 * p * (1 - p)


def grade_response(task, response, mapping, thinking=False):
    if response.get("finish_reason") == "length":
        return {"success": False, "status": "truncated"}
    if thinking and "</think>" not in response["text"]:
        return {"success": False, "status": "unfinished_reasoning"}
    return grade(task, response["text"], mapping)


def summarize(rows):
    grouped, per_task = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for r in rows:
        grouped[r["variant"]].append(r)
        per_task[r["task_id"]][r["variant"]].append(int(r["success"]))
    scores = []
    for task_id, variants in per_task.items():
        rates = {v: sum(xs) / len(xs) for v, xs in variants.items()}
        if "canonical" not in rates:
            continue
        scores.append({"task_id": task_id, "rates": rates,
                       "failure_reward": 1 - rates["canonical"],
                       "frontier_reward": frontier(rates["canonical"]),
                       "robust_frontier_reward": min(map(frontier, rates.values())),
                       "presentation_range": max(rates.values()) - min(rates.values())})
    # Cluster by task: attempts/presentations of one task are not independent tasks.
    gaps = [s["rates"]["renamed_reordered"] - s["rates"]["canonical"] for s in scores if "renamed_reordered" in s["rates"]]
    rng = random.Random(1234)
    boot = sorted(sum(rng.choice(gaps) for _ in gaps) / len(gaps) for _ in range(2000)) if gaps else []
    return {
        "unique_tasks": len(per_task), "calls": len(rows),
        "by_variant": {v: {"attempts": len(xs), "success_rate": sum(r["success"] for r in xs) / len(xs),
                            "statuses": dict(Counter(r["status"] for r in xs))} for v, xs in grouped.items()},
        "paired_renamed_reordered_minus_canonical": {
            "paired_tasks": len(gaps),
            "mean": sum(gaps) / len(gaps) if gaps else None,
            "task_bootstrap_95_interval": [boot[50], boot[1949]] if boot else None,
            "note": "Exploratory task-cluster bootstrap; a tiny pilot cannot establish a general effect."},
        "task_scores": scores,
    }


def proposer_prompt(index, existing, mode="claim"):
    prompt = (
        "Invent a finite planning puzzle you can solve. Return only JSON with exactly these fields: "
        'modulus (integer 7), start ([x,y]), goal ([x,y]), actions (four distinct allowed names), '
        'max_steps (integer 2 to 4), witness (array of allowed action names). '
        "Coordinates must be integers 0 through 6. All arithmetic is modulo 7. "
        "The witness MUST reach goal from start within max_steps. Start must differ from goal. "
        "Try to require at least two steps even for the shortest solution. "
        "Use a new combination of start, goal and actions. "
        "Rules: " + json.dumps(RULES) + f"\nProposal number: {index}\n"
        "Avoid these earlier valid tasks: " + json.dumps(existing[-4:])
    )
    if mode == "construct":
        prompt = (
            "Create a planning puzzle by choosing a starting state and executing a path. "
            "The environment will compute the target from your path. Return ONLY JSON with exactly "
            "modulus (integer 7), start (two integers 0 through 6), actions (four distinct allowed names), "
            "max_steps (integer 2 to 4), witness (a sequence of 2 to max_steps chosen actions). "
            "Do not return a goal field. Try to make a path whose result requires at least two steps "
            "even under the shortest solution; avoid redundant actions and zero-effect paths. "
            "Rules: " + json.dumps(RULES) + f"\nProposal number: {index}\n"
            "Avoid these earlier valid paths: " + json.dumps([{k: v for k, v in t.items() if k != "goal"} for t in existing[-4:]])
        )
    return prompt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["oracle", "random", "mlx", "endpoint"], default="oracle")
    ap.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    ap.add_argument("--revision")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--prompt-version", choices=["v1", "v2"], default="v1")
    ap.add_argument("--base-url", default=os.environ.get("MODEL_BASE_URL", ""))
    ap.add_argument("--tasks", type=int, default=24)
    ap.add_argument("--depths", default="1,2,3")
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--proposals", type=int, default=0)
    ap.add_argument("--proposal-mode", choices=["claim", "construct"], default="claim")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-calls", type=int, default=400)
    ap.add_argument("--max-minutes", type=float, default=20)
    ap.add_argument("--variants", default="canonical,reordered,renamed,renamed_reordered")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    variants = args.variants.split(",")
    depths = tuple(map(int, args.depths.split(",")))
    if args.tasks < 1 or args.attempts < 1 or args.proposals < 0 or args.max_tokens < 1 or args.max_minutes <= 0:
        ap.error("Invalid count or limit")
    if len(set(variants)) != len(variants) or "canonical" not in variants or set(variants) - {"canonical", "renamed", "reordered", "renamed_reordered"}:
        ap.error("Invalid variants; canonical is required")
    if not depths or min(depths) < 1 or max(depths) > 12:
        ap.error("depths must be between 1 and 12")
    planned_calls = args.proposals + (args.tasks + args.proposals) * args.attempts * len(variants)
    if planned_calls > args.max_calls:
        ap.error(f"Worst-case {planned_calls} calls exceeds --max-calls")
    if args.backend == "endpoint" and not args.base_url:
        ap.error("Endpoint mode requires --base-url or MODEL_BASE_URL")
    if args.thinking and args.backend != "mlx":
        ap.error("--thinking currently supports only the MLX backend")
    if args.backend in ("oracle", "random") and args.proposals:
        ap.error("LLM proposals require a model backend")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    backend = None
    if args.backend == "mlx": backend = MLXBackend(args.model, args.revision, args.thinking, args.top_k)
    if args.backend == "endpoint": backend = EndpointBackend(args.model, args.base_url)
    config = vars(args).copy()
    config["out"] = str(args.out)
    config.pop("base_url")  # URLs may contain credentials; do not log them.
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    (args.out / "source").mkdir()
    for p in Path(__file__).parent.glob("*.py"):
        (args.out / "source" / p.name).write_bytes(p.read_bytes())
    manifest = {"config": config, "model": backend.metadata if backend else {"backend": args.backend},
                "started_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                "source_sha256": source_hashes, "training": False, "max_planned_calls": planned_calls}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    tasks = generate_tasks(args.tasks, args.seed, depths)
    origins = {t.identity: "procedural" for t in tasks}
    proposals = []
    rows = []
    calls = 0
    def complete(prompt, seed):
        nonlocal calls
        if calls >= args.max_calls or time.monotonic() - started > args.max_minutes * 60:
            raise RuntimeError("Run limit reached; partial results retained")
        calls += 1
        return backend.complete(prompt, seed, args.max_tokens, args.temperature)
    try:
        with (args.out / "proposals.jsonl").open("w") as f:
            for i in range(args.proposals):
                prompt = proposer_prompt(i, [asdict(t) for t in tasks if origins[t.identity] == "llm"], args.proposal_mode)
                response = complete(prompt, args.seed + 100000 + i)
                record = {"index": i, "prompt": prompt, **response, "valid": False}
                try:
                    if response.get("finish_reason") == "length":
                        raise ValueError("Truncated proposal")
                    if args.thinking and "</think>" not in response["text"]:
                        raise ValueError("Unfinished proposal reasoning")
                    obj = parse_object(response["text"])
                    task = construct_task(obj) if args.proposal_mode == "construct" else Task.from_dict(obj)
                    if task.modulus != 7 or not 2 <= task.max_steps <= 4 or len(task.actions) != 4:
                        raise ValueError("Proposal violates requested grammar")
                    if len(task.shortest_plan()) < 2:
                        raise ValueError("Shortest path is too easy")
                    if task.identity in origins:
                        raise ValueError("Duplicate proposal")
                    tasks.append(task)
                    origins[task.identity] = "llm"
                    record.update(valid=True, task_id=task.identity)
                except (ValueError, TypeError, KeyError) as exc:
                    record["rejection"] = str(exc)
                proposals.append(record)
                f.write(json.dumps(record) + "\n"); f.flush()
                print(f"proposal {i + 1}/{args.proposals}: {'valid' if record['valid'] else record['rejection']}", flush=True)
        (args.out / "tasks.jsonl").write_text("".join(json.dumps({**asdict(t), "task_id": t.identity, "origin": origins[t.identity]}) + "\n" for t in tasks))
        jobs = [(i, variant, attempt) for i in range(len(tasks)) for variant in variants for attempt in range(args.attempts)]
        random.Random(args.seed).shuffle(jobs)
        with (args.out / "rollouts.jsonl").open("w") as f:
            for index, (i, variant, attempt) in enumerate(jobs):
                task = tasks[i]
                prompt, mapping = render(task, variant, args.seed + i, args.prompt_version)
                seed = args.seed + i * 100 + attempt
                t0 = time.monotonic()
                if backend:
                    response = complete(prompt, seed)
                else:
                    inverse = {a: label for label, a in mapping.items()}
                    plan = task.shortest_plan()
                    if args.backend == "random":
                        rng = random.Random(seed)
                        plan = [rng.choice(sorted(task.actions)) for _ in range(rng.randint(1, task.max_steps))]
                    response = {"text": json.dumps({"plan": [inverse[a] for a in plan]}),
                                "prompt_tokens": 0, "completion_tokens": 0}
                row = {"task_id": task.identity, "origin": origins[task.identity], "depth": len(task.shortest_plan()),
                       "variant": variant, "attempt": attempt, "seed": seed,
                       "prompt": prompt, **response, **grade_response(task, response, mapping, args.thinking),
                       "seconds": time.monotonic() - t0}
                rows.append(row)
                f.write(json.dumps(row) + "\n"); f.flush()
                if (index + 1) % 8 == 0:
                    print(f"rollouts {index + 1}/{len(jobs)}; solved {sum(r['success'] for r in rows)}", flush=True)
    finally:
        result = summarize(rows)
        result.update(proposals_attempted=len(proposals), proposals_accepted=sum(p["valid"] for p in proposals),
                      elapsed_seconds=time.monotonic() - started, training=False,
                      complete=len(rows) == len(tasks) * len(variants) * args.attempts,
                      model_calls=calls)
        # Partial jobs may omit a task's canonical variant; see summarize handling.
        (args.out / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "task_scores"}, indent=2))


if __name__ == "__main__":
    main()
