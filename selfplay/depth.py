"""Freeze depth-balanced splits and collect constrained or free action plans."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

from .environment import Task, generate_tasks, render
from .pilot import grade_response
from .strict import StrictBackend

MODEL = "mlx-community/Qwen3-4B-4bit"
REVISION = "4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25"


def family(task):
    """Group step budgets and the exact x/y swap symmetry together."""
    fields = {k: v for k, v in asdict(task).items() if k not in ("witness", "max_steps")}
    fields["actions"] = sorted(fields["actions"])
    swap = {"inc_x": "inc_y", "inc_y": "inc_x", "sum_x": "sum_y", "sum_y": "sum_x",
            "double_x": "double_y", "double_y": "double_x", "swap": "swap"}
    dual = {**fields, "start": list(reversed(task.start)), "goal": list(reversed(task.goal)),
            "actions": sorted(swap[a] for a in task.actions)}
    canonical = min(json.dumps(fields, sort_keys=True), json.dumps(dual, sort_keys=True))
    return hashlib.sha256(canonical.encode()).hexdigest()


def read_tasks(path):
    result = []
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        task = Task.from_dict({k: row[k] for k in Task.__dataclass_fields__})
        if task.identity != row["task_id"] or family(task) != row["family_id"]:
            raise ValueError("Task identity mismatch")
        result.append((row, task))
    return result


def render_depth(row, task, variant="canonical"):
    prompt, mapping = render(task, variant, row["render_seed"], "v2")
    return prompt + f"\nThis is a fixed-horizon task: return exactly {task.max_steps} tool calls in your plan.\n", mapping


def prepare(out, candidates=256):
    out.mkdir(parents=True, exist_ok=False)
    seen = set()
    # Every previously inspected puzzle remains development material.
    for path in Path("runs").glob("*/tasks.jsonl"):
        for row in map(json.loads, path.read_text().splitlines()):
            task = Task.from_dict({k: row[k] for k in Task.__dataclass_fields__})
            seen.add(family(task))
    specifications = [("calibration", 16, 7, (2, 3, 4, 5), 6121),
                      ("candidates", candidates, 7, (2, 3, 4, 5), 7319),
                      ("validation", 32, 7, (2, 3, 4, 5), 8111),
                      ("test", 64, 7, (2, 3, 4, 5), 9199),
                      ("test_ood", 32, 11, (2, 3, 4, 5), 10111)]
    manifest = {"seed_protocol": specifications, "model": MODEL, "revision": REVISION,
                "excluded_prior_families": len(seen), "split_files": {},
                "family_rule": "Ignore step budget and witness; collapse exact x/y swap symmetry. Other semantic similarities remain."}
    for name, count, modulus, depths, seed in specifications:
        rows = []
        cursor = 0
        while len(rows) < count:
            depth = depths[len(rows) % len(depths)]
            task = generate_tasks(1, seed + cursor, (depth,), modulus)[0]
            cursor += 1
            if family(task) in seen:
                continue
            seen.add(family(task))
            rows.append({**asdict(task), "task_id": task.identity, "family_id": family(task),
                         "split": name, "depth": depth, "render_seed": seed + len(rows)})
        path = out / (name + ".jsonl")
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        manifest["split_files"][name] = {"count": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def collect(args):
    task_rows = read_tasks(args.tasks)
    resuming = getattr(args, "resume", False)
    args.out.mkdir(parents=True, exist_ok=resuming)
    existing, prior_seconds = [], 0
    if resuming:
        original = json.loads((args.out / "manifest.json").read_text())
        for key in ("tasks", "attempts", "formats", "variants", "seed", "adapter", "batch_size"):
            if str(original["config"][key]) != str(getattr(args, key)):
                raise ValueError("Resume configuration differs: " + key)
        if original["task_manifest_sha256"] != hashlib.sha256(args.tasks.read_bytes()).hexdigest():
            raise ValueError("Resume task manifest differs")
        if original["source_sha256"]["strict.py"] != hashlib.sha256(Path(__file__).with_name("strict.py").read_bytes()).hexdigest():
            raise ValueError("Inference implementation changed; start a separate run")
        existing = [json.loads(line) for line in (args.out / "rollouts.jsonl").read_text().splitlines()]
        prior_seconds = json.loads((args.out / "summary.json").read_text())["elapsed_seconds"]
    model = StrictBackend(MODEL, REVISION, args.adapter)
    if args.adapter:
        model.metadata["adapter_sha256"] = hashlib.sha256((Path(args.adapter) / "adapters.safetensors").read_bytes()).hexdigest()
    config = vars(args).copy()
    config.update(tasks=str(args.tasks), out=str(args.out))
    segment = str(time.time_ns())
    snapshot = args.out / ("source-resume-" + segment if resuming else "source"); snapshot.mkdir()
    for file in Path(__file__).parent.glob("*.py"):
        (snapshot / file.name).write_bytes(file.read_bytes())
    manifest = {"config": config, "model": model.metadata, "training": False,
                "task_manifest_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.glob("*.py")}}
    (args.out / ("resume-" + segment + ".json" if resuming else "manifest.json")).write_text(json.dumps(manifest, indent=2))
    if not resuming:
        (args.out / "tasks.jsonl").write_bytes(args.tasks.read_bytes())
    jobs = [(i, v, f, a) for i in range(len(task_rows)) for v in args.variants.split(",")
            for f in args.formats.split(",") for a in range(args.attempts)]
    random.Random(args.seed).shuffle(jobs)
    indices = {task.identity: i for i, (_, task) in enumerate(task_rows)}
    done = {(indices[r["task_id"]], r["variant"], r["format"], r["attempt"]) for r in existing}
    if len(done) != len(existing) or not done <= set(jobs):
        raise ValueError("Invalid saved rollout identities")
    for result in existing:
        row, task = task_rows[indices[result["task_id"]]]
        prompt, mapping = render_depth(row, task, result["variant"])
        if prompt != result["prompt"] or grade_response(task, result, mapping)["success"] != result["success"]:
            raise ValueError("Saved rollout did not reverify")
    total_jobs = len(jobs)
    jobs = [j for j in jobs if j not in done]
    started, results = time.monotonic(), existing
    try:
        with (args.out / "rollouts.jsonl").open("a" if resuming else "w") as sink:
            for offset in range(0, len(jobs), args.batch_size):
                if time.monotonic() - started > args.max_minutes * 60:
                    raise RuntimeError("Time admission limit reached")
                batch = jobs[offset:offset + args.batch_size]
                prepared = []
                for i, variant, fmt, attempt in batch:
                    row, task = task_rows[i]
                    prompt, mapping = render_depth(row, task, variant)
                    seed = args.seed + i * 100 + attempt
                    prepared.append((prompt, list(mapping), task.max_steps, seed, fmt == "strict"))
                t0 = time.monotonic()
                responses = model.complete_batch(prepared)
                duration = time.monotonic() - t0
                for (i, variant, fmt, attempt), request, response in zip(batch, prepared, responses):
                    row, task = task_rows[i]
                    prompt, mapping = render_depth(row, task, variant)
                    result = {"task_id": task.identity, "family_id": family(task), "depth": row["depth"],
                              "variant": variant, "format": fmt, "attempt": attempt, "seed": request[3],
                              "prompt": prompt, **response, **grade_response(task, response, mapping),
                              "amortized_batch_seconds": duration / len(batch)}
                    results.append(result)
                    sink.write(json.dumps(result) + "\n")
                sink.flush()
                if len(results) % 64 == 0 or len(results) == total_jobs:
                    print(f"completed {len(results)}/{total_jobs}; solved {sum(r['success'] for r in results)}", flush=True)
    finally:
        grouped = defaultdict(list)
        for row in results:
            grouped[(row["format"], row["variant"], row["depth"])].append(row)
        summary = {"complete": len(results) == total_jobs, "calls": len(results),
                   "successes": sum(r["success"] for r in results),
                   "by_group": {str(k): {"calls": len(rs), "successes": sum(r["success"] for r in rs),
                                            "statuses": dict(Counter(r["status"] for r in rs))} for k, rs in grouped.items()},
                   "completion_tokens": sum(r["completion_tokens"] for r in results),
                   "elapsed_seconds": prior_seconds + time.monotonic() - started, "training": False}
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--candidates", type=int, default=256)
    run = sub.add_parser("collect")
    run.add_argument("--tasks", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--attempts", type=int, default=8)
    run.add_argument("--formats", default="strict")
    run.add_argument("--variants", default="canonical")
    run.add_argument("--seed", type=int, default=12011)
    run.add_argument("--max-minutes", type=float, default=60)
    run.add_argument("--adapter", default=None)
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.command == "prepare":
        prepare(args.out, args.candidates)
    else:
        if args.attempts < 1 or args.batch_size < 1 or set(args.formats.split(",")) - {"strict", "free"}:
            ap.error("Invalid attempts or formats")
        collect(args)


if __name__ == "__main__":
    main()
