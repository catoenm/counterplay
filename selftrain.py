"""Prepare verified curricula and train small local QLoRA adapters."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import time
from types import SimpleNamespace

from .depth import MODEL, REVISION, read_tasks, render_depth
from .environment import parse_object
from .pilot import frontier, grade_response


def load_run(path):
    path = Path(path)
    if not json.loads((path / "summary.json").read_text())["complete"]:
        raise ValueError("Incomplete collection run")
    tasks = {row["task_id"]: (row, task) for row, task in read_tasks(path / "tasks.jsonl")}
    grouped = defaultdict(list)
    identities = set()
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
    tasks, grouped = load_run(args.candidates)
    valid_tasks, valid_grouped = load_run(args.validation)
    train_families = {t[0]["family_id"] for t in tasks.values()}
    if train_families & {t[0]["family_id"] for t in valid_tasks.values()}:
        raise ValueError("Training/validation family overlap")
    for filename in ("test.jsonl", "test_ood.jsonl"):
        test_families = {r["family_id"] for r, _ in read_tasks(args.splits / filename)}
        if test_families & (train_families | {t[0]["family_id"] for t in valid_tasks.values()}):
            raise ValueError("Evaluation leakage")
    eligible, scores, supervision = defaultdict(list), {}, {}
    for tid, rows in grouped.items():
        if {r["attempt"] for r in rows} != set(range(16)):
            raise ValueError("Need eight scoring and eight supervision attempts per candidate")
        scoring = [r for r in rows if r["attempt"] < 8]
        answers = [r for r in rows if r["attempt"] >= 8 and r["success"]]
        p = sum(r["success"] for r in scoring) / 8
        scores[tid] = {"score_rate": p, "frontier": frontier(p), "independent_rate": len(answers) / 8}
        if answers:
            eligible[tasks[tid][0]["depth"]].append(tid)
            supervision[tid] = sorted(answers, key=lambda x: x["attempt"])[0]
    ranked = {d: [tid for tid in ids if scores[tid]["frontier"] > 0] for d, ids in eligible.items()}
    capacity = sum(len(ids) for ids in ranked.values())
    limit = next((n for n in (32, 20, 16) if n <= capacity), 0)
    if not limit or len([d for d, ids in ranked.items() if ids]) < 2:
        raise ValueError(f"Insufficient independent curriculum: capacities { {d:len(v) for d,v in ranked.items()} }")
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
        successes = sorted([r for r in rows if r["success"]], key=lambda r: r["attempt"])
        if successes:
            valid_examples.append(example(tid, successes[0], valid_tasks[tid][0]))
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
    print(json.dumps(manifest, indent=2))


class DirectDataset:
    def __init__(self, rows, tokenizer):
        self.rows, self.tokenizer = rows, tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def process(self, row):
        prefix = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        prompt_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(row["answer"], add_special_tokens=False)
        return prompt_ids + answer_ids + [self.tokenizer.eos_token_id], len(prompt_ids)


def train_adapter(args):
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load
    from mlx_lm.lora import CONFIG_DEFAULTS, train_model
    from mlx_lm.tuner.callbacks import TrainingCallback
    from huggingface_hub import snapshot_download
    manifest = json.loads((args.data.parent / "manifest.json").read_text())
    for path, expected in manifest["files_sha256"].items():
        if hashlib.sha256((args.data.parent / path).read_bytes()).hexdigest() != expected:
            raise ValueError("Dataset manifest changed")
    args.out.mkdir(parents=True, exist_ok=False)
    model_path = snapshot_download(MODEL, revision=REVISION)
    model, tokenizer = load(model_path)
    datasets = [DirectDataset([json.loads(x) for x in (args.data / f"{name}.jsonl").read_text().splitlines()], tokenizer)
                for name in ("train", "valid")]
    lengths = [[(len(ds.process(row)[0]), len(ds.process(row)[0])-ds.process(row)[1]) for row in ds.rows] for ds in datasets]
    if max(n for group in lengths for n, _ in group) > 512:
        raise ValueError("Sequence cap would truncate a training example")
    options = deepcopy(CONFIG_DEFAULTS)
    options.update(model=model_path, data=str(args.data), train=True, adapter_path=str(args.out),
                   num_layers=8, batch_size=4, iters=80, val_batches=-1, learning_rate=1e-4,
                   steps_per_report=10, steps_per_eval=80, save_every=80, max_seq_length=512,
                   grad_checkpoint=True, seed=71, mask_prompt=True,
                   lora_parameters={"rank": 8, "dropout": 0.0, "scale": 16.0})
    metadata = {"base_model": MODEL, "revision": REVISION, "training": True, "method": "verified filtered self-training, QLoRA",
                "train_tasks": len(datasets[0]), "validation_tasks": len(datasets[1]),
                "train_tokens_per_epoch": sum(n for n, _ in lengths[0]),
                "target_tokens_per_epoch": sum(n for _, n in lengths[0]),
                "dataset_manifest_sha256": hashlib.sha256((args.data.parent / "manifest.json").read_bytes()).hexdigest(),
                "options": options, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.out / "run-manifest.json").write_text(json.dumps(metadata, indent=2))
    (args.out / "training-source.py").write_bytes(Path(__file__).read_bytes())

    class LogCallback(TrainingCallback):
        def write(self, kind, info):
            loss = info.get("train_loss", info.get("val_loss"))
            if not math.isfinite(loss):
                raise FloatingPointError("Non-finite loss; stopping")
            with (args.out / "metrics.jsonl").open("a") as f:
                f.write(json.dumps({"kind": kind, **info}) + "\n")

        def on_train_loss_report(self, info):
            self.write("train", info)

        def on_val_loss_report(self, info):
            self.write("validation", info)

    np.random.seed(71)
    started = time.monotonic()
    train_model(SimpleNamespace(**options), model, *datasets, training_callback=LogCallback())
    arrays = mx.load(str(args.out / "adapters.safetensors"))
    b_weights = [v for k, v in arrays.items() if k.endswith("lora_b")]
    if not b_weights or not any(mx.any(v != 0).item() for v in b_weights):
        raise ValueError("Adapter did not change from zero-initialized B weights")
    result = {"complete": True, "updates": 80, "elapsed_seconds": time.monotonic() - started,
              "adapter_sha256": hashlib.sha256((args.out / "adapters.safetensors").read_bytes()).hexdigest(),
              "adapter_tensors": len(arrays), "nonzero_lora_b": True, "cloud_spend_usd": 0}
    (args.out / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--candidates", type=Path, required=True)
    prep.add_argument("--validation", type=Path, required=True)
    prep.add_argument("--splits", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    train = sub.add_parser("train")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        train_adapter(args)


if __name__ == "__main__":
    main()
