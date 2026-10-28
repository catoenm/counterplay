"""Train a local QLoRA adapter on verified model-generated final plans."""
import argparse
from copy import deepcopy
from functools import partial
import hashlib
import json
import math
import importlib.metadata
from pathlib import Path
import time
from types import SimpleNamespace

from .curriculum import check_splits
from .depth import MODEL, REVISION


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


def completion_loss(model, batch, lengths):
    """Predict assistant tokens through EOS, excluding all padding positions."""
    import mlx.core as mx
    import mlx.nn as nn
    targets = batch[:, 1:]
    logits = model(batch[:, :-1])
    positions = mx.arange(1, targets.shape[1] + 1)
    mask = (positions >= lengths[:, 0:1]) & (positions < lengths[:, 1:])
    count = mask.sum()
    loss = (nn.losses.cross_entropy(logits, targets) * mask).astype(mx.float32).sum() / count
    return loss, count


def train_adapter(args):
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load
    from mlx_lm.lora import CONFIG_DEFAULTS, train_model
    import mlx_lm.lora as lora_module
    from mlx_lm.tuner.callbacks import TrainingCallback
    from mlx_lm.tuner.datasets import CacheDataset
    from mlx_lm.tuner.trainer import evaluate
    from huggingface_hub import snapshot_download
    manifest = json.loads((args.data.parent / "manifest.json").read_text())
    updates, seed = args.iters, args.seed
    if updates < 1 or seed < 0:
        raise ValueError("Invalid training schedule")
    if "updates" in manifest and (updates != manifest["updates"] or seed != manifest["seed"]):
        raise ValueError("Training schedule differs from dataset protocol")
    protocol = args.protocol
    if manifest.get("protocol_sha256") and hashlib.sha256(protocol.read_bytes()).hexdigest() != manifest["protocol_sha256"]:
        raise ValueError("Training protocol changed")
    split_dir = Path(manifest["split_dir"])
    check_splits(split_dir)
    if hashlib.sha256((split_dir / "manifest.json").read_bytes()).hexdigest() != manifest["split_manifest_sha256"]:
        raise ValueError("Task manifest changed")
    for path, expected in manifest["files_sha256"].items():
        if hashlib.sha256((args.data.parent / path).read_bytes()).hexdigest() != expected:
            raise ValueError("Dataset manifest changed")
    args.out.mkdir(parents=True, exist_ok=False)
    model_path = snapshot_download(MODEL, revision=REVISION)
    model, tokenizer = load(model_path)
    datasets = [DirectDataset([json.loads(x) for x in (args.data / f"{name}.jsonl").read_text().splitlines()], tokenizer)
                for name in ("train", "valid")]
    lengths = []
    for ds in datasets:
        lengths.append([])
        for row in ds.rows:
            tokens, offset = ds.process(row)
            lengths[-1].append((len(tokens), len(tokens)-offset))
    if max(n for group in lengths for n, _ in group) > 512:
        raise ValueError("Sequence cap would truncate an example")
    options = deepcopy(CONFIG_DEFAULTS)
    options.update(model=model_path, data=str(args.data), train=True, adapter_path=str(args.out),
                   num_layers=8, batch_size=4, iters=updates, val_batches=-1, learning_rate=1e-4,
                   steps_per_report=10, steps_per_eval=updates, save_every=updates, max_seq_length=512,
                   grad_checkpoint=True, seed=seed, mask_prompt=True,
                   lora_parameters={"rank": 8, "dropout": 0.0, "scale": 16.0})
    metadata = {"base_model": MODEL, "revision": REVISION, "training": True,
                "method": manifest.get("method", "verified filtered self-training, QLoRA"),
                "train_tasks": len({r["task_id"] for r in datasets[0].rows}),
                "validation_tasks": len({r["task_id"] for r in datasets[1].rows}),
                "train_examples": len(datasets[0]), "validation_examples": len(datasets[1]),
                "train_tokens_per_epoch": sum(n for n, _ in lengths[0]),
                "target_tokens_per_epoch": sum(n for _, n in lengths[0]),
                "dataset_manifest_sha256": hashlib.sha256((args.data.parent / "manifest.json").read_bytes()).hexdigest(),
                "options": options, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    metadata["runtime_versions"] = {name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm", "transformers")}
    metadata["training_example_kinds"] = {}
    for row, (total, target) in zip(datasets[0].rows, lengths[0]):
        stats = metadata["training_example_kinds"].setdefault(row.get("kind", "plan"),
                                                             {"examples": 0, "tokens": 0, "target_tokens": 0})
        stats["examples"] += 1
        stats["tokens"] += total
        stats["target_tokens"] += target
    metadata["protocol_sha256"] = hashlib.sha256(protocol.read_bytes()).hexdigest()
    (args.out / "run-manifest.json").write_text(json.dumps(metadata, indent=2))
    (args.out / "training-source.py").write_bytes(Path(__file__).read_bytes())
    (args.out / "protocol.md").write_bytes(protocol.read_bytes())

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

    np.random.seed(seed)
    started = time.monotonic()
    initial_validation_loss = evaluate(model, CacheDataset(datasets[1]), 1, -1, max_seq_length=512, loss=completion_loss)
    original_train = lora_module.train
    lora_module.train = partial(original_train, loss=completion_loss)
    try:
        train_model(SimpleNamespace(**options), model, *datasets, training_callback=LogCallback())
    finally:
        lora_module.train = original_train
    final_validation_loss = evaluate(model, CacheDataset(datasets[1]), 1, -1, max_seq_length=512, loss=completion_loss)
    arrays = mx.load(str(args.out / "adapters.safetensors"))
    b_weights = [v for k, v in arrays.items() if k.endswith("lora_b")]
    if not b_weights or not any(mx.any(v != 0).item() for v in b_weights):
        raise ValueError("Adapter did not change from zero-initialized B weights")
    if not all(mx.all(mx.isfinite(v)).item() for v in arrays.values()):
        raise ValueError("Non-finite adapter tensor")
    reports = [json.loads(line) for line in (args.out / "metrics.jsonl").read_text().splitlines()]
    last = [r for r in reports if r["kind"] == "train"][-1]
    if last["iteration"] != updates:
        raise ValueError("Training did not finish the prescribed update count")
    if updates * 4 % len(datasets[0]) == 0:
        expected_tokens = metadata["target_tokens_per_epoch"] * (updates * 4 // len(datasets[0]))
        if last["trained_tokens"] != expected_tokens:
            raise ValueError("Supervised token count differs from full dataset passes")
    result = {"complete": True, "updates": updates, "elapsed_seconds": time.monotonic() - started,
              "adapter_sha256": hashlib.sha256((args.out / "adapters.safetensors").read_bytes()).hexdigest(),
              "adapter_tensors": len(arrays), "nonzero_lora_b": True, "cloud_spend_usd": 0,
              "trained_target_tokens": last["trained_tokens"], "peak_memory_gb": last["peak_memory"]}
    result.update(initial_validation_loss=initial_validation_loss, final_validation_loss=final_validation_loss)
    (args.out / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--protocol", type=Path, default=Path("docs/depth-training-protocol.md"))
    train_adapter(ap.parse_args())
