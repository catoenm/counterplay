"""Explicit deadline-bounded environment/model staging; never launches training."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from selfplay.program_cardinal_gpu import MODEL_REPO, MODEL_REVISION, VERSIONS, index_model, verify_run
from scripts.cloud.train_program_rl import atomic_json, admission


def stage(args, manifest):
    admission(manifest['config'])
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
        '-r', str(ROOT / 'scripts/cloud/cardinal-requirements.txt')], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    import torch
    if torch.__version__.split('+')[0] != '2.8.0' or torch.version.cuda != '12.8':
        raise ValueError('Use the pinned torch2.8.0 CUDA12.8 image; no silent Torch replacement')
    actual = {name: importlib.metadata.version(name).split('+')[0] for name in VERSIONS}
    if actual != VERSIONS:
        raise ValueError('Exact numerical dependency versions differ')
    from transformers import AutoModelForImageTextToText, Qwen3_5ForConditionalGeneration
    from huggingface_hub import snapshot_download
    from peft import prepare_model_for_kbit_training
    admission(manifest['config'])
    snapshot_download(repo_id=MODEL_REPO, revision=MODEL_REVISION, local_dir=args.model_dir, max_workers=8)
    admission(manifest['config'])
    index = index_model(args.model_dir, args.run / 'inputs/model-info.json')
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze', '--all'], check=True, capture_output=True, text=True).stdout
    target = args.run / 'staging-report.json'
    report = {'python': sys.version, 'versions': actual, 'torch_cuda': torch.version.cuda,
        'model_index': index, 'pip_freeze': freeze, 'cuda_model_validation': 'Not performed; no weight loading or model computation'}
    admission(manifest['config'])
    if target.exists():
        if json.loads(target.read_text()) != report:
            raise ValueError('Existing staging report differs')
    else:
        atomic_json(target, report)
    print(json.dumps({'staged': True, 'model_dir': str(args.model_dir), 'report': str(target)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    manifest, _ = verify_run(args.run)
    admission(manifest['config'])
    if args._worker:
        return stage(args, manifest)
    deadline = json.loads((args.run / 'deployment.json').read_text())['deadline_unix']
    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--run', str(args.run.resolve()),
        '--model-dir', str(args.model_dir.resolve()), '--_worker'], start_new_session=True, cwd=ROOT)
    try:
        code = child.wait(timeout=max(.1, deadline-time.time()))
        if code:
            raise RuntimeError(f'Staging worker exited{code}')
    except BaseException:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)
        raise


if __name__ == '__main__':
    main()
