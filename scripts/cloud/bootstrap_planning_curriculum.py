"""Explicit deadline-bounded environment/model staging; never launches training."""
from __future__ import annotations
import argparse
from datetime import datetime
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.cloud.train_program_rl import atomic_json



def stage_deadline(deployment):
    if deployment.get('phase')!='planning_curriculum':raise ValueError('Planning deployment required')
    created=datetime.fromisoformat(deployment['created_at'])
    if created.tzinfo is None:raise ValueError('Timezone-aware preparation timestamp required')
    rates=[deployment[k] for k in ('gpu_hourly_rate','guard_hourly_rate','storage_hourly_rate')]
    other=deployment['other_reserved_cost_usd']
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in rates+[other]) or sum(rates)<=0:
        raise ValueError('Finite conservative tariffs required')
    if deployment['stage_caps_usd']!={'admission':15.,'comparison':65.,'protected':10.}:
        raise ValueError('Stage allowances differ')
    return min(deployment['deadline_unix']-3600,created.timestamp()+(15-other)/sum(rates)*3600)


def admission(config):
    deadline=config['deadline_unix']
    if type(deadline) not in (int,float) or not math.isfinite(deadline) or time.time()>=deadline:
        raise TimeoutError('Immutable mixed arena deadline exhausted or invalid')


def approved_files(metadata,repo,revision):
    if metadata.get('model')!=repo or metadata.get('revision')!=revision:
        raise ValueError('Authoritative original model identity differs')
    names=[row['name'] for row in metadata['files']]
    if (not names or len(set(names))!=len(names) or any(Path(n).name!=n for n in names)
            or not {'config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json'}<=set(names)):
        raise ValueError('Unsafe or incomplete model filename inventory')
    return names


def protocol():
    from selfplay.planning_curriculum import experiment
    return experiment


def stage(args, manifest):
    admission({'deadline_unix':stage_deadline(json.loads((args.run/'deployment.json').read_text()))})
    if sys.prefix == sys.base_prefix:
        raise ValueError('Create and use the reviewed --system-site-packages virtual environment before staging')
    runtime = protocol()
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--disable-pip-version-check',
        '-r', str(ROOT / 'scripts/cloud/arena-mixed-scalar-requirements.txt')], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    import torch
    if torch.__version__.split('+')[0] != '2.8.0' or torch.version.cuda != '12.8':
        raise ValueError('Use the pinned torch2.8.0 CUDA12.8 image; no silent Torch replacement')
    actual = {name: importlib.metadata.version(name).split('+')[0] for name in runtime.VERSIONS}
    if actual != runtime.VERSIONS:
        raise ValueError('Exact numerical dependency versions differ')
    from transformers import AutoModelForCausalLM, Qwen3ForCausalLM
    from huggingface_hub import snapshot_download
    from peft import prepare_model_for_kbit_training
    admission({'deadline_unix':stage_deadline(json.loads((args.run/'deployment.json').read_text()))})
    metadata=json.loads((args.run/'inputs/model-info.json').read_text())
    names=approved_files(metadata,runtime.MODEL_REPO,runtime.MODEL_REVISION)
    snapshot_download(repo_id=runtime.MODEL_REPO, revision=runtime.MODEL_REVISION,
        allow_patterns=names,local_dir=args.model_dir,max_workers=8)
    admission({'deadline_unix':stage_deadline(json.loads((args.run/'deployment.json').read_text()))})
    index = runtime.index_model(args.model_dir, args.run / 'inputs/model-info.json')
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze', '--all'], check=True, capture_output=True, text=True).stdout
    target = args.run / 'staging-report.json'
    report = {'python': sys.version, 'versions': actual, 'torch_cuda': torch.version.cuda,
        'model_index': index, 'pip_freeze': freeze, 'cuda_model_validation': 'Not performed; no weight loading or model computation'}
    admission({'deadline_unix':stage_deadline(json.loads((args.run/'deployment.json').read_text()))})
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
    manifest, _ = protocol().verify_run(args.run)
    admission({'deadline_unix':stage_deadline(json.loads((args.run/'deployment.json').read_text()))})
    if args._worker:
        return stage(args, manifest)
    deadline = stage_deadline(json.loads((args.run / 'deployment.json').read_text()))
    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--run', str(args.run.resolve()),
        '--model-dir', str(args.model_dir.resolve()), '--_worker'], start_new_session=True, cwd=ROOT,
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PIP_NO_CACHE_DIR':'1'})
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
