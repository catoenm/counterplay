"""Prepare a distinct deployment with an earlier, actual-guard-compatible deadline.

No provider calls or reservations. The prior failed deployment and all frozen
scientific/lifecycle sources remain unchanged. The 25h estimate is an upper
envelope; the immutable actual deadline is at most 23h55m from preparation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import secrets
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import runpod_revision_rl as controller
from scripts.cloud.cardinal_guard import Guard

base = controller.base


def prepare(folder, previous, ssh_public_key):
    folder, previous = Path(folder), Path(previous)
    if folder.exists():
        raise FileExistsError('New deployment already exists')
    prior = base.load(previous/'plan.json')
    state = base.load(previous/'state.json')
    failure = base.load(previous/'guard-startup-failure.json')
    review = base.load(previous/'terminated-liability-review.json')
    if (state['plan_sha256'] != base.digest(previous/'plan.json')
            or set(state['roles']) != {'guard'}
            or state['roles']['guard']['state'] != 'terminated_confirmed'
            or failure.get('kind') != 'definite_guard_startup_configuration_failure'
            or not failure.get('guard_404_confirmed')
            or failure.get('gpu_created') is not False
            or failure.get('training_wrapper_launched') is not False
            or review.get('passed') is not True
            or review['intent'] != prior['intent']
            or review['cleanup']['fresh_exact_guard_http_status'] != 404
            or review['cleanup']['matching_current_intent_pods'] != 0):
        raise ValueError('Previous definite failure and cleanup are not established')
    public = Path(ssh_public_key).read_text().strip()
    if not public.startswith('ssh-ed25519 ') or '\n' in public:
        raise ValueError('Expected one project Ed25519 public key')
    intent = uuid.uuid4().hex
    created = time.time()
    deadline = float(math.floor(min(prior['deadline_unix'], created + 23*3600 + 55*60)))
    if deadline <= created or deadline >= prior['deadline_unix']:
        raise ValueError('Expected a strictly earlier future deadline')
    private = ROOT/'.secrets/cardinal-gpu'/intent
    secret_file = private/'guard.json'
    guard = ROOT/'scripts/cloud/cardinal_guard.py'
    plan = {'intent': intent, 'project': base.PROJECT, 'created_at': base.now(),
        'deadline_unix': deadline, 'maximum_hours': 25.0,
        'reservation_usd': controller.RESERVATION,
        'estimate_usd': (5.29+.14+170*.10/(28*24))*25,
        'gpu_id': 'NVIDIA H200', 'maximum_gpu_rate_usd_hour': 5.29,
        'cpu_id': 'cpu5c', 'cpu_vcpu': 4, 'maximum_cpu_rate_usd_hour': .14,
        'gpu_disk_gb': 160, 'guard_disk_gb': 10,
        'gpu_image': 'runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404',
        'guard_image': 'python:3.12-slim', 'ssh_public_key': public,
        'guard_source_sha256': base.digest(guard), 'guard_secret_path': str(secret_file),
        'inherited_controller_sha256': base.digest(Path(base.__file__)),
        'revision_rl_controller_sha256': base.digest(Path(controller.__file__)),
        'guard_compatible_preparer_sha256': base.digest(Path(__file__)),
        'prior_failed_deployment_plan_sha256': base.digest(previous/'plan.json'),
        'prior_failed_deployment': str(previous),
        'deadline_reason': 'New initial deadline below actual frozen guard24h admission and earlier than prior failed deployment; 25h is only a conservative reservation envelope.',
        'cleanup': prior['cleanup']}
    controller.validate(plan)
    # Exercise the actual frozen constructor before any disk mutation or rental.
    # Constructor performs no provider requests; None prevents accidental API use.
    Guard(None, intent, 'offline-preparation-check', deadline, 'x'*40,
          plan['guard_source_sha256'], clock=lambda: created)
    folder.mkdir(parents=True)
    private.mkdir(parents=True, mode=0o700)
    os.chmod(private, 0o700)
    fd = os.open(secret_file, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as handle:
        json.dump({'token': secrets.token_urlsafe(40)}, handle)
    base.atomic_json(folder/'plan.json', plan)
    base.atomic_json(folder/'state.json', {'plan_sha256': base.digest(folder/'plan.json'),
        'created_at': base.now(), 'reservation_applied': False, 'roles': {}, 'events': []})
    return {'deployment': str(folder), 'intent': intent, 'deadline_unix': deadline,
        'reservation_usd': controller.RESERVATION, 'actual_guard_constructor_passed': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--ssh-public-key', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.deployment, args.previous, args.ssh_public_key), indent=2))
