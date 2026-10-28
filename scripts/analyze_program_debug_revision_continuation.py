"""CPU-only audit of the linked fixed-ten-group continuation and evaluation.

The original throughput failure stays immutable. Inherited artifacts retain
their original manifest identities; new operational phases have separate
manifests. Neural evidence is validated from saved vectors and CPU tensors,
never by loading a language model or requesting fresh controller games.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_program_debug_revision_rl as original

previous, calibration, debug, gpu = original.previous, original.calibration, original.debug, original.gpu
read, sha, same, require, safe_path, value_sha = (original.read, original.sha,
    original.same, original.require, original.safe_path, original.value_sha)
KIND = 'linked_mixed_native_revision_continuation'
OPERATIONAL_CONFIG = {'deadline', 'worker_seconds'}
PHASES = {'training': (18, 110, 61200), 'evaluation': (10, 65, 35100)}
DESIGN_SHA256 = '4a51cca4fc6a0221d90ce95e109a35cd644e45e130eb756c61229fe0245c3442'


def checked_path(root, relative):
    """Scientific inventories permit regular, relative, non-aliased files only."""
    rel = Path(relative)
    require(isinstance(relative, str) and not rel.is_absolute() and relative == rel.as_posix()
            and '..' not in rel.parts and '.' not in rel.parts, 'Unsafe artifact member')
    path = safe_path(root, relative)
    cursor = Path(root)
    for part in rel.parts:
        cursor /= part
        require(not cursor.is_symlink(), 'Symlink in scientific evidence: ' + relative)
    require(path.is_file(), 'Missing scientific evidence: ' + relative)
    return path


def validate_scientific_config(config, parent, phase):
    require(phase in PHASES, 'Unknown continuation phase')
    require(set(config) == set(parent), 'Continuation config key set differs')
    same({k: v for k, v in config.items() if k not in OPERATIONAL_CONFIG},
         {k: v for k, v in parent.items() if k not in OPERATIONAL_CONFIG},
         'Continuation changed scientific settings')
    require(config['worker_seconds'] == PHASES[phase][2], 'Phase worker allowance differs')
    datetime.fromisoformat(config['deadline'])


def excluded_inherited(relative):
    return (relative in {'runtime.json', 'endpoint-selection.json'}
            or relative.startswith(('smoke-', 'admissions/', 'continuation-smoke-', 'continuation/')))


def validate_inherited(run, parent, declared, origins, journal):
    """Accept only the complete explicit inherited namespace, without rebasing."""
    parent_status = read(parent / 'status.json')
    expected = {p: d for p, d in parent_status['artifacts'].items() if not excluded_inherited(p)}
    same(declared, expected, 'Inherited journal inventory differs')
    for relative, digest in expected.items():
        require(journal.get(relative) == digest, 'Inherited artifact overwritten: ' + relative)
        require(sha(checked_path(parent, relative)) == digest
                == sha(checked_path(run, relative)), 'Inherited bytes differ: ' + relative)
    labels = {Path(p).parent.name for p in expected
              if p.startswith('checkpoints/') and p.endswith('/metadata.json')}
    require(set(origins) == labels, 'Inherited checkpoint origin mapping differs')
    for label in sorted(labels):
        relative = f'checkpoints/{label}/metadata.json'
        metadata = read(run / relative)
        require(metadata['manifest_sha256'] == origins[label],
                'Inherited checkpoint was rewritten under another manifest')
        for name, digest in metadata['files'].items():
            require(declared.get(f'checkpoints/{label}/{name}') == digest,
                    'Inherited checkpoint file lacks original journal binding')
    return labels


def no_evaluation(artifacts):
    require(not any(p.startswith(('evaluation/', 'generations/eval-', 'batches/eval-', 'feedback/eval-'))
                    for p in artifacts), 'Evaluation observed before its separate phase')


def validate_parent_terminal(phase, manifest, status, audit):
    require(audit['integrity_passed'] is True, 'Parent independent audit did not pass')
    require(audit['manifest_sha256'] == status['manifest_sha256'], 'Parent audit manifest differs')
    no_evaluation(status['artifacts'])
    labels = {Path(p).parent.name for p in status['artifacts']
              if p.startswith('checkpoints/') and p.endswith('/metadata.json')}
    last = 3 if phase == 'training' else 10
    require(labels == {'base'} | {f'cycle{i}' for i in range(1, last + 1)},
            'Parent checkpoint prefix is not the prescribed boundary')
    require(audit['planned_comparisons_available'] is False,
            'Parent falsely claims a completed comparison')
    if phase == 'training':
        require(manifest['kind'] == 'native_nonthinking_revision_rl'
                and status['state'] == 'throughput_failed' and audit['recorded_state'] == 'throughput_failed'
                and audit['planned_study_complete'] is False and audit['selected_endpoint'] is None,
                'Training requires original audited throughput failure')
    else:
        require(manifest['kind'] == KIND and manifest['phase'] == 'training'
                and status['state'] == 'training_complete' and audit['training_complete'] is True
                and audit['selected_endpoint'] == 10 and audit['total_attempted_groups'] == 10,
                'Evaluation requires audited fixed-cycle10 training completion')
    groups = audit['training_groups']
    require([r['cycle'] for r in groups if r['checkpoint_committed']] == list(range(1, last + 1)),
            'Parent audit lacks complete prescribed training groups')


def validate_counts(phase, attempted, actual, inherited_attempted, inherited_actual):
    require(type(attempted) is int and type(actual) is int and type(inherited_actual) is int,
            'Nonintegral optimizer/group counts')
    require(inherited_attempted == (3 if phase == 'training' else 10)
            and inherited_attempted <= attempted <= 10
            and 0 <= inherited_actual <= inherited_attempted
            and inherited_actual <= actual <= attempted, 'Invalid inherited/new update counts')
    if phase == 'evaluation':
        require(attempted == 10 and actual == inherited_actual, 'Evaluation performed training')
    return {'total_attempted_groups': attempted, 'inherited_attempted_groups': inherited_attempted,
            'new_attempted_groups': attempted - inherited_attempted,
            'genuine_RL_optimizer_steps': actual, 'genuine_optimizer_steps': actual,
            'inherited_optimizer_steps': inherited_actual, 'new_optimizer_steps': actual - inherited_actual}

