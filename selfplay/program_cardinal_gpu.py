"""Batched, budget-bounded CUDA cardinal-controller self-play RL.

Preparation and model indexing are CPU-only. The run command requires staged
original HF weights and holds both neural/run locks. All source samples, pool
snapshots, updates, and final evaluation draws are immutable journal artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

from scripts.cloud import train_program_rl as common
from selfplay import program_study as journal
from selfplay import program_cardinal_cycle as cardinal
from selfplay.program_cardinal import cardinal_model_spec
from selfplay.program_cardinal_cuda import CardinalCudaPolicy, checkpoint_state

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / 'runs/local-neural.lock'
MODEL_REPO = 'Qwen/Qwen3.8-27B'
MODEL_REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
VERSIONS = {'torch': '2.8.0', 'transformers': '5.20.2', 'peft': '0.21.2',
            'bitsandbytes': '0.50.2', 'accelerate': '1.15.0', 'safetensors': '0.8.0'}
SOURCES = list(dict.fromkeys(cardinal.SOURCES + ['selfplay/program_cardinal_cuda.py',
    'selfplay/program_cardinal_gpu.py', 'scripts/cloud/cardinal-requirements.txt',
    'scripts/cloud/bootstrap_cardinal.py']))
FILES = ('adapter.safetensors', 'optimizer.pt')


def object_hash(value):
    return hashlib.sha256(common.canonical(value).encode()).hexdigest()


def public_deployment(value):
    keys = ('intent', 'project', 'created_at', 'deadline_unix', 'maximum_hours',
            'reservation_usd', 'estimate_usd', 'gpu_id', 'maximum_gpu_rate_usd_hour',
            'maximum_cpu_rate_usd_hour', 'gpu_disk_gb', 'guard_disk_gb', 'gpu_image',
            'guard_source_sha256', 'cleanup')
    result = {key: value[key] for key in keys}
    if (result['project'] != 'cardinal-gpu' or not 0 < result['maximum_hours'] <= 17
            or not 0 < result['reservation_usd'] <= 80
            or not math.isfinite(result['deadline_unix'])
            or not 0 < result['maximum_gpu_rate_usd_hour'] <= 5.29
            or not 0 < result['estimate_usd'] <= result['reservation_usd']):
        raise ValueError('Deployment exceeds the reserved envelope')
    return result


def config(deployment):
    return {**common.DEFAULTS, 'kind': 'cardinal_gpu_league_rl', 'condition': 'clean-cardinal-gpu-v1',
        'model_repo': MODEL_REPO, 'model_revision': MODEL_REVISION, 'loader': 'image_text_to_text',
        'backend': 'cuda', 'algorithm': 'full_completion_RLOO', 'iterations': 100,
        'endpoint_choices': [50, 100], 'group_size': 8, 'batch_size': 8,
        'learning_rate': 1e-5, 'max_new_tokens': 2048, 'max_context_tokens': 4096,
        'sample_seconds': 300, 'worker_seconds': int(deployment['maximum_hours'] * 3600),
        'deadline': datetime.fromtimestamp(deployment['deadline_unix'], timezone.utc).isoformat(),
        'capability_samples': 8, 'minimum_productive_programs': 4,
        'minimum_productive_distinct_ir': 2, 'minimum_positive_calibration_games': 12,
        'opponent_count': 2, 'opponent_attempt_cap': 8, 'eval_samples': 32,
        'map_seeds_per_group': 2, 'max_rounds': 48, 'league_refresh_every': 5,
        'maximum_model_opponents': 6, 'profile_attempt_cap': 3, 'timing_safety_factor': 1.5,
        'audit_export_reserve_seconds': 1800, 'evaluation_batch_reserve_seconds': 2400,
        'train_seed_start': 8200000, 'calibration_seed_start': 8100000,
        'holdout_seed_start': 8300000,
        'reward': 'Mean relative return; invalid authorship or any candidate-seat interpreter fault=-1',
        'scope': 'Fresh current-learner controller gradients versus a bounded historical controller league; frozen final evaluation roster'}


def schedule():
    return {'calibration_maps': list(range(8100000, 8100006)),
        'train_maps': {str(i): [8200000+i*100+j for j in range(2)] for i in range(1, 101)},
        'holdout_maps': list(range(8300000, 8300006)),
        'capability_generation_seeds': list(range(10100000, 10100008)),
        'train_generation_seeds': {str(i): [10200000+i*100+j for j in range(8)] for i in range(1, 101)},
        'eval_generation_seeds': list(range(10300000, 10300032))}


def plan(deployment):
    cfg = config(deployment)
    return {'kind': 'cardinal_gpu_league_rl_plan', 'config': cfg, 'schedule': schedule(),
        'prompt': 'Clean cardinal specification; no successful or failed API controller and no repair feedback.',
        'model': 'Original pinned HF weights, newly initialized GPU NF4 FP32 LoRA baseline; independent of local MLX adapter.',
        'capability_gate': {'draws': 8, 'games_per_compiled_program': 24,
            'productive': 'Fault-free candidate seat across24 calibration games and positive banking in at least12.',
            'required_productive': 4, 'required_distinct_ir': 2, 'select': 'First two qualifying distinct IRs.',
            'all_draws_retained': True, 'failure': 'Stop before training; no resampling or gate relaxation.'},
        'training': {'attempted_groups': [50, 100], 'draws_per_group': 8,
            'maps_per_group': 2, 'opponents_per_group': 4, 'both_seats': True,
            'algorithm': 'RLOO advantages, summed full-completion log probabilities including actual EOS; one accumulated AdamW step per nonzero group.',
            'zero_signal': 'Retain group and copy exact adapter/optimizer; zero genuine steps; no retry.',
            'initial_optimizer': 'Fresh zero-step AdamW at declared1e-5; no local adapter or optimizer inherited.',
            'numerics': 'FP32, full-softmax, token0.002/sequence0.02 likelihood gates in eval and gradient modes; no tolerance relaxation.',
            'profile': 'At most3 prescribed groups to observe first genuine update; choose endpoint using only elapsed time and immutable budget. No heldout inspection.',
            'no_invalid_streak_stop_after_profile': True},
        'league': {'maximum_model_controllers': 6, 'initial_two_retained': True,
            'refresh': 'Every5 completed groups, first2 distinct runtime-valid controllers banking in at least half of prescribed training games, selected in fixed draw order.',
            'pruning': 'Keep initial2 plus newest4 distinct additions; never delete source/checkpoint files.',
            'match_snapshot': 'Nearest/value plus newest model controller and deterministic rotating different historical model, frozen before sampling/scoring group.',
            'evaluation_pool': 'Initial two capability controllers plus nearest/value, frozen throughout.'},
        'evaluation': {'endpoints': ['base', 'timing-selected cycle50 or cycle100'],
            'draws_per_endpoint': 32, 'paired_generation_seeds': True, 'holdout_maps': 6,
            'opponents': 4, 'both_seats': True, 'games_per_compiled_program': 48,
            'seal_all64_before_scoring': True,
            'primary': 'Banked-point yield over all32 prescribed program draws; invalid authorship is zero yield/no fabricated game, missing is not zero.',
            'runtime_validity_and_faults': 'Report separately; relative outcomes and all-rest frequency are descriptive.',
            'incomplete': 'No final learned-improvement comparison unless fixed endpoint and all64 draws/2cells complete.',
            'uncertainty': 'One training seed. Paired controller-generation clusters n32, not individual games; exploratory intervals only.'},
        'resources': {'reservation_usd': deployment['reservation_usd'], 'cumulative_cap_usd': 100,
            'maximum_authoring_draws': {'50': 472, '100': 872},
            'projection': '1.5×maximum measured profile-group time per remaining group, plus2400s for8 evaluation batches and1800s audit/export.',
            'hard_stop': 'Earlier of immutable deployment deadline and nonrenewable worker budget; external guard owns rental deletion.'}}


def validate_configuration(cfg, timetable, deployment):
    if cfg != config(deployment) or timetable != schedule():
        raise ValueError('GPU configuration differs from preregistration')
    families = [timetable['calibration_maps'], timetable['holdout_maps'],
                [s for row in timetable['train_maps'].values() for s in row]]
    if sum(map(len, families)) != len(set(s for row in families for s in row)):
        raise ValueError('Map seed families overlap')


def prepare(run, deployment_path, model_info_path):
    run = Path(run)
    if run.exists():
        raise FileExistsError('New GPU run directory required')
    deployment = public_deployment(common.load(deployment_path))
    metadata = common.load(model_info_path)
    if metadata.get('sha') != MODEL_REVISION or metadata.get('id') != MODEL_REPO:
        raise ValueError('Authoritative HF model metadata differs')
    for relative in SOURCES:
        if not (ROOT / relative).is_file():
            raise ValueError('Missing source before freeze: ' + relative)
    timetable, cfg = schedule(), config(deployment)
    fingerprints = cardinal.map_fingerprints(timetable)
    run.mkdir(parents=True)
    (run / 'inputs').mkdir()
    shutil.copyfile(model_info_path, run / 'inputs/model-info.json')
    shutil.copyfile(cardinal.PRIOR / 'map-fingerprints.json', run / 'inputs/prior-map-fingerprints.json')
    common.atomic_json(run / 'deployment.json', deployment)
    common.atomic_json(run / 'plan.json', plan(deployment))
    common.atomic_json(run / 'schedule.json', timetable)
    common.atomic_json(run / 'map-fingerprints.json', fingerprints)
    (run / 'prompts').mkdir()
    (run / 'prompts/cardinal.txt').write_text(cardinal_model_spec())
    inputs = {p: common.digest(run / p) for p in ('inputs/model-info.json',
        'inputs/prior-map-fingerprints.json', 'deployment.json', 'prompts/cardinal.txt')}
    sources = {}
    for relative in SOURCES:
        target = run / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
        sources[relative] = common.digest(target)
    manifest = {'format_version': 1, 'kind': cfg['kind'], 'created': common.now(), 'config': cfg,
        'plan_sha256': common.digest(run / 'plan.json'), 'schedule_sha256': common.digest(run / 'schedule.json'),
        'map_fingerprints_sha256': common.digest(run / 'map-fingerprints.json'),
        'source_sha256': sources, 'input_sha256': inputs,
        'prompt_sha256': {'cardinal': inputs['prompts/cardinal.txt']},
        'expected_installed_versions': VERSIONS, 'claims_require_complete': True,
        'baseline': 'New GPU NF4 FP32 model and fresh LoRA, not the local MLX checkpoint'}
    common.atomic_json(run / 'manifest.json', manifest)
    common.atomic_json(run / 'status.json', {'manifest_sha256': common.digest(run / 'manifest.json'),
        'artifacts': {}, 'state': 'prepared', 'updated': common.now(), 'stage': 'prepared', 'claims_allowed': False})
    verify_run(run)
    return manifest


def verify_run(run, require_active=True):
    run = Path(run)
    manifest, status = common.load(run / 'manifest.json'), common.load(run / 'status.json')
    deployment = public_deployment(common.load(run / 'deployment.json'))
    if manifest['kind'] != 'cardinal_gpu_league_rl' or status['manifest_sha256'] != common.digest(run / 'manifest.json'):
        raise ValueError('GPU manifest identity differs')
    validate_configuration(manifest['config'], common.load(run / 'schedule.json'), deployment)
    if common.load(run / 'plan.json') != plan(deployment) or set(manifest['source_sha256']) != set(SOURCES):
        raise ValueError('Plan/source coverage differs')
    expected = {'plan.json': manifest['plan_sha256'], 'schedule.json': manifest['schedule_sha256'],
        'map-fingerprints.json': manifest['map_fingerprints_sha256'], **manifest['input_sha256'], **status['artifacts']}
    for relative, digest in expected.items():
        if common.digest(run / relative) != digest:
            raise ValueError('Committed artifact changed: ' + relative)
    for relative, digest in manifest['source_sha256'].items():
        if common.digest(run / 'source' / relative) != digest or (require_active and common.digest(ROOT / relative) != digest):
            raise ValueError('Archived/active source changed: ' + relative)
    return manifest, status


def index_model(model_dir, metadata_path):
    """Verify each staged HF file against authoritative LFS/Git object identities."""
    model_dir = Path(model_dir)
    metadata = common.load(metadata_path)
    if metadata.get('sha') != MODEL_REVISION or metadata.get('id') != MODEL_REPO:
        raise ValueError('Pinned HF revision differs')
    expected = {r['rfilename']: r for r in metadata['siblings']}
    indexed = {}
    for path in sorted(model_dir.rglob('*')):
        if not path.is_file() or '.cache' in path.parts or path.name == 'model-index.json':
            continue
        name = str(path.relative_to(model_dir))
        if name not in expected:
            raise ValueError('Unrecognized staged model file: ' + name)
        row = expected[name]
        if path.stat().st_size != row['size']:
            raise ValueError('Staged file size differs: ' + name)
        sha = common.digest(path)
        if row.get('lfs'):
            if sha != row['lfs']['sha256']:
                raise ValueError('LFS weight hash differs: ' + name)
        else:
            h = hashlib.sha1(b'blob ' + str(path.stat().st_size).encode() + b'\0')
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 << 20), b''):
                    h.update(block)
            if h.hexdigest() != row['blobId']:
                raise ValueError('HF metadata blob differs: ' + name)
        indexed[name] = sha
    required = {r['rfilename'] for r in metadata['siblings'] if r['rfilename'].endswith('.safetensors')}
    required |= {'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors.index.json'}
    if not required <= set(indexed):
        raise ValueError('Incomplete staged pinned snapshot')
    result = {'repo': MODEL_REPO, 'revision': MODEL_REVISION, 'files': indexed,
              'authoritative_metadata_sha256': common.digest(metadata_path)}
    target = model_dir / 'model-index.json'
    if target.exists() and common.load(target) != result:
        raise ValueError('Existing model index differs')
    if not target.exists():
        common.atomic_json(target, result)
    return result


def endpoint_decision(profile_reports, remaining_seconds, cfg):
    if not profile_reports or not any(r['optimizer_steps'] == 1 for r in profile_reports):
        return {'passed': False, 'reason': 'No genuine update available for throughput profiling', 'endpoint': None}
    per_group = max(float(r['elapsed_seconds']) for r in profile_reports)
    if not math.isfinite(per_group) or per_group <= 0 or not math.isfinite(remaining_seconds):
        raise ValueError('Invalid throughput timing')
    reserve = cfg['audit_export_reserve_seconds'] + cfg['evaluation_batch_reserve_seconds']
    projections = {str(n): cfg['timing_safety_factor'] * per_group * (n-len(profile_reports)) + reserve for n in cfg['endpoint_choices']}
    fits = [n for n in cfg['endpoint_choices'] if projections[str(n)] <= remaining_seconds]
    return {'passed': bool(fits), 'endpoint': max(fits) if fits else None,
        'reason': 'Timing-only admission; no heldout or reward-based endpoint selection',
        'remaining_seconds': remaining_seconds, 'profile_groups': len(profile_reports),
        'maximum_measured_group_seconds': per_group, 'safety_factor': cfg['timing_safety_factor'],
        'reserved_final_and_export_seconds': reserve, 'projected_remaining_seconds': projections}


def validate_batch(batch, seeds, cfg):
    if len(batch.get('records', [])) != len(seeds) or len(seeds) != 8:
        raise ValueError('Incomplete eight-row batch')
    first_prompt = batch['records'][0].get('prompt_ids')
    duration = batch.get('timing', {}).get('batch_elapsed_seconds')
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid batch wall time')
    for i, (seed, row) in enumerate(zip(seeds, batch['records'])):
        tokens, logps = row.get('completion_ids'), row.get('sampled_token_logprobs')
        if (row.get('seed') != seed or row.get('sampling_batch_row', i) != i
                or row.get('sampling_batch_size', 8) != 8):
            raise ValueError('Backend batch seed/order differs')
        if (not isinstance(tokens, list) or not tokens or len(tokens) > cfg['max_new_tokens']
                or any(type(t) is not int or t < 0 for t in tokens)
                or not isinstance(logps, list) or len(logps) != len(tokens)
                or any(not isinstance(x, (float, int)) or not math.isfinite(x) or x > 1e-6 for x in logps)
                or row.get('output_tokens') != len(tokens)):
            raise ValueError('Sampled token/log-probability coverage differs')
        eos = row.get('eos_token_ids')
        if (not isinstance(eos, list) or not eos or any(type(t) is not int for t in eos)
                or row.get('finish_reason') not in ('eos', 'time', 'length')
                or any(t in eos for t in tokens[:-1])
                or (tokens[-1] in eos) != (row['finish_reason'] == 'eos')):
            raise ValueError('Sampled EOS termination differs')
        if (not first_prompt or row.get('prompt_ids') != first_prompt
                or row.get('input_tokens') != len(first_prompt)):
            raise ValueError('Shared batch prompt differs')
    if batch.get('timing', {}).get('total_output_tokens') != sum(len(r['completion_ids']) for r in batch['records']):
        raise ValueError('Batch token total differs')


def refresh_roster(initial, prior, rows, generation_hashes, source_metadata_sha256, cycle):
    """Outcome-based opponents, never outcome-based training samples/endpoints."""
    models = [r for r in prior if r['origin'] == 'model']
    seen = {r['ir_sha256'] for r in models}
    added = []
    for row in rows:
        if not row['runtime_valid'] or not row['matches'] or row['positive_banking_games']*2 < len(row['matches']):
            continue
        ir = row['classification']['program']['compiled_ir']
        fingerprint = common.compile_program(ir).digest
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        name = row['generation']
        added.append({'id': name, 'origin': 'model', 'compiled_ir': ir, 'ir_sha256': fingerprint,
            'generation_sha256': generation_hashes[name], 'source_path': f'generations/{name}.json',
            'source_checkpoint': 'base' if cycle == 1 else f'cycle{cycle-1}',
            'source_metadata_sha256': source_metadata_sha256, 'admitted_after_cycle': cycle})
        if len(added) == 2:
            break
    fixed = [r for r in initial if r['origin'] == 'model']
    fixed_ids = {r['id'] for r in fixed}
    tail = [r for r in models + added if r['id'] not in fixed_ids][-4:]
    return fixed + tail, added


class GPUCycle(cardinal.Cycle):
    def __init__(self, run, backend):
        self.run, self.backend = Path(run), backend
        self.manifest, self.status = verify_run(self.run)
        self.config = self.manifest['config']
        self.schedule = common.load(self.run / 'schedule.json')
        self.prompts = {'cardinal': (self.run / 'prompts/cardinal.txt').read_text()}

    def inspect_checkpoint(self, folder):
        return checkpoint_state(folder, expected_learning_rate=self.config['learning_rate'])

    def checkpoint(self, label):
        relative = f'checkpoints/{label}/metadata.json'
        if relative not in self.status['artifacts']:
            return None
        metadata = self.existing(relative)
        if (metadata['label'] != label or not metadata['complete'] or metadata['manifest_sha256'] != self.status['manifest_sha256']
                or set(metadata['files']) != set(FILES)):
            raise ValueError('Checkpoint metadata differs')
        folder = self.run / 'checkpoints' / label
        for name, digest in metadata['files'].items():
            if common.digest(folder / name) != digest or self.status['artifacts'].get(f'checkpoints/{label}/{name}') != digest:
                raise ValueError('Checkpoint bytes differ')
        if self.inspect_checkpoint(folder) != metadata['optimizer_identity']:
            raise ValueError('Checkpoint optimizer identity differs')
        return metadata

    def activate(self, label):
        metadata = self.checkpoint(label)
        if metadata is None:
            raise ValueError('Missing committed checkpoint')
        self.backend.load_checkpoint(self.run / 'checkpoints' / label, metadata['files'])
        return metadata

    def initialize(self):
        if self.checkpoint('base') is None:
            self.save_checkpoint('base', None, None)
        self.activate('base')

    def save_checkpoint(self, label, parent, report, copy_from=None, update_relative=None):
        if self.checkpoint(label):
            raise ValueError('Checkpoint already committed')
        final = self.run / 'checkpoints' / label
        if final.exists():
            final.rename(final.with_name(label + '.incomplete-' + uuid.uuid4().hex))
        temporary = final.with_name('.' + label + '.pending-' + uuid.uuid4().hex)
        temporary.parent.mkdir(parents=True, exist_ok=True)
        if copy_from is None:
            files = self.backend.save_checkpoint(temporary)
        else:
            temporary.mkdir()
            for name in FILES:
                shutil.copyfile(copy_from / name, temporary / name)
            files = {name: common.digest(temporary / name) for name in FILES}
        identity = self.inspect_checkpoint(temporary)
        before_step = self.checkpoint(parent)['optimizer_identity']['step'] if parent else 0
        expected_step = before_step + (report['optimizer_steps'] if report else 0)
        if identity['step'] != expected_step or identity['learning_rate'] != self.config['learning_rate']:
            raise ValueError('Actual optimizer step/LR differs')
        metadata = {'label': label, 'parent': parent, 'files': files, 'report': report,
            'optimizer_identity': identity, 'complete': True, 'manifest_sha256': self.status['manifest_sha256'],
            'model_repo': self.config['model_repo'], 'model_revision': self.config['model_revision']}
        common.atomic_json(temporary / 'metadata.json', metadata)
        temporary.rename(final)
        transaction = {f'checkpoints/{label}/{name}': digest for name, digest in files.items()}
        transaction[f'checkpoints/{label}/metadata.json'] = common.digest(final / 'metadata.json')
        if update_relative:
            self.existing(update_relative)
            common.atomic_json(self.run / update_relative, report)
            transaction[update_relative] = common.digest(self.run / update_relative)
        self.status['artifacts'].update(transaction)
        self.state()
        return metadata

    def ensure_batch(self, batch_id, names, seeds, checkpoint):
        if len(names) != 8 or len(seeds) != 8:
            raise ValueError('Exactly eight generation requests required')
        source_hash = self.status['artifacts'][f'checkpoints/{checkpoint}/metadata.json']
        context = {'batch_id': batch_id, 'names': names, 'seeds': seeds, 'checkpoint': checkpoint,
            'source_metadata_sha256': source_hash, 'prompt_condition': 'cardinal',
            'prompt_sha256': self.manifest['prompt_sha256']['cardinal']}
        relative = f'batches/{batch_id}.json'
        saved = self.existing(relative)
        if saved is None:
            self.admit()
            self.activate(checkpoint)
            self.state(stage='batch/' + batch_id)
            batch = self.backend.sample_batch(self.prompts['cardinal'], seeds)
            validate_batch(batch, seeds, self.config)
            saved = self.commit(relative, {**context, **batch})
        if any(saved.get(k) != v for k, v in context.items()):
            raise ValueError('Committed batch identity differs')
        validate_batch(saved, seeds, self.config)
        output = []
        for i, (name, seed, row) in enumerate(zip(names, seeds, saved['records'])):
            record = {**row, 'name': name, 'seed': seed, 'checkpoint': checkpoint,
                'prompt_condition': 'cardinal', 'prompt_sha256': context['prompt_sha256'],
                'source_metadata_sha256': source_hash, 'model_repo': MODEL_REPO, 'model_revision': MODEL_REVISION,
                'authorship': 'fresh_learner_sample', 'api_training_data': False,
                'batch_path': relative, 'batch_sha256': self.status['artifacts'][relative], 'batch_index': i}
            if not record['completion_ids'] or len(record['completion_ids']) != len(record['sampled_token_logprobs']):
                raise ValueError('Empty/incomplete sampled completion')
            existing = self.existing(f'generations/{name}.json')
            if existing is None:
                self.commit(f'generations/{name}.json', record)
            elif existing != record:
                raise ValueError('Generation differs from committed batch row')
            output.append(record)
        return output

    def generation(self, name, seed, checkpoint, prompt='cardinal'):
        record = self.existing(f'generations/{name}.json')
        if record is None:
            raise ValueError('A committed full batch is required before per-record access')
        if (record['seed'], record['checkpoint'], record['prompt_condition']) != (seed, checkpoint, prompt):
            raise ValueError('Generation identity differs')
        return record

    def capability(self):
        if self.existing('capability.json') is None:
            self.ensure_batch('capability', [f'capability-{i}' for i in range(8)], self.schedule['capability_generation_seeds'], 'base')
        return super().capability()

    def roster(self, number, initial):
        relative = f'groups/{number}/opponents.json'
        old = self.existing(relative)
        if old is not None:
            return old
        latest = (number-1)//5*5
        if latest:
            population = self.existing(f'league/after-{latest}.json')
            if population is None:
                raise ValueError('Missing preceding league snapshot')
            models = population['models']
            population_hash = self.status['artifacts'][f'league/after-{latest}.json']
        else:
            models = [r for r in initial['selected'] if r['origin'] == 'model']
            population_hash = self.status['artifacts']['opponents.json']
        anchors = [r for r in initial['selected'] if r['origin'] == 'scripted']
        newest = models[-1]
        older = [r for r in models if r['id'] != newest['id']]
        historical = older[(number-1) % len(older)]
        return self.commit(relative, {'cycle': number, 'selected': anchors+[newest, historical],
            'population_sha256': population_hash, 'frozen_before_generation': True})

    def finish_group_timing(self, number, report):
        relative = f'groups/{number}/timing.json'
        if self.existing(relative) is None:
            start = self.existing(f'groups/{number}/timing-start.json')
            elapsed = (datetime.now(timezone.utc)-datetime.fromisoformat(start['started'])).total_seconds()
            if elapsed <= 0:
                raise ValueError('Nonpositive group wall time')
            self.commit(relative, {'cycle': number, 'elapsed_seconds': elapsed,
                'optimizer_steps': report['optimizer_steps'], 'includes_resume_downtime': True,
                'update_sha256': self.status['artifacts'][f'groups/{number}/update.json']})
        return report

    def group(self, number, opponents):
        label = f'cycle{number}'
        if self.checkpoint(label):
            return self.finish_group_timing(number, self.checkpoint(label)['report'])
        if self.existing(f'groups/{number}/timing-start.json') is None:
            self.commit(f'groups/{number}/timing-start.json', {'started': common.now(), 'cycle': number})
        started = time.monotonic()
        source = 'base' if number == 1 else f'cycle{number-1}'
        self.activate(source)
        folder = f'groups/{number}'
        records = self.ensure_batch(f'train-{number}', [f'train-cycle{number}-{i}' for i in range(8)],
            self.schedule['train_generation_seeds'][str(number)], source)
        hashes = {r['name']: self.status['artifacts'][f"generations/{r['name']}.json"] for r in records}
        opponents_hash = self.status['artifacts'][folder + '/opponents.json']
        scores = self.existing(folder + '/scores.json')
        if scores is None:
            rows = []
            for record in records:
                self.admit()
                rows.append({'generation': record['name'], 'generation_sha256': hashes[record['name']],
                    **cardinal.score(record, opponents['selected'], self.schedule['train_maps'][str(number)], self.config)})
            rewards = [r['reward'] for r in rows]
            adv = journal.advantages(rewards)
            scores = self.commit(folder + '/scores.json', {'cycle': number, 'source_checkpoint': source,
                'source_metadata_sha256': self.status['artifacts'][f'checkpoints/{source}/metadata.json'],
                'opponents_sha256': opponents_hash, 'seeds': self.schedule['train_maps'][str(number)],
                'rows': rows, 'rewards': rewards, 'advantages': adv, 'zero_signal': not any(adv)})
        if (scores['source_checkpoint'] != source or {r['generation']: r['generation_sha256'] for r in scores['rows']} != hashes
                or journal.advantages(scores['rewards']) != scores['advantages'] or scores['opponents_sha256'] != opponents_hash):
            raise ValueError('Saved on-policy group provenance differs')
        report = {'cycle': number, 'source_checkpoint': source,
            'source_metadata_sha256': self.status['artifacts'][f'checkpoints/{source}/metadata.json'],
            'scores_sha256': self.status['artifacts'][folder+'/scores.json'], 'generation_sha256': hashes,
            'rewards': scores['rewards'], 'advantages': scores['advantages'],
            'training_tokens': sum(len(r['completion_ids']) for r in records),
            'all_runtime_invalid': not any(r['runtime_valid'] for r in scores['rows'])}
        copy_from = None
        if scores['zero_signal']:
            report.update(outcome='skipped_zero_variance', optimizer_steps=0, adapter_l2_delta=0.0,
                likelihood_gate_path=None, likelihood_gate_sha256=None)
            copy_from = self.run / 'checkpoints' / source
        else:
            self.admit()
            self.state(stage=folder+'/likelihood')
            gate = self.backend.likelihood_gate(records)
            gate.update(generation_sha256=hashes, source_checkpoint=source,
                source_metadata_sha256=report['source_metadata_sha256'], generation_names=list(hashes))
            gate_path = folder+'/gate-'+uuid.uuid4().hex+'.json'
            self.commit(gate_path, gate)
            if not gate['passed']:
                raise ValueError('Full completion likelihood gate failed')
            self.admit()
            self.state(stage=folder+'/optimizer')
            update = self.backend.update(records, scores['advantages'])
            if update['optimizer_steps'] != 1 or update['adapter_l2_delta'] <= 0:
                raise ValueError('No genuine optimizer update')
            update['gradient_likelihood_gate']['generation_names'] = list(hashes)
            report.update(update, outcome='updated', likelihood_gate_path=gate_path,
                likelihood_gate_sha256=self.status['artifacts'][gate_path])
        report['invocation_seconds_before_checkpoint'] = time.monotonic()-started
        report['batch_elapsed_seconds'] = self.existing(f'batches/train-{number}.json')['timing']['batch_elapsed_seconds']
        report['timing_note'] = 'Endpoint projection uses separate timing.json wall time including checkpoint save and resume downtime; per-row sample times are never summed.'
        self.save_checkpoint(label, source, report, copy_from=copy_from, update_relative=folder+'/update.json')
        return self.finish_group_timing(number, report)

    def refresh(self, number, initial):
        if number % 5:
            return
        relative = f'league/after-{number}.json'
        if self.existing(relative) is not None:
            return
        prior = self.existing(f'league/after-{number-5}.json')['models'] if number > 5 else initial['selected']
        scores = self.existing(f'groups/{number}/scores.json')
        hashes = {r['generation']: r['generation_sha256'] for r in scores['rows']}
        models, added = refresh_roster(initial['selected'], prior, scores['rows'], hashes,
            scores['source_metadata_sha256'], number)
        self.commit(relative, {'after_cycle': number, 'models': models, 'added_ids': [r['id'] for r in added],
            'scores_sha256': self.status['artifacts'][f'groups/{number}/scores.json'],
            'retained_initial_two': True, 'maximum_models': 6})

    def evaluate(self, opponents, endpoint):
        endpoints = ('base', f'cycle{endpoint}')
        if any(self.checkpoint(label) is None for label in endpoints):
            raise ValueError('Both fixed endpoints required')
        paths = []
        for label in endpoints:
            for offset in range(0, 32, 8):
                names = [f'eval-{label}-{i}' for i in range(offset, offset+8)]
                self.ensure_batch(f'eval-{label}-{offset//8}', names,
                    self.schedule['eval_generation_seeds'][offset:offset+8], label)
                paths.extend(f'generations/{name}.json' for name in names)
        seal = {'checkpoints': {e: self.status['artifacts'][f'checkpoints/{e}/metadata.json'] for e in endpoints},
            'generations': {p: self.status['artifacts'][p] for p in paths}, 'draws': 64,
            'endpoint_selection_sha256': self.status['artifacts']['endpoint-selection.json'], 'scoring_allowed': True}
        previous = self.existing('evaluation/sealed.json')
        if previous is None:
            self.commit('evaluation/sealed.json', seal)
        elif previous != seal:
            raise ValueError('Evaluation seal differs')
        for label in endpoints:
            relative = f'evaluation/{label}/holdout.json'
            if self.existing(relative) is not None:
                continue
            rows = []
            for i in range(32):
                self.admit()
                name = f'eval-{label}-{i}'
                record = self.existing(f'generations/{name}.json')
                rows.append({'generation': name, 'generation_sha256': self.status['artifacts'][f'generations/{name}.json'],
                    **cardinal.score(record, opponents['selected'], self.schedule['holdout_maps'], self.config)})
            self.commit(relative, {'checkpoint': label, 'prompt_condition': 'cardinal', 'split': 'holdout',
                'training_use': False, 'seeds': self.schedule['holdout_maps'], 'opponents': opponents['selected'],
                'opponents_sha256': self.status['artifacts']['opponents.json'],
                'seal_sha256': self.status['artifacts']['evaluation/sealed.json'], 'prescribed_program_draws': 32, 'scores': rows})
        self.admit()
        updates = sum(self.checkpoint(f'cycle{i}')['report']['optimizer_steps'] for i in range(1, endpoint+1))
        self.state(state='complete', stage='complete', claims_allowed=True, attempted_groups=endpoint, actual_optimizer_steps=updates)

    def run_all(self):
        self.initialize()
        capability = self.capability()
        if not capability['passed']:
            self.state(state='capability_failed', stage='productive_play_gate', claims_allowed=False)
            return
        opponents = self.opponents(capability)
        selection = self.existing('endpoint-selection.json')
        if selection is None:
            reports = []
            for number in range(1, self.config['profile_attempt_cap']+1):
                self.group(number, self.roster(number, opponents))
                reports.append(self.existing(f'groups/{number}/timing.json'))
                if reports[-1]['optimizer_steps']:
                    break
            _, remaining = journal.execution_budget(self.config, self.status['execution_budget'])
            selection = endpoint_decision(reports, remaining, self.config)
            selection['profile_update_sha256'] = {str(i+1): self.status['artifacts'][f'groups/{i+1}/update.json'] for i in range(len(reports))}
            selection['profile_timing_sha256'] = {str(i+1): self.status['artifacts'][f'groups/{i+1}/timing.json'] for i in range(len(reports))}
            self.commit('endpoint-selection.json', selection)
        if not selection['passed']:
            self.state(state='throughput_or_signal_failed', stage='endpoint_admission', claims_allowed=False)
            return
        endpoint = selection['endpoint']
        for number in range(1, endpoint+1):
            if self.checkpoint(f'cycle{number}') is None:
                _, remaining = journal.execution_budget(self.config, self.status['execution_budget'])
                if remaining < self.config['audit_export_reserve_seconds'] + self.config['evaluation_batch_reserve_seconds']:
                    self.state(state='budget_incomplete', stage=f'before_cycle{number}', claims_allowed=False)
                    return
            self.group(number, self.roster(number, opponents))
            self.refresh(number, opponents)
        self.evaluate(opponents, endpoint)


def worker(run, model_dir, lock_fds):
    journal.verify_lock_descriptors(run, lock_fds)
    manifest, status = verify_run(run)
    cfg = manifest['config']
    if 'execution_budget' not in status:
        raise ValueError('Parent must commit durable execution budget')
    journal.execution_budget(cfg, status['execution_budget'])
    actual = {name: importlib.metadata.version(name).split('+')[0] for name in VERSIONS}
    if actual != VERSIONS:
        raise ValueError('Installed CUDA dependency versions differ: ' + repr(actual))
    import torch
    if torch.version.cuda != '12.8':
        raise ValueError('The pinned Torch CUDA12.8 build is required even on direct/resumed runs')
    index_model(model_dir, Path(run) / 'inputs/model-info.json')
    index = common.verify_model_index(model_dir, cfg)
    if index['authoritative_metadata_sha256'] != manifest['input_sha256']['inputs/model-info.json']:
        raise ValueError('Staged model index is not bound to authoritative metadata')
    engine = GPUCycle(run, None)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    prompt_ids = tokenizer.apply_chat_template([{'role': 'user', 'content': engine.prompts['cardinal']}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False)
    measurement = {'prompt_tokens': len(prompt_ids), 'completion_cap': cfg['max_new_tokens'],
        'context_cap': cfg['max_context_tokens'], 'admitted': len(prompt_ids)+cfg['max_new_tokens'] <= cfg['max_context_tokens']}
    old = engine.existing('prompt-measurement.json')
    if old is None:
        engine.commit('prompt-measurement.json', measurement)
    elif old != measurement:
        raise ValueError('Tokenizer/prompt measurement changed')
    if not measurement['admitted']:
        raise ValueError('Prompt plus completion exceeds context budget')
    del tokenizer
    backend = CardinalCudaPolicy(model_dir, cfg)
    engine = GPUCycle(run, backend)
    runtime = {'model_index': index, **backend.runtime}
    old = engine.existing('runtime.json')
    if old is None:
        engine.commit('runtime.json', runtime)
    elif old != runtime:
        raise ValueError('Runtime identity changed')
    engine.initialize()
    engine.admit()
    engine.state(stage='numerical_smoke')
    try:
        smoke = backend.mechanics_smoke(engine.prompts['cardinal'])
    except Exception as error:
        smoke = {'passed': False, 'training_use': False, 'optimizer_steps': 0,
            'error_type': type(error).__name__, 'reason': str(error)[:2000],
            'incomplete_diagnostic': True}
    smoke.update(source_checkpoint='base', source_metadata_sha256=engine.status['artifacts']['checkpoints/base/metadata.json'])
    engine.commit('smoke-' + str(time.time_ns()) + '.json', smoke)
    if not smoke['passed']:
        engine.state(state='numerical_failed', stage='numerical_smoke', claims_allowed=False)
        return engine.status
    engine.run_all()
    return engine.status


def launch(run, model_dir):
    run, model_dir = Path(run).resolve(), Path(model_dir).resolve()
    manifest, status = verify_run(run)
    if status['state'] in {'complete', 'capability_failed', 'throughput_or_signal_failed', 'budget_incomplete', 'numerical_failed'}:
        return status
    with common.process_lock(LOCK) as global_fd, common.process_lock(run / 'run.lock') as run_fd:
        status = common.load(run / 'status.json')
        budget, timeout = journal.execution_budget(manifest['config'], status.get('execution_budget'))
        status.update(execution_budget=budget, updated=common.now())
        common.atomic_json(run / 'status.json', status)
        child = subprocess.Popen([sys.executable, '-m', 'selfplay.program_cardinal_gpu', '_worker',
            '--run', str(run), '--model-dir', str(model_dir), '--lock-fds', f'{global_fd},{run_fd}'],
            cwd=ROOT, pass_fds=(global_fd, run_fd), start_new_session=True)
        started = time.monotonic()
        try:
            while child.poll() is None:
                if time.monotonic()-started >= timeout:
                    raise TimeoutError('Immutable deployment deadline or persistent worker budget reached')
                time.sleep(1)
            if child.returncode:
                raise RuntimeError(f'CUDA worker exited{child.returncode}; committed artifacts remain resumable')
        except BaseException as error:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
            status = common.load(run / 'status.json')
            status.update(state='stopped', updated=common.now(), claims_allowed=False,
                error_type=type(error).__name__, reason=str(error))
            common.atomic_json(run / 'status.json', status)
            raise
    return common.load(run / 'status.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'prepare', 'index-model', 'run', '_worker'))
    parser.add_argument('--run', type=Path)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--deployment', type=Path)
    parser.add_argument('--model-info', type=Path, default=ROOT / 'runs/cardinal-gpu-preflight/model-info.json')
    parser.add_argument('--lock-fds')
    args = parser.parse_args()
    if args.command in ('plan', 'prepare') and args.deployment is None:
        parser.error('--deployment is required')
    if args.command == 'plan':
        result = plan(public_deployment(common.load(args.deployment)))
    elif args.command == 'prepare':
        if args.run is None:
            parser.error('--run required')
        result = prepare(args.run, args.deployment, args.model_info)
    elif args.command == 'index-model':
        if args.model_dir is None:
            parser.error('--model-dir required')
        result = index_model(args.model_dir, args.model_info)
    else:
        if args.run is None or args.model_dir is None:
            parser.error('--run and --model-dir required')
        result = launch(args.run, args.model_dir) if args.command == 'run' else worker(args.run, args.model_dir, args.lock_fds)
    print(common.canonical(result))


if __name__ == '__main__':
    main()
