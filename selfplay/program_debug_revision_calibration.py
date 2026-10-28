"""Separately frozen native-nonthinking-revision delivery/productivity calibration; no training.

CPU preparation/auditing do not construct a model. CUDA execution inherits the
strict cardinal checkpoint identity and accumulated policy-gradient machinery.
Every sampled draft and revision is retained, including invalid/truncated ones.
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
from selfplay import program_cardinal_gpu as gpu
from selfplay import program_cardinal_cycle as cardinal
from selfplay import program_study as journal
from selfplay import program_debug_revision_harness as harness
from selfplay import program_debug_calibration as preceding
from selfplay import program_debug_rl as previous
from selfplay.program_debug_revision_cuda import RevisionCudaPolicy, stage_metadata
from selfplay.program_cardinal_cuda import checkpoint_state

ROOT, LOCK = gpu.ROOT, gpu.LOCK
MODEL_REPO, MODEL_REVISION = gpu.MODEL_REPO, gpu.MODEL_REVISION
VERSIONS = {**gpu.VERSIONS, 'tokenizers': '0.23.3'}
index_model = gpu.index_model
INITIAL = ROOT / 'runs/cardinal-warmstart-gpu-v2'
TOKENIZER_INPUT = ROOT / 'runs/cardinal-debug-preflight-v1/tokenizer'
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'generation_config.json', 'config.json', 'chat_template.jinja')
PRIOR_DEBUG = ROOT / 'runs/program-debug-gpu-v1'
PRIOR_CALIBRATION = ROOT / 'runs/program-debug-calibration-gpu-v1'
DESIGN_PLAN = ROOT / 'runs/debug-mixed-thinking-calibration-design-v2.json'
CONTEXT_REVIEW = ROOT / 'runs/debug-revision-calibration-preflight-v1/context-stress-review.json'
SOURCES = list(dict.fromkeys(preceding.SOURCES + [
    'selfplay/program_debug_revision_harness.py', 'selfplay/program_debug_revision_cuda.py',
    'selfplay/program_debug_revision_calibration.py', 'scripts/analyze_program_debug_revision_calibration.py',
    'scripts/cloud/run_revision_calibration_job.py', 'scripts/cloud/bootstrap_revision_calibration.py',
    'scripts/revision_calibration_supervisor.py']))
TERMINAL = {'complete', 'budget_incomplete', 'numerical_failed', 'stopped', 'failed'}


def config(deployment):
    if (deployment['maximum_hours'] != 3 or deployment['reservation_usd'] != 25
            or deployment['maximum_gpu_rate_usd_hour'] > 5.29):
        raise ValueError('Calibration requires the fixed three-hour/$25 envelope')
    return {**gpu.config(deployment), 'kind': 'native_nonthinking_revision_calibration',
        'condition': 'debug-revision-calibration-v1', 'seed': 325401, 'algorithm': 'none_calibration_only',
        'thinking': True, 'reasoning_effort': 'low',
        'turn_native_modes': {'draft': stage_metadata('draft'), 'revision': stage_metadata('revision')}, 'max_new_tokens': 4096,
        'max_context_tokens': 16384, 'sample_seconds': 900, 'worker_seconds': 9000,
        'batch_size': 2, 'group_size': 8, 'iterations': 0, 'endpoint_choices': [],
        'profile_attempt_cap': 0, 'authoring_calls': 2, 'practice_maps_per_trajectory': 2,
        'capability_samples': 8, 'minimum_productive_programs': 4,
        'minimum_productive_distinct_ir': 2, 'minimum_positive_calibration_games': 12,
        'opponent_attempt_cap': 0, 'eval_samples': 0,
        'scope': 'Fixed eight-trajectory authoring capability calibration; no learning or improvement comparison',
        'baseline': 'Exact prior SFT-step32 weights and zero-step RL Adam; no further SFT or RL',
        'history_transform': harness.REVISION_CONTEXT_POLICY,
        'reward': 'Diagnostic final controller return/banking/validity only; never used for training'}


def schedule():
    return {'practice_maps': {str(i): [191000000+i*10+j for j in range(2)] for i in range(8)},
        'scoring_maps': list(range(192000000,192000006)),
        'generation_seeds': [201000000+i*2 for i in range(8)],
        'revision_seed_rule': 'draft_seed+1; paired consecutive seeds, no retry'}


def plan(deployment, design_path=DESIGN_PLAN):
    return {'kind':'native_nonthinking_revision_calibration_plan', 'config':config(deployment), 'schedule':schedule(),
        'design_plan_sha256':common.digest(design_path),
        'authoring':'Exactly two calls per trajectory, draft then fixed factual practice feedback and mandatory revision. Eight trajectories, batch2, all draws retained.',
        'reasoning':'Draft uses native low thinking; revision uses native enable_thinking=False with a fixed empty thinking prefix. Both caps4096. Initial messages and factual feedback identical to previous calibration. Draft boundary parser unchanged; whole revision completion uses existing cardinal parser.',
        'baseline':'Exact prior SFT32/zero-RL-step adapter and optimizer. No zero-SFT comparison, SFT, API authoring or optimizer updates.',
        'scoring':'Seal all8 final revisions, pass full16-turn unchanged eval/train likelihood admission, then score six new maps against nearest/value in both seats,24games per compiled final; invalid authorship zero yield with no fabricated games.',
        'gate':'At least4/8 final controllers fault-free on all24 games and bank-positive on at least12 games; at least2 distinct compiled IRs. Missing output means incomplete, not failure or zero.',
        'numerics':'Fresh mixed-native batch2 32-token derivative smoke; then full unchanged eval/train replay of all16 actual turns before any final scoring. Token0.002/sequence0.02 tolerances unchanged. No update; no full-length backward feasibility claim.',
        'interpretation':'Fresh descriptive calibration of native revision thinking disabled; unchanged draft/feedback/caps and exact SFT32 base. Fresh maps/seeds do not supply a paired causal comparison. No learned improvement claim.',
        'resources':{'cumulative_ceiling_usd':200,'reservation_usd':25,'rental_maximum_seconds':10800,
            'persistent_worker_seconds':9000,'maximum_sample_batches':8,'sample_batch_seconds':900,
            'worst_case_sample_seconds':7200,'worker_smoke_feedback_export_allowance_seconds':1800,
            'rental_setup_export_allowance_seconds':1800,'deadline_unix':deployment['deadline_unix'],
            'completion_guaranteed':False},
        'next':'Only a complete, audited passing calibration may inform a separately frozen longer RL run; no automatic training in this job.'}


def fingerprints(timetable, prior):
    known=set()
    def collect(value):
        if isinstance(value,dict):
            if 'seed' in value and 'layout' in value and 'sha256' in value: known.add(value['sha256'])
            if 'prior_seed_fingerprints' in value:known.update(value['prior_seed_fingerprints'].values())
            for child in value.values():collect(child)
        elif isinstance(value,list):
            for child in value:collect(child)
    collect(prior);groups={}
    for family in ('practice_maps','scoring_maps'):
        value=timetable[family]
        seeds=[s for row in value.values() for s in row] if isinstance(value,dict) else value
        groups[family]=[]
        for seed in seeds:
            layout=journal.map_layout(seed);digest=gpu.object_hash(layout)
            if digest in known:raise ValueError('Observable map overlaps prior or another calibration map')
            known.add(digest);groups[family].append({'seed':seed,'sha256':digest,'layout':layout})
    return {'families':groups,'prior_evidence_sha256':gpu.object_hash(prior),
        'definition':'Exact observable initial grid/site order/homes/capacity/energy/horizon'}


def initial_roster():
    return {'selected':[{'id':n,'origin':'scripted','authorship':'scripted_anchor',
        'compiled_ir':common.reference_programs()[n]} for n in ('nearest','value')], 'frozen':True}


def prepare(run, deployment_path, model_info_path):
    run = Path(run)
    if run.exists(): raise FileExistsError('New run directory required')
    deployment = gpu.public_deployment(common.load(deployment_path))
    initial_manifest, initial_status = common.load(INITIAL/'manifest.json'), common.load(INITIAL/'status.json')
    if initial_status['state'] != 'capability_failed' or initial_status['manifest_sha256'] != common.digest(INITIAL/'manifest.json'):
        raise ValueError('Expected preserved failed warmstart run')
    audit = common.load(INITIAL/'final-audit.json')
    if (audit.get('integrity_passed') is not True
            or audit.get('manifest_sha256') != common.digest(INITIAL/'manifest.json')
            or audit.get('status_sha256') != common.digest(INITIAL/'status.json')
            or audit.get('genuine_supervised_optimizer_steps') != 32
            or audit.get('genuine_RL_optimizer_steps') != 0 or audit.get('optimizer_reset_verified') is not True):
        raise ValueError('Initial32-SFT/0-RL checkpoint lacks a current passed independent audit')
    initial_meta = common.load(INITIAL/'checkpoints/base/metadata.json')
    if common.digest(INITIAL/'checkpoints/base/metadata.json') != initial_status['artifacts']['checkpoints/base/metadata.json']:
        raise ValueError('Initial checkpoint metadata changed')
    identity = checkpoint_state(INITIAL/'checkpoints/base', 1e-5)
    if identity != initial_meta['optimizer_identity'] or identity['step'] != 0:
        raise ValueError('Exact zero-RL-step initial checkpoint required')
    previous_manifest = common.load(PRIOR_CALIBRATION/'manifest.json')
    previous_status = common.load(PRIOR_CALIBRATION/'status.json')
    previous_audit = common.load(PRIOR_CALIBRATION/'final-audit.json')
    if (previous_status.get('state') != 'complete'
            or previous_status.get('qualification_passed') is not False
            or previous_status.get('actual_optimizer_steps') != 0
            or previous_status['manifest_sha256'] != common.digest(PRIOR_CALIBRATION/'manifest.json')
            or previous_audit.get('integrity_passed') is not True
            or previous_audit.get('manifest_sha256') != common.digest(PRIOR_CALIBRATION/'manifest.json')
            or previous_audit.get('status_sha256') != common.digest(PRIOR_CALIBRATION/'status.json')
            or previous_manifest['initial_checkpoint_identity'] != identity
            or common.digest(PRIOR_CALIBRATION/'map-fingerprints.json') != previous_manifest['input_sha256']['map-fingerprints.json']):
        raise ValueError('Prior failed calibration identity or unchanged baseline differs')
    design = common.load(DESIGN_PLAN)
    for key, name in [('manifest_sha256','manifest.json'),('status_sha256','status.json'),
                      ('final_audit_sha256','final-audit.json'),('map_fingerprints_sha256','map-fingerprints.json')]:
        if design['predecessor'][key] != common.digest(PRIOR_CALIBRATION/name):
            raise ValueError('Authoritative design predecessor differs: '+key)
    cfg, timetable = config(deployment), schedule()
    prior = {'warmstart':common.load(INITIAL/'map-fingerprints.json'),
             'debug':common.load(PRIOR_DEBUG/'map-fingerprints.json'),
             'calibration':common.load(PRIOR_CALIBRATION/'map-fingerprints.json')}
    parent_debug=common.load(PRIOR_DEBUG/'manifest.json')
    if common.digest(PRIOR_DEBUG/'map-fingerprints.json') != parent_debug['input_sha256']['map-fingerprints.json']:
        raise ValueError('Prior debug map identity differs')
    maps = fingerprints(timetable, prior)
    roster = initial_roster()
    metadata = common.load(model_info_path)
    if (metadata['id'], metadata['sha']) != (MODEL_REPO, MODEL_REVISION):
        raise ValueError('Authoritative model identity differs')
    authoritative = {r['rfilename']:r for r in metadata['siblings']}
    for name in TOKENIZER_FILES:
        raw, row = (TOKENIZER_INPUT/name).read_bytes(), authoritative[name]
        if row.get('lfs'):
            actual, expected = hashlib.sha256(raw).hexdigest(), row['lfs']['sha256']
        else:
            actual, expected = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(), row['blobId']
        if len(raw) != row['size'] or actual != expected:
            raise ValueError('Original HF tokenizer/config bytes differ: '+name)
    for relative in SOURCES:
        if not (ROOT/relative).is_file(): raise ValueError('Source not ready: '+relative)
    run.mkdir(parents=True)
    inputs = {}
    def copy(source, relative):
        target = run/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        inputs[relative] = common.digest(target)
    for name in ('manifest.json', 'status.json', 'final-audit.json', 'map-fingerprints.json'):
        copy(INITIAL/name, 'inputs/initial-'+name)
    for name in ('adapter.safetensors', 'optimizer.pt', 'metadata.json'):
        copy(INITIAL/'checkpoints/base'/name, 'inputs/initial-checkpoint/'+name)
    copy(model_info_path, 'inputs/model-info.json')
    for name in TOKENIZER_FILES:
        copy(TOKENIZER_INPUT/name, 'inputs/tokenizer/'+name)
    copy(PRIOR_DEBUG/'manifest.json','inputs/prior-debug-manifest.json')
    copy(PRIOR_DEBUG/'map-fingerprints.json','inputs/prior-debug-map-fingerprints.json')
    for name in ('manifest.json','status.json','final-audit.json','map-fingerprints.json'):
        copy(PRIOR_CALIBRATION/name,'inputs/previous-calibration-'+name)
    copy(DESIGN_PLAN,'inputs/design-plan.json')
    copy(ROOT/'runs/debug-mixed-thinking-calibration-design-v1.json','inputs/design-plan-v1.json')
    copy(ROOT/'runs/debug-calibration-preflight-v1/context-stress-review.json','inputs/context-stress-review.json')
    copy(ROOT/'runs/debug-calibration-preflight-v1/context-stress-review-v2.json','inputs/context-stress-review-v2.json')
    copy(CONTEXT_REVIEW,'inputs/revision-context-stress-review.json')
    for relative, value in [('deployment.json', deployment), ('plan.json', plan(deployment)),
                            ('schedule.json', timetable), ('map-fingerprints.json', maps), ('initial-opponents.json', roster)]:
        common.atomic_json(run/relative, value)
        inputs[relative] = common.digest(run/relative)
    (run/'prompts').mkdir()
    (run/'prompts/debug.txt').write_text(harness.debug_model_spec())
    inputs['prompts/debug.txt'] = common.digest(run/'prompts/debug.txt')
    sources = {}
    for relative in SOURCES:
        target = run/'source'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, target)
        sources[relative] = common.digest(target)
    manifest = {'format_version': 1, 'kind': cfg['kind'], 'created': common.now(), 'config': cfg,
        'source_sha256': sources, 'input_sha256': inputs, 'prompt_sha256': {'debug': inputs['prompts/debug.txt']},
        'expected_installed_versions': VERSIONS, 'claims_require_complete': True,
        'initial_checkpoint_files': initial_meta['files'], 'initial_checkpoint_identity': identity,
        'baseline': cfg['baseline']}
    common.atomic_json(run/'manifest.json', manifest)
    common.atomic_json(run/'status.json', {'manifest_sha256': common.digest(run/'manifest.json'),
        'artifacts': {}, 'state': 'prepared', 'stage': 'prepared', 'updated': common.now(), 'claims_allowed': False})
    verify_run(run)
    return manifest


def verify_run(run, require_active=True):
    run = Path(run)
    manifest, status = common.load(run/'manifest.json'), common.load(run/'status.json')
    deployment = gpu.public_deployment(common.load(run/'deployment.json'))
    if (manifest['kind'] != 'native_nonthinking_revision_calibration' or manifest['config'] != config(deployment)
            or status['manifest_sha256'] != common.digest(run/'manifest.json')
            or common.load(run/'schedule.json') != schedule() or common.load(run/'plan.json') != plan(deployment, design_path=run/'inputs/design-plan.json')
            or set(manifest['source_sha256']) != set(SOURCES)):
        raise ValueError('Frozen debug protocol identity differs')
    for relative, expected in {**manifest['input_sha256'], **status['artifacts']}.items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or common.digest(run/path) != expected:
            raise ValueError('Committed artifact identity differs: '+relative)
    for relative, expected in manifest['source_sha256'].items():
        if common.digest(run/'source'/relative) != expected or (require_active and common.digest(ROOT/relative) != expected):
            raise ValueError('Archived/active source differs: '+relative)
    return manifest, status


def validate_batch(batch, contexts, seeds, cfg, authoring_stage):
    mode = stage_metadata(authoring_stage)
    records = batch.get('records', [])
    if len(records) != len(seeds) or len(seeds) != 2 or len(contexts) != 2:
        raise ValueError('Exactly two complete contexts and batch rows required')
    for i, (row, context, seed) in enumerate(zip(records, contexts, seeds)):
        ids, lps, prefix = row.get('completion_ids'), row.get('sampled_token_logprobs'), row.get('prompt_ids')
        if (row.get('messages') != context or row.get('seed') != seed or row.get('sampling_batch_row') != i
                or row.get('sampling_batch_size') != 2
                or any(row.get(k) != v or type(row.get(k)) is not type(v) for k,v in mode.items())
                or row.get('completion_token_limit') != cfg['max_new_tokens'] or row.get('calibration_only') is not True):
            raise ValueError('Sampled row context/seed/identity differs')
        if (not isinstance(prefix, list) or not prefix or not isinstance(ids, list) or not ids
                or len(ids) > cfg['max_new_tokens'] or len(prefix)+cfg['max_new_tokens'] > cfg['max_context_tokens']
                or any(type(x) is not int or x < 0 for x in prefix+ids)
                or len(lps or []) != len(ids) or any(type(x) not in (int,float) or not math.isfinite(x) or x > 1e-6 for x in lps)
                or row.get('input_tokens') != len(prefix) or row.get('output_tokens') != len(ids)):
            raise ValueError('Sampled token/logprob coverage differs')
        eos = row.get('eos_token_ids')
        if (not isinstance(eos, list) or not eos or row.get('finish_reason') not in ('eos','length','time')
                or any(t in eos for t in ids[:-1]) or (ids[-1] in eos) != (row['finish_reason']=='eos')):
            raise ValueError('Sampled termination differs')
    duration = batch.get('timing', {}).get('batch_elapsed_seconds', 0)
    if not isinstance(duration, (int,float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid batch timing')
    if batch['timing'].get('total_output_tokens') != sum(len(r['completion_ids']) for r in records):
        raise ValueError('Batch token total differs')


def score(record, opponents, seeds, cfg):
    classification, program = harness.classify_revision(record['text'], record['finish_reason'])
    if program is None:
        result = cardinal.score({'text': '', 'finish_reason': 'length'}, opponents, seeds, cfg)
    else:
        result = cardinal.score({'text': classification['final_answer'], 'finish_reason': 'eos'}, opponents, seeds, cfg)
    result['debug_classification'] = classification
    return result


def validate_full_gate(gate, records, cfg):
    """CPU check of recorded admission evidence, not a neural recomputation."""
    flags = ('adapter_unchanged','optimizer_unchanged','rng_restored','gradients_restored',
             'config_restored','model_mode_restored','frozen_state_unchanged')
    if (gate.get('passed') is not True or gate.get('optimizer_steps') != 0
            or gate.get('training_use') is not False or gate.get('expected_records') != 16
            or gate.get('token_tolerance') != cfg['token_logprob_tolerance']
            or gate.get('sequence_tolerance') != cfg['sequence_logprob_tolerance']
            or any(gate.get(k) is not True for k in flags)
            or not gate.get('state_before') or gate.get('state_before') != gate.get('state_after')
            or len(records) != 16 or len(gate.get('records',[])) != 16
            or gate.get('sampled_tokens') != sum(len(r['completion_ids']) for r in records)):
        raise ValueError('Full completion gate failed or changed state')
    for i,(row,record) in enumerate(zip(gate['records'], records)):
        if (row.get('index') != i or row.get('record_sha256') != gpu.object_hash(record)
                or row.get('authoring_stage') != ('draft' if i%2==0 else 'revision')
                or row.get('authoring_stage') != record['authoring_stage']
                or row.get('seed') != record['seed'] or row.get('passed') is not True
                or row.get('completion_tokens') != len(record['completion_ids'])
                or row.get('sampled_token_logprobs') != record['sampled_token_logprobs']):
            raise ValueError('Full completion gate record coverage differs')
        for mode in ('eval','train'):
            vector = row.get(mode+'_token_logprobs')
            if (not isinstance(vector,list) or len(vector) != len(record['completion_ids'])
                    or row.get(mode+'_all_finite') is not True
                    or row.get(mode+'_grad_enabled') is not (mode=='train')
                    or any(type(x) not in (int,float) or not math.isfinite(x) for x in vector)):
                raise ValueError('Full completion gate likelihood vector differs')
            check = common.check_likelihoods(record['sampled_token_logprobs'], vector,
                cfg['token_logprob_tolerance'],cfg['sequence_logprob_tolerance'])
            if not check['passed'] or row.get(mode+'_check') != check:
                raise ValueError('Full completion gate likelihood bound failed')
    return True


class Calibration(gpu.GPUCycle):
    def __init__(self, run, backend):
        self.run, self.backend = Path(run), backend
        self.manifest, self.status = verify_run(run)
        self.config, self.schedule = self.manifest['config'], common.load(self.run/'schedule.json')
        self.prompts = {'debug': (self.run/'prompts/debug.txt').read_text()}

    def initialize(self):
        if self.checkpoint('base') is None:
            self.backend.load_checkpoint(self.run/'inputs/initial-checkpoint', self.manifest['initial_checkpoint_files'])
            self.save_checkpoint('base', None, None, copy_from=self.run/'inputs/initial-checkpoint')
        self.activate('base')
        if self.existing('opponents.json') is None:
            self.commit('opponents.json', common.load(self.run/'initial-opponents.json'))

    def ensure_raw_batch(self, batch_id, contexts, seeds, checkpoint, authoring_stage):
        path = f'batches/{batch_id}.json'
        context = {'checkpoint': checkpoint, 'source_metadata_sha256': self.status['artifacts'][f'checkpoints/{checkpoint}/metadata.json'],
                   'contexts_sha256': gpu.object_hash(contexts), 'seeds': seeds, 'authoring_stage': authoring_stage}
        batch = self.existing(path)
        if batch is None:
            self.admit(); self.activate(checkpoint); self.state(stage='batch/'+batch_id)
            # Durable intent ensures an interrupted/ambiguous sample is never silently redrawn.
            intent = f'batches/{batch_id}-intent.json'
            if self.existing(intent) is not None:
                raise ValueError('Uncommitted sampled batch intent exists; preserve failure, never redraw')
            self.commit(intent, {**context, 'started': common.now()})
            sampled = self.backend.sample_messages_batch(contexts, seeds, authoring_stage=authoring_stage)
            # Retain the returned raw draw even if admission discovers a
            # mechanical inconsistency. Validation never triggers resampling.
            batch = self.commit(path, {**context, **sampled})
        if any(batch.get(k) != v for k,v in context.items()): raise ValueError('Batch provenance differs')
        validate_batch(batch, contexts, seeds, self.config, authoring_stage)
        return batch, path

    def trajectories(self, batch_id, names, seeds, checkpoint, practice_maps):
        if len(names) != 2 or len(seeds) != 2 or len(practice_maps) != 2:
            raise ValueError('Two fixed independent trajectories per batch required')
        initial = harness.initial_messages()
        if initial != [{'role':'user','content':self.prompts['debug']}]: raise ValueError('Frozen prompt changed')
        drafts, draft_path = self.ensure_raw_batch(batch_id+'-draft', [initial]*2, seeds, checkpoint, 'draft')
        feedbacks, contexts = [], []
        for i, draft in enumerate(drafts['records']):
            path = f'feedback/{names[i]}.json'
            evidence = self.existing(path)
            if evidence is None:
                self.admit()
                evidence = self.commit(path, {'draft_batch_sha256': self.status['artifacts'][draft_path],
                    'draft_batch_index': i, 'practice_seeds': practice_maps[i],
                    **harness.practice_feedback(draft, practice_maps[i], max_rounds=self.config['max_rounds'])})
            if (evidence['draft_batch_sha256'] != self.status['artifacts'][draft_path]
                    or evidence['draft_batch_index'] != i or evidence['practice_seeds'] != practice_maps[i]):
                raise ValueError('Practice feedback provenance differs')
            feedbacks.append((path, evidence))
            contexts.append(harness.revision_messages(initial, draft, evidence))
        revisions, revision_path = self.ensure_raw_batch(batch_id+'-revision', contexts, [s+1 for s in seeds], checkpoint, 'revision')
        results = []
        for i, name in enumerate(names):
            draft, revision = drafts['records'][i], revisions['records'][i]
            feedback_path, _ = feedbacks[i]
            trajectory = {**revision, 'name': name, 'checkpoint': checkpoint,
                'source_metadata_sha256': self.status['artifacts'][f'checkpoints/{checkpoint}/metadata.json'],
                'prompt_condition': 'native_nonthinking_revision_calibration', 'authorship': 'fresh_learner_two_call_trajectory',
                'model_repo': MODEL_REPO, 'model_revision': MODEL_REVISION, 'api_training_data': False,
                'trajectory_seed': seeds[i], 'turns': [draft, revision],
                'turn_sha256': [gpu.object_hash(draft), gpu.object_hash(revision)],
                'draft_batch_path': draft_path, 'draft_batch_sha256': self.status['artifacts'][draft_path],
                'revision_batch_path': revision_path, 'revision_batch_sha256': self.status['artifacts'][revision_path],
                'batch_index': i, 'feedback_path': feedback_path,
                'feedback_sha256': self.status['artifacts'][feedback_path], 'practice_seeds': practice_maps[i],
                'final_turn_index': 1, 'training_tokens': len(draft['completion_ids'])+len(revision['completion_ids']),
                'history_transform': self.config['history_transform']}
            relative = f'generations/{name}.json'
            old = self.existing(relative)
            if old is None: self.commit(relative, trajectory)
            elif old != trajectory: raise ValueError('Trajectory differs from committed batches')
            results.append(trajectory)
        return results

    def run_all(self):
        self.initialize()
        for offset in range(0,8,2):
            self.trajectories(f'calibration-{offset//2}', [f'calibration-{i}' for i in range(offset,offset+2)],
                self.schedule['generation_seeds'][offset:offset+2], 'base',
                [self.schedule['practice_maps'][str(i)] for i in range(offset,offset+2)])
        paths=[f'generations/calibration-{i}.json' for i in range(8)]
        seal={'generations':{p:self.status['artifacts'][p] for p in paths},'draws':8,'sampled_turns':16,
            'checkpoint_sha256':self.status['artifacts']['checkpoints/base/metadata.json'],
            'scoring_allowed':False,'requires_full_likelihood_gate':True}
        old=self.existing('calibration/sealed.json')
        if old is None:self.commit('calibration/sealed.json',seal)
        elif old!=seal:raise ValueError('Calibration seal differs')
        records=[turn for path in paths for turn in self.existing(path)['turns']]
        binding={'source_checkpoint':'base',
            'source_metadata_sha256':self.status['artifacts']['checkpoints/base/metadata.json'],
            'seal_sha256':self.status['artifacts']['calibration/sealed.json'],
            'generation_sha256':{path:self.status['artifacts'][path] for path in paths},
            'turn_sha256':[gpu.object_hash(record) for record in records]}
        gate_path='calibration/full-likelihood-gate.json'
        gate=self.existing(gate_path)
        if gate is None:
            self.admit();self.state(stage='full_completion_likelihood_gate')
            intent_path='calibration/full-likelihood-gate-intent.json'
            if self.existing(intent_path) is not None:
                self.state(state='numerical_failed',stage='full_completion_likelihood_gate',
                    claims_allowed=False,reason='Interrupted full likelihood admission; no automatic retry')
                return
            self.commit(intent_path,{**binding,'started':common.now()})
            gate=self.commit(gate_path,{**self.backend.full_completion_likelihood_gate(records),**binding})
        try:
            if any(gate.get(k)!=v for k,v in binding.items()):
                raise ValueError('Full likelihood gate provenance differs')
            validate_full_gate(gate,records,self.config)
        except (ValueError,KeyError,TypeError) as error:
            self.state(state='numerical_failed',stage='full_completion_likelihood_gate',
                claims_allowed=False,reason=str(error),actual_optimizer_steps=0)
            return
        scores=self.existing('calibration/scores.json')
        if scores is None:
            rows=[]
            for i,path in enumerate(paths):
                self.admit()
                rows.append({'generation':f'calibration-{i}','generation_sha256':self.status['artifacts'][path],
                    **score(self.existing(path),self.existing('opponents.json')['selected'],
                            self.schedule['scoring_maps'],self.config)})
            scores=self.commit('calibration/scores.json',{'rows':rows,'seeds':self.schedule['scoring_maps'],
                'seal_sha256':self.status['artifacts']['calibration/sealed.json'],
                'full_likelihood_gate_sha256':self.status['artifacts'][gate_path],
                'opponents_sha256':self.status['artifacts']['opponents.json'],
                'prescribed_program_draws':8,'training_use':False})
        decision=cardinal.capability_decision(scores['rows'],self.config)
        decision.update(scores_sha256=self.status['artifacts']['calibration/scores.json'],
            full_likelihood_gate_sha256=self.status['artifacts'][gate_path],
            all_draws_retained=True,optimizer_steps=0,training_permitted_in_this_job=False)
        old=self.existing('calibration/decision.json')
        if old is None:self.commit('calibration/decision.json',decision)
        elif old!=decision:raise ValueError('Calibration decision differs')
        if self.checkpoint('unchanged-final') is None:
            self.save_checkpoint('unchanged-final','base',{'optimizer_steps':0,'outcome':'calibration_no_training'})
        final=self.checkpoint('unchanged-final');base=self.checkpoint('base')
        if (final['files']['adapter.safetensors']!=base['files']['adapter.safetensors']
                or {k:v for k,v in final['optimizer_identity'].items() if k!='files'} !=
                   {k:v for k,v in base['optimizer_identity'].items() if k!='files'}):
            raise ValueError('Calibration changed adapter, Adam, frozen base or RNG values')
        self.admit()
        self.state(state='complete',stage='complete',claims_allowed=False,calibration_complete=True,
            qualification_passed=decision['passed'],actual_optimizer_steps=0,learning_comparison_available=False)


def make_runtime_report(index, backend_runtime, actual_versions):
    """Record every verified dependency, including those omitted by the base backend."""
    if actual_versions != VERSIONS:
        raise ValueError('Exact CUDA versions differ')
    for name, version in backend_runtime.get('versions', {}).items():
        if name in VERSIONS and version.split('+')[0] != VERSIONS[name]:
            raise ValueError('Backend dependency report differs: '+name)
    return {**backend_runtime, 'model_index': index, 'versions': dict(actual_versions),
            'thinking': True, 'authoring_calls': 2, 'reasoning_effort':'low', 'calibration_only':True,
            'turn_native_modes': {'draft': stage_metadata('draft'), 'revision': stage_metadata('revision')},
            'sampling_batch': 'Left-padded variable native chat prompts, independent row RNG; checked against unpadded likelihoods'}


def worker(run,model_dir,lock_fds):
    journal.verify_lock_descriptors(run,lock_fds)
    manifest,status=verify_run(run);cfg=manifest['config']
    journal.execution_budget(cfg,status['execution_budget'])
    actual={name:importlib.metadata.version(name).split('+')[0] for name in VERSIONS}
    if actual!=VERSIONS:raise ValueError('Exact CUDA versions differ')
    import torch
    if torch.version.cuda!='12.8':raise ValueError('Pinned CUDA12.8 required')
    index_model(model_dir,Path(run)/'inputs/model-info.json')
    index=common.verify_model_index(model_dir,cfg)
    if index['authoritative_metadata_sha256']!=manifest['input_sha256']['inputs/model-info.json']:
        raise ValueError('Model metadata identity differs')
    backend=RevisionCudaPolicy(model_dir,cfg);engine=Calibration(run,backend)
    runtime=make_runtime_report(index,backend.runtime,actual)
    engine.commit('runtime.json',runtime);engine.initialize();engine.admit();engine.state(stage='numerical_smoke')
    # Mixed prompt lengths and native multi-turn prefixes exercise the newly
    # introduced padding/history path before any experiment rollout or update.
    contexts=[harness.initial_messages() for _ in range(2)]
    for i in range(1,2):
        contexts[i]=contexts[i]+[{'role':'assistant','content':"def act():\n    return 'rest'"},
            {'role':'user','content':'Unscored numerical fixture. Revise the draft. '+('Check your code. '*i)}]
    smoke=backend.mechanics_smoke({'batch_messages':contexts})
    smoke.update(source_checkpoint='base',source_metadata_sha256=engine.status['artifacts']['checkpoints/base/metadata.json'],
                 mixed_length_native_chat_prompts=True, authoring_stages=['draft','revision'])
    engine.commit('smoke-'+str(time.time_ns())+'.json',smoke)
    if not smoke['passed']:
        engine.state(state='numerical_failed',stage='numerical_smoke',claims_allowed=False);return engine.status
    engine.run_all();return engine.status


def launch(run,model_dir):
    run,model_dir=Path(run).resolve(),Path(model_dir).resolve()
    manifest,status=verify_run(run)
    if status['state'] in TERMINAL:return status
    with common.process_lock(LOCK) as global_fd,common.process_lock(run/'run.lock') as run_fd:
        status=common.load(run/'status.json')
        budget,timeout=journal.execution_budget(manifest['config'],status.get('execution_budget'))
        status.update(execution_budget=budget,updated=common.now());common.atomic_json(run/'status.json',status)
        child=subprocess.Popen([sys.executable,'-u','-m','selfplay.program_debug_revision_calibration','_worker','--run',str(run),
            '--model-dir',str(model_dir),'--lock-fds',f'{global_fd},{run_fd}'],cwd=ROOT,
            pass_fds=(global_fd,run_fd),start_new_session=True)
        try:
            code=child.wait(timeout=timeout)
            if code:raise RuntimeError(f'CUDA worker exited {code}; no automatic retry')
        except BaseException as error:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
            status=common.load(run/'status.json');status.update(state='stopped',updated=common.now(),claims_allowed=False,
                error_type=type(error).__name__,reason=str(error));common.atomic_json(run/'status.json',status)
            raise
    return common.load(run/'status.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','plan','run','_worker'))
    parser.add_argument('--run',type=Path);parser.add_argument('--model-dir',type=Path)
    parser.add_argument('--deployment',type=Path);parser.add_argument('--lock-fds')
    parser.add_argument('--model-info',type=Path,default=ROOT/'runs/cardinal-gpu-preflight/model-info.json')
    args=parser.parse_args()
    if args.command=='prepare':result=prepare(args.run,args.deployment,args.model_info)
    elif args.command=='plan':result=plan(gpu.public_deployment(common.load(args.deployment)))
    elif args.command=='run':result=launch(args.run,args.model_dir)
    else:result=worker(args.run,args.model_dir,args.lock_fds)
    print(common.canonical(result))


if __name__=='__main__':main()
