"""Frozen two-call thinking/test/debug trajectories and historical-league RL.

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
from selfplay import program_debug as harness
from selfplay.program_debug_cuda import DebugCudaPolicy, flatten_trajectories
from selfplay.program_cardinal_cuda import checkpoint_state

ROOT, LOCK = gpu.ROOT, gpu.LOCK
MODEL_REPO, MODEL_REVISION = gpu.MODEL_REPO, gpu.MODEL_REVISION
VERSIONS = {**gpu.VERSIONS, 'tokenizers': '0.23.3'}
index_model = gpu.index_model
INITIAL = ROOT / 'runs/cardinal-warmstart-gpu-v2'
TOKENIZER_INPUT = ROOT / 'runs/cardinal-debug-preflight-v1/tokenizer'
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'generation_config.json', 'config.json', 'chat_template.jinja')
SOURCES = list(dict.fromkeys(gpu.SOURCES + [
    'selfplay/program_debug.py', 'selfplay/program_debug_cuda.py', 'selfplay/program_debug_rl.py',
    'scripts/cloud/run_debug_job.py', 'scripts/cloud/bootstrap_debug.py',
    'scripts/debug_supervisor.py', 'scripts/runpod_debug.py',
    'scripts/runpod_cardinal.py', 'scripts/runpod_pilot.py', 'scripts/cloud/cardinal_guard.py',
    'scripts/cloud/cardinal-warmstart-requirements.txt',
    'scripts/analyze_program_cardinal_gpu.py',
    'selfplay/frontier_api.py', 'selfplay/__init__.py',
    'scripts/analyze_program_debug_rl.py']))
TERMINAL = {'complete', 'throughput_failed', 'budget_incomplete', 'numerical_failed', 'stopped', 'failed'}


def config(deployment):
    return {**gpu.config(deployment), 'kind': 'thinking_debug_league_rl',
        'condition': 'two-call-practice-debug-v1', 'seed': 225301,
        'thinking': True, 'max_new_tokens': 1536, 'max_context_tokens': 8192,
        'sample_seconds': 360, 'endpoint_choices': [25, 50, 100],
        'profile_attempt_cap': 3, 'authoring_calls': 2, 'practice_maps_per_trajectory': 2,
        'evaluation_batch_reserve_seconds': 6360, 'audit_export_reserve_seconds': 1200,
        'capability_samples': 0, 'minimum_productive_programs': 0,
        'minimum_productive_distinct_ir': 0, 'opponent_attempt_cap': 0,
        'scope': 'RL improvement beyond a fixed SFT checkpoint using the same two-call thinking/debug harness',
        'baseline': 'Exact prior SFT-step32 weights and zero-step RL Adam; no further SFT',
        'history_transform': 'Exact extracted final draft, or unmodified incomplete draft; raw reasoning retained in trajectory but not revision context',
        'reward': 'Final revised controller mean relative return; invalid or candidate interpreter faults=-1'}


def schedule():
    value = {
        'train_practice_maps': {str(i): [151000000+i*10+j for j in range(2)] for i in range(1, 101)},
        'train_maps': {str(i): [152000000+i*10+j for j in range(2)] for i in range(1, 101)},
        'eval_practice_maps': {str(i): [153000000+i*10+j for j in range(2)] for i in range(32)},
        'holdout_maps': list(range(154000000, 154000006)),
        'train_generation_seeds': {str(i): [161000000+i*100+j*2 for j in range(8)] for i in range(1, 101)},
        'eval_generation_seeds': [162000000+i*2 for i in range(32)],
        'revision_seed_rule': 'draft_seed+1; disjoint even/odd seeds within each independent trajectory'}
    # Pre-neural layout-only collision review found one duplicate among the
    # proposed families. Replacement is fixed before any sampled controller.
    value['train_maps']['22'][0] = 159000000
    return value


def plan(deployment):
    return {'kind': 'thinking_debug_league_rl_plan', 'config': config(deployment), 'schedule': schedule(),
        'authoring': 'Exactly two thinking-enabled calls: draft, deterministic practice feedback, final revision; no selection between draft and revision and no repair/retry.',
        'training': 'Eight independent two-call trajectories per group. Leave-one-out advantage times summed log probabilities of both complete sampled turns, including reasoning/EOS. Context and feedback tokens are not targets.',
        'numerics': 'Unchanged FP32 full-softmax distribution and .002 token/.02 sequence gates in eval and gradient modes; variable left-padded sampling additionally checked against unpadded teacher forcing.',
        'initialization': 'Exact prior zero-RL-step SFT baseline, original NF4 base and fresh Adam. No additional SFT, API generation, or successful-answer substitution.',
        'league': 'Nearest/value plus first two distinct verified teacher controllers initially; all identities labeled. Every five groups add first two productive distinct final revisions, retain at most six model controllers including initial two.',
        'weak_policies': 'No productive-play prerequisite. All zero-signal groups retained with unchanged adapter/Adam and zero actual optimizer steps.',
        'endpoint': 'Exactly three profile groups, then choose largest of25/50/100 fitting measured maximum time with1.5 safety factor and reserved evaluation/export. No score dependence.',
        'evaluation': 'Same two-call harness,32 paired trajectory seeds/checkpoint, six heldout maps, four frozen initial opponents, both seats. Seal all64 final outputs before scoring. Practice maps never overlap scoring maps.',
        'primary': 'Mean banked-point yield across all32 prescribed final revisions. Invalid outputs count zero yield with no invented games. Missing outputs mean incomplete, not zero.',
        'uncertainty': 'One training seed, exploratory paired controller-cluster intervals. No general or continual-improvement claim.',
        'budget': {'cumulative_cap_usd': 100, 'reservation_usd': deployment['reservation_usd'],
                   'hard_deadline_unix': deployment['deadline_unix']}}


def fingerprints(timetable, prior):
    known = set()
    def collect(value):
        if isinstance(value, dict):
            if 'seed' in value and 'layout' in value and 'sha256' in value:
                known.add(value['sha256'])
            if 'prior_seed_fingerprints' in value:
                known.update(value['prior_seed_fingerprints'].values())
            for child in value.values(): collect(child)
        elif isinstance(value, list):
            for child in value: collect(child)
    collect(prior)
    groups = {}
    for family in ('train_practice_maps', 'train_maps', 'eval_practice_maps', 'holdout_maps'):
        value = timetable[family]
        seeds = [s for row in value.values() for s in row] if isinstance(value, dict) else value
        groups[family] = []
        for seed in seeds:
            layout = journal.map_layout(seed)
            fingerprint = gpu.object_hash(layout)
            if fingerprint in known:
                raise ValueError('Observable map overlaps prior or another registered map')
            known.add(fingerprint)
            groups[family].append({'seed': seed, 'sha256': fingerprint, 'layout': layout})
    return {'families': groups, 'definition': 'Exact observable initial grid/site order/homes/capacity/energy/horizon',
            'prior_evidence_sha256': gpu.object_hash(prior)}


def initial_roster(corpus):
    selected = [{'id': name, 'origin': 'scripted', 'authorship': 'scripted_anchor',
        'compiled_ir': common.reference_programs()[name]} for name in ('nearest', 'value')]
    seen = set()
    for demo in corpus['demonstrations']:
        ir = demo['program']['compiled_ir']
        digest = common.compile_program(ir).digest
        if digest in seen: continue
        seen.add(digest)
        selected.append({'id': 'teacher-'+str(demo['index']), 'origin': 'model',
            'authorship': 'external_teacher_off_policy_opponent', 'compiled_ir': ir,
            'ir_sha256': digest, 'generation_sha256': demo['generation_sha256'],
            'source_checkpoint': 'external_teacher', 'source_path': 'inputs/teacher-corpus.json'})
        if len(selected) == 4: break
    if len(selected) != 4:
        raise ValueError('Two distinct verified historical teacher opponents required')
    return {'selected': selected, 'frozen': True, 'external_teacher_opponents_are_not_RL_samples': True}


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
    cfg, timetable = config(deployment), schedule()
    prior = common.load(INITIAL/'map-fingerprints.json')
    maps = fingerprints(timetable, prior)
    corpus = common.load(INITIAL/'inputs/teacher/corpus.json')
    if common.digest(INITIAL/'inputs/teacher/corpus.json') != initial_manifest['input_sha256']['inputs/teacher/corpus.json']:
        raise ValueError('Frozen teacher corpus changed')
    roster = initial_roster(corpus)
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
    copy(INITIAL/'inputs/teacher/corpus.json', 'inputs/teacher-corpus.json')
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
    if (manifest['kind'] != 'thinking_debug_league_rl' or manifest['config'] != config(deployment)
            or status['manifest_sha256'] != common.digest(run/'manifest.json')
            or common.load(run/'schedule.json') != schedule() or common.load(run/'plan.json') != plan(deployment)
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


def validate_batch(batch, contexts, seeds, cfg):
    records = batch.get('records', [])
    if len(records) != len(seeds) or len(seeds) != 8 or len(contexts) != 8:
        raise ValueError('Exactly eight complete contexts and batch rows required')
    for i, (row, context, seed) in enumerate(zip(records, contexts, seeds)):
        ids, lps, prefix = row.get('completion_ids'), row.get('sampled_token_logprobs'), row.get('prompt_ids')
        if (row.get('messages') != context or row.get('seed') != seed or row.get('sampling_batch_row') != i
                or row.get('sampling_batch_size') != 8 or row.get('thinking_enabled') is not True):
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
    classification, program = harness.classify_debug(record['text'], record['finish_reason'])
    if program is None:
        result = cardinal.score({'text': '', 'finish_reason': 'length'}, opponents, seeds, cfg)
    else:
        result = cardinal.score({'text': classification['final_answer'], 'finish_reason': 'eos'}, opponents, seeds, cfg)
    result['debug_classification'] = classification
    return result


def endpoint_decision(reports, remaining, cfg):
    if len(reports) != 3:
        raise ValueError('Exactly three prescribed profile groups required regardless of rewards')
    times = [r['elapsed_seconds'] for r in reports]
    if any(type(t) not in (int,float) or not math.isfinite(t) or t <= 0 for t in times):
        raise ValueError('Invalid measured group timing')
    reserve = cfg['audit_export_reserve_seconds']+cfg['evaluation_batch_reserve_seconds']
    projections = {str(n): max(times)*cfg['timing_safety_factor']*(n-3)+reserve for n in cfg['endpoint_choices']}
    fits = [n for n in cfg['endpoint_choices'] if projections[str(n)] <= remaining]
    return {'passed': bool(fits), 'endpoint': max(fits) if fits else None,
        'reason': 'Timing and immutable budget only; no reward, validity or heldout selection',
        'remaining_seconds': remaining, 'profile_groups': 3, 'maximum_measured_group_seconds': max(times),
        'safety_factor': cfg['timing_safety_factor'], 'reserved_final_and_export_seconds': reserve,
        'projected_remaining_seconds': projections}


class DebugCycle(gpu.GPUCycle):
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

    def ensure_raw_batch(self, batch_id, contexts, seeds, checkpoint):
        path = f'batches/{batch_id}.json'
        context = {'checkpoint': checkpoint, 'source_metadata_sha256': self.status['artifacts'][f'checkpoints/{checkpoint}/metadata.json'],
                   'contexts_sha256': gpu.object_hash(contexts), 'seeds': seeds}
        batch = self.existing(path)
        if batch is None:
            self.admit(); self.activate(checkpoint); self.state(stage='batch/'+batch_id)
            # Durable intent ensures an interrupted/ambiguous sample is never silently redrawn.
            intent = f'batches/{batch_id}-intent.json'
            if self.existing(intent) is not None:
                raise ValueError('Uncommitted sampled batch intent exists; preserve failure, never redraw')
            self.commit(intent, {**context, 'started': common.now()})
            sampled = self.backend.sample_messages_batch(contexts, seeds)
            # Retain the returned raw draw even if admission discovers a
            # mechanical inconsistency. Validation never triggers resampling.
            batch = self.commit(path, {**context, **sampled})
        if any(batch.get(k) != v for k,v in context.items()): raise ValueError('Batch provenance differs')
        validate_batch(batch, contexts, seeds, self.config)
        return batch, path

    def trajectories(self, batch_id, names, seeds, checkpoint, practice_maps):
        if len(names) != 8 or len(seeds) != 8 or len(practice_maps) != 8:
            raise ValueError('Eight fixed independent trajectories required')
        initial = harness.initial_messages()
        if initial != [{'role':'user','content':self.prompts['debug']}]: raise ValueError('Frozen prompt changed')
        drafts, draft_path = self.ensure_raw_batch(batch_id+'-draft', [initial]*8, seeds, checkpoint)
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
        revisions, revision_path = self.ensure_raw_batch(batch_id+'-revision', contexts, [s+1 for s in seeds], checkpoint)
        results = []
        for i, name in enumerate(names):
            draft, revision = drafts['records'][i], revisions['records'][i]
            feedback_path, _ = feedbacks[i]
            trajectory = {**revision, 'name': name, 'checkpoint': checkpoint,
                'source_metadata_sha256': self.status['artifacts'][f'checkpoints/{checkpoint}/metadata.json'],
                'prompt_condition': 'thinking_debug', 'authorship': 'fresh_learner_two_call_trajectory',
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

    def group(self, number, opponents):
        label, folder = f'cycle{number}', f'groups/{number}'
        if self.checkpoint(label): return self.finish_group_timing(number, self.checkpoint(label)['report'])
        if self.existing(folder+'/timing-start.json') is None:
            self.commit(folder+'/timing-start.json', {'started': common.now(), 'cycle': number})
        source = 'base' if number == 1 else f'cycle{number-1}'
        self.activate(source)
        records = self.trajectories(f'train-{number}', [f'train-cycle{number}-{i}' for i in range(8)],
            self.schedule['train_generation_seeds'][str(number)], source,
            [self.schedule['train_practice_maps'][str(number)]]*8)
        hashes = {r['name']:self.status['artifacts'][f"generations/{r['name']}.json"] for r in records}
        scores = self.existing(folder+'/scores.json')
        if scores is None:
            rows = []
            for record in records:
                self.admit()
                rows.append({'generation':record['name'],'generation_sha256':hashes[record['name']],
                    **score(record, opponents['selected'], self.schedule['train_maps'][str(number)], self.config)})
            rewards = [r['reward'] for r in rows]
            advantages = journal.advantages(rewards)
            scores = self.commit(folder+'/scores.json', {'cycle':number,'source_checkpoint':source,
                'source_metadata_sha256':self.status['artifacts'][f'checkpoints/{source}/metadata.json'],
                'opponents_sha256':self.status['artifacts'][folder+'/opponents.json'],
                'seeds':self.schedule['train_maps'][str(number)], 'rows':rows,'rewards':rewards,
                'advantages':advantages,'zero_signal':not any(advantages)})
        if ({r['generation']:r['generation_sha256'] for r in scores['rows']} != hashes
                or scores['source_checkpoint'] != source or journal.advantages(scores['rewards']) != scores['advantages']
                or scores['opponents_sha256'] != self.status['artifacts'][folder+'/opponents.json']):
            raise ValueError('Group on-policy provenance differs')
        report = {'cycle':number,'source_checkpoint':source,'source_metadata_sha256':scores['source_metadata_sha256'],
            'scores_sha256':self.status['artifacts'][folder+'/scores.json'],'generation_sha256':hashes,
            'turn_sha256':{r['name']:r['turn_sha256'] for r in records},
            'rewards':scores['rewards'],'advantages':scores['advantages'],
            'training_tokens':sum(r['training_tokens'] for r in records),'objective_divisor':8,
            'all_runtime_invalid':not any(r['runtime_valid'] for r in scores['rows'])}
        copy_from = None
        if scores['zero_signal']:
            report.update(outcome='skipped_zero_variance',optimizer_steps=0,adapter_l2_delta=0.0,
                          likelihood_gate_path=None,likelihood_gate_sha256=None)
            copy_from = self.run/'checkpoints'/source
        else:
            self.admit(); self.state(stage=folder+'/likelihood')
            flat, _ = flatten_trajectories(records, scores['advantages'])
            gate = self.backend.likelihood_gate(flat)
            gate.update(generation_sha256=hashes,turn_sha256=report['turn_sha256'],source_checkpoint=source,
                        source_metadata_sha256=report['source_metadata_sha256'])
            path = folder+'/gate-'+uuid.uuid4().hex+'.json'
            self.commit(path,gate)
            if not gate['passed']: raise ValueError('Full two-turn likelihood gate failed')
            self.admit(); self.state(stage=folder+'/optimizer')
            update = self.backend.update_trajectories(records,scores['advantages'])
            if update['optimizer_steps'] != 1 or update['adapter_l2_delta'] <= 0: raise ValueError('No genuine update')
            report.update(update,outcome='updated',likelihood_gate_path=path,likelihood_gate_sha256=self.status['artifacts'][path])
        self.save_checkpoint(label,source,report,copy_from=copy_from,update_relative=folder+'/update.json')
        return self.finish_group_timing(number,report)

    def evaluate(self, opponents, endpoint):
        endpoints, paths = ('base',f'cycle{endpoint}'), []
        for label in endpoints:
            for offset in range(0,32,8):
                names=[f'eval-{label}-{i}' for i in range(offset,offset+8)]
                self.trajectories(f'eval-{label}-{offset//8}',names,
                    self.schedule['eval_generation_seeds'][offset:offset+8],label,
                    [self.schedule['eval_practice_maps'][str(i)] for i in range(offset,offset+8)])
                paths.extend(f'generations/{name}.json' for name in names)
        seal={'checkpoints':{e:self.status['artifacts'][f'checkpoints/{e}/metadata.json'] for e in endpoints},
            'generations':{p:self.status['artifacts'][p] for p in paths},'draws':64,'sampled_turns':128,
            'endpoint_selection_sha256':self.status['artifacts']['endpoint-selection.json'],'scoring_allowed':True}
        old=self.existing('evaluation/sealed.json')
        if old is None:self.commit('evaluation/sealed.json',seal)
        elif old!=seal:raise ValueError('Evaluation seal differs')
        for label in endpoints:
            path=f'evaluation/{label}/holdout.json'
            if self.existing(path) is not None:continue
            rows=[]
            for i in range(32):
                self.admit(); name=f'eval-{label}-{i}'
                record=self.existing(f'generations/{name}.json')
                rows.append({'generation':name,'generation_sha256':self.status['artifacts'][f'generations/{name}.json'],
                    **score(record,opponents['selected'],self.schedule['holdout_maps'],self.config)})
            self.commit(path,{'checkpoint':label,'prompt_condition':'thinking_debug','split':'holdout','training_use':False,
                'seeds':self.schedule['holdout_maps'],'opponents':opponents['selected'],
                'opponents_sha256':self.status['artifacts']['opponents.json'],
                'seal_sha256':self.status['artifacts']['evaluation/sealed.json'],'prescribed_program_draws':32,'scores':rows})
        updates=sum(self.checkpoint(f'cycle{i}')['report']['optimizer_steps'] for i in range(1,endpoint+1))
        self.admit()
        self.state(state='complete',stage='complete',claims_allowed=True,attempted_groups=endpoint,actual_optimizer_steps=updates)

    def run_all(self):
        self.initialize()
        opponents=self.existing('opponents.json')
        selection=self.existing('endpoint-selection.json')
        if selection is None:
            for number in range(1,4):self.group(number,self.roster(number,opponents))
            _,remaining=journal.execution_budget(self.config,self.status['execution_budget'])
            reports=[self.existing(f'groups/{i}/timing.json') for i in range(1,4)]
            selection=endpoint_decision(reports,remaining,self.config)
            selection['profile_timing_sha256']={str(i):self.status['artifacts'][f'groups/{i}/timing.json'] for i in range(1,4)}
            self.commit('endpoint-selection.json',selection)
        if not selection['passed']:
            self.state(state='throughput_failed',stage='endpoint_admission',claims_allowed=False);return
        endpoint=selection['endpoint']
        for number in range(1,endpoint+1):
            if self.checkpoint(f'cycle{number}') is None:
                _,remaining=journal.execution_budget(self.config,self.status['execution_budget'])
                if remaining < self.config['audit_export_reserve_seconds']+self.config['evaluation_batch_reserve_seconds']:
                    self.state(state='budget_incomplete',stage=f'before_cycle{number}',claims_allowed=False);return
            self.group(number,self.roster(number,opponents));self.refresh(number,opponents)
        self.evaluate(opponents,endpoint)


def make_runtime_report(index, backend_runtime, actual_versions):
    """Record every verified dependency, including those omitted by the base backend."""
    if actual_versions != VERSIONS:
        raise ValueError('Exact CUDA versions differ')
    for name, version in backend_runtime.get('versions', {}).items():
        if name in VERSIONS and version.split('+')[0] != VERSIONS[name]:
            raise ValueError('Backend dependency report differs: '+name)
    return {**backend_runtime, 'model_index': index, 'versions': dict(actual_versions),
            'thinking': True, 'authoring_calls': 2,
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
    backend=DebugCudaPolicy(model_dir,cfg);engine=DebugCycle(run,backend)
    runtime=make_runtime_report(index,backend.runtime,actual)
    engine.commit('runtime.json',runtime);engine.initialize();engine.admit();engine.state(stage='numerical_smoke')
    # Mixed prompt lengths and native multi-turn prefixes exercise the newly
    # introduced padding/history path before any experiment rollout or update.
    contexts=[harness.initial_messages() for _ in range(8)]
    for i in range(1,8):
        contexts[i]=contexts[i]+[{'role':'assistant','content':"def act():\n    return 'rest'"},
            {'role':'user','content':'Unscored numerical fixture. Revise the draft. '+('Check your code. '*i)}]
    smoke=backend.mechanics_smoke({'batch_messages':contexts})
    smoke.update(source_checkpoint='base',source_metadata_sha256=engine.status['artifacts']['checkpoints/base/metadata.json'],
                 mixed_length_native_chat_prompts=True)
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
        child=subprocess.Popen([sys.executable,'-u','-m','selfplay.program_debug_rl','_worker','--run',str(run),
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
