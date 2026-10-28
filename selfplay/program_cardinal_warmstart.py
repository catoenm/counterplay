"""Verified teacher SFT followed by fresh on-policy cardinal league RL.

This separate protocol preserves previous experiment sources. Preparation only
copies verified CPU artifacts. Explicit run is the sole neural entry point.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import uuid

from selfplay import program_cardinal_gpu as gpu
from selfplay.program_cardinal_sft import CardinalSftPolicy, SFT_LEARNING_RATE

common, journal, ROOT = gpu.common, gpu.journal, gpu.ROOT
MODEL_REPO, MODEL_REVISION = gpu.MODEL_REPO, gpu.MODEL_REVISION
VERSIONS = {**gpu.VERSIONS, 'tokenizers': '0.23.3'}
public_deployment, index_model = gpu.public_deployment, gpu.index_model
SOURCES = list(dict.fromkeys(gpu.SOURCES + [
    'selfplay/program_cardinal_teacher.py', 'selfplay/program_model_screen.py',
    'scripts/collect_cardinal_teacher.py',
    'selfplay/program_cardinal_teacher_low.py', 'scripts/collect_cardinal_teacher_low.py',
    'selfplay/program_cardinal_teacher_none.py', 'scripts/collect_cardinal_teacher_none.py',
    'selfplay/program_cardinal_teacher_named.py', 'scripts/collect_cardinal_teacher_named.py',
    'selfplay/program_cardinal_sft.py', 'selfplay/program_cardinal_warmstart.py',
    'scripts/cloud/run_cardinal_warmstart_job.py',
    'scripts/cloud/bootstrap_cardinal_warmstart.py',
    'scripts/cloud/cardinal-warmstart-requirements.txt',
    'scripts/cardinal_warmstart_supervisor.py',
    'scripts/analyze_program_cardinal_gpu.py',
    'scripts/analyze_program_cardinal_warmstart.py']))
SFT_SHUFFLE_SEED = 115000000
TERMINAL = {'complete', 'capability_failed', 'throughput_or_signal_failed',
            'budget_incomplete', 'numerical_failed'}


def schedule():
    result = {'teacher_maps': list(range(91000000, 91000006)),
        'calibration_maps': list(range(92000000, 92000006)),
        'train_maps': {str(i): [93000000+i*100+j for j in range(2)] for i in range(1, 101)},
        'holdout_maps': list(range(94000000, 94000006)),
        'capability_generation_seeds': list(range(112000000, 112000008)),
        'train_generation_seeds': {str(i): [113000000+i*100+j for j in range(8)] for i in range(1, 101)},
        'eval_generation_seeds': list(range(114000000, 114000032)),
        'map_seed_replacements': [{'family': 'train_maps', 'group': 45, 'index': 0,
            'proposed_seed': 93004500, 'replacement_seed': 93004502, 'prior_equal_layout_seed': 8203401,
            'reason': 'Exact observable-map collision detected before teacher collection or new neural work.'}]}
    # CPU layout preflight, before any samples, found 93004500 == old8203401.
    result['train_maps']['45'][0] = 93004502
    return result


def sft_schedule(indices):
    if len(indices) != 16 or len(set(indices)) != 16 or indices != sorted(indices):
        raise ValueError('Exactly16 selected teacher indices in fixed order required')
    result = []
    for epoch in range(8):
        order = list(indices)
        random.Random(SFT_SHUFFLE_SEED + epoch).shuffle(order)
        for offset in range(0, 16, 4):
            result.append({'step': len(result)+1, 'epoch': epoch+1,
                'shuffle_seed': SFT_SHUFFLE_SEED+epoch, 'teacher_indices': order[offset:offset+4]})
    return result


def config(deployment):
    return {**gpu.config(deployment), 'kind': 'cardinal_teacher_warmstart_league_rl',
        'condition': 'verified-teacher-named-cardinal-gpu-v1', 'seed': 125301,
        'calibration_seed_start': 92000000, 'train_seed_start': 93000000,
        'holdout_seed_start': 94000000, 'sft_learning_rate': SFT_LEARNING_RATE,
        'sft_updates': 32, 'sft_batch_size': 4, 'sft_epochs': 8,
        'sft_completion_cap': 700, 'sft_shuffle_seed': SFT_SHUFFLE_SEED,
        'baseline': 'fixed verified-teacher SFT adapter, fresh zero-step RL Adam',
        'scope': 'Incremental on-policy league RL beyond a fixed supervised teacher warmstart'}


def plan(deployment, corpus_sha256):
    result = gpu.plan(deployment)
    result.update(kind='cardinal_teacher_warmstart_league_rl_plan', config=config(deployment),
        schedule=schedule(), teacher_corpus_sha256=corpus_sha256,
        model='Original pinned HF NF4 FP32 base; supervised teacher warmstart followed by fresh on-policy learner rollouts.')
    result['teacher'] = {'attempts': 32, 'selected_demonstrations': 16,
        'condition': 'reasoning-disabled requested; separate fresh cohort with uniform teacher-only local-variable naming clarification after the previous32 outputs were evaluated',
        'delivery_gate': 'First prescribed attempt must finish stop with nonempty final code before calls1..31; no strategy scoring, replacement or extra attempt.',
        'generation_seeds': list(range(141000000, 141000032)),
        'map_reuse': 'Teacher qualification maps reused after observation; learner calibration, training and final evaluation maps remain unobserved before this experiment.',
        'selection': 'First16 productive distinct IRs; all32 attempts retained and independently replay verified.',
        'completion_cap_including_eos': 700, 'verification_games': 24,
        'criterion': 'Zero candidate-seat interpreter faults and positive banking in at least12/24 teacher-only games.',
        'phase_api_cap_usd': 2, 'training_data': 'Explicit off-policy supervised teacher targets, never RL trajectories.',
        'learner_prompt': 'Clean specification; teacher design hints and reasoning are excluded.'}
    result['sft'] = {'updates': 32, 'epochs': 8, 'batch_size': 4, 'learning_rate': 5e-5,
        'gradient_clip': .5, 'shuffle_seed': SFT_SHUFFLE_SEED,
        'objective': 'Mean of four per-program mean completion-token cross-entropies, including actual EOS; prompt masked.',
        'checkpoints': 'Initial plus all32 adapter/Adam states retained separately under warmstart/checkpoints.',
        'numerics': 'Every update compares full teacher-forced eval/train log-probability vectors; finite nonzero gradients and actual deltas.',
        'transition': 'After exactly32 updates, preserve exact adapter file bytes; replace Adam with empty moments/step0/LR1e-5. Both optimizer states retained.',
        'fixed': 'No adaptive epochs, demonstration replacement, or relaxed post-SFT capability criterion.'}
    result['map_preflight_override'] = {'group': 45, 'rejected_seed': 93004500,
        'replacement_seed': 93004502, 'colliding_prior_seed': 8203401,
        'reason': 'Exact observable map collision found on CPU before any samples; next unused family seed.'}
    result['training']['initial_optimizer'] = 'Fresh RL Adam step0/LR1e-5 on exact SFT-step32 adapter; no supervised moments inherited.'
    result['training']['profile'] += ' Warmstart/setup time has already consumed the immutable remaining budget.'
    result['evaluation']['endpoints'] = ['base (fixed SFT baseline)', 'timing-selected cycle50 or cycle100']
    result['evaluation']['interpretation'] = 'Estimate subsequent RL effect beyond SFT; no pure self-play-from-base claim. No original-base arm.'
    return result


def validate_configuration(cfg, timetable, deployment):
    if cfg != config(deployment) or timetable != schedule():
        raise ValueError('Warmstart configuration/schedule differs from preregistration')
    families = [timetable['teacher_maps'], timetable['calibration_maps'], timetable['holdout_maps'],
        [seed for row in timetable['train_maps'].values() for seed in row]]
    flattened = [seed for family in families for seed in family]
    if len(flattened) != len(set(flattened)):
        raise ValueError('Teacher, training and heldout map seeds overlap')


def teacher_evidence(folder, replay=True, require_active=True):
    """The collector verifies all32 attempts; the run then binds every byte."""
    from scripts import collect_cardinal_teacher_named as collector
    return collector.verify_corpus(folder, replay=replay, require_active=require_active)


def verify_teacher_manifest_binding(corpus, teacher_plan, timetable):
    if teacher_plan.get('kind') != 'verified_cardinal_teacher_named_corpus':
        raise ValueError('This run requires the distinct naming-clarified teacher cohort')
    if (corpus['state'] != 'ready_for_sft' or corpus['all_attempts'] != 32
            or len(corpus['demonstrations']) != 16 or len(corpus['selected_indices']) != 16
            or [r['index'] for r in corpus['demonstrations']] != corpus['selected_indices']):
        raise ValueError('Full admitted teacher corpus required before GPU preparation')
    expected = {**timetable, 'capability_maps': timetable['calibration_maps'],
        'teacher_generation_seeds': list(range(141000000, 141000032))}
    del expected['calibration_maps']
    if teacher_plan['schedule'] != expected:
        raise ValueError('Teacher and learner frozen map schedules differ')
    if (teacher_plan['learner']['model'], teacher_plan['learner']['revision']) != (MODEL_REPO, MODEL_REVISION):
        raise ValueError('Teacher tokenizer model/revision differs')
    for row in corpus['demonstrations']:
        if not 2 <= len(row['completion_ids']) <= 700:
            raise ValueError('Teacher token bound differs')


def prepare(run, deployment_path, model_info_path, teacher_run):
    run, teacher_run = Path(run), Path(teacher_run)
    if run.exists():
        raise FileExistsError('New warmstart run directory required')
    deployment = public_deployment(common.load(deployment_path))
    metadata = common.load(model_info_path)
    if (metadata.get('sha'), metadata.get('id')) != (MODEL_REVISION, MODEL_REPO):
        raise ValueError('Authoritative HF metadata differs')
    teacher_evidence(teacher_run, replay=True)
    corpus, teacher_plan = common.load(teacher_run/'corpus.json'), common.load(teacher_run/'plan.json')
    timetable, cfg = schedule(), config(deployment)
    verify_teacher_manifest_binding(corpus, teacher_plan, timetable)
    for relative in SOURCES:
        if not (ROOT/relative).is_file():
            raise ValueError('Source not ready for freeze: '+relative)
    run.mkdir(parents=True)
    (run/'inputs').mkdir()
    # Self-contained copy includes failures and all replay evidence, not just selected targets.
    shutil.copytree(teacher_run, run/'inputs/teacher', ignore=shutil.ignore_patterns('*.lock', '__pycache__'))
    teacher_evidence(run/'inputs/teacher', replay=False)
    shutil.copyfile(model_info_path, run/'inputs/model-info.json')
    shutil.copyfile(teacher_run/'map-fingerprints.json', run/'map-fingerprints.json')
    (run/'prompts').mkdir()
    shutil.copyfile(teacher_run/'prompts/learner.txt', run/'prompts/cardinal.txt')
    if (run/'prompts/cardinal.txt').read_text() != gpu.cardinal_model_spec():
        raise ValueError('Teacher learner prompt differs from clean frozen cardinal specification')
    common.atomic_json(run/'deployment.json', deployment)
    common.atomic_json(run/'schedule.json', timetable)
    common.atomic_json(run/'sft-schedule.json', sft_schedule(corpus['selected_indices']))
    common.atomic_json(run/'plan.json', plan(deployment, common.digest(run/'inputs/teacher/corpus.json')))
    inputs = {str(path.relative_to(run)): common.digest(path)
        for path in sorted((run/'inputs').rglob('*')) if path.is_file()}
    inputs.update({p: common.digest(run/p) for p in ('deployment.json', 'prompts/cardinal.txt', 'sft-schedule.json')})
    sources = {}
    for relative in SOURCES:
        target = run/'source'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, target)
        sources[relative] = common.digest(target)
    manifest = {'format_version': 1, 'kind': cfg['kind'], 'created': common.now(), 'config': cfg,
        'plan_sha256': common.digest(run/'plan.json'), 'schedule_sha256': common.digest(run/'schedule.json'),
        'map_fingerprints_sha256': common.digest(run/'map-fingerprints.json'),
        'source_sha256': sources, 'input_sha256': inputs,
        'prompt_sha256': {'cardinal': inputs['prompts/cardinal.txt']},
        'expected_installed_versions': VERSIONS, 'claims_require_complete': True,
        'baseline': cfg['baseline'], 'teacher_corpus_sha256': inputs['inputs/teacher/corpus.json']}
    common.atomic_json(run/'manifest.json', manifest)
    common.atomic_json(run/'status.json', {'manifest_sha256': common.digest(run/'manifest.json'),
        'artifacts': {}, 'state': 'prepared', 'updated': common.now(), 'stage': 'prepared', 'claims_allowed': False})
    verify_run(run)
    return manifest


def verify_run(run, require_active=True):
    run = Path(run)
    manifest, status = common.load(run/'manifest.json'), common.load(run/'status.json')
    deployment = public_deployment(common.load(run/'deployment.json'))
    if manifest['kind'] != 'cardinal_teacher_warmstart_league_rl' or status['manifest_sha256'] != common.digest(run/'manifest.json'):
        raise ValueError('Warmstart manifest identity differs')
    validate_configuration(manifest['config'], common.load(run/'schedule.json'), deployment)
    corpus = common.load(run/'inputs/teacher/corpus.json')
    teacher_plan = common.load(run/'inputs/teacher/plan.json')
    verify_teacher_manifest_binding(corpus, teacher_plan, schedule())
    if (common.load(run/'plan.json') != plan(deployment, manifest['teacher_corpus_sha256'])
            or common.load(run/'sft-schedule.json') != sft_schedule(corpus['selected_indices'])
            or set(manifest['source_sha256']) != set(SOURCES)):
        raise ValueError('Warmstart plan/source/schedule differs')
    expected = {'plan.json': manifest['plan_sha256'], 'schedule.json': manifest['schedule_sha256'],
        'map-fingerprints.json': manifest['map_fingerprints_sha256'], **manifest['input_sha256'], **status['artifacts']}
    if manifest['teacher_corpus_sha256'] != manifest['input_sha256']['inputs/teacher/corpus.json']:
        raise ValueError('Teacher corpus identity differs')
    for relative, digest in expected.items():
        if common.digest(run/relative) != digest:
            raise ValueError('Committed artifact changed: '+relative)
    for relative, digest in manifest['source_sha256'].items():
        if common.digest(run/'source'/relative) != digest or (require_active and common.digest(ROOT/relative) != digest):
            raise ValueError('Archived/active source changed: '+relative)
    return manifest, status


def prompt_measurement(tokenizer, prompt, cfg):
    template = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(template, add_special_tokens=False)
    if not ids or any(type(i) is not int for i in ids):
        raise ValueError('Tokenizer did not return actual prompt token IDs')
    return {'prompt_tokens': len(ids), 'prompt_ids': ids, 'prompt_template': template,
        'method': 'rendered chat template then actual tokenizer.encode; never len(BatchEncoding)',
        'completion_cap': cfg['max_new_tokens'], 'context_cap': cfg['max_context_tokens'],
        'admitted': len(ids)+cfg['max_new_tokens'] <= cfg['max_context_tokens']}


class WarmstartCycle(gpu.GPUCycle):
    def __init__(self, run, backend):
        self.run, self.backend = Path(run), backend
        self.manifest, self.status = verify_run(self.run)
        self.config = self.manifest['config']
        self.schedule = common.load(self.run/'schedule.json')
        self.prompts = {'cardinal': (self.run/'prompts/cardinal.txt').read_text()}
        self.corpus = common.load(self.run/'inputs/teacher/corpus.json')
        self.sft_schedule = common.load(self.run/'sft-schedule.json')

    def warm_checkpoint(self, step):
        relative = f'warmstart/checkpoints/step{step}/metadata.json'
        metadata = self.existing(relative)
        if metadata is None:
            return None
        folder = self.run/f'warmstart/checkpoints/step{step}'
        if (metadata['step'] != step or metadata['complete'] is not True
                or metadata['manifest_sha256'] != self.status['manifest_sha256']
                or set(metadata['files']) != set(gpu.FILES)):
            raise ValueError('Supervised checkpoint metadata differs')
        for name, digest in metadata['files'].items():
            if (common.digest(folder/name) != digest
                    or self.status['artifacts'].get(f'warmstart/checkpoints/step{step}/{name}') != digest):
                raise ValueError('Supervised checkpoint bytes differ')
        actual = gpu.checkpoint_state(folder, expected_learning_rate=SFT_LEARNING_RATE)
        if actual != metadata['optimizer_identity'] or actual['step'] != step:
            raise ValueError('Supervised optimizer identity/step differs')
        return metadata

    def activate_warm(self, step):
        metadata = self.warm_checkpoint(step)
        if metadata is None or self.backend.config['learning_rate'] != SFT_LEARNING_RATE:
            raise ValueError('Missing SFT checkpoint or wrong optimizer phase')
        self.backend.load_checkpoint(self.run/f'warmstart/checkpoints/step{step}', metadata['files'])
        return metadata

    def save_warm_checkpoint(self, step, report=None):
        if self.warm_checkpoint(step):
            raise ValueError('Supervised checkpoint already committed')
        if step and (report is None or report['optimizer_steps'] != 1):
            raise ValueError('Each prescribed supervised update must be genuine')
        folder = f'warmstart/checkpoints/step{step}'
        final = self.run/folder
        if final.exists():
            final.rename(final.with_name(final.name+'.incomplete-'+uuid.uuid4().hex))
        temporary = final.with_name('.'+final.name+'.pending-'+uuid.uuid4().hex)
        temporary.parent.mkdir(parents=True, exist_ok=True)
        files = self.backend.save_checkpoint(temporary)
        identity = gpu.checkpoint_state(temporary, expected_learning_rate=SFT_LEARNING_RATE)
        if identity['step'] != step:
            raise ValueError('Saved supervised Adam step differs')
        parent_hash = self.status['artifacts'][f'warmstart/checkpoints/step{step-1}/metadata.json'] if step else None
        metadata = {'step': step, 'parent_step': step-1 if step else None,
            'parent_metadata_sha256': parent_hash, 'files': files, 'report': report,
            'optimizer_identity': identity, 'complete': True, 'manifest_sha256': self.status['manifest_sha256'],
            'teacher_corpus_sha256': self.manifest['teacher_corpus_sha256'],
            'model_repo': MODEL_REPO, 'model_revision': MODEL_REVISION}
        common.atomic_json(temporary/'metadata.json', metadata)
        temporary.rename(final)
        transaction = {folder+'/'+name: digest for name, digest in files.items()}
        transaction[folder+'/metadata.json'] = common.digest(final/'metadata.json')
        if step:
            path = f'warmstart/updates/{step}.json'
            self.existing(path)
            common.atomic_json(self.run/path, report)
            transaction[path] = common.digest(self.run/path)
        self.status['artifacts'].update(transaction)
        self.state()
        return metadata

    def teacher_records(self):
        output = {}
        for demo in self.corpus['demonstrations']:
            index = demo['index']
            prompt, text = [message['content'] for message in demo['messages']]
            if prompt != self.prompts['cardinal']:
                raise ValueError('Teacher target prompt differs from clean learner specification')
            record = self.backend.teacher_record(prompt, text)
            if record['completion_ids'] != demo['completion_ids']:
                raise ValueError('Actual GPU tokenizer differs from verified teacher target IDs')
            record.update(teacher_index=index, corpus_sha256=self.manifest['teacher_corpus_sha256'],
                teacher_generation_sha256=demo['generation_sha256'],
                teacher_program=demo['program'])
            relative = f'warmstart/targets/{index}.json'
            saved = self.existing(relative)
            if saved is None:
                self.commit(relative, record)
            elif saved != record:
                raise ValueError('Frozen supervised target changed')
            output[index] = record
        return output

    def ensure_warm_initial(self):
        if self.warm_checkpoint(0) is None:
            if self.backend.config['learning_rate'] != SFT_LEARNING_RATE:
                raise ValueError('Initial supervised optimizer LR differs')
            self.save_warm_checkpoint(0)

    def warmstart(self):
        if self.checkpoint('base'):
            self.activate('base')
            return
        self.ensure_warm_initial()
        records = self.teacher_records()
        for row in self.sft_schedule:
            step = row['step']
            if self.warm_checkpoint(step):
                continue
            self.admit()
            self.activate_warm(step-1)
            self.state(stage=f'warmstart/update{step}')
            batch = [records[index] for index in row['teacher_indices']]
            report = self.backend.sft_update(batch)
            if (report['optimizer_steps'] != 1 or report['adapter_l2_delta'] <= 0
                    or report['adam_step_before'] != step-1 or report['adam_step_after'] != step
                    or report['learning_rate'] != SFT_LEARNING_RATE):
                raise ValueError('Supervised update failed actual step/delta checks')
            report.update(step=step, schedule=row,
                source_metadata_sha256=self.status['artifacts'][f'warmstart/checkpoints/step{step-1}/metadata.json'],
                target_artifact_sha256={str(i): self.status['artifacts'][f'warmstart/targets/{i}.json'] for i in row['teacher_indices']},
                teacher_corpus_sha256=self.manifest['teacher_corpus_sha256'])
            self.save_warm_checkpoint(step, report)
        self.admit()
        self.activate_warm(32)
        transition = self.backend.reset_optimizer(learning_rate=self.config['learning_rate'])
        transition.update(source_checkpoint='warmstart/step32', optimizer_steps=0,
            source_metadata_sha256=self.status['artifacts']['warmstart/checkpoints/step32/metadata.json'])
        self.save_rl_base(transition)
        self.activate('base')

    def save_rl_base(self, transition):
        """Atomically commit reset and base only after adapter file equality."""
        if self.checkpoint('base'):
            raise ValueError('RL base already committed')
        source = self.warm_checkpoint(32)
        if source is None:
            raise ValueError('Exactly32 committed supervised updates required')
        final = self.run/'checkpoints/base'
        if final.exists():
            final.rename(final.with_name('base.incomplete-'+uuid.uuid4().hex))
        temporary = final.with_name('.base.pending-'+uuid.uuid4().hex)
        temporary.parent.mkdir(parents=True, exist_ok=True)
        files = self.backend.save_checkpoint(temporary)
        identity = self.inspect_checkpoint(temporary)
        if (identity['step'] != 0 or identity['moments']
                or files['adapter.safetensors'] != source['files']['adapter.safetensors']):
            raise ValueError('RL optimizer reset changed adapter bytes or inherited Adam state')
        transition.update(adapter_file_sha256=files['adapter.safetensors'],
            adapter_file_equality_verified=True, adapter_file_equality_still_required=False,
            sft_optimizer_identity=source['optimizer_identity'], rl_optimizer_identity=identity)
        metadata = {'label': 'base', 'parent': None, 'files': files, 'report': transition,
            'optimizer_identity': identity, 'complete': True, 'manifest_sha256': self.status['manifest_sha256'],
            'model_repo': MODEL_REPO, 'model_revision': MODEL_REVISION,
            'baseline': self.config['baseline']}
        common.atomic_json(temporary/'metadata.json', metadata)
        temporary.rename(final)
        path = 'warmstart/optimizer-transition.json'
        self.existing(path)
        common.atomic_json(self.run/path, transition)
        self.status['artifacts'].update({**{f'checkpoints/base/{n}': h for n, h in files.items()},
            'checkpoints/base/metadata.json': common.digest(final/'metadata.json'), path: common.digest(self.run/path)})
        self.state()
        return metadata

    def initialize(self):
        self.warmstart()

    def smoke(self, phase, source_path):
        self.admit()
        self.state(stage=phase+'/numerical_smoke')
        before = dict(self.status['artifacts'])
        try:
            report = self.backend.mechanics_smoke(self.prompts['cardinal'])
        except Exception as error:
            report = {'passed': False, 'training_use': False, 'optimizer_steps': 0,
                'error_type': type(error).__name__, 'reason': str(error)[:2000], 'incomplete_diagnostic': True}
        report.update(phase=phase, source_checkpoint=source_path,
            source_metadata_sha256=before[source_path])
        # Checkpoint files are immutable throughout either unscored smoke.
        for relative, digest in before.items():
            if '/checkpoints/' in relative or relative.startswith('checkpoints/'):
                if common.digest(self.run/relative) != digest:
                    report.update(passed=False, reason='Smoke changed a committed checkpoint file')
        self.commit(phase+'/smoke-'+str(time.time_ns())+'.json', report)
        if not report['passed']:
            self.state(state='numerical_failed', stage=phase+'/numerical_smoke', claims_allowed=False)
        return report['passed']

    def run_all(self):
        self.initialize()
        if not self.smoke('rl-admission', 'checkpoints/base/metadata.json'):
            return
        # Inherited initialize merely reactivates the exact committed RL base.
        # No capability draw occurs until the fresh post-SFT smoke passes.
        super().run_all()


def worker(run, model_dir, lock_fds):
    journal.verify_lock_descriptors(run, lock_fds)
    manifest, status = verify_run(run)
    cfg = manifest['config']
    if 'execution_budget' not in status:
        raise ValueError('Parent must commit durable execution budget')
    journal.execution_budget(cfg, status['execution_budget'])
    actual = {name: importlib.metadata.version(name).split('+')[0] for name in VERSIONS}
    if actual != VERSIONS:
        raise ValueError('Installed dependency versions differ')
    import torch
    if torch.version.cuda != '12.8':
        raise ValueError('Pinned Torch CUDA12.8 is required')
    index_model(model_dir, Path(run)/'inputs/model-info.json')
    index = common.verify_model_index(model_dir, cfg)
    if index['authoritative_metadata_sha256'] != manifest['input_sha256']['inputs/model-info.json']:
        raise ValueError('Staged model is not bound to authoritative metadata')
    engine = WarmstartCycle(run, None)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    measurement = prompt_measurement(tokenizer, engine.prompts['cardinal'], cfg)
    previous = engine.existing('prompt-measurement.json')
    if previous is None:
        engine.commit('prompt-measurement.json', measurement)
    elif previous != measurement:
        raise ValueError('Actual tokenizer prompt measurement changed')
    if not measurement['admitted']:
        raise ValueError('Actual prompt plus full completion reserve exceeds context')
    del tokenizer
    phase_lr = cfg['learning_rate'] if engine.checkpoint('base') else SFT_LEARNING_RATE
    backend = CardinalSftPolicy(model_dir, {**cfg, 'learning_rate': phase_lr})
    engine = WarmstartCycle(run, backend)
    runtime = {'model_index': index, **backend.runtime, 'versions': actual}
    previous = engine.existing('runtime.json')
    if previous is None:
        engine.commit('runtime.json', runtime)
    elif previous != runtime:
        raise ValueError('Runtime identity changed')
    if not engine.checkpoint('base'):
        engine.ensure_warm_initial()
        latest = max(i for i in range(33) if engine.warm_checkpoint(i))
        engine.activate_warm(latest)
        source_path = f'warmstart/checkpoints/step{latest}/metadata.json'
        if not engine.smoke('sft-admission', source_path):
            return engine.status
    engine.run_all()
    return engine.status


def launch(run, model_dir):
    run, model_dir = Path(run).resolve(), Path(model_dir).resolve()
    manifest, status = verify_run(run)
    if status['state'] in TERMINAL:
        return status
    with common.process_lock(gpu.LOCK) as global_fd, common.process_lock(run/'run.lock') as run_fd:
        status = common.load(run/'status.json')
        budget, timeout = journal.execution_budget(manifest['config'], status.get('execution_budget'))
        status.update(execution_budget=budget, updated=common.now())
        common.atomic_json(run/'status.json', status)
        child = subprocess.Popen([sys.executable, '-m', 'selfplay.program_cardinal_warmstart', '_worker',
            '--run', str(run), '--model-dir', str(model_dir), '--lock-fds', f'{global_fd},{run_fd}'],
            cwd=ROOT, pass_fds=(global_fd, run_fd), start_new_session=True)
        try:
            code = child.wait(timeout=timeout)
            if code:
                raise RuntimeError(f'Warmstart CUDA worker exited{code}; committed artifacts remain resumable')
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
            status = common.load(run/'status.json')
            status.update(state='stopped', updated=common.now(), claims_allowed=False,
                error_type=type(error).__name__, reason=str(error))
            common.atomic_json(run/'status.json', status)
            raise
    return common.load(run/'status.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'prepare', 'index-model', 'run', '_worker'))
    parser.add_argument('--run', type=Path)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--deployment', type=Path)
    parser.add_argument('--teacher-run', type=Path)
    parser.add_argument('--model-info', type=Path, default=ROOT/'runs/cardinal-gpu-preflight/model-info.json')
    parser.add_argument('--lock-fds')
    args = parser.parse_args()
    if args.command in ('plan', 'prepare') and (args.deployment is None or args.teacher_run is None):
        parser.error('--deployment and --teacher-run required')
    if args.command == 'plan':
        result = plan(public_deployment(common.load(args.deployment)), common.digest(args.teacher_run/'corpus.json'))
    elif args.command == 'prepare':
        if args.run is None:
            parser.error('--run required')
        result = prepare(args.run, args.deployment, args.model_info, args.teacher_run)
    elif args.command == 'index-model':
        result = index_model(args.model_dir, args.model_info)
    else:
        if args.run is None or args.model_dir is None:
            parser.error('--run and --model-dir required')
        result = launch(args.run, args.model_dir) if args.command == 'run' else worker(args.run, args.model_dir, args.lock_fds)
    print(common.canonical(result))


if __name__ == '__main__':
    main()
