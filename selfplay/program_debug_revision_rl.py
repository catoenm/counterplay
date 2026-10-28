"""Mixed-native two-call historical-league RLOO; separately frozen long run.

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
from selfplay import program_debug_revision_calibration as calibrated
from selfplay import program_debug_rl as previous
from selfplay.program_debug_revision_cuda import stage_metadata
from selfplay.program_debug_revision_train import TrainRevisionCudaPolicy, validate_checkpoint_transition
from selfplay.program_cardinal_cuda import checkpoint_state

ROOT, LOCK = gpu.ROOT, gpu.LOCK
MODEL_REPO, MODEL_REVISION = gpu.MODEL_REPO, gpu.MODEL_REVISION
VERSIONS, index_model = calibrated.VERSIONS, gpu.index_model
PARENT = ROOT/'runs/program-debug-revision-calibration-gpu-v1'
DESIGN_PLAN = ROOT/'runs/debug-mixed-native-long-rl-design-v3.json'
ACCEPTANCE = ROOT/'runs/debug-mixed-native-long-rl-root-acceptance-v1.json'
MAP_REVIEW = ROOT/'runs/debug-revision-rl-preflight-v1/map-collision-design-review.json'
SOURCES = list(dict.fromkeys(calibrated.SOURCES + [
    'selfplay/program_debug_revision_rl.py','selfplay/program_debug_revision_train.py',
    'scripts/analyze_program_debug_revision_rl.py','scripts/runpod_revision_rl.py',
    'scripts/cloud/run_revision_rl_job.py','scripts/cloud/bootstrap_revision_rl.py',
    'scripts/revision_rl_supervisor.py','tests/test_program_debug_revision_rl.py',
    'tests/test_program_debug_revision_train.py','tests/test_analyze_program_debug_revision_rl.py',
    'tests/test_runpod_revision_rl.py','tests/test_revision_rl_job_supervisor.py']))
TERMINAL = {'complete','throughput_failed','budget_incomplete','numerical_failed','stopped','failed'}
EVALUATION_RESERVE = 32400


def public_deployment(value):
    keys=('intent','project','created_at','deadline_unix','maximum_hours','reservation_usd',
          'estimate_usd','gpu_id','maximum_gpu_rate_usd_hour','maximum_cpu_rate_usd_hour',
          'gpu_disk_gb','guard_disk_gb','gpu_image','guard_source_sha256','cleanup')
    result={key:value[key] for key in keys}
    if (result['project']!='cardinal-gpu' or result['maximum_hours']!=25
            or result['reservation_usd']!=145 or not math.isfinite(result['deadline_unix'])
            or not 0<result['maximum_gpu_rate_usd_hour']<=5.29
            or not 0<result['maximum_cpu_rate_usd_hour']<=.14
            or not 0<result['estimate_usd']<=145):
        raise ValueError('Expected strict25h/$145 deployment envelope')
    return result


def config(deployment):
    public_deployment(deployment)
    return {**gpu.config(deployment),'kind':'native_nonthinking_revision_rl',
        'condition':'mixed-native-revision-rl-v1','seed':425401,'thinking':True,'reasoning_effort':'low',
        'turn_native_modes':{s:stage_metadata(s) for s in ('draft','revision')},
        'max_new_tokens':4096,'max_context_tokens':16384,'sample_seconds':900,
        'worker_seconds':86400,'batch_size':2,'group_size':8,'iterations':25,
        'endpoint_choices':[10,15,25],'profile_attempt_cap':3,'authoring_calls':2,
        'practice_maps_per_trajectory':2,'eval_samples':16,'capability_samples':0,
        'opponent_attempt_cap':0,'maximum_model_opponents':2,'timing_safety_factor':1.5,
        'evaluation_generation_reserve_seconds':28800,'evaluation_cpu_reserve_seconds':1800,
        'audit_export_reserve_seconds':1800,'evaluation_total_reserve_seconds':EVALUATION_RESERVE,
        'full_update_checkpoint_allowance_seconds':120,
        'scope':'Incremental RL beyond SFT32 with identical two-call mixed-native harness',
        'baseline':'Exact SFT32 adapter and zero-step RL Adam; no new SFT/API targets',
        'history_transform':harness.REVISION_CONTEXT_POLICY,
        'reward':'Final relative return; invalid authorship or candidate interpreter fault=-1'}


def schedule():
    result={'train_practice_maps':{str(g):[211000000+g*100+i*2+j for i in range(8) for j in range(2)] for g in range(1,26)},
        'train_maps':{str(g):[212000000+g*10+j for j in range(2)] for g in range(1,26)},
        'eval_practice_maps':{str(i):[231000000+i*2+j for j in range(2)] for i in range(16)},
        'holdout_maps':list(range(232000000,232000006)),
        'train_generation_seeds':{str(g):[221000000+g*100+i*2 for i in range(8)] for g in range(1,26)},
        'eval_generation_seeds':[241000000+i*2 for i in range(16)],
        'revision_seed_rule':'draft_seed+1; fixed disjoint authoring families; no retries',
        'map_seed_replacements':[{'family':'train_practice_maps','group':'8','index':11,
            'rejected_seed':211000811,'collision_with_seed':151000251,'replacement_seed':219000000}]}
    result['train_practice_maps']['8'][11]=219000000
    return result


def plan(deployment,design_path=DESIGN_PLAN,acceptance_path=ACCEPTANCE):
    return {'kind':'mixed_native_revision_rl_plan','config':config(deployment),'schedule':schedule(),
        'design_sha256':common.digest(design_path),'root_acceptance_sha256':common.digest(acceptance_path),
        'objective':'Mean8 trajectory RLOO advantage times summed sampled draft+revision logprob; prompt/feedback masked.',
        'profile':'Exactly3 attempts; numerical feasibility and timing only; zero-signal groups retained.',
        'endpoint':'Largest10/15/25 fitting1.5 times full-priced group costs plus32400s reserved once.',
        'evaluation':'Base and fixed final16 paired trajectories each, collected after training; all32 sealed before scoring.',
        'failure':'Preserve incomplete/numerical/ambiguous intent; no replacement draws, alternate endpoints or automatic retries.',
        'budget':{'cumulative_ceiling_usd':200,'reservation_usd':145,'rental_seconds':90000,'worker_seconds':86400}}


fingerprints = previous.fingerprints


def prior_maps(parent):
    return {'warmstart':common.load(parent/'inputs/initial-map-fingerprints.json'),
        'debug':common.load(parent/'inputs/prior-debug-map-fingerprints.json'),
        'low_calibration':common.load(parent/'inputs/previous-calibration-map-fingerprints.json'),
        'mixed_calibration':common.load(parent/'map-fingerprints.json')}


def rosters(scores,decision,corpus):
    anchors=[{'id':n,'origin':'scripted','authorship':'scripted_anchor','compiled_ir':common.reference_programs()[n]} for n in ('nearest','value')]
    models=[]
    for i in decision['selected_indices']:
        row=scores['rows'][i];program=row['classification']['program']
        models.append({'id':'calibration-'+str(i),'origin':'model','authorship':'inherited_SFT32_controller',
            'source_checkpoint':'SFT32-base','compiled_ir':program['compiled_ir'],'ir_sha256':program['ir_sha256'],
            'generation_sha256':row['generation_sha256'],'source_path':f'inputs/calibration-generations/calibration-{i}.json'})
    if len(models)!=2 or len({r['ir_sha256'] for r in models})!=2:raise ValueError('Two fixed calibration controllers required')
    seen={r['ir_sha256'] for r in models};held=[]
    for demo in sorted(corpus['demonstrations'],key=lambda r:r['index']):
        p=demo['program'];ir=p['ir_sha256']
        if ir in seen:continue
        seen.add(ir);held.append({'id':'teacher-'+str(demo['index']),'origin':'model',
            'authorship':'external_teacher_off_policy_anchor','was_prior_SFT_example':True,
            'source_checkpoint':'external_teacher','compiled_ir':p['compiled_ir'],'ir_sha256':ir,
            'generation_sha256':demo['generation_sha256'],'source_path':'inputs/teacher-corpus.json'})
        if len(held)==2:break
    if len(held)!=2:raise ValueError('Two reserved distinct teacher anchors required')
    return ({'selected':anchors+models,'frozen':True},{'selected':anchors+held,'frozen':True,
        'teacher_anchors_reserved_from_RL_roster_not_unseen_to_weights':True})


def prepare(run,deployment_path,model_info_path):
    run=Path(run)
    if run.exists():raise FileExistsError('New run directory required')
    deployment=public_deployment(common.load(deployment_path));cfg=config(deployment)
    parent_manifest,parent_status=calibrated.verify_run(PARENT)
    audit=common.load(PARENT/'final-audit.json')
    if (parent_status['state']!='complete' or parent_status.get('qualification_passed') is not True
            or not audit.get('integrity_passed') or not audit.get('qualification_passed')
            or audit.get('manifest_sha256')!=common.digest(PARENT/'manifest.json')
            or audit.get('status_sha256')!=common.digest(PARENT/'status.json')
            or audit.get('genuine_RL_optimizer_steps')!=0
            or audit.get('full_likelihood_admission',{}).get('verified_turns')!=16
            or not audit['full_likelihood_admission']['passed']):
        raise ValueError('Complete independently audited mixed-native calibration required')
    identity=checkpoint_state(PARENT/'inputs/initial-checkpoint',1e-5)
    if identity!=parent_manifest['initial_checkpoint_identity'] or identity['step']!=0:
        raise ValueError('Exact SFT32/zero-RL-step baseline differs')
    acceptance=common.load(ACCEPTANCE)
    if acceptance.get('root_accepted') is not True or acceptance['design_sha256']!=common.digest(DESIGN_PLAN):
        raise ValueError('Root design acceptance differs')
    if common.digest(model_info_path)!=parent_manifest['input_sha256']['inputs/model-info.json']:
        raise ValueError('Pinned model metadata differs')
    initial_manifest=common.load(PARENT/'inputs/initial-manifest.json')
    corpus_path=calibrated.INITIAL/'inputs/teacher/corpus.json'
    if common.digest(corpus_path)!=initial_manifest['input_sha256']['inputs/teacher/corpus.json']:
        raise ValueError('PriorSFT teacher corpus differs')
    scores=common.load(PARENT/'calibration/scores.json');decision=common.load(PARENT/'calibration/decision.json')
    training,evaluation=rosters(scores,decision,common.load(corpus_path))
    maps=fingerprints(schedule(),prior_maps(PARENT))
    for relative in SOURCES:
        if not (ROOT/relative).is_file():raise ValueError('Source not ready: '+relative)
    run.mkdir(parents=True);inputs={}
    def copy(source,relative):
        target=run/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        inputs[relative]=common.digest(target)
    # Preserve the exact baseline, tokenizer and full earlier evidence chain.
    for relative in parent_manifest['input_sha256']:
        if relative.startswith('inputs/') and relative not in ('inputs/design-plan.json','inputs/design-plan-v1.json'):
            copy(PARENT/relative,relative)
    for name in ('manifest.json','status.json','final-audit.json','map-fingerprints.json'):
        copy(PARENT/name,'inputs/calibration-'+name)
    for name in ('scores.json','decision.json','sealed.json','full-likelihood-gate.json'):
        copy(PARENT/'calibration'/name,'inputs/calibration-'+name)
    for i in decision['selected_indices']:
        copy(PARENT/f'generations/calibration-{i}.json',f'inputs/calibration-generations/calibration-{i}.json')
    copy(corpus_path,'inputs/teacher-corpus.json');copy(DESIGN_PLAN,'inputs/design-plan.json')
    copy(ACCEPTANCE,'inputs/root-acceptance.json')
    copy(MAP_REVIEW,'inputs/map-collision-design-review.json')
    receipt=ROOT/acceptance['authoritative_budget_now']['accounting_receipt']
    copy(receipt,'inputs/accounting-review.json')
    for relative,value in [('deployment.json',deployment),('plan.json',plan(deployment)),('schedule.json',schedule()),
        ('map-fingerprints.json',maps),('initial-opponents.json',training),('evaluation-opponents.json',evaluation)]:
        common.atomic_json(run/relative,value);inputs[relative]=common.digest(run/relative)
    (run/'prompts').mkdir();(run/'prompts/debug.txt').write_text(harness.debug_model_spec())
    inputs['prompts/debug.txt']=common.digest(run/'prompts/debug.txt');sources={}
    for relative in SOURCES:
        target=run/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,target)
        sources[relative]=common.digest(target)
    manifest={'format_version':1,'kind':cfg['kind'],'created':common.now(),'config':cfg,
        'source_sha256':sources,'input_sha256':inputs,'prompt_sha256':{'debug':inputs['prompts/debug.txt']},
        'expected_installed_versions':VERSIONS,'claims_require_complete':True,
        'initial_checkpoint_files':parent_manifest['initial_checkpoint_files'],'initial_checkpoint_identity':identity,
        'baseline':cfg['baseline']}
    common.atomic_json(run/'manifest.json',manifest)
    common.atomic_json(run/'status.json',{'manifest_sha256':common.digest(run/'manifest.json'),
        'artifacts':{},'state':'prepared','stage':'prepared','updated':common.now(),'claims_allowed':False})
    verify_run(run);return manifest


def verify_run(run,require_active=True):
    run=Path(run);manifest,status=common.load(run/'manifest.json'),common.load(run/'status.json')
    deployment=public_deployment(common.load(run/'deployment.json'))
    if (manifest['kind']!='native_nonthinking_revision_rl' or manifest['config']!=config(deployment)
            or status['manifest_sha256']!=common.digest(run/'manifest.json')
            or common.load(run/'schedule.json')!=schedule()
            or common.load(run/'plan.json')!=plan(deployment,run/'inputs/design-plan.json',run/'inputs/root-acceptance.json')
            or set(manifest['source_sha256'])!=set(SOURCES)):
        raise ValueError('Frozen mixed-native RL identity differs')
    for relative,expected in {**manifest['input_sha256'],**status['artifacts']}.items():
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts or common.digest(run/path)!=expected:
            raise ValueError('Committed artifact identity differs: '+relative)
    for relative,expected in manifest['source_sha256'].items():
        if common.digest(run/'source'/relative)!=expected or (require_active and common.digest(ROOT/relative)!=expected):
            raise ValueError('Archived/active source differs: '+relative)
    return manifest,status


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
                or row.get('completion_token_limit') != cfg['max_new_tokens'] or row.get('calibration_only') is not False or row.get('training_capable') is not True):
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


class RevisionRL(gpu.GPUCycle):
    def __init__(self, run, backend):
        self.run, self.backend = Path(run), backend
        self.manifest, self.status = verify_run(run)
        self.config, self.schedule = self.manifest['config'], common.load(self.run/'schedule.json')
        self.prompts = {'debug': (self.run/'prompts/debug.txt').read_text()}
        self.active_source = None

    def activate(self,label):
        value=super().activate(label);self.active_source=label;return value

    def initialize(self):
        if self.checkpoint('base') is None:
            self.backend.load_checkpoint(self.run/'inputs/initial-checkpoint', self.manifest['initial_checkpoint_files'])
            self.save_checkpoint('base', None, None, copy_from=self.run/'inputs/initial-checkpoint')
        self.activate('base')
        if self.existing('opponents.json') is None:
            self.commit('opponents.json', common.load(self.run/'initial-opponents.json'))
        if self.existing('heldout-opponents.json') is None:
            self.commit('heldout-opponents.json',common.load(self.run/'evaluation-opponents.json'))

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
            for name in gpu.FILES:
                shutil.copyfile(copy_from / name, temporary / name)
            files = {name: common.digest(temporary / name) for name in gpu.FILES}
        identity = self.inspect_checkpoint(temporary)
        before_step = self.checkpoint(parent)['optimizer_identity']['step'] if parent else 0
        expected_step = before_step + (report['optimizer_steps'] if report else 0)
        if identity['step'] != expected_step or identity['learning_rate'] != self.config['learning_rate']:
            raise ValueError('Actual optimizer step/LR differs')
        if parent is not None:
            validate_checkpoint_transition(report, self.checkpoint(parent)['optimizer_identity'], identity)
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

    def ensure_raw_batch(self, batch_id, contexts, seeds, checkpoint, authoring_stage):
        path = f'batches/{batch_id}.json'
        context = {'checkpoint': checkpoint, 'source_metadata_sha256': self.status['artifacts'][f'checkpoints/{checkpoint}/metadata.json'],
                   'contexts_sha256': gpu.object_hash(contexts), 'seeds': seeds, 'authoring_stage': authoring_stage}
        batch = self.existing(path)
        if batch is None:
            self.admit()
            if self.active_source != checkpoint:raise ValueError('Batch source was not explicitly activated')
            self.state(stage='batch/'+batch_id)
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
                'prompt_condition': 'native_nonthinking_revision_rl', 'authorship': 'fresh_learner_two_call_trajectory',
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

    def roster(self,number):
        relative=f'groups/{number}/opponents.json';old=self.existing(relative)
        if old is not None:return old
        latest=(number-1)//5*5
        path=f'league/after-{latest}.json' if latest else 'opponents.json'
        population=self.existing(path)
        if population is None:raise ValueError('Missing preceding frozen league roster')
        selected=population['selected']
        reserved={r['ir_sha256'] for r in self.existing('heldout-opponents.json')['selected'] if r['origin']=='model'}
        model_ids=[r['ir_sha256'] for r in selected if r['origin']=='model']
        if len(selected)!=4 or len(set(model_ids))!=2 or set(model_ids)&reserved:
            raise ValueError('Training roster overlaps reserved model anchors')
        return self.commit(relative,{'cycle':number,'selected':selected,
            'population_sha256':self.status['artifacts'][path],'frozen_before_generation':True})

    def refresh(self,number):
        if number%5:return
        path=f'league/after-{number}.json'
        if self.existing(path) is not None:return
        current=self.existing(f'groups/{number}/opponents.json')['selected']
        anchors=[r for r in current if r['origin']=='scripted']
        candidates=[r for r in current if r['origin']=='model'];admissions=[]
        excluded={r['ir_sha256'] for r in self.existing('heldout-opponents.json')['selected'] if r['origin']=='model'}
        evidence={}
        for group in range(number-4,number+1):
            score_path=f'groups/{group}/scores.json';scores=self.existing(score_path)
            evidence[score_path]=self.status['artifacts'][score_path]
            for i,row in enumerate(scores['rows']):
                if (not row['runtime_valid'] or not row['matches'] or row['candidate_program_faults']
                        or row['positive_banking_games']*2<len(row['matches'])):continue
                program=row['classification']['program'];ir=program['ir_sha256']
                if ir in excluded:continue
                candidate={'id':row['generation'],'origin':'model','authorship':'fresh_learner_final_controller',
                    'source_checkpoint':scores['source_checkpoint'],'compiled_ir':program['compiled_ir'],
                    'ir_sha256':ir,'generation_sha256':row['generation_sha256'],
                    'source_path':f"generations/{row['generation']}.json",'admitted_after_group':number}
                candidates=[r for r in candidates if r['ir_sha256']!=ir]+[candidate]
                admissions.append({'group':group,'candidate_index':i,'controller':candidate})
        models=candidates[-2:]
        if len(models)!=2 or len({r['ir_sha256'] for r in models})!=2:raise ValueError('League lost two distinct models')
        self.commit(path,{'after_group':number,'selected':anchors+models,'admissions':admissions,
            'scores_sha256':evidence,'reserved_model_ir_sha256':sorted(excluded),
            'rule':'Latest two distinct qualifying IRs by ascending group/index; previous retained if insufficient; no new draws/games'})

    def numerical(self,path,records,binding,method,probe=False):
        report=self.existing(path)
        if report is None:
            intent_path=path.replace('.json','-intent.json')
            if self.existing(intent_path) is not None:
                self.state(state='numerical_failed',stage=path,claims_allowed=False,reason='Interrupted numerical intent; no automatic retry')
                return None
            self.admit();self.commit(intent_path,{**binding,'started':common.now()});start=time.monotonic()
            report={**getattr(self.backend,method)(records),**binding,'elapsed_seconds':time.monotonic()-start}
            self.commit(path,report)
        try:
            if any(report.get(k)!=v for k,v in binding.items()):raise ValueError('Numerical source binding differs')
            validate_full_gate(report,records,self.config)
            if probe and (not report.get('derivative',{}).get('finite_nonzero')
                    or report.get('coefficients')!=[1.0]*16 or report.get('objective_divisor')!=16):
                raise ValueError('Full backward probe lacks fixed finite complete derivatives')
        except (ValueError,KeyError,TypeError) as error:
            self.state(state='numerical_failed',stage=path,claims_allowed=False,reason=str(error));return None
        return report

    def group(self,number):
        label,folder=f'cycle{number}',f'groups/{number}'
        target=self.checkpoint(label)
        if target is not None:
            self.refresh(number);return self.finish_timing(number,target['report'])
        if self.existing(folder+'/optimizer-intent.json') is not None:
            self.state(state='failed',stage=folder+'/ambiguous_optimizer',claims_allowed=False,
                reason='Optimizer intent without committed target; no reapplication');return None
        if self.existing(folder+'/timing-start.json') is None:
            self.commit(folder+'/timing-start.json',{'started':common.now(),'cycle':number})
        opponents=self.roster(number);source='base' if number==1 else f'cycle{number-1}'
        self.activate(source);records=[];seeds=self.schedule['train_generation_seeds'][str(number)]
        maps=self.schedule['train_practice_maps'][str(number)]
        for offset in range(0,8,2):
            records.extend(self.trajectories(f'train-{number}-{offset//2}',[f'train-{number}-{i}' for i in range(offset,offset+2)],
                seeds[offset:offset+2],source,[maps[i*2:i*2+2] for i in range(offset,offset+2)]))
        hashes={f"generations/{r['name']}.json":self.status['artifacts'][f"generations/{r['name']}.json"] for r in records}
        flat=[turn for r in records for turn in r['turns']]
        seal={'cycle':number,'source_checkpoint':source,
            'source_metadata_sha256':self.status['artifacts'][f'checkpoints/{source}/metadata.json'],
            'generation_sha256':hashes,'turn_sha256':[gpu.object_hash(t) for t in flat],
            'opponents_sha256':self.status['artifacts'][folder+'/opponents.json'],'draws':8,'sampled_turns':16}
        old=self.existing(folder+'/seal.json')
        if old is None:self.commit(folder+'/seal.json',seal)
        elif old!=seal:raise ValueError('Group seal differs')
        binding={**seal,'seal_sha256':self.status['artifacts'][folder+'/seal.json']}
        gate=self.numerical(folder+'/gate.json',flat,binding,'full_completion_likelihood_gate')
        if gate is None:return None
        if number==1:
            probe=self.numerical(folder+'/probe.json',flat,binding,'full_backward_probe',probe=True)
            if probe is None:return None
        probe=self.existing('groups/1/probe.json')
        if probe is None or not probe['passed']:raise ValueError('Initial full backward admission required')
        scores=self.existing(folder+'/scores.json')
        if scores is None:
            rows=[]
            for record in records:
                self.admit();path=f"generations/{record['name']}.json"
                rows.append({'generation':record['name'],'generation_sha256':hashes[path],
                    **score(record,opponents['selected'],self.schedule['train_maps'][str(number)],self.config)})
            rewards=[r['reward'] for r in rows];advantages=journal.advantages(rewards)
            scores=self.commit(folder+'/scores.json',{'cycle':number,'source_checkpoint':source,
                'source_metadata_sha256':seal['source_metadata_sha256'],'seal_sha256':binding['seal_sha256'],
                'opponents_sha256':seal['opponents_sha256'],'gate_sha256':self.status['artifacts'][folder+'/gate.json'],
                'probe_sha256':self.status['artifacts']['groups/1/probe.json'],
                'seeds':self.schedule['train_maps'][str(number)],'rows':rows,'rewards':rewards,
                'advantages':advantages,'zero_signal':not any(advantages)})
        if (scores['source_checkpoint']!=source or journal.advantages(scores['rewards'])!=scores['advantages']
                or {f"generations/{r['generation']}.json":r['generation_sha256'] for r in scores['rows']}!=hashes
                or scores['seal_sha256']!=binding['seal_sha256']):raise ValueError('Group score provenance differs')
        report={'cycle':number,'source_checkpoint':source,'source_metadata_sha256':seal['source_metadata_sha256'],
            'scores_sha256':self.status['artifacts'][folder+'/scores.json'],
            'generation_sha256':{r['name']:hashes[f"generations/{r['name']}.json"] for r in records},
            'trajectory_turn_sha256':{r['name']:r['turn_sha256'] for r in records},
            'turn_sha256':[gpu.object_hash(t) for t in flat],'rewards':scores['rewards'],
            'advantages':scores['advantages'],'objective_divisor':8,'trajectory_count':8,'sampled_turn_count':16,
            'training_tokens':sum(r['training_tokens'] for r in records),'tool_feedback_target_tokens':0,
            'likelihood_gate_path':folder+'/gate.json','likelihood_gate_sha256':self.status['artifacts'][folder+'/gate.json'],
            'full_backward_probe_sha256':self.status['artifacts']['groups/1/probe.json']}
        copy_from=None;compute_seconds=0.0
        if scores['zero_signal']:
            # Backend validates actual clean state without forward/backward/step;
            # the preserved report must match both exact copied checkpoints.
            update=self.existing(folder+'/skip-result.json')
            if update is None:
                update=self.backend.update_trajectories(records,scores['advantages'])
                self.commit(folder+'/skip-result.json',update)
            if (not update.get('passed') or update.get('outcome')!='skipped_zero_variance'
                    or update.get('optimizer_steps')!=0):
                self.state(state='numerical_failed',stage=folder+'/skip',claims_allowed=False,
                    reason='Zero-signal state proof failed');return None
            report.update(update)
            copy_from=self.run/'checkpoints'/source
        else:
            self.admit();self.state(stage=folder+'/optimizer')
            self.commit(folder+'/optimizer-intent.json',{'source_checkpoint':source,
                'source_metadata_sha256':seal['source_metadata_sha256'],'scores_sha256':report['scores_sha256'],
                'gate_sha256':report['likelihood_gate_sha256'],'probe_sha256':report['full_backward_probe_sha256'],
                'started':common.now()})
            start=time.monotonic();update=self.backend.update_trajectories(records,scores['advantages']);compute_seconds=time.monotonic()-start
            self.commit(folder+'/optimizer-result.json',update)
            if (not update.get('passed') or update.get('outcome')!='updated'
                    or update.get('optimizer_steps')!=1 or update.get('adapter_l2_delta',0)<=0):
                self.state(state='numerical_failed',stage=folder+'/optimizer',claims_allowed=False,
                    reason='Failed or ambiguous actual update; preserved result, no retry');return None
            report.update(update)
        result_path=folder+('/skip-result.json' if scores['zero_signal'] else '/optimizer-result.json')
        report.update(backend_result_path=result_path,backend_result_sha256=self.status['artifacts'][result_path])
        report['update_compute_seconds']=compute_seconds
        save_start=time.monotonic()
        self.save_checkpoint(label,source,report,copy_from=copy_from,update_relative=folder+'/update.json')
        checkpoint_seconds=time.monotonic()-save_start
        self.commit(folder+'/checkpoint-timing.json',{'elapsed_seconds':checkpoint_seconds,
            'checkpoint_metadata_sha256':self.status['artifacts'][f'checkpoints/{label}/metadata.json']})
        self.refresh(number);return self.finish_timing(number,report)

    def finish_timing(self,number,report):
        path=f'groups/{number}/timing.json'
        if self.existing(path) is not None:return report
        start=self.existing(f'groups/{number}/timing-start.json')
        finished_at=datetime.now(timezone.utc)
        elapsed=(finished_at-datetime.fromisoformat(start['started'])).total_seconds()
        checkpoint=self.existing(f'groups/{number}/checkpoint-timing.json')
        # A checkpoint may commit immediately before process loss. Its duration
        # is conservatively bounded by total durable walltime on resume.
        checkpoint_seconds=checkpoint['elapsed_seconds'] if checkpoint else elapsed
        actual_stage=(report['update_compute_seconds']+checkpoint_seconds) if report['optimizer_steps'] else 0.
        probe=self.existing('groups/1/probe.json');probe_seconds=probe['elapsed_seconds'] if number==1 else 0.
        nonupdate=max(0.000001,elapsed-probe_seconds-actual_stage)
        self.commit(path,{'cycle':number,'elapsed_seconds':elapsed,'finished_at':finished_at.isoformat(),
            'timing_start_sha256':self.status['artifacts'][f'groups/{number}/timing-start.json'],
            'probe_elapsed_seconds':probe_seconds,
            'update_compute_seconds':report['update_compute_seconds'],'checkpoint_seconds':checkpoint_seconds,
            'actual_update_stage_seconds':actual_stage,'non_update_seconds':nonupdate,
            'optimizer_steps':report['optimizer_steps'],'includes_resume_downtime':True,
            'checkpoint_timing_recovered_upper_bound':checkpoint is None,
            'update_sha256':self.status['artifacts'][f'groups/{number}/update.json']})
        return report

    def full_cost(self):
        rows=[self.existing(f'groups/{i}/timing.json') for i in range(1,26) if f'groups/{i}/timing.json' in self.status['artifacts']]
        if not rows:raise ValueError('No committed full-priced group timings')
        probe=self.existing('groups/1/probe.json')
        price=max(probe['elapsed_seconds']+self.config['full_update_checkpoint_allowance_seconds'],
            *(r['actual_update_stage_seconds'] for r in rows))
        return {'full_update_price_seconds':price,'maximum_full_group_seconds':max(r['non_update_seconds']+price for r in rows),
            'timing_sha256':{str(r['cycle']):self.status['artifacts'][f"groups/{r['cycle']}/timing.json"] for r in rows},
            'probe_sha256':self.status['artifacts']['groups/1/probe.json']}

    def evaluate(self,endpoint):
        endpoints=('base',f'cycle{endpoint}');paths=[];opponents=self.existing('heldout-opponents.json')
        for label in endpoints:
            self.activate(label)
            for offset in range(0,16,2):
                names=[f'eval-{label}-{i}' for i in range(offset,offset+2)]
                self.trajectories(f'eval-{label}-{offset//2}',names,
                    self.schedule['eval_generation_seeds'][offset:offset+2],label,
                    [self.schedule['eval_practice_maps'][str(i)] for i in range(offset,offset+2)])
                paths.extend(f'generations/{name}.json' for name in names)
        seal={'checkpoints':{e:self.status['artifacts'][f'checkpoints/{e}/metadata.json'] for e in endpoints},
            'generations':{p:self.status['artifacts'][p] for p in paths},'draws':32,'sampled_turns':64,
            'endpoint_selection_sha256':self.status['artifacts']['endpoint-selection.json'],'scoring_allowed':True}
        old=self.existing('evaluation/sealed.json')
        if old is None:self.commit('evaluation/sealed.json',seal)
        elif old!=seal:raise ValueError('Evaluation seal differs')
        for label in endpoints:
            path=f'evaluation/{label}/holdout.json'
            if self.existing(path) is not None:continue
            rows=[]
            for i in range(16):
                self.admit();name=f'eval-{label}-{i}';record=self.existing(f'generations/{name}.json')
                rows.append({'generation':name,'generation_sha256':self.status['artifacts'][f'generations/{name}.json'],
                    **score(record,opponents['selected'],self.schedule['holdout_maps'],self.config)})
            self.commit(path,{'checkpoint':label,'prompt_condition':'native_nonthinking_revision_rl','split':'holdout',
                'training_use':False,'seeds':self.schedule['holdout_maps'],'opponents':opponents['selected'],
                'opponents_sha256':self.status['artifacts']['heldout-opponents.json'],
                'seal_sha256':self.status['artifacts']['evaluation/sealed.json'],'prescribed_program_draws':16,'scores':rows})
        updates=sum(self.checkpoint(f'cycle{i}')['report']['optimizer_steps'] for i in range(1,endpoint+1))
        self.admit();self.state(state='complete',stage='complete',claims_allowed=True,attempted_groups=endpoint,
            actual_optimizer_steps=updates,final_checkpoint=f'cycle{endpoint}',learning_comparison_available=True)

    def run_all(self):
        self.initialize()
        if self.status['state'] in TERMINAL:return
        selection=self.existing('endpoint-selection.json')
        if selection is None:
            for number in range(1,4):
                if self.group(number) is None:return
            selected_at=datetime.now(timezone.utc)
            _,remaining=journal.execution_budget(self.config,self.status['execution_budget'],at=selected_at)
            cost=self.full_cost();selection=endpoint_decision(cost,remaining,self.config)
            self.commit('endpoint-selection.json',{**selection,**cost,'selected_at':selected_at.isoformat()})
        if not selection['passed']:
            self.state(state='throughput_failed',stage='endpoint_admission',claims_allowed=False);return
        endpoint=selection['endpoint']
        for number in range(1,endpoint+1):
            if self.checkpoint(f'cycle{number}') is None:
                checked_at=datetime.now(timezone.utc)
                _,remaining=journal.execution_budget(self.config,self.status['execution_budget'],at=checked_at)
                cost=self.full_cost();needed=self.config['timing_safety_factor']*cost['maximum_full_group_seconds']+EVALUATION_RESERVE
                path=f'admissions/before-group-{number}.json'
                if self.existing(path) is None:self.commit(path,{**cost,'remaining_seconds':remaining,
                    'required_seconds':needed,'passed':remaining>=needed,'checked_at':checked_at.isoformat(),
                    'endpoint_selection_sha256':self.status['artifacts']['endpoint-selection.json']})
                if remaining<needed:
                    self.state(state='budget_incomplete',stage=f'before_cycle{number}',claims_allowed=False);return
            if self.group(number) is None:return
        self.evaluate(endpoint)


def endpoint_decision(cost,remaining,cfg):
    if len(cost['timing_sha256'])!=3:raise ValueError('Exactly3 prescribed profiling groups required')
    maximum=cost['maximum_full_group_seconds']
    if not math.isfinite(maximum) or maximum<=0 or not math.isfinite(remaining) or remaining<0:
        raise ValueError('Invalid full-priced timing evidence')
    projections={str(n):cfg['timing_safety_factor']*maximum*(n-3)+EVALUATION_RESERVE for n in cfg['endpoint_choices']}
    fits=[n for n in cfg['endpoint_choices'] if projections[str(n)]<=remaining]
    return {'passed':bool(fits),'endpoint':max(fits) if fits else None,'profile_groups':3,
        'remaining_seconds':remaining,'safety_factor':cfg['timing_safety_factor'],
        'reserved_final_and_export_seconds':EVALUATION_RESERVE,'projected_remaining_seconds':projections,
        'reason':'Timing/numerical feasibility only; no reward, validity or successful-update-count selection'}


def make_runtime_report(index,backend_runtime,actual_versions):
    report=calibrated.make_runtime_report(index,backend_runtime,actual_versions)
    return {**report,'calibration_only':False,'training_capable':True}


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
    backend=TrainRevisionCudaPolicy(model_dir,cfg);engine=RevisionRL(run,backend)
    runtime=make_runtime_report(index,backend.runtime,actual)
    old=engine.existing('runtime.json')
    if old is None:engine.commit('runtime.json',runtime)
    elif old!=runtime:raise ValueError('Runtime identity differs on resume')
    engine.initialize();engine.admit();engine.state(stage='numerical_smoke')
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
        child=subprocess.Popen([sys.executable,'-u','-m','selfplay.program_debug_revision_rl','_worker','--run',str(run),
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
    elif args.command=='plan':result=plan(public_deployment(common.load(args.deployment)))
    elif args.command=='run':result=launch(args.run,args.model_dir)
    else:result=worker(args.run,args.model_dir,args.lock_fds)
    print(common.canonical(result))


if __name__=='__main__':main()
