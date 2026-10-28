"""Evaluation-only closure of the terminal three-update mixed-native27B study.

No optimizer operation is permitted. The original run remains incomplete;
this separately registered measurement compares its exact base and cycle3.
Preparation and verification are CPU-only and never allocate resources.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

from selfplay import program_debug_revision_rl as original
from selfplay.program_cardinal_cuda import checkpoint_state

common, gpu, journal, harness = original.common, original.gpu, original.journal, original.harness
ROOT, LOCK = original.ROOT, original.LOCK
MODEL_REPO, MODEL_REVISION = original.MODEL_REPO, original.MODEL_REVISION
VERSIONS, index_model = original.VERSIONS, original.index_model
PARENT = ROOT/'runs/program-debug-revision-rl-gpu-v2'
ENDPOINTS = ('base', 'cycle3')
KIND = 'mixed_native_revision_evaluation_only'
SOURCES = list(dict.fromkeys(original.SOURCES + [
    'selfplay/program_debug_revision_evaluation.py',
    'tests/test_program_debug_revision_evaluation.py',
    'scripts/runpod_arena_mixed.py',
    'scripts/cloud/bootstrap_revision_evaluation.py',
    'scripts/cloud/run_revision_evaluation_job.py',
    'scripts/revision_evaluation_supervisor.py',
    'tests/test_revision_evaluation_job_supervisor.py',
    'scripts/analyze_program_debug_revision_evaluation.py',
    'tests/test_analyze_program_debug_revision_evaluation.py']))
TERMINAL = {'complete', 'budget_incomplete', 'numerical_failed', 'stopped', 'failed'}


def public_deployment(value):
    keys = ('intent','project','created_at','deadline_unix','maximum_hours','reservation_usd',
        'estimate_usd','gpu_id','maximum_gpu_rate_usd_hour','maximum_cpu_rate_usd_hour',
        'gpu_disk_gb','guard_disk_gb','gpu_image','guard_source_sha256','cleanup',
        'phase','continuation_id','combined_funding_usd','funding_training_intent',
        'evaluation_intent','coding_evaluation_intent','funding_training_deployment',
        'evaluation_deployment','coding_evaluation_deployment',
        'predecessor_deployment','predecessor_evidence_sha256',
        'mixed_controller_sha256','inherited_controller_sha256')
    result = {key:value[key] for key in keys}
    if (result['project'] != 'cardinal-gpu' or result['maximum_hours'] != 10
            or result['reservation_usd'] != 65 or not math.isfinite(result['deadline_unix'])
            or result['phase'] != 'coding_evaluation' or result['combined_funding_usd'] != 240
            or result['intent'] != result['coding_evaluation_intent']
            or not 0 < result['estimate_usd'] <= 65
            or not 0 < result['maximum_gpu_rate_usd_hour'] <= 5.29
            or not 0 < result['maximum_cpu_rate_usd_hour'] <= .14):
        raise ValueError('Expected strict10h/$65 evaluation deployment envelope')
    identities=[result[k] for k in ('continuation_id','funding_training_intent','evaluation_intent','coding_evaluation_intent')]
    if any(not isinstance(v,str) or uuid.UUID(v).hex!=v for v in identities) or len(set(identities[1:])) != 3:
        raise ValueError('Invalid or ambiguous phase funding identities')
    return result


def config(deployment, parent_config):
    public_deployment(deployment)
    return {**parent_config, 'kind':KIND, 'condition':'cycle3_evaluation_only_v1',
        'deadline':datetime.fromtimestamp(deployment['deadline_unix'],timezone.utc).isoformat(),
        'worker_seconds':35100, 'evaluation_only':True, 'training_allowed':False,
        'fixed_evaluation_checkpoints':list(ENDPOINTS), 'new_optimizer_steps':0,
        'scope':'Separate paired evaluation of three existing RL updates beyond SFT32; original run stays incomplete'}


def parent_evidence(parent, require_active=True):
    """Verify the terminal parent and require that its final evaluation is unseen."""
    parent = Path(parent)
    manifest, status = original.verify_run(parent, require_active=require_active)
    audit = common.load(parent/'final-audit.json')
    if (status.get('state') != 'throughput_failed' or status.get('claims_allowed') is not False
            or audit.get('integrity_passed') is not True
            or audit.get('planned_study_complete') is not False
            or audit.get('planned_comparisons_available') is not False
            or audit.get('manifest_sha256') != common.digest(parent/'manifest.json')
            or audit.get('status_sha256') != common.digest(parent/'status.json')
            or audit.get('genuine_RL_optimizer_steps') != 3
            or audit.get('selected_endpoint') is not None):
        raise ValueError('Audited terminal three-update parent required')
    selection = common.load(parent/'endpoint-selection.json')
    if selection.get('passed') is not False or selection.get('endpoint') is not None:
        raise ValueError('Parent timing endpoint must have failed without a selected endpoint')
    if any(p.startswith('evaluation/') or p.startswith('generations/eval-')
            or p.startswith('batches/eval-') or p.startswith('feedback/eval-')
            for p in status['artifacts']):
        raise ValueError('Parent evaluation has already been sampled or scored')
    if any((parent/'evaluation').glob('**/*')):
        raise ValueError('Uncommitted parent evaluation files exist')
    if any(any((parent/folder).glob('eval-*')) for folder in ('batches','generations','feedback')):
        raise ValueError('Uncommitted parent evaluation draws exist')
    expected = {'base':0,'cycle1':1,'cycle2':2,'cycle3':3}
    metadata = {}
    for label, step in expected.items():
        relative = f'checkpoints/{label}/metadata.json'
        row = common.load(parent/relative)
        identity = checkpoint_state(parent/'checkpoints'/label, 1e-5)
        if (status['artifacts'].get(relative) != common.digest(parent/relative)
                or row['label'] != label or row.get('complete') is not True
                or row['manifest_sha256'] != common.digest(parent/'manifest.json')
                or identity != row['optimizer_identity'] or identity['step'] != step
                or row['model_repo'] != MODEL_REPO or row['model_revision'] != MODEL_REVISION
                or any(status['artifacts'].get(f'checkpoints/{label}/{n}') != h
                    or common.digest(parent/f'checkpoints/{label}/{n}') != h for n,h in row['files'].items())):
            raise ValueError('Parent exact checkpoint lineage differs: '+label)
        if label != 'base' and (row['parent'] != ('base' if step == 1 else f'cycle{step-1}')
                or row['report']['optimizer_steps'] != 1):
            raise ValueError('Expected three genuine inherited optimizer updates')
        metadata[label] = row
    return manifest, status, audit, metadata


def prepare(run, deployment_path, model_info_path=None, parent=PARENT):
    run, parent = Path(run), Path(parent)
    if run.exists(): raise FileExistsError('New evaluation directory required')
    deployment = public_deployment(common.load(deployment_path))
    pm, ps, audit, checkpoints = parent_evidence(parent)
    cfg = config(deployment, pm['config'])
    model_info_path = Path(model_info_path) if model_info_path else parent/'inputs/model-info.json'
    if common.digest(model_info_path) != pm['input_sha256']['inputs/model-info.json']:
        raise ValueError('Original pinned model metadata differs')
    for relative in SOURCES:
        if not (ROOT/relative).is_file(): raise ValueError('Source not ready: '+relative)
    run.mkdir(parents=True)
    inputs = {}
    def copy(source, relative):
        target=run/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target);inputs[relative]=common.digest(target)
    # The complete original scientific journal is portable and remains byte-for-byte
    # identifiable under its own manifest. No metadata is relabeled to this run.
    parent_paths = set(pm['input_sha256']) | set(ps['artifacts']) | {'manifest.json','status.json','final-audit.json'}
    parent_paths.update('source/'+p for p in pm['source_sha256'])
    for relative in sorted(parent_paths): copy(parent/relative, 'parent/'+relative)
    for relative in ('schedule.json','prompts/debug.txt','heldout-opponents.json','inputs/model-info.json'):
        copy(parent/relative, relative)
    for relative in pm['input_sha256']:
        if relative.startswith('inputs/tokenizer/'): copy(parent/relative,relative)
    inherited = {}
    initial_artifacts = {}
    for label in ENDPOINTS:
        metadata=checkpoints[label]
        for name in (*gpu.FILES,'metadata.json'):
            relative=f'checkpoints/{label}/{name}';copy(parent/relative,relative)
            initial_artifacts[relative]=inputs[relative]
        inherited[label]={'origin_manifest_sha256':common.digest(parent/'manifest.json'),
            'metadata_sha256':inputs[f'checkpoints/{label}/metadata.json'],
            'files':metadata['files'],'optimizer_identity':metadata['optimizer_identity'],
            'origin_checkpoint':label,'prior_SFT_steps':32,'prior_RL_steps':metadata['optimizer_identity']['step']}
    plan = {'kind':KIND,'created':common.now(),'config':cfg,
        'fixed_endpoints':list(ENDPOINTS),'paired_authoring_seeds':16,'trajectories':32,'sampled_turns':64,
        'matches_per_compiled_final':48,'heldout_maps':6,'opponents':4,'seats':2,
        'parent_manifest_sha256':common.digest(parent/'manifest.json'),
        'parent_status_sha256':common.digest(parent/'status.json'),
        'parent_audit_sha256':common.digest(parent/'final-audit.json'),
        'numerical_protocol':'Fresh unchanged mixed-native32-token derivative/likelihood smoke at each loaded endpoint; original evaluation has no additional full64-turn teacher-forcing gate.',
        'unchanged_state':'Each endpoint cohort is followed by exact adapter/Adam/RNG semantic comparison against its inherited checkpoint.',
        'selection':'Fixed cycle3 before any heldout draw; all32 final revisions sealed before any heldout scoring.',
        'interpretation':'Closes three-update comparison only; does not complete or validate the abandoned original10/15/25-endpoint study.',
        'time_budget':{'rental_seconds':36000,'worker_seconds':35100,'sampling_worst_case_seconds':28800,
            'scoring_seconds':1800,'export_seconds':1800,'initial_total_reserve_seconds':32400},
        'scope':'Evaluation only; no SFT, RL, repairs, replacement draws or changed opponent policies.'}
    for relative,value in [('deployment.json',deployment),('plan.json',plan)]:
        common.atomic_json(run/relative,value);inputs[relative]=common.digest(run/relative)
    sources={}
    for relative in SOURCES:
        target=run/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,target);sources[relative]=common.digest(target)
    manifest={'format_version':1,'kind':KIND,'phase':'coding_evaluation','created':common.now(),'config':cfg,
        'source_sha256':sources,'input_sha256':inputs,'expected_installed_versions':VERSIONS,
        'prompt_sha256':{'debug':inputs['prompts/debug.txt']},'inherited_checkpoints':inherited,
        'parent_manifest_sha256':plan['parent_manifest_sha256'],'parent_status_sha256':plan['parent_status_sha256'],
        'parent_audit_sha256':plan['parent_audit_sha256'],'claims_require_complete':True,
        'new_optimizer_steps':0,'parent_training_complete':False,'evaluation_plan_sha256':inputs['plan.json']}
    common.atomic_json(run/'manifest.json',manifest)
    common.atomic_json(run/'status.json',{'manifest_sha256':common.digest(run/'manifest.json'),
        'artifacts':initial_artifacts,'state':'prepared','stage':'prepared','updated':common.now(),
        'claims_allowed':False,'new_optimizer_steps':0,'inherited_optimizer_steps':3})
    verify_run(run)
    return manifest


def verify_run(run, require_active=True):
    run=Path(run);manifest,status=common.load(run/'manifest.json'),common.load(run/'status.json')
    deployment=public_deployment(common.load(run/'deployment.json'))
    pm=common.load(run/'parent/manifest.json')
    if (manifest.get('kind') != KIND or manifest.get('phase') != 'coding_evaluation'
            or manifest['config'] != config(deployment,pm['config'])
            or set(manifest['source_sha256']) != set(SOURCES)
            or status['manifest_sha256'] != common.digest(run/'manifest.json')
            or manifest['evaluation_plan_sha256'] != common.digest(run/'plan.json')):
        raise ValueError('Evaluation manifest/config identity differs')
    for relative,expected in {**manifest['input_sha256'],**status['artifacts']}.items():
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts or common.digest(run/path) != expected:
            raise ValueError('Committed evaluation/input bytes differ: '+relative)
    for relative,expected in manifest['source_sha256'].items():
        if common.digest(run/'source'/relative) != expected or (require_active and common.digest(ROOT/relative) != expected):
            raise ValueError('Frozen evaluation source differs: '+relative)
    for name,key in [('manifest.json','parent_manifest_sha256'),('status.json','parent_status_sha256'),('final-audit.json','parent_audit_sha256')]:
        if common.digest(run/'parent'/name) != manifest[key]: raise ValueError('Parent evidence binding differs')
    # Cheap byte bindings during every resume; full CPU tensor parent proof is
    # performed at prepare, inherited checkpoint activation and final audit.
    if (common.load(run/'schedule.json') != common.load(run/'parent/schedule.json')
            or common.digest(run/'heldout-opponents.json') != common.digest(run/'parent/heldout-opponents.json')
            or common.digest(run/'prompts/debug.txt') != common.digest(run/'parent/prompts/debug.txt')):
        raise ValueError('Original unseen evaluation protocol differs')
    return manifest,status


def semantic(identity):
    return {k:v for k,v in identity.items() if k != 'files'}


class EvaluationPolicy(original.TrainRevisionCudaPolicy):
    """The exact historical sampler, with every optimizer entry point disabled."""
    def update(self,*args,**kwargs): raise RuntimeError('Evaluation forbids optimizer operations')
    update_trajectories=update
    full_backward_probe=update


class Evaluation(original.RevisionRL):
    def __init__(self,run,backend):
        self.run,self.backend=Path(run),backend
        self.manifest,self.status=verify_run(run)
        self.config,self.schedule=self.manifest['config'],common.load(self.run/'schedule.json')
        self.prompts={'debug':(self.run/'prompts/debug.txt').read_text()}
        self.active_source=None

    def checkpoint(self,label):
        if label not in ENDPOINTS: raise ValueError('Only fixed inherited evaluation endpoints exist')
        binding=self.manifest['inherited_checkpoints'][label]
        relative=f'checkpoints/{label}/metadata.json';metadata=self.existing(relative)
        if (metadata is None or metadata['label'] != label or metadata.get('complete') is not True
                or metadata['manifest_sha256'] != binding['origin_manifest_sha256']
                or self.status['artifacts'][relative] != binding['metadata_sha256']
                or metadata['files'] != binding['files']
                or metadata['optimizer_identity'] != binding['optimizer_identity']):
            raise ValueError('Inherited checkpoint origin/metadata differs')
        for name,digest in metadata['files'].items():
            path=f'checkpoints/{label}/{name}'
            if self.status['artifacts'].get(path) != digest or common.digest(self.run/path) != digest:
                raise ValueError('Inherited checkpoint file differs')
        if self.inspect_checkpoint(self.run/'checkpoints'/label) != metadata['optimizer_identity']:
            raise ValueError('Inherited actual optimizer/tensor identity differs')
        return metadata

    def inspect_checkpoint(self,folder):
        return checkpoint_state(folder,self.config['learning_rate'])

    def initialize(self):
        for label in ENDPOINTS:self.checkpoint(label)

    def save_checkpoint(self,*args,**kwargs):
        raise RuntimeError('Evaluation cannot create or update learned checkpoints')

    def group(self,*args,**kwargs):
        raise RuntimeError('Evaluation cannot run training groups')

    def smoke(self,label):
        self.admit();self.state(stage='numerical_smoke/'+label)
        contexts=[harness.initial_messages(),harness.initial_messages()+[
            {'role':'assistant','content':"def act():\n    return 'rest'"},
            {'role':'user','content':'Unscored numerical fixture. Revise the draft. Check your code. '}]]
        report=self.backend.mechanics_smoke({'batch_messages':contexts})
        report.update(source_checkpoint=label,
            source_metadata_sha256=self.status['artifacts'][f'checkpoints/{label}/metadata.json'],
            mixed_length_native_chat_prompts=True,authoring_stages=['draft','revision'],evaluation_only=True)
        path=f'smoke/{label}-{time.time_ns()}.json';self.commit(path,report)
        if not report.get('passed'):
            self.state(state='numerical_failed',stage='numerical_smoke/'+label,claims_allowed=False)
            return None
        return path

    def remaining_admission(self,label):
        checked_at=datetime.now(timezone.utc)
        _,remaining=journal.execution_budget(self.config,self.status['execution_budget'],at=checked_at)
        pending=[]
        for endpoint in ENDPOINTS:
            for batch in range(8):
                for stage in ('draft','revision'):
                    path=f'batches/eval-{endpoint}-{batch}-{stage}.json'
                    if path not in self.status['artifacts']:pending.append(path)
        required=len(pending)*self.config['sample_seconds']+3600
        passed=remaining>=required
        self.commit(f'admissions/{label}-{time.time_ns()}.json',{'checked_at':checked_at.isoformat(),
            'remaining_seconds':remaining,'pending_batch_paths':pending,'sampling_seconds_per_batch':900,
            'scoring_and_export_reserve_seconds':3600,'required_seconds':required,'passed':passed})
        if not passed:self.state(state='budget_incomplete',stage='before_'+label,claims_allowed=False)
        return passed

    def unchanged(self,label,smoke_path):
        path=f'endpoint-state/{label}/proof.json'
        old=self.existing(path)
        if old is not None:return old
        self.admit()
        folder=self.run/'endpoint-state'/label/'snapshot'
        if folder.exists():folder.rename(folder.with_name('snapshot.uncommitted-'+uuid.uuid4().hex))
        files=self.backend.save_checkpoint(folder)
        actual=self.inspect_checkpoint(folder);expected=self.checkpoint(label)['optimizer_identity']
        if semantic(actual) != semantic(expected):
            # Preserve actual bytes and the failed comparison before stopping.
            passed=False
        else:passed=True
        for name,digest in files.items():
            if common.digest(folder/name) != digest:raise ValueError('Unchanged-state save hash differs')
            self.status['artifacts'][f'endpoint-state/{label}/snapshot/{name}']=digest
        proof={'checkpoint':label,'source_metadata_sha256':self.status['artifacts'][f'checkpoints/{label}/metadata.json'],
            'smoke_sha256':self.status['artifacts'][smoke_path],'actual_identity':actual,'files':files,
            'unchanged_adapter_Adam_RNG':passed,'new_optimizer_steps':0,
            'generations':{f'generations/eval-{label}-{i}.json':self.status['artifacts'][f'generations/eval-{label}-{i}.json'] for i in range(16)}}
        self.commit(path,proof)
        if not passed:self.state(state='numerical_failed',stage='endpoint_state/'+label,claims_allowed=False)
        return proof

    def run_all(self):
        if self.status['state'] in TERMINAL:return
        self.initialize()
        paths=[]
        for label in ENDPOINTS:
            names=[f'eval-{label}-{i}' for i in range(16)]
            paths.extend(f'generations/{name}.json' for name in names)
            if self.existing(f'endpoint-state/{label}/proof.json') is not None:continue
            self.activate(label)
            smoke_path=self.smoke(label)
            if smoke_path is None or not self.remaining_admission(label):return
            for offset in range(0,16,2):
                self.trajectories(f'eval-{label}-{offset//2}',names[offset:offset+2],
                    self.schedule['eval_generation_seeds'][offset:offset+2],label,
                    [self.schedule['eval_practice_maps'][str(i)] for i in range(offset,offset+2)])
            if not self.unchanged(label,smoke_path)['unchanged_adapter_Adam_RNG']:return
        seal={'checkpoints':{e:self.status['artifacts'][f'checkpoints/{e}/metadata.json'] for e in ENDPOINTS},
            'generations':{p:self.status['artifacts'][p] for p in paths},'draws':32,'sampled_turns':64,
            'endpoint_state_proofs':{e:self.status['artifacts'][f'endpoint-state/{e}/proof.json'] for e in ENDPOINTS},
            'evaluation_plan_sha256':self.manifest['evaluation_plan_sha256'],'scoring_allowed':True,'new_optimizer_steps':0}
        old=self.existing('evaluation/sealed.json')
        if old is None:self.commit('evaluation/sealed.json',seal)
        elif old!=seal:raise ValueError('Evaluation seal differs')
        opponents=common.load(self.run/'heldout-opponents.json')
        for label in ENDPOINTS:
            relative=f'evaluation/{label}/holdout.json'
            if self.existing(relative) is not None:continue
            rows=[]
            for i in range(16):
                self.admit();name=f'eval-{label}-{i}';record=self.existing(f'generations/{name}.json')
                rows.append({'generation':name,'generation_sha256':self.status['artifacts'][f'generations/{name}.json'],
                    **original.score(record,opponents['selected'],self.schedule['holdout_maps'],self.config)})
            self.commit(relative,{'checkpoint':label,'prompt_condition':'native_nonthinking_revision_rl','split':'holdout',
                'training_use':False,'seeds':self.schedule['holdout_maps'],'opponents':opponents['selected'],
                'opponents_sha256':self.manifest['input_sha256']['heldout-opponents.json'],
                'seal_sha256':self.status['artifacts']['evaluation/sealed.json'],'prescribed_program_draws':16,'scores':rows})
        self.admit();self.state(state='complete',stage='complete',claims_allowed=True,new_optimizer_steps=0,
            inherited_optimizer_steps=3,final_checkpoint='cycle3',learning_comparison_available=True,
            parent_training_complete=False,comparison_scope='Three inherited RL updates versus exact SFT32 baseline')


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
    backend=EvaluationPolicy(model_dir,cfg);engine=Evaluation(run,backend)
    runtime={**original.make_runtime_report(index,backend.runtime,actual),'evaluation_only':True,
        'training_allowed':False,'new_optimizer_steps':0}
    old=engine.existing('runtime.json')
    if old is None:engine.commit('runtime.json',runtime)
    elif old!=runtime:raise ValueError('Runtime identity differs on resume')
    engine.run_all();return engine.status


def launch(run,model_dir):
    run,model_dir=Path(run).resolve(),Path(model_dir).resolve()
    manifest,status=verify_run(run)
    if status['state'] in TERMINAL:return status
    with common.process_lock(LOCK) as global_fd,common.process_lock(run/'run.lock') as run_fd:
        status=common.load(run/'status.json')
        budget,timeout=journal.execution_budget(manifest['config'],status.get('execution_budget'))
        status.update(execution_budget=budget,updated=common.now());common.atomic_json(run/'status.json',status)
        child=subprocess.Popen([sys.executable,'-u','-m','selfplay.program_debug_revision_evaluation','_worker',
            '--run',str(run),'--model-dir',str(model_dir),'--lock-fds',f'{global_fd},{run_fd}'],
            cwd=ROOT,pass_fds=(global_fd,run_fd),start_new_session=True)
        try:
            code=child.wait(timeout=timeout)
            if code:raise RuntimeError(f'Evaluation worker exited {code}; no automatic retry')
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
    parser.add_argument('command',choices=('prepare','run','_worker'))
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--model-dir',type=Path)
    parser.add_argument('--deployment',type=Path);parser.add_argument('--lock-fds')
    parser.add_argument('--model-info',type=Path);parser.add_argument('--parent',type=Path,default=PARENT)
    args=parser.parse_args()
    if args.command=='prepare':result=prepare(args.run,args.deployment,args.model_info,args.parent)
    elif args.command=='run':result=launch(args.run,args.model_dir)
    else:result=worker(args.run,args.model_dir,args.lock_fds)
    print(common.canonical(result))


if __name__=='__main__':main()
