"""Separately frozen scalar-execution/rational-score five-environment study."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time
import uuid

from . import envs
from .cuda_policy import MODEL as MODEL_REPO, REVISION as MODEL_REVISION, default_config, compare_actor_reports
from .runner import Journal, canonical, collect_cohort, digest, exclusive_json, file_sha, now, run_pipeline

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = {'market': .24, 'maze': .24, 'chess': .24, 'planning': .24, 'coding': .04}
VERSIONS = {'torch':'2.8.0', 'transformers':'5.20.2', 'peft':'0.21.2',
    'bitsandbytes':'0.50.2', 'accelerate':'1.15.0', 'safetensors':'0.8.0',
    'tokenizers':'0.23.3', 'python-chess':'1.999', 'chess':'1.11.2', 'numpy':'2.5.4'}
TERMINAL = {'training_complete', 'complete', 'throughput_failed', 'failed'}


def load(path):
    return json.loads(Path(path).read_text())


def atomic_status(run, value):
    target = Path(run)/'status.json'
    temporary = target.with_name('.status-'+uuid.uuid4().hex)
    try:
        with temporary.open('x') as f:
            f.write(canonical(value)+'\n'); f.flush(); os.fsync(f.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def public_deployment(plan):
    phase = plan['phase']
    expected = {'mixed_training':(20,125), 'mixed_evaluation':(8,50)}
    if phase not in expected:
        raise ValueError('Wrong mixed experiment deployment phase')
    hours, reservation = expected[phase]
    if (plan['maximum_hours'] != hours or plan['reservation_usd'] != reservation
            or plan['project'] != 'cardinal-gpu' or plan['deadline_unix'] <= 0
            or plan['maximum_gpu_rate_usd_hour'] > 5.29):
        raise ValueError('Mixed deployment budget/deadline differs')
    keys = ('phase','intent','project','created_at','deadline_unix','maximum_hours',
        'reservation_usd','estimate_usd','gpu_id','maximum_gpu_rate_usd_hour',
        'maximum_cpu_rate_usd_hour','gpu_disk_gb','guard_disk_gb','gpu_image',
        'guard_source_sha256','cleanup','continuation_id','funding_training_intent',
        'evaluation_intent','coding_evaluation_intent','evaluation_deployment','coding_evaluation_deployment')
    return {k:plan[k] for k in keys}


def index_model(model_dir, metadata_path):
    metadata, model_dir = load(metadata_path), Path(model_dir)
    if metadata['model'] != MODEL_REPO or metadata['revision'] != MODEL_REVISION:
        raise ValueError('Pinned original model revision differs')
    indexed = {}
    for row in metadata['files']:
        path = model_dir/row['name']
        if not path.is_file() or path.stat().st_size != row['size']:
            raise ValueError('Missing or wrong-size model file: '+row['name'])
        sha, git = hashlib.sha256(), hashlib.sha1(b'blob '+str(row['size']).encode()+b'\0')
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8<<20),b''):
                sha.update(block); git.update(block)
        if row['lfs_sha256']:
            if sha.hexdigest() != row['lfs_sha256']: raise ValueError('Model LFS hash differs')
        elif git.hexdigest() != row['blob_id']:
            raise ValueError('Model Git blob identity differs')
        indexed[row['name']] = sha.hexdigest()
    return {'repo':MODEL_REPO,'revision':MODEL_REVISION,'files':indexed,
            'authoritative_metadata_sha256':file_sha(metadata_path)}


def sources():
    from selfplay.arena_mixed.experiment import sources as original_sources
    names=set(original_sources()) | set(envs.source_hashes())
    names |= {str(p.relative_to(ROOT)) for p in (ROOT/'selfplay/arena_mixed_scalar').glob('*.py')}
    names |= {'scripts/runpod_arena_mixed_scalar.py','scripts/arena_mixed_scalar_supervisor.py',
        'scripts/cloud/bootstrap_arena_mixed_scalar.py','scripts/cloud/run_arena_mixed_scalar_job.py',
        'scripts/cloud/arena-mixed-scalar-requirements.txt',
        'scripts/analyze_arena_mixed_scalar.py','scripts/arena_mixed_scalar_action_audit.py'}
    return {name:file_sha(ROOT/name) for name in sorted(names)}


def make_schedules(prior_fingerprints=()):
    """Allocate maximum training and the entire evaluation before any sampling."""
    schedules, selections, excluded = {}, {}, set(prior_fingerprints)
    for env_index, name in enumerate(envs.ENVIRONMENTS):
        roles = envs.protocol_config()['roles'][name]
        selection = envs.select_scenarios(name, 60, 842000+env_index*10000,
                                          excluded=excluded)
        selections[name] = selection
        chosen = selection['selected']
        excluded.update(s['scenario_sha256'] for s in chosen)
        for index in range(22):
            stage = f'calibration-{index+1}' if index < 2 else f'train-{index-1}'
            split = 'calibration' if index < 2 else 'train'
            for offset, item in enumerate(chosen[index*2:index*2+2]):
                episode = f'{stage}-{name}-{offset}'
                schedules.setdefault(stage,[]).append({k:v for k,v in {
                    **item,'initial_state':None,'episode_id':episode,'split':split,
                    'scenario_family_id': name+':'+item['scenario_sha256'],
                    'learner_seats':roles,'opponent':'self' if len(roles)==2 else 'none',
                    'paired_seed_key':episode}.items() if k!='initial_state'})
        for offset, item in enumerate(chosen[44:60]):
            combinations = [(seat,opponent) for seat in roles for opponent in ('base','heuristic')] if len(roles)==2 else [(0,'none')]
            for seat, opponent in combinations:
                pair = f'eval-{name}-{offset}-{seat}-{opponent}'
                for endpoint in ('base','final'):
                    schedules.setdefault('eval-'+endpoint,[]).append({k:v for k,v in {
                        **item,'initial_state':None,'episode_id':endpoint+'-'+pair,'split':'eval',
                        'scenario_family_id':name+':'+item['scenario_sha256'],
                        'learner_seats':[seat],'opponent':opponent,
                        'paired_seed_key':pair}.items() if k!='initial_state'})
    verification = envs.validate_schedule(schedules, prior_fingerprints=prior_fingerprints)
    if len(schedules['eval-base']) != 224 or len(schedules['eval-final']) != 224:
        raise ValueError('Full 448-episode held-out evaluation required')
    return schedules, selections, verification


def prepare(run, deployment, metadata, file_identities, *, predecessor=None, prior_inventory=None, diagnostic_fixtures=None):
    run = Path(run)
    if run.exists(): raise FileExistsError('New run directory required')
    public = public_deployment(load(deployment))
    phase = 'training' if public['phase']=='mixed_training' else 'evaluation'
    sha = load(file_identities)
    policy_config = default_config(WEIGHTS, model_index_sha256=sha['model.safetensors.index.json'],
        tokenizer_files_sha256=digest({k:sha[k] for k in ('tokenizer.json','tokenizer_config.json')}))
    manifest = {'kind':'mixed_arena_scalar_v1','phase':phase,'created_at':now(),'deployment':public,
        'config':{'policy':policy_config,'deadline_unix':public['deadline_unix'],
            'endpoint_choices':[20], 'episodes_per_cohort':10, 'episodes_per_environment':2,'profile_safety_factor':1.5,
            'training_export_reserve_seconds':1800,'full_evaluation_episodes':448,
            'evaluation_funded_separately':True,'objective':'bounded-lag PPO; one shared model; max lag1',
            'worker_reserve_before_guard_seconds':120}, 'sources':sources(), 'inputs':{}}
    run.mkdir(parents=True)
    for relative, expected in manifest['sources'].items():
        target=run/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,target)
        if file_sha(target)!=expected:raise ValueError('Source changed while archiving')
    for source, relative in ((metadata,'inputs/model-info.json'),(file_identities,'inputs/model-file-sha256.json')):
        target=run/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    diagnostic_source = Path(diagnostic_fixtures) if diagnostic_fixtures is not None else Path(predecessor)/'inputs/execution-diagnostic-fixtures.json' if predecessor is not None else None
    if diagnostic_source is None or not diagnostic_source.is_file():raise ValueError('Frozen observed numerical diagnostic fixtures required')
    shutil.copyfile(diagnostic_source,run/'inputs/execution-diagnostic-fixtures.json')
    for name in ('config.json','tokenizer.json','tokenizer_config.json'):
        source=Path(metadata).parent/'tokenizer'/name
        if file_sha(source)!=sha[name]:raise ValueError('Portable original tokenizer identity differs')
        target=run/'inputs/tokenizer'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    exclusive_json(run/'deployment.json',public)
    if phase=='training':
        if prior_inventory is None:raise ValueError('Explicit historical scenario inventory required')
        inventory=load(prior_inventory)
        prior_fingerprints=inventory['fingerprints']
        if inventory.get('verified') is not True or not prior_fingerprints:
            raise ValueError('Verified nonempty historical scenario inventory required')
        exclusive_json(run/'inputs/prior-scenario-inventory.json',inventory)
        schedules,selections,verification=make_schedules(prior_fingerprints)
        exclusive_json(run/'inputs/schedule.json',schedules)
        exclusive_json(run/'inputs/scenario-selections.json',selections)
        exclusive_json(run/'inputs/schedule-verification.json',verification)
        exclusive_json(run/'inputs/prior-fingerprints.json',list(prior_fingerprints))
    else:
        if predecessor is None:raise ValueError('Completed audited training predecessor required')
        parent,status=verify_run(predecessor)
        if status['state']!='training_complete':raise ValueError('Training has not reached its fixed endpoint')
        audit=load(Path(predecessor)/'final-audit.json')
        if (audit.get('integrity_passed') is not True or audit.get('complete') is not True
                or audit.get('training_complete') is not True or audit.get('state')!='training_complete'
                or audit.get('manifest_sha256')!=file_sha(Path(predecessor)/'manifest.json')
                or audit.get('status_sha256')!=file_sha(Path(predecessor)/'status.json')
                or audit.get('source_auditor_sha256')!=parent['sources']['scripts/analyze_arena_mixed_scalar.py']):
            raise ValueError('Independent complete predecessor audit bound to exact manifest required')
        completion=load(Path(predecessor)/'training-complete.json')
        index=completion['attempted_cohorts']
        if audit.get('attempted_cohorts')!=index or audit.get('actual_optimizer_steps')!=completion['actual_optimizer_steps']:
            raise ValueError('Audited predecessor endpoint or actual steps differ')
        for name in ('final-audit.json','manifest.json','status.json','training-complete.json'):
            target=run/'inputs/parent'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(Path(predecessor)/name,target)
        for label,folder in [('base','base'),('final',f'cycle{index}')]:
            shutil.copytree(Path(predecessor)/'checkpoints'/folder,run/'inputs/checkpoints'/label)
        for name in ('schedule.json','model-info.json','model-file-sha256.json','prior-fingerprints.json'):
            shutil.copyfile(Path(predecessor)/'inputs'/name,run/'inputs'/name)
        exclusive_json(run/'inputs/predecessor.json',{'manifest_sha256':file_sha(Path(predecessor)/'manifest.json'),
            'audit_sha256':file_sha(Path(predecessor)/'final-audit.json'),'completion':completion})
        if parent['config']['policy']!=policy_config:raise ValueError('Evaluation policy config differs')
    manifest['inputs']={str(p.relative_to(run)):file_sha(p) for p in sorted((run/'inputs').rglob('*')) if p.is_file()}
    exclusive_json(run/'manifest.json',manifest)
    atomic_status(run,{'manifest_sha256':file_sha(run/'manifest.json'),'state':'prepared','stage':'prepared','artifacts':{},'at':now()})
    verify_run(run)
    return manifest


def verify_run(run, require_active=False):
    run=Path(run);manifest,status=load(run/'manifest.json'),load(run/'status.json')
    if manifest['kind']!='mixed_arena_scalar_v1' or status['manifest_sha256']!=file_sha(run/'manifest.json'):
        raise ValueError('Manifest identity differs')
    if load(run/'deployment.json')!=manifest['deployment'] or manifest['config']['deadline_unix']!=manifest['deployment']['deadline_unix']:
        raise ValueError('Immutable deadline/deployment differs')
    for name, expected in manifest['sources'].items():
        path=ROOT/name
        if not path.resolve().is_relative_to(ROOT) or file_sha(path)!=expected:raise ValueError('Frozen source changed: '+name)
        if file_sha(run/'source'/name)!=expected:raise ValueError('Archived frozen source differs: '+name)
    for mapping in (manifest['inputs'],status['artifacts']):
        for name,expected in mapping.items():
            path=run/name
            if not path.resolve().is_relative_to(run.resolve()) or file_sha(path)!=expected:raise ValueError('Committed artifact changed: '+name)
    if require_active and time.time()>=manifest['config']['deadline_unix']:
        raise TimeoutError('Immutable deadline exhausted')
    return manifest,status


def select_endpoint(profile_seconds, remaining_seconds, *, export_seconds=1800):
    if not profile_seconds or any(not math.isfinite(t) or t<=0 for t in profile_seconds):
        raise ValueError('Measured full-cohort costs required')
    unit=1.5*max(profile_seconds)
    choices={str(n):n*unit+export_seconds for n in (20,)}
    eligible=[int(n) for n,required in choices.items() if required<=remaining_seconds]
    return {'endpoint':max(eligible) if eligible else None,'full_cohort_upper_seconds':unit,
        'projected_seconds':choices,'remaining_seconds':remaining_seconds,
        'selection_inputs':'timing only; no training or held-out scores','complete_evaluation_reserved_separately':True}


def commit_status(run, manifest, state, stage):
    artifacts={str(p.relative_to(run)):file_sha(p) for p in sorted(run.rglob('*')) if p.is_file()
        and p.suffix in {'.json','.safetensors','.pt'} and p.name not in {'manifest.json','status.json'}
        and not p.name.startswith('.') and 'inputs' not in p.relative_to(run).parts
        and not p.name.startswith('remote-')}
    atomic_status(run,{'manifest_sha256':file_sha(run/'manifest.json'),'state':state,'stage':stage,
        'artifacts':artifacts,'at':now(),'claims_allowed':state=='complete'})


def run_experiment(run, model_dir, phase):
    from .processes import ProcessClient,LearnerClient,checkpoint_reference,GPUMonitor
    from .cuda_policy import checkpoint_state
    run=Path(run).resolve();manifest,status=verify_run(run,require_active=True)
    if phase!=manifest['phase'] or status['state']!='prepared':raise ValueError('New prepared matching phase required')
    actual={name:importlib.metadata.version(name).split('+')[0] for name in VERSIONS}
    if actual!=VERSIONS:raise ValueError('Pinned CUDA dependency versions differ')
    model_index=index_model(model_dir,run/'inputs/model-info.json')
    if model_index['files']!=load(run/'inputs/model-file-sha256.json'):raise ValueError('Staged model file identities differ')
    deadline=manifest['config']['deadline_unix']-120
    journal=Journal(run,file_sha(run/'manifest.json'),deadline)
    clients=[]
    with (run/'worker.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        journal.intent('execution-intent.json','mixed_experiment_execution',{'phase':phase,'deadline_unix':deadline})
        try:
            commit_status(run,manifest,'running','model_loading')
            learner=ProcessClient('learner',model_dir,manifest['config']['policy'],journal);clients.append(learner)
            actor=ProcessClient('actor',model_dir,manifest['config']['policy'],journal);clients.append(actor)
            learner_ready,actor_ready=learner.ready(),actor.ready()
            if learner_ready['base_identity']!=actor_ready['base_identity']:raise ValueError('Independent model copies differ')
            if phase=='training':
                base=learner.call('publish_base',{})
            else:
                base=checkpoint_reference(run/'inputs/checkpoints/base',checkpoint_state(run/'inputs/checkpoints/base'))
                learner.call('load',base)
            actor.call('load',base)
            schedules=load(run/'inputs/schedule.json')
            fixture=load(run/'inputs/execution-diagnostic-fixtures.json')
            for client,label in ((actor,'actor'),(learner,'learner')):
                diagnostic=client.call('execution_diagnostic',{'fixture':fixture})
                journal.write(label+'-execution-diagnostic.json',diagnostic)
                if not diagnostic['passed'] or not diagnostic['state_restored']:
                    raise ValueError('Scalar execution diagnostic failed; no retry')
            requests=[]
            # One distinct public starting observation per environment.
            for spec in schedules['calibration-1'][::2]:
                instance=envs.make_env(spec['environment'],spec['seed'],spec['regime'])
                player=instance.active_players()[0];view=envs.policy_view(instance,player)
                requests.append({'episode_id':'numerical-'+spec['environment'],'environment':spec['environment'],
                    'scenario_family_id':spec['scenario_family_id'],'turn':0,'player':player,
                    'observation':view['observation'],'actions':view['legal_actions'],'sample_seed':731991+len(requests)})
            smoke_a=actor.call('smoke',{'requests':requests});journal.write('actor-smoke.json',smoke_a)
            smoke_l=learner.call('smoke',{'requests':requests});journal.write('learner-smoke.json',smoke_l)
            cross=compare_actor_reports(smoke_a,smoke_l);journal.write('cross-process-admission.json',cross)
            if not cross['passed']:raise ValueError('Independent actor/learner admission failed')
            if phase=='training':
                commit_status(run,manifest,'running','calibration')
                first=actor.submit(schedules['calibration-1'],base,'calibration/1').result()
                started=time.monotonic()
                monitor=GPUMonitor().start()
                try:
                    second=actor.submit(schedules['calibration-2'],base,'calibration/2')
                    profile=learner.call('profile',{'cohort':str(run/'calibration/1'),'current':base,'behavior':base,'prefix':'calibration/1'})
                    if not profile['passed'] or profile.get('temporary_optimizer_steps')!=1:
                        raise ValueError('Full real-update profile must prove one restored Adam update')
                    second_seal=second.result()
                finally:
                    telemetry=monitor.stop();journal.write('overlap-gpu-telemetry.json',telemetry)
                if not telemetry['samples']:raise ValueError('No observed GPU memory evidence for overlap admission')
                overlap_seconds=time.monotonic()-started
                journal.write('overlap-profile.json',{'actor_seconds':second_seal['elapsed_seconds'],
                    'full_update_seconds':profile['elapsed_seconds'],'observed_overlap_seconds':overlap_seconds,
                    'parallel_processes':True,'speedup_claim':False})
                # Gate cost is included in observed elapsed overlap, not profile's backward-only timer.
                costs=[first['elapsed_seconds']+overlap_seconds,
                       second_seal['elapsed_seconds']+overlap_seconds]
                endpoint=select_endpoint(costs,deadline-time.time())
                endpoint.update(full_cost_inputs_seconds=costs,worker_deadline_unix=deadline,
                                selected_at_unix=deadline-endpoint['remaining_seconds'])
                journal.write('endpoint-selection.json',endpoint)
                if endpoint['endpoint'] is None:
                    commit_status(run,manifest,'throughput_failed','pretraining_timing_admission');return
                commit_status(run,manifest,'running','training')
                checkpoints={0:base}
                run_pipeline(actor,LearnerClient(learner,checkpoints),journal,schedules,endpoint['endpoint'],base)
                commit_status(run,manifest,'training_complete','fixed_endpoint_exported')
            else:
                final=checkpoint_reference(run/'inputs/checkpoints/final',checkpoint_state(run/'inputs/checkpoints/final'))
                for label,reference in [('base',base),('final',final)]:
                    actor.call('load',reference)
                    collect_cohort(actor.choose,schedules['eval-'+label],reference['binding'],journal,'evaluation/'+label,
                        batch_size=manifest['config']['policy']['inference_batch_size'],
                        opponent_decide=learner.choose,opponent_binding=base['binding'])
                journal.write('evaluation-seal.json',{'baseline_seal_sha256':file_sha(run/'evaluation/base/seal.json'),
                    'final_seal_sha256':file_sha(run/'evaluation/final/seal.json'),'episodes':448,
                    'paired_scenarios_per_environment':16,'sealed_at':now()})
                commit_status(run,manifest,'complete','full_evaluation_sealed')
        except BaseException as error:
            # Quiesce both writers before binding terminal retrieval to files.
            # The actor may still be completing a draw when the learner fails.
            stop_errors=[]
            for client in clients:
                try:client.stop()
                except BaseException as stop_error:
                    stop_errors.append({'type':type(stop_error).__name__,'reason':str(stop_error)})
            if any(client.process.is_alive() for client in clients):
                raise RuntimeError('Worker still alive; terminal artifact snapshot refused') from error
            clients.clear()
            journal.write('failure.json',{'error_type':type(error).__name__,'reason':str(error),'at':now(),
                'retry_allowed':False,'workers_stopped_before_artifact_scan':True,'worker_stop_errors':stop_errors})
            commit_status(run,manifest,'failed','failed')
            raise
        finally:
            for client in clients:client.stop()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','verify'))
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--phase',choices=('training','evaluation'))
    parser.add_argument('--model-dir',type=Path)
    parser.add_argument('--deployment',type=Path)
    parser.add_argument('--metadata',type=Path)
    parser.add_argument('--file-identities',type=Path)
    parser.add_argument('--predecessor',type=Path)
    parser.add_argument('--prior-inventory',type=Path)
    parser.add_argument('--diagnostic-fixtures',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':result=prepare(args.run,args.deployment,args.metadata,args.file_identities,predecessor=args.predecessor,prior_inventory=args.prior_inventory,diagnostic_fixtures=args.diagnostic_fixtures)
    elif args.command=='verify':result=verify_run(args.run)
    else:
        signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('termination requested')))
        result=run_experiment(args.run,args.model_dir,args.phase)
    print(canonical(result))


if __name__=='__main__':main()
