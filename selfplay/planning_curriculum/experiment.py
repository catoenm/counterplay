"""Fixed admission and four-arm planning study, one immutable allocation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import time

from selfplay.arena_mixed_scalar.experiment import (
    atomic_status, commit_status, index_model, VERSIONS, sources as scalar_sources)
from . import envs, schedule
from .admission import signal_gate, timing_gate, build_context_bound
from .cuda_policy import MODEL as MODEL_REPO, REVISION as MODEL_REVISION, default_config, compare_actor_reports
from .runner import Journal, canonical, digest, exclusive_json, file_sha, load_sealed_cohort, now, run_pipeline

ROOT = Path(__file__).resolve().parents[2]
TERMINAL = {'complete', 'admission_signal_failed', 'admission_timing_failed', 'failed'}
OVERHEAD_CONTRACT = {
    'fixed_cpu_ipc_publication_seconds': 1800,
    'load_units': 165, 'roundtrip_units': 88, 'margin_units': 27,
    'load_unit': 'max(initial_actor_checkpoint_load, probe_checkpoint_save_readback)',
    'roundtrip_unit': 'max(mature_Adam_state_roundtrip, probe_unattributed_full_state_overhead)',
    'covered_operations': {'actor_training_loads': 80, 'learner_behavior_load_restore': 80,
        'evaluation_loads': 5, 'arm_exact_resets': 8, 'additional_margin': 27},
    'model_loads_after_admission': 0,
    'outer_safety_factor': 1.5,
}


def load(path):
    return json.loads(Path(path).read_text())


def public_deployment(plan):
    # Validate locally without exposing guard tokens or SSH private paths.
    from scripts.runpod_planning_curriculum import validate, STORAGE_RATE
    if 'guard_secret_path' in plan:
        validate(plan)
    keys = ('phase', 'intent', 'project', 'created_at', 'deadline_unix', 'maximum_hours',
        'reservation_usd', 'estimate_usd', 'gpu_id', 'maximum_gpu_rate_usd_hour',
        'maximum_cpu_rate_usd_hour', 'gpu_disk_gb', 'guard_disk_gb', 'gpu_image',
        'guard_source_sha256', 'primitives_sha256', 'controller_sha256', 'stage_caps_usd',
        'protected_export_seconds', 'gpu_hourly_rate', 'guard_hourly_rate',
        'storage_hourly_rate', 'other_reserved_cost_usd')
    public = {key: plan[key] for key in keys}
    fixed = {'phase':'planning_curriculum', 'project':'cardinal-gpu', 'maximum_hours':12,
        'reservation_usd':90, 'stage_caps_usd':{'admission':15.,'comparison':65.,'protected':10.},
        'protected_export_seconds':3600, 'maximum_cpu_rate_usd_hour':.14,
        'guard_hourly_rate':.14, 'storage_hourly_rate':STORAGE_RATE,
        'other_reserved_cost_usd':0, 'gpu_disk_gb':160, 'guard_disk_gb':10,
        'gpu_image':'runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404'}
    if any(public.get(k)!=v for k,v in fixed.items()):
        raise ValueError('Public lease envelope differs')
    created = datetime.fromisoformat(public['created_at'])
    rate, deadline = public['gpu_hourly_rate'], public['deadline_unix']
    if (created.tzinfo is None or type(rate) not in (int,float) or not math.isfinite(rate)
            or not 0 < rate <= 5.29 or rate != public['maximum_gpu_rate_usd_hour']
            or type(deadline) not in (int,float) or not math.isfinite(deadline)
            or int(deadline)!=deadline or not 43200-1 < deadline-created.timestamp() <= 43200
            or abs(public['estimate_usd']-12*(rate+.14+STORAGE_RATE))>1e-9
            or not isinstance(public['intent'],str) or len(public['intent'])!=32
            or any(c not in '0123456789abcdef' for c in public['intent'])):
        raise ValueError('Public rates, deadline or intent differ')
    for key,name in (('guard_source_sha256','scripts/cloud/cardinal_guard.py'),
            ('primitives_sha256','scripts/runpod_cardinal.py'),
            ('controller_sha256','scripts/runpod_planning_curriculum.py')):
        if public[key] != file_sha(ROOT/name):
            raise ValueError('Public lifecycle source identity differs')
    return public


def sources():
    names = set(scalar_sources())
    names.update(str(p.relative_to(ROOT)) for p in (ROOT/'selfplay/planning_curriculum').glob('*.py'))
    names.update(('scripts/runpod_planning_curriculum.py', 'scripts/planning_curriculum_supervisor.py',
        'scripts/cloud/bootstrap_planning_curriculum.py', 'scripts/cloud/run_planning_curriculum_job.py',
        'scripts/planning_curriculum_dispatch.py','scripts/analyze_planning_curriculum.py'))
    names.update(str(p.relative_to(ROOT)) for p in (ROOT/'tests').glob('test_*planning*.py'))
    names.update(('tests/test_arena_mixed_cuda_policy.py','tests/test_runpod_cardinal.py'))
    return {name: file_sha(ROOT/name) for name in sorted(names)}


def prepare(run, deployment, preflight, model_inputs):
    run, preflight, model_inputs = Path(run), Path(preflight), Path(model_inputs)
    if run.exists():
        raise FileExistsError('New run directory required')
    plan = public_deployment(load(deployment))
    reviewed = load(preflight/'implementation-review.json')
    current_sources = sources()
    if reviewed.get('passed') is not True or reviewed.get('sources') != current_sources:
        raise ValueError('Independent implementation review must bind every live source')
    context = load(preflight/'context-bound.json')
    proof = build_context_bound(model_inputs/'tokenizer')
    if {k:v for k,v in context.items() if k!='review_sha256'} != proof:
        raise ValueError('Reviewed CPU context proof differs')
    if context.get('review_sha256') != file_sha(preflight/'context-review.json'):
        raise ValueError('Context peer review binding differs')
    context_review = load(preflight/'context-review.json')
    if (context_review.get('passed') is not True
            or context_review.get('proof_file_sha256') != file_sha(preflight/'context-proof.json')
            or context_review.get('proof_sha256') != proof['proof_sha256']):
        raise ValueError('Context proof did not pass peer review')
    overhead = load(preflight/'overhead-contract.json')
    if overhead != OVERHEAD_CONTRACT or reviewed.get('overhead_contract_sha256') != digest(overhead):
        raise ValueError('Timing fixed overhead must be reviewed before inference')
    inventory = load(preflight/'prior-scenario-inventory.json')
    if (inventory.get('kind')!='planning_curriculum_historical_inventory_v1'
            or inventory.get('verified') is not True or not inventory.get('fingerprints')
            or not inventory.get('source_sha256')):
        raise ValueError('Verified nonempty historical family exclusion inventory required')
    for relative, expected in inventory['source_sha256'].items():
        if file_sha(ROOT/relative) != expected:
            raise ValueError('Historical source inventory changed')
    stages = load(preflight/'schedule.json')
    verification = schedule.validate_schedules(stages, inventory['fingerprints'])
    if verification != load(preflight/'schedule-verification.json'):
        raise ValueError('Frozen split validation differs')
    sha = load(model_inputs/'model-file-sha256.json')
    config = default_config(model_index_sha256=sha['model.safetensors.index.json'],
        tokenizer_files_sha256=digest({k:sha[k] for k in ('tokenizer.json','tokenizer_config.json')}))
    manifest = {'kind': 'planning_curriculum_study_v1', 'phase': 'planning_curriculum',
        'created_at': now(), 'deployment': plan, 'config': {'policy': config,
        'deadline_unix': plan['deadline_unix'], 'maximum_policy_episodes': 544,
        'attempted_retained_updates': 80, 'max_policy_lag': 1,
        'numerical_diagnostics': {'requests': 'first fixed hard-screen initial public observation',
            'actor_draws': 1, 'learner_draws': 1, 'new_episodes': 0,
            'full_vector_gate': .002, 'derivative_and_exact_restore': True},
        'temporary_probe': 'first two fixed depth6 hard-screen episodes, one step, exact restoration',
        'overhead_contract': overhead}, 'sources': current_sources, 'inputs': {}}
    run.mkdir(parents=True)
    for relative, expected in current_sources.items():
        destination = run/'source'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, destination)
        if file_sha(destination) != expected:
            raise ValueError('Source changed during freezing')
    for source in sorted(preflight.glob('*.json')):
        destination = run/'inputs'/source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    for name in ('model-info.json', 'model-file-sha256.json'):
        shutil.copyfile(model_inputs/name, run/'inputs'/name)
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
        source, destination = model_inputs/'tokenizer'/name, run/'inputs/tokenizer'/name
        if file_sha(source) != sha[name]:
            raise ValueError('Pinned tokenizer/config hash differs')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    exclusive_json(run/'deployment.json', plan)
    manifest['inputs'] = {str(p.relative_to(run)):file_sha(p) for p in sorted((run/'inputs').rglob('*')) if p.is_file()}
    exclusive_json(run/'manifest.json', manifest)
    atomic_status(run, {'manifest_sha256': file_sha(run/'manifest.json'), 'state':'prepared',
        'stage':'prepared', 'artifacts': {}, 'at': now()})
    verify_run(run)
    return manifest


def verify_run(run, require_active=False):
    run = Path(run).resolve()
    manifest, status = load(run/'manifest.json'), load(run/'status.json')
    if manifest['kind'] != 'planning_curriculum_study_v1' or status['manifest_sha256'] != file_sha(run/'manifest.json'):
        raise ValueError('Planning manifest identity differs')
    if load(run/'deployment.json') != manifest['deployment'] or manifest['config']['deadline_unix'] != manifest['deployment']['deadline_unix']:
        raise ValueError('Immutable deployment/deadline differs')
    for relative, expected in manifest['sources'].items():
        path = (ROOT/relative).resolve()
        if not path.is_relative_to(ROOT) or file_sha(path) != expected or file_sha(run/'source'/relative) != expected:
            raise ValueError('Frozen source changed: '+relative)
    for mapping in (manifest['inputs'], status['artifacts']):
        for relative, expected in mapping.items():
            path = (run/relative).resolve()
            if not path.is_relative_to(run) or file_sha(path) != expected:
                raise ValueError('Committed input/artifact changed: '+relative)
    if require_active and time.time() >= manifest['deployment']['deadline_unix']-3600:
        raise TimeoutError('Scientific work deadline exhausted')
    return manifest, status


def lease_clocks(plan, checked=None, comparison_started=None):
    checked = time.time() if checked is None else checked
    started = datetime.fromisoformat(plan['created_at']).timestamp()
    hourly = sum(plan[k] for k in ('gpu_hourly_rate','guard_hourly_rate','storage_hourly_rate'))
    admission_stop = min(plan['deadline_unix']-3600,
        started+(15-plan['other_reserved_cost_usd'])/hourly*3600)
    result = {'started': started, 'checked': checked, 'hourly': hourly,
        'admission_stop': admission_stop, 'scientific_stop': plan['deadline_unix']-3600,
        'admission_spent': max(0., checked-started)/3600*hourly+plan['other_reserved_cost_usd']}
    if comparison_started is not None:
        result['comparison_stop'] = min(plan['deadline_unix']-3600, comparison_started+65/hourly*3600)
    return result


def main_overhead(profile, base_load_elapsed):
    for value in (base_load_elapsed, profile['elapsed_seconds'], profile['gate_seconds'],
            profile['temporary_update']['elapsed_seconds'], profile['checkpoint_seconds'],
            profile['mature_state_roundtrip']['seconds']):
        if type(value) not in (int,float) or not math.isfinite(value) or value <= 0:
            raise ValueError('Finite positive observed full-work timing required')
    remainder = max(0., profile['elapsed_seconds']-profile['gate_seconds']-
        profile['temporary_update']['elapsed_seconds']-profile['checkpoint_seconds'])
    load_unit = max(base_load_elapsed, profile['checkpoint_seconds'])
    roundtrip = profile['mature_state_roundtrip']
    if (roundtrip.get('passed') is not True or not isinstance(roundtrip.get('state_before'),dict)
            or roundtrip.get('state_before') != roundtrip.get('state_after')
            or roundtrip.get('optimizer_steps') != 0 or roundtrip.get('model_forwards') != 0):
        raise ValueError('Measured mature Adam state-only restoration is required')
    roundtrip_unit = max(roundtrip['seconds'], remainder)
    return {'contract': OVERHEAD_CONTRACT, 'initial_actor_checkpoint_load_seconds': base_load_elapsed,
        'probe_unattributed_full_state_seconds': remainder, 'load_unit_seconds':load_unit,
        'roundtrip_unit_seconds':roundtrip_unit,
        'fixed_main_overhead_seconds':1800+165*load_unit+88*roundtrip_unit+27*max(load_unit,roundtrip_unit)}


def run_experiment(run, model_dir):
    from .processes import ProcessClient, ArmActor, ArmLearner
    run = Path(run).resolve()
    manifest, status = verify_run(run, require_active=True)
    if status['state'] != 'prepared':
        raise ValueError('A new prepared run is required; no resume')
    actual = {name:importlib.metadata.version(name).split('+')[0] for name in VERSIONS}
    if actual != VERSIONS:
        raise ValueError('Pinned CUDA dependency versions differ')
    if index_model(model_dir, run/'inputs/model-info.json')['files'] != load(run/'inputs/model-file-sha256.json'):
        raise ValueError('Staged model identities differ')
    plan = manifest['deployment']
    clocks = lease_clocks(plan)
    outer = Journal(run, file_sha(run/'manifest.json'), clocks['scientific_stop'])
    admission = Journal(run, outer.run_sha256, clocks['admission_stop'])
    clients = []
    terminal, terminal_stage = 'failed', 'failed'
    with (run/'worker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        admission.intent('execution-intent.json', 'planning_curriculum_execution',
            {'deadline_unix': plan['deadline_unix'], 'admission_deadline_unix': admission.deadline})
        try:
            commit_status(run, manifest, 'running', 'admission_model_loading')
            learner = ProcessClient('learner', model_dir, manifest['config']['policy'], outer); clients.append(learner)
            actor = ProcessClient('actor', model_dir, manifest['config']['policy'], outer); clients.append(actor)
            for client in clients: client.journal = admission
            learner_ready, actor_ready = learner.ready(), actor.ready()
            if learner_ready['base_identity'] != actor_ready['base_identity']:
                raise ValueError('Independent actor/learner model identities differ')
            base = learner.call('publish_base', {})
            before = time.monotonic()
            actor.call('load', base)
            base_load_elapsed = time.monotonic()-before
            stages = load(run/'inputs/schedule.json')
            spec = stages['screen-hard'][0]
            instance = envs.make_env('planning', spec['seed'], spec['regime'], depth=spec['depth'])
            view = envs.policy_view(instance, 0)
            requests = [{'episode_id':'numerical-planning', 'environment':'planning',
                'scenario_family_id':spec['scenario_family_id'], 'turn':0, 'player':0,
                'observation':view['observation'], 'actions':view['legal_actions'], 'sample_seed':731991}]
            smokes = []
            for client in (actor, learner):
                smoke = client.call('smoke', {'requests':requests})
                admission.write(f'admission/{client.role}-smoke.json', smoke)
                if not smoke['passed'] or not smoke['state_restored']:
                    raise ValueError('Numerical derivative/full-vector/exact-restore admission failed')
                smokes.append(smoke)
            cross = compare_actor_reports(*smokes)
            admission.write('admission/cross-process.json', cross)
            if not cross['passed']:
                raise ValueError('Actor/learner numerical agreement failed')
            commit_status(run, manifest, 'running', 'admission_screen')
            screen_specs = stages['screen-hard']+stages['screen-easy']
            actor.call('collect_profiled', {'specs':screen_specs, 'checkpoint':base, 'prefix':'admission/screen'})
            _, screen = load_sealed_cohort(run/'admission/screen')
            signal = signal_gate(screen, screen_specs)
            admission.write('admission/signal-gate.json', signal)
            if not signal['passed']:
                terminal, terminal_stage = 'admission_signal_failed', 'signal_gate'
                return
            profile = learner.call('timing_probe', {'cohort':str(run/'admission/screen'),
                'episode_ids':signal['timing_probe_episode_ids']})
            overhead = main_overhead(profile, base_load_elapsed) if profile['passed'] else None
            if overhead is None:
                raise ValueError('Once-only restored timing probe failed')
            admission.write('admission/main-overhead.json', overhead)
            measurements = [load(p) for p in sorted((run/'admission/screen/batches').glob('*.json')) if not p.name.endswith('-intent.json')]
            checked = time.time()
            clocks = lease_clocks(plan, checked=checked)
            budget = {k:plan[k] for k in ('gpu_hourly_rate','guard_hourly_rate','storage_hourly_rate','other_reserved_cost_usd')}
            budget.update(aggregate_started_at=plan['created_at'], checked_at=datetime.fromtimestamp(checked,timezone.utc).isoformat(),
                hard_stop=datetime.fromtimestamp(plan['deadline_unix'], timezone.utc).isoformat(),
                admission_spent_usd=clocks['admission_spent'],
                fixed_main_overhead_seconds=overhead['fixed_main_overhead_seconds'],
                overhead_review_sha256=file_sha(run/'inputs/implementation-review.json'))
            admission.write('admission/timing-budget.json', budget)
            decision = timing_gate(profile, measurements, load(run/'inputs/context-bound.json'), budget)
            admission.write('admission/timing-gate.json', decision)
            if not decision['passed']:
                terminal, terminal_stage = 'admission_timing_failed', 'timing_gate'
                return
            admission.admit()
            comparison_started = time.time()
            main = Journal(run, outer.run_sha256, lease_clocks(plan, comparison_started=comparison_started)['comparison_stop'])
            main.intent('comparison-admission.json', 'planning_comparison_admitted',
                {'signal_sha256':file_sha(run/'admission/signal-gate.json'),
                 'timing_sha256':file_sha(run/'admission/timing-gate.json'),
                 'comparison_started_unix':comparison_started, 'comparison_deadline_unix':main.deadline,
                 'admission_spent_upper_usd':lease_clocks(plan, checked=comparison_started)['admission_spent'],
                 'retained_updates':80, 'evaluation_episodes':320})
            for client in clients: client.journal = main
            endpoints = {'base':base}
            for seed in schedule.TRAINING_SEEDS:
                for arm in schedule.ARMS:
                    label = f'{seed}-{arm}'
                    prefix = 'arms/'+label
                    arm_journal = Journal(run/prefix, outer.run_sha256, main.deadline)
                    commit_status(run, manifest, 'running', 'training_'+label)
                    for client in (actor, learner):
                        reset = client.call('reset_base', {'binding':base['binding']})
                        arm_journal.write(client.role+'-base-restoration.json', reset)
                    completion = run_pipeline(ArmActor(actor,prefix), ArmLearner(learner,prefix,base),
                        arm_journal, schedule.training_stages(stages,seed,arm),20,base)
                    endpoints[label] = completion['endpoint']
            for label, reference in endpoints.items():
                commit_status(run, manifest, 'running', 'evaluation_'+label)
                actor.submit(stages['eval-'+label], reference, 'evaluation/'+label).result()
            main.admit()
            main.write('comparison-complete.json', {'kind':'planning_comparison_complete_v1',
                'attempted_updates':80, 'evaluation_episodes':320, 'endpoints':endpoints,
                'all_draws_preserved':True, 'finished_at':now()})
            terminal, terminal_stage = 'complete', 'full_comparison_exported'
        except BaseException as error:
            # Do not bind a terminal artifact list while any child writer exists.
            outer.write('failure.json', {'error_type':type(error).__name__, 'reason':str(error),
                'at':now(), 'retry_allowed':False})
            raise
        finally:
            shutdown_errors = []
            for client in clients:
                try: client.stop()
                except BaseException as stop_error:
                    shutdown_errors.append({'role':client.role,'error_type':type(stop_error).__name__})
            if any(client.process.is_alive() for client in clients):
                raise RuntimeError('Refuse terminal snapshot while child writers remain')
            if shutdown_errors:
                outer.write('shutdown-errors.json', {'errors':shutdown_errors, 'all_children_stopped':True})
                terminal, terminal_stage = 'failed', 'shutdown_errors'
            commit_status(run, manifest, terminal, terminal_stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare','run','verify'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--deployment', type=Path)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--model-inputs', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.run,args.deployment,args.preflight,args.model_inputs)
        print(canonical({'manifest_sha256':file_sha(args.run/'manifest.json'),'kind':result['kind']}))
    elif args.command == 'verify':
        manifest,status = verify_run(args.run)
        print(canonical({'manifest_sha256':file_sha(args.run/'manifest.json'),'state':status['state']}))
    else:
        run_experiment(args.run,args.model_dir)


if __name__ == '__main__': main()
