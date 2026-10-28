"""Fresh, fixed final evaluation after the audited FP32 training run.

No checkpoint selection, training, paid service, or original-run mutation.
The protocol is fixed before the final checkpoint exists. At most 288 games.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import uuid

from . import arena_v2, league

SOURCE_KEY = 'selfplay/arena_holdout.py'
ANALYSIS_KEYS = ('scripts/analyze_arena_holdout.py', 'scripts/analyze_arena.py')
PROTOCOL = {
    'version': 1, 'candidates': ['base', 'cycle3'],
    'opponents': ['heuristic', 'base'], 'scenarios_per_cell': 12,
    'seed_start': 3900000, 'paired_seats': True,
    'selection': 'Fixed final cycle3, regardless of preliminary evaluation outcomes',
    'purpose': 'Fresh final validation; no training or model selection on these games',
}


def make_schedule(environments):
    schedule = {}
    # Interleave candidates per scenario so deadline-limited results are paired.
    for game_index, game in enumerate(environments):
        for opponent in PROTOCOL['opponents']:
            specs = []
            for i in range(PROTOCOL['scenarios_per_cell']):
                seed = PROTOCOL['seed_start'] + game_index * 1000 + i * 2
                for candidate in PROTOCOL['candidates']:
                    for seat in (0, 1):
                        policies = [opponent, opponent]; policies[seat] = candidate
                        specs.append({'environment': game, 'seed': seed, 'regime': 'eval',
                            'policies': policies, 'candidate': candidate, 'opponent': opponent,
                            'candidate_seat': seat, 'iteration': 3, 'evaluation_phase': 'final_holdout',
                            'episode_id': f'holdout-{candidate}-{opponent}-{game}-{i}-{seat}'})
            schedule[f'eval-{game}-{opponent}'] = specs
    return schedule


def scenario_families(schedule):
    return {(r['environment'], r['seed']//2 if r['environment']=='chess' else r['seed'])
            for group in schedule.values() for r in group}


def content_sha(value):
    return hashlib.sha256(league.normalized(value).encode()).hexdigest()


def registration_path(out):
    out = Path(out).resolve()
    return out.with_name(out.name+'-preregistration.json')


def live_sources():
    root = Path(__file__).resolve().parents[1]
    return {SOURCE_KEY: Path(__file__), 'selfplay/league.py': Path(league.__file__),
            'selfplay/arena_v2.py': Path(arena_v2.__file__),
            **{key: root/key for key in ANALYSIS_KEYS}}


def historical_schedules(parent):
    """Include earlier runs that supplied the inherited base/cycle1 adapters."""
    evidence, families, seen = [],set(),set()
    current = Path(parent).resolve()
    while True:
        if str(current) in seen:
            raise ValueError('Cycle in parent experiment history')
        seen.add(str(current))
        manifest = json.loads((current/'manifest.json').read_text())
        schedule = json.loads((current/'schedule.json').read_text())
        evidence.append({'run':str(current),'manifest_sha256':league.sha(current/'manifest.json'),
                         'schedule_sha256':league.sha(current/'schedule.json')})
        families |= scenario_families(schedule)
        if not manifest.get('parent_run'):
            return evidence,families
        ancestor = Path(manifest['parent_run']).resolve()
        league.same(league.sha(ancestor/'manifest.json'),manifest['parent_manifest_sha256'],'Historical parent manifest changed')
        current = ancestor


def preregister(parent, out, deadline=league.DEFAULT_DEADLINE):
    """Freeze the protocol and executable/audit sources before cycle3 exists."""
    parent, out = Path(parent).resolve(), Path(out).resolve()
    initial = json.loads((parent/'manifest.json').read_text())
    path = registration_path(out)
    if path.exists():
        record = read_registration(path)
        validate_registration(record, parent, initial, deadline)
        for key, source in live_sources().items():
            league.same(league.sha(source), record['source_sha256'][key], 'Preregistered source changed')
        return record
    if (parent/'checkpoints/cycle3').exists():
        raise ValueError('Cannot preregister after the final checkpoint directory exists')
    if datetime.fromisoformat(deadline).tzinfo is None:
        raise ValueError('Deadline must include timezone')
    for key in ('selfplay/league.py', 'selfplay/arena_v2.py'):
        league.same(league.sha(live_sources()[key]),initial['source_sha256'][key],'Live parent helper changed')
    sources = {**initial['source_sha256'], **{key:league.sha(source) for key,source in live_sources().items()}}
    schedule = make_schedule(initial['environments'])
    history,families = historical_schedules(parent)
    if scenario_families(schedule) & families:
        raise ValueError('Holdout scenarios overlap historical parent data')
    record = {'format_version':1,'protocol':PROTOCOL,'deadline':deadline,'parent_run':str(parent),
              'parent_manifest_sha256':league.sha(parent/'manifest.json'),
              'parent_schedule_sha256':league.sha(parent/'schedule.json'),
              'schedule_sha256':content_sha(schedule),'scheduled_episodes':sum(map(len,schedule.values())),
              'source_sha256':sources,'created_at':league.timestamp(),'cycle3_absent_at_registration':True}
    record['historical_schedules'] = history
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(record,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    # Check again after publication. A race fails closed rather than calling it preregistered.
    if (parent/'checkpoints/cycle3').exists():
        raise ValueError('Final checkpoint appeared during preregistration; no valid hash published')
    with path.with_suffix('.sha256').open('x') as stream:
        stream.write(league.sha(path)+'\n');stream.flush();os.fsync(stream.fileno())
    return record


def read_registration(path):
    path = Path(path)
    league.same(path.with_suffix('.sha256').read_text().strip(),league.sha(path),'Preregistration hash differs')
    return json.loads(path.read_text())


def validate_registration(record, parent, initial, deadline):
    league.same(record['protocol'],PROTOCOL,'Preregistered protocol differs')
    league.same(record['parent_run'],str(parent.resolve()),'Preregistered parent differs')
    league.same(record['parent_manifest_sha256'],league.sha(parent/'manifest.json'),'Preregistered parent manifest changed')
    league.same(record['parent_schedule_sha256'],league.sha(parent/'schedule.json'),'Preregistered parent schedule changed')
    league.same(record['deadline'],deadline,'Preregistered deadline differs')
    league.same(record['schedule_sha256'],content_sha(make_schedule(initial['environments'])),'Preregistered schedule differs')
    league.same(record['scheduled_episodes'],sum(map(len,make_schedule(initial['environments']).values())),
                'Preregistered episode count differs')
    history,families = historical_schedules(parent)
    league.same(record['historical_schedules'],history,'Preregistered historical schedules changed')
    if scenario_families(make_schedule(initial['environments'])) & families:
        raise ValueError('Holdout scenarios overlap historical parent data')
    if record.get('cycle3_absent_at_registration') is not True:
        raise ValueError('Missing final-checkpoint absence attestation')
    for key, expected in initial['source_sha256'].items():
        league.same(record['source_sha256'][key],expected,'Preregistered parent source differs')


def validate_checkpoint(name, record, metadata, manifest, audit):
    for key,value in record.items():
        league.same(value,audit['checkpoint_verification'][name].get(key),'Checkpoint differs from complete parent audit')
    if name == 'cycle3':
        league.same(metadata['policy_arithmetic'],manifest['policy_arithmetic'],'Final checkpoint arithmetic differs')
        league.same(metadata['policy_arithmetic_sha256'],manifest['policy_arithmetic_sha256'],'Final checkpoint arithmetic hash differs')
        checks = metadata['arithmetic_verification']
        if checks.get('integer_weights_exactly_unchanged') is not True or checks.get('all_floating_parameters_float32') is not True:
            raise ValueError('Final checkpoint precision verification is incomplete')


def validate_parent(parent, manifest):
    summary = json.loads((parent/'summary.json').read_text())
    audit = json.loads((parent/'analysis.json').read_text())
    if not summary.get('complete') or not audit.get('complete') or not audit.get('all_completed_episodes_verified'):
        raise ValueError('Parent must complete and pass its independent audit first')
    league.same(audit['manifest_sha256'], league.sha(parent/'manifest.json'), 'Parent audit manifest differs')
    league.same(manifest['rating_compatibility'], arena_v2.COMPATIBILITY, 'Parent arithmetic pool differs')
    league.same(manifest['policy_arithmetic'], arena_v2.ARITHMETIC, 'Parent arithmetic profile differs')
    league.same(manifest['policy_arithmetic_sha256'], arena_v2.ARITHMETIC_SHA, 'Parent arithmetic hash differs')
    league.same(audit['policy_arithmetic_sha256'],manifest['policy_arithmetic_sha256'],'Parent audit arithmetic differs')
    league.same(audit['rating_compatibility'],manifest['rating_compatibility'],'Parent audit arithmetic pool differs')
    league.same(summary['policy_arithmetic_sha256'],manifest['policy_arithmetic_sha256'],'Parent summary arithmetic differs')
    if audit.get('missing_episode_ids'):
        raise ValueError('Parent audit is missing scheduled episodes')
    schedule,_,_ = arena_v2.make_schedule(manifest['environments'])
    league.same(json.loads((parent/'schedule.json').read_text()),schedule,'Parent schedule differs from frozen protocol')
    expected_files = {stage+'.jsonl' for stage in schedule}
    league.same(sorted(audit['rollout_files']),sorted(expected_files),'Parent audit does not cover the full schedule')
    for filename, evidence in audit['rollout_files'].items():
        league.same(league.sha(parent/filename),evidence['sha256'],'Parent audited rollout bytes changed')
        league.same(evidence['episodes'],len(schedule[filename[:-6]]),'Parent audited episode count differs')
    total = sum(map(len,schedule.values()))
    league.same(audit['scheduled_episodes'],total,'Parent audit schedule count differs')
    league.same(audit['episodes_verified'],total,'Parent audit episode count differs')
    for key,expected in manifest['source_sha256'].items():
        league.same(league.sha(parent/'source'/Path(key).name),expected,'Parent source archive changed')
    return audit


def prepare(parent, out, deadline=league.DEFAULT_DEADLINE):
    parent, out = Path(parent).resolve(), Path(out).resolve()
    if datetime.fromisoformat(deadline).tzinfo is None:
        raise ValueError('Deadline must include timezone')
    initial = json.loads((parent/'manifest.json').read_text())
    registration = read_registration(registration_path(out))
    validate_registration(registration,parent,initial,deadline)
    for key, source in live_sources().items():
        league.same(league.sha(source),registration['source_sha256'][key],'Preregistered live source changed')
    audit = validate_parent(parent,initial)
    if out.exists():
        manifest = json.loads((out/'manifest.json').read_text())
        league.same(manifest['parent_run'], str(parent), 'Holdout parent differs')
        league.same(manifest['deadline'], deadline, 'Holdout deadline differs')
        league.same(manifest['holdout_protocol'], PROTOCOL, 'Holdout protocol differs')
        league.same(manifest['source_sha256'],registration['source_sha256'],'Holdout source set differs from preregistration')
        league.same(league.sha(out/'preregistration.json'),manifest['preregistration_sha256'],'Archived preregistration changed')
        league.same(manifest['preregistration_sha256'],league.sha(registration_path(out)),'Preregistration identity differs')
        for key, expected in manifest['source_sha256'].items():
            league.same(league.sha(out/'source'/Path(key).name), expected, 'Holdout archive differs')
        league.same(league.sha(parent/'manifest.json'), manifest['parent_manifest_sha256'], 'Parent manifest changed')
        league.same(league.sha(out/'parent-analysis.json'), manifest['parent_analysis_sha256'], 'Archived parent audit changed')
        league.same(league.sha(parent/'analysis.json'),manifest['parent_analysis_sha256'],'Parent audit changed since preparation')
        for name, expected in manifest['inherited_checkpoints'].items():
            actual, metadata = league.checkpoint_record(expected['path'], name, manifest['model'])
            league.same(actual, expected, 'Inherited checkpoint changed')
            validate_checkpoint(name,actual,metadata,manifest,audit)
        schedule = make_schedule(manifest['environments'])
        league.same(json.loads((out/'schedule.json').read_text()), schedule, 'Holdout schedule changed')
        return manifest, schedule
    schedule = make_schedule(initial['environments'])
    if scenario_families(schedule) & scenario_families(json.loads((parent/'schedule.json').read_text())):
        raise ValueError('Holdout scenarios overlap parent training/evaluation')
    inherited = {}
    for name in PROTOCOL['candidates']:
        old = initial['inherited_checkpoints'].get(name)
        directory = Path(old['path']) if old else parent/'checkpoints'/name
        record, metadata = league.checkpoint_record(directory, name, initial['model'])
        validate_checkpoint(name,record,metadata,initial,audit)
        inherited[name] = record
    sources = registration['source_sha256']
    manifest = {**initial, 'experiment': 'arena-final-holdout-v1', 'holdout_protocol': PROTOCOL,
        'parent_run': str(parent), 'parent_manifest_sha256': league.sha(parent/'manifest.json'),
        'parent_analysis_sha256': league.sha(parent/'analysis.json'), 'deadline': deadline,
        'preregistration_sha256':league.sha(registration_path(out)),
        'inherited_checkpoints': inherited, 'source_sha256': sources, 'continuation_cycles': [],
        'initial_evaluation_stages': [], 'evaluation_rosters': {}, 'rollout_chunk_episodes': 4,
        'training': 'None. Fixed original and final policy evaluated on fresh scenarios.',
        'created_at': league.timestamp(), 'external_spend_usd': 0}
    temporary = out.parent/('.'+out.name+'.preparing-'+uuid.uuid4().hex)
    temporary.mkdir(parents=True); (temporary/'source').mkdir(); (temporary/'rollout-parts').mkdir()
    for key, expected in sources.items():
        source = live_sources().get(key,parent/'source'/Path(key).name)
        league.same(league.sha(source), expected, 'Source changed before holdout')
        shutil.copyfile(source, temporary/'source'/Path(key).name)
    shutil.copyfile(parent/'analysis.json', temporary/'parent-analysis.json')
    shutil.copyfile(registration_path(out),temporary/'preregistration.json')
    shutil.copyfile(registration_path(out).with_suffix('.sha256'),temporary/'preregistration.sha256')
    league.atomic_json(temporary/'manifest.json', manifest)
    league.atomic_json(temporary/'schedule.json', schedule)
    league.atomic_json(temporary/'status.json', {'status':'prepared','stages':{},'external_spend_usd':0})
    temporary.rename(out)
    return manifest, schedule


class Holdout(league.Continuation):
    def recovered_rollouts(self,stage):
        # Do not let ordinary resume re-canonicalization replace a prior byte
        # commitment with a new hash after a valid-looking record was altered.
        prior = self.status.get('stages',{}).get(stage,{})
        if prior.get('sha256'):
            league.same(league.sha(self.out/(stage+'.jsonl')),prior['sha256'],
                        'Completed holdout rollout bytes changed before resume')
        return super().recovered_rollouts(stage)

    def run(self):
        try:
            for stage in self.schedule:
                if self.rollout_stage(stage) is None:
                    return self.status
            league.atomic_json(self.out/'summary.json', {'complete':True, 'training':False,
                'protocol':PROTOCOL, 'finished_at':league.timestamp(),
                'episodes':sum(map(len,self.schedule.values())), 'external_spend_usd':0,
                'stage_records':self.status['stages']})
            self.record('complete', None)
            return self.status
        except Exception as error:
            self.record('failed', self.status.get('current_stage'), error=f'{type(error).__name__}: {error}')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=Path('runs/arena-v2'))
    parser.add_argument('--out', type=Path, default=Path('runs/arena-final-holdout-v1'))
    parser.add_argument('--deadline', default=league.DEFAULT_DEADLINE)
    parser.add_argument('--preregister',action='store_true',help='Freeze protocol and source hashes only; no model allocation')
    args = parser.parse_args()
    if args.preregister:
        with league.process_lock(args.out):
            preregister(args.parent,args.out,args.deadline)
        print(json.dumps({'status':'preregistered','path':str(registration_path(args.out)),
                          'sha256':league.sha(registration_path(args.out)),'external_spend_usd':0}))
        return
    with league.process_lock(Path(__file__).resolve().parents[1]/'runs/local-neural'), league.process_lock(args.out):
        manifest, schedule = prepare(args.parent, args.out, args.deadline)
        for name, expected in manifest['versions'].items():
            league.same(importlib.metadata.version(name), expected, 'Holdout runtime version changed')
        arena, policy, experiment = league.archived_arena(args.out, manifest)
        runner = Holdout(args.out, manifest, schedule, arena, policy, experiment,
                        policy_factory=lambda:arena_v2.precision_policy(policy,manifest['model']))
        result = runner.run()
    print(json.dumps({'status':result['status'], 'run':str(args.out.resolve()), 'external_spend_usd':0}))


if __name__ == '__main__':
    main()
