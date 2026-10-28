"""Durable explicit-depth planning rollouts and a bounded-lag actor/learner schedule.

This module imports no neural runtime. Every external computation is preceded
by an exclusive intent. An interrupted invocation is evidence, never permission
to redraw or repeat an optimizer step. Completed cohorts are immutable files.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import uuid

from . import envs


def canonical(value):
    return envs.canonical(value)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def exclusive_json(path, value):
    """Publish complete bytes without replacing an earlier attempt/result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    try:
        with temporary.open('x') as stream:
            stream.write(canonical(value) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return file_sha(path)


def stable_seed(*parts):
    return int(digest(list(parts))[:15], 16)


class Journal:
    def __init__(self, folder, run_sha256, deadline):
        self.folder, self.run_sha256, self.deadline = Path(folder), run_sha256, deadline

    def admit(self):
        if time.time() >= self.deadline:
            raise TimeoutError('Immutable worker deadline reached')

    def path(self, relative):
        path = self.folder / relative
        if not path.resolve().is_relative_to(self.folder.resolve()):
            raise ValueError('Journal path escapes run')
        return path

    def write(self, relative, value):
        return exclusive_json(self.path(relative), value)

    def intent(self, relative, kind, payload):
        self.admit()
        value = {'kind': kind, 'run_sha256': self.run_sha256, 'at': now(), **payload}
        return self.write(relative, value)


def verify_transitions(episode):
    """Bind neural policy rows to the replayed decision view, not just rewards."""
    replay = envs.verify_episode(episode)
    expected = {(event['turn'], player): event for event in episode['events']
                for player in event['active_players'] if player in episode['learner_seats']}
    observed = {}
    for row in episode['transitions']:
        key = (row['turn'], row['player'])
        if key in observed or key not in expected:
            raise ValueError('Duplicate or undeclared learner action')
        event = expected[key]
        player = str(row['player'])
        normalized = json.loads(canonical(event))
        checks = ((row['observation'], normalized['observations'][player]),
                  (row['actions'], normalized['legal_actions'][player]),
                  (row['action'], normalized['actions'][player]),
                  (row['terminal_return'], episode['returns'][row['player']]),
                  (row['checkpoint'], episode['behavior_binding']))
        if any(canonical(a) != canonical(b) for a, b in checks):
            raise ValueError('Neural row differs from exact replay decision')
        if row['episode_id'] != episode['episode_id'] or row['environment'] != episode['environment']:
            raise ValueError('Neural row belongs to another episode')
        observed[key] = row
    if set(observed) != set(expected):
        raise ValueError('Incomplete neural action coverage')
    return {**replay, 'learner_actions': len(observed)}


def verify_batch_timing(batch, profiled):
    """Check the saved timing envelope without changing any sampled row."""
    if type(profiled) is not bool or batch.get('profiled') is not profiled:
        raise ValueError('Sampling timing mode differs from immutable intent')
    if not profiled:
        if 'timing' in batch:
            raise ValueError('Unprofiled draw contains undeclared timing')
        return
    timing = batch.get('timing')
    if (not isinstance(timing, dict) or timing.get('kind') != 'planning_screen_sampling_timing_v1'
            or timing.get('row_sha256') != [digest(row) for row in batch['rows']]
            or not isinstance(timing.get('forward_timings'), list)
            or len(timing['forward_timings']) != len(batch['rows'])):
        raise ValueError('Profiled draw timing lacks exact sampled-row coverage')
    for name in ('elapsed_seconds', 'nonforward_seconds'):
        value = timing.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid finite sampling timing')
    for row in timing['forward_timings']:
        seconds = row.get('seconds')
        if (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0
                or type(row.get('input_tokens')) is not int or row['input_tokens'] < 1
                or row.get('training') is not False or row.get('grad_enabled') is not False
                or not isinstance(row.get('row_sha256'), str) or len(row['row_sha256']) != 64):
            raise ValueError('Invalid scalar sampling forward timing')
    expected_nonforward = max(0., timing['elapsed_seconds'] - sum(row['seconds'] for row in timing['forward_timings']))
    if timing['nonforward_seconds'] != expected_nonforward:
        raise ValueError('Sampling timing accounting differs')


def collect_cohort(decide, specs, binding, journal, prefix, *, batch_size=8,
                   opponent_decide=None, opponent_binding=None, profiled=False):
    """Batch independent decision requests; step only after all players choose.

    ``decide`` has the backend choose(requests,binding) signature. A frozen base
    opponent may be a separate process via ``opponent_decide``. No outcome or
    omniscient snapshot is passed to either policy. Training uses both seats;
    single-player tasks use only their one real seat.
    """
    if not specs or type(profiled) is not bool or type(batch_size) is not int or batch_size < 1 or len({s['episode_id'] for s in specs}) != len(specs):
        raise ValueError('Nonempty unique episode schedule required')
    started = time.monotonic()
    intent_sha = journal.intent(prefix + '/intent.json', 'mixed_cohort_draw',
        {'specs': specs, 'binding': binding, 'opponent_binding': opponent_binding, 'profiled': profiled})
    instances, records = [], []
    for spec in specs:
        instance = envs.make_env(spec['environment'], spec['seed'], spec['regime'], depth=spec['depth'])
        seats = spec['learner_seats']
        roles = envs.protocol_config()['roles'][instance.name]
        if (type(seats) is not list or not seats or any(type(p) is not int for p in seats)
                or len(set(seats)) != len(seats) or not set(seats) <= set(roles)):
            raise ValueError('Invalid real learner seats')
        record = envs.episode_header(instance, spec['episode_id'], learner_seats=seats,
            behavior_binding=binding, schedule=spec, transitions=[], opponent_transitions=[])
        if record['scenario_sha256'] != spec['scenario_sha256']:
            raise ValueError('Initial scenario differs from frozen schedule')
        instances.append(instance)
        records.append(record)
    batch_number = 0
    while any(not instance.done for instance in instances):
        journal.admit()
        grouped = {'learner': [], 'opponent': []}
        actions, menus = [{} for _ in specs], [{} for _ in specs]
        for index, (instance, spec) in enumerate(zip(instances, specs, strict=True)):
            for player in instance.active_players():
                view = envs.policy_view(instance, player)
                menu = view['legal_actions']
                # Endpoint is omitted: held-out baseline/final use common seeds.
                seed_key = spec.get('paired_seed_key', spec['episode_id'])
                menu_seed = stable_seed('menu', seed_key, instance.turn, player)
                random.Random(menu_seed).shuffle(menu)
                menus[index][player] = menu
                request = {'episode_id': spec['episode_id'], 'environment': instance.name,
                    'scenario_family_id': records[index]['scenario_family_id'],
                    'turn': instance.turn, 'player': player, 'observation': view['observation'],
                    'actions': menu, 'menu_seed': menu_seed,
                    'sample_seed': stable_seed('action', seed_key, instance.turn, player)}
                if player in spec['learner_seats']:
                    grouped['learner'].append((index, request))
                elif spec['opponent'] == 'heuristic':
                    actions[index][player] = instance.heuristic_action(player)
                elif spec['opponent'] == 'base' and opponent_decide is not None:
                    grouped['opponent'].append((index, request))
                else:
                    raise ValueError('Opponent or separate frozen policy unavailable')
        for role, requests in grouped.items():
            for offset in range(0, len(requests), batch_size):
                batch = requests[offset:offset+batch_size]
                selected_binding = binding if role == 'learner' else opponent_binding
                request_rows = [row for _, row in batch]
                batch_prefix = f'{prefix}/batches/{batch_number:06d}'
                batch_intent = journal.intent(batch_prefix+'-intent.json', 'mixed_action_batch',
                    {'role': role, 'binding': selected_binding, 'requests': request_rows, 'profiled': profiled})
                call = decide if role == 'learner' else opponent_decide
                before = time.monotonic()
                response = call(request_rows, selected_binding)
                elapsed = time.monotonic()-before
                results = response.get('rows') if profiled and isinstance(response,dict) else response
                # Commit raw output before any validation or environment scoring.
                raw = {'intent_sha256': batch_intent, 'role': role, 'rows': results,
                       'profiled': profiled, 'elapsed_seconds': elapsed, 'finished_at': now()}
                if profiled:
                    raw['timing'] = response.get('timing') if isinstance(response,dict) else None
                    if not isinstance(response,dict) or set(response) != {'rows','timing'}:
                        raw['malformed_profiled_response'] = response
                journal.write(batch_prefix+'.json', raw)
                if not isinstance(results,list) or len(results) != len(batch):
                    raise ValueError('Policy failed to return every action; no redraw')
                if 'malformed_profiled_response' in raw:
                    raise ValueError('Profiled policy response must contain exactly rows and timing')
                verify_batch_timing(raw, profiled)
                for (index, request), result in zip(batch, results, strict=True):
                    if any(canonical(result.get(k)) != canonical(v) for k,v in request.items()):
                        raise ValueError('Returned policy request provenance differs')
                    if result['checkpoint'] != selected_binding:
                        raise ValueError('Policy switched checkpoint within a cohort')
                    if result['action'] not in request['actions']:
                        raise ValueError('Policy returned an illegal action')
                    actions[index][request['player']] = result['action']
                    records[index]['transitions' if role=='learner' else 'opponent_transitions'].append(result)
                batch_number += 1
        for index, instance in enumerate(instances):
            if not instance.done:
                records[index]['events'].append(envs.record_step(instance, actions[index], menus[index]))
    completed, replay_reports = [], []
    for record, instance in zip(records, instances, strict=True):
        result = envs.finish_episode(record, instance)
        for row in result['transitions'] + result['opponent_transitions']:
            row['terminal_return'] = result['returns'][row['player']]
        replay_reports.append(verify_transitions(result))
        completed.append(result)
    episodes_sha = journal.write(prefix+'/episodes.json', completed)
    replay_sha = journal.write(prefix+'/replay-verification.json', replay_reports)
    seal = {'kind': 'mixed_cohort_seal', 'intent_sha256': intent_sha,
        'binding': binding, 'specs_sha256': digest(specs), 'episodes_sha256': episodes_sha,
        'replay_sha256': replay_sha, 'episodes': len(completed),
        'learner_actions': sum(len(e['transitions']) for e in completed),
        'raw_batches': batch_number, 'profiled': profiled, 'elapsed_seconds': time.monotonic()-started, 'sealed_at': now()}
    seal['draw_files'] = {str(path.relative_to(journal.path(prefix))): file_sha(path)
        for path in sorted(journal.path(prefix).rglob('*.json'))}
    journal.write(prefix+'/seal.json', seal)
    return seal


def load_sealed_cohort(folder):
    folder = Path(folder)
    seal = json.loads((folder/'seal.json').read_text())
    episodes = json.loads((folder/'episodes.json').read_text())
    for relative, expected in seal['draw_files'].items():
        path = folder/relative
        if not path.resolve().is_relative_to(folder.resolve()) or file_sha(path) != expected:
            raise ValueError('Raw draw or intent changed after sealing')
    intent = json.loads((folder/'intent.json').read_text())
    if file_sha(folder/'intent.json') != seal['intent_sha256'] or digest(intent['specs']) != seal['specs_sha256']:
        raise ValueError('Cohort intent differs')
    if intent['binding'] != seal['binding'] or len(intent['specs']) != len(episodes):
        raise ValueError('Cohort behavior or scheduled episode coverage differs')
    if type(intent.get('profiled')) is not bool or seal.get('profiled') is not intent['profiled']:
        raise ValueError('Cohort timing mode differs')
    if (file_sha(folder/'episodes.json') != seal['episodes_sha256']
            or file_sha(folder/'replay-verification.json') != seal['replay_sha256']
            or len(episodes) != seal['episodes']):
        raise ValueError('Sealed cohort files differ')
    specifications = {}
    for spec,episode in zip(intent['specs'],episodes,strict=True):
        if (canonical(episode['schedule']) != canonical(spec)
                or episode['episode_id'] != spec['episode_id']
                or episode['environment'] != spec['environment']
                or episode['seed'] != spec['seed'] or episode['regime'] != spec['regime']
                or episode['depth'] != spec['depth']
                or episode['learner_seats'] != spec['learner_seats']
                or any(type(p) is not int for p in episode['learner_seats'])
                or not set(episode['learner_seats']) <= set(envs.protocol_config()['roles'][episode['environment']])):
            raise ValueError('Episode role denominator or schedule differs from immutable intent')
        specifications[spec['episode_id']] = spec
        verify_transitions(episode)
        if episode['behavior_binding'] != seal['binding']:
            raise ValueError('Mixed behavior versions in sealed cohort')
    raw, final = {}, {}
    for index in range(seal['raw_batches']):
        name = f'batches/{index:06d}'
        batch_intent = json.loads((folder/(name+'-intent.json')).read_text())
        batch = json.loads((folder/(name+'.json')).read_text())
        if batch['intent_sha256'] != file_sha(folder/(name+'-intent.json')) or batch['role'] != batch_intent['role']:
            raise ValueError('Raw batch is not bound to its sampling intent')
        if batch_intent.get('profiled') is not intent['profiled']:
            raise ValueError('Raw batch timing mode differs')
        verify_batch_timing(batch, intent['profiled'])
        if len(batch['rows']) != len(batch_intent['requests']):
            raise ValueError('Raw batch request coverage differs')
        for request, row in zip(batch_intent['requests'], batch['rows'], strict=True):
            if any(canonical(row.get(k)) != canonical(v) for k,v in request.items()) or row['checkpoint'] != batch_intent['binding']:
                raise ValueError('Raw batch request or checkpoint differs')
            key = (row['episode_id'], row['turn'], row['player'])
            spec = specifications[row['episode_id']]
            role = 'learner' if row['player'] in spec['learner_seats'] else 'opponent'
            seed_key = spec.get('paired_seed_key',spec['episode_id'])
            if (batch['role'] != role
                    or row['sample_seed'] != stable_seed('action',seed_key,row['turn'],row['player'])
                    or row['menu_seed'] != stable_seed('menu',seed_key,row['turn'],row['player'])
                    or batch_intent['binding'] != intent['binding' if role=='learner' else 'opponent_binding']):
                raise ValueError('Raw action role, random seed or checkpoint differs from frozen spec')
            if key in raw:
                raise ValueError('Duplicate raw neural action')
            raw[key] = row
    for episode in episodes:
        for field in ('transitions', 'opponent_transitions'):
            for row in episode[field]:
                key = (row['episode_id'], row['turn'], row['player'])
                if key in final:
                    raise ValueError('Duplicate final neural action')
                final[key] = {k:v for k,v in row.items() if k != 'terminal_return'}
        if episode['schedule'].get('opponent') == 'base':
            opponent_episode = {**episode, 'learner_seats': [p for p in envs.protocol_config()['roles'][episode['environment']] if p not in episode['learner_seats']],
                'transitions': episode['opponent_transitions'], 'behavior_binding': intent['opponent_binding']}
            verify_transitions(opponent_episode)
    if set(raw) != set(final) or any(canonical(raw[k]) != canonical(final[k]) for k in raw):
        raise ValueError('Episode actions differ from original immutable neural draws')
    return seal, episodes


def pipeline_plan(count):
    """One outstanding cohort; immutable actor version never exceeds lag one."""
    if type(count) is not int or count < 1:
        raise ValueError('Positive fixed cohort count required')
    return [{'cohort': index, 'behavior_version': max(0, index-2),
             'learner_source_version': index-1, 'published_version': index,
             'policy_lag': 0 if index==1 else 1}
            for index in range(1, count+1)]


def run_pipeline(actor, learner, journal, schedules, count, base_checkpoint):
    """Drive futures/RPC clients; actor collection overlaps the prior update.

    Actor.submit(specs,checkpoint,prefix) returns a future with result().
    Learner.update(folder,checkpoint,index) commits exactly one cohort/checkpoint.
    Clients own subprocesses, deadlines and durable neural-operation intents.
    """
    plan = pipeline_plan(count)
    journal.intent('training-intent.json', 'mixed_training', {'plan': plan})
    published = {0: base_checkpoint}
    pending = actor.submit(schedules['train-1'], published[0], 'cohorts/1')
    updates = []
    for step in plan:
        index = step['cohort']
        seal = pending.result()
        if seal['binding']['version'] != step['behavior_version']:
            raise ValueError('Actor returned an unexpected behavior version')
        # Snapshot used for the next rollout is published before this update.
        if index < count:
            pending = actor.submit(schedules[f'train-{index+1}'], published[index-1], f'cohorts/{index+1}')
        outcome = learner.update(journal.path(f'cohorts/{index}'), published[index-1], index)
        checkpoint = outcome['checkpoint']
        if checkpoint['binding']['version'] != index:
            raise ValueError('Learner published an unexpected version')
        published[index] = checkpoint
        updates.append(outcome)
        journal.write(f'cohorts/{index}/pipeline-commit.json', outcome)
    result = {'attempted_cohorts': count, 'actual_optimizer_steps': sum(u['report']['optimizer_steps'] for u in updates),
              'endpoint': published[count], 'all_updates_committed': True}
    journal.write('training-complete.json', result)
    return result
