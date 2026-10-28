"""Explicit planning depth with unchanged public rules, reward and horizon.

Only initial task generation changes. Depth, witnesses and shortest paths remain
outside observations. Family hashes are identical to historical mixed planning.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from ..arena_mixed.planning import PlanningEnv as HistoricalPlanningEnv
from ..environment import Task, generate_tasks


class PlanningEnv(HistoricalPlanningEnv):
    def __init__(self, seed, regime='train', max_rounds=12, *, depth):
        if type(seed) is not int or seed < 0 or regime not in ('train','eval'):
            raise ValueError('Nonnegative integer seed and original modulus11 regime required')
        if type(depth) is not int or not 1 <= depth <= 6 or type(max_rounds) is not int or max_rounds != 12:
            raise ValueError('Explicit integer depth1..6 and unchanged horizon12 required')
        self.seed, self.regime, self.max_rounds, self.depth = seed, regime, max_rounds, depth
        original = generate_tasks(1, seed=seed, depths=(depth,), modulus=11)[0]
        self._task = Task(original.modulus, original.start, original.goal, original.actions, max_rounds, original.witness)
        if len(self._task.shortest_plan()) != depth:
            raise ValueError('Generated task exact shortest depth differs')
        self._initial_shortest_depth = depth
        self.state, self.turn, self.done = self._task.start, 0, False
        self.history, self.finish_reason = [], None


ENVIRONMENTS = ('planning',)
_PROTOCOL = {
    'version':1, 'name':'planning-explicit-depth-curriculum-v1',
    'environments':['planning'], 'horizons':{'planning':12}, 'roles':{'planning':[0]},
    'reward_contract':{'planning':'single_player_score'}, 'max_legal_actions':256,
    'depths':[1,2,3,4,5,6], 'modulus':11,
    'scenario_identity':'Historical planning canonical task with x/y symmetry; witness, depth label, seed and budget excluded',
    'reward':'Unchanged verified goal1, otherwise0 at12 actions; inactive role return0',
    'policy_boundary':'Public observation/rules/menu unchanged; no depth, witness or solver result in requests',
}
def canonical(value):
    # Simulator dictionaries use integer player/account keys; disk JSON uses
    # strings. Normalize keys before sorting so negative account IDs round-trip.
    value = json.loads(json.dumps(value, allow_nan=False))
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json_equal(left, right, message):
    # JSON comparison also distinguishes bools from integers (Python == does not).
    _require(canonical(left) == canonical(right), message)


def protocol_config():
    return deepcopy(_PROTOCOL)


def _protocol(protocol):
    if protocol is not None:
        _json_equal(protocol, _PROTOCOL, 'Unregistered environment protocol')
    return _PROTOCOL


def source_hashes():
    root=Path(__file__).resolve().parents[2]
    names=('selfplay/__init__.py','selfplay/environment.py','selfplay/arena_mixed/__init__.py',
           'selfplay/arena_mixed/planning.py','selfplay/planning_curriculum/__init__.py',
           'selfplay/planning_curriculum/envs.py')
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}


def make_env(name, seed, regime='train', *, depth, protocol=None):
    _protocol(protocol)
    _require(name == 'planning', 'Planning-only experiment')
    env=PlanningEnv(seed,regime=regime,depth=depth)
    validate_env(env)
    return env


def policy_view(env, player):
    _require(type(player) is int and player in (0, 1), 'Invalid player')
    _require(player in env.active_players(), 'Player is not active')
    observation = deepcopy(env.observation(player))
    actions = deepcopy(env.legal_actions(player))
    _require(isinstance(observation, dict) and 'rules' in observation,
             'Observation must document the game rules')
    _require(isinstance(actions, list) and 0 < len(actions) <= _PROTOCOL['max_legal_actions']
             and all(isinstance(action, dict) for action in actions), 'Invalid legal-action menu')
    encoded = [canonical(a) for a in actions]
    _require(len(encoded) == len(set(encoded)), 'Duplicate legal actions')
    canonical(observation)
    return {'observation': observation, 'legal_actions': actions}


def validate_env(env):
    _require(env.name in ENVIRONMENTS and type(env.done) is bool and type(env.turn) is int
             and 0 <= env.turn <= env.max_rounds, 'Invalid simulator lifecycle state')
    _require(env.max_rounds == _PROTOCOL['horizons'][env.name], 'Simulator horizon differs')
    active = env.active_players()
    _require(isinstance(active, list) and len(active) == len(set(active))
             and all(type(p) is int and p in (0, 1) for p in active), 'Invalid active-player list')
    if env.done:
        _require(active == [], 'Terminal simulator still has active players')
        result = env.returns()
        _require(isinstance(result, list) and len(result) == 2
                 and all(type(x) in (float, int) and math.isfinite(x) and -1 <= x <= 1 for x in result),
                 'Terminal returns are not bounded finite scores')
        if _PROTOCOL['reward_contract'][env.name] == 'zero_sum':
            _require(abs(math.fsum(result)) <= 1e-12, 'Competitive returns are not zero-sum')
        else:
            _require(0 <= result[0] <= 1 and result[1] == 0, 'Invalid single-player score contract')
    else:
        expected = _PROTOCOL['roles'][env.name]
        _require((len(active) == 1 and active[0] in expected) if env.name == 'chess' else active == expected,
                 'Active-player turn semantics differ')
        for player in active:
            policy_view(env, player)
    canonical(env.snapshot()); canonical(env.metrics())
    return {'environment': env.name, 'turn': env.turn, 'done': env.done, 'active_players': active,
            'role_count': len(_PROTOCOL['roles'][env.name])}


def assert_private_boundary(env):
    return env.assert_private_boundary()


def scenario_description(environment, initial_state):
    _require(environment == 'planning' and initial_state['turn'] == 0, 'Initial planning state required')
    return {'environment':'planning','task':deepcopy(initial_state['scenario_identity'])}


def scenario_fingerprint(environment='planning', seed=None, regime='train', *, depth=None, initial_state=None):
    if initial_state is None:
        initial_state=make_env(environment,seed,regime,depth=depth).snapshot()
    return digest(scenario_description(environment,initial_state))


def episode_header(env, episode_id, **metadata):
    _require(env.turn == 0 and not env.done, 'Episode must begin at a nonterminal reset')
    _require(isinstance(episode_id, str) and episode_id, 'Episode ID is required')
    reserved = {'format_version', 'episode_id', 'environment', 'seed', 'regime', 'environment_protocol',
                'environment_protocol_sha256', 'scenario_sha256', 'scenario_family_id', 'role_count',
                'initial_state', 'events', 'metrics', 'returns', 'depth'}
    _require(not set(metadata) & reserved, 'Episode metadata overrides required provenance')
    state = deepcopy(env.snapshot())
    fingerprint = scenario_fingerprint(env.name, initial_state=state)
    return {**deepcopy(metadata), 'format_version': 1, 'episode_id': episode_id, 'environment': env.name,
            'seed': env.seed, 'regime': env.regime, 'depth': env.depth, 'environment_protocol': protocol_config(),
            'environment_protocol_sha256': digest(_PROTOCOL),
            'scenario_sha256': fingerprint, 'scenario_family_id': env.name + ':' + fingerprint,
            'role_count': len(_PROTOCOL['roles'][env.name]),
            'initial_state': state, 'events': []}


def _players(mapping, expected):
    _require(isinstance(mapping, dict), 'Player mapping must be an object')
    result = {}
    for key, value in mapping.items():
        _require(type(key) is int and key in (0, 1) or type(key) is str and key in ('0', '1'),
                 'Invalid player mapping key')
        player = int(key)
        _require(player not in result, 'Aliased duplicate player key')
        result[player] = value
    _require(set(result) == set(expected), 'Player mapping lacks exactly the active players')
    return result


def _menu(menu, legal):
    _require(isinstance(menu, list), 'Recorded action menu is not a list')
    _require(sorted(map(canonical, menu)) == sorted(map(canonical, legal)),
             'Recorded action menu is not the complete legal-action permutation')


def record_step(env, actions, menus=None):
    """Validate the entire submitted batch before changing simulator state."""
    validate_env(env)
    _require(not env.done, 'Cannot step a terminal episode')
    players = env.active_players()
    actions = _players(actions, players)
    menus = _players(menus, players) if menus is not None else None
    views = {p: policy_view(env, p) for p in players}
    for player in players:
        legal = views[player]['legal_actions']
        if menus is not None:
            _menu(menus[player], legal)
            views[player]['legal_actions'] = deepcopy(menus[player])
        _require(canonical(actions[player]) in set(map(canonical, legal)), 'Action is not strictly legal')
    assert_private_boundary(env)
    event = {'turn': env.turn, 'active_players': list(players),
             'observations': {p: v['observation'] for p, v in views.items()},
             'legal_actions': {p: v['legal_actions'] for p, v in views.items()}, 'actions': deepcopy(actions)}
    env.step(deepcopy(actions))
    validate_env(env)
    event['state'] = deepcopy(env.snapshot())
    return event


def finish_episode(record, env):
    _require(env.done and record['environment'] == env.name and len(record['events']) == env.turn,
             'Cannot finish incomplete or mismatched episode')
    result = deepcopy(record)
    result.update(metrics=deepcopy(env.metrics()), returns=env.returns())
    return result


def verify_episode(record):
    """Reconstruct every saved decision view, full legal menu, action and state."""
    _require(record['format_version'] == 1, 'Unknown episode format')
    _protocol(record['environment_protocol'])
    _require(record['environment_protocol_sha256'] == digest(_PROTOCOL), 'Protocol hash differs')
    env = make_env(record['environment'], record['seed'], record['regime'], depth=record['depth'])
    _json_equal(record['initial_state'], env.snapshot(), 'Initial state differs')
    _require(record['scenario_sha256'] == scenario_fingerprint(env.name, initial_state=env.snapshot()),
             'Scenario identity differs')
    _require(record['scenario_family_id'] == env.name + ':' + record['scenario_sha256']
             and record['role_count'] == len(_PROTOCOL['roles'][env.name]), 'Scenario family/role denominator differs')
    _require(isinstance(record['events'], list) and 0 < len(record['events']) <= env.max_rounds,
             'Invalid episode event count')
    decisions = 0
    for saved in record['events']:
        _require(not env.done, 'Replay continues after a true terminal outcome')
        event = record_step(env, saved['actions'], saved['legal_actions'])
        _json_equal(saved, event, 'Replayed decision/action/state differs')
        decisions += len(event['active_players'])
    _require(env.done, 'Incomplete episode cannot have a terminal reward')
    _json_equal(record['metrics'], env.metrics(), 'Terminal metrics differ')
    _json_equal(record['returns'], env.returns(), 'Terminal returns differ')
    return {'verified': True, 'episode_id': record['episode_id'], 'environment': env.name,
            'turns': env.turn, 'decisions': decisions, 'scenario_sha256': record['scenario_sha256'],
            'finish_reason': env.metrics().get('finish_reason', 'settlement'), 'returns': env.returns()}


