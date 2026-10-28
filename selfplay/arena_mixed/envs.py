"""Longer-horizon arena contract, complete replay, and scenario isolation.

This module never loads a model. The three competitive simulators are reused
unchanged; planning and coding repair are separate verified tasks. Observations and legal menus are policy inputs; omniscient snapshots,
scenario fingerprints, and evaluator metrics are restricted to replay/audit.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from ..arena.market import MarketEnv
from ..arena.maze import MazeEnv

ENVIRONMENTS = ('market', 'maze', 'chess', 'planning', 'coding')
_CLASSES = {'market': MarketEnv, 'maze': MazeEnv}
_PROTOCOL = {
    'version': 1,
    'name': 'mixed-arena-long-horizon-v1',
    'environments': list(ENVIRONMENTS),
    'horizons': {'market': 16, 'maze': 32, 'chess': 96, 'planning': 12, 'coding': 12},
    'horizon_units': {'market': 'simultaneous rounds', 'maze': 'simultaneous rounds', 'chess': 'plies',
                      'planning': 'actions', 'coding': 'edits/tests/submission actions'},
    'roles': {'market': [0, 1], 'maze': [0, 1], 'chess': [0, 1], 'planning': [0], 'coding': [0]},
    'reward_contract': {'market': 'zero_sum', 'maze': 'zero_sum', 'chess': 'zero_sum',
                        'planning': 'single_player_score', 'coding': 'single_player_score'},
    'dependency_versions': {'python-chess': '1.999', 'chess': '1.11.2'},
    'max_legal_actions': 256,
    'simulators': 'Unmodified selfplay.arena market/maze/chess; separately versioned witnessed planning and bounded coding repair',
    'terminal_rewards': 'Competitive games retain original zero-sum returns and labeled caps; single-player tasks return [verified score,0]',
    'scenario_identity': 'Initial scenario excluding seed, regime label and horizon; chess canonicalized under color/rank mirror',
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
    """Relative archive keys needed to reproduce this factory and its games."""
    package = Path(__file__).resolve().parents[1]
    paths = ['arena_mixed/__init__.py', 'arena_mixed/envs.py', 'arena_mixed/planning.py',
             'arena_mixed/coding.py', 'environment.py', 'arena/__init__.py',
             'arena/market.py', 'arena/maze.py', 'arena/chess_game.py']
    return {'selfplay/' + p: hashlib.sha256((package / p).read_bytes()).hexdigest() for p in paths}


def make_env(name, seed, regime='train', *, protocol=None):
    cfg = _protocol(protocol)
    _require(name in ENVIRONMENTS, 'Unknown arena environment')
    _require(type(seed) is int and seed >= 0, 'Scenario seed must be a nonnegative integer')
    _require(regime in ({'train', 'eval', 'ood', 'standard'} if name == 'chess'
                        else {'train', 'eval', 'ood'}), 'Unknown environment regime')
    if name == 'chess':
        from ..arena.chess_game import ChessEnv
        cls = ChessEnv
    elif name == 'planning':
        from .planning import PlanningEnv
        cls = PlanningEnv
    elif name == 'coding':
        from .coding import CodingEnv
        cls = CodingEnv
    else:
        cls = _CLASSES[name]
    env = cls(seed, regime=regime, max_rounds=cfg['horizons'][name])
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
    """Counterfactual check of inaccessible market state, without advancing play.

    Current public quotes remain fixed. The hidden settlement value, opponent
    signal, event ordering/noise, and future event schedule cannot affect either
    player's current observation or legal menu. No probe state is returned.
    """
    if env.name in ('planning', 'coding') and not env.done:
        return env.assert_private_boundary()
    if env.name != 'market' or env.done:
        return {'checked': False, 'reason': 'Full-information game or terminal state'}
    for player in env.active_players():
        before = policy_view(env, player)
        other = deepcopy(env)
        other._terminal_value += 10007
        other._signals[1 - player] -= 10009
        event = other._schedule[other.turn]
        event['priority'] = list(reversed(event['priority']))
        event['noise_side'] = 'sell' if event['noise_side'] == 'buy' else 'buy'
        event['noise_quantity'] += 17
        event['noise_offset'] += 19
        for future in other._schedule[other.turn + 1:]:
            future['mid'] += 10011
            future['priority'] = list(reversed(future['priority']))
        _json_equal(policy_view(other, player), before, 'Private market state leaked to current policy view')
    return {'checked': True, 'players': 2}


def scenario_description(environment, initial_state):
    """Outcome-independent identity; never pass this omniscient object to policy."""
    _require(environment in ENVIRONMENTS, 'Unknown scenario environment')
    state = deepcopy(initial_state)
    if environment in ('planning', 'coding'):
        _require(state['turn'] == 0, 'Scenario fingerprint requires initial task state')
        return {'environment': environment, 'task': state['scenario_identity']}
    if environment == 'maze':
        _require(state['round'] == 0, 'Scenario fingerprint requires initial maze state')
        return {'environment': environment, 'size': state['size'], 'start': state['start'],
                'goal': state['goal'], 'terrain': sorted(state['terrain']),
                'positions': state['positions'], 'walls': [sorted(w) for w in state['walls']],
                'blocks_remaining': state['blocks_remaining']}
    if environment == 'chess':
        import chess
        _require(state['ply'] == 0, 'Scenario fingerprint requires initial chess state')
        board = chess.Board(state['initial_fen'])
        _require(board.is_valid(), 'Invalid initial chess position')
        return {'environment': environment, 'mirror_canonical_fen': min(board.fen(), board.mirror().fen())}
    _require(state['turn'] == 0, 'Scenario fingerprint requires initial market state')
    # Preserve private/exogenous randomness, including both signals and priority.
    # Their omission would conflate materially different market scenarios.
    return {'environment': environment, **{key: state[key] for key in
            ('terminal_value', 'private_signals', 'accounts', 'orders', 'schedule')}}


def scenario_fingerprint(environment, seed=None, regime='train', *, initial_state=None):
    if initial_state is None:
        initial_state = make_env(environment, seed, regime).snapshot()
    return digest(scenario_description(environment, initial_state))


def episode_header(env, episode_id, **metadata):
    _require(env.turn == 0 and not env.done, 'Episode must begin at a nonterminal reset')
    _require(isinstance(episode_id, str) and episode_id, 'Episode ID is required')
    reserved = {'format_version', 'episode_id', 'environment', 'seed', 'regime', 'environment_protocol',
                'environment_protocol_sha256', 'scenario_sha256', 'scenario_family_id', 'role_count',
                'initial_state', 'events', 'metrics', 'returns'}
    _require(not set(metadata) & reserved, 'Episode metadata overrides required provenance')
    state = deepcopy(env.snapshot())
    fingerprint = scenario_fingerprint(env.name, initial_state=state)
    return {**deepcopy(metadata), 'format_version': 1, 'episode_id': episode_id, 'environment': env.name,
            'seed': env.seed, 'regime': env.regime, 'environment_protocol': protocol_config(),
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
    env = make_env(record['environment'], record['seed'], record['regime'])
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


def select_scenarios(environment, count, seed_start, *, regime='train', excluded=(), maximum_attempts=100000):
    """Choose the first unused initial layouts, before models/outcomes exist.

    Exclusions are scenario hashes, not merely integer seeds. Chess increments
    by two because adjacent seeds intentionally represent the same mirrored
    position family. The returned rejection log makes every substitution clear.
    """
    _require(type(count) is int and count > 0 and type(seed_start) is int and seed_start >= 0,
             'Invalid scenario allocation request')
    _require(type(maximum_attempts) is int and maximum_attempts >= count, 'Invalid allocation bound')
    seen = set(excluded)
    _require(all(isinstance(x, str) and len(x) == 64 and set(x) <= set('0123456789abcdef') for x in seen),
             'Invalid excluded fingerprint')
    selected, rejected = [], []
    stride = 2 if environment == 'chess' else 1
    for offset in range(maximum_attempts):
        seed = seed_start + offset * stride
        env = make_env(environment, seed, regime)
        snapshot = env.snapshot()
        fingerprint = scenario_fingerprint(environment, initial_state=snapshot)
        if fingerprint in seen:
            rejected.append({'seed': seed, 'scenario_sha256': fingerprint, 'reason': 'duplicate_initial_scenario'})
            continue
        seen.add(fingerprint)
        selected.append({'environment': environment, 'seed': seed, 'regime': regime,
                         'scenario_sha256': fingerprint, 'initial_state': snapshot})
        if len(selected) == count:
            return {'selected': selected, 'rejected': rejected,
                    'selection': 'first distinct initial scenarios; no gameplay or model-dependent selection'}
    raise ValueError('Cannot allocate the required distinct scenario count within the fixed bound')


def validate_schedule(schedule, *, prior_fingerprints=()):
    """Check actual train/calibration/evaluation separation and paired repeats.

    Each spec declares split and scenario_id. Intentional seat/opponent repeats
    share that scenario_id within one split. Reusing a layout under a different
    scenario_id or split is rejected, including mirrored chess aliases.
    """
    _require(isinstance(schedule, dict) and schedule, 'Schedule must contain named stages')
    historical = set(prior_fingerprints)
    identities, scenarios, aliases, counts = set(), {}, {}, {}
    for stage, specs in schedule.items():
        _require(isinstance(stage, str) and isinstance(specs, list) and specs, 'Invalid schedule stage')
        for spec in specs:
            episode = spec['episode_id']
            _require(isinstance(episode, str) and episode and episode not in identities, 'Duplicate episode identity')
            identities.add(episode)
            split = spec['split']
            scenario_id = spec.get('scenario_family_id', spec.get('scenario_id'))
            if 'scenario_family_id' in spec and 'scenario_id' in spec:
                _require(spec['scenario_family_id'] == spec['scenario_id'], 'Conflicting scenario family names')
            _require(split in {'train', 'calibration', 'validation', 'eval', 'holdout'}
                     and isinstance(scenario_id, str) and scenario_id, 'Missing split/scenario family')
            env = make_env(spec['environment'], spec['seed'], spec.get('regime', 'train'))
            fingerprint = scenario_fingerprint(env.name, initial_state=env.snapshot())
            if 'scenario_sha256' in spec:
                _require(spec['scenario_sha256'] == fingerprint, 'Scheduled scenario fingerprint differs')
            _require(fingerprint not in historical, 'Scenario duplicates historical evidence')
            key = (env.name, fingerprint)
            family = (env.name, split, scenario_id)
            _require(key not in scenarios or scenarios[key] == family, 'Scenario aliases another split/family')
            _require(family not in aliases or aliases[family] == fingerprint, 'Scenario family names different layouts')
            scenarios[key], aliases[family] = family, fingerprint
            counts[split] = counts.get(split, 0) + 1
    return {'verified': True, 'episodes': len(identities), 'distinct_scenarios': len(scenarios),
            'episodes_by_split': counts, 'scenario_sha256': sorted(fp for _, fp in scenarios),
            'schedule_sha256': digest(schedule), 'environment_protocol_sha256': digest(_PROTOCOL)}
