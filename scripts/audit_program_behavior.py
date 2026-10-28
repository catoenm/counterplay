"""Read-only behavior/interaction audit plus paired idle-opponent counterfactuals.

No models are queried and no source programs or game rules are modified.
Counterfactual scores are diagnostics, not additional held-out evaluation trials.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selfplay.program_arena import compile_program, play, verify_replay
from selfplay.program_arena.examples import idle_program


def sign(number):
    return (number > 0) - (number < 0)


def player_metrics(record, player):
    end = record['events'][-1]['state']['robots'][player]
    result = {key: end[key] for key in ('score', 'banked_units', 'harvested_units', 'moves', 'rests', 'deposits', 'failed_actions')}
    result['final_unbanked_value'] = sum(end['cargo'])
    result['final_unbanked_units'] = len(end['cargo'])
    end_at_home = (end['x'], end['y']) == (end['home_x'], end['home_y'])
    result['final_unbanked_at_home_value'] = sum(end['cargo']) if end_at_home else 0
    result['unbanked_home_turns'] = 0
    result['full_home_rest_turns'] = 0
    result['partial_home_rest_turns'] = 0
    result['partial_home_rest_with_resources'] = 0
    result['partial_home_rest_empty_map'] = 0
    for event in record['events']:
        obs, action = event['observations'][player], event['decisions'][player]['action']
        home = obs['self_x'] == obs['home_x'] and obs['self_y'] == obs['home_y']
        if home and obs['carrying'] > 0 and action['action'] != 'deposit':
            result['unbanked_home_turns'] += 1
            if action['action'] == 'rest':
                result['full_home_rest_turns'] += obs['carrying'] == obs['capacity']
                result['partial_home_rest_turns'] += obs['carrying'] < obs['capacity']
                if obs['carrying'] < obs['capacity']:
                    key = 'partial_home_rest_with_resources' if any(s['units'] for s in obs['sites']) else 'partial_home_rest_empty_map'
                    result[key] += 1
    return result


def interactions(record):
    harvests = [Counter(), Counter()]
    cross_half = [0, 0]
    same_cell_rounds = simultaneous_harvests = last_unit_contests = 0
    for event in record['events']:
        observations, decisions = event['observations'], event['decisions']
        coords = [(o['self_x'], o['self_y']) for o in observations]
        outcomes = event['state']['history'][-1]['outcomes']
        same_cell_rounds += coords[0] == coords[1]
        both_harvest = all(d['action']['action'] == 'harvest' for d in decisions) and coords[0] == coords[1]
        if both_harvest:
            simultaneous_harvests += 1
            site = next((s for s in observations[0]['sites'] if (s['x'], s['y']) == coords[0]), None)
            last_unit_contests += site is not None and site['units'] == 1
        for player in (0, 1):
            if outcomes[str(player)] == 'harvested':
                harvests[player][coords[player]] += 1
                cross_half[player] += coords[player][0] > 3 if player == 0 else coords[player][0] < 3
    shared = set(harvests[0]) & set(harvests[1])
    return {'shared_harvest_sites': len(shared),
            'units_from_shared_sites': [sum(h[p] for p in shared) for h in harvests],
            'harvested_units': [sum(h.values()) for h in harvests],
            'cross_half_harvests': cross_half, 'same_cell_rounds': same_cell_rounds,
            'simultaneous_harvests': simultaneous_harvests, 'last_unit_contests': last_unit_contests,
            'site_harvests': [{f'{x},{y}': count for (x, y), count in h.items()} for h in harvests]}


def audit(matches_path):
    content = matches_path.read_bytes()
    wrappers = [json.loads(line) for line in content.splitlines() if line]
    cache, rows = {}, []
    idle = compile_program(idle_program())
    for wrapper in wrappers:
        record = wrapper['record']
        verify_replay(record)
        solo = []
        for player in (0, 1):
            program = record['programs'][player]
            key = (program['digest'], record['seed'], player, record['max_rounds'])
            if key not in cache:
                choices = [idle, idle]
                choices[player] = compile_program(program['source'])
                origins = ['scripted', 'scripted']
                origins[player] = record['origins'][player]
                counterfactual = play(choices, record['seed'], record['max_rounds'], origins=origins)
                verify_replay(counterfactual)
                cache[key] = counterfactual
            solo.append(cache[key])
        outcome = interactions(record)
        outcome.update(episode_id=wrapper['episode_id'], players=wrapper['players'], seed=record['seed'])
        outcome['player_metrics'] = [player_metrics(record, p) for p in (0, 1)]
        outcome['solo_metrics'] = [player_metrics(solo[p], p) for p in (0, 1)]
        outcome['score_changed_by_active_opponent'] = [record['metrics']['scores'][p] != solo[p]['metrics']['scores'][p] for p in (0, 1)]
        outcome['score_delta_active_minus_solo'] = [record['metrics']['scores'][p] - solo[p]['metrics']['scores'][p] for p in (0, 1)]
        outcome['action_turns_changed_by_active_opponent'] = [sum(e['decisions'][p]['action'] != s['decisions'][p]['action']
            for e, s in zip(record['events'], solo[p]['events'], strict=True)) for p in (0, 1)]
        actual_margin = record['metrics']['scores'][0] - record['metrics']['scores'][1]
        solo_margin = solo[0]['metrics']['scores'][0] - solo[1]['metrics']['scores'][1]
        outcome['outcome_sign_differs_from_independent_solo_ranking'] = sign(actual_margin) != sign(solo_margin)
        outcome['strict_winner_reversal_vs_independent_solo_ranking'] = actual_margin * solo_margin < 0
        outcome['solo_pair_scores'] = [solo[p]['metrics']['scores'][p] for p in (0, 1)]
        rows.append(outcome)

    by_player = defaultdict(list)
    for row in rows:
        for p, identity in enumerate(row['players']):
            by_player[identity].append((row, p))
    players = {}
    for identity, cases in by_player.items():
        keys = cases[0][0]['player_metrics'][cases[0][1]]
        players[identity] = {
            'games': len(cases),
            'actual_totals': {k: sum(r['player_metrics'][p][k] for r, p in cases) for k in keys},
            'solo_totals_same_cases': {k: sum(r['solo_metrics'][p][k] for r, p in cases) for k in keys},
            'games_score_changed_by_active_opponent': sum(r['score_changed_by_active_opponent'][p] for r, p in cases),
            'games_actions_changed_by_active_opponent': sum(r['action_turns_changed_by_active_opponent'][p] > 0 for r, p in cases),
            'action_turns_changed_by_active_opponent': sum(r['action_turns_changed_by_active_opponent'][p] for r, p in cases),
            'points_delta_active_minus_solo': sum(r['score_delta_active_minus_solo'][p] for r, p in cases),
            'cross_half_harvests': sum(r['cross_half_harvests'][p] for r, p in cases),
            'units_from_shared_sites': sum(r['units_from_shared_sites'][p] for r, p in cases),
            'games_with_final_unbanked_cargo': sum(r['player_metrics'][p]['final_unbanked_value'] > 0 for r, p in cases)}
        contexts = defaultdict(dict)
        for row, player in cases:
            contexts[(row['seed'], player)][row['players'][1-player]] = row['player_metrics'][player]['score']
        comparisons = [values for values in contexts.values() if len(values) > 1]
        players[identity]['matched_contexts_with_multiple_opponent_types'] = len(comparisons)
        players[identity]['contexts_score_changes_with_opponent_type'] = sum(len(set(v.values())) > 1 for v in comparisons)
    total = {'matches': len(rows), 'unique_scenario_seeds': len({r['seed'] for r in rows}),
             'games_with_shared_harvest_sites': sum(r['shared_harvest_sites'] > 0 for r in rows),
             'shared_site_occurrences': sum(r['shared_harvest_sites'] for r in rows),
             'harvested_units': sum(sum(r['harvested_units']) for r in rows),
             'units_from_shared_sites': sum(sum(r['units_from_shared_sites']) for r in rows),
             'cross_half_harvests': sum(sum(r['cross_half_harvests']) for r in rows),
             'same_cell_rounds': sum(r['same_cell_rounds'] for r in rows),
             'simultaneous_harvests': sum(r['simultaneous_harvests'] for r in rows),
             'last_unit_contests': sum(r['last_unit_contests'] for r in rows),
             'outcome_sign_changed_vs_independent_solo_ranking': sum(r['outcome_sign_differs_from_independent_solo_ranking'] for r in rows)}
    total['strict_winner_reversals_vs_independent_solo_ranking'] = sum(r['strict_winner_reversal_vs_independent_solo_ranking'] for r in rows)
    inputs = {}
    for wrapper in wrappers:
        for p, identity in enumerate(wrapper['players']):
            if identity in inputs:
                continue
            read_names = set()
            stack = [wrapper['record']['programs'][p]['source']]
            while stack:
                obj = stack.pop()
                if isinstance(obj, dict):
                    if set(obj) == {'var'}:
                        read_names.add(obj['var'])
                    stack.extend(obj.values())
                elif isinstance(obj, list):
                    stack.extend(obj)
            inputs[identity] = {'reads_opponent_sensors': any(n.startswith('opponent_') for n in read_names),
                                'reads_horizon_or_turn': bool({'turn', 'horizon'} & read_names)}
    return {'kind': 'post-hoc mechanistic audit, not a new held-out evaluation',
            'source_matches': str(matches_path), 'source_sha256': hashlib.sha256(content).hexdigest(),
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'all_original_and_counterfactual_replays_verified': True,
            'unique_counterfactual_matches': len(cache), 'new_model_calls': 0,
            'definitions': {'solo': 'same program/seat/seed/horizon against idle opponent; the map is unchanged',
              'shared_site': 'both players successfully harvest at least one unit from the same coordinate within the match',
              'cross_half': 'a successful harvest on the opponent home half (x>3 for player0, x<3 for player1)',
              'last_unit_contest': 'both robots choose harvest at the same coordinate with exactly one unit available',
              'outcome_sign_changed': 'actual banked-point margin sign differs from the margin of separate solo runs'},
            'totals': total, 'players': players, 'program_input_usage': inputs, 'episodes': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matches', type=Path, default=Path('runs/program-screen-v3/matches.jsonl'))
    parser.add_argument('--out', type=Path, default=Path('runs/program-behavior-v3.json'))
    args = parser.parse_args()
    result = audit(args.matches)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in ('totals', 'players', 'unique_counterfactual_matches')}, indent=2))


if __name__ == '__main__':
    main()
