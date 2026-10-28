"""CPU-only source-bound scenario inventory; no model loading or game play."""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from selfplay.arena_mixed import envs
from selfplay.arena_mixed.planning import task_family_payload
from selfplay.environment import Task
from selfplay.arena.chess_game import ChessEnv

ARENA_RUNS = ('arena-v1', 'arena-v2', 'arena-final-holdout-v1')


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(root=ROOT):
    root = Path(root)
    sources, fingerprints, rows, counts = {}, set(), [], Counter()
    def bind(path):
        relative = str(path.relative_to(root))
        sources[relative] = file_sha(path)
        return relative
    for name in ARENA_RUNS:
        folder = root/'runs'/name
        manifest_path, schedule_path = folder/'manifest.json', folder/'schedule.json'
        manifest = json.loads(manifest_path.read_text())
        schedule = json.loads(schedule_path.read_text())
        bind(manifest_path); bind(schedule_path)
        for game, module in (('market', 'market.py'), ('maze', 'maze.py'), ('chess', 'chess_game.py')):
            relative = 'selfplay/arena/' + module
            if file_sha(root/relative) != manifest['source_sha256'][relative]:
                raise ValueError('Historical simulator bytes changed: '+relative)
            bind(root/relative)
        actual = {}
        for path in sorted(folder.glob('*.jsonl')):
            episode_rows = []
            for number, line in enumerate(path.read_text().splitlines(), 1):
                value = json.loads(line)
                if isinstance(value, dict) and 'initial_state' in value and 'environment' in value:
                    episode_rows.append((number, value))
            if not episode_rows:
                continue
            relative = bind(path)
            for number, value in episode_rows:
                key = (value['environment'], value['seed'], value['regime'])
                actual.setdefault(key, []).append((value['initial_state'], relative, number))
                counts['actual_arena_episode_records'] += 1
        scheduled = {}
        for stage, specs in schedule.items():
            for spec in specs:
                key = (spec['environment'], spec['seed'], spec['regime'])
                scheduled.setdefault(key, []).append({'stage':stage, 'episode_id':spec['episode_id']})
        for key in sorted(set(scheduled) | set(actual)):
            game, seed, regime = key
            cls = ChessEnv if game == 'chess' else envs._CLASSES[game]
            old = cls(seed, regime=regime, max_rounds=manifest['horizons'][game])
            initial = old.snapshot()
            for observed, path, line in actual.get(key, []):
                if envs.canonical(observed) != envs.canonical(initial):
                    raise ValueError(f'Historical initial state differs: {path}:{line}')
            original = envs.scenario_fingerprint(game, initial_state=initial)
            projected = envs.scenario_fingerprint(game, seed, regime)
            fingerprints.update((original, projected))
            rows.append({'environment':game, 'run':name, 'seed':seed, 'regime':regime,
                'original_horizon':manifest['horizons'][game], 'scenario_sha256':original,
                'new_horizon_projection_sha256':projected,
                'scheduled_episodes':scheduled.get(key, []),
                'actual_records':[{'path':p, 'line':line} for _, p, line in actual.get(key, [])]})
            counts[game+'_run_scenarios'] += 1
    task_keys = set(Task.__dataclass_fields__)
    for path in sorted((root/'runs').glob('*/tasks.jsonl')):
        relative = bind(path)
        for line, raw in enumerate(path.read_text().splitlines(), 1):
            value = json.loads(raw)
            task = Task.from_dict({key:value[key] for key in task_keys})
            identity = {'environment':'planning', 'canonical_task':task_family_payload(task)}
            fingerprint = envs.scenario_fingerprint('planning', initial_state={
                'turn':0, 'scenario_identity':identity})
            fingerprints.add(fingerprint)
            rows.append({'environment':'planning', 'path':relative, 'line':line,
                         'scenario_sha256':fingerprint, 'canonical_task':identity['canonical_task']})
            counts['planning_task_records'] += 1
    for relative in ('selfplay/environment.py', 'selfplay/arena_mixed/envs.py',
                     'selfplay/arena_mixed/planning.py', 'scripts/inventory_arena_mixed_prior.py'):
        bind(root/relative)
    return {'kind':'mixed_arena_prior_scenario_inventory', 'format_version':1, 'verified':True,
        'method':{'arena_runs':list(ARENA_RUNS),
            'include_unplayed_scheduled_scenarios':True,
            'actual_initial_states_reconstructed_and_compared':True,
            'planning':'All top-level runs/*/tasks.jsonl; valid Task and x/y-symmetry canonicalization; ignore witness/budget.',
            'horizon_projection':'Include both original-state fingerprint and same seed/regime under new horizon. Projection is conservative seed exclusion, not a claim of equal dynamics: market public signal depends on horizon.',
            'coding':'No prior bounded-repair tasks. Previous free-form controller games are a different environment.',
            'no_outcomes_or_policy_calls_used':True},
        'source_sha256':sources, 'counts':dict(counts),
        'fingerprints':sorted(fingerprints), 'distinct_fingerprints':len(fingerprints),
        'rows':rows, 'environment_protocol_sha256':envs.digest(envs.protocol_config())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = inventory()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        f.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'path':str(args.output), 'sha256':file_sha(args.output),
                      'counts':value['counts'], 'distinct_fingerprints':value['distinct_fingerprints']}))


if __name__ == '__main__':
    main()
