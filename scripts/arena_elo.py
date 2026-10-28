"""CPU-only, per-environment Elo from completed, replay-verified arena matches.

Ratings summarize this league's adjudicated outcomes. They are not chess Elo,
financial skill scores, confidence intervals, or proof of continual improvement.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import re
import sys


DEFAULT_FRAGMENT = Path('/Users/mitchellcatoen/.codex/visualizations/2026/10/20/'
                        '01a15b4f-05a3-7053-9034-4d1bc9f8d726/arena-rating-history.html')
INITIAL_RATING = 1500.0
K_FACTOR = 24.0


def canonical(value):
    return json.dumps(json.loads(json.dumps(value)), sort_keys=True, separators=(',', ':'), allow_nan=False)


def outcome_score(reward):
    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
        raise ValueError('Expected a finite numeric return')
    if not -1 <= reward <= 1:
        raise ValueError('Return outside [-1, 1]')
    return 1.0 if reward > 0 else 0.0 if reward < 0 else 0.5


def elo_update(first, second, score, k=K_FACTOR):
    if score not in (0.0, 0.5, 1.0) or not all(math.isfinite(v) for v in (first, second, k)) or k <= 0:
        raise ValueError('Invalid Elo update')
    expected = 1 / (1 + 10 ** ((second - first) / 400))
    delta = k * (score - expected)
    return first + delta, second - delta


def iteration_of(identity):
    if identity in {'base', 'heuristic'}:
        return 0
    match = re.fullmatch(r'cycle(\d+)', identity)
    if match:
        return int(match[1])
    raise ValueError(f'Unknown checkpoint identity {identity!r}; use cycleN or record iteration explicitly')


def match_iteration(row):
    if 'iteration' in row:
        value = row['iteration']
        if type(value) is not int or value < 0:
            raise ValueError('iteration must be a nonnegative integer')
        return value
    return max(iteration_of(identity) for identity in row['policies'])


def order_key(row):
    # Reproducible adjudication order, independent of batched completion time.
    opponent = row.get('opponent')
    phase = 0 if opponent == 'heuristic' else 1 if opponent == 'base' else 2
    return (match_iteration(row), phase, row.get('_source', ''), row['seed'],
            row.get('candidate_seat', -1), row['episode_id'])


def identical_policies(row, fingerprints=None):
    first, second = row['policies']
    fingerprints = fingerprints or {}
    return first == second or (first in fingerprints and second in fingerprints
                               and fingerprints[first] == fingerprints[second])


def policy_profile(manifest):
    """Arithmetic is part of the effective policy, not just adapter weights."""
    declaration = manifest.get('rating_compatibility')
    arithmetic = manifest.get('policy_arithmetic')
    recorded_hash = manifest.get('policy_arithmetic_sha256')
    if arithmetic is None and recorded_hash is None:
        if declaration not in (None, 'arena-v1'):
            raise ValueError('A nonlegacy rating pool requires policy_arithmetic and its hash')
        return {'rating_compatibility': 'arena-v1', 'policy_arithmetic': None,
                'policy_arithmetic_sha256': 'legacy-unspecified'}
    if not isinstance(declaration, str) or not declaration:
        raise ValueError('Explicit policy arithmetic requires rating_compatibility')
    if not isinstance(arithmetic, dict) or not arithmetic:
        raise ValueError('policy_arithmetic must be a nonempty object')
    actual_hash = hashlib.sha256(canonical(arithmetic).encode()).hexdigest()
    if recorded_hash != actual_hash:
        raise ValueError('policy_arithmetic_sha256 does not match canonical arithmetic profile')
    return {'rating_compatibility': declaration, 'policy_arithmetic': arithmetic,
            'policy_arithmetic_sha256': actual_hash}


def policy_fingerprint(metadata, profile, inherited=False):
    """Inherited bytes may be re-evaluated using a new run's declared arithmetic."""
    metadata_has_profile = any(key in metadata for key in ('policy_arithmetic', 'policy_arithmetic_sha256',
                                                          'rating_compatibility'))
    if metadata_has_profile:
        metadata_profile = policy_profile(metadata)
        if not inherited and metadata_profile != profile:
            raise ValueError('Checkpoint arithmetic differs from run manifest')
    elif not inherited and profile['policy_arithmetic'] is not None:
        raise ValueError('Local checkpoint lacks declared policy arithmetic metadata')
    identity = {key: metadata[key] for key in ('model', 'revision', 'adapter_sha256')}
    identity.update(rating_compatibility=profile['rating_compatibility'],
                    policy_arithmetic_sha256=profile['policy_arithmetic_sha256'])
    return canonical(identity)


def evaluation_specs(schedule):
    result = {}
    for stage, rows in schedule.items():
        # Continuation training IDs are league-train-*, so episode-name prefix
        # filtering alone incorrectly counts training as missing rated games.
        if stage.startswith('train-'):
            continue
        for row in rows:
            identity = row['episode_id']
            if identity in result:
                raise ValueError(f'Duplicate scheduled episode: {identity}')
            result[identity] = row
    return result


def build_ratings(rows, initial=INITIAL_RATING, k=K_FACTOR, fingerprints=None):
    ratings, histories = defaultdict(dict), defaultdict(lambda: defaultdict(list))
    counts, stages, adjudications = defaultdict(Counter), defaultdict(dict), defaultdict(Counter)
    match_log, excluded, seen = [], [], set()
    grouped = defaultdict(list)
    for row in rows:
        identity = row['episode_id']
        if identity in seen:
            raise ValueError(f'Duplicate episode: {identity}')
        seen.add(identity)
        if len(row['policies']) != 2 or len(row['returns']) != 2:
            raise ValueError('Expected exactly two players')
        for reward in row['returns']:
            outcome_score(reward)
        if abs(sum(row['returns'])) > 1e-9:
            raise ValueError('Returns must be zero-sum')
        if identical_policies(row, fingerprints):
            excluded.append({'episode_id': identity, 'environment': row['environment'],
                             'reason': 'same persistent policy or verified identical effective policy in both seats'})
            continue
        grouped[(match_iteration(row), row['environment'])].append(row)
    observed = sorted({iteration for iteration, _ in grouped})
    for iteration in observed:
        games_at_stage = sorted({game for stage, game in grouped if stage == iteration})
        # A game gets a stage point only when it has real new matches at that stage.
        for game in games_at_stage:
            new_counts = Counter()
            stage_reasons = Counter()
            for row in sorted(grouped[(iteration, game)], key=order_key):
                first, second = row['policies']
                for identity in (first, second):
                    ratings[game].setdefault(identity, float(initial))
                before = [ratings[game][first], ratings[game][second]]
                score = outcome_score(row['returns'][0])
                after = elo_update(*before, score, k=k)
                ratings[game][first], ratings[game][second] = after
                reason = row['metrics'].get('finish_reason') or 'settlement'
                adjudications[game][reason] += 1
                stage_reasons[reason] += 1
                for identity in (first, second):
                    counts[game][identity] += 1
                    new_counts[identity] += 1
                match_log.append({'episode_id': row['episode_id'], 'environment': game,
                                  'iteration': iteration, 'policies': row['policies'],
                                  'return_player0': row['returns'][0], 'score_player0': score,
                                  'termination': reason, 'rating_before': before,
                                  'rating_after': list(after), 'source': row.get('_source')})
            stage_matches = len(grouped[(iteration, game)])
            stages[game][iteration] = {'matches': stage_matches, 'termination_counts': dict(stage_reasons)}
            for identity, rating in ratings[game].items():
                histories[game][identity].append({'iteration': iteration, 'rating': rating,
                    'games': counts[game][identity], 'new_games': new_counts[identity]})
            if abs(sum(ratings[game].values()) - len(ratings[game]) * initial) > 1e-7:
                raise ValueError('Elo conservation failed')
    games = {}
    for game in sorted(ratings):
        games[game] = {'ratings': ratings[game], 'series': dict(histories[game]),
                       'stages': stages[game], 'rated_matches': sum(adjudications[game].values()),
                       'termination_counts': dict(adjudications[game]),
                       'same_policy_matches_excluded': sum(e['environment'] == game for e in excluded)}
    return {'initial_rating': initial, 'k_factor': k, 'observed_iterations': observed,
            'games': games, 'matches': match_log, 'excluded_matches': excluded,
            'score_rule': 'Positive terminal return = 1; exact zero = 0.5; negative = 0. '
                          'Distance/material caps count as adjudicated outcomes, not actual finishes/checkmates.',
            'match_order': 'iteration, heuristic/base/other opponent phase, source filename, seed, candidate seat, episode ID',
            'warnings': ['Exploratory league-relative ratings; fixed opponent matching and small samples.',
                         'Elo is order-dependent; ratings are not calibrated across environments.',
                         'Historical ratings can change without weight updates when new matches arrive.',
                         'A new checkpoint enters at 1500 only when its first rated match is completed.',
                         'Same-policy self-play cannot distinguish persistent policy strength and is excluded.',
                         'No unobserved training iterations or matches are extrapolated.']}


def archived_factory(run):
    manifest = json.loads((run / 'manifest.json').read_text())
    for original, expected in manifest['source_sha256'].items():
        archive = run / 'source' / Path(original).name
        if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen source mismatch: {archive}')
    for package in ('chess', 'python-chess'):
        frozen = manifest.get('versions', {}).get(package)
        if frozen and importlib.metadata.version(package) != frozen:
            raise ValueError(f'Install frozen {package}=={frozen} before replay')
    name = '_arena_elo_' + hashlib.sha256((run / 'manifest.json').read_bytes()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, run / 'source' / '__init__.py',
                                                 submodule_search_locations=[str(run / 'source')])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return package.make_env, manifest


def verify_match(row, factory):
    env = factory(row['environment'], row['seed'], row['regime'])
    if canonical(env.snapshot()) != canonical(row['initial_state']):
        raise ValueError(f'Initial state mismatch: {row["episode_id"]}')
    for event in row['events']:
        if env.done or env.turn != event['turn']:
            raise ValueError(f'Invalid event sequence: {row["episode_id"]}')
        env.step({int(player): action for player, action in event['actions'].items()})
        if canonical(env.snapshot()) != canonical(event['state']):
            raise ValueError(f'State mismatch: {row["episode_id"]} / {event["turn"]}')
    if not env.done or canonical(env.returns()) != canonical(row['returns']) or canonical(env.metrics()) != canonical(row['metrics']):
        raise ValueError(f'Terminal outcome mismatch: {row["episode_id"]}')


def analyze(runs):
    if isinstance(runs, (str, Path)):
        runs = [runs]
    runs = [Path(run).resolve() for run in runs]
    if len(set(runs)) != len(runs):
        raise ValueError('Duplicate run directory')
    rows, provenance, incomplete, compatibility, expected, seen_scheduled = [], {}, [], {}, {}, set()
    checkpoint_hashes, checkpoint_fingerprints = {}, {}
    reference_rules, reference_profile = None, None
    for run in runs:
        factory, manifest = archived_factory(run)
        profile = policy_profile(manifest)
        if reference_profile is None:
            reference_profile = profile
        elif profile != reference_profile:
            raise ValueError(f'Incompatible rating pool or policy arithmetic: {run}; analyze separate pools')
        checkpoint_entries = {path.parent.name: {'path': str(path.parent)}
                              for path in (run / 'checkpoints').glob('*/metadata.json')}
        for identity, entry in manifest.get('inherited_checkpoints', {}).items():
            if identity in checkpoint_entries:
                raise ValueError(f'Checkpoint declared as both inherited and local: {identity}')
            checkpoint_entries[identity] = entry
        for identity, entry in checkpoint_entries.items():
            directory = Path(entry['path'])
            metadata_bytes = (directory / 'metadata.json').read_bytes()
            metadata = json.loads(metadata_bytes)
            if metadata.get('checkpoint') != identity:
                raise ValueError(f'Checkpoint metadata identity mismatch: {identity}')
            if {key: metadata[key] for key in ('model', 'revision')} != manifest['model']:
                raise ValueError(f'Checkpoint base model differs from run manifest: {identity}')
            adapter_hash = hashlib.sha256((directory / 'adapters.safetensors').read_bytes()).hexdigest()
            if adapter_hash != metadata['adapter_sha256'] or adapter_hash != entry.get('adapter_sha256', adapter_hash):
                raise ValueError(f'Checkpoint adapter hash mismatch: {identity}')
            if entry.get('metadata_sha256') and hashlib.sha256(metadata_bytes).hexdigest() != entry['metadata_sha256']:
                raise ValueError(f'Inherited checkpoint metadata hash mismatch: {identity}')
            if identity in checkpoint_hashes and checkpoint_hashes[identity] != adapter_hash:
                raise ValueError(f'Persistent checkpoint identity has different weights: {identity}')
            checkpoint_hashes[identity] = adapter_hash
            fingerprint = policy_fingerprint(metadata, profile,
                                             inherited=identity in manifest.get('inherited_checkpoints', {}))
            if identity in checkpoint_fingerprints and checkpoint_fingerprints[identity] != fingerprint:
                raise ValueError(f'Persistent policy identity changed: {identity}')
            checkpoint_fingerprints[identity] = fingerprint
        rules = {'horizons': manifest['horizons'],
                 'versions': {name: manifest.get('versions', {}).get(name) for name in ('chess', 'python-chess')},
                 'sources': {
            Path(path).name: value for path, value in manifest['source_sha256'].items()
            if Path(path).name in {'__init__.py', 'market.py', 'maze.py', 'chess_game.py'}}}
        if reference_rules is None:
            reference_rules = rules
        elif rules != reference_rules:
            raise ValueError(f'Incompatible game source hashes or horizons: {run}')
        declaration = manifest.get('rating_compatibility')
        compatibility[run.name] = {'declared_rating_compatibility': declaration,
                                  'effective_rating_compatibility': profile['rating_compatibility'],
                                  'policy_arithmetic_sha256': profile['policy_arithmetic_sha256'],
                                  'game_rules_and_horizons_match': True,
                                  'verification': 'Exact factory/game-source hashes, horizons, rating-pool declaration, and verified arithmetic profile'}
        schedule_path = run / 'schedule.json'
        run_expected = {}
        if schedule_path.exists():
            run_expected = evaluation_specs(json.loads(schedule_path.read_text()))
            for identity, row in run_expected.items():
                if identity in expected:
                    raise ValueError(f'Duplicate scheduled identity across runs: {identity}')
                expected[identity] = row
        files = sorted(set(run.glob('eval-*.jsonl')) | set(run.glob('league-*.jsonl')) |
                       set(run.glob('matches-*.jsonl')))
        for path in files:
            content = path.read_bytes()
            lines = content.splitlines(keepends=True)
            provenance[str(path)] = hashlib.sha256(content).hexdigest()
            if lines and not lines[-1].endswith(b'\n'):
                lines.pop(); incomplete.append(str(path))
            for raw in lines:
                row = json.loads(raw)
                if run_expected:
                    scheduled = run_expected.get(row['episode_id'])
                    if scheduled is None or any(row.get(key) != value for key, value in scheduled.items()):
                        raise ValueError(f'Match differs from frozen schedule: {row["episode_id"]}')
                    seen_scheduled.add(row['episode_id'])
                verify_match(row, factory)
                if any(identity != 'heuristic' and identity not in checkpoint_hashes for identity in row['policies']):
                    raise ValueError(f'Match has an unverified checkpoint identity: {row["episode_id"]}')
                rows.append({**row, '_source': run.name + '/' + path.name})
    result = build_ratings(rows, fingerprints=checkpoint_fingerprints)
    coverage = {}
    for iteration in result['observed_iterations']:
        planned = [row for row in expected.values() if not identical_policies(row, checkpoint_fingerprints)
                   and match_iteration(row) == iteration]
        completed = [row for row in rows if not identical_policies(row, checkpoint_fingerprints)
                     and match_iteration(row) == iteration]
        coverage[iteration] = {'scheduled_rated_matches': len(planned),
                               'completed_rated_matches': len(completed),
                               'partial': bool(planned) and len(completed) < len(planned)}
    result.update(run='+'.join(run.name for run in runs), simulator_replay_verified=True,
                  completed_matches_verified=len(rows), game_rule_compatibility=compatibility,
                  checkpoint_identity_hashes=checkpoint_hashes,
                  checkpoint_effective_policy_sha256={identity: hashlib.sha256(value.encode()).hexdigest()
                                                     for identity, value in checkpoint_fingerprints.items()},
                  rating_pool=reference_profile,
                  stage_coverage=coverage,
                  compatible_game_rules=reference_rules, rollout_sha256=provenance,
                  ignored_incomplete_writes=incomplete,
                  scheduled_matches=len(expected), missing_scheduled_match_ids=sorted(set(expected) - seen_scheduled),
                  scope='Completed evaluation/league matches only; no training self-play or neural inference.')
    return result


def embed(result, fragment):
    # Per-match audit details stay in JSON; the visual needs stage aggregates only.
    compact = {key: result[key] for key in ('run', 'initial_rating', 'k_factor', 'observed_iterations',
                                           'games', 'warnings', 'simulator_replay_verified', 'stage_coverage',
                                           'rating_pool', 'completed_matches_verified')}
    encoded = json.dumps(compact, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    text, count = re.subn(r'(<script id="er-data" type="application/json">).*?(</script>)',
                         lambda match: match[1] + encoded + match[2], fragment.read_text(), flags=re.S)
    if count != 1 or len(text.encode()) > 1_000_000:
        raise ValueError('Expected one data block and fragment under 1 MB')
    fragment.write_text(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, action='append', help='Repeat to pool compatible recorded runs')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--fragment', type=Path, default=DEFAULT_FRAGMENT)
    args = ap.parse_args()
    runs = args.run or [Path('runs/arena-v1')]
    result = analyze(runs)
    destination = args.out or runs[0] / 'elo.json'
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(destination)
    embed(result, args.fragment)
    print(json.dumps({'verified_matches': result['completed_matches_verified'],
                      'rated_matches': len(result['matches']), 'iterations': result['observed_iterations'],
                      'output': str(destination.resolve()), 'fragment': str(args.fragment)}))


if __name__ == '__main__':
    main()
