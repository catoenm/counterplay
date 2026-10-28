"""Descriptive controller Elo from audited, committed warmstart RL games only.

No model calls, compilation, game execution, or checkpoint-level pooled rating.
The independent audit supplies replay verification; this reader binds every used
artifact to that exact audited status and rechecks those bytes before export.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import plot_cardinal_warmstart as report

require, sha = report.require, report.sha
INITIAL_RATING, K = 1500.0, 24.0


def value_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def elo_step(a, b, score):
    require(score in (0.0, 0.5, 1.0), 'Elo requires win, draw, or loss')
    expected = 1 / (1 + 10 ** ((b-a)/400))
    change = K*(score-expected)
    return a+change, b-change


class Reader:
    def __init__(self, audit_path):
        self.audit, self.manifest, self.status, self.schedule, self.bindings = report.bind_inputs(audit_path)
        self.bindings['warmstart_plot_helper'] = self.bindings.pop('reporter')
        self.bindings['reporter'] = {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)}
        self.run = Path(self.audit['run']).resolve()
        self.cache, self.nodes, self.history, self.draws, self.games = {}, {}, [], [], []

    def artifact(self, relative):
        part = Path(relative)
        require(not part.is_absolute() and '..' not in part.parts, 'Unsafe artifact path')
        require(relative in self.status['artifacts'], 'Artifact is not committed: '+relative)
        if relative not in self.cache:
            path = self.run/part
            require(path.resolve().is_relative_to(self.run), 'Artifact escapes the run')
            raw = path.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
            require(digest == self.status['artifacts'][relative], 'Committed artifact changed: '+relative)
            self.bindings['artifact/'+relative] = {'path': str(path), 'sha256': digest}
            self.cache[relative] = json.loads(raw)
        return self.cache[relative]

    def learner(self, name):
        match = re.fullmatch(r'(?:capability-(\d+)|train-cycle(\d+)-(\d+))', name)
        require(match is not None, 'Only learner capability/training generations are admitted')
        if match[1] is not None:
            index, label = int(match[1]), 'base'
            seeds = self.schedule['capability_generation_seeds']
        else:
            group, index = int(match[2]), int(match[3])
            label = 'base' if group == 1 else f'cycle{group-1}'
            seeds = self.schedule['train_generation_seeds'][str(group)]
        require(0 <= index < len(seeds), 'Generation index exceeds its schedule')
        relative = f'generations/{name}.json'; row = self.artifact(relative)
        require(row['name'] == name and row['seed'] == seeds[index] and row['checkpoint'] == label,
                'Learner generation schedule/checkpoint differs')
        require(row.get('authorship') == 'fresh_learner_sample' and row.get('api_training_data') is False,
                'Teacher/API output cannot be presented as a learner controller')
        require(row.get('prompt_condition') == 'cardinal', 'Generation prompt condition differs')
        cfg = self.manifest['config']
        require((row['model_repo'], row['model_revision']) == (cfg['model_repo'], cfg['model_revision']),
                'Learner model identity differs')
        metadata = f'checkpoints/{label}/metadata.json'; checkpoint = self.artifact(metadata)
        require(row['source_metadata_sha256'] == self.status['artifacts'][metadata], 'Generation checkpoint hash differs')
        step = self.audit['tensor_reports'][label]['actual_optimizer_step']
        require(type(step) is int and step >= 0, 'Actual source optimizer step missing')
        alias = {'identity': name, 'authorship': 'fresh_learner_sample', 'source_checkpoint': label,
            'source_RL_optimizer_step': step, 'source_checkpoint_description':
                'Teacher-SFT baseline; zero RL steps' if label == 'base' else f'{label}; {step} actual RL steps',
            'model_repo': row['model_repo'], 'model_revision': row['model_revision'],
            'generation_path': relative, 'generation_sha256': self.status['artifacts'][relative],
            'checkpoint_metadata_sha256': self.status['artifacts'][metadata],
            'adapter_sha256': checkpoint['files']['adapter.safetensors']}
        return row, alias

    def node(self, ir, alias, expected=None):
        digest = value_sha(ir)
        require(expected is None or digest == expected, 'Controller IR identity differs')
        key = 'ir:'+digest
        if key not in self.nodes:
            self.nodes[key] = {'controller': key, 'ir_sha256': digest, 'aliases': [],
                'rating': INITIAL_RATING, 'rated_games': 0, 'played_seat_exposures': 0,
                'self_IR_seat_exposures': 0, 'wins': 0, 'draws': 0, 'losses': 0,
                'program_faults': 0, 'faulted_seat_exposures': 0}
        if alias not in self.nodes[key]['aliases']: self.nodes[key]['aliases'].append(alias)
        return key

    def opponent(self, row):
        if row['origin'] == 'scripted':
            require(row['id'] in ('nearest', 'value'), 'Unexpected scripted anchor')
            alias = {'identity': row['id'], 'authorship': 'scripted_anchor', 'source_checkpoint': None}
        else:
            require(row['origin'] == 'model', 'Unknown controller origin')
            _, alias = self.learner(row['id'])
            require(row['source_path'] == alias['generation_path']
                and row['generation_sha256'] == alias['generation_sha256']
                and row['source_checkpoint'] == alias['source_checkpoint']
                and row['source_metadata_sha256'] == alias['checkpoint_metadata_sha256'],
                'Historical opponent provenance differs')
        return self.node(row['compiled_ir'], alias, row.get('ir_sha256'))

    def game(self, match, candidate, opponent, expected, group, name, index):
        rival, seed, seat = expected; replay = match['replay']
        require(match['opponent_id'] == rival['id'] and type(match['candidate_seat']) is int
            and match['candidate_seat'] == seat and replay['seed'] == seed and replay['max_rounds'] == 48,
            'Recorded match order/seat/map differs')
        keys = [opponent, opponent]; keys[seat] = candidate
        origins = [rival['origin'], rival['origin']]; origins[seat] = 'model'
        require(replay['origins'] == origins and len(replay['programs']) == 2, 'Replay origins differ')
        for p, key in enumerate(keys):
            entry = replay['programs'][p]
            require(value_sha(entry['source']) == entry['digest'] == self.nodes[key]['ir_sha256'],
                    'Replay IR differs from scored controller')
        points, faults = replay['metrics']['scores'], replay['metrics']['program_faults']
        require(len(points) == len(faults) == 2 and all(type(x) is int and x >= 0 for x in points+faults),
                'Invalid game scores/faults')
        score = 1.0 if points[0] > points[1] else 0.0 if points[0] < points[1] else 0.5
        identical = keys[0] == keys[1]
        for p, key in enumerate(keys):
            node = self.nodes[key]; own_score = score if p == 0 else 1-score
            node['played_seat_exposures'] += 1; node['program_faults'] += faults[p]
            node['faulted_seat_exposures'] += int(faults[p] > 0)
            node['wins' if own_score == 1 else 'losses' if own_score == 0 else 'draws'] += 1
            node['self_IR_seat_exposures'] += int(identical)
        event = {'played_game_index': len(self.games)+1, 'cycle': group, 'generation': name,
            'match_index': index, 'opponent_id': rival['id'], 'candidate_seat': seat, 'seed': seed,
            'controllers': keys, 'banked_points': points, 'program_faults': faults,
            'replay_sha256': value_sha(replay), 'rated': not identical}
        if not identical:
            before = [self.nodes[k]['rating'] for k in keys]
            after = elo_step(*before, score)
            for key, value in zip(keys, after):
                self.nodes[key]['rating'] = value; self.nodes[key]['rated_games'] += 1
            self.history.append({**event, 'rated_game_index': len(self.history)+1,
                'seat0_score': score, 'ratings_before': before, 'ratings_after': list(after)})
        self.games.append(event)
        return faults[seat], int(faults[seat] > 0)

    def build(self):
        audit = self.audit; groups = audit.get('training_groups', [])
        require([g['cycle'] for g in groups] == list(range(1, len(groups)+1)), 'Training group order differs')
        require(all(type(g.get('checkpoint_committed')) is bool for g in groups), 'Missing commit status')
        committed = [g for g in groups if g['checkpoint_committed']]
        require([g['cycle'] for g in committed] == list(range(1, len(committed)+1)), 'Committed prefix has a gap')
        if committed:
            require(audit.get('genuine_supervised_optimizer_steps') == 32 and audit.get('optimizer_reset_verified') is True,
                    'Learner RL games require completed SFT and verified optimizer reset')
        group_rows = []
        for summary in committed:
            n = summary['cycle']; source = 'base' if n == 1 else f'cycle{n-1}'
            folder = f'groups/{n}'; scores = self.artifact(folder+'/scores.json')
            opponents = self.artifact(folder+'/opponents.json')['selected']
            checkpoint = self.artifact(f'checkpoints/cycle{n}/metadata.json')
            require(checkpoint['report']['scores_sha256'] == self.status['artifacts'][folder+'/scores.json'],
                    'Checkpoint does not commit the scored group')
            require(scores['cycle'] == n and scores['source_checkpoint'] == source
                and scores['source_metadata_sha256'] == self.status['artifacts'][f'checkpoints/{source}/metadata.json']
                and scores['opponents_sha256'] == self.status['artifacts'][folder+'/opponents.json']
                and scores['seeds'] == self.schedule['train_maps'][str(n)], 'Group provenance differs')
            require(len(opponents) == 4 and len({o['id'] for o in opponents}) == 4, 'Expected four frozen opponents')
            keys = {o['id']: self.opponent(o) for o in opponents}
            checked = report.previous.checked_cell(summary, [f'train-cycle{n}-{i}' for i in range(8)], 16)
            require(len(scores['rows']) == 8, 'All eight prescribed draws must be retained')
            first = len(self.draws)
            for i, row in enumerate(scores['rows']):
                name = f'train-cycle{n}-{i}'; generation, alias = self.learner(name)
                require(row['generation'] == name and row['generation_sha256'] == alias['generation_sha256'],
                        'Scored generation differs')
                valid = row['classification']['valid']
                require(type(valid) is bool and valid == row['authorship_valid'], 'Authorship classification differs')
                require(row['runtime_valid'] == (valid and row['candidate_program_faults'] == 0), 'Runtime classification differs')
                audited = summary['programs'][i]
                for key in ('authorship_valid', 'runtime_valid', 'candidate_program_faults', 'banked_point_yield'):
                    require(row[key] == audited[key], 'Score and audited draw differ: '+key)
                expected = [(o, seed, seat) for o in opponents for seed in scores['seeds'] for seat in (0, 1)] if valid else []
                require(len(row['matches']) == len(expected), 'Invalid authorship has games or compiled draw is incomplete')
                candidate = None; faults = fault_games = 0
                if valid:
                    program = row['classification']['program']
                    candidate = self.node(program['compiled_ir'], alias, program['ir_sha256'])
                    for j, (match, planned) in enumerate(zip(row['matches'], expected)):
                        count, failed = self.game(match, candidate, keys[planned[0]['id']], planned, n, name, j)
                        faults += count; fault_games += failed
                require(faults == row['candidate_program_faults'], 'Candidate fault total differs')
                self.draws.append({'cycle': n, 'generation': name, 'generation_sha256': alias['generation_sha256'],
                    'source_checkpoint': source, 'source_RL_optimizer_step': alias['source_RL_optimizer_step'],
                    'controller': candidate, 'category': checked['programs'][i]['category'],
                    'played_games': len(expected), 'candidate_program_faults': faults,
                    'candidate_faulted_games': fault_games})
            own = self.draws[first:]
            group_rows.append({'cycle': n, **draw_counts(own), 'rated_games_after_group': len(self.history)})
        leaderboard = sorted(self.nodes.values(), key=lambda r: (-r['rating'], r['ir_sha256']))
        for node in leaderboard:
            node['aliases'].sort(key=lambda r: (r['authorship'], r['identity']))
            node['authorships'] = sorted({a['authorship'] for a in node['aliases']})
            node['ranked'] = node['rated_games'] > 0
            if not node['ranked']: node['rating'] = None
        report.previous.verify_bindings(self.bindings)
        return {'format_version': 1, 'kind': 'cardinal_warmstart_training_controller_elo',
            'state': 'rated_played_games' if self.history else 'no_ratable_committed_games',
            'test_fixture': audit.get('test_fixture') is True, 'bound_inputs': self.bindings,
            'source_audit_complete': audit.get('planned_study_complete') is True,
            'committed_training_groups': len(committed),
            'excluded_uncommitted_cycles': [g['cycle'] for g in groups if not g['checkpoint_committed']],
            'initial_rating': INITIAL_RATING, 'K': K, 'draw_score': 0.5,
            'rating_unit': 'Immutable compiled controller IR; exact duplicates share a rating and retain all source aliases.',
            'order': 'Committed cycle ascending, prescribed draw index ascending, recorded opponent/map/seat order.',
            'outcome': 'Final banked-points win/draw/loss; runtime-fault games retained, same-IR games excluded from Elo only.',
            'counts': {**draw_counts(self.draws), 'rated_games': len(self.history),
                'self_IR_excluded_games': len(self.games)-len(self.history),
                'distinct_replay_hashes': len({g['replay_sha256'] for g in self.games}),
                'distinct_controller_IRs': len(self.nodes)},
            'groups': group_rows, 'draws': self.draws, 'controllers': leaderboard,
            'rating_history': self.history, 'played_games': self.games,
            'caveats': ['This is controller-level descriptive training Elo, not checkpoint or average model ability.',
                'Maps and opponent exposure change; Elo is order-dependent and can hide nontransitive matchups.',
                'Only compiled controllers play. Invalid draws remain in denominators and receive no invented losses.',
                'Historical opponents are selected survivors; unequal exposure and correlated games prevent generalization claims.',
                'Authorship aliases retain scripted versus learner origins; teacher/API demonstrations are not rated.',
                'Replay validity comes from the bound independent audit; this reporter executes no game or model.',
                'Controllers with no rated games have a null rating, not an inferred tie at1500.',
                'Played exposure counts seats, including both seats of an excluded same-IR game.']}


def draw_counts(draws):
    n = len(draws)
    invalid = sum(r['category'] == 'invalid_authorship' for r in draws)
    faults = sum(r['category'] == 'runtime_fault' for r in draws)
    return {'prescribed_program_draws': n, 'invalid_authorship_draws': invalid, 'runtime_fault_draws': faults,
        'invalid_authorship_rate_all_draws': invalid/n if n else None,
        'runtime_fault_rate_all_draws': faults/n if n else None,
        'compiled_program_draws': n-invalid, 'played_games': sum(r['played_games'] for r in draws),
        'candidate_faulted_games': sum(r['candidate_faulted_games'] for r in draws)}


def export(audit_path, output):
    result = Reader(audit_path).build()
    require(not result['test_fixture'], 'Synthetic fixtures cannot be exported as experiment ratings')
    data = (json.dumps(result, indent=2, allow_nan=False)+'\n').encode()
    report.previous.verify_bindings(result['bound_inputs'])
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream: stream.write(data)
    return {'output': str(output.resolve()), 'sha256': sha(output), 'counts': result['counts']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(export(args.audit, args.out), indent=2))


if __name__ == '__main__': main()
