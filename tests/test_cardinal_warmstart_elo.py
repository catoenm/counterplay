"""Explicit synthetic reporting fixtures; no inference or games are executed."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts import cardinal_warmstart_elo as elo


def fixture(folder, empty=False, all_invalid=False):
    """Build a hash-bound reporting fixture, never a claimed independent audit."""
    run = Path(folder); artifacts = {}
    def put(relative, value):
        path = run/relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value)); artifacts[relative] = elo.sha(path)
        return artifacts[relative]
    schedule = {'capability_generation_seeds': list(range(8)),
        'train_generation_seeds': {'1': list(range(100, 108))}, 'train_maps': {'1': [200, 201]}}
    (run/'schedule.json').write_text(json.dumps(schedule))
    cfg = {'model_repo': 'explicit/reporting-fixture', 'model_revision': 'fixture-not-model'}
    manifest = {'kind': 'cardinal_teacher_warmstart_league_rl', 'source_sha256': {},
        'schedule_sha256': elo.sha(run/'schedule.json'), 'config': cfg}
    (run/'manifest.json').write_text(json.dumps(manifest))
    base_sha = put('checkpoints/base/metadata.json', {'files': {'adapter.safetensors': 'b'*64}})
    def generation(name, seed):
        return put(f'generations/{name}.json', {'name': name, 'seed': seed, 'checkpoint': 'base',
            **cfg, 'source_metadata_sha256': base_sha, 'authorship': 'fresh_learner_sample',
            'api_training_data': False, 'prompt_condition': 'cardinal'})
    irs = [{'explicit_fixture_IR': i} for i in range(4)]
    opponents = [{'id': name, 'origin': 'scripted', 'compiled_ir': irs[i]}
                 for i, name in enumerate(('nearest', 'value'))]
    for i in range(2):
        name = f'capability-{i}'; digest = generation(name, i)
        opponents.append({'id': name, 'origin': 'model', 'compiled_ir': irs[i+2],
            'ir_sha256': elo.value_sha(irs[i+2]), 'source_path': f'generations/{name}.json',
            'generation_sha256': digest, 'source_checkpoint': 'base', 'source_metadata_sha256': base_sha})
    opponents_sha = put('groups/1/opponents.json', {'selected': opponents})
    rows, programs = [], []
    for i in range(8):
        name = f'train-cycle1-{i}'; digest = generation(name, 100+i)
        valid = i < 2 and not all_invalid; matches = []
        if valid:
            for opponent in opponents:
                for seed in schedule['train_maps']['1']:
                    for seat in (0, 1):
                        sources = [opponent['compiled_ir']]*2; sources[seat] = irs[0]
                        points = [1, 1]; points[seat] = 2
                        faults = [0, 0]
                        if i == 1 and len(matches) == 4: faults[seat] = 3
                        origins = [opponent['origin']]*2; origins[seat] = 'model'
                        matches.append({'opponent_id': opponent['id'], 'candidate_seat': seat,
                            'replay': {'seed': seed, 'max_rounds': 48, 'origins': origins,
                                'programs': [{'source': p, 'digest': elo.value_sha(p)} for p in sources],
                                'metrics': {'scores': points, 'program_faults': faults}}})
        row = {'generation': name, 'generation_sha256': digest,
            'classification': {'valid': valid}, 'matches': matches, 'authorship_valid': valid,
            'runtime_valid': valid and i == 0, 'candidate_program_faults': 3 if valid and i == 1 else 0,
            'banked_point_yield': 2.0 if valid else 0.0}
        if valid: row['classification']['program'] = {'compiled_ir': irs[0], 'ir_sha256': elo.value_sha(irs[0])}
        rows.append(row)
        programs.append({key: row[key] for key in ('generation', 'authorship_valid', 'runtime_valid',
                                                   'candidate_program_faults', 'banked_point_yield')} | {'games': len(matches)})
    score_sha = put('groups/1/scores.json', {'cycle': 1, 'source_checkpoint': 'base',
        'source_metadata_sha256': base_sha, 'opponents_sha256': opponents_sha,
        'seeds': schedule['train_maps']['1'], 'rows': rows})
    put('checkpoints/cycle1/metadata.json', {'report': {'scores_sha256': score_sha}})
    group = {'cycle': 1, 'checkpoint_committed': True, 'prescribed_program_clusters': 8,
        'observed_program_clusters': 8, 'missing_program_clusters': 0, 'complete': True,
        'primary_mean_banked_point_yield': 0 if all_invalid else .5,
        'compiled_programs': 0 if all_invalid else 2, 'invalid_no_game_programs': 8 if all_invalid else 6,
        'fault_free_programs': 0 if all_invalid else 1, 'played_games': 0 if all_invalid else 32, 'programs': programs}
    status = {'manifest_sha256': elo.sha(run/'manifest.json'), 'artifacts': artifacts, 'state': 'fixture'}
    (run/'status.json').write_text(json.dumps(status))
    audit = {'kind': 'independent_cpu_cardinal_teacher_warmstart_audit', 'integrity_passed': True,
        'test_fixture': True, 'run': str(run), 'manifest_sha256': elo.sha(run/'manifest.json'),
        'status_sha256': elo.sha(run/'status.json'), 'training_groups': [] if empty else [group],
        'genuine_supervised_optimizer_steps': 32, 'optimizer_reset_verified': True,
        'tensor_reports': {'base': {'actual_optimizer_step': 0}, 'cycle1': {'actual_optimizer_step': 1}},
        'planned_study_complete': False}
    path = run/'audit.json'; path.write_text(json.dumps(audit)); return path


def rewrite_bound(folder, relative, mutate):
    """Fixture-only re-signing lets structural checks be tested independently."""
    run = Path(folder); path = run/relative; value = json.loads(path.read_text())
    mutate(value); path.write_text(json.dumps(value))
    status_path = run/'status.json'; status = json.loads(status_path.read_text())
    status['artifacts'][relative] = elo.sha(path); status_path.write_text(json.dumps(status))
    audit_path = run/'audit.json'; audit = json.loads(audit_path.read_text())
    audit['status_sha256'] = elo.sha(status_path); audit_path.write_text(json.dumps(audit))


class ControllerEloTests(unittest.TestCase):
    def test_known_win_draw_math_and_conservation(self):
        self.assertEqual(elo.elo_step(1500, 1500, 1), (1512, 1488))
        self.assertEqual(elo.elo_step(1500, 1500, .5), (1500, 1500))
        a, b = elo.elo_step(1600, 1400, 0)
        self.assertAlmostEqual(a+b, 3000); self.assertLess(a, 1600)

    def test_all_draw_denominators_fault_games_and_same_IR_aliases(self):
        with tempfile.TemporaryDirectory() as d:
            value = elo.Reader(fixture(d)).build()
            self.assertEqual(value['counts']['prescribed_program_draws'], 8)
            self.assertEqual(value['counts']['invalid_authorship_rate_all_draws'], .75)
            self.assertEqual(value['counts']['runtime_fault_rate_all_draws'], .125)
            self.assertEqual(value['counts']['played_games'], 32)
            self.assertEqual(value['counts']['rated_games'], 24)
            self.assertEqual(value['counts']['self_IR_excluded_games'], 8)
            self.assertEqual(sum(r['rated_games'] for r in value['controllers']), 48)
            self.assertEqual(sum(r['played_seat_exposures'] for r in value['controllers']), 64)
            node = next(r for r in value['controllers'] if len(r['aliases']) == 3)
            self.assertEqual(node['authorships'], ['fresh_learner_sample', 'scripted_anchor'])
            self.assertEqual(node['program_faults'], 3)
            self.assertEqual(node['self_IR_seat_exposures'], 16)
            self.assertTrue(all(a['source_checkpoint'] in (None, 'base') for a in node['aliases']))
            first = value['rating_history'][0]
            self.assertEqual(first['ratings_before'], [1500, 1500])
            self.assertEqual(first['ratings_after'], [1512, 1488])
            self.assertTrue(any(sum(g['program_faults']) for g in value['rating_history']))
            self.assertAlmostEqual(sum(r['rating'] for r in value['controllers']), 4*1500)

    def test_deterministic_order_and_exact_audit_binding(self):
        with tempfile.TemporaryDirectory() as d:
            path = fixture(d); first = elo.Reader(path).build(); second = elo.Reader(path).build()
            self.assertEqual(first, second)
            Path(d, 'groups/1/scores.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Committed artifact changed'): elo.Reader(path).build()
        with tempfile.TemporaryDirectory() as d:
            path = fixture(d); Path(d, 'status.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'status identity'): elo.Reader(path)

    def test_teacher_or_API_authorship_cannot_be_presented_as_learner(self):
        for field, value in [('authorship', 'external_teacher_distillation'), ('api_training_data', True)]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as d:
                path = fixture(d)
                rewrite_bound(d, 'generations/train-cycle1-0.json', lambda r: r.update({field: value}))
                with self.assertRaisesRegex(ValueError, 'Teacher/API'): elo.Reader(path).build()

    def test_checkpoint_and_historical_source_alias_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = fixture(d)
            rewrite_bound(d, 'generations/capability-0.json', lambda r: r.update(checkpoint='cycle1'))
            with self.assertRaisesRegex(ValueError, 'schedule/checkpoint'): elo.Reader(path).build()

    def test_empty_prepared_snapshot_has_no_ratings_and_null_rates(self):
        with tempfile.TemporaryDirectory() as d:
            value = elo.Reader(fixture(d, empty=True)).build()
            self.assertEqual(value['state'], 'no_ratable_committed_games')
            self.assertEqual(value['controllers'], []); self.assertEqual(value['rating_history'], [])
            self.assertEqual(value['counts']['prescribed_program_draws'], 0)
            self.assertIsNone(value['counts']['invalid_authorship_rate_all_draws'])

    def test_all_invalid_committed_draws_have_no_invented_anchor_tie(self):
        with tempfile.TemporaryDirectory() as d:
            value = elo.Reader(fixture(d, all_invalid=True)).build()
            self.assertEqual(value['counts']['prescribed_program_draws'], 8)
            self.assertEqual(value['counts']['invalid_authorship_rate_all_draws'], 1)
            self.assertEqual(value['counts']['played_games'], 0)
            self.assertEqual(value['rating_history'], [])
            self.assertTrue(value['controllers'])
            self.assertTrue(all(r['rating'] is None and not r['ranked'] and r['played_seat_exposures'] == 0
                                for r in value['controllers']))

    def test_uncommitted_group_is_not_rated_or_counted_as_zero(self):
        with tempfile.TemporaryDirectory() as d:
            path = fixture(d); audit = json.loads(path.read_text())
            audit['training_groups'][0]['checkpoint_committed'] = False
            path.write_text(json.dumps(audit)); value = elo.Reader(path).build()
            self.assertEqual(value['excluded_uncommitted_cycles'], [1])
            self.assertEqual(value['counts']['prescribed_program_draws'], 0)

    def test_fixture_export_refused_and_source_rechecked(self):
        with tempfile.TemporaryDirectory() as d:
            path = fixture(d); reader = elo.Reader(path)
            with self.assertRaisesRegex(ValueError, 'Synthetic fixtures'):
                elo.export(path, Path(d)/'no-real-rankings.json')
            self.assertFalse(Path(d, 'no-real-rankings.json').exists())
            path.write_text(path.read_text()+'\n')
            with self.assertRaisesRegex(ValueError, 'audit changed'): reader.build()


if __name__ == '__main__': unittest.main()
