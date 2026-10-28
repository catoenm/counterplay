import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from arena_elo import (build_ratings, elo_update, outcome_score, evaluation_specs,
                       policy_profile, policy_fingerprint, canonical, analyze)


def match(identity, players, reward, seed=1, game='maze', reason='distance_adjudication_at_cap'):
    return {'episode_id': identity, 'environment': game, 'policies': players,
            'returns': [reward, -reward], 'seed': seed, 'metrics': {'finish_reason': reason}}


class EloTests(unittest.TestCase):
    def test_pair_conservation_equal_draw_and_expected_win(self):
        self.assertEqual(elo_update(1500, 1500, .5), (1500, 1500))
        self.assertEqual(elo_update(1500, 1500, 1), (1512, 1488))
        for a, b, result in ((1700, 1300, 0), (1300, 1700, .5), (1500, 1560, 1)):
            updated = elo_update(a, b, result)
            self.assertAlmostEqual(sum(updated), a+b)
        first, second = elo_update(1700, 1300, .5)
        self.assertLess(first, 1700)
        self.assertGreater(second, 1300)

    def test_scoring_is_adjudicated_sign_not_fractional_reward(self):
        self.assertEqual([outcome_score(v) for v in (-1, -.001, 0, .001, 1)], [0, 0, .5, 1, 1])
        for invalid in (True, math.nan, math.inf, 2, 'win'):
            with self.assertRaises(ValueError):
                outcome_score(invalid)

    def test_deterministic_input_order_and_persistent_identity(self):
        rows = [match('base-h', ['base', 'heuristic'], -.2),
                match('new-h', ['cycle1', 'heuristic'], .2, 2),
                match('new-old', ['cycle1', 'base'], -.1, 3),
                match('final-old', ['cycle2', 'cycle1'], .1, 4)]
        expected = build_ratings(rows)
        shuffled = copy.deepcopy(rows)
        random.Random(17).shuffle(shuffled)
        self.assertEqual(expected, build_ratings(shuffled))
        series = expected['games']['maze']['series']
        self.assertEqual([p['iteration'] for p in series['cycle2']], [2])
        self.assertEqual(series['base'][2]['new_games'], 0)
        self.assertEqual(series['base'][1]['rating'], series['base'][2]['rating'])
        self.assertNotEqual(series['base'][0]['rating'], series['base'][1]['rating'])
        self.assertEqual(expected['observed_iterations'], [0, 1, 2])
        self.assertAlmostEqual(sum(expected['games']['maze']['ratings'].values()), 6000)

    def test_independent_games_and_identical_policy_exclusion(self):
        rows = [match('mm', ['base', 'base'], 1),
                match('maze', ['base', 'heuristic'], 1),
                match('market', ['base', 'heuristic'], -1, game='market', reason='settlement')]
        result = build_ratings(rows)
        self.assertEqual(result['games']['maze']['ratings']['base'], 1512)
        self.assertEqual(result['games']['market']['ratings']['base'], 1488)
        self.assertEqual(len(result['matches']), 2)
        self.assertEqual(len(result['excluded_matches']), 1)
        self.assertEqual(result['games']['maze']['termination_counts'], {'distance_adjudication_at_cap': 1})

    def test_no_future_stage_and_duplicate_rejection(self):
        row = match('observed', ['base', 'heuristic'], 0)
        result = build_ratings([row])
        self.assertEqual(result['observed_iterations'], [0])
        self.assertNotIn('cycle1', result['games']['maze']['series'])
        with self.assertRaises(ValueError):
            build_ratings([row, row])
        self.assertEqual(build_ratings([])['observed_iterations'], [])

    def test_identical_weights_under_distinct_names_are_not_rated(self):
        row = match('aliases', ['cycle1', 'cycle2'], 1)
        result = build_ratings([row], fingerprints={'cycle1': 'same-model-revision-and-adapter',
                                                   'cycle2': 'same-model-revision-and-adapter'})
        self.assertEqual(result['matches'], [])
        self.assertEqual(len(result['excluded_matches']), 1)

    def test_merged_schedule_counts_exclude_both_training_id_formats(self):
        from selfplay.arena.experiment import game_specs
        from selfplay.league import make_schedule
        arena_schedule = {f'train-{i}': game_specs('train', cycle=i) for i in range(2)}
        for checkpoint in ('base', 'cycle1', 'cycle2'):
            for opponent in ('base', 'heuristic'):
                arena_schedule[f'eval-{checkpoint}-{opponent}'] = game_specs('eval', candidate=checkpoint, opponent=opponent)
        arena_schedule['league-cycle2-cycle1'] = game_specs('league', candidate='cycle2', opponent='cycle1')
        continuation_schedule, _ = make_schedule(['market', 'maze', 'chess'])
        initial, continuation = evaluation_specs(arena_schedule), evaluation_specs(continuation_schedule)
        self.assertEqual(len(initial), 126)
        self.assertEqual(len(continuation), 162)
        self.assertFalse(set(initial) & set(continuation))
        merged = {**initial, **continuation}
        self.assertEqual(len(merged), 288)
        self.assertEqual(sum(row['policies'][0] != row['policies'][1] for row in merged.values()), 270)
        self.assertTrue(all(row['regime'] == 'eval' for row in merged.values()))


class PolicyArithmeticTests(unittest.TestCase):
    @staticmethod
    def manifest(dtype='float32', tag='arena-v2-fp32'):
        arithmetic = {'floating_parameters': dtype, 'activations': dtype, 'integer_quantized_storage': 'unchanged',
                      'policy': 'single-token legal categorical'}
        return {'rating_compatibility': tag, 'policy_arithmetic': arithmetic,
                'policy_arithmetic_sha256': hashlib.sha256(canonical(arithmetic).encode()).hexdigest(),
                'horizons': {'maze': 8}, 'source_sha256': {}, 'model': {'model': 'fixture', 'revision': 'fixture'}}

    def test_profile_hash_validated_and_legacy_is_separate(self):
        legacy = policy_profile({})
        self.assertEqual(legacy, policy_profile({'rating_compatibility': 'arena-v1'}))
        self.assertEqual(legacy['policy_arithmetic_sha256'], 'legacy-unspecified')
        explicit = self.manifest()
        self.assertNotEqual(legacy, policy_profile(explicit))
        explicit['policy_arithmetic']['activations'] = 'bfloat16'
        with self.assertRaisesRegex(ValueError, 'canonical arithmetic'):
            policy_profile(explicit)
        for broken in ({'rating_compatibility': 'arena-v2-fp32'},
                       {'policy_arithmetic': {'activations': 'float32'}},
                       {'policy_arithmetic_sha256': 'missing-profile'}):
            with self.assertRaises(ValueError):
                policy_profile(broken)

    def test_identical_weights_in_different_arithmetic_have_distinct_fingerprints(self):
        metadata = {'model': 'fixture', 'revision': 'fixture', 'adapter_sha256': 'same-bytes'}
        fp32 = policy_fingerprint(metadata, policy_profile(self.manifest()), inherited=True)
        bf16 = policy_fingerprint(metadata, policy_profile(self.manifest('bfloat16', 'bf16-fixture')), inherited=True)
        self.assertNotEqual(fp32, bf16)
        row = match('different-arithmetic', ['cycle1', 'cycle2'], 1)
        result = build_ratings([row], fingerprints={'cycle1': fp32, 'cycle2': bf16})
        self.assertEqual(len(result['matches']), 1)
        same = build_ratings([row], fingerprints={'cycle1': fp32, 'cycle2': fp32})
        self.assertEqual(len(same['excluded_matches']), 1)

    def test_local_metadata_must_match_but_inherited_legacy_bytes_are_allowed(self):
        metadata = {'model': 'fixture', 'revision': 'fixture', 'adapter_sha256': 'same-bytes'}
        profile = policy_profile(self.manifest())
        with self.assertRaisesRegex(ValueError, 'lacks declared'):
            policy_fingerprint(metadata, profile)
        inherited = policy_fingerprint(metadata, profile, inherited=True)
        metadata.update({key: value for key, value in self.manifest().items()
                         if key in ('rating_compatibility', 'policy_arithmetic', 'policy_arithmetic_sha256')})
        self.assertEqual(inherited, policy_fingerprint(metadata, profile))
        other = policy_profile(self.manifest('bfloat16', 'bf16-fixture'))
        with self.assertRaisesRegex(ValueError, 'differs from run'):
            policy_fingerprint(metadata, other)

    def test_analysis_rejects_cross_precision_and_different_pool_tags(self):
        with tempfile.TemporaryDirectory() as temporary:
            runs = [Path(temporary) / name for name in ('first-fixture', 'second-fixture')]
            for run in runs:
                run.mkdir()
            legacy = {'horizons': {'maze': 8}, 'source_sha256': {}, 'model': {'model': 'fixture', 'revision': 'fixture'}}
            for first, second in ((legacy, self.manifest()),
                                  (self.manifest(), self.manifest('bfloat16', 'arena-v2-fp32')),
                                  (self.manifest(), self.manifest(tag='another-fp32-pool'))):
                manifests = dict(zip((run.resolve() for run in runs), (first, second)))
                with patch('arena_elo.archived_factory', side_effect=lambda run: (None, manifests[run])):
                    with self.assertRaisesRegex(ValueError, 'Incompatible rating pool or policy arithmetic'):
                        analyze(runs)
            with patch('arena_elo.archived_factory', return_value=(None, self.manifest())):
                result = analyze(runs)
                self.assertEqual(result['rating_pool']['rating_compatibility'], 'arena-v2-fp32')
                self.assertEqual(result['completed_matches_verified'], 0)

    def test_analysis_inherited_checkpoint_effective_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / 'fp32-fixture'
            old_checkpoint = Path(temporary) / 'original-base-fixture'
            run.mkdir()
            old_checkpoint.mkdir()
            weights = b'explicit CPU-test fixture, not neural weights'
            (old_checkpoint / 'adapters.safetensors').write_bytes(weights)
            metadata = {'checkpoint': 'base', 'model': 'fixture', 'revision': 'fixture',
                        'adapter_sha256': hashlib.sha256(weights).hexdigest()}
            metadata_bytes = json.dumps(metadata).encode()
            (old_checkpoint / 'metadata.json').write_bytes(metadata_bytes)
            manifest = self.manifest()
            manifest['inherited_checkpoints'] = {'base': {'path': str(old_checkpoint),
                'adapter_sha256': metadata['adapter_sha256'], 'metadata_sha256': hashlib.sha256(metadata_bytes).hexdigest()}}
            with patch('arena_elo.archived_factory', return_value=(None, manifest)):
                result = analyze([run])
            expected = hashlib.sha256(policy_fingerprint(metadata, policy_profile(manifest), inherited=True).encode()).hexdigest()
            self.assertEqual(result['checkpoint_effective_policy_sha256']['base'], expected)
            self.assertEqual((old_checkpoint / 'metadata.json').read_bytes(), metadata_bytes)


if __name__ == '__main__':
    unittest.main()
