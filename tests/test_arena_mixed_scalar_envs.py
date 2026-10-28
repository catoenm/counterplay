"""CPU-only rational-score and immutable native-dynamics regression fixtures."""
from copy import deepcopy
from decimal import Decimal, localcontext
import json
import math
import random
import unittest

from selfplay.arena_mixed import envs as original
from selfplay.arena_mixed_scalar import envs


def without_score_metrics(snapshot):
    value = deepcopy(snapshot)
    metrics = value['metrics']
    for key in ('returns', 'reward_formula', 'reward_normalization',
                'relative_pnl_numerator', 'relative_reward_denominator'):
        metrics.pop(key, None)
    return value


def fixture(name, seed=61):
    game = envs.make_env(name, seed)
    record = envs.episode_header(game, f'scalar-fixture-{name}-{seed}')
    chooser = random.Random(seed + 400)
    while not game.done:
        actions = {p: chooser.choice(game.legal_actions(p)) for p in game.active_players()}
        record['events'].append(envs.record_step(game, actions))
    return envs.finish_episode(record, game)


class ScalarEnvironmentTests(unittest.TestCase):
    def test_exact_rational_reference_and_sign_monotonicity(self):
        values = [-10**30, -1024, -128, -15, -10, -6, -1, 0, 1, 6, 10, 15, 128, 1024, 10**30]
        with localcontext() as context:
            context.prec = 100
            expected = [float(Decimal(n) / Decimal(128 + abs(n))) for n in values]
        observed = [envs.rational_market_score(n) for n in values]
        self.assertEqual([v.hex() for v in observed], [v.hex() for v in expected])
        self.assertEqual(observed, sorted(observed))
        for n, score in zip(values, observed):
            self.assertLessEqual(abs(score), 1.)
            self.assertEqual(envs.rational_market_score(-n), -score)
        for invalid in (True, False, 1., '1'):
            with self.assertRaises(ValueError): envs.rational_market_score(invalid)

    def test_same_actions_preserve_native_integer_market_and_public_inputs(self):
        for seed in range(8):
            old = original.make_env('market', seed)
            new = envs.make_env('market', seed)
            chooser = random.Random(1700 + seed)
            with self.assertRaises(ValueError): new.returns()
            while not old.done:
                self.assertEqual(old.done, new.done)
                self.assertEqual(envs.canonical(without_score_metrics(old.snapshot())),
                                 envs.canonical(without_score_metrics(new.snapshot())))
                for player in (0, 1):
                    view = new.observation(player)
                    self.assertIn('d/(128+abs(d))', view['rules'].pop('terminal_reward'))
                    self.assertEqual(view, old.observation(player))
                    self.assertEqual(new.legal_actions(player), old.legal_actions(player))
                actions = {p: chooser.choice(old.legal_actions(p)) for p in (0, 1)}
                old.step(deepcopy(actions)); new.step(deepcopy(actions))
            self.assertEqual(envs.canonical(without_score_metrics(old.snapshot())),
                             envs.canonical(without_score_metrics(new.snapshot())))
            self.assertEqual(new.metrics()['absolute_pnl'], old.metrics()['absolute_pnl'])
            self.assertEqual(new.metrics()['cash'], old.metrics()['cash'])
            self.assertEqual(new.metrics()['inventory'], old.metrics()['inventory'])

    def test_all_saved_score_locations_and_json_roundtrip_are_exact(self):
        record = fixture('market')
        expected = record['returns']
        metrics = record['metrics']
        difference = metrics['absolute_pnl'][0] - metrics['absolute_pnl'][1]
        self.assertEqual(metrics['relative_pnl_numerator'], difference)
        self.assertEqual(metrics['relative_reward_denominator'], 128 + abs(difference))
        self.assertEqual(expected, [envs.rational_market_score(difference), -envs.rational_market_score(difference)])
        self.assertEqual(metrics['reward_formula'], envs.MARKET_REWARD_FORMULA)
        self.assertEqual(record['events'][-1]['state']['metrics'], metrics)
        for _ in range(3):
            record = json.loads(json.dumps(record, sort_keys=True))
            self.assertTrue(envs.verify_episode(record)['verified'])
            self.assertEqual([v.hex() for v in record['returns']], [v.hex() for v in expected])

    def test_protocol_is_new_and_cross_protocol_relabeling_is_rejected(self):
        self.assertNotEqual(envs.protocol_config(), original.protocol_config())
        self.assertEqual(envs.protocol_config()['horizons'], original.protocol_config()['horizons'])
        for name in envs.ENVIRONMENTS:
            self.assertEqual(envs.scenario_fingerprint(name, 90), original.scenario_fingerprint(name, 90))
            record = fixture(name)
            self.assertTrue(envs.verify_episode(json.loads(json.dumps(record)))['verified'])
            with self.assertRaisesRegex(ValueError, 'protocol'): original.verify_episode(record)
        with self.assertRaisesRegex(ValueError, 'protocol'):
            envs.make_env('market', 0, protocol=original.protocol_config())

    def test_no_blanket_tolerance_for_one_ulp_score_or_integer_state_mutation(self):
        record = json.loads(json.dumps(fixture('market')))
        for path in ('returns', 'metrics', 'state'):
            bad = deepcopy(record)
            target = bad['returns'] if path == 'returns' else (bad['metrics']['returns'] if path == 'metrics'
                                                              else bad['events'][-1]['state']['metrics']['returns'])
            target[0] = math.nextafter(target[0], math.inf)
            with self.subTest(path=path), self.assertRaises(ValueError): envs.verify_episode(bad)
        bad = deepcopy(record); bad['events'][0]['state']['accounts']['0']['cash'] += 1
        with self.assertRaises(ValueError): envs.verify_episode(bad)

    def test_no_private_pnl_in_live_policy_view_and_required_source_closure(self):
        game = envs.make_env('market', 7)
        self.assertTrue(envs.assert_private_boundary(game)['checked'])
        for player in (0, 1):
            view = game.observation(player)
            self.assertNotIn('absolute_pnl', view)
            self.assertNotIn('relative_pnl_numerator', view)
        sources = envs.source_hashes()
        self.assertIn('selfplay/arena_mixed_scalar/envs.py', sources)
        self.assertIn('selfplay/arena/market.py', sources)
        self.assertIn('selfplay/arena_mixed/coding.py', sources)
        self.assertNotIn('selfplay/arena_mixed/envs.py', sources)


if __name__ == '__main__':
    unittest.main()
