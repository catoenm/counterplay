"""CPU contract/replay fixtures; no model calls or experimental outcomes."""
from copy import deepcopy
import json
import random
import unittest

from selfplay.arena_mixed import envs


def episode(name, seed=47, regime='train'):
    env = envs.make_env(name, seed, regime)
    record = envs.episode_header(env, f'fixture-{name}-{seed}')
    chooser = random.Random(seed + 8191)
    while not env.done:
        menus = {p: env.legal_actions(p) for p in env.active_players()}
        for menu in menus.values():
            chooser.shuffle(menu)
        actions = {p: env.heuristic_action(p) for p in env.active_players()}
        record['events'].append(envs.record_step(env, actions, menus))
    return envs.finish_episode(record, env)


class MixedArenaContractTests(unittest.TestCase):
    def test_registered_horizons_and_immutable_config(self):
        expected = {'market': 16, 'maze': 32, 'chess': 96, 'planning': 12, 'coding': 12}
        for name, limit in expected.items():
            env = envs.make_env(name, 0)
            self.assertEqual(env.max_rounds, limit)
            self.assertEqual(env.snapshot(), envs.make_env(name, 0).snapshot())
        config = envs.protocol_config(); config['horizons']['maze'] = 8
        self.assertEqual(envs.protocol_config()['horizons']['maze'], 32)
        with self.assertRaisesRegex(ValueError, 'protocol'):
            envs.make_env('maze', 0, protocol=config)
        for seed in (False, -1, 1.5):
            with self.assertRaises(ValueError): envs.make_env('maze', seed)
        with self.assertRaises(ValueError): envs.make_env('unknown', 1)

    def test_replay_covers_all_active_players_after_json_roundtrip(self):
        for name in envs.ENVIRONMENTS:
            with self.subTest(name=name):
                original = episode(name)
                record = json.loads(json.dumps(original))
                result = envs.verify_episode(record)
                self.assertTrue(result['verified'])
                self.assertEqual(result['decisions'], result['turns'] * (2 if name in ('market', 'maze') else 1))
                for event in record['events']:
                    self.assertEqual(set(event['observations']), set(event['legal_actions']))
                    self.assertEqual(set(event['observations']), set(event['actions']))
                self.assertEqual(record, json.loads(json.dumps(episode(name))))

    def test_independent_tampering_rejected_at_each_evidence_boundary(self):
        record = json.loads(json.dumps(episode('market')))
        mutants = []
        row = deepcopy(record); row['events'][0]['observations']['0']['private_signal'] += 1; mutants.append(row)
        row = deepcopy(record); row['events'][0]['observations']['0']['terminal_value'] = row['initial_state']['terminal_value']; mutants.append(row)
        row = deepcopy(record); row['events'][0]['legal_actions']['0'].pop(); mutants.append(row)
        row = deepcopy(record); row['events'][0]['actions']['0'] = {'type': 'hold', 'extra': True}; mutants.append(row)
        row = deepcopy(record); row['events'][0]['state']['terminal_value'] += 1; mutants.append(row)
        row = deepcopy(record); row['returns'][0] += .01; mutants.append(row)
        row = deepcopy(record); row['metrics']['absolute_pnl'][0] += 1; mutants.append(row)
        row = deepcopy(record); row['events'].pop(); mutants.append(row)
        row = deepcopy(record); row['events'].append(deepcopy(row['events'][-1])); mutants.append(row)
        row = deepcopy(record); row['scenario_sha256'] = '0' * 64; mutants.append(row)
        for index, mutant in enumerate(mutants):
            with self.subTest(index=index), self.assertRaises(ValueError):
                envs.verify_episode(mutant)

    def test_actions_are_atomic_and_type_strict_even_when_python_equality_is_not(self):
        env = envs.make_env('market', 7)
        before = deepcopy(env.snapshot())
        first = next(a for a in env.legal_actions(0) if a.get('quantity') == 1)
        bad = {**first, 'quantity': True}
        for actions in ({0: bad, 1: {'type': 'hold'}},
                        {0: first}, {False: first, True: {'type': 'hold'}},
                        {0: first, '0': first, 1: {'type': 'hold'}}):
            with self.assertRaises(ValueError): envs.record_step(env, actions)
            self.assertEqual(env.snapshot(), before)
        menus = {p: env.legal_actions(p) for p in (0, 1)}
        menus[1] = menus[1][:-1]
        with self.assertRaises(ValueError): envs.record_step(env, {0: first, 1: {'type': 'hold'}}, menus)
        self.assertEqual(env.snapshot(), before)

    def test_private_market_probe_checks_future_signals_and_menu_not_only_observation(self):
        env = envs.make_env('market', 29)
        before = deepcopy(env.snapshot())
        self.assertTrue(envs.assert_private_boundary(env)['checked'])
        self.assertEqual(env.snapshot(), before)
        from unittest.mock import patch
        cls = type(env); legal = cls.legal_actions
        with patch.object(cls, 'legal_actions', lambda self, p: legal(self, p) + [{'type': 'leak', 'value': self._terminal_value}]):
            with self.assertRaisesRegex(ValueError, 'leaked'): envs.assert_private_boundary(env)

    def test_maze_reachability_and_block_budget_survive_longer_random_games(self):
        for seed in range(6):
            env = envs.make_env('maze', seed)
            rng = random.Random(seed)
            while not env.done:
                envs.record_step(env, {p: rng.choice(env.legal_actions(p)) for p in env.active_players()})
                for p in (0, 1):
                    todo = [env.positions[p]]; reached = set(todo)
                    while todo:
                        r, c = todo.pop()
                        for cell in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                            if (0 <= cell[0] < env.size and 0 <= cell[1] < env.size
                                    and cell not in env.walls[p] and cell not in reached):
                                reached.add(cell); todo.append(cell)
                    self.assertIn(env.goal, reached)
                    self.assertEqual(env.blocks_placed[p] + env.blocks_remaining[p], 2)

    def test_single_player_rewards_roles_and_task_privacy(self):
        for name in ('planning', 'coding'):
            env = envs.make_env(name, 7)
            self.assertEqual(env.active_players(), [0])
            self.assertEqual(envs.validate_env(env)['role_count'], 1)
            self.assertTrue(envs.assert_private_boundary(env)['checked'])
            with self.assertRaises(ValueError): envs.policy_view(env, 1)
            record = episode(name, 7)
            self.assertEqual(record['role_count'], 1)
            self.assertEqual(record['returns'], [1., 0.])
            self.assertTrue(envs.verify_episode(record)['verified'])
            bad = deepcopy(record); bad['role_count'] = 2
            with self.assertRaisesRegex(ValueError, 'denominator'): envs.verify_episode(bad)
            bad = deepcopy(record); bad['returns'][1] = -1.
            with self.assertRaises(ValueError): envs.verify_episode(bad)


class ScenarioIsolationTests(unittest.TestCase):
    @staticmethod
    def spec(name, seed, split, scenario, episode_id):
        return {'environment': name, 'seed': seed, 'regime': 'eval' if split == 'eval' else 'train',
                'split': split, 'scenario_id': scenario, 'episode_id': episode_id}

    def test_chess_color_mirrors_are_one_family_and_labels_do_not_hide_aliases(self):
        self.assertEqual(envs.scenario_fingerprint('chess', 40, 'train'),
                         envs.scenario_fingerprint('chess', 41, 'eval'))
        schedule = {'train': [self.spec('chess', 40, 'train', 'one', 'a')],
                    'eval': [self.spec('chess', 41, 'eval', 'two', 'b')]}
        with self.assertRaisesRegex(ValueError, 'aliases'): envs.validate_schedule(schedule)
        schedule['eval'][0].update(split='train', scenario_id='one')
        self.assertEqual(envs.validate_schedule(schedule)['distinct_scenarios'], 1)

    def test_market_hidden_event_identity_and_seed_independent_maze_identity(self):
        env = envs.make_env('market', 31); original = env.snapshot(); altered = deepcopy(original)
        altered['seed'] += 100
        self.assertEqual(envs.scenario_fingerprint('market', initial_state=original),
                         envs.scenario_fingerprint('market', initial_state=altered))
        altered['schedule'][-1]['priority'].reverse()
        self.assertNotEqual(envs.scenario_fingerprint('market', initial_state=original),
                            envs.scenario_fingerprint('market', initial_state=altered))
        original = envs.make_env('maze', 1).snapshot(); altered = deepcopy(original)
        altered.update(seed=900, regime='eval', round_limit=100)
        self.assertEqual(envs.scenario_fingerprint('maze', initial_state=original),
                         envs.scenario_fingerprint('maze', initial_state=altered))

    def test_fresh_allocation_skips_structural_aliases_and_rejects_historical_overlap(self):
        forbidden = envs.scenario_fingerprint('chess', 60)
        result = envs.select_scenarios('chess', 3, 60, excluded=[forbidden])
        self.assertEqual(result['rejected'][0]['seed'], 60)
        self.assertEqual(len({r['scenario_sha256'] for r in result['selected']}), 3)
        self.assertNotIn(forbidden, {r['scenario_sha256'] for r in result['selected']})
        self.assertEqual(result, envs.select_scenarios('chess', 3, 60, excluded=[forbidden]))
        schedule = {'eval': [self.spec('chess', 61, 'eval', 'one', 'a')]}
        with self.assertRaisesRegex(ValueError, 'historical'):
            envs.validate_schedule(schedule, prior_fingerprints=[forbidden])
        with self.assertRaisesRegex(ValueError, 'fixed bound'):
            envs.select_scenarios('chess', 1, 60, excluded=[forbidden], maximum_attempts=1)


if __name__ == '__main__':
    unittest.main()
