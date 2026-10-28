import copy
import json
import random
import unittest

from selfplay.arena.market import MarketEnv


class MarketTests(unittest.TestCase):
    def cleared(self, regime="train"):
        env = MarketEnv(14, regime=regime)
        for owner in env._accounts:
            env._cancel(owner)
        return env

    def test_price_time_priority_partial_fill_and_reservations(self):
        env = self.cleared()
        expensive = env._submit(1, "sell", 102, 1)
        first = env._submit(env.MAKER, "sell", 100, 2)
        second = env._submit(env.NOISE, "sell", 100, 2)
        env._submit(0, "buy", 103, 3)
        self.assertEqual([t["resting_order_id"] for t in env._trades], [first, second])
        self.assertEqual([t["price"] for t in env._trades], [100, 100])
        self.assertEqual([t["quantity"] for t in env._trades], [2, 1])
        self.assertEqual(env._accounts[0].cash, 300)
        self.assertEqual(env._accounts[0].inventory, 7)
        self.assertEqual(env._reserved(0), (0, 0))
        self.assertEqual(env._reserved(env.NOISE), (0, 1))
        self.assertIn(expensive, [o.order_id for o in env._orders])

    def test_cash_inventory_reservations_and_cancel_ownership(self):
        env = self.cleared()
        order = env._submit(0, "buy", 100, 5)
        self.assertEqual(env._available(0), (100, 4))
        with self.assertRaises(ValueError):
            env._submit(0, "buy", 100, 2)
        self.assertEqual(env._cancel(1, order), 0)
        self.assertEqual(env._available(0), (100, 4))
        self.assertEqual(env._cancel(0, order), 5)
        env._submit(0, "sell", 105, 4)
        with self.assertRaises(ValueError):
            env._submit(0, "sell", 105, 1)
        env._submit(1, "buy", 105, 2)
        self.assertEqual(env._reserved(0), (0, 2))
        self.assertEqual(env._cancel(0), 2)
        self.assertEqual(env._available(0), (810, 2))

    def test_fees_no_self_trade_and_conservation(self):
        env = self.cleared("ood")
        env._submit(0, "sell", 100, 1)
        env._submit(0, "buy", 100, 1)
        self.assertEqual(env._trades, [])
        # Self-trade prevention cancels the incoming remainder, leaving no
        # crossed same-owner quotes and never skipping a better-priced order.
        self.assertEqual(env._reserved(0), (0, 1))
        env._submit(1, "buy", 100, 1)
        self.assertEqual(len(env._trades), 1)
        self.assertEqual(env._fees_collected, 2)
        self.assertEqual(env._accounts[0].cash, 699)
        self.assertEqual(env._accounts[1].cash, 499)
        env._assert_invariants()

    def test_atomic_illegal_actions_and_snapshot_privacy(self):
        env = MarketEnv(6)
        before = env.snapshot()
        with self.assertRaises(ValueError):
            env.step({0: env.legal_actions(0)[1], 1: {"type": "steal"}})
        self.assertEqual(before, env.snapshot())
        obs = copy.deepcopy(env.observation(0))
        # Changing only hidden state must not alter player 0's observation.
        env._signals[1] += 1000
        env._terminal_value += 1000
        self.assertEqual(obs, env.observation(0))
        self.assertNotIn("terminal_value", obs)
        self.assertNotIn("private_signals", obs)
        self.assertEqual(env.metrics()["terminal_value"], None)
        with self.assertRaises(ValueError):
            env.returns()
        self.assertEqual(json.loads(json.dumps(obs)), obs)

    def test_random_rollouts_invariants_determinism_and_reward_signal(self):
        nonzero = 0
        for regime in ("train", "eval", "ood"):
            for seed in range(50):
                env, twin = MarketEnv(seed, regime), MarketEnv(seed, regime)
                chooser = random.Random(seed + 90)
                schedule = copy.deepcopy(env._schedule)
                while not env.done:
                    actions = {}
                    for player in env.active_players():
                        legal = env.legal_actions(player)
                        self.assertLessEqual(len(legal), 14)
                        actions[player] = chooser.choice(legal)
                    env.step(actions)
                    twin.step(actions)
                    env._assert_invariants()
                    self.assertEqual(env.snapshot(), twin.snapshot())
                reward = env.returns()
                self.assertEqual(sum(reward), 0)
                self.assertTrue(all(-1 <= r <= 1 for r in reward))
                self.assertEqual(env._schedule, schedule)
                self.assertFalse(env.active_players())
                self.assertEqual(env.legal_actions(0), [])
                nonzero += reward[0] != 0
        self.assertGreater(nonzero, 120)

    def test_terminal_marking_and_hold_baseline(self):
        env = MarketEnv(1)
        while not env.done:
            env.step({0: {"type": "hold"}, 1: {"type": "hold"}})
        self.assertEqual(env.metrics()["absolute_pnl"], [0, 0])
        self.assertEqual(env.returns(), [0, 0])
        # Last trade is explicitly irrelevant to terminal accounting.
        old_returns = env.returns()
        env._trades.append({"price": 10_000})
        self.assertEqual(env.returns(), old_returns)

    def test_signal_bot_legal_and_beats_random_on_aggregate(self):
        total_profit, relative = 0, 0
        for seed in range(100):
            env = MarketEnv(seed)
            chooser = random.Random(seed + 500)
            while not env.done:
                bot = env.heuristic_action(0)
                self.assertIn(bot, env.legal_actions(0))
                env.step({0: bot, 1: chooser.choice(env.legal_actions(1))})
            total_profit += env.metrics()["absolute_pnl"][0]
            relative += env.returns()[0]
        self.assertGreater(total_profit, 0)
        self.assertGreater(relative, 10)


if __name__ == "__main__":
    unittest.main()
