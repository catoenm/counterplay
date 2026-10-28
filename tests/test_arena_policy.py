"""Checks of the RL update's direction, independence, and checkpoint isolation."""
from copy import deepcopy
import importlib.util
import unittest

from selfplay.arena.policy import ActionPolicy, clipped_objective, episode_advantages


class EpisodeBaselineTests(unittest.TestCase):
    def test_baseline_excludes_own_episode_and_other_role(self):
        episodes = [
            {'environment': 'maze', 'episode_id': 'm0', 'returns': [1.0, -1.0]},
            {'environment': 'maze', 'episode_id': 'm1', 'returns': [-0.5, 0.5]},
            {'environment': 'maze', 'episode_id': 'm2', 'returns': [0.0, 0.0]},
            {'environment': 'market', 'episode_id': 'x0', 'returns': [100.0, -100.0]},
        ]
        first = episode_advantages(episodes)
        self.assertAlmostEqual(first[('m0', 0)], 1.25)
        altered = deepcopy(episodes)
        altered[0]['returns'] = [0.25, 999.0]
        altered[1]['returns'][1] = -999.0
        second = episode_advantages(altered)
        # The inferred baseline stays at -0.25 despite changing the learner's
        # own return, its same-episode opponent, and another role's return.
        self.assertAlmostEqual(1.0 - first[('m0', 0)],
                               0.25 - second[('m0', 0)])

    def test_turn_count_cannot_change_return_baseline(self):
        episodes = [
            {'environment': 'chess', 'episode_id': 'c0', 'returns': [1.0, -1.0]},
            {'environment': 'chess', 'episode_id': 'c1', 'returns': [-1.0, 1.0]},
        ]
        before = episode_advantages(episodes)
        episodes[0]['transitions'] = [{}] * 100
        episodes[1]['transitions'] = [{}]
        self.assertEqual(before, episode_advantages(episodes))


@unittest.skipUnless(importlib.util.find_spec('mlx'), 'MLX is optional outside Apple Silicon')
class PolicyGradientTests(unittest.TestCase):
    def objective(self, logits, advantage, old_probability=0.5):
        import mlx.core as mx
        lp = logits - mx.logsumexp(logits)
        return clipped_objective(
            lp[0:1], mx.array([old_probability]).log(),
            mx.array([advantage]), mx.array([1.0]),
            mx.array([0.0]), mx.array([0.0]),
            entropy_coef=0.0, kl_coef=0.0,
        )

    def test_positive_and_negative_returns_move_probability_oppositely(self):
        import mlx.core as mx
        initial = mx.array([0.0, 0.0])
        for advantage, expected in [(1.0, 'increase'), (-1.0, 'decrease')]:
            grad = mx.grad(lambda x: self.objective(x, advantage))(initial)
            updated = initial - 0.1 * grad
            probability = float(mx.softmax(updated)[0])
            if expected == 'increase':
                self.assertGreater(probability, 0.5)
            else:
                self.assertLess(probability, 0.5)

    def test_clipping_stops_favorable_out_of_range_updates(self):
        import mlx.core as mx
        for probabilities, advantage in [([0.8, 0.2], 1.0), ([0.2, 0.8], -1.0)]:
            logits = mx.log(mx.array(probabilities))
            grad = mx.grad(lambda x: self.objective(x, advantage))(logits)
            self.assertLess(float(mx.max(mx.abs(grad))), 1e-6)

    def test_saved_checkpoint_is_not_modified_by_training_or_switches(self):
        import mlx.core as mx
        import mlx.nn as nn
        policy = ActionPolicy.__new__(ActionPolicy)
        policy.mx = mx
        policy.model = nn.Linear(2, 1, bias=False)
        policy.model.update({'weight': mx.array([[1.0, 2.0]])})
        policy.checkpoints, policy.active = {}, 'base'
        policy.save_memory('base')
        # Optimizers replace parameter arrays; use that same mutation pattern.
        policy.model.update({'weight': policy.model.weight + 5.0})
        policy.save_memory('cycle1')
        policy.active = 'cycle1'
        sample = mx.array([[1.0, 1.0]])
        self.assertAlmostEqual(float(policy.model(sample).item()), 13.0)
        policy.activate('base')
        self.assertAlmostEqual(float(policy.model(sample).item()), 3.0)
        policy.activate('cycle1')
        self.assertAlmostEqual(float(policy.model(sample).item()), 13.0)
        policy.activate('base')
        self.assertAlmostEqual(float(policy.model(sample).item()), 3.0)


if __name__ == '__main__':
    unittest.main()
