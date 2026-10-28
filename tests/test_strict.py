from dataclasses import replace
from itertools import product
import json
import unittest

from selfplay.depth import family
from selfplay.environment import Task
from selfplay.strict import PlanGrammar


class StrictTests(unittest.TestCase):
    def test_grammar_all_legal_sequences_including_wrong_answers(self):
        labels = ["tool_0", "tool_1", "tool_2", "tool_3"]
        grammar = PlanGrammar(lambda s: list(s.encode()), labels, 3, [256])
        self.assertEqual(grammar.plans, 85)
        for length in range(4):
            for plan in product(labels, repeat=length):
                tokens = list(json.dumps({"plan": plan}, separators=(",", ":")).encode()) + [256]
                for i, token in enumerate(tokens):
                    self.assertIn(token, grammar.allowed(tokens[:i]))
                self.assertEqual(grammar.allowed(tokens), (256,))
        for bad in ['{"plan":["tool0"]}', '{"plan":["tool_4"]}',
                    '{"plan":["tool_0","tool_0","tool_0","tool_0"]}']:
            with self.assertRaises(ValueError):
                grammar.allowed(list(bad.encode()))

    def test_family_groups_budget_and_coordinate_swap(self):
        a = Task(7, (0, 0), (2, 0), ("inc_x", "inc_y", "sum_x", "swap"), 2, ("inc_x", "inc_x"))
        b = Task(7, (0, 0), (0, 2), ("inc_y", "inc_x", "sum_y", "swap"), 2, ("inc_y", "inc_y"))
        self.assertEqual(family(a), family(b))
        self.assertEqual(family(a), family(replace(a, max_steps=3)))
        self.assertNotEqual(a.identity, b.identity)

    def test_fixed_horizon_keeps_every_action_combination(self):
        grammar = PlanGrammar(lambda s: list(s.encode()), ["a", "b"], 3, [256], exact_steps=True)
        self.assertEqual(grammar.plans, 8)
        for plan in product(["a", "b"], repeat=3):
            ids = list(json.dumps({"plan": plan}, separators=(",", ":")).encode())
            self.assertIn(256, grammar.allowed(ids))
        with self.assertRaises(ValueError):
            grammar.allowed(list(b'{"plan":[]}'))


if __name__ == "__main__":
    unittest.main()
