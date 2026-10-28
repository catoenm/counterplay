import itertools
import json
import unittest
from dataclasses import asdict

from selfplay.environment import RULES, Task, construct_task, generate_tasks, grade, render, step
from selfplay.pilot import frontier, summarize, grade_response


class EnvironmentTests(unittest.TestCase):
    def test_transition_reference(self):
        # Independent affine matrix specification, exhaustive over small worlds.
        matrices = {
            "inc_x": (1, 0, 0, 1, 1, 0), "inc_y": (1, 0, 0, 1, 0, 1),
            "swap": (0, 1, 1, 0, 0, 0), "sum_x": (1, 1, 0, 1, 0, 0),
            "sum_y": (1, 0, 1, 1, 0, 0), "double_x": (2, 0, 0, 1, 0, 0),
            "double_y": (1, 0, 0, 2, 0, 0),
        }
        for m in (3, 7, 11):
            for x, y in itertools.product(range(m), repeat=2):
                for op, (a, b, c, d, e, f) in matrices.items():
                    self.assertEqual(step((x, y), op, m), ((a*x+b*y+e) % m, (c*x+d*y+f) % m))

    def test_witnesses_depths_and_identity(self):
        tasks = generate_tasks(120, seed=5, depths=(1, 2, 3, 4))
        self.assertEqual(len({t.identity for t in tasks}), 120)
        for i, t in enumerate(tasks):
            self.assertTrue(t.accepts(t.witness))
            self.assertEqual(len(t.shortest_plan()), (1, 2, 3, 4)[i % 4])
            self.assertEqual(Task.from_dict(asdict(t)), t)
            d = asdict(t); d["actions"] = tuple(reversed(t.actions))
            self.assertEqual(Task.from_dict(d).identity, t.identity)

    def test_renderer_preserves_all_short_plans(self):
        for task in generate_tasks(12):
            for variant in ("canonical", "reordered", "renamed", "renamed_reordered"):
                prompt, mapping = render(task, variant, 4)
                self.assertNotIn("witness", prompt)
                inverse = {a: label for label, a in mapping.items()}
                for n in range(task.max_steps + 1):
                    for plan in itertools.product(task.actions, repeat=n):
                        text = json.dumps({"plan": [inverse[a] for a in plan]})
                        self.assertEqual(grade(task, text, mapping)["success"], task.accepts(plan))

    def test_reject_false_witness_and_malformed_output(self):
        t = generate_tasks(1)[0]
        d = asdict(t); d["witness"] = []
        with self.assertRaises(ValueError): Task.from_dict(d)
        _, mapping = render(t)
        for text in ('{}', '[]', '{"plan":[true]}', '{"plan":[{}]}', '{"plan":"tool_0"}', 'some text'):
            self.assertFalse(grade(t, text, mapping)["success"])
        for invalid in (True, 100, -1):
            d = asdict(t); d["modulus"] = invalid
            with self.assertRaises(ValueError): Task.from_dict(d)

    def test_reward_and_partial_summary(self):
        self.assertEqual([frontier(p) for p in (0, .5, 1)], [0, 1, 0])
        r = summarize([{"task_id": "x", "variant": "renamed", "success": False, "status": "wrong_goal"}])
        self.assertEqual(r["unique_tasks"], 1)
        self.assertEqual(r["paired_renamed_reordered_minus_canonical"]["paired_tasks"], 0)

    def test_constructive_target(self):
        d = {"modulus": 7, "start": [1, 2], "actions": ["inc_x", "inc_y", "swap", "sum_x"],
             "max_steps": 3, "witness": ["inc_x", "swap", "sum_x"]}
        t = construct_task(d)
        self.assertEqual(t.goal, (4, 2))
        self.assertTrue(t.accepts(t.witness))
        with self.assertRaises(ValueError): construct_task({**d, "goal": [3, 4]})
        with self.assertRaises(ValueError): construct_task({**d, "witness": ["unknown"]})

    def test_reasoning_and_truncation(self):
        t = generate_tasks(1)[0]
        prompt, mapping = render(t, "renamed", prompt_version="v2")
        self.assertNotIn("tool_0", prompt)
        inverse = {a: label for label, a in mapping.items()}
        answer = json.dumps({"plan": [inverse[a] for a in t.witness]})
        r = {"text": "Checking arithmetic.\n</think>\n" + answer, "finish_reason": "stop"}
        self.assertTrue(grade_response(t, r, mapping, True)["success"])
        self.assertEqual(grade_response(t, {**r, "finish_reason": "length"}, mapping, True)["status"], "truncated")
        self.assertEqual(grade_response(t, {"text": answer}, mapping, True)["status"], "unfinished_reasoning")


if __name__ == "__main__":
    unittest.main()
