import copy
import json
import random
import unittest

from selfplay.program_arena import RobotArena, compile_program, play, verify_replay, ProgramValidationError
from selfplay.program_arena.dsl import MAX_BYTES, MAX_STEPS, INTEGER_BOUND
from selfplay.program_arena.examples import reference_programs, idle_program, var, op, assign, emit


class ProgramLanguageTests(unittest.TestCase):
    def test_host_escape_and_invalid_types_rejected(self):
        attacks = [
            '__import__("os").system("true")',
            {"version": 1, "body": [{"exec": "import os"}]},
            {"version": 1, "body": [assign("v_x", {"var": "__class__"})]},
            {"version": 1, "body": [assign("v_x", {"site": {"index": 0, "field": "__dict__"}})]},
            {"version": 1, "body": [assign("energy", 100)]},
            {"version": 1, "body": [assign("m16", 1)]},
            {"version": 1, "body": [assign("v_x", {"op": "**", "args": [2, 1000000]})]},
            {"version": 1, "body": [assign("v_x", True)]},
            {"version": 1, "body": [assign("v_x", 1.5)]},
            {"version": 1, "body": [{"repeat": True, "index": "v_i", "body": [emit("rest")]}]},
            {"version": 1, "body": [{"while": 1, "body": [emit("rest")]}]},
            {"version": 1, "body": [emit("rest", file="/tmp/escape")]},
            {"version": True, "body": [emit("rest")]},
            '{"version":1,"body":[{"emit":{"action":"rest","action":"harvest"}}]}',
        ]
        for attack in attacks:
            with self.subTest(attack=attack), self.assertRaises(ProgramValidationError):
                compile_program(attack)

    def test_size_depth_node_and_local_limits(self):
        with self.assertRaises(ProgramValidationError):
            compile_program(" " * (MAX_BYTES + 1))
        with self.assertRaises(ProgramValidationError):
            compile_program({"version": 1, "body": [emit("rest")] * 513})
        with self.assertRaises(ProgramValidationError):
            compile_program({"version": 1, "body": [assign(f"v_a{i}", 0) for i in range(33)]})
        deep = 0
        for _ in range(22):
            deep = op("abs", deep)
        with self.assertRaises(ProgramValidationError):
            compile_program({"version": 1, "body": [assign("v_x", deep)]})

    def test_bounded_nested_loops_and_atomic_memory_on_runtime_fault(self):
        body = [assign("m0", op("+", var("m0"), 1))]
        for name in ("v_i", "v_j", "v_k"):
            body = [{"repeat": 16, "index": name, "body": body}]
        program = compile_program({"version": 1, "body": [assign("m0", 999), *body, emit("harvest")]})
        decision = program.run(RobotArena(7).observation(0))
        self.assertEqual(decision.fault, "step_limit")
        self.assertEqual(decision.steps, MAX_STEPS)
        self.assertEqual(decision.action, {"action": "rest"})
        self.assertEqual(decision.memory, (0,) * 16)

    def test_memory_arithmetic_faults_and_no_emit(self):
        obs = RobotArena(2).observation(0)
        program = compile_program({"version": 1, "body": [assign("m0", op("+", var("m0"), 1)),
                                  assign("m1", op("*", 1000000, 1000000)), emit("rest")]})
        first = program.run(obs)
        second = program.run(obs, first.memory)
        self.assertEqual(second.memory[:2], (2, INTEGER_BOUND))
        for expr, fault in ((op("//", 1, 0), "division_by_zero"), (op("%", 1, 0), "division_by_zero")):
            p = compile_program({"version": 1, "body": [assign("m0", 1), assign("m1", expr), emit("rest")]})
            d = p.run(obs)
            self.assertEqual(d.fault, fault)
            self.assertEqual(d.memory, (0,) * 16)
        self.assertEqual(compile_program({"version": 1, "body": [assign("v_x", 1)]}).run(obs).fault, "no_action")
        invalid = compile_program({"version": 1, "body": [emit("move", dx=1, dy=1)]})
        self.assertEqual(invalid.run(obs).fault, "invalid_move")

    def test_compiler_snapshot_and_observation_are_not_mutated(self):
        source = idle_program()
        p = compile_program(source)
        source["body"][0]["emit"]["action"] = "harvest"
        obs = RobotArena(4).observation(0)
        original = copy.deepcopy(obs)
        self.assertEqual(p.run(obs).action, {"action": "rest"})
        self.assertEqual(original, obs)
        self.assertEqual(p.digest, compile_program(json.loads(p.source)).digest)


class RobotCompetitionTests(unittest.TestCase):
    def test_symmetric_resources_observation_boundary_and_atomic_validation(self):
        env = RobotArena(42)
        sites = {(s["x"], s["y"], s["units"], s["value"]) for s in env.sites}
        self.assertEqual(sites, {(6-x, y, units, value) for x, y, units, value in sites})
        observation = env.observation(0)
        self.assertNotIn("seed", observation)
        self.assertNotIn("priority", observation)
        env.priority.reverse()
        self.assertEqual(observation, env.observation(0))
        before = env.snapshot()
        with self.assertRaises(ValueError):
            env.step({0: {"action": "rest"}, 1: {"action": "move", "dx": True, "dy": 0}})
        self.assertEqual(before, env.snapshot())
        with self.assertRaises(ValueError):
            env.step({0: {"action": "rest"}, 1: {"action": "deposit", "score": 999}})

    def test_contested_last_unit_priority_and_value_conservation(self):
        env = RobotArena(9)
        site = env.sites[0]
        # Place all but one unit into the bank, preserving fixture conservation.
        removed = site["units"] - 1
        env.robots[0]["banked_units"] += removed
        env.robots[0]["score"] += removed * site["value"]
        site["units"] = 1
        for r in env.robots:
            r.update(x=site["x"], y=site["y"])
        first, second = env.priority[0]
        env.step({0: {"action": "harvest"}, 1: {"action": "harvest"}})
        self.assertEqual(len(env.robots[first]["cargo"]), 1)
        self.assertEqual(len(env.robots[second]["cargo"]), 0)
        self.assertEqual(env.history[-1]["outcomes"][second], "no_resource")
        env._assert_invariants()

    def test_random_action_sequences_conserve_resources(self):
        actions = [{"action": "rest"}, {"action": "harvest"}, {"action": "deposit"}]
        actions += [{"action": "move", "dx": dx, "dy": dy} for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        for seed in range(30):
            env = RobotArena(seed)
            chooser = random.Random(seed + 1000)
            while not env.done:
                env.step({p: chooser.choice(actions) for p in (0, 1)})
                env._assert_invariants()
            self.assertEqual(sum(env.returns()), 0)
            self.assertTrue(all(-1 <= r <= 1 for r in env.returns()))

    def test_real_algorithms_score_differently_and_replay_exactly(self):
        programs = reference_programs()
        nearest = play([programs["nearest"], programs["idle"]], 1, origins=["scripted", "scripted"])
        self.assertGreater(nearest["metrics"]["scores"][0], 0)
        self.assertEqual(nearest["metrics"]["scores"][1], 0)
        self.assertEqual(nearest["metrics"]["program_faults"], [0, 0])
        self.assertFalse(nearest["model_generated"])
        self.assertFalse(nearest["gradient_training"])
        self.assertTrue(verify_replay(json.loads(json.dumps(nearest))))
        tampered = copy.deepcopy(nearest)
        tampered["events"][0]["decisions"][0]["memory"][0] += 1
        with self.assertRaises(ValueError):
            verify_replay(tampered)
        nonzero = 0
        for seed in range(12):
            match = play([programs["nearest"], programs["value"]], seed)
            self.assertEqual(match["metrics"]["program_faults"], [0, 0])
            nonzero += match["returns"][0] != 0
        # Algorithm choices affect results; symmetric maps may legitimately tie.
        self.assertGreater(nonzero, 0)


if __name__ == "__main__":
    unittest.main()
