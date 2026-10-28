"""CPU fixtures for a syntax-only interface; these are not model trajectories."""
import copy
import hashlib
import json
import unittest

from selfplay.program_arena import RobotArena, compile_program, play, verify_replay
from selfplay.program_cardinal import (
    ACTION_NAMES, CARDINAL_FRONTEND_VERSION, CardinalProgramValidationError,
    cardinal_model_spec, classify_cardinal, compile_cardinal_program,
)
from selfplay.program_python import compile_python_program


ACTIONS = {
    'north': {'action': 'move', 'dx': 0, 'dy': -1},
    'south': {'action': 'move', 'dx': 0, 'dy': 1},
    'east': {'action': 'move', 'dx': 1, 'dy': 0},
    'west': {'action': 'move', 'dx': -1, 'dy': 0},
    'rest': {'action': 'rest'}, 'harvest': {'action': 'harvest'}, 'deposit': {'action': 'deposit'},
}


def literal(name):
    return compile_cardinal_program(f'def act():\n    return {name!r}\n')


class CardinalFrontendTests(unittest.TestCase):
    def test_all_seven_actions_lower_to_exact_legacy_ir_and_preserve_source(self):
        self.assertEqual(set(ACTION_NAMES), set(ACTIONS))
        for name, action in ACTIONS.items():
            with self.subTest(name=name):
                source = f'# retained comment\ndef act():\n    return {name!r}  # exact original\n'
                p = compile_cardinal_program(source)
                legacy = compile_python_program(f'def act():\n    return {action!r}\n')
                self.assertEqual(p.compiled.digest, legacy.compiled.digest)
                self.assertEqual(p.run(RobotArena(11).observation(0)), legacy.run(RobotArena(11).observation(0)))
                self.assertEqual(p.run(RobotArena(11).observation(0)).action, action)
                saved = p.to_dict()
                self.assertEqual(saved['frontend_version'], CARDINAL_FRONTEND_VERSION)
                self.assertEqual(saved['python_source'], source)
                self.assertEqual(saved['python_sha256'], hashlib.sha256(source.encode()).hexdigest())
                self.assertEqual(saved['lowered_python_sha256'], hashlib.sha256(saved['lowered_python_source'].encode()).hexdigest())
                self.assertEqual(compile_python_program(saved['lowered_python_source']).compiled.digest, saved['ir_sha256'])
                self.assertEqual(compile_program(saved['compiled_ir']).digest, saved['ir_sha256'])
                self.assertEqual(saved['compiled_ir'], json.loads(p.compiled.source))

    def compare_transition(self, env, name, expected):
        env._assert_invariants()
        reference = copy.deepcopy(env)
        decision = literal(name).run(env.observation(0))
        self.assertIsNone(decision.fault)
        env.step({0: decision.action, 1: {'action': 'rest'}})
        reference.step({0: ACTIONS[name], 1: {'action': 'rest'}})
        self.assertEqual(env.snapshot(), reference.snapshot())
        self.assertEqual(env.history[-1]['outcomes'][0], expected)
        self.assertEqual(env.turn, 1)

    def test_boundary_and_zero_energy_actions_fail_without_repair_or_runtime_fault(self):
        for name, x, y in [('north', 3, 0), ('south', 3, 6), ('east', 6, 3), ('west', 0, 3)]:
            with self.subTest(name=name):
                env = RobotArena(13)
                env.robots[0].update(x=x, y=y)
                self.compare_transition(env, name, 'outside_grid')
                self.assertEqual(env.robots[0]['energy'], 10)
        for name in ('north', 'south', 'east', 'west', 'harvest'):
            with self.subTest(no_energy=name):
                env = RobotArena(13)
                env.robots[0].update(x=3, y=3, energy=0)
                self.compare_transition(env, name, 'no_energy')
                self.assertEqual(env.robots[0]['energy'], 0)
        env = RobotArena(13)
        env.robots[0]['energy'] = 0
        self.compare_transition(env, 'rest', 'rested')
        self.assertEqual(env.robots[0]['energy'], 3)

    @staticmethod
    def cargo(env, units):
        for site in env.sites:
            while units and site['units']:
                site['units'] -= 1
                env.robots[0]['cargo'].append(site['value'])
                env.robots[0]['harvested_units'] += 1
                units -= 1
        assert units == 0

    def test_cargo_harvest_and_deposit_keep_underlying_rules(self):
        env = RobotArena(14)
        self.cargo(env, 3)
        site = next(s for s in env.sites if s['units'])
        env.robots[0].update(x=site['x'], y=site['y'])
        self.compare_transition(env, 'harvest', 'cargo_full')
        env = RobotArena(14)
        self.cargo(env, 2)
        value = sum(env.robots[0]['cargo'])
        env.robots[0]['energy'] = 0
        self.compare_transition(env, 'deposit', 'deposited')
        self.assertEqual(env.robots[0]['score'], value)
        self.assertEqual(env.robots[0]['cargo'], [])
        self.assertEqual(env.robots[0]['energy'], 0)
        env = RobotArena(14)
        self.cargo(env, 1)
        env.robots[0]['x'] = 1
        self.compare_transition(env, 'deposit', 'not_at_home')
        self.assertEqual(len(env.robots[0]['cargo']), 1)
        env = RobotArena(14)
        self.compare_transition(env, 'harvest', 'no_resource')
        env = RobotArena(14)
        site = env.sites[0]
        env.robots[0].update(x=site['x'], y=site['y'])
        self.compare_transition(env, 'harvest', 'harvested')
        self.assertEqual(len(env.robots[0]['cargo']), 1)
        self.assertEqual(env.robots[0]['energy'], 9)

    def test_diagonal_targets_require_controller_choice_and_preserve_memory(self):
        source = '''def act():
    m0 += 1
    if self_x < home_x:
        return 'east'
    if self_x > home_x:
        return 'west'
    if self_y < home_y:
        return 'south'
    if self_y > home_y:
        return 'north'
    return 'deposit'
'''
        program = compile_cardinal_program(source)
        legacy = compile_python_program(program.lowered_python_source)
        cases = [(0, 0, 6, 3, 'east'), (6, 6, 0, 3, 'west'),
                 (0, 0, 0, 3, 'south'), (6, 6, 6, 3, 'north'), (0, 3, 0, 3, 'deposit')]
        for x, y, hx, hy, name in cases:
            obs = RobotArena(15).observation(0)
            obs.update(self_x=x, self_y=y, home_x=hx, home_y=hy)
            got = program.run(obs, [9] + [0] * 15)
            self.assertEqual(got, legacy.run(obs, [9] + [0] * 15))
            self.assertEqual(got.action, ACTIONS[name])
            self.assertEqual(got.memory[0], 10)

    def test_unsafe_dynamic_and_nonliteral_returns_are_rejected(self):
        returns = ["'move'", "'northeast'", "'North'", "{'action':'north'}", "{'action':'rest'}",
                   "{'action':'move','dx':1,'dy':1}", "name", "'north' if energy else 'rest'",
                   "'n' + 'orth'", "str(1)", "True", "None", "1", "f'north'", ""]
        for value in returns:
            with self.subTest(value=value), self.assertRaises(CardinalProgramValidationError):
                compile_cardinal_program('def act():\n    return ' + value + '\n')
        statements = ['import os', 'energy = 4', 'x = [1]', 'x = self_x.real',
                      'while 1:\n        pass', 'x = open("ignored")',
                      'for i in range(6):\n        break', "name = 'north'"]
        for statement in statements:
            with self.subTest(statement=statement), self.assertRaises(CardinalProgramValidationError):
                compile_cardinal_program('def act():\n    ' + statement + "\n    return 'rest'\n")
        for source in ("def act(x):\n    return 'rest'", "def helper():\n    return 'rest'",
                       "@decorator\ndef act():\n    return 'rest'", "def act():\n    return 'rest'\ndef helper():\n    pass"):
            with self.assertRaises(CardinalProgramValidationError):
                compile_cardinal_program(source)

    def test_original_and_legacy_limits_and_runtime_rollback(self):
        for source in (' ' * 16385, 'def act():\n    ' + ';'.join(['pass'] * 2046),
                       'def act():\n' + '    m0 = 0\n' * 257 + "    return 'rest'\n"):
            with self.assertRaises(CardinalProgramValidationError):
                compile_cardinal_program(source)
        source = "def act():\n    m0 = 17\n    x = 1 // 0\n    return 'north'\n"
        result = compile_cardinal_program(source).run(RobotArena(16).observation(0), [2] * 16)
        self.assertEqual(result.action, {'action': 'rest'})
        self.assertIsNotNone(result.fault)
        self.assertEqual(result.memory, (2,) * 16)
        self.assertEqual(compile_cardinal_program('def act():\n    pass').run(RobotArena(16).observation(0)).fault, 'no_action')
        source = '''def act():
    m0 = 17
    for i in range(16):
        for j in range(16):
            for k in range(16):
                m0 += 1
    return 'east'
'''
        result = compile_cardinal_program(source).run(RobotArena(16).observation(0))
        self.assertEqual(result.fault, 'step_limit')
        self.assertEqual(result.memory, (0,) * 16)

    def test_classification_preserves_raw_response_and_only_removes_single_fence(self):
        source = "def act():\n    return 'east'\n"
        text = 'Controller:\n```python\n' + source + '```\nEnd.'
        row, inner = classify_cardinal(text, 'eos')
        self.assertTrue(row['valid'])
        self.assertEqual(row['normalization'], 'extracted_single_python_fence_with_prose')
        self.assertEqual(row['program']['python_source'], source)
        self.assertEqual(row['original_response_sha256'], hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(inner.digest, row['program']['ir_sha256'])
        for broken in (text + '\n```python\npass\n```', '```py\n' + source + '```',
                       'Prefix ' + source, source + '\nnot Python prose', '<think>x</think>' + source):
            row, p = classify_cardinal(broken, 'eos')
            self.assertFalse(row['valid'])
            self.assertIsNone(p)
        row, p = classify_cardinal(source, 'length')
        self.assertEqual(row['reason'], 'generation_length')
        self.assertIsNone(p)

    def test_complete_replay_equivalence_with_scripted_fixture_only(self):
        source = '''def act():
    m0 += 1
    if m0 % 3 == 0:
        return 'rest'
    if self_x < 3:
        return 'east'
    return 'harvest'
'''
        cardinal = compile_cardinal_program(source)
        legacy = compile_python_program(cardinal.lowered_python_source)
        idle = literal('rest')
        for seed in (21, 22):
            got = play([cardinal.compiled, idle.compiled], seed, origins=['scripted', 'scripted'])
            expected = play([legacy.compiled, idle.compiled], seed, origins=['scripted', 'scripted'])
            self.assertEqual(got, expected)
            self.assertTrue(verify_replay(got))
            self.assertEqual(got['metrics']['program_faults'], [0, 0])

    def test_spec_is_complete_without_a_strategy_or_rest_controller_example(self):
        spec = cardinal_model_spec()
        for name in ACTION_NAMES:
            self.assertIn(repr(name), spec)
        self.assertIn('North changes y by -1; south y by +1', spec)
        self.assertIn('<=512 nodes', spec)
        self.assertNotIn("return 'rest'", spec)
        self.assertNotIn('Minimal', spec)


if __name__ == '__main__':
    unittest.main()
