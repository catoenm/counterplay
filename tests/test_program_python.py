import json
import unittest

from selfplay.program_python import compile_python_program, PythonProgramValidationError, FRONTEND_VERSION
from selfplay.program_arena import compile_program, RobotArena, play, verify_replay
from selfplay.program_arena.examples import var, op, assign, emit, branch


class PythonFrontendTests(unittest.TestCase):
    def test_parser_cap_amendment_preserves_ir_cap(self):
        # Python ASTs count syntactic load/store/operator nodes absent from the
        # bounded IR. A simple program can exceed 512 parser nodes and still
        # comfortably fit the original IR budget.
        source='def act():\n'+'    m0 = 0\n'*150+"    return {'action':'rest'}\n"
        program=compile_python_program(source)
        self.assertEqual(FRONTEND_VERSION,2)
        self.assertEqual(program.to_dict()['frontend_version'],2)
        self.assertLess(program.compiled.node_count,512)
        self.assertIsNone(program.run(RobotArena(1).observation(0)).fault)
        excessive_parser='def act():\n    '+';'.join(['pass']*2046)
        with self.assertRaisesRegex(PythonProgramValidationError,'Python AST exceeds 2048'):
            compile_python_program(excessive_parser)
        excessive_ir='def act():\n'+'    m0 = 0\n'*257+"    return {'action':'rest'}\n"
        with self.assertRaisesRegex(PythonProgramValidationError,'lowered program exceeds'):
            compile_python_program(excessive_ir)

    def test_equivalence_to_handwritten_ir_and_source_preservation(self):
        source = '''def act():
    m0 += 1
    total = 0
    for i in range(6):
        total += site(i, 'units') * site(i, 'value')
    m1 = total
    if energy == 0:
        return {'action': 'rest'}
    if self_x < home_x:
        return {'action':'move','dx':1,'dy':0}
    elif self_x > home_x:
        return {'action':'move','dx':-1,'dy':0}
    return {'action':'deposit'}
'''
        reference = {'version': 1, 'body': [assign('m0', op('+', var('m0'), 1)), assign('v_total', 0),
          {'repeat':6, 'index':'v_i', 'body':[assign('v_total', op('+', var('v_total'),
             op('*', {'site':{'index':var('v_i'),'field':'units'}}, {'site':{'index':var('v_i'),'field':'value'}})))]},
          assign('m1', var('v_total')), branch(op('==',var('energy'),0),[emit('rest')]),
          branch(op('<',var('self_x'),var('home_x')),[emit('move',dx=1,dy=0)],
                 [branch(op('>',var('self_x'),var('home_x')),[emit('move',dx=-1,dy=0)])]),emit('deposit')]}
        candidate, expected = compile_python_program(source), compile_program(reference)
        for seed in range(10):
            for player in (0,1):
                obs = RobotArena(seed).observation(player)
                for energy in (0,4,10):
                    for x in (0,3,6):
                        obs.update(energy=energy,self_x=x)
                        a,b=candidate.run(obs),expected.run(obs)
                        self.assertEqual((a.action,a.memory,a.fault),(b.action,b.memory,b.fault))
        preserved=candidate.to_dict()
        self.assertEqual(preserved['python_source'],source)
        self.assertEqual(compile_program(preserved['compiled_ir']).digest,candidate.compiled.digest)
        self.assertEqual(json.loads(candidate.compiled.source),preserved['compiled_ir'])

    def test_short_circuit_boolean_operand_values_and_comparisons(self):
        source='''def act():
    m0 = 0 and 1 // 0
    m1 = 7 or 1 // 0
    m2 = 5 and 9
    m3 = 0 or 12
    m4 = 2 < 1 < 1 // 0
    m5 = 1 < 2 < 3
    m6 = 8 if energy > 0 else 1 // 0
    m7 = abs(-3) + min(2, 9) + max(1, 4)
    return {'action':'rest'}
'''
        decision=compile_python_program(source).run(RobotArena(3).observation(0))
        self.assertIsNone(decision.fault)
        self.assertEqual(decision.memory[:8],(0,7,9,12,0,1,8,9))

    def test_reject_escapes_dynamic_execution_and_unbounded_constructs(self):
        statements=["import os", "return __import__('os').system('true')", "x = (1).__class__",
                    "x = self_x.__class__", "x = globals()", "return act()", "while 1:\n        pass",
                    "x = [i for i in range(6)]", "x = lambda: 1", "x = site(0, '__class__')",
                    "x = site(0, 'x')[0]", "x = 2 ** 999999", "x = 3 / 2", "x = True",
                    "energy = 999", "global m0", "for i in range(1000000):\n        pass",
                    "for i in range(site_count):\n        pass", "x = min(1, y=2)", "x = open('/tmp/escape')",
                    "return {'action':'rest','action':'harvest'}", "for i in range(6):\n        break"]
        for statement in statements:
            with self.subTest(statement=statement), self.assertRaises(PythonProgramValidationError):
                compile_python_program('def act():\n    '+statement+"\n    return {'action':'rest'}\n")
        for source in ("import os\ndef act():\n    pass", "@something\ndef act():\n    pass",
                       "def act(x):\n    pass", "def act():\n    pass\ndef helper():\n    pass",
                       "def act():\n    def helper():\n        pass\n    return {'action':'rest'}"):
            with self.assertRaises(PythonProgramValidationError):
                compile_python_program(source)

    def test_runtime_limit_memory_rollback_and_missing_return(self):
        source='''def act():
    m0 = 17
    for i in range(16):
        for j in range(16):
            for k in range(16):
                m0 += 1
    return {'action':'rest'}
'''
        decision=compile_python_program(source).run(RobotArena(1).observation(0))
        self.assertEqual(decision.fault,'step_limit')
        self.assertEqual(decision.memory,(0,)*16)
        self.assertEqual(compile_python_program('def act():\n    pass').run(RobotArena(1).observation(0)).fault,'no_action')
        with self.assertRaises(PythonProgramValidationError):
            compile_python_program(' ' * 16385)
        many='def act():\n'+''.join(f'    variable{i} = {i}\n' for i in range(33))+"    return {'action':'rest'}\n"
        with self.assertRaises(PythonProgramValidationError):
            compile_python_program(many)

    def test_compiled_program_competition_replays_without_python_execution(self):
        idle=compile_python_program("def act():\n    return {'action':'rest'}\n")
        patrol=compile_python_program('''def act():
    m0 += 1
    if self_x < 3:
        return {'action':'move','dx':1,'dy':0}
    return {'action':'rest'}
''')
        result=play([idle.compiled,patrol.compiled],seed=22,origins=['scripted','scripted'])
        self.assertTrue(verify_replay(result))
        self.assertEqual(result['metrics']['program_faults'],[0,0])


if __name__=='__main__':
    unittest.main()
