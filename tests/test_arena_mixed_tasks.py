import ast
from copy import deepcopy
from dataclasses import replace
import json
import random
import unittest

from selfplay.arena_mixed.planning import PlanningEnv, task_family_payload
from selfplay.arena_mixed.coding import (CodingEnv, ProgramError, apply_edit, edit_sites,
    evaluate, parse_program, reference, run_program, task_source)


class PlanningTests(unittest.TestCase):
    def test_verified_tasks_and_real_transitions(self):
        for seed in range(16):
            env=PlanningEnv(seed)
            self.assertEqual(len(env._task.shortest_plan()),3+seed%4)
            for action in env._task.witness:
                if env.done:break
                env.step({0:{'tool':action}})
            self.assertTrue(env.done)
            self.assertEqual(env.returns(),[1.,0.])
            self.assertEqual(env.state,env._task.goal)
            self.assertEqual(env.active_players(),[])
            with self.assertRaises(RuntimeError):env.step({0:{'tool':'swap'}})

    def test_observation_omits_witness_and_solver(self):
        env=PlanningEnv(33)
        self.assertTrue(env.assert_private_boundary()['checked'])
        public=json.dumps(env.observation(0))
        self.assertNotIn('witness',public)
        self.assertNotIn('shortest',public)
        self.assertIn('audit_only',env.snapshot())
        with self.assertRaises(ValueError):env.observation(1)
        with self.assertRaises(RuntimeError):env.returns()

    def test_family_is_color_swap_and_budget_invariant(self):
        env=PlanningEnv(27);task=env._task
        dual={'inc_x':'inc_y','inc_y':'inc_x','sum_x':'sum_y','sum_y':'sum_x',
              'double_x':'double_y','double_y':'double_x','swap':'swap'}
        swapped=replace(task,start=tuple(reversed(task.start)),goal=tuple(reversed(task.goal)),
                        actions=tuple(dual[a] for a in task.actions),witness=tuple(dual[a] for a in task.witness))
        self.assertEqual(task_family_payload(task),task_family_payload(swapped))
        self.assertEqual(task_family_payload(task),task_family_payload(replace(task,max_steps=11)))

    def test_replay_and_strict_roles(self):
        a,b=PlanningEnv(19),PlanningEnv(19)
        while not a.done:
            action=a.heuristic_action(0);a.step({0:action});b.step({0:action})
            self.assertEqual(a.snapshot(),b.snapshot())
        self.assertEqual(a.returns(),[1.,0.])
        for bad in ({1:{'tool':'swap'}},{False:{'tool':'swap'}},{0:{'tool':'unknown'}}):
            env=PlanningEnv(19);before=deepcopy(env.snapshot())
            with self.assertRaises(ValueError):env.step(bad)
            self.assertEqual(before,env.snapshot())


class CodingTests(unittest.TestCase):
    def test_interpreter_against_independent_trusted_semantics(self):
        rng=random.Random(91)
        for family in ('sum_shift','count_at_least','maximum_with_floor'):
            for p in range(-3,4):
                source=task_source(family,p)
                for xs in ([],[-3,0,4],[p,p-1,p+1],*[ [rng.randint(-8,8) for _ in range(rng.randrange(9))] for _ in range(8)]):
                    self.assertEqual(run_program(source,xs),reference(family,p,xs))

    def test_disallowed_execution_and_bounds(self):
        for body in ('import os\nfor item in xs:\n    result = 0\nreturn result',
                     'result = 0\nfor item in xs:\n    result = item.real\nreturn result',
                     'result = 0\nfor item in xs:\n    result = len(xs)\nreturn result',
                     'result = 0\nfor item in xs:\n    for item in xs:\n        result = 0\nreturn result',
                     'result = 0\nfor item in xs:\n    result = item // 0\nreturn result'):
            with self.assertRaises(ProgramError):parse_program('def solve(xs):\n'+'\n'.join('    '+x for x in body.splitlines()))
        source=task_source('sum_shift',1)
        for xs in ([True],[1.1],[65],[0]*9):
            with self.assertRaises(ProgramError):run_program(source,xs)
        with self.assertRaises(ProgramError):run_program('def solve(xs):\n    result=2\n    for item in xs:\n        result=result*result\n    return result\n',[0]*8)

    def test_real_ast_edit_and_visible_slot_menu(self):
        source=task_source('count_at_least',2)
        slot=next(s['slot'] for s in edit_sites(source) if s['kind']=='comparison')
        changed=apply_edit(source,{'type':'edit','slot':slot,'value':'Lt'})
        self.assertNotEqual(ast.dump(parse_program(source)),ast.dump(parse_program(changed)))
        self.assertEqual(run_program(source,[1,2,3]),2)
        self.assertEqual(run_program(changed,[1,2,3]),1)
        with self.assertRaises(ProgramError):apply_edit(source,{'type':'edit','slot':slot,'value':'GtE'})
        with self.assertRaises(ProgramError):apply_edit(source,{'type':'edit','slot':999,'value':1})

    def test_hidden_tests_and_solution_never_affect_policy_view(self):
        for seed in range(9):
            env=CodingEnv(seed)
            self.assertTrue(env.assert_private_boundary()['checked'])
            view=json.dumps(env.observation(0))
            for key in ('hidden_tests','reference_solution','final_hidden_results'):
                self.assertNotIn(key,view)
            public={tuple(r['input']) for r in env.public_tests}
            hidden=[tuple(r['input']) for r in env._hidden_tests]
            self.assertFalse(public & set(hidden));self.assertEqual(len(hidden),len(set(hidden)))
            self.assertTrue(all(r['passed'] for r in evaluate(env._solution,env._hidden_tests)))
            self.assertFalse(all(r['passed'] for r in evaluate(env.program,env.public_tests)))
            self.assertLessEqual(len(env.legal_actions(0)),256)

    def test_public_feedback_then_hidden_scoring_and_cap(self):
        env=CodingEnv(4,max_rounds=3)
        env.step({0:{'type':'run_public_tests'}})
        self.assertEqual(env.public_feedback,evaluate(env.program,env.public_tests))
        self.assertIsNone(env.metrics()['hidden_pass_fraction'])
        env.program=env._solution  # An exact trusted solution fixture, never a learner repair.
        env.step({0:{'type':'submit'}})
        self.assertEqual(env.returns(),[1.,0.]);self.assertEqual(env.active_players(),[])
        self.assertEqual(env.metrics()['finish_reason'],'submitted')
        capped=CodingEnv(5,max_rounds=3)
        for _ in range(3):capped.step({0:{'type':'run_public_tests'}})
        self.assertTrue(capped.done);self.assertEqual(capped.finish_reason,'repair_step_cap')
        expected=sum(r['passed'] for r in evaluate(capped.program,capped._hidden_tests))/16
        self.assertEqual(capped.returns(),[expected,0.])

    def test_public_heuristic_ignores_hidden_cases_and_solution(self):
        env=CodingEnv(23);other=deepcopy(env)
        other._hidden_tests=[{'input':[9],'expected':100}];other._solution='withheld'
        self.assertEqual(env.heuristic_action(0),other.heuristic_action(0))

    def test_replay_and_rejected_action_no_mutation(self):
        a,b=CodingEnv(34),CodingEnv(34)
        while not a.done:
            action=a.heuristic_action(0);a.step({0:action});b.step({0:action})
            self.assertEqual(a.snapshot(),b.snapshot())
        for bad in ({False:{'type':'submit'}},{1:{'type':'submit'}},{0:{'type':'edit','slot':999,'value':1}}):
            env=CodingEnv(9);before=deepcopy(env.snapshot())
            with self.assertRaises(ValueError):env.step(bad)
            self.assertEqual(before,env.snapshot())

    def test_initial_identity_excludes_witness_and_is_not_current_program(self):
        env=CodingEnv(7);identity=deepcopy(env.fingerprint_payload())
        env._solution='not part of identity'
        env._hidden_tests=[{'input':[64],'expected':-999}]
        env.step({0:next(a for a in env.legal_actions(0) if a['type']=='edit')})
        self.assertEqual(identity,env.fingerprint_payload())
        self.assertNotIn('reference_solution',json.dumps(identity))
        self.assertNotIn('hidden_tests',json.dumps(identity))


if __name__=='__main__':unittest.main()
