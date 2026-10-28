"""CPU-only hand-written fixtures; these are never presented as model output."""
import copy
import hashlib
import unittest
from unittest.mock import patch

from selfplay.program_arena import verify_replay
from selfplay.program_cardinal import classify_cardinal, compile_cardinal_program
from selfplay.program_debug import (
    MODEL_CALLS, REVISION_CONTEXT_POLICY, TRACE_TURNS, assert_map_separation,
    classify_debug, debug_model_spec, initial_messages, practice_feedback, revision_messages,
)


SEEDS = [152000000, 152000001]
IDLE = "def act():\n    return 'rest'\n"


def record(source=IDLE, *, prefix='I will inspect the rules.\n</think>\n', finish='eos'):
    return {'text': prefix + source, 'finish_reason': finish, 'completion_ids': [7, 8, 9],
            'sampled_logps': [-0.1, -0.2, -0.3], 'fixture_only': True}


class DebugExtractionTests(unittest.TestCase):
    def test_final_boundary_with_native_or_explicit_opening_preserves_exact_source(self):
        for prefix in ('reasoning\n</think>', '<think>reasoning\n</think>', '  <think>reasoning\n</think>'):
            raw = prefix + '\n```python\n' + IDLE + '```\n'
            result, program = classify_debug(raw, 'eos')
            self.assertTrue(result['valid'])
            self.assertEqual(program.digest, compile_cardinal_program(IDLE).compiled.digest)
            self.assertEqual(result['program']['python_source'], IDLE)
            self.assertEqual(result['original_response_sha256'], hashlib.sha256(raw.encode()).hexdigest())
            extract = result['debug_extraction']
            self.assertEqual(result['final_answer'], raw[extract['final_start']:])
            self.assertEqual(extract['reasoning_sha256'], hashlib.sha256(raw[extract['reasoning_start']:extract['reasoning_end']].encode()).hexdigest())
            self.assertEqual(result['cardinal_classification'], classify_cardinal(result['final_answer'], 'eos')[0])

    def test_never_executes_or_recovers_code_from_reasoning(self):
        raw = '<think>\n' + IDLE + '</think>\nThis answer is not Python.'
        result, program = classify_debug(raw, 'eos')
        self.assertFalse(result['valid'])
        self.assertIsNone(program)
        for text in (IDLE, '<think>' + IDLE, '</think>' + IDLE + '</think>',
                     '<think><think>thought</think>' + IDLE,
                     'prose <think>thought</think>' + IDLE,
                     'thought</think><think>' + IDLE):
            with self.subTest(text=text):
                result, program = classify_debug(text, 'eos')
                self.assertFalse(result['valid'])
                self.assertIsNone(program)

    def test_truncation_and_ambiguous_fences_are_not_repaired(self):
        result, program = classify_debug(record()['text'], 'length')
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'generation_length')
        self.assertIsNone(program)
        result, program = classify_debug('</think>\n```python\n'+IDLE+'```\n```python\n'+IDLE+'```', 'eos')
        self.assertFalse(result['valid'])
        self.assertIsNone(program)
        self.assertEqual(classify_debug(None, 'eos')[0]['reason'], 'invalid_response_type')

    def test_prompt_explains_lifecycle_and_two_calls_without_worked_controller(self):
        text = debug_model_spec()
        for required in ('AT ITS OWN HOME', 'empty cargo', 'exactly once', 'ONE action',
                         'Locals reset', 'two model calls', 'second', 'read-only', '40 lines', 'concise'):
            self.assertIn(required, text)
        self.assertNotIn("return 'rest'", text)
        self.assertEqual(MODEL_CALLS, 2)
        self.assertEqual(initial_messages(), [{'role': 'user', 'content': text}])


class DebugPracticeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rest = practice_feedback(record(), SEEDS)

    def test_practice_is_deterministic_eight_real_games_with_candidate_only_totals(self):
        evidence = self.rest
        self.assertEqual(evidence, practice_feedback(record(), SEEDS))
        self.assertEqual(evidence['record'], record())
        self.assertEqual(evidence['summary']['games_played'], 8)
        self.assertEqual(evidence['summary']['banked_point_yield'], 0)
        self.assertEqual(evidence['summary']['candidate_totals']['rests'], 8*48)
        self.assertEqual(evidence['summary']['candidate_program_faults'], 0)
        self.assertEqual(evidence['summary']['positive_banking_games'], 0)
        self.assertTrue(evidence['summary']['runtime_valid'])
        self.assertFalse(evidence['generated_python_executed'])
        self.assertEqual([(m['opponent_id'], m['replay']['seed'], m['candidate_seat']) for m in evidence['matches']],
                         [(op, seed, seat) for op in ('nearest', 'value') for seed in SEEDS for seat in (0, 1)])
        for match in evidence['matches']:
            self.assertTrue(verify_replay(match['replay']))
        traces = evidence['feedback']['traces']
        self.assertEqual(len(traces), 2)
        self.assertEqual([t['seat'] for t in traces], [0, 1])
        for trace in traces:
            self.assertEqual(trace['seed'], SEEDS[0])
            self.assertEqual(trace['opponent'], 'nearest')
            self.assertEqual([r[0] for r in trace['rows']], list(TRACE_TURNS))
        self.assertLess(len(evidence['feedback_text']), 12000)

    def test_compiler_error_and_no_final_have_no_fabricated_games(self):
        for draft in (record('def act():\n    score = 9\n    return \'rest\'\n'), record(prefix='')):
            evidence = practice_feedback(draft, SEEDS)
            self.assertFalse(evidence['classification']['valid'])
            self.assertEqual(evidence['matches'], [])
            self.assertEqual(evidence['summary']['games_played'], 0)
            self.assertIsNone(evidence['summary']['game_return'])
            self.assertEqual(evidence['summary']['banked_point_yield'], 0)
            self.assertEqual(evidence['feedback']['traces'], [])
            self.assertIn('reason', evidence['feedback']['classification'])

    def test_ordinary_failed_actions_are_not_interpreter_faults(self):
        evidence = practice_feedback(record("def act():\n    return 'west'\n"), SEEDS)
        summary = evidence['summary']
        self.assertTrue(summary['runtime_valid'])
        self.assertEqual(summary['candidate_program_faults'], 0)
        self.assertGreater(summary['candidate_totals']['failed_actions'], 0)
        self.assertGreater(summary['candidate_outcomes']['outside_grid'], 0)

    def test_runtime_faults_are_attributed_to_candidate_not_opponent(self):
        faulty = compile_cardinal_program("def act():\n    x = 1 // 0\n    return 'rest'\n").compiled
        with patch('selfplay.program_debug.reference_programs', return_value={'nearest': faulty, 'value': faulty}):
            evidence = practice_feedback(record(), SEEDS)
        self.assertEqual(evidence['summary']['opponent_program_faults'], 8*48)
        self.assertEqual(evidence['summary']['candidate_program_faults'], 0)
        self.assertTrue(evidence['summary']['runtime_valid'])
        bad = practice_feedback(record("def act():\n    x = 1 // 0\n    return 'rest'\n"), SEEDS)
        self.assertEqual(bad['summary']['candidate_program_faults'], 8*48)
        self.assertFalse(bad['summary']['runtime_valid'])
        self.assertTrue(bad['summary']['candidate_fault_reasons'])

    def test_revision_uses_exact_final_source_and_retains_raw_reasoning_in_evidence(self):
        draft = record('```python\n'+IDLE+'```\n')
        evidence = practice_feedback(draft, SEEDS)
        messages = revision_messages(initial_messages(), draft, evidence)
        self.assertEqual([m['role'] for m in messages], ['user', 'assistant', 'user'])
        self.assertEqual(messages[1]['content'], IDLE)
        self.assertNotIn('I will inspect', messages[1]['content'])
        self.assertIn('I will inspect', evidence['record']['text'])
        self.assertEqual(messages[2]['content'], evidence['feedback_text'])
        self.assertEqual(evidence['revision_context']['policy'], REVISION_CONTEXT_POLICY)
        self.assertEqual(evidence['final_attempt_index'], 1)
        for draft in (record('not code'), record(prefix='')):
            evidence = practice_feedback(draft, SEEDS)
            messages = revision_messages(initial_messages(), draft, evidence)
            expected = evidence['classification'].get('final_answer')
            self.assertEqual(messages[1]['content'], draft['text'] if expected is None else expected)

    def test_revision_rejects_mismatched_record_feedback_prompt_and_prefix(self):
        for mutate in (
            lambda e: e['record'].update(text='edited'),
            lambda e: e.update(feedback_text=e['feedback_text']+'edited'),
            lambda e: e['classification'].update(valid=False),
            lambda e: e['revision_context'].update(assistant_content_sha256='wrong'),
        ):
            evidence = copy.deepcopy(self.rest)
            mutate(evidence)
            with self.assertRaises(ValueError):
                revision_messages(initial_messages(), record(), evidence)
        with self.assertRaises(ValueError):
            revision_messages([{'role': 'user', 'content': 'Different rules'}], record(), self.rest)

    def test_seed_and_observable_layout_overlap_guard(self):
        proof = assert_map_separation(SEEDS, [152000002])
        self.assertEqual(len(proof['practice']), 2)
        self.assertNotIn('layout', proof['reserved'][0])
        for seeds in ([True, 2], [1], [1, 1], [1, 2, 3]):
            with self.assertRaises(ValueError):
                practice_feedback(record(), seeds)
        with self.assertRaises(ValueError):
            assert_map_separation(SEEDS, [SEEDS[0]])
        # Distinct integer seeds can still describe identical observable maps.
        real = proof['practice'][0]
        def collision(seed, max_rounds=48):
            return {'seed': seed, 'sha256': real['sha256'], 'layout': real['layout']}
        with patch('selfplay.program_debug._map', side_effect=collision):
            with self.assertRaisesRegex(ValueError, 'identical observable'):
                assert_map_separation(SEEDS, [])
        with self.assertRaises(ValueError):
            practice_feedback(record(), SEEDS, max_rounds=24)


if __name__ == '__main__':
    unittest.main()
