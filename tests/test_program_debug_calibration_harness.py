"""CPU-only fixed-harness tests; synthetic text and bounded game fixtures."""
from copy import deepcopy
import unittest

from selfplay import program_debug_calibration_harness as harness
from selfplay import program_debug as previous

IDLE = "def act():\n    return 'rest'\n"


def record(text='fixture analysis\n</think>\n'+IDLE, finish='eos'):
    return {'text': text, 'raw_text': text+'<eos>', 'finish_reason': finish,
            'completion_ids': [1, 2], 'sampled_token_logprobs': [-.3, -.5],
            'output_tokens': 2, 'completion_token_limit': 4096, 'reasoning_effort': 'low'}


class CalibrationHarnessTests(unittest.TestCase):
    def test_clean_prompt_retains_lifecycle_rules_without_internal_delimiter_instructions(self):
        prompt = harness.initial_messages()[0]['content']
        for tag in ('<think>', '</think>'):
            self.assertNotIn(tag, prompt)
            self.assertNotIn(tag, harness.CHECKLIST)
        self.assertIn('AT ITS OWN HOME, empty cargo', prompt)
        self.assertIn('exactly two model calls', prompt)
        self.assertIn('read-only', prompt)
        self.assertEqual(prompt.split('LIFECYCLE:', 1)[1], previous.debug_model_spec().split('LIFECYCLE:', 1)[1])

    def test_full_record_and_fixed_practice_preserved_without_repair(self):
        draft = record(); before = deepcopy(draft)
        evidence = harness.practice_feedback(draft, [171000000, 171000001])
        self.assertEqual(draft, before)
        self.assertEqual(evidence['record'], before)
        self.assertEqual(len(evidence['matches']), 8)
        self.assertEqual(evidence['summary']['games_played'], 8)
        self.assertTrue(evidence['summary']['runtime_valid'])
        self.assertEqual(evidence['summary']['banked_point_yield'], 0.)
        messages = harness.revision_messages(harness.initial_messages(), draft, evidence)
        self.assertEqual([m['role'] for m in messages], ['user', 'assistant', 'user'])
        self.assertEqual(messages[1]['content'], '\n'+IDLE)
        self.assertNotIn('fixture analysis', messages[1]['content'])
        self.assertEqual(evidence['feedback']['delivery']['reasoning_effort_requested'], 'low')
        self.assertFalse(evidence['feedback']['delivery']['token_limit_reached'])
        self.assertEqual(evidence['final_attempt_index'], 1)

    def test_truncated_draft_has_zero_games_and_exact_incomplete_history(self):
        draft = record('unfinished draft thoughts', 'length')
        draft['output_tokens'] = 4096
        evidence = harness.practice_feedback(draft, [171000010, 171000011])
        self.assertFalse(evidence['classification']['valid'])
        self.assertEqual(evidence['classification']['reason'], 'missing_thinking_boundary')
        self.assertEqual(evidence['matches'], [])
        self.assertTrue(evidence['feedback']['delivery']['token_limit_reached'])
        self.assertEqual(evidence['feedback']['delivery']['output_tokens'], 4096)
        messages = harness.revision_messages(harness.initial_messages(), draft, evidence)
        self.assertEqual(messages[1]['content'], draft['text'])
        # No guessed final code, boundary insertion, or third call is introduced.
        self.assertEqual(harness.MODEL_CALLS, 2)
        self.assertEqual(len(messages), 3)

    def test_ambiguous_final_and_tampered_feedback_rejected(self):
        draft = record('first</think>second</think>'+IDLE)
        evidence = harness.practice_feedback(draft, [171000020, 171000021])
        self.assertEqual(evidence['classification']['reason'], 'ambiguous_thinking_boundary')
        self.assertEqual(evidence['matches'], [])
        changed = deepcopy(evidence)
        changed['feedback']['delivery']['output_tokens'] = 99
        changed['feedback_text'] = harness._feedback_text(changed['feedback'])
        changed['feedback_sha256'] = previous._sha(changed['feedback_text'])
        with self.assertRaisesRegex(ValueError, 'Delivery feedback'):
            harness.revision_messages(harness.initial_messages(), draft, changed)
        changed = deepcopy(evidence); changed['record']['text'] += ' repaired'
        with self.assertRaisesRegex(ValueError, 'exact draft'):
            harness.revision_messages(harness.initial_messages(), draft, changed)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            harness.assert_map_separation([171000000, 171000001], [171000000])


if __name__ == '__main__':
    unittest.main()
