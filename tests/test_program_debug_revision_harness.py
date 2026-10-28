"""CPU-only native-disabled final extraction; no generated Python execution."""
from copy import deepcopy
import unittest
from selfplay import program_debug_revision_harness as harness
from selfplay import program_debug_calibration_harness as previous

CODE="def act():\n    return 'rest'\n"

class RevisionHarnessTests(unittest.TestCase):
    def test_prompts_and_factual_practice_are_byte_identical(self):
        self.assertEqual(harness.initial_messages(),previous.initial_messages())
        self.assertEqual(harness.CHECKLIST,previous.CHECKLIST)
        draft={'text':'brief reasoning\n</think>\n'+CODE,'finish_reason':'eos',
               'authoring_stage':'draft','thinking_enabled':True,'reasoning_effort':'low',
               'reasoning_mode':'native_low','output_tokens':9,'completion_token_limit':4096}
        before=deepcopy(draft)
        actual=harness.practice_feedback(draft,[191000000,191000001])
        expected=previous.practice_feedback(draft,[191000000,191000001])
        self.assertEqual(actual,expected)
        self.assertEqual(actual['record'],before)
        self.assertEqual(len(actual['matches']),8)
        self.assertEqual(harness.revision_messages(harness.initial_messages(),draft,actual),
                         previous.revision_messages(previous.initial_messages(),draft,expected))
        self.assertNotIn('brief reasoning',harness.revision_messages(harness.initial_messages(),draft,actual)[1]['content'])

    def test_revision_grades_whole_completion_and_retains_normalization(self):
        for text in (CODE,'```python\n'+CODE+'```'):
            original=text
            c,p=harness.classify_revision(text,'eos')
            self.assertTrue(c['valid'],c);self.assertIsNotNone(p)
            self.assertEqual(c['final_answer'],original)
            self.assertEqual(c['revision_extraction']['end'],len(text))
            self.assertEqual(c['revision_extraction']['injected_boundaries'],0)
            self.assertFalse(c['thinking_enabled'])
        c,_=harness.classify_debug(CODE,'eos')
        self.assertEqual(c['reason'],'missing_thinking_boundary')

    def test_native_disabled_completion_never_extracts_or_repairs_thinking(self):
        for text,finish,reason in [(CODE,'length','generation_length'),
            ('analysis</think>'+CODE,'eos','unexpected_thinking'),
            ('<think>analysis</think>'+CODE,'eos','unexpected_thinking'),
            ('commentary\n'+CODE,'eos','invalid_cardinal_python')]:
            c,p=harness.classify_revision(text,finish)
            self.assertFalse(c['valid']);self.assertIsNone(p)
            self.assertEqual(c['reason'],reason);self.assertEqual(c['final_answer'],text)
        draft={'text':'unfinished reasoning','finish_reason':'length','output_tokens':4096,
               'completion_token_limit':4096,'reasoning_effort':'low'}
        feedback=harness.practice_feedback(draft,[191000010,191000011])
        self.assertEqual(feedback['matches'],[])
        self.assertEqual(harness.revision_messages(harness.initial_messages(),draft,feedback)[1]['content'],draft['text'])
        self.assertEqual(harness.MODEL_CALLS,2)

if __name__=='__main__':unittest.main()
