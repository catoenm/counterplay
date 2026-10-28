import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from selfplay.challenges import proposal_prompt, task_row, validate_proposal, summarize
from selfplay.depth import family, render_depth
from selfplay.environment import generate_tasks
from selfplay.frontier_api import OpenAIBackend, request_payload, reserve_amount


class ChallengeTests(unittest.TestCase):
    def test_claimed_goal_is_not_repaired(self):
        t = generate_tasks(1, 220, (2,), 11)[0]
        row = task_row(t, 'test', 0)
        obj = {k: row[k] for k in t.__dataclass_fields__}
        text = '</think>' + json.dumps(obj)
        good = validate_proposal({'text': text}, 0, set(), True)
        self.assertTrue(good['accepted'])
        duplicate = validate_proposal({'text': text}, 0, {family(t)}, True)
        self.assertTrue(duplicate['valid'])
        self.assertFalse(duplicate['accepted'])
        obj['goal'] = [(t.goal[0]+1) % 11, t.goal[1]]
        self.assertFalse(validate_proposal({'text': '</think>' + json.dumps(obj)}, 0, set(), True)['valid'])
        self.assertFalse(validate_proposal({'text': text, 'finish_reason': 'length'}, 0, set(), True)['valid'])

    def test_solver_has_no_witness_and_grammar_schema_has_no_goal(self):
        t = generate_tasks(1, 221, (4,), 11)[0]
        prompt, mapping = render_depth(task_row(t, 'test', 0), t)
        self.assertNotIn('witness', prompt)
        body = request_payload({'prompt': prompt, 'labels': list(mapping), 'horizon': 4})
        schema = body['text']['format']['schema']
        self.assertEqual(schema['properties']['plan']['items']['enum'], list(mapping))
        self.assertNotIn('goal', json.dumps(schema))
        self.assertLess(reserve_amount(body) * 114, 35)

    def test_truncations_do_not_count_as_useful_challenges(self):
        rows = [{'task_id': 'a', 'origin': 'test', 'variant': 'canonical', 'depth': 2,
                 'success': i == 0, 'status': 'solved' if i == 0 else 'truncated',
                 'completion_tokens': 1, 'prompt_tokens': 1} for i in range(4)]
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(summarize(rows, Path(d), 4)['useful_mixed_task_ids'], [])
            for r in rows[1:]: r['status'] = 'wrong_goal'
            self.assertEqual(summarize(rows, Path(d), 4)['useful_mixed_task_ids'], ['a'])

    def test_budget_retains_unknown_and_does_not_retry(self):
        with tempfile.TemporaryDirectory() as d, patch('selfplay.frontier_api.api_key', return_value='test'):
            p = Path(d)
            (p / 'budget.json').write_text(json.dumps({'cloud_spend_to_date': 0, 'approved_pilot_cap': 100}))
            backend = OpenAIBackend(p / 'ledger.json', p / 'budget.json')
            with patch('urllib.request.urlopen', side_effect=TimeoutError) as send:
                with self.assertRaises(RuntimeError): backend.complete({'prompt': 'test', 'seed': 1})
                with self.assertRaises(RuntimeError): backend.complete({'prompt': 'test', 'seed': 1})
                self.assertEqual(send.call_count, 1)
            r = next(iter(json.loads((p / 'ledger.json').read_text())['requests'].values()))
            self.assertGreater(r['charged_upper_usd'], 0)
            self.assertEqual(r['state'], 'unknown')

    def test_budget_refuses_before_network(self):
        with tempfile.TemporaryDirectory() as d, patch('selfplay.frontier_api.api_key', return_value='test'):
            p = Path(d)
            (p / 'budget.json').write_text(json.dumps({'cloud_spend_to_date': 100, 'approved_pilot_cap': 100}))
            backend = OpenAIBackend(p / 'ledger.json', p / 'budget.json')
            with patch('urllib.request.urlopen') as send:
                with self.assertRaises(RuntimeError): backend.complete({'prompt': 'test', 'seed': 1})
                send.assert_not_called()

    def test_successful_request_is_journaled_and_reused(self):
        import io
        with tempfile.TemporaryDirectory() as d, patch('selfplay.frontier_api.api_key', return_value='test'):
            p = Path(d)
            (p / 'budget.json').write_text(json.dumps({'cloud_spend_to_date': 0, 'approved_pilot_cap': 100}))
            backend = OpenAIBackend(p / 'ledger.json', p / 'budget.json')
            result = {'status': 'completed', 'usage': {'input_tokens': 25, 'output_tokens': 12},
                      'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': '{"plan":[]}'}]}]}
            with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(result).encode())) as send:
                first = backend.complete({'prompt': 'test', 'seed': 1})
                second = backend.complete({'prompt': 'test', 'seed': 1})
                self.assertEqual(first, second)
                self.assertEqual(send.call_count, 1)


if __name__ == '__main__':
    unittest.main()
