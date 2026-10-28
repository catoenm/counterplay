import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from selfplay.openrouter_api import MODEL, OpenRouterBackend, payload, reservation


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.budget = self.root / 'budget.json'
        self.budget.write_text(json.dumps({'cloud_spend_to_date': 0, 'approved_pilot_cap': 100}))
        with patch('selfplay.openrouter_api.api_key', return_value='not-a-key'):
            self.backend = OpenRouterBackend(self.root / 'ledger.json', self.budget)
        self.request = {'prompt': 'JSON plan, use foo then bar', 'labels': ['foo', 'bar'], 'horizon': 2, 'seed': 1}

    def result(self, **changes):
        result = {'model': MODEL, 'provider': 'DeepInfra', 'id': 'test',
                  'usage': {'prompt_tokens': 200, 'completion_tokens': 100, 'cost': .0005,
                            'completion_tokens_details': {'reasoning_tokens': 90}},
                  'choices': [{'finish_reason': 'stop', 'message': {'content': '{"plan":["foo","bar"]}'}}]}
        result.update(changes)
        return io.BytesIO(json.dumps(result).encode())

    def test_pinned_schema_and_budget(self):
        body = payload(self.request)
        self.assertFalse(body['provider']['allow_fallbacks'])
        self.assertTrue(body['provider']['require_parameters'])
        self.assertEqual(body['provider']['only'], ['deepinfra/fp8'])
        schema = body['response_format']['json_schema']['schema']
        self.assertEqual(schema['properties']['plan']['items']['enum'], ['foo', 'bar'])
        self.assertNotIn('goal', json.dumps(schema))
        self.assertLess(reservation(body) * 115, 10)

    def test_journal_reuse_includes_reasoning_cost(self):
        with patch('urllib.request.urlopen', return_value=self.result()) as send:
            first = self.backend.complete(self.request)
            self.assertEqual(first, self.backend.complete(self.request))
            self.assertEqual(send.call_count, 1)
        self.assertAlmostEqual(first['cost_upper_usd'], .00052)

    def test_timeout_stops_all_subsequent_requests(self):
        with patch('urllib.request.urlopen', side_effect=TimeoutError) as send:
            with self.assertRaises(RuntimeError): self.backend.complete(self.request)
            with self.assertRaises(RuntimeError): self.backend.complete({**self.request, 'seed': 2})
            self.assertEqual(send.call_count, 1)

    def test_cap_refuses_before_network(self):
        self.budget.write_text(json.dumps({'cloud_spend_to_date': 100, 'approved_pilot_cap': 100}))
        with patch('urllib.request.urlopen') as send:
            with self.assertRaises(RuntimeError): self.backend.complete(self.request)
            send.assert_not_called()

    def test_wrong_provider_is_not_a_scored_response(self):
        with patch('urllib.request.urlopen', return_value=self.result(provider='Other')) as send:
            with self.assertRaises(RuntimeError): self.backend.complete(self.request)
            with self.assertRaises(RuntimeError): self.backend.complete({**self.request, 'seed': 2})
            self.assertEqual(send.call_count, 1)

    def test_missing_cost_retains_reservation(self):
        with patch('urllib.request.urlopen', return_value=self.result(usage={'prompt_tokens': 200, 'completion_tokens': 100})):
            with self.assertRaises(RuntimeError): self.backend.complete(self.request)
        entry = next(iter(json.loads(self.backend.ledger.read_text())['requests'].values()))
        self.assertEqual(entry['state'], 'unknown_usage')
        self.assertEqual(entry['charged_upper_usd'], entry['reserved_usd'])

    def test_provider_cost_overrun_stops_future_requests(self):
        with patch('urllib.request.urlopen', return_value=self.result(usage={'prompt_tokens': 200, 'completion_tokens': 100, 'cost': 1})) as send:
            with self.assertRaises(RuntimeError): self.backend.complete(self.request)
            with self.assertRaises(RuntimeError): self.backend.complete({**self.request, 'seed': 2})
            self.assertEqual(send.call_count, 1)


if __name__ == '__main__':
    unittest.main()
