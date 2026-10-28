import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from selfplay import program_api as api


class ProgramAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cwd = contextlib.chdir(self.temp.name)
        self.cwd.__enter__()
        Path('budget.json').write_text(json.dumps({'cloud_spend_to_date': 0, 'approved_pilot_cap': 100}))
        Path('runs').mkdir()
        self.key = patch.object(api, 'api_key', return_value='test-key')
        self.key.start()

    def tearDown(self):
        self.key.stop()
        self.cwd.__exit__(None, None, None)
        self.temp.cleanup()

    @staticmethod
    def response():
        return io.BytesIO(json.dumps({'id': 'test-response', 'model': api.MODEL, 'provider': 'DeepInfra',
            'usage': {'prompt_tokens': 100, 'completion_tokens': 100, 'cost': 0.00039},
            'choices': [{'finish_reason': 'stop', 'message': {'content': '{}'}}]}).encode())

    def test_identical_request_reuses_saved_response(self):
        with patch.object(api.urllib.request, 'urlopen', side_effect=lambda *a, **k: self.response()) as call:
            first = api.complete('Write a controller.', 'one')
            self.assertEqual(first, api.complete('Write a controller.', 'one'))
            self.assertEqual(call.call_count, 1)

    def test_unknown_request_retains_reservation_and_halts_screen(self):
        with patch.object(api.urllib.request, 'urlopen', side_effect=urllib.error.URLError('offline')) as call:
            with self.assertRaises(RuntimeError):
                api.complete('Write a controller.', 'one')
            with self.assertRaises(RuntimeError):
                api.complete('Write another controller.', 'two')
            self.assertEqual(call.call_count, 1)
        record = next(iter(json.loads(api.LEDGER.read_text())['requests'].values()))
        self.assertEqual(record['state'], 'unknown')
        self.assertEqual(api.ledger_commitment(api.LEDGER), record['reserved_usd'])

    def test_prior_unknown_cost_and_cloud_reservation_count_toward_total(self):
        Path('budget.json').write_text(json.dumps({'cloud_spend_to_date': 99, 'cloud_reserved_usd': 0.8,
                                                 'approved_pilot_cap': 100}))
        Path('runs/openrouter-api-ledger.json').write_text(json.dumps({'requests': {'old': {
            'state': 'interrupted_unknown', 'reserved_usd': 0.3, 'charged_upper_usd': 0}}}))
        with patch.object(api.urllib.request, 'urlopen') as call, self.assertRaises(RuntimeError):
            api.complete('Write a controller.', 'one')
        call.assert_not_called()

    def test_provider_mismatch_halts_without_releasing_reservation(self):
        response = json.loads(self.response().read())
        response['provider'] = 'unexpected-provider'
        with patch.object(api.urllib.request, 'urlopen', return_value=io.BytesIO(json.dumps(response).encode())):
            with self.assertRaises(RuntimeError):
                api.complete('Write a controller.', 'one')
        record = next(iter(json.loads(api.LEDGER.read_text())['requests'].values()))
        self.assertEqual(record['state'], 'invalid_response')
        self.assertGreaterEqual(record['charged_upper_usd'], record['reserved_usd'])

    def test_oversized_prompt_rejected_without_a_request(self):
        with patch.object(api.urllib.request, 'urlopen') as call, self.assertRaises(ValueError):
            api.complete('x' * 40001, 'one')
        call.assert_not_called()

    def test_thinking_diagnostic_preserves_model_and_total_output_cap(self):
        original = api.body_for('Write a controller.', 'high')
        diagnostic = api.body_for('Write a controller.', 'off')
        self.assertEqual(original.pop('reasoning'), {'effort': 'high'})
        self.assertEqual(diagnostic.pop('reasoning'), {'enabled': False})
        self.assertEqual(original, diagnostic)


if __name__ == '__main__':
    unittest.main()
