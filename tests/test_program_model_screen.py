"""Mock-only tests. No credentials, provider requests, or GPU allocation."""
import contextlib
import copy
import fcntl
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from selfplay import program_model_screen as client


class GatewayScreenTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cwd = contextlib.chdir(self.temp.name)
        self.cwd.__enter__()
        Path('runs').mkdir()
        self.budget = {'cloud_spend_to_date': 0, 'approved_pilot_cap': 100}
        self.write_budget()
        self.key = patch.object(client, 'api_key', return_value='private-test-key')
        self.preflight = patch.object(client, 'verify_endpoint', return_value={'explicit_fixture': True})
        self.key.start()
        self.preflight.start()

    def tearDown(self):
        self.preflight.stop()
        self.key.stop()
        self.cwd.__exit__(None, None, None)
        self.temp.cleanup()

    def write_budget(self):
        Path('budget.json').write_text(json.dumps(self.budget))

    @staticmethod
    def response(profile_name='qwen3-32b', **changes):
        profile = client.PROFILES[profile_name]
        value = {'id': 'explicit-fixture-response', 'model': profile['model'], 'provider': profile['returned_provider'],
                 'usage': {'prompt_tokens': 100, 'completion_tokens': 100, 'cost': 0.0001},
                 'choices': [{'finish_reason': 'stop', 'message': {'content': "def act():\n    return {'action':'rest'}"}}]}
        value.update(changes)
        return io.BytesIO(json.dumps(value).encode())

    def entries(self):
        return json.loads(client.LEDGER.read_text())['requests']

    def test_request_fixed_profiles_final_only_and_private_cache(self):
        for name, profile in client.PROFILES.items():
            body = client.body_for('exact prompt', name)
            self.assertEqual(body['model'], profile['model'])
            self.assertEqual(body['messages'], [{'role': 'user', 'content': 'exact prompt'}])
            self.assertEqual(body['reasoning'], {'enabled': False})
            self.assertEqual(body['max_tokens'], 4096)
            self.assertEqual((body['temperature'], body['top_p'], body['top_k']), (.6, .95, 20))
            self.assertEqual(body['provider']['quantizations'], ['fp8'])
            self.assertEqual(body['provider']['only'], [profile['provider']])
            self.assertIs(body['provider']['allow_fallbacks'], False)
            self.assertIs(body['provider']['require_parameters'], True)
        with patch.object(client.urllib.request, 'urlopen', side_effect=lambda *a, **k: self.response()) as call:
            first = client.complete('exact prompt', 'one', 'qwen3-32b')
            self.assertEqual(first, client.complete('exact prompt', 'one', 'qwen3-32b'))
            self.assertEqual(call.call_count, 1)
            self.assertEqual(first['generation_client'], 'program_model_screen')
            self.assertEqual(first['reasoning_mode'], 'off')
            self.assertEqual(first['prompt_sha256'], client.sha(b'exact prompt'))
            self.assertNotIn('private-test-key', client.LEDGER.read_text())

    def test_unknown_outcome_holds_reservation_without_retry_or_secret_leak(self):
        with patch.object(client.urllib.request, 'urlopen', side_effect=urllib.error.URLError('private-test-key')) as call:
            for sample, profile in (('one', 'qwen3-32b'), ('one', 'qwen3-32b'), ('two', 'qwen38-27b')):
                with self.assertRaises(RuntimeError) as error:
                    client.complete('same prompt', sample, profile)
                self.assertNotIn('private-test-key', str(error.exception))
            self.assertEqual(call.call_count, 1)
        entry = next(iter(self.entries().values()))
        self.assertEqual(entry['state'], 'unknown')
        self.assertEqual(client.ledger_commitment(client.LEDGER), entry['reserved_usd'])
        self.assertNotIn('private-test-key', client.LEDGER.read_text())

    def test_total_budget_includes_all_prior_ledgers_and_cloud_reservation(self):
        self.budget.update(cloud_spend_to_date=98.5, cloud_reserved_usd=.8)
        self.write_budget()
        for path in client.PRIOR_LEDGERS:
            path.write_text(json.dumps({'requests': {'old': {'state': 'unknown', 'reserved_usd': .3,
                                                           'charged_upper_usd': .1}}}))
        with patch.object(client.urllib.request, 'urlopen') as call:
            with self.assertRaisesRegex(RuntimeError, 'Cumulative pilot'):
                client.complete('same prompt', 'one', 'qwen3-32b')
            call.assert_not_called()
        self.assertFalse(client.LEDGER.exists())

    def test_subcap_and_per_profile_and_total_call_caps(self):
        with patch.object(client.urllib.request, 'urlopen', side_effect=lambda request, **kw:
                          self.response('qwen38-27b' if json.loads(request.data)['model'] == client.PROFILES['qwen38-27b']['model'] else 'qwen3-32b')) as call:
            for sample in ('one', 'two'):
                client.complete('same prompt', sample, 'qwen3-32b')
            with self.assertRaisesRegex(RuntimeError, 'call cap'):
                client.complete('same prompt', 'third-same-model', 'qwen3-32b')
            for sample in ('three', 'four'):
                client.complete('same prompt', sample, 'qwen38-27b')
            with self.assertRaisesRegex(RuntimeError, 'call cap'):
                client.complete('same prompt', 'fifth', 'qwen38-27b')
            self.assertEqual(call.call_count, 4)
        self.assertEqual(len(self.entries()), 4)
        # Lower administrative subcaps also take precedence over the frozen $1.
        client.LEDGER.unlink()
        self.budget['program_model_screen_subcap_usd'] = .000001
        self.write_budget()
        with patch.object(client.urllib.request, 'urlopen') as call, self.assertRaisesRegex(RuntimeError, 'subcap'):
            client.complete('same prompt', 'small-cap', 'qwen3-32b')
        call.assert_not_called()

    def test_rejects_wrong_model_provider_usage_and_token_limit(self):
        cases = [({'model': 'wrong-model'}, 'invalid_response'),
                 ({'provider': 'wrong-provider'}, 'invalid_response'),
                 ({'usage': {}}, 'unknown_usage'),
                 ({'usage': {'prompt_tokens': 100, 'completion_tokens': 4097, 'cost': .001}}, 'invalid_response'),
                 ({'usage': {'prompt_tokens': 100, 'completion_tokens': 100, 'cost': 2.0}}, 'invalid_response')]
        for changes, state in cases:
            with self.subTest(changes=changes):
                client.LEDGER.unlink(missing_ok=True)
                with patch.object(client.urllib.request, 'urlopen', return_value=self.response(**changes)) as call:
                    with self.assertRaises(RuntimeError):
                        client.complete('same prompt', 'one', 'qwen3-32b')
                    with self.assertRaises(RuntimeError):
                        client.complete('same prompt', 'two', 'qwen38-27b')
                    self.assertEqual(call.call_count, 1)
                entry = next(iter(self.entries().values()))
                self.assertEqual(entry['state'], state)
                self.assertGreaterEqual(entry['charged_upper_usd'], entry['reserved_usd'])

    def test_global_lock_is_held_during_paid_request(self):
        def response_while_locked(*args, **kwargs):
            with client.BUDGET_LOCK.open('a') as other:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            entry = next(iter(self.entries().values()))
            self.assertEqual(entry['state'], 'reserved')
            return self.response()
        with patch.object(client.urllib.request, 'urlopen', side_effect=response_while_locked):
            client.complete('same prompt', 'one', 'qwen3-32b')

    def test_changed_prompt_or_sample_id_payload_cannot_replace_request(self):
        with patch.object(client.urllib.request, 'urlopen', side_effect=lambda *a, **kw: self.response()) as call:
            client.complete('same prompt', 'one', 'qwen3-32b')
            with self.assertRaisesRegex(RuntimeError, 'Sample ID'):
                client.complete('modified prompt', 'one', 'qwen3-32b')
            with self.assertRaisesRegex(RuntimeError, 'unchanged prompt'):
                client.complete('modified prompt', 'two', 'qwen38-27b')
            self.assertEqual(call.call_count, 1)

    def test_invalid_budget_or_oversized_prompt_fails_before_request(self):
        self.budget['cloud_reserved_usd'] = float('nan')
        self.write_budget()
        with patch.object(client.urllib.request, 'urlopen') as call:
            with self.assertRaises(ValueError):
                client.complete('same prompt', 'one', 'qwen3-32b')
            with self.assertRaisesRegex(ValueError, '40 KB'):
                client.complete('x' * 40001, 'two', 'qwen38-27b')
            call.assert_not_called()


class PublicEndpointTests(unittest.TestCase):
    @staticmethod
    def catalog():
        p = client.PROFILES['qwen3-32b']
        return {'data': {'id': p['model'], 'endpoints': [{
            'tag': p['provider'], 'provider_name': p['returned_provider'], 'quantization': 'fp8',
            'pricing': {'prompt': '0.00000008', 'completion': '0.00000028'},
            'supported_parameters': sorted(client.REQUIRED_PARAMETERS), 'max_completion_tokens': 4096}]}}

    def test_current_exact_endpoint_quote_is_checked_without_authentication(self):
        value = self.catalog()
        with patch.object(client.urllib.request, 'urlopen', return_value=io.BytesIO(json.dumps(value).encode())) as call:
            quote = client.verify_endpoint('qwen3-32b')
        self.assertIsInstance(call.call_args.args[0], str)
        self.assertEqual(quote['endpoint_sha256'], client.sha(client.canonical(value['data']['endpoints'][0]).encode()))

    def test_missing_provider_wrong_model_price_or_parameters_fail_closed(self):
        original = self.catalog()
        cases = []
        for field, value in (('quantization', 'fp4'), ('tag', 'other/fp8'),
                             ('supported_parameters', ['max_tokens']), ('max_completion_tokens', 1024),
                             ('pricing', {'prompt': '0.000008', 'completion': '0.000028'})):
            catalog = copy.deepcopy(original)
            catalog['data']['endpoints'][0][field] = value
            cases.append(catalog)
        wrong_model = copy.deepcopy(original)
        wrong_model['data']['id'] = 'another-model'
        cases.append(wrong_model)
        for value in cases:
            with self.subTest(value=value), patch.object(client.urllib.request, 'urlopen', return_value=io.BytesIO(json.dumps(value).encode())):
                with self.assertRaisesRegex(RuntimeError, 'preflight failed'):
                    client.verify_endpoint('qwen3-32b')


if __name__ == '__main__':
    unittest.main()
