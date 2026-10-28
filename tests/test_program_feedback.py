"""Explicit mocked-generation fixtures only. No paid API or GPU requests."""
import contextlib
import copy
import fcntl
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from selfplay import program_feedback as client

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('feedback_runner_test', ROOT / 'scripts/run_program_feedback.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def context(lineage='qwen38-27b-1', round_index=1):
    return {'plan_sha256': 'a'*64, 'lineage': lineage, 'round_index': round_index,
            'previous_generation_sha256': 'b'*64, 'feedback_sha256': 'c'*64, 'original_prompt_sha256': 'd'*64}


def response(**changes):
    value = {'id': 'explicit-test-fixture', 'model': 'qwen/qwen3.8-27b', 'provider': 'Chutes',
             'usage': {'prompt_tokens': 100, 'completion_tokens': 100, 'cost': .0001,
                       'completion_tokens_details': {'reasoning_tokens': 0}},
             'choices': [{'finish_reason': 'stop', 'message': {'content': "def act():\n    return {'action': 'rest'}\n"}}]}
    value.update(changes)
    return io.BytesIO(json.dumps(value).encode())


class FeedbackClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cwd = contextlib.chdir(self.temp.name)
        self.cwd.__enter__()
        Path('runs').mkdir()
        self.budget = {'cloud_spend_to_date': 0, 'approved_pilot_cap': 100}
        self.write_budget()
        self.key = patch.object(client, 'api_key', return_value='test-secret-never-log')
        self.quote = patch.object(client, 'verify_endpoint', return_value={'explicit_fixture': True})
        self.key.start(); self.quote.start()

    def tearDown(self):
        self.quote.stop(); self.key.stop()
        self.cwd.__exit__(None, None, None)
        self.temp.cleanup()

    def write_budget(self):
        Path('budget.json').write_text(json.dumps(self.budget))

    def test_four_calls_two_rounds_identity_sampling_and_cache(self):
        body = client.body_for('spec + own code + feedback', 'qwen38-27b')
        self.assertEqual(body['provider']['only'], ['chutes/fp8'])
        self.assertEqual(body['provider']['quantizations'], ['fp8'])
        self.assertFalse(body['provider']['allow_fallbacks'])
        self.assertEqual((body['temperature'], body['top_p'], body['top_k'], body['max_tokens']), (.6, .95, 20, 4096))
        with patch.object(client.urllib.request, 'urlopen', side_effect=lambda *a, **kw: response()) as call:
            for r in (1, 2):
                for name in client.LINEAGES:
                    c = context(name, r)
                    result = client.complete('different feedback ' + name + str(r), f'{name}-r{r}', c)
                    self.assertEqual(result['feedback_context'], c)
                    self.assertEqual(result['reasoning_condition_status'], 'confirmed_off')
                    self.assertEqual(result, client.complete(result['prompt'], result['sample_id'], c))
            self.assertEqual(call.call_count, 4)
            with self.assertRaises(ValueError):
                client.complete('extra', 'bad', context(round_index=3))
        self.assertNotIn('test-secret-never-log', client.LEDGER.read_text())

    def test_context_chain_sequential_plan_and_sample_guards(self):
        with patch.object(client.urllib.request, 'urlopen', side_effect=lambda *a, **kw: response()) as call:
            with self.assertRaisesRegex(RuntimeError, 'sequential'):
                client.complete('p', 'qwen38-27b-1-r2', context(round_index=2))
            client.complete('p', 'qwen38-27b-1-r1', context())
            changed = context('qwen38-27b-2'); changed['plan_sha256'] = 'e'*64
            with self.assertRaisesRegex(RuntimeError, 'frozen plan'):
                client.complete('p', 'qwen38-27b-2-r1', changed)
            with self.assertRaisesRegex(RuntimeError, 'Sample ID'):
                client.complete('changed', 'qwen38-27b-1-r1', context())
            self.assertEqual(call.call_count, 1)

    def test_shared_budget_counts_old_commitments_and_cloud(self):
        self.budget.update(cloud_spend_to_date=99, cloud_reserved_usd=.9)
        self.write_budget()
        Path('runs/program-model-screen-ledger.json').write_text(json.dumps({'requests': {'old': {
            'state': 'unknown', 'reserved_usd': .2, 'charged_upper_usd': .01}}}))
        with patch.object(client.urllib.request, 'urlopen') as call:
            with self.assertRaisesRegex(RuntimeError, 'Cumulative'):
                client.complete('p', 'qwen38-27b-1-r1', context())
            call.assert_not_called()
        self.budget.update(cloud_spend_to_date=0, cloud_reserved_usd=0, program_feedback_subcap_usd=.001)
        self.write_budget()
        with patch.object(client.urllib.request, 'urlopen') as call:
            with self.assertRaisesRegex(RuntimeError, 'subcap'):
                client.complete('p', 'qwen38-27b-1-r1', context())
            call.assert_not_called()

    def test_unknown_outcome_no_retry_and_lock_held(self):
        def fail(*args, **kwargs):
            with client.BUDGET_LOCK.open('a') as lock:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            raise urllib.error.URLError('test-secret-never-log')
        with patch.object(client.urllib.request, 'urlopen', side_effect=fail) as call:
            for name in client.LINEAGES:
                with self.assertRaises(RuntimeError) as err:
                    client.complete('p', name+'-r1', context(name))
                self.assertNotIn('test-secret-never-log', str(err.exception))
            self.assertEqual(call.call_count, 1)
        self.assertGreater(client.ledger_commitment(client.LEDGER), 0)

    def test_reasoning_condition_explicitly_distinguishes_unknown_and_violated(self):
        for tokens, expected in ((7, 'violated'), (None, 'unknown')):
            client.LEDGER.unlink(missing_ok=True)
            usage = {'prompt_tokens': 100, 'completion_tokens': 100, 'cost': .0001}
            if tokens is not None:
                usage['completion_tokens_details'] = {'reasoning_tokens': tokens}
            with patch.object(client.urllib.request, 'urlopen', return_value=response(usage=usage)):
                result = client.complete('p', 'qwen38-27b-1-r1', context())
            self.assertEqual(result['reasoning_condition_status'], expected)


class FeedbackRunnerTests(unittest.TestCase):
    def fixture_original(self, root):
        original = root / 'explicit-fixture-original'
        real = ROOT / 'runs/program-qwen38-27b-v1'
        for file in ['plan.json', 'prompt.txt'] + ['source/' + p.name for p in (real/'source').iterdir() if p.is_file()]:
            target = original / file
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((real / file).read_bytes())
        for i in (1, 2):
            generation = json.loads((real / f'generation-{i}.json').read_text())
            generation.update(text="def act():\n    return {'action': 'rest'}\n", finish_reason='stop',
                              response_id='explicit-fixture-no-paid-call', cost_usd=0, cost_upper_usd=0)
            (original/f'generation-{i}.json').write_bytes(runner.encoded(generation))
        return original

    def test_fixture_full_chain_holdout_gate_no_selection_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); original = self.fixture_original(root); run = root/'diagnostic'
            runner.prepare(run, original)
            with self.assertRaises(FileNotFoundError):
                runner.holdout(run)
            self.assertFalse((run/'holdout').exists())
            calls = []
            ledger_patch = patch.object(client, 'LEDGER', root/'fixture-feedback-ledger.json')
            ledger_patch.start()
            self.addCleanup(ledger_patch.stop)
            def generate(prompt, sample_id, ctx):
                calls.append(sample_id)
                plan = runner.read(run/'plan.json')
                p = client.PROFILES['qwen38-27b']
                # Last revision intentionally invalid: retain it instead of falling back.
                text = "def act():\n    return {'action': 'rest'}\n" if ctx['round_index']==1 else 'def act():\n    break\n'
                record = {'sample_id':sample_id, 'prompt':prompt, 'prompt_sha256':runner.sha(prompt.encode()),
                        'text':text, 'finish_reason':'stop', 'feedback_context':ctx, 'generation_client':'program_feedback',
                        'client_source_sha256':plan['diagnostic_source_sha256']['program_feedback.py'],
                        'generation_client_configuration_sha256':runner.sha(client.canonical(client.configuration()).encode()),
                        'generation_client_profile':{'name':'qwen38-27b',**p},
                        'returned_model':p['model'], 'returned_provider':p['returned_provider'],
                        'request_parameters':{k:v for k,v in client.body_for(prompt,'qwen38-27b').items() if k!='messages'},
                        'usage':{'completion_tokens_details':{'reasoning_tokens':0}}, 'reasoning_condition_status':'confirmed_off',
                        'cost_usd':0,'cost_upper_usd':0,'explicit_fixture':True}
                source_hash = client.sha(Path(client.__file__).read_bytes())
                journal = runner.read(client.LEDGER) if client.LEDGER.exists() else client.new_ledger(source_hash)
                body = client.body_for(prompt, 'qwen38-27b')
                identity = client.sha(client.canonical({'body':body,'sample_id':sample_id,
                    'client_source_sha256':source_hash,'feedback_context':ctx}).encode())
                journal['requests'][identity] = {'state':'completed','response':record,'explicit_fixture':True}
                client.atomic_json(client.LEDGER, journal)
                return record
            with patch.object(client, 'complete', side_effect=generate):
                for _ in range(4):
                    runner.advance(run)
                self.assertEqual(runner.advance(run)['status'], 'all_four_revisions_complete_ready_for_holdout')
            self.assertEqual(len(calls), 4)
            self.assertFalse((run/'holdout').exists())
            result = runner.holdout(run)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(runner.holdout(run), result)
            summary = runner.read(run/'summary.json')
            for name in client.LINEAGES:
                self.assertEqual(summary['holdout'][name+'-r2']['matches'], 0)
                self.assertFalse(summary['holdout'][name+'-r2']['classification']['valid'])
                self.assertEqual(summary['holdout'][name]['matches'], 24)
                self.assertFalse((run/'holdout'/(name+'-r1')).exists())
                prompt = (run/'prompts'/(name+'-r1.txt')).read_text()
                self.assertNotIn('740000', prompt)
                self.assertNotIn('740005', prompt)
            # Changing only model text must fail even with otherwise valid self-reported metadata.
            path = run/'generations/qwen38-27b-1-r2.json'
            original_bytes = path.read_bytes()
            modified = runner.read(path); modified['text'] = "def act():\n    return {'action':'rest'}\n"
            path.write_bytes(runner.encoded(modified))
            with self.assertRaisesRegex(ValueError, 'paid-request journal'):
                runner.holdout(run)
            path.write_bytes(original_bytes)
            artifact = run/'practice/qwen38-27b-1/matches.jsonl'
            with artifact.open('ab') as stream:
                stream.write(b'{')
            with self.assertRaisesRegex(ValueError, 'Partial'):
                runner.holdout(run)

    def test_feedback_turn_failures_and_fixed_trace_selection(self):
        from selfplay.program_arena import play
        from selfplay.program_python import compile_python_program
        import scripts.evaluate_program_screen as evaluator
        p = compile_python_program("def act():\n    return {'action':'move','dx':1,'dy':1}\n")
        r = play([p.compiled,p.compiled], 730000)
        wrapper = {'players':['model','nearest'],'record':r}
        f = runner.feedback_from([wrapper], 'model', {'valid':True,'failure':None,'normalization':'none'}, evaluator)
        self.assertEqual(len(f['traces']),1)
        self.assertEqual(len(f['traces'][0]['rows']),48)
        groups = f['match_summaries'][0]['failed_turn_groups']
        self.assertEqual(groups[0]['turns'],list(range(48)))
        self.assertEqual(groups[0]['action_outcome_fault_points_opponent_points'][1], 'rested')
        self.assertIsNotNone(groups[0]['action_outcome_fault_points_opponent_points'][2])
        self.assertEqual(f['overall']['banked_points'],0)


if __name__ == '__main__':
    unittest.main()
