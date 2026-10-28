"""NONE-condition CPU fixtures only; mocked gateways never invoke paid inference."""
import contextlib
import copy
import fcntl
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error

from selfplay import program_cardinal_teacher_none as client
from selfplay import program_arena
from selfplay.program_cardinal import cardinal_model_spec
from scripts import collect_cardinal_teacher_none as collector


PLAN = 'a' * 64


def response(**changes):
    value = {'id': 'explicit-mocked-fixture', 'model': client.PROFILE['model'], 'provider': 'DeepInfra',
        'usage': {'prompt_tokens': 100, 'completion_tokens': 100, 'cost': .0001,
            'completion_tokens_details': {'reasoning_tokens': 70}},
        'choices': [{'finish_reason': 'stop', 'message': {'content': "def act():\n    return 'rest'\n"}}]}
    value.update(changes)
    return io.BytesIO(json.dumps(value).encode())


class TeacherClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.cwd = contextlib.chdir(self.temp.name); self.cwd.__enter__()
        self.addCleanup(self.cwd.__exit__, None, None, None)
        Path('runs').mkdir()
        self.budget = {'cloud_spend_to_date': 0, 'cloud_reserved_usd': 80, 'approved_pilot_cap': 100}
        self.write_budget()
        for target, name, value in ((client, 'verify_endpoint', {'explicit_fixture': True}),
            (client.gateway, 'api_key', 'fixture-secret-never-log')):
            p = patch.object(target, name, return_value=value); p.start(); self.addCleanup(p.stop)

    def write_budget(self):
        Path('budget.json').write_text(json.dumps(self.budget))

    def opener(self, **kwargs):
        return patch.object(client.urllib.request, 'build_opener', return_value=SimpleNamespace(open=lambda *a, **k: response(**kwargs)))

    def test_pinned_reasoning_payload_and_all32_fixed_attempts(self):
        body = client.body_for('spec', 4)
        self.assertEqual(body['reasoning'], {'effort': 'none'})
        self.assertEqual(body['provider']['only'], ['deepinfra/fp8'])
        self.assertEqual(body['provider']['quantizations'], ['fp8'])
        self.assertFalse(body['provider']['allow_fallbacks'])
        self.assertEqual(body['max_tokens'], 16384)
        self.assertEqual(body['seed'], 131000004)
        with self.opener() as calls:
            for i in range(32):
                record = client.complete('spec '+str(i), i, PLAN)
                self.assertEqual(client.complete('spec '+str(i), i, PLAN), record)
                client.verify_completed_response('spec '+str(i), i, PLAN, record)
            self.assertEqual(calls.call_count, 32)
        with self.assertRaises(ValueError): client.body_for('extra', 32)
        self.assertNotIn('fixture-secret-never-log', client.LEDGER.read_text())

    def test_reasoning_usage_remains_separate_from_delivery(self):
        with self.opener(): first = client.complete('p',0,PLAN)
        self.assertEqual(first['reasoning_requested'], {'effort':'none'})
        self.assertEqual(first['reported_reasoning_tokens'],70)
        with self.opener(usage={'prompt_tokens':100,'completion_tokens':100,'cost':.0001}):
            second=client.complete('p2',1,PLAN)
        self.assertIsNone(second['reported_reasoning_tokens'])
        self.assertTrue(client.delivery_passed(first))
        self.assertTrue(client.delivery_passed(second))

    def test_unknown_call_retains_reservation_and_cannot_repeat(self):
        def fail(*args, **kwargs):
            with client.BUDGET_LOCK.open('a') as lock:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            raise urllib.error.URLError('fixture-secret-never-log')
        with patch.object(client.urllib.request, 'build_opener', return_value=SimpleNamespace(open=fail)) as calls:
            with self.assertRaisesRegex(RuntimeError, 'reservation held'): client.complete('p', 0, PLAN)
            with self.assertRaisesRegex(RuntimeError, 'Unresolved'): client.complete('p', 0, PLAN)
            with self.assertRaisesRegex(RuntimeError, 'Unresolved'): client.complete('p2', 1, PLAN)
            self.assertEqual(calls.call_count, 1)
        self.assertGreater(client.gateway.ledger_commitment(client.LEDGER), 0)
        self.assertNotIn('fixture-secret-never-log', client.LEDGER.read_text())

    def test_global_and_phase_admission_include_every_ledger(self):
        self.budget.update(cloud_spend_to_date=19.9); self.write_budget()
        Path('runs/other-ledger.json').write_text(json.dumps({'requests': {'old': {
            'state': 'unknown', 'reserved_usd': .2, 'charged_upper_usd': 0}}}))
        with self.opener() as calls:
            with self.assertRaisesRegex(RuntimeError, 'budget admission'): client.complete('p', 0, PLAN)
            calls.assert_not_called()
        self.budget.update(cloud_spend_to_date=0, program_cardinal_teacher_none_subcap_usd=.001); self.write_budget()
        with self.opener() as calls:
            with self.assertRaisesRegex(RuntimeError, 'budget admission'): client.complete('p', 0, PLAN)
            calls.assert_not_called()

    def test_identity_failure_is_retained_and_blocks_followups(self):
        with self.opener(model='wrong/model'):
            with self.assertRaisesRegex(RuntimeError, 'identity/usage'): client.complete('p', 0, PLAN)
        entry = next(iter(json.loads(client.LEDGER.read_text())['requests'].values()))
        self.assertEqual(entry['raw_response']['model'], 'wrong/model')
        self.assertEqual(entry['state'], 'invalid_response')
        with self.assertRaisesRegex(RuntimeError, 'Unresolved'): client.complete('p2', 1, PLAN)

    def test_saved_output_text_mutation_and_prompt_replacement_rejected(self):
        with self.opener(): record = client.complete('p', 0, PLAN)
        altered = copy.deepcopy(record); altered['text'] += '# altered'
        with self.assertRaisesRegex(ValueError, 'journal'): client.verify_completed_response('p', 0, PLAN, altered)
        with self.assertRaisesRegex(RuntimeError, 'sequential'): client.complete('changed prompt', 0, PLAN)
        with self.assertRaisesRegex(RuntimeError, 'sequential'): client.complete('skipped index', 2, PLAN)
        with self.assertRaisesRegex(ValueError, 'protocol'): client.complete('p', 1, 'b'*64)

    def test_truncation_is_completed_attempt_not_replaced(self):
        with self.opener(choices=[{'finish_reason': 'length', 'message': {'content': ''}}]):
            record = client.complete('p', 0, PLAN)
        self.assertEqual(record['finish_reason'], 'length')
        client.verify_completed_response('p', 0, PLAN, record)
        with self.opener() as calls:
            with self.assertRaisesRegex(RuntimeError, 'delivery gate failed'): client.complete('p2', 1, PLAN)
            calls.assert_not_called()
        self.assertEqual(client.checked_ledger(PLAN)['requests'][record['request_identity']]['response'], record)

    def test_delivery_gate_is_not_a_compile_or_strategy_gate(self):
        with self.opener(choices=[{'finish_reason': 'stop', 'message': {'content': 'invalid code but delivered'}}]):
            client.complete('p', 0, PLAN)
        with self.opener() as calls:
            client.complete('p2', 1, PLAN)
            self.assertEqual(calls.call_count, 1)

    def test_empty_stop_fails_delivery_and_has_no_dynamic_fallback(self):
        with self.opener(choices=[{'finish_reason': 'stop', 'message': {'content': '   '}}]):
            client.complete('p', 0, PLAN)
        with self.opener() as calls:
            with self.assertRaisesRegex(RuntimeError, 'delivery gate failed'): client.complete('p2', 1, PLAN)
            calls.assert_not_called()


class TeacherCollectorTests(unittest.TestCase):
    def row(self, i, **changes):
        row = {'index': i, 'classification': {'valid': True, 'program': {'ir_sha256': str(i)}},
            'learner_tokens_including_eos': 700, 'candidate_program_faults': 0,
            'positive_banking_games': 12, 'matches': [{}] * 24}
        row.update(changes); return row

    def test_selection_fixed_first16_unique_and_no_shortfall_fallback(self):
        rows = [self.row(i) for i in range(32)]
        rows[1]['classification']['program']['ir_sha256'] = '0'
        rows[2]['learner_tokens_including_eos'] = 701
        rows[3]['candidate_program_faults'] = 1
        rows[4]['positive_banking_games'] = 11
        chosen = collector.select(rows)
        self.assertEqual(chosen, [0] + list(range(5, 20)))
        for row in rows[19:]: row['positive_banking_games'] = 0
        self.assertEqual(collector.select(rows), [])
        with self.assertRaisesRegex(ValueError, 'All32'): collector.select(rows[:-1])

    def test_candidate_faults_only_and_invalid_has_no_fake_games(self):
        row = self.row(0, opponent_program_faults=100)
        self.assertTrue(collector.qualifies(row))
        row['classification'] = {'valid': False}; row['learner_tokens_including_eos'] = None; row['matches'] = []
        self.assertFalse(collector.qualifies(row))
        row = self.row(1, matches=[{}]*23)
        self.assertFalse(collector.qualifies(row))

    def test_teacher_prompts_are_diverse_but_learner_spec_unchanged(self):
        spec = cardinal_model_spec(); prompts = [collector.teacher_prompt(spec, i) for i in range(32)]
        self.assertEqual(len(set(prompts)), 32)
        self.assertTrue(all(p.endswith(spec) for p in prompts))
        self.assertTrue(all('return \'rest\'' not in p[:p.index(spec)] for p in prompts))
        self.assertLessEqual(sum(client.reservation(p, i) for i, p in enumerate(prompts)), 2)

    def test_map_families_disjoint_in_seed_and_exact_observable_layout(self):
        timetable = collector.schedule()
        result = collector.map_fingerprints(program_arena, timetable, [])
        all_rows = [r for rows in result['families'].values() for r in rows]
        self.assertEqual(len(all_rows), 218)
        self.assertEqual(len({r['sha256'] for r in all_rows}), 218)
        prior = {'prior_seed_fingerprints': {'999999': all_rows[0]['sha256']}, 'families': {}}
        with self.assertRaisesRegex(ValueError, 'overlaps'): collector.map_fingerprints(program_arena, timetable, [prior])
        bad = copy.deepcopy(timetable); bad['holdout_maps'] = bad['teacher_maps']
        with self.assertRaisesRegex(ValueError, 'overlaps'): collector.map_fingerprints(program_arena, bad, [])

    def test_delivery_receipt_crash_recovery_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.chdir(temp):
            Path('runs').mkdir(); Path('budget.json').write_text(json.dumps({
                'cloud_spend_to_date': 0, 'cloud_reserved_usd': 80, 'approved_pilot_cap': 100}))
            run = Path(temp)/'fixture-low'
            with patch.object(client, 'verify_endpoint', return_value={'explicit_fixture': True}):
                collector.prepare(run, collector.ROOT/'runs/cardinal-teacher-low-v1',
                    collector.ROOT/'runs/program-cardinal-teacher-low-ledger.json')
                plan = collector.verify_plan(run)
                with patch.object(client.gateway, 'api_key', return_value='fixture-secret'), patch.object(
                    client.urllib.request, 'build_opener', return_value=SimpleNamespace(open=lambda *a, **k: response())) as calls:
                    first = client.complete((run/plan['teacher_prompts']['0']['path']).read_text(), 0, collector.sha(run/'plan.json'))
                    collector.write_new(run/'generations/teacher-00.json', first)
                    self.assertFalse((run/'delivery-gate.json').exists())
                    collector.advance(run)
                    self.assertTrue(collector.read(run/'delivery-gate.json')['passed'])
                    self.assertEqual(calls.call_count, 2)
                    gate = collector.read(run/'delivery-gate.json'); gate['passed'] = False
                    (run/'delivery-gate.json').write_text(json.dumps(gate))
                    with self.assertRaisesRegex(ValueError, 'delivery gate differs'): collector.advance(run)
                    self.assertEqual(calls.call_count, 2)

    def test_high_stop_and_ungarded_map_reuse_bound_offline(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)/'fixture-low'
            with patch.object(client, 'verify_endpoint', return_value={'explicit_fixture': True}):
                collector.prepare(run, collector.ROOT/'runs/cardinal-teacher-low-v1',
                    collector.ROOT/'runs/program-cardinal-teacher-low-ledger.json')
            plan = collector.verify_plan(run)
            self.assertEqual(plan['schedule']['teacher_generation_seeds'], list(range(131000000,131000032)))
            high = run/'inputs/high-cohort'
            self.assertEqual(plan['map_fingerprints_sha256'], collector.sha(high/'map-fingerprints.json'))
            with patch.object(collector, 'ROOT', Path(temp)/'absent'):
                collector.verify_plan(run, require_active=False)
            (high/'evaluation').mkdir()
            with self.assertRaisesRegex(ValueError, 'gameplay grades'): collector.verify_plan(run)
            (high/'evaluation').rmdir()
            stop = collector.read(high/'administrative-stop-final.json'); stop['no_gameplay_scoring'] = False
            (high/'administrative-stop-final.json').write_text(json.dumps(stop))
            with self.assertRaisesRegex(ValueError, 'administrative stop'): collector.verify_plan(run)

    def test_low_failure_audit_is_required_and_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)/'fixture-none'
            with patch.object(client, 'verify_endpoint', return_value={'explicit_fixture': True}):
                collector.prepare(run, collector.ROOT/'runs/cardinal-teacher-low-v1',
                    collector.ROOT/'runs/program-cardinal-teacher-low-ledger.json')
            low = run/'inputs/low-cohort'
            plan = collector.verify_plan(run)
            self.assertEqual(plan['map_reuse']['low_delivery_audit_sha256'], collector.sha(low/'independent-delivery-audit.json'))
            (low/'evaluation').mkdir()
            with self.assertRaisesRegex(ValueError, 'Low cohort has gameplay grades'): collector.verify_plan(run)
            (low/'evaluation').rmdir()
            audit = collector.read(low/'independent-delivery-audit.json'); audit['generation_sha256']='f'*64
            (low/'independent-delivery-audit.json').write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, 'Low completed response'): collector.verify_plan(run)

    def test_complete_fixture_corpus_replay_token_and_offline_receipt_binding(self):
        """Handwritten fixture only; never a demonstration in a real corpus."""
        tokenizer = collector.ROOT/'runs/cardinal-teacher-inputs/tokenizer.json'
        if not tokenizer.exists():
            self.skipTest('Pinned public tokenizer fixture is unavailable')
        source = '''def act():
    m15 = VARIANT
    if self_x == home_x and self_y == home_y and carrying > 0:
        return 'deposit'
    if energy == 0:
        return 'rest'
    tx = home_x
    ty = home_y
    if carrying == 0:
        best = 1000000
        for i in range(6):
            if site(i, 'units') > 0:
                d = abs(self_x-site(i,'x'))+abs(self_y-site(i,'y'))
                if d < best:
                    best = d
                    tx = site(i,'x')
                    ty = site(i,'y')
    if self_x < tx:
        return 'east'
    if self_x > tx:
        return 'west'
    if self_y < ty:
        return 'south'
    if self_y > ty:
        return 'north'
    return 'harvest'
'''
        with tempfile.TemporaryDirectory() as temp, contextlib.chdir(temp):
            Path('runs').mkdir(); Path('budget.json').write_text(json.dumps({
                'cloud_spend_to_date': 0, 'cloud_reserved_usd': 80, 'approved_pilot_cap': 100}))
            run = Path(temp)/'fixture-corpus'
            with patch.object(client, 'verify_endpoint', return_value={'explicit_fixture': True}):
                collector.prepare(run, collector.ROOT/'runs/cardinal-teacher-low-v1', collector.ROOT/'runs/program-cardinal-teacher-low-ledger.json')
                plan = collector.verify_plan(run)
                calls = []
                def fake(*args, **kwargs):
                    index = len(calls); calls.append(index)
                    code = source.replace('VARIANT', str(index)) if index < 16 else 'def act():\n    break\n'
                    return response(choices=[{'finish_reason': 'stop', 'message': {'content': code}}])
                with patch.object(client.gateway, 'api_key', return_value='explicit-fixture-secret'), patch.object(
                    client.urllib.request, 'build_opener', return_value=SimpleNamespace(open=fake)):
                    for i in range(32): collector.advance(run)
                self.assertEqual(calls, list(range(32)))
                result = collector.evaluate(run)
                self.assertEqual(result['selected_indices'], list(range(16)))
                # Portable verification must work without the original ledger,
                # API credentials, or active source tree.
                client.LEDGER.unlink()
                with patch.object(collector, 'ROOT', Path(temp)/'absent-active-tree'):
                    verified = collector.verify_corpus(run, replay=True, require_active=False)
                self.assertEqual(len(verified['demonstrations']), 16)
                self.assertTrue(verified['replays_independently_verified'])
                self.assertEqual(verified['demonstrations'][0]['messages'][0]['content'], cardinal_model_spec())
                self.assertEqual(verified['demonstrations'][0]['completion_ids'][-1], 248046)
                altered = collector.read(run/'corpus.json'); altered['demonstrations'][0]['completion_ids'][0] += 1
                (run/'corpus.json').write_text(json.dumps(altered))
                with self.assertRaisesRegex(ValueError, 'demonstrations differ'):
                    collector.verify_corpus(run, replay=False)


if __name__ == '__main__':
    unittest.main()
