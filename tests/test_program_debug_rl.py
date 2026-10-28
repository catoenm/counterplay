"""Independent CPU integration fixtures; no neural model, API, or rental.

Fake tensors/likelihoods only exercise journal mechanics. The small practice
tests use actual bounded game replays; larger orchestration tests label and
mock scores rather than claim them as scientific evidence.
"""
from contextlib import contextmanager
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from selfplay import program_debug as harness
from selfplay import program_debug_rl as debug
from selfplay import program_cardinal_gpu as gpu
from selfplay.program_debug_cuda import DebugCudaPolicy, flatten_trajectories
from tests.test_program_cardinal_gpu import FakePolicy, deployment, fake_identity


IDLE = "def act():\n    return 'rest'\n"
ORIGINAL_TOKENIZER = debug.ROOT/'runs/cardinal-debug-preflight-v1/tokenizer'


class FakeDebugPolicy(FakePolicy):
    def __init__(self, invalid_drafts=False):
        super().__init__()
        self.invalid_drafts = invalid_drafts
        self.sample_calls, self.updated_turns = [], []
        self.fail_sample = False

    def sample_messages_batch(self, contexts, seeds):
        self.sample_calls.append(copy.deepcopy((contexts, seeds)))
        if self.fail_sample:
            self.fail_sample = False
            raise RuntimeError('Fixture interrupted sampling; outcome unknown')
        records = []
        for i, (context, seed) in enumerate(zip(contexts, seeds)):
            is_draft = len(context) == 1
            # Alternate invalid FINAL revisions to ensure no draft substitution.
            final = IDLE if is_draft or i % 2 else 'invalid final controller'
            text = ('fixture private thoughts\n</think>\n' + final)
            if is_draft and self.invalid_drafts:
                text = 'fixture truncated thoughts with no final boundary'
            ids = [30+i, 99]
            records.append({'text': text, 'raw_text': text+'<eos>', 'messages': copy.deepcopy(context),
                'seed': seed, 'finish_reason': 'eos', 'prompt_ids': [1, 2] if is_draft else [1, 2, 3],
                'completion_ids': ids, 'sampled_token_logprobs': [-.5, -.7],
                'input_tokens': 2 if is_draft else 3, 'output_tokens': 2,
                'eos_token_ids': [99], 'sampling_batch_size': 8, 'sampling_batch_row': i,
                'thinking_enabled': True, 'fixture_only': True})
        return {'records': records, 'timing': {'batch_elapsed_seconds': .1, 'total_output_tokens': 16}}

    def update_trajectories(self, records, advantages):
        self.updated_turns.append(copy.deepcopy(records))
        flat, weights = flatten_trajectories(records, advantages)
        assert len(flat) == len(weights) == 16
        self.step += 1
        self.weight += .01
        self.updates += 1
        return {'optimizer_steps': 1, 'adapter_l2_delta': .01,
                'gradient_likelihood_gate': self.likelihood_gate(flat), 'loss': 1.,
                'trajectory_count': 8, 'sampled_turn_count': 16, 'tool_feedback_target_tokens': 0}


@contextmanager
def fixture(invalid_drafts=False):
    with tempfile.TemporaryDirectory() as d, patch.object(debug, 'SOURCES', []), patch.object(gpu, 'checkpoint_state', fake_identity):
        run = Path(d)
        dep = deployment()
        cfg = debug.config(dep)
        for name, value in [('deployment.json', dep), ('plan.json', debug.plan(dep)),
                            ('schedule.json', debug.schedule()), ('map-fingerprints.json', {})]:
            debug.common.atomic_json(run/name, value)
        (run/'prompts').mkdir()
        (run/'prompts/debug.txt').write_text(harness.debug_model_spec())
        backend = FakeDebugPolicy(invalid_drafts)
        files = backend.save_checkpoint(run/'inputs/initial-checkpoint')
        ir = harness.classify_debug('</think>'+IDLE, 'eos')[0]['program']['compiled_ir']
        roster = {'selected': [{'id': name, 'origin': 'scripted' if i < 2 else 'model', 'compiled_ir': ir,
                                'ir_sha256': str(i)} for i, name in enumerate(('nearest', 'value', 'teacher0', 'teacher1'))]}
        debug.common.atomic_json(run/'initial-opponents.json', roster)
        inputs = {name: debug.common.digest(run/name) for name in ('deployment.json', 'prompts/debug.txt', 'initial-opponents.json')}
        manifest = {'kind': cfg['kind'], 'config': cfg, 'source_sha256': {}, 'input_sha256': inputs,
                    'initial_checkpoint_files': files, 'prompt_sha256': {'debug': inputs['prompts/debug.txt']}}
        debug.common.atomic_json(run/'manifest.json', manifest)
        budget, _ = debug.journal.execution_budget(cfg)
        debug.common.atomic_json(run/'status.json', {'manifest_sha256': debug.common.digest(run/'manifest.json'),
            'artifacts': {}, 'state': 'prepared', 'claims_allowed': False, 'execution_budget': budget})
        engine = debug.DebugCycle(run, backend)
        engine.initialize()
        yield run, engine, backend


def trajectories(engine, name='fixture'):
    return engine.trajectories(name, [name+'-'+str(i) for i in range(8)],
        [171000000+i*2 for i in range(8)], 'base', [[172000000, 172000001]]*8)


def fake_score(record, *args):
    classification, _ = harness.classify_debug(record['text'], record['finish_reason'])
    valid = classification['valid']
    return {'classification': classification, 'debug_classification': classification,
            'runtime_valid': valid, 'authorship_valid': valid, 'reward': .2 if valid else -1.,
            'game_return': .2 if valid else None, 'matches': [], 'candidate_program_faults': 0,
            'opponent_program_faults': 0, 'banked_point_yield': 0., 'positive_banking_games': 0,
            'fixture_only': True}


class RuntimeReportTests(unittest.TestCase):
    def test_all_verified_versions_recorded_without_mutating_base_backend(self):
        backend = {'versions': {'torch': '2.8.0+cu128', 'transformers': '5.20.2'},
                   'cuda': '12.8', 'trainable_parameters': 12, 'model_index': 'stale fixture'}
        original = copy.deepcopy(backend)
        index = {'repo': debug.MODEL_REPO, 'revision': debug.MODEL_REVISION}
        report = debug.make_runtime_report(index, backend, dict(debug.VERSIONS))
        self.assertEqual(report['versions'], debug.VERSIONS)
        self.assertEqual(report['versions']['tokenizers'], '0.23.3')
        self.assertEqual(report['versions']['safetensors'], '0.8.0')
        self.assertEqual(report['model_index'], index)
        self.assertEqual(report['trainable_parameters'], 12)
        self.assertTrue(report['thinking'])
        self.assertEqual(report['authoring_calls'], 2)
        self.assertEqual(backend, original)

    def test_missing_or_conflicting_runtime_versions_rejected(self):
        missing = dict(debug.VERSIONS)
        missing.pop('tokenizers')
        with self.assertRaisesRegex(ValueError, 'Exact CUDA'):
            debug.make_runtime_report({}, {'versions': {}}, missing)
        with self.assertRaisesRegex(ValueError, 'Backend dependency'):
            debug.make_runtime_report({}, {'versions': {'torch': '2.7.0'}}, debug.VERSIONS)


class DebugRunnerIntegrationTests(unittest.TestCase):
    def test_actual_practice_and_two_turn_journal_resume_without_resample(self):
        with fixture() as (run, engine, backend):
            rows = trajectories(engine)
            self.assertEqual(len(backend.sample_calls), 2)
            self.assertEqual([len(c[0][0]) for c in backend.sample_calls], [1, 3])
            for i, row in enumerate(rows):
                self.assertEqual(row['final_turn_index'], 1)
                self.assertEqual(row['text'], row['turns'][1]['text'])
                self.assertEqual(row['training_tokens'], 4)
                self.assertEqual(row['turn_sha256'], [gpu.object_hash(t) for t in row['turns']])
                self.assertEqual([t['seed'] for t in row['turns']], [171000000+i*2, 171000001+i*2])
                self.assertIn('fixture private thoughts', row['turns'][0]['text'])
                self.assertEqual(row['turns'][1]['messages'][1]['content'], '\n'+IDLE)
                self.assertNotIn('fixture private thoughts', row['turns'][1]['messages'][1]['content'])
                evidence = engine.existing(row['feedback_path'])
                self.assertEqual(len(evidence['matches']), 8)
                self.assertEqual(evidence['record'], row['turns'][0])
                self.assertEqual(row['feedback_sha256'], engine.status['artifacts'][row['feedback_path']])
            resumed = debug.DebugCycle(run, backend)
            self.assertEqual(trajectories(resumed), rows)
            self.assertEqual(len(backend.sample_calls), 2)

    def test_interruption_after_draft_feedback_reuses_draft_and_generates_one_revision(self):
        with fixture(invalid_drafts=True) as (run, engine, backend):
            original = engine.commit
            def crash(relative, value):
                if relative == 'feedback/fixture-4.json':
                    raise RuntimeError('Fixture interruption after four feedback records')
                return original(relative, value)
            with patch.object(engine, 'commit', crash), self.assertRaisesRegex(RuntimeError, 'four'):
                trajectories(engine)
            self.assertEqual(len(backend.sample_calls), 1)
            rows = trajectories(debug.DebugCycle(run, backend))
            self.assertEqual(len(rows), 8)
            self.assertEqual(len(backend.sample_calls), 2)
            self.assertEqual(rows[0]['turns'][1]['messages'][1]['content'], rows[0]['turns'][0]['text'])

    def test_interruption_after_revision_batch_materializes_without_third_call(self):
        with fixture(invalid_drafts=True) as (run, engine, backend):
            original = engine.commit
            def crash(relative, value):
                if relative == 'generations/fixture-4.json':
                    raise RuntimeError('Fixture interruption after four trajectories')
                return original(relative, value)
            with patch.object(engine, 'commit', crash), self.assertRaisesRegex(RuntimeError, 'four'):
                trajectories(engine)
            self.assertEqual(len(backend.sample_calls), 2)
            rows = trajectories(debug.DebugCycle(run, backend))
            self.assertEqual(len(rows), 8)
            self.assertEqual(len(backend.sample_calls), 2)

    def test_unknown_sampling_intent_cannot_redraw(self):
        with fixture(invalid_drafts=True) as (run, engine, backend):
            backend.fail_sample = True
            with self.assertRaisesRegex(RuntimeError, 'unknown'):
                trajectories(engine)
            self.assertIn('batches/fixture-draft-intent.json', engine.status['artifacts'])
            with self.assertRaisesRegex(ValueError, 'never redraw'):
                trajectories(debug.DebugCycle(run, backend))
            self.assertEqual(len(backend.sample_calls), 1)

    def test_score_uses_only_final_revision_and_keeps_real_failure_classification(self):
        opponents = [{'id': 'nearest', 'origin': 'scripted', 'compiled_ir': debug.common.reference_programs()['nearest']}]
        cfg = {'max_rounds': 48}
        draft = {'text': 'private\n</think>\n'+IDLE, 'finish_reason': 'eos'}
        revision = {'text': 'private\n</think>\ninvalid final controller', 'finish_reason': 'eos', 'turns': [draft]}
        result = debug.score(revision, opponents, [172000010], cfg)
        self.assertEqual(result['reward'], -1)
        self.assertEqual(result['matches'], [])
        self.assertFalse(result['debug_classification']['valid'])
        good = debug.score(draft, opponents, [172000010], cfg)
        self.assertEqual(len(good['matches']), 2)
        self.assertEqual(good['classification']['program']['python_source'], '\n'+IDLE)
        self.assertTrue(good['runtime_valid'])

    def test_changed_context_and_bad_batch_identity_are_rejected(self):
        with fixture(invalid_drafts=True) as (run, engine, backend):
            contexts = [harness.initial_messages()]*8
            seeds = list(range(8))
            batch = backend.sample_messages_batch(contexts, seeds)
            batch['records'][0]['messages'] = [{'role': 'user', 'content': 'wrong'}]
            with self.assertRaisesRegex(ValueError, 'context/seed/identity'):
                debug.validate_batch(batch, contexts, seeds, engine.config)
            rows = trajectories(engine)
            changed = copy.deepcopy(rows[0]['turns'][0])
            changed['text'] += 'different'
            with self.assertRaises(ValueError):
                harness.revision_messages(harness.initial_messages(), changed, engine.existing(rows[0]['feedback_path']))

    def test_zero_signal_copies_exact_checkpoint_and_mixed_signal_updates_once(self):
        with fixture(invalid_drafts=True) as (run, engine, backend):
            initial = engine.existing('opponents.json')
            base = engine.checkpoint('base')['files']
            def zero(record, *args):
                row = fake_score(record); row['reward'] = -1.; return row
            with patch.object(debug, 'score', zero):
                report = engine.group(1, engine.roster(1, initial))
            self.assertEqual(report['optimizer_steps'], 0)
            self.assertEqual(engine.checkpoint('cycle1')['files'], base)
            self.assertEqual(report['training_tokens'], 32)
            self.assertEqual(backend.updates, 0)
            with patch.object(debug, 'score', fake_score):
                report = engine.group(2, engine.roster(2, initial))
            self.assertEqual(report['optimizer_steps'], 1)
            self.assertEqual(report['training_tokens'], 32)
            self.assertEqual(report['objective_divisor'], 8)
            self.assertEqual(backend.updates, 1)
            self.assertEqual(len(backend.updated_turns[0]), 8)
            self.assertEqual(engine.checkpoint('cycle2')['optimizer_identity']['step'], 1)
            with patch.object(debug, 'score', fake_score):
                engine.group(2, engine.roster(2, initial))
            self.assertEqual(backend.updates, 1)

    def test_full_fixed_twentyfive_group_fake_run_seals_eval_before_scoring(self):
        with fixture(invalid_drafts=True) as (run, engine, backend):
            def scoring(record, *args):
                if record['name'].startswith('eval-'):
                    seal = engine.existing('evaluation/sealed.json')
                    self.assertEqual(seal['draws'], 64)
                    self.assertEqual(seal['sampled_turns'], 128)
                    self.assertEqual(len(seal['generations']), 64)
                return fake_score(record)
            with patch.object(debug, 'score', scoring), patch.object(debug, 'endpoint_decision', return_value={'passed': True, 'endpoint': 25}):
                engine.run_all()
            self.assertEqual(engine.status['state'], 'complete')
            self.assertEqual(engine.status['attempted_groups'], 25)
            self.assertEqual(engine.status['actual_optimizer_steps'], 25)
            self.assertEqual(backend.updates, 25)
            self.assertEqual(len(backend.sample_calls), 2*(25+8))
            self.assertEqual(engine.checkpoint('cycle25')['optimizer_identity']['step'], 25)
            self.assertEqual(len(engine.existing('evaluation/base/holdout.json')['scores']), 32)
            self.assertEqual(len(engine.existing('evaluation/cycle25/holdout.json')['scores']), 32)

    @unittest.skipUnless((ORIGINAL_TOKENIZER/'tokenizer.json').exists(), 'Authoritative original tokenizer fixture absent')
    def test_actual_native_tokenizer_prefix_has_no_reasoning_duplication_or_feedback_targets(self):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            self.skipTest('Tokenizer library absent; no installation or network')
        tokenizer = AutoTokenizer.from_pretrained(str(ORIGINAL_TOKENIZER), local_files_only=True)
        policy = object.__new__(DebugCudaPolicy)
        policy.tokenizer = tokenizer
        draft = {'text': 'fixture private thoughts\n</think>\n'+IDLE, 'finish_reason': 'eos'}
        evidence = harness.practice_feedback(draft, [172000000, 172000001])
        messages = harness.revision_messages(harness.initial_messages(), draft, evidence)
        rendered, ids = policy.template(messages)
        self.assertEqual(tokenizer.decode(ids, skip_special_tokens=False), rendered)
        self.assertTrue(rendered.endswith('<think>\n'))
        self.assertNotIn('fixture private thoughts', rendered)
        # The pinned native chat template trims surrounding content whitespace;
        # the stored message retains it and no source tokens are rewritten.
        self.assertEqual(messages[1]['content'], '\n'+IDLE)
        self.assertEqual(rendered.count(IDLE.strip()), 1)
        self.assertIn('PRACTICE TEST RESULT', rendered)
        self.assertNotIn('154000000', rendered)  # heldout seed family stays absent
        self.assertLess(len(ids)+1536, 8192)
        boundary = tokenizer.encode('</think>', add_special_tokens=False)
        self.assertEqual(tokenizer.decode(boundary, skip_special_tokens=True), '</think>')
        turn = {'completion_ids': [1, 2], 'sampled_token_logprobs': [-.2, -.3], 'prompt_ids': ids}
        flat, weights = flatten_trajectories([{'turns': [turn, turn]}], [.5])
        self.assertEqual(weights, [1., 1.])
        self.assertEqual(sum(len(t['completion_ids']) for t in flat), 4)


if __name__ == '__main__':
    unittest.main()
