"""CPU Torch/tokenizer tests only; no pretrained-model, CUDA or network calls."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from selfplay import program_debug_calibration_cuda as backend
from selfplay import program_debug_calibration_harness as harness
from tests.test_program_cardinal_cuda import HAS_TORCH, fixture
from tests.test_program_debug_cuda import sampler as previous_sampler

if HAS_TORCH:
    import torch

ORIGINAL_TOKENIZER = Path(__file__).resolve().parents[1]/'runs/cardinal-debug-preflight-v1/tokenizer'


def sampler():
    old = previous_sampler()
    p = object.__new__(backend.CalibrationCudaPolicy)
    p.__dict__.update(old.__dict__)
    original = p.tokenizer.apply_chat_template
    def apply(messages, tokenize, add_generation_prompt, enable_thinking, reasoning_effort):
        assert reasoning_effort == 'low'
        return '<|im_start|>system\nReasoning effort is set to low.\n<|im_end|>\n'+original(messages, tokenize, add_generation_prompt, enable_thinking)
    p.tokenizer.apply_chat_template = apply
    return p


@unittest.skipUnless(HAS_TORCH, 'Optional CPU Torch fixture unavailable')
class CalibrationBackendTests(unittest.TestCase):
    def test_two_row_exact_rng_and_low_effort_metadata(self):
        contexts = [[{'role':'user','content': 'x'*i}] for i in (1, 9)]
        p = sampler(); batch = p.sample_messages_batch(contexts, [23, 24])
        self.assertEqual(len(batch['records']), 2)
        for i, row in enumerate(batch['records']):
            separate = sampler().sample_messages_batch([contexts[i]], [23+i])['records'][0]
            for key in ('completion_ids', 'sampled_token_logprobs', 'prompt_ids', 'finish_reason'):
                self.assertEqual(row[key], separate[key])
            self.assertEqual(row['reasoning_effort'], 'low')
            self.assertTrue(row['calibration_only'])
            self.assertEqual(row['completion_token_limit'], p.config['max_new_tokens'])
        self.assertTrue(all(len(call[0]) == 2 for call in p.model.calls))
        quoted = [{'role':'user','content':'task'},
                  {'role':'assistant','content':'Earlier quoted text: Reasoning effort is set to xhigh.'},
                  {'role':'user','content':'Revise this draft.'}]
        rendered, _ = p.template(quoted)
        self.assertTrue(rendered.startswith('<|im_start|>system\nReasoning effort is set to low.'))

    def test_large_batches_missing_effort_and_context_overflow_rejected(self):
        p = sampler(); context = [{'role':'user','content':'x'}]
        for seeds in ([], [1,2,3]):
            with self.assertRaisesRegex(ValueError, 'one or two'):
                p.sample_messages_batch([context]*len(seeds), seeds)
        _, ids = p.template(context)
        p.config['max_context_tokens'] = len(ids)+p.config['max_new_tokens']-1
        with self.assertRaisesRegex(ValueError, 'context'):
            p.sample_messages_batch([context], [1])
        p.tokenizer.apply_chat_template = lambda *args, **kwargs: 'Reasoning effort is set to xhigh.<think>'
        with self.assertRaisesRegex(ValueError, 'explicit low'):
            p.template(context)

    def test_optimizer_entrypoints_are_disabled(self):
        p = sampler()
        for method in (p.update, p.update_trajectories):
            with self.assertRaisesRegex(RuntimeError, 'never performs optimizer'):
                method([], [])

    def test_two_row_derivative_smoke_preserves_all_state_and_failure_evidence(self):
        p = object.__new__(backend.CalibrationCudaPolicy); p.__dict__.update(fixture().__dict__)
        p.config.update(seed=225401, max_new_tokens=4096, token_logprob_tolerance=.002,
                        sequence_logprob_tolerance=.02)
        p.model.train(True)
        for value in p.parameters.values():
            value.grad = torch.ones_like(value)*.3
        grads = {n: v.grad.clone() for n,v in p.parameters.items()}
        config, cpu_rng = p.config, torch.get_rng_state().clone()
        gpu_rng = [cpu_rng.clone()]; calls = []
        def sample(contexts, seeds):
            calls.append((deepcopy(contexts), list(seeds), p.config['max_new_tokens']))
            torch.manual_seed(999)
            return {'records':[{'completion_ids':[1,2], 'sampled_token_logprobs':[-.5,-.7]}
                               for _ in seeds], 'timing':{'batch_elapsed_seconds':.1}}
        p.sample_messages_batch = sample
        def logps(row):
            total = sum(v.sum() for v in p.parameters.values())
            return torch.tensor(row['sampled_token_logprobs'])+(total-total.detach())
        p.token_logps = logps
        contexts = [[{'role':'user','content': 'a'}], [{'role':'user','content':'longer fixture'}]]
        with patch.object(torch.cuda, 'get_rng_state_all', side_effect=lambda:[v.clone() for v in gpu_rng]), \
             patch.object(torch.cuda, 'set_rng_state_all', side_effect=lambda values:gpu_rng.__setitem__(slice(None),values)):
            good = p.mechanics_smoke({'batch_messages':contexts})
            self.assertTrue(good['passed'], good)
            self.assertEqual(good['batch_size'], 2)
            self.assertEqual(good['optimizer_steps'], 0)
            self.assertTrue(good['derivative']['finite_nonzero'])
            self.assertEqual(calls[0][2],32);self.assertEqual(len(calls[0][1]),2)
            self.assertEqual(p._optimizer_identity()['step'],0)
            self.assertIs(p.config, config);self.assertTrue(p.model.training)
            self.assertTrue(torch.equal(torch.get_rng_state(), cpu_rng))
            for n,v in p.parameters.items():self.assertTrue(torch.equal(v.grad,grads[n]))
            p.token_logps = lambda row: logps(row)+.1
            failed = p.mechanics_smoke({'batch_messages':contexts})
            self.assertFalse(failed['passed'])
            for key in ('adapter_unchanged','optimizer_unchanged','rng_restored','model_mode_restored','config_restored'):
                self.assertTrue(failed[key])


def context_stress(tokenizer):
    """Deterministic synthetic 4096-token-or-shorter cases, never model outputs."""
    policy = object.__new__(backend.CalibrationCudaPolicy)
    policy.tokenizer = tokenizer
    initial = harness.initial_messages()
    def fit(prefix, repeated, suffix, cap=4095):
        low, high = 0, 100000
        while low < high:
            mid = (low+high+1)//2
            if len(tokenizer.encode(prefix+repeated*mid+suffix, add_special_tokens=False)) <= cap:
                low = mid
            else:
                high = mid-1
        return prefix+repeated*low+suffix
    cases = [
        ('long_valid_comments', fit('brief analysis\n</think>\ndef act():\n', '    # a\n', "    return 'rest'\n"), 'eos'),
        ('runtime_fault', "brief analysis\n</think>\ndef act():\n    return 'north'\n", 'eos'),
        ('long_unsupported_identifier', fit('brief analysis\n</think>\ndef act():\n    ', 'a', " = 1\n    return 'rest'\n"), 'eos'),
        ('deep_nesting_error', 'brief analysis\n</think>\ndef act():\n    x = '+('('*250)+'1'+(')'*250)+"\n    return 'rest'\n", 'eos'),
        ('unterminated_string', fit('brief analysis\n</think>\ndef act():\n    x = "', ' a', ''), 'eos'),
        ('missing_boundary', fit('', 'thought ', '', cap=4096), 'length'),
        ('ambiguous_boundaries', fit('quoted </think> then ', 'thought ', ' </think>', cap=4096), 'length'),
        ('over_byte_limit_source', 'brief analysis\n</think>\ndef act():\n    # '+' '*17000+"\n    return 'rest'\n", 'eos'),
    ]
    rows=[]
    for name,text,finish in cases:
        ids=tokenizer.encode(text,add_special_tokens=False)
        if finish=='eos':ids += [tokenizer.eos_token_id]
        assert len(ids)<=4096, (name,len(ids))
        draft={'text':tokenizer.decode(ids,skip_special_tokens=True),
            'raw_text':tokenizer.decode(ids,skip_special_tokens=False),'completion_ids':ids,
            'finish_reason':finish,'output_tokens':len(ids),'completion_token_limit':4096,
            'reasoning_effort':'low','fixture_only':True}
        evidence=harness.practice_feedback(draft,[171000000,171000001])
        messages=harness.revision_messages(initial,draft,evidence)
        _,prefix=policy.template(messages)
        rows.append({'fixture':name,'sampled_tokens':len(ids),'source_bytes':len(text.encode()),
            'classification':evidence['classification'].get('reason','valid'),
            'compiler_error_chars':len(evidence['classification'].get('detail','')),
            'practice_games':len(evidence['matches']),'revision_prefix_tokens':len(prefix),
            'completion_reserve':4096,'total_context_required':len(prefix)+4096,
            'within_16384':len(prefix)+4096<=16384})
    return rows


class OriginalTokenizerTests(unittest.TestCase):
    @unittest.skipUnless((ORIGINAL_TOKENIZER/'tokenizer_config.json').exists(), 'Original HF tokenizer absent')
    def test_native_low_effort_and_final_only_history(self):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            self.skipTest('Optional tokenizer dependency absent')
        tok = AutoTokenizer.from_pretrained(str(ORIGINAL_TOKENIZER), local_files_only=True)
        p = object.__new__(backend.CalibrationCudaPolicy);p.tokenizer=tok
        initial=harness.initial_messages();rendered,ids=p.template(initial)
        self.assertIn('Reasoning effort is set to low.',rendered)
        self.assertNotIn('Reasoning effort is set to xhigh.',rendered)
        self.assertTrue(rendered.endswith('<think>\n'))
        self.assertEqual(tok.decode(ids,skip_special_tokens=False),rendered)
        draft={'text':"private reasoning\n</think>\ndef act():\n    return 'rest'\n",'finish_reason':'eos'}
        feedback=harness.practice_feedback(draft,[171000000,171000001])
        messages=harness.revision_messages(initial,draft,feedback)
        rendered,ids=p.template(messages)
        self.assertNotIn('private reasoning',rendered)
        self.assertEqual(rendered.count('def act():'),2)  # specification and exactly one draft
        self.assertLess(len(ids)+4096,16384)
        self.assertEqual(tok.decode(tok.encode('</think>',add_special_tokens=False),skip_special_tokens=True),'</think>')

    @unittest.skipUnless((ORIGINAL_TOKENIZER/'tokenizer_config.json').exists(), 'Original HF tokenizer absent')
    def test_stressed_delivery_and_feedback_fit_full_revision_reserve(self):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            self.skipTest('Optional tokenizer dependency absent')
        tokenizer=AutoTokenizer.from_pretrained(str(ORIGINAL_TOKENIZER),local_files_only=True)
        rows=context_stress(tokenizer)
        self.assertEqual(len(rows),8)
        self.assertTrue(all(row['within_16384'] for row in rows),rows)
        self.assertGreaterEqual(max(row['sampled_tokens'] for row in rows),4095)
        self.assertTrue(any(row['practice_games']==8 for row in rows))
        self.assertTrue(any(row['compiler_error_chars'] for row in rows))


if __name__ == '__main__':
    unittest.main()
