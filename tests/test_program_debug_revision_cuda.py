"""CPU tensor and pinned-tokenizer fixtures; no model downloads or GPU calls."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from selfplay import program_debug_revision_cuda as backend
from selfplay import program_debug_revision_harness as harness
from tests.test_program_debug_cuda import sampler as old_sampler
from tests.test_program_cardinal_cuda import HAS_TORCH, fixture

if HAS_TORCH:import torch
ORIGINAL_TOKENIZER=Path(__file__).resolve().parents[1]/'runs/cardinal-debug-preflight-v1/tokenizer'


def sampler():
    old=old_sampler();p=object.__new__(backend.RevisionCudaPolicy);p.__dict__.update(old.__dict__)
    def apply(messages,tokenize,add_generation_prompt,enable_thinking,reasoning_effort=None):
        assert not tokenize and add_generation_prompt
        header='<|im_start|>system\nReasoning effort is set to low.\n<|im_end|>\n' if enable_thinking else ''
        if enable_thinking:assert reasoning_effort=='low'
        else:assert reasoning_effort is None
        history=''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n' for m in messages)
        tail='<|im_start|>assistant\n'+('<think>\n' if enable_thinking else '<think>\n\n</think>\n\n')
        return header+history+tail
    p.tokenizer.apply_chat_template=apply;p.config['max_context_tokens']=16384
    return p


def replay_fixture():
    p=object.__new__(backend.RevisionCudaPolicy);p.__dict__.update(fixture().__dict__)
    p.config.update(seed=325401,max_new_tokens=15,max_context_tokens=16384,sample_seconds=30,
                    token_logprob_tolerance=.002,sequence_logprob_tolerance=.02)
    p.tokenizer=sampler().tokenizer;p.eos=[0]
    records=[]
    for i in range(8):
        contexts=[[{'role':'user','content':'x'*(i+1)}],[{'role':'user','content':'question'},
                   {'role':'assistant','content':'def act(): return "rest"'},
                   {'role':'user','content':'Recorded factual fixture feedback '+str(i)}]]
        records.extend(sampler()._sample_native_batch(contexts,[100+i*2,101+i*2],['draft','revision'])['records'])
    reference={r['seed']:list(r['sampled_token_logprobs']) for r in records}
    calls=[]
    def logps(record):
        calls.append((record['seed'],p.model.training,torch.is_grad_enabled()))
        total=sum(v.sum() for v in p.parameters.values())
        return torch.tensor(reference[record['seed']])+(total-total.detach())
    p.token_logps=logps
    return p,records,calls


@unittest.skipUnless(HAS_TORCH,'Optional CPU Torch unavailable')
class RevisionBackendTests(unittest.TestCase):
    def test_mixed_prefix_padding_independent_rng_and_post_eos_continuation(self):
        contexts=[[{'role':'user','content':'short'}],[{'role':'user','content':'different longer context'}]]
        observed_continuation=False
        for first_seed in (0,2,4,6):
            p=sampler();batch=p._sample_native_batch(contexts,[first_seed,first_seed+1],['draft','revision'])
            self.assertTrue(all(len(c[0])==2 for c in p.model.calls))
            self.assertEqual(batch['timing']['authoring_stages'],['draft','revision'])
            for i,row in enumerate(batch['records']):
                single=sampler().sample_messages_batch([contexts[i]],[first_seed+i],authoring_stage=['draft','revision'][i])['records'][0]
                for key in ('prompt_ids','completion_ids','sampled_token_logprobs','finish_reason'):
                    self.assertEqual(row[key],single[key])
                self.assertTrue(p.validate_record(row))
                expected=len(batch['timing']['active_sequences_by_step'])-row['output_tokens'] if row['finish_reason']=='eos' else 0
                self.assertEqual(row['post_eos_cache_forwards'],expected)
                observed_continuation |= expected>0
                if row['finish_reason']=='eos':
                    self.assertEqual(row['completion_ids'][-1],0);self.assertNotIn(0,row['completion_ids'][:-1])
                self.assertEqual(row['thinking_enabled'],i==0)
                self.assertEqual(row['reasoning_effort'],'low' if i==0 else None)
                self.assertEqual(row['reasoning_mode'],'native_low' if i==0 else 'native_disabled')
        self.assertTrue(observed_continuation)

    def test_false_mode_template_token_and_eos_metadata_rejected(self):
        p=sampler();context=[{'role':'user','content':'x'}]
        row=p.sample_messages_batch([context],[4],authoring_stage='revision')['records'][0]
        edits=[('thinking_enabled',True),('thinking_enabled',0),('reasoning_effort','low'),
               ('reasoning_mode','native_low'),('authoring_stage','draft'),('prompt_ids',[99]),
               ('text','repaired source'),('eos_token_ids',[99]),('finish_reason','unknown')]
        for k,v in edits:
            r=deepcopy(row);r[k]=v
            with self.subTest(k=k,v=v),self.assertRaises(ValueError):p.validate_record(r)
        r=deepcopy(row);r['completion_ids'].insert(0,0);r['sampled_token_logprobs'].insert(0,-1.);r['output_tokens']+=1
        with self.assertRaises(ValueError):p.validate_record(r)
        for stage in ('unknown',None,True):
            with self.assertRaises(ValueError):p.sample_messages_batch([context],[1],authoring_stage=stage)
        for seeds in ([],[1,2,3],[True]):
            with self.assertRaises(ValueError):p.sample_messages_batch([context]*len(seeds),seeds)
        p.config['max_context_tokens']=1
        with self.assertRaisesRegex(ValueError,'context'):p.sample_messages_batch([context],[1])
        for name in ('update','update_trajectories'):
            with self.assertRaisesRegex(RuntimeError,'never performs optimizer'):getattr(p,name)([],[])

    def test_full_vectors_both_modes_and_exact_state_restore_without_step(self):
        p,records,calls=replay_fixture();before=deepcopy(records)
        p.model.train(True)
        for parameter in p.parameters.values():parameter.grad=torch.ones_like(parameter)*.3
        cpu_rng=torch.get_rng_state().clone();gpu_rng=[cpu_rng.clone()]
        with patch.object(torch.cuda,'get_rng_state_all',side_effect=lambda:[v.clone() for v in gpu_rng]), \
             patch.object(torch.cuda,'set_rng_state_all',side_effect=lambda v:gpu_rng.__setitem__(slice(None),v)), \
             patch.object(p.optimizer,'step',side_effect=AssertionError('Forbidden optimizer step')):
            result=p.full_completion_likelihood_gate(records)
            self.assertTrue(result['passed'],result)
            self.assertEqual(result['state_before'],result['state_after'])
            self.assertEqual(result['optimizer_steps'],0);self.assertEqual(p._optimizer_identity()['step'],0)
            self.assertEqual(len(calls),32)
            self.assertTrue(all(mode==grad for _,mode,grad in calls))
            self.assertEqual(records,before)
            self.assertEqual(result['sampled_tokens'],sum(len(r['completion_ids']) for r in records))
            for source,row in zip(records,result['records']):
                self.assertEqual(row['record_sha256'],backend.object_hash(source))
                for mode in ('eval','train'):
                    self.assertEqual(row[mode+'_token_logprobs'],source['sampled_token_logprobs'])
                    self.assertTrue(row[mode+'_all_finite']);self.assertTrue(row[mode+'_check']['passed'])
                    self.assertEqual(row[mode+'_grad_enabled'],mode=='train')
            self.assertTrue(torch.equal(cpu_rng,torch.get_rng_state()))

    def test_failed_full_gate_keeps_all_records_vectors_and_unchanged_state(self):
        p,records,calls=replay_fixture();original=p.token_logps
        def altered(record):
            values=original(record)
            if record['seed']==records[0]['seed']:values=values+.1
            if record['seed']==records[3]['seed']:values=values*float('nan')
            return values
        p.token_logps=altered;gpu_rng=[]
        with patch.object(torch.cuda,'get_rng_state_all',return_value=[]),patch.object(torch.cuda,'set_rng_state_all'):
            result=p.full_completion_likelihood_gate(records)
            self.assertFalse(result['passed']);self.assertEqual(len(result['records']),16)
            self.assertEqual(len(calls),32);self.assertEqual(result['state_before'],result['state_after'])
            self.assertFalse(result['records'][0]['eval_check']['passed'])
            self.assertEqual(set(result['records'][3]['train_token_logprobs']),{'nan'})
            self.assertFalse(result['records'][3]['train_all_finite'])
            self.assertTrue(result['records'][-1]['passed'])
            missing=p.full_completion_likelihood_gate(records[:-1])
            self.assertFalse(missing['passed']);self.assertIn('sixteen',missing['reason'])
            self.assertEqual(p._optimizer_identity()['step'],0)

    def test_mixed_smoke_derivative_restores_config_rng_grads_and_mode(self):
        p,records,calls=replay_fixture();p.config['max_new_tokens']=4096
        contexts=[r['messages'] for r in records[:2]]
        def sample(contexts,seeds,stages):
            q=sampler();q.config['max_new_tokens']=32
            batch=q._sample_native_batch(contexts,seeds,stages)
            for row in batch['records']:
                row['sampled_token_logprobs']=[-.5]*len(row['completion_ids'])
            return batch
        p._sample_native_batch=sample
        def logps(record):
            total=sum(v.sum() for v in p.parameters.values())
            return torch.tensor(record['sampled_token_logprobs'])+(total-total.detach())
        p.token_logps=logps
        with patch.object(torch.cuda,'get_rng_state_all',return_value=[]),patch.object(torch.cuda,'set_rng_state_all'), \
             patch.object(p.optimizer,'step',side_effect=AssertionError('Forbidden optimizer step')):
            result=p.mechanics_smoke({'batch_messages':contexts})
            self.assertTrue(result['passed'],result)
            self.assertEqual(result['state_before'],result['state_after'])
            self.assertTrue(result['derivative']['finite_nonzero'])
            self.assertEqual(result['authoring_stages'],['draft','revision'])
            self.assertEqual(result['max_new_tokens'],32)
            self.assertEqual([r['thinking_enabled'] for r in result['batch']['records']],[True,False])
            self.assertEqual(p.config['max_new_tokens'],4096)
            self.assertEqual(p._optimizer_identity()['step'],0)
            p.token_logps=lambda record:logps(record)+.1
            failure=p.mechanics_smoke({'batch_messages':contexts})
            self.assertFalse(failure['passed']);self.assertEqual(failure['state_before'],failure['state_after'])


class OriginalTokenizerTests(unittest.TestCase):
    @unittest.skipUnless((ORIGINAL_TOKENIZER/'tokenizer_config.json').exists(),'Pinned original tokenizer absent')
    def test_native_modes_actual_tokens_and_final_only_history(self):
        try:from transformers import AutoTokenizer
        except ImportError:self.skipTest('Optional tokenizer package absent')
        tok=AutoTokenizer.from_pretrained(str(ORIGINAL_TOKENIZER),local_files_only=True)
        p=object.__new__(backend.RevisionCudaPolicy);p.tokenizer=tok
        initial=harness.initial_messages();draft,di=p.template(initial,'draft')
        self.assertTrue(draft.startswith('<|im_start|>system\nReasoning effort is set to low.'))
        self.assertTrue(draft.endswith('<|im_start|>assistant\n<think>\n'))
        raw={'text':"private reasoning\n</think>\ndef act():\n    return 'rest'\n",'finish_reason':'eos'}
        feedback=harness.practice_feedback(raw,[191000020,191000021])
        messages=harness.revision_messages(initial,raw,feedback)
        rendered,ids=p.template(messages,'revision')
        self.assertTrue(rendered.startswith('<|im_start|>user\n'))
        self.assertTrue(rendered.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n'))
        self.assertNotIn('private reasoning',rendered)
        self.assertNotIn('Reasoning effort is set to low.',rendered)
        self.assertEqual(tok.decode(ids,skip_special_tokens=False),rendered)
        self.assertLess(len(ids)+4096,16384)

if __name__=='__main__':unittest.main()
