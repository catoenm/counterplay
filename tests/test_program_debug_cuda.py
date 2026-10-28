"""CPU Torch fixtures for padded sampling and the two-call policy-gradient sum.

These exercise tensor plumbing, not pretrained Qwen numerical equivalence. The
frozen mixed-context GPU smoke remains mandatory before real rollouts/updates.
"""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from selfplay import program_debug_cuda as debug
from tests.test_program_cardinal_cuda import HAS_TORCH, fixture

if HAS_TORCH:
    import torch


def sampler():
    class Tokenizer:
        pad_token_id=9
        def apply_chat_template(self,messages,tokenize,add_generation_prompt,enable_thinking):
            assert not tokenize and add_generation_prompt and enable_thinking
            return '|'.join(m['role']+':'+m['content'] for m in messages)+'<think>'
        def encode(self,text,add_special_tokens):
            assert not add_special_tokens
            return [ord(c)%4+1 for c in text]
        def decode(self,ids,skip_special_tokens):
            return ','.join(str(i) for i in ids if not skip_special_tokens or i!=0)
    class MaskedTable:
        def __init__(self):self.calls=[]
        def eval(self):return self
        @staticmethod
        def table():return torch.tensor([[.4,.1,-.2,.3,.0],[.1,.4,.2,-.1,.0],
                                        [.3,-.1,.2,.4,.0],[.1,.3,.4,.0,-.2]])
        def __call__(self,input_ids,attention_mask,position_ids,past_key_values,use_cache,logits_to_keep):
            assert use_cache and logits_to_keep==1
            mask=attention_mask[:,-input_ids.shape[1]:]
            expected=(attention_mask.cumsum(-1)-1).masked_fill(attention_mask==0,0)[:,-input_ids.shape[1]:]
            assert torch.equal(position_ids,expected)
            self.calls.append((input_ids.clone(),attention_mask.clone(),position_ids.clone()))
            total=(input_ids*mask).sum(-1)
            if past_key_values is not None:total+=past_key_values
            phases=(total+position_ids[:,-1])%4
            return SimpleNamespace(logits=self.table()[phases][:,None,:],past_key_values=total)
    p=object.__new__(debug.DebugCudaPolicy)
    p.torch,p.model,p.eos,p.tokenizer=torch,MaskedTable(),[0],Tokenizer()
    p.config={'max_new_tokens':15,'max_context_tokens':512,'sample_seconds':30}
    p._sample_device='cpu'
    return p


class FlattenTests(unittest.TestCase):
    def test_exact_turn_identity_and_mean_trajectory_normalization(self):
        trajectories=[{'turns':[{'id':'a1'},{'id':'a2'}],'feedback':'must never be a target'},
                      {'turns':[{'id':'b1'},{'id':'b2'}],'feedback':'ignored'}]
        original=deepcopy(trajectories)
        records,advantages=debug.flatten_trajectories(trajectories,[.25,-.5])
        self.assertEqual(advantages,[.5,.5,-1.,-1.])
        self.assertEqual([r['id'] for r in records],['a1','a2','b1','b2'])
        self.assertIs(records[0],trajectories[0]['turns'][0])
        self.assertEqual(trajectories,original)
        logps=[-3.,-5.,-7.,-11.]
        flattened=-sum(a*l for a,l in zip(advantages,logps))/len(records)
        expected=-(.25*(-3-5)-.5*(-7-11))/2
        self.assertEqual(flattened,expected)

    def test_malformed_trajectory_or_advantage_rejected(self):
        good={'turns':[{},{}]}
        for rows,advantages in [([],[]),([good]*9,[1]*9),([good],[]),([good],[True]),
                                ([good],[float('nan')]),([good],[float('inf')]),
                                ([{'turns':[{}]}],[1]),([{'turns':[{}, {}, {}]}],[1])]:
            with self.subTest(rows=len(rows),advantages=advantages):
                with self.assertRaises(ValueError):debug.flatten_trajectories(rows,advantages)


@unittest.skipUnless(HAS_TORCH,'Requires optional CPU Torch; no CUDA/pretrained models')
class DebugCudaTests(unittest.TestCase):
    def test_differing_prefix_batch_matches_individual_rng_and_unpadded_likelihoods(self):
        contexts=[[{'role':'user','content':'a'*i}] for i in range(1,9)]
        original=deepcopy(contexts);p=sampler();batch=p.sample_messages_batch(contexts,list(range(8)))
        self.assertEqual(contexts,original)
        self.assertTrue(all(len(call[0])==8 for call in p.model.calls))
        ids,mask,positions=p.model.calls[0]
        self.assertEqual(mask[0].tolist()[:7],[0]*7)
        self.assertEqual(positions[0,7].item(),0)
        self.assertEqual(positions[-1,0].item(),0)
        for seed,row in enumerate(batch['records']):
            single=sampler().sample_messages_batch([contexts[seed]],[seed])['records'][0]
            for key in ('prompt_ids','completion_ids','sampled_token_logprobs','finish_reason','raw_text'):
                self.assertEqual(row[key],single[key])
            self.assertEqual(row['left_padding_tokens'],7-seed)
            prefix=list(row['prompt_ids']);reference=[]
            for token in row['completion_ids']:
                logits=p.model.table()[(sum(prefix)+len(prefix)-1)%4]
                reference.append(float(torch.log_softmax(logits,dim=-1)[token]));prefix.append(token)
            self.assertEqual(row['sampled_token_logprobs'],reference)
            if row['finish_reason']=='eos':
                self.assertEqual(row['completion_ids'][-1],0)
                self.assertNotIn(0,row['completion_ids'][:-1])
        self.assertEqual(batch['timing']['total_output_tokens'],sum(r['output_tokens'] for r in batch['records']))
        self.assertGreater(batch['timing']['finished_eos'],0)

    def test_thinking_native_history_and_strict_context_admission(self):
        p=sampler();messages=[{'role':'user','content':'draft'},{'role':'assistant','content':'code'},
                              {'role':'user','content':'actual feedback'}]
        rendered,ids=p.template(messages)
        self.assertTrue(rendered.endswith('<think>'));self.assertIn('actual feedback',rendered)
        row=p.sample_batch({'batch_messages':[messages]},[7])['records'][0]
        self.assertEqual(row['messages'],messages);self.assertEqual(row['prompt_ids'],ids)
        self.assertTrue(row['thinking_enabled'])
        for bad in ([{'role':'tool','content':'x'}],[{'role':'user','content':'x','extra':1}]):
            with self.assertRaises(ValueError):p.template(bad)
        p.config['max_context_tokens']=len(ids)+p.config['max_new_tokens']-1
        with self.assertRaisesRegex(ValueError,'context'):p.sample_messages_batch([messages],[7])

    def test_time_bound_seed_validation_and_eos_padding_fallback(self):
        p=sampler();context=[{'role':'user','content':'x'}]
        for seeds in ([],list(range(9)),[True],[-1],[2**63]):
            with self.assertRaises(ValueError):p.sample_messages_batch([context]*len(seeds),seeds)
        p.tokenizer.pad_token_id=None
        rows=p.sample_messages_batch([context,context],[11,11])['records']
        self.assertEqual(rows[0]['completion_ids'],rows[1]['completion_ids'])
        p.config['sample_seconds']=0
        row=p.sample_messages_batch([context],[7])['records'][0]
        self.assertEqual(row['finish_reason'],'time');self.assertEqual(row['completion_ids'],[])

    def policy_for_update(self):
        p=object.__new__(debug.DebugCudaPolicy);p.__dict__.update(fixture().__dict__)
        p.config.update(gradient_clip=.5,token_logprob_tolerance=.002,sequence_logprob_tolerance=.02)
        def logps(record):
            z=sum((i+1)*v.sum() for i,v in enumerate(p.parameters.values()))*.02
            logits=torch.stack([z,-z,torch.zeros_like(z)])
            return torch.log_softmax(logits,dim=-1)[record['completion_ids']]
        p.token_logps=logps
        trajectories=[]
        for pair in (([0,1],[0,0,1]),([1],[2,1])):
            turns=[]
            for completion in pair:
                record={'completion_ids':completion,'prompt_ids':[999,999]}
                record['sampled_token_logprobs']=logps(record).detach().tolist();turns.append(record)
            trajectories.append({'turns':turns,'feedback':'never target'})
        return p,trajectories

    def test_real_adam_step_matches_full_two_turn_objective_and_gradients(self):
        p,trajectories=self.policy_for_update();advantages=[1.,-1.]
        loss=-sum(a*sum(p.token_logps(r).sum() for r in t['turns']) for a,t in zip(advantages,trajectories))/2
        expected_loss=float(loss.detach());grads=torch.autograd.grad(loss,list(p.parameters.values()))
        expected_norm=float(torch.sqrt(sum(g.square().sum() for g in grads)))
        before={n:v.detach().clone() for n,v in p.parameters.items()};frozen=p.verify_frozen()
        report=p.update_trajectories(trajectories,advantages)
        self.assertAlmostEqual(report['loss'],expected_loss,places=6)
        self.assertAlmostEqual(report['gradient_norm_before_clip'],expected_norm,places=6)
        self.assertEqual(report['adam_step_before'],0);self.assertEqual(report['adam_step_after'],1)
        self.assertEqual(report['sampled_turn_count'],4);self.assertEqual(report['trajectory_count'],2)
        self.assertEqual(report['tool_feedback_target_tokens'],0)
        self.assertTrue(report['gradient_likelihood_gate']['passed']);self.assertGreater(report['adapter_l2_delta'],0)
        self.assertEqual(p.verify_frozen(),frozen)
        self.assertTrue(any(not torch.equal(before[n],v) for n,v in p.parameters.items()))

    def test_likelihood_mismatch_prevents_actual_adam_step(self):
        p,trajectories=self.policy_for_update()
        trajectories[1]['turns'][1]['sampled_token_logprobs'][0]+=.1
        before={n:v.detach().clone() for n,v in p.parameters.items()}
        with self.assertRaisesRegex(ValueError,'likelihood gate'):p.update_trajectories(trajectories,[1.,-1.])
        self.assertEqual(p._optimizer_identity()['step'],0)
        self.assertTrue(all(torch.equal(before[n],v) for n,v in p.parameters.items()))


if __name__=='__main__':unittest.main()
