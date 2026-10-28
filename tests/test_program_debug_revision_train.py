"""CPU-only synthetic tensor tests; no pretrained model, API or game calls."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from selfplay import program_debug_revision_train as backend
from selfplay.program_debug_revision_cuda import RevisionCudaPolicy
from selfplay.program_cardinal_cuda import checkpoint_state
from tests.test_program_debug_revision_cuda import replay_fixture, sampler
from tests.test_program_cardinal_cuda import HAS_TORCH

if HAS_TORCH:
    import torch


def training_fixture(long_record=False):
    policy, records, _ = replay_fixture()
    policy.__class__ = backend.TrainRevisionCudaPolicy
    policy.config.update(gradient_clip=.5, max_new_tokens=4096, group_size=8)
    policy.model.eval()
    calls = []
    for row in records:
        row.update(calibration_only=False, training_capable=True, completion_token_limit=4096)
        if row['finish_reason'] == 'length':
            row['completion_ids'][-1] = 0
            row['finish_reason'] = 'eos'
        row['text'] = policy.tokenizer.decode(row['completion_ids'], skip_special_tokens=True)
        row['raw_text'] = policy.tokenizer.decode(row['completion_ids'], skip_special_tokens=False)
    if long_record:
        records[0].update(completion_ids=[1, 2] * 2048, output_tokens=4096, finish_reason='length')
        records[0]['text'] = policy.tokenizer.decode(records[0]['completion_ids'], skip_special_tokens=True)
        records[0]['raw_text'] = policy.tokenizer.decode(records[0]['completion_ids'], skip_special_tokens=False)
    def logps(record):
        calls.append((record['seed'], policy.model.training, torch.is_grad_enabled()))
        # A small differentiable full-softmax fixture, with all adapter tensors
        # involved and varied turn lengths/labels. This is not a language model.
        z = sum((i+1)*p.sum() for i,p in enumerate(policy.parameters.values())) * .02
        z = z * (1 + record['seed'] % 7 * .15)
        logits = torch.stack([z, -z, z*.4, z*.2, torch.zeros_like(z)])
        return torch.log_softmax(logits, dim=-1)[record['completion_ids']]
    policy.token_logps = logps
    for row in records:
        row['sampled_token_logprobs'] = logps(row).detach().tolist()
        row['output_tokens'] = len(row['completion_ids'])
    calls.clear()
    trajectories = [{'turns': records[i:i+2], 'feedback': 'NEVER A TARGET'} for i in range(0,16,2)]
    return policy, records, trajectories, calls


@unittest.skipUnless(HAS_TORCH, 'Optional CPU Torch unavailable')
class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.gpu_rng = [torch.get_rng_state().clone()]
        self.get_rng = patch.object(torch.cuda, 'get_rng_state_all', side_effect=lambda:[x.clone() for x in self.gpu_rng])
        self.set_rng = patch.object(torch.cuda, 'set_rng_state_all', side_effect=lambda x:self.gpu_rng.__setitem__(slice(None),x))
        self.get_rng.start(); self.set_rng.start()
        self.addCleanup(self.get_rng.stop); self.addCleanup(self.set_rng.stop)

    def test_sampling_changes_only_scope_metadata_and_preserves_frozen_gates(self):
        p=sampler();p.__class__=backend.TrainRevisionCudaPolicy
        contexts=[[{'role':'user','content':'x'}],[{'role':'user','content':'longer'}]]
        actual=p.sample_messages_batch(contexts,[7,8],authoring_stage='revision')
        expected=sampler().sample_messages_batch(contexts,[7,8],authoring_stage='revision')
        for a,b in zip(actual['records'],expected['records']):
            self.assertFalse(a.pop('calibration_only'));self.assertTrue(a.pop('training_capable'))
            b.pop('calibration_only')
            for k in ('elapsed_seconds',):a.pop(k);b.pop(k)
            self.assertEqual(a,b)
        self.assertIs(backend.TrainRevisionCudaPolicy.full_completion_likelihood_gate,
                      RevisionCudaPolicy.full_completion_likelihood_gate)
        self.assertIs(backend.TrainRevisionCudaPolicy.mechanics_smoke,RevisionCudaPolicy.mechanics_smoke)
        with self.assertRaisesRegex(RuntimeError,'complete eight-trajectory'):p.update([],[])

    def test_full_length_probe_covers_all_tokens_without_update_and_restores_exact_state(self):
        p,records,_,calls=training_fixture(long_record=True)
        p.model.train()
        for value in p.parameters.values():value.grad=torch.full_like(value,.25)
        original=deepcopy(records);before=p._state_identity()
        expected=-sum(p.token_logps(row).sum() for row in records)/16
        expected_loss=float(expected.detach())
        grads=torch.autograd.grad(expected,list(p.parameters.values()))
        expected_norm=float(torch.sqrt(sum(x.double().square().sum() for x in grads)))
        calls.clear()
        with patch.object(p.optimizer,'step',side_effect=AssertionError('Probe may not optimize')):
            report=p.full_backward_probe(records)
        self.assertTrue(report['passed'],report)
        self.assertEqual(report['state_before'],before);self.assertEqual(report['state_after'],before)
        self.assertEqual(report['optimizer_steps'],0);self.assertFalse(report['training_use'])
        self.assertEqual(report['coefficients'],[1.]*16);self.assertEqual(report['objective_divisor'],16)
        self.assertEqual(report['sampled_tokens'],sum(len(r['completion_ids']) for r in records))
        self.assertGreaterEqual(report['sampled_tokens'],4096)
        self.assertAlmostEqual(report['loss'],expected_loss,delta=.002)
        self.assertAlmostEqual(report['derivative']['gradient_l2_norm'],expected_norm,delta=.0001)
        self.assertEqual(report['derivative']['missing_gradient_names'],[])
        self.assertEqual(len(calls),32);self.assertTrue(all(mode==grad for _,mode,grad in calls))
        self.assertEqual(records,original)
        for row,source in zip(report['records'],records):
            self.assertTrue(row['backward_completed']);self.assertTrue(row['passed'])
            self.assertEqual(row['record_sha256'],backend.object_hash(source))
            self.assertEqual(row['eval_token_logprobs'],source['sampled_token_logprobs'])
            self.assertEqual(row['train_token_logprobs'],source['sampled_token_logprobs'])
            self.assertGreaterEqual(row['timing']['backward_seconds'],0)

    def test_probe_failure_preserves_partial_evidence_and_original_gradients(self):
        p,records,_,_=training_fixture();p.model.train()
        for value in p.parameters.values():value.grad=torch.full_like(value,.5)
        before=p._state_identity();original=p.token_logps
        def interrupted(row):
            if row['seed']==records[3]['seed'] and p.model.training:
                raise RuntimeError('Synthetic out of memory; no retry')
            return original(row)
        p.token_logps=interrupted
        with patch.object(p.optimizer,'step',side_effect=AssertionError('No step after failure')):
            report=p.full_backward_probe(records)
        self.assertFalse(report['passed']);self.assertEqual(len(report['records']),4)
        self.assertEqual(len(report['microbatch_losses']),3)
        self.assertFalse(report['records'][-1]['backward_completed'])
        self.assertIsNone(report['records'][-1]['train_token_logprobs'])
        self.assertEqual(report['state_before'],before);self.assertEqual(report['state_after'],before)
        self.assertEqual(p._optimizer_identity()['step'],0)

    def test_actual_adam_matches_full_two_turn_sum_without_length_normalization(self):
        p,records,trajectories,_=training_fixture();advantages=[1.,-1.,.5,-.5,.75,-.75,.25,-.25]
        expected=-sum(a*sum(p.token_logps(r).sum() for r in t['turns']) for a,t in zip(advantages,trajectories))/8
        expected_loss=float(expected.detach());grads=torch.autograd.grad(expected,list(p.parameters.values()))
        expected_norm=float(torch.sqrt(sum(x.double().square().sum() for x in grads)))
        reference=[torch.nn.Parameter(v.detach().clone()) for v in p.parameters.values()]
        opt=torch.optim.AdamW(reference,lr=1e-5,weight_decay=0)
        for target,gradient in zip(reference,grads):target.grad=gradient.clone()
        torch.nn.utils.clip_grad_norm_(reference,.5,error_if_nonfinite=True);opt.step()
        before=p._state_identity();report=p.update_trajectories(trajectories,advantages)
        self.assertTrue(report['passed'],report);self.assertEqual(report['outcome'],'updated')
        self.assertAlmostEqual(report['loss'],expected_loss,places=6)
        self.assertAlmostEqual(report['gradient_norm_before_clip'],expected_norm,places=6)
        self.assertEqual(report['adam_step_before'],0);self.assertEqual(report['adam_step_after'],1)
        self.assertEqual(report['optimizer_steps'],1);self.assertTrue(report['optimizer_step_returned'])
        self.assertGreater(report['adapter_l2_delta'],0);self.assertEqual(len(report['microbatch_losses']),16)
        self.assertEqual(report['objective_divisor'],8)
        self.assertEqual((report['prompt_target_tokens'],report['tool_feedback_target_tokens']),(0,0))
        self.assertEqual(report['state_after']['rng'],before['rng'])
        self.assertEqual(report['state_after']['frozen_state_sha256'],before['frozen_state_sha256'])
        self.assertTrue(all(p.grad is None for p in p.parameters.values()))
        for target,actual in zip(reference,p.parameters.values()):self.assertTrue(torch.allclose(target,actual,atol=2e-8,rtol=0))
        for i,(row,source) in enumerate(zip(report['records'],records)):
            self.assertEqual(row['train_token_logprobs'],source['sampled_token_logprobs'])
            self.assertAlmostEqual(row['train_loss'],-advantages[i//2]*sum(source['sampled_token_logprobs'])/8,places=6)
            self.assertEqual(row['record_sha256'],backend.object_hash(source))
            self.assertTrue(row['train_grad_enabled']);self.assertTrue(row['train_check']['passed'])

    def test_actual_inherited_token_loss_masks_full_history_and_keeps_sampled_eos(self):
        p,records,_,_=training_fixture();row=deepcopy(records[1]);seen={}
        row.update(completion_ids=[2,1,0],output_tokens=3,finish_reason='eos',
                   sampled_token_logprobs=[-1.,-1.,-1.])
        row['text']=p.tokenizer.decode(row['completion_ids'],skip_special_tokens=True)
        row['raw_text']=p.tokenizer.decode(row['completion_ids'],skip_special_tokens=False)
        def tensor(data,**kwargs):
            if kwargs.get('device')=='cuda':kwargs['device']='cpu'
            return torch.tensor(data,**kwargs)
        def forward(input_ids,use_cache):
            seen.update(ids=input_ids.tolist(),use_cache=use_cache)
            n=input_ids.shape[1]
            logits=torch.arange(n*5,dtype=torch.float32).reshape(1,n,5)/50
            seen['logits']=logits
            return SimpleNamespace(logits=logits)
        p.torch=SimpleNamespace(tensor=tensor,long=torch.long,nn=torch.nn)
        p.model.forward=forward
        # Invoke the actual inherited method, not this fixture's analytic stub.
        values=RevisionCudaPolicy.token_logps(p,row)
        self.assertFalse(seen['use_cache'])
        self.assertEqual(seen['ids'],[row['prompt_ids']+row['completion_ids'][:-1]])
        start=len(row['prompt_ids'])-1
        expected=-torch.nn.functional.cross_entropy(seen['logits'][0,start:start+3],
                                                     torch.tensor([2,1,0]),reduction='none')
        self.assertTrue(torch.equal(values,expected));self.assertEqual(len(values),3)

    def test_zero_advantages_skip_exactly_without_forwards_or_optimizer_and_commit_binding(self):
        p,_,trajectories,calls=training_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p.save_checkpoint(root/'before');source=checkpoint_state(root/'before',1e-5)
            with patch.object(p.optimizer,'step',side_effect=AssertionError('Zero variance must not optimize')):
                report=p.update_trajectories(trajectories,[0.]*8)
            self.assertTrue(report['passed'],report);self.assertEqual(report['outcome'],'skipped_zero_variance')
            self.assertEqual(calls,[]);self.assertEqual(report['state_before'],report['state_after'])
            p.save_checkpoint(root/'after');target=checkpoint_state(root/'after',1e-5)
            self.assertTrue(backend.validate_checkpoint_transition(report,source,target))

    def test_actual_checkpoint_commit_rejects_wrong_source_target_and_ambiguous_report(self):
        p,_,trajectories,_=training_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p.save_checkpoint(root/'before');source=checkpoint_state(root/'before',1e-5)
            report=p.update_trajectories(trajectories,[1.,-1.]*4)
            self.assertTrue(report['passed'],report)
            p.save_checkpoint(root/'after');target=checkpoint_state(root/'after',1e-5)
            self.assertTrue(backend.validate_checkpoint_transition(report,source,target))
            for src,tgt,rep in [(source,source,report),(target,target,report),
                    (source,target,{**report,'optimizer_steps':0}),
                    (source,target,{**report,'passed':False}),
                    (source,{**target,'rng':source['adapter_tensors']},report)]:
                with self.assertRaises(ValueError):backend.validate_checkpoint_transition(rep,src,tgt)

    def test_late_likelihood_mismatch_and_missing_gradient_never_step(self):
        for fault in ('likelihood','missing_gradient'):
            p,records,trajectories,_=training_fixture();before=p._state_identity()
            if fault=='likelihood':records[-1]['sampled_token_logprobs'][0]-=.1
            else:
                last=list(p.parameters.values())[-1];last.requires_grad_(False)
                # Avoid changing frozen membership; simulate an unused trainable
                # graph by removing its term while keeping the true schema.
                last.requires_grad_(True)
                def logps(row):
                    z=sum(x.sum() for x in list(p.parameters.values())[:-1])*.02
                    return torch.log_softmax(torch.stack([z,-z,z*.4,z*.2,torch.zeros_like(z)]),-1)[row['completion_ids']]
                p.token_logps=logps
                for row in records:row['sampled_token_logprobs']=logps(row).detach().tolist()
            with patch.object(p.optimizer,'step',side_effect=AssertionError('No optimizer after bad gradients')):
                report=p.update_trajectories(trajectories,[1.,-1.]*4)
            self.assertFalse(report['passed']);self.assertFalse(report['optimizer_step_attempted'])
            self.assertEqual(report['optimizer_steps'],0);self.assertEqual(report['state_before'],before)
            self.assertEqual(report['state_after'],before)
            if fault=='missing_gradient':self.assertTrue(report['derivative']['missing_gradient_names'])

    def test_throwing_optimizer_preserves_ambiguous_mutation_never_rolls_back_or_retries(self):
        p,_,trajectories,_=training_fixture();original=p.optimizer.step;calls=[]
        def failing_step():
            calls.append(True);original();raise RuntimeError('Synthetic error after Adam mutated')
        with patch.object(p.optimizer,'step',side_effect=failing_step):
            report=p.update_trajectories(trajectories,[1.,-1.]*4)
        self.assertFalse(report['passed']);self.assertEqual(len(calls),1)
        self.assertTrue(report['optimizer_step_attempted']);self.assertFalse(report['optimizer_step_returned'])
        self.assertIsNone(report['optimizer_steps']);self.assertEqual(report['adam_step_after'],1)
        self.assertNotEqual(report['state_before']['adapter_sha256'],report['state_after']['adapter_sha256'])
        self.assertTrue(report['gradients_cleared'])

    def test_invalid_groups_and_expired_deadline_cannot_update(self):
        p,records,trajectories,_=training_fixture()
        for ts,advs in [(trajectories[:-1],[1.,-1.]*4),(trajectories,[1.]*8),
                        (trajectories,[True]+[0.]*7),(trajectories,[float('nan')]*8)]:
            report=p.update_trajectories(ts,advs)
            self.assertFalse(report['passed']);self.assertFalse(report['optimizer_step_attempted'])
        original=records[0]['calibration_only'];records[0]['calibration_only']=True
        report=p.full_backward_probe(records);self.assertFalse(report['passed'])
        records[0]['calibration_only']=original
        p.config['deadline']='2000-01-01T00:00:00+00:00'
        report=p.update_trajectories(trajectories,[1.,-1.]*4)
        self.assertFalse(report['passed']);self.assertEqual(report['error_type'],'TimeoutError')
        self.assertEqual(p._optimizer_identity()['step'],0)


if __name__=='__main__':unittest.main()
