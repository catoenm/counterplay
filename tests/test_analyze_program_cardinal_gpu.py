"""CPU audit fixtures, explicitly synthetic tensors/gate reports; no model loaded."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts import analyze_program_cardinal_gpu as audit

HAS_TORCH=importlib.util.find_spec('torch') is not None


class ScalarAuditTests(unittest.TestCase):
    def test_paired_denominator_missing_is_not_zero(self):
        seeds=list(range(32));a=[{'banked_point_yield':0.} for _ in seeds];b=deepcopy(a);b[-1]['banked_point_yield']=32.
        result=audit.paired_yield(a,b,seeds,1000)
        self.assertEqual(result['mean_yield_difference'],1.)
        self.assertEqual(result['program_cluster_n'],32)
        with self.assertRaises(ValueError):audit.paired_yield(a,b[:-1],seeds,100)

    def test_timing_only_endpoint_does_not_use_rewards(self):
        cfg={'evaluation_batch_reserve_seconds':2400,'audit_export_reserve_seconds':1800}
        rows=[{'elapsed_seconds':100,'optimizer_steps':1}]
        self.assertEqual(audit.independently_select_endpoint(rows,20000,cfg),100)
        self.assertEqual(audit.independently_select_endpoint(rows,12000,cfg),50)
        self.assertIsNone(audit.independently_select_endpoint(rows,10000,cfg))
        self.assertIsNone(audit.independently_select_endpoint([{'elapsed_seconds':1,'optimizer_steps':0}],90000,cfg))

    def test_actual_context_uses_token_ids_and_full_completion_reserve(self):
        cfg={'max_new_tokens':2048,'max_context_tokens':4096}
        record={'prompt_ids':[1]*863,'input_tokens':863,'prompt_template':'frozen prompt'}
        check=audit.actual_context_admission(record,cfg,'frozen prompt')
        self.assertEqual(check['maximum_total_tokens'],2911)
        record['input_tokens']=2
        with self.assertRaisesRegex(ValueError,'count differs'):
            audit.actual_context_admission(record,cfg)
        record.update(prompt_ids=[1]*2050,input_tokens=2050)
        with self.assertRaisesRegex(ValueError,'reserved completion'):
            audit.actual_context_admission(record,cfg)

    def test_actual_frozen_prepared_run(self):
        result=audit.audit(Path('runs/cardinal-gpu-v1'))
        self.assertTrue(result['integrity_passed'])
        self.assertFalse(result['planned_study_complete'])
        self.assertIsNone(result['paired_yield'])


@contextmanager
def actual_tensor_failed_gate_fixture():
    import torch
    from tests.test_program_cardinal_cuda import fixture as tiny_policy
    from selfplay import program_cardinal_gpu as gpu
    from tests.test_program_cardinal_gpu import FakePolicy
    with tempfile.TemporaryDirectory() as d:
        run=Path(d)/'run';shutil.copytree(Path('runs/cardinal-gpu-v1'),run)
        status=gpu.common.load(run/'status.json');status['artifacts']={};status['state']='prepared';status['claims_allowed']=False
        cfg=gpu.common.load(run/'manifest.json')['config'];status['execution_budget']=gpu.journal.execution_budget(cfg)[0]
        gpu.common.atomic_json(run/'status.json',status)
        policy=tiny_policy();policy.config=cfg
        fake=FakePolicy();base_sample=fake.sample_batch
        def sample(prompt,seeds):
            batch=base_sample(prompt,seeds)
            for row in batch['records']:
                row['text']='deliberately invalid fixture';row['raw_text']=row['text'];row['prompt_template']=prompt
            return batch
        policy.sample_batch=sample
        info=gpu.common.load(run/'inputs/model-info.json')
        policy.runtime.update(cuda='12.8',loader='image_text_to_text',eos_token_ids=[2],
            arithmetic='NF4 frozen base; FP32 compute/LoRA; TF32 off; eager attention',
            versions={n:v for n,v in gpu.VERSIONS.items() if n!='safetensors'},
            trainable_parameters=sum(v.numel() for v in policy.parameters.values()),
            model_index={'repo':gpu.MODEL_REPO,'revision':gpu.MODEL_REVISION,
                'authoritative_metadata_sha256':gpu.common.digest(run/'inputs/model-info.json'),
                'files':{r['rfilename']:r['lfs']['sha256'] for r in info['siblings'] if r['rfilename'].endswith('.safetensors')}})
        rng=[torch.get_rng_state().clone()]
        with patch.object(torch.cuda,'get_rng_state_all',side_effect=lambda:[x.clone() for x in rng]),patch.object(torch.cuda,'set_rng_state_all',side_effect=lambda v:rng.__setitem__(slice(None),v)):
            engine=gpu.GPUCycle(run,policy)
            engine.commit('runtime.json',policy.runtime)
            engine.commit('prompt-measurement.json',{'prompt_tokens':1,'completion_cap':2048,'context_cap':4096,'admitted':True})
            engine.initialize()
            samples=base_sample('fixture',list(range(8)))['records']
            for row in samples:row['prompt_template']=engine.prompts['cardinal']
            gate={'passed':True,'records':[{'passed':True,'tokens':1,'max_token_error':0.,'sequence_error':0.,'recorded_sequence_logprob':-.5} for _ in range(8)]}
            engine.commit('smoke-fixture.json',{'passed':True,'training_use':False,'optimizer_steps':0,
                'source_checkpoint':'base','source_metadata_sha256':engine.status['artifacts']['checkpoints/base/metadata.json'],
                'adapter_unchanged':True,'optimizer_unchanged':True,'rng_restored':True,'config_restored':True,
                'optimizer_state_before_sha256':'same','optimizer_state_after_sha256':'same',
                'batch':{'records':samples},'eval_likelihood_gate':gate,'train_likelihood_gate':gate,
                'derivative':{'finite_nonzero':True,'gradient_l2_norm':1.,'missing_gradient_names':[]}})
            engine.run_all()
            yield run,engine,policy


@unittest.skipUnless(HAS_TORCH,'Optional CPU torch required')
class TensorAuditTests(unittest.TestCase):
    def test_real_cuda_checkpoint_schema_failed_gate_all_eight_retained(self):
        with actual_tensor_failed_gate_fixture() as (run,engine,policy):
            result=audit.audit(run)
            self.assertTrue(result['integrity_passed']);self.assertEqual(result['recorded_state'],'capability_failed')
            self.assertEqual(result['committed_generation_draws'],8)
            self.assertEqual(result['genuine_optimizer_steps'],0)
            self.assertEqual(result['capability']['selected_indices'],[])
            self.assertFalse(result['planned_comparisons_available'])

    def test_non_authoritative_precheck_mismatch_is_an_explicit_erratum(self):
        from scripts.cloud import train_program_rl as common
        with actual_tensor_failed_gate_fixture() as (run,engine,policy):
            path='prompt-measurement.json';measurement=common.load(run/path);measurement['prompt_tokens']=2
            common.atomic_json(run/path,measurement);engine.status['artifacts'][path]=common.digest(run/path);engine.state()
            result=audit.audit(run)
            self.assertTrue(result['integrity_passed'])
            self.assertEqual(result['context_admission']['authoritative_backend_measurement']['actual_prompt_tokens'],1)
            self.assertEqual(result['context_admission']['authoritative_backend_measurement']['maximum_total_tokens'],2049)
            self.assertEqual(result['metadata_errata'][0]['recorded_precheck_prompt_tokens'],2)
            self.assertFalse(result['metadata_errata'][0]['frozen_artifacts_modified'])

    def test_changed_generation_rejected_even_if_journal_rehashed(self):
        from scripts.cloud import train_program_rl as common
        with actual_tensor_failed_gate_fixture() as (run,engine,policy):
            path='generations/capability-0.json';value=common.load(run/path);value['text']='different invalid fixture'
            common.atomic_json(run/path,value);engine.status['artifacts'][path]=common.digest(run/path);engine.state()
            with self.assertRaisesRegex(ValueError,'batch row'):
                audit.audit(run)

    def test_committed_group_with_real_adam_step_and_archived_game_replay(self):
        import torch
        from tests.test_program_cardinal_cuda import one_step
        from selfplay import program_cardinal_gpu as gpu
        controller="""def act():
    m0 = m0 + 1
    if energy == 0:
        return 'rest'
    if self_x == home_x and self_y == home_y and carrying > 0:
        return 'deposit'
    tx = home_x
    ty = home_y
    target = -1
    best = 1000000
    if carrying < capacity:
        for i in range(6):
            if site(i, 'units') > 0:
                cost = 100 * (abs(site(i, 'x') - self_x) + abs(site(i, 'y') - self_y)) - site(i, 'value')
                if cost < best:
                    best = cost
                    tx = site(i, 'x')
                    ty = site(i, 'y')
                    target = i
    if carrying > 0 and horizon - turn <= abs(home_x - self_x) + abs(home_y - self_y) + 4:
        tx = home_x
        ty = home_y
        target = -1
    if self_x == tx and self_y == ty:
        if target >= 0:
            return 'harvest'
        return 'rest'
    if self_x < tx:
        return 'east'
    if self_x > tx:
        return 'west'
    if self_y < ty:
        return 'south'
    return 'north'
"""
        # Use a fresh copy of the synthetic fixture and remove only its already
        # generated failed capability artifacts; this is test-only construction.
        with actual_tensor_failed_gate_fixture() as (run,old,policy):
            keep={p:v for p,v in old.status['artifacts'].items() if not (p.startswith('generations/') or p.startswith('batches/') or p=='capability.json')}
            old.status['artifacts']=keep;old.status['state']='running';old.state()
            engine=gpu.GPUCycle(run,policy)
            from tests.test_program_cardinal_gpu import FakePolicy
            generator=FakePolicy()
            def samples(prompt,seeds):
                batch=generator.sample_batch(prompt,seeds)
                for i,row in enumerate(batch['records']):
                    text=controller if i%2 else controller.replace('m0','m1')
                    if seeds[0]>=10200000 and i%2==0:text='deliberately invalid fixture'
                    row.update(text=text,raw_text=text,prompt_template=prompt)
                return batch
            policy.sample_batch=samples
            def gate(records):
                return {'passed':True,'records':[{'passed':True,'tokens':len(r['completion_ids']),
                    'max_token_error':0.,'sequence_error':0.,'recorded_sequence_logprob':sum(r['sampled_token_logprobs'])} for r in records]}
            policy.likelihood_gate=gate
            def update(records,advantages):
                before={n:p.detach().clone() for n,p in policy.parameters.items()};one_step(policy)
                delta=sum(float((p.detach().double()-before[n].double()).square().sum()) for n,p in policy.parameters.items())**.5
                losses=[-a*sum(r['sampled_token_logprobs'])/8 for a,r in zip(advantages,records)]
                return {'optimizer_steps':1,'adapter_l2_delta':delta,'gradient_likelihood_gate':gate(records),
                    'gradient_norm_before_clip':1.,'microbatch_losses':losses,'loss':sum(losses)}
            policy.update=update
            cap=engine.capability();self.assertTrue(cap['passed']);opponents=engine.opponents(cap)
            engine.group(1,engine.roster(1,opponents))
            result=audit.audit(run)
            self.assertEqual(result['genuine_optimizer_steps'],1)
            self.assertGreater(result['verified_unique_game_replays'],0)
            self.assertEqual(result['tensor_reports']['cycle1']['actual_optimizer_step'],1)
            self.assertFalse(result['planned_study_complete'])

    def test_actual_tensor_delta_and_optimizer_rate_tampering(self):
        import torch
        from tests.test_program_cardinal_cuda import one_step
        from selfplay.program_cardinal_cuda import checkpoint_state
        with actual_tensor_failed_gate_fixture() as (run,engine,policy):
            before=run/'checkpoints/base';one_step(policy);after=run/'changed';policy.save_checkpoint(after)
            result=audit.checkpoint_delta(after,before)
            self.assertGreater(result['adapter_l2_delta'],0);self.assertGreater(result['changed_parameters'],0)
            state=torch.load(after/'optimizer.pt',map_location='cpu',weights_only=True)
            state['optimizer']['param_groups'][0]['lr']=1e-6;torch.save(state,after/'optimizer.pt')
            with self.assertRaises(ValueError):checkpoint_state(after,expected_learning_rate=1e-5)


if __name__=='__main__':unittest.main()
