"""Independent warmstart audit with real small CPU Adam states, no model load."""
from contextlib import contextmanager
import contextlib
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import analyze_program_cardinal_warmstart as audit
from selfplay import program_cardinal_warmstart as warm
from tests.test_program_cardinal_warmstart import fixture, Tokenizer


@contextmanager
def tensors():
    import torch
    from tests.test_program_cardinal_sft import policy
    backend=policy();gpu_rng=[torch.get_rng_state().clone()]
    with fixture(backend,real_tensors=True) as (run,engine,backend), \
            patch.object(torch.cuda,'get_rng_state_all',side_effect=lambda:[r.clone() for r in gpu_rng]), \
            patch.object(torch.cuda,'set_rng_state_all',side_effect=lambda values:gpu_rng.__setitem__(slice(None),values)):
        # Tiny explicit fixture tokenizer EOS; no pretrained tokenizer is loaded.
        path=run/'inputs/teacher/plan.json';plan=warm.common.load(path);plan['learner']['eos_token_id']=0
        warm.common.atomic_json(path,plan)
        manifest=warm.common.load(run/'manifest.json')
        manifest['input_sha256']['inputs/teacher/plan.json']=warm.common.digest(path)
        warm.common.atomic_json(run/'manifest.json',manifest)
        engine.manifest=manifest;engine.status['manifest_sha256']=warm.common.digest(run/'manifest.json');engine.state()
        engine.commit('runtime.json',{**backend.runtime,'eos_token_ids':[0]})
        engine.commit('prompt-measurement.json',warm.prompt_measurement(Tokenizer(),'fixture',engine.config))
        engine.initialize()
        obj=audit.Audit(run);obj.teacher=engine.corpus;obj.cuda_inspector=warm.gpu.checkpoint_state
        yield run,engine,backend,obj


class ScheduleTests(unittest.TestCase):
    def test_independent_schedule_matches_and_keeps_collision_override(self):
        self.assertEqual(audit.registered_schedule(),warm.schedule())
        self.assertEqual(audit.registered_schedule()['train_maps']['45'],[93004502,93004501])
        self.assertEqual(audit.registered_sft_schedule(list(range(16))),warm.sft_schedule(list(range(16))))


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('safetensors'),'Optional CPU Torch')
class TensorAuditTests(unittest.TestCase):
    def test_all32_real_SFT_commits_and_exact_zero_step_RL_reset(self):
        with tensors() as (run,engine,backend,obj):
            base=obj.checkpoint('base')
            self.assertEqual(base['optimizer_identity']['step'],0)
            self.assertEqual(len(obj.supervised_reports),32)
            self.assertTrue(all(row['adapter_l2_delta']>0 for row in obj.supervised_reports))
            self.assertEqual(sorted(obj.warm_checkpoints),list(range(33)))
            self.assertEqual(len(obj.targets),16)
            self.assertTrue(obj.tensor_reports['base']['same_adapter_as_sft_step32'])
            self.assertEqual(obj.tensor_reports['sft_step32']['actual_learning_rate'],5e-5)

    def test_saved_vector_loss_and_EOS_coverage_tampering_rejected(self):
        with tensors() as (run,engine,backend,obj):
            obj.warm_checkpoint(1)
            report=engine.existing('warmstart/updates/1.json')
            targets=[obj.target(i) for i in report['schedule']['teacher_indices']]
            before=obj.warm_checkpoint(0)['optimizer_identity'];after=obj.warm_checkpoint(1)['optimizer_identity']
            delta=audit.gpu.checkpoint_delta(run/'warmstart/checkpoints/step1',run/'warmstart/checkpoints/step0')
            altered=deepcopy(report);altered['loss']+=.05
            with self.assertRaisesRegex(ValueError,'program losses'):
                audit.validate_supervised_report(altered,targets,before,after,delta,engine.config)
            altered=deepcopy(report);altered['teacher_forced_train_logprobs'][0].pop()
            with self.assertRaisesRegex(ValueError,'completion/EOS'):
                audit.validate_supervised_report(altered,targets,before,after,delta,engine.config)
            altered=deepcopy(report);altered['teacher_forced_train_logprobs'][0][0]-=.01
            with self.assertRaisesRegex(ValueError,'likelihood gate'):
                audit.validate_supervised_report(altered,targets,before,after,delta,engine.config)

    def test_sft_data_mislabeled_on_policy_rejected(self):
        with tensors() as (run,engine,backend,obj):
            path='warmstart/targets/0.json';record=engine.existing(path);record['on_policy']=True
            warm.common.atomic_json(run/path,record);engine.status['artifacts'][path]=warm.common.digest(run/path);engine.state()
            obj=audit.Audit(run);obj.teacher=engine.corpus
            with self.assertRaisesRegex(ValueError,'ownership'):
                obj.target(0)

    def test_actual_RL_optimizer_LR_or_moments_cannot_inherit_SFT(self):
        import torch
        with tensors() as (run,engine,backend,obj):
            path=run/'checkpoints/base/optimizer.pt'
            state=torch.load(path,map_location='cpu',weights_only=True)
            state['optimizer']['param_groups'][0]['lr']=5e-5
            torch.save(state,path)
            with self.assertRaises(ValueError):obj.checkpoint('base')


class PreparedIntegrationTests(unittest.TestCase):
    def test_full_mock_teacher_receipts_archives_and_prepared_audit(self):
        """Only hand-written fixture controllers; no gateway or model calls."""
        from scripts import collect_cardinal_teacher_named as collector
        from selfplay import program_cardinal_teacher_named as client
        from tests.test_program_cardinal_teacher import response
        from tests.test_program_cardinal_gpu import deployment
        tokenizer=collector.ROOT/'runs/cardinal-teacher-inputs/tokenizer.json'
        if not tokenizer.exists():self.skipTest('Pinned public tokenizer fixture unavailable')
        prior=collector.ROOT/'runs/cardinal-teacher-none-v1'
        prior_auditor=collector.ROOT/'scripts/audit_cardinal_teacher_none.py'
        if not (prior/'independent-audit.json').exists() or not prior_auditor.exists():
            self.skipTest('Portable audited insufficient-corpus evidence unavailable')
        source="""def act():
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
"""
        with tempfile.TemporaryDirectory() as temporary,contextlib.chdir(temporary):
            folder=Path(temporary);Path('runs').mkdir()
            Path('budget.json').write_text(json.dumps({'cloud_spend_to_date':0,'cloud_reserved_usd':80,'approved_pilot_cap':100}))
            teacher=folder/'teacher';run=folder/'learner'
            calls=[]
            def fake(*args,**kwargs):
                index=len(calls);calls.append(index)
                code=source.replace('VARIANT',str(index)) if index<16 else 'def act():\n    break\n'
                return response(choices=[{'finish_reason':'stop','message':{'content':code}}])
            with patch.object(client,'verify_endpoint',return_value={'explicit_fixture':True}):
                collector.prepare(teacher,prior,prior_auditor)
                with patch.object(client.gateway,'api_key',return_value='fixture-never-network'),patch.object(
                        client.urllib.request,'build_opener',return_value=SimpleNamespace(open=fake)):
                    for i in range(32):collector.advance(teacher)
                collector.evaluate(teacher)
            deployment_path=folder/'deployment.json';warm.common.atomic_json(deployment_path,deployment())
            warm.prepare(run,deployment_path,collector.ROOT/'runs/cardinal-gpu-preflight/model-info.json',teacher)
            result=audit.audit(run)
            self.assertTrue(result['integrity_passed'])
            self.assertFalse(result['planned_comparisons_available'])
            self.assertEqual(result['genuine_supervised_optimizer_steps'],0)
            self.assertEqual(result['genuine_RL_optimizer_steps'],0)
            self.assertEqual(result['teacher_replays_verified'],384)
            self.assertTrue(result['map_reuse_provenance']['teacher_maps_previously_graded'])
            self.assertFalse(result['map_reuse_provenance']['learner_maps_previously_scored'])
            self.assertFalse(result['map_reuse_provenance']['prior_outputs_eligible_for_new_selection'])
            self.assertEqual(len(calls),32)


if __name__=='__main__':unittest.main()
