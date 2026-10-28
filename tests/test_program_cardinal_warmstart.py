"""CPU-only warmstart transactions and phase isolation; no model weights."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from selfplay import program_cardinal_warmstart as warm
from tests.test_program_cardinal_gpu import deployment
from tests.test_program_cardinal_sft import Tokenizer


def fake_identity(folder, expected_learning_rate=None):
    result = warm.common.load(folder/'optimizer.pt')
    if result['learning_rate'] != expected_learning_rate:
        raise ValueError('Optimizer LR differs')
    return result


class FakePolicy:
    def __init__(self, lr=5e-5):
        self.config = {'learning_rate': lr}
        self.step, self.weight, self.updates = 0, 0., 0
        self.fail_step = None
        self.reset_weight_change = False
        self.smoke_passed = True
        self.smokes = []

    def teacher_record(self, prompt, text):
        return warm_record(prompt, text)

    def save_checkpoint(self, folder):
        folder.mkdir(parents=True)
        warm.common.atomic_json(folder/'adapter.safetensors', {'weight': self.weight})
        warm.common.atomic_json(folder/'optimizer.pt', {'step': self.step,
            'learning_rate': self.config['learning_rate'], 'moments': {'q': self.step} if self.step else {}})
        if self.step == self.fail_step:
            self.fail_step = None
            raise RuntimeError('Interrupted save')
        return {name: warm.common.digest(folder/name) for name in warm.gpu.FILES}

    def load_checkpoint(self, folder, hashes):
        if any(warm.common.digest(folder/n) != h for n,h in hashes.items()):
            raise ValueError('bytes')
        state = fake_identity(folder, self.config['learning_rate'])
        self.step = state['step']
        self.weight = warm.common.load(folder/'adapter.safetensors')['weight']

    def sft_update(self, records):
        assert len(records) == 4 and self.config['learning_rate'] == 5e-5
        before = self.step
        self.step += 1; self.weight += .01; self.updates += 1
        return {'optimizer_steps': 1, 'adapter_l2_delta': .01,
            'adam_step_before': before, 'adam_step_after': self.step, 'learning_rate': 5e-5}

    def reset_optimizer(self, learning_rate):
        assert self.step == 32 and learning_rate == 1e-5
        self.step = 0; self.config = {'learning_rate': learning_rate}
        if self.reset_weight_change:
            self.weight += 1
        return {'adapter_tensors_unchanged': True, 'rl_moments_empty': True}

    def mechanics_smoke(self, prompt):
        self.smokes.append(self.config['learning_rate'])
        return {'passed': self.smoke_passed, 'optimizer_steps': 0,
            'adapter_unchanged': True, 'optimizer_unchanged': True,
            'rng_restored': True, 'config_restored': True}


def warm_record(prompt, text):
    from selfplay.program_cardinal_sft import make_teacher_record
    return make_teacher_record(Tokenizer(), prompt, text, [0])


@contextmanager
def fixture(backend=None, real_tensors=False):
    with tempfile.TemporaryDirectory() as directory, patch.object(warm, 'SOURCES', []):
        run = Path(directory); dep = deployment(); cfg = warm.config(dep)
        demos = [{'index': i, 'messages': [{'role':'user','content':'fixture'},
            {'role':'assistant','content':'x'*(i+1)}],
            'completion_ids': warm_record('fixture','x'*(i+1))['completion_ids'],
            'generation_sha256': str(i)*64, 'program': {'fixture': i}} for i in range(16)]
        corpus = {'state': 'ready_for_sft', 'all_attempts':32, 'demonstrations':demos,
            'selected_indices':list(range(16))}
        teacher_schedule = {**warm.schedule(), 'capability_maps':warm.schedule()['calibration_maps'],
            'teacher_generation_seeds':list(range(141000000,141000032))}
        del teacher_schedule['calibration_maps']
        teacher_plan = {'kind':'verified_cardinal_teacher_named_corpus','schedule':teacher_schedule,
            'learner':{'model':warm.MODEL_REPO,'revision':warm.MODEL_REVISION}}
        values = {'deployment.json':dep, 'schedule.json':warm.schedule(), 'map-fingerprints.json':{},
            'sft-schedule.json':warm.sft_schedule(corpus['selected_indices']),
            'inputs/teacher/corpus.json':corpus, 'inputs/teacher/plan.json':teacher_plan}
        for name,value in values.items(): warm.common.atomic_json(run/name,value)
        corpus_sha = warm.common.digest(run/'inputs/teacher/corpus.json')
        warm.common.atomic_json(run/'plan.json',warm.plan(dep,corpus_sha))
        (run/'prompts').mkdir(); (run/'prompts/cardinal.txt').write_text('fixture')
        inputs = {name:warm.common.digest(run/name) for name in
            ('deployment.json','sft-schedule.json','inputs/teacher/corpus.json','inputs/teacher/plan.json','prompts/cardinal.txt')}
        manifest = {'kind':cfg['kind'],'config':cfg,'source_sha256':{},'input_sha256':inputs,
            'plan_sha256':warm.common.digest(run/'plan.json'),'schedule_sha256':warm.common.digest(run/'schedule.json'),
            'map_fingerprints_sha256':warm.common.digest(run/'map-fingerprints.json'),
            'prompt_sha256':{'cardinal':inputs['prompts/cardinal.txt']},'teacher_corpus_sha256':corpus_sha}
        warm.common.atomic_json(run/'manifest.json',manifest)
        budget,_ = warm.journal.execution_budget(cfg)
        warm.common.atomic_json(run/'status.json',{'manifest_sha256':warm.common.digest(run/'manifest.json'),
            'artifacts':{},'state':'prepared','claims_allowed':False,'execution_budget':budget})
        backend = backend or FakePolicy()
        if real_tensors:
            yield run,warm.WarmstartCycle(run,backend),backend
        else:
            with patch.object(warm.gpu,'checkpoint_state',fake_identity):
                yield run,warm.WarmstartCycle(run,backend),backend


class WarmstartTests(unittest.TestCase):
    def test_earlier_cohorts_cannot_be_substituted_for_named_cohort(self):
        with fixture() as (run,engine,backend):
            plan=warm.common.load(run/'inputs/teacher/plan.json')
            for start in (111000000,121000000,131000000):
                plan['schedule']['teacher_generation_seeds']=list(range(start,start+32))
                with self.assertRaisesRegex(ValueError,'schedules differ'):
                    warm.verify_teacher_manifest_binding(engine.corpus,plan,warm.schedule())
            for kind in ('verified_cardinal_teacher_corpus','verified_cardinal_teacher_low_corpus',
                    'verified_cardinal_teacher_none_corpus'):
                plan['kind']=kind
                with self.assertRaisesRegex(ValueError,'naming-clarified'):
                    warm.verify_teacher_manifest_binding(engine.corpus,plan,warm.schedule())
            self.assertEqual(engine.config['condition'],'verified-teacher-named-cardinal-gpu-v1')

    def test_fixed_sft_schedule_has_8_complete_epochs_and_no_duplicate_batch(self):
        schedule = warm.sft_schedule(list(range(16)))
        self.assertEqual(len(schedule),32)
        for epoch in range(8):
            rows = schedule[epoch*4:epoch*4+4]
            self.assertEqual(sorted(i for row in rows for i in row['teacher_indices']),list(range(16)))
            self.assertTrue(all(len(set(row['teacher_indices'])) == 4 for row in rows))
        self.assertEqual(schedule,warm.sft_schedule(list(range(16))))

    def test_insufficient_teacher_corpus_cannot_enter_SFT(self):
        with fixture() as (run,engine,backend):
            plan=warm.common.load(run/'inputs/teacher/plan.json')
            corpus={**engine.corpus,'state':'insufficient_verified_demonstrations',
                'qualified_attempts':list(range(12)),'selected_indices':[],
                'demonstrations':[]}
            with self.assertRaisesRegex(ValueError,'Full admitted teacher corpus'):
                warm.verify_teacher_manifest_binding(corpus,plan,warm.schedule())
            self.assertEqual(backend.updates,0)

    def test_32_sft_updates_then_exact_adapter_and_fresh_RL_optimizer(self):
        with fixture() as (run,engine,backend):
            engine.initialize()
            self.assertEqual(backend.updates,32)
            self.assertEqual(backend.step,0)
            self.assertEqual(backend.config['learning_rate'],1e-5)
            final = engine.warm_checkpoint(32); base = engine.checkpoint('base')
            self.assertEqual(final['optimizer_identity']['step'],32)
            self.assertEqual(final['optimizer_identity']['learning_rate'],5e-5)
            self.assertEqual(base['optimizer_identity']['moments'],{})
            self.assertEqual(base['files']['adapter.safetensors'],final['files']['adapter.safetensors'])
            self.assertEqual(len([p for p in engine.status['artifacts'] if p.startswith('warmstart/updates/')]),32)
            resumed = warm.WarmstartCycle(run,backend); resumed.initialize()
            self.assertEqual(backend.updates,32)
            self.assertTrue(resumed.existing('warmstart/optimizer-transition.json')['adapter_file_equality_verified'])

    def test_interrupted_sft_update_restarts_from_parent_without_double_update(self):
        with fixture() as (run,engine,backend):
            backend.fail_step = 7
            with self.assertRaisesRegex(RuntimeError,'Interrupted'): engine.initialize()
            self.assertEqual(backend.step,7)
            self.assertIsNone(engine.warm_checkpoint(7))
            fresh = FakePolicy()
            resumed = warm.WarmstartCycle(run,fresh); resumed.initialize()
            self.assertEqual(fresh.updates,26)
            self.assertAlmostEqual(fresh.weight,.32)
            self.assertEqual(resumed.warm_checkpoint(32)['optimizer_identity']['step'],32)
            self.assertEqual(resumed.checkpoint('base')['optimizer_identity']['step'],0)

    def test_reset_interruption_reuses_SFT32_and_adapter_mismatch_never_commits(self):
        with fixture() as (run,engine,backend):
            backend.reset_weight_change = True
            with self.assertRaisesRegex(ValueError,'adapter bytes'):engine.initialize()
            self.assertIsNone(engine.checkpoint('base'))
            self.assertNotIn('warmstart/optimizer-transition.json',engine.status['artifacts'])
            fresh = FakePolicy(); resumed = warm.WarmstartCycle(run,fresh); resumed.initialize()
            self.assertEqual(fresh.updates,0)
            self.assertEqual(resumed.checkpoint('base')['optimizer_identity']['step'],0)
            self.assertAlmostEqual(fresh.weight,.32)

    def test_failed_fresh_postSFT_smoke_prevents_capability_and_resume_rechecks(self):
        with fixture() as (run,engine,backend):
            backend.smoke_passed = False
            with patch.object(engine,'capability',side_effect=AssertionError('must not sample')):
                engine.run_all()
            self.assertEqual(backend.updates,32)
            self.assertEqual(engine.status['state'],'numerical_failed')
            self.assertEqual(backend.smokes,[1e-5])
            self.assertFalse(any(p.startswith('generations/') for p in engine.status['artifacts']))

    def test_actual_token_measurement_is_not_BatchEncoding_key_count(self):
        measured = warm.prompt_measurement(Tokenizer(),'fixture',{'max_new_tokens':2048,'max_context_tokens':4096})
        self.assertEqual(measured['prompt_tokens'],len(measured['prompt_ids']))
        self.assertGreater(measured['prompt_tokens'],2)
        self.assertTrue(measured['admitted'])

    def test_frozen_teacher_tokens_are_checked_before_supervised_updates(self):
        with fixture() as (run,engine,backend):
            engine.corpus['demonstrations'][0]['completion_ids'] = [123,0]
            with self.assertRaisesRegex(ValueError,'tokenizer'):engine.initialize()
            self.assertEqual(backend.updates,0)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('safetensors'), 'Optional CPU Torch')
class ActualTensorTests(unittest.TestCase):
    def test_32_real_Adam_commits_reset_and_JSON_resume(self):
        import torch
        from tests.test_program_cardinal_sft import policy
        backend = policy()
        gpu_rng = [torch.get_rng_state().clone()]
        with fixture(backend,real_tensors=True) as (run,engine,backend), \
                patch.object(torch.cuda,'get_rng_state_all',side_effect=lambda:[r.clone() for r in gpu_rng]), \
                patch.object(torch.cuda,'set_rng_state_all',side_effect=lambda values:gpu_rng.__setitem__(slice(None),values)):
            engine.initialize()
            self.assertEqual(engine.warm_checkpoint(32)['optimizer_identity']['step'],32)
            self.assertEqual(engine.checkpoint('base')['optimizer_identity']['step'],0)
            self.assertEqual(engine.checkpoint('base')['optimizer_identity']['moments'],{})
            self.assertEqual(engine.checkpoint('base')['files']['adapter.safetensors'],
                engine.warm_checkpoint(32)['files']['adapter.safetensors'])
            engine.initialize()
            self.assertEqual(backend._optimizer_identity()['step'],0)


if __name__ == '__main__':unittest.main()
