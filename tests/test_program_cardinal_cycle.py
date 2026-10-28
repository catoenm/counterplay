"""CPU fixtures for cardinal RL provenance/resume; never loads a model."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from safetensors.numpy import load_file, save_file

from selfplay import program_cardinal_cycle as cycle


def zero_files(folder, lr=1e-6):
    folder.mkdir(parents=True)
    arrays = {f"self_attn.{p}.lora_{s}": np.full((2, 2), .1 if s == "a" else 0, np.float32)
              for p in ("q_proj", "v_proj") for s in ("a", "b")}
    optimizer = {name+suffix: np.zeros_like(value) for name, value in arrays.items() for suffix in (".m", ".v")}
    optimizer.update(step=np.array(0, dtype=np.uint64), learning_rate=np.array(lr, np.float32))
    save_file(arrays, str(folder / "adapter.safetensors"))
    save_file(optimizer, str(folder / "optimizer.safetensors"))


class FakePolicy:
    def __init__(self):
        self.arrays, self.adam = {}, {}
        self.samples, self.updates = [], 0
        self.fail_save = False

    def load_checkpoint(self, folder, hashes):
        for name, expected in hashes.items():
            assert cycle.common.digest(folder/name) == expected
        self.arrays = load_file(str(folder/'adapter.safetensors'))
        self.adam = load_file(str(folder/'optimizer.safetensors'))
        assert self.adam['learning_rate'].item() == np.float32(1e-5).item()

    def save_checkpoint(self, folder):
        folder.mkdir(parents=True)
        save_file(self.arrays, str(folder/'adapter.safetensors'))
        save_file(self.adam, str(folder/'optimizer.safetensors'))
        if self.fail_save:
            self.fail_save = False
            raise RuntimeError('Simulated interrupted checkpoint')
        return {f: cycle.common.digest(folder/f) for f in ('adapter.safetensors','optimizer.safetensors')}

    def sample(self, prompt, seed):
        self.samples.append(seed)
        return {'text': "def act():\n    return 'rest'", 'raw_text': "def act():\n    return 'rest'",
                'seed': seed, 'finish_reason': 'eos', 'prompt_ids': [1], 'completion_ids': [2],
                'sampled_token_logprobs': [-.5], 'input_tokens': 1, 'output_tokens': 1, 'eos_token_ids': [2]}

    def likelihood_gate(self, records):
        return {'passed': True, 'records': [{'tokens': 1, 'passed': True, 'max_token_error': 0,
                    'sequence_error': 0, 'recorded_sequence_logprob': -.5} for _ in records]}

    def update(self, records, advantages):
        assert len(records) == len(advantages) == 8 and any(advantages)
        self.updates += 1
        self.adam['step'] += np.uint64(1)
        for name, value in self.arrays.items():
            value += np.float32(.001)
            self.adam[name+'.m'] += np.float32(.1)
            self.adam[name+'.v'] += np.float32(.01)
        return {'optimizer_steps': 1, 'adapter_l2_delta': .004,
                'gradient_likelihood_gate': self.likelihood_gate(records), 'loss': 1.,
                'gradient_norm_before_clip': 1., 'microbatch_losses': [.125]*8, 'peak_memory_bytes': 1}


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as d, patch.object(cycle, 'SOURCES', []):
        run = Path(d)
        zero_files(run/'inputs/parent-base')
        cycle.initialize_optimizer_copy(run/'inputs/parent-base', run/'inputs/base', 1e-5)
        files = {n: cycle.common.digest(run/'inputs/base'/n) for n in ('adapter.safetensors','optimizer.safetensors')}
        metadata = {'complete': True, 'label': 'base', 'parent': None, 'report': None, 'files': files,
                    'model_repo': cycle.local.MODEL_REPO, 'model_revision': cycle.local.MODEL_REVISION}
        cycle.common.atomic_json(run/'inputs/base/metadata.json', metadata)
        for name, value in [('plan.json',cycle.plan()),('schedule.json',cycle.schedule()),('map-fingerprints.json',{})]:
            cycle.common.atomic_json(run/name,value)
        (run/'prompts').mkdir();(run/'prompts/cardinal.txt').write_text('fixture')
        copied = ['inputs/base/'+n for n in ['metadata.json',*files]]+['prompts/cardinal.txt']
        manifest = {'kind': cycle.config()['kind'], 'config': cycle.config(), 'source_sha256': {},
            'plan_sha256':cycle.common.digest(run/'plan.json'),'schedule_sha256':cycle.common.digest(run/'schedule.json'),
            'map_fingerprints_sha256':cycle.common.digest(run/'map-fingerprints.json'),
            'input_sha256':{p:cycle.common.digest(run/p) for p in copied},
            'prompt_sha256':{'cardinal':cycle.common.digest(run/'prompts/cardinal.txt')}}
        cycle.common.atomic_json(run/'manifest.json',manifest)
        cycle.common.atomic_json(run/'status.json',{'manifest_sha256':cycle.common.digest(run/'manifest.json'),
            'artifacts':{},'state':'prepared','claims_allowed':False})
        backend=FakePolicy();engine=cycle.Cycle(run,backend);engine.initialize()
        yield run,engine,backend


def fake_score(record, opponents, seeds, cfg):
    valid=bool(record['seed']%2)
    return {'classification':{'valid':valid},'runtime_valid':valid,'authorship_valid':valid,
            'reward':.2 if valid else -1.,'game_return':.2 if valid else None,'matches':[],
            'candidate_program_faults':0,'opponent_program_faults':0,'banked_point_yield':0.,'positive_banking_games':0}


def gate_row(action='rest', positive=12, faults=0):
    classification,_=cycle.classify_cardinal(f"def act():\n    return '{action}'",'eos')
    return {'runtime_valid':not faults,'matches':[{}]*24,'positive_banking_games':positive,
            'classification':classification}


class CardinalCycleTests(unittest.TestCase):
    def test_optimizer_initialization_changes_only_lr_and_rejects_nonzero_state(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);zero_files(root/'original')
            report=cycle.initialize_optimizer_copy(root/'original',root/'new',1e-5)
            self.assertEqual(report['changed_tensor_names'],['learning_rate'])
            self.assertEqual(report['before']['adapter_sha256'],report['after']['adapter_sha256'])
            before=load_file(str(root/'original/optimizer.safetensors'));after=load_file(str(root/'new/optimizer.safetensors'))
            for key in before:
                if key!='learning_rate':self.assertTrue(np.array_equal(before[key],after[key]))
            self.assertEqual(after['learning_rate'].item(),np.float32(1e-5).item())
            with self.assertRaisesRegex(ValueError,'learning rate'):
                cycle.optimizer_identity(root/'new',1e-6)
            before['step']=np.array(1,np.uint64);save_file(before,str(root/'original/optimizer.safetensors'))
            with self.assertRaisesRegex(ValueError,'step'):
                cycle.initialize_optimizer_copy(root/'original',root/'rejected',1e-5)

    def test_all_eight_gate_and_first_two_distinct_selection(self):
        rows=[gate_row('rest'),gate_row('rest'),gate_row('deposit'),gate_row('harvest')]+[gate_row(positive=11)]*4
        decision=cycle.capability_decision(rows,cycle.config())
        self.assertTrue(decision['passed']);self.assertEqual(decision['selected_indices'],[0,2])
        with self.assertRaisesRegex(ValueError,'all eight'):
            cycle.capability_decision(rows[:7],cycle.config())
        rows[3]=gate_row(positive=11)
        self.assertFalse(cycle.capability_decision(rows,cycle.config())['passed'])
        self.assertEqual(cycle.capability_decision(rows,cycle.config())['selected_indices'],[])
        self.assertFalse(cycle.capability_decision([gate_row()]*8,cycle.config())['passed'])

    def test_penalty_is_candidate_fault_only_and_failed_legal_actions_remain_valid(self):
        fault="def act():\n    x = 1 // 0\n    return 'rest'"
        ir=cycle.classify_cardinal(fault,'eos')[0]['program']['compiled_ir']
        rival=[{'id':'faulty','origin':'model','compiled_ir':ir}]
        row=cycle.score({'text':"def act():\n    return 'harvest'",'finish_reason':'eos'},rival,[7890123],cycle.config())
        self.assertTrue(row['runtime_valid']);self.assertEqual(row['reward'],0)
        self.assertGreater(row['opponent_program_faults'],0)
        row=cycle.score({'text':fault,'finish_reason':'eos'},rival,[7890123],cycle.config())
        self.assertFalse(row['runtime_valid']);self.assertEqual(row['reward'],-1)

    def test_failed_global_gate_retains_eight_draws_and_never_trains(self):
        with fixture() as (run,engine,backend):
            engine.run_all()
            self.assertEqual(engine.status['state'],'capability_failed')
            self.assertEqual(len(backend.samples),8);self.assertEqual(backend.updates,0)
            self.assertFalse((run/'opponents.json').exists())
            self.assertFalse((run/'evaluation').exists())

    def test_resume_interrupted_update_restarts_from_parent_without_resampling(self):
        with fixture() as (run,engine,backend),patch.object(cycle,'score',fake_score):
            pool=engine.commit('opponents.json',{'selected':[]})
            backend.fail_save=True
            with self.assertRaisesRegex(RuntimeError,'interrupted'):
                engine.group(1,pool)
            self.assertIsNone(engine.checkpoint('cycle1'))
            self.assertEqual(len(backend.samples),8)
            resumed=cycle.Cycle(run,backend);resumed.group(1,pool)
            self.assertEqual(len(backend.samples),8)
            self.assertEqual(resumed.checkpoint('cycle1')['optimizer_identity']['step'],1)
            old_updates=backend.updates;resumed.group(1,pool)
            self.assertEqual(backend.updates,old_updates)
            self.assertTrue(list((run/'checkpoints').glob('.cycle1.pending-*')))

    def test_zero_signal_copies_exact_checkpoint_and_all_eval_draws_are_sealed_first(self):
        def zero(*args):
            result=fake_score(*args);result.update(reward=0.,runtime_valid=True);return result
        with fixture() as (run,engine,backend),patch.object(cycle,'score',zero):
            pool=engine.commit('opponents.json',{'selected':[]})
            base=engine.checkpoint('base')['files']
            for number in range(1,4):
                report=engine.group(number,pool)
                self.assertEqual(report['optimizer_steps'],0)
                self.assertEqual(engine.checkpoint(f'cycle{number}')['files'],base)
            def eval_score(*args):
                self.assertIn('evaluation/sealed.json',engine.status['artifacts'])
                self.assertEqual(sum(p.startswith('generations/eval-') for p in engine.status['artifacts']),24)
                return zero(*args)
            with patch.object(cycle,'score',eval_score):engine.evaluate(pool)
            self.assertEqual(engine.status['state'],'complete')
            self.assertEqual(backend.updates,0)
            self.assertEqual(len(backend.samples),48) #24 train+24 eval; capability not invoked in fixture.

    def test_durable_deadline_blocks_completion_even_before_absolute_deadline(self):
        with fixture() as (run,engine,backend):
            first=datetime.now(timezone.utc)-timedelta(hours=7)
            stop=first+timedelta(seconds=engine.config['worker_seconds'])
            engine.status['execution_budget']={'first_started':first.isoformat(),'hard_stop':stop.isoformat(),
                'maximum_seconds':engine.config['worker_seconds'],'resume_counts_wall_time':True}
            with self.assertRaises(TimeoutError):engine.admit()
            self.assertFalse(engine.status['claims_allowed'])


if __name__=='__main__':unittest.main()
