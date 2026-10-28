"""CPU-only orchestration fixtures: no Torch/model/network needed."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from selfplay import program_cardinal_gpu as gpu


def deployment():
    return {'intent': 'fixture', 'project': 'cardinal-gpu', 'created_at': gpu.common.now(),
        'deadline_unix': (datetime.now(timezone.utc)+timedelta(hours=17)).timestamp(),
        'maximum_hours': 17, 'reservation_usd': 80, 'estimate_usd': 60, 'gpu_id': 'fixture',
        'maximum_gpu_rate_usd_hour': 3.19, 'maximum_cpu_rate_usd_hour': .06,
        'gpu_disk_gb': 160, 'guard_disk_gb': 10, 'gpu_image': 'fixture',
        'guard_source_sha256': '0'*64, 'cleanup': 'fixture'}


def fake_identity(folder, expected_learning_rate=None):
    value = gpu.common.load(folder/'optimizer.pt')
    if value['learning_rate'] != expected_learning_rate:
        raise ValueError('Optimizer LR differs')
    return {**value, 'trainable_names': ['q', 'v']}


class FakePolicy:
    def __init__(self):
        self.step, self.weight, self.batches, self.updates = 0, 0., [], 0
        self.fail_save = False

    def save_checkpoint(self, folder):
        folder.mkdir(parents=True)
        gpu.common.atomic_json(folder/'adapter.safetensors', {'weight': self.weight})
        gpu.common.atomic_json(folder/'optimizer.pt', {'step': self.step, 'learning_rate': 1e-5})
        if self.fail_save:
            self.fail_save = False
            raise RuntimeError('Simulated interrupted checkpoint')
        return {n: gpu.common.digest(folder/n) for n in gpu.FILES}

    def load_checkpoint(self, folder, hashes):
        assert all(gpu.common.digest(folder/n)==v for n,v in hashes.items())
        self.step = gpu.common.load(folder/'optimizer.pt')['step']
        self.weight = gpu.common.load(folder/'adapter.safetensors')['weight']

    def sample_batch(self, prompt, seeds):
        self.batches.append(list(seeds))
        records = [{'text': "def act():\n    return 'rest'", 'raw_text': "def act():\n    return 'rest'",
            'seed': s, 'finish_reason': 'eos', 'prompt_ids': [1], 'completion_ids': [2],
            'sampled_token_logprobs': [-.5], 'input_tokens': 1, 'output_tokens': 1,
            'eos_token_ids': [2]} for s in seeds]
        return {'records': records, 'timing': {'batch_elapsed_seconds': .1, 'total_output_tokens': 8}}

    def likelihood_gate(self, records):
        return {'passed': True, 'records': [{'passed': True, 'tokens': 1, 'max_token_error': 0.,
            'sequence_error': 0., 'recorded_sequence_logprob': -.5} for r in records]}

    def update(self, records, advantages):
        assert len(records)==len(advantages)==8 and any(advantages)
        self.step+=1;self.weight+=.01;self.updates+=1
        return {'optimizer_steps': 1, 'adapter_l2_delta': .01,
            'gradient_likelihood_gate': self.likelihood_gate(records), 'loss': 1.}


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as d, patch.object(gpu,'SOURCES',[]),patch.object(gpu,'checkpoint_state',fake_identity):
        run=Path(d);dep=deployment();cfg=gpu.config(dep)
        for name,value in [('deployment.json',dep),('plan.json',gpu.plan(dep)),('schedule.json',gpu.schedule()),('map-fingerprints.json',{})]:
            gpu.common.atomic_json(run/name,value)
        (run/'prompts').mkdir();(run/'prompts/cardinal.txt').write_text('fixture')
        inputs={p:gpu.common.digest(run/p) for p in ['deployment.json','prompts/cardinal.txt']}
        manifest={'kind':cfg['kind'],'config':cfg,'source_sha256':{},'input_sha256':inputs,
            'plan_sha256':gpu.common.digest(run/'plan.json'),'schedule_sha256':gpu.common.digest(run/'schedule.json'),
            'map_fingerprints_sha256':gpu.common.digest(run/'map-fingerprints.json'),
            'prompt_sha256':{'cardinal':inputs['prompts/cardinal.txt']}}
        gpu.common.atomic_json(run/'manifest.json',manifest)
        budget,_=gpu.journal.execution_budget(cfg)
        gpu.common.atomic_json(run/'status.json',{'manifest_sha256':gpu.common.digest(run/'manifest.json'),
            'artifacts':{},'state':'prepared','claims_allowed':False,'execution_budget':budget})
        backend=FakePolicy();engine=gpu.GPUCycle(run,backend);engine.initialize()
        yield run,engine,backend


def fake_score(record,*args):
    valid=bool(record['seed']%2)
    return {'classification':{'valid':valid},'runtime_valid':valid,'authorship_valid':valid,
        'reward':.2 if valid else -1.,'game_return':.2 if valid else None,'matches':[],
        'candidate_program_faults':0,'opponent_program_faults':0,'banked_point_yield':0.,'positive_banking_games':0}


def pool(engine):
    return engine.commit('opponents.json',{'selected':[{'id':name,'origin':'scripted' if i<2 else 'model',
        'ir_sha256':name,'compiled_ir':{}} for i,name in enumerate(['nearest','value','initial0','initial1'])]})


class GPUCycleTests(unittest.TestCase):
    def test_batch_first_resume_materializes_missing_records_without_resampling(self):
        with fixture() as (run,engine,backend):
            original=engine.commit
            def crash(relative,value):
                if relative=='generations/capability-4.json':raise RuntimeError('crash after four')
                return original(relative,value)
            names=[f'capability-{i}' for i in range(8)];seeds=engine.schedule['capability_generation_seeds']
            with patch.object(engine,'commit',crash),self.assertRaisesRegex(RuntimeError,'four'):
                engine.ensure_batch('capability',names,seeds,'base')
            self.assertIn('batches/capability.json',engine.status['artifacts'])
            resumed=gpu.GPUCycle(run,backend);rows=resumed.ensure_batch('capability',names,seeds,'base')
            self.assertEqual(len(backend.batches),1);self.assertEqual(len(rows),8)
            self.assertEqual([r['batch_index'] for r in rows],list(range(8)))
            self.assertTrue(all(r['batch_sha256']==resumed.status['artifacts']['batches/capability.json'] for r in rows))

    def test_mislabeled_backend_seed_or_token_rows_rejected_before_commit(self):
        with fixture() as (run,engine,backend):
            seeds=engine.schedule['capability_generation_seeds'];batch=backend.sample_batch('fixture',seeds)
            batch['records'][0]['seed']=seeds[1]
            with patch.object(backend,'sample_batch',return_value=batch),self.assertRaisesRegex(ValueError,'seed/order'):
                engine.ensure_batch('capability',[f'capability-{i}' for i in range(8)],seeds,'base')
            self.assertNotIn('batches/capability.json',engine.status['artifacts'])
            batch['records'][0]['seed']=seeds[0];batch['records'][0]['sampled_token_logprobs']=[]
            with self.assertRaisesRegex(ValueError,'coverage'):
                gpu.validate_batch(batch,seeds,engine.config)

    def test_interrupted_update_reloads_parent_and_reuses_batch(self):
        with fixture() as (run,engine,backend),patch.object(gpu.cardinal,'score',fake_score):
            initial=pool(engine);opponents=engine.roster(1,initial);backend.fail_save=True
            with self.assertRaisesRegex(RuntimeError,'interrupted'):engine.group(1,opponents)
            self.assertIsNone(engine.checkpoint('cycle1'));self.assertEqual(backend.step,1)
            resumed=gpu.GPUCycle(run,backend);resumed.group(1,resumed.roster(1,initial))
            self.assertEqual(len(backend.batches),1);self.assertEqual(backend.step,1)
            self.assertEqual(resumed.checkpoint('cycle1')['optimizer_identity']['step'],1)
            before=backend.updates;resumed.group(1,resumed.roster(1,initial));self.assertEqual(backend.updates,before)
            self.assertIn('groups/1/timing.json',resumed.status['artifacts'])

    def test_zero_signal_exact_copy_and_actual_step_count(self):
        def zero(*args):
            row=fake_score(*args);row.update(reward=0.,runtime_valid=True);return row
        with fixture() as (run,engine,backend),patch.object(gpu.cardinal,'score',zero):
            initial=pool(engine);base=engine.checkpoint('base')['files']
            report=engine.group(1,engine.roster(1,initial))
            self.assertEqual(report['optimizer_steps'],0);self.assertEqual(backend.updates,0)
            self.assertEqual(engine.checkpoint('cycle1')['files'],base)
            self.assertIsNone(report['likelihood_gate_path'])

    def test_timing_endpoint_only_and_reserve(self):
        cfg=gpu.config(deployment())
        self.assertEqual(gpu.endpoint_decision([{'optimizer_steps':1,'elapsed_seconds':100}],20000,cfg)['endpoint'],100)
        self.assertEqual(gpu.endpoint_decision([{'optimizer_steps':1,'elapsed_seconds':100}],12000,cfg)['endpoint'],50)
        self.assertFalse(gpu.endpoint_decision([{'optimizer_steps':1,'elapsed_seconds':100}],10000,cfg)['passed'])
        self.assertFalse(gpu.endpoint_decision([{'optimizer_steps':0,'elapsed_seconds':1}],90000,cfg)['passed'])

    def test_final64_draws_sealed_before_any_scoring(self):
        with fixture() as (run,engine,backend):
            initial=pool(engine)
            # Only endpoint generation/sealing mechanics are tested here; intermediate audits are separate.
            report={'optimizer_steps':0,'adapter_l2_delta':0.}
            for n in range(1,51):engine.save_checkpoint(f'cycle{n}','base' if n==1 else f'cycle{n-1}',report,copy_from=run/'checkpoints/base')
            engine.commit('endpoint-selection.json',{'passed':True,'endpoint':50})
            def score(*args):
                self.assertIn('evaluation/sealed.json',engine.status['artifacts'])
                self.assertEqual(sum(p.startswith('generations/eval-') for p in engine.status['artifacts']),64)
                return fake_score(*args)
            with patch.object(gpu.cardinal,'score',score):engine.evaluate(initial,50)
            self.assertEqual(engine.status['state'],'complete');self.assertEqual(len(backend.batches),8)
            a=engine.existing('evaluation/base/holdout.json');b=engine.existing('evaluation/cycle50/holdout.json')
            self.assertEqual(len(a['scores']),32);self.assertEqual(len(b['scores']),32)
            self.assertEqual(engine.status['actual_optimizer_steps'],0)

    def test_roster_is_frozen_before_group_and_bounded_pruning(self):
        with fixture() as (run,engine,backend):
            initial=pool(engine);snapshot=engine.roster(1,initial)
            self.assertEqual([r['id'] for r in snapshot['selected']],['nearest','value','initial1','initial0'])
            self.assertFalse(backend.batches)
            self.assertEqual(engine.roster(1,initial),snapshot)
        def row(i):
            text=f"def act():\n    if energy > {i}:\n        return 'rest'\n    return 'harvest'"
            classification,_=gpu.cardinal.classify_cardinal(text,'eos')
            return {'generation':f'g{i}','runtime_valid':True,'matches':[{}]*16,
                'positive_banking_games':8,'classification':classification}
        initial=[{'id':'i0','origin':'model','ir_sha256':'i0'},{'id':'i1','origin':'model','ir_sha256':'i1'}]
        prior=initial;seen=[]
        for cycle in [5,10,15,20]:
            rows=[row(cycle+j) for j in range(2)];hashes={r['generation']:'a'*64 for r in rows}
            prior,added=gpu.refresh_roster(initial,prior,rows,hashes,'b'*64,cycle);seen+=added
            self.assertLessEqual(len(prior),6);self.assertEqual(prior[:2],initial)
        self.assertEqual([r['id'] for r in prior[2:]],[r['id'] for r in seen[-4:]])

    def test_fifty_group_journal_and_fixed_final_endpoint_complete(self):
        with fixture() as (run,engine,backend),patch.object(gpu.cardinal,'score',fake_score):
            initial=pool(engine)
            with patch.object(engine,'capability',return_value={'passed':True}),patch.object(engine,'opponents',return_value=initial),patch.object(gpu,'endpoint_decision',return_value={'passed':True,'endpoint':50}):
                engine.run_all()
            self.assertEqual(engine.status['state'],'complete')
            self.assertEqual(engine.status['attempted_groups'],50)
            self.assertEqual(engine.status['actual_optimizer_steps'],50)
            self.assertEqual(backend.updates,50)
            self.assertEqual(len(backend.batches),58) #50 groups plus8 final evaluation batches
            self.assertEqual(engine.checkpoint('cycle50')['optimizer_identity']['step'],50)
            self.assertEqual(sum(p.startswith('league/after-') for p in engine.status['artifacts']),10)
            self.assertEqual(len(engine.existing('evaluation/sealed.json')['generations']),64)

    def test_model_index_binds_authoritative_metadata_and_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);model=root/'model';model.mkdir();siblings=[]
            names=['config.json','generation_config.json','tokenizer.json','tokenizer_config.json',
                   'model.safetensors.index.json','model-00001-of-00001.safetensors']
            for name in names:
                data=b'fixture-'+name.encode();(model/name).write_bytes(data)
                row={'rfilename':name,'size':len(data),'blobId':hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()}
                if name.endswith('.safetensors'):row['lfs']={'sha256':hashlib.sha256(data).hexdigest()}
                siblings.append(row)
            metadata={'id':gpu.MODEL_REPO,'sha':gpu.MODEL_REVISION,'siblings':siblings};gpu.common.atomic_json(root/'info.json',metadata)
            indexed=gpu.index_model(model,root/'info.json')
            self.assertEqual(indexed['repo'],gpu.MODEL_REPO)
            checked=gpu.common.verify_model_index(model,gpu.config(deployment()))
            self.assertEqual(checked['authoritative_metadata_sha256'],gpu.common.digest(root/'info.json'))
            (model/'config.json').write_text('mutated')
            with self.assertRaises(ValueError):gpu.index_model(model,root/'info.json')

    def test_durable_budget_and_schedule_override_rejected(self):
        cfg=gpu.config(deployment());table=gpu.schedule();table['holdout_maps'][0]=table['calibration_maps'][0]
        with self.assertRaises(ValueError):gpu.validate_configuration(cfg,table,deployment())
        now=datetime.now(timezone.utc);budget,_=gpu.journal.execution_budget(cfg,at=now)
        with self.assertRaises(TimeoutError):gpu.journal.execution_budget(cfg,budget,at=now+timedelta(hours=18))


if __name__=='__main__':unittest.main()
