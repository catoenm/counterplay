"""CPU-only precision-protocol, admission, likelihood, and continuation tests."""
from contextlib import redirect_stdout
from datetime import datetime
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from selfplay import arena_v2 as v2,league
from selfplay.arena import experiment
from selfplay.arena.policy import action_prompt
from tests.test_league import FakeGame,FakePolicy,MODEL,LABELS,fake_loader,write_checkpoint


def diagnostic():
    return {'checkpoint':'cycle1','floating_parameter_dtype':'float32',
            'quantized_integer_weights_unchanged':True,'tests':[
                {'mode':mode,'batch_size':size,'rows':32,'errors':[1e-5]*32,'max_full_vector_error':1e-5}
                for mode,size in [('eval',8),('eval',4),('eval',1),('train',4)]]}


class V2FakePolicy(FakePolicy):
    def __init__(self,tracker):
        super().__init__(tracker)
        self.model=SimpleNamespace(train=lambda:None)
        self.mx=SimpleNamespace(eval=lambda *values:None)
    def tokens(self,prompt):return [1]
    def arrays(self,rows):return (rows,)
    def log_probs(self,model,rows):
        return np.array([r['old_log_probs'] for r in rows])
    def train(self,episodes,source,target,out,seed,**kwargs):
        summary=super().train(episodes,source,target,out,seed,**kwargs)
        summary.update(updates=math.ceil(sum(len(e['transitions']) for e in episodes)/kwargs['batch_size']))
        league.atomic_json(Path(out)/'summary.json',summary)
        return summary


class ArenaV2Tests(unittest.TestCase):
    def test_precision_diagnostic_requires_every_completed_condition(self):
        self.assertEqual(v2.validate_diagnostic(diagnostic())['conditions'],4)
        for mutate in (lambda d:d['tests'].pop(),lambda d:d['tests'][0]['errors'].pop(),
                       lambda d:d['tests'][1].update(errors=[.01]*32,max_full_vector_error=.01)):
            report=diagnostic();mutate(report)
            with self.assertRaises(ValueError):v2.validate_diagnostic(report)

    def test_cast_verification_rejects_integer_changes_and_non_float32(self):
        integers={'packed':v2.tensor_digest(np.array([1,2],dtype=np.uint32))}
        after={'packed':np.array([1,2],dtype=np.uint32),'norm':np.ones(3,dtype=np.float32)}
        self.assertTrue(v2.verify_precision_parameters(integers,after,{'norm'})['integer_weights_exactly_unchanged'])
        for key,value in [('packed',np.array([1,3],dtype=np.uint32)),('norm',np.ones(3,dtype=np.float16))]:
            with self.assertRaises(ValueError):v2.verify_precision_parameters(integers,{**after,key:value},{'norm'})

    def test_fresh_schedule_has_baselines_disjoint_seed_families_and_bounded_rosters(self):
        schedule,rosters,initial=v2.make_schedule(['market','maze','chess'])
        rows=[r for group in schedule.values() for r in group]
        self.assertEqual(len(initial),4);self.assertEqual(len(rows),246)
        self.assertEqual(len(rows),len({r['episode_id'] for r in rows}))
        families=lambda regime:{(r['environment'],r['seed']//2 if r['environment']=='chess' else r['seed'])
                                for r in rows if r['regime']==regime}
        self.assertFalse(families('train')&families('eval'))
        self.assertEqual(len(families('train')),48)
        self.assertEqual(rosters['cycle2'],['heuristic','base','cycle1'])
        self.assertTrue(all(len(roster)+1<=8 for roster in rosters.values()))

    def test_full_vector_gate_catches_unselected_action_error_before_update(self):
        class Policy:
            active=None
            model=SimpleNamespace(train=lambda:None)
            mx=SimpleNamespace(eval=lambda *x:None)
            def activate(self,name):self.active=name
            def tokens(self,prompt):return [1]
            def arrays(self,rows):return (rows,)
            def log_probs(self,model,rows):return np.array([[math.log(.5),math.log(.5)+.1] for _ in rows])
        episodes=[{'episode_id':'one','transitions':[{'checkpoint':'cycle1','prompt':'test',
                    'actions':[{},{}],'action_index':0,'old_log_probs':[math.log(.5)]*2,'turn':0,'player':0}]}]
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'gate.json'
            with self.assertRaises(ValueError):v2.full_likelihood_gate(Policy(),episodes,'cycle1',4,17,path)
            report=json.loads(path.read_text())
            self.assertFalse(report['complete']);self.assertEqual(report['rows'][0]['selected_error'],0)
            self.assertGreater(report['rows'][0]['full_vector_error'],.09)

    def fixture(self,root):
        root=Path(root);out=root/'v2';out.mkdir();(out/'checkpoints').mkdir();(out/'rollout-parts').mkdir()
        inherited={}
        for name in ('base','cycle1'):
            path=root/'parent/checkpoints'/name;write_checkpoint(path,name)
            inherited[name]=league.checkpoint_record(path,name,MODEL)[0]
        schedule,rosters,initial=v2.make_schedule(['fake'])
        manifest={'model':MODEL,'deadline':'2030-01-01T08:00:00-08:00','continuation_cycles':[2,3],
                  'inherited_checkpoints':inherited,'parent_run':str(root/'parent'),'evaluation_rosters':rosters,
                  'initial_evaluation_stages':initial,'inference_batch_size':8,'minibatch_size':4,
                  'rollout_chunk_episodes':6,'learning_rate':2e-5,'likelihood_tolerance':v2.TOLERANCE}
        league.atomic_json(out/'status.json',{'status':'prepared','stages':{}})
        tracker={'constructors':0,'train_calls':0,'training_ids':[],'played':[]}
        def play(policy,specs,path,batch_size):
            tracker['played'] += [s['episode_id'] for s in specs]
            with patch.object(experiment,'make_env',FakeGame),redirect_stdout(io.StringIO()):
                return experiment.play(policy,specs,path,batch_size)
        def runner():
            return v2.PrecisionContinuation(out,manifest,schedule,SimpleNamespace(make_env=FakeGame),
                    SimpleNamespace(action_prompt=action_prompt),
                    SimpleNamespace(play=play,summarize_eval=experiment.summarize_eval),
                    policy_factory=lambda:V2FakePolicy(tracker),checkpoint_loader=fake_loader,
                    now=lambda:datetime.fromisoformat('2026-10-22T00:00:00-07:00'))
        return out,tracker,runner

    def test_two_fresh_cycles_and_baselines_resume_without_duplicate_update_or_model(self):
        with tempfile.TemporaryDirectory() as root,redirect_stdout(io.StringIO()):
            out,tracker,runner=self.fixture(root)
            self.assertEqual(runner().run()['status'],'complete')
            self.assertEqual(tracker['train_calls'],2)
            self.assertEqual(len(tracker['played']),82)
            self.assertEqual(len(set(tracker['played'])),82)
            constructions=tracker['constructors'];runner().run()
            self.assertEqual(tracker['constructors'],constructions);self.assertEqual(tracker['train_calls'],2)
            self.assertEqual(json.loads((out/'likelihood-cycle2.json').read_text())['expected_rows'],32)
            (out/'likelihood-cycle2.json').write_text('{}')
            with self.assertRaises(ValueError):runner().run()

    def test_incomplete_update_restarts_from_original_source_and_same_fresh_rollouts(self):
        with tempfile.TemporaryDirectory() as root,redirect_stdout(io.StringIO()):
            out,tracker,runner=self.fixture(root);tracker['crash_training']=True
            with self.assertRaises(RuntimeError):runner().run()
            runner().run()
            self.assertEqual(tracker['train_calls'],3)
            self.assertEqual(tracker['training_ids'][0],tracker['training_ids'][1])
            self.assertEqual(len(list((out/'checkpoints').glob('cycle2.incomplete-*'))),1)
            self.assertEqual(len(tracker['played']),len(set(tracker['played'])))

    def test_prepare_accepts_verified_partial_parent_and_freezes_precision_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);parent=root/'parent';(parent/'source').mkdir(parents=True)
            source=parent/'source/__init__.py';source.write_text('# fake archived games\n')
            manifest={'model':MODEL,'environments':['fake'],'versions':{},
                      'source_sha256':{'selfplay/arena/__init__.py':league.sha(source)}}
            league.atomic_json(parent/'manifest.json',manifest)
            checkpoints={}
            for name in ('base','cycle1'):
                directory=parent/'checkpoints'/name;write_checkpoint(directory,name)
                checkpoints[name]={'adapter_sha256':league.sha(directory/'adapters.safetensors')}
            league.atomic_json(parent/'analysis.json',{'complete':False,'all_completed_episodes_verified':True,
                'manifest_sha256':league.sha(parent/'manifest.json'),'checkpoint_verification':checkpoints})
            report=root/'diagnostic.json';league.atomic_json(report,diagnostic())
            out=root/'v2';saved,schedule=v2.prepare(parent,out,report)
            self.assertFalse(saved['parent_complete']);self.assertEqual(saved['minibatch_size'],4)
            self.assertEqual(saved['policy_arithmetic_sha256'],v2.ARITHMETIC_SHA)
            self.assertEqual(v2.prepare(parent,out,report),(saved,schedule))
            parent_audit=(out/'parent-analysis.json').read_bytes()
            (out/'parent-analysis.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'parent analysis changed'):v2.prepare(parent,out,report)
            (out/'parent-analysis.json').write_bytes(parent_audit)
            (out/'precision-diagnostic.json').write_text('{}')
            with self.assertRaises(ValueError):v2.prepare(parent,out,report)


if __name__=='__main__':unittest.main()
