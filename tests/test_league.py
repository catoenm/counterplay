"""CPU-only crash/recovery tests for the bounded continuation runner."""
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from selfplay import league
from selfplay.arena import experiment
from selfplay.arena.policy import action_prompt


MODEL={'model':'fake-model','revision':'fixed-revision'}
LABELS=['A','B']


def write_checkpoint(directory,name,source=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    (directory/'adapters.safetensors').write_text('weights:'+name)
    metadata={**MODEL,'checkpoint':name,'labels':LABELS,'label_token_ids':[1,2],
              'adapter_sha256':league.sha(directory/'adapters.safetensors')}
    league.atomic_json(directory/'metadata.json',metadata)
    if name!='base':
        summary={'complete':True,'source':source or ('base' if name=='cycle1' else f'cycle{int(name[5:])-1}'),
                 'target':name,'parameter_delta_l2':1.,'checkpoint':metadata}
        league.atomic_json(directory/'summary.json',summary)
        return summary


class FakeGame:
    def __init__(self,name,seed,regime):
        self.name,self.seed,self.regime=name,seed,regime
        self.turn,self.done,self.scores=0,False,[0,0]
    def active_players(self):return [] if self.done else [0,1]
    def legal_actions(self,player):return [{'type':'add','n':0},{'type':'add','n':1}]
    def observation(self,player):return {'game':self.name,'player':player,'turn':self.turn,'score':self.scores[player]}
    def heuristic_action(self,player):return {'type':'add','n':1}
    def step(self,actions):
        for player in (0,1):self.scores[player]+=actions[player]['n']
        self.turn+=1;self.done=self.turn==2
    def returns(self):
        value=(self.scores[0]-self.scores[1])/2
        return [value,-value]
    def metrics(self):return {'scores':list(self.scores),'done':self.done}
    def snapshot(self):return {'seed':self.seed,'turn':self.turn,'scores':list(self.scores),'done':self.done}


class FakePolicy:
    def __init__(self,tracker):
        self.tracker=tracker;self.checkpoints={};self.active=None;self.value=None;self.labels=LABELS
        tracker['constructors']+=1
    def activate(self,name):
        if self.active!=name:self.value=self.checkpoints[name];self.active=name
    def choose(self,requests,checkpoint):
        self.activate(checkpoint);results=[]
        for row in requests:
            lp=np.full(len(row['actions']),-np.log(len(row['actions'])))
            probs=np.exp(lp);probs/=probs.sum()
            pick=int(np.random.default_rng(row['sample_seed']).choice(len(lp),p=probs))
            results.append({'action_index':pick,'action':row['actions'][pick],
                'action_label':LABELS[pick],'old_log_prob':float(lp[pick]),'old_log_probs':lp.tolist(),
                'entropy':float(-(probs*np.log(probs)).sum()),'prompt_tokens':1,'checkpoint':checkpoint})
        return results
    def train(self,episodes,source,target,out,seed,**kwargs):
        self.tracker['train_calls']+=1
        if self.value!='weights:'+source:raise AssertionError('Update did not restart from original source')
        self.tracker['training_ids'].append([e['episode_id'] for e in episodes])
        if self.tracker.get('crash_training'):
            self.tracker['crash_training']=False
            Path(out).mkdir();(Path(out)/'summary.json').write_text('{"complete":')
            self.value='half-updated-weights'
            raise RuntimeError('simulated interrupted optimizer')
        summary=write_checkpoint(out,target,source)
        self.checkpoints[target]='weights:'+target;self.active=target;self.value=self.checkpoints[target]
        return summary


def fake_loader(policy,name,directory):
    record,_=league.checkpoint_record(directory,name,MODEL)
    policy.checkpoints[name]=(Path(directory)/'adapters.safetensors').read_text()
    policy.active=None;policy.activate(name)
    return record


class LeagueTests(unittest.TestCase):
    def fixture(self,root,tracker=None):
        tracker=tracker or {'constructors':0,'train_calls':0,'training_ids':[],'played':[]}
        out=Path(root)/'league';out.mkdir()
        (out/'checkpoints').mkdir();(out/'rollout-parts').mkdir()
        inherited={}
        for name in ('base','cycle1','cycle2'):
            path=Path(root)/'parent/checkpoints'/name;write_checkpoint(path,name)
            inherited[name]=league.checkpoint_record(path,name,MODEL)[0]
        schedule,rosters=league.make_schedule(['fake'],cycles=(3,))
        manifest={'model':MODEL,'deadline':'2030-01-01T08:00:00-08:00','continuation_cycles':[3],
            'evaluation_rosters':rosters,'inherited_checkpoints':inherited,'parent_run':str(Path(root)/'parent'),
            'inference_batch_size':8,'rollout_chunk_episodes':6,'minibatch_size':4,'learning_rate':2e-5}
        league.atomic_json(out/'status.json',{'status':'prepared','stages':{}})
        return out,manifest,schedule,tracker

    def runner(self,fixture,play_override=None,now=None):
        out,manifest,schedule,tracker=fixture
        def play(policy,specs,path,batch_size):
            tracker['played'].extend(s['episode_id'] for s in specs)
            with patch.object(experiment,'make_env',FakeGame),redirect_stdout(io.StringIO()):
                return experiment.play(policy,specs,path,batch_size)
        real_play=play
        if play_override:play=lambda *a,**kw:play_override(real_play,*a,**kw)
        return league.Continuation(out,manifest,schedule,SimpleNamespace(make_env=FakeGame),
            SimpleNamespace(action_prompt=action_prompt),SimpleNamespace(play=play),
            policy_factory=lambda:FakePolicy(tracker),checkpoint_loader=fake_loader,
            now=now or (lambda:datetime.fromisoformat('2026-10-21T22:00:00-07:00')))

    def test_roster_is_bounded_deterministic_and_keeps_recent_anchors(self):
        available=['base']+[f'cycle{i}' for i in range(1,20)]
        result=league.evaluation_roster(20,available)
        self.assertLessEqual(len(result)+1,8)
        self.assertEqual(result[:4],['heuristic','base','cycle19','cycle18'])
        self.assertEqual(result,league.evaluation_roster(20,list(reversed(available))))

    def test_adapter_validation_rejects_missing_extra_wrong_shape_and_nonfinite_tensors(self):
        expected={'layer.lora_a':np.zeros((2,3)),'layer.lora_b':np.zeros((3,2))}
        saved=league.validate_adapter_tensors(expected,expected)
        self.assertEqual(set(saved),set(expected))
        bad=[{'layer.lora_a':expected['layer.lora_a']},
             {**expected,'unexpected':np.zeros((1,))},
             {**expected,'layer.lora_a':np.zeros((3,2))},
             {**expected,'layer.lora_a':np.full((2,3),np.nan)}]
        for tensors in bad:
            with self.subTest(keys=list(tensors)):
                with self.assertRaises(ValueError):league.validate_adapter_tensors(tensors,expected)

    def test_loaded_adapter_must_equal_saved_values(self):
        saved={'lora_a':np.ones((2,3)),'lora_b':np.zeros((3,2))}
        league.verify_loaded_adapter_tensors(saved,{key:value.copy() for key,value in saved.items()})
        with self.assertRaises(ValueError):
            league.verify_loaded_adapter_tensors(saved,{**saved,'lora_b':np.ones((3,2))})

    def test_schedules_have_disjoint_families_and_unique_ids(self):
        schedule,_=league.make_schedule(['market','maze','chess'])
        rows=[row for group in schedule.values() for row in group]
        self.assertEqual(len(rows),len({r['episode_id'] for r in rows}))
        families=lambda rows:{(r['environment'],r['seed']//2 if r['environment']=='chess' else r['seed']) for r in rows}
        train=[r for r in rows if r['regime']=='train'];evaluation=[r for r in rows if r['regime']=='eval']
        self.assertFalse(families(train)&families(evaluation))
        self.assertEqual(len(families(train)),48)
        self.assertTrue(all(r['policies'][0]==r['policies'][1] for r in train))

    def test_completed_episode_survives_trailing_write_and_is_not_replayed(self):
        with tempfile.TemporaryDirectory() as root:
            fixture=self.fixture(root);failed=[False]
            def interrupt(play,policy,specs,path,batch_size):
                if not failed[0]:
                    failed[0]=True;play(policy,specs[:1],path,batch_size)
                    with Path(path).open('a') as handle:handle.write('{"incomplete":')
                    raise RuntimeError('simulated rollout crash')
                return play(policy,specs,path,batch_size)
            with self.assertRaises(RuntimeError):self.runner(fixture,interrupt).run()
            result=self.runner(fixture).run()
            self.assertEqual(result['status'],'complete')
            self.assertTrue(all(n==1 for n in Counter(fixture[3]['played']).values()))
            self.assertEqual(len(league.complete_jsonl_rows(fixture[0]/'train-rollouts-3.jsonl')),8)
            self.assertEqual(fixture[3]['train_calls'],1)

    def test_incomplete_update_is_preserved_and_restarts_from_source(self):
        with tempfile.TemporaryDirectory() as root:
            fixture=self.fixture(root);fixture[3]['crash_training']=True
            with self.assertRaises(RuntimeError):self.runner(fixture).run()
            self.runner(fixture).run()
            self.assertEqual(fixture[3]['train_calls'],2)
            self.assertEqual(fixture[3]['training_ids'][0],fixture[3]['training_ids'][1])
            self.assertEqual(len(list((fixture[0]/'checkpoints').glob('cycle3.incomplete-*'))),1)
            self.assertEqual(len(Counter(fixture[3]['played'])),len(fixture[3]['played']))

    def test_finished_checkpoint_is_reused_after_evaluation_crash(self):
        with tempfile.TemporaryDirectory() as root:
            fixture=self.fixture(root);failed=[False]
            def interrupt(play,policy,specs,path,batch_size):
                if specs[0]['regime']=='eval' and not failed[0]:
                    failed[0]=True;raise RuntimeError('simulated evaluation crash')
                return play(policy,specs,path,batch_size)
            with self.assertRaises(RuntimeError):self.runner(fixture,interrupt).run()
            self.assertEqual(fixture[3]['train_calls'],1)
            self.runner(fixture).run()
            self.assertEqual(fixture[3]['train_calls'],1)
            # A fully completed run can be revalidated without model allocation.
            count=fixture[3]['constructors'];self.runner(fixture).run()
            self.assertEqual(fixture[3]['constructors'],count)

    def test_expired_deadline_never_allocates_model_or_starts_episode(self):
        with tempfile.TemporaryDirectory() as root:
            fixture=self.fixture(root)
            result=self.runner(fixture,now=lambda:datetime.fromisoformat('2030-01-02T08:00:00-08:00')).run()
            self.assertEqual(result['status'],'deadline_reached')
            self.assertEqual(fixture[3]['constructors'],0)
            self.assertEqual(fixture[3]['played'],[])

    def test_corrupted_completed_episode_is_rejected_before_resume(self):
        with tempfile.TemporaryDirectory() as root:
            fixture=self.fixture(root);self.runner(fixture).run()
            path=fixture[0]/'train-rollouts-3.jsonl'
            rows=league.complete_jsonl_rows(path)
            rows[0]['transitions'][0]['observation']['hidden_state']='leak'
            league.atomic_rows(path,rows)
            with self.assertRaises(ValueError):self.runner(fixture).run()

    def test_process_lock_refuses_duplicate_jobs(self):
        with tempfile.TemporaryDirectory() as root:
            out=Path(root)/'league'
            with league.process_lock(out):
                with self.assertRaises(RuntimeError):
                    with league.process_lock(out):pass
            with league.process_lock(out):pass

    def test_completed_checkpoint_identity_cannot_be_rewritten_on_resume(self):
        with tempfile.TemporaryDirectory() as root:
            fixture=self.fixture(root);self.runner(fixture).run()
            directory=fixture[0]/'checkpoints/cycle3'
            (directory/'adapters.safetensors').write_text('changed-but-self-consistent')
            metadata=json.loads((directory/'metadata.json').read_text())
            metadata['adapter_sha256']=league.sha(directory/'adapters.safetensors')
            league.atomic_json(directory/'metadata.json',metadata)
            summary=json.loads((directory/'summary.json').read_text());summary['checkpoint']=metadata
            league.atomic_json(directory/'summary.json',summary)
            with self.assertRaises(ValueError):self.runner(fixture).run()

    def test_preparation_freezes_configuration_and_requires_audited_parent(self):
        with tempfile.TemporaryDirectory() as root:
            parent=Path(root)/'parent';(parent/'source').mkdir(parents=True)
            source=parent/'source/__init__.py';source.write_text('# frozen test factory\n')
            manifest={'environments':['fake'],'model':MODEL,'versions':{},
                      'source_sha256':{'selfplay/arena/__init__.py':league.sha(source)}}
            league.atomic_json(parent/'manifest.json',manifest)
            league.atomic_json(parent/'summary.json',{'complete':True})
            checkpoints={}
            for name in ('base','cycle1','cycle2'):
                path=parent/'checkpoints'/name;write_checkpoint(path,name)
                checkpoints[name]={'adapter_sha256':league.sha(path/'adapters.safetensors')}
            audit={'complete':False,'all_completed_episodes_verified':True,
                'manifest_sha256':league.sha(parent/'manifest.json'),'checkpoint_verification':checkpoints}
            league.atomic_json(parent/'analysis.json',audit)
            out=Path(root)/'prepared'
            with self.assertRaises(ValueError):league.prepare(parent,out)
            self.assertFalse(out.exists())
            audit['complete']=True;league.atomic_json(parent/'analysis.json',audit)
            first,schedule=league.prepare(parent,out)
            second,again=league.prepare(parent,out)
            self.assertEqual(first,second);self.assertEqual(schedule,again)
            with self.assertRaises(ValueError):league.prepare(parent,out,'2030-01-01T08:00:00-08:00')
            (out/'source/__init__.py').write_text('# changed\n')
            with self.assertRaises(ValueError):league.prepare(parent,out)


if __name__=='__main__':
    unittest.main()
