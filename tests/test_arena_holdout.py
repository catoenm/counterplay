"""Final-evaluation pairing, deadline, no-training, and resume behavior."""
from collections import Counter
from copy import deepcopy
from contextlib import redirect_stdout
from datetime import datetime
import io
import json
from pathlib import Path
import tempfile
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from selfplay import arena_holdout as holdout, arena_v2, league
from selfplay.arena import experiment
from selfplay.arena.policy import action_prompt
from scripts.analyze_arena_holdout import canonical_rows
from tests.test_league import FakeGame,FakePolicy,MODEL,fake_loader,write_checkpoint


class HoldoutTests(unittest.TestCase):
    def parent_fixture(self,root):
        parent=Path(root)/'parent';(parent/'source').mkdir(parents=True)
        sources={}
        for key,path in {'selfplay/league.py':Path(league.__file__),
                         'selfplay/arena_v2.py':Path(arena_v2.__file__)}.items():
            shutil.copyfile(path,parent/'source'/path.name);sources[key]=league.sha(path)
        write_checkpoint(parent/'checkpoints/base','base')
        base=league.checkpoint_record(parent/'checkpoints/base','base',MODEL)[0]
        manifest={'model':MODEL,'environments':['fake'],'versions':{},'horizons':{'fake':2},
                  'source_sha256':sources,'inherited_checkpoints':{'base':base},
                  'rating_compatibility':arena_v2.COMPATIBILITY,'policy_arithmetic':arena_v2.ARITHMETIC,
                  'policy_arithmetic_sha256':arena_v2.ARITHMETIC_SHA,'inference_batch_size':8}
        league.atomic_json(parent/'manifest.json',manifest)
        schedule,_,_=arena_v2.make_schedule(['fake']);league.atomic_json(parent/'schedule.json',schedule)
        return parent,manifest,schedule

    def complete_parent(self,parent,manifest,schedule):
        directory=parent/'checkpoints/cycle3';write_checkpoint(directory,'cycle3')
        meta=json.loads((directory/'metadata.json').read_text())
        meta.update(policy_arithmetic=arena_v2.ARITHMETIC,policy_arithmetic_sha256=arena_v2.ARITHMETIC_SHA,
                    arithmetic_verification={'integer_weights_exactly_unchanged':True,'all_floating_parameters_float32':True})
        league.atomic_json(directory/'metadata.json',meta)
        files={}
        for stage,specs in schedule.items():
            path=parent/(stage+'.jsonl');league.atomic_rows(path,[{'episode_id':s['episode_id']} for s in specs])
            files[path.name]={'sha256':league.sha(path),'episodes':len(specs)}
        total=sum(map(len,schedule.values()))
        audit={'complete':True,'all_completed_episodes_verified':True,'missing_episode_ids':[],
               'manifest_sha256':league.sha(parent/'manifest.json'),'policy_arithmetic_sha256':arena_v2.ARITHMETIC_SHA,
               'rating_compatibility':arena_v2.COMPATIBILITY,'rollout_files':files,
               'scheduled_episodes':total,'episodes_verified':total,
               'checkpoint_verification':{name:league.checkpoint_record(parent/'checkpoints'/name,name,MODEL)[0]
                                          for name in ('base','cycle3')}}
        league.atomic_json(parent/'analysis.json',audit)
        league.atomic_json(parent/'summary.json',{'complete':True,'policy_arithmetic_sha256':arena_v2.ARITHMETIC_SHA})
        return audit

    def test_fixed_final_pairing_and_unseen_scenarios(self):
        schedule = holdout.make_schedule(['market','maze','chess'])
        rows = [r for group in schedule.values() for r in group]
        self.assertEqual(len(rows),288)
        self.assertEqual(len({r['episode_id'] for r in rows}),288)
        parent,_,_ = arena_v2.make_schedule(['market','maze','chess'])
        self.assertFalse(holdout.scenario_families(schedule)&holdout.scenario_families(parent))
        for specs in schedule.values():
            for i in range(0,len(specs),4):
                group = specs[i:i+4]
                self.assertEqual(len({r['seed'] for r in group}),1)
                self.assertEqual({(r['candidate'],r['candidate_seat']) for r in group},
                                 {('base',0),('base',1),('cycle3',0),('cycle3',1)})
        self.assertTrue(all(r['regime']=='eval' for r in rows))

    def fixture(self,root):
        out = Path(root)/'holdout';out.mkdir();(out/'rollout-parts').mkdir()
        inherited = {}
        for name in ('base','cycle3'):
            path = Path(root)/'parent/checkpoints'/name;write_checkpoint(path,name)
            inherited[name] = league.checkpoint_record(path,name,MODEL)[0]
        manifest = {'model':MODEL,'deadline':'2030-01-01T08:00:00-08:00','continuation_cycles':[],
                    'inherited_checkpoints':inherited,'inference_batch_size':8,'rollout_chunk_episodes':4}
        schedule = holdout.make_schedule(['fake'])
        schedule = {key:value[:8] for key,value in schedule.items()}
        tracker = {'constructors':0,'train_calls':0,'training_ids':[],'played':[]}
        league.atomic_json(out/'status.json',{'status':'prepared','stages':{}})
        return out,manifest,schedule,tracker

    def runner(self,fixture,interrupt=False,expired=False):
        out,manifest,schedule,tracker = fixture
        def play(policy,specs,path,batch_size):
            selected = specs[:1] if interrupt else specs
            tracker['played'].extend(s['episode_id'] for s in selected)
            with patch.object(experiment,'make_env',FakeGame),redirect_stdout(io.StringIO()):
                result = experiment.play(policy,selected,path,batch_size)
            if interrupt:raise RuntimeError('interrupted after one verified episode')
            return result
        return holdout.Holdout(out,manifest,schedule,SimpleNamespace(make_env=FakeGame),
            SimpleNamespace(action_prompt=action_prompt),SimpleNamespace(play=play),
            policy_factory=lambda:FakePolicy(tracker),checkpoint_loader=fake_loader,
            now=lambda:datetime.fromisoformat('2031-01-01T00:00:00-08:00' if expired else '2026-10-22T00:00:00-07:00'))

    def test_resume_keeps_completed_games_and_never_trains(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = self.fixture(root)
            with self.assertRaises(RuntimeError):self.runner(fixture,interrupt=True).run()
            self.assertEqual(self.runner(fixture).run()['status'],'complete')
            tracker = fixture[3]
            self.assertEqual(tracker['train_calls'],0)
            self.assertTrue(all(n==1 for n in Counter(tracker['played']).values()))
            count = tracker['constructors'];self.runner(fixture).run()
            self.assertEqual(count,tracker['constructors'])
            self.assertTrue(json.loads((fixture[0]/'summary.json').read_text())['complete'])
            path=fixture[0]/(next(iter(fixture[2]))+'.jsonl')
            rows=league.complete_jsonl_rows(path);rows[0]['transitions'][0]['entropy']+=.01
            league.atomic_rows(path,rows)
            with self.assertRaisesRegex(ValueError,'rollout bytes changed before resume'):
                self.runner(fixture).run()
            self.assertEqual(count,tracker['constructors'])

    def test_expiry_does_not_allocate_model(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = self.fixture(root)
            self.assertEqual(self.runner(fixture,expired=True).run()['status'],'deadline_reached')
            self.assertEqual(fixture[3]['constructors'],0)
            self.assertEqual(fixture[3]['played'],[])

    def test_preregistration_precedes_final_and_prepare_resume_bind_all_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            parent,manifest,schedule=self.parent_fixture(root);out=Path(root)/'holdout'
            registration=holdout.preregister(parent,out)
            self.assertFalse(out.exists())
            self.assertEqual(registration['scheduled_episodes'],96)
            self.assertEqual(holdout.read_registration(holdout.registration_path(out)),registration)
            self.complete_parent(parent,manifest,schedule)
            saved,scheduled=holdout.prepare(parent,out)
            self.assertEqual(saved['preregistration_sha256'],league.sha(holdout.registration_path(out)))
            self.assertEqual(saved['rollout_chunk_episodes'],4)
            self.assertEqual(holdout.prepare(parent,out),(saved,scheduled))
            for key in holdout.ANALYSIS_KEYS:
                self.assertEqual(league.sha(out/'source'/Path(key).name),saved['source_sha256'][key])
            path=parent/(next(iter(schedule))+'.jsonl');path.write_text(path.read_text()+'{}\n')
            with self.assertRaisesRegex(ValueError,'rollout bytes changed'):
                holdout.prepare(parent,out)

    def test_late_registration_and_preregistered_source_drift_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            parent,manifest,schedule=self.parent_fixture(root);out=Path(root)/'holdout'
            holdout.preregister(parent,out);self.complete_parent(parent,manifest,schedule)
            with self.assertRaisesRegex(ValueError,'final checkpoint directory exists'):
                holdout.preregister(parent,Path(root)/'late')
            changed=Path(root)/'changed.py';changed.write_text('# modified runner\n')
            sources={**holdout.live_sources(),holdout.SOURCE_KEY:changed}
            with patch.object(holdout,'live_sources',return_value=sources):
                with self.assertRaisesRegex(ValueError,'Preregistered live source changed'):
                    holdout.prepare(parent,out)

    def test_parent_checkpoint_metadata_is_bound_to_audit_before_prepare(self):
        with tempfile.TemporaryDirectory() as root:
            parent,manifest,schedule=self.parent_fixture(root);out=Path(root)/'holdout'
            holdout.preregister(parent,out);self.complete_parent(parent,manifest,schedule)
            path=parent/'checkpoints/cycle3/metadata.json';meta=json.loads(path.read_text())
            meta['label_token_ids']=[99,100];league.atomic_json(path,meta)
            with self.assertRaisesRegex(ValueError,'complete parent audit'):
                holdout.prepare(parent,out)

    def test_inherited_ancestor_scenarios_also_remain_held_out(self):
        with tempfile.TemporaryDirectory() as root:
            parent,manifest,schedule=self.parent_fixture(root)
            ancestor=Path(root)/'ancestor';ancestor.mkdir()
            league.atomic_json(ancestor/'manifest.json',{'model':MODEL})
            league.atomic_json(ancestor/'schedule.json',{'old-train':[{'environment':'fake','seed':3900000}]})
            manifest.update(parent_run=str(ancestor),parent_manifest_sha256=league.sha(ancestor/'manifest.json'))
            league.atomic_json(parent/'manifest.json',manifest)
            with self.assertRaisesRegex(ValueError,'overlap historical parent data'):
                holdout.preregister(parent,Path(root)/'holdout')

    def test_arithmetic_validation_rejects_wrong_final_pool_even_with_matching_hash_record(self):
        manifest={'policy_arithmetic':arena_v2.ARITHMETIC,'policy_arithmetic_sha256':arena_v2.ARITHMETIC_SHA}
        record={'name':'cycle3','adapter_sha256':'a','metadata_sha256':'b'}
        audit={'checkpoint_verification':{'cycle3':record}}
        meta={**manifest,'arithmetic_verification':{'integer_weights_exactly_unchanged':True,'all_floating_parameters_float32':True}}
        holdout.validate_checkpoint('cycle3',record,meta,manifest,audit)
        changed=deepcopy(meta);changed['policy_arithmetic']['activation_arithmetic']='bfloat16'
        with self.assertRaisesRegex(ValueError,'arithmetic differs'):
            holdout.validate_checkpoint('cycle3',record,changed,manifest,audit)

    def test_canonical_audit_checks_stage_hash_order_and_completed_coverage(self):
        with tempfile.TemporaryDirectory() as root:
            run=Path(root);stage='eval-fake-base';specs=[{'episode_id':'a'},{'episode_id':'b'}]
            path=run/(stage+'.jsonl');league.atomic_rows(path,specs)
            status={'stages':{stage:{'status':'complete','sha256':league.sha(path),'completed_episodes':2}}}
            self.assertEqual(canonical_rows(run,stage,specs,status,True)[0],specs)
            changed=[{**specs[0],'extra':'valid JSON mutation'},specs[1]];league.atomic_rows(path,changed)
            with self.assertRaisesRegex(ValueError,'rollout bytes changed'):
                canonical_rows(run,stage,specs,status,True)
            league.atomic_rows(path,list(reversed(specs)))
            with self.assertRaisesRegex(ValueError,'schedule order'):
                canonical_rows(run,stage,specs,{},False)
            path.unlink()
            with self.assertRaisesRegex(ValueError,'missing its canonical file'):
                canonical_rows(run,stage,specs,status,True)


if __name__=='__main__':unittest.main()
