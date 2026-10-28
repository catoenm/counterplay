"""CPU-only schedule, public-boundary and immutable rollout tests."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest

from selfplay.environment import Task
from selfplay.arena_mixed.planning import PlanningEnv as HistoricalPlanningEnv, task_family_payload
from selfplay.planning_curriculum import envs, runner, schedule


def fake_rows(requests, binding):
    return [{**deepcopy(request), 'checkpoint':deepcopy(binding), 'action':deepcopy(request['actions'][0]),
             'action_index':0, 'old_log_prob':-1.3862943611198906,
             'old_log_probs':[-1.3862943611198906]*4, 'entropy':1.3862943611198906}
            for request in requests]


def profiled_rows(requests, binding):
    rows=fake_rows(requests,binding)
    return {'rows':rows,'timing':{'kind':'planning_screen_sampling_timing_v1',
        'elapsed_seconds':float(len(rows)), 'nonforward_seconds':0.,
        'row_sha256':[runner.digest(row) for row in rows],
        'forward_timings':[{'row_sha256':runner.digest(row),'input_tokens':100,
            'training':False,'grad_enabled':False,'seconds':1.} for row in rows]}}


class PlanningCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schedules,cls.selection,cls.verification=schedule.make_schedules([])

    def test_exact_depth_goal_and_public_schema(self):
        for depth in range(1,7):
            env=envs.make_env('planning',1234+depth,depth=depth)
            self.assertEqual(len(env._task.shortest_plan()),depth)
            self.assertTrue(env.assert_private_boundary())
            view=envs.policy_view(env,0)
            self.assertFalse({'depth','witness','audit_only','initial_shortest_depth'} & set(view['observation']))
            header=envs.episode_header(env,f'depth-{depth}')
            for tool in env._task.shortest_plan():
                header['events'].append(envs.record_step(env,{0:{'tool':tool}}))
            episode=envs.finish_episode(header,env)
            self.assertEqual(episode['returns'],[1.,0.])
            self.assertEqual(envs.verify_episode(json.loads(json.dumps(episode)))['turns'],depth)
        for invalid in (0,7,True,1.5):
            with self.assertRaises(ValueError): envs.make_env('planning',1,depth=invalid)

    def test_unchanged_historical_depths_and_canonical_symmetry(self):
        for seed in range(12):
            old=HistoricalPlanningEnv(seed)
            new=envs.make_env('planning',seed,depth=(3,4,5,6)[seed%4])
            self.assertEqual(old.observation(0),new.observation(0))
            self.assertEqual(old.snapshot(),new.snapshot())
            self.assertEqual(envs.scenario_fingerprint(initial_state=old.snapshot()),envs.scenario_fingerprint(seed=seed,depth=(3,4,5,6)[seed%4]))
        task=envs.PlanningEnv(100,depth=2)._task
        dual_names={'inc_x':'inc_y','inc_y':'inc_x','sum_x':'sum_y','sum_y':'sum_x',
                    'double_x':'double_y','double_y':'double_x','swap':'swap'}
        dual=Task(task.modulus,tuple(reversed(task.start)),tuple(reversed(task.goal)),
                  tuple(dual_names[x] for x in task.actions),12,tuple(dual_names[x] for x in task.witness))
        self.assertEqual(task_family_payload(task),task_family_payload(dual))

    def test_complete_balanced_schedule_and_declared_pairing(self):
        self.assertEqual(self.verification['episodes'],544)
        self.assertEqual(self.verification['episodes_by_split'],{'screen':64,'train':160,'eval':320})
        self.assertEqual(self.verification['distinct_scenarios'],248)
        self.assertEqual(self.selection['timing_probe_episode_ids'],['screen-hard-depth6-00','screen-hard-depth6-01'])
        for seed in schedule.TRAINING_SEEDS:
            stages=schedule.training_stages(self.schedules,seed,'curriculum')
            self.assertEqual(set(stages),{f'train-{i}' for i in range(1,21)})
            for i in range(1,21):
                a=self.schedules[f'train-{seed}-control-{i}']; b=stages[f'train-{i}']
                self.assertEqual([x['paired_seed_key'] for x in a],[x['paired_seed_key'] for x in b])
                self.assertEqual([x['depth'] for x in b], [1,2] if i<=10 else ([3,4] if i%2 else [5,6]))
                self.assertEqual([x['scenario_sha256'] for x in a]==[x['scenario_sha256'] for x in b],i>10)
        self.assertEqual(schedule.validate_schedules(json.loads(json.dumps(self.schedules))),self.verification)

    def test_historical_exclusion_changes_selection_without_outcomes(self):
        excluded=self.schedules['screen-hard'][0]['scenario_sha256']
        replacement,receipt,verification=schedule.make_schedules([excluded])
        self.assertNotIn(excluded,verification['scenario_sha256'])
        self.assertEqual(receipt['rejected'][0]['scenario_sha256'],excluded)
        self.assertNotEqual(replacement['screen-hard'][0]['seed'],self.schedules['screen-hard'][0]['seed'])

    def test_schedule_mutations_rejected(self):
        bad=deepcopy(self.schedules);bad['screen-easy'][1]['pair_id']='easy-depth1-pair01'
        with self.assertRaises(ValueError): schedule.validate_schedules(bad)
        bad=deepcopy(self.schedules);bad['eval-base'][0]['depth']=2
        with self.assertRaises(ValueError): schedule.validate_schedules(bad)
        bad=deepcopy(self.schedules)
        donor=bad['train-742101-control-1'][0];recipient=bad['train-742102-control-1'][0]
        for key in ('seed','scenario_sha256','scenario_family_id'):recipient[key]=donor[key]
        with self.assertRaisesRegex(ValueError,'reuse'): schedule.validate_schedules(bad)
        bad=deepcopy(self.schedules);bad['train-742101-curriculum-11'][0]['paired_seed_key']='different'
        with self.assertRaises(ValueError): schedule.validate_schedules(bad)

    def test_historical_inventory_binds_prepared_schedule_and_legacy_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'runs'/'prepared'/'inputs';folder.mkdir(parents=True)
            old=HistoricalPlanningEnv(100)
            spec={'environment':'planning','seed':100,'regime':'train',
                  'scenario_sha256':envs.scenario_fingerprint(initial_state=old.snapshot())}
            raw=json.dumps({'calibration':[spec],'eval':[spec]}).encode()
            (folder/'schedule.json').write_bytes(raw)
            (folder.parent/'manifest.json').write_text(json.dumps({'inputs':{'inputs/schedule.json':hashlib.sha256(raw).hexdigest()}}))
            (root/'runs'/'tasks.jsonl').write_text(json.dumps(asdict(envs.PlanningEnv(101,depth=1)._task))+'\n')
            inventory=schedule.historical_inventory(root)
            self.assertTrue(inventory['verified']);self.assertEqual(inventory['distinct_planning_families'],2)
            self.assertIn(spec['scenario_sha256'],inventory['fingerprints'])
            self.assertIn('runs/prepared/inputs/schedule.json',inventory['source_sha256'])
            (folder/'schedule.json').write_text(json.dumps({'calibration':[{**spec,'seed':101}]}))
            with self.assertRaisesRegex(ValueError,'manifest hash'): schedule.historical_inventory(root)

    def test_replay_raw_seal_timing_and_once_only(self):
        specs=self.schedules['screen-easy'][:2];binding={'version':0,'adapter_sha256':'a'*64}
        with tempfile.TemporaryDirectory() as tmp:
            journal=runner.Journal(tmp,'b'*64,time.time()+30)
            seal=runner.collect_cohort(profiled_rows,specs,binding,journal,'screen',batch_size=2,profiled=True)
            saved,episodes=runner.load_sealed_cohort(Path(tmp)/'screen')
            self.assertEqual(saved,seal);self.assertEqual(len(episodes),2)
            self.assertEqual([x['depth'] for x in episodes],[1,1])
            for episode in episodes:
                for row in episode['transitions']:
                    self.assertNotIn('depth',row);self.assertNotIn('audit_only',row['observation'])
            called=[]
            with self.assertRaises(FileExistsError):
                runner.collect_cohort(lambda *args:called.append(args),specs,binding,journal,'screen')
            self.assertEqual(called,[])
            path=Path(tmp)/'screen/batches/000000.json'
            value=json.loads(path.read_text());value['timing']['elapsed_seconds']+=1
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'changed after sealing'): runner.load_sealed_cohort(Path(tmp)/'screen')

    def test_bad_profile_response_is_retained_no_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal=runner.Journal(tmp,'b'*64,time.time()+30)
            with self.assertRaises(ValueError):
                runner.collect_cohort(lambda *args:{'rows':[],'unexpected':'retained'},self.schedules['screen-easy'][:2],
                                      {'version':0},journal,'screen',profiled=True)
            raw=json.loads((Path(tmp)/'screen/batches/000000.json').read_text())
            self.assertEqual(raw['malformed_profiled_response'],{'rows':[],'unexpected':'retained'})
            self.assertFalse((Path(tmp)/'screen/seal.json').exists())
            with self.assertRaises(FileExistsError):
                runner.collect_cohort(fake_rows,self.schedules['screen-easy'][:2],{'version':0},journal,'screen')

    def test_unprofiled_rows_and_menu_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal=runner.Journal(tmp,'b'*64,time.time()+30)
            runner.collect_cohort(fake_rows,self.schedules['screen-hard'][:2],{'version':0},journal,'cohort')
            _,episodes=runner.load_sealed_cohort(Path(tmp)/'cohort')
            episode=deepcopy(episodes[0]);event=episode['events'][0]
            event['legal_actions']['0' if '0' in event['legal_actions'] else 0].pop()
            with self.assertRaisesRegex(ValueError,'complete legal-action'):envs.verify_episode(episode)

    def test_expired_deadline_creates_no_draw_intent(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal=runner.Journal(tmp,'b'*64,time.time()-1)
            with self.assertRaises(TimeoutError):
                runner.collect_cohort(fake_rows,self.schedules['screen-easy'][:2],{'version':0},journal,'screen')
            self.assertFalse((Path(tmp)/'screen/intent.json').exists())


if __name__=='__main__':unittest.main()
