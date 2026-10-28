"""Independent scalar protocol integration/mutations; no pretrained models."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest

from selfplay.arena_mixed_scalar import envs, runner


def binding(version=0):
    return {'checkpoint_id': 'base' if version == 0 else f'cycle{version}',
            'version': version, 'adapter_sha256': str(version % 10) * 64,
            'base_identity': {'fixture': True}}


def spec(name, seed, seats=None, opponent='self'):
    fingerprint = envs.scenario_fingerprint(name, seed)
    return {'episode_id': f'fixture-{name}-{seed}', 'environment': name, 'seed': seed,
            'regime': 'train', 'split': 'train', 'scenario_sha256': fingerprint,
            'scenario_family_id': name + ':' + fingerprint,
            'learner_seats': envs.protocol_config()['roles'][name] if seats is None else seats,
            'opponent': opponent}


def decide(requests, identity):
    # Deterministic fixture: choose a visible action, with no strategy access.
    return [{**deepcopy(row), 'checkpoint': deepcopy(identity),
             'action': deepcopy(row['actions'][0]), 'fixture_draw': i}
            for i, row in enumerate(requests)]


def read(path):
    return json.loads(Path(path).read_text())


def replace_json(path, value):
    # Deliberately mutate fixture evidence to test independent validation.
    Path(path).write_text(runner.canonical(value) + '\n')


def reseal(folder):
    seal = read(folder/'seal.json')
    for relative in seal['draw_files']:
        seal['draw_files'][relative] = runner.file_sha(folder/relative)
    seal['episodes_sha256'] = runner.file_sha(folder/'episodes.json')
    seal['replay_sha256'] = runner.file_sha(folder/'replay-verification.json')
    replace_json(folder/'seal.json', seal)


class IndependentRolloutReview(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.journal = runner.Journal(self.folder, 'a'*64, time.time()+60)

    def collect(self, specs=None, **kwargs):
        runner.collect_cohort(decide, specs or [spec('market', 17)], binding(),
                              self.journal, 'cohort', **kwargs)
        return self.folder/'cohort'

    def test_all_five_real_replays_have_exact_raw_action_coverage(self):
        folder = self.collect([spec(name, 19) for name in envs.ENVIRONMENTS])
        seal, episodes = runner.load_sealed_cohort(folder)
        self.assertEqual(seal['episodes'], 5)
        for episode in episodes:
            self.assertTrue(envs.verify_episode(episode)['verified'])
            expected = sum(len(event['active_players']) for event in episode['events'])
            self.assertEqual(len(episode['transitions']), expected)
            for row in episode['transitions']:
                self.assertEqual(row['scenario_family_id'], episode['scenario_family_id'])
                self.assertNotIn('initial_state', row)
                self.assertNotIn('audit_only', row['observation'])
        solo = [e for e in episodes if e['role_count'] == 1]
        self.assertTrue(all(e['learner_seats'] == [0] and e['returns'][1] == 0 for e in solo))

    def test_two_episode_leave_one_out_and_real_role_weights(self):
        from selfplay.arena_mixed_scalar.cuda_policy import cohort_rows
        weights={'market':.24,'maze':.24,'chess':.24,'planning':.24,'coding':.04}
        folder=self.collect([spec(name,seed) for name in envs.ENVIRONMENTS for seed in (19,21)])
        _,episodes=runner.load_sealed_cohort(folder)
        rows=cohort_rows(episodes,weights)
        identities={e['episode_id']:e for e in episodes}
        for row in rows:
            episode=identities[row['episode_id']]
            other=next(e for e in episodes if e['environment']==episode['environment']
                       and e['episode_id']!=episode['episode_id'])
            self.assertEqual(row['advantage'],episode['returns'][row['player']]-other['returns'][row['player']])
            self.assertEqual(row['weight'],weights[episode['environment']]/2/len(episode['learner_seats']))
        self.assertTrue(all(r['player']==0 for r in rows if r['environment'] in ('planning','coding')))
        aliased=deepcopy(episodes)
        aliased[1]['scenario_family_id']=aliased[0]['scenario_family_id']
        with self.assertRaisesRegex(ValueError,'unique'):
            cohort_rows(aliased,weights)

    def test_existing_intent_prevents_another_sampler_invocation(self):
        calls = []
        def counted(*args):
            calls.append(True); return decide(*args)
        specs = [spec('planning', 17)]
        runner.collect_cohort(counted, specs, binding(), self.journal, 'cohort')
        before = len(calls)
        with self.assertRaises(FileExistsError):
            runner.collect_cohort(counted, specs, binding(), self.journal, 'cohort')
        self.assertEqual(len(calls), before)

    def test_resealed_episode_cannot_change_original_sample_metadata(self):
        folder = self.collect()
        episodes = read(folder/'episodes.json')
        episodes[0]['transitions'][0]['fixture_draw'] = 999
        replace_json(folder/'episodes.json', episodes); reseal(folder)
        with self.assertRaisesRegex(ValueError, 'original|draw|request'):
            runner.load_sealed_cohort(folder)

    def test_resealed_episode_cannot_reclassify_a_real_learner_seat(self):
        folder = self.collect()
        episodes = read(folder/'episodes.json'); episode = episodes[0]
        episode['learner_seats'] = [0]; episode['schedule']['learner_seats'] = [0]
        episode['schedule']['opponent'] = 'heuristic'
        episode['opponent_transitions'] = [r for r in episode['transitions'] if r['player'] == 1]
        episode['transitions'] = [r for r in episode['transitions'] if r['player'] == 0]
        replace_json(folder/'episodes.json', episodes); reseal(folder)
        with self.assertRaises(ValueError):
            runner.load_sealed_cohort(folder)

    def test_base_opponent_missing_decisions_fail_even_after_resealing(self):
        folder = self.collect([spec('market', 17, [0], 'base')],
                              opponent_decide=decide, opponent_binding=binding())
        runner.load_sealed_cohort(folder)
        episodes = read(folder/'episodes.json')
        episodes[0]['opponent_transitions'].pop()
        replace_json(folder/'episodes.json', episodes); reseal(folder)
        with self.assertRaises(ValueError):
            runner.load_sealed_cohort(folder)

    def test_wrong_model_response_is_retained_before_validation_failure(self):
        def wrong(requests, identity):
            rows = decide(requests, identity)
            rows[0]['observation']['undeclared_private_value'] = 123
            return rows
        with self.assertRaisesRegex(ValueError, 'provenance'):
            runner.collect_cohort(wrong, [spec('planning', 3)], binding(), self.journal, 'cohort')
        self.assertTrue((self.folder/'cohort/batches/000000.json').is_file())
        self.assertFalse((self.folder/'cohort/seal.json').exists())

    def test_short_batch_response_is_retained_without_retry(self):
        calls = []
        def short(requests, identity):
            calls.append(True)
            return []
        with self.assertRaises(ValueError):
            runner.collect_cohort(short, [spec('planning', 3)], binding(), self.journal, 'cohort')
        self.assertEqual(calls, [True])
        self.assertTrue((self.folder/'cohort/batches/000000.json').is_file())
        self.assertEqual(read(self.folder/'cohort/batches/000000.json')['rows'], [])
        self.assertFalse((self.folder/'cohort/seal.json').exists())

    def test_pipeline_submits_next_actor_before_prior_update_with_actual_lag_one(self):
        events = []
        class Future:
            def __init__(self, number, checkpoint): self.number, self.checkpoint = number, checkpoint
            def result(self):
                events.append(('resolved', self.number))
                return {'binding': self.checkpoint['binding']}
        class Actor:
            def submit(self, specs, checkpoint, prefix):
                number = int(prefix.rsplit('/', 1)[1])
                events.append(('submitted', number, checkpoint['binding']['version']))
                return Future(number, checkpoint)
        class Learner:
            def update(self, cohort, checkpoint, index):
                events.append(('updated', index, checkpoint['binding']['version']))
                return {'checkpoint': {'binding': binding(index)},
                        'report': {'optimizer_steps': int(index != 2)}}
        result = runner.run_pipeline(Actor(), Learner(), self.journal,
                    {f'train-{i}': [] for i in range(1, 5)}, 4, {'binding': binding()})
        self.assertEqual([e for e in events if e[0] == 'submitted'],
                         [('submitted', 1, 0), ('submitted', 2, 0),
                          ('submitted', 3, 1), ('submitted', 4, 2)])
        for i in range(1, 4):
            self.assertLess(events.index(('submitted', i+1, i-1)), events.index(('updated', i, i-1)))
        self.assertEqual(result['actual_optimizer_steps'], 3)
        self.assertEqual(result['endpoint']['binding']['version'], 4)



from scripts.arena_mixed_scalar_action_audit import get_tokenizer, verify_action_rows
from selfplay.arena_mixed_scalar import cuda_policy, envs, runner
import torch

ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT/'runs/arena-mixed-preflight-v1'
WEIGHTS = {'market':.24,'maze':.24,'chess':.24,'planning':.24,'coding':.04}


class ScalarNativeActionAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory(); run=Path(cls.folder.name)
        (run/'inputs').mkdir()
        identities=json.loads((PRE/'model-file-sha256.json').read_text())
        cls.config=cuda_policy.default_config(WEIGHTS,model_index_sha256=identities['model.safetensors.index.json'],
            tokenizer_files_sha256=cuda_policy.digest({k:identities[k] for k in ('tokenizer.json','tokenizer_config.json')}))
        (run/'manifest.json').write_text(json.dumps({'config':{'policy':cls.config}}))
        (run/'inputs/model-file-sha256.json').write_text(json.dumps(identities))
        cls.tokenizer=get_tokenizer(run,PRE/'tokenizer')
        cls.labels,cls.label_ids=cuda_policy.select_labels(cls.tokenizer)
        cls.base={'checkpoint_id':'base','version':0,'adapter_sha256':'a'*64,
            'base_identity':{**{k:cls.config[k] for k in ('model','revision','model_index_sha256','tokenizer_files_sha256')},
                             'quantization':'nf4-double-fp32','frozen_state_sha256':'b'*64}}
        cls.journal=runner.Journal(run,'c'*64,time.time()+120)
        cls.requests=[spec('planning',955),spec('coding',955),spec('market',955,[0],'base')]
        runner.collect_cohort(cls.choose,cls.requests,cls.base,cls.journal,'fixtures',
                              opponent_decide=cls.choose,opponent_binding=cls.base)
        cls.episodes=runner.load_sealed_cohort(run/'fixtures')[1]

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    @classmethod
    def choose(cls, requests, binding):
        result=[]
        for request in requests:
            row=deepcopy(request);actions=row['actions']
            prompt=cuda_policy.action_prompt(row['observation'],actions,cls.labels)
            rendered=cls.tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,
                          add_generation_prompt=True,enable_thinking=False)
            ids=cls.tokenizer.encode(rendered,add_special_tokens=False)
            vector=torch.zeros(len(actions),dtype=torch.float32).log_softmax(0)
            probs=torch.softmax(vector.double(),0)
            pick=int(torch.multinomial(probs,1,generator=torch.Generator().manual_seed(row['sample_seed'])))
            row.update(prompt=prompt,rendered_prompt=rendered,input_ids=ids,
                input_ids_sha256=cuda_policy.digest(ids),prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                menu_sha256=cuda_policy.digest(actions),labels_sha256=cuda_policy.digest([cls.labels,cls.label_ids]),
                physical_execution_contract_sha256=cuda_policy.digest(cls.config['physical_execution_contract']),
                actor_config_sha256=cuda_policy.digest(cls.config),base_identity=deepcopy(binding['base_identity']),
                action_index=pick,action=deepcopy(actions[pick]),action_label=cls.labels[pick],
                action_token_id=cls.label_ids[pick],old_log_prob=float(vector[pick]),old_log_probs=vector.tolist(),
                entropy=float(-(probs*probs.log()).sum()),checkpoint=deepcopy(binding),
                policy_version=binding['version'],actor_snapshot_sha256=binding['adapter_sha256'],thinking=False)
            result.append(row)
        return result

    def verify(self, episodes):
        return verify_action_rows(episodes,self.config,self.tokenizer,opponent_binding=self.base)

    def test_original_native_prompts_all_roles_and_categorical_draws_verify(self):
        report=self.verify(self.episodes)
        self.assertTrue(report['verified'])
        self.assertEqual(report['actions'],sum(len(e['transitions'])+len(e['opponent_transitions']) for e in self.episodes))
        self.assertEqual(report['roles']['opponent'],16)
        self.assertLess(report['maximum_prompt_tokens'],4096)

    def test_prompt_token_label_and_native_mode_mutations_fail(self):
        for key,value in (('prompt','secret answer'),('rendered_prompt','<think>open'),
            ('input_ids',[1,2]),('labels_sha256','0'*64),('action_token_id',0),
            ('thinking',True),('actor_config_sha256','0'*64),('physical_execution_contract_sha256','0'*64)):
            data=deepcopy(self.episodes);data[0]['transitions'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.verify(data)

    def test_invalid_vector_selected_logprob_entropy_and_seed_fail(self):
        first=self.episodes[0]['transitions'][0]
        for key,value in (('old_log_probs',[0.]*len(first['actions'])),('old_log_prob',42.),
            ('entropy',42.),('sample_seed',first['sample_seed']+1),('menu_seed',0)):
            data=deepcopy(self.episodes);data[0]['transitions'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.verify(data)

    def test_wrong_checkpoint_and_missing_real_opponent_coverage_fail(self):
        data=deepcopy(self.episodes);data[-1]['opponent_transitions'][0]['checkpoint']['adapter_sha256']='f'*64
        with self.assertRaises(ValueError):self.verify(data)
        data=deepcopy(self.episodes);data[-1]['opponent_transitions'].pop()
        with self.assertRaises(ValueError):self.verify(data)
        with self.assertRaisesRegex(ValueError,'Explicit'):
            verify_action_rows(self.episodes,self.config,self.tokenizer)

    def test_original_tokenizer_hash_tampering_rejected_before_loading(self):
        run=Path(self.folder.name)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            for name in ('config.json','tokenizer.json','tokenizer_config.json'):
                (folder/name).write_bytes((PRE/'tokenizer'/name).read_bytes())
            with (folder/'tokenizer_config.json').open('ab') as f:f.write(b' ')
            with self.assertRaisesRegex(ValueError,'bytes'):
                get_tokenizer(run,folder)


class ScalarScheduleReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from selfplay.arena_mixed_scalar.experiment import make_schedules
        cls.inventory = read(ROOT/'runs/arena-mixed-scalar-preflight-v1/prior-scenarios.json')
        cls.schedules, cls.selections, cls.report = make_schedules(cls.inventory['fingerprints'])

    def test_all_previous_prepared_tasks_excluded_and_exact_fixed_counts(self):
        self.assertTrue(self.inventory['verified'])
        self.assertEqual(self.report['distinct_scenarios'],300)
        self.assertEqual(self.report['episodes'],668)
        self.assertEqual(self.report['episodes_by_split'],{'calibration':20,'train':200,'eval':448})
        excluded=set(self.inventory['fingerprints'])
        for name, selection in self.selections.items():
            selected=selection['selected']
            self.assertEqual(len(selected),60)
            self.assertFalse(excluded.intersection(s['scenario_sha256'] for s in selected))
        for stage, specs in self.schedules.items():
            if stage.startswith(('train-','calibration-')):
                self.assertEqual(len(specs),10)
                for name in envs.ENVIRONMENTS:
                    rows=[s for s in specs if s['environment']==name]
                    self.assertEqual(len(rows),2)
                    self.assertEqual(len({r['scenario_family_id'] for r in rows}),2)
                    self.assertTrue(all(r['learner_seats']==envs.protocol_config()['roles'][name] for r in rows))

    def test_heldout_pairing_keeps_real_roles_and_full_scenario_denominator(self):
        baseline,final=self.schedules['eval-base'],self.schedules['eval-final']
        self.assertEqual(len(baseline),224);self.assertEqual(len(final),224)
        for a,b in zip(baseline,final,strict=True):
            self.assertEqual({k:v for k,v in a.items() if k!='episode_id'},
                             {k:v for k,v in b.items() if k!='episode_id'})
        for name in envs.ENVIRONMENTS:
            rows=[s for s in baseline if s['environment']==name]
            families={r['scenario_family_id'] for r in rows}
            self.assertEqual(len(families),16)
            for family in families:
                actual={(tuple(s['learner_seats']),s['opponent']) for s in rows if s['scenario_family_id']==family}
                expected={((0,),'none')} if name in ('planning','coding') else {
                    ((seat,),opponent) for seat in (0,1) for opponent in ('base','heuristic')}
                self.assertEqual(actual,expected)

    def test_cross_split_alias_and_historical_collision_are_rejected(self):
        data=deepcopy(self.schedules)
        original=data['train-1'][0]
        leaked={**deepcopy(original),'episode_id':'invented-eval-id','split':'eval',
                'scenario_family_id':'different-family-name'}
        data['eval-base'][0]=leaked
        with self.assertRaisesRegex(ValueError,'aliases'):
            envs.validate_schedule(data,prior_fingerprints=self.inventory['fingerprints'])
        with self.assertRaisesRegex(ValueError,'historical'):
            envs.validate_schedule(self.schedules,prior_fingerprints=[original['scenario_sha256']])


if __name__=='__main__':unittest.main()


