"""CPU-only calibration journal fixtures; no neural/model/network calls.

Fake serialized adapters and sampled tokens test orchestration only. Real
bounded practice games are exercised for two synthetic controllers; large
complete-run tests explicitly mock scoring and are not research evidence.
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from selfplay import program_debug_revision_calibration as calibration
from selfplay import program_debug_revision_harness as harness
from selfplay import program_cardinal_gpu as gpu
from tests.test_program_cardinal_gpu import FakePolicy, deployment, fake_identity

IDLE = "def act():\n    return 'rest'\n"


class FakeCalibrationPolicy(FakePolicy):
    def __init__(self, bad_drafts=False, malformed_batch=False):
        super().__init__()
        self.bad_drafts, self.malformed_batch = bad_drafts, malformed_batch
        self.calls = []; self.fail_sample = False; self.mutate_on_sample = False
        self.gate_calls = 0; self.fail_gate = False; self.truncate_gate = False

    def sample_messages_batch(self, contexts, seeds, authoring_stage='draft'):
        self.calls.append(deepcopy((contexts, seeds)))
        if self.fail_sample:
            self.fail_sample = False
            raise RuntimeError('Fixture interrupted sample, outcome unknown')
        rows=[]
        for i,(context,seed) in enumerate(zip(contexts,seeds)):
            is_draft=len(context)==1
            code=IDLE if is_draft or i else 'invalid mandatory final'
            text=('fixture reasoning\n</think>\n' if is_draft else '')+code
            if is_draft and self.bad_drafts:text='unfinished raw draft thoughts'
            prefix=[1,2] if is_draft else [1,2,3]
            rows.append({'text':text,'raw_text':text+'<eos>','messages':deepcopy(context),
                'prompt_template':'fixture only','prompt_ids':prefix,'completion_ids':[41+i,99],
                'sampled_token_logprobs':[-.2,-.4],'seed':seed,'finish_reason':'eos',
                'input_tokens':len(prefix),'output_tokens':2,'elapsed_seconds':.1,
                'eos_token_ids':[99],'sampling_batch_size':2,'sampling_batch_row':i,
                **calibration.stage_metadata(authoring_stage),'completion_token_limit':4096,
                'calibration_only':True,'fixture_only':True})
        if self.malformed_batch:rows[0]['seed']+=1000
        if self.mutate_on_sample:self.weight+=1.
        return {'records':rows,'timing':{'batch_elapsed_seconds':.1,'total_output_tokens':4}}

    def full_completion_likelihood_gate(self, records):
        self.gate_calls += 1
        rows=[]
        for i,record in enumerate(records):
            vector=record['sampled_token_logprobs'][:]
            check=calibration.common.check_likelihoods(vector,vector,.002,.02)
            rows.append({'index':i,'record_sha256':gpu.object_hash(record),
                'authoring_stage':record['authoring_stage'],'seed':record['seed'],
                'completion_tokens':len(vector),'sampled_token_logprobs':vector,
                'eval_token_logprobs':vector,'train_token_logprobs':vector,
                'eval_check':check,'train_check':check,'eval_all_finite':True,'train_all_finite':True,
                'eval_grad_enabled':False,'train_grad_enabled':True,'passed':True})
        if self.truncate_gate:rows=rows[:-1]
        identity={'fixture_only':True,'weight':self.weight,'step':self.updates}
        return {'kind':'revision_full_completion_likelihood_admission','passed':not self.fail_gate,
            'optimizer_steps':0,'training_use':False,'expected_records':16,
            'token_tolerance':.002,'sequence_tolerance':.02,'records':rows,
            'sampled_tokens':sum(len(r['completion_ids']) for r in records),
            'state_before':identity,'state_after':identity,
            **{key:True for key in ('adapter_unchanged','optimizer_unchanged','rng_restored',
                'gradients_restored','config_restored','model_mode_restored','frozen_state_unchanged')}}

    def update(self,*args,**kwargs):
        raise AssertionError('Calibration must not call an optimizer update')

    update_trajectories=update


@contextmanager
def fixture(bad_drafts=False):
    with tempfile.TemporaryDirectory() as directory, patch.object(calibration,'SOURCES',[]), \
         patch.object(gpu,'checkpoint_state',fake_identity):
        run=Path(directory);dep=deployment()
        dep.update(maximum_hours=3,reservation_usd=25,estimate_usd=20,maximum_gpu_rate_usd_hour=5.29,
            deadline_unix=(datetime.now(timezone.utc)+timedelta(hours=3)).timestamp())
        (run/'inputs').mkdir();calibration.common.atomic_json(run/'inputs/design-plan.json',{'fixture_only':True})
        cfg=calibration.config(dep)
        for name,value in [('deployment.json',dep),
            ('plan.json',calibration.plan(dep,design_path=run/'inputs/design-plan.json')),
            ('schedule.json',calibration.schedule()),('map-fingerprints.json',{}),
            ('initial-opponents.json',calibration.initial_roster())]:
            calibration.common.atomic_json(run/name,value)
        (run/'prompts').mkdir();(run/'prompts/debug.txt').write_text(harness.debug_model_spec())
        policy=FakeCalibrationPolicy(bad_drafts);files=policy.save_checkpoint(run/'inputs/initial-checkpoint')
        names=['deployment.json','plan.json','schedule.json','map-fingerprints.json',
               'initial-opponents.json','prompts/debug.txt','inputs/design-plan.json']
        inputs={n:calibration.common.digest(run/n) for n in names}
        manifest={'kind':cfg['kind'],'config':cfg,'source_sha256':{},'input_sha256':inputs,
            'initial_checkpoint_files':files,'prompt_sha256':{'debug':inputs['prompts/debug.txt']}}
        calibration.common.atomic_json(run/'manifest.json',manifest)
        budget,_=calibration.journal.execution_budget(cfg)
        calibration.common.atomic_json(run/'status.json',{'manifest_sha256':calibration.common.digest(run/'manifest.json'),
            'artifacts':{},'state':'prepared','stage':'prepared','claims_allowed':False,'execution_budget':budget})
        engine=calibration.Calibration(run,policy);engine.initialize()
        yield run,engine,policy


def trajectories(engine):
    return engine.trajectories('fixture',[f'fixture-{i}' for i in range(2)],
        [201000000,201000002],'base',[engine.schedule['practice_maps'][str(i)] for i in range(2)])


def fixture_score(record,*unused):
    classification,_=harness.classify_revision(record['text'],record['finish_reason'])
    return {'authorship_valid':classification['valid'],'runtime_valid':classification['valid'],
        'classification':classification,'debug_classification':classification,'matches':[],
        'candidate_program_faults':0,'opponent_program_faults':0,'positive_banking_games':0,
        'banked_point_yield':0.,'game_return':0. if classification['valid'] else None,
        'reward':0. if classification['valid'] else -1.,'fixture_only':True}


class CalibrationRunnerTests(unittest.TestCase):
    def test_exactly_two_calls_real_practice_and_resume_without_resampling(self):
        with fixture() as (run,engine,policy):
            rows=trajectories(engine)
            self.assertEqual(len(policy.calls),2)
            for i,row in enumerate(rows):
                self.assertEqual(row['text'],row['turns'][1]['text'])
                self.assertEqual(row['turn_sha256'],[gpu.object_hash(t) for t in row['turns']])
                self.assertEqual(row['final_turn_index'],1)
                evidence=engine.existing(row['feedback_path'])
                self.assertEqual(len(evidence['matches']),8)
                self.assertEqual(evidence['record'],row['turns'][0])
                self.assertNotIn('fixture reasoning',row['turns'][1]['messages'][1]['content'])
            self.assertFalse(harness.classify_revision(rows[0]['text'],'eos')[0]['valid'])
            self.assertTrue(harness.classify_debug(rows[0]['turns'][0]['text'],'eos')[0]['valid'])
            self.assertEqual(trajectories(calibration.Calibration(run,policy)),rows)
            self.assertEqual(len(policy.calls),2)

    def test_crash_after_draft_or_revision_commit_never_adds_call(self):
        for failure_path,expected_calls in [('feedback/fixture-1.json',1),('generations/fixture-1.json',2)]:
            with self.subTest(path=failure_path),fixture(bad_drafts=True) as (run,engine,policy):
                commit=engine.commit
                def crash(relative,value):
                    if relative==failure_path:raise RuntimeError('Fixture crash after durable sample')
                    return commit(relative,value)
                with patch.object(engine,'commit',crash),self.assertRaisesRegex(RuntimeError,'durable'):
                    trajectories(engine)
                self.assertEqual(len(policy.calls),expected_calls)
                rows=trajectories(calibration.Calibration(run,policy))
                self.assertEqual(len(rows),2);self.assertEqual(len(policy.calls),2)
                self.assertEqual(rows[0]['turns'][1]['messages'][1]['content'],rows[0]['turns'][0]['text'])

    def test_unknown_intent_and_malformed_completed_batch_preserve_evidence_without_retry(self):
        with fixture(bad_drafts=True) as (run,engine,policy):
            policy.fail_sample=True
            with self.assertRaisesRegex(RuntimeError,'unknown'):trajectories(engine)
            with self.assertRaisesRegex(ValueError,'never redraw'):
                trajectories(calibration.Calibration(run,policy))
            self.assertEqual(len(policy.calls),1)
            self.assertIn('batches/fixture-draft-intent.json',engine.status['artifacts'])
        with fixture(bad_drafts=True) as (run,engine,policy):
            policy.malformed_batch=True
            with self.assertRaisesRegex(ValueError,'identity'):trajectories(engine)
            self.assertIn('batches/fixture-draft.json',engine.status['artifacts'])
            with self.assertRaisesRegex(ValueError,'identity'):
                trajectories(calibration.Calibration(run,policy))
            self.assertEqual(len(policy.calls),1)

    def test_fixed_eight_trajectories_sealed_before_any_scoring_and_zero_updates(self):
        with fixture(bad_drafts=True) as (run,engine,policy):
            calls=[]
            def scoring(record,*args):
                seal=engine.existing('calibration/sealed.json')
                self.assertEqual(seal['draws'],8);self.assertEqual(seal['sampled_turns'],16)
                self.assertEqual(len(seal['generations']),8)
                self.assertTrue(engine.existing('calibration/full-likelihood-gate.json')['passed'])
                self.assertEqual(policy.gate_calls,1)
                calls.append(record['name']);return fixture_score(record)
            with patch.object(calibration,'score',scoring):engine.run_all()
            self.assertEqual(calls,[f'calibration-{i}' for i in range(8)])
            self.assertEqual(len(policy.calls),8)
            self.assertEqual(engine.status['state'],'complete')
            self.assertFalse(engine.status['qualification_passed'])
            self.assertFalse(engine.status['claims_allowed'])
            self.assertFalse(engine.status['learning_comparison_available'])
            self.assertEqual(engine.status['actual_optimizer_steps'],0)
            self.assertEqual(policy.updates,0)
            self.assertEqual(engine.checkpoint('unchanged-final')['files'],engine.checkpoint('base')['files'])
            with patch.object(calibration,'score',side_effect=AssertionError('No rescore')):
                calibration.Calibration(run,policy).run_all()
            self.assertEqual(len(policy.calls),8)

    def test_full_gate_failure_or_missing_turn_stops_before_scores_without_retry(self):
        for bad_flag in ('fail_gate','truncate_gate'):
            with self.subTest(flag=bad_flag),fixture(bad_drafts=True) as (run,engine,policy):
                setattr(policy,bad_flag,True)
                with patch.object(calibration,'score',side_effect=AssertionError('No score before full gate')):
                    engine.run_all()
                    self.assertEqual(engine.status['state'],'numerical_failed')
                    self.assertIn('calibration/sealed.json',engine.status['artifacts'])
                    self.assertIn('calibration/full-likelihood-gate.json',engine.status['artifacts'])
                    self.assertNotIn('calibration/scores.json',engine.status['artifacts'])
                    calibration.Calibration(run,policy).run_all()
                self.assertEqual(policy.gate_calls,1)
                self.assertEqual(len(policy.calls),8)

    def test_full_gate_coverage_and_gradient_mode_checks_are_not_just_passed_flag(self):
        with fixture(bad_drafts=True) as (_,engine,policy):
            with patch.object(calibration,'score',fixture_score):engine.run_all()
            records=[turn for i in range(8) for turn in engine.existing(f'generations/calibration-{i}.json')['turns']]
            report=engine.existing('calibration/full-likelihood-gate.json')
            self.assertTrue(calibration.validate_full_gate(report,records,engine.config))
            for mutate in ('hash','train_mode','likelihood','state'):
                changed=deepcopy(report)
                if mutate=='hash':changed['records'][1]['record_sha256']='x'
                elif mutate=='train_mode':changed['records'][1]['train_grad_enabled']=False
                elif mutate=='likelihood':changed['records'][1]['train_token_logprobs'][0]-=.1
                else:changed['state_after']['step']=1
                with self.subTest(mutate=mutate),self.assertRaises(ValueError):
                    calibration.validate_full_gate(changed,records,engine.config)

    def test_interrupted_full_gate_is_not_replayed_on_resume(self):
        with fixture(bad_drafts=True) as (run,engine,policy):
            with patch.object(policy,'full_completion_likelihood_gate',side_effect=RuntimeError('Interrupted gate')):
                with self.assertRaisesRegex(RuntimeError,'Interrupted gate'):
                    engine.run_all()
            self.assertIn('calibration/full-likelihood-gate-intent.json',engine.status['artifacts'])
            self.assertNotIn('calibration/full-likelihood-gate.json',engine.status['artifacts'])
            with patch.object(policy,'full_completion_likelihood_gate',side_effect=AssertionError('Never retry')):
                resumed=calibration.Calibration(run,policy)
                resumed.run_all()
            self.assertEqual(resumed.status['state'],'numerical_failed')
            self.assertNotIn('calibration/scores.json',resumed.status['artifacts'])
            self.assertEqual(len(policy.calls),8)

    def test_native_revision_mode_must_be_false_and_effort_null(self):
        with fixture() as (_,engine,policy):
            contexts=[harness.initial_messages()]*2;seeds=[2,4]
            batch=policy.sample_messages_batch(contexts,seeds,authoring_stage='revision')
            calibration.validate_batch(batch,contexts,seeds,engine.config,'revision')
            for field,value in [('thinking_enabled',True),('reasoning_effort','low'),('reasoning_mode','native_low'),('authoring_stage','draft')]:
                changed=deepcopy(batch);changed['records'][0][field]=value
                with self.subTest(field=field),self.assertRaises(ValueError):
                    calibration.validate_batch(changed,contexts,seeds,engine.config,'revision')

    def test_accidental_weight_mutation_prevents_completion(self):
        with fixture(bad_drafts=True) as (run,engine,policy):
            policy.mutate_on_sample=True
            with patch.object(calibration,'score',fixture_score),self.assertRaisesRegex(ValueError,'Calibration changed'):
                engine.run_all()
            self.assertNotEqual(engine.status['state'],'complete')
            self.assertFalse(engine.status['claims_allowed'])

    def test_config_and_sampling_identity_are_fixed_and_deadline_does_not_reset(self):
        with fixture() as (_,engine,policy):
            cfg=engine.config
            self.assertEqual((cfg['max_new_tokens'],cfg['max_context_tokens'],cfg['sample_seconds']),(4096,16384,900))
            self.assertEqual((cfg['capability_samples'],cfg['batch_size'],cfg['authoring_calls']),(8,2,2))
            self.assertEqual(cfg['iterations'],0)
            self.assertEqual(cfg['reasoning_effort'],'low')
            contexts=[harness.initial_messages()]*2;seeds=[2,4]
            batch=policy.sample_messages_batch(contexts,seeds)
            for key,value in [('reasoning_effort','xhigh'),('calibration_only',False),('completion_token_limit',1536)]:
                changed=deepcopy(batch);changed['records'][0][key]=value
                with self.assertRaisesRegex(ValueError,'identity'):
                    calibration.validate_batch(changed,contexts,seeds,cfg,'draft')
            budget=engine.status['execution_budget'];at=datetime.fromisoformat(budget['first_started'])+timedelta(seconds=100)
            later,remaining=calibration.journal.execution_budget(cfg,budget,at=at)
            self.assertEqual(later,budget);self.assertEqual(remaining,8900.)


if __name__=='__main__':unittest.main()
