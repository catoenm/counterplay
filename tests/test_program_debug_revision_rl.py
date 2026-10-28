"""CPU-only journal fixtures; synthetic samples/weights are not experiment data.

The real transition validator checks semantic fake tensor/Adam identities here;
real Torch derivatives/Adam transitions have separate backend tests. No model,
API, GPU, rental, or actual heldout result is produced by these tests.
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from selfplay import program_debug_revision_rl as rl
from selfplay import program_cardinal_gpu as gpu
from tests.test_program_debug_revision_calibration import FakeCalibrationPolicy, fixture_score
from tests.test_program_cardinal_gpu import deployment


def identity(weight,step):
    return {'step':step,'learning_rate':1e-5,'trainable_names':['q'],
        'adapter_tensors':{'q':{'shape':[1],'dtype':'float32','sha256':gpu.object_hash(weight)}},
        'parameter_schema':{'q':{'shape':[1],'dtype':'float32'}},
        'parameter_groups':[{'params':['q'],'lr':1e-5}],
        'state_sha256':gpu.object_hash({'step':step,'lr':1e-5}),
        'rng':{'cpu':'fixture','cuda':[]},'frozen_state_sha256':'frozen-fixture',
        'quantization_state_sha256':'quant-fixture'}


def fake_identity(folder,expected_learning_rate=None):
    opt=rl.common.load(folder/'optimizer.pt');weight=rl.common.load(folder/'adapter.safetensors')['weight']
    if opt['learning_rate']!=expected_learning_rate:raise ValueError('LR mismatch')
    return identity(weight,opt['step'])


def state(ident):
    return {'adapter_sha256':gpu.object_hash(ident['adapter_tensors']),
        'optimizer_state_sha256':ident['state_sha256'],'rng':ident['rng'],
        'frozen_state_sha256':ident['frozen_state_sha256'],
        'gradients_sha256':gpu.object_hash({'q':None}),'config_sha256':'fixture','model_training':False}


class FakeTrainPolicy(FakeCalibrationPolicy):
    def __init__(self,bad_drafts=True):
        super().__init__(bad_drafts=bad_drafts)
        self.probe_calls=0;self.update_calls=0;self.fail_probe=False;self.ambiguous_update=False

    def sample_messages_batch(self,*args,**kwargs):
        batch=super().sample_messages_batch(*args,**kwargs)
        for row in batch['records']:row.update(calibration_only=False,training_capable=True)
        return batch

    def full_backward_probe(self,records):
        self.probe_calls+=1
        report=super().full_completion_likelihood_gate(records)
        report.update(kind='revision_full_backward_probe',coefficients=[1.]*16,objective_divisor=16,
            derivative={'finite_nonzero':not self.fail_probe},passed=not self.fail_probe)
        return report

    def update_trajectories(self,rows,advantages):
        self.update_calls+=1
        before=identity(self.weight,self.step)
        if any(advantages):
            self.step+=1;self.weight+=.01;self.updates+=1
            if self.ambiguous_update:raise RuntimeError('Actual step happened; result unknown')
        after=identity(self.weight,self.step)
        return {'passed':True,'outcome':'updated' if any(advantages) else 'skipped_zero_variance',
            'optimizer_steps':int(any(advantages)),'adapter_l2_delta':.01 if any(advantages) else 0.,
            'optimizer_step_attempted':bool(any(advantages)),'optimizer_step_returned':bool(any(advantages)),
            'adam_step_before':before['step'],'adam_step_after':after['step'],
            'state_before':state(before),'state_after':state(after),'advantages':advantages,
            'turn_sha256':[gpu.object_hash(t) for row in rows for t in row['turns']],
            'objective_divisor':8,'trajectory_count':8,'sampled_turn_count':16,
            'training_use':True,'fixture_only':True}


@contextmanager
def fixture(bad_drafts=True):
    with tempfile.TemporaryDirectory() as directory,patch.object(rl,'SOURCES',[]),patch.object(gpu,'checkpoint_state',fake_identity):
        run=Path(directory);dep=deployment()
        dep.update(maximum_hours=25,reservation_usd=145,estimate_usd=136.4,maximum_gpu_rate_usd_hour=5.29,
            maximum_cpu_rate_usd_hour=.14,deadline_unix=(datetime.now(timezone.utc)+timedelta(hours=25)).timestamp())
        (run/'inputs').mkdir()
        for name in ('design-plan.json','root-acceptance.json'):rl.common.atomic_json(run/'inputs'/name,{'fixture_only':True})
        cfg=rl.config(dep)
        training={'selected':[{'id':n,'origin':'scripted' if i<2 else 'model','ir_sha256':n,'compiled_ir':{},
            'source_checkpoint':'fixture'} for i,n in enumerate(('nearest','value','initial0','initial1'))]}
        held=deepcopy(training)
        for i in (2,3):held['selected'][i].update(id='held'+str(i),ir_sha256='held'+str(i))
        values=[('deployment.json',dep),('plan.json',rl.plan(dep,run/'inputs/design-plan.json',run/'inputs/root-acceptance.json')),
            ('schedule.json',rl.schedule()),('map-fingerprints.json',{}),('initial-opponents.json',training),
            ('evaluation-opponents.json',held)]
        for name,value in values:rl.common.atomic_json(run/name,value)
        (run/'prompts').mkdir();(run/'prompts/debug.txt').write_text(rl.harness.debug_model_spec())
        policy=FakeTrainPolicy(bad_drafts);files=policy.save_checkpoint(run/'inputs/initial-checkpoint')
        names=[n for n,_ in values]+['prompts/debug.txt','inputs/design-plan.json','inputs/root-acceptance.json']
        inputs={n:rl.common.digest(run/n) for n in names}
        manifest={'kind':cfg['kind'],'config':cfg,'source_sha256':{},'input_sha256':inputs,
            'initial_checkpoint_files':files,'prompt_sha256':{'debug':inputs['prompts/debug.txt']}}
        rl.common.atomic_json(run/'manifest.json',manifest)
        budget,_=rl.journal.execution_budget(cfg)
        rl.common.atomic_json(run/'status.json',{'manifest_sha256':rl.common.digest(run/'manifest.json'),
            'artifacts':{},'state':'prepared','stage':'prepared','claims_allowed':False,'execution_budget':budget})
        engine=rl.RevisionRL(run,policy);engine.initialize()
        yield run,engine,policy


def two(engine):
    return engine.trajectories('fixture',['fixture0','fixture1'],[301000000,301000002],'base',[[401000000,401000001],[401000002,401000003]])


def zero_score(record,*args):
    result=fixture_score(record,*args);result['reward']=-1.;return result


class RevisionRLRunnerTests(unittest.TestCase):
    def test_one_group_eight_trajectories_all16_gate_probe_and_actual_transition(self):
        with fixture() as (run,engine,policy),patch.object(rl,'score',fixture_score):
            report=engine.group(1)
            self.assertEqual((len(policy.calls),policy.probe_calls,policy.updates),(8,1,1))
            self.assertEqual(report['optimizer_steps'],1)
            self.assertEqual(engine.checkpoint('cycle1')['optimizer_identity']['step'],1)
            self.assertEqual(len(report['turn_sha256']),16)
            self.assertEqual(len(report['trajectory_turn_sha256']),8)
            self.assertEqual(report['backend_result_sha256'],engine.status['artifacts']['groups/1/optimizer-result.json'])
            self.assertTrue(rl.validate_checkpoint_transition(report,engine.checkpoint('base')['optimizer_identity'],engine.checkpoint('cycle1')['optimizer_identity']))
            timing=engine.existing('groups/1/timing.json');start=engine.existing('groups/1/timing-start.json')
            self.assertEqual(timing['elapsed_seconds'],(datetime.fromisoformat(timing['finished_at'])-datetime.fromisoformat(start['started'])).total_seconds())
            before=(len(policy.calls),policy.update_calls)
            rl.RevisionRL(run,policy).group(1)
            self.assertEqual(before,(len(policy.calls),policy.update_calls))

    def test_skip_preserves_exact_files_and_three_zero_profiles_choose_by_time_only(self):
        with fixture() as (run,engine,policy),patch.object(rl,'score',zero_score):
            for n in range(1,4):self.assertEqual(engine.group(n)['optimizer_steps'],0)
            self.assertEqual(policy.updates,0);self.assertEqual(policy.probe_calls,1)
            self.assertEqual(engine.checkpoint('base')['files'],engine.checkpoint('cycle3')['files'])
            self.assertIn('groups/1/skip-result.json',engine.status['artifacts'])
            self.assertNotIn('groups/1/optimizer-intent.json',engine.status['artifacts'])
            cost=engine.full_cost();self.assertGreaterEqual(cost['full_update_price_seconds'],120.)
            self.assertEqual(rl.endpoint_decision(cost,86000,engine.config)['endpoint'],25)

    def test_ambiguous_optimizer_and_failed_save_never_repeat_step(self):
        for mode in ('step_then_throw','save_after_step'):
            with self.subTest(mode=mode),fixture() as (run,engine,policy),patch.object(rl,'score',fixture_score):
                policy.ambiguous_update=mode=='step_then_throw';policy.fail_save=mode=='save_after_step'
                with self.assertRaises(RuntimeError):engine.group(1)
                self.assertEqual(policy.updates,1)
                resumed=rl.RevisionRL(run,policy)
                self.assertIsNone(resumed.group(1));self.assertEqual(policy.updates,1)
                self.assertEqual(resumed.status['state'],'failed')
                self.assertIsNone(resumed.checkpoint('cycle1'))

    def test_failed_probe_preserved_without_update_or_scoring(self):
        with fixture() as (run,engine,policy),patch.object(rl,'score',side_effect=AssertionError('No scores allowed')):
            policy.fail_probe=True
            self.assertIsNone(engine.group(1));self.assertEqual(policy.update_calls,0)
            self.assertIn('groups/1/probe.json',engine.status['artifacts'])
            self.assertEqual(engine.status['state'],'numerical_failed')

    def test_actual_target_transition_rejected_before_atomic_commit(self):
        with fixture() as (run,engine,policy),patch.object(rl,'score',fixture_score):
            update=policy.update_trajectories
            def tamper(*args):
                report=update(*args);report['state_after']['adapter_sha256']='not-actual';return report
            with patch.object(policy,'update_trajectories',tamper),self.assertRaisesRegex(ValueError,'Saved checkpoint'):
                engine.group(1)
            self.assertIsNone(engine.checkpoint('cycle1'))
            self.assertNotIn('groups/1/update.json',engine.status['artifacts'])

    def test_complete25_group_state_machine_seals_all32_before_scoring(self):
        with fixture() as (run,engine,policy):
            def scoring(record,*args):
                if record['name'].startswith('eval-'):
                    seal=engine.existing('evaluation/sealed.json')
                    self.assertEqual((seal['draws'],seal['sampled_turns']),(32,64))
                    self.assertEqual(len(seal['generations']),32)
                return fixture_score(record,*args)
            with patch.object(rl,'score',scoring):engine.run_all()
            self.assertEqual(engine.status['state'],'complete')
            self.assertEqual(engine.status['actual_optimizer_steps'],25)
            self.assertEqual(engine.status['attempted_groups'],25)
            self.assertEqual(len(engine.existing('evaluation/base/holdout.json')['scores']),16)
            self.assertEqual(len(engine.existing('evaluation/cycle25/holdout.json')['scores']),16)
            for i in range(16):
                a=engine.existing(f'generations/eval-base-{i}.json');b=engine.existing(f'generations/eval-cycle25-{i}.json')
                self.assertEqual(a['trajectory_seed'],b['trajectory_seed'])
                self.assertEqual(a['practice_seeds'],b['practice_seeds'])
            selection=engine.existing('endpoint-selection.json');stop=datetime.fromisoformat(engine.status['execution_budget']['hard_stop'])
            self.assertEqual(selection['remaining_seconds'],(stop-datetime.fromisoformat(selection['selected_at'])).total_seconds())
            self.assertEqual(selection['reserved_final_and_export_seconds'],32400)

    def test_batch_recovery_materializes_without_new_draw_and_unknown_intent_stops(self):
        with fixture() as (run,engine,policy):
            original=engine.commit
            def crash(path,value):
                if path=='generations/fixture1.json':raise RuntimeError('after sample')
                return original(path,value)
            with patch.object(engine,'commit',crash),self.assertRaises(RuntimeError):two(engine)
            self.assertEqual(len(policy.calls),2)
            resumed=rl.RevisionRL(run,policy);resumed.activate('base');self.assertEqual(len(two(resumed)),2)
            self.assertEqual(len(policy.calls),2)
        with fixture() as (run,engine,policy):
            policy.fail_sample=True
            with self.assertRaises(RuntimeError):two(engine)
            with self.assertRaisesRegex(ValueError,'never redraw'):two(engine)
            self.assertEqual(len(policy.calls),1)

    def test_timing_boundaries_reserve_once_and_full_price_for_skips(self):
        cfg={'timing_safety_factor':1.5,'endpoint_choices':[10,15,25]}
        cost={'timing_sha256':{'1':'a','2':'b','3':'c'},'maximum_full_group_seconds':2000}
        self.assertEqual(rl.endpoint_decision(cost,32400+3000*22,cfg)['endpoint'],25)
        self.assertEqual(rl.endpoint_decision(cost,32400+3000*22-1,cfg)['endpoint'],15)
        self.assertEqual(rl.endpoint_decision(cost,32400+3000*12-1,cfg)['endpoint'],10)
        self.assertFalse(rl.endpoint_decision(cost,32400+3000*7-1,cfg)['passed'])

    def test_league_rejects_reserved_ir_and_selects_latest_distinct_without_extra_games(self):
        with fixture() as (run,engine,policy):
            initial=engine.roster(1)
            engine.commit('groups/5/opponents.json',initial)
            for n in range(1,6):
                rows=[]
                for i,ir in enumerate(('new'+str(n),'held2')):
                    rows.append({'runtime_valid':True,'matches':[{},{}],'candidate_program_faults':0,
                        'positive_banking_games':1,'generation':f'train-{n}-{i}','generation_sha256':'fixture',
                        'classification':{'program':{'ir_sha256':ir,'compiled_ir':{}}}})
                engine.commit(f'groups/{n}/scores.json',{'source_checkpoint':'base','rows':rows})
            engine.refresh(5);roster=engine.existing('league/after-5.json')
            self.assertEqual([r['ir_sha256'] for r in roster['selected'][2:]],['new4','new5'])
            self.assertEqual(len(policy.calls),0)

    def test_all_new_layouts_disjoint_and_fixed_collision_replacement(self):
        maps=rl.fingerprints(rl.schedule(),rl.prior_maps(rl.PARENT))
        self.assertEqual(sum(map(len,maps['families'].values())),488)
        self.assertEqual(rl.schedule()['train_practice_maps']['8'][11],219000000)


if __name__=='__main__':unittest.main()
