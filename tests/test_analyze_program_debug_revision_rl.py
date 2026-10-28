"""Independent RL audit fixtures are synthetic CPU evidence, never real results."""
import copy
import math
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

from scripts import analyze_program_debug_revision_rl as audit

CONFIG = {'token_logprob_tolerance': .002, 'sequence_logprob_tolerance': .02}


class RevisionRLAuditHelpers(unittest.TestCase):
    def rows(self, prefix):
        return [{'generation': f'{prefix}-{i}', 'authorship_valid': True, 'runtime_valid': True,
                 'games': 48, 'banked_point_yield': float(i)} for i in range(16)]

    def test_paired_denominator_retains_invalid_and_played_faults(self):
        base, final = self.rows('base'), self.rows('final')
        final[0].update(authorship_valid=False, runtime_valid=False, games=0, banked_point_yield=0.)
        final[1].update(runtime_valid=False, banked_point_yield=3.)
        result = audit.paired_summary(base, final, list(range(16)), 200)
        self.assertEqual(result['program_cluster_n'], 16)
        self.assertEqual(result['mean_yield_difference'], 2 / 16)
        self.assertEqual(result, audit.paired_summary(base, final, list(range(16)), 200))
        for broken in (final[:-1], [final[0]] * 16):
            with self.assertRaises(ValueError): audit.paired_summary(base, broken, list(range(16)), 200)
        bad = copy.deepcopy(final); bad[0]['games'] = 48
        with self.assertRaises(ValueError): audit.paired_summary(base, bad, list(range(16)), 200)
        bad = copy.deepcopy(final); bad[2]['games'] = 47
        with self.assertRaises(ValueError): audit.paired_summary(base, bad, list(range(16)), 200)

    def test_timing_prices_full_groups_and_reserves_evaluation_once(self):
        # Update counts/rewards cannot enter this pure timing-only selector.
        costs = [2000., 2100., 1900.]
        boundary = 1.5 * 2100 * 7 + 32400
        self.assertEqual(audit.select_endpoint(costs, boundary)['endpoint'], 10)
        self.assertIsNone(audit.select_endpoint(costs, boundary - .001)['endpoint'])
        self.assertEqual(audit.select_endpoint(costs, 1.5 * 2100 * 12 + 32400)['endpoint'], 15)
        self.assertEqual(audit.select_endpoint(costs, 1.5 * 2100 * 22 + 32400)['endpoint'], 25)
        for costs in ([1., 2.], [1., 2., float('nan')], [1., 2., 0.]):
            with self.assertRaises(ValueError): audit.select_endpoint(costs, 90000)

    def trajectories(self):
        return [{'name': f'train-1-{i}', 'checkpoint': 'base', 'training_tokens': 5,
                 'turns': [{'seed': i * 2, 'authoring_stage':'draft', 'completion_ids': [1, 2], 'sampled_token_logprobs': [-.2, -.3]},
                           {'seed': i * 2 + 1, 'authoring_stage':'revision', 'completion_ids': [1, 2, 3], 'sampled_token_logprobs': [-.4, -.5, -.6]}]}
                for i in range(8)]

    def update(self, trajectories, rewards):
        advantages = audit.previous.advantages(rewards)
        checks, losses = [], []
        for i, trajectory in enumerate(trajectories):
            for turn in trajectory['turns']:
                total = math.fsum(turn['sampled_token_logprobs'])
                check=audit.calibration.vector_check(turn['sampled_token_logprobs'],turn['sampled_token_logprobs'],CONFIG)
                checks.append({**check,'index':len(checks),'record_sha256':audit.value_sha(turn),'seed':turn['seed'],
                    'authoring_stage':turn['authoring_stage'],'train_grad_enabled':True,'backward_completed':True,
                    'sampled_token_logprobs':turn['sampled_token_logprobs'][:],
                    'train_token_logprobs':turn['sampled_token_logprobs'][:], 'train_all_finite':True,
                    'train_check':check,'coefficient':advantages[i],'objective_divisor':8})
                losses.append(-advantages[i] * total / 8)
        return advantages, {'objective_divisor': 8, 'training_tokens': 40, 'optimizer_steps': 1,
            'trajectory_count': 8, 'sampled_turn_count': 16, 'tool_feedback_target_tokens': 0,
            'outcome': 'updated', 'gradient_likelihood_gate': {'passed': True, 'records': checks},
            'coefficients':[float(a) for a in advantages for _ in range(2)],
            'microbatch_losses': losses, 'loss': math.fsum(losses), 'gradient_norm_before_clip': .8,
            'adam_step_before': 0, 'adam_step_after': 1}

    def test_rloo_sum_both_turns_divided_by_eight_not_sixteen_or_tokens(self):
        trajectories = self.trajectories(); rewards = [-1., 0., .5, 1., -.5, .2, .3, -.2]
        advantages, update = self.update(trajectories, rewards)
        self.assertFalse(audit.validate_objective(update, trajectories, rewards, advantages, CONFIG)['zero_signal'])
        for factor in (.5, 1 / 5):
            bad = copy.deepcopy(update)
            bad['microbatch_losses'] = [x * factor for x in bad['microbatch_losses']]
            bad['loss'] = math.fsum(bad['microbatch_losses'])
            with self.assertRaisesRegex(ValueError, 'objective'): audit.validate_objective(bad, trajectories, rewards, advantages, CONFIG)
        bad = copy.deepcopy(update); bad['gradient_likelihood_gate']['records'].pop()
        with self.assertRaises(ValueError): audit.validate_objective(bad, trajectories, rewards, advantages, CONFIG)
        bad = copy.deepcopy(update); bad['adam_step_after'] = 2
        with self.assertRaises(ValueError): audit.validate_objective(bad, trajectories, rewards, advantages, CONFIG)

    def test_zero_signal_is_preserved_without_artificial_optimizer_step(self):
        trajectories = self.trajectories(); rewards = [-1.] * 8
        update = {'objective_divisor': 8, 'training_tokens': 40, 'optimizer_steps': 0,
                  'outcome': 'skipped_zero_variance'}
        self.assertTrue(audit.validate_objective(update, trajectories, rewards, [0.] * 8, CONFIG)['zero_signal'])
        update['optimizer_steps'] = 1
        with self.assertRaises(ValueError): audit.validate_objective(update, trajectories, rewards, [0.] * 8, CONFIG)


class RevisionRLAuditTensorEvidence(unittest.TestCase):
    """Actual CPU tensors/Adam; no pretrained model or inference."""
    def setUp(self):
        import torch
        self.rng=[torch.get_rng_state().clone()]
        get=patch.object(torch.cuda,'get_rng_state_all',side_effect=lambda:[x.clone() for x in self.rng])
        put=patch.object(torch.cuda,'set_rng_state_all',side_effect=lambda x:self.rng.__setitem__(slice(None),x))
        get.start();put.start();self.addCleanup(get.stop);self.addCleanup(put.stop)

    def test_real_full_backward_probe_and_partial_failure(self):
        from tests.test_program_debug_revision_train import training_fixture
        from selfplay.program_cardinal_cuda import checkpoint_state
        for failed in (False,True):
            with self.subTest(failed=failed),tempfile.TemporaryDirectory() as directory:
                policy,records,_,_=training_fixture(long_record=True)
                source=Path(directory)/'source';policy.save_checkpoint(source)
                identity=checkpoint_state(source,1e-5)
                if failed:
                    original=policy.token_logps
                    def interrupted(row):
                        if row['seed']==records[3]['seed'] and policy.model.training:
                            raise RuntimeError('Fixture backward admission failure')
                        return original(row)
                    policy.token_logps=interrupted
                report=policy.full_backward_probe(records)
                checked=audit.validate_probe(report,records,identity,policy.config)
                self.assertEqual(checked['passed'],not failed)
                self.assertEqual(checked['verified_turns'],3 if failed else 16)
                bad=copy.deepcopy(report);bad['coefficients'][0]=0.
                with self.assertRaises(ValueError):audit.validate_probe(bad,records,identity,policy.config)
                bad=copy.deepcopy(report);bad['state_before']['optimizer_state_sha256']='0'*64
                with self.assertRaises(ValueError):audit.validate_probe(bad,records,identity,policy.config)
                if not failed:
                    bad=copy.deepcopy(report);bad['derivative']['gradient_tensors'].pop(next(iter(bad['derivative']['gradient_tensors'])))
                    with self.assertRaisesRegex(ValueError,'coverage'):audit.validate_probe(bad,records,identity,policy.config)

    def test_actual_adam_transition_and_zero_signal_tensor_identity(self):
        from tests.test_program_debug_revision_train import training_fixture
        from selfplay.program_cardinal_cuda import checkpoint_state
        for skipped in (False,True):
            with self.subTest(skipped=skipped),tempfile.TemporaryDirectory() as directory:
                policy,_,trajectories,_=training_fixture()
                for i,row in enumerate(trajectories):
                    row.update(name=f'train-1-{i}',checkpoint='base',training_tokens=sum(len(t['completion_ids']) for t in row['turns']))
                rewards=[-1.]*8 if skipped else [1.,-1.,.5,-.5,.75,-.75,.25,-.25]
                advantages=audit.previous.advantages(rewards)
                source=Path(directory)/'source';target=Path(directory)/'target'
                policy.save_checkpoint(source);before=checkpoint_state(source,1e-5)
                report=policy.update_trajectories(trajectories,advantages)
                report['training_tokens']=sum(r['training_tokens'] for r in trajectories)
                policy.save_checkpoint(target);after=checkpoint_state(target,1e-5)
                self.assertEqual(audit.validate_objective(report,trajectories,rewards,advantages,policy.config)['zero_signal'],skipped)
                audit.validate_checkpoint_update(report,before,after,policy.config)
                bad=copy.deepcopy(after);bad['learning_rate']=1e-4
                with self.assertRaises(ValueError):audit.validate_checkpoint_update(report,before,bad,policy.config)
                if not skipped:
                    bad=copy.deepcopy(report);bad['gradient_likelihood_gate']['records'][0]['train_token_logprobs'][0]-=.1
                    with self.assertRaises(ValueError):audit.validate_objective(bad,trajectories,rewards,advantages,policy.config)
                    bad=copy.deepcopy(report);bad['derivative_after_clip']['gradient_l2_norm']=99.
                    with self.assertRaises(ValueError):audit.validate_checkpoint_update(bad,before,after,policy.config)


from datetime import datetime, timedelta, timezone
from selfplay import program_debug_revision_rl as runner
from tests.test_analyze_program_debug_revision_calibration import Backend as CalibrationBackend
from tests.test_program_cardinal_gpu import deployment


class SyntheticJournalBackend(CalibrationBackend):
    """Synthetic draws/gates, real checkpoint serialization and tokenizer."""
    def sample_messages_batch(self,*args,**kwargs):
        result=super().sample_messages_batch(*args,**kwargs)
        smoke=kwargs.get('maximum')==32
        for row in result['records']:
            text=('r</think>' if row['authoring_stage']=='draft' else '')+'invalid fixture'
            ids=self.tokenizer.encode(text,add_special_tokens=False)+[248046]
            row.update(completion_ids=ids,output_tokens=len(ids),sampled_token_logprobs=[-.1]*len(ids),
                text=self.tokenizer.decode(ids,skip_special_tokens=True),raw_text=self.tokenizer.decode(ids,skip_special_tokens=False),
                calibration_only=smoke)
            if not smoke:row['training_capable']=True
        lengths=[r['output_tokens'] for r in result['records']]
        counts=[sum(n>i for n in lengths) for i in range(max(lengths))]
        for r in result['records']:r['post_eos_cache_forwards']=len(counts)-r['output_tokens']
        result['timing'].update(total_output_tokens=sum(lengths),active_sequences_by_step=counts,batch_steps=len(counts))
        return result
    def full_backward_probe(self,records):
        result=self.vectors(records)
        losses=[]
        for row in result['records']:
            loss=-math.fsum(row['train_token_logprobs'])/16;losses.append(loss)
            row.update(coefficient=1.,objective_divisor=16,backward_completed=True,train_loss=loss)
        result.update(kind='revision_full_backward_probe',expected_records=16,coefficients=[1.]*16,
            objective_divisor=16,microbatch_losses=losses,loss=math.fsum(losses),
            sampled_tokens=sum(len(r['completion_ids']) for r in records),
            derivative={'gradient_l2_norm':1.,'missing_gradient_names':[], 'nonfinite_gradient_names':[], 'finite_nonzero':True,
                'gradient_tensors':{r['name']:{'shape':r['shape'],'dtype':r['dtype'],'squared_l2_norm':float(i==0)}
                    for i,r in enumerate(self.identity['parameter_schema'])}},
            fixture_only=True)
        return result
    def update_trajectories(self,rows,advantages):
        if any(advantages):raise AssertionError('This journal fixture only exercises exact zero-signal copies')
        ident=copy.deepcopy(self.identity);ident.pop('files',None)
        state=self.state_identity()
        return {'passed':True,'outcome':'skipped_zero_variance','optimizer_steps':0,'adapter_l2_delta':0.,
            'optimizer_step_attempted':False,'optimizer_step_returned':False,
            'adam_step_before':ident['step'],'adam_step_after':ident['step'],
            'optimizer_identity_before':ident,'optimizer_identity_after':copy.deepcopy(ident),
            'state_before':state,'state_after':copy.deepcopy(state),
            **{k:True for k in ('rng_restored','config_restored','model_mode_restored','frozen_state_unchanged','gradients_cleared')},
            'advantages':advantages,'turn_sha256':[audit.value_sha(t) for row in rows for t in row['turns']],
            'objective_divisor':8,'trajectory_count':8,'sampled_turn_count':16,'training_use':True,'fixture_only':True}


def prepared(folder):
    root=Path(folder);dep=deployment()
    # Fixed shorter remaining window deliberately selects ten attempts. This
    # is a temporary CPU fixture, never a real deployment or live deadline.
    dep.update(maximum_hours=25,reservation_usd=145,estimate_usd=136.4,
        maximum_gpu_rate_usd_hour=5.29,maximum_cpu_rate_usd_hour=.14,
        deadline_unix=int((datetime.now(timezone.utc)+timedelta(seconds=34050)).timestamp()))
    runner.common.atomic_json(root/'deployment.json',dep)
    run=root/'run';runner.prepare(run,root/'deployment.json',runner.ROOT/'runs/cardinal-gpu-preflight/model-info.json')
    return run


def initialized(run):
    backend=SyntheticJournalBackend();engine=runner.RevisionRL(run,backend);backend.config=engine.config
    budget,_=runner.journal.execution_budget(engine.config);engine.status['execution_budget']=budget;engine.state()
    runtime=audit.read(runner.PARENT/'runtime.json')
    runtime.update(calibration_only=False,training_capable=True,turn_native_modes=engine.config['turn_native_modes'])
    engine.commit('runtime.json',runtime);engine.initialize()
    initial=runner.harness.initial_messages()
    contexts=[initial,initial+[{'role':'assistant','content':'fixture'},{'role':'user','content':'Check fixture.'}]]
    seeds=[70000000+engine.config['seed']+i for i in range(2)]
    batch=backend.sample_messages_batch(contexts,seeds,stages=['draft','revision'],maximum=32)
    engine.commit('smoke-fixture.json',{**backend.vectors(batch['records']),
        'kind':'revision_mixed_native_derivative_admission','batch_size':2,'max_new_tokens':32,
        'authoring_stages':['draft','revision'],'seeds':seeds,'source_checkpoint':'base',
        'source_metadata_sha256':engine.status['artifacts']['checkpoints/base/metadata.json'],'batch':batch,
        'derivative':{'finite_nonzero':True,'gradient_l2_norm':.1,'missing_gradient_names':[]},'fixture_only':True})
    return engine,backend


class RevisionRLAuditJournalTests(unittest.TestCase):
    def test_real_prepared_parent_and_complete_zero_signal_with_paired16(self):
        with tempfile.TemporaryDirectory() as directory:
            run=prepared(directory);result=audit.audit(run)
            self.assertTrue(result['integrity_passed']);self.assertFalse(result['planned_study_complete'])
            self.assertEqual(audit.registered_schedule(),runner.schedule())
            engine,backend=initialized(run);engine.run_all()
            self.assertEqual(engine.status['state'],'complete')
            result=audit.audit(run)
            self.assertTrue(result['planned_study_complete']);self.assertEqual(result['selected_endpoint'],10)
            self.assertEqual(result['genuine_RL_optimizer_steps'],0)
            self.assertEqual(result['committed_trajectories'],112)
            self.assertEqual(result['committed_raw_sampled_turns'],224)
            self.assertEqual(result['paired_yield']['program_cluster_n'],16)
            self.assertEqual(result['paired_yield']['mean_yield_difference'],0.)
            self.assertEqual(result['full_backward_admission']['verified_turns'],16)
            for label in ('base','cycle10'):
                cell=result['evaluation_cells'][label]
                self.assertEqual(cell['prescribed_program_clusters'],16)
                self.assertTrue(all(r['games']==0 for r in cell['programs']))
            calls=backend.calls;runner.RevisionRL(run,backend).run_all();self.assertEqual(calls,backend.calls)
            bad=run/'generations/eval-cycle10-15.json';row=audit.read(bad);row['text']='substitution'
            runner.common.atomic_json(bad,row)
            with self.assertRaisesRegex(ValueError,'Artifact'):audit.audit(run)


if __name__ == '__main__': unittest.main()
