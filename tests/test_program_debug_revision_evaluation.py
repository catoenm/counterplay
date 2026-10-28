"""CPU fixtures only: synthetic neural bytes/values, no model or new experiment."""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from selfplay import program_debug_revision_evaluation as evaluation
from selfplay import program_debug_revision_rl as original
from tests.test_program_debug_revision_rl import fixture as parent_fixture, FakeTrainPolicy, fake_identity
from tests.test_program_debug_revision_calibration import fixture_score


class FakeEvaluationPolicy(FakeTrainPolicy):
    def __init__(self):
        super().__init__(bad_drafts=True)
        self.smokes=[];self.fail_smoke=False
    def mechanics_smoke(self,contexts):
        self.smokes.append(self.step)
        return {'passed':not self.fail_smoke,'optimizer_steps':0,'fixture_only':True}
    def update(self,*args,**kwargs):raise AssertionError('Evaluation optimizer forbidden')
    update_trajectories=update
    full_backward_probe=update


@contextmanager
def prepared():
    with parent_fixture() as (parent,engine,_), tempfile.TemporaryDirectory() as temporary, \
            patch.object(evaluation,'SOURCES',[]),patch.object(evaluation,'checkpoint_state',fake_identity):
        # Add the opaque model-info fixture to the originally synthetic parent.
        # Rebind this fixture's one base metadata before its first update.
        info=parent/'inputs/model-info.json';evaluation.common.atomic_json(info,{'fixture_only':True})
        manifest=evaluation.common.load(parent/'manifest.json')
        manifest['input_sha256']['inputs/model-info.json']=evaluation.common.digest(info)
        evaluation.common.atomic_json(parent/'manifest.json',manifest)
        engine.manifest=manifest;engine.status['manifest_sha256']=evaluation.common.digest(parent/'manifest.json')
        meta=parent/'checkpoints/base/metadata.json';row=evaluation.common.load(meta)
        row['manifest_sha256']=engine.status['manifest_sha256'];evaluation.common.atomic_json(meta,row)
        engine.status['artifacts']['checkpoints/base/metadata.json']=evaluation.common.digest(meta);engine.state()
        with patch.object(original,'score',fixture_score):
            for n in (1,2,3):engine.group(n)
        engine.commit('endpoint-selection.json',{'passed':False,'endpoint':None,'fixture_only':True})
        engine.state(state='throughput_failed',stage='endpoint_admission',claims_allowed=False)
        evaluation.common.atomic_json(parent/'final-audit.json',{'integrity_passed':True,
            'planned_study_complete':False,'planned_comparisons_available':False,'selected_endpoint':None,
            'manifest_sha256':evaluation.common.digest(parent/'manifest.json'),
            'status_sha256':evaluation.common.digest(parent/'status.json'),'genuine_RL_optimizer_steps':3,
            'fixture_only':True})
        base=Path(temporary);deployment=evaluation.common.load(parent/'deployment.json')
        deployment.update(maximum_hours=10,reservation_usd=65,estimate_usd=55,
            deadline_unix=(datetime.now(timezone.utc)+timedelta(hours=10)).timestamp(),
            phase='coding_evaluation',continuation_id='a'*32,combined_funding_usd=240.,
            intent='d'*32,funding_training_intent='b'*32,evaluation_intent='c'*32,
            coding_evaluation_intent='d'*32,funding_training_deployment='/fixture/training',
            evaluation_deployment='/fixture/evaluation',coding_evaluation_deployment='/fixture/coding',
            predecessor_deployment=str(parent),predecessor_evidence_sha256={'fixture':'unused'},
            mixed_controller_sha256='fixture',inherited_controller_sha256='fixture')
        evaluation.common.atomic_json(base/'deployment.json',deployment)
        run=base/'run';evaluation.prepare(run,base/'deployment.json',info,parent)
        status=evaluation.common.load(run/'status.json')
        status['execution_budget']=evaluation.journal.execution_budget(evaluation.common.load(run/'manifest.json')['config'])[0]
        evaluation.common.atomic_json(run/'status.json',status)
        policy=FakeEvaluationPolicy();runner=evaluation.Evaluation(run,policy)
        yield run,runner,policy,parent,engine


class EvaluationTests(unittest.TestCase):
    def test_parent_checkpoint_metadata_preserved_and_fixed_three_update_comparison(self):
        with prepared() as (run,engine,policy,parent,_):
            for label,step in [('base',0),('cycle3',3)]:
                for name in (*evaluation.gpu.FILES,'metadata.json'):
                    self.assertEqual((run/f'checkpoints/{label}/{name}').read_bytes(),(parent/f'checkpoints/{label}/{name}').read_bytes())
                self.assertEqual(engine.checkpoint(label)['optimizer_identity']['step'],step)
            self.assertNotEqual(engine.checkpoint('cycle3')['manifest_sha256'],engine.status['manifest_sha256'])
            self.assertEqual(policy.calls,[])
            for method in (engine.group,engine.save_checkpoint,policy.update,policy.update_trajectories):
                with self.assertRaises((RuntimeError,AssertionError)):method()

    def test_all32_paired_draws_sealed_before_scores_and_no_state_updates(self):
        with prepared() as (run,engine,policy,parent,_):
            parent_hash=evaluation.common.digest(parent/'status.json')
            def score(record,*args):
                seal=engine.existing('evaluation/sealed.json')
                self.assertEqual((seal['draws'],seal['sampled_turns']),(32,64))
                self.assertEqual(len(seal['generations']),32)
                self.assertEqual(len(seal['endpoint_state_proofs']),2)
                return fixture_score(record,*args)
            with patch.object(original,'score',score):engine.run_all()
            self.assertEqual(engine.status['state'],'complete')
            self.assertEqual(engine.status['new_optimizer_steps'],0)
            self.assertEqual(engine.status['inherited_optimizer_steps'],3)
            self.assertFalse(engine.status['parent_training_complete'])
            self.assertEqual(policy.smokes,[0,3]);self.assertEqual(len(policy.calls),32)
            for label in evaluation.ENDPOINTS:
                self.assertEqual(len(engine.existing(f'evaluation/{label}/holdout.json')['scores']),16)
                self.assertTrue(engine.existing(f'endpoint-state/{label}/proof.json')['unchanged_adapter_Adam_RNG'])
            for i in range(16):
                a=engine.existing(f'generations/eval-base-{i}.json');b=engine.existing(f'generations/eval-cycle3-{i}.json')
                self.assertEqual(a['trajectory_seed'],b['trajectory_seed'])
                self.assertEqual(a['practice_seeds'],b['practice_seeds'])
            self.assertEqual(evaluation.common.digest(parent/'status.json'),parent_hash)
            evaluation.Evaluation(run,policy).run_all();self.assertEqual(len(policy.calls),32)

    def test_crash_after_durable_sample_materializes_without_redrawing(self):
        with prepared() as (run,engine,policy,_,_):
            commit=engine.commit
            def crash(path,value):
                if path=='generations/eval-base-1.json':raise RuntimeError('Fixture after durable batch')
                return commit(path,value)
            with patch.object(engine,'commit',crash),self.assertRaises(RuntimeError):engine.run_all()
            self.assertEqual(len(policy.calls),2)
            resumed=evaluation.Evaluation(run,policy)
            with patch.object(original,'score',fixture_score):resumed.run_all()
            self.assertEqual(resumed.status['state'],'complete');self.assertEqual(len(policy.calls),32)
            self.assertEqual(policy.smokes,[0,0,3])

    def test_unknown_sample_and_mutated_checkpoint_never_become_valid_measurement(self):
        with prepared() as (run,engine,policy,_,_):
            policy.fail_sample=True
            with self.assertRaisesRegex(RuntimeError,'unknown'):engine.run_all()
            with self.assertRaisesRegex(ValueError,'never redraw'):evaluation.Evaluation(run,policy).run_all()
            self.assertEqual(len(policy.calls),1)
        with prepared() as (run,engine,policy,_,_):
            policy.mutate_on_sample=True
            with patch.object(original,'score',side_effect=AssertionError('No scoring after mutation')):engine.run_all()
            self.assertEqual(engine.status['state'],'numerical_failed')
            proof=engine.existing('endpoint-state/base/proof.json')
            self.assertFalse(proof['unchanged_adapter_Adam_RNG'])
            self.assertNotIn('evaluation/sealed.json',engine.status['artifacts'])

    def test_failed_smoke_retained_before_any_sampling(self):
        with prepared() as (run,engine,policy,_,_):
            policy.fail_smoke=True;engine.run_all()
            self.assertEqual(engine.status['state'],'numerical_failed')
            self.assertEqual(policy.calls,[])
            self.assertEqual(len([p for p in engine.status['artifacts'] if p.startswith('smoke/')]),1)

    def test_unchanged_input_and_source_hashes_and_no_observed_parent_holdout(self):
        with prepared() as (run,engine,policy,parent,parent_engine):
            (run/'schedule.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'bytes differ'):evaluation.verify_run(run)
            parent_engine.commit('batches/eval-base-0-draft-intent.json',{'fixture_only':True})
            parent_engine.state(state='throughput_failed',claims_allowed=False)
            audit=evaluation.common.load(parent/'final-audit.json');audit['status_sha256']=evaluation.common.digest(parent/'status.json')
            evaluation.common.atomic_json(parent/'final-audit.json',audit)
            with self.assertRaisesRegex(ValueError,'already been sampled'):evaluation.parent_evidence(parent)

    def test_reserve_counts_only_unrun_batches_and_persistent_hard_stop(self):
        with prepared() as (run,engine,policy,_,_):
            self.assertTrue(engine.remaining_admission('base'))
            proof=[engine.existing(p) for p in engine.status['artifacts'] if p.startswith('admissions/')][-1]
            self.assertEqual((len(proof['pending_batch_paths']),proof['required_seconds']),(32,32400))
            previous=engine.status['execution_budget']
            budget,remaining=evaluation.journal.execution_budget(engine.config,previous)
            self.assertEqual(budget['hard_stop'],previous['hard_stop'])
            self.assertLessEqual(remaining,35100)
            engine.status['execution_budget']=evaluation.journal.execution_budget(engine.config,
                at=datetime.now(timezone.utc)-timedelta(seconds=4000))[0]
            self.assertFalse(engine.remaining_admission('base'))
            self.assertEqual(engine.status['state'],'budget_incomplete')


if __name__=='__main__':unittest.main()
