"""Independent audit fixtures from real tiny Torch policy updates; no model calls."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
import time

from scripts.analyze_arena_mixed import check_gate, check_objective, uncommitted_operations, check_profile_checkpoint
from selfplay.arena_mixed.processes import _dispatch
from selfplay.arena_mixed.runner import Journal
from selfplay.arena_mixed import experiment
from tests import test_arena_mixed_cuda_policy as toy


class ObjectiveAuditTests(unittest.TestCase):
    setUp = toy.MixedCudaTests.setUp
    tearDown = toy.MixedCudaTests.tearDown
    base = toy.MixedCudaTests.base

    def update(self, *, lag_one=False, zero=False):
        obj = toy.fixture(**({'entropy_coefficient': 0., 'kl_coefficient': 0.} if zero else {}))
        self.base(obj)
        data = toy.episodes(obj, unequal_lengths=True, rewards=not zero)
        binding = deepcopy(obj.binding)
        raw = [row for episode in data for row in episode['transitions']]
        gate = obj.full_likelihood_gate(raw, binding)
        if lag_one:
            obj.train_cohort(data, binding, gate)
            obj.save_checkpoint(Path(self.temp.name)/'cycle1', 'cycle1', 1)
        report = obj.train_cohort(data, binding, gate)
        return obj.config, data, gate, report

    def test_actual_full_vectors_recompute_lag_zero_and_one_objective(self):
        for lag in (False, True):
            with self.subTest(lag_one=lag):
                cfg, data, gate, report = self.update(lag_one=lag)
                raw = [row for episode in data for row in episode['transitions']]
                check_gate(gate, raw, report['behavior'], cfg['likelihood_tolerance'])
                result = check_objective(report, data, cfg)
                self.assertEqual(report['policy_lag'], int(lag))
                self.assertEqual(report['optimizer_steps'], 1)
                self.assertEqual(result['rows'], len(raw))
                self.assertAlmostEqual(result['recomputed_loss'], report['loss'], places=6)

    def test_actual_zero_signal_skip_remains_auditable(self):
        cfg, data, _, report = self.update(zero=True)
        self.assertEqual(report['optimizer_steps'], 0)
        self.assertFalse(report['reward_learning_signal'])
        self.assertEqual(check_objective(report, data, cfg)['recomputed_loss'], 0.)

    def test_changed_advantage_old_vector_and_checkpoint_binding_rejected(self):
        cfg, data, _, report = self.update()
        mutations = {
            'advantage': lambda row: row.__setitem__('advantage', row['advantage']+1),
            'weight': lambda row: row.__setitem__('weight', row['weight']*2),
            'old_vector': lambda row: row.__setitem__('old_vector_sha256', 'f'*64),
            'source': lambda row: row['source_binding'].__setitem__('adapter_sha256', 'f'*64),
            'behavior': lambda row: row['behavior_binding'].__setitem__('adapter_sha256', 'f'*64),
        }
        for name, mutate in mutations.items():
            with self.subTest(corruption=name):
                bad = deepcopy(report)
                # Separate the record from possible shared Python object aliases.
                bad['learner_records'][0] = deepcopy(bad['learner_records'][0])
                mutate(bad['learner_records'][0])
                with self.assertRaises(ValueError): check_objective(bad, data, cfg)
        changed = deepcopy(data)
        changed[0]['transitions'][0]['old_log_probs'][0] -= .1
        with self.assertRaises(ValueError): check_objective(report, changed, cfg)

    def test_loss_partial_nonfinite_and_changed_learner_vectors_rejected(self):
        cfg, data, _, report = self.update()
        changes = {
            'loss': lambda x: x.__setitem__('loss', x['loss']+.1),
            'missing_row': lambda x: x['learner_records'].pop(),
            'missing_action': lambda x: x['learner_records'][0]['learner_log_probs'].pop(),
            'nonfinite': lambda x: x['learner_records'][0]['learner_log_probs'].__setitem__(0, float('nan')),
            'unnormalized': lambda x: x['learner_records'][0]['learner_log_probs'].__setitem__(0, -100.),
        }
        for name, change in changes.items():
            with self.subTest(corruption=name):
                bad = deepcopy(report); change(bad)
                with self.assertRaises(ValueError): check_objective(bad, data, cfg)

    def test_behavior_gate_full_vectors_modes_and_rows_rejected(self):
        cfg, data, gate, report = self.update()
        raw = [row for episode in data for row in episode['transitions']]
        changes = (
            lambda x: x['records'].pop(),
            lambda x: x['records'][0].__setitem__('grad_enabled', True),
            lambda x: x['records'][0]['actual_log_probs'].__setitem__(0, -100.),
            lambda x: x['row_sha256'].__setitem__(0, 'f'*64),
        )
        for change in changes:
            bad = deepcopy(gate); change(bad)
            with self.assertRaises(ValueError):
                check_gate(bad, raw, report['behavior'], cfg['likelihood_tolerance'])

    def test_actual_worker_checkpoint_gate_profile_and_update_api(self):
        obj = toy.fixture()
        journal = Journal(self.temp.name, 'a'*64, time.time()+30)
        reference = _dispatch(obj, {'method':'publish_base','data':{}}, journal)
        data = toy.episodes(obj)
        payload = {'cohort':'unused-toy-seal','current':reference,'behavior':reference,
            'prefix':'calibration/1'}
        with patch('selfplay.arena_mixed.processes.load_sealed_cohort',
                   return_value=({'binding':reference['binding']}, data)):
            profile = _dispatch(obj, {'method':'profile','data':payload}, journal)
            self.assertTrue(profile['passed'])
            self.assertEqual(obj.binding, reference['binding'])
            base = toy.policy.checkpoint_state(Path(self.temp.name)/'checkpoints/base')
            check_profile_checkpoint(Path(self.temp.name),profile,base,obj.config)
            for key,value in (('optimizer_step',2),('files',{})):
                bad=deepcopy(profile);bad['temporary_checkpoint'][key]=value
                with self.assertRaises(ValueError):
                    check_profile_checkpoint(Path(self.temp.name),bad,base,obj.config)
            bad=deepcopy(profile);bad['temporary_update']['parameter_delta_l2']+=.1
            with self.assertRaises(ValueError):
                check_profile_checkpoint(Path(self.temp.name),bad,base,obj.config)
            payload.update(prefix='cohorts/1', index=1)
            update = _dispatch(obj, {'method':'update','data':payload}, journal)
        self.assertEqual(update['checkpoint']['binding']['version'], 1)
        self.assertEqual(update['checkpoint']['optimizer_step'], 1)
        self.assertEqual(obj.binding, update['checkpoint']['binding'])
        check_objective(update['report'], data, obj.config)


class PredecessorAuditAdmissionTests(unittest.TestCase):
    def test_stale_status_incomplete_or_wrong_auditor_cannot_admit_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary); parent=folder/'parent'; parent.mkdir()
            metadata=folder/'model-info.json'; metadata.write_text('{}')
            tokenizer=folder/'tokenizer';tokenizer.mkdir()
            identities={'model.safetensors.index.json':'a'*64}
            for name in ('config.json','tokenizer.json','tokenizer_config.json'):
                (tokenizer/name).write_text(name)
                identities[name]=hashlib.sha256(name.encode()).hexdigest()
            identity_file=folder/'identities.json';identity_file.write_text(json.dumps(identities))
            deployment=folder/'deployment.json';deployment.write_text('{}')
            (parent/'manifest.json').write_text('{}'); (parent/'status.json').write_text('{}')
            status={'state':'training_complete'}
            parent_manifest={'sources':{'scripts/analyze_arena_mixed.py':'c'*64}}
            base={'integrity_passed':True,'complete':True,'training_complete':True,
                'state':'training_complete','manifest_sha256':experiment.file_sha(parent/'manifest.json'),
                'status_sha256':experiment.file_sha(parent/'status.json'),'source_auditor_sha256':'c'*64}
            for index,(key,value) in enumerate((('status_sha256','f'*64),('complete',False),
                                                 ('source_auditor_sha256','f'*64))):
                bad={**base,key:value};(parent/'final-audit.json').write_text(json.dumps(bad))
                with self.subTest(corruption=key), \
                     patch.object(experiment,'public_deployment',return_value={'phase':'mixed_evaluation','deadline_unix':time.time()+60}), \
                     patch.object(experiment,'sources',return_value={}), \
                     patch.object(experiment,'verify_run',return_value=(parent_manifest,status)):
                    with self.assertRaisesRegex(ValueError,'Independent complete predecessor audit'):
                        experiment.prepare(folder/f'evaluation-{index}',deployment,metadata,identity_file,predecessor=parent)

    def test_zero_temporary_adam_step_cannot_reach_production_timing_or_updates(self):
        class Future:
            def result(self):return {'elapsed_seconds':1.}
        class Client:
            all=[]
            def __init__(self,*args,**kwargs):self.stopped=False;Client.all.append(self)
            def ready(self):return {'base_identity':{'fixture':True}}
            def call(self,method,data):
                if method=='publish_base':return {'binding':{'version':0}}
                if method=='profile':return {'passed':True,'temporary_optimizer_steps':0}
                return {'passed':True}
            def submit(self,*args):return Future()
            def stop(self):self.stopped=True
        class Monitor:
            def start(self):return self
            def stop(self):return {'samples':[{'fixture':True}]}
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);(folder/'inputs').mkdir();(folder/'manifest.json').write_text('{}')
            (folder/'inputs/model-file-sha256.json').write_text('{}')
            (folder/'inputs/schedule.json').write_text(json.dumps({'calibration-1':[],'calibration-2':[]}))
            manifest={'phase':'training','config':{'policy':{},'deadline_unix':time.time()+3600}}
            with patch.object(experiment,'verify_run',return_value=(manifest,{'state':'prepared'})), \
                 patch.object(experiment.importlib.metadata,'version',side_effect=experiment.VERSIONS.__getitem__), \
                 patch.object(experiment,'index_model',return_value={'files':{}}), \
                 patch('selfplay.arena_mixed.processes.ProcessClient',Client), \
                 patch.object(experiment,'compare_actor_reports',return_value={'passed':True}), \
                 patch('selfplay.arena_mixed.processes.GPUMonitor',Monitor), \
                 patch.object(experiment,'select_endpoint') as select, \
                 patch.object(experiment,'run_pipeline') as production:
                with self.assertRaisesRegex(ValueError,'real-update profile'):
                    experiment.run_experiment(folder,folder/'no-model','training')
                select.assert_not_called();production.assert_not_called()
            self.assertTrue(all(c.stopped for c in Client.all))
            self.assertEqual(json.loads((folder/'status.json').read_text())['state'],'failed')


class InterruptedOptimizerEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
        self.journal=Journal(self.folder,'a'*64,time.time()+30)

    def test_durable_rpc_intent_without_correct_success_is_unknown(self):
        self.journal.intent('processes/learner-000001-intent.json','mixed_rpc',
            {'request_id':'learner-000001','method':'update','data_sha256':'b'*64})
        self.journal.intent('processes/actor-000001-intent.json','mixed_rpc',
            {'request_id':'actor-000001','method':'collect','data_sha256':'b'*64})
        pending=uncommitted_operations(self.folder)
        self.assertEqual(len(pending),1)
        self.assertIn('unknown',pending[0]['outcome'])
        self.assertIsNone(pending[0]['result_sha256'])
        self.journal.write('processes/learner-000001-result.json',{'id':'wrong','passed':True})
        pending=uncommitted_operations(self.folder)
        self.assertEqual(len(pending),1)
        self.assertEqual(pending[0]['result_sha256'],experiment.file_sha(self.folder/'processes/learner-000001-result.json'))

    def test_only_exact_successful_rpc_receipt_clears_its_ambiguity(self):
        self.journal.intent('processes/learner-000002-intent.json','mixed_rpc',
            {'request_id':'learner-000002','method':'profile','data_sha256':'b'*64})
        self.journal.write('processes/learner-000002-result.json',
            {'id':'learner-000002','passed':True,'result':{'passed':True}})
        self.assertEqual(uncommitted_operations(self.folder),[])

    def test_optimizer_result_without_publication_keeps_failure_evidence(self):
        self.journal.intent('cohorts/4/optimizer-intent.json','mixed_optimizer_update',{})
        self.journal.write('cohorts/4/optimizer-result.json',{'optimizer_steps':1})
        self.journal.write('processes/learner-failure.json',
            {'neural_failure':{'optimizer_step_attempted':True,'optimizer_steps':None}})
        pending=uncommitted_operations(self.folder)
        self.assertEqual(len(pending),1)
        self.assertEqual(pending[0]['kind'],'optimizer_update')
        self.assertIn('unknown',pending[0]['outcome'])
        self.assertEqual(set(pending[0]['preserved_evidence_sha256']),
            {'cohorts/4/optimizer-result.json','processes/learner-failure.json'})
        self.journal.write('cohorts/4/pipeline-commit.json',{'fixture':'publication checked by full audit'})
        self.assertEqual(uncommitted_operations(self.folder),[])


if __name__ == '__main__': unittest.main()
