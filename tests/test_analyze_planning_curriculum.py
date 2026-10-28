"""Independent audit fixtures: real tiny Torch/Adam, no pretrained model or GPU."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from scripts import analyze_planning_curriculum as audit
from selfplay.planning_curriculum import cuda_policy as policy
from selfplay.planning_curriculum import schedule
from tests.test_planning_curriculum_cuda_policy import fixture, episodes


class AuditTensorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.cuda_rng=[torch.get_rng_state().clone()]
        for name,fn in [('get_rng_state_all',lambda:[x.clone() for x in self.cuda_rng]),
            ('set_rng_state_all',lambda rows:self.cuda_rng.__setitem__(slice(None),[x.clone() for x in rows]))]:
            p=patch.object(torch.cuda,name,side_effect=fn);p.start();self.addCleanup(p.stop)
        self.obj=fixture();self.base_folder=self.root/'base';self.obj.save_checkpoint(self.base_folder)
        self.base=policy.checkpoint_state(self.base_folder);self.data=episodes(self.obj)
    def probe(self):
        self.target=self.root/'probe'
        return self.obj.timing_probe(self.data,[e['episode_id'] for e in self.data],self.target)
    def check(self,report):
        return audit.check_probe(report,self.data,self.base,self.obj.config,self.base_folder,self.target)
    def test_real_temporary_adam_and_unpublished_mature_copy(self):
        report=self.probe();self.assertTrue(report['passed'],report)
        result=self.check(report)
        self.assertEqual(result['retained_steps'],0);self.assertEqual(result['optimizer_steps'],1)
        self.assertGreater(result['parameter_delta_l2'],0)
        self.assertIsNone(report['mature_state_roundtrip']['state_after']['binding'])
        self.assertEqual(report['mature_state_roundtrip']['state_after']['pending_version'],1)
        self.assertEqual(self.obj._state_identity(),report['state_before'])
    def test_real_objective_vector_advantage_loss_and_binding_corruption(self):
        report=self.probe()
        edits=[lambda r:r['temporary_update']['learner_records'][0].__setitem__('advantage',0.),
            lambda r:r['temporary_update']['learner_records'][0].__setitem__('old_vector_sha256','0'*64),
            lambda r:r['temporary_update']['learner_records'][0]['source_binding'].__setitem__('version',3),
            lambda r:r['temporary_update'].__setitem__('loss',r['temporary_update']['loss']+.01),
            lambda r:r['temporary_update']['learner_records'][0]['learner_log_probs'].__setitem__(0,-90.),
            lambda r:r['mature_state_roundtrip']['state_after'].__setitem__('pending_version',None)]
        for edit in edits:
            bad=deepcopy(report);edit(bad)
            with self.subTest(edit=edit),self.assertRaises((ValueError,KeyError)):self.check(bad)
    def test_actual_saved_optimizer_lr_and_weight_delta_rejected(self):
        report=self.probe();path=self.target/'optimizer.pt'
        saved=torch.load(path,weights_only=True);saved['optimizer']['param_groups'][0]['lr']=1e-5
        torch.save(saved,path)
        with self.assertRaises(ValueError):self.check(report)
    def test_ambiguous_step_keeps_failed_probe_and_exact_restoration(self):
        original=self.obj.optimizer.step
        def broken():original();raise RuntimeError('synthetic step-after-mutation failure')
        with patch.object(self.obj.optimizer,'step',side_effect=broken):report=self.probe()
        result=self.check(report)
        self.assertFalse(result['passed']);self.assertIsNone(result['temporary_step_outcome'])
        self.assertEqual(result['retained_steps'],0)
    def test_restored_ambiguous_probe_remains_unknown(self):
        from selfplay.planning_curriculum.runner import exclusive_json
        exclusive_json(self.root/'admission/probe-optimizer-intent.json',{'fixture':'attempt'})
        exclusive_json(self.root/'admission/probe.json',{'passed':False,'state_restored':True,'temporary_optimizer_steps':None})
        pending=audit.pending_operations(self.root)
        self.assertEqual(len(pending),1);self.assertIn('unknown temporary step',pending[0]['outcome'])
        self.assertTrue(pending[0]['state_restoration_reported'])

    def test_behavior_gate_restoration_must_be_actual_source_state(self):
        report=self.probe();state=deepcopy(report['state_before']);state['rng']['cpu']='0'*64
        with self.assertRaises(ValueError):audit.check_state(state,self.base,self.obj.config)


class ComparisonTests(unittest.TestCase):
    def rows(self):
        return {endpoint:[{'scenario_family_id':str(i),'depth':3+i//16,
            'returns':[float(i<({'base':8}.get(endpoint,12 if endpoint.endswith('control') else 20))),0.]}
            for i in range(64)] for endpoint in schedule.ENDPOINTS}
    def test_fixed_64_family_2_seed_bootstrap_and_operational_gate(self):
        result=audit.comparisons(self.rows())
        self.assertEqual(result['paired_families'],64);self.assertEqual(result['training_seed_replicates'],2)
        self.assertEqual(result['bootstrap_seed'],20261025)
        self.assertEqual(result['mean_curriculum_minus_control'],8/64)
        self.assertTrue(result['operational_followup_gate']['passed'])
        self.assertFalse(result['operational_followup_gate']['authorization_for_more_work'])
        self.assertEqual(result,audit.comparisons(self.rows()))
    def test_incomplete_or_repaired_family_cannot_score(self):
        bad=self.rows();bad.pop('base')
        with self.assertRaises(ValueError):audit.comparisons(bad)
        bad=self.rows();bad['742101-curriculum'][0]['scenario_family_id']='other'
        with self.assertRaises(ValueError):audit.comparisons(bad)


if __name__=='__main__':unittest.main()

class PrefixTests(AuditTensorTests):
    """Incomplete raw draws are inspected without assigning terminal rewards."""
    def test_raw_prefix_and_missing_result_are_preserved_without_scores(self):
        import time
        from selfplay.planning_curriculum import runner
        from selfplay.arena_mixed_scalar import cuda_policy as scalar
        stages=schedule.make_schedules([])[0];specs=stages['screen-hard'][:2]
        journal=runner.Journal(self.root,'d'*64,time.time()+60);calls=0
        def draw(requests,binding):
            nonlocal calls
            calls+=1
            if calls==2:raise TimeoutError('synthetic interrupted raw draw')
            return self.obj.choose(requests,binding)
        with self.assertRaises(TimeoutError):runner.collect_cohort(draw,specs,self.obj.binding,journal,'prefix')
        with patch.object(scalar,'select_labels',return_value=(self.obj.labels,self.obj.label_ids)):
            result=audit.check_unsealed_prefix(self.root/'prefix',specs,self.obj.binding,self.obj.config,self.obj.tokenizer,'d'*64,False)
        self.assertEqual(result['sampled_actions_verified'],2)
        self.assertEqual(result['pending_batch'],'000001');self.assertTrue(result['returns_not_scored'])
        self.assertNotIn('returns',result)
        path=self.root/'prefix/batches/000000-intent.json';saved=audit.load(path)
        saved['requests'][0]['observation']['turn']=999
        path.write_text(audit.canonical(saved))
        with patch.object(scalar,'select_labels',return_value=(self.obj.labels,self.obj.label_ids)),self.assertRaises(ValueError):
            audit.check_unsealed_prefix(self.root/'prefix',specs,self.obj.binding,self.obj.config,self.obj.tokenizer,'d'*64,False)
    def test_unresolved_optimizer_is_never_a_zero_step_claim(self):
        from selfplay.planning_curriculum.runner import exclusive_json
        path=self.root/'processes/learner-000003-intent.json'
        exclusive_json(path,{'method':'update','request_id':'learner-000003'})
        rows=audit.pending_operations(self.root)
        self.assertEqual(len(rows),1);self.assertIn('unknown',rows[0]['outcome'])
        exclusive_json(self.root/'arms/742101-control/cohorts/1/optimizer-intent.json',{'fixture':'attempted'})
        self.assertEqual(len(audit.pending_operations(self.root)),2)

class TerminalIntegrationTests(unittest.TestCase):
    """Real runner/replay files; neural math mocked here, separately tested above."""
    @classmethod
    def setUpClass(cls):
        from tests.test_planning_curriculum_experiment import PlanningExperimentTests
        PlanningExperimentTests.setUpClass()
    def fixture(self,signal_pass=True):
        from tests import test_planning_curriculum_experiment as integration
        case=integration.PlanningExperimentTests('test_failed_signal_stops_without_probe_or_training');case.setUp()
        self.addCleanup(case.doCleanups)
        case.manifest['sources']={'scripts/analyze_planning_curriculum.py':audit.file_sha(audit.__file__)}
        case.manifest['config'].update(policy=policy.default_config(),maximum_policy_episodes=544,attempted_retained_updates=80,max_policy_lag=1)
        integration.json_write(case.run/'manifest.json',case.manifest)
        integration.json_write(case.run/'inputs/prior-scenario-inventory.json',{'fingerprints':[]})
        integration.json_write(case.run/'inputs/schedule-verification.json',schedule.validate_schedules(case.stages,[]))
        parent_policy=integration.FakePolicy;parent_client=integration.FakeClient
        class EvidencePolicy(parent_policy):
            def timing_probe(self,*args):
                return {**super().timing_probe(*args),'state_restored':True,'temporary_optimizer_steps':1}
            def gate_at_checkpoint(self,rows,folder,files):
                return {**super().gate_at_checkpoint(rows,folder,files),'learner_state_before':{}}
            def train_cohort(self,rows,binding,gate):
                source=deepcopy(self.binding);result=super().train_cohort(rows,binding,gate)
                return {**result,'source':source,'behavior':binding,'policy_lag':source['version']-binding['version'],
                    'gate_sha256':audit.digest(gate),'cohort_sha256':audit.digest(rows)}
        class EvidenceClient(parent_client):
            def request(self,method,data):
                future=super().request(method,data);identifier=future.command['id']
                path=self.journal.folder/('processes/'+identifier+'-intent.json');row=audit.load(path)
                row.update(request_id=identifier,kind='mixed_rpc');integration.json_write(path,row)
                return future
        EvidenceClient.signal_pass=signal_pass
        self.integration_patches=[patch.object(integration,'FakePolicy',EvidencePolicy),patch.object(integration,'FakeClient',EvidenceClient)]
        for item in self.integration_patches:item.start();self.addCleanup(item.stop)
        return case
    def run_audit(self,case):
        def checkpoint(folder,*unused):
            folder=Path(folder);saved=audit.load(folder/'optimizer.pt');metadata=audit.load(folder/'policy.json')
            return {'files':{name:audit.file_sha(folder/name) for name in ('adapter.safetensors','optimizer.pt','policy.json')},
                'policy':{**metadata,'config':case.manifest['config']['policy']},'step':saved['step'],
                'moments':{} if not saved['step'] else {'fixture':True}}
        def transition(report,previous,current,*unused):
            if current['step']!=previous['step']+1:raise ValueError('Fixture step lineage differs')
            return {'optimizer_steps':1,'parameter_delta_l2':1.,'weights_changed':True}
        with patch.object(audit,'verify_run',return_value=(case.manifest,audit.load(case.run/'status.json'))), \
            patch.object(policy,'checkpoint_state',side_effect=checkpoint),patch.object(audit,'get_tokenizer',return_value=None), \
            patch.object(audit,'check_smoke',return_value={'passed':True}),patch.object(policy,'compare_actor_reports',return_value={'passed':True}), \
            patch.object(audit,'verify_action_rows',side_effect=lambda episodes,*_: {'actions':sum(len(e['transitions']) for e in episodes)}), \
            patch.object(audit,'check_probe',return_value={'passed':True,'retained_steps':0}), \
            patch.object(audit,'timing_gate',side_effect=lambda *args:audit.load(case.run/'admission/timing-gate.json')), \
            patch.object(audit,'check_gate_evidence',return_value={'passed':True}),patch.object(audit,'check_gate'), \
            patch.object(audit,'check_state'),patch.object(audit,'check_objective',return_value={'fixture_only':True}), \
            patch.object(audit,'check_transition',side_effect=transition):
            return audit.audit(case.run,require_receipt=False)
    def test_failed_signal_has_all64_replays_no_probe_no_comparisons(self):
        case=self.fixture(False);case.execute()
        result=self.run_audit(case)
        self.assertTrue(result['integrity_passed']);self.assertFalse(result['complete'])
        self.assertEqual(result['sealed_rollouts']['admission/screen']['episodes'],64)
        self.assertEqual(result['actual_retained_optimizer_steps'],0)
        self.assertFalse(result['comparisons_available'])
        self.assertEqual(result['rpc_journal']['verified_successful_rpcs'],5)
    def test_complete_all80_commits_and320_matched_outcomes_then_corruption(self):
        case=self.fixture();case.execute()
        result=self.run_audit(case)
        self.assertTrue(result['complete']);self.assertEqual(result['committed_training_cohorts'],80)
        self.assertEqual(result['actual_retained_optimizer_steps'],80);self.assertEqual(result['evaluation_episodes'],320)
        # A successful RPC cannot be detached from the committed update file.
        path=case.run/'processes/learner-000004-result.json';row=audit.load(path)
        row['result']['report']['optimizer_steps']=0;path.write_text(audit.canonical(row))
        from selfplay.planning_curriculum.experiment import commit_status
        commit_status(case.run,case.manifest,'complete','fixture-corruption')
        with self.assertRaises(ValueError):self.run_audit(case)
    def test_active_status_refuses_even_preserved_partial_evaluation(self):
        case=self.fixture(False);case.execute()
        from selfplay.planning_curriculum.experiment import commit_status
        commit_status(case.run,case.manifest,'running','evaluation_fixture')
        with self.assertRaisesRegex(ValueError,'Only prepared or terminal'):self.run_audit(case)
