"""CPU-only staging/supervision fixtures; no SSH, cloud calls or model loads."""
import json
from datetime import datetime, timezone
import time
from pathlib import Path
import signal
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from scripts.cloud import run_planning_curriculum_job as job
from scripts import planning_curriculum_supervisor as supervisor
from scripts.cloud import bootstrap_planning_curriculum as bootstrap


def deployment():
    return {'phase':'planning_curriculum','deadline_unix':time.time()+43200,
        'created_at':datetime.now(timezone.utc).isoformat(),
        'gpu_hourly_rate':5.29,'guard_hourly_rate':.14,'storage_hourly_rate':170*.1/(28*24),
        'other_reserved_cost_usd':0,'stage_caps_usd':{'admission':15.,'comparison':65.,'protected':10.}}


def write(path,value):
    path.write_text(json.dumps(value))


class BootstrapTests(unittest.TestCase):
    def test_system_python_rejected_before_pip_or_model_download(self):
        with patch.object(bootstrap.sys, 'prefix', '/system'), patch.object(bootstrap.sys, 'base_prefix', '/system'), \
                patch.object(bootstrap, 'admission'), patch.object(bootstrap, 'stage_deadline'), patch.object(bootstrap.json, 'loads'), \
                patch.object(Path, 'read_text', return_value='{}'), patch.object(bootstrap.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'virtual environment'):
                bootstrap.stage(SimpleNamespace(run=Path('/fixture')), {'config': {}})
            run.assert_not_called()

    def test_deadline_and_exact_download_filename_inventory(self):
        with patch.object(bootstrap.time,'time',return_value=100):
            bootstrap.admission({'deadline_unix':101})
            for deadline in (99,float('nan'),float('inf'),True):
                with self.assertRaises(TimeoutError):bootstrap.admission({'deadline_unix':deadline})
        names=['config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json']
        metadata={'model':'Qwen/Qwen3-8B','revision':'frozen','files':[{'name':n} for n in names]}
        self.assertEqual(bootstrap.approved_files(metadata,metadata['model'],'frozen'),names)
        metadata['files'].append({'name':'../unapproved'})
        with self.assertRaises(ValueError):bootstrap.approved_files(metadata,metadata['model'],'frozen')


    def test_bootstrap_meter_includes_guard_disk_and_elapsed_preparation(self):
        p=deployment();created=datetime.fromisoformat(p['created_at']).timestamp()
        expected=created+15/(5.29+.14+170*.1/(28*24))*3600
        self.assertAlmostEqual(bootstrap.stage_deadline(p),expected)
        p['other_reserved_cost_usd']=3
        self.assertAlmostEqual(bootstrap.stage_deadline(p),created+12/(5.29+.14+170*.1/(28*24))*3600)
        for value in (float('nan'),True,-1):
            p['other_reserved_cost_usd']=value
            with self.assertRaises(ValueError):bootstrap.stage_deadline(p)


class JobTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.run=Path(self.temp.name);write(self.run/'manifest.json',{'fixture':True})
        write(self.run/'deployment.json',deployment())
        self.expected=job.digest(self.run/'manifest.json')
        self.args=SimpleNamespace(run=self.run,model_dir=self.run/'models',expected_manifest_sha256=self.expected)
    def verifier(self,path):return {},{'state':'prepared','manifest_sha256':self.expected}

    def test_success_is_one_attempt_and_receipt_binds_start(self):
        child=MagicMock(returncode=0)
        with patch.object(job.subprocess,'Popen',return_value=child) as popen:
            self.assertEqual(job.execute(self.args,self.verifier),0)
            self.assertEqual(popen.call_count,2)
            for call in popen.call_args_list:
                self.assertEqual(call.kwargs['env']['PYTHONDONTWRITEBYTECODE'],'1')
                self.assertEqual(call.kwargs['env']['PIP_NO_CACHE_DIR'],'1')
            commands=[call.args[0] for call in popen.call_args_list]
            self.assertIn("scripts/cloud/bootstrap_planning_curriculum.py",commands[0])
            self.assertIn("selfplay.planning_curriculum.experiment",commands[1])
            self.assertNotIn('--phase',commands[1])
            self.assertLess(child.wait.call_args_list[0].kwargs['timeout'],3*3600)
            self.assertGreater(child.wait.call_args_list[1].kwargs['timeout'],10*3600)
        receipt=json.loads((self.run/'remote-exit.json').read_text())
        self.assertEqual(receipt['start_sha256'],job.digest(self.run/'remote-job-start.json'))
        self.assertTrue(receipt['verification_completed'])
        original=(self.run/'remote-exit.json').read_bytes()
        with self.assertRaisesRegex(RuntimeError,'already attempted'):job.execute(self.args,self.verifier)
        self.assertEqual((self.run/'remote-exit.json').read_bytes(),original)
        self.assertFalse((self.run/'remote-launch-failure.json').exists())

    def test_early_verification_failure_has_separate_non_success_receipt(self):
        with patch.object(job.subprocess,'Popen') as popen:
            self.assertEqual(job.execute(self.args,MagicMock(side_effect=ValueError('fixture failure'))),1)
            popen.assert_not_called()
        receipt=json.loads((self.run/'remote-launch-failure.json').read_text())
        self.assertFalse(receipt['verification_completed']);self.assertEqual(receipt['manifest_sha256'],self.expected)
        self.assertFalse((self.run/'remote-exit.json').exists())

    def test_changed_manifest_fails_before_import_or_staging(self):
        write(self.run/'manifest.json',{'different':True})
        verifier=MagicMock()
        self.assertEqual(job.execute(self.args,verifier),1);verifier.assert_not_called()
        receipt=json.loads((self.run/'remote-launch-failure.json').read_text())
        self.assertNotEqual(receipt['observed_manifest_sha256'],receipt['manifest_sha256'])

    def test_wrong_phase_rejected_before_staging(self):
        write(self.run/'deployment.json',{'deadline_unix':10**10,'phase':'coding_evaluation'})
        with patch.object(job.subprocess,'Popen') as popen:
            self.assertEqual(job.execute(self.args,self.verifier),1);popen.assert_not_called()
        self.assertFalse(json.loads((self.run/'remote-launch-failure.json').read_text())['verification_completed'])

    def test_timeout_interrupts_phase_parent_before_any_kill(self):
        child=MagicMock(pid=123,returncode=None)
        child.poll.return_value=None
        child.wait.side_effect=[subprocess.TimeoutExpired('fixture',1),0]
        with patch.object(job.subprocess,'Popen',return_value=child),patch.object(job.os,'killpg') as kill:
            self.assertEqual(job.execute(self.args,self.verifier),1)
        kill.assert_called_once_with(123,signal.SIGINT)
        receipt=json.loads((self.run/'remote-exit.json').read_text())
        self.assertEqual(receipt['child_cleanup'],'sigint_parent_cleanup_completed')

    def test_kill_only_after_interrupt_grace_and_no_receipt_overwrite(self):
        child=MagicMock(pid=123);child.poll.return_value=None
        child.wait.side_effect=[subprocess.TimeoutExpired('fixture',30),0]
        with patch.object(job.os,'killpg') as kill:
            self.assertEqual(job.stop_child(child),'forced_kill_external_guard_still_required')
        self.assertEqual([c.args for c in kill.call_args_list],[(123,signal.SIGINT),(123,signal.SIGKILL)])
        target=self.run/'one.json';job.exclusive_json(target,{'x':1})
        with self.assertRaises(FileExistsError):job.exclusive_json(target,{'x':2})
        self.assertEqual(json.loads(target.read_text()),{'x':1})


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name);self.run=root/'run';self.deployment=root/'deployment'
        self.run.mkdir();self.deployment.mkdir()
        self.public={'intent':'fixture','deadline_unix':1000,'phase':'planning_curriculum'}
        write(self.deployment/'plan.json',self.public);write(self.run/'deployment.json',self.public)
        write(self.run/'manifest.json',{'fixture':True});self.expected=job.digest(self.run/'manifest.json')
        self.args=SimpleNamespace(run=self.run,deployment=self.deployment,remote_run='/workspace/fixture',host='fixture',port=22,user='root')
        self.p1=patch.object(supervisor,'public_deployment',side_effect=lambda x:x);self.p1.start();self.addCleanup(self.p1.stop)
        self.p2=patch.object(supervisor,'verify_run',return_value=({'phase':'evaluation'}, {'manifest_sha256':self.expected,'state':'complete','artifacts':{}}))
        self.verifier=self.p2.start();self.addCleanup(self.p2.stop)
        self.origin=supervisor.capture_origin(self.args)

    def receipt(self,validated=True):
        start={'manifest_sha256':self.expected,'attempt_id':'fixed-attempt'}
        write(self.run/'remote-job-start.json',start)
        receipt={'manifest_sha256':self.expected,'attempt_id':'fixed-attempt',
            'start_sha256':job.digest(self.run/'remote-job-start.json'),'exit_code':0 if validated else 1,
            'verification_completed':validated,'observed_manifest_sha256':job.digest(self.run/'manifest.json'),
            'kind':'planning_curriculum_job_exit' if validated else 'planning_curriculum_launch_failure','phase':'planning_curriculum'}
        write(self.run/('remote-exit.json' if validated else 'remote-launch-failure.json'),receipt)

    def test_origin_is_persisted_before_first_pull_and_normal_exit_verified(self):
        self.assertEqual(supervisor.load(self.deployment/'supervisor-origin.json'),self.origin)
        self.receipt();result=supervisor.verify_receipt(self.args,self.origin)
        self.assertTrue(result['scientific_artifacts_verified'])
        self.assertEqual(result['study_state'],'complete')

    def test_zero_exit_requires_terminal_state(self):
        self.receipt()
        self.verifier.return_value=({'phase':'evaluation'}, {'manifest_sha256':self.expected,'state':'running','artifacts':{}})
        with self.assertRaisesRegex(ValueError,'terminal'):supervisor.verify_receipt(self.args,self.origin)
        for state in ('complete','admission_signal_failed','admission_timing_failed','failed'):
            self.verifier.return_value=({'phase':'evaluation'}, {'manifest_sha256':self.expected,'state':state,'artifacts':{}})
            self.assertEqual(supervisor.verify_receipt(self.args,self.origin)['study_state'],state)

    def test_wrong_deployment_or_replaced_expected_manifest_rejected(self):
        self.receipt();write(self.run/'deployment.json',{'intent':'other','deadline_unix':1000})
        with self.assertRaisesRegex(ValueError,'another deployment'):supervisor.verify_receipt(self.args,self.origin)
        write(self.run/'deployment.json',self.public);write(self.run/'manifest.json',{'wrong':True});self.receipt()
        with self.assertRaisesRegex(ValueError,'replaced prepared'):supervisor.verify_receipt(self.args,self.origin)
        self.assertEqual(supervisor.capture_origin(self.args),self.origin)  # Restart retains original expected hash.

    def test_early_failure_preserves_failed_validation_without_claiming_success(self):
        write(self.run/'manifest.json',{'bad launch source':True});self.receipt(False)
        self.verifier.reset_mock()
        result=supervisor.verify_receipt(self.args,self.origin)
        self.assertFalse(result['scientific_artifacts_verified']);self.assertEqual(result['verified_artifacts'],0)
        self.verifier.assert_not_called()

    def test_guard_cleanup_retry_does_not_pull_deleted_gpu(self):
        self.receipt()
        instance=MagicMock()
        instance.plan=self.public
        instance.state={'roles':{'gpu':{'state':'identified'},'guard':{'state':'identified'}}}
        instance.gpu_resolved.side_effect=lambda:instance.state['roles']['gpu']['state']=='terminated_confirmed'
        failures=[True]
        def delete(role):
            if role=='guard' and failures:
                failures.pop();raise RuntimeError('transient provider failure')
            instance.state['roles'][role]['state']='terminated_confirmed'
        instance.delete.side_effect=delete
        with patch.object(supervisor,'Deployment',return_value=instance),patch.object(supervisor,'pull',return_value=0) as pull:
            with self.assertRaisesRegex(RuntimeError,'transient'):
                supervisor.supervisor_step(self.args,self.origin)
            self.assertTrue((self.deployment/'supervisor-cleanup-intent.json').exists())
            self.assertEqual(instance.state['roles']['gpu']['state'],'terminated_confirmed')
            result=supervisor.supervisor_step(self.args,self.origin)
            self.assertEqual(pull.call_count,1)
            self.assertTrue(result['gpu_404_confirmed']);self.assertTrue(result['guard_404_confirmed'])
            self.assertTrue(instance.state['creation_closed'])
            instance.guard_health.assert_not_called()

    def test_tampered_start_or_conflicting_receipts_rejected(self):
        self.receipt();write(self.run/'remote-job-start.json',{'manifest_sha256':self.expected,'attempt_id':'other'})
        with self.assertRaisesRegex(ValueError,'receipt/start'):supervisor.verify_receipt(self.args,self.origin)
        self.receipt(False)
        with self.assertRaisesRegex(ValueError,'Conflicting'):supervisor.verify_receipt(self.args,self.origin)


if __name__=='__main__':unittest.main()
