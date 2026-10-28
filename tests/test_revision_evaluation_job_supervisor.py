"""CPU-only evaluation lifecycle tests; no network, SSH, model or provider calls."""
import json
from pathlib import Path
import signal
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from scripts.cloud import run_revision_evaluation_job as job
from scripts.cloud import bootstrap_revision_evaluation as bootstrap
from scripts import revision_evaluation_supervisor as supervisor


def write(path,value):path.write_text(json.dumps(value))


class BootstrapTests(unittest.TestCase):
    def test_routing_and_global_python_rejected_before_network(self):
        self.assertEqual(bootstrap.protocol().__name__,'selfplay.program_debug_revision_evaluation')
        with patch.object(bootstrap.sys,'prefix','/system'),patch.object(bootstrap.sys,'base_prefix','/system'), \
                patch.object(bootstrap,'admission'),patch.object(bootstrap.subprocess,'run') as run:
            with self.assertRaisesRegex(ValueError,'virtual environment'):bootstrap.stage(SimpleNamespace(),{'config':{}})
            run.assert_not_called()


class JobTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.run=Path(self.temporary.name);write(self.run/'manifest.json',{'fixture':True})
        write(self.run/'deployment.json',{'deadline_unix':10**10,'phase':'coding_evaluation'})
        self.expected=job.digest(self.run/'manifest.json')
        self.args=SimpleNamespace(run=self.run,model_dir=self.run/'models',expected_manifest_sha256=self.expected)
    def verifier(self,path):return {'phase':'coding_evaluation'},{'state':'prepared','manifest_sha256':self.expected}

    def test_exact_evaluation_route_and_no_second_job_attempt(self):
        child=MagicMock(returncode=0)
        with patch.object(job.subprocess,'Popen',return_value=child) as popen:
            self.assertEqual(job.execute(self.args,self.verifier),0)
            commands=[c.args[0] for c in popen.call_args_list]
            self.assertEqual(len(commands),2)
            self.assertIn('scripts/cloud/bootstrap_revision_evaluation.py',commands[0])
            self.assertIn('selfplay.program_debug_revision_evaluation',commands[1])
            self.assertNotIn('--phase',commands[1])
        receipt=json.loads((self.run/'remote-exit.json').read_text())
        self.assertEqual(receipt['phase'],'coding_evaluation')
        self.assertEqual(receipt['start_sha256'],job.digest(self.run/'remote-job-start.json'))
        before=(self.run/'remote-exit.json').read_bytes()
        with self.assertRaisesRegex(RuntimeError,'already attempted'):job.execute(self.args,self.verifier)
        self.assertEqual(before,(self.run/'remote-exit.json').read_bytes())

    def test_training_phase_and_mismatched_manifest_rejected_before_any_job(self):
        write(self.run/'deployment.json',{'deadline_unix':10**10,'phase':'mixed_training'})
        with patch.object(job.subprocess,'Popen') as popen:
            self.assertEqual(job.execute(self.args,self.verifier),1);popen.assert_not_called()
        row=json.loads((self.run/'remote-launch-failure.json').read_text())
        self.assertFalse(row['verification_completed'])

    def test_timeout_preserves_receipt_and_interrupts_before_kill(self):
        child=MagicMock(pid=123,returncode=None);child.poll.return_value=None
        child.wait.side_effect=[subprocess.TimeoutExpired('fixture',1),0]
        with patch.object(job.subprocess,'Popen',return_value=child),patch.object(job.os,'killpg') as kill:
            self.assertEqual(job.execute(self.args,self.verifier),1)
        kill.assert_called_once_with(123,signal.SIGINT)
        row=json.loads((self.run/'remote-exit.json').read_text())
        self.assertEqual(row['child_cleanup'],'sigint_parent_cleanup_completed')


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        root=Path(self.temporary.name);self.run=root/'run';self.deployment=root/'deployment'
        self.run.mkdir();self.deployment.mkdir()
        self.public={'intent':'fixture','phase':'coding_evaluation','deadline_unix':1000}
        for p in (self.run/'deployment.json',self.deployment/'plan.json'):write(p,self.public)
        write(self.run/'manifest.json',{'fixture':True});self.expected=job.digest(self.run/'manifest.json')
        self.args=SimpleNamespace(run=self.run,deployment=self.deployment,remote_run='/fixture',host='fixture',port=22,user='root')
        p=patch.object(supervisor,'public_deployment',side_effect=lambda x:x);p.start();self.addCleanup(p.stop)
        p=patch.object(supervisor,'verify_run',return_value=({'phase':'coding_evaluation'},
            {'manifest_sha256':self.expected,'state':'complete','artifacts':{}}))
        self.verifier=p.start();self.addCleanup(p.stop)
        self.origin=supervisor.capture_origin(self.args)

    def receipt(self):
        write(self.run/'remote-job-start.json',{'manifest_sha256':self.expected,'attempt_id':'fixture'})
        write(self.run/'remote-exit.json',{'manifest_sha256':self.expected,'attempt_id':'fixture',
            'start_sha256':job.digest(self.run/'remote-job-start.json'),'exit_code':0,
            'verification_completed':True,'observed_manifest_sha256':job.digest(self.run/'manifest.json'),
            'kind':'cardinal_gpu_job_exit','phase':'coding_evaluation'})

    def test_completed_evaluation_and_terminal_failures_allow_bound_cleanup(self):
        self.receipt()
        for state in ('complete','budget_incomplete','numerical_failed','stopped','failed'):
            self.verifier.return_value=({'phase':'coding_evaluation'},
                {'manifest_sha256':self.expected,'state':state,'artifacts':{}})
            result=supervisor.verify_receipt(self.args,self.origin)
            self.assertTrue(result['scientific_artifacts_verified']);self.assertEqual(result['study_state'],state)

    def test_training_or_active_state_cannot_become_successful_evaluation(self):
        self.receipt()
        for state in ('running','training_complete'):
            self.verifier.return_value=({'phase':'coding_evaluation'},
                {'manifest_sha256':self.expected,'state':state,'artifacts':{}})
            with self.assertRaisesRegex(ValueError,'terminal'):supervisor.verify_receipt(self.args,self.origin)
        self.verifier.return_value=({'phase':'training'},
            {'manifest_sha256':self.expected,'state':'complete','artifacts':{}})
        with self.assertRaisesRegex(ValueError,'phase'):supervisor.verify_receipt(self.args,self.origin)

    def test_replaced_manifest_or_foreign_deployment_rejected(self):
        self.receipt();write(self.run/'deployment.json',{'intent':'foreign'})
        with self.assertRaisesRegex(ValueError,'another deployment'):supervisor.verify_receipt(self.args,self.origin)
        write(self.run/'deployment.json',self.public);write(self.run/'manifest.json',{'foreign':True});self.receipt()
        with self.assertRaisesRegex(ValueError,'replaced prepared'):supervisor.verify_receipt(self.args,self.origin)
        self.assertEqual(supervisor.capture_origin(self.args),self.origin)


if __name__=='__main__':unittest.main()
