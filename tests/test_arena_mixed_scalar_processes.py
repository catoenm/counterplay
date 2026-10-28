"""Scalar process contracts without CUDA initialization or model calls."""
from pathlib import Path
import json
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from selfplay.arena_mixed_scalar import processes
from selfplay.arena_mixed_scalar.runner import Journal
from tests.test_arena_mixed_processes import fake_worker


class ScalarProcessTests(unittest.TestCase):
    def test_diagnostic_dispatch_keeps_exact_fixture(self):
        fixture={'kind':'frozen fixture'};seen=[]
        policy=SimpleNamespace(execution_diagnostic=lambda value:seen.append(value) or {'passed':True})
        result=processes._dispatch(policy,{'method':'execution_diagnostic','data':{'fixture':fixture}},None)
        self.assertEqual(seen,[fixture]);self.assertEqual(result,{'passed':True})

    def test_failed_full_gate_is_durable_and_never_enters_optimizer_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal=Journal(tmp,'a'*64,time.time()+30)
            checkpoint={'binding':{'version':0},'folder':'frozen','files':{'a':'b'}}
            calls=[];gate={'passed':False,'records':[{'passed':False}]}
            policy=SimpleNamespace(role='learner',binding=checkpoint['binding'],
                gate_at_checkpoint=lambda *args:gate,
                profile_cohort=lambda *args,**kwargs:calls.append('profile'))
            with patch.object(processes,'load_sealed_cohort',return_value=({'bound':True},[{'transitions':[{}]}])):
                with self.assertRaisesRegex(ValueError,'Full behavior likelihood gate failed'):
                    processes._dispatch(policy,{'method':'profile','data':{'cohort':'unused','current':checkpoint,
                        'behavior':checkpoint,'prefix':'calibration/1'}},journal)
            self.assertEqual(calls,[])
            self.assertTrue((Path(tmp)/'calibration/1/gate-intent.json').is_file())
            self.assertEqual(json.loads((Path(tmp)/'calibration/1/gate.json').read_text()),gate)
            self.assertFalse((Path(tmp)/'calibration/1/optimizer-intent.json').exists())

    def test_queue_one_result_binding_and_no_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal=Journal(tmp,'a'*64,time.time()+30)
            client=processes.ProcessClient('actor','unused',{},journal,target=fake_worker)
            try:
                client.ready();future=client.request('echo',{'scalar':True})
                with self.assertRaises(ValueError):client.request('echo',{})
                self.assertEqual(future.result(),{'scalar':True})
                with self.assertRaises(ValueError):future.result()
                with self.assertRaises(RuntimeError):client.call('wrong',{})
                with self.assertRaises(ValueError):client.request('echo',{})
                self.assertIsNotNone(client.pending)
                self.assertTrue((Path(tmp)/'processes/actor-000001-result.json').is_file())
            finally:client.stop()

    def test_scalar_worker_import_and_new_runner_are_separate(self):
        self.assertEqual(processes.worker.__module__,'selfplay.arena_mixed_scalar.processes')
        self.assertEqual(processes.Journal.__module__,'selfplay.arena_mixed_scalar.runner')
        self.assertEqual(processes.collect_cohort.__module__,'selfplay.arena_mixed_scalar.runner')
        self.assertEqual(processes.load_sealed_cohort.__module__,'selfplay.arena_mixed_scalar.runner')

    def test_broken_stop_pipe_still_terminates_joins_and_closes(self):
        for error in (BrokenPipeError, EOFError, OSError):
            with self.subTest(error=error):
                client=object.__new__(processes.ProcessClient)
                client.role='actor';client.pending=None
                client.connection=Mock();client.connection.send.side_effect=error('closed')
                client.process=Mock();alive=[True]
                client.process.is_alive.side_effect=lambda:alive[0]
                client.process.terminate.side_effect=lambda:alive.__setitem__(0,False)
                client.stop()
                client.process.terminate.assert_called_once_with()
                client.process.kill.assert_not_called()
                self.assertEqual(client.process.join.call_args_list[0].kwargs,{'timeout':10})
                client.connection.close.assert_called_once_with()

    def test_stop_kills_stuck_worker_and_refuses_live_artifact_snapshot(self):
        client=object.__new__(processes.ProcessClient)
        client.role='learner';client.pending='unresolved'
        client.connection=Mock();client.process=Mock();client.process.is_alive.return_value=True
        with self.assertRaisesRegex(RuntimeError,'survived bounded shutdown'):client.stop()
        client.connection.send.assert_not_called()
        client.process.terminate.assert_called_once_with();client.process.kill.assert_called_once_with()
        self.assertTrue(all(call.kwargs['timeout']<=10 for call in client.process.join.call_args_list))
        client.connection.close.assert_called_once_with()


if __name__=='__main__':unittest.main()
