import json
from pathlib import Path
import tempfile
import time
import unittest

from selfplay.arena_mixed.processes import ProcessClient, checkpoint_reference
from selfplay.arena_mixed.runner import Journal
from selfplay.arena_mixed.experiment import select_endpoint


def fake_worker(connection, role, model_dir, config, folder, run_sha256, deadline):
    connection.send({'ready':True,'runtime':{'fake':True},'base_identity':{'pin':'test'}})
    while True:
        command=connection.recv()
        if command['method']=='stop':return
        if command['method']=='die':return
        if command['method']=='wrong':
            connection.send({'id':'wrong','passed':True,'result':1})
        else:
            connection.send({'id':command['id'],'passed':True,'result':command['data']})


class ProcessContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.journal=Journal(self.tmp.name,'a'*64,time.time()+30)
        self.client=ProcessClient('actor','unused',{},self.journal,target=fake_worker)
        self.client.ready()

    def tearDown(self):
        self.client.stop();self.tmp.cleanup()

    def test_queue_capacity_one_and_no_duplicate_future_consumption(self):
        first=self.client.request('echo',{'version':2})
        with self.assertRaises(ValueError):self.client.request('echo',{'version':3})
        self.assertEqual(first.result(),{'version':2})
        with self.assertRaises(ValueError):first.result()
        self.assertEqual(self.client.call('echo',{'version':3}),{'version':3})
        self.assertTrue((Path(self.tmp.name)/'processes/actor-000001-intent.json').is_file())

    def test_unbound_result_remains_poisoned_and_preserved(self):
        with self.assertRaises(RuntimeError):self.client.call('wrong',{})
        self.assertIsNotNone(self.client.pending)
        with self.assertRaises(ValueError):self.client.request('echo',{})
        saved=json.loads((Path(self.tmp.name)/'processes/actor-000000-result.json').read_text())
        self.assertEqual(saved['id'],'wrong')

    def test_child_death_cannot_become_success_or_automatic_retry(self):
        with self.assertRaises((RuntimeError,EOFError)):self.client.call('die',{})
        self.assertIsNotNone(self.client.pending)


class AdmissionTests(unittest.TestCase):
    def test_fixed_endpoint_uses_full_cost_and_export_reserve(self):
        self.assertEqual(select_endpoint([100,120],9000)['endpoint'],40)
        self.assertEqual(select_endpoint([100,120],5400)['endpoint'],20)
        self.assertIsNone(select_endpoint([100,120],5399)['endpoint'])
        self.assertEqual(select_endpoint([120,100],5400)['endpoint'],20)

    def test_invalid_profile_never_admits_training(self):
        for costs in ([],[0],[float('nan')],[-1]):
            with self.assertRaises(ValueError):select_endpoint(costs,999999)

    def test_checkpoint_reference_keeps_actual_adam_distinct_from_version(self):
        state={'files':{'adapter.safetensors':'a'},'step':3,'policy':{'binding':{'version':5}}}
        ref=checkpoint_reference('/tmp/test',state)
        self.assertEqual(ref['optimizer_step'],3)
        self.assertEqual(ref['binding']['version'],5)


if __name__=='__main__':unittest.main()
