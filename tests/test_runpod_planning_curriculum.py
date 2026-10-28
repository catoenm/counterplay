"""New lifecycle fixtures; only temporary files and fake HTTP are permitted."""
from copy import deepcopy
from datetime import datetime
import io
import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import MagicMock,patch
import urllib.error

from scripts import runpod_planning_curriculum as c
from tests.test_runpod_cardinal import FakeProvider


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);(self.root/'runs').mkdir();(self.root/'scripts/cloud').mkdir(parents=True)
        shutil.copyfile(c.ROOT/'scripts/cloud/cardinal_guard.py',self.root/'scripts/cloud/cardinal_guard.py')
        patcher=patch.object(c,'ROOT',self.root);patcher.start();self.addCleanup(patcher.stop)
        self.key=self.root/'ssh.pub';self.key.write_text('ssh-ed25519 fixture-public-only')
        self.folder=self.root/'new-lease'
        self.budget={'approved_pilot_cap':350.,'cloud_spend_to_date':0.,'cloud_reserved_usd':237.,
            'cloud_reservations':{'old-unresolved':{'amount_usd':237.,'state':'reserved','deployment':'untouched'}}}
        c.atomic_json(self.root/'budget.json',self.budget)
        for i in range(8):
            c.atomic_json(self.root/f'runs/fixture-{i}-ledger.json',{'requests':{'call':{
                'state':'interrupted_unknown','reserved_usd':.576147678/8}}})
        self.prepared=c.prepare(self.folder,self.key)
        self.provider=FakeProvider();self.provider.next_cost=.14
        self.d=c.Deployment(self.folder,self.provider)

    def test_first_write_fixed_clock_real_guard_no_funding_or_legacy_dispatch(self):
        self.assertTrue(self.prepared['actual_guard_constructor_passed'])
        self.assertEqual(self.provider.calls,[]);self.assertFalse(self.d.state['reservation_applied'])
        self.assertEqual(c.load(self.root/'budget.json'),self.budget)
        start=datetime.fromisoformat(self.d.plan['created_at']).timestamp()
        self.assertLessEqual(self.d.plan['deadline_unix']-start,43200)
        self.assertGreater(self.d.plan['deadline_unix']-start,43199)
        old=(self.folder/'plan.json').read_bytes()
        with self.assertRaises(FileExistsError):c.prepare(self.folder,self.key)
        self.assertEqual((self.folder/'plan.json').read_bytes(),old)
        with patch.object(c.primitives,'Deployment',side_effect=AssertionError('old controller forbidden')):
            c.Deployment(self.folder,self.provider).reserve()

    def test_single90_all_eight_ledgers_and_old_liability_unchanged(self):
        self.d.reserve();self.d.reserve()
        budget=c.load(self.root/'budget.json')
        self.assertEqual(budget['cloud_reserved_usd'],327)
        self.assertAlmostEqual(c.budget_total(budget),327.576147678)
        self.assertEqual(budget['cloud_reservations']['old-unresolved'],self.budget['cloud_reservations']['old-unresolved'])
        self.assertEqual(len(budget['cloud_reservations']),2)
        self.assertEqual(self.provider.calls,[])
        self.assertEqual(self.d.plan['stage_caps_usd'],{'admission':15.,'comparison':65.,'protected':10.})

    def test_lower_and_hard_caps_unknown_liability_fail_before_write(self):
        for cap,amount in [(300,237),(1000,261)]:
            b=deepcopy(self.budget);b['approved_pilot_cap']=cap;b['cloud_reserved_usd']=amount
            b['cloud_reservations']['old-unresolved']['amount_usd']=amount;c.atomic_json(self.root/'budget.json',b)
            before=(self.root/'budget.json').read_bytes()
            with self.assertRaisesRegex(ValueError,'cap'):self.d.reserve()
            self.assertEqual(before,(self.root/'budget.json').read_bytes())
        self.assertEqual(self.provider.calls,[])

    def test_source_guard_validation_precedes_any_prepare_writes(self):
        with patch.object(c,'Guard',side_effect=ValueError('actual constructor rejection')):
            with self.assertRaisesRegex(ValueError,'constructor'):c.prepare(self.root/'bad',self.key)
        self.assertFalse((self.root/'bad').exists())
        for key,value in [('deadline_unix',True),('maximum_gpu_rate_usd_hour',float('nan')),
                ('stage_caps_usd',{'admission':90}),('maximum_hours',24),('created_at','2026-10-25T00:00:00')]:
            p=deepcopy(self.d.plan);p[key]=value
            with self.assertRaises(ValueError):c.validate(p)

    def test_one_post_unknown_response_no_retry(self):
        self.d.reserve();self.provider.lose_create_response=True
        with self.assertRaisesRegex(RuntimeError,'no repeat'):self.d.create('guard')
        with self.assertRaisesRegex(ValueError,'already attempted'):self.d.create('guard')
        self.assertEqual(self.provider.calls.count(('POST','/pods')),1)
        self.assertEqual(self.d.state['roles']['guard']['state'],'create_unknown')
        self.d.reconcile();self.assertEqual(self.d.state['roles']['guard']['state'],'identified')
        self.assertNotIn(self.provider.key,(self.folder/'state.json').read_text())

    def test_guard_must_be_fresh_unique_and_rate_verified(self):
        self.d.reserve();self.d.create('guard')
        health={'ready':True,'guard_pod_id':'testpod','last_inventory_unix':time.time()}
        for mutate in ({'ready':False},{'last_inventory_unix':0},{'duplicate_intent_anomaly':True}):
            with patch.object(self.d,'guard_health',return_value={**health,**mutate}):
                with self.assertRaisesRegex(ValueError,'freshly armed'):self.d.create('gpu')
        self.assertEqual(self.provider.calls.count(('POST','/pods')),1)
        self.provider.rows[0]['cost']=None
        with patch.object(self.d,'guard_health',return_value=health):
            with self.assertRaisesRegex(ValueError,'GPU creation blocked'):self.d.create('gpu')
        self.assertEqual(self.d.state['roles']['guard']['state'],'terminated_confirmed')

    def test_duplicate_exact_ids_cleanup_only_after_get_and_confirm404(self):
        self.d.reserve();self.d.create('guard')
        original=deepcopy(self.provider.rows[0]);other=deepcopy(original);other['id']='duplicate'
        unrelated=deepcopy(original);unrelated['id']='unrelated';unrelated['name']='unrelated'
        self.provider.rows.extend([other,unrelated]);self.d.reconcile();self.d.cleanup_if_needed()
        self.assertTrue(self.d.state['roles']['guard']['duplicate_intent_anomaly'])
        self.assertEqual(set(self.d.state['roles']['guard']['confirmed_404_ids']),{'testpod','duplicate'})
        self.assertEqual([p['id'] for p in self.provider.rows],['unrelated'])
        deletes=[path for method,path in self.provider.calls if method=='DELETE']
        self.assertEqual(set(deletes),{'/pods/testpod','/pods/duplicate'})
        for path in deletes:
            at=self.provider.calls.index(('DELETE',path))
            self.assertEqual(self.provider.calls[at-1],('GET',path))
            self.assertEqual(self.provider.calls[at+1],('GET',path))

    def test_unknown_gpu_retains_guard_even_with_empty_inventory(self):
        self.d.reserve();self.d.create('guard')
        self.d.state['roles']['gpu']={'state':'create_unknown'};self.d.save()
        with self.assertRaisesRegex(ValueError,'protection'):self.d.delete('guard')
        self.d.cleanup_if_needed(force=True)
        self.assertEqual(self.d.state['roles']['guard']['state'],'identified')
        self.assertEqual(len(self.provider.rows),1)
        self.d.state['creation_closed']=True;self.d.save();self.d.reconcile()
        self.assertFalse(self.d.gpu_resolved())
        self.assertEqual(self.d.state['roles']['gpu']['state'],'create_unknown')
        with self.assertRaisesRegex(ValueError,'unresolved'):self.d.guard_health(finish=True)
        with self.assertRaisesRegex(ValueError,'protection'):self.d.delete('guard')

    def test_no_gpu_attempt_allows_exact_guard_cleanup(self):
        self.d.reserve();self.d.create('guard')
        self.assertTrue(self.d.gpu_resolved());self.d.delete('guard')
        self.assertEqual(self.d.state['roles']['guard']['state'],'terminated_confirmed')

    def test_overrate_or_unverified_created_resource_immediately_cleaned(self):
        self.d.reserve();self.provider.next_cost=float('nan')
        self.d.create('guard')
        self.assertEqual(self.d.state['roles']['guard']['state'],'terminated_confirmed')
        self.assertTrue(self.d.state['cleanup_started'])
        with self.assertRaisesRegex(ValueError,'closed|cleanup'):self.d.create('gpu')

    def test_reservation_tampering_blocks_create(self):
        self.d.reserve();b=c.load(self.root/'budget.json');b['cloud_reservations'][self.d.plan['intent']]['amount_usd']=0
        c.atomic_json(self.root/'budget.json',b)
        with self.assertRaisesRegex(ValueError,'reservation'):self.d.create('guard')
        self.assertEqual(self.provider.calls,[])

    def test_delayed_create_cannot_bypass_admission_budget_clock(self):
        self.d.reserve()
        created=datetime.fromisoformat(self.d.plan['created_at']).timestamp()
        with patch.object(c.time,'time',return_value=created+3*3600):
            with self.assertRaisesRegex(ValueError,'admission budget'):self.d.create('guard')
        self.assertEqual(self.provider.calls,[])

    def test_http_error_body_redacted_and_status_durable(self):
        self.d.reserve()
        token=c.load(self.d.plan['guard_secret_path'])['token'];api='rpa_private-api-secret'
        payload=json.dumps({'error':'capacity unavailable '+api+' '+token,'request':{'env':{'RUNPOD_API_KEY':api}},
                            'code':'NO_STOCK','message':'Provider has no available GPU'}).encode()
        opener=MagicMock();opener.open.side_effect=urllib.error.HTTPError('https://fixture',503,'unavailable',{},io.BytesIO(payload))
        client=c.Client(api,self.folder/'provider-errors',opener=opener);self.d.client=client
        with self.assertRaisesRegex(RuntimeError,'no repeat'):self.d.create('guard')
        row=self.d.state['roles']['guard'];self.assertEqual(row['provider_error']['http_status'],503)
        self.assertIn('NO_STOCK',row['provider_error']['redacted_body'])
        self.assertIn('capacity unavailable',row['provider_error']['redacted_body'])
        receipts=list((self.folder/'provider-errors').glob('*.json'));self.assertEqual(len(receipts),1)
        saved=(self.folder/'state.json').read_text()+receipts[0].read_text()
        self.assertNotIn(api,saved);self.assertNotIn(token,saved)
        with self.assertRaisesRegex(ValueError,'already attempted'):self.d.create('guard')
        self.assertEqual(opener.open.call_count,1)

    def test_paginated_inventory_and_repeated_cursor_fail_closed(self):
        client=c.Client('fake');client.call=MagicMock(side_effect=[
            {'pods':[{'id':'a'}],'pagination':{'hasNextPage':True,'nextCursor':'p2'}},
            {'pods':[{'id':'b'}],'pagination':{'hasNextPage':False}}])
        self.assertEqual(client.pods(),[{'id':'a'},{'id':'b'}]);self.assertIn('cursor=p2',client.call.call_args.args[1])
        page={'pods':[],'pagination':{'hasNextPage':True,'nextCursor':'same'}}
        client.call=MagicMock(return_value=page)
        with self.assertRaisesRegex(ValueError,'pagination'):client.pods()

    def test_oversized_error_cannot_reveal_a_truncated_credential(self):
        secret='private-long-reflected-credential'
        raw=b'x'*65530+secret.encode()
        opener=MagicMock();opener.open.side_effect=urllib.error.HTTPError('https://fixture',400,'bad',{},io.BytesIO(raw))
        client=c.Client(secret,opener=opener)
        with self.assertRaises(c.RecordedProviderError) as caught:client.call('POST','/pods',{})
        receipt=caught.exception.receipt
        self.assertTrue(receipt['body_truncated']);self.assertEqual(receipt['captured_body_bytes'],65536)
        self.assertIn('withheld',receipt['redacted_body'])
        self.assertNotIn(secret[:6],receipt['redacted_body'])


if __name__=='__main__':unittest.main()
