"""Loopback fake-provider HTTP tests. No Runpod requests or paid resources."""
import contextlib
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import unittest
from urllib.parse import urlparse, parse_qs
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from unittest.mock import patch

from scripts.cloud import cardinal_guard as cg

INTENT='e28b40ea-430f-4e58-b159-5405656ef6a1'
TOKEN='test-only-'+('x'*40)


def pod(identifier,role='gpu'):
    return {'id':identifier,'name':f'selfplay-cardinal-{INTENT}-{role}',
            'env':{'SELFPLAY_PROJECT':'cardinal-gpu','SELFPLAY_INTENT':INTENT,'SELFPLAY_ROLE':role},
            'status':'RUNNING'}


class Provider:
    def __init__(self):
        self.pods={'guard':pod('guard','guard')}
        self.requests=[]
        self.pages=None
        self.inventory_code=200
        self.delete_delayed=set()
        self.get_override={}

    def handler(self):
        owner=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def dispatch(self,method):
                parsed=urlparse(self.path)
                owner.requests.append((method,self.path))
                if self.headers.get('Authorization')!='Bearer fake-provider-secret':
                    return self.reply(401,{'provider_secret':'must not escape'})
                if parsed.path=='/v2/pods' and method=='GET':
                    if owner.inventory_code!=200:return self.reply(owner.inventory_code,{'secret':'provider-body-secret'})
                    cursor=parse_qs(parsed.query).get('cursor',[''])[0]
                    if owner.pages is not None:return self.reply(200,owner.pages[cursor])
                    return self.reply(200,{'pods':list(owner.pods.values()),'pagination':{'hasNextPage':False,'nextCursor':None}})
                ident=parsed.path.removeprefix('/v2/pods/')
                if method=='GET':
                    if ident in owner.get_override:return self.reply(200,owner.get_override[ident])
                    return self.reply(200,owner.pods[ident]) if ident in owner.pods else self.reply(404,{'secret':'none'})
                if method=='DELETE':
                    if ident not in owner.delete_delayed:owner.pods.pop(ident,None)
                    return self.reply(204,None)
                self.reply(404,{})
            def reply(self,code,body):
                data=json.dumps(body).encode() if body is not None else b''
                self.send_response(code);self.send_header('Content-Length',str(len(data)))
                self.end_headers();self.wfile.write(data)
            def do_GET(self):self.dispatch('GET')
            def do_DELETE(self):self.dispatch('DELETE')
        return Handler


@contextlib.contextmanager
def server(handler):
    http=ThreadingHTTPServer(('127.0.0.1',0),handler)
    thread=threading.Thread(target=http.serve_forever,daemon=True);thread.start()
    try:yield 'http://127.0.0.1:'+str(http.server_port)
    finally:http.shutdown();http.server_close();thread.join()


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.provider=Provider();self.now=1000.
        self.context=server(self.provider.handler());base=self.context.__enter__()
        self.guard=cg.Guard(cg.API('fake-provider-secret',base+'/v2'),INTENT,'guard',1100.,TOKEN,'a'*64,lambda:self.now)
    def tearDown(self):self.context.__exit__(None,None,None)
    def deletes(self):return [p for method,p in self.provider.requests if method=='DELETE']

    def test_pending_create_keeps_guard_alive_even_after_deadline(self):
        self.guard.tick();self.assertTrue(self.guard.health()['ready'])
        self.now=1200;self.guard.tick();self.now=5000;self.guard.tick()
        self.assertEqual(self.deletes(),[])
        self.assertFalse(self.guard.health()['self_delete_requested'])
        self.provider.pods['late']=pod('late')
        self.guard.tick();self.assertEqual(self.deletes(),['/v2/pods/late'])

    def test_explicit_closed_empty_intent_can_finish_early(self):
        self.guard.tick()
        self.guard.finish({'intent':INTENT,'creation_closed':True})
        self.guard.tick();self.assertEqual(self.deletes(),[])
        self.now+=31;self.guard.tick()
        self.assertEqual(self.deletes(),['/v2/pods/guard'])

    def test_deadline_cleanup_confirms404_then_waits_quiet_period(self):
        self.provider.pods['gpu']=pod('gpu')
        self.provider.pods['unrelated']={'id':'unrelated','name':'other','env':{}}
        self.guard.tick();self.assertEqual(self.deletes(),[])
        self.now=1100;self.guard.tick();self.assertEqual(self.deletes(),['/v2/pods/gpu'])
        self.assertEqual(self.guard.health()['confirmed_404_gpu_ids'],['gpu'])
        self.now=1115;self.guard.tick();self.assertIn('guard',self.provider.pods)
        self.now=1146;self.guard.tick()
        self.assertEqual(self.deletes(),['/v2/pods/gpu','/v2/pods/guard'])
        self.assertIn('unrelated',self.provider.pods)

    def test_duplicates_cleaned_but_near_matches_untouched(self):
        self.provider.pods.update({n:pod(n) for n in ('gpu1','gpu2','wrongproject','wrongname','wrongrole')})
        self.provider.pods['wrongproject']['env']['SELFPLAY_PROJECT']='other'
        self.provider.pods['wrongname']['name']+='suffix'
        self.provider.pods['wrongrole']['env']['SELFPLAY_ROLE']='guard'
        self.now=1100;self.guard.tick()
        self.assertEqual(self.deletes(),['/v2/pods/gpu1','/v2/pods/gpu2'])
        self.assertTrue(self.guard.health()['duplicate_intent_anomaly'])
        self.assertEqual(set(self.provider.pods),{'guard','wrongproject','wrongname','wrongrole'})

    def test_direct_get_identity_mismatch_prevents_delete(self):
        self.provider.pods['gpu']=pod('gpu')
        changed=pod('gpu');changed['env']['SELFPLAY_INTENT']='different'
        self.provider.get_override['gpu']=changed
        self.now=1100;self.guard.tick()
        self.assertEqual(self.deletes(),[])
        self.assertEqual(self.guard.health()['error'],'identity_mismatch')
        self.assertFalse(self.guard.health()['ready'])

    def test_guard_identity_mismatch_prevents_any_mutation(self):
        self.provider.pods['guard']['env']['SELFPLAY_ROLE']='other'
        self.provider.pods['gpu']=pod('gpu');self.now=1100;self.guard.tick()
        self.assertEqual(self.deletes(),[])
        self.assertEqual(self.guard.health()['error'],'guard_identity_unverified')

    def test_all_inventory_pages_and_pagination_cycle_fails_closed(self):
        self.provider.pods['gpu']=pod('gpu')
        self.provider.pages={'':{'pods':[pod('guard','guard')],'pagination':{'hasNextPage':True,'nextCursor':'p2'}},
            'p2':{'pods':[pod('gpu')],'pagination':{'hasNextPage':False,'nextCursor':None}}}
        self.guard.tick();self.assertEqual(self.guard.health()['known_gpu_ids'],['gpu'])
        self.assertTrue(any('cursor=p2' in p for _,p in self.provider.requests))
        self.provider.pages['p2']['pagination']={'hasNextPage':True,'nextCursor':'p2'}
        self.now=1100;self.guard.tick()
        self.assertEqual(self.deletes(),[])
        self.assertEqual(self.guard.health()['error'],'inventory_pagination')

    def test_delete_success_is_not_absence_and_provider_errors_are_redacted(self):
        self.provider.pods['gpu']=pod('gpu');self.provider.delete_delayed.add('gpu')
        self.now=1100;self.guard.tick();self.now=1400;self.guard.tick()
        self.assertIn('guard',self.provider.pods);self.assertEqual(self.guard.health()['confirmed_404_gpu_ids'],[])
        self.provider.inventory_code=500;self.guard.tick()
        output=json.dumps(self.guard.health())
        self.assertIn('inventory_http_500',output)
        self.assertNotIn('secret',output);self.assertNotIn(TOKEN,output)

    def test_authenticated_routes_have_no_exec_and_finish_body_is_strict(self):
        self.guard.tick()
        with server(cg.handler_for(self.guard)) as base:
            def call(path,method='GET',body=None,token=TOKEN):
                data=json.dumps(body).encode() if body is not None else None
                req=Request(base+path,data=data,method=method,headers={'X-Guard-Token':token})
                try:
                    with urlopen(req,timeout=2) as response:return response.status,json.load(response)
                except HTTPError as error:return error.code,json.load(error)
            self.assertEqual(call('/health',token='bad')[0],403)
            code,body=call('/health');self.assertEqual(code,200);self.assertTrue(body['ready'])
            self.assertEqual(call('/exec')[0],404)
            self.assertEqual(call('/finish','POST',{'intent':INTENT,'creation_closed':1})[0],400)
            self.assertFalse(self.guard.finish_requested)
            self.assertEqual(call('/finish','POST',{'intent':INTENT,'creation_closed':True})[0],202)
            self.assertTrue(self.guard.finish_requested)
            self.assertEqual(self.deletes(),[])

    def test_controller_hex_uuid_is_accepted_and_invalid_identity_rejected(self):
        value=INTENT.replace('-', '')
        other=cg.Guard(self.guard.api,value,'guard',1100.,TOKEN,'a'*64,lambda:self.now)
        self.assertEqual(other.intent,value)
        with self.assertRaises((cg.GuardError,ValueError)):
            cg.Guard(self.guard.api,'not-a-uuid','guard',1100.,TOKEN,'a'*64,lambda:self.now)
        with self.assertRaises(cg.GuardError):
            cg.Guard(self.guard.api,value,'../pods/unrelated',1100.,TOKEN,'a'*64,lambda:self.now)

    def test_code_hash_mismatch_aborts_before_api(self):
        with patch.dict(cg.os.environ,{'SELFPLAY_GUARD_SHA256':'0'*64}),patch.object(cg,'API') as api:
            with self.assertRaisesRegex(cg.GuardError,'source_mismatch'):cg.main()
            api.assert_not_called()


if __name__=='__main__':unittest.main()
