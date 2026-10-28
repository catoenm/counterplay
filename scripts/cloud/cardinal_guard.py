"""Project-scoped Runpod deadline guard; stdlib only, no neural work or shell API.

Run only on the separately created CPU guard. /health and POST /finish require
X-Guard-Token. finish means creation_closed=true: no later GPU create is allowed.
Never echoes provider bodies, environment, API keys, or request credentials.
"""
from __future__ import annotations
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, build_opener, HTTPRedirectHandler
import uuid

PROJECT = 'cardinal-gpu'
POLL_SECONDS = 15
QUIET_SECONDS = 30


class GuardError(Exception):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise GuardError('provider_redirect_rejected')


class API:
    def __init__(self, key, base='https://api.runpod.io/v2'):
        self.key, self.base = key, base
        self.opener = build_opener(NoRedirect)

    def request(self, method, path):
        req = Request(self.base+path, method=method,
                      headers={'Authorization':'Bearer '+self.key, 'Accept':'application/json'})
        try:
            with self.opener.open(req, timeout=8) as response:
                raw = response.read(8*1024*1024+1)
                if len(raw)>8*1024*1024: raise GuardError('provider_response_too_large')
                return response.status, json.loads(raw) if raw else None
        except HTTPError as error:
            code = error.code
            error.close()
            return code, None
        except GuardError:
            raise
        except Exception:
            raise GuardError('provider_transport_or_json_error') from None

    def inventory(self):
        pods, cursor, seen = [], None, set()
        started = time.monotonic()
        for _ in range(100):
            query = {'limit':100, 'includeClusterPods':'true'}
            if cursor is not None: query['cursor'] = cursor
            code, body = self.request('GET','/pods?'+urlencode(query))
            if code != 200 or not isinstance(body,dict): raise GuardError('inventory_http_'+str(code))
            if not isinstance(body.get('pods'),list): raise GuardError('inventory_schema')
            pods.extend(body['pods'])
            page = body.get('pagination',{})
            if page.get('hasNextPage') is False: return pods
            cursor = page.get('nextCursor')
            if page.get('hasNextPage') is not True or not isinstance(cursor,str) or not cursor or cursor in seen:
                raise GuardError('inventory_pagination')
            seen.add(cursor)
            if time.monotonic()-started > 30: raise GuardError('inventory_time_limit')
        raise GuardError('inventory_page_limit')


def valid_id(value):
    return isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9_-]{1,128}',value) is not None


class Guard:
    def __init__(self, api, intent, own_id, deadline, token, code_hash, clock=time.time):
        if intent not in (str(uuid.UUID(intent)),uuid.UUID(intent).hex) or not valid_id(own_id):
            raise GuardError('invalid_guard_identity')
        if not math.isfinite(deadline) or deadline<=0 or deadline>clock()+24*3600:
            raise GuardError('invalid_deadline')
        if len(token)<32: raise GuardError('weak_guard_token')
        self.api, self.intent, self.own_id = api, intent, own_id
        self.deadline, self.token, self.code_hash, self.clock = deadline, token, code_hash, clock
        self.lock = threading.RLock()
        self.known, self.absent = set(), set()
        self.creation_closed, self.finish_requested, self.self_delete_requested = False, False, False
        self.last_scan, self.empty_since, self.error = None, None, None
        self.duplicate_intent, self.guard_verified = False, False

    def matches(self, pod, role):
        if not isinstance(pod,dict): return False
        env = pod.get('env')
        return (valid_id(pod.get('id')) and isinstance(env,dict) and
                pod.get('name') == f'selfplay-cardinal-{self.intent}-{role}' and
                env.get('SELFPLAY_PROJECT') == PROJECT and env.get('SELFPLAY_INTENT') == self.intent and
                env.get('SELFPLAY_ROLE') == role)

    def get(self, pod_id):
        if not valid_id(pod_id): raise GuardError('invalid_pod_id')
        return self.api.request('GET','/pods/'+pod_id)

    def delete_verified(self, pod_id, role):
        code, pod = self.get(pod_id)
        if code == 404: return True
        if code != 200: raise GuardError('identity_get_http_'+str(code))
        if not self.matches(pod,role) or pod['id'] != pod_id: raise GuardError('identity_mismatch')
        code, _ = self.api.request('DELETE','/pods/'+pod_id)
        if code not in (200,202,204,404): raise GuardError('delete_http_'+str(code))
        code, _ = self.get(pod_id)
        if code not in (200,404): raise GuardError('confirmation_http_'+str(code))
        return code == 404

    def finish(self, payload):
        if not isinstance(payload,dict) or set(payload) != {'intent','creation_closed'} or payload['intent'] != self.intent or payload['creation_closed'] is not True:
            raise GuardError('invalid_finish_body')
        with self.lock:
            self.creation_closed = self.finish_requested = True
            return self.health()

    def health(self):
        with self.lock:
            return {'intent':self.intent, 'project':PROJECT, 'guard_pod_id':self.own_id,
                'source_sha256':self.code_hash, 'deadline_unix':self.deadline,
                'ready':self.guard_verified and self.last_scan is not None and self.clock()-self.last_scan<60 and self.error is None,
                'last_inventory_unix':self.last_scan, 'known_gpu_ids':sorted(self.known),
                'confirmed_404_gpu_ids':sorted(self.absent), 'duplicate_intent_anomaly':self.duplicate_intent,
                'creation_closed':self.creation_closed, 'finish_requested':self.finish_requested,
                'self_delete_requested':self.self_delete_requested, 'error':self.error}

    def tick(self):
        """One complete scan; errors retain the guard and never widen its scope."""
        try:
            code, own = self.get(self.own_id)
            if code != 200 or not self.matches(own,'guard') or own['id'] != self.own_id:
                raise GuardError('guard_identity_unverified')
            pods = self.api.inventory()
            matching = {p['id'] for p in pods if self.matches(p,'gpu')}
            if self.own_id in matching: raise GuardError('guard_gpu_identity_collision')
            now = self.clock()
            with self.lock:
                self.guard_verified = True
                self.known.update(matching)
                self.duplicate_intent |= len(self.known)>1
                self.last_scan, self.error = now, None
                cleanup = now>=self.deadline or self.finish_requested
            if not cleanup: return
            absent = set()
            for pod_id in sorted(self.known):
                if self.delete_verified(pod_id,'gpu'): absent.add(pod_id)
            with self.lock:
                self.absent = absent
                # No observed GPU leaves an ambiguous creation unresolved. Root
                # must explicitly close creation; never infer it from emptiness.
                safe = (self.known or self.creation_closed) and absent == self.known and not matching
                if not safe:
                    self.empty_since = None
                    return
                if self.empty_since is None: self.empty_since = now
                if now-self.empty_since < QUIET_SECONDS: return
                self.self_delete_requested = True
            self.delete_verified(self.own_id,'guard')
        except GuardError as error:
            with self.lock: self.error = str(error)
        except Exception:
            with self.lock: self.error = 'unexpected_guard_error'


def handler_for(guard):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def log_message(self, *args):
            pass

        def respond(self, code, body):
            encoded=json.dumps(body,allow_nan=False).encode()
            self.send_response(code)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(encoded)))
            self.send_header('Cache-Control','no-store')
            self.send_header('Connection','close')
            self.end_headers()
            self.wfile.write(encoded)
            self.close_connection=True

        def authorized(self):
            received=self.headers.get('X-Guard-Token','')
            if not hmac.compare_digest(received.encode(),guard.token.encode()):
                self.respond(403,{'error':'forbidden'})
                return False
            return True

        def do_GET(self):
            if not self.authorized(): return
            if self.path != '/health': return self.respond(404,{'error':'unknown_path'})
            self.respond(200,guard.health())

        def do_POST(self):
            if not self.authorized(): return
            if self.path != '/finish': return self.respond(404,{'error':'unknown_path'})
            try:
                size=int(self.headers.get('Content-Length','-1'))
                if not 0<size<=1024 or self.headers.get('Transfer-Encoding'): raise GuardError('invalid_finish_body')
                payload=json.loads(self.rfile.read(size))
                result=guard.finish(payload)
            except Exception:
                return self.respond(400,{'error':'invalid_finish_body'})
            self.respond(202,result)
    return Handler


def main():
    code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if not hmac.compare_digest(code_hash,os.environ['SELFPLAY_GUARD_SHA256']): raise GuardError('guard_source_mismatch')
    guard=Guard(API(os.environ['RUNPOD_API_KEY']), os.environ['SELFPLAY_INTENT'], os.environ['RUNPOD_POD_ID'],
        float(os.environ['SELFPLAY_DEADLINE_UNIX']),os.environ['SELFPLAY_GUARD_TOKEN'],code_hash)
    def watch():
        while True:
            guard.tick()
            time.sleep(POLL_SECONDS)
    threading.Thread(target=watch,daemon=True).start()
    ThreadingHTTPServer(('0.0.0.0',8080),handler_for(guard)).serve_forever()


if __name__=='__main__':
    main()
