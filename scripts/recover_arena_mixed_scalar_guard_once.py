"""Explicit administrative recovery of one unknown CPU guard; never launches GPU/model.

The first attempt is retained as unknown. A separately funded and reviewed recovery
attempt may be sent exactly once under the original identity and deadline. Existing
duplicate detection and cleanup remain in charge if the first guard later appears.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import runpod_cardinal as base
from scripts.runpod_arena_mixed_scalar import Deployment, budget_total

FOLDER = ROOT/'runs/cardinal-arena-mixed-scalar-training-deployment-v1'
RUN = ROOT/'runs/arena-mixed-scalar-training-gpu-v1'
INTENT = '30f7647634854202be7aa07b37bdd25b'
MANIFEST = 'aad493a27e256857237b405c390fdc71a9be3ce623cf21152a74493cd1a91e6e'
DEADLINE = 1792817444
EXTRA_RESERVATION = 4.0
AUTHORIZATION = 'Why is the 8B run still blocked? Please resolve the provisioning issue and get training going across all five environments. I wanted it running overnight.'


def exclusive(path, value):
    path = Path(path)
    temp = path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    try:
        with temp.open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def sanitize_error(raw, secrets):
    """Only public error-description fields; never retain echoed requests/headers."""
    try:
        body = json.loads(raw)
    except Exception:
        return {'body_sha256':hashlib.sha256(raw).hexdigest(), 'json':False}
    result = {'body_sha256':hashlib.sha256(raw).hexdigest(), 'json':True}
    if not isinstance(body,dict): return result
    for field in ('type','title','status','code','message','detail','error'):
        value = body.get(field)
        if isinstance(value,(int,float,bool)) or value is None:
            if field in body: result[field]=value
        elif isinstance(value,str):
            for secret in sorted((x for x in secrets if x),key=len,reverse=True):
                value=value.replace(secret,'[REDACTED]')
            result[field]=value[:1500]
    return result


class CapturingClient(base.Client):
    def create_guard_once(self, body):
        request=urllib.request.Request(base.API+'/pods',data=json.dumps(body).encode(),method='POST',
            headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
        secrets=[self.key,*[str(v) for k,v in body['env'].items() if k not in base.PUBLIC_ENV]]
        try:
            with self.opener.open(request,timeout=45) as response:
                raw=response.read(8*1024*1024+1)
                if len(raw)>8*1024*1024: raise ValueError('Oversized provider response')
                return json.loads(raw),{'http_status':response.status}
        except urllib.error.HTTPError as error:
            raw=error.read(256*1024)
            report={'http_status':error.code,'error':sanitize_error(raw,secrets)}
            error.close()
            return None,report
        except Exception as error:
            return None,{'http_status':None,'error_type':type(error).__name__}


def validate_local(deployment, now):
    from selfplay.arena_mixed_scalar.experiment import verify_run
    p,s=deployment.plan,deployment.state
    if p['intent']!=INTENT or p['deadline_unix']!=DEADLINE or now>=DEADLINE-3600:
        raise ValueError('Wrong deployment or insufficient original time remains')
    if s.get('creation_closed') or s.get('cleanup_started') or set(s['roles'])!={'guard'}:
        raise ValueError('Creation closed or another role has already been attempted')
    guard=s['roles']['guard']
    if (guard.get('state')!='create_unknown' or guard.get('known_ids') or guard.get('matching_ids')
            or guard.get('pod_id') or guard.get('duplicate_intent_anomaly')):
        raise ValueError('Original guard must remain wholly unidentified for this recovery')
    for name in ('guard-recovery-intent-v1.json','guard-recovery-result-v1.json',
                 'launch-intent.json','hardware-check.json','supervisor-origin.json'):
        if (FOLDER/name).exists(): raise ValueError('Recovery or model staging already attempted')
    if base.digest(RUN/'manifest.json')!=MANIFEST: raise ValueError('Manifest changed')
    _,status=verify_run(RUN)
    if status['state']!='prepared' or status['artifacts']:
        raise ValueError('Scientific run has already started')
    for name in ('remote-dispatch-intent.json','remote-job-start.json','execution-intent.json'):
        if (RUN/name).exists(): raise ValueError('Neural execution already attempted')
    return deepcopy(guard)


def recover(review_path):
    review=base.load(review_path)
    if (review.get('approved') is not True or review.get('script_sha256')!=base.digest(__file__)
            or review.get('deployment_intent')!=INTENT or review.get('extra_guard_reservation_usd')!=EXTRA_RESERVATION):
        raise ValueError('Independent review does not approve this exact recovery script')
    client=CapturingClient()
    with base.lock(FOLDER/'deployment.lock'):
        d=Deployment(FOLDER,client);original=validate_local(d,time.time())
        d.reconcile()
        validate_local(d,time.time())
        body=d.body('guard')
        request_sha=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
        if request_sha!=original['request_sha256']: raise ValueError('Guard identity/source/request changed')
        # Four dollars covers the entire extra guard life, including a full-hour
        # envelope, while the original $125 and future $50 remain fully held.
        upper=math.ceil(max(0,DEADLINE-time.time())/3600)*(.14+10*.10/(28*24))
        if upper>EXTRA_RESERVATION: raise ValueError('Extra guard envelope exceeds funding')
        recovery_id=uuid.uuid4().hex
        with base.lock(ROOT/'runs/external-budget.lock'):
            budget=base.load(ROOT/'budget.json');d.check_pair(budget['cloud_reservations'])
            total=budget_total(budget)
            if total+EXTRA_RESERVATION>min(350,budget['approved_pilot_cap']): raise ValueError('Budget ceiling')
            exclusive(FOLDER/'guard-recovery-original-unknown-v1.json',{'at':base.now(),
                'original_state':deepcopy(d.state),'original_state_sha256':base.digest(FOLDER/'state.json'),
                'original_outcome':'unknown; no absence-based rejection claim'})
            exclusive(FOLDER/'budget-before-guard-recovery-v1.json',budget)
            intent={'kind':'explicit_guard_recovery_attempt','at':base.now(),'recovery_id':recovery_id,
                'deployment_intent':INTENT,'deadline_unix':DEADLINE,'manifest_sha256':MANIFEST,
                'request_sha256':request_sha,'original_guard_attempt':original,
                'original_outcome_remains_unknown':True,'maximum_create_requests':1,
                'extra_guard_reservation_usd':EXTRA_RESERVATION,'extra_guard_cost_envelope_usd':upper,
                'cumulative_before_usd':total,'cumulative_after_usd':total+EXTRA_RESERVATION,
                'user_authorization':AUTHORIZATION,'independent_review':str(Path(review_path).resolve()),
                'independent_review_sha256':base.digest(review_path),'script_sha256':base.digest(__file__)}
            exclusive(FOLDER/'guard-recovery-intent-v1.json',intent)
            budget['cloud_reservations'][recovery_id]={'state':'reserved','amount_usd':EXTRA_RESERVATION,
                'deployment':str(FOLDER),'purpose':'one additional CPU guard recovery attempt; original unknown fully retained',
                'original_deployment_intent':INTENT,'created_at':base.now(),'deadline_unix':DEADLINE}
            budget['cloud_reserved_usd']+=EXTRA_RESERVATION
            base.atomic_json(ROOT/'budget.json',budget)
        d.state.setdefault('guard_creation_attempts',[]).append({'kind':'original_unknown','record':original})
        d.state['guard_creation_attempts'].append({'kind':'explicit_recovery','recovery_id':recovery_id,
            'attempted_at':base.now(),'request_sha256':request_sha,'result':'pending'})
        d.state['roles']['guard'].update(state='create_attempted',recovery_id=recovery_id,
            original_create_outcome_unknown=True,recovery_attempted_at=base.now())
        d.save() # Intent, funding and state are durable before the sole new POST.
        pod,report=client.create_guard_once(body)
        if isinstance(pod,dict) and 'pod' in pod: pod=pod['pod']
        identified=isinstance(pod,dict) and base.matches(pod,INTENT,'guard')
        if identified:
            d.state['roles']['guard'].update(state='identified',pod_id=base.pod_id(pod['id']),latest=base.identity(pod))
        else:
            d.state['roles']['guard'].update(state='create_unknown',recovery_error=report)
        d.state['guard_creation_attempts'][-1].update(result='identified' if identified else 'unknown',provider_report=report)
        d.save()
        result={'at':base.now(),'recovery_id':recovery_id,'identified':identified,'provider_report':report,
            'original_outcome_remains_unknown':True,'new_gpu_requests':0,'new_model_launches':0,
            'pod_id':pod['id'] if identified else None,'reservation_retained_usd':EXTRA_RESERVATION}
        exclusive(FOLDER/'guard-recovery-result-v1.json',result)
        d.reconcile();d.cleanup_if_needed()
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--independent-review',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(recover(args.independent_review),indent=2))
