"""Read-only Runpod deployment planner. There is deliberately no launch command.

This module performs no I/O on import. The CLI reads already captured official
schemas/catalog data and local ledgers, then writes a reviewable dry-run plan.
It never provisions, reserves funds, downloads models, or sends API mutations.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import math
from pathlib import Path
import sys
import time

DEADLINE = datetime.fromisoformat('2026-10-22T08:00:00-07:00').timestamp()
SUBCAP = 20.0
DISK_GB = 160
GPU = 'NVIDIA A100 80GB PCIe'
IMAGE = 'runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404'
SOURCES = {
    'schema':'https://api.runpod.io/v2/openapi.json',
    'catalog':'https://docs.runpod.io/api-reference-v2/catalog/list-gpu-types',
    'create':'https://docs.runpod.io/api-reference-v2/pods/create-a-pod',
    'delete':'https://docs.runpod.io/api-reference-v2/pods/terminate-a-pod',
    'pod_pricing':'https://docs.runpod.io/pods/pricing',
    'serverless_pricing':'https://docs.runpod.io/serverless/pricing',
    'serverless_settings':'https://docs.runpod.io/serverless/endpoints/endpoint-configurations',
    'job_policy':'https://docs.runpod.io/serverless/endpoints/send-requests',
    'operations':'https://docs.runpod.io/serverless/endpoints/operation-reference',
    'cli_source':'https://github.com/runpod/runpodctl/blob/416403f44d2c5b81b3b30b9aadddb37fa8bd3294/cmd/pod/create.go',
}


def finite_money(value):
    if type(value) not in (int,float) or not math.isfinite(value) or value<0:
        raise ValueError('Accounting amounts must be finite nonnegative numbers')
    return float(value)


def ledger_commitment(ledger):
    entries=ledger.get('requests')
    if not isinstance(entries,dict):
        raise ValueError('Unrecognized ledger schema; manual accounting review required')
    total=0.0
    for row in entries.values():
        state=row.get('state')
        if not isinstance(state,str) or not state:raise ValueError('Missing ledger state')
        charged=finite_money(row.get('charged_upper_usd',0))
        reserved=finite_money(row.get('reserved_usd',0))
        if state=='completed' and 'charged_upper_usd' not in row:
            raise ValueError('Completed ledger entry lacks accounting')
        if state!='completed' and reserved<=0:
            raise ValueError('Unknown call must retain a positive reservation')
        total+=max(charged,reserved if state!='completed' else 0.0)
    return total


def budget_snapshot(root):
    """One consistent READ under the same lock used by current gateway clients."""
    root=Path(root)
    with (root/'runs/external-budget.lock').open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_SH)
        budget=json.loads((root/'budget.json').read_text())
        ledgers={}
        for path in sorted((root/'runs').glob('*ledger.json')):
            ledgers[path.name]=ledger_commitment(json.loads(path.read_text()))
        total=finite_money(budget['cloud_spend_to_date'])+finite_money(budget.get('cloud_reserved_usd',0))+sum(ledgers.values())
        cap=min(100.0,finite_money(budget['approved_pilot_cap']))
        return {'pilot_cap_usd':cap,'cloud_spend_to_date':budget['cloud_spend_to_date'],
                'cloud_reserved_usd':budget.get('cloud_reserved_usd',0),'ledger_commitments':ledgers,
                'cumulative_commitment_usd':total,'available_usd':cap-total,
                'proposed_reservation_usd':SUBCAP,'reservation_fits':total+SUBCAP<=cap,
                'reservation_applied':False}


def estimate(rate,seconds,serverless=False,disk_gb=DISK_GB):
    rate=finite_money(rate)
    if type(seconds) is not int or seconds<0:raise ValueError('Invalid billed seconds')
    if type(disk_gb) is not int or not 1<=disk_gb<=160:raise ValueError('Invalid disk size')
    # 28-day month overestimates the documented $0.10/GB/month disk price.
    disk_seconds=math.ceil(seconds/300)*300 if serverless else seconds
    compute=rate*seconds/3600
    disk=disk_gb*0.10*disk_seconds/(28*24*3600)
    return {'compute_usd':compute,'disk_usd':disk,'total_usd':compute+disk,
            'billed_seconds_assumed':seconds,'disk_billed_seconds':disk_seconds,
            'disk_usd_per_gb_month':0.10,'month_hours_conservative':28*24,
            'hard_upper_bound_proven':False}


def schema_properties(spec,name):
    """Discover published property names; not a general JSON Schema validator."""
    schema=spec['components']['schemas'][name]
    def walk(node):
        found=set(node.get('properties',{}))
        if '$ref' in node:found|=walk(spec['components']['schemas'][node['$ref'].split('/')[-1]])
        for child in node.get('allOf',[]):found|=walk(child)
        return found
    return walk(schema)


def make_plan(spec,quote,budget,now):
    if not math.isfinite(now) or now>=DEADLINE:raise ValueError('Overnight deadline has elapsed')
    gpu=quote['gpu']
    if gpu['id']!=GPU or gpu['memory']!=80:raise ValueError('Unexpected quoted GPU')
    remaining=int(DEADLINE-now)
    # Reserve 30 minutes for startup and five minutes for artifact retrieval.
    execution=min(6*3600,remaining-1800-300)
    if execution<300:raise ValueError('Insufficient time before fixed deadline')
    pod_seconds=execution+1800+300
    serverless_seconds=execution+1800+5
    pod_rate=finite_money(gpu['price']['secure'])
    serverless_rate=max(finite_money(gpu['price']['serverless']),0.00076*3600)
    pod={'name':'selfplay-program-rl-UNIQUE_INTENT_ID','cloud':'SECURE','image':IMAGE,
         'gpu':{'id':GPU,'count':1,'minCudaVersion':'12.8'},'disk':DISK_GB,
         'ports':['22/tcp'],'startSsh':True,'startJupyter':False,
         'env':{'SELFPLAY_PROJECT':'program-rl','SELFPLAY_INTENT':'UNIQUE_INTENT_ID'},
         'mounts':{}}
    endpoint={'name':'selfplay-program-rl-UNIQUE_INTENT_ID','type':'QUEUE_BASED',
              'image':'UNBUILT_SERVERLESS_HANDLER_IMAGE','disk':DISK_GB,
              'gpu':{'pools':['AMPERE_80'],'count':1,'minCudaVersion':'12.8'},
              'workers':{'min':0,'max':1,'idleTimeout':5},
              'scaling':{'type':'QUEUE_DELAY','queueDelay':4},
              'flashboot':'OFF','timeout':execution*1000,'env':{},'ports':[]}
    for schema,body in [('CreatePodRequest',pod),('CreateEndpointRequest',endpoint)]:
        if set(body)-schema_properties(spec,schema):raise ValueError('Request uses unpublished fields')
    blockers=[
        'Strategic capability gate failed; no GPU training currently justified.',
        'No independent always-on Pod cleanup host is configured.',
        'REST v2 Pod create has no verified provider-enforced termination deadline.',
        'Serverless startup billing is outside executionTimeout; no verified provider startup cap.',
        'No documented retryLimit=0 field in RESTv2 endpoint or per-job policy.',
        'TTL deletes job data; documentation does not explicitly establish initializing-worker shutdown.',
        'Serverless worker image and large-artifact delivery are not built or validated.',
    ]
    if gpu.get('availability') not in {'LOW','MEDIUM','HIGH'}:blockers.append('Quoted GPU is currently unavailable.')
    if now-quote['checked_at_unix']>300: blockers.append('Catalog quote is older than five minutes.')
    if not budget['reservation_fits']:blockers.append('Proposed $20 reservation exceeds remaining cumulative budget.')
    return {'format_version':1,'state':'read_only_dry_run','can_launch':False,'external_actions':0,
            'deadline':'2026-10-22T08:00:00-07:00','planned_at_unix':now,'budget':budget,
            'gpu_quote':quote,'pod_request_example_NOT_SUBMITTED':pod,
            'pod_conditional_cost_estimate':estimate(pod_rate,pod_seconds),
            'serverless_request_example_NOT_SUBMITTED':endpoint,
            'serverless_job_policy':{'executionTimeout':execution*1000,'ttl':(execution+1800)*1000},
            'serverless_conditional_cost_estimate':estimate(serverless_rate,serverless_seconds,True),
            'serverless_rate_used_usd_per_hour':serverless_rate,
            'cost_interpretation':'Estimates assume bounded worker startup/termination. They are NOT proven billing caps.',
            'blockers':blockers,'sources':SOURCES,
            'future_launch_contract':[
                'Require a passed strategic capability gate and resolved independent termination bound.',
                'Take external-budget.lock exclusively, read every ledger, reserve $20 in budget.cloud_reserved_usd before any POST.',
                'Persist one random project intent and exact request hash before one create POST; never automatically retry ambiguous creation.',
                'Arm an independent remote deadline before Pod creation; bind only its returned pod ID after verifying name/project intent.',
                'Local heartbeat copies hashed artifacts periodically and begins final recovery five minutes before cleanup.',
                'Terminate only the recorded project pod/endpoint ID; GET confirms 404 or TERMINATED; no stop-only cleanup or unrelated resources.',
                'Preserve full reservation after uncertain POST, deletion, or billing reconciliation; release only against verified scoped billing.',
                'Serverless alternative: one async /run POST, no manual retry, persist job/endpoint IDs, poll results before30-minute retention expires.',
            ]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,default=Path('runs/runpod-deployment-plan'))
    parser.add_argument('--root',type=Path,default=Path('.'))
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    spec_path=args.evidence/'openapi.json';quote_path=args.evidence/'a100-quote.json'
    plan=make_plan(json.loads(spec_path.read_text()),json.loads(quote_path.read_text()),budget_snapshot(args.root),time.time())
    files=[p for p in args.evidence.iterdir() if p.is_file() and p.name!='plan.json']+[Path(__file__)]
    plan['evidence_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    out=args.out or args.evidence/'plan.json'
    if out.exists():raise ValueError('Refusing to overwrite an earlier infrastructure plan')
    out.write_text(json.dumps(plan,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'out':str(out.resolve()),'can_launch':False,'reservation_applied':False,
                      'prior_commitment_usd':plan['budget']['cumulative_commitment_usd'],
                      'pod_conditional_cost_usd':plan['pod_conditional_cost_estimate']['total_usd'],
                      'serverless_conditional_cost_usd':plan['serverless_conditional_cost_estimate']['total_usd']}))


if __name__=='__main__':main()
