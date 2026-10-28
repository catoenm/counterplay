"""New $75 debug-rental envelope; reuse the frozen scoped lifecycle controller.

Preparation creates local identity files only. Reserve/create are separate,
explicit commands; inherited one-POST, guard, rate and cleanup rules apply.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import secrets
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import runpod_cardinal as base
from scripts.runpod_pilot import ledger_commitment, finite_money

RESERVATION = 75.0
lock, now = base.lock, base.now


def validate(plan):
    if (plan.get('reservation_usd') != RESERVATION or plan.get('project') != base.PROJECT
            or not 0 < plan['maximum_hours'] <= 12
            or not 0 < plan['maximum_gpu_rate_usd_hour'] <= 5.29
            or plan['cpu_id'] != 'cpu5c' or plan['cpu_vcpu'] != 4
            or plan['maximum_cpu_rate_usd_hour'] != .14
            or not math.isfinite(plan['deadline_unix'])
            or int(plan['deadline_unix']) != plan['deadline_unix']):
        raise ValueError('Debug rental envelope differs')
    expected = (plan['maximum_gpu_rate_usd_hour'] + .14 + 170*.10/(28*24))*plan['maximum_hours']
    if (plan['gpu_disk_gb'], plan['guard_disk_gb']) != (160, 10) or abs(expected-plan['estimate_usd']) > 1e-9 or expected+10 > RESERVATION:
        raise ValueError('Estimate plus contingency exceeds debug reservation')
    if plan['inherited_controller_sha256'] != base.digest(Path(base.__file__)):
        raise ValueError('Frozen lifecycle controller changed')


def prepare(folder, gpu_id, gpu_rate, ssh_public_key, hours=11.5):
    folder = Path(folder)
    if folder.exists(): raise FileExistsError('Deployment already exists')
    public = Path(ssh_public_key).read_text().strip()
    if not public.startswith('ssh-ed25519 ') or '\n' in public:
        raise ValueError('Expected one project Ed25519 public key')
    intent = uuid.uuid4().hex
    private = ROOT/'.secrets/cardinal-gpu'/intent
    secret_file = private/'guard.json'
    guard = ROOT/'scripts/cloud/cardinal_guard.py'
    plan = {'intent':intent, 'project':base.PROJECT, 'created_at':now(),
        'deadline_unix':float(math.floor(time.time()+hours*3600)), 'maximum_hours':hours,
        'reservation_usd':RESERVATION,
        'estimate_usd':(gpu_rate+.14+170*.10/(28*24))*hours,
        'gpu_id':gpu_id, 'maximum_gpu_rate_usd_hour':gpu_rate,
        'cpu_id':'cpu5c', 'cpu_vcpu':4, 'maximum_cpu_rate_usd_hour':.14,
        'gpu_disk_gb':160, 'guard_disk_gb':10,
        'gpu_image':'runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404',
        'guard_image':'python:3.12-slim', 'ssh_public_key':public,
        'guard_source_sha256':base.digest(guard), 'guard_secret_path':str(secret_file),
        'inherited_controller_sha256':base.digest(Path(base.__file__)),
        'debug_controller_sha256':base.digest(Path(__file__)),
        'cleanup':'Independent CPU deadline guard plus local exact-identity backup; DELETE, not stop. API outages are not a provider-enforced billing cap.'}
    validate(plan)
    folder.mkdir(parents=True)
    private.mkdir(parents=True, mode=0o700);os.chmod(private,0o700)
    fd=os.open(secret_file,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f: json.dump({'token':secrets.token_urlsafe(40)},f)
    base.atomic_json(folder/'plan.json',plan)
    base.atomic_json(folder/'state.json',{'plan_sha256':base.digest(folder/'plan.json'),
        'created_at':now(),'reservation_applied':False,'roles':{},'events':[]})
    return {'deployment':str(folder),'intent':intent,'estimate_usd':plan['estimate_usd'],
        'reservation_usd':RESERVATION,'deadline_unix':plan['deadline_unix']}


class Deployment(base.Deployment):
    def __init__(self, folder, client=None):
        super().__init__(folder,client)
        validate(self.plan)
        if self.plan['debug_controller_sha256'] != base.digest(Path(__file__)):
            raise ValueError('Debug controller changed after preparation')

    def reserve(self):
        with lock(ROOT/'runs/external-budget.lock'):
            budget=base.load(ROOT/'budget.json');records=budget.setdefault('cloud_reservations',{})
            held=finite_money(budget.get('cloud_reserved_usd',0))
            listed=0.
            for row in records.values():
                amount=finite_money(row['amount_usd'])
                if row['state'] in ('reserved','terminated_pending_billing'):listed+=amount
                elif amount:raise ValueError('Unknown outstanding reservation state')
            if abs(held-listed)>1e-8:raise ValueError('Aggregate pending cloud liability differs')
            intent=self.plan['intent']
            if intent not in records:
                total=finite_money(budget.get('cloud_spend_to_date',0))+held
                total+=sum(ledger_commitment(base.load(p)) for p in (ROOT/'runs').glob('*ledger.json'))
                if total+RESERVATION>min(100,finite_money(budget['approved_pilot_cap'])):
                    raise ValueError('Cumulative pilot cap would be exceeded')
                records[intent]={'amount_usd':RESERVATION,'state':'reserved','deployment':str(self.folder),'created_at':now()}
                budget['cloud_reserved_usd']=held+RESERVATION
                base.atomic_json(ROOT/'budget.json',budget)
            elif records[intent]['amount_usd']!=RESERVATION or records[intent]['state']!='reserved' or Path(records[intent]['deployment']).resolve()!=self.folder.resolve():
                raise ValueError('Existing reservation is incompatible')
        self.state['reservation_applied']=True;self.save()
        return {'reserved_usd':RESERVATION}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','reserve','create','status','health','finish','delete','watch'))
    p.add_argument('--deployment',type=Path,required=True)
    p.add_argument('--role',choices=('guard','gpu'))
    p.add_argument('--gpu-id',default='NVIDIA H200');p.add_argument('--gpu-rate',type=float,default=5.29)
    p.add_argument('--hours',type=float,default=11.5);p.add_argument('--ssh-public-key',type=Path)
    args=p.parse_args()
    if args.command=='prepare':result=prepare(args.deployment,args.gpu_id,args.gpu_rate,args.ssh_public_key,args.hours)
    elif args.command=='watch':result=Deployment(args.deployment).watch()
    else:
        with lock(args.deployment/'deployment.lock'):
            d=Deployment(args.deployment)
            if args.command=='reserve':result=d.reserve()
            elif args.command=='create':result=d.create(args.role)
            elif args.command=='status':result=d.reconcile()
            elif args.command=='health':result=d.guard_health()
            elif args.command=='finish':result=d.guard_health(finish=True)
            else:result=d.delete(args.role)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
