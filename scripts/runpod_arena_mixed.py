"""Three independently guarded leases funded together under the $350 project cap.

Training reserves $125 plus $50 mixed evaluation and $65 coding evaluation. Each
phase gets its initial immutable deadline only after its predecessor is retrieved
and terminated; future allocations have no running clock or resource attached.
No provider operation or budget write occurs on import or local preparation.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import secrets
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import runpod_cardinal as base
from scripts.cloud.cardinal_guard import Guard
from scripts.runpod_pilot import ledger_commitment,finite_money

PHASES={'mixed_training':{'hours':20.,'reservation':125.,'contingency':8.},
        'mixed_evaluation':{'hours':8.,'reservation':50.,'contingency':6.},
        'coding_evaluation':{'hours':10.,'reservation':65.,'contingency':8.}}
INTENT_KEYS={'mixed_training':'funding_training_intent','mixed_evaluation':'evaluation_intent','coding_evaluation':'coding_evaluation_intent'}
PATH_KEYS={'mixed_training':'funding_training_deployment','mixed_evaluation':'evaluation_deployment','coding_evaluation':'coding_evaluation_deployment'}
COMBINED_RESERVATION=240.
CUMULATIVE_CEILING=350.
lock,now=base.lock,base.now


def uuid_hex(value):
    return isinstance(value,str) and uuid.UUID(value).hex==value


def validate(plan):
    phase=plan.get('phase');spec=PHASES.get(phase)
    if spec is None:raise ValueError('Unknown continuation phase')
    if (plan.get('project')!=base.PROJECT or plan['maximum_hours']!=spec['hours']
            or plan['reservation_usd']!=spec['reservation'] or plan['combined_funding_usd']!=240.
            or type(plan['maximum_gpu_rate_usd_hour']) not in (int,float)
            or not 0<plan['maximum_gpu_rate_usd_hour']<=5.29
            or plan['maximum_cpu_rate_usd_hour']!=.14 or plan['cpu_id']!='cpu5c' or plan['cpu_vcpu']!=4
            or (plan['gpu_disk_gb'],plan['guard_disk_gb'])!=(160,10)):
        raise ValueError('Continuation phase envelope differs')
    for key in ('intent','continuation_id','funding_training_intent','evaluation_intent','coding_evaluation_intent'):
        if not uuid_hex(plan[key]):raise ValueError('Invalid continuation identity')
    if (len({plan[k] for k in INTENT_KEYS.values()})!=3
            or plan['intent']!=plan[INTENT_KEYS[phase]]):
        raise ValueError('Phase/funding identity differs')
    created_time=datetime.fromisoformat(plan['created_at'])
    if created_time.tzinfo is None:raise ValueError('Timezone-aware initial creation time required')
    created=created_time.timestamp();deadline=plan['deadline_unix']
    if (type(deadline) not in (int,float) or not math.isfinite(deadline) or int(deadline)!=deadline
            or not 0<deadline-created<=spec['hours']*3600 or deadline-created>=24*3600):
        raise ValueError('Initial phase deadline differs')
    estimate=(plan['maximum_gpu_rate_usd_hour']+.14+170*.10/(28*24))*spec['hours']
    if (type(plan['estimate_usd']) not in (int,float) or not math.isfinite(plan['estimate_usd'])
            or abs(plan['estimate_usd']-estimate)>1e-9 or estimate+spec['contingency']>spec['reservation']):
        raise ValueError('Phase estimate plus contingency exceeds reservation')
    if (any(not Path(plan[k]).is_absolute() for k in PATH_KEYS.values())
            or len({plan[k] for k in PATH_KEYS.values()})!=3
            or plan['inherited_controller_sha256']!=base.digest(Path(base.__file__))
            or plan['mixed_controller_sha256']!=base.digest(Path(__file__))
            or plan['guard_source_sha256']!=base.digest(ROOT/'scripts/cloud/cardinal_guard.py')):
        raise ValueError('Frozen lifecycle source or future deployment differs')


def terminated_predecessor(folder,phase):
    """Read terminal retrieval and exact-ID deletion evidence; never delete here."""
    folder=Path(folder).resolve();plan=base.load(folder/'plan.json');state=base.load(folder/'state.json')
    receipt=base.load(folder/'supervisor-status.json');origin=base.load(folder/'supervisor-origin.json')
    if (state['plan_sha256']!=base.digest(folder/'plan.json') or state.get('creation_closed') is not True
            or set(state['roles'])!={'guard','gpu'} or receipt.get('scientific_artifacts_verified') is not True
            or receipt.get('gpu_404_confirmed') is not True or receipt.get('guard_404_confirmed') is not True
            or receipt.get('pull_returncode')!=0 or receipt.get('expected_manifest_sha256')!=origin['manifest_sha256']
            or origin['deployment']['intent']!=plan['intent']):
        raise ValueError('Predecessor retrieval and closed creation are not verified')
    expected='training_complete' if phase=='mixed_evaluation' else 'throughput_failed'
    exit=receipt.get('remote_exit',{})
    if (receipt.get('study_state')!=expected or exit.get('exit_code')!=0
            or exit.get('verification_completed') is not True or exit.get('manifest_sha256')!=origin['manifest_sha256']):
        raise ValueError('Predecessor has not reached the required terminal phase')
    for role in ('guard','gpu'):
        row=state['roles'][role];known=set(row.get('known_ids',[]))|{row.get('pod_id')}
        if (row.get('state')!='terminated_confirmed' or None in known or not known
                or known!=set(row.get('confirmed_404_ids',[])) or row.get('matching_ids')!=[]):
            raise ValueError('Predecessor exact resource deletion remains unresolved')
    hashes={name:base.digest(folder/name) for name in ('plan.json','state.json','supervisor-status.json','supervisor-origin.json')}
    return plan,hashes


def prepare(folder,phase,ssh_public_key,previous_deployment,evaluation_deployment=None,
            coding_evaluation_deployment=None,funding_deployment=None,gpu_id='NVIDIA H200',gpu_rate=5.29):
    folder=Path(folder).resolve();previous_deployment=Path(previous_deployment).resolve()
    if folder.exists():raise FileExistsError('New deployment required')
    if phase not in PHASES:raise ValueError('Unknown phase')
    previous,evidence=terminated_predecessor(previous_deployment,phase)
    public=Path(ssh_public_key).read_text().strip()
    if not public.startswith('ssh-ed25519 ') or '\n' in public:raise ValueError('Project Ed25519 public key required')
    if phase=='mixed_training':
        if evaluation_deployment is None or coding_evaluation_deployment is None:
            raise ValueError('Both future evaluation deployment paths are required')
        paths={'funding_training_deployment':str(folder),
            'evaluation_deployment':str(Path(evaluation_deployment).resolve()),
            'coding_evaluation_deployment':str(Path(coding_evaluation_deployment).resolve())}
        if len(set(paths.values()))!=3 or any(Path(v).exists() for v in paths.values()):
            raise ValueError('Three distinct unprepared deployment paths required')
        continuation=uuid.uuid4().hex;intents={key:uuid.uuid4().hex for key in INTENT_KEYS.values()}
    else:
        funding=previous_deployment if phase=='mixed_evaluation' else Path(funding_deployment).resolve()
        master=base.load(funding/'plan.json');validate(master)
        if master['phase']!='mixed_training':raise ValueError('Future phase must use its original training funding plan')
        if phase=='mixed_evaluation' and previous['intent']!=master['funding_training_intent']:
            raise ValueError('Mixed evaluation must follow exact linked training cleanup')
        continuation=master['continuation_id'];intents={key:master[key] for key in INTENT_KEYS.values()}
        paths={key:master[key] for key in PATH_KEYS.values()}
        if folder!=Path(paths[PATH_KEYS[phase]]).resolve():
            raise ValueError('Future phase path differs from upfront funding')
    intent=intents[INTENT_KEYS[phase]];spec=PHASES[phase];created=time.time()
    private=ROOT/'.secrets/cardinal-gpu'/intent;secret_file=private/'guard.json'
    plan={'intent':intent,'project':base.PROJECT,'created_at':datetime.fromtimestamp(created,timezone.utc).isoformat(),
        'deadline_unix':float(math.floor(created+spec['hours']*3600)),'maximum_hours':spec['hours'],
        'reservation_usd':spec['reservation'],'combined_funding_usd':240.,'phase':phase,'continuation_id':continuation,
        **intents,**paths,'predecessor_deployment':str(previous_deployment),
        'predecessor_evidence_sha256':evidence,'gpu_id':gpu_id,'maximum_gpu_rate_usd_hour':gpu_rate,
        'estimate_usd':(gpu_rate+.14+170*.10/(28*24))*spec['hours'],
        'cpu_id':'cpu5c','cpu_vcpu':4,'maximum_cpu_rate_usd_hour':.14,'gpu_disk_gb':160,'guard_disk_gb':10,
        'gpu_image':'runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404','guard_image':'python:3.12-slim',
        'ssh_public_key':public,'guard_secret_path':str(secret_file),
        'guard_source_sha256':base.digest(ROOT/'scripts/cloud/cardinal_guard.py'),
        'inherited_controller_sha256':base.digest(Path(base.__file__)),
        'mixed_controller_sha256':base.digest(Path(__file__)),
        'cleanup':'Independent CPU deadline guard plus local exact-identity backup. Retrieve verified artifacts then DELETE both roles. API outages are not a provider-enforced billing cap.'}
    validate(plan)
    Guard(None,intent,'offline-preparation-check',plan['deadline_unix'],'x'*40,plan['guard_source_sha256'],clock=lambda:created)
    folder.mkdir(parents=True);private.mkdir(parents=True,mode=0o700);os.chmod(private,0o700)
    fd=os.open(secret_file,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as stream:json.dump({'token':secrets.token_urlsafe(40)},stream)
    base.atomic_json(folder/'plan.json',plan)
    base.atomic_json(folder/'state.json',{'plan_sha256':base.digest(folder/'plan.json'),'created_at':now(),
        'reservation_applied':False,'roles':{},'events':[]})
    return {'deployment':str(folder),'intent':intent,'continuation_id':continuation,'phase':phase,
        'deadline_unix':plan['deadline_unix'],'reservation_usd':spec['reservation'],'combined_funding_usd':240.,
        'actual_guard_constructor_passed':True}


def budget_total(budget):
    held=finite_money(budget.get('cloud_reserved_usd',0));listed=0.
    for row in budget.get('cloud_reservations',{}).values():
        amount=finite_money(row['amount_usd'])
        if row['state'] in ('reserved','terminated_pending_billing'):listed+=amount
        elif amount:raise ValueError('Unknown outstanding reservation state')
    if abs(held-listed)>1e-8:raise ValueError('Aggregate reserved liability differs')
    return finite_money(budget.get('cloud_spend_to_date',0))+held+sum(ledger_commitment(base.load(p)) for p in (ROOT/'runs').glob('*ledger.json'))


class Deployment(base.Deployment):
    def __init__(self,folder,client=None):
        super().__init__(folder,client);validate(self.plan)

    def expected_pair(self):
        p=self.plan
        return {p[INTENT_KEYS[phase]]:{'amount_usd':spec['reservation'],'phase':phase,
            'deployment':p[PATH_KEYS[phase]]} for phase,spec in PHASES.items()}

    def check_pair(self,records):
        for intent,expected in self.expected_pair().items():
            if intent not in records:raise ValueError('Incomplete linked reservation requires review')
            row=records[intent]
            if (row.get('continuation_id')!=self.plan['continuation_id'] or row.get('phase')!=expected['phase']
                    or Path(row['deployment']).resolve()!=Path(expected['deployment']).resolve()):
                raise ValueError('Existing linked funding identity differs')
            if expected['phase']!=self.plan['phase'] and row['state']=='terminated_pending_billing':
                if not 0<=finite_money(row['amount_usd'])<=expected['amount_usd']:
                    raise ValueError('Reviewed prior phase liability differs')
            elif expected['phase']!=self.plan['phase'] and row['state']=='settled' and row['amount_usd']==0:
                pass
            elif row['state']!='reserved' or row['amount_usd']!=expected['amount_usd']:
                raise ValueError('Existing phase funding is incompatible')

    def reserve(self):
        with lock(ROOT/'runs/external-budget.lock'):
            budget=base.load(ROOT/'budget.json');records=budget.setdefault('cloud_reservations',{})
            total=budget_total(budget);pair=self.expected_pair();present=set(pair)&set(records)
            cap=min(CUMULATIVE_CEILING,finite_money(budget['approved_pilot_cap']))
            if not present:
                if self.plan['phase']!='mixed_training':raise ValueError('Future evaluation was not funded upfront')
                if total+240>cap:raise ValueError('Combined continuation funding exceeds cumulative cap')
                for intent,expected in pair.items():
                    records[intent]={**expected,'state':'reserved','continuation_id':self.plan['continuation_id'],
                        'created_at':now(),'funded_together_usd':240.,'future_phase':expected['phase']!='mixed_training'}
                budget['cloud_reserved_usd']=finite_money(budget.get('cloud_reserved_usd',0))+240
                base.atomic_json(ROOT/'budget.json',budget)
            else:
                if present!=set(pair):raise ValueError('Incomplete paired reservation requires review')
                self.check_pair(records)
                if total>cap:raise ValueError('Current aggregate commitment exceeds cumulative cap')
        self.state['reservation_applied']=True;self.save()
        return {'phase_reserved_usd':self.plan['reservation_usd'],'combined_funding_usd':240.,'new_provider_calls':0}

    def create(self,role):
        with lock(ROOT/'runs/external-budget.lock'):
            budget=base.load(ROOT/'budget.json');self.check_pair(budget.get('cloud_reservations',{}))
            if budget_total(budget)>min(CUMULATIVE_CEILING,finite_money(budget['approved_pilot_cap'])):
                raise ValueError('Current commitment exceeds cap')
        return super().create(role)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','reserve','create','status','health','finish','delete','watch'))
    p.add_argument('--deployment',type=Path,required=True);p.add_argument('--role',choices=('guard','gpu'))
    p.add_argument('--phase',choices=tuple(PHASES));p.add_argument('--previous-deployment',type=Path)
    p.add_argument('--evaluation-deployment',type=Path);p.add_argument('--coding-evaluation-deployment',type=Path)
    p.add_argument('--funding-deployment',type=Path);p.add_argument('--ssh-public-key',type=Path)
    p.add_argument('--gpu-id',default='NVIDIA H200');p.add_argument('--gpu-rate',type=float,default=5.29)
    a=p.parse_args()
    if a.command=='prepare':result=prepare(a.deployment,a.phase,a.ssh_public_key,a.previous_deployment,a.evaluation_deployment,a.coding_evaluation_deployment,a.funding_deployment,a.gpu_id,a.gpu_rate)
    elif a.command=='watch':result=Deployment(a.deployment).watch()
    else:
        with lock(a.deployment/'deployment.lock'):
            d=Deployment(a.deployment)
            if a.command=='reserve':result=d.reserve()
            elif a.command=='create':result=d.create(a.role)
            elif a.command=='status':result=d.reconcile()
            elif a.command=='health':result=d.guard_health()
            elif a.command=='finish':result=d.guard_health(finish=True)
            else:result=d.delete(a.role)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
