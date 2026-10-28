"""Single $90 planning-curriculum lease; no operations occur on import.

This controller owns its operations. It imports only pure legacy primitives; it
never instantiates or dispatches the previous controller. Creates are durable
single attempts. Uncertain creation retains both protection and liability.
"""
from __future__ import annotations
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import runpod_cardinal as primitives
from scripts.cloud.cardinal_guard import Guard
from scripts.runpod_pilot import ledger_commitment, finite_money
atomic_json,digest,load=primitives.atomic_json,primitives.digest,primitives.load
lock,now,pod_id,identity,matches=primitives.lock,primitives.now,primitives.pod_id,primitives.identity,primitives.matches
NoRedirect,ProviderError=primitives.NoRedirect,primitives.ProviderError
API,PROJECT=primitives.API,primitives.PROJECT
RESERVATION=90.
CUMULATIVE_CEILING=350.
STAGE_CAPS={'admission':15.,'comparison':65.,'protected':10.}
STORAGE_RATE=170*.10/(28*24)


def redact_error(raw, sensitive):
    """Preserve a bounded diagnostic body, never reflected request credentials."""
    def string(value):
        for secret in sorted(set(sensitive),key=len,reverse=True):
            if secret:value=value.replace(secret,'[REDACTED]')
        value=re.sub(r'(?i)(bearer\s+)[^\s\"<>]+',r'\1[REDACTED]',value)
        value=re.sub(r'(?i)\b(?:rpa_|sk-or-v1-|sk-)[A-Za-z0-9_-]+','[REDACTED]',value)
        return value
    def clean(value):
        if isinstance(value,dict):
            return {string(str(k)):('[REDACTED]' if re.search(r'(?i)authorization|api.?key|secret|token|password|environment|^env$|headers|request|source_b64',str(k)) else clean(v)) for k,v in value.items()}
        if isinstance(value,list):return [clean(v) for v in value]
        return string(value) if isinstance(value,str) else value
    text=raw.decode('utf-8',errors='replace')
    try:return json.dumps(clean(json.loads(text)),sort_keys=True,ensure_ascii=True)[:4096]
    except (ValueError,RecursionError):return string(text)[:4096]


class RecordedProviderError(ProviderError):
    def __init__(self,status,receipt):
        super().__init__(status);self.receipt=receipt


class Client:
    def __init__(self,key=None,receipt_dir=None,opener=None):
        self._key=key;self.receipt_dir=Path(receipt_dir) if receipt_dir else None
        self.opener=opener or urllib.request.build_opener(NoRedirect)
        self.sensitive=set()

    @property
    def key(self):
        if self._key is None:self._key=primitives.api_key('RUNPOD_API_KEY')
        return self._key

    def call(self,method,path,body=None):
        if not path.startswith('/pods') or '://' in path:raise ValueError('Only scoped pod endpoints supported')
        self.sensitive.add(self.key)
        if body is not None:
            # Redact every environment value, including the guard token/source,
            # even when an HTTP error reflects it under a different JSON key.
            self.sensitive.update(str(v) for v in body.get('env',{}).values())
        data=json.dumps(body).encode() if body is not None else None
        request=urllib.request.Request(API+path,data=data,method=method,
            headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
        try:
            with self.opener.open(request,timeout=45) as response:
                raw=response.read();return json.loads(raw) if raw else None
        except urllib.error.HTTPError as error:
            raw=error.read(65537);truncated=len(raw)>65536;raw=raw[:65536]
            receipt={'at':now(),'method':method,'path':path,'http_status':error.code,
                'captured_body_bytes':len(raw),'body_truncated':truncated,
                'captured_body_sha256':hashlib.sha256(raw).hexdigest(),
                'redacted_body':('[Oversized body withheld: truncation could split a reflected credential]'
                    if truncated else redact_error(raw,self.sensitive))}
            if self.receipt_dir:
                self.receipt_dir.mkdir(parents=True,exist_ok=True)
                target=self.receipt_dir/(uuid.uuid4().hex+'.json')
                with target.open('x') as stream:json.dump(receipt,stream,sort_keys=True,allow_nan=False)
                receipt['receipt_path']=str(target)
            raise RecordedProviderError(error.code,receipt) from None

    def pods(self):
        rows=[];cursor=None;seen=set()
        for _ in range(100):
            query={'includeClusterPods':'true'}
            if cursor:query['cursor']=cursor
            page=self.call('GET','/pods?'+urllib.parse.urlencode(query))
            if not isinstance(page.get('pods'),list):raise ValueError('Invalid inventory rows')
            rows.extend(page['pods']);pagination=page['pagination']
            if pagination['hasNextPage'] is False:return rows
            cursor=pagination['nextCursor']
            if not isinstance(cursor,str) or not cursor or cursor in seen:raise ValueError('Invalid inventory pagination')
            seen.add(cursor)
        raise ValueError('Inventory pagination exceeded bounded limit')


def validate(plan):
    if (plan.get('phase')!='planning_curriculum' or plan.get('project')!=PROJECT
            or plan.get('maximum_hours')!=12 or plan.get('reservation_usd')!=90
            or plan.get('stage_caps_usd')!=STAGE_CAPS or plan.get('protected_export_seconds')!=3600
            or plan.get('cpu_id')!='cpu5c' or plan.get('cpu_vcpu')!=4
            or (plan.get('gpu_disk_gb'),plan.get('guard_disk_gb'))!=(160,10)
            or plan.get('maximum_cpu_rate_usd_hour')!=.14):
        raise ValueError('Planning lease envelope differs')
    if not isinstance(plan.get('intent'),str) or re.fullmatch('[0-9a-f]{32}',plan['intent']) is None:
        raise ValueError('Invalid lease identity')
    rate=finite_money(plan['maximum_gpu_rate_usd_hour'])
    if not 0<rate<=5.29:raise ValueError('GPU rate ceiling differs')
    created=datetime.fromisoformat(plan['created_at'])
    if created.tzinfo is None:raise ValueError('Timezone-aware creation required')
    deadline=plan['deadline_unix']
    if (type(deadline) not in (int,float) or not math.isfinite(deadline) or int(deadline)!=deadline
            or not 12*3600-1<deadline-created.timestamp()<=12*3600):
        raise ValueError('Initial immutable deadline differs')
    if (plan.get('gpu_hourly_rate')!=rate or plan.get('guard_hourly_rate')!=.14
            or plan.get('storage_hourly_rate')!=STORAGE_RATE or plan.get('other_reserved_cost_usd')!=0
            or plan.get('gpu_image')!='runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404'
            or plan.get('guard_image')!='python:3.12-slim'):
        raise ValueError('Reviewed tariff/image identity differs')
    estimate=(rate+.14+STORAGE_RATE)*12
    if abs(finite_money(plan['estimate_usd'])-estimate)>1e-9 or estimate+10>90:
        raise ValueError('Estimate and protected allowance exceed reservation')
    if (plan['guard_source_sha256']!=digest(ROOT/'scripts/cloud/cardinal_guard.py')
            or plan['primitives_sha256']!=digest(Path(primitives.__file__))
            or plan['controller_sha256']!=digest(Path(__file__))):
        raise ValueError('Reviewed lifecycle source changed')
    if not Path(plan['guard_secret_path']).is_absolute():raise ValueError('Absolute private token path required')


def prepare(folder,ssh_public_key,gpu_id='NVIDIA H200',gpu_rate=5.29):
    """First write only. Call immediately before allocation, not during coding."""
    folder=Path(folder).resolve()
    if folder.exists():raise FileExistsError('New deployment directory required')
    public=Path(ssh_public_key).read_text().strip()
    if not public.startswith('ssh-ed25519 ') or '\n' in public:raise ValueError('Project Ed25519 public key required')
    created=time.time();intent=uuid.uuid4().hex
    private=ROOT/'.secrets/cardinal-gpu'/intent;secret_file=private/'guard.json'
    plan={'intent':intent,'phase':'planning_curriculum','project':PROJECT,
        'created_at':datetime.fromtimestamp(created,timezone.utc).isoformat(),
        'deadline_unix':math.floor(created+12*3600),'maximum_hours':12,'reservation_usd':90.,
        'stage_caps_usd':dict(STAGE_CAPS),'protected_export_seconds':3600,
        'gpu_id':gpu_id,'maximum_gpu_rate_usd_hour':gpu_rate,'maximum_cpu_rate_usd_hour':.14,
        'gpu_hourly_rate':gpu_rate,'guard_hourly_rate':.14,'storage_hourly_rate':STORAGE_RATE,
        'other_reserved_cost_usd':0.,'estimate_usd':(gpu_rate+.14+STORAGE_RATE)*12,
        'cpu_id':'cpu5c','cpu_vcpu':4,'gpu_disk_gb':160,'guard_disk_gb':10,
        'gpu_image':'runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404','guard_image':'python:3.12-slim',
        'ssh_public_key':public,'guard_secret_path':str(secret_file),
        'guard_source_sha256':digest(ROOT/'scripts/cloud/cardinal_guard.py'),
        'primitives_sha256':digest(Path(primitives.__file__)),'controller_sha256':digest(Path(__file__)),
        'cleanup':'Independent exact-identity deadline guard and local backup; unknown creates retain protection/liability. Provider API outages are not a billing cap.'}
    validate(plan)
    Guard(None,intent,'offline-preparation-check',plan['deadline_unix'],'x'*40,plan['guard_source_sha256'],clock=lambda:created)
    folder.mkdir(parents=True);private.mkdir(parents=True,mode=0o700);os.chmod(private,0o700)
    fd=os.open(secret_file,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as stream:json.dump({'token':secrets.token_urlsafe(40)},stream)
    atomic_json(folder/'plan.json',plan)
    atomic_json(folder/'state.json',{'plan_sha256':digest(folder/'plan.json'),'created_at':now(),
        'reservation_applied':False,'roles':{},'events':[]})
    return {'deployment':str(folder),'intent':intent,'deadline_unix':plan['deadline_unix'],
        'reservation_usd':90.,'estimate_usd':plan['estimate_usd'],'actual_guard_constructor_passed':True}


def budget_total(budget):
    held=finite_money(budget.get('cloud_reserved_usd',0));listed=0.
    for row in budget.get('cloud_reservations',{}).values():
        amount=finite_money(row['amount_usd'])
        if row['state'] in ('reserved','terminated_pending_billing'):listed+=amount
        elif amount:raise ValueError('Unknown outstanding reservation state')
    if abs(held-listed)>1e-8:raise ValueError('Aggregate reserved liability differs')
    return finite_money(budget.get('cloud_spend_to_date',0))+held+sum(
        ledger_commitment(load(p)) for p in sorted((ROOT/'runs').glob('*ledger.json')))


class Deployment:
    def __init__(self,folder,client=None):
        self.folder=Path(folder).resolve();self.plan=load(self.folder/'plan.json');self.state=load(self.folder/'state.json')
        if self.state['plan_sha256']!=digest(self.folder/'plan.json'):raise ValueError('Deployment plan changed')
        validate(self.plan)
        self.client=client or Client(receipt_dir=self.folder/'provider-errors')

    def save(self):
        self.state['updated_at']=now();atomic_json(self.folder/'state.json',self.state)

    def check_reservation(self):
        with lock(ROOT/'runs/external-budget.lock'):
            budget=load(ROOT/'budget.json');row=budget.get('cloud_reservations',{}).get(self.plan['intent'],{})
            if (row.get('amount_usd')!=90 or row.get('state')!='reserved'
                    or row.get('deployment')!=str(self.folder) or row.get('phase')!='planning_curriculum'
                    or row.get('plan_sha256')!=digest(self.folder/'plan.json')):
                raise ValueError('Exact planning reservation required')
            if budget_total(budget)>min(350,finite_money(budget['approved_pilot_cap']))+1e-9:
                raise ValueError('Current cumulative cap exceeded')

    def reserve(self):
        with lock(ROOT/'runs/external-budget.lock'):
            budget=load(ROOT/'budget.json');records=budget.setdefault('cloud_reservations',{});intent=self.plan['intent']
            total=budget_total(budget);cap=min(350,finite_money(budget['approved_pilot_cap']))
            if intent not in records:
                if total+90>cap+1e-9:raise ValueError('Cumulative cap would be exceeded')
                records[intent]={'amount_usd':90.,'state':'reserved','phase':'planning_curriculum',
                    'deployment':str(self.folder),'created_at':now(),'plan_sha256':digest(self.folder/'plan.json')}
                budget['cloud_reserved_usd']=finite_money(budget.get('cloud_reserved_usd',0))+90
                atomic_json(ROOT/'budget.json',budget)
            elif (records[intent].get('amount_usd')!=90 or records[intent].get('state')!='reserved'
                    or records[intent].get('deployment')!=str(self.folder)
                    or records[intent].get('plan_sha256')!=digest(self.folder/'plan.json')
                    or records[intent].get('phase')!='planning_curriculum' or total>cap+1e-9):
                raise ValueError('Existing reservation is incompatible or cap exceeded')
        self.state['reservation_applied']=True;self.save()
        return {'reserved_usd':90.,'cumulative_commitment_usd':budget_total(budget),'old_liabilities_unchanged':True}

    def body(self, role):
        p = self.plan
        env = {"SELFPLAY_PROJECT": PROJECT, "SELFPLAY_INTENT": p["intent"], "SELFPLAY_ROLE": role}
        body = {"name": f"selfplay-cardinal-{p['intent']}-{role}", "cloud": "SECURE",
                "env": env, "mounts": {}, "startJupyter": False}
        if role == "guard":
            guard = ROOT / "scripts/cloud/cardinal_guard.py"
            if digest(guard) != p["guard_source_sha256"]:
                raise ValueError("Guard source changed after review")
            env.update(RUNPOD_API_KEY=self.client.key,
                SELFPLAY_GUARD_TOKEN=load(p["guard_secret_path"])["token"],
                SELFPLAY_GUARD_SHA256=p["guard_source_sha256"],
                SELFPLAY_DEADLINE_UNIX=str(p["deadline_unix"]),
                SELFPLAY_GUARD_SOURCE_B64=base64.b64encode(guard.read_bytes()).decode())
            boot = "import os,base64; p='/tmp/cardinal_guard.py'; open(p,'wb').write(base64.b64decode(os.environ['SELFPLAY_GUARD_SOURCE_B64'])); os.execv('/usr/local/bin/python',['python','-u',p])"
            body.update(image=p["guard_image"], cpu={"id": p["cpu_id"], "vcpuCount": p["cpu_vcpu"]},
                disk=p["guard_disk_gb"], ports=["8080/http"], entrypoint=["python", "-u", "-c"], cmd=[boot])
        elif role == "gpu":
            env["PUBLIC_KEY"] = p["ssh_public_key"]
            body.update(image=p["gpu_image"], gpu={"id": p["gpu_id"], "count": 1, "minCudaVersion": "12.8"},
                        disk=p["gpu_disk_gb"], ports=["22/tcp"], startSsh=True)
        else:
            raise ValueError("Unknown deployment role")
        return body

    def guard_health(self, finish=False):
        if finish and not self.gpu_resolved():
            raise ValueError('Cannot finish guard protection while GPU creation or deletion remains unresolved')
        row = self.state["roles"].get("guard", {})
        pod_id = row.get("pod_id")
        if not pod_id:
            raise ValueError("Guard has not been identified")
        token = load(self.plan["guard_secret_path"])["token"]
        payload = json.dumps({"intent": self.plan["intent"], "creation_closed": True}).encode() if finish else None
        request = urllib.request.Request(f"https://{pod_id}-8080.proxy.runpod.net/" + ("finish" if finish else "health"),
            data=payload, headers={"X-Guard-Token": token, "Content-Type": "application/json",
                                  "User-Agent": "Mozilla/5.0"})
        with urllib.request.build_opener(NoRedirect).open(request, timeout=20) as response:
            health = json.load(response)
        if health.get("intent") != self.plan["intent"] or health.get("deadline_unix") != self.plan["deadline_unix"] or health.get("source_sha256") != self.plan["guard_source_sha256"]:
            raise ValueError("Guard readiness identity differs")
        if finish:
            if health.get("creation_closed") is not True:
                raise ValueError("Guard did not acknowledge creation closure")
            self.state["creation_closed"] = True
        self.state["guard_health"] = health
        self.state["guard_checked_at_unix"] = time.time()
        self.save()
        return health

    def create(self, role):
        self.check_reservation()
        if not self.state["reservation_applied"]:
            raise ValueError("Reserve before any create")
        if self.state.get("creation_closed") or self.state.get("cleanup_started"):
            raise ValueError("Creation has been explicitly closed or cleanup started")
        if role in self.state["roles"]:
            raise ValueError("Create already attempted; use read-only reconciliation")
        if time.time() > self.plan["deadline_unix"] - 3600:
            raise ValueError("Insufficient time left in the fixed rental window")
        admission_end=datetime.fromisoformat(self.plan['created_at']).timestamp()+15/(
            self.plan['gpu_hourly_rate']+self.plan['guard_hourly_rate']+self.plan['storage_hourly_rate'])*3600
        if time.time()>=admission_end:
            raise ValueError('Conservative admission budget exhausted before create')
        if role == "gpu":
            health = self.guard_health()
            if (health.get("ready") is not True
                    or health.get("guard_pod_id") != self.state["roles"]["guard"]["pod_id"]
                    or health.get("finish_requested") or health.get("self_delete_requested")
                    or health.get("creation_closed") or health.get("duplicate_intent_anomaly")
                    or not isinstance(health.get("last_inventory_unix"), (int, float))
                    or not 0 <= time.time()-health["last_inventory_unix"] < 60):
                raise ValueError("Independent guard is not freshly armed and ready")
            self.reconcile()
            guard = self.state["roles"].get("guard", {})
            if guard.get("state") != "identified" or guard.get("rate_within_plan") is not True or guard.get("duplicate_intent_anomaly"):
                self.cleanup_if_needed(force=True)
                raise ValueError("Guard cost or unique identity is unverified; GPU creation blocked")
        body = self.body(role)
        request_hash = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        self.state["roles"][role] = {"state": "create_attempted", "request_sha256": request_hash, "attempted_at": now()}
        self.save()  # Durable before exactly one POST.
        try:
            pod = self.client.call("POST", "/pods", body)
            if isinstance(pod, dict) and "pod" in pod:
                pod = pod["pod"]
            if not matches(pod, self.plan["intent"], role):
                raise ValueError("Created response identity is not the requested project role")
            self.state["roles"][role].update(state="identified", pod_id=pod_id(pod["id"]), latest=identity(pod))
        except Exception as error:
            self.state["roles"][role].update(state="create_unknown", error_type=type(error).__name__,
                provider_error=getattr(error, "receipt", None))
            self.save()
            raise RuntimeError("Create outcome requires read-only reconciliation; no repeat POST") from None
        self.save()
        self.reconcile()
        self.cleanup_if_needed()
        return {"intent": self.plan["intent"], "roles": self.state["roles"]}

    def watch(self):
        """Independent local backup; no creates, no unattended reservation release."""
        while True:
            try:
                with lock(self.folder / "deployment.lock"):
                    current = Deployment(self.folder, self.client)
                    current.reconcile()
                    current.cleanup_if_needed()
                    guard = current.state["roles"].get("guard", {})
                    if guard.get("state") == "terminated_confirmed" and current.gpu_resolved():
                        return {"state": "all_identified_resources_terminated", "reservation_retained": True}
            except Exception as error:
                print(json.dumps({"at": now(), "watch_error_type": type(error).__name__}), flush=True)
            time.sleep(30)

    def rate(self, pod, role):
        value = pod.get("cost")
        verified = type(value) in (int, float) and math.isfinite(value) and value >= 0
        maximum = self.plan["maximum_gpu_rate_usd_hour" if role == "gpu" else "maximum_cpu_rate_usd_hour"]
        disk = self.plan["gpu_disk_gb" if role == "gpu" else "guard_disk_gb"] * .10 / (28*24)
        return {"verified": verified, "within_plan": verified and value <= maximum + disk + 1e-6}

    def reconcile(self):
        """Read every page and directly confirm every previously observed ID."""
        pods = self.client.pods()
        for role in ("guard", "gpu"):
            found = {pod_id(p["id"]): p for p in pods if matches(p, self.plan["intent"], role)}
            if role not in self.state["roles"] and not found:
                continue
            row = self.state["roles"].setdefault(role, {"state": "discovered_by_exact_intent"})
            known = set(row.get("known_ids", [])) | set(row.get("matching_ids", [])) | set(found)
            if row.get("pod_id"): known.add(row["pod_id"])
            absent = set()
            for identifier in sorted(known - set(found)):
                try:
                    p = self.client.call("GET", "/pods/" + pod_id(identifier))
                    if not matches(p, self.plan["intent"], role) or p.get("id") != identifier:
                        raise ValueError("Previously identified deletion identity changed")
                    found[identifier] = p
                except ProviderError as error:
                    if error.status != 404: raise
                    absent.add(identifier)
            row.update(known_ids=sorted(known), matching_ids=sorted(found), confirmed_404_ids=sorted(absent))
            row["duplicate_intent_anomaly"] = row.get("duplicate_intent_anomaly", False) or len(known)>1
            if found:
                rates = {identifier:self.rate(p, role) for identifier,p in found.items()}
                row.update(rate_checks=rates, rate_verified=all(r["verified"] for r in rates.values()),
                           rate_within_plan=all(r["within_plan"] for r in rates.values()))
                if len(found)>1:
                    row.update(state="ambiguous_multiple", latest_by_id={i:identity(p) for i,p in found.items()})
                else:
                    identifier,p = next(iter(found.items()))
                    row.update(state="identified", pod_id=identifier, latest=identity(p))
            elif known and absent == known:
                row.update(state="terminated_confirmed", confirmed_at=now())
            else:
                row["inventory_no_match_at"] = now()
                if self.state.get("creation_closed"):
                    row["creation_closed_without_resolution"] = True
        self.save()
        return {"intent": self.plan["intent"], "roles": self.state["roles"]}

    def gpu_resolved(self):
        row = self.state["roles"].get("gpu")
        if row is None: return True  # No create was ever attempted or discovered.
        if row.get("matching_ids"): return False
        known = set(row.get("known_ids", []))
        if known and set(row.get("confirmed_404_ids", [])) == known:
            return True
        # A human/local finish closes future dispatch; it cannot prove the
        # outcome of a POST whose provider ID has never been observed.
        return False

    def cleanup_if_needed(self, force=False):
        active = [r for r in self.state["roles"].values() if r.get("matching_ids")]
        unsafe = any(r.get("rate_within_plan") is not True or r.get("duplicate_intent_anomaly") for r in active)
        if not (force or unsafe or self.state.get("cleanup_started") or time.time() >= self.plan["deadline_unix"]): return
        # Cleanup permanently forbids later creates. Neither an explicit finish
        # nor an empty inventory resolves a never-observed unknown POST.
        self.state["cleanup_started"] = True
        self.save()
        gpu = self.state["roles"].get("gpu", {})
        if gpu.get("known_ids"): self.delete("gpu")
        if self.gpu_resolved() and self.state["roles"].get("guard", {}).get("known_ids"):
            self.delete("guard")

    def delete(self, role):
        self.reconcile()
        if role == "guard" and not self.gpu_resolved():
            raise ValueError("Guard protection required while GPU is live or creation remains ambiguous")
        row = self.state["roles"].get(role, {})
        identifiers = row.get("known_ids", [])
        if not identifiers: raise ValueError("No exact identified project resource to delete")
        for identifier in identifiers:
            identifier = pod_id(identifier)
            try:
                p = self.client.call("GET", "/pods/" + identifier)
                if not matches(p, self.plan["intent"], role) or p.get("id") != identifier:
                    raise ValueError("Deletion identity does not match this project")
                self.client.call("DELETE", "/pods/" + identifier)
                # DELETE acceptance alone is not termination confirmation.
                try:
                    remaining = self.client.call("GET", "/pods/" + identifier)
                    if not matches(remaining, self.plan["intent"], role) or remaining.get("id") != identifier:
                        raise ValueError("Deletion confirmation identity differs")
                except ProviderError as error:
                    if error.status != 404: raise
            except ProviderError as error:
                if error.status != 404: raise
            self.state["events"].append({"action":"delete_requested", "role":role, "pod_id":identifier, "at":now()})
            self.save()
        return self.reconcile()



def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','reserve','create','status','health','finish','delete','watch'))
    p.add_argument('--deployment',type=Path,required=True);p.add_argument('--role',choices=('guard','gpu'))
    p.add_argument('--ssh-public-key',type=Path);p.add_argument('--gpu-id',default='NVIDIA H200')
    p.add_argument('--gpu-rate',type=float,default=5.29);args=p.parse_args()
    if args.command=='prepare':result=prepare(args.deployment,args.ssh_public_key,args.gpu_id,args.gpu_rate)
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
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
