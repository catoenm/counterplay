"""Explicit, journaled RunPod rental operations for the cardinal GPU experiment.

No network or mutations on import. Creates are sent once; an unknown outcome is
reconciled by reads, never by another create. Provider responses are redacted.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
import re
from pathlib import Path
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from selfplay.frontier_api import api_key
from scripts.cloud.train_program_rl import atomic_json, digest, load
from scripts.runpod_pilot import ledger_commitment

API = "https://api.runpod.io/v2"
PROJECT = "cardinal-gpu"
RESERVATION = 80.0
PUBLIC_ENV = ("SELFPLAY_PROJECT", "SELFPLAY_INTENT", "SELFPLAY_ROLE")


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def lock(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def pod_id(value):
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value) is None:
        raise ValueError("Invalid provider resource ID")
    return value


def identity(pod):
    pod = dict(pod)
    cost = pod.get("cost")
    if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
        pod["cost"] = None
    return {**{k: pod.get(k) for k in ("id", "name", "status", "cost", "createdAt", "startedAt", "ssh", "gpu", "cpu", "dataCenterId")},
            "tags": {k: pod.get("env", {}).get(k) for k in PUBLIC_ENV}}


def matches(pod, intent, role):
    if not isinstance(pod, dict) or not isinstance(pod.get("env"), dict):
        return False
    env = pod["env"]
    return (pod.get("name") == f"selfplay-cardinal-{intent}-{role}"
            and env.get("SELFPLAY_PROJECT") == PROJECT
            and env.get("SELFPLAY_INTENT") == intent and env.get("SELFPLAY_ROLE") == role)


class ProviderError(RuntimeError):
    def __init__(self, status):
        self.status = status
        super().__init__(f"RunPod HTTP status {status}; response body withheld")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise RuntimeError("Authenticated redirect rejected")


class Client:
    def __init__(self, key=None):
        self.key = key or api_key("RUNPOD_API_KEY")
        self.opener = urllib.request.build_opener(NoRedirect)

    def call(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(API + path, data=data, method=method,
            headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        try:
            with self.opener.open(request, timeout=45) as response:
                raw = response.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as error:
            raise ProviderError(error.code) from None

    def pods(self):
        rows, cursor, seen = [], None, set()
        while True:
            query = {"includeClusterPods": "true"}
            if cursor:
                query["cursor"] = cursor
            page = self.call("GET", "/pods?" + urllib.parse.urlencode(query))
            rows.extend(page["pods"])
            pagination = page["pagination"]
            if not pagination["hasNextPage"]:
                return rows
            cursor = pagination["nextCursor"]
            if not cursor or cursor in seen:
                raise RuntimeError("Invalid inventory pagination")
            seen.add(cursor)


def prepare(folder, gpu_id, gpu_rate, ssh_public_key, hours=17):
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError("Deployment directory already exists")
    if not 0 < gpu_rate <= 5.29 or not 0 < hours <= 17:
        raise ValueError("Rental rate/duration exceed this reviewed envelope")
    guard = ROOT / "scripts/cloud/cardinal_guard.py"
    if not guard.is_file():
        raise ValueError("Remote guard is not implemented")
    intent = uuid.uuid4().hex
    deadline = time.time() + hours * 3600
    estimate = (gpu_rate + .06 + 170 * .10 / (28*24)) * hours
    if not math.isfinite(estimate) or estimate + 10 > RESERVATION:
        raise ValueError("Estimate plus contingency exceeds reservation")
    public = Path(ssh_public_key).read_text().strip()
    if not public.startswith("ssh-ed25519 ") or "\n" in public:
        raise ValueError("Expected one project Ed25519 public key")
    folder.mkdir(parents=True)
    private = ROOT / ".secrets" / "cardinal-gpu" / intent
    private.mkdir(parents=True, mode=0o700)
    os.chmod(private, 0o700)
    secret_file = private / "guard.json"
    fd = os.open(secret_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"token": secrets.token_urlsafe(40)}, f)
    plan = {"intent": intent, "project": PROJECT, "created_at": now(),
            "deadline_unix": deadline, "maximum_hours": hours,
            "reservation_usd": RESERVATION, "estimate_usd": estimate,
            "gpu_id": gpu_id, "maximum_gpu_rate_usd_hour": gpu_rate,
            "cpu_id": "cpu3c", "cpu_vcpu": 2, "maximum_cpu_rate_usd_hour": .06,
            "gpu_disk_gb": 160, "guard_disk_gb": 10,
            "gpu_image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
            "guard_image": "python:3.12-slim", "ssh_public_key": public,
            "guard_source_sha256": digest(guard), "guard_secret_path": str(secret_file),
            "cleanup": "Independent CPU deadline guard plus local exact-identity backup; DELETE, not stop. API outages are not a provider-enforced billing cap."}
    atomic_json(folder / "plan.json", plan)
    atomic_json(folder / "state.json", {"plan_sha256": digest(folder / "plan.json"),
        "created_at": now(), "reservation_applied": False, "roles": {}, "events": []})
    return {"deployment": str(folder), "intent": intent, "estimate_usd": estimate,
            "reservation_usd": RESERVATION, "deadline_unix": deadline}


class Deployment:
    def __init__(self, folder, client=None):
        self.folder = Path(folder)
        self.plan = load(self.folder / "plan.json")
        self.state = load(self.folder / "state.json")
        if self.state["plan_sha256"] != digest(self.folder / "plan.json"):
            raise ValueError("Deployment plan changed")
        self.client = client or Client()

    def save(self):
        self.state["updated_at"] = now()
        atomic_json(self.folder / "state.json", self.state)

    def reserve(self):
        with lock(ROOT / "runs/external-budget.lock"):
            budget = load(ROOT / "budget.json")
            records = budget.setdefault("cloud_reservations", {})
            intent = self.plan["intent"]
            if intent not in records:
                total = float(budget.get("cloud_spend_to_date", 0)) + float(budget.get("cloud_reserved_usd", 0))
                total += sum(ledger_commitment(load(p)) for p in (ROOT / "runs").glob("*ledger.json"))
                if total + RESERVATION > min(100, budget["approved_pilot_cap"]):
                    raise ValueError("Cumulative $100 cap would be exceeded")
                records[intent] = {"amount_usd": RESERVATION, "state": "reserved", "deployment": str(self.folder), "created_at": now()}
                budget["cloud_reserved_usd"] = float(budget.get("cloud_reserved_usd", 0)) + RESERVATION
                atomic_json(ROOT / "budget.json", budget)
            elif records[intent]["amount_usd"] != RESERVATION or records[intent]["state"] != "reserved":
                raise ValueError("Existing reservation is not compatible")
        self.state["reservation_applied"] = True
        self.save()
        return {"reserved_usd": RESERVATION}

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
        if not self.state["reservation_applied"]:
            raise ValueError("Reserve before any create")
        if self.state.get("creation_closed") or self.state.get("cleanup_started"):
            raise ValueError("Creation has been explicitly closed or cleanup started")
        if role in self.state["roles"]:
            raise ValueError("Create already attempted; use read-only reconciliation")
        if time.time() > self.plan["deadline_unix"] - 3600:
            raise ValueError("Insufficient time left in the fixed rental window")
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
            self.state["roles"][role].update(state="create_unknown", error_type=type(error).__name__)
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
                    row.update(state="creation_closed_absent")
        self.save()
        return {"intent": self.plan["intent"], "roles": self.state["roles"]}

    def gpu_resolved(self):
        row = self.state["roles"].get("gpu")
        if row is None: return True  # No create was ever attempted or discovered.
        if row.get("matching_ids"): return False
        known = set(row.get("known_ids", []))
        if known and set(row.get("confirmed_404_ids", [])) == known:
            return True
        return self.state.get("creation_closed") is True and not known

    def cleanup_if_needed(self, force=False):
        active = [r for r in self.state["roles"].values() if r.get("matching_ids")]
        unsafe = any(r.get("rate_within_plan") is not True or r.get("duplicate_intent_anomaly") for r in active)
        if not (force or unsafe or self.state.get("cleanup_started") or time.time() >= self.plan["deadline_unix"]): return
        # Any cleanup decision permanently forbids later create calls, but only
        # an explicit authenticated finish resolves a never-observed unknown POST.
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "reserve", "create", "status", "health", "finish", "delete", "watch"))
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--role", choices=("guard", "gpu"))
    parser.add_argument("--gpu-id", default="NVIDIA H100 NVL")
    parser.add_argument("--gpu-rate", type=float, default=3.19)
    parser.add_argument("--hours", type=float, default=17)
    parser.add_argument("--ssh-public-key", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.deployment, args.gpu_id, args.gpu_rate, args.ssh_public_key, args.hours)
    elif args.command == "watch":
        result = Deployment(args.deployment).watch()
    else:
        with lock(args.deployment / "deployment.lock"):
            d = Deployment(args.deployment)
            if args.command == "reserve": result = d.reserve()
            elif args.command == "create": result = d.create(args.role)
            elif args.command == "status": result = d.reconcile()
            elif args.command == "health": result = d.guard_health()
            elif args.command == "finish": result = d.guard_health(finish=True)
            else: result = d.delete(args.role)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
