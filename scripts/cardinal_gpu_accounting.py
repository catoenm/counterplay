"""Read-only, exact-pod billing capture and conservative accounting preview.

Neither command mutates budget.json, resources, or ledgers. ``capture`` makes
only GET /v2/billing/pods calls, one per previously recorded project pod ID.
Billing buckets have no published settlement marker: even after confirmed
deletion, this tool NEVER releases the deployment reservation automatically.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.runpod_pilot import ledger_commitment

API = "https://api.runpod.io/v2"
SCHEMA_URL = API + "/openapi.json"
FIELDS = ("totalAmount", "gpuAmount", "cpuAmount", "diskAmount")
TOLERANCE = Decimal("0.000001")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def money(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Invalid nonnegative USD amount")
    return Decimal(str(value))


def instant(text):
    if not isinstance(text, str):
        raise ValueError("Expected timestamp")
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("Timestamp lacks timezone")
    return value.astimezone(timezone.utc)


def stamp(value):
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def resource_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        raise ValueError("Invalid resource ID")
    return value


def write_new(path, value):
    # An evidence capture is immutable; never replace an earlier response.
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def snapshot(root, deployment):
    """Take one local snapshot under existing shared locks, with no writes."""
    root, deployment = Path(root).resolve(), Path(deployment).resolve()
    files = {}

    def read(path):
        raw = path.read_bytes()
        files[str(path)] = sha(raw)
        return json.loads(raw)

    with (deployment / "deployment.lock").open("r") as deploy_lock:
        fcntl.flock(deploy_lock, fcntl.LOCK_SH)
        with (root / "runs/external-budget.lock").open("r") as budget_lock:
            fcntl.flock(budget_lock, fcntl.LOCK_SH)
            plan = read(deployment / "plan.json")
            state = read(deployment / "state.json")
            budget = read(root / "budget.json")
            ledgers = {p.name: read(p) for p in sorted((root / "runs").glob("*ledger.json"))}
    if state["plan_sha256"] != files[str(deployment / "plan.json")]:
        raise ValueError("Deployment plan hash changed")
    if plan.get("project") != "cardinal-gpu" or not re.fullmatch(r"[0-9a-f-]{32,36}", plan.get("intent", "")):
        raise ValueError("Unexpected project intent")
    files[str(Path(__file__).resolve())] = sha(Path(__file__).read_bytes())
    helper = Path(__file__).resolve().with_name("runpod_pilot.py")
    files[str(helper)] = sha(helper.read_bytes())
    schema = root / "runs/cardinal-gpu-preflight/openapi.json"
    if schema.exists():
        files[str(schema)] = sha(schema.read_bytes())
    roles, ids = {}, set()
    for role in ("guard", "gpu"):
        row = state.get("roles", {}).get(role, {})
        known = set(row.get("known_ids", [])) | set(row.get("matching_ids", []))
        if row.get("pod_id"):
            known.add(row["pod_id"])
        # Include all budget-recorded IDs for this exact intent, including
        # duplicates that disappeared from a later live inventory.
        for recorded in budget.get("cloud_resources_created", []):
            if recorded.get("intent") != plan["intent"]:
                continue
            if recorded.get("role") not in ("guard", "gpu"):
                raise ValueError("Unknown resource role in project budget")
            location = Path(recorded["deployment"])
            if not location.is_absolute():
                location = root / location
            if location.resolve() != deployment:
                raise ValueError("Project resource deployment disagrees")
            if recorded["role"] == role:
                known.add(recorded["id"])
        known = sorted(resource_id(i) for i in known)
        if ids.intersection(known):
            raise ValueError("Same pod ID appears under both roles")
        ids.update(known)
        observations = list(row.get("latest_by_id", {}).values())
        if isinstance(row.get("latest"), dict):
            observations.append(row["latest"])
        for observed in observations:
            expected_tags = {"SELFPLAY_PROJECT": "cardinal-gpu", "SELFPLAY_INTENT": plan["intent"], "SELFPLAY_ROLE": role}
            if observed.get("id") not in known or observed.get("name") != f"selfplay-cardinal-{plan['intent']}-{role}" or observed.get("tags") != expected_tags:
                raise ValueError("Saved resource identity disagrees with project")
        roles[role] = {"ids": known, "state": row.get("state", "not_attempted"),
            "all_known_ids_confirmed_404": bool(known) and set(known) <= set(row.get("confirmed_404_ids", [])) and not row.get("matching_ids"),
            "ambiguous_unobserved_create": bool(row) and not known and row.get("state") != "creation_closed_absent",
            "duplicate_intent_anomaly": bool(row.get("duplicate_intent_anomaly")) or len(known) > 1}
    return {"plan": plan, "state": state, "budget": budget, "ledgers": ledgers,
            "roles": roles, "inputs": files, "deployment": str(deployment)}


def validate_response(response, query):
    """Do not accept unscoped, overlapping, truncated, or inconsistent totals."""
    records, metadata = response["records"], response["metadata"]
    echoed = metadata["query"]
    if (not isinstance(records, list) or echoed.get("podId") != query["podId"]
            or echoed.get("bucketSize") != query["bucketSize"]
            or any(instant(echoed[k]) != instant(query[k]) for k in ("startTime", "endTime"))):
        raise ValueError("Billing query metadata does not match exact request")
    start, end = instant(query["startTime"]), instant(query["endTime"])
    if start >= end or query["bucketSize"] != "hour" or any(t.minute or t.second or t.microsecond for t in (start, end)):
        raise ValueError("Billing query must be a boundary-aligned hourly range")
    total = {key: Decimal(0) for key in FIELDS}
    previous = None
    for row in sorted(records, key=lambda r: instant(r["startTime"])):
        a, b = instant(row["startTime"]), instant(row["endTime"])
        if row["podId"] != query["podId"] or a < start or b > end or a >= b or (b-a).total_seconds() != 3600 or a.minute or a.second or a.microsecond:
            raise ValueError("Billing row has wrong ID or interval")
        if previous is not None and a < previous:
            raise ValueError("Duplicate or overlapping billing buckets")
        previous = b
        amounts = {key: money(row[key]) for key in FIELDS}
        if abs(amounts["totalAmount"] - sum(amounts[k] for k in FIELDS[1:])) > TOLERANCE:
            raise ValueError("Billing components do not sum to total")
        total = {key: total[key] + amounts[key] for key in FIELDS}
    if type(metadata["recordCount"]) is not int or metadata["recordCount"] != len(records) or type(metadata["uniquePodCount"]) is not int or metadata["uniquePodCount"] != int(bool(records)):
        raise ValueError("Billing metadata counts disagree")
    if any(abs(money(metadata["totals"][key]) - total[key]) > TOLERANCE for key in FIELDS):
        raise ValueError("Billing metadata amounts disagree")
    return {key: float(value) for key, value in total.items()}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise RuntimeError("Authenticated redirect rejected")


def billing_get(query):
    # Lazy credential access: offline preview does not read credentials.
    from selfplay.frontier_api import api_key
    request = urllib.request.Request(API + "/billing/pods?" + urllib.parse.urlencode(query),
        method="GET", headers={"Authorization": "Bearer " + api_key("RUNPOD_API_KEY"), "User-Agent": "Mozilla/5.0"})
    with urllib.request.build_opener(NoRedirect).open(request, timeout=45) as response:
        raw = response.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("Billing response exceeds size limit")
    return json.loads(raw)


def capture(snap, at, getter=billing_get):
    from datetime import timedelta
    start = instant(snap["plan"]["created_at"]).replace(minute=0, second=0, microsecond=0)
    end = at.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    if end <= start:
        raise ValueError("Capture clock predates deployment")
    entries = []
    for role, identity in snap["roles"].items():
        for identifier in identity["ids"]:
            query = {"podId": identifier, "startTime": stamp(start), "endTime": stamp(end), "bucketSize": "hour"}
            entry = {"role": role, "pod_id": identifier, "query": query}
            try:
                response = getter(query)
                validate_response(response, query)
                entry.update(state="captured", response=response)
            except Exception as error:
                # No provider body, credential, or arbitrary exception text.
                entry.update(state="unavailable", error_type=type(error).__name__)
                if isinstance(error, urllib.error.HTTPError):
                    entry["http_status"] = error.code
            entries.append(entry)
    return {"kind": "cardinal_project_pod_billing_capture", "format_version": 1,
        "intent": snap["plan"]["intent"], "plan_sha256": snap["state"]["plan_sha256"],
        "captured_at": stamp(at), "endpoint": API + "/billing/pods", "entries": entries,
        "input_sha256": snap["inputs"], "settlement_confirmed": False}


def preview(snap, captures, at):
    plan, budget = snap["plan"], snap["budget"]
    reservation = budget.get("cloud_reservations", {}).get(plan["intent"])
    if not reservation or reservation.get("state") != "reserved" or money(reservation["amount_usd"]) != money(plan["reservation_usd"]):
        raise ValueError("Expected intact project reservation")
    ids = {i: role for role, r in snap["roles"].items() for i in r["ids"]}
    observations, input_hashes = {}, dict(snap["inputs"])
    for path, raw in captures:
        input_hashes[str(Path(path).resolve())] = sha(raw)
        value = json.loads(raw)
        if value.get("kind") != "cardinal_project_pod_billing_capture" or value.get("intent") != plan["intent"] or value.get("plan_sha256") != snap["state"]["plan_sha256"] or value.get("endpoint") != API + "/billing/pods":
            raise ValueError("Billing capture belongs to another deployment")
        captured = instant(value["captured_at"])
        if captured > at:
            raise ValueError("Billing capture timestamp is in the future")
        seen = set()
        for row in value["entries"]:
            identifier = row["pod_id"]
            if identifier not in ids or row["role"] != ids[identifier] or identifier in seen or row["query"]["podId"] != identifier:
                raise ValueError("Billing capture has unknown/duplicate project resource")
            seen.add(identifier)
            if row["state"] == "captured":
                totals = validate_response(row["response"], row["query"])
                if instant(row["query"]["startTime"]) > instant(plan["created_at"]):
                    raise ValueError("Billing capture omits deployment start")
                candidate = {"pod_id": identifier, "role": ids[identifier], "state": "provider_reported",
                    "captured_at": stamp(captured), "totals_usd": totals, "records": len(row["response"]["records"]),
                    "query": row["query"], "capture_sha256": sha(raw), "settlement_confirmed": False}
                if identifier not in observations or captured > instant(observations[identifier]["captured_at"]):
                    observations[identifier] = candidate
            elif row["state"] != "unavailable":
                raise ValueError("Unknown capture state")
    rows = [observations.get(i, {"pod_id": i, "role": role, "state": "no_valid_billing_capture", "settlement_confirmed": False}) for i, role in sorted(ids.items())]
    totals = {key: sum((money(r["totals_usd"][key]) for r in rows if "totals_usd" in r), Decimal(0)) for key in FIELDS}
    ledgers = {name: ledger_commitment(value) for name, value in snap["ledgers"].items()}
    reserve = money(reservation["amount_usd"])
    all_reserved = money(budget.get("cloud_reserved_usd", 0))
    listed_reserved = sum((money(v["amount_usd"]) for v in budget.get("cloud_reservations", {}).values() if v.get("state") == "reserved"), Decimal(0))
    if all_reserved != listed_reserved or all_reserved < reserve:
        raise ValueError("Cloud reservation aggregate disagrees with reservation records")
    conservative = money(budget["cloud_spend_to_date"]) + all_reserved + sum((money(x) for x in ledgers.values()), Decimal(0))
    # Reported charges are already covered by this reservation. Do not add
    # them twice. Raise the preview if charges have already exceeded it.
    conservative += max(Decimal(0), totals["totalAmount"] - reserve)
    cap = min(Decimal(100), money(budget["approved_pilot_cap"]))
    return {"kind": "cardinal_gpu_accounting_preview", "format_version": 1, "at": stamp(at),
        "intent": plan["intent"], "deployment": snap["deployment"], "roles": snap["roles"],
        "provider_reported_pod_charges_usd": {k: float(v) for k, v in totals.items()}, "pod_billing": rows,
        "budget": {"pilot_cap_usd": float(cap), "api_ledger_commitments_usd": ledgers,
            "other_recorded_cloud_spend_usd": float(money(budget["cloud_spend_to_date"])),
            "all_cloud_reservations_usd": float(all_reserved), "project_reservation_usd": float(reserve),
            "conservative_commitment_usd": float(conservative), "uncommitted_headroom_usd": float(cap-conservative),
            "suggested_reservation_release_usd": 0.0, "reservation_retained": True, "budget_mutated": False},
        "settlement_confirmed": False, "hard_billing_upper_bound_proven": False,
        "limitations": ["Provider-reported bucket totals may be incomplete, delayed, or subsequently adjusted; no settlement marker is published.",
            "Zero or absent billing rows do not establish zero cost. Known-ID capture cannot rule out a delayed, never-observed creation.",
            "Pod diskAmount is included. No account-wide charges or unrelated resource costs are attributed to this project.",
            "The reservation remains fully committed; provider-reported charges covered by it are not added a second time.",
            "Termination is the saved local controller's direct-404 evidence, not a fresh provider inventory check by this preview."],
        "input_sha256": input_hashes, "official_schema": SCHEMA_URL}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("capture", "preview"))
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--billing-capture", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    snap = snapshot(ROOT, args.deployment)
    at = datetime.now(timezone.utc)
    value = capture(snap, at) if args.command == "capture" else preview(snap, [(p, p.read_bytes()) for p in args.billing_capture], at)
    write_new(args.out, value)
    print(json.dumps({"output": str(args.out), "sha256": sha(args.out.read_bytes()), "budget_mutated": False, "resource_mutations": 0}))


if __name__ == "__main__":
    main()
