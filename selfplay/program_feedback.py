"""Four-call, $0.10 practice-feedback diagnostic. No automatic retries or training.

Execution is opt-in through the CLI; importing this module performs no I/O.
Every paid request holds the shared external-budget lock through its journal
commit. Prior uncertain calls remain reserved; this screen never retries them.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import time
import urllib.error
import urllib.request
import uuid


PROFILES = {
    "qwen38-27b": {
        "model": "qwen/qwen3.8-27b", "provider": "chutes/fp8", "returned_provider": "Chutes",
        "quantization": "fp8", "input_usd_per_million_upper": 0.039,
        "output_usd_per_million_upper": 1.25,
        "hugging_face_model": "Qwen/Qwen3.8-27B",
        "hugging_face_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    },

}
OUTPUT_CAP = 4096
SCREEN_CAP = 0.10
MAX_CALLS = 4
LINEAGES = ("qwen38-27b-1", "qwen38-27b-2")
ROUNDS_PER_LINEAGE = 2
MAX_REQUEST_BYTES = 40000
LEDGER = Path("runs/program-feedback-ledger.json")
BUDGET = Path("budget.json")
BUDGET_LOCK = Path("runs/external-budget.lock")
PRIOR_LEDGERS = (Path("runs/program-api-ledger.json"), Path("runs/openrouter-api-ledger.json"),
                 Path("runs/challenge-api-ledger.json"))
REQUIRED_PARAMETERS = {"reasoning", "max_tokens", "temperature", "top_p", "top_k"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def api_key():
    value = os.environ.get("OPENROUTER_API_KEY")
    if value:
        return value
    path = Path(".env")
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip().removeprefix("export ")
            if line.startswith("OPENROUTER_API_KEY="):
                parts = shlex.split(line.split("=", 1)[1], comments=True)
                if len(parts) == 1 and parts[0]:
                    return parts[0]
    raise RuntimeError("OPENROUTER_API_KEY is not configured")


def money(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid nonnegative finite accounting value: {label}")
    return float(value)


def ledger_commitment(path):
    if not path.exists():
        return 0.0
    entries = json.loads(path.read_text())["requests"]
    if not isinstance(entries, dict):
        raise ValueError(f"Invalid ledger requests: {path}")
    total = 0.0
    for entry in entries.values():
        state = entry["state"]
        reserved = money(entry.get("reserved_usd", 0), "reserved_usd")
        charged = money(entry.get("charged_upper_usd", 0), "charged_upper_usd")
        if not isinstance(state, str) or not state:
            raise ValueError(f"Invalid ledger state: {path}")
        if state == "completed" and "charged_upper_usd" not in entry:
            raise ValueError(f"Completed request lacks accounting: {path}")
        if state != "completed" and ("reserved_usd" not in entry or reserved <= 0):
            raise ValueError(f"Unresolved request lacks a positive reservation: {path}")
        total += max(charged, reserved if state != "completed" else 0.0)
    return total


def configuration():
    return {"format_version": 1, "generation_client": "program_feedback", "profiles": PROFILES,
            "reasoning": "off", "max_output_tokens": OUTPUT_CAP, "temperature": 0.6, "top_p": 0.95,
            "top_k": 20, "max_calls": MAX_CALLS, "lineages": list(LINEAGES), "rounds_per_lineage": ROUNDS_PER_LINEAGE,
            "screen_cap_usd": SCREEN_CAP, "cumulative_pilot_cap_usd": 100,
            "pricing_checked": "2026-10-22", "live_endpoint_check_required": True,
            "interpretation": "Bounded practice-feedback authoring diagnostic, no gradient training or best-of outcome selection."}


def body_for(prompt, profile_name):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Nonempty prompt text required")
    profile = PROFILES[profile_name]
    return {"model": profile["model"], "messages": [{"role": "user", "content": prompt}],
            "reasoning": {"enabled": False}, "max_tokens": OUTPUT_CAP,
            "temperature": 0.6, "top_p": 0.95, "top_k": 20,
            "provider": {"only": [profile["provider"]], "quantizations": [profile["quantization"]],
                         "allow_fallbacks": False, "require_parameters": True,
                         "max_price": {"prompt": profile["input_usd_per_million_upper"],
                                       "completion": profile["output_usd_per_million_upper"]}}}


def verify_endpoint(profile_name):
    """Public metadata GET only; never sends credentials or invokes inference."""
    profile = PROFILES[profile_name]
    url = "https://openrouter.ai/api/v1/models/" + profile["model"] + "/endpoints"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            catalog = json.load(response)
        if catalog["data"].get("id") != profile["model"]:
            raise ValueError("Endpoint catalog model identity differs")
        endpoints = catalog["data"]["endpoints"]
        matching = [e for e in endpoints if e.get("tag") == profile["provider"]
                    and e.get("provider_name") == profile["returned_provider"]
                    and e.get("quantization") == profile["quantization"]]
        if len(matching) != 1:
            raise ValueError("Pinned endpoint is missing or ambiguous")
        endpoint = matching[0]
        for key, limit in (("prompt", profile["input_usd_per_million_upper"]),
                           ("completion", profile["output_usd_per_million_upper"])):
            price = money(float(endpoint["pricing"][key]), key)
            if price > limit / 1_000_000 + 1e-15:
                raise ValueError("Pinned endpoint price exceeds frozen maximum")
        if not REQUIRED_PARAMETERS <= set(endpoint.get("supported_parameters", [])):
            raise ValueError("Pinned endpoint lacks requested parameter support")
        if type(endpoint.get("max_completion_tokens")) is not int or endpoint["max_completion_tokens"] < OUTPUT_CAP:
            raise ValueError("Pinned endpoint completion cap is too small")
    except Exception as error:
        # Do not echo HTTP bodies, credentials, or arbitrary server errors.
        raise RuntimeError("Public endpoint preflight failed: " + type(error).__name__) from None
    return {"catalog_url": url, "checked_at_unix": time.time(), "endpoint": endpoint,
            "endpoint_sha256": sha(canonical(endpoint).encode())}


def new_ledger(source_hash):
    config = configuration()
    return {"configuration": config, "configuration_sha256": sha(canonical(config).encode()),
            "client_source_sha256": source_hash, "requests": {}}


def validate_context(context):
    keys = {"plan_sha256", "lineage", "round_index", "previous_generation_sha256",
            "feedback_sha256", "original_prompt_sha256"}
    if not isinstance(context, dict) or set(context) != keys:
        raise ValueError("Exact feedback revision context required")
    if context["lineage"] not in LINEAGES or type(context["round_index"]) is not int or context["round_index"] not in (1, 2):
        raise ValueError("Only two revisions per original lineage are authorized")
    for key in keys - {"lineage", "round_index"}:
        if not isinstance(context[key], str) or re.fullmatch(r"[0-9a-f]{64}", context[key]) is None:
            raise ValueError("Feedback context must contain exact SHA256 values")


def complete(prompt, sample_id, context):
    validate_context(context)
    profile_name = "qwen38-27b"
    if sample_id != f'{context["lineage"]}-r{context["round_index"]}':
        raise ValueError("Sample ID must identify its lineage and revision")
    if not isinstance(sample_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", sample_id) is None:
        raise ValueError("A short alphanumeric sample ID is required")
    body = body_for(prompt, profile_name)
    encoded = canonical(body).encode()
    if len(encoded) > MAX_REQUEST_BYTES:
        raise ValueError("Controller prompt exceeds the 40 KB admission limit")
    profile = PROFILES[profile_name]
    input_rate, output_rate = (profile[k] / 1_000_000 for k in
                              ("input_usd_per_million_upper", "output_usd_per_million_upper"))
    # UTF-8/JSON bytes bound input tokenization conservatively, plus framing.
    reserved = (len(encoded) + 2048) * input_rate + (OUTPUT_CAP + 64) * output_rate
    source_hash = sha(Path(__file__).read_bytes())
    identity = sha(canonical({"body": body, "sample_id": sample_id, "client_source_sha256": source_hash, "feedback_context": context}).encode())
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with BUDGET_LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        budget = json.loads(BUDGET.read_text())
        ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else new_ledger(source_hash)
        expected = new_ledger(source_hash)
        if any(ledger.get(key) != expected[key] for key in ("configuration", "configuration_sha256", "client_source_sha256")):
            raise ValueError("Incompatible model screen ledger; retain and audit it")
        entries = ledger["requests"]
        if identity in entries and entries[identity]["state"] == "completed":
            return entries[identity]["response"]
        if any(entry["state"] != "completed" for entry in entries.values()):
            raise RuntimeError("Screen request unresolved; reservation retained and no retry permitted")
        if any(entry["charged_upper_usd"] > entry["reserved_usd"] + 1e-9 for entry in entries.values()):
            raise RuntimeError("Prior screen accounting overrun requires audit")
        if any(entry["sample_id"] == sample_id for entry in entries.values()):
            raise RuntimeError("Sample ID already has a different request; no replacement permitted")
        if any(entry["feedback_context"][key] != context[key] for entry in entries.values()
               for key in ("plan_sha256", "original_prompt_sha256")):
            raise RuntimeError("Feedback calls must share the frozen plan and original specification")
        prior_rounds = [entry["feedback_context"]["round_index"] for entry in entries.values()
                        if entry["feedback_context"]["lineage"] == context["lineage"]]
        if len(entries) >= MAX_CALLS or context["round_index"] != len(prior_rounds) + 1:
            raise RuntimeError("Feedback call cap or sequential revision rule reached")
        committed = ledger_commitment(LEDGER)
        prior_paths = set(PRIOR_LEDGERS) | set(Path("runs").glob("program-*-ledger.json"))
        prior_paths.discard(LEDGER)
        prior_breakdown = {str(path): ledger_commitment(path) for path in sorted(prior_paths)}
        prior = money(budget["cloud_spend_to_date"], "cloud_spend_to_date")
        prior += money(budget.get("cloud_reserved_usd", 0), "cloud_reserved_usd")
        prior += sum(prior_breakdown.values())
        subcap = min(SCREEN_CAP, money(budget.get("program_feedback_subcap_usd", SCREEN_CAP), "screen subcap"))
        if committed + reserved > subcap:
            raise RuntimeError("Feedback $0.10 subcap admission rejected")
        if prior + committed + reserved > min(100.0, money(budget["approved_pilot_cap"], "approved_pilot_cap")):
            raise RuntimeError("Cumulative pilot budget admission rejected")
        quote = verify_endpoint(profile_name)
        key = api_key()  # Missing credentials do not create a paid-call reservation.
        entry = {"state": "reserved", "sample_id": sample_id, "profile_name": profile_name,
                 "reserved_usd": reserved, "charged_upper_usd": reserved, "prior_commitment_usd": prior,
                 "prior_ledger_commitments": prior_breakdown, "started": time.time(),
                 "prompt_sha256": sha(prompt.encode()), "endpoint_quote": quote,
                 "feedback_context": context,
                 "request_parameters": {k: v for k, v in body.items() if k != "messages"}}
        entries[identity] = entry
        atomic_json(LEDGER, ledger)
        request = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", encoded,
                    {"Content-Type": "application/json", "Authorization": "Bearer " + key})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.load(response)
        except Exception as error:
            entry.update(state="unknown", error_type=type(error).__name__)
            if isinstance(error, urllib.error.HTTPError):
                entry["http_status"] = error.code
            atomic_json(LEDGER, ledger)
            raise RuntimeError("Model screen request failed; reservation retained; no automatic retry") from None
        if not isinstance(result, dict):
            entry.update(state="invalid_response", error_type="NonObjectResponse")
            atomic_json(LEDGER, ledger)
            raise RuntimeError("Malformed response; reservation retained")
        usage = result.get("usage") or {}
        cost = usage.get("cost") if isinstance(usage, dict) else None
        if (not isinstance(usage, dict)
                or not all(type(usage.get(k)) is int and usage[k] >= 0 for k in ("prompt_tokens", "completion_tokens"))
                or type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0):
            entry.update(state="unknown_usage", response_id=result.get("id"))
            atomic_json(LEDGER, ledger)
            raise RuntimeError("Missing valid usage/cost; reservation retained")
        charged = max(cost, usage["prompt_tokens"] * input_rate + usage["completion_tokens"] * output_rate)
        choices = result.get("choices")
        choice = choices[0] if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict) else {}
        message = choice.get("message") or {}
        message = message if isinstance(message, dict) else {}
        text = message.get("content")
        text = "" if text is None else text
        details = usage.get("completion_tokens_details")
        reasoning_tokens = details.get("reasoning_tokens") if isinstance(details, dict) else None
        condition_status = ("confirmed_off" if type(reasoning_tokens) is int and reasoning_tokens == 0
                            else "violated" if type(reasoning_tokens) is int and reasoning_tokens > 0 else "unknown")
        record = {"sample_id": sample_id, "prompt": prompt, "prompt_sha256": sha(prompt.encode()),
                  "client_source_sha256": source_hash, "generation_client": "program_feedback",
                  "generation_client_profile": {"name": profile_name, **profile},
                  "generation_client_configuration_sha256": ledger["configuration_sha256"],
                  "reasoning_mode": "off", "reasoning_condition_status": condition_status,
                  "reported_reasoning_tokens": reasoning_tokens, "request_parameters": entry["request_parameters"],
                  "endpoint_quote": quote, "feedback_context": context, "text": text, "reasoning": message.get("reasoning"),
                  "finish_reason": choice.get("finish_reason"), "usage": usage,
                  "response_id": result.get("id"), "returned_model": result.get("model"),
                  "returned_provider": result.get("provider"), "cost_usd": cost, "cost_upper_usd": charged}
        valid = (result.get("model") == profile["model"] and result.get("provider") == profile["returned_provider"]
                 and choice.get("finish_reason") in ("stop", "length") and isinstance(text, str)
                 and usage["completion_tokens"] <= OUTPUT_CAP and charged <= reserved + 1e-9)
        entry.update(state="completed" if valid else "invalid_response", response=record,
                     charged_upper_usd=charged if valid else max(charged, reserved))
        atomic_json(LEDGER, ledger)
        if not valid:
            raise RuntimeError("Model/provider identity, token cap, or accounting mismatch; stop and audit")
        return record



def verify_completed_response(prompt, sample_id, context, record):
    """Bind saved output bytes-by-value to the durable paid-request journal."""
    source_hash = sha(Path(__file__).read_bytes())
    body = body_for(prompt, "qwen38-27b")
    identity = sha(canonical({"body": body, "sample_id": sample_id,
                             "client_source_sha256": source_hash, "feedback_context": context}).encode())
    ledger = json.loads(LEDGER.read_text())
    expected = new_ledger(source_hash)
    if any(ledger.get(key) != expected[key] for key in ("configuration", "configuration_sha256", "client_source_sha256")):
        raise ValueError("Feedback journal configuration differs")
    entry = ledger["requests"].get(identity)
    if entry is None or entry.get("state") != "completed" or entry.get("response") != record:
        raise ValueError("Saved generation differs from its completed paid-request journal")
    return identity


def write_once(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Refusing to replace archived file: {path}")
        return
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Refusing to overwrite an existing generation record")
    source = args.out.parent / "source"
    source.mkdir(parents=True, exist_ok=True)
    write_once(source / "program_feedback.py", Path(__file__).read_bytes())
    write_once(source / "feedback-profile.json", (json.dumps(configuration(), indent=2) + "\n").encode())
    record = complete(args.prompt.read_text(), args.sample_id, json.loads(args.context.read_text()))
    # The journal makes a crash before output publication recoverable without
    # another request. Exclusive creation protects concurrent output names.
    write_once(args.out, (json.dumps(record, indent=2, allow_nan=False) + "\n").encode())
    print(json.dumps({key: record[key] for key in ("sample_id", "returned_model", "finish_reason", "cost_usd")}))


if __name__ == "__main__":
    main()
