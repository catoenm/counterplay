"""Pinned, bounded teacher authoring client; importing never invokes inference.

Thirty-two fixed attempts, at most $2 phase commitment, cumulative $100 cap.
Completed responses are reused exactly; uncertain requests are never retried.
The collector owns frozen prompts and CPU-only gameplay qualification.
"""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

from . import program_model_screen as gateway

PROFILE = {'model': 'deepseek/deepseek-v4-pro-0813', 'provider': 'deepinfra/fp8',
    'returned_provider': 'DeepInfra', 'quantization': 'fp8',
    'input_usd_per_million_upper': 1.30, 'output_usd_per_million_upper': 2.60}
MAX_CALLS, OUTPUT_CAP, PHASE_CAP = 32, 16384, 2.0
LEDGER = Path('runs/program-cardinal-teacher-named-ledger.json')
BUDGET = Path('budget.json')
BUDGET_LOCK = Path('runs/external-budget.lock')
REQUIRED_PARAMETERS = {'reasoning', 'max_tokens', 'temperature', 'top_p', 'seed'}
canonical, sha, atomic_json = gateway.canonical, gateway.sha, gateway.atomic_json


def configuration():
    return {'format_version': 1, 'generation_client': 'program_cardinal_teacher_named',
        'profile': PROFILE, 'max_calls': MAX_CALLS, 'phase_cap_usd': PHASE_CAP,
        'cumulative_pilot_cap_usd': 100, 'max_output_tokens': OUTPUT_CAP,
        'reasoning': {'effort': 'none'}, 'temperature': .6, 'top_p': .95,
        'no_automatic_retries': True, 'retain_all_attempts': True,
        'delivery_gate': {'index': 0, 'nonempty_final': True, 'finish_reason': 'stop',
            'failure': 'Stop incomplete; no remaining requests, replacement, or gameplay grading.'}}


def body_for(prompt, index):
    if type(index) is not int or not 0 <= index < MAX_CALLS or not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('Fixed teacher index0..31 and nonempty frozen prompt required')
    return {'model': PROFILE['model'], 'messages': [{'role': 'user', 'content': prompt}],
        'reasoning': {'effort': 'none'}, 'max_tokens': OUTPUT_CAP,
        'temperature': .6, 'top_p': .95, 'seed': 141000000 + index,
        'provider': {'only': [PROFILE['provider']], 'quantizations': ['fp8'],
            'allow_fallbacks': False, 'require_parameters': True,
            'max_price': {'prompt': 1.30, 'completion': 2.60}}}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise RuntimeError('Authenticated redirect rejected')


def verify_endpoint():
    """Public metadata only; credentials and inference are never used here."""
    url = 'https://openrouter.ai/api/v1/models/' + PROFILE['model'] + '/endpoints'
    with urllib.request.urlopen(url, timeout=30) as response:
        catalog = json.load(response)
    if catalog['data']['id'] != PROFILE['model']:
        raise ValueError('Teacher catalog model differs')
    matches = [e for e in catalog['data']['endpoints'] if e.get('tag') == PROFILE['provider']
        and e.get('provider_name') == PROFILE['returned_provider'] and e.get('quantization') == 'fp8']
    if len(matches) != 1:
        raise ValueError('Teacher endpoint missing or ambiguous')
    endpoint = matches[0]
    for field, maximum in (('prompt', 1.30), ('completion', 2.60)):
        actual = gateway.money(float(endpoint['pricing'][field]), field)
        if actual > maximum / 1_000_000 + 1e-15:
            raise ValueError('Teacher price exceeds pinned maximum')
    if not REQUIRED_PARAMETERS <= set(endpoint.get('supported_parameters', [])):
        raise ValueError('Teacher endpoint lacks requested parameters')
    if type(endpoint.get('max_completion_tokens')) is not int or endpoint['max_completion_tokens'] < OUTPUT_CAP:
        raise ValueError('Teacher completion limit too small')
    models_url = 'https://openrouter.ai/api/v1/models'
    with urllib.request.urlopen(models_url, timeout=30) as response:
        models = json.load(response)
    model_rows = [m for m in models['data'] if m.get('id') == PROFILE['model']]
    if (len(model_rows) != 1 or 'none' not in model_rows[0].get('reasoning', {}).get('supported_efforts', [])
            or model_rows[0].get('reasoning', {}).get('mandatory') is not False):
        raise ValueError('Pinned model catalog does not advertise disabled reasoning')
    capability = {'id': model_rows[0]['id'], 'reasoning': model_rows[0]['reasoning']}
    return {'url': url, 'checked_at_unix': time.time(), 'endpoint': endpoint,
        'endpoint_sha256': sha(canonical(endpoint).encode()),
        'model_catalog_url': models_url, 'reasoning_capability': capability,
        'reasoning_capability_sha256': sha(canonical(capability).encode())}


def source_hashes():
    return {'selfplay/program_cardinal_teacher_named.py': sha(Path(__file__).read_bytes()),
        'selfplay/program_model_screen.py': sha(Path(gateway.__file__).read_bytes())}


def request_identity(prompt, index, plan_sha256):
    if len(plan_sha256) != 64 or any(c not in '0123456789abcdef' for c in plan_sha256):
        raise ValueError('Exact frozen plan SHA256 required')
    return sha(canonical({'body': body_for(prompt, index), 'index': index,
        'plan_sha256': plan_sha256, 'source_sha256': source_hashes()}).encode())


def reservation(prompt, index):
    encoded = canonical(body_for(prompt, index)).encode()
    if len(encoded) > 40000:
        raise ValueError('Teacher request exceeds40KB')
    return (len(encoded) + 2048) * 1.30e-6 + (OUTPUT_CAP + 64) * 2.60e-6


def new_ledger(plan_sha256):
    return {'configuration': configuration(), 'plan_sha256': plan_sha256,
        'source_sha256': source_hashes(), 'requests': {}}


def checked_ledger(plan_sha256, path=None):
    path = LEDGER if path is None else Path(path)
    expected = new_ledger(plan_sha256)
    ledger = json.loads(path.read_text()) if path.exists() else expected
    if any(ledger.get(k) != expected[k] for k in ('configuration', 'plan_sha256', 'source_sha256')):
        raise ValueError('Teacher ledger belongs to another frozen protocol')
    return ledger


def verify_completed_response(prompt, index, plan_sha256, record, ledger_path=None):
    ledger = checked_ledger(plan_sha256, ledger_path)
    identity = request_identity(prompt, index, plan_sha256)
    entry = ledger['requests'].get(identity)
    if not entry or entry.get('state') != 'completed' or entry.get('response') != record:
        raise ValueError('Teacher output differs from completed paid-request journal')
    return identity


def delivery_passed(record):
    return (record.get('finish_reason') == 'stop' and isinstance(record.get('text'), str)
        and bool(record['text'].strip()))


def complete(prompt, index, plan_sha256):
    """Exactly one paid POST at most; only an explicit caller invokes this."""
    body = body_for(prompt, index); encoded = canonical(body).encode()
    reserved = reservation(prompt, index)
    identity = request_identity(prompt, index, plan_sha256)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with BUDGET_LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = checked_ledger(plan_sha256); entries = ledger['requests']
        if identity in entries and entries[identity]['state'] == 'completed':
            return entries[identity]['response']
        if any(e.get('state') != 'completed' for e in entries.values()):
            raise RuntimeError('Unresolved teacher call: reservation retained, no retry')
        if sorted(e['index'] for e in entries.values()) != list(range(index)) or index >= MAX_CALLS:
            raise RuntimeError('Teacher attempts must be fixed, sequential and unreplaced')
        if index > 0:
            first = next(e for e in entries.values() if e['index'] == 0)
            if not delivery_passed(first['response']):
                raise RuntimeError('First teacher delivery gate failed; cohort incomplete, no later requests')
        if any(e['charged_upper_usd'] > e['reserved_usd'] + 1e-9 for e in entries.values()):
            raise RuntimeError('Prior teacher accounting overrun requires audit')
        budget = json.loads(BUDGET.read_text())
        commitment = gateway.ledger_commitment(LEDGER)
        prior_paths = set(Path('runs').glob('*ledger.json')); prior_paths.discard(LEDGER)
        prior = {str(p): gateway.ledger_commitment(p) for p in sorted(prior_paths)}
        total = gateway.money(budget['cloud_spend_to_date'], 'cloud spend')
        total += gateway.money(budget.get('cloud_reserved_usd', 0), 'cloud reservation') + sum(prior.values())
        cap = min(PHASE_CAP, gateway.money(budget.get('program_cardinal_teacher_named_subcap_usd', PHASE_CAP), 'phase cap'))
        if commitment + reserved > cap or total + commitment + reserved > min(100, gateway.money(budget['approved_pilot_cap'], 'pilot cap')):
            raise RuntimeError('Teacher phase or cumulative pilot budget admission rejected')
        quote = verify_endpoint(); key = gateway.api_key()
        entry = {'state': 'reserved', 'index': index, 'reserved_usd': reserved,
            'charged_upper_usd': reserved, 'started': time.time(), 'prior_ledger_commitments': prior,
            'prior_commitment_usd': total, 'prompt_sha256': sha(prompt.encode()),
            'endpoint_quote': quote, 'request_parameters': {k: v for k, v in body.items() if k != 'messages'}}
        entries[identity] = entry; atomic_json(LEDGER, ledger)
        request = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions', encoded,
            {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key}, method='POST')
        try:
            with urllib.request.build_opener(NoRedirect).open(request, timeout=300) as response:
                raw = response.read(4 * 1024 * 1024 + 1)
            if len(raw) > 4 * 1024 * 1024:
                raise ValueError('Oversized teacher response')
            result = json.loads(raw)
        except Exception as error:
            entry.update(state='unknown', error_type=type(error).__name__)
            if isinstance(error, urllib.error.HTTPError): entry['http_status'] = error.code
            atomic_json(LEDGER, ledger)
            raise RuntimeError('Teacher request failed; reservation held; no automatic retry') from None
        entry['raw_response'] = result
        try:
            usage = result['usage']; choices = result['choices']
            if not isinstance(usage, dict) or not all(type(usage.get(k)) is int and usage[k] >= 0 for k in ('prompt_tokens', 'completion_tokens')):
                raise ValueError('Invalid usage')
            cost = gateway.money(usage['cost'], 'teacher usage cost')
            charged = max(cost, usage['prompt_tokens'] * 1.30e-6 + usage['completion_tokens'] * 2.60e-6)
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError('Expected exactly one teacher choice')
            choice = choices[0]; text = choice['message'].get('content') or ''
            if not isinstance(text, str) or choice['finish_reason'] not in ('stop', 'length'):
                raise ValueError('Invalid text/finish')
            if result.get('model') != PROFILE['model'] or result.get('provider') != PROFILE['returned_provider']:
                raise ValueError('Teacher model/provider changed')
            if charged > reserved + 1e-9 or usage['completion_tokens'] > OUTPUT_CAP:
                raise ValueError('Teacher cost/token cap exceeded')
        except Exception as error:
            entry.update(state='invalid_response', error_type=type(error).__name__)
            if 'charged' in locals(): entry['charged_upper_usd'] = max(reserved, charged)
            atomic_json(LEDGER, ledger)
            raise RuntimeError('Teacher response identity/usage invalid; original retained, no retry') from None
        details = usage.get('completion_tokens_details') or {}
        record = {'generation_client': 'program_cardinal_teacher_named', 'index': index,
            'plan_sha256': plan_sha256, 'request_identity': identity, 'prompt': prompt,
            'prompt_sha256': sha(prompt.encode()), 'source_sha256': source_hashes(),
            'text': text, 'finish_reason': choice['finish_reason'], 'usage': usage,
            'reasoning_requested': {'effort': 'none'},
            'reported_reasoning_tokens': details.get('reasoning_tokens') if isinstance(details, dict) else None,
            'returned_model': result['model'], 'returned_provider': result['provider'],
            'response_id': result.get('id'), 'endpoint_quote': quote,
            'request_parameters': entry['request_parameters'], 'cost_usd': cost, 'cost_upper_usd': charged,
            'raw_response_sha256': sha(raw)}
        entry.update(state='completed', charged_upper_usd=charged, response=record)
        atomic_json(LEDGER, ledger)
        return record
