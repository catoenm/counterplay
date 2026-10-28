"""Budgeted OpenAI Responses client. No retries, tools or shared conversation."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import time
import urllib.error
import urllib.request

MODEL = 'gpt-6-astra'
INPUT_RATE = 12.50 / 1_000_000  # conservative: includes maximum cache-write rate
OUTPUT_RATE = 50 / 1_000_000
RUN_CAP = 35.0
MAX_CALLS = 114
OUTPUT_CAP = 4096


def api_key(name='OPENAI_API_KEY'):
    if os.environ.get(name):
        return os.environ[name]
    path = Path('.env')
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip().removeprefix('export ')
            if line.startswith(name + '='):
                parts = shlex.split(line.split('=', 1)[1], comments=True)
                if len(parts) == 1 and parts[0]:
                    return parts[0]
    raise RuntimeError(name + ' is not configured in the environment or project .env')


def atomic_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    with tmp.open('rb') as f:
        os.fsync(f.fileno())
    tmp.replace(path)


def request_payload(request):
    body = {'model': MODEL, 'input': [{'role': 'user', 'content': request['prompt']}],
            'reasoning': {'effort': 'medium'}, 'max_output_tokens': OUTPUT_CAP,
            'store': False, 'service_tier': 'default'}
    if request.get('labels'):
        body['text'] = {'format': {'type': 'json_schema', 'name': 'action_plan', 'strict': True,
            'schema': {'type': 'object', 'properties': {'plan': {'type': 'array',
                'items': {'type': 'string', 'enum': request['labels']},
                'minItems': request['horizon'], 'maxItems': request['horizon']}},
                'required': ['plan'], 'additionalProperties': False}}}
    return body


def reserve_amount(body):
    size = len(json.dumps(body).encode())
    if size > 6000:
        raise ValueError('Request exceeds frozen 6000-byte admission limit')
    return (size + 1024) * INPUT_RATE + OUTPUT_CAP * OUTPUT_RATE


class OpenAIBackend:
    def __init__(self, ledger=Path('runs/challenge-api-ledger.json'), budget=Path('budget.json')):
        self.key = api_key()
        self.ledger, self.budget = Path(ledger), Path(budget)
        self.metadata = {'model': MODEL, 'backend': 'openai_responses', 'reasoning': 'medium',
                         'max_output_tokens': OUTPUT_CAP, 'seed_supported': False,
                         'pricing_checked': '2026-10-21', 'input_usd_per_million_upper': 12.5,
                         'output_usd_per_million': 50, 'api_run_cap_usd': RUN_CAP}

    def complete_batch(self, requests):
        return [self.complete(r) for r in requests]

    def complete(self, request):
        body = request_payload(request)
        reservation = reserve_amount(body)
        identity = hashlib.sha256(json.dumps({'body': body, 'sample_id': request['seed']}, sort_keys=True).encode()).hexdigest()
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        # Held through request + journal commit to prevent concurrent budget admissions.
        with self.ledger.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            budget = json.loads(self.budget.read_text())
            ledger = json.loads(self.ledger.read_text()) if self.ledger.exists() else {
                'model': MODEL, 'baseline_external_spend_usd': budget['cloud_spend_to_date'], 'requests': {}}
            if ledger['model'] != MODEL:
                raise RuntimeError('Ledger model differs')
            if any(r.get('charged_upper_usd', 0) > r.get('reserved_usd', 0) + 1e-9
                   for r in ledger['requests'].values()):
                raise RuntimeError('Prior accounting overrun requires audit; no further requests admitted')
            if identity in ledger['requests']:
                old = ledger['requests'][identity]
                if old['state'] != 'completed':
                    raise RuntimeError('Previous request outcome unknown; reservation retained, no retry')
                return old['response']
            entries = ledger['requests']
            committed = sum(r['charged_upper_usd'] for r in entries.values())
            if len(entries) >= MAX_CALLS or committed + reservation > RUN_CAP:
                raise RuntimeError('Calibration API cap reached')
            # Also respect external spending recorded since this calibration began.
            prior = max(ledger['baseline_external_spend_usd'], budget['cloud_spend_to_date'])
            if prior + committed + reservation > min(100.0, budget['approved_pilot_cap']):
                raise RuntimeError('Cumulative pilot budget admission rejected')
            entries[identity] = {'state': 'reserved', 'charged_upper_usd': reservation,
                                 'reserved_usd': reservation, 'sample_id': request['seed'], 'started': time.time()}
            atomic_json(self.ledger, ledger)
            http_request = urllib.request.Request('https://api.openai.com/v1/responses',
                json.dumps(body).encode(), {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.key})
            try:
                with urllib.request.urlopen(http_request, timeout=240) as resp:
                    result = json.load(resp)
            except Exception as exc:
                # Never print provider bodies or authentication headers.
                entries[identity]['state'] = 'unknown'
                entries[identity]['error_type'] = type(exc).__name__
                atomic_json(self.ledger, ledger)
                raise RuntimeError('API request failed; conservative charge retained; no automatic retry') from None
            usage = result.get('usage') or {}
            if not all(type(usage.get(k)) is int and usage[k] >= 0 for k in ('input_tokens', 'output_tokens')):
                entries[identity]['state'] = 'unknown_usage'
                atomic_json(self.ledger, ledger)
                raise RuntimeError('Missing provider usage; reservation retained')
            charged = usage['input_tokens'] * INPUT_RATE + usage['output_tokens'] * OUTPUT_RATE
            text = ''.join(c.get('text', '') for o in result.get('output', []) if o.get('type') == 'message'
                           for c in o.get('content', []) if c.get('type') == 'output_text')
            incomplete = result.get('incomplete_details') or {}
            response = {'text': text, 'prompt_tokens': usage['input_tokens'], 'completion_tokens': usage['output_tokens'],
                        'finish_reason': 'length' if incomplete.get('reason') == 'max_output_tokens' else result.get('status'),
                        'usage': usage, 'response_id': result.get('id'), 'returned_model': result.get('model'),
                        'service_tier': result.get('service_tier'), 'cost_upper_usd': charged}
            entries[identity].update(state='completed', response=response, charged_upper_usd=charged)
            atomic_json(self.ledger, ledger)
            if charged > reservation + 1e-9:
                raise RuntimeError('Provider accounting exceeded reservation; stop and audit before continuing')
            return response
