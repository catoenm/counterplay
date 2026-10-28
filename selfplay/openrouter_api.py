"""Pinned DeepSeek calibration via OpenRouter; durable spend admission, no retries."""
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.request

from .frontier_api import api_key, atomic_json, request_payload

MODEL = 'deepseek/deepseek-v4-pro-0813'
PROVIDER = 'deepinfra/fp8'
INPUT_RATE = 1.30 / 1_000_000
OUTPUT_RATE = 2.60 / 1_000_000
OUTPUT_CAP = 4096
RUN_CAP = 10.0
MAX_CALLS = 115  # one plumbing check plus the original 114-call ceiling
PROTOCOL = Path('docs/openrouter-calibration-protocol.md')


def payload(request):
    body = {'model': MODEL, 'messages': [{'role': 'user', 'content': request['prompt']}],
            'reasoning': {'effort': 'high'}, 'max_tokens': OUTPUT_CAP,
            'temperature': 0.6, 'top_p': 0.95,
            'provider': {'only': [PROVIDER], 'allow_fallbacks': False,
                         'require_parameters': True,
                         'max_price': {'prompt': 1.30, 'completion': 2.60}}}
    if request.get('labels'):
        fmt = request_payload(request)['text']['format'].copy()
        fmt.pop('type')
        body['response_format'] = {'type': 'json_schema', 'json_schema': fmt}
    return body


def reservation(body):
    size = len(json.dumps(body).encode())
    if size > 6000:
        raise ValueError('Request exceeds 6000-byte admission limit')
    # Tokenization overhead + a small allowance for provider end tokens.
    return (size + 1024) * INPUT_RATE + (OUTPUT_CAP + 32) * OUTPUT_RATE


class OpenRouterBackend:
    def __init__(self, ledger=Path('runs/openrouter-api-ledger.json'), budget=Path('budget.json')):
        self.key = api_key('OPENROUTER_API_KEY')
        self.ledger, self.budget = Path(ledger), Path(budget)
        self.metadata = {'model': MODEL, 'backend': 'openrouter_chat', 'provider': PROVIDER,
                         'reasoning': 'high', 'max_output_tokens': OUTPUT_CAP,
                         'temperature': 0.6, 'top_p': 0.95, 'seed_sent': False,
                         'pricing_checked': '2026-10-21', 'input_usd_per_million_upper': 1.30,
                         'output_usd_per_million_upper': 2.60, 'api_run_cap_usd': RUN_CAP,
                         'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()}

    def complete_batch(self, requests):
        return [self.complete(r) for r in requests]

    def complete(self, request):
        body = payload(request)
        reserved = reservation(body)
        identity = hashlib.sha256(json.dumps({'body': body, 'sample_id': request['seed']}, sort_keys=True).encode()).hexdigest()
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            budget = json.loads(self.budget.read_text())
            ledger = json.loads(self.ledger.read_text()) if self.ledger.exists() else {
                'model': MODEL, 'provider': PROVIDER, 'baseline_external_spend_usd': budget['cloud_spend_to_date'],
                'requests': {}}
            if ledger['model'] != MODEL or ledger['provider'] != PROVIDER:
                raise RuntimeError('Ledger configuration differs')
            entries = ledger['requests']
            if any(r['state'] != 'completed' or r['charged_upper_usd'] > r['reserved_usd'] + 1e-9
                   for r in entries.values()):
                raise RuntimeError('Unresolved request or accounting overrun; stop and audit, no retry')
            if identity in entries:
                return entries[identity]['response']
            committed = sum(r['charged_upper_usd'] for r in entries.values())
            if len(entries) >= MAX_CALLS or committed + reserved > min(RUN_CAP, budget.get('openrouter_api_subcap_usd', RUN_CAP)):
                raise RuntimeError('OpenRouter calibration cap reached')
            # cloud_spend_to_date excludes this active ledger until final reconciliation.
            prior = max(ledger['baseline_external_spend_usd'], budget['cloud_spend_to_date'])
            if prior + committed + reserved > min(100.0, budget['approved_pilot_cap']):
                raise RuntimeError('Cumulative pilot budget admission rejected')
            entries[identity] = {'state': 'reserved', 'reserved_usd': reserved, 'charged_upper_usd': reserved,
                                 'sample_id': request['seed'], 'started': time.time()}
            atomic_json(self.ledger, ledger)
            req = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',
                    json.dumps(body).encode(), {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.key})
            try:
                with urllib.request.urlopen(req, timeout=240) as resp:
                    result = json.load(resp)
            except Exception as exc:
                entries[identity].update(state='unknown', error_type=type(exc).__name__)
                if isinstance(exc, urllib.error.HTTPError):
                    entries[identity]['http_status'] = exc.code
                atomic_json(self.ledger, ledger)
                raise RuntimeError('OpenRouter request failed; reservation retained; no automatic retry') from None
            usage = result.get('usage') or {}
            cost = usage.get('cost')
            if (not all(type(usage.get(k)) is int and usage[k] >= 0 for k in ('prompt_tokens', 'completion_tokens'))
                    or type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0):
                entries[identity].update(state='unknown_usage', response_id=result.get('id'))
                atomic_json(self.ledger, ledger)
                raise RuntimeError('Missing valid usage/cost; reservation retained')
            charged = max(cost, usage['prompt_tokens'] * INPUT_RATE + usage['completion_tokens'] * OUTPUT_RATE)
            choices = result.get('choices') or []
            choice = choices[0] if len(choices) == 1 else {}
            message = choice.get('message') or {}
            response = {'text': message.get('content') or '', 'reasoning': message.get('reasoning'),
                        'prompt_tokens': usage['prompt_tokens'], 'completion_tokens': usage['completion_tokens'],
                        'finish_reason': choice.get('finish_reason'), 'usage': usage,
                        'response_id': result.get('id'), 'returned_model': result.get('model'),
                        'returned_provider': result.get('provider'), 'cost_usd': cost, 'cost_upper_usd': charged}
            valid = (result.get('provider') == 'DeepInfra' and result.get('model') == MODEL
                     and choice.get('finish_reason') in ('stop', 'length')
                     and isinstance(response['text'], str))
            entries[identity].update(state='completed' if valid else 'invalid_response',
                                     response=response, charged_upper_usd=charged)
            atomic_json(self.ledger, ledger)
            if not valid or charged > reserved + 1e-9:
                raise RuntimeError('Provider identity, response, or accounting mismatch; stop and audit')
            return response
