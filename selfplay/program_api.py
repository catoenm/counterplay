"""Small, explicitly budgeted controller-generation screen; no training or retries.

Separate from the stopped puzzle experiment. Its unresolved request remains
reserved in the cumulative budget. This module never retries that request.
"""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.request

from .frontier_api import api_key, atomic_json

MODEL = 'deepseek/deepseek-v4-pro-0813'
PROVIDER = 'deepinfra/fp8'
INPUT_RATE, OUTPUT_RATE = 1.30e-6, 2.60e-6
OUTPUT_CAP = 16384
SCREEN_CAP = 2.0
MAX_CALLS = 6  # Four preserved JSON-interface attempts plus two Python-interface diagnostics.
LEDGER = Path('runs/program-api-ledger.json')


def ledger_commitment(path):
    if not path.exists():
        return 0.0
    ledger = json.loads(path.read_text())
    return sum(max(float(x.get('charged_upper_usd', 0)),
                   float(x.get('reserved_usd', 0)) if x['state'] != 'completed' else 0)
               for x in ledger['requests'].values())


def body_for(prompt, reasoning='high'):
    if reasoning not in ('high', 'off'):
        raise ValueError('Unsupported reasoning configuration')
    return {'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}],
            'reasoning': {'effort': 'high'} if reasoning == 'high' else {'enabled': False}, 'max_tokens': OUTPUT_CAP,
            'temperature': 0.6, 'top_p': 0.95,
            'provider': {'only': [PROVIDER], 'allow_fallbacks': False,
                         'require_parameters': True,
                         'max_price': {'prompt': 1.30, 'completion': 2.60}}}


def complete(prompt, sample_id, reasoning='high'):
    body = body_for(prompt, reasoning)
    encoded = json.dumps(body).encode()
    if len(encoded) > 40000:
        raise ValueError('Controller prompt exceeds the 40 KB admission limit')
    reserved = (len(encoded) + 2048) * INPUT_RATE + (OUTPUT_CAP + 64) * OUTPUT_RATE
    identity = hashlib.sha256(json.dumps({'body': body, 'sample_id': sample_id}, sort_keys=True).encode()).hexdigest()
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with Path('runs/external-budget.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        budget = json.loads(Path('budget.json').read_text())
        ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {
            'model': MODEL, 'provider': PROVIDER, 'purpose': 'generated-controller capability screen',
            'cap_usd': SCREEN_CAP, 'requests': {}}
        if ledger['model'] != MODEL or ledger['provider'] != PROVIDER:
            raise ValueError('Incompatible screen ledger')
        entries = ledger['requests']
        if any(x['state'] != 'completed' for x in entries.values()):
            raise RuntimeError('Screen request unresolved; reservation retained and no retry permitted')
        if identity in entries:
            return entries[identity]['response']
        committed = ledger_commitment(LEDGER)
        prior = float(budget['cloud_spend_to_date']) + ledger_commitment(Path('runs/openrouter-api-ledger.json'))
        prior += ledger_commitment(Path('runs/challenge-api-ledger.json'))
        # Cloud jobs must reserve their worst-case cost before admission here.
        prior += float(budget.get('cloud_reserved_usd', 0))
        if len(entries) >= MAX_CALLS or committed + reserved > SCREEN_CAP:
            raise RuntimeError('Controller screen cap reached')
        if prior + committed + reserved > min(100.0, float(budget['approved_pilot_cap'])):
            raise RuntimeError('Cumulative pilot budget admission rejected')
        entries[identity] = {'state': 'reserved', 'sample_id': sample_id,
                             'reserved_usd': reserved, 'charged_upper_usd': reserved,
                             'prior_commitment_usd': prior, 'started': time.time(),
                             'request_parameters': {k: v for k, v in body.items() if k != 'messages'}}
        atomic_json(LEDGER, ledger)
        request = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions', encoded,
                    {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + api_key('OPENROUTER_API_KEY')})
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                result = json.load(response)
        except Exception as error:
            entries[identity].update(state='unknown', error_type=type(error).__name__)
            if isinstance(error, urllib.error.HTTPError):
                entries[identity]['http_status'] = error.code
            atomic_json(LEDGER, ledger)
            raise RuntimeError('Controller request failed; cost reserved; no automatic retry') from None
        usage = result.get('usage') or {}
        cost = usage.get('cost')
        if (not all(type(usage.get(k)) is int and usage[k] >= 0 for k in ('prompt_tokens', 'completion_tokens'))
                or type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0):
            entries[identity].update(state='unknown_usage', response_id=result.get('id'))
            atomic_json(LEDGER, ledger)
            raise RuntimeError('Missing usage; controller reservation retained')
        charged = max(cost, usage['prompt_tokens'] * INPUT_RATE + usage['completion_tokens'] * OUTPUT_RATE)
        choices = result.get('choices') or []
        choice = choices[0] if len(choices) == 1 else {}
        message = choice.get('message') or {}
        response = {'text': message.get('content') or '', 'reasoning': message.get('reasoning'),
                    'finish_reason': choice.get('finish_reason'), 'usage': usage,
                    'response_id': result.get('id'), 'returned_model': result.get('model'),
                    'returned_provider': result.get('provider'), 'cost_usd': cost,
                    'cost_upper_usd': charged}
        valid = (result.get('model') == MODEL and result.get('provider') == 'DeepInfra'
                 and choice.get('finish_reason') in ('stop', 'length')
                 and isinstance(response['text'], str) and charged <= reserved + 1e-9)
        entries[identity].update(state='completed' if valid else 'invalid_response',
                                 response=response, charged_upper_usd=max(charged, reserved) if not valid else charged)
        atomic_json(LEDGER, ledger)
        if not valid:
            raise RuntimeError('Controller response identity or budget mismatch; stop and audit')
        return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt', required=True, type=Path)
    parser.add_argument('--sample-id', required=True)
    parser.add_argument('--reasoning', choices=('high', 'off'), default='high')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Refusing to overwrite a controller-generation record')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt.read_text()
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = complete(prompt, args.sample_id, args.reasoning)
    atomic_json(args.out, {'sample_id': args.sample_id, 'prompt': prompt,
                          'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                          'client_source_sha256': source_sha, 'reasoning_mode': args.reasoning,
                          'request_parameters': {k:v for k,v in body_for(prompt,args.reasoning).items() if k != 'messages'},
                          **result})
    print(json.dumps({'out': str(args.out), 'finish_reason': result['finish_reason'],
                      'cost_usd': result['cost_usd'], 'completion_tokens': result['usage']['completion_tokens']}))


if __name__ == '__main__':
    main()
