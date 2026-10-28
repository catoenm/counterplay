"""Frozen, verified cross-model challenge calibration. No weight updates."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

from .depth import MODEL, REVISION, family, read_tasks, render_depth
from .environment import RULES, Task, generate_tasks, parse_object
from .pilot import MLXBackend, grade_response
from .strict import PlanGrammar

PROTOCOL = Path('docs/challenge-calibration-protocol.md')
ROOT = Path('experiments/challenge-v1')
CAP = 4096


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()] if Path(path).exists() else []


def append(path, row):
    with Path(path).open('a') as f:
        f.write(json.dumps(row) + '\n')
        f.flush()


def prepare(root=ROOT):
    root.mkdir(parents=True, exist_ok=False)
    seen = set()
    for base in ('experiments', 'runs'):
        for p in Path(base).rglob('*.jsonl'):
            for r in read_jsonl(p):
                if set(Task.__dataclass_fields__) <= set(r):
                    seen.add(family(Task.from_dict({k: r[k] for k in Task.__dataclass_fields__})))
    excluded = sorted(seen)
    rows = []
    for i, depth in enumerate((2, 4, 6, 2, 4, 6)):
        for offset in range(10000):
            t = generate_tasks(1, 221001 + i * 10000 + offset, (depth,), 11)[0]
            if family(t) not in seen:
                break
        else:
            raise ValueError('No fresh procedural task')
        seen.add(family(t))
        rows.append(task_row(t, 'procedural', i))
    (root / 'procedural.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (root / 'excluded.json').write_text(json.dumps(excluded))
    (root / 'protocol.md').write_bytes(PROTOCOL.read_bytes())
    manifest = {'training': False, 'model': MODEL, 'revision': REVISION,
                'protocol_sha256': digest(PROTOCOL), 'excluded_families': len(excluded),
                'files': {n: digest(root / n) for n in ('procedural.jsonl', 'excluded.json', 'protocol.md')}}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def task_row(task, origin, index):
    return {**asdict(task), 'task_id': task.identity, 'family_id': family(task),
            'origin': origin, 'depth': len(task.shortest_plan()), 'render_seed': 223001 + index}


def proposal_prompt(index, previous):
    depth = (2, 4, 6, 2, 4, 6)[index]
    return (
        'Invent a finite planning puzzle and a correct solution. Your final answer must be one JSON object '
        'with exactly these fields: modulus (11), start ([x,y]), goal ([x,y]), actions '
        '(four distinct allowed action names), max_steps (' + str(depth) + '), witness '
        '(exactly ' + str(depth) + ' allowed action names). Coordinates are integers from 0 to 10. '
        'Execute your witness mentally and supply the correct resulting goal. All arithmetic is modulo 11. '
        'The arbiter executes your witness; an incorrect goal rejects the proposal. '
        'Aim for a shortest solution of exactly ' + str(depth) + ' steps, avoiding shortcuts. '
        'Try to challenge another reasoning model without making the problem invalid. '
        'Use a different start, goal and action-set combination from prior proposals. '
        'Rules: ' + json.dumps(RULES) + '\nEarlier accepted proposals: ' +
        json.dumps([{k: r[k] for k in Task.__dataclass_fields__} for r in previous])
    )


def validate_proposal(response, index, excluded, thinking):
    result = {'valid': False, 'accepted': False, 'requested_depth': (2, 4, 6, 2, 4, 6)[index]}
    try:
        if response.get('finish_reason') == 'length':
            raise ValueError('truncated')
        if thinking and '</think>' not in response['text']:
            raise ValueError('unfinished_reasoning')
        task = Task.from_dict(parse_object(response['text']))
        if task.modulus != 11 or len(task.actions) != 4 or task.max_steps != result['requested_depth'] or len(task.witness) != task.max_steps:
            raise ValueError('proposal_contract')
        depth = len(task.shortest_plan())
        result.update(valid=True, task=asdict(task), depth=depth,
                      depth_compliant=depth == result['requested_depth'], family_id=family(task))
        if family(task) in excluded:
            result['rejection'] = 'duplicate_or_prior_family'
        elif depth < 2:
            result['rejection'] = 'too_easy'
        else:
            result['accepted'] = True
    except (ValueError, KeyError, TypeError) as e:
        result['rejection'] = str(e)
    return result


class ThinkingBatch(MLXBackend):
    """Free reasoning followed by syntax-only legal final-plan decoding."""
    def __init__(self, thinking=True):
        super().__init__(MODEL, REVISION, thinking=thinking, top_k=20)
        self.grammars = {}
        end = self.tokenizer.encode('</think>', add_special_tokens=False)
        if len(end) != 1:
            raise ValueError('Expected one reasoning delimiter token')
        self.end_think = end[0]
        self.metadata.update(protocol='challenge-thinking-v1', max_tokens=CAP)

    def complete_batch(self, requests):
        from mlx_lm.generate import BatchGenerator
        from mlx_lm.sample_utils import apply_top_k, apply_top_p
        mx = self.mx
        prompts, samplers, processors = [], [], []

        def sampler(seed, state):
            key = mx.random.key(seed)
            def sample(logprobs):
                nonlocal key
                key, draw = mx.random.split(key)
                allowed = state.get('allowed')
                if allowed is not None:
                    ids = mx.array(allowed)
                    probs = apply_top_p(logprobs[:, ids], 0.95)
                    if len(allowed) > 20:
                        probs = apply_top_k(probs, 20)
                    return ids[mx.random.categorical(probs / 0.6, key=draw)]
                return mx.random.categorical(apply_top_k(apply_top_p(logprobs, 0.95), 20) / 0.6, key=draw)
            return sample

        def processor(grammar, n, state):
            def constrain(tokens, logits):
                generated = tokens[n:].tolist()
                if self.thinking:
                    if self.end_think not in generated:
                        return logits
                    generated = generated[generated.index(self.end_think) + 1:]
                allowed = grammar.allowed(generated)
                state['allowed'] = allowed
                mask = mx.full(logits.shape, -float('inf'), dtype=logits.dtype)
                mask[:, mx.array(allowed)] = 0
                return logits + mask
            return constrain

        for r in requests:
            formatted = self.tokenizer.apply_chat_template([{'role': 'user', 'content': r['prompt']}],
                         tokenize=False, add_generation_prompt=True, enable_thinking=self.thinking)
            ids = self.tokenizer.encode(formatted, add_special_tokens=False)
            prompts.append(ids)
            state = {}
            samplers.append(sampler(r['seed'], state))
            if r.get('labels'):
                key = (tuple(r['labels']), r['horizon'])
                if key not in self.grammars:
                    self.grammars[key] = PlanGrammar(lambda s: self.tokenizer.encode(s, add_special_tokens=False),
                        r['labels'], r['horizon'], self.tokenizer.eos_token_ids, exact_steps=True)
                processors.append([processor(self.grammars[key], len(ids), state)])
            else:
                processors.append([])
        gen = BatchGenerator(self.model, max_tokens=CAP, stop_tokens=[[x] for x in self.tokenizer.eos_token_ids],
                             completion_batch_size=len(requests), prefill_batch_size=len(requests))
        uids = gen.insert(prompts, samplers=samplers, logits_processors=processors)
        outputs, finished = {uid: [] for uid in uids}, {}
        try:
            while responses := gen.next_generated():
                for r in responses:
                    if r.finish_reason != 'stop':
                        outputs[r.uid].append(r.token)
                    if r.finish_reason:
                        finished[r.uid] = r.finish_reason
        finally:
            gen.close()
        if len(finished) != len(uids):
            raise RuntimeError('Incomplete batch')
        return [{'text': self.tokenizer.decode(outputs[uid]), 'prompt_tokens': len(prompts[i]),
                 'completion_tokens': len(outputs[uid]) + int(finished[uid] == 'stop'),
                 'finish_reason': finished[uid]} for i, uid in enumerate(uids)]


def model_backend(name):
    if name in ('qwen', 'qwen-off'):
        return ThinkingBatch(thinking=name == 'qwen')
    if name == 'openrouter':
        from .openrouter_api import OpenRouterBackend
        return OpenRouterBackend()
    from .frontier_api import OpenAIBackend
    return OpenAIBackend()


def audit_root(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, sha in manifest['files'].items():
        if digest(root / name) != sha:
            raise ValueError('Frozen calibration input changed: ' + name)


def run_manifest(out, config):
    if config.get('backend') == 'openrouter':
        from .openrouter_api import PROTOCOL
        config['provider_protocol_sha256'] = digest(PROTOCOL)
    hashes = {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')}
    path = out / 'manifest.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['config'] != config or saved['source_sha256'] != hashes:
            raise ValueError('Run configuration or source changed; cannot resume')
    else:
        out.mkdir(parents=True, exist_ok=False)
        (out / 'source').mkdir()
        for p in Path(__file__).parent.glob('*.py'):
            (out / 'source' / p.name).write_bytes(p.read_bytes())
        path.write_text(json.dumps({'config': config, 'source_sha256': hashes, 'training': False}, indent=2))


def propose(args):
    audit_root(args.root)
    out = args.out
    run_manifest(out, {'stage': 'propose', 'backend': args.backend, 'root_sha256': digest(args.root / 'manifest.json'),
                       'max_tokens': CAP, 'count': 6})
    saved = read_jsonl(out / 'proposals.jsonl')
    if [r['index'] for r in saved] != list(range(len(saved))):
        raise ValueError('Invalid proposal sequence')
    excluded = set(json.loads((args.root / 'excluded.json').read_text()))
    excluded.update(r['family_id'] for r in read_jsonl(args.root / 'procedural.jsonl'))
    tasks = []
    for r in saved:
        expected = validate_proposal(r, r['index'], excluded, args.backend == 'qwen')
        if expected != r['verification'] or r['prompt'] != proposal_prompt(r['index'], tasks):
            raise ValueError('Saved proposal did not reverify')
        if expected['accepted']:
            t = Task.from_dict(expected['task'])
            tasks.append(task_row(t, args.backend, r['index']))
            excluded.add(family(t))
    backend = model_backend(args.backend) if len(saved) < 6 else None
    for i in range(len(saved), 6):
        prompt = proposal_prompt(i, tasks)
        start = time.monotonic()
        response = backend.complete_batch([{'prompt': prompt, 'seed': 231001 + i}])[0]
        verification = validate_proposal(response, i, excluded, args.backend == 'qwen')
        row = {'index': i, 'prompt': prompt, 'seed': 231001 + i, **response, 'verification': verification,
               'seconds': time.monotonic() - start, 'model': backend.metadata}
        append(out / 'proposals.jsonl', row)
        saved.append(row)
        if verification['accepted']:
            t = Task.from_dict(verification['task'])
            tasks.append(task_row(t, args.backend, i))
            excluded.add(family(t))
        (out / 'tasks.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in tasks))
        print(f"proposal {i+1}/6: {verification}", flush=True)
    (out / 'tasks.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in tasks))
    summary = {'complete': len(saved) == 6, 'proposals': len(saved), 'accepted': len(tasks),
               'valid': sum(r['verification']['valid'] for r in saved),
               'depth_compliant': sum(r['verification'].get('depth_compliant', False) for r in saved),
               'accepted_depths': dict(Counter(r['depth'] for r in tasks)),
               'unique_action_sets': len({tuple(sorted(r['actions'])) for r in tasks}),
               'rejections': dict(Counter(r['verification'].get('rejection') for r in saved if not r['verification']['accepted'])),
               'completion_tokens': sum(r['completion_tokens'] for r in saved), 'training': False}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def solve(args):
    audit_root(args.root)
    paths = args.tasks
    task_rows = [pair for p in paths for pair in read_tasks(p)]
    # Retain source memberships if two proposers independently discover the same task.
    unique = {}
    for row, task in task_rows:
        if task.identity not in unique:
            unique[task.identity] = (row, task)
    task_rows = list(unique.values())
    config = {'stage': 'solve', 'backend': args.backend, 'max_tokens': CAP,
              'tasks_sha256': {str(p): digest(p) for p in paths}, 'batch_size': args.batch_size,
              'root_sha256': digest(args.root / 'manifest.json')}
    run_manifest(args.out, config)
    (args.out / 'tasks.jsonl').write_text(''.join(json.dumps(r) + '\n' for r, _ in task_rows))
    jobs = []
    for row, task in task_rows:
        for variant, n in (('canonical', 4), ('renamed_reordered', 2)):
            prompt, mapping = render_depth(row, task, variant)
            prompt += '\nUse the exact listed tool names. Finish with only the JSON plan.\n'
            for attempt in range(n):
                # Same task/attempt seed across presentations and across source pools.
                seed = (int(task.identity[:8], 16) + 241001 + attempt) % (2**32)
                jobs.append({'task': task, 'row': row, 'variant': variant, 'attempt': attempt,
                             'seed': seed, 'prompt': prompt, 'mapping': mapping,
                             'key': (task.identity, variant, attempt)})
    random.Random(251001).shuffle(jobs)
    by_key = {j['key']: j for j in jobs}
    saved = read_jsonl(args.out / 'rollouts.jsonl')
    done = set()
    for r in saved:
        key = (r['task_id'], r['variant'], r['attempt'])
        if key in done or key not in by_key:
            raise ValueError('Invalid saved rollout identity')
        j = by_key[key]
        grade = grade_response(j['task'], r, j['mapping'], args.backend == 'qwen')
        if r['prompt'] != j['prompt'] or r['seed'] != j['seed'] or any(r[k] != v for k, v in grade.items()):
            raise ValueError('Saved rollout did not reverify')
        done.add(key)
    pending = [j for j in jobs if j['key'] not in done]
    backend = model_backend(args.backend) if pending else None
    try:
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset:offset + args.batch_size]
            start = time.monotonic()
            responses = backend.complete_batch([{'prompt': j['prompt'], 'seed': j['seed'],
                                'labels': list(j['mapping']), 'horizon': j['task'].max_steps} for j in batch])
            duration = time.monotonic() - start
            for j, response in zip(batch, responses, strict=True):
                result = {k: j[k] for k in ('variant', 'attempt', 'seed', 'prompt')}
                result.update(task_id=j['task'].identity, origin=j['row']['origin'], depth=j['row']['depth'],
                              **response, **grade_response(j['task'], response, j['mapping'], args.backend == 'qwen'),
                              amortized_seconds=duration / len(batch), model=backend.metadata)
                append(args.out / 'rollouts.jsonl', result)
                saved.append(result)
            summarize(saved, args.out, len(jobs))
            print(f"{args.backend} solved {sum(r['success'] for r in saved)}/{len(saved)}; {len(saved)}/{len(jobs)} calls", flush=True)
    finally:
        summarize(saved, args.out, len(jobs))


def summarize(rows, out, total):
    groups, tasks = defaultdict(list), defaultdict(list)
    for r in rows:
        groups[(r['origin'], r['variant'], r['depth'])].append(r)
        if r['variant'] == 'canonical':
            tasks[r['task_id']].append(r)
    useful = [tid for tid, rs in tasks.items() if len(rs) == 4 and 0 < sum(r['success'] for r in rs) < 4
              and all(r['status'] in ('solved', 'wrong_goal') for r in rs)]
    summary = {'complete': len(rows) == total, 'calls': len(rows), 'expected_calls': total,
               'successes': sum(r['success'] for r in rows), 'statuses': dict(Counter(r['status'] for r in rows)),
               'canonical_tasks': len(tasks), 'canonical_tasks_solved': sum(any(r['success'] for r in rs) for rs in tasks.values()),
               'canonical_mixed_tasks': sum(0 < sum(r['success'] for r in rs) < len(rs) for rs in tasks.values()),
               'useful_mixed_task_ids': useful, 'completion_tokens': sum(r['completion_tokens'] for r in rows),
               'prompt_tokens': sum(r['prompt_tokens'] for r in rows),
               'by_group': {str(k): {'calls': len(rs), 'successes': sum(r['success'] for r in rs),
                                    'statuses': dict(Counter(r['status'] for r in rs))} for k, rs in groups.items()},
               'training': False}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=('prepare', 'propose', 'solve'))
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--backend', choices=('qwen', 'qwen-off', 'frontier', 'openrouter'), default='qwen')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--tasks', nargs='+', type=Path)
    ap.add_argument('--batch-size', type=int, default=4)
    args = ap.parse_args()
    if args.stage == 'prepare':
        prepare(args.root)
    elif args.out is None or not 1 <= args.batch_size <= 4:
        ap.error('Output path and batch size 1–4 required')
    elif args.stage == 'propose':
        if args.backend == 'qwen-off':
            ap.error('No thinking-disabled proposal arm')
        propose(args)
    elif not args.tasks:
        ap.error('Task files required')
    else:
        solve(args)


if __name__ == '__main__':
    main()
