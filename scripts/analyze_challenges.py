"""Reverify challenge calibration records and compare paired task outcomes."""
import argparse
from collections import Counter, defaultdict
import json
from itertools import product
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selfplay.challenges import digest, read_jsonl
from selfplay.depth import read_tasks, render_depth
from selfplay.pilot import grade_response


def inspect_run(path):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    for name, sha in manifest['source_sha256'].items():
        if digest(path / 'source' / name) != sha:
            raise ValueError('Archived source changed')
    tasks = {t.identity: (r, t) for r, t in read_tasks(path / 'tasks.jsonl')}
    rows = read_jsonl(path / 'rollouts.jsonl')
    seen = set()
    for r in rows:
        row, task = tasks[r['task_id']]
        prompt, mapping = render_depth(row, task, r['variant'])
        prompt += '\nUse the exact listed tool names. Finish with only the JSON plan.\n'
        grade = grade_response(task, r, mapping, manifest['config']['backend'] == 'qwen')
        key = (r['task_id'], r['variant'], r['attempt'])
        if key in seen or prompt != r['prompt'] or any(r[k] != v for k, v in grade.items()):
            raise ValueError('Rollout did not reverify')
        seen.add(key)
    return manifest, tasks, rows


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r['origin'], r['variant'])].append(r)
    result = {}
    for key, rs in groups.items():
        tasks = defaultdict(list)
        for r in rs: tasks[r['task_id']].append(r)
        result[str(key)] = {
            'tasks': len(tasks), 'attempts': len(rs), 'successes': sum(r['success'] for r in rs),
            'success_rate': sum(r['success'] for r in rs) / len(rs),
            'tasks_solved': sum(any(r['success'] for r in ss) for ss in tasks.values()),
            'mixed_tasks': sum(0 < sum(r['success'] for r in ss) < len(ss) for ss in tasks.values()),
            'useful_mixed_tasks': sum(len(ss) == 4 and 0 < sum(r['success'] for r in ss) < 4 and
                all(r['status'] in ('solved', 'wrong_goal') for r in ss) for ss in tasks.values()) if key[1] == 'canonical' else None,
            'statuses': dict(Counter(r['status'] for r in rs)),
            'completion_tokens': sum(r['completion_tokens'] for r in rs),
            'prompt_tokens': sum(r['prompt_tokens'] for r in rs),
            'seconds': sum(r['amortized_seconds'] for r in rs)}
    return result


def compare(a, b):
    def aggregate(rows):
        tasks = defaultdict(list)
        for r in rows:
            if r['variant'] == 'canonical': tasks[r['task_id']].append(r)
        return {t: rs for t, rs in tasks.items() if len(rs) == 4}
    aa, bb = aggregate(a), aggregate(b)
    common = sorted(aa.keys() & bb.keys())
    gaps = []
    for t in common:
        if {r['prompt'] for r in aa[t]} != {r['prompt'] for r in bb[t]}:
            raise ValueError('Comparison prompts differ')
        gaps.append(sum(r['success'] for r in aa[t])/4 - sum(r['success'] for r in bb[t])/4)
    rng = random.Random(271001)
    boot = sorted(sum(rng.choice(gaps) for _ in gaps)/len(gaps) for _ in range(5000)) if gaps else []
    return {'paired_tasks': len(common), 'a_minus_b': sum(gaps)/len(gaps) if gaps else None,
            'task_bootstrap_95_interval': [boot[125], boot[4874]] if boot else None,
            'note': 'Exploratory task-cluster bootstrap; this small calibration set cannot establish generalization.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', nargs='+', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    combined, details, random_reference = defaultdict(list), {}, {}
    for path in args.runs:
        manifest, tasks, rows = inspect_run(path)
        name = manifest['config']['backend']
        combined[name].extend(rows)
        for tid, (_, task) in tasks.items():
            successes = sum(task.accepts(p) for p in product(task.actions, repeat=task.max_steps))
            random_reference[tid] = {'depth': task.max_steps, 'solving_sequences': successes,
                                    'legal_sequences': len(task.actions)**task.max_steps,
                                    'success_rate': successes / len(task.actions)**task.max_steps}
        details[str(path)] = {'backend': name, 'complete': len(rows) == 6*len(tasks),
                             'task_count': len(tasks), 'calls': len(rows), 'reverified': True}
    # Do not silently double-count an evaluation when passed overlapping run directories.
    for name, rows in combined.items():
        keys = {(r['task_id'], r['variant'], r['attempt']) for r in rows}
        if len(keys) != len(rows): raise ValueError('Overlapping evaluation runs: ' + name)
    result = {'training': False, 'runs': details, 'models': {k: summarize(v) for k,v in combined.items()},
              'comparisons': {}, 'uniform_random_legal_plan_reference': random_reference}
    if 'qwen' in combined and 'qwen-off' in combined:
        result['comparisons']['thinking_minus_disabled'] = compare(combined['qwen'], combined['qwen-off'])
    if 'frontier' in combined and 'qwen' in combined:
        result['comparisons']['frontier_minus_qwen'] = compare(combined['frontier'], combined['qwen'])
    if 'openrouter' in combined and 'qwen' in combined:
        result['comparisons']['deepseek_minus_qwen'] = compare(combined['openrouter'], combined['qwen'])
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
