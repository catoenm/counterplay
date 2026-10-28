"""Embed reverified, recorded trajectories in the calibration replay."""
import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selfplay.challenges import read_jsonl
from selfplay.depth import render_depth
from selfplay.environment import Task, parse_object, step
from selfplay.pilot import grade_response


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('fragment', type=Path)
    ap.add_argument('--proposals', type=Path, default=Path('runs/challenge-proposals-qwen'))
    ap.add_argument('--solvers', nargs='+', type=Path, default=[Path('runs/challenge-solve-qwen')])
    ap.add_argument('--proposal-index', type=int)
    ap.add_argument('--data-out', type=Path, default=Path('runs/challenge-visualization-data.json'))
    args = ap.parse_args()
    accepted = [r for r in read_jsonl(args.proposals / 'proposals.jsonl') if r['verification']['accepted']]
    if not accepted: raise ValueError('No accepted proposal to visualize')
    proposal = accepted[0] if args.proposal_index is None else next(r for r in accepted if r['index'] == args.proposal_index)
    task = Task.from_dict(proposal['verification']['task'])
    rows = [r for p in args.solvers for r in read_jsonl(p / 'tasks.jsonl')]
    row = next(r for r in rows if r['task_id'] == task.identity)
    def trace(actions, label, status, tokens):
        states = [list(task.start)]
        for a in actions: states.append(list(step(states[-1], a, task.modulus)))
        return {'actions': list(actions), 'states': states, 'label': label, 'status': status, 'tokens': tokens}
    traces = [trace(task.witness, 'Proposer witness', 'target reached', proposal['completion_tokens'])]
    samples = [(p, r) for p in args.solvers for r in read_jsonl(p / 'rollouts.jsonl') if r['task_id'] == task.identity]
    for path, r in sorted(samples, key=lambda pair: (str(pair[0]), pair[1]['variant'], pair[1]['attempt'])):
        _, mapping = render_depth(row, task, r['variant'])
        backend = json.loads((path / 'manifest.json').read_text())['config']['backend']
        grade = grade_response(task, r, mapping, backend == 'qwen')
        if any(r[k] != v for k, v in grade.items()): raise ValueError('Saved result does not reverify')
        legal = r['status'] in ('solved', 'wrong_goal')
        actions = [mapping[a] for a in parse_object(r['text'])['plan']] if legal else []
        status = 'target reached' if r['success'] else r['status'].replace('_', ' ')
        presentation = 'renamed' if r['variant'] == 'renamed_reordered' else 'canonical'
        solver = {'qwen': 'Qwen', 'openrouter': 'DeepSeek'}.get(backend, backend)
        traces.append(trace(actions, f"{solver} · {presentation} · {r['attempt']+1}", status, r['completion_tokens']))
    data = {'proposal_number': proposal['index']+1, 'task': row, 'traces': traces,
            'source': [str(args.proposals)] + [str(p) for p in args.solvers], 'training': False}
    encoded = json.dumps(data, separators=(',', ':')).replace('<', '\\u003c')
    text = args.fragment.read_text()
    text, count = re.subn(r'(<script id="ac-data" type="application/json">).*?(</script>)',
                         lambda m: m[1]+encoded+m[2], text, flags=re.S)
    if count != 1: raise ValueError('Expected one data block')
    args.fragment.write_text(text)
    args.data_out.write_text(json.dumps(data, indent=2))
    print(f'Exported {len(traces)} independently replayed trajectories')


if __name__ == '__main__': main()
