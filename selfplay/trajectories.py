"""Fresh task splits and verified presentation/trajectory augmentation."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

from .curriculum import check_splits, load_run
from .depth import MODEL, REVISION, family, read_tasks, render_depth
from .environment import Task, generate_tasks, parse_object, step
from .pilot import grade_response


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(out):
    seen = set()
    for path in list(Path('runs').glob('*/tasks.jsonl')) + list(Path('experiments').glob('*/*.jsonl')):
        for r in map(json.loads, path.read_text().splitlines()):
            if all(k in r for k in Task.__dataclass_fields__):
                seen.add(family(Task.from_dict({k:r[k] for k in Task.__dataclass_fields__})))
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'model':MODEL, 'revision':REVISION, 'excluded_prior_families':len(seen),
                'split_files':{}, 'protocol_sha256':digest('docs/trajectory-training-protocol.md')}
    for name, n, modulus, seed in [('candidates',1536,7,410011), ('validation',128,7,510011),
                                  ('test',128,7,610011), ('test_ood',64,11,710011)]:
        rows, cursor = [], 0
        while len(rows) < n:
            depth = 2 + len(rows) % 4
            task = generate_tasks(1, seed+cursor, (depth,), modulus)[0]
            cursor += 1
            if family(task) in seen:
                continue
            seen.add(family(task))
            rows.append({**asdict(task),'task_id':task.identity,'family_id':family(task),
                         'split':name,'depth':depth,'render_seed':seed+len(rows)})
        path = out/(name+'.jsonl')
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        manifest['split_files'][name] = {'count':n,'sha256':digest(path),'seed':seed}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


def augment(spec, task, response, variant):
    source_prompt, source_mapping = render_depth(spec, task)
    if response['prompt'] != source_prompt or not grade_response(task,response,source_mapping)['success']:
        raise ValueError('Unverified source response')
    actions = [source_mapping[label] for label in parse_object(response['text'])['plan']]
    if len(actions) != task.max_steps or not task.accepts(actions):
        raise ValueError('Source plan horizon or goal mismatch')
    prompt, mapping = render_depth(spec, task, variant)
    inverse = {action:label for label,action in mapping.items()}
    plan = [inverse[a] for a in actions]
    answer = json.dumps({'plan':plan},separators=(',',':'))
    if not grade_response(task,{'text':answer,'finish_reason':'stop'},mapping)['success']:
        raise ValueError('Augmented answer failed verification')
    state, states = task.start, []
    for action in actions:
        state = step(state,action,task.modulus)
        states.append(list(state))
    if state != task.goal:
        raise ValueError('Trajectory target mismatch')
    # Preserve presentation order exactly while omitting the target and answer.
    tools_text = prompt.split('Tools:\n',1)[1].split('\nThis is a fixed-horizon',1)[0]
    simulation = ('Execute the proposed tool plan from the starting state. '
                  'All calculations use the state before that tool call. '
                  'Return only a JSON object with key states, containing [x,y] after each action in order. '
                  'Do not include the starting state.\n'
                  f'm = {task.modulus}\nStart (x,y) = {task.start}\nTools:\n{tools_text}\n'
                  'Proposed plan: '+json.dumps(plan,separators=(',',':'))+'\n')
    common = {'task_id':task.identity,'family_id':family(task),'depth':spec['depth'],
              'variant':variant,'source_attempt':response['attempt']}
    direct = {**common,'kind':'plan','prompt':prompt,'answer':answer}
    trace = {**common,'kind':'state_trajectory','prompt':simulation,
             'answer':json.dumps({'states':states},separators=(',',':')),
             'environment_actions':actions,'verified_states':states}
    return direct, trace


def build(args):
    check_splits(args.splits)
    candidate_tasks, candidates = load_run(args.candidates)
    validation_tasks, validation = load_run(args.validation)
    for run,tasks,grouped in [(args.candidates,candidate_tasks,candidates),
                              (args.validation,validation_tasks,validation)]:
        attempts=json.loads((run/'manifest.json').read_text())['config']['attempts']
        if set(grouped)!=set(tasks) or any({r['attempt'] for r in rows}!=set(range(attempts))
                                         for rows in grouped.values()):
            raise ValueError('Missing collection attempts')
    for run,split in [(args.candidates,'candidates'),(args.validation,'validation')]:
        if (run/'tasks.jsonl').read_bytes() != (args.splits/(split+'.jsonl')).read_bytes():
            raise ValueError('Collection differs from frozen split')
    sets = [{r['family_id'] for r,_ in read_tasks(args.splits/(name+'.jsonl'))}
            for name in ('candidates','validation','test','test_ood')]
    for i,a in enumerate(sets):
        if any(a & b for b in sets[i+1:]):
            raise ValueError('Parent family leakage')
    def successes(grouped):
        return {tid:min([r for r in rows if r['success']],key=lambda r:r['attempt'])
                for tid,rows in grouped.items() if any(r['success'] for r in rows)}
    answers, valid_answers = successes(candidates), successes(validation)
    pools = {d:sorted(t for t in answers if candidate_tasks[t][0]['depth']==d) for d in range(2,6)}
    if min(map(len,pools.values())) < 8 or len(answers) < 96 or len(valid_answers) < 8:
        raise ValueError(f'Collection gate failed: depths={ {d:len(v) for d,v in pools.items()} }, validation={len(valid_answers)}')
    count = min(128,len(answers)//4*4)
    counts=Counter()
    while sum(counts.values()) < count:
        for d in range(2,6):
            if counts[d]<len(pools[d]) and sum(counts.values())<count:
                counts[d]+=1
    rng=random.Random(173)
    ids=[]
    for d in range(2,6):
        rng.shuffle(pools[d]); ids.extend(pools[d][:counts[d]])
    variants=('canonical','reordered','renamed','renamed_reordered')
    plans,traces=[],[]
    for tid in ids:
        spec,task=candidate_tasks[tid]
        for variant in variants:
            p,t=augment(spec,task,answers[tid],variant)
            p['source_run']=t['source_run']=str(args.candidates)
            plans.append(p); traces.append(t)
    valid=[]
    for tid in sorted(valid_answers):
        spec,task=validation_tasks[tid]
        for variant in variants:
            p,_=augment(spec,task,valid_answers[tid],variant)
            p['source_run']=str(args.validation); valid.append(p)
    args.out.mkdir(parents=True,exist_ok=False)
    manifest={'method':'verified model plans plus environment-labelled trajectory augmentation',
              'seed':173,'parent_tasks':count,'depth_counts':dict(counts),'selected_task_ids':ids,
              'available_by_depth':{d:len(v) for d,v in pools.items()},'validation_parents':len(valid_answers),
              'variants':variants,'examples_per_arm':8*count,'updates':2*count,
              'split_dir':str(args.splits),'split_manifest_sha256':digest(args.splits/'manifest.json'),
              'source_runs':{str(p):digest(p/'rollouts.jsonl') for p in (args.candidates,args.validation)},
              'files_sha256':{},'protocol_sha256':digest('docs/trajectory-training-protocol.md'),
              'source_sha256':digest(__file__)}
    for arm,rows in [('plans',plans+plans),('mixed',plans+traces)]:
        directory=args.out/arm; directory.mkdir()
        rng=random.Random(173); rng.shuffle(rows)
        for name,data in [('train',rows),('valid',valid)]:
            p=directory/(name+'.jsonl');p.write_text(''.join(json.dumps(r)+'\n' for r in data))
            manifest['files_sha256'][str(p.relative_to(args.out))]=digest(p)
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.out/'augmentation-source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in manifest.items() if k!='selected_task_ids'},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('freeze');p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('build')
    for key in ('candidates','validation','splits','out'):
        p.add_argument('--'+key,type=Path,required=True)
    args=ap.parse_args()
    freeze(args.out) if args.command=='freeze' else build(args)
