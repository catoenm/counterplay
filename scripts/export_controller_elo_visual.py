"""Reverify the frozen controller tournament and expose its existing Elo history.

No new games, models, authoring calls, or training. The original recorded order
and all seven controller identities are retained. The run itself is read-only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_program_tournament import runtime_for

DEFAULT_FRAGMENT=Path('/Users/mitchellcatoen/.codex/visualizations/2026/10/20/'
                      '01a15b4f-05a3-7053-9034-4d1bc9f8d726/controller-tournament-rating-history.html')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def reconstruct(wrappers,identities,initial=1500.,k=24.):
    ratings={identity:float(initial) for identity in identities}
    series={identity:[{'game':0,'rating':float(initial),'played':0}] for identity in identities}
    counts={identity:0 for identity in identities}
    history=[];seen=set()
    for game,row in enumerate(wrappers,1):
        if row['episode_id'] in seen:raise ValueError('Duplicate rated episode')
        seen.add(row['episode_id'])
        a,b=row['players']
        if a==b or a not in ratings or b not in ratings:raise ValueError('Invalid rating identity')
        scores=row['record']['metrics']['scores']
        score=1. if scores[0]>scores[1] else .5 if scores[0]==scores[1] else 0.
        before=[ratings[a],ratings[b]]
        delta=k*(score-1/(1+10**((before[1]-before[0])/400)))
        ratings[a]+=delta;ratings[b]-=delta
        counts[a]+=1;counts[b]+=1
        history.append({'episode_id':row['episode_id'],'players':[a,b],'score_player0':score,
                        'before':before,'after':[ratings[a],ratings[b]]})
        for identity in identities:
            series[identity].append({'game':game,'rating':ratings[identity],'played':counts[identity]})
        if not math.isclose(sum(ratings.values()),len(identities)*initial,abs_tol=1e-7):
            raise ValueError('Elo conservation failed')
    return ratings,series,history


def label(identity,entry):
    if entry['origin']=='scripted':return identity.title()+' bot'
    if identity=='controller-python-1':return 'DeepSeek V4 Pro · P1'
    suffix='feedback 2' if entry['authoring_version']=='feedback_round_2' else 'original'
    number=re.fullmatch(r'qwen38-27b-(\d)(?:-r2)?',identity)
    if not number:raise ValueError('Unknown authoring identity')
    return f'Qwen3.8-27B · P{number[1]} {suffix}'


def build_payload(run):
    run=Path(run)
    paths=[run/'plan.json',run/'summary.json',run/'matches.jsonl']
    source_hashes={str(p):sha(p.read_bytes()) for p in paths}
    summary=json.loads(paths[1].read_text())
    raw=paths[2].read_bytes()
    if not raw.endswith(b'\n'):raise ValueError('Incomplete final match')
    wrappers=[json.loads(line) for line in raw.splitlines()]
    if (not summary['complete'] or summary['matches_sha256']!=sha(raw)
        or summary['plan_sha256']!=source_hashes[str(paths[0])]):raise ValueError('Unbound completed summary')
    with runtime_for(run) as (plan,evaluator,runtime,programs,python_programs):
        if plan['rating']['initial']!=1500. or plan['rating']['k_factor']!=24.:
            raise ValueError('Unexpected frozen Elo settings')
        if len(wrappers)!=len(plan['schedule']):raise ValueError('Incomplete tournament')
        for row,spec in zip(wrappers,plan['schedule']):
            evaluator.verify_wrapper(row,spec,runtime,programs,python_programs)
        ratings,series,history=reconstruct(wrappers,plan['identities'])
        recorded=summary['elo']
        if (recorded['initial']!=1500. or recorded['k_factor']!=24.
            or recorded['match_history']!=history or recorded['ratings']!=ratings):
            raise ValueError('Recorded Elo differs from actual game order or outcomes')
        versions={identity:{'label':label(identity,entry),
                  **{k:entry.get(k) for k in ('authoring_model','authoring_version','origin','ir_sha256','python_sha256','generation_sha256')},
                  'rating':ratings[identity], 'played':series[identity][-1]['played']}
                  for identity,entry in plan['roster'].items()}
        seed_blocks=[]
        for i,row in enumerate(wrappers,1):
            if i==len(wrappers) or wrappers[i]['record']['seed']!=row['record']['seed']:
                seed_blocks.append({'seed':row['record']['seed'],'game':i})
        records=[{'episode_id':row['episode_id'],
                  'sha256':sha(json.dumps(row['record'],sort_keys=True,separators=(',',':'),allow_nan=False).encode())}
                 for row in wrappers]
    for p in paths:
        if sha(p.read_bytes())!=source_hashes[str(p)]:raise ValueError('Source changed during export')
    return {'run':run.name,'initial':1500.,'k_factor':24.,'games':len(wrappers),
            'identities':plan['identities'],'controllers':versions,'series':series,'seed_blocks':seed_blocks,
            'records':records,'source_sha256':source_hashes,'all_replays_verified':True,
            'order':recorded['order'],'score':recorded['score'],
            'rating_identity':'Frozen controller authoring versions; not RL checkpoints or average model ability',
            'training':False,'revision':sha(json.dumps(source_hashes,sort_keys=True).encode())[:16]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'runs/program-tournament-v1')
    parser.add_argument('--fragment',type=Path,default=DEFAULT_FRAGMENT)
    parser.add_argument('--out',type=Path,default=ROOT/'runs/controller-tournament-visual/payload.json')
    args=parser.parse_args();data=build_payload(args.run)
    encoded=json.dumps(data,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    text,count=re.subn(r'(<script id="ct-data" type="application/json">).*?(</script>)',
                       lambda m:m[1]+encoded+m[2],args.fragment.read_text(),flags=re.S)
    if count!=1 or len(text.encode())>=1_000_000:raise ValueError('Invalid fragment size/data block')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');args.fragment.write_text(text)
    print(json.dumps({'verified_games':data['games'],'controllers':len(data['identities']),
                      'bytes':len(text.encode()),'fragment':str(args.fragment)}))


if __name__=='__main__':main()
