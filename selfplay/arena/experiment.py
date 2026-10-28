"""Run two full on-policy play -> RL update -> replay cycles across a game suite."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time

from . import ENVIRONMENTS, make_env
from .policy import ActionPolicy, action_prompt


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def append(path, row):
    with Path(path).open('a') as f:
        f.write(json.dumps(row)+'\n'); f.flush()


def game_specs(kind, cycle=0, candidate='base', opponent='heuristic', seeds=3):
    specs = []
    for index, game in enumerate(ENVIRONMENTS):
        if kind == 'train':
            for i in range(8):
                specs.append({'environment':game, 'seed':810000+cycle*10000+index*1000+i*2,
                              'regime':'train', 'policies':[candidate,candidate],
                              'episode_id':f'train-{cycle}-{game}-{i}'})
        else:
            for i in range(seeds):
                seed = 910000+index*1000+i*2
                for seat in (0,1):
                    names = [opponent,opponent]; names[seat]=candidate
                    specs.append({'environment':game, 'seed':seed, 'regime':'eval', 'policies':names,
                                  'candidate':candidate, 'candidate_seat':seat, 'opponent':opponent,
                                  'episode_id':f'{kind}-{candidate}-{opponent}-{game}-{i}-{seat}'})
    return specs


def play(policy, specs, path, batch_size=8):
    path = Path(path)
    if path.exists(): raise ValueError('Refusing to overwrite completed/partial rollout file')
    path.parent.mkdir(parents=True, exist_ok=True)
    states = [{'env':make_env(s['environment'],s['seed'],s['regime']),
               'record':{**s,'transitions':[],'events':[]}, 'initial_saved':False} for s in specs]
    completed=[]; started=time.monotonic()
    while states:
        pending=defaultdict(list)
        for index,s in enumerate(states):
            env,r=s['env'],s['record']
            if not s['initial_saved']:
                r['initial_state']=deepcopy(env.snapshot());s['initial_saved']=True
            if env.done: raise ValueError('Environment unexpectedly starts terminal')
            s['actions']={}
            for player in env.active_players():
                name=r['policies'][player]
                menu=deepcopy(env.legal_actions(player))
                if not menu or len(menu)>256:raise ValueError('Invalid action-menu size')
                # Same random streams across checkpoints; separate from environment RNG.
                sample_seed=r['seed']*100000+env.turn*37+player*11
                random.Random(sample_seed+907).shuffle(menu)
                obs=deepcopy(env.observation(player))
                if name=='heuristic':
                    action=env.heuristic_action(player)
                    if action not in menu:raise ValueError('Heuristic emitted illegal action')
                    s['actions'][player]=deepcopy(action)
                else:
                    request={'prompt':action_prompt(obs,menu,policy.labels),'actions':menu,
                             'sample_seed':sample_seed,'player':player,'turn':env.turn,'observation':obs}
                    pending[name].append((index,player,request))
        for checkpoint, jobs in pending.items():
            for offset in range(0,len(jobs),batch_size):
                batch=jobs[offset:offset+batch_size]
                responses=policy.choose([j[2] for j in batch],checkpoint)
                for (index,player,request),response in zip(batch,responses,strict=True):
                    if response['action'] not in request['actions']:raise ValueError('Policy emitted illegal action')
                    s=states[index]
                    s['actions'][player]=response['action']
                    s['record']['transitions'].append({**request,**response})
        remaining=[]
        for s in states:
            env,r=s['env'],s['record']
            before=env.turn
            env.step(s['actions'])
            r['events'].append({'turn':before,'actions':deepcopy(s['actions']),'state':deepcopy(env.snapshot())})
            if env.done:
                r.update(returns=env.returns(),metrics=env.metrics(),elapsed_batch_seconds=time.monotonic()-started)
                if len(r['returns'])!=2 or abs(sum(r['returns']))>1e-6:raise ValueError('Non-zero-sum game rewards')
                for transition in r['transitions']:transition['terminal_return']=r['returns'][transition['player']]
                append(path,r);completed.append(r)
                print(json.dumps({'stage':path.stem,'complete':len(completed),'total':len(specs),
                                  'game':r['environment'],'returns':r['returns']}),flush=True)
            else:
                if env.turn>100:raise ValueError('Non-terminating game')
                remaining.append(s)
        states=remaining
    return completed


def evaluate(policy, candidate, out):
    rows=[]
    for opponent in ('heuristic','base'):
        rows += play(policy,game_specs('eval',candidate=candidate,opponent=opponent),out/f'eval-{candidate}-{opponent}.jsonl')
    return rows


def summarize_eval(rows):
    groups=defaultdict(list)
    for row in rows:
        groups[(row['candidate'],row['opponent'],row['environment'])].append(row)
    return {' / '.join(key):{'games':len(rs),'mean_return':sum(r['returns'][r['candidate_seat']] for r in rs)/len(rs),
                'positive_outcomes':sum(r['returns'][r['candidate_seat']]>1e-8 for r in rs),
                'zero_outcomes':sum(abs(r['returns'][r['candidate_seat']])<=1e-8 for r in rs),
                'negative_outcomes':sum(r['returns'][r['candidate_seat']]<-1e-8 for r in rs),
                'termination_reasons':dict(Counter(str(r['metrics'].get('finish_reason','settlement')) for r in rs)),
                'scenario_seeds':len({r['seed'] for r in rs}),
                'raw_paired_scores':[{'seed':seed,'mean_return':sum(r['returns'][r['candidate_seat']] for r in rs if r['seed']==seed)/2}
                                     for seed in sorted({r['seed'] for r in rs})]}
            for key,rs in groups.items()}


def freeze(out, model):
    out.mkdir(parents=True,exist_ok=False)
    paths=list(Path('selfplay/arena').glob('*.py'))+[Path('docs/arena-rl-protocol.md')]
    config={'model':model,'cycles':2,'environments':list(ENVIRONMENTS),'selfplay_episodes_per_game_per_cycle':8,
            'evaluation_scenario_seeds_per_game_opponent':3,'evaluation_seat_swaps':True,
            'evaluation_opponents':['base','heuristic'],'horizons':{'market':6,'maze':8,'chess':16},
            'optimizer':'Adam','learning_rate':2e-5,'epochs_per_cycle':1,'minibatch_size':4,
            'inference_batch_size':8,
            'clip_ratio':.2,'entropy_coefficient':.01,'kl_to_rollout_coefficient':.02,'gradient_clip':1.,
            'baseline':'leave-one-episode-out, environment and player conditioned',
            'discount':1.,'objective':'equal environment/episode/seat weight; sum action score gradients',
            'training':'same current checkpoint in both seats, fresh episodes each cycle',
            'external_spend_usd':0,'source_sha256':{str(p):sha(p) for p in paths},
            'versions':{n:importlib.metadata.version(n) for n in ('mlx','mlx-lm','chess','python-chess')}}
    (out/'manifest.json').write_text(json.dumps(config,indent=2))
    (out/'source').mkdir()
    for p in paths:(out/'source'/p.name).write_bytes(p.read_bytes())
    all_specs={}
    for c in range(2):all_specs[f'train-{c}']=game_specs('train',cycle=c,candidate='base' if c==0 else 'cycle1')
    for c in ('base','cycle1','cycle2'):
        for opp in ('base','heuristic'):all_specs[f'eval-{c}-{opp}']=game_specs('eval',candidate=c,opponent=opp)
    all_specs['league-cycle2-cycle1']=game_specs('league',candidate='cycle2',opponent='cycle1')
    (out/'schedule.json').write_text(json.dumps(all_specs,indent=2))
    return config


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=Path('runs/arena-v1'))
    args=ap.parse_args()
    model=json.loads(Path('experiments/arena-v1/model.json').read_text())
    manifest=freeze(args.out,model)
    started=time.monotonic()
    policy=ActionPolicy(model)
    policy.save(args.out/'checkpoints/base','base')
    eval_rows=evaluate(policy,'base',args.out)
    training=[]
    (args.out/'evaluation.json').write_text(json.dumps(summarize_eval(eval_rows),indent=2))
    for cycle in range(2):
        source='base' if cycle==0 else 'cycle1';target=f'cycle{cycle+1}'
        episodes=play(policy,game_specs('train',cycle=cycle,candidate=source),args.out/f'train-rollouts-{cycle+1}.jsonl')
        training.append(policy.train(episodes,source,target,args.out/f'checkpoints/{target}',seed=72021+cycle))
        eval_rows+=evaluate(policy,target,args.out)
        (args.out/'evaluation.json').write_text(json.dumps(summarize_eval(eval_rows),indent=2))
    eval_rows+=play(policy,game_specs('league',candidate='cycle2',opponent='cycle1'),args.out/'league-cycle2-cycle1.jsonl')
    # A real model-state check: restoring base must exactly recover its adapter tensors.
    policy.activate('base')
    from mlx.utils import tree_flatten
    restored=dict(tree_flatten(policy.model.trainable_parameters()))
    if any(not policy.mx.array_equal(restored[k],v).item() for k,v in policy.checkpoints['base']):
        raise ValueError('Base checkpoint restoration failed')
    result={'complete':True,'cycles':2,'training':training,'evaluation':summarize_eval(eval_rows),
            'elapsed_seconds':time.monotonic()-started,'external_spend_usd':0,
            'base_restore_verified':True,'source_unchanged':all(sha(p)==v for p,v in manifest['source_sha256'].items()),
            'limitation':'Small action-policy RL pilot; no free-form reasoning training or general capability claim.'}
    (args.out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'complete':True,'cycles':2,'elapsed_seconds':result['elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
