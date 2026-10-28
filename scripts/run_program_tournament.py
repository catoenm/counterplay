"""Frozen CPU controller tournament: authoring versions, not RL checkpoints.

No API calls or neural inference. Model Python is parsed into the bounded IR,
never executed as Python. Prepare freezes roster, source archives and schedule.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
SEEDS = list(range(750000, 750006))
INITIAL_RATING = 1500.0
K_FACTOR = 24.0
EVALUATOR = ROOT / 'scripts/evaluate_program_screen.py'
CORE = ['selfplay/program_arena/'+name+'.py' for name in ('__init__','demo','dsl','examples','game','runner','spec')] + ['selfplay/program_python.py']


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()


def read(path):
    return json.loads(path.read_bytes())


def publish(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f'Incompatible immutable artifact: {path}')
        return
    temp = path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    try:
        with temp.open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.link(temp,path)
    finally:
        temp.unlink(missing_ok=True)


def save(path, value):
    publish(path,encoded(value))


def load_module(path, expected=None):
    raw = path.read_bytes()
    if expected is not None and sha(raw) != expected:
        raise ValueError(f'Archived source mismatch: {path}')
    name = '_tournament_evaluator_'+uuid.uuid4().hex
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(name,loader=None))
    module.__file__ = str(path)
    exec(compile(raw,str(path),'exec'),module.__dict__)
    return module


def deduplicate(roster):
    unique, aliases, seen = [], {}, {}
    for name, entry in sorted(roster.items()):
        fingerprint = entry['ir_sha256']
        if fingerprint in seen:
            aliases[name] = seen[fingerprint]
        else:
            unique.append(name); seen[fingerprint] = name
    return unique, aliases


def schedule_for(roster, seeds):
    identities, aliases = deduplicate(roster)
    specs = []
    for seed in seeds:
        for a,b in itertools.combinations(identities,2):
            for seat in (0,1):
                names = [a,b] if seat==0 else [b,a]
                origins = [roster[name]['origin'] for name in names]
                kind = 'peer_crossplay' if origins==['model','model'] else 'scripted_anchor' if origins==['scripted','scripted'] else 'fixed_reference'
                specs.append({'episode_id':f'tournament-{len(specs)+1:04d}','players':names,'seed':seed,
                              'max_rounds':48,'match_kind':kind})
    return specs, aliases


def elo_update(a,b,score):
    if score not in (0.,.5,1.) or not all(math.isfinite(v) for v in (a,b)):
        raise ValueError('Invalid Elo input')
    delta=K_FACTOR*(score-1/(1+10**((b-a)/400)))
    return a+delta,b-delta


def ratings_for(wrappers, identities):
    ratings={name:INITIAL_RATING for name in identities}
    history=[]; seen=set()
    for wrapper in wrappers:
        if wrapper['episode_id'] in seen:
            raise ValueError('Duplicate rated match')
        seen.add(wrapper['episode_id'])
        a,b=wrapper['players']
        if a==b:
            raise ValueError('Identical identity cannot be rated')
        scores=wrapper['record']['metrics']['scores']
        result=1. if scores[0]>scores[1] else 0. if scores[0]<scores[1] else .5
        before=[ratings[a],ratings[b]]
        ratings[a],ratings[b]=elo_update(*before,result)
        history.append({'episode_id':wrapper['episode_id'],'players':[a,b],'score_player0':result,
                        'before':before,'after':[ratings[a],ratings[b]]})
    return {'initial':INITIAL_RATING,'k_factor':K_FACTOR,'ratings':ratings,'match_history':history,
            'interpretation':'League-relative Elo for controller authoring versions, not RL checkpoints. Small fixed roster and six paired seeds; order dependent, not population skill or evidence of gradient training.',
            'order':'Frozen schedule: ascending seed, lexicographically sorted identity pairs, first orientation then swapped seats.',
            'score':'Actual final banked-point difference: win=1, draw=0.5, loss=0; no shaped adjudication.'}


def source_entries(evaluator):
    deep=ROOT/'runs/program-screen-v3'; qwen=ROOT/'runs/program-qwen38-27b-v1'; feedback=ROOT/'runs/program-feedback-v1'
    definitions=[('controller-python-1',deep,'generation-1.json',False),
                 ('qwen38-27b-1',qwen,'generation-1.json',False),
                 ('qwen38-27b-2',qwen,'generation-2.json',False),
                 ('qwen38-27b-1-r2',feedback,'generations/qwen38-27b-1-r2.json',True),
                 ('qwen38-27b-2-r2',feedback,'generations/qwen38-27b-2-r2.json',True)]
    core_hashes=read(deep/'plan.json')['source_sha256']
    result={}; files={}
    for identity,run,filename,is_feedback in definitions:
        plan=read(run/('original-plan.json' if is_feedback else 'plan.json'))
        if any(plan['source_sha256'][key]!=core_hashes[key] for key in CORE):
            raise ValueError('Tournament simulator/frontend sources are incompatible')
        response_bytes=(run/filename).read_bytes(); response=json.loads(response_bytes)
        if response['sample_id']!=identity:
            raise ValueError('Retained generation identity differs')
        summary=read(run/'summary.json')
        if not summary['complete']:
            raise ValueError('Retained run incomplete')
        if is_feedback:
            expected=summary['generation_sha256'][identity]
            prior=summary['holdout'][identity]['classification']
            if summary['selection_rule']!='round_2_even_if_invalid':
                raise ValueError('Feedback version is not mechanically selected final')
            if sha(encoded(response))!=summary['holdout'][identity]['generation_content_sha256']:
                raise ValueError('Feedback response does not match retained evaluation')
            model=plan['model']
        else:
            evaluation=read(run/'evaluation.json')
            expected=evaluation['generation_sha256'][filename]
            prior=next(g for g in summary['generations'] if g['sample_id']==identity)
            model=plan['model']
        if sha(response_bytes)!=expected or response['returned_model']!=model:
            raise ValueError('Retained generation bytes or authoring model differ')
        with evaluator.archived_runtime(run,plan) as runtime:
            classification,program=evaluator.classify_generation(response,runtime)
            if program is None or program.to_dict()!=prior['compiled_program']:
                raise ValueError('Retained source fails to reproduce the original compiled program')
        result[identity]={'origin':'model','authoring_model':model,'provider':response['returned_provider'],
                          'source_run':str(run),'source_generation':filename,'generation_sha256':expected,
                          'source_plan_sha256':sha((run/'plan.json').read_bytes()),
                          'source_summary_sha256':sha((run/'summary.json').read_bytes()),
                          'python_sha256':program.source_sha256,'ir_sha256':program.compiled.digest,
                          'python_program':program.to_dict(),
                          'authoring_version':'feedback_round_2' if is_feedback else 'original_retained',
                          'posthoc_parser_amendment':identity=='controller-python-1'}
        # All provenance artifacts are copied; no dependence on mutable summaries later.
        for source_name in ('plan.json','summary.json',filename):
            target=f'inputs/{identity}/'+('generation.json' if source_name==filename else source_name)
            files[target]=(run/source_name).read_bytes()
    return result,files,deep


def prepare(run):
    evaluator=load_module(EVALUATOR)
    roster,files,base=source_entries(evaluator)
    base_plan=read(base/'plan.json')
    with evaluator.archived_runtime(base,base_plan) as runtime:
        for name in ('nearest','value'):
            program=runtime.compile_program(runtime.reference_programs()[name])
            roster[name]={'origin':'scripted','authoring_model':None,'source_run':str(base),
                          'source_generation':None,'generation_sha256':None,'python_sha256':None,
                          'ir_sha256':program.digest,'ir_source':json.loads(program.source),'authoring_version':'scripted_anchor'}
    schedule,aliases=schedule_for(roster,SEEDS)
    if len(roster)>7 or len(schedule)>252:
        raise ValueError('Tournament exceeds seven identities / 252 games')
    for name,raw in files.items():publish(run/name,raw)
    publish(run/'runtime-plan.json',(base/'plan.json').read_bytes())
    for original in base_plan['source_sha256']:
        name=Path(original).name
        publish(run/'source'/name,(base/'source'/name).read_bytes())
    publish(run/'diagnostic_source/evaluator.py',EVALUATOR.read_bytes())
    publish(run/'diagnostic_source/tournament.py',Path(__file__).read_bytes())
    plan={'format_version':1,'kind':'frozen_controller_tournament','training':False,'new_api_calls':0,
          'external_spend_usd':0,'seeds':SEEDS,'max_rounds':48,'paired_seats':True,
          'roster':roster,'identities':deduplicate(roster)[0],'excluded_identical_aliases':aliases,
          'schedule':schedule,'schedule_sha256':sha(encoded(schedule)),
          'runtime_plan_sha256':sha((run/'runtime-plan.json').read_bytes()),
          'input_sha256':{name:sha(raw) for name,raw in files.items()},
          'source_sha256':base_plan['source_sha256'],
          'evaluator_sha256':sha(EVALUATOR.read_bytes()),'tournament_source_sha256':sha(Path(__file__).read_bytes()),
          'representative_rule':'Every model-versus-model pair, first new seed 750000, both seats. No selection by result.',
          'excluded_versions':['qwen38-27b-1-r1','qwen38-27b-2-r1'],
          'excluded_version_reason':'Intermediate feedback revisions excluded; first lineage round1 also has identical compiled fingerprint to mandatory round2.',
          'rating':{'initial':INITIAL_RATING,'k_factor':K_FACTOR,'identity':'controller authoring versions, not RL checkpoints'},
          'interpretation':'A small frozen controller league. All five retained model-authored programs remain visible, including failures. No policy weights are updated.'}
    save(run/'plan.json',plan)
    return {'status':'prepared_no_games','identities':len(plan['identities']),'scheduled_games':len(schedule),'plan_sha256':sha((run/'plan.json').read_bytes())}


@contextmanager
def runtime_for(run):
    plan=read(run/'plan.json')
    if (plan['seeds']!=SEEDS or plan['max_rounds']!=48 or plan['training'] is not False
            or len(plan['roster'])>7 or len(plan['schedule'])>252
            or plan['new_api_calls']!=0 or plan['external_spend_usd']!=0):
        raise ValueError('Frozen tournament limits differ')
    expected,aliases=schedule_for(plan['roster'],plan['seeds'])
    if (expected!=plan['schedule'] or aliases!=plan['excluded_identical_aliases']
            or deduplicate(plan['roster'])[0]!=plan['identities'] or sha(encoded(expected))!=plan['schedule_sha256']):
        raise ValueError('Frozen schedule differs')
    if sha(Path(__file__).read_bytes())!=plan['tournament_source_sha256']:
        raise ValueError('Live tournament runner differs from frozen source')
    if sha((run/'diagnostic_source/tournament.py').read_bytes())!=plan['tournament_source_sha256']:
        raise ValueError('Archived tournament runner differs')
    if sha((run/'runtime-plan.json').read_bytes())!=plan['runtime_plan_sha256']:
        raise ValueError('Archived runtime plan differs')
    for name,expected_hash in plan['input_sha256'].items():
        if sha((run/name).read_bytes())!=expected_hash:
            raise ValueError(f'Frozen provenance input differs: {name}')
    if read(run/'runtime-plan.json')['source_sha256']!=plan['source_sha256']:
        raise ValueError('Declared simulator source hashes differ')
    evaluator=load_module(run/'diagnostic_source/evaluator.py',plan['evaluator_sha256'])
    with evaluator.archived_runtime(run,read(run/'runtime-plan.json')) as runtime:
        programs={};python_programs={}
        for identity,entry in plan['roster'].items():
            if entry['origin']=='model':
                program=runtime.compile_python_program(entry['python_program']['python_source'])
                if program.to_dict()!=entry['python_program'] or program.source_sha256!=entry['python_sha256']:
                    raise ValueError('Frozen Python/IR provenance differs')
                # Source text must be identical to the retained program's published classification.
                generation_path=run/f'inputs/{identity}/generation.json'
                generation=read(generation_path)
                if (sha(generation_path.read_bytes())!=entry['generation_sha256']
                        or generation['sample_id']!=identity or generation['returned_model']!=entry['authoring_model']
                        or generation['returned_provider']!=entry['provider']):
                    raise ValueError('Authoring model/generation identity differs from frozen provenance')
                source_summary=read(run/f'inputs/{identity}/summary.json')
                source_classification=(source_summary['holdout'][identity]['classification'] if entry['authoring_version']=='feedback_round_2'
                    else next(g for g in source_summary['generations'] if g['sample_id']==identity))
                if source_classification['compiled_program']!=program.to_dict():
                    raise ValueError('Frozen roster changed the retained program')
                python_programs[identity]=program;programs[identity]=program.compiled
            else:
                programs[identity]=runtime.compile_program(entry['ir_source'])
            if programs[identity].digest!=entry['ir_sha256']:
                raise ValueError('Frozen compiled fingerprint differs')
        yield plan,evaluator,runtime,programs,python_programs


def summarize(plan,wrappers,evaluator):
    stats={}
    for identity in plan['identities']:
        rows=[];groups={}
        for w in wrappers:
            if identity in w['players']:
                seat=w['players'].index(identity);opponent=w['players'][1-seat]
                row=(w['record'],seat);rows.append(row);groups.setdefault(opponent,[]).append(row)
        stats[identity]={'origin':plan['roster'][identity]['origin'],
                         'authoring_model':plan['roster'][identity]['authoring_model'],
                         'overall':evaluator.aggregate(rows),
                         'by_opponent':{name:evaluator.aggregate(group) for name,group in groups.items()}}
    elo=ratings_for(wrappers,plan['identities'])
    return {'complete':len(wrappers)==len(plan['schedule']),'training':False,'new_api_calls':0,'external_spend_usd':0,
            'controller_identity':'controller authoring versions, not RL checkpoints','seeds':plan['seeds'],
            'matches':len(wrappers),'peer_matches':sum(w['match_kind']=='peer_crossplay' for w in wrappers),
            'all_replays_verified':True,'players':stats,'elo':elo,'interpretation':plan['interpretation'],
            'caveat':'Six seeds with paired seats and fixed opponents are a small, correlated sample. Elo depends on roster and match order; authoring/feedback differences are not RL checkpoint gains.',
            'scoring':'W/D/L is actual final banked-point difference at the fixed horizon. It is the game objective, not heuristic adjudication.'}


def run_matches(run):
    with runtime_for(run) as (plan,evaluator,runtime,programs,python_programs):
        path=run/'matches.jsonl';result_path=run/'summary.json'
        raw=path.read_bytes() if path.exists() else b''
        if raw and not raw.endswith(b'\n'):raise ValueError('Partial final JSONL line retained; no automatic replacement')
        wrappers=[json.loads(line) for line in raw.splitlines()]
        if len(wrappers)>len(plan['schedule']):raise ValueError('Extra games beyond frozen schedule')
        if result_path.exists() and len(wrappers)!=len(plan['schedule']):raise ValueError('Completed summary has incomplete matches')
        for wrapper,spec in zip(wrappers,plan['schedule']):
            evaluator.verify_wrapper(wrapper,spec,runtime,programs,python_programs)
        with path.open('ab') as stream:
            for spec in plan['schedule'][len(wrappers):]:
                names=spec['players']
                record=runtime.play([programs[name] for name in names],spec['seed'],48,
                                    origins=[plan['roster'][name]['origin'] for name in names])
                wrapper={'episode_id':spec['episode_id'],'players':names,'match_kind':spec['match_kind'],
                         'record':record,'python_programs':{name:python_programs[name].to_dict() for name in names if name in python_programs}}
                evaluator.verify_wrapper(wrapper,spec,runtime,programs,python_programs)
                stream.write((json.dumps(wrapper,separators=(',',':'),allow_nan=False)+'\n').encode())
                stream.flush();os.fsync(stream.fileno());wrappers.append(wrapper)
        summary=summarize(plan,wrappers,evaluator)
        summary.update(plan_sha256=sha((run/'plan.json').read_bytes()),matches_sha256=sha(path.read_bytes()))
        save(result_path,summary)
        selected=[w for w in wrappers if w['match_kind']=='peer_crossplay' and w['record']['seed']==SEEDS[0]]
        publish(run/'representative-matches.jsonl',b''.join((json.dumps(w,separators=(',',':'),allow_nan=False)+'\n').encode() for w in selected))
        save(run/'representatives.json',{'rule':plan['representative_rule'],'episode_ids':[w['episode_id'] for w in selected],
             'source_matches_sha256':summary['matches_sha256'],'representatives_sha256':sha((run/'representative-matches.jsonl').read_bytes())})
        return {'status':'complete','games':summary['matches'],'peer_games':summary['peer_matches'],
                'ratings':summary['elo']['ratings'],'representative_peer_games':len(selected)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--play',action='store_true')
    args=parser.parse_args();run=args.run.resolve();run.mkdir(parents=True,exist_ok=True)
    with (run/'.tournament.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=prepare(run) if args.prepare else run_matches(run)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
