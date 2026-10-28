"""CPU-only independent replay, checkpoint, objective and completeness audit."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from selfplay.arena_mixed import envs
from selfplay.arena_mixed.cuda_policy import checkpoint_state, cohort_rows
from selfplay.arena_mixed.experiment import verify_run,select_endpoint
from selfplay.arena_mixed.runner import digest, exclusive_json, file_sha, load_sealed_cohort, pipeline_plan


def require(condition,message):
    if not condition:raise ValueError(message)


def load(path):return json.loads(Path(path).read_text())


def uncommitted_operations(run):
    """Distinguish committed checkpoint counts from interrupted optimizer work."""
    run=Path(run);pending=[]
    for path in sorted((run/'processes').glob('*-intent.json')):
        intent=load(path);payload=intent
        method=payload.get('method')
        if method not in {'update','profile'}:continue
        receipt=path.with_name(path.name.replace('-intent.json','-result.json'))
        result=load(receipt) if receipt.is_file() else None
        if result and result.get('passed') is True and result.get('id')==payload.get('request_id'):continue
        pending.append({'kind':'optimizer_rpc','method':method,'intent':str(path.relative_to(run)),
            'intent_sha256':file_sha(path),'result_sha256':file_sha(receipt) if receipt.is_file() else None,
            'outcome':'unknown; RPC intent is not evidence of a completed optimizer step'})
    for path in sorted(run.glob('cohorts/*/optimizer-intent.json'))+sorted(run.glob('calibration/*/optimizer-intent.json')):
        probe=path.parts[-3]=='calibration'
        terminal=path.parent/('profile.json' if probe else 'pipeline-commit.json')
        if terminal.is_file():continue
        evidence={str(p.relative_to(run)):file_sha(p) for p in (
            path.parent/'optimizer-result.json',path.parent/'update.json',run/'processes/learner-failure.json') if p.is_file()}
        pending.append({'kind':'optimizer_probe' if probe else 'optimizer_update',
            'intent':str(path.relative_to(run)),'intent_sha256':file_sha(path),
            'preserved_evidence_sha256':evidence,'outcome':'unknown; no complete bound publication'})
    return pending


def checkpoint_delta(before,after):
    from safetensors.torch import load_file
    a,b=load_file(str(Path(before)/'adapter.safetensors')),load_file(str(Path(after)/'adapter.safetensors'))
    require(set(a)==set(b),'Adapter schemas changed')
    return math.sqrt(sum(float((b[k].double()-a[k].double()).square().sum()) for k in a))


def check_checkpoint_config(state,config):
    require(state['policy']['config']==config and state['learning_rate']==config['learning_rate'],
            'Checkpoint configuration/learning rate differs from frozen protocol')


def check_profile_checkpoint(run,profile,base,config):
    folder=Path(run)/'calibration/1/profile-checkpoint'
    temporary=checkpoint_state(folder)
    check_checkpoint_config(temporary,config)
    proof=profile['temporary_checkpoint']
    require(temporary['files']==proof['files'] and temporary['policy']['binding']==proof['binding'],
            'Saved temporary profile checkpoint differs from report')
    require(temporary['step']==proof['optimizer_step']==base['step']+1,
            'Saved temporary profile checkpoint does not prove one Adam update')
    delta=checkpoint_delta(Path(run)/'checkpoints/base',folder)
    require(math.isclose(delta,profile['temporary_update']['parameter_delta_l2'],rel_tol=1e-6,abs_tol=1e-9),
            'Temporary profile checkpoint adapter delta differs')
    return {'saved_temporary_optimizer_steps':temporary['step']-base['step'],'parameter_delta_l2':delta}


def check_gate(gate,rows,binding,tolerance):
    require(gate['passed'] is True and gate['binding']==binding,'Missing passed exact behavior gate')
    require(gate['row_sha256']==[digest(r) for r in rows],'Gate rows differ from sealed draws')
    require(gate['tolerance']==tolerance and gate['optimizer_steps']==0,'Numerical gate contract differs')
    require(len(gate['records'])==len(rows)*2,'Incomplete eval/train vector coverage')
    for mode,grad in [('eval',False),('train',True)]:
        records=[r for r in gate['records'] if r['mode']==mode]
        require([r['index'] for r in records]==list(range(len(rows))),'Gate ordering/coverage differs')
        for raw,observed in zip(rows,records,strict=True):
            vector=observed['actual_log_probs']
            require(observed['passed'] and observed['finite'] and observed['grad_enabled']==grad,'Gate mode/finite evidence differs')
            require(observed['row_sha256']==digest(raw) and len(vector)==len(raw['old_log_probs']),'Gate vector identity differs')
            require(all(math.isfinite(v) for v in vector) and all(math.isfinite(v) for v in raw['old_log_probs']),
                    'Gate contains a nonfinite actual or behavior entry')
            error=max(abs(a-b) for a,b in zip(vector,raw['old_log_probs'],strict=True))
            require(math.isfinite(error) and error<=tolerance,'Gate vector exceeds fixed tolerance')
    if 'learner_restored' in gate:
        require(gate['learner_restored'] and gate['learner_state_before']==gate['learner_state_after'], 'Stale-behavior gate changed learner state')


def check_objective(report,episodes,config):
    rows=cohort_rows(episodes,config['environment_weights'])
    records=report['learner_records']
    raw=[r for e in episodes for r in e['transitions']]
    require(len(rows)==len(records),'Missing pre-update learner vectors')
    total=0.
    for index,(row,record) in enumerate(zip(rows,records,strict=True)):
        require(record['index']==index and record['row_sha256']==digest(raw[index]),'Learner objective row identity differs')
        require(record['source_binding']==report['source'] and record['behavior_binding']==report['behavior'],'Objective checkpoint binding differs')
        require(record['advantage']==row['advantage'] and record['weight']==row['weight']
            and record['action_index']==row['action_index'] and record['old_vector_sha256']==digest(row['old_log_probs']), 'Objective advantage/weight/action differs')
        values=record['learner_log_probs']
        require(len(values)==len(row['actions']) and all(math.isfinite(v) for v in values),'Learner vector coverage/finiteness differs')
        require(abs(sum(math.exp(v) for v in values)-1)<1e-5,'Learner vector not normalized')
        old=row['old_log_probs'];pick=row['action_index'];advantage=row['advantage'];weight=row['weight']
        ratio=math.exp(values[pick]-old[pick])
        clipped=max(1-config['clip_ratio'],min(1+config['clip_ratio'],ratio))
        entropy=-sum(math.exp(v)*v for v in values)
        kl=sum(math.exp(a)*(a-b) for a,b in zip(old,values,strict=True))
        total+=weight*(-min(ratio*advantage,clipped*advantage)-config['entropy_coefficient']*entropy+config['kl_coefficient']*kl)
    require(math.isclose(total,report['loss'],rel_tol=1e-4,abs_tol=.002),'Recorded clipped PPO objective differs')
    return {'rows':len(rows),'recomputed_loss':total,'reported_loss':report['loss']}


def audit(run):
    run=Path(run);manifest,status=verify_run(run)
    schedules=load(run/'inputs/schedule.json');config=manifest['config']['policy']
    history=load(run/'inputs/prior-fingerprints.json')
    schedule_report=envs.validate_schedule(schedules,prior_fingerprints=history)
    result={'kind':'mixed_arena_independent_audit_v1','manifest_sha256':file_sha(run/'manifest.json'),
        'status_sha256':file_sha(run/'status.json'),
        'source_auditor_sha256':file_sha(__file__),'state':status['state'],'phase':manifest['phase'],
        'schedule':schedule_report,'integrity_passed':False,'complete':False,
        'comparisons_available':False,'actual_optimizer_steps':0,'attempted_cohorts':0,
        'limitations':['One training seed; finite task families and bounded horizons.',
            'Chess uses tactical endgames; capped material is not full-game Elo.',
            'Coding is bounded AST repair, not freeform code authoring.',
            'Environment objective coefficients are not measured gradient-contribution percentages.',
            'Bounded lag one PPO is not exact on-policy RL.']}
    pending=uncommitted_operations(run)
    result.update(uncommitted_operations=pending,optimizer_outcome_unknown=bool(pending),
        optimizer_count_scope='Only independently verified committed checkpoints; unresolved operations are not counted as zero.')
    if status['state'] in {'training_complete','complete'}:
        require(not pending,'Completed phase contains an unresolved optimizer operation')
    if status['state']=='prepared':
        result['integrity_passed']=True;return result
    from scripts.arena_mixed_action_audit import get_tokenizer,verify_action_rows
    tokenizer=get_tokenizer(run)
    if status['state'] in {'training_complete','complete'}:
        require(all((run/name).is_file() for name in ('actor-smoke.json','learner-smoke.json','cross-process-admission.json')),
                'Completed phase lacks mandatory numerical/derivative/cross-process admission')
    if (run/'cross-process-admission.json').exists():
        for name in ('actor-smoke.json','learner-smoke.json'):
            smoke=load(run/name)
            require(smoke['passed'] and smoke['state_restored'] and smoke['state_before']==smoke['state_after'],'Numerical smoke did not restore complete state')
            require(smoke['optimizer_steps']==0 and smoke['derivative']['finite_nonzero'],'Invalid numerical derivative admission')
        from selfplay.arena_mixed.cuda_policy import compare_actor_reports
        recomputed=compare_actor_reports(load(run/'actor-smoke.json'),load(run/'learner-smoke.json'))
        require(recomputed==load(run/'cross-process-admission.json') and recomputed['passed'],'Cross-process identity differs')
    if manifest['phase']=='training':
        base=checkpoint_state(run/'checkpoints/base') if (run/'checkpoints/base').is_dir() else None
        if base:
            require(base['step']==0 and base['policy']['binding']['version']==0,'Fresh baseline Adam must be zero')
            check_checkpoint_config(base,config)
        checkpoints={0:base};groups=[]
        for directory in sorted((run/'cohorts').glob('*'),key=lambda p:int(p.name)) if (run/'cohorts').is_dir() else []:
            index=int(directory.name)
            if not (directory/'pipeline-commit.json').exists():continue
            require(index==len(groups)+1,'Committed cohort gap')
            seal,episodes=load_sealed_cohort(directory)
            require([e['schedule'] for e in episodes]==schedules[f'train-{index}'],'Training episodes differ from frozen schedule')
            raw=[r for e in episodes for r in e['transitions']]
            action_provenance=verify_action_rows(episodes,config,tokenizer)
            gate=load(directory/'gate.json');update=load(directory/'update.json');report=update['report']
            require(update==load(directory/'pipeline-commit.json'),'Pipeline publication differs from optimizer checkpoint')
            prior=checkpoints[index-1];behavior=checkpoints[max(0,index-2)]
            require(seal['binding']==behavior['policy']['binding'] and report['source']==prior['policy']['binding'],'Learner/actor source checkpoint differs')
            require(report['behavior']==seal['binding'] and report['policy_lag']==(0 if index==1 else 1),'Actual policy lag differs')
            check_gate(gate,raw,seal['binding'],config['likelihood_tolerance'])
            require(report['gate_sha256']==digest(gate) and report['cohort_sha256']==digest(episodes),'Optimizer objective sources differ')
            objective=check_objective(report,episodes,config)
            current=checkpoint_state(run/'checkpoints'/f'cycle{index}')
            check_checkpoint_config(current,config)
            require(current['step']==prior['step']+report['optimizer_steps'],'Actual saved Adam increment differs')
            require(report['adam_step_before']==prior['step'] and report['adam_step_after']==current['step'],'Report Adam count differs from checkpoints')
            require(current['policy']['binding']==update['checkpoint']['binding'] and current['files']==update['checkpoint']['files'],'Published checkpoint bytes differ')
            require(current['frozen_state_sha256']==base['frozen_state_sha256'],'Frozen base changed')
            prior_folder='base' if index==1 else f'cycle{index-1}'
            delta=checkpoint_delta(run/'checkpoints'/prior_folder,run/'checkpoints'/f'cycle{index}')
            require(math.isclose(delta,report['parameter_delta_l2'],rel_tol=1e-6,abs_tol=1e-9),'Saved adapter change differs from reported delta')
            checkpoints[index]=current
            groups.append({'cohort':index,'episodes':len(episodes),'decisions':len(raw),
                'policy_lag':report['policy_lag'],'optimizer_steps':report['optimizer_steps'],
                'reward_learning_signal':report['reward_learning_signal'],'objective':objective,
                'parameter_delta_l2':delta,'action_provenance':action_provenance})
        result.update(groups=groups,attempted_cohorts=len(groups),actual_optimizer_steps=sum(g['optimizer_steps'] for g in groups))
        if (run/'endpoint-selection.json').exists():
            selection=load(run/'endpoint-selection.json')
            first=load(run/'calibration/1/seal.json');second=load(run/'calibration/2/seal.json')
            overlap=load(run/'overlap-profile.json')
            costs=[first['elapsed_seconds']+overlap['observed_overlap_seconds'],second['elapsed_seconds']+overlap['observed_overlap_seconds']]
            require(selection['full_cost_inputs_seconds']==costs,'Timing selector omitted measured full costs')
            require(selection['worker_deadline_unix']==manifest['config']['deadline_unix']-120,'Timing selector deadline differs')
            require(abs(selection['remaining_seconds']-(selection['worker_deadline_unix']-selection['selected_at_unix']))<1e-6,'Remaining wall-time calculation differs')
            computed=select_endpoint(costs,selection['remaining_seconds'])
            require(all(selection.get(k)==v for k,v in computed.items()),'Timing-only endpoint rule differs')
            result['endpoint_selection']=selection
        if status['state']=='training_complete':
            completion=load(run/'training-complete.json');endpoint=result['endpoint_selection']['endpoint']
            require(endpoint in (20,40) and len(groups)==endpoint,'Prescribed training endpoint incomplete')
            require(completion['actual_optimizer_steps']==result['actual_optimizer_steps'],'Completion optimizer count differs')
            require(completion['attempted_cohorts']==endpoint and completion['endpoint']['files']==checkpoints[endpoint]['files'],'Endpoint export differs')
            require((run/'calibration/1/profile.json').is_file(),'Missing actual full update timing probe')
            profile=load(run/'calibration/1/profile.json')
            require(profile['passed'] and profile['state_restored'] and profile['state_before']==profile['state_after'] and profile['retained_optimizer_steps']==0,'Timing probe altered scientific baseline')
            require(profile.get('temporary_optimizer_steps')==1,'Timing probe did not prove an actual restored Adam update')
            result['profile_checkpoint_proof']=check_profile_checkpoint(run,profile,base,config)
            calibration_seal,calibration_episodes=load_sealed_cohort(run/'calibration/1')
            verify_action_rows(calibration_episodes,config,tokenizer)
            calibration_gate=load(run/'calibration/1/gate.json')
            check_gate(calibration_gate,[r for e in calibration_episodes for r in e['transitions']],base['policy']['binding'],config['likelihood_tolerance'])
            check_objective(profile['temporary_update'],calibration_episodes,config)
            result['complete']=True
            result['training_complete']=True
        result['comparisons_available']=False
    else:
        if status['state']=='complete':
            require((run/'evaluation-seal.json').is_file(),'Evaluation never fully sealed')
            all_episodes={}
            endpoints={name:checkpoint_state(run/'inputs/checkpoints'/name) for name in ('base','final')}
            for endpoint in endpoints.values():check_checkpoint_config(endpoint,config)
            for name in ('base','final'):
                seal,episodes=load_sealed_cohort(run/'evaluation'/name)
                require([e['schedule'] for e in episodes]==schedules['eval-'+name],'Held-out cohort differs from fixed paired schedule')
                require(seal['binding']==endpoints[name]['policy']['binding'],'Evaluated checkpoint identity differs')
                require(len(episodes)==224,'Full held-out cohort missing')
                verify_action_rows(episodes,config,tokenizer,opponent_binding=endpoints['base']['policy']['binding'])
                all_episodes[name]=episodes
            seal=load(run/'evaluation-seal.json')
            require(seal['baseline_seal_sha256']==file_sha(run/'evaluation/base/seal.json') and seal['final_seal_sha256']==file_sha(run/'evaluation/final/seal.json'),'Overall evaluation seal differs')
            paired=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
            metrics={}
            for endpoint,episodes in all_episodes.items():
                metrics[endpoint]={}
                for episode in episodes:
                    env=episode['environment'];family=episode['scenario_family_id'];seat=episode['learner_seats'][0]
                    paired[env][family][endpoint].append(episode['returns'][seat])
                    metrics[endpoint].setdefault(env,[]).append({'episode_id':episode['episode_id'],
                        'family':family,'learner_seat':seat,'opponent':episode['schedule']['opponent'],
                        'return':episode['returns'][seat],'metrics':episode['metrics']})
            comparisons={}
            for environment,families in paired.items():
                require(len(families)==16,'Expected sixteen independent paired scenario families')
                values=[]
                for family,sides in families.items():
                    require(set(sides)=={'base','final'} and len(sides['base'])==len(sides['final']),'Paired family imbalance')
                    a=sum(sides['base'])/len(sides['base']);b=sum(sides['final'])/len(sides['final'])
                    values.append({'scenario_family_id':family,'base':a,'final':b,'delta':b-a})
                comparisons[environment]={'paired_clusters':16,'base_mean':sum(v['base'] for v in values)/16,
                    'final_mean':sum(v['final'] for v in values)/16,'delta':sum(v['delta'] for v in values)/16,
                    'clusters':values}
            require(set(comparisons)==set(config['environment_weights']),'Missing environment in evaluation')
            result.update(complete=True,comparisons_available=True,comparisons=comparisons,metrics=metrics,
                weighted_return_delta=sum(config['environment_weights'][k]*v['delta'] for k,v in comparisons.items()),
                actual_optimizer_steps=endpoints['final']['step'],evaluation_episodes=448)
    result['integrity_passed']=True
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve previous audit; choose a new path')
    try:report=audit(args.run)
    except Exception as error:
        report={'kind':'mixed_arena_independent_audit_v1','integrity_passed':False,'complete':False,
            'error_type':type(error).__name__,'reason':str(error)}
    exclusive_json(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k in ('integrity_passed','complete','state','phase','actual_optimizer_steps','attempted_cohorts','reason')}))
    if not report['integrity_passed']:sys.exit(1)


if __name__=='__main__':main()
