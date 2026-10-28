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
from selfplay.arena_mixed_scalar import envs
from selfplay.arena_mixed_scalar.cuda_policy import checkpoint_state, cohort_rows
from selfplay.arena_mixed_scalar.experiment import verify_run,select_endpoint
from selfplay.arena_mixed_scalar.runner import digest, exclusive_json, file_sha, load_sealed_cohort, pipeline_plan


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


def check_gate_evidence(gate,rows,binding,tolerance,*,require_restoration=True):
    """Verify complete admission evidence even when its scientific result is FAIL."""
    require(gate['binding']==binding and gate['tolerance']==tolerance and gate['optimizer_steps']==0,'Gate identity/config differs')
    require(gate['row_sha256']==[digest(r) for r in rows] and len(gate['records'])==len(rows)*2,'Gate vector coverage differs')
    failures=[]
    for mode,grad in [('eval',False),('train',True)]:
        records=[r for r in gate['records'] if r['mode']==mode]
        require([r['index'] for r in records]==list(range(len(rows))),'Gate ordering differs')
        for raw,observed in zip(rows,records,strict=True):
            require(observed['row_sha256']==digest(raw) and observed['grad_enabled']==grad,'Gate source/mode differs')
            vector=observed['actual_log_probs']
            finite=isinstance(vector,list) and len(vector)==len(raw['old_log_probs']) and all(isinstance(v,(int,float)) and math.isfinite(v) for v in vector)
            error=max(abs(a-b) for a,b in zip(vector,raw['old_log_probs'],strict=True)) if finite else None
            passed=finite and error<=tolerance
            require(observed['finite']==finite and observed['passed']==passed and observed['max_error']==error,'Gate finite/error/pass evidence differs')
            if not passed:failures.append({'mode':mode,'index':observed['index'],'max_error':error})
    require(gate['passed']==(not failures),'Gate aggregate result differs')
    if require_restoration:
        require(gate.get('learner_restored') and gate['learner_state_before']==gate['learner_state_after'],'Gate failed to restore learner')
    return {'passed':not failures,'records':len(gate['records']),'failures':failures,'source_sha256':digest(gate)}


def check_execution_diagnostic(report,fixture,base,config,tokenizer):
    """Check numerical evidence without invoking a model or accepting a failed gate.

    A well-preserved failed diagnostic is auditable, but never admits training.
    Legacy errors are descriptive; the precommitted scalar contract is mandatory.
    """
    import hashlib
    import torch
    from selfplay.arena_mixed_scalar import cuda_policy as backend
    backend.validate_config(config)
    binding=base['policy']['binding'];contract=backend.PHYSICAL_EXECUTION_CONTRACT
    require(type(report.get('passed')) is bool,'Diagnostic result must be an explicit Boolean')
    require(report['kind']=='mixed_scalar_physical_execution_diagnostic_v1'
        and report['fixture_sha256']==digest(fixture) and report['binding']==binding
        and report['physical_execution_contract']==contract and report['optimizer_steps']==0
        and report['training_use'] is False,'Diagnostic fixture/checkpoint/contract differs')
    require(fixture['kind']=='mixed_scalar_execution_fixtures_v1'
        and fixture['physical_execution_contract']==contract
        and fixture['outlier_actor_index']==6 and fixture['outlier_gate_index']==0,
        'Unexpected diagnostic fixture protocol')
    require(len(fixture['actor_requests'])==8 and len(fixture['gate_requests'])==2
        and len(fixture['smoke_requests'])==5
        and {x['environment'] for x in fixture['smoke_requests']}==set(config['environment_weights'])
        and fixture['actor_requests'][6]==fixture['gate_requests'][0]
        and next(x for x in fixture['smoke_requests'] if x['environment']=='chess')==fixture['gate_requests'][0],
        'Diagnostic requests/outlier/representative coverage differs')
    before,after=report['state_before'],report['state_after']
    require(report['state_restored'] is True and before==after,'Diagnostic did not restore complete state')
    require(before['binding']==binding and binding['version']==0 and base['step']==0
        and before['pending_version'] is None and before['adapter']==base['adapter_tensors']
        and before['optimizer']['state_sha256']==base['state_sha256']
        and before['optimizer']['step']==0 and not before['optimizer']['moments']
        and before['rng']==base['rng'] and before['config_sha256']==digest(config)
        and before['frozen_state_sha256']==base['frozen_state_sha256']
        and before['model_training'] is False and set(before['gradients'])==set(base['adapter_tensors'])
        and all(v is None for v in before['gradients'].values()),'Diagnostic is not the exact fresh base state')
    labels,label_ids=backend.select_labels(tokenizer)
    def request_fields(request):
        prompt=backend.action_prompt(request['observation'],request['actions'],labels)
        rendered=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,
            add_generation_prompt=True,enable_thinking=False)
        ids=tokenizer.encode(rendered,add_special_tokens=False)
        require(0<len(ids)<=config['max_context_tokens'],'Diagnostic prompt outside context cap')
        return {'prompt':prompt,'rendered_prompt':rendered,'input_ids':ids,'input_ids_sha256':digest(ids),
            'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'menu_sha256':digest(request['actions']),
            'labels_sha256':digest([labels,label_ids]),'actor_config_sha256':digest(config),
            'base_identity':binding['base_identity'],'physical_execution_contract_sha256':digest(contract)}
    def full_vector(vector,n):
        require(isinstance(vector,list) and len(vector)==n
            and all(type(v) in (int,float) and math.isfinite(v) for v in vector),
            'Missing/nonfinite diagnostic full legal vector')
        require(abs(math.fsum(math.exp(v) for v in vector)-1)<1e-5,'Diagnostic legal vector is not normalized')
    groups=report['groups'];require(set(groups)<=set(('actor_requests','gate_requests')),'Unexpected diagnostic group')
    for name,group in groups.items():
        requests=fixture[name];records=group['records'];fields=[request_fields(x) for x in requests]
        require(group['legacy_physical_batch_size']==len(requests) and group['scalar_physical_batch_size']==1
            and group['legacy_padded_tokens']==max(len(x['input_ids']) for x in fields)
            and [x['index'] for x in records]==list(range(len(requests))),'Diagnostic geometry/coverage differs')
        for request,expected,observed in zip(requests,fields,records,strict=True):
            n=len(request['actions'])
            require(observed['request_sha256']==digest(request)
                and observed['input_ids_sha256']==expected['input_ids_sha256']
                and observed['input_tokens']==len(expected['input_ids']) and observed['legal_actions']==n,
                'Diagnostic original token/request binding differs')
            full_vector(observed['legacy_log_probs'],n);full_vector(observed['scalar_log_probs'],n)
            error=max(abs(a-b) for a,b in zip(observed['legacy_log_probs'],observed['scalar_log_probs'],strict=True))
            require(observed['all_finite'] is True and observed['max_error']==error,'Diagnostic comparison error differs')
    scalar_group_pass=False
    if set(groups)=={'actor_requests','gate_requests'}:
        a,b=groups['actor_requests']['records'][6],groups['gate_requests']['records'][0]
        comparison=report['outlier_comparison']
        legacy_error=max(abs(x-y) for x,y in zip(a['legacy_log_probs'],b['legacy_log_probs'],strict=True))
        scalar_error=max(abs(x-y) for x,y in zip(a['scalar_log_probs'],b['scalar_log_probs'],strict=True))
        require(comparison=={'legacy_b8_b2_max_error':legacy_error,'scalar_group_max_error':scalar_error,
            'tolerance':.002,'legacy_condition_selected':False},'Diagnostic outlier comparison differs')
        scalar_group_pass=scalar_error<=.002
    smoke=report.get('scalar_smoke');smoke_pass=False;smoke_result=None
    if smoke is not None:
        require(type(smoke.get('passed')) is bool,'Scalar smoke result must be an explicit Boolean')
        require(smoke['kind']=='mixed_arena_numerical_derivative_smoke' and smoke['binding']==binding
            and smoke['optimizer_steps']==0 and smoke['training_use'] is False
            and smoke['state_restored'] is True and smoke['state_before']==smoke['state_after']==before,
            'Scalar smoke origin/restored state differs')
        rows=smoke.get('rows')
        if rows is not None:
            require(len(rows)==5,'Scalar smoke must cover all five fixed requests')
            for request,row in zip(fixture['smoke_requests'],rows,strict=True):
                expected={**request,**request_fields(request)}
                require(all(row.get(k)==v for k,v in expected.items()),'Scalar smoke request or native tokens differ')
                require(row['checkpoint']==binding and row['policy_version']==0
                    and row['actor_snapshot_sha256']==binding['adapter_sha256'] and row['thinking'] is False,
                    'Scalar smoke sampled checkpoint differs')
                full_vector(row['old_log_probs'],len(row['actions']));pick=row['action_index']
                require(type(pick) is int and 0<=pick<len(row['actions'])
                    and row['old_log_prob']==row['old_log_probs'][pick] and row['action']==row['actions'][pick]
                    and row['action_label']==labels[pick] and row['action_token_id']==label_ids[pick],
                    'Scalar smoke selected legal token differs')
                gen=torch.Generator(device='cpu').manual_seed(row['sample_seed'])
                probs=torch.softmax(torch.tensor(row['old_log_probs'],dtype=torch.float64),-1)
                require(int(torch.multinomial(probs,1,generator=gen))==pick,'Scalar smoke seeded draw differs')
                entropy=float(-(probs*probs.log()).sum())
                require(math.isclose(row['entropy'],entropy,rel_tol=1e-12,abs_tol=1e-12),'Scalar smoke entropy differs')
        gate=smoke.get('gate');derivative=smoke.get('derivative');gate_pass=False;derivative_pass=False
        if gate is not None:
            require(rows is not None,'Scalar gate lacks its sampled rows')
            smoke_result=check_gate_evidence(gate,rows,binding,.002,require_restoration=False)
            gate_pass=smoke_result['passed']
        if derivative is not None:
            require(gate is not None and rows is not None,'Derivative lacks complete source likelihood evidence')
            norm=derivative['gradient_norm'];finite=derivative['all_finite'];missing=derivative['missing_gradients']
            require(isinstance(missing,list) and set(missing)<=set(base['adapter_tensors'])
                and type(finite) is bool,'Malformed derivative coverage report')
            derivative_pass=finite and not missing and type(norm) in (int,float) and math.isfinite(norm) and norm>0
            require(derivative['finite_nonzero']==derivative_pass,'Derivative gate result differs')
            train=[x for x in gate['records'] if x['mode']=='train']
            if all(x['actual_log_probs'] is not None for x in train):
                loss=-math.fsum(x['actual_log_probs'][row['action_index']] for x,row in zip(train,rows,strict=True))/5
                require(type(derivative['loss']) in (int,float) and math.isfinite(derivative['loss'])
                    and math.isclose(loss,derivative['loss'],rel_tol=1e-5,abs_tol=1e-5),'Derivative loss differs from full scalar vectors')
        smoke_pass=gate_pass and derivative_pass
        if rows is None or gate is None or derivative is None:
            require(smoke['passed'] is False and smoke.get('reason'),'Incomplete smoke lacks preserved failure reason')
        if smoke['passed'] is True:require(smoke_pass,'False successful scalar derivative admission')
        else:require(not smoke_pass or smoke.get('reason'),'Unexplained failed scalar smoke')
        smoke_pass=smoke['passed'] is True
    computed=scalar_group_pass and smoke_pass
    if report['passed'] is True:
        require(computed,'False successful physical execution admission')
    else:
        require(not computed or report.get('reason'),'Unexplained failed execution diagnostic')
        require(smoke is not None or report.get('reason'),'Missing failed diagnostic evidence')
    return {'passed':report['passed'],'fixture_sha256':digest(fixture),'report_sha256':digest(report),
        'legacy_groups_present':sorted(groups),'scalar_gate':smoke_result,'optimizer_steps':0,
        'restored_exact_base':True,'gradient_evidence':'Reported finite nonzero derivative; loss independently recomputed from full vectors. No CPU model forward.'}


def verify_sealed_cohort_integrity(run,manifest,status,schedules,config,tokenizer):
    """Audit all completed rollouts even if later admission/update failed.

    A seal is rollout evidence, not optimizer publication. Partial evaluation is
    read only after the run is terminal, and never emits comparison scores here.
    """
    run=Path(run);phase=manifest['phase'];verified={};skipped=[]
    paths=sorted(run.glob('calibration/*/seal.json'))+sorted(run.glob('cohorts/*/seal.json'))+sorted(run.glob('evaluation/*/seal.json'))
    for path in paths:
        folder=path.parent;relative=str(folder.relative_to(run));category,label=relative.split('/')
        if category=='evaluation' and status['state'] not in {'failed','throughput_failed','complete'}:
            skipped.append(relative);continue
        if category=='calibration':
            require(phase=='training' and label in ('1','2'),'Unexpected calibration seal')
            schedule_key='calibration-'+label;checkpoint_folder=run/'checkpoints/base'
        elif category=='cohorts':
            require(phase=='training' and label.isdigit() and 1<=int(label)<=20,'Unexpected production seal')
            index=int(label);schedule_key='train-'+label;version=max(0,index-2)
            checkpoint_folder=run/'checkpoints'/('base' if version==0 else 'cycle'+str(version))
        else:
            require(phase=='evaluation' and label in ('base','final'),'Unexpected evaluation seal')
            schedule_key='eval-'+label;checkpoint_folder=run/'inputs/checkpoints'/label
        require(schedule_key in schedules,'Sealed rollout has no frozen schedule')
        seal,episodes=load_sealed_cohort(folder)
        require([e['schedule'] for e in episodes]==schedules[schedule_key], 'Sealed rollout differs from frozen schedule')
        require(len(episodes)==(224 if category=='evaluation' else 10),'Sealed rollout denominator differs')
        checkpoint=checkpoint_state(checkpoint_folder);check_checkpoint_config(checkpoint,config)
        require(seal['binding']==checkpoint['policy']['binding'],'Sealed behavior differs from actual saved checkpoint')
        opponent=None
        if category=='evaluation':
            base=checkpoint_state(run/'inputs/checkpoints/base');check_checkpoint_config(base,config)
            opponent=base['policy']['binding']
        provenance=verify_native_actions(episodes,config,tokenizer,opponent_binding=opponent)
        summary={'seal_sha256':file_sha(path),'episodes':len(episodes),'replay_verified':True,
            'actual_checkpoint_sha256':checkpoint['files'],'action_provenance':provenance,
            'optimizer_publication_claim':False,'comparison_scores_emitted':False}
        if (folder/'gate.json').exists():
            summary['gate_evidence']=check_gate_evidence(load(folder/'gate.json'),
                [r for e in episodes for r in e['transitions']],seal['binding'],config['likelihood_tolerance'])
        verified[relative]=summary
    return {'verified':verified,'active_evaluation_seals_not_read':skipped,
        'scope':'Every present sealed rollout on terminal runs, including uncommitted production and partial evaluation; no learning/comparison claim.'}


def verify_native_actions(episodes,config,tokenizer,**kwargs):
    from scripts.arena_mixed_scalar_action_audit import verify_action_rows
    return verify_action_rows(episodes,config,tokenizer,**kwargs)


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
    result={'kind':'mixed_arena_scalar_independent_audit_v1','manifest_sha256':file_sha(run/'manifest.json'),
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
    from scripts.arena_mixed_scalar_action_audit import get_tokenizer,verify_action_rows
    tokenizer=get_tokenizer(run)
    require(config['likelihood_tolerance']==.002,'Full-vector tolerance must remain unchanged')
    result['sealed_rollout_integrity']=verify_sealed_cohort_integrity(run,manifest,status,schedules,config,tokenizer)
    diagnostics={}
    for label in ('actor','learner'):
        path=run/(label+'-execution-diagnostic.json')
        if path.exists():
            diagnostic=load(path)
            base_folder=run/('checkpoints/base' if manifest['phase']=='training' else 'inputs/checkpoints/base')
            diagnostic_base=checkpoint_state(base_folder)
            check_checkpoint_config(diagnostic_base,config)
            diagnostics[label]=check_execution_diagnostic(diagnostic,load(run/'inputs/execution-diagnostic-fixtures.json'),
                diagnostic_base,config,tokenizer)
            if status['state'] in {'training_complete','complete'}:
                require(diagnostic['passed'],'Completed run has failed scalar diagnostic')
        elif status['state'] in {'training_complete','complete'}:
            raise ValueError('Complete run lacks scalar execution diagnostic')
    result['execution_diagnostics']=diagnostics
    if status['state'] in {'training_complete','complete'}:
        require(all((run/name).is_file() for name in ('actor-smoke.json','learner-smoke.json','cross-process-admission.json')),
                'Completed phase lacks mandatory numerical/derivative/cross-process admission')
    if (run/'cross-process-admission.json').exists():
        for name in ('actor-smoke.json','learner-smoke.json'):
            smoke=load(run/name)
            require(smoke['passed'] and smoke['state_restored'] and smoke['state_before']==smoke['state_after'],'Numerical smoke did not restore complete state')
            require(smoke['optimizer_steps']==0 and smoke['derivative']['finite_nonzero'],'Invalid numerical derivative admission')
        from selfplay.arena_mixed_scalar.cuda_policy import compare_actor_reports
        recomputed=compare_actor_reports(load(run/'actor-smoke.json'),load(run/'learner-smoke.json'))
        require(recomputed==load(run/'cross-process-admission.json') and recomputed['passed'],'Cross-process identity differs')
    if manifest['phase']=='training':
        base=checkpoint_state(run/'checkpoints/base') if (run/'checkpoints/base').is_dir() else None
        if base:
            require(base['step']==0 and base['policy']['binding']['version']==0,'Fresh baseline Adam must be zero')
            check_checkpoint_config(base,config)
        calibration=[]
        for index in (1,2):
            directory=run/'calibration'/str(index)
            if not (directory/'seal.json').is_file():continue
            seal,episodes=load_sealed_cohort(directory)
            require([e['schedule'] for e in episodes]==schedules[f'calibration-{index}'], 'Calibration differs from frozen schedule')
            require(len(episodes)==10,'Full ten-episode calibration cohort required')
            provenance=verify_action_rows(episodes,config,tokenizer)
            entry={'index':index,'episodes':len(episodes),'seal_sha256':file_sha(directory/'seal.json'),'replay_verified':True,'action_provenance':provenance}
            if (directory/'gate.json').is_file():
                gate=load(directory/'gate.json');rows=[r for e in episodes for r in e['transitions']]
                entry['gate_evidence']=check_gate_evidence(gate,rows,seal['binding'],config['likelihood_tolerance'])
            calibration.append(entry)
        result['calibration']=calibration
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
            require(endpoint == 20 and len(groups)==endpoint,'Prescribed training endpoint incomplete')
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
        report={'kind':'mixed_arena_scalar_independent_audit_v1','integrity_passed':False,'complete':False,
            'error_type':type(error).__name__,'reason':str(error)}
    exclusive_json(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k in ('integrity_passed','complete','state','phase','actual_optimizer_steps','attempted_cohorts','reason')}))
    if not report['integrity_passed']:sys.exit(1)


if __name__=='__main__':main()
