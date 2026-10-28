"""Independent terminal planning-study audit; CPU tensors/tokenizer, never a model.

Prepared studies receive integrity-only preflight. Terminal failures retain partial
rollouts and uncertain optimizer outcomes without emitting held-out comparisons.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
import json
import math
from pathlib import Path
import random
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from selfplay.planning_curriculum import envs,schedule
from selfplay.planning_curriculum import cuda_policy as policy
from selfplay.planning_curriculum.admission import signal_gate,timing_gate
from selfplay.planning_curriculum.experiment import verify_run,TERMINAL,lease_clocks,main_overhead
from selfplay.planning_curriculum.runner import digest,canonical,file_sha,exclusive_json,load_sealed_cohort,stable_seed,verify_batch_timing
from scripts.arena_mixed_scalar_action_audit import get_tokenizer,verify_action_rows
from scripts.analyze_arena_mixed_scalar import check_gate,check_gate_evidence,checkpoint_delta


def require(value,message):
    if not value:raise ValueError(message)


def load(path):return json.loads(Path(path).read_text())


def close(a,b,message):
    require(type(a) in (int,float) and type(b) in (int,float) and math.isfinite(a) and math.isfinite(b)
        and math.isclose(a,b,rel_tol=1e-5,abs_tol=2e-5),message)


def check_objective(report,episodes,config):
    """Independently compute each clipped PPO/entropy/KL loss from full vectors."""
    policy.validate_config(config)
    require(report['passed'] is True and report['global_loss_multiplier']==.24
        and report['objective_contract']==policy.OBJECTIVE_CONTRACT,'Objective contract differs')
    raw=[r for e in episodes for r in e['transitions']]
    require(len(episodes)==2 and episodes[0]['scenario_family_id']!=episodes[1]['scenario_family_id'],
        'Two independent planning trajectories required')
    expected=[]
    for i,e in enumerate(episodes):
        require(e['learner_seats']==[0] and e['returns'] in ([0.,0.],[1.,0.]),'Planning objective roles/rewards differ')
        advantage=e['returns'][0]-episodes[1-i]['returns'][0]
        expected.extend((r,advantage,.5) for r in e['transitions'])
    require(len(report['learner_records'])==len(expected),'Missing actual learner full vectors')
    terms=[]
    for index,((row,advantage,weight),record) in enumerate(zip(expected,report['learner_records'],strict=True)):
        require(record['index']==index and record['row_sha256']==digest(row)
            and record['source_binding']==report['source'] and record['behavior_binding']==report['behavior']
            and record['advantage']==advantage and record['weight']==weight
            and record['action_index']==row['action_index'] and record['old_vector_sha256']==digest(row['old_log_probs'])
            and record['all_finite'] is True,'Objective row/advantage/weight/binding differs')
        values=record['learner_log_probs'];old=row['old_log_probs'];n=len(row['actions'])
        require(len(values)==len(old)==n and all(type(x) in (int,float) and math.isfinite(x) for x in values+old),
            'Missing/nonfinite full learner or behavior vector')
        require(abs(math.fsum(math.exp(v) for v in values)-1)<1e-5,'Learner vector not normalized')
        pick=row['action_index'];ratio=math.exp(values[pick]-old[pick]);clipped=max(.8,min(1.2,ratio))
        entropy=-math.fsum(math.exp(v)*v for v in values)
        kl=math.fsum(math.exp(a)*(a-b) for a,b in zip(old,values,strict=True))
        w=.24*weight
        terms.append({'policy_loss':-w*min(ratio*advantage,clipped*advantage),
            'entropy_loss':-w*.01*entropy,'kl_loss':w*.02*kl,
            'weighted_entropy':w*entropy,'weighted_kl':w*kl,
            'weighted_clip_fraction':w*float(abs(ratio-1)>.2)})
    covered=[]
    for batch in report['microbatches']:
        indices=batch['row_indices'];require(indices==list(range(len(covered),min(len(covered)+2,len(raw)))),'Microbatch coverage differs')
        covered+=indices
        for key in terms[0]:close(batch[key],math.fsum(terms[i][key] for i in indices),'Microbatch '+key+' differs')
        close(batch['loss'],math.fsum(terms[i][k] for i in indices for k in ('policy_loss','entropy_loss','kl_loss')),'Microbatch loss differs')
    require(covered==list(range(len(raw))),'Incomplete objective microbatches')
    require(len(report['microbatch_losses'])==len(report['microbatches']),'Loss count differs')
    for loss,batch in zip(report['microbatch_losses'],report['microbatches'],strict=True):close(loss,batch['loss'],'Loss list differs')
    for key in terms[0]:close(report[key],math.fsum(t[key] for t in terms),'Aggregate '+key+' differs')
    total=math.fsum(t[k] for t in terms for k in ('policy_loss','entropy_loss','kl_loss'))
    close(report['loss'],total,'Global .24 clipped PPO objective differs')
    signal=sum(abs(a)>1e-12 for _,a,_ in expected)
    require(report['nonzero_advantage_actions']==signal and report['reward_learning_signal']==bool(signal)
        and report['episodes']==2 and report['transitions']==len(raw),'Reward-signal denominator differs')
    return {'decisions':len(raw),'nonzero_advantage_actions':signal,'recomputed_loss':total,
        'global_loss_multiplier':.24,'no_realized_length_normalization':True}


def check_state(state,checkpoint,config,*,unpublished=False):
    require(state['adapter']==checkpoint['adapter_tensors']
        and state['optimizer']['state_sha256']==checkpoint['state_sha256']
        and state['optimizer']['step']==checkpoint['step'] and state['rng']==checkpoint['rng']
        and (state['binding'] is None and state['pending_version']==checkpoint['policy']['binding']['version']
            if unpublished else state['binding']==checkpoint['policy']['binding'] and state['pending_version'] is None)
        and state['config_sha256']==digest(config) and state['frozen_state_sha256']==checkpoint['frozen_state_sha256']
        and state['model_training'] is False and set(state['gradients'])==set(checkpoint['adapter_tensors'])
        and all(x is None for x in state['gradients'].values()),'Saved full state differs from actual checkpoint')


def check_transition(report,previous,current,source_folder,target_folder,config):
    steps=report['optimizer_steps']
    require(type(steps) is int and steps in (0,1) and current['step']==previous['step']+steps
        and report['adam_step_before']==previous['step'] and report['adam_step_after']==current['step'],
        'Actual persistent Adam step count differs')
    for item in (previous,current):
        require(item['policy']['config']==config and item['learning_rate']==config['learning_rate'],
            'Actual checkpoint config/LR differs')
    require(report['optimizer_before']['state_sha256']==previous['state_sha256']
        and report['optimizer_after']['state_sha256']==current['state_sha256']
        and current['frozen_state_sha256']==previous['frozen_state_sha256']
        and current['rng']==previous['rng'],'Optimizer/frozen/RNG checkpoint lineage differs')
    norm=report['gradient_norm_before_clip']
    require(type(norm) in (int,float) and math.isfinite(norm) and norm>=0 and bool(norm>0)==bool(steps)
        and report['gradient_clip']==1.,'Actual gradient/step contract differs')
    delta=checkpoint_delta(source_folder,target_folder)
    close(delta,report['parameter_delta_l2'],'Adapter delta differs from saved tensors')
    if steps==0:
        require(delta==0 and current['state_sha256']==previous['state_sha256'],'Zero-step checkpoint changed weights/Adam')
    return {'optimizer_steps':steps,'parameter_delta_l2':delta,'weights_changed':delta>0}


def check_probe(profile,selected,base,config,base_folder,target_folder):
    """Verify the real temporary Adam step separately from retained updates."""
    require(profile['episode_ids']==[e['episode_id'] for e in selected]
        and profile['cohort_sha256']==digest(selected)
        and profile['decisions']==sum(len(e['transitions']) for e in selected)
        and profile['training_use'] is False and profile['retained_optimizer_steps']==0,
        'Probe source/retained-step contract differs')
    require(profile['state_restored'] is True and profile['state_before']==profile['state_after'],
        'Probe restoration not proven')
    check_state(profile['state_before'],base,config)
    gate=None
    if profile.get('gate'):
        gate=check_gate_evidence(profile['gate'],[r for e in selected for r in e['transitions']],
            base['policy']['binding'],.002,require_restoration=False)
    if not profile['passed']:
        return {'passed':False,'retained_steps':0,'reason':profile.get('reason'),
            'temporary_step_outcome':profile.get('temporary_optimizer_steps')}
    require(gate and gate['passed'] and profile['temporary_optimizer_steps']==1,'No passed full gate and actual temporary Adam step')
    temporary=policy.checkpoint_state(target_folder,config['learning_rate'])
    require(profile['temporary_checkpoint']['files']==temporary['files']
        and profile['temporary_checkpoint']['binding']==temporary['policy']['binding']
        and profile['temporary_checkpoint']['optimizer_step']==temporary['step']==1,'Temporary checkpoint identity differs')
    update=profile['temporary_update']
    require(update['source']==update['behavior']==base['policy']['binding'] and update['policy_lag']==0
        and update['gate_sha256']==digest(profile['gate']) and update['cohort_sha256']==digest(selected),
        'Temporary objective source/gate differs')
    objective=check_objective(update,selected,config)
    transition=check_transition(update,base,temporary,base_folder,target_folder,config)
    mature=profile['mature_state_roundtrip']
    require(mature['passed'] is True and mature['state_before']==mature['state_after']
        and mature['optimizer_steps']==0 and mature['model_forwards']==0
        and type(mature['seconds']) in (int,float) and math.isfinite(mature['seconds']) and mature['seconds']>0,
        'Mature-state measurement changed state')
    check_state(mature['state_after'],temporary,config,unpublished=True)
    return {'passed':True,'retained_steps':0,'objective':objective,**transition}


def check_receipt(run,manifest_sha):
    start=load(run/'remote-job-start.json');receipt=load(run/'remote-exit.json')
    require(start['kind']=='planning_curriculum_job_attempt' and receipt['kind']=='planning_curriculum_job_exit'
        and start['manifest_sha256']==receipt['manifest_sha256']==manifest_sha
        and receipt['observed_manifest_sha256']==manifest_sha and receipt['start_sha256']==file_sha(run/'remote-job-start.json')
        and start['attempt_id']==receipt['attempt_id'] and receipt['verification_completed'] is True
        and receipt['phase']=='planning_curriculum' and type(receipt['exit_code']) is int,
        'Terminal wrapper receipt is not bound to the single expected launch')
    return {'sha256':file_sha(run/'remote-exit.json'),'exit_code':receipt['exit_code'],'attempt_id':receipt['attempt_id']}


def pending_operations(run):
    rows=[]
    for path in sorted((run/'processes').glob('*-intent.json')):
        intent=load(path)
        if intent.get('method') not in ('update','timing_probe'):continue
        result=path.with_name(path.name.replace('-intent.json','-result.json'))
        saved=load(result) if result.is_file() else None
        if saved and saved.get('passed') is True and saved.get('id')==intent.get('request_id'):continue
        rows.append({'intent':str(path.relative_to(run)),'sha256':file_sha(path),'kind':intent['method'],
            'outcome':'unknown; unresolved RPC is not a zero-step result'})
    paths=list(run.glob('arms/*/cohorts/*/optimizer-intent.json'))
    if (run/'admission/probe-optimizer-intent.json').exists():paths.append(run/'admission/probe-optimizer-intent.json')
    for path in sorted(paths):
        probe=path.parent==run/'admission';result=path.parent/('probe.json' if probe else 'pipeline-commit.json')
        saved=load(result) if result.is_file() else None
        if saved and (not probe or (saved.get('state_restored') is True
                and saved.get('passed') is True and saved.get('temporary_optimizer_steps')==1)):continue
        rows.append({'intent':str(path.relative_to(run)),'sha256':file_sha(path),
            'kind':'temporary_probe' if probe else 'retained_update',
            'state_restoration_reported':saved.get('state_restored') if probe and saved else None,
            'outcome':'unknown temporary step; restoration does not establish whether Adam stepped' if probe else
                'unknown; no complete checkpoint/restoration publication'})
    return rows


def verify_rpc_journal(run,manifest):
    """Reconstruct once-only commands and bind successful replies to publications."""
    for item in (run/'processes').glob('*.json'):
        if item.name.endswith(('-intent.json','-result.json')):
            require(re.fullmatch(r'(actor|learner)-\d{6}-(intent|result)\.json',item.name) is not None,
                'Noncanonical RPC filename')
            if item.name.endswith('-result.json'):
                require(item.with_name(item.name.replace('-result.json','-intent.json')).is_file(),
                    'RPC result has no exact durable intent')
    plan=manifest['deployment'];admission_stop=lease_clocks(plan)['admission_stop']
    initial=run/'processes/learner-000000-result.json'
    response=load(initial) if initial.exists() else None
    if not response or response.get('passed') is not True:
        intents=sorted((run/'processes').glob('*-intent.json'))
        require(len(intents)<=1,'Later RPC dispatched without completed base publication')
        if intents:
            intent=load(intents[0])
            require(intents[0].name=='learner-000000-intent.json' and intent['request_id']=='learner-000000'
                and intent['run_sha256']==file_sha(run/'manifest.json') and intent['method']=='publish_base'
                and intent['data_sha256']==digest({'stage_deadline_unix':admission_stop}),'Initial publication intent differs')
        return {'verified_successful_rpcs':0,'base_publication_available':False,
            'unresolved_or_failed_rpcs':['learner-000000'] if intents else []}
    base=response['result'];remote=Path(base['folder']).parent.parent
    require(Path(base['folder']).is_absolute() and str(Path(base['folder']).relative_to(remote))=='checkpoints/base','Base remote path differs')
    actual=policy.checkpoint_state(run/'checkpoints/base',manifest['config']['policy']['learning_rate'])
    require(base['files']==actual['files'] and base['binding']==actual['policy']['binding'] and base['optimizer_step']==0,'Base RPC publication differs')
    expected={'actor':[],'learner':[]}
    def add(role,method,data,path,deadline):
        expected[role].append((method,{**data,'stage_deadline_unix':deadline},path))
    def path(relative):return str(remote/relative)
    add('learner','publish_base',{},None,admission_stop)
    add('actor','load',base,None,admission_stop)
    specs=load(run/'inputs/schedule.json');first=specs['screen-hard'][0]
    instance=envs.make_env('planning',first['seed'],first['regime'],depth=first['depth']);view=envs.policy_view(instance,0)
    request={'episode_id':'numerical-planning','environment':'planning','scenario_family_id':first['scenario_family_id'],
        'turn':0,'player':0,'observation':view['observation'],'actions':view['legal_actions'],'sample_seed':731991}
    for role in ('actor','learner'):add(role,'smoke',{'requests':[request]},f'admission/{role}-smoke.json',admission_stop)
    add('actor','collect_profiled',{'specs':specs['screen-hard']+specs['screen-easy'],'checkpoint':base,'prefix':'admission/screen'},'admission/screen/seal.json',admission_stop)
    signal_path=run/'admission/signal-gate.json'
    if signal_path.exists() and load(signal_path)['passed']:
        add('learner','timing_probe',{'cohort':path('admission/screen'),'episode_ids':load(signal_path)['timing_probe_episode_ids']},'admission/probe.json',admission_stop)
    main_path=run/'comparison-admission.json'
    if main_path.exists():
        main=load(main_path);started=main['comparison_started_unix'];deadline=lease_clocks(plan,comparison_started=started)['comparison_stop']
        require(main['kind']=='planning_comparison_admitted' and main['run_sha256']==file_sha(run/'manifest.json')
            and main['comparison_deadline_unix']==deadline and started<admission_stop
            and main['signal_sha256']==file_sha(signal_path) and main['timing_sha256']==file_sha(run/'admission/timing-gate.json')
            and main['retained_updates']==80 and main['evaluation_episodes']==320,'Comparison admission identity/time differs')
        close(main['admission_spent_upper_usd'],lease_clocks(plan,checked=started)['admission_spent'],'Comparison admission cost differs')
        all_done=True;endpoints={'base':base}
        for label in schedule.ENDPOINTS[1:]:
            prefix='arms/'+label;refs={0:base}
            for role in ('actor','learner'):add(role,'reset_base',{'binding':base['binding']},prefix+'/'+role+'-base-restoration.json',deadline)
            for index in range(1,21):
                actor_version=max(0,index-2);current_version=index-1
                if actor_version in refs:
                    add('actor','collect',{'specs':specs[f'train-{label}-{index}'],'checkpoint':refs[actor_version],
                        'prefix':f'{prefix}/cohorts/{index}'},f'{prefix}/cohorts/{index}/seal.json',deadline)
                if current_version not in refs:break
                data={'cohort':path(f'{prefix}/cohorts/{index}'),'current':refs[current_version],
                    'behavior':refs[actor_version],'index':index,'prefix':f'{prefix}/cohorts/{index}',
                    'checkpoint_relative':f'{prefix}/checkpoints/cycle{index}'}
                add('learner','update',data,f'{prefix}/cohorts/{index}/update.json',deadline)
                publication=run/f'{prefix}/cohorts/{index}/pipeline-commit.json'
                if publication.exists():refs[index]=load(publication)['checkpoint']
                else:
                    # One next actor cohort may already be dispatched while
                    # the current learner update is incomplete.
                    if index<20:
                        add('actor','collect',{'specs':specs[f'train-{label}-{index+1}'],'checkpoint':refs[current_version],
                            'prefix':f'{prefix}/cohorts/{index+1}'},f'{prefix}/cohorts/{index+1}/seal.json',deadline)
                    break
            if 20 not in refs:all_done=False;break
            endpoints[label]=refs[20]
        if all_done:
            for label,reference in endpoints.items():add('actor','collect',{'specs':specs['eval-'+label],
                'checkpoint':reference,'prefix':'evaluation/'+label},'evaluation/'+label+'/seal.json',deadline)
    passed=0;pending=[]
    for role in ('actor','learner'):
        intents=sorted((run/'processes').glob(role+'-??????-intent.json'))
        require(len(intents)<=len(expected[role]),'More RPCs than the fixed study permits')
        for index,(intent_path,command) in enumerate(zip(intents,expected[role])):
            identifier=f'{role}-{index:06d}';method,data,target=command;intent=load(intent_path)
            require(intent_path.name==identifier+'-intent.json' and intent['request_id']==identifier
                and intent['run_sha256']==file_sha(run/'manifest.json') and intent['method']==method
                and intent['data_sha256']==digest(data),'RPC command/stage/source differs from reconstructed fixed protocol')
            receipt=intent_path.with_name(identifier+'-result.json')
            if not receipt.exists():
                require(index==len(intents)-1,'Later same-role RPC follows unresolved earlier request')
                pending.append(identifier);continue
            reply=load(receipt)
            if reply.get('passed') is not True:
                require(index==len(intents)-1,'Later same-role RPC follows failed earlier request')
                pending.append(identifier);continue
            require(reply['id']==identifier,'RPC result identity differs')
            require(reply['result']==(load(run/target) if target else base),'Successful RPC differs from durable publication')
            passed+=1
    return {'verified_successful_rpcs':passed,'unresolved_or_failed_rpcs':pending,'base_publication_available':True}


def check_smoke(report,request,base,config,tokenizer):
    import hashlib
    import torch
    from selfplay.arena_mixed_scalar import cuda_policy as renderer
    require(report['kind']=='mixed_arena_numerical_derivative_smoke' and report['optimizer_steps']==0
        and report['training_use'] is False and report['binding']==base['policy']['binding'],'Smoke contract differs')
    require(report['state_restored'] is True and report['state_before']==report['state_after'],'Smoke restoration failed')
    check_state(report['state_before'],base,config)
    rows=report.get('rows');gate=report.get('gate');derivative=report.get('derivative');passed=False
    if rows is not None:
        require(len(rows)==1 and all(rows[0].get(k)==v for k,v in request.items()),'Smoke fixed request differs')
        row=rows[0];labels,tokens=renderer.select_labels(tokenizer)
        prompt=renderer.action_prompt(request['observation'],request['actions'],labels)
        rendered=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=tokenizer.encode(rendered,add_special_tokens=False)
        require(row['prompt']==prompt and row['rendered_prompt']==rendered and row['input_ids']==ids
            and row['input_ids_sha256']==digest(ids) and row['prompt_sha256']==hashlib.sha256(prompt.encode()).hexdigest()
            and row['menu_sha256']==digest(row['actions']) and row['labels_sha256']==digest([labels,tokens])
            and row['actor_config_sha256']==digest(config) and row['thinking'] is False
            and row['physical_execution_contract_sha256']==digest(config['physical_execution_contract'])
            and row['checkpoint']==base['policy']['binding'],'Smoke native tokens/config/checkpoint differs')
        values=row['old_log_probs'];pick=row['action_index']
        require(len(values)==len(row['actions']) and all(type(v) in (int,float) and math.isfinite(v) for v in values)
            and abs(math.fsum(math.exp(v) for v in values)-1)<1e-5,'Smoke full vector malformed')
        probs=torch.softmax(torch.tensor(values,dtype=torch.float64),-1)
        chosen=int(torch.multinomial(probs,1,generator=torch.Generator().manual_seed(request['sample_seed'])))
        require(type(pick) is int and pick==chosen and row['action']==row['actions'][pick]
            and row['old_log_prob']==values[pick] and row['action_label']==labels[pick] and row['action_token_id']==tokens[pick],
            'Smoke sampled action differs')
        if gate is not None:
            evidence=check_gate_evidence(gate,rows,base['policy']['binding'],.002,require_restoration=False)
            if derivative is not None:
                norm=derivative['gradient_norm'];finite=derivative['all_finite']
                derivative_pass=finite is True and derivative['missing_gradients']==[] and type(norm) in (int,float) and math.isfinite(norm) and norm>0
                require(derivative['finite_nonzero']==derivative_pass,'Smoke derivative result differs')
                train=[r for r in gate['records'] if r['mode']=='train']
                if train[0]['actual_log_probs'] is not None:close(derivative['loss'],-train[0]['actual_log_probs'][pick],'Smoke derivative loss differs')
                passed=evidence['passed'] and derivative_pass
    if report['passed'] is True:require(passed,'False successful numerical smoke')
    else:require(not passed or report.get('reason'),'Failed smoke has no diagnostic reason')
    return {'passed':report['passed'],'optimizer_steps':0,'report_sha256':digest(report)}


def check_raw_action(row,request,binding,config,tokenizer):
    """Native provenance for an actual sampled row without inventing a return."""
    import hashlib
    import torch
    from selfplay.arena_mixed_scalar import cuda_policy as renderer
    require(all(canonical(row.get(k))==canonical(v) for k,v in request.items()),'Raw request differs')
    require(row['checkpoint']==binding and row['policy_version']==binding['version']
        and row['actor_snapshot_sha256']==binding['adapter_sha256'] and row['base_identity']==binding['base_identity']
        and row['actor_config_sha256']==digest(config) and row['thinking'] is False
        and row['physical_execution_contract_sha256']==digest(config['physical_execution_contract']), 'Raw policy identity differs')
    labels,ids=renderer.select_labels(tokenizer)
    prompt=renderer.action_prompt(request['observation'],request['actions'],labels)
    rendered=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    tokens=tokenizer.encode(rendered,add_special_tokens=False)
    require(row['prompt']==prompt and row['rendered_prompt']==rendered and row['input_ids']==tokens
        and 0<len(tokens)<=config['max_context_tokens'] and row['input_ids_sha256']==digest(tokens)
        and row['prompt_sha256']==hashlib.sha256(prompt.encode()).hexdigest()
        and row['menu_sha256']==digest(request['actions']) and row['labels_sha256']==digest([labels,ids]),'Raw native prompt differs')
    values,pick=row['old_log_probs'],row['action_index']
    require(type(values) is list and len(values)==len(request['actions'])
        and all(type(v) in (int,float) and math.isfinite(v) and v<=1e-6 for v in values)
        and abs(math.fsum(math.exp(v) for v in values)-1)<1e-5,'Raw full vector differs')
    probs=torch.softmax(torch.tensor(values,dtype=torch.float64),-1)
    actual=int(torch.multinomial(probs,1,generator=torch.Generator().manual_seed(request['sample_seed'])))
    require(type(pick) is int and pick==actual and row['action']==request['actions'][pick]
        and row['old_log_prob']==values[pick] and row['action_label']==labels[pick]
        and row['action_token_id']==ids[pick], 'Raw categorical action differs')
    require(math.isclose(row['entropy'],float(-(probs*probs.log()).sum()),rel_tol=1e-12,abs_tol=1e-12),'Raw entropy differs')


def check_unsealed_prefix(folder,specs,binding,config,tokenizer,run_sha,profiled):
    """Replay only actual saved action prefixes; never manufacture episodes."""
    folder=Path(folder);intent=load(folder/'intent.json')
    require(intent['kind']=='mixed_cohort_draw' and intent['run_sha256']==run_sha
        and intent['specs']==specs and intent['binding']==binding and intent['opponent_binding'] is None
        and intent['profiled'] is profiled,'Unsealed intent schedule/checkpoint differs')
    instances=[envs.make_env('planning',s['seed'],s['regime'],depth=s['depth']) for s in specs]
    existing={p.name for p in (folder/'batches').glob('*.json')};seen=set();index=0;actions_checked=0;pending=None
    while any(not e.done for e in instances):
        requests=[]
        for i,(instance,spec) in enumerate(zip(instances,specs,strict=True)):
            if instance.done:continue
            view=envs.policy_view(instance,0);seed_key=spec.get('paired_seed_key',spec['episode_id'])
            menu_seed=stable_seed('menu',seed_key,instance.turn,0);random.Random(menu_seed).shuffle(view['legal_actions'])
            requests.append((i,{'episode_id':spec['episode_id'],'environment':'planning','scenario_family_id':spec['scenario_family_id'],
                'turn':instance.turn,'player':0,'observation':view['observation'],'actions':view['legal_actions'],
                'menu_seed':menu_seed,'sample_seed':stable_seed('action',seed_key,instance.turn,0)}))
        choices={}
        for offset in range(0,len(requests),config['inference_batch_size']):
            batch=requests[offset:offset+config['inference_batch_size']];name=f'{index:06d}'
            ip=folder/'batches'/(name+'-intent.json');rp=folder/'batches'/(name+'.json')
            if not ip.exists():
                require(existing==seen,'Raw files appear beyond missing batch intent')
                return {'sampled_actions_verified':actions_checked,'pending_batch':None,'sealed':False,'returns_not_scored':True}
            seen.add(ip.name);saved=load(ip)
            require(saved['kind']=='mixed_action_batch' and saved['run_sha256']==run_sha and saved['role']=='learner'
                and saved['binding']==binding and saved['profiled'] is profiled
                and canonical(saved['requests'])==canonical([r for _,r in batch]),'Unsealed batch intent/order/public state differs')
            if not rp.exists():
                require(existing==seen,'Raw files appear beyond unresolved batch')
                return {'sampled_actions_verified':actions_checked,'pending_batch':name,'sealed':False,'returns_not_scored':True}
            seen.add(rp.name);raw=load(rp)
            require(raw['intent_sha256']==file_sha(ip) and raw['role']=='learner','Unsealed raw result binding differs')
            require(type(raw['rows']) is list and len(raw['rows'])==len(batch),'Unsealed raw row coverage differs')
            verify_batch_timing(raw,profiled)
            for (i,request),row in zip(batch,raw['rows'],strict=True):
                check_raw_action(row,request,binding,config,tokenizer);choices[i]=row['action'];actions_checked+=1
            index+=1
        for i,action in choices.items():instances[i].step({0:action})
    require(existing==seen,'Extra raw batches after terminal states')
    return {'sampled_actions_verified':actions_checked,'pending_batch':pending,'sealed':False,'returns_not_scored':True}


def comparisons(episodes):
    require(set(episodes)==set(schedule.ENDPOINTS),'All five held-out endpoints required')
    rows=[]
    for i,base in enumerate(episodes['base']):
        family=base['scenario_family_id'];values={}
        for endpoint in schedule.ENDPOINTS:
            episode=episodes[endpoint][i]
            require(episode['scenario_family_id']==family and episode['depth']==base['depth'],'Paired evaluation family differs')
            values[endpoint]=episode['returns'][0]
        differences={str(seed):values[f'{seed}-curriculum']-values[f'{seed}-control'] for seed in schedule.TRAINING_SEEDS}
        rows.append({'family':family,'depth':base['depth'],'returns':values,'curriculum_minus_control_by_seed':differences,
            'mean_curriculum_minus_control':math.fsum(differences.values())/2})
    require(len(rows)==64 and len({r['family'] for r in rows})==64,'Exactly64 paired held-out families required')
    rng=random.Random(20261025);boot=[]
    for _ in range(10000):boot.append(math.fsum(rows[rng.randrange(64)]['mean_curriculum_minus_control'] for _ in range(64))/64)
    boot.sort()
    by_seed={str(seed):math.fsum(r['curriculum_minus_control_by_seed'][str(seed)] for r in rows)/64 for seed in schedule.TRAINING_SEEDS}
    means={e:math.fsum(r['returns'][e] for r in rows)/64 for e in schedule.ENDPOINTS}
    average=math.fsum(by_seed.values())/2
    treatment_vs_base=math.fsum(means[f'{seed}-curriculum']-means['base'] for seed in schedule.TRAINING_SEEDS)/2
    return {'paired_families':64,'training_seed_replicates':2,'rows':rows,
        'endpoint_success_rates':means,'curriculum_minus_control_by_seed':by_seed,
        'mean_curriculum_minus_control':average,'mean_curriculum_minus_base':treatment_vs_base,
        'each_endpoint_minus_base':{e:v-means['base'] for e,v in means.items() if e!='base'},
        'per_depth_success':{str(d):{e:math.fsum(r['returns'][e] for r in rows if r['depth']==d)/16 for e in schedule.ENDPOINTS} for d in (3,4,5,6)},
        'operational_followup_gate':{'passed':all(v>0 for v in by_seed.values()) and average>=4/64 and treatment_vs_base>0,
            'authorization_for_more_work':False,'interpretation':'Predeclared diagnostic threshold, not a significance claim.'},
        'scenario_bootstrap_95_percentile':[boot[249],boot[9749]],'bootstrap_seed':20261025,
        'uncertainty':'Descriptive paired scenario bootstrap, conditional on only two training seeds; not broad training-run uncertainty.'}


def behavior_summary(episodes):
    rows=[r for e in episodes for r in e['transitions']]
    revisits=0
    for episode in episodes:
        seen=set()
        for row in episode['transitions']:
            state=canonical(row['observation']['state'])
            revisits+=state in seen;seen.add(state)
    return {'episodes':len(episodes),'successes':sum(e['returns'][0]==1 for e in episodes),
        'decisions':len(rows),'mean_actions':len(rows)/len(episodes) if episodes else None,
        'mean_sampled_categorical_entropy':math.fsum(r['entropy'] for r in rows)/len(rows) if rows else None,
        'revisited_pre_action_states':revisits,'revisit_definition':'A public state previously observed before an action in the same episode.'}


def audit(run,*,require_receipt=True):
    run=Path(run).resolve();manifest,status=verify_run(run)
    require(status['state']=='prepared' or status['state'] in TERMINAL,'Only prepared or terminal studies may be audited; no active evaluation scoring')
    require(manifest['sources'].get('scripts/analyze_planning_curriculum.py')==file_sha(__file__),
        'Independent auditor is not bound to frozen source closure')
    config=manifest['config']['policy'];policy.validate_config(config)
    require(manifest['config']['maximum_policy_episodes']==544 and manifest['config']['attempted_retained_updates']==80
        and manifest['config']['max_policy_lag']==1,'Fixed study denominator/config differs')
    schedules=load(run/'inputs/schedule.json');inventory=load(run/'inputs/prior-scenario-inventory.json')
    verified=schedule.validate_schedules(schedules,inventory['fingerprints'])
    require(verified==load(run/'inputs/schedule-verification.json'),'Frozen schedule verification differs')
    result={'kind':'planning_curriculum_independent_audit_v1','manifest_sha256':file_sha(run/'manifest.json'),
        'status_sha256':file_sha(run/'status.json'),'source_auditor_sha256':file_sha(__file__),
        'state':status['state'],'integrity_passed':False,'complete':False,'comparisons_available':False,
        'schedule':verified,'actual_retained_optimizer_steps':0,'committed_training_cohorts':0,
        'limitations':['Single-player planning curriculum, not adversarial self-play.',
            'One-token legal actions without freeform reasoning; physical scalar NF4/FP32 execution.',
            'Two paired training seeds; fixed64-family evaluation. No model gradient replay in this CPU audit.']}
    if status['state']=='prepared':result['integrity_passed']=True;return result
    if require_receipt:
        result['wrapper_receipt']=check_receipt(run,result['manifest_sha256'])
        if status['state']!='failed':require(result['wrapper_receipt']['exit_code']==0,'Successful terminal state has nonzero wrapper exit')
    for tree in ('admission','arms','evaluation','checkpoints','processes'):
        for path in (run/tree).rglob('*') if (run/tree).exists() else []:
            if path.is_file() and path.suffix in ('.json','.pt','.safetensors') and not path.name.startswith('.'):
                require(status['artifacts'].get(str(path.relative_to(run)))==file_sha(path),'Terminal artifact is unbound: '+str(path.relative_to(run)))
    result['rpc_journal']=verify_rpc_journal(run,manifest)
    pending=pending_operations(run);result['unresolved_optimizer_operations']=pending;result['optimizer_outcome_unknown']=bool(pending)
    tokenizer=get_tokenizer(run);cache={}
    def checkpoint(relative):
        if relative not in cache:
            item=policy.checkpoint_state(run/relative,config['learning_rate'])
            require(item['policy']['config']==config,'Checkpoint exact policy config differs');cache[relative]=item
        return cache[relative]
    base=checkpoint('checkpoints/base') if (run/'checkpoints/base/policy.json').exists() else None
    if base:require(base['step']==0 and base['policy']['binding']['checkpoint_id']=='base'
        and base['policy']['binding']['version']==0 and not base['moments'],'Shared baseline is not zero-step Adam')
    # Every existing seal is replayed, including uncommitted groups and failed
    # terminal evaluation. Partial evaluation never emits comparisons below.
    cohorts={};unsealed={};all_eval={};actual=0;groups=[]
    folders=[run/'admission/screen']+[p.parent for p in sorted((run/'arms').glob('*/cohorts/*/intent.json'))]+[p.parent for p in sorted((run/'evaluation').glob('*/intent.json'))]
    for folder in folders:
        if not (folder/'intent.json').exists():
            require(not folder.exists() or not list((folder/'batches').glob('*.json')),'Raw draws without a cohort intent')
            continue
        relative=str(folder.relative_to(run));parts=relative.split('/')
        if parts[0]=='admission':specs=schedules['screen-hard']+schedules['screen-easy'];source='checkpoints/base';expected_profile=True
        elif parts[0]=='arms':
            label,index=parts[1],int(parts[3]);seed,arm=label.split('-');require(label in schedule.ENDPOINTS[1:] and 1<=index<=20,'Undeclared arm/cohort')
            specs=schedules[f'train-{seed}-{arm}-{index}'];version=max(0,index-2)
            source='checkpoints/base' if version==0 else f'arms/{label}/checkpoints/cycle{version}';expected_profile=False
        else:
            endpoint=parts[1];require(endpoint in schedule.ENDPOINTS,'Undeclared evaluation endpoint');specs=schedules['eval-'+endpoint]
            source='checkpoints/base' if endpoint=='base' else f'arms/{endpoint}/checkpoints/cycle20';expected_profile=False
        source_state=checkpoint(source)
        if not (folder/'seal.json').exists():
            unsealed[relative]=check_unsealed_prefix(folder,specs,source_state['policy']['binding'],config,tokenizer,
                result['manifest_sha256'],expected_profile)
            continue
        seal,episodes=load_sealed_cohort(folder)
        require([e['schedule'] for e in episodes]==specs and seal['binding']==source_state['policy']['binding']
            and seal['profiled'] is expected_profile,'Sealed rollout source/schedule/timing mode differs')
        provenance=verify_action_rows(episodes,config,tokenizer)
        cohorts[relative]={'episodes':len(episodes),'actions':provenance['actions'],'seal_sha256':file_sha(folder/'seal.json')}
        if parts[0]=='evaluation':all_eval[parts[1]]=episodes
        if (folder/'gate.json').exists():
            cohorts[relative]['gate']=check_gate_evidence(load(folder/'gate.json'),[r for e in episodes for r in e['transitions']],seal['binding'],.002)
    result['sealed_rollouts']=cohorts;result['unsealed_draw_prefixes']=unsealed
    smokes={}
    for role in ('actor','learner'):
        path=run/f'admission/{role}-smoke.json'
        if path.exists():
            require(base is not None,'Smoke lacks actual base checkpoint');spec=schedules['screen-hard'][0]
            instance=envs.make_env('planning',spec['seed'],spec['regime'],depth=spec['depth']);view=envs.policy_view(instance,0)
            request={'episode_id':'numerical-planning','environment':'planning','scenario_family_id':spec['scenario_family_id'],
                'turn':0,'player':0,'observation':view['observation'],'actions':view['legal_actions'],'sample_seed':731991}
            smokes[role]=check_smoke(load(path),request,base,config,tokenizer)
    result['numerical_smokes']=smokes
    cross=run/'admission/cross-process.json'
    if cross.exists():
        recomputed=policy.compare_actor_reports(load(run/'admission/actor-smoke.json'),load(run/'admission/learner-smoke.json'))
        require(recomputed==load(cross) and recomputed['passed'],'Exact cross-process agreement differs')
    if cohorts or unsealed:require(len(smokes)==2 and all(v['passed'] for v in smokes.values()) and cross.exists(),'Experimental draws lack passed numerical admission')
    signal_path=run/'admission/signal-gate.json';signal=None
    if signal_path.exists():
        _,screen=load_sealed_cohort(run/'admission/screen')
        signal=signal_gate(screen,schedules['screen-hard']+schedules['screen-easy'])
        require(signal==load(signal_path),'Fixed64 signal gate differs');result['signal_gate']=signal
    profile_path=run/'admission/probe.json';profile=None
    if profile_path.exists():
        require(signal and signal['passed'],'Optimizer probe occurred without fixed signal admission')
        profile=load(profile_path);_,screen=load_sealed_cohort(run/'admission/screen')
        selected=[e for e in screen if e['episode_id'] in signal['timing_probe_episode_ids']]
        require([e['episode_id'] for e in selected]==signal['timing_probe_episode_ids'],'Prescribed probe pair order differs')
        result['temporary_probe']=check_probe(profile,selected,base,config,run/'checkpoints/base',run/'admission/probe-checkpoint')
    decision_path=run/'admission/timing-gate.json';decision=None
    if decision_path.exists():
        require(profile and profile['passed'],'Timing decision lacks actual full-step probe')
        decision=load(decision_path)
        # Budget is reconstructed from immutable tariff/prepare-clock inputs.
        admission_receipt=load(run/'admission/timing-budget.json')
        recomputed=timing_gate(profile,[load(p) for p in sorted((run/'admission/screen/batches').glob('*.json')) if not p.name.endswith('-intent.json')],
            load(run/'inputs/context-bound.json'),admission_receipt)
        require(recomputed==decision,'Full-work timing gate differs')
        overhead=load(run/'admission/main-overhead.json')
        require(overhead==main_overhead(profile,overhead['initial_actor_checkpoint_load_seconds'])
            and admission_receipt['fixed_main_overhead_seconds']==overhead['fixed_main_overhead_seconds']
            and admission_receipt['overhead_review_sha256']==file_sha(run/'inputs/implementation-review.json'),
            'Measured full-state/load overhead or review binding differs')
        plan=manifest['deployment'];checked=datetime.fromisoformat(admission_receipt['checked_at']).timestamp()
        clocks=lease_clocks(plan,checked=checked)
        for key in ('gpu_hourly_rate','guard_hourly_rate','storage_hourly_rate','other_reserved_cost_usd'):
            require(admission_receipt[key]==plan[key],'Timing tariff differs')
        close(admission_receipt['admission_spent_usd'],clocks['admission_spent'],'Admission elapsed-cost meter differs')
        require(admission_receipt['aggregate_started_at']==plan['created_at']
            and datetime.fromisoformat(admission_receipt['hard_stop']).timestamp()==plan['deadline_unix'],'Timing clock was reset')
        result['timing_gate']=decision
    production=any(k.startswith(('arms/','evaluation/')) for k in list(cohorts)+list(unsealed))
    if production:
        require(signal and signal['passed'] and decision and decision['passed'] and (run/'comparison-admission.json').exists(),'Production without both admission gates')
    for label in schedule.ENDPOINTS[1:]:
        arm=run/'arms'/label;previous=base;source_folder=run/'checkpoints/base';count=0
        for role in ('actor','learner'):
            path=arm/(role+'-base-restoration.json')
            if path.exists():
                restoration=load(path);require(restoration['passed'] is True and restoration['optimizer_steps']==0
                    and restoration['after']==restoration['shared_base_state'],'Arm baseline reset differs')
                check_state(restoration['after'],base,config)
        for index in range(1,21):
            folder=arm/f'cohorts/{index}'
            if not (folder/'pipeline-commit.json').exists():continue
            require(index==count+1,'Committed arm checkpoint gap')
            require(all((arm/(r+'-base-restoration.json')).exists() for r in ('actor','learner')),'Arm did not restore shared zero-step base')
            seal,episodes=load_sealed_cohort(folder);update=load(folder/'update.json');report=update['report']
            require(update==load(folder/'pipeline-commit.json') and report==load(folder/'optimizer-result.json'),'Update publication differs')
            behavior=base if index<=2 else checkpoint(f'arms/{label}/checkpoints/cycle{index-2}')
            require(report['source']==previous['policy']['binding'] and report['behavior']==behavior['policy']['binding']
                and report['policy_lag']==(0 if index==1 else 1) and seal['binding']==report['behavior'],'Lag/source lineage differs')
            gate=load(folder/'gate.json');check_gate(gate,[r for e in episodes for r in e['transitions']],seal['binding'],.002)
            check_state(gate['learner_state_before'],previous,config)
            require(report['gate_sha256']==digest(gate) and report['cohort_sha256']==digest(episodes),'Objective provenance differs')
            objective=check_objective(report,episodes,config);target=arm/f'checkpoints/cycle{index}';current=checkpoint(str(target.relative_to(run)))
            require(update['checkpoint']['files']==current['files'] and update['checkpoint']['binding']==current['policy']['binding']
                and update['checkpoint']['optimizer_step']==current['step'] and current['policy']['binding']['version']==index,'Actual target checkpoint differs')
            transition=check_transition(report,previous,current,source_folder,target,config)
            groups.append({'arm':label,'index':index,'objective':objective,'behavior':behavior_summary(episodes),
                'zero_advantage_cohort':episodes[0]['returns'][0]==episodes[1]['returns'][0],**transition});actual+=transition['optimizer_steps'];count+=1
            previous=current;source_folder=target
        if (arm/'training-complete.json').exists():
            completion=load(arm/'training-complete.json')
            require(count==20 and completion['attempted_cohorts']==20 and completion['all_updates_committed'] is True
                and completion['actual_optimizer_steps']==previous['step'] and completion['endpoint']['files']==previous['files'],
                'Arm completion is not its fixed cycle20 endpoint')
    result.update(groups=groups,committed_training_cohorts=len(groups),actual_retained_optimizer_steps=actual,
        optimizer_count_scope='Only actual saved checkpoint transitions with complete pipeline commits; unknown intents are separate.')
    if status['state']=='admission_signal_failed':
        require(signal and signal['passed'] is False and not profile_path.exists() and not production and not groups
            and not (run/'admission/probe-optimizer-intent.json').exists()
            and not (run/'comparison-admission.json').exists(),'Failed signal performed undeclared optimizer/comparison work')
    if status['state']=='admission_timing_failed':
        require(decision and decision['passed'] is False and not production and not groups
            and not (run/'comparison-admission.json').exists(),'Failed timing performed retained training/evaluation')
    if status['state']=='complete':
        require(not pending and not unsealed and not result['rpc_journal']['unresolved_or_failed_rpcs']
            and signal and signal['passed'] and decision and decision['passed']
            and all((run/'arms'/label/'training-complete.json').is_file() for label in schedule.ENDPOINTS[1:])
            and len(groups)==80 and set(all_eval)==set(schedule.ENDPOINTS),'Complete comparison is missing required work')
        completion=load(run/'comparison-complete.json')
        require(completion['attempted_updates']==80 and completion['evaluation_episodes']==320
            and completion['all_draws_preserved'] is True,'Comparison terminal denominator differs')
        for label in schedule.ENDPOINTS:
            expected=base if label=='base' else checkpoint(f'arms/{label}/checkpoints/cycle20')
            require(completion['endpoints'][label]['files']==expected['files'] and completion['endpoints'][label]['binding']==expected['policy']['binding'],
                'Final fixed endpoint was substituted')
        result.update(complete=True,comparisons_available=True,evaluation_episodes=320,comparisons=comparisons(all_eval),
            evaluation_behavior={label:behavior_summary(rows) for label,rows in all_eval.items()})
    result['integrity_passed']=True
    require(result['status_sha256']==file_sha(run/'status.json') and result['manifest_sha256']==file_sha(run/'manifest.json'),'Run changed while auditing')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Audit output is exclusive; preserve prior evidence')
    try:report=audit(args.run)
    except Exception as error:
        report={'kind':'planning_curriculum_independent_audit_v1','integrity_passed':False,'complete':False,'error_type':type(error).__name__,'reason':str(error)}
    exclusive_json(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k in ('integrity_passed','complete','state','actual_retained_optimizer_steps','reason')}))
    if not report['integrity_passed']:sys.exit(1)


if __name__=='__main__':main()
