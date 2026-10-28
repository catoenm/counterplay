"""Complete-only scalar-execution five-environment scientific reporting.

No game/model execution or partial held-out scoring. The failed padded-batch
condition and 27B authoring runs are not pooled with this distinct protocol.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile

ENVIRONMENTS=('market','maze','chess','planning','coding')
WEIGHTS=dict(zip(ENVIRONMENTS,(.24,.24,.24,.24,.04)))
MODEL='Qwen/Qwen3-8B'
REVISION='b968826d9c46dd6066d109eabc6255188de91218'
AUDITOR='scripts/analyze_arena_mixed_scalar.py'
SCALAR_SOURCES={AUDITOR,*(f'selfplay/arena_mixed_scalar/{n}.py' for n in
    ('experiment','runner','processes','cuda_policy','envs'))}
PHYSICAL_EXECUTION_CONTRACT={'version':1,'physical_batch_size':1,'input_padding':'none_per_request',
    'logical_inference_batch_size':8,'logical_training_microbatch_size':2,'likelihood_tolerance':.002}
BOOTSTRAP_SEED=20261024
BOOTSTRAP_DRAWS=10000


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def finite(value):return type(value) in (int,float) and math.isfinite(value)
def close(a,b):return finite(a) and finite(b) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10)
def mean(values):return math.fsum(values)/len(values)


def complete_audit(audit):
    require(audit.get('kind')=='mixed_arena_scalar_independent_audit_v1'
        and audit.get('integrity_passed') is True and audit.get('complete') is True
        and audit.get('comparisons_available') is True and audit.get('state')=='complete'
        and audit.get('phase')=='evaluation' and audit.get('evaluation_episodes')==448
        and audit.get('optimizer_outcome_unknown') is False
        and audit.get('uncommitted_operations')==[],
        'Only a complete passing 448-episode mixed evaluation audit can be reported')


def bind_inputs(run,audit_path):
    """Check completion before reading any evaluation evidence; bind exact bytes."""
    run=Path(run).resolve();audit_path=Path(audit_path).resolve()
    raw=audit_path.read_bytes();audit=json.loads(raw);complete_audit(audit)
    bindings={'audit':{'path':str(audit_path),'sha256':hashlib.sha256(raw).hexdigest()},
        'reporter':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)}}
    def read(name,expected=None):
        p=(run/name).resolve();require(p.is_relative_to(run),'Evidence path escapes run')
        raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest()
        if expected is not None:require(h==expected,'Audited evidence changed: '+name)
        bindings[name]={'path':str(p),'sha256':h}
        return json.loads(raw)
    manifest=read('manifest.json',audit['manifest_sha256'])
    status=read('status.json',audit['status_sha256'])
    require(manifest.get('kind')=='mixed_arena_scalar_v1' and manifest.get('phase')=='evaluation'
        and status.get('state')=='complete' and status.get('claims_allowed') is True
        and status.get('manifest_sha256')==audit['manifest_sha256'],'Current completed protocol differs')
    cfg=manifest['config']['policy']
    require(cfg.get('model')==MODEL and cfg.get('revision')==REVISION
        and cfg.get('environment_weights')==WEIGHTS and cfg.get('thinking') is False
        and cfg.get('physical_execution_contract')==PHYSICAL_EXECUTION_CONTRACT
        and cfg.get('likelihood_tolerance')==.002,
        'Fresh pinned scalar 8B direct-action policy identity differs')
    require(SCALAR_SOURCES<=set(manifest['sources']),'Scalar scientific source closure missing')
    require(manifest['sources'].get(AUDITOR)==audit['source_auditor_sha256'],
        'Auditor is not the frozen scientific source')
    for namespace,base,entries in (('source',run/'source',manifest['sources']),
        ('input',run,manifest['inputs']),('artifact',run,status['artifacts'])):
        for relative,expected in entries.items():
            part=Path(relative);p=(base/part).resolve()
            require(not part.is_absolute() and '..' not in part.parts and p.is_relative_to(base.resolve()),'Unsafe bound source/input path')
            require(sha(p)==expected,'Audited '+namespace+' changed: '+relative)
            bindings[namespace+'/'+relative]={'path':str(p),'sha256':expected}
    require('evaluation-seal.json' in status['artifacts'],'Missing complete evaluation seal')
    seal=read('evaluation-seal.json',status['artifacts']['evaluation-seal.json'])
    require(seal.get('episodes')==448 and seal.get('paired_scenarios_per_environment')==16,
        'Full evaluation denominator differs')
    for endpoint,key in (('base','baseline_seal_sha256'),('final','final_seal_sha256')):
        name=f'evaluation/{endpoint}/seal.json'
        require(name in status['artifacts'] and seal.get(key)==status['artifacts'][name],
            'Both complete endpoint seals must be bound')
    schedule=read('inputs/schedule.json',manifest['inputs']['inputs/schedule.json'])
    policies={}
    for endpoint in ('base','final'):
        name=f'inputs/checkpoints/{endpoint}/policy.json'
        policies[endpoint]=read(name,manifest['inputs'][name])
        require(policies[endpoint]['config']==cfg,'Endpoint policy configuration differs')
    base,final=(policies[e]['binding'] for e in ('base','final'))
    require(base['version']==0 and base['checkpoint_id']=='base'
        and final['version']==20 and final['checkpoint_id']==f"cycle{final['version']}"
        and base['base_identity']==final['base_identity'],'Fixed fresh-base/final checkpoint pair differs')
    require(type(audit['actual_optimizer_steps']) is int and 0<=audit['actual_optimizer_steps']<=final['version'],
        'Actual Adam count differs from attempted cohorts')
    parent_manifest=read('inputs/parent/manifest.json',manifest['inputs']['inputs/parent/manifest.json'])
    parent_status=read('inputs/parent/status.json',manifest['inputs']['inputs/parent/status.json'])
    parent_audit=read('inputs/parent/final-audit.json',manifest['inputs']['inputs/parent/final-audit.json'])
    completion=read('inputs/parent/training-complete.json',manifest['inputs']['inputs/parent/training-complete.json'])
    require(parent_manifest.get('kind')=='mixed_arena_scalar_v1' and parent_manifest.get('phase')=='training'
        and parent_manifest['config']['policy']==cfg
        and parent_manifest['config']['endpoint_choices']==[20]
        and parent_manifest['config']['episodes_per_cohort']==10
        and parent_manifest['config']['episodes_per_environment']==2,
        'Fixed scalar training protocol differs')
    require(parent_audit.get('manifest_sha256')==manifest['inputs']['inputs/parent/manifest.json']
        and parent_audit.get('status_sha256')==manifest['inputs']['inputs/parent/status.json']
        and parent_status.get('manifest_sha256')==parent_audit['manifest_sha256']
        and parent_status.get('state')=='training_complete'
        and parent_audit.get('source_auditor_sha256')==parent_manifest['sources'][AUDITOR]
        and parent_manifest['sources'][AUDITOR]==manifest['sources'][AUDITOR],
        'Complete parent training audit source/status binding differs')
    training=summarize_training(parent_audit)
    require(completion['attempted_cohorts']==20 and completion['actual_optimizer_steps']==training['actual_optimizer_steps']
        and completion['all_updates_committed'] is True
        and completion['endpoint']['binding']==final
        and training['actual_optimizer_steps']==audit['actual_optimizer_steps'],
        'Endpoint/Adam count differs from complete training evidence')
    policies['_training_summary']=training
    return audit,manifest,status,schedule,policies,bindings


def verify_bindings(bindings):
    for key,value in bindings.items():
        require(sha(value['path'])==value['sha256'],'Bound '+key+' changed during reporting')


def summarize_training(audit):
    require(audit.get('kind')=='mixed_arena_scalar_independent_audit_v1'
        and audit.get('integrity_passed') is True and audit.get('complete') is True
        and audit.get('training_complete') is True and audit.get('state')=='training_complete'
        and audit.get('phase')=='training' and audit.get('comparisons_available') is False
        and audit.get('optimizer_outcome_unknown') is False and audit.get('uncommitted_operations')==[],
        'Complete independently audited scalar training required')
    groups=audit['groups']
    require(len(groups)==audit['attempted_cohorts']==20
        and [g['cohort'] for g in groups]==list(range(1,21)),
        'All twenty attempted cohorts must be committed')
    for g in groups:
        require(g['episodes']==10 and type(g['optimizer_steps']) is int and g['optimizer_steps'] in (0,1)
            and type(g['reward_learning_signal']) is bool,'Malformed cohort count or reward signal')
    steps=sum(g['optimizer_steps'] for g in groups)
    require(type(audit['actual_optimizer_steps']) is int and audit['actual_optimizer_steps']==steps,
        'Actual optimizer increments differ from cohort count')
    return {'attempted_cohorts':20,'actual_optimizer_steps':steps,'zero_gradient_skipped_cohorts':20-steps,
        'cohorts_with_nonzero_reward_advantages':sum(g['reward_learning_signal'] for g in groups),
        'updates_with_nonzero_reward_advantages':sum(g['optimizer_steps'] for g in groups if g['reward_learning_signal']),
        'updates_without_nonzero_reward_advantages':sum(g['optimizer_steps'] for g in groups if not g['reward_learning_signal']),
        'episodes_per_cohort':10,'episodes_per_environment_per_cohort':2,
        'temporary_calibration_step_excluded':True,
        'interpretation':'A retained Adam step may be driven only by entropy/KL terms when reward advantages are zero. Counts do not establish learning or environment gain.'}


def percentile(values,p):
    values=sorted(values);position=p*(len(values)-1);lo=int(position);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(position-lo)


def paired_bootstrap(pairs,seed,draws=BOOTSTRAP_DRAWS):
    require(len(pairs)==16 and type(draws) is int and draws>=100,'Sixteen pairs and fixed bootstrap count required')
    rng=random.Random(seed);samples={'base':[],'final':[],'delta':[]}
    for _ in range(draws):
        sampled=[pairs[rng.randrange(16)] for _ in range(16)]
        a=mean([row[0] for row in sampled]);b=mean([row[1] for row in sampled])
        samples['base'].append(a);samples['final'].append(b);samples['delta'].append(b-a)
    intervals={k:[percentile(v,.025),percentile(v,.975)] for k,v in samples.items()}
    return intervals,samples


def validate_metrics(audit,schedule):
    """Reconcile audit aggregates with all prescribed seat/opponent rows."""
    complete_audit(audit)
    require(set(audit['comparisons'])==set(ENVIRONMENTS)
        and set(audit['metrics'])=={'base','final'},'Missing environment or endpoint')
    grouped={};seen=set()
    for endpoint in ('base','final'):
        specs=schedule['eval-'+endpoint]
        require(len(specs)==224 and len({s['episode_id'] for s in specs})==224,'Fixed 224 episode schedule differs')
        lookup={s['episode_id']:s for s in specs}
        require(set(audit['metrics'][endpoint])==set(ENVIRONMENTS),'Missing metric environment')
        for env in ENVIRONMENTS:
            rows=audit['metrics'][endpoint][env];expected=64 if env in ENVIRONMENTS[:3] else 16
            require(len(rows)==expected,'Per-environment episode denominator differs')
            families=defaultdict(list)
            for row in rows:
                identifier=row['episode_id'];require(identifier in lookup and identifier not in seen,'Duplicate or unprescribed evaluation episode')
                seen.add(identifier);spec=lookup[identifier]
                require(spec['environment']==env and row['family']==spec['scenario_family_id']
                    and spec['learner_seats']==[row['learner_seat']] and row['opponent']==spec['opponent'],
                    'Metric row family, seat or opponent differs from schedule')
                require(finite(row['return']) and (-1 if env in ENVIRONMENTS[:3] else 0)<=row['return']<=1,'Finite bounded return required')
                families[row['family']].append(row)
            require(len(families)==16,'Sixteen independent scenario families required')
            combinations={(seat,opponent) for seat in (0,1) for opponent in ('base','heuristic')} if expected==64 else {(0,'none')}
            require(all(len(rows)==len(combinations) and {(r['learner_seat'],r['opponent']) for r in rows}==combinations
                for rows in families.values()),'Paired seat/opponent cluster coverage differs')
            grouped[(endpoint,env)]=families
    require(len(seen)==448,'All 448 episode identities required')
    for env in ENVIRONMENTS:
        comparison=audit['comparisons'][env];clusters=comparison['clusters']
        require(comparison['paired_clusters']==16 and len(clusters)==16
            and len({r['scenario_family_id'] for r in clusters})==16,'Comparison cluster denominator differs')
        keys=set(grouped[('base',env)])
        require(keys==set(grouped[('final',env)])=={r['scenario_family_id'] for r in clusters},'Pairing across endpoints differs')
        for row in clusters:
            family=row['scenario_family_id'];a=mean([r['return'] for r in grouped[('base',env)][family]])
            b=mean([r['return'] for r in grouped[('final',env)][family]])
            require(close(row['base'],a) and close(row['final'],b) and close(row['delta'],b-a),'Audited paired scenario means differ')
        require(close(comparison['base_mean'],mean([r['base'] for r in clusters]))
            and close(comparison['final_mean'],mean([r['final'] for r in clusters]))
            and close(comparison['delta'],mean([r['delta'] for r in clusters])),'Aggregate return mean differs')
    require(close(audit['weighted_return_delta'],math.fsum(WEIGHTS[e]*audit['comparisons'][e]['delta'] for e in ENVIRONMENTS)),
        'Weighted task mean differs')
    return grouped


def numeric_metric(row,key,seat=False,boolean=False):
    values=row['metrics']
    if key not in values:return None
    value=values[key]
    if seat:
        require(isinstance(value,list) and len(value)==2,'Malformed role-specific native metric')
        value=value[row['learner_seat']]
    require(type(value) is bool if boolean else finite(value),'Malformed native metric: '+key)
    return float(value)


def metric_value(row,key):
    if key=='goal_reached':
        distance=numeric_metric(row,'shortest_distances',seat=True)
        return None if distance is None else float(distance==0)
    if key=='checkmate_win':
        checkmate=numeric_metric(row,'checkmate',boolean=True)
        return None if checkmate is None else float(bool(checkmate) and row['return']>0)
    if key=='cap_rate':
        reason=row['metrics'].get('finish_reason')
        return None if reason is None else float(reason in {'material_adjudication_at_cap','distance_adjudication_at_cap'})
    roles={'absolute_pnl','filled_units','fees','moves','blocks_placed'}
    booleans={'verified_goal','hidden_all_passed'}
    return numeric_metric(row,key,key in roles,key in booleans)


NATIVE={'market':[('absolute_pnl','Absolute P&L'),('filled_units','Filled units'),('fees','Fees')],
    'maze':[('goal_reached','Goal reached fraction'),('cap_rate','Distance cap fraction'),('moves','Moves')],
    'chess':[('checkmate_win','Learner checkmate win fraction'),('cap_rate','Material cap fraction')],
    'planning':[('verified_goal','Verified goal fraction'),('turns','Actions')],
    'coding':[('hidden_all_passed','All hidden tests passed fraction'),('hidden_pass_fraction','Hidden test pass fraction'),
              ('hidden_runtime_faults','Hidden test runtime faults')]}


def summarize(audit,schedule,policies,*,draws=BOOTSTRAP_DRAWS):
    grouped=validate_metrics(audit,schedule);panels={};bootstrap={};native={}
    for index,env in enumerate(ENVIRONMENTS):
        clusters=sorted(audit['comparisons'][env]['clusters'],key=lambda r:r['scenario_family_id'])
        pairs=[(r['base'],r['final']) for r in clusters]
        intervals,samples=paired_bootstrap(pairs,BOOTSTRAP_SEED+997*index,draws)
        bootstrap[env]=samples['delta']
        panels[env]={**audit['comparisons'][env],'clusters':clusters,'descriptive_95_intervals':intervals,
            'episodes_per_checkpoint':64 if env in ENVIRONMENTS[:3] else 16,'objective_coefficient':WEIGHTS[env]}
        native[env]={}
        for number,(key,label) in enumerate(NATIVE[env]):
            extracted={endpoint:{family:[metric_value(r,key) for r in rows] for family,rows in grouped[(endpoint,env)].items()}
                for endpoint in ('base','final')}
            counts={e:sum(v is not None for values in families.values() for v in values) for e,families in extracted.items()}
            expected=panels[env]['episodes_per_checkpoint']
            result={'label':label,'available':all(v==expected for v in counts.values()),'observed_values':counts,
                'required_values_per_checkpoint':expected}
            if result['available']:
                pairs=[(mean(extracted['base'][family]),mean(extracted['final'][family])) for family in sorted(extracted['base'])]
                ci,_=paired_bootstrap(pairs,BOOTSTRAP_SEED+10000+997*index+31*number,draws)
                a,b=mean([p[0] for p in pairs]),mean([p[1] for p in pairs])
                result.update(base_mean=a,final_mean=b,delta=b-a,paired_clusters=16,descriptive_95_delta_interval=ci['delta'])
            native[env][key]=result
    macro_samples=[math.fsum(WEIGHTS[e]*bootstrap[e][i] for e in ENVIRONMENTS) for i in range(draws)]
    steps=audit['actual_optimizer_steps'];version=policies['final']['binding']['version']
    training=policies['_training_summary']
    require(version==20 and training['attempted_cohorts']==20 and training['actual_optimizer_steps']==steps,
        'Fixed endpoint and actual update scope differ')
    return {'kind':'complete_mixed_arena_scalar_reporting_data_v1','model':MODEL,'revision':REVISION,
        'baseline':policies['base']['binding'],'final':policies['final']['binding'],
        'attempted_cohorts':version,'actual_optimizer_steps':steps,'evaluation_episodes':448,
        'training':training,'physical_execution_contract':PHYSICAL_EXECUTION_CONTRACT,
        'condition':'Scalar physical execution with separately declared rational market reward',
        'panels':panels,'native_metrics':native,
        'weighted_macro_delta':{'weights':WEIGHTS,'delta':audit['weighted_return_delta'],
            'descriptive_95_interval':[percentile(macro_samples,.025),percentile(macro_samples,.975)],
            'interpretation':'Weighted mean of five task return changes; units are bounded task returns, not Elo or measured gradient shares.'},
        'bootstrap':{'seed':BOOTSTRAP_SEED,'draws':draws,'paired_clusters_per_environment':16,
            'method':'Percentile bootstrap of 16 paired scenario families per task; seat/opponent games stay together. Independent resampling across environments for weighted mean.'},
        'interpretation':('Fixed fresh 8B baseline versus final shared-policy weights; one training seed, descriptive uncertainty only.' if steps else
            'Zero actual updates: repeat evaluation of unchanged weights; differences cannot establish learned improvement.'),
        'caveats':['Chess is capped tactical endgame return, not full-game Elo.',
            'Market reward is d/(128+abs(d)) for the learner\'s integer profit difference; absolute P&L is reported separately.',
            'Planning and coding are bounded single-player tasks; coding is AST repair, not freeform authoring.',
            'Legal-action masking guarantees executable menu choices; validity is not a learned gain.',
            'No pooling with the failed padded-batch condition, 27B authoring, teacher, or historical runs; five task comparisons are descriptive, not multiplicity-adjusted.']}


def make_figure(data,plt,fixture=False):
    colors=('#667586','#176B87');fig,axes=plt.subplots(2,3,figsize=(14,9.5))
    fig.subplots_adjust(left=.065,right=.97,top=.79,bottom=.29,wspace=.30,hspace=1.0)
    for ax,env in zip(axes.flat,ENVIRONMENTS):
        panel=data['panels'][env]
        for row in panel['clusters']:
            ax.plot([0,1],[row['base'],row['final']],color='#CCD4DB',alpha=.75,linewidth=.7,zorder=1)
        for i,endpoint in enumerate(('base','final')):
            value=panel[endpoint+'_mean'];lo,hi=panel['descriptive_95_intervals'][endpoint]
            ax.plot([i,i],[lo,hi],color=colors[i],linewidth=3,zorder=3)
            ax.scatter([i],[value],color=colors[i],s=55,zorder=4)
        low,high=panel['descriptive_95_intervals']['delta']
        title={'market':'Market (rational return)','coding':'Coding repair (4% objective)'}.get(env,env.capitalize())
        ax.set(title=title,xticks=[0,1],xticklabels=['Fresh base','Final checkpoint'],
            xlim=(-.25,1.25),ylim=(-1.08,1.08) if env in ENVIRONMENTS[:3] else (-.08,1.08),ylabel='Learner task return')
        ax.set_title(title,pad=30)
        ax.axhline(0,color='#D4DBE1',linewidth=.6);ax.grid(axis='y',color='#E8EDF1',linewidth=.6)
        ax.text(.5,1.03,f"Δ {panel['delta']:+.3f}  [{low:+.3f}, {high:+.3f}]",transform=ax.transAxes,ha='center',fontsize=9,color='#244D60')
        key=NATIVE[env][0][0];metric=data['native_metrics'][env][key]
        text=(f"{metric['label']}\n{metric['base_mean']:.3f} → {metric['final_mean']:.3f}" if metric['available'] else
            f"{metric['label']}: unavailable")
        ax.text(.5,-.24,text,transform=ax.transAxes,ha='center',va='top',fontsize=8.5,color='#475569')
        for spine in ('top','right'):ax.spines[spine].set_visible(False)
    ax=axes.flat[5];macro=data['weighted_macro_delta'];lo,hi=macro['descriptive_95_interval']
    ax.axvline(0,color='#BAC6CF',linewidth=1);ax.plot([lo,hi],[0,0],color=colors[1],linewidth=4)
    ax.scatter([macro['delta']],[0],marker='D',color=colors[1],s=65)
    extent=max(.02,abs(lo),abs(hi),abs(macro['delta']))*1.3
    ax.set(xlim=(-extent,extent),ylim=(-.5,.5),yticks=[],xlabel='Weighted final − base task return',title='Weighted task mean change')
    ax.text(.5,.84,f"{macro['delta']:+.3f}  [{lo:+.3f}, {hi:+.3f}]",transform=ax.transAxes,ha='center',weight='bold',fontsize=11)
    ax.text(.5,.13,'Weights: .24 each for market, maze, chess, planning; .04 coding.\nThese are objective coefficients, not measured gradient shares.',
        transform=ax.transAxes,ha='center',fontsize=8,linespacing=1.5)
    for spine in ('top','right','left'):ax.spines[spine].set_visible(False)
    if fixture:fig.text(.055,.985,'SYNTHETIC TEST FIXTURE',fontsize=10,color='#A33A21',weight='bold')
    fig.text(.055,.955,'Shared 8B policy: scalar execution across five tasks',weight='bold',fontsize=19,color='#182B3A')
    fig.text(.055,.918,f"Qwen3-8B · fresh base → {data['final']['checkpoint_id']} · {data['attempted_cohorts']} attempted cohorts · {data['actual_optimizer_steps']} actual Adam updates",fontsize=11,color='#475569')
    training=data['training']
    fig.text(.055,.889,f"{training['zero_gradient_skipped_cohorts']} zero-gradient skips · {training['cohorts_with_nonzero_reward_advantages']}/20 cohorts with nonzero reward advantages · temporary calibration step excluded",fontsize=9,color='#475569')
    fig.text(.055,.856,'448 held-out episodes · 16 paired families per task · lines pair families; intervals are descriptive 95% bootstraps',fontsize=10,color='#475569')
    fig.text(.055,.145,'Competitive tasks: 64 games/checkpoint averaged within 16 scenario clusters. Planning/coding: 16 episodes/checkpoint. One training seed.',fontsize=9,color='#475569')
    fig.text(.055,.108,'Chess is capped tactical return, not Elo. Market uses rational relative P&L. Competitive scores always use the learner\'s seat; coding is bounded AST repair.',fontsize=9,color='#475569')
    fig.text(.055,.071,'Legal-action formatting is enforced by the harness. No validity gain, general transfer, or perpetual improvement follows from these figures.',fontsize=9,color='#475569')
    fig.text(.055,.034,data['interpretation'],fontsize=9,color='#475569')
    return fig


def render(run,audit_path,prefix):
    audit,manifest,status,schedule,policies,bindings=bind_inputs(run,audit_path)
    require(not audit.get('test_fixture'),'Synthetic fixtures cannot be published as research results')
    data=summarize(audit,schedule,policies)
    prefix=Path(prefix);targets={s:Path(str(prefix)+s) for s in ('.png','.svg','.json')}
    if any(p.exists() for p in targets.values()):raise FileExistsError('Choose a fresh output prefix; reports are immutable')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    prefix.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.mixed-scalar-report-',dir=prefix.parent) as tmp:
        fig=make_figure(data,plt);staged={}
        try:
            for suffix in ('.png','.svg'):
                p=Path(tmp)/('figure'+suffix);fig.savefig(p,dpi=200,facecolor='white');staged[suffix]=p
        finally:plt.close(fig)
        record={'data':data,'bound_inputs':bindings,'matplotlib_version':matplotlib.__version__,
            'outputs':{str(targets[s].resolve()):sha(staged[s]) for s in ('.png','.svg')}}
        staged['.json']=Path(tmp)/'figure.json';staged['.json'].write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        verify_bindings(bindings)
        if any(p.exists() for p in targets.values()):raise FileExistsError('Output appeared while rendering; preserve it')
        for suffix,target in targets.items():os.link(staged[suffix],target)
    return {s:str(p.resolve()) for s,p in targets.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--out-prefix',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(render(args.run,args.audit,args.out_prefix),indent=2))


if __name__=='__main__':main()
