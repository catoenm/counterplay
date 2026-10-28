"""Complete-only static reporting for audited mixed-native two-call RL.

Reads audited reporting data and hashes committed artifacts; never loads a
model, invokes an API, scores a game, or reports partial heldout results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import plot_cardinal_gpu as previous

require,close=previous.require,previous.close
ENDPOINTS=(10,15,25)
N=16
AUDITOR='scripts/analyze_program_debug_revision_rl.py'
KINDS=previous.KINDS


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def complete_audit(audit):
    require(audit.get('kind')=='independent_cpu_debug_revision_rl_audit'
        and audit.get('integrity_passed') is True and audit.get('planned_study_complete') is True
        and audit.get('planned_comparisons_available') is True and audit.get('recorded_state')=='complete',
        'Figures require a complete passing independent mixed-native RL audit')


def bind_inputs(audit_path):
    """Bind the exact parsed audit bytes, then all frozen/committed evidence."""
    path=Path(audit_path).resolve();raw=path.read_bytes();audit=json.loads(raw)
    complete_audit(audit)  # No evaluation files are read for a partial audit.
    run=Path(audit['run']).resolve()
    bindings={'audit':{'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()},
        'reporter':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},
        'reporting_helper':{'path':str(Path(previous.__file__).resolve()),'sha256':sha(previous.__file__)}}
    parsed={}
    for name in ('manifest','status','schedule'):
        target=run/(name+'.json');data=target.read_bytes();digest=hashlib.sha256(data).hexdigest()
        expected=audit[name+'_sha256'] if name!='schedule' else parsed['manifest']['input_sha256']['schedule.json']
        require(digest==expected,'Audited '+name+' binding differs')
        bindings[name]={'path':str(target),'sha256':digest};parsed[name]=json.loads(data)
    manifest,status=parsed['manifest'],parsed['status'];cfg=manifest['config']
    require(manifest['kind']=='native_nonthinking_revision_rl'
        and status['manifest_sha256']==bindings['manifest']['sha256'],'Protocol or manifest/status identity differs')
    require(cfg['authoring_calls']==2 and cfg['eval_samples']==N and cfg['group_size']==8
        and cfg['batch_size']==2 and cfg['endpoint_choices']==list(ENDPOINTS)
        and cfg['turn_native_modes']['draft']['thinking_enabled'] is True
        and cfg['turn_native_modes']['draft']['reasoning_effort']=='low'
        and cfg['turn_native_modes']['revision']['thinking_enabled'] is False
        and cfg['turn_native_modes']['revision']['reasoning_effort'] is None,'Mixed native protocol differs')
    require(status.get('state')=='complete' and status.get('claims_allowed') is True,
        'Current status must allow a complete comparison')
    # No new analysis of generated text or games: only byte-level integrity.
    for namespace,folder,entries in (('source',run/'source',manifest['source_sha256']),
        ('input',run,manifest['input_sha256']),('artifact',run,status['artifacts'])):
        for relative,expected in entries.items():
            part=Path(relative);target=(folder/part).resolve()
            require(not part.is_absolute() and '..' not in part.parts
                and target.is_relative_to(folder.resolve()),'Unsafe bound evidence path')
            require(sha(target)==expected,'Audited '+namespace+' changed: '+relative)
            bindings[namespace+'/'+relative]={'path':str(target),'sha256':expected}
    require(audit['auditor_sha256']==manifest['source_sha256'][AUDITOR],
        'Independent auditor differs from frozen source')
    sealpath=run/'evaluation/sealed.json'
    require('evaluation/sealed.json' in status['artifacts'],'Complete evaluation seal is missing')
    raw=sealpath.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==status['artifacts']['evaluation/sealed.json'],'Evaluation seal changed')
    return audit,manifest,status,parsed['schedule'],json.loads(raw),bindings


def levels(values,tolerance=1e-12):
    """Report numerical variation at the registered zero-signal tolerance."""
    values=sorted(values);representatives=[]
    for value in values:
        if not representatives or value-representatives[-1]>tolerance:representatives.append(value)
    return len(representatives)


def training_data(audit):
    complete_audit(audit)
    groups=audit.get('training_groups',[]);endpoint=audit['selected_endpoint']
    require(type(endpoint) is int and endpoint in ENDPOINTS and len(groups)==endpoint
        and [g['cycle'] for g in groups]==list(range(1,endpoint+1))
        and all(g.get('checkpoint_committed') is True for g in groups),'Fixed committed training endpoint is incomplete')
    tensors=audit['tensor_reports'];require(tensors['base']['actual_optimizer_step']==0,'Base Adam is not zero step')
    rows=[];prior_step=0
    for group in groups:
        number=group['cycle'];cell=previous.checked_cell(group,[f'train-{number}-{i}' for i in range(8)],16)
        rewards=group['rewards'];advantages=group['advantages']
        require(isinstance(rewards,list) and len(rewards)==8 and all(type(r) in (int,float)
            and math.isfinite(r) and -1<=r<=1 for r in rewards),'Eight finite bounded training rewards required')
        require(close(math.fsum(rewards)/8,group['mean_penalized_task_reward']),'Training reward mean differs')
        raw=group['programs'];valid_returns=[]
        for reward,program in zip(rewards,raw):
            require(close(reward,program['reward']),'Audited reward/program order differs')
            if program['runtime_valid']:
                ret=program['game_return']
                require(type(ret) in (int,float) and math.isfinite(ret) and -1<=ret<=1
                    and close(reward,ret),'Fault-free gameplay reward differs')
                valid_returns.append(ret)
            else:require(reward==-1.,'Invalid authoring/runtime penalty differs')
        zero=max(rewards)-min(rewards)<=1e-12
        expected=[0.]*8 if zero else [r-(math.fsum(rewards)-r)/7 for r in rewards]
        require(isinstance(advantages,list) and len(advantages)==8
            and all(close(a,b) for a,b in zip(advantages,expected)),'RLOO advantages differ')
        step=tensors[f'cycle{number}']['actual_optimizer_step']
        require(type(step) is int and step-prior_step==int(not zero),'Actual Adam steps differ from zero-signal skips')
        signal='no_reward_variation' if zero else 'validity_contrast_only' if levels(valid_returns)<=1 else 'game_return_variation'
        rows.append({'cycle':number,'rewards':rewards,'advantages':advantages,'mean_reward':math.fsum(rewards)/8,
            'reward_std':math.sqrt(math.fsum((r-math.fsum(rewards)/8)**2 for r in rewards)/8),
            'reward_levels':levels(rewards),'fault_free_game_return_levels':levels(valid_returns),
            'fault_free_programs':len(valid_returns),'reward_diversity':signal,
            'mean_yield':cell['mean'],'compiled_fraction':(8-cell['counts']['invalid_authorship'])/8,
            'fault_free_fraction':len(valid_returns)/8,
            'positive_yield_fraction':sum(p['banked_point_yield']>0 for p in cell['programs'])/8,
            'actual_optimizer_step':step,'update_committed':step>prior_step,'counts':cell['counts']})
        prior_step=step
    require(prior_step==audit['genuine_RL_optimizer_steps']==audit['genuine_optimizer_steps'],'Actual step totals disagree')
    return {'groups':rows,'attempted_groups':endpoint,'actual_optimizer_steps':prior_step,
        'skipped_groups':sum(not r['update_committed'] for r in rows),
        'diversity_counts':{label:sum(r['reward_diversity']==label for r in rows)
            for label in ('no_reward_variation','validity_contrast_only','game_return_variation')},
        'reward_level_tolerance':1e-12,'complete':True,
        'interpretation':'Training diagnostics only: maps and historical opponents change. Reward variation is not evidence of heldout improvement or of which behavior a gradient learned.'}


def final_data(audit,status,schedule,seal):
    complete_audit(audit)
    require(status.get('state')=='complete' and status.get('claims_allowed') is True
        and status.get('learning_comparison_available') is True,'Complete current status is required')
    endpoint=audit.get('selected_endpoint')
    require(type(endpoint) is int and endpoint in ENDPOINTS,'Unexpected fixed endpoint')
    require(audit.get('new_supervised_optimizer_steps')==0,'Unexpected new supervised updates')
    total=8*endpoint+32
    require(audit.get('committed_trajectories')==total and audit.get('committed_raw_sampled_turns')==2*total
        and audit.get('raw_turns_pending_trajectory_materialization')==0,'All 32 final trajectories and 64 sampled turns are required')
    labels=('base',f'cycle{endpoint}')
    paths=[f'generations/eval-{label}-{i}.json' for label in labels for i in range(N)]
    require(isinstance(seal,dict) and seal.get('draws')==32 and seal.get('sampled_turns')==64
        and seal.get('scoring_allowed') is True and set(seal.get('generations',{}))==set(paths)
        and set(seal.get('checkpoints',{}))==set(labels),'All 32 final revisions must be sealed before scoring')
    require(all(seal['generations'][p]==status['artifacts'].get(p) for p in paths),'Sealed generation hashes differ')
    require(all(seal['checkpoints'][label]==status['artifacts'].get(f'checkpoints/{label}/metadata.json') for label in labels)
        and seal['endpoint_selection_sha256']==status['artifacts'].get('endpoint-selection.json'),'Fixed checkpoint/endpoint seal differs')
    require(set(audit['evaluation_cells'])==set(labels),'Both predetermined evaluation endpoints required')
    cells={label:previous.checked_cell(audit['evaluation_cells'][label],[f'eval-{label}-{i}' for i in range(N)],48) for label in labels}
    seeds=schedule['eval_generation_seeds'];contrast=audit['paired_yield']
    require(len(seeds)==len(set(seeds))==N and all(type(s) is int for s in seeds)
        and contrast['paired_generation_seeds']==seeds and contrast['program_cluster_n']==N,'Sixteen paired authoring seeds differ')
    differences=[b['banked_point_yield']-a['banked_point_yield'] for a,b in zip(cells['base']['programs'],cells[labels[1]]['programs'])]
    require(len(contrast['paired_yield_differences'])==N and all(close(a,b) for a,b in zip(differences,contrast['paired_yield_differences'])),
        'Paired program yield differences disagree')
    mean=math.fsum(differences)/N
    require(close(mean,contrast['mean_yield_difference']) and close(cells['base']['mean'],contrast['base_mean_yield'])
        and close(cells[labels[1]]['mean'],contrast['final_mean_yield']),'All-draw paired means disagree')
    interval=previous.bootstrap_interval(differences,contrast['bootstrap_draws'],contrast['bootstrap_seed'])
    require(len(contrast['descriptive_bootstrap_95_interval'])==2 and all(close(a,b) for a,b in zip(interval,contrast['descriptive_bootstrap_95_interval'])),
        'Descriptive paired-trajectory bootstrap differs')
    progress=training_data(audit)
    require(status['attempted_groups']==endpoint and status['actual_optimizer_steps']==progress['actual_optimizer_steps']
        and status['final_checkpoint']==labels[1],'Terminal attempted/update/endpoint counts differ')
    final={'endpoint':endpoint,'labels':labels,'cells':cells,'generation_seeds':seeds,'paired_differences':differences,
        'mean_difference':mean,'descriptive_bootstrap_95_interval':interval,'program_cluster_n':N,
        'actual_optimizer_steps':progress['actual_optimizer_steps'],'skipped_groups':progress['skipped_groups'],
        'authoring_calls_per_trajectory':2,'mandatory_final_turn':1,'new_supervised_optimizer_steps':0,
        'baseline':'Exact inherited 32-step SFT adapter, zero RL Adam; same native-low draft and native-nonthinking revision harness',
        'uncertainty':'Descriptive 95% paired-trajectory bootstrap;16 clusters, one training seed. Turns and 48 games per compiled controller are dependent.',
        'opponents':'Two scripted anchors and two fixed teacher controllers that were prior SFT examples; excluded from the RL roster, not unseen to weights.',
        'interpretation':('Incremental RL under one fixed harness/opponent pool; no causal claim about the thinking toggle, teacher equivalence, general transfer or continual improvement.'
            if progress['actual_optimizer_steps'] else 'Zero actual updates: repeated evaluation of unchanged weights; any endpoint differences are not learned improvement.')}
    return {'final':final,'training':progress}


def final_figure(data,plt,fixture=False):
    from matplotlib.lines import Line2D
    colors=('#6A7885','#1B7CA1');labels=data['labels'];n=data['program_cluster_n']
    fig=plt.figure(figsize=(13.5,10.2));grid=fig.add_gridspec(2,2,width_ratios=(1.25,1),height_ratios=(1,1),
        left=.13,right=.96,top=.79,bottom=.28,hspace=.60,wspace=.28)
    ax=fig.add_subplot(grid[:,0]);diff=fig.add_subplot(grid[0,1]);rates=fig.add_subplot(grid[1,1])
    maximum=max(1.,max(r['banked_point_yield'] for c in data['cells'].values() for r in c['programs']))
    for i in range(n):
        y=n-1-i;values=[data['cells'][label]['programs'][i]['banked_point_yield'] for label in labels]
        ax.plot(values,[y-.16,y+.16],color='#CED5DC',linewidth=.8,zorder=1)
        for j,label in enumerate(labels):
            row=data['cells'][label]['programs'][i];category=row['category']
            marker='X' if category=='invalid_authorship' else '^' if category=='runtime_fault' else 'o'
            ax.scatter(row['banked_point_yield'],y+(-.16 if j==0 else .16),marker=marker,s=34,
                facecolors='white' if category=='valid_zero' else colors[j],edgecolors=colors[j],linewidths=1,zorder=3)
    ax.set(yticks=list(range(n-1,-1,-1)),yticklabels=[str(s) for s in data['generation_seeds']],
        xlabel='Mean banked points per held-out game',ylabel='Paired two-call authoring seed',
        xlim=(-.05*maximum,1.08*maximum),ylim=(-.6,n-.4),title='All 16 mandatory final revisions per checkpoint')
    ax.tick_params(axis='y',labelsize=8);ax.grid(axis='x',color='#E8EDF1',linewidth=.7)
    lo,hi=data['descriptive_bootstrap_95_interval'];mean=data['mean_difference']
    # Plot bootstrap limits directly: skewed intervals need not contain the sample mean.
    jitter=[((i*7)%n-(n-1)/2)*.012 for i in range(n)]
    diff.scatter(data['paired_differences'],[.3+y for y in jitter],s=20,color='#9AB6C3',label='16 paired trajectories')
    diff.plot([lo,hi],[0,0],color=colors[1],linewidth=3);diff.scatter([mean],[0],marker='D',s=55,color=colors[1])
    diff.axvline(0,color='#AFB7BE',linewidth=1);diff.set(yticks=[],ylim=(-.2,.5),xlabel='After RL − before RL banked points',title='Mean paired change + descriptive 95% interval')
    diff.text(.02,.91,f'{mean:+.3f}  [{lo:+.3f}, {hi:+.3f}]',transform=diff.transAxes,color='#244657',weight='bold',fontsize=11)
    category_colors=('#B56B63','#D69A42','#C7D0D7','#488E7C')
    category_names=('Invalid authoring','Runtime fault','Fault-free zero','Fault-free positive')
    left=[0,0]
    for kind,color,name in zip(KINDS,category_colors,category_names):
        counts=[data['cells'][label]['counts'][kind] for label in labels]
        bars=rates.barh([1,0],counts,left=left,height=.5,color=color,label=name)
        for bar,count in zip(bars,counts):
            if count:rates.text(bar.get_x()+count/2,bar.get_y()+.25,str(count),ha='center',va='center',fontsize=9,color='#24313A')
        left=[x+c for x,c in zip(left,counts)]
    rates.set(xlim=(0,n),ylim=(-.35,1.95),xticks=[0,4,8,12,16],yticks=[1,0],yticklabels=['Before RL','After RL'],
        xlabel='Programs out of 16 prescribed draws',title='Validity and useful play · same denominator')
    rates.legend(loc='upper center',bbox_to_anchor=(.5,1.0),ncol=2,frameon=False,fontsize=8.5)
    for axis in (ax,diff,rates):previous.style_axis(axis)
    prefix='TEST FIXTURE · ' if fixture else ''
    fig.text(.075,.96,prefix+'Thinking, practice, then code: held-out RL results',fontsize=20,weight='bold',color='#17212B')
    fig.text(.075,.925,f"Same two-call harness · {data['endpoint']} attempted groups · {data['actual_optimizer_steps']} actual Adam updates · {data['skipped_groups']} skips",fontsize=11,color='#4A535D')
    fig.legend(handles=[Line2D([0],[0],color=c,marker='o',linestyle='none',label=label)
        for c,label in zip(colors,('Before RL · inherited 32-step SFT weights',
            'After RL · fixed endpoint' if data['actual_optimizer_steps'] else 'After attempts · unchanged weights'))],loc='upper left',bbox_to_anchor=(.067,.903),ncol=2,frameon=False,fontsize=10)
    fig.legend(handles=[Line2D([0],[0],color='#526372',marker=m,linestyle='none',markerfacecolor=face,label=label)
        for m,face,label in [('X','#526372','Invalid authoring; no game'),('^','#526372','Compiled with runtime faults'),('o','white','Fault-free zero yield'),('o','#526372','Fault-free positive yield')]],loc='upper left',bbox_to_anchor=(.067,.868),ncol=4,frameon=False,fontsize=8.7)
    before,after=(data['cells'][label] for label in labels)
    fig.text(.075,.205,f"Mean of all 16 draws: before {before['mean']:.3f} → after {after['mean']:.3f} banked points/game.",fontsize=11,weight='bold',color='#244657')
    fig.text(.075,.148,'Each compiled final plays 6 held-out maps × 4 fixed opponents × 2 seats = 48 correlated games. Invalid authoring contributes zero yield and no games.',fontsize=9,color='#4A535D')
    fig.text(.075,.113,'Uncertainty resamples 16 paired authoring trajectories, including invalid draws. One training seed; turns and game replays are not independent samples.',fontsize=9,color='#4A535D')
    fig.text(.075,.078,'Draft uses native low thinking; revision uses native thinking-off. Both endpoints inherit the same 32 SFT steps; there is no additional SFT here.',fontsize=9,color='#4A535D')
    caveat=('Two evaluation teachers were prior SFT examples, reserved from the RL roster. This fixed-pool result does not establish general or perpetual improvement.'
        if data['actual_optimizer_steps'] else 'No policy update occurred. This is repeat evaluation of unchanged weights; any endpoint differences are not learned improvement.')
    fig.text(.075,.043,caveat,fontsize=9,color='#4A535D')
    return fig


def training_figure(data,plt,fixture=False):
    fig,axes=plt.subplots(2,2,figsize=(12.8,9.3));fig.subplots_adjust(left=.085,right=.965,top=.82,bottom=.20,hspace=.43,wspace=.26)
    rows=data['groups'];x=[r['cycle'] for r in rows]
    for i in range(8):axes[0,0].scatter([r['cycle']+(i-3.5)*.035 for r in rows],[r['rewards'][i] for r in rows],s=12,color='#91ADBC',alpha=.7)
    axes[0,0].plot(x,[r['mean_reward'] for r in rows],color='#246E94',linewidth=1.4,label='Mean of 8')
    axes[0,0].set(title='Final-controller reward · all 8 trajectories',ylabel='Penalized reward',ylim=(-1.08,1.08));axes[0,0].legend(frameon=False,fontsize=9)
    for key,label,color in [('compiled_fraction','Compiled','#7C858E'),('fault_free_fraction','Fault-free','#1B7CA1'),('positive_yield_fraction','Positive banked yield','#488E7C')]:
        axes[0,1].plot(x,[r[key] for r in rows],label=label,color=color,linewidth=1.3)
    axes[0,1].set(title='Validity and useful play',ylabel='Fraction of all 8 trajectories',ylim=(-.04,1.06));axes[0,1].legend(frameon=False,fontsize=9,loc='lower left')
    axes[1,0].plot([0]+x,[0]+x,color='#B9C2CA',linestyle='--',linewidth=1.2,label='Attempted groups')
    axes[1,0].step([0]+x,[0]+[r['actual_optimizer_step'] for r in rows],where='post',color='#725A98',linewidth=1.8,label='Actual Adam steps')
    skipped=[r for r in rows if not r['update_committed']]
    if skipped:axes[1,0].scatter([r['cycle'] for r in skipped],[r['actual_optimizer_step'] for r in skipped],marker='x',color='#BD8438',s=30,label='No update; zero variation')
    axes[1,0].set(title='Attempts and genuine updates',ylabel='Cumulative count');axes[1,0].legend(frameon=False,fontsize=8.5,loc='upper left')
    axes[1,1].plot(x,[r['reward_levels'] for r in rows],color='#246E94',marker='o',markersize=3,linewidth=1.2,label='Reward levels · all 8')
    axes[1,1].plot(x,[r['fault_free_game_return_levels'] for r in rows],color='#488E7C',marker='s',markersize=3,linewidth=1.2,label='Game-return levels · fault-free only')
    axes[1,1].set(title='Within-group reward diversity',ylabel='Distinct values (tolerance 1e−12)',ylim=(-.3,8.4),yticks=[0,2,4,6,8]);axes[1,1].legend(frameon=False,fontsize=8.5,loc='upper left')
    for axis in axes.flat:
        axis.set_xlabel('Attempted training group');axis.set_xlim(.4,max(x)+.6);previous.style_axis(axis)
    prefix='TEST FIXTURE · ' if fixture else '';counts=data['diversity_counts']
    fig.text(.065,.955,prefix+'Mixed-native controller RL: training diagnostics',fontsize=20,weight='bold',color='#17212B')
    fig.text(.065,.912,f"{data['attempted_groups']} attempted groups · {data['actual_optimizer_steps']} actual updates · {data['skipped_groups']} skips · 8 two-call trajectories/group",fontsize=11,color='#4A535D')
    fig.text(.065,.876,f"Reward contrast: {counts['game_return_variation']} groups with varied fault-free game returns; {counts['validity_contrast_only']} with only a validity contrast; {counts['no_reward_variation']} with no variation.",fontsize=9.5,color='#4A535D')
    fig.text(.065,.125,'Training maps and historical opponents change between groups. These curves cannot establish held-out improvement.',fontsize=10,weight='bold',color='#4A535D')
    fig.text(.065,.091,'Final revision earns reward from 2 maps × 4 opponents × 2 seats. Invalid authoring or candidate runtime faults receive −1; every draw stays in the denominator.',fontsize=9,color='#4A535D')
    fig.text(.065,.057,'Only variation among fault-free returns distinguishes their gameplay outcomes. A validity-only contrast does not establish that an update learned better play.',fontsize=9,color='#4A535D')
    return fig


def render(audit_path,prefix):
    audit,manifest,status,schedule,seal,bindings=bind_inputs(audit_path)
    require(audit.get('test_fixture') is not True,'Synthetic fixtures cannot be exported as actual results')
    data=final_data(audit,status,schedule,seal);prefix=Path(prefix)
    targets={(kind,suffix):Path(str(prefix)+'-'+kind+suffix) for kind in data for suffix in ('.png','.svg','.json')}
    if any(p.exists() for p in targets.values()):raise FileExistsError('Existing figures are retained; choose a new output prefix')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    prefix.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.revision-rl-report-',dir=prefix.parent) as tmp:
        staged={}
        for kind,value in data.items():
            fig=(final_figure if kind=='final' else training_figure)(value,plt)
            try:
                for suffix in ('.png','.svg'):
                    path=Path(tmp)/(kind+suffix);fig.savefig(path,dpi=200,facecolor='white');staged[(kind,suffix)]=path
            finally:plt.close(fig)
            record={'kind':'mixed_native_revision_RL_reporting_figure','plot':kind,'source_audit_complete':True,
                'bound_inputs':bindings,'matplotlib_version':matplotlib.__version__,'data':value,
                'outputs':{str(targets[(kind,s)].resolve()):sha(staged[(kind,s)]) for s in ('.png','.svg')}}
            path=Path(tmp)/(kind+'.json');path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');staged[(kind,'.json')]=path
        previous.verify_bindings(bindings)
        for key,path in targets.items():
            with path.open('xb') as stream:stream.write(staged[key].read_bytes())
    return {kind:{suffix:str(targets[(kind,suffix)].resolve()) for suffix in ('.png','.svg','.json')} for kind in data}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output-prefix',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(render(args.audit,args.output_prefix),indent=2))


if __name__=='__main__':main()
