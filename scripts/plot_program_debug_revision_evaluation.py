"""Complete-only 27B closure reporting; no neural execution or game scoring.

This separate evaluation measures three existing RL updates. It does not make
its interrupted parent training study complete and performs no new updates.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import plot_cardinal_gpu as helper

require,close=helper.require,helper.close
N=16
LABELS=('base','cycle3')
AUDITOR='scripts/analyze_program_debug_revision_evaluation.py'
MODEL='Qwen/Qwen3.8-27B'
REVISION='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def complete_audit(audit):
    require(audit.get('kind')=='independent_cpu_revision_evaluation_audit'
        and audit.get('integrity_passed') is True and audit.get('planned_study_complete') is True
        and audit.get('planned_comparisons_available') is True and audit.get('recorded_state')=='complete',
        'Only a complete passing independent evaluation-closure audit can be reported')
    require(audit.get('parent_training_complete') is False and audit.get('inherited_RL_optimizer_steps')==3
        and audit.get('new_supervised_optimizer_steps')==0 and audit.get('new_RL_optimizer_steps')==0
        and audit.get('fixed_final_checkpoint')=='cycle3','Three inherited updates and zero new updates required')
    require(audit.get('committed_trajectories')==32 and audit.get('committed_raw_sampled_turns')==64
        and audit.get('raw_turns_pending_trajectory_materialization')==0,
        'All 32 evaluation trajectories and 64 sampled turns are required')


def bind_inputs(audit_path):
    path=Path(audit_path).resolve();raw=path.read_bytes();audit=json.loads(raw)
    complete_audit(audit)  # Never read heldout evidence for a partial audit.
    run=Path(audit['run']).resolve()
    bindings={'audit':{'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()},
        'reporter':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},
        'helper':{'path':str(Path(helper.__file__).resolve()),'sha256':sha(helper.__file__)}}
    def read(relative,expected):
        p=(run/relative).resolve();require(p.is_relative_to(run),'Evidence path escapes run')
        raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();require(h==expected,'Audited evidence changed: '+relative)
        bindings[relative]={'path':str(p),'sha256':h};return json.loads(raw)
    manifest=read('manifest.json',audit['manifest_sha256']);status=read('status.json',audit['status_sha256'])
    cfg=manifest['config']
    require(manifest['kind']=='mixed_native_revision_evaluation_only' and manifest['phase']=='coding_evaluation'
        and manifest['parent_training_complete'] is False and manifest['new_optimizer_steps']==0
        and manifest['parent_manifest_sha256']==audit['parent_manifest_sha256']
        and status['manifest_sha256']==audit['manifest_sha256'],'Closure origin/protocol identity differs')
    require(cfg['model_repo']==MODEL and cfg['model_revision']==REVISION and cfg['authoring_calls']==2
        and cfg['eval_samples']==N and cfg['batch_size']==2 and cfg['evaluation_only'] is True
        and cfg['training_allowed'] is False and cfg['new_optimizer_steps']==0
        and cfg['fixed_evaluation_checkpoints']==list(LABELS)
        and cfg['turn_native_modes']['draft']['thinking_enabled'] is True
        and cfg['turn_native_modes']['draft']['reasoning_effort']=='low'
        and cfg['turn_native_modes']['revision']['thinking_enabled'] is False
        and cfg['turn_native_modes']['revision']['reasoning_effort'] is None,'Original two-call native protocol differs')
    require(manifest['source_sha256'][AUDITOR]==audit['auditor_sha256'],'Auditor differs from frozen source')
    for namespace,base,entries in (('source',run/'source',manifest['source_sha256']),
        ('input',run,manifest['input_sha256']),('artifact',run,status['artifacts'])):
        for relative,expected in entries.items():
            q=Path(relative);target=(base/q).resolve()
            require(not q.is_absolute() and '..' not in q.parts and target.is_relative_to(base.resolve()),'Unsafe bound evidence path')
            require(sha(target)==expected,'Audited '+namespace+' changed: '+relative)
            bindings[namespace+'/'+relative]={'path':str(target),'sha256':expected}
    for relative,key in [('parent/manifest.json','parent_manifest_sha256'),('parent/status.json','parent_status_sha256'),
        ('parent/final-audit.json','parent_audit_sha256')]:
        require(manifest['input_sha256'][relative]==manifest[key],'Parent origin binding differs')
    for label,step in zip(LABELS,(0,3)):
        inherited=manifest['inherited_checkpoints'][label];name=f'checkpoints/{label}/metadata.json'
        require(inherited['metadata_sha256']==status['artifacts'][name]
            and inherited['origin_manifest_sha256']==manifest['parent_manifest_sha256']
            and inherited['prior_SFT_steps']==32 and inherited['prior_RL_steps']==step
            and inherited['optimizer_identity']['step']==step,'Inherited baseline/cycle3 lineage differs')
    schedule=read('schedule.json',manifest['input_sha256']['schedule.json'])
    seal=read('evaluation/sealed.json',status['artifacts']['evaluation/sealed.json'])
    return audit,manifest,status,schedule,seal,bindings


def data(audit,manifest,status,schedule,seal):
    complete_audit(audit)
    require(status.get('state')=='complete' and status.get('claims_allowed') is True
        and status.get('learning_comparison_available') is True and status.get('parent_training_complete') is False
        and status.get('new_optimizer_steps')==0 and status.get('inherited_optimizer_steps')==3
        and status.get('final_checkpoint')=='cycle3','Completed closure status/scope differs')
    paths=[f'generations/eval-{label}-{i}.json' for label in LABELS for i in range(N)]
    require(seal.get('draws')==32 and seal.get('sampled_turns')==64 and seal.get('scoring_allowed') is True
        and seal.get('new_optimizer_steps')==0 and set(seal.get('generations',{}))==set(paths)
        and set(seal.get('checkpoints',{}))==set(LABELS)
        and set(seal.get('endpoint_state_proofs',{}))==set(LABELS),'All 32 final revisions and endpoint state proofs must be sealed')
    require(all(seal['generations'][p]==status['artifacts'].get(p) for p in paths),'Sealed generation hashes differ')
    require(seal['evaluation_plan_sha256']==manifest['evaluation_plan_sha256'],'Evaluation plan seal differs')
    for label,step in zip(LABELS,(0,3)):
        require(seal['checkpoints'][label]==status['artifacts'].get(f'checkpoints/{label}/metadata.json')
            and seal['endpoint_state_proofs'][label]==status['artifacts'].get(f'endpoint-state/{label}/proof.json'),
            'Exact endpoint or unchanged-state proof seal differs')
        t=audit['tensor_reports'][label]
        require(t['actual_optimizer_step']==step and t['unchanged_import'] is True
            and t['origin_manifest_sha256']==manifest['parent_manifest_sha256'],'Audited inherited tensor lineage differs')
    require(set(audit['evaluation_cells'])==set(LABELS),'Both fixed endpoints required')
    cells={label:helper.checked_cell(audit['evaluation_cells'][label],[f'eval-{label}-{i}' for i in range(N)],48) for label in LABELS}
    seeds=schedule['eval_generation_seeds'];paired=audit['paired_yield']
    require(len(seeds)==len(set(seeds))==N and all(type(s) is int for s in seeds)
        and paired['paired_generation_seeds']==seeds and paired['program_cluster_n']==N,'Sixteen paired authoring seeds differ')
    delta=[b['banked_point_yield']-a['banked_point_yield'] for a,b in zip(cells['base']['programs'],cells['cycle3']['programs'])]
    require(len(paired['paired_yield_differences'])==N and all(close(a,b) for a,b in zip(delta,paired['paired_yield_differences'])),
        'Paired yield differences disagree')
    mean=math.fsum(delta)/N
    require(close(mean,paired['mean_yield_difference']) and close(cells['base']['mean'],paired['base_mean_yield'])
        and close(cells['cycle3']['mean'],paired['final_mean_yield']),'All-draw paired means differ')
    ci=helper.bootstrap_interval(delta,paired['bootstrap_draws'],paired['bootstrap_seed'])
    require(len(paired['descriptive_bootstrap_95_interval'])==2 and all(close(a,b) for a,b in zip(ci,paired['descriptive_bootstrap_95_interval'])),
        'Paired-trajectory bootstrap differs')
    counts={label:{'prescribed':N,'compiled':N-cell['counts']['invalid_authorship'],
        'fault_free':cell['counts']['valid_zero']+cell['counts']['valid_positive'],
        'productive':cell['counts']['valid_positive'],'invalid_authorship':cell['counts']['invalid_authorship'],
        'runtime_fault':cell['counts']['runtime_fault'],'played_games':sum(r['games'] for r in cell['programs'])}
        for label,cell in cells.items()}
    return {'kind':'complete_revision_evaluation_closure_report','model':MODEL,'model_revision':REVISION,
        'labels':list(LABELS),'cells':cells,'counts':counts,'generation_seeds':seeds,'program_cluster_n':N,
        'paired_differences':delta,'mean_difference':mean,'descriptive_bootstrap_95_interval':ci,
        'bootstrap':{'draws':paired['bootstrap_draws'],'seed':paired['bootstrap_seed'],'unit':'Paired two-call authoring trajectory'},
        'inherited_SFT_steps_both_endpoints':32,'inherited_RL_steps':3,'new_SFT_steps':0,'new_RL_steps':0,
        'parent_training_complete':False,'scope':'Separate measurement of three existing RL updates; the original longer training run remains incomplete.',
        'productive_definition':'Fault-free compiled final controller with positive mean banked points across its 48 prescribed heldout games. This is descriptive and is not the earlier capability-admission criterion.',
        'uncertainty':'Descriptive 95% paired-trajectory bootstrap; 16 clusters and one training seed. The two turns and 48 games per compiled controller are dependent.',
        'caveats':['All 16 authoring draws per endpoint remain in the mean. Invalid authorship yields zero with no invented games; missing draws prevent reporting.',
            'Compiled controllers with runtime faults retain their actual banked yield and are shown separately.',
            'The same native-low draft, practice feedback and native-nonthinking revision harness is used at both endpoints.',
            'Two fixed evaluation teacher opponents were prior SFT examples, excluded from the RL roster but not unseen to weights.',
            'This is not a completed 10/15/25-group study, a causal harness comparison, evidence of general transfer or guaranteed continual improvement.']}


def make_figure(report,plt,fixture=False):
    from matplotlib.lines import Line2D
    colors=('#687785','#177EA2');labels=report['labels'];n=N
    fig=plt.figure(figsize=(13.5,10.4));grid=fig.add_gridspec(2,2,width_ratios=(1.2,1),
        left=.13,right=.96,top=.78,bottom=.29,hspace=.63,wspace=.30)
    ax=fig.add_subplot(grid[:,0]);contrast=fig.add_subplot(grid[0,1]);rates=fig.add_subplot(grid[1,1])
    maximum=max(1.,max(p['banked_point_yield'] for cell in report['cells'].values() for p in cell['programs']))
    for i in range(n):
        y=n-1-i;rows=[report['cells'][label]['programs'][i] for label in labels]
        ax.plot([r['banked_point_yield'] for r in rows],[y-.15,y+.15],color='#CBD5DF',lw=.8,zorder=1)
        for j,row in enumerate(rows):
            cat=row['category'];marker='X' if cat=='invalid_authorship' else '^' if cat=='runtime_fault' else 'o'
            ax.scatter(row['banked_point_yield'],y+(-.15 if j==0 else .15),marker=marker,s=35,
                facecolors='white' if cat=='valid_zero' else colors[j],edgecolors=colors[j],lw=1,zorder=3)
    ax.set(yticks=list(range(n-1,-1,-1)),yticklabels=[str(s) for s in report['generation_seeds']],
        xlabel='Mean banked points per held-out game',ylabel='Paired two-call authoring seed',
        xlim=(-.05*maximum,1.08*maximum),ylim=(-.6,n-.4),title='All 16 final controllers per checkpoint')
    ax.tick_params(axis='y',labelsize=8);ax.grid(axis='x',color='#E7EDF2',lw=.6)
    lo,hi=report['descriptive_bootstrap_95_interval'];avg=report['mean_difference']
    contrast.scatter(report['paired_differences'],[.3+(((i*7)%n)-7.5)*.012 for i in range(n)],s=21,color='#99B8C6')
    contrast.plot([lo,hi],[0,0],color=colors[1],lw=3);contrast.scatter([avg],[0],marker='D',s=55,color=colors[1])
    contrast.axvline(0,color='#B3BEC7',lw=1);contrast.set(yticks=[],ylim=(-.2,.5),xlabel='Cycle3 − baseline banked points',title='Paired change + descriptive 95% interval')
    contrast.text(.03,.91,f'{avg:+.3f}  [{lo:+.3f}, {hi:+.3f}]',transform=contrast.transAxes,weight='bold',color='#254D63')
    for j,label in enumerate(labels):
        values=[report['counts'][label][k] for k in ('compiled','fault_free','productive')]
        bars=rates.barh([2-.17+j*.34,1-.17+j*.34,-.17+j*.34],values,height=.28,color=colors[j])
        for bar,value in zip(bars,values):rates.text(value+.2,bar.get_y()+bar.get_height()/2,str(value),va='center',fontsize=9)
    rates.set(yticks=[2,1,0],yticklabels=['Compiled','Fault-free','Fault-free\n+ banking'],xlim=(0,18),xticks=[0,4,8,12,16],
        xlabel='Programs out of 16 prescribed draws',title='Usable controllers · overlapping counts')
    for axis in (ax,contrast,rates):helper.style_axis(axis)
    prefix='SYNTHETIC TEST FIXTURE · ' if fixture else ''
    fig.text(.075,.957,prefix+'27B evaluation: three existing RL updates',fontsize=19,weight='bold',color='#172B3B')
    fig.text(.075,.918,'Exact SFT32 baseline versus cycle3 · zero new training updates · original longer run remains incomplete',fontsize=11,color='#475569')
    fig.legend(handles=[Line2D([0],[0],color=c,marker='o',ls='none',label=s) for c,s in zip(colors,('Baseline · SFT32, RL0','Cycle3 · SFT32, RL3'))],
        loc='upper left',bbox_to_anchor=(.067,.895),ncol=2,frameon=False,fontsize=10)
    fig.legend(handles=[Line2D([0],[0],color='#526372',marker=mark,ls='none',markerfacecolor=face,label=text)
        for mark,face,text in [('X','#526372','Invalid authoring; no games'),('^','#526372','Runtime faults'),('o','white','Fault-free zero yield'),('o','#526372','Fault-free positive yield')]],
        loc='upper left',bbox_to_anchor=(.067,.858),ncol=4,frameon=False,fontsize=8.5)
    fig.text(.075,.216,f"Mean of all 16 draws: baseline {report['cells']['base']['mean']:.3f} → cycle3 {report['cells']['cycle3']['mean']:.3f} banked points/game.",fontsize=11,weight='bold',color='#254D63')
    fig.text(.075,.164,'Each compiled final plays 6 maps × 4 fixed opponents × 2 seats = 48 correlated games. Invalid authorship stays zero; missing draws block reporting.',fontsize=9,color='#475569')
    fig.text(.075,.127,'Uncertainty resamples 16 paired authoring trajectories, including invalid draws. One training seed; turns and individual games are not independent.',fontsize=9,color='#475569')
    fig.text(.075,.090,'Both endpoints use native-low draft thinking, the same practice feedback, then a native-nonthinking code revision. No new SFT or RL occurs here.',fontsize=9,color='#475569')
    fig.text(.075,.053,'Two evaluation teachers were prior SFT examples. This closes only the three-update comparison; it does not complete the abandoned longer study.',fontsize=9,color='#475569')
    return fig


def render(audit_path,prefix):
    audit,manifest,status,schedule,seal,bindings=bind_inputs(audit_path)
    require(not audit.get('test_fixture'),'Synthetic fixtures cannot be published as actual results')
    report=data(audit,manifest,status,schedule,seal);prefix=Path(prefix)
    targets={s:Path(str(prefix)+s) for s in ('.png','.svg','.json')}
    if any(p.exists() for p in targets.values()):raise FileExistsError('Preserve existing figures; choose a fresh prefix')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    prefix.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.revision-evaluation-report-',dir=prefix.parent) as tmp:
        fig=make_figure(report,plt);staged={}
        try:
            for suffix in ('.png','.svg'):
                path=Path(tmp)/('figure'+suffix);fig.savefig(path,dpi=200,facecolor='white');staged[suffix]=path
        finally:plt.close(fig)
        record={'data':report,'bound_inputs':bindings,'matplotlib_version':matplotlib.__version__,
            'outputs':{str(targets[s].resolve()):sha(staged[s]) for s in ('.png','.svg')}}
        staged['.json']=Path(tmp)/'figure.json';staged['.json'].write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        helper.verify_bindings(bindings)
        if any(p.exists() for p in targets.values()):raise FileExistsError('Output appeared during rendering; preserve it')
        for suffix,target in targets.items():os.link(staged[suffix],target)
    return {s:str(p.resolve()) for s,p in targets.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--out-prefix',type=Path,required=True)
    args=p.parse_args();print(json.dumps(render(args.audit,args.out_prefix),indent=2))


if __name__=='__main__':main()
