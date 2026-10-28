"""Reporting-only figures for a verified teacher warmstart and subsequent RL.

No model calls or game scoring. Final figures require all64 sealed outputs and
a complete independent audit. Progress shows committed updates only. Synthetic
fixtures may exercise data checks in tests but may never be rendered as figures.
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

sha, require = previous.sha, previous.require


def bind_inputs(audit_path):
    path=Path(audit_path).resolve();raw=path.read_bytes();audit=json.loads(raw)
    require(audit.get('kind')=='independent_cpu_cardinal_teacher_warmstart_audit'
        and audit.get('integrity_passed') is True,'A passing independent warmstart audit is required')
    run=Path(audit['run']).resolve()
    bindings={'audit':{'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()},
        'reporter':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},
        'reporting_helper':{'path':str(Path(previous.__file__).resolve()),'sha256':sha(previous.__file__)}}
    parsed={}
    for name in ('manifest','status','schedule'):
        target=run/(name+'.json');data=target.read_bytes();digest=hashlib.sha256(data).hexdigest()
        expected=audit[name+'_sha256'] if name!='schedule' else parsed['manifest']['schedule_sha256']
        require(digest==expected,'Audit '+name+' identity differs')
        bindings[name]={'path':str(target),'sha256':digest};parsed[name]=json.loads(data)
    manifest,status=parsed['manifest'],parsed['status']
    require(status['manifest_sha256']==bindings['manifest']['sha256'],'Status and manifest differ')
    require(manifest['kind']=='cardinal_teacher_warmstart_league_rl','Different learner protocol')
    for relative,expected in manifest['source_sha256'].items():
        part=Path(relative)
        require(not part.is_absolute() and '..' not in part.parts,'Unsafe archived source path')
        target=run/'source'/part
        require(sha(target)==expected,'Archived source changed: '+relative)
        bindings['source/'+relative]={'path':str(target),'sha256':expected}
    return audit,manifest,status,parsed['schedule'],bindings


def sft_data(audit):
    rows=audit.get('supervised_updates',[])
    require([r.get('step') for r in rows]==list(range(1,len(rows)+1)) and len(rows)<=32,
        'Supervised updates must be a committed prefix of32')
    require(type(audit.get('genuine_supervised_optimizer_steps')) is int
        and audit['genuine_supervised_optimizer_steps']==len(rows),'Supervised actual update count differs')
    tensor=audit.get('tensor_reports',{})
    output=[]
    for row in rows:
        n=row['step'];identity=tensor.get(f'sft_step{n}',{})
        require(identity.get('actual_optimizer_step')==n and identity.get('actual_learning_rate')==5e-5,
            'Supervised Adam lineage/rate differs')
        value=row.get('mean_program_token_CE')
        require(type(value) in (int,float) and math.isfinite(value) and value>=0,'Missing/nonfinite supervised loss')
        require(type(row.get('completion_tokens')) is int and 8<=row['completion_tokens']<=2800,
            'Supervised batch4 completion count differs')
        require(math.isfinite(row['adapter_l2_delta']) and row['adapter_l2_delta']>0
            and type(row['changed_parameters']) is int and row['changed_parameters']>0,'SFT update changed no tensor')
        output.append({'step':n,'mean_program_token_CE':value,'completion_tokens':row['completion_tokens'],
            'adapter_l2_delta':row['adapter_l2_delta']})
    return {'updates':output,'genuine_supervised_optimizer_steps':len(rows),
        'complete':len(rows)==32,'loss_definition':'Mean of four per-program mean completion-token cross-entropies including EOS; prompt masked.',
        'interpretation':'Training loss on the fixed16 teacher demonstrations, repeated8 epochs. Not held-out gameplay or RL reward.'}


def progress_data(audit):
    require(audit.get('integrity_passed') is True,'Independent integrity audit required')
    supervised=sft_data(audit)
    require(audit.get('genuine_RL_optimizer_steps')==audit.get('genuine_optimizer_steps'),
        'RL and legacy actual-step fields disagree')
    committed=[g for g in audit.get('training_groups',[]) if g.get('checkpoint_committed') is True]
    rl=previous.progress_data(audit) if committed else None
    if rl is None:
        require(audit['genuine_RL_optimizer_steps']==0,'RL steps without committed training groups')
    else:
        require(supervised['complete'] and audit.get('optimizer_reset_verified') is True,
            'RL progress without full SFT and verified fresh-Adam reset')
    require(supervised['updates'] or rl is not None,'No committed supervised or RL updates to plot')
    return {'supervised':supervised,'RL':rl,'genuine_RL_optimizer_steps':audit['genuine_RL_optimizer_steps'],
        'complete':audit.get('planned_study_complete') is True,
        'interpretation':'SFT loss and RL step counts are separate. Changing training maps/opponents prevent a held-out improvement claim from progress curves.'}


def final_data(audit,status,schedule):
    supervised=sft_data(audit)
    require(supervised['complete'] and audit.get('optimizer_reset_verified') is True,
        'Fixed SFT baseline requires32 committed supervised updates and verified optimizer reset')
    require(audit.get('genuine_RL_optimizer_steps')==audit.get('genuine_optimizer_steps'),
        'RL optimizer counts disagree')
    result=previous.final_data(audit,status,schedule)
    return {**result,'supervised_optimizer_steps':32,'RL_optimizer_steps':audit['genuine_RL_optimizer_steps'],
        'baseline':'Fixed teacher-SFT adapter, before any RL optimizer update',
        'interpretation':'Incremental RL beyond teacher SFT on a fixed held-out opponent pool. One training seed; descriptive paired-controller bootstrap only.'}


def progress_figure(data,plt):
    fig,axes=plt.subplots(2,2,figsize=(12.5,9.4))
    fig.subplots_adjust(left=.085,right=.96,top=.82,bottom=.19,hspace=.42,wspace=.29)
    supervised=data['supervised']['updates']
    if supervised:
        axes[0,0].plot([r['step'] for r in supervised],[r['mean_program_token_CE'] for r in supervised],
            color='#7A569B',marker='o',markersize=3,linewidth=1.4)
        axes[0,0].set_xlim(.5,max(1,len(supervised))+.5)
    else:axes[0,0].text(.5,.5,'No committed SFT updates',transform=axes[0,0].transAxes,ha='center')
    axes[0,0].set(title='Teacher SFT · training token loss',xlabel='Committed supervised update',ylabel='Mean per-program token cross-entropy')
    rl=data['RL']
    if rl:
        rows=rl['groups'];x=[r['cycle'] for r in rows]
        axes[0,1].plot(x,[r['mean_yield'] for r in rows],color='#258773',marker='o',markersize=3,linewidth=1.4)
        for key,label,color in [('compiled_fraction','Compiled','#758391'),('fault_free_fraction','Fault-free','#267AA1'),
                ('positive_yield_fraction','Positive banking','#258773')]:
            axes[1,0].plot(x,[r[key] for r in rows],label=label,color=color,linewidth=1.4)
        axes[1,0].legend(loc='lower left',frameon=False,fontsize=9)
        axes[1,1].step([0]+x,[0]+[r['actual_optimizer_step'] for r in rows],where='post',color='#267AA1',linewidth=1.6)
        skipped=[r for r in rows if not r['update_committed']]
        if skipped:
            axes[1,1].scatter([r['cycle'] for r in skipped],[r['actual_optimizer_step'] for r in skipped],
                marker='x',color='#BD7A2A',label='Zero variance; no update',s=25)
            axes[1,1].legend(frameon=False,fontsize=9)
        for ax in (axes[0,1],axes[1,0],axes[1,1]):ax.set_xlim(0,max(x)+.5)
    else:
        for ax in (axes[0,1],axes[1,0],axes[1,1]):
            ax.text(.5,.5,'No committed RL group',transform=ax.transAxes,ha='center',color='#758391')
    axes[0,1].set(title='RL training · all8-output banking yield',xlabel='Attempted RL group',ylabel='Mean banked points per game')
    axes[0,1].set_ylim(bottom=0)
    axes[1,0].set(title='RL authoring and useful play',xlabel='Attempted RL group',ylabel='Fraction of all8 prescribed draws',ylim=(-.03,1.08))
    axes[1,1].set(title='Actual RL optimizer steps',xlabel='Attempted RL group',ylabel='Cumulative committed Adam steps')
    for ax in axes.flat:previous.style_axis(ax)
    fig.text(.075,.95,'Teacher warmstart + RL: training diagnostics',fontsize=20,weight='bold',color='#17212B')
    state='Completed audited study' if data['complete'] else 'Partial audited snapshot'
    fig.text(.075,.905,f"{state} · {len(supervised)} supervised steps · {data['genuine_RL_optimizer_steps']} actual RL steps",fontsize=11,color='#4A535D')
    fig.text(.075,.12,'SFT: fixed16 verified teacher demonstrations,8 epochs; each update averages4 per-program token means. This is training loss.',fontsize=9.2,color='#4A535D')
    fig.text(.075,.085,'RL: changing maps and historical opponents. Yield/validity include every prescribed output; invalid authoring contributes zero yield.',fontsize=9.2,color='#4A535D')
    fig.text(.075,.05,'These curves do not establish held-out improvement. Missing or uncommitted groups are omitted; zero-signal groups are not optimizer updates.',fontsize=9.2,color='#4A535D')
    return fig


def final_figure(data,plt):
    """Reuse the audited n32 geometry; relabel every baseline as SFT explicitly."""
    fig=previous.final_figure(data,plt,fixture=False)
    replacements={
        'Cardinal controller RL: held-out program yield':'Teacher warmstart + RL: held-out program yield',
        f"Qwen3.8-27B · base versus cycle {data['endpoint']} · {data['actual_optimizer_steps']} actual optimizer steps · all 64 evaluation outputs":
            f"Qwen3.8-27B · SFT baseline versus cycle {data['endpoint']} · 32 supervised + {data['RL_optimizer_steps']} RL steps · all64 outputs",
        'Fresh GPU base':'SFT baseline · 32 supervised steps',
        f"Cycle {data['endpoint']}":f"RL final · cycle {data['endpoint']}",
    }
    for text in fig.texts:
        current=text.get_text()
        if current in replacements:text.set_text(replacements[current])
        elif current.startswith('Mean over all 32 outputs:'):
            text.set_text(current.replace('base ','SFT baseline ').replace('final ','RL final '))
        elif current.startswith('Base: '):text.set_text(current.replace('Base: ','SFT baseline: ').replace('Final: ','RL final: '))
    for legend in fig.legends:
        for text in legend.get_texts():
            if text.get_text() in replacements:text.set_text(replacements[text.get_text()])
    fig.axes[1].set_xlabel('RL final − SFT baseline banked points')
    return fig


def render(audit_path,prefix,mode='both'):
    require(mode in ('progress','final','both'),'Unknown report mode')
    audit,manifest,status,schedule,bindings=bind_inputs(audit_path)
    require(audit.get('test_fixture') is not True,'Synthetic fixtures must not be rendered as actual experiment figures')
    data={}
    if mode in ('progress','both'):data['progress']=progress_data(audit)
    if mode in ('final','both'):data['final']=final_data(audit,status,schedule)
    prefix=Path(prefix)
    targets={(kind,suffix):Path(str(prefix)+'-'+kind+suffix) for kind in data for suffix in ('.png','.svg','.json')}
    if any(path.exists() for path in targets.values()):raise FileExistsError('Choose a new report prefix; previous figures are retained')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    prefix.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.warmstart-report-',dir=prefix.parent) as temporary:
        staged={}
        for kind,value in data.items():
            fig=(progress_figure if kind=='progress' else final_figure)(value,plt)
            try:
                for suffix in ('.png','.svg'):
                    path=Path(temporary)/(kind+suffix);fig.savefig(path,dpi=200,facecolor='white');staged[(kind,suffix)]=path
            finally:plt.close(fig)
            record={'kind':'cardinal_teacher_warmstart_reporting_figure','plot':kind,'bound_inputs':bindings,
                'source_audit_complete':audit.get('planned_study_complete') is True,
                'matplotlib_version':matplotlib.__version__,'data':value,
                'outputs':{str(targets[(kind,s)].resolve()):sha(staged[(kind,s)]) for s in ('.png','.svg')}}
            path=Path(temporary)/(kind+'.json');path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');staged[(kind,'.json')]=path
        previous.verify_bindings(bindings)
        for key,path in targets.items():
            with path.open('xb') as stream:stream.write(staged[key].read_bytes())
    return {kind:{s:str(targets[(kind,s)].resolve()) for s in ('.png','.svg','.json')} for kind in data}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True);parser.add_argument('--output-prefix',type=Path,required=True)
    parser.add_argument('--mode',choices=('progress','final','both'),default='both')
    args=parser.parse_args();print(json.dumps(render(args.audit,args.output_prefix,args.mode),indent=2))


if __name__=='__main__':main()
