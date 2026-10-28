"""Static reporting for the completed, independently audited two-call RL study.

No model, API, scoring, or partial-evaluation access. Both figures require the
fixed endpoint and all64 sealed trajectories (128 sampled turns). Existing plot
geometry is reused, with explicit same-harness before/after labels.
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

sha,require,close=previous.sha,previous.require,previous.close


def bind_inputs(audit_path):
    """Hash exactly the bytes parsed, then recheck every binding before export."""
    path=Path(audit_path).resolve();raw=path.read_bytes();audit=json.loads(raw)
    require(audit.get('kind')=='independent_cpu_program_debug_rl_audit' and audit.get('integrity_passed') is True,
            'A passing independent two-call RL audit is required')
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
    manifest,status=parsed['manifest'],parsed['status']
    require(manifest['kind']=='thinking_debug_league_rl' and status['manifest_sha256']==bindings['manifest']['sha256'],
            'Different learner protocol or status/manifest identity')
    cfg=manifest['config']
    require(cfg['thinking'] is True and cfg['authoring_calls']==2 and cfg['eval_samples']==32
            and cfg['group_size']==8 and cfg['endpoint_choices']==[25,50,100],'Same two-call protocol differs')
    for relative,expected in manifest['source_sha256'].items():
        part=Path(relative);require(not part.is_absolute() and '..' not in part.parts,'Unsafe source path')
        target=run/'source'/part
        require(sha(target)==expected,'Audited source archive changed: '+relative)
        bindings['source/'+relative]={'path':str(target),'sha256':expected}
    require(audit.get('auditor_sha256')==manifest['source_sha256']['scripts/analyze_program_debug_rl.py'],
            'Independent auditor differs from frozen source')
    # The independent audit already verified all games; this reporter additionally
    # binds the current complete-output seal rather than relying on a count label.
    seal_path=run/'evaluation/sealed.json'
    if 'evaluation/sealed.json' in status.get('artifacts',{}):
        data=seal_path.read_bytes();digest=hashlib.sha256(data).hexdigest()
        require(digest==status['artifacts']['evaluation/sealed.json'],'Heldout seal changed')
        seal=json.loads(data);bindings['evaluation_seal']={'path':str(seal_path),'sha256':digest}
    else:seal=None
    return audit,manifest,status,parsed['schedule'],seal,bindings


def final_data(audit,status,schedule,seal):
    require(audit.get('planned_study_complete') is True and audit.get('planned_comparisons_available') is True
            and audit.get('integrity_passed') is True and audit.get('recorded_state')=='complete'
            and status.get('state')=='complete' and status.get('claims_allowed') is True,
            'Figures require complete status and a complete independent audit')
    endpoint=audit.get('selected_endpoint')
    require(type(endpoint) is int and endpoint in (25,50,100),'Unexpected fixed endpoint')
    require(audit.get('new_supervised_optimizer_steps')==0,'This comparison contains unexpected new SFT')
    require(audit.get('genuine_RL_optimizer_steps')==audit.get('genuine_optimizer_steps'), 'Actual RL step fields disagree')
    total=8*endpoint+64
    require(audit.get('committed_trajectories')==total and audit.get('committed_raw_sampled_turns')==2*total
            and audit.get('raw_turns_pending_trajectory_materialization')==0,'All64 final two-call trajectories are required')
    labels=('base',f'cycle{endpoint}')
    names=[f'generations/eval-{label}-{i}.json' for label in labels for i in range(32)]
    require(isinstance(seal,dict) and seal.get('draws')==64 and seal.get('sampled_turns')==128
            and seal.get('scoring_allowed') is True and set(seal.get('generations',{}))==set(names)
            and set(seal.get('checkpoints',{}))==set(labels),'All64 final outputs must be sealed before scoring')
    require(all(seal['generations'][p]==status['artifacts'].get(p) for p in names), 'Sealed generation hashes differ')
    require(all(seal['checkpoints'][label]==status['artifacts'].get(f'checkpoints/{label}/metadata.json') for label in labels)
            and seal['endpoint_selection_sha256']==status['artifacts'].get('endpoint-selection.json'), 'Seal checkpoint/endpoint binding differs')
    require(set(audit['evaluation_cells'])==set(labels),'Both fixed heldout endpoints required')
    cells={label:previous.checked_cell(audit['evaluation_cells'][label],[f'eval-{label}-{i}' for i in range(32)],48) for label in labels}
    seeds=schedule['eval_generation_seeds'];contrast=audit['paired_yield']
    require(len(seeds)==len(set(seeds))==32 and all(type(s) is int for s in seeds)
            and contrast['paired_generation_seeds']==seeds and contrast['program_cluster_n']==32,'Paired trajectory seeds differ')
    delta=[b['banked_point_yield']-a['banked_point_yield'] for a,b in zip(cells['base']['programs'],cells[labels[1]]['programs'])]
    require(len(contrast['paired_yield_differences'])==32 and all(close(a,b) for a,b in zip(delta,contrast['paired_yield_differences'])),
            'Paired final-revision differences disagree')
    require(close(cells['base']['mean'],contrast['base_mean_yield']) and close(cells[labels[1]]['mean'],contrast['final_mean_yield'])
            and close(math.fsum(delta)/32,contrast['mean_yield_difference']),'All-draw paired means disagree')
    interval=previous.bootstrap_interval(delta,contrast['bootstrap_draws'],contrast['bootstrap_seed'])
    require(len(contrast['descriptive_bootstrap_95_interval'])==2 and
            all(close(a,b) for a,b in zip(interval,contrast['descriptive_bootstrap_95_interval'])),'Paired trajectory bootstrap differs')
    progress=previous.progress_data(audit)
    require(progress['committed_groups']==endpoint and not progress['scored_uncommitted_cycles'],'Fixed training endpoint incomplete')
    require(status.get('attempted_groups')==endpoint and status.get('actual_optimizer_steps')==audit['genuine_RL_optimizer_steps'],
            'Terminal optimizer count differs')
    skipped=sum(not row['update_committed'] for row in progress['groups'])
    require(skipped+audit['genuine_RL_optimizer_steps']==endpoint,'Skipped/updated groups do not cover endpoint')
    final={'endpoint':endpoint,'labels':labels,'cells':cells,'generation_seeds':seeds,'paired_differences':delta,
        'mean_difference':math.fsum(delta)/32,'descriptive_bootstrap_95_interval':interval,'program_cluster_n':32,
        'actual_optimizer_steps':audit['genuine_RL_optimizer_steps'],'skipped_groups':skipped,
        'baseline':'Imported SFT adapter with zero RL steps, using the same two-call thinking/practice/revision harness',
        'authoring_calls_per_trajectory':2,'mandatory_final_turn':1,'new_supervised_optimizer_steps':0,
        'interpretation':'Incremental RL under the same two-call harness and fixed heldout opponent pool. One training seed; descriptive paired-trajectory uncertainty. No teacher-equivalence or general-transfer claim.'}
    return {'final':final,'training':{**progress,'skipped_groups':skipped,'authoring_calls_per_trajectory':2}}


def final_figure(data,plt):
    fig=previous.final_figure(data,plt,fixture=False)
    replacements={
        'Cardinal controller RL: held-out program yield':'Thinking + practice + revision: held-out yield',
        f"Qwen3.8-27B · base versus cycle {data['endpoint']} · {data['actual_optimizer_steps']} actual optimizer steps · all 64 evaluation outputs":
            f"Same two-call harness before/after RL · {data['actual_optimizer_steps']} actual updates · {data['skipped_groups']} skipped groups · all 64 final revisions",
        'Fresh GPU base':'Before RL · imported SFT weights',
        f"Cycle {data['endpoint']}":f"After RL · fixed group {data['endpoint']}",
    }
    for text in fig.texts:
        value=text.get_text()
        if value in replacements:text.set_text(replacements[value])
        elif value.startswith('Mean over all 32 outputs:'):
            text.set_text(value.replace('base ','before RL ').replace('final ','after RL '))
        elif value.startswith('Base: '):text.set_text(value.replace('Base: ','Before RL: ').replace('Final: ','After RL: '))
        elif value.startswith('Each compiled controller:'):
            text.set_text('Each compiled final revision: 6 held-out maps × 4 fixed opponents × 2 seats = 48 correlated games. Invalid authoring: zero yield, no game.')
        elif value.startswith('Interval: descriptive'):
            text.set_text('Interval: descriptive 95% paired-trajectory bootstrap (n = 32); one training seed. Each trajectory is draft→practice→revision. Missing is never zero.')
        elif value.startswith('Evidence is specific'):
            text.set_text('Inherited 32-step SFT weights are shared. This measures added RL under one fixed harness/pool; it is not a teacher-performance comparison.')
    for legend in fig.legends:
        for text in legend.get_texts():
            if text.get_text() in replacements:text.set_text(replacements[text.get_text()])
    fig.axes[0].set_ylabel('Paired two-call trajectory seed')
    fig.axes[0].set_title('All 32 mandatory final revisions per checkpoint',loc='left',fontsize=12,pad=12)
    fig.axes[1].set_xlabel('After RL − before RL banked points')
    return fig


def training_figure(data,plt):
    fig=previous.progress_figure(data,plt,fixture=False)
    for text in fig.texts:
        value=text.get_text()
        if value=='Cardinal controller RL: training progress':text.set_text('Two-call controller RL: committed training groups')
        elif value.startswith('Completed study ·'):
            text.set_text(f"{data['committed_groups']} groups · {data['genuine_optimizer_steps']} actual RL updates · {data['skipped_groups']} zero-signal skips · 8 trajectories/group")
        elif value.startswith('Each compiled controller plays'):
            text.set_text('Only final revisions earn game reward: 2 maps × 4 opponents × 2 seats. All 8 draws remain in yield/validity denominators; invalid yields zero.')
        elif value.startswith('Runtime faults remain'):
            text.set_text('Each trajectory has two sampled calls with practice feedback between them. A skipped group copies the adapter and Adam state without updating.')
    fig.axes[0].set_title('Final-revision training reward · all 8 trajectories')
    fig.axes[1].set_title('Final-revision banking · all 8 trajectories')
    fig.axes[2].set_title('Final-revision validity and useful play')
    return fig


def render(audit_path,prefix):
    audit,manifest,status,schedule,seal,bindings=bind_inputs(audit_path)
    require(audit.get('test_fixture') is not True,'Synthetic fixtures cannot be rendered as actual study results')
    data=final_data(audit,status,schedule,seal)
    prefix=Path(prefix)
    targets={(kind,suffix):Path(str(prefix)+'-'+kind+suffix) for kind in data for suffix in ('.png','.svg','.json')}
    if any(p.exists() for p in targets.values()):raise FileExistsError('Choose a new prefix; existing scientific figures are retained')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    prefix.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.debug-report-',dir=prefix.parent) as temporary:
        staged={}
        for kind,value in data.items():
            fig=(final_figure if kind=='final' else training_figure)(value,plt)
            try:
                for suffix in ('.png','.svg'):
                    path=Path(temporary)/(kind+suffix);fig.savefig(path,dpi=200,facecolor='white');staged[(kind,suffix)]=path
            finally:plt.close(fig)
            record={'kind':'two_call_debug_RL_reporting_figure','plot':kind,'source_audit_complete':True,
                'bound_inputs':bindings,'matplotlib_version':matplotlib.__version__,'data':value,
                'outputs':{str(targets[(kind,s)].resolve()):sha(staged[(kind,s)]) for s in ('.png','.svg')}}
            path=Path(temporary)/(kind+'.json');path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');staged[(kind,'.json')]=path
        previous.verify_bindings(bindings)
        for key,path in targets.items():
            with path.open('xb') as stream:stream.write(staged[key].read_bytes())
    return {kind:{suffix:str(targets[(kind,suffix)].resolve()) for suffix in ('.png','.svg','.json')} for kind in data}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output-prefix',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(render(args.audit,args.output_prefix),indent=2))


if __name__=='__main__':main()
