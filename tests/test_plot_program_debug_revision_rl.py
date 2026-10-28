"""Synthetic reporting fixtures only; no active-run data, neural calls or games."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import plot_program_debug_revision_rl as plot


def cell(names,games,*,final=False,all_invalid=False,one_valid_return=False):
    rows=[]
    for i,name in enumerate(names):
        authored=(not all_invalid) and i%4!=0;fault=authored and i%4==1
        valid=authored and not fault
        bank=2. if fault else float(i%5+1)+(1. if final else 0.) if valid and i%4==3 else 0.
        ret=(.2 if one_valid_return else -.5 if i%4==2 else .2) if authored else None
        rows.append({'generation':name,'banked_point_yield':bank,'authorship_valid':authored,
            'runtime_valid':valid,'candidate_program_faults':int(fault),'games':games if authored else 0,
            'game_return':ret,'reward':ret if valid else -1.})
    return {'complete':True,'prescribed_program_clusters':len(names),'observed_program_clusters':len(names),
        'missing_program_clusters':0,'programs':rows,
        'primary_mean_banked_point_yield':math.fsum(r['banked_point_yield'] for r in rows)/len(rows),
        'compiled_programs':sum(r['authorship_valid'] for r in rows),
        'invalid_no_game_programs':sum(not r['authorship_valid'] for r in rows),
        'fault_free_programs':sum(r['runtime_valid'] for r in rows),'played_games':sum(r['games'] for r in rows),
        'mean_penalized_task_reward':math.fsum(r['reward'] for r in rows)/len(rows)}


def fixture(endpoint=10,zero=False):
    groups=[];tensors={'base':{'actual_optimizer_step':0}};step=0
    for n in range(1,endpoint+1):
        group=cell([f'train-{n}-{i}' for i in range(8)],16,all_invalid=zero or n%3==0,one_valid_return=n%3==2)
        rewards=[r['reward'] for r in group['programs']];signal=max(rewards)-min(rewards)>1e-12;step+=signal
        group.update(cycle=n,checkpoint_committed=True,rewards=rewards,
            advantages=[r-(math.fsum(rewards)-r)/7 for r in rewards] if signal else [0.]*8)
        tensors[f'cycle{n}']={'actual_optimizer_step':int(step)};groups.append(group)
    labels=('base',f'cycle{endpoint}')
    cells={label:cell([f'eval-{label}-{i}' for i in range(16)],48,final=label!='base',all_invalid=zero) for label in labels}
    delta=[b['banked_point_yield']-a['banked_point_yield'] for a,b in zip(cells['base']['programs'],cells[labels[1]]['programs'])]
    seeds=[241000000+2*i for i in range(16)]
    audit={'kind':'independent_cpu_debug_revision_rl_audit','integrity_passed':True,'test_fixture':True,
        'planned_study_complete':True,'planned_comparisons_available':True,'recorded_state':'complete',
        'selected_endpoint':endpoint,'genuine_optimizer_steps':int(step),'genuine_RL_optimizer_steps':int(step),
        'new_supervised_optimizer_steps':0,'tensor_reports':tensors,'training_groups':groups,'evaluation_cells':cells,
        'committed_trajectories':8*endpoint+32,'committed_raw_sampled_turns':16*endpoint+64,
        'raw_turns_pending_trajectory_materialization':0,
        'paired_yield':{'program_cluster_n':16,'paired_generation_seeds':seeds,
            'base_mean_yield':cells['base']['primary_mean_banked_point_yield'],
            'final_mean_yield':cells[labels[1]]['primary_mean_banked_point_yield'],
            'mean_yield_difference':math.fsum(delta)/16,'paired_yield_differences':delta,
            'bootstrap_draws':200,'bootstrap_seed':20261024,
            'descriptive_bootstrap_95_interval':plot.previous.bootstrap_interval(delta,200,20261024)}}
    schedule={'eval_generation_seeds':seeds[:]}
    generations={f'generations/eval-{label}-{i}.json':f'{label}-{i}-fixturehash' for label in labels for i in range(16)}
    checkpoints={label:label+'-fixturehash' for label in labels}
    status={'state':'complete','claims_allowed':True,'learning_comparison_available':True,'attempted_groups':endpoint,
        'actual_optimizer_steps':int(step),'final_checkpoint':labels[1],
        'artifacts':{**generations,**{f'checkpoints/{label}/metadata.json':h for label,h in checkpoints.items()},'endpoint-selection.json':'endpointfixture'}}
    seal={'draws':32,'sampled_turns':64,'scoring_allowed':True,'generations':generations,'checkpoints':checkpoints,
        'endpoint_selection_sha256':'endpointfixture'}
    return audit,status,schedule,seal


def write_fixture(directory):
    run=Path(directory);audit,status,schedule,seal=fixture()
    for relative in list(status['artifacts']):
        path=run/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps({'synthetic_fixture':relative}))
        status['artifacts'][relative]=plot.sha(path)
    seal.update(generations={p:status['artifacts'][p] for p in seal['generations']},
        checkpoints={label:status['artifacts'][f'checkpoints/{label}/metadata.json'] for label in seal['checkpoints']},
        endpoint_selection_sha256=status['artifacts']['endpoint-selection.json'])
    sealpath=run/'evaluation/sealed.json';sealpath.parent.mkdir();sealpath.write_text(json.dumps(seal))
    status['artifacts']['evaluation/sealed.json']=plot.sha(sealpath)
    (run/'schedule.json').write_text(json.dumps(schedule))
    source=run/'source'/plot.AUDITOR;source.parent.mkdir(parents=True);source.write_text('SYNTHETIC AUDITOR FIXTURE')
    manifest={'kind':'native_nonthinking_revision_rl','config':{'authoring_calls':2,'eval_samples':16,'group_size':8,
        'batch_size':2,'endpoint_choices':[10,15,25],'turn_native_modes':{
            'draft':{'thinking_enabled':True,'reasoning_effort':'low'},'revision':{'thinking_enabled':False,'reasoning_effort':None}}},
        'source_sha256':{plot.AUDITOR:plot.sha(source)},'input_sha256':{'schedule.json':plot.sha(run/'schedule.json')}}
    (run/'manifest.json').write_text(json.dumps(manifest));status['manifest_sha256']=plot.sha(run/'manifest.json')
    (run/'status.json').write_text(json.dumps(status))
    audit.update(run=str(run.resolve()),manifest_sha256=plot.sha(run/'manifest.json'),status_sha256=plot.sha(run/'status.json'),auditor_sha256=plot.sha(source))
    path=run/'audit.json';path.write_text(json.dumps(audit));return path


class RevisionRLPlotTests(unittest.TestCase):
    def test_exact_n16_categories_steps_and_reward_diversity(self):
        data=plot.final_data(*fixture());final,training=data['final'],data['training']
        self.assertEqual(final['program_cluster_n'],16);self.assertEqual(final['mean_difference'],.25)
        self.assertEqual((training['attempted_groups'],training['actual_optimizer_steps'],training['skipped_groups']),(10,7,3))
        self.assertEqual(training['diversity_counts'],{'no_reward_variation':3,'validity_contrast_only':3,'game_return_variation':4})
        self.assertEqual(final['cells']['base']['counts'],dict(zip(plot.KINDS,[4]*4)))
        self.assertEqual(final['cells']['base']['programs'][1]['banked_point_yield'],2.)
        self.assertEqual(final['cells']['base']['programs'][0]['games'],0)
        for endpoint in (15,25):self.assertEqual(plot.final_data(*fixture(endpoint))['final']['endpoint'],endpoint)

    def test_partial_and_missing_never_inspect_evaluation(self):
        for key in ('integrity_passed','planned_study_complete','planned_comparisons_available'):
            args=list(fixture());args[0][key]=False;args[0]['evaluation_cells']='must not be touched'
            with self.assertRaisesRegex(ValueError,'complete'):plot.final_data(*args)
            with self.assertRaisesRegex(ValueError,'complete'):plot.training_data(args[0])
        for field in ('committed_trajectories','committed_raw_sampled_turns'):
            args=list(fixture());args[0][field]-=1;args[0]['evaluation_cells']='must not be touched'
            with self.assertRaisesRegex(ValueError,'All 32'):plot.final_data(*args)
        args=list(fixture());args[3]['generations'].pop(next(iter(args[3]['generations'])))
        with self.assertRaisesRegex(ValueError,'sealed'):plot.final_data(*args)
        with tempfile.TemporaryDirectory() as tmp:
            path=write_fixture(tmp);audit=json.loads(path.read_text());audit['planned_study_complete']=False;path.write_text(json.dumps(audit))
            (Path(tmp)/'evaluation/sealed.json').unlink()
            with self.assertRaisesRegex(ValueError,'complete'):plot.bind_inputs(path)

    def test_paired_order_interval_and_invalid_denominator_are_verified(self):
        for change,match in [(lambda a:a[0]['evaluation_cells']['base']['programs'].reverse(),'identity/order'),
            (lambda a:a[0]['paired_yield']['descriptive_bootstrap_95_interval'].__setitem__(0,999),'bootstrap'),
            (lambda a:a[2]['eval_generation_seeds'].reverse(),'paired'),
            (lambda a:a[0]['evaluation_cells']['base']['programs'][0].__setitem__('games',48),'fabricated'),
            (lambda a:a[0]['evaluation_cells']['base']['programs'][2].__setitem__('banked_point_yield',None),'not zero')]:
            args=list(fixture());change(args)
            with self.subTest(match=match),self.assertRaisesRegex(ValueError,match):plot.final_data(*args)

    def test_zero_updates_are_honest_and_validity_is_not_gameplay_variation(self):
        data=plot.final_data(*fixture(zero=True));self.assertEqual(data['final']['actual_optimizer_steps'],0)
        self.assertEqual(data['training']['skipped_groups'],10)
        self.assertIn('not learned improvement',data['final']['interpretation'])
        self.assertEqual(data['final']['descriptive_bootstrap_95_interval'],[0.,0.])
        self.assertEqual(data['training']['groups'][0]['fault_free_game_return_levels'],0)
        args=list(fixture());args[0]['tensor_reports']['cycle3']['actual_optimizer_step']+=1
        with self.assertRaisesRegex(ValueError,'Adam steps'):plot.final_data(*args)
        args=list(fixture());args[0]['training_groups'][0]['advantages'][0]+=1.
        with self.assertRaisesRegex(ValueError,'advantages'):plot.final_data(*args)
        args=list(fixture());args[0]['training_groups'][0]['programs'][0]['reward']=0.
        with self.assertRaisesRegex(ValueError,'reward/program'):plot.final_data(*args)

    def test_all_audit_status_source_and_committed_bytes_bound(self):
        for relative in ('audit.json','status.json','evaluation/sealed.json','generations/eval-base-0.json','source/'+plot.AUDITOR):
            with self.subTest(relative=relative),tempfile.TemporaryDirectory() as tmp:
                path=write_fixture(tmp);*_,bindings=plot.bind_inputs(path);plot.previous.verify_bindings(bindings)
                target=Path(tmp)/relative;target.write_text(target.read_text()+'\n')
                with self.assertRaisesRegex(ValueError,'changed'):plot.previous.verify_bindings(bindings)
                if relative!='audit.json':
                    with self.assertRaises(ValueError):plot.bind_inputs(path)

    def test_public_render_refuses_synthetic_and_existing_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=write_fixture(tmp);prefix=Path(tmp)/'must-not-exist'
            with self.assertRaisesRegex(ValueError,'Synthetic'):plot.render(path,prefix)
            self.assertEqual(list(Path(tmp).glob('must-not-exist*')),[])
            values=list(plot.bind_inputs(path));values[0]['test_fixture']=False
            existing=Path(str(prefix)+'-final.png');existing.write_bytes(b'KEEP')
            with patch.object(plot,'bind_inputs',return_value=values),self.assertRaises(FileExistsError):plot.render(path,prefix)
            self.assertEqual(existing.read_bytes(),b'KEEP')

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'),'Isolated reporting runtime optional')
    def test_fixture_figures_labels_and_midrender_recheck(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        data=plot.final_data(*fixture())
        for kind,make in [('final',plot.final_figure),('training',plot.training_figure)]:
            fig=make(data[kind],plt,fixture=True)
            texts=' '.join(t.get_text() for t in fig.texts)
            self.assertIn('TEST FIXTURE',texts)
            self.assertNotIn('32 paired',texts)
            if kind=='final':self.assertIn('16 paired authoring trajectories',texts)
            else:self.assertIn('cannot establish held-out improvement',texts)
            plt.close(fig)
        with tempfile.TemporaryDirectory() as tmp:
            path=write_fixture(tmp);values=list(plot.bind_inputs(path));values[0]['test_fixture']=False
            original=plot.final_figure
            def mutate(data,plt):
                fig=original(data,plt,fixture=True)
                (Path(tmp)/'status.json').write_text('{}')
                return fig
            with patch.object(plot,'bind_inputs',return_value=values),patch.object(plot,'final_figure',side_effect=mutate),self.assertRaisesRegex(ValueError,'status changed'):
                plot.render(path,Path(tmp)/'rejected')
            self.assertEqual(list(Path(tmp).glob('rejected*')),[])
        with tempfile.TemporaryDirectory() as tmp:
            path=write_fixture(tmp);values=list(plot.bind_inputs(path));values[0]['test_fixture']=False
            prefix=Path(tmp)/'racing';existing=Path(str(prefix)+'-final.png')
            def race(data,plt):
                fig=original(data,plt,fixture=True);existing.write_bytes(b'KEEP RACING OUTPUT');return fig
            with patch.object(plot,'bind_inputs',return_value=values),patch.object(plot,'final_figure',side_effect=race),self.assertRaises(FileExistsError):
                plot.render(path,prefix)
            self.assertEqual(existing.read_bytes(),b'KEEP RACING OUTPUT')
            self.assertEqual(list(Path(tmp).glob('racing*')),[existing])


if __name__=='__main__':unittest.main()
