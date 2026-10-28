"""Synthetic reporting fixtures only: no live evaluation, replay or model calls."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import plot_program_debug_revision_evaluation as plot
from tests.test_plot_program_debug_revision_rl import cell


def fixture(all_invalid=False):
    cells={label:cell([f'eval-{label}-{i}' for i in range(16)],48,final=label=='cycle3',all_invalid=all_invalid)
        for label in plot.LABELS}
    delta=[b['banked_point_yield']-a['banked_point_yield'] for a,b in zip(cells['base']['programs'],cells['cycle3']['programs'])]
    seeds=[241000000+2*i for i in range(16)];origin='parentmanifestfixture'
    audit={'kind':'independent_cpu_revision_evaluation_audit','integrity_passed':True,'planned_study_complete':True,
        'planned_comparisons_available':True,'recorded_state':'complete','parent_training_complete':False,
        'inherited_RL_optimizer_steps':3,'new_supervised_optimizer_steps':0,'new_RL_optimizer_steps':0,
        'fixed_final_checkpoint':'cycle3','committed_trajectories':32,'committed_raw_sampled_turns':64,
        'raw_turns_pending_trajectory_materialization':0,'test_fixture':True,'evaluation_cells':cells,'parent_manifest_sha256':origin,
        'tensor_reports':{label:{'actual_optimizer_step':step,'unchanged_import':True,'origin_manifest_sha256':origin}
            for label,step in zip(plot.LABELS,(0,3))},
        'paired_yield':{'program_cluster_n':16,'paired_generation_seeds':seeds,'paired_yield_differences':delta,
            'base_mean_yield':cells['base']['primary_mean_banked_point_yield'],'final_mean_yield':cells['cycle3']['primary_mean_banked_point_yield'],
            'mean_yield_difference':math.fsum(delta)/16,'bootstrap_draws':200,'bootstrap_seed':20261024,
            'descriptive_bootstrap_95_interval':plot.helper.bootstrap_interval(delta,200,20261024)}}
    generation={f'generations/eval-{label}-{i}.json':f'{label}-{i}-fixture' for label in plot.LABELS for i in range(16)}
    checkpoint={label:label+'checkpointfixture' for label in plot.LABELS};proof={label:label+'prooffixture' for label in plot.LABELS}
    status={'state':'complete','claims_allowed':True,'learning_comparison_available':True,'parent_training_complete':False,
        'new_optimizer_steps':0,'inherited_optimizer_steps':3,'final_checkpoint':'cycle3','artifacts':{**generation,
            **{f'checkpoints/{label}/metadata.json':v for label,v in checkpoint.items()},
            **{f'endpoint-state/{label}/proof.json':v for label,v in proof.items()}}}
    manifest={'kind':'mixed_native_revision_evaluation_only','phase':'coding_evaluation','parent_training_complete':False,
        'new_optimizer_steps':0,'parent_manifest_sha256':origin,'evaluation_plan_sha256':'planfixture',
        'config':{'model_repo':plot.MODEL,'model_revision':plot.REVISION,'authoring_calls':2,'eval_samples':16,'batch_size':2,
            'evaluation_only':True,'training_allowed':False,'new_optimizer_steps':0,'fixed_evaluation_checkpoints':['base','cycle3'],
            'turn_native_modes':{'draft':{'thinking_enabled':True,'reasoning_effort':'low'},'revision':{'thinking_enabled':False,'reasoning_effort':None}}}}
    seal={'draws':32,'sampled_turns':64,'scoring_allowed':True,'new_optimizer_steps':0,'generations':generation,
        'checkpoints':checkpoint,'endpoint_state_proofs':proof,'evaluation_plan_sha256':'planfixture'}
    return audit,manifest,status,{'eval_generation_seeds':seeds[:]},seal


def write_fixture(directory):
    run=Path(directory);audit,manifest,status,schedule,seal=fixture();inputs={}
    def write(relative,value,table):
        p=run/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));table[relative]=plot.sha(p)
    for relative in list(status['artifacts']):write(relative,{'synthetic_fixture':relative},status['artifacts'])
    for relative,key in [('parent/manifest.json','parent_manifest_sha256'),('parent/status.json','parent_status_sha256'),
        ('parent/final-audit.json','parent_audit_sha256')]:
        write(relative,{'synthetic_fixture':relative},inputs);manifest[key]=inputs[relative]
    audit['parent_manifest_sha256']=manifest['parent_manifest_sha256']
    for report in audit['tensor_reports'].values():report['origin_manifest_sha256']=manifest['parent_manifest_sha256']
    write('schedule.json',schedule,inputs);write('plan.json',{'synthetic_fixture':True},inputs)
    manifest['evaluation_plan_sha256']=inputs['plan.json'];manifest['inherited_checkpoints']={}
    for label,step in zip(plot.LABELS,(0,3)):
        manifest['inherited_checkpoints'][label]={'metadata_sha256':status['artifacts'][f'checkpoints/{label}/metadata.json'],
            'origin_manifest_sha256':manifest['parent_manifest_sha256'],'prior_SFT_steps':32,'prior_RL_steps':step,'optimizer_identity':{'step':step}}
    seal.update(generations={p:status['artifacts'][p] for p in seal['generations']},
        checkpoints={label:status['artifacts'][f'checkpoints/{label}/metadata.json'] for label in plot.LABELS},
        endpoint_state_proofs={label:status['artifacts'][f'endpoint-state/{label}/proof.json'] for label in plot.LABELS},
        evaluation_plan_sha256=manifest['evaluation_plan_sha256'])
    write('evaluation/sealed.json',seal,status['artifacts'])
    source=run/'source'/plot.AUDITOR;source.parent.mkdir(parents=True);source.write_text('SYNTHETIC AUDITOR')
    manifest.update(input_sha256=inputs,source_sha256={plot.AUDITOR:plot.sha(source)})
    (run/'manifest.json').write_text(json.dumps(manifest));status['manifest_sha256']=plot.sha(run/'manifest.json')
    (run/'status.json').write_text(json.dumps(status))
    audit.update(run=str(run.resolve()),manifest_sha256=plot.sha(run/'manifest.json'),status_sha256=plot.sha(run/'status.json'),auditor_sha256=plot.sha(source))
    path=run/'audit.json';path.write_text(json.dumps(audit));return path


class ClosureReportingTests(unittest.TestCase):
    def test_n16_full_denominator_fault_yield_and_scope(self):
        report=plot.data(*fixture())
        self.assertEqual(report['program_cluster_n'],16);self.assertEqual(report['mean_difference'],.25)
        self.assertEqual(report['counts']['base'],{'prescribed':16,'compiled':12,'fault_free':8,'productive':4,
            'invalid_authorship':4,'runtime_fault':4,'played_games':576})
        self.assertEqual(report['cells']['base']['programs'][1]['banked_point_yield'],2.)
        self.assertEqual(report['cells']['base']['programs'][0]['games'],0)
        self.assertEqual((report['inherited_RL_steps'],report['new_RL_steps'],report['new_SFT_steps']),(3,0,0))
        self.assertFalse(report['parent_training_complete']);self.assertIn('remains incomplete',report['scope'])

    def test_partial_and_missing_refused_before_holdout_read(self):
        for key,value in [('planned_study_complete',False),('planned_comparisons_available',False),('integrity_passed',False),
                          ('committed_trajectories',31),('committed_raw_sampled_turns',63)]:
            args=list(fixture());args[0][key]=value;args[0]['evaluation_cells']='MUST NOT READ'
            with self.assertRaises(ValueError):plot.data(*args)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'partial.json';p.write_text(json.dumps({'kind':'partial','run':'/nonexistent'}))
            with self.assertRaisesRegex(ValueError,'complete'):plot.bind_inputs(p)
        args=list(fixture());args[-1]['generations'].pop(next(iter(args[-1]['generations'])))
        with self.assertRaisesRegex(ValueError,'sealed'):plot.data(*args)

    def test_fixed_checkpoint_scope_and_state_proofs(self):
        changes=[lambda a:a[0].__setitem__('parent_training_complete',True),lambda a:a[0].__setitem__('new_RL_optimizer_steps',1),
            lambda a:a[0].__setitem__('fixed_final_checkpoint','cycle10'),lambda a:a[0]['tensor_reports']['cycle3'].__setitem__('actual_optimizer_step',10),
            lambda a:a[2].__setitem__('parent_training_complete',True),lambda a:a[4]['endpoint_state_proofs'].__setitem__('base','changed')]
        for change in changes:
            args=list(fixture());change(args)
            with self.assertRaises(ValueError):plot.data(*args)

    def test_paired_bootstrap_order_and_invalid_no_game_validation(self):
        changes=[lambda a:a[0]['evaluation_cells']['base']['programs'].reverse(),
            lambda a:a[0]['paired_yield']['descriptive_bootstrap_95_interval'].__setitem__(0,99),
            lambda a:a[3]['eval_generation_seeds'].reverse(),
            lambda a:a[0]['evaluation_cells']['base']['programs'][0].__setitem__('games',48),
            lambda a:a[0]['evaluation_cells']['base']['programs'][2].__setitem__('banked_point_yield',None)]
        for change in changes:
            args=list(fixture());change(args)
            with self.assertRaises(ValueError):plot.data(*args)
        report=plot.data(*fixture(all_invalid=True));self.assertEqual(report['descriptive_bootstrap_95_interval'],[0.,0.])
        self.assertEqual(report['counts']['base']['played_games'],0);self.assertEqual(report['counts']['base']['productive'],0)
        self.assertEqual(report['inherited_RL_steps'],3)  # Invalid eval does not erase actual prior updates.

    def test_exact_audit_status_parent_source_and_generation_bindings(self):
        for relative in ('audit.json','status.json','parent/manifest.json','source/'+plot.AUDITOR,'generations/eval-base-0.json'):
            with self.subTest(relative=relative),tempfile.TemporaryDirectory() as tmp:
                path=write_fixture(tmp);*_,bindings=plot.bind_inputs(path);plot.helper.verify_bindings(bindings)
                target=Path(tmp)/relative;target.write_text(target.read_text()+'\n')
                with self.assertRaisesRegex(ValueError,'changed'):plot.helper.verify_bindings(bindings)
                if relative!='audit.json':
                    with self.assertRaises(ValueError):plot.bind_inputs(path)

    def test_synthetic_publication_and_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=write_fixture(tmp);prefix=Path(tmp)/'output'
            with self.assertRaisesRegex(ValueError,'Synthetic'):plot.render(path,prefix)
            self.assertEqual(list(Path(tmp).glob('output*')),[])
            values=list(plot.bind_inputs(path));values[0]['test_fixture']=False
            target=Path(str(prefix)+'.png');target.write_bytes(b'KEEP')
            with patch.object(plot,'bind_inputs',return_value=values),self.assertRaises(FileExistsError):plot.render(path,prefix)
            self.assertEqual(target.read_bytes(),b'KEEP')

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'),'Isolated reporting dependencies optional')
    def test_figure_labels_and_render_race_or_evidence_mutation(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        report=plot.data(*fixture());fig=plot.make_figure(report,plt,fixture=True)
        texts=' '.join(t.get_text() for t in fig.texts)
        self.assertIn('SYNTHETIC TEST FIXTURE',texts);self.assertIn('three existing RL updates',texts)
        self.assertIn('zero new training updates',texts);self.assertIn('original longer run remains incomplete',texts)
        self.assertEqual(len(fig.axes),3);plt.close(fig)
        for scenario in ('mutation','race'):
            with tempfile.TemporaryDirectory() as tmp:
                path=write_fixture(tmp);values=list(plot.bind_inputs(path));values[0]['test_fixture']=False
                prefix=Path(tmp)/'output';target=Path(str(prefix)+'.png');original=plot.make_figure
                def changed(data,plt):
                    fig=original(data,plt,fixture=True)
                    if scenario=='mutation':(Path(tmp)/'status.json').write_text('{}')
                    else:target.write_bytes(b'KEEP RACE')
                    return fig
                with patch.object(plot,'bind_inputs',return_value=values),patch.object(plot,'make_figure',side_effect=changed),self.assertRaises((ValueError,FileExistsError)):
                    plot.render(path,prefix)
                if scenario=='mutation':self.assertEqual(list(Path(tmp).glob('output*')),[])
                else:self.assertEqual(target.read_bytes(),b'KEEP RACE')


if __name__=='__main__':unittest.main()
