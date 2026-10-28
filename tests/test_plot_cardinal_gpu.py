"""Reporting fixtures only; none of these synthetic values are research results."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import plot_cardinal_gpu as plot


def cell(names, games, final=False):
    rows=[]
    for i,name in enumerate(names):
        authored=i%4!=0
        faults=int(i%4==1)
        value=float((i%5)+1) if authored and i%4==3 else 0.
        if final and value: value += 1. if i%2 else -1.
        rows.append({"generation":name,"banked_point_yield":value,"authorship_valid":authored,
            "runtime_valid":authored and not faults,"candidate_program_faults":faults,
            "games":games if authored else 0})
    return {"complete":True,"prescribed_program_clusters":len(names),"observed_program_clusters":len(names),
        "missing_program_clusters":0,"programs":rows,
        "primary_mean_banked_point_yield":sum(r['banked_point_yield'] for r in rows)/len(rows),
        "compiled_programs":sum(r['authorship_valid'] for r in rows),
        "invalid_no_game_programs":sum(not r['authorship_valid'] for r in rows),
        "fault_free_programs":sum(r['runtime_valid'] for r in rows),
        "played_games":sum(r['games'] for r in rows)}


def fixture(endpoint=50):
    groups=[];step=0;tensors={'base':{'actual_optimizer_step':0}}
    for n in range(1,endpoint+1):
        updated=n%3!=0
        step+=int(updated)
        tensors[f'cycle{n}']={'actual_optimizer_step':step}
        group=cell([f'train-cycle{n}-{i}' for i in range(8)],16)
        rewards=[-1.,-.5,0.,.5,-1.,0.,.5,1.] if updated else [0.]*8
        group.update(cycle=n,checkpoint_committed=True,
                     gradient_signal='gameplay_and_validity_signal' if updated else 'no_update',
                     rewards=rewards,mean_penalized_task_reward=sum(rewards)/8)
        groups.append(group)
    evaluation={label:cell([f'eval-{label}-{i}' for i in range(32)],48,label!='base')
                for label in ('base',f'cycle{endpoint}')}
    differences=[b['banked_point_yield']-a['banked_point_yield'] for a,b in
        zip(evaluation['base']['programs'],evaluation[f'cycle{endpoint}']['programs'])]
    seeds=list(range(10300000,10300032))
    return {'kind':'independent_cpu_cardinal_gpu_audit','integrity_passed':True,'test_fixture':True,
        'planned_study_complete':True,'planned_comparisons_available':True,'recorded_state':'complete',
        'selected_endpoint':endpoint,'genuine_optimizer_steps':step,'tensor_reports':tensors,
        'training_groups':groups,'evaluation_cells':evaluation,'committed_generation_draws':8+8*endpoint+64,
        'paired_yield':{'program_cluster_n':32,'paired_generation_seeds':seeds,
            'base_mean_yield':evaluation['base']['primary_mean_banked_point_yield'],
            'final_mean_yield':evaluation[f'cycle{endpoint}']['primary_mean_banked_point_yield'],
            'mean_yield_difference':sum(differences)/32,'paired_yield_differences':differences,
            'bootstrap_draws':100,'bootstrap_seed':20261024,
            'descriptive_bootstrap_95_interval':plot.bootstrap_interval(differences,100,20261024)}}


def write_fixture(directory, endpoint=50):
    run=Path(directory);run.mkdir(parents=True,exist_ok=True)
    schedule={'eval_generation_seeds':list(range(10300000,10300032))}
    (run/'schedule.json').write_text(json.dumps(schedule))
    manifest={'schedule_sha256':plot.sha(run/'schedule.json'),'source_sha256':{}}
    (run/'manifest.json').write_text(json.dumps(manifest))
    status={'state':'complete','claims_allowed':True,'manifest_sha256':plot.sha(run/'manifest.json')}
    (run/'status.json').write_text(json.dumps(status))
    data=fixture(endpoint)
    data.update(run=str(run.resolve()),manifest_sha256=plot.sha(run/'manifest.json'),status_sha256=plot.sha(run/'status.json'))
    (run/'audit.json').write_text(json.dumps(data,indent=2)+'\n')
    return run/'audit.json'


class CardinalGPUPlotTests(unittest.TestCase):
    def test_full_denominators_pairing_and_actual_step_counts(self):
        data=fixture();progress=plot.progress_data(data)
        self.assertEqual(progress['committed_groups'],50)
        self.assertEqual(progress['genuine_optimizer_steps'],34)
        self.assertFalse(progress['groups'][2]['update_committed'])
        result=plot.final_data(data,{'state':'complete','claims_allowed':True},
                               {'eval_generation_seeds':data['paired_yield']['paired_generation_seeds']})
        self.assertEqual(result['program_cluster_n'],32)
        self.assertEqual(result['cells']['base']['counts'],dict(zip(plot.KINDS,[8,8,8,8])))
        self.assertEqual(result['mean_difference'],.25)
        self.assertEqual(result['actual_optimizer_steps'],34)

    def test_final_rejects_partial_missing_and_false_completion(self):
        data=fixture();status={'state':'complete','claims_allowed':True}
        schedule={'eval_generation_seeds':data['paired_yield']['paired_generation_seeds']}
        for field in ('integrity_passed','planned_study_complete','planned_comparisons_available'):
            bad=deepcopy(data);bad[field]=False
            with self.assertRaisesRegex(ValueError,'complete independent'):
                plot.final_data(bad,status,schedule)
        for badstatus in ({'state':'running','claims_allowed':True},{'state':'complete','claims_allowed':False}):
            with self.assertRaises(ValueError):plot.final_data(data,badstatus,schedule)
        bad=deepcopy(data);bad['committed_generation_draws']-=1
        with self.assertRaisesRegex(ValueError,'64 evaluation'):plot.final_data(bad,status,schedule)
        bad=deepcopy(data);bad['evaluation_cells']['base']['programs'].pop()
        with self.assertRaisesRegex(ValueError,'identity/order'):plot.final_data(bad,status,schedule)

    def test_invalid_authorship_missing_values_and_faults_remain_distinct(self):
        summary=cell([f'x{i}' for i in range(8)],16)
        checked=plot.checked_cell(summary,[f'x{i}' for i in range(8)],16)
        self.assertEqual(checked['counts'],dict(zip(plot.KINDS,[2,2,2,2])))
        bad=deepcopy(summary);bad['programs'][0]['games']=16
        with self.assertRaisesRegex(ValueError,'fabricated'):plot.checked_cell(bad,[f'x{i}' for i in range(8)],16)
        bad=deepcopy(summary);bad['programs'][2]['banked_point_yield']=None
        with self.assertRaisesRegex(ValueError,'not zero'):plot.checked_cell(bad,[f'x{i}' for i in range(8)],16)
        bad=deepcopy(summary);bad['programs'][1]['runtime_valid']=True
        with self.assertRaisesRegex(ValueError,'faults'):plot.checked_cell(bad,[f'x{i}' for i in range(8)],16)

    def test_paired_order_differences_and_bootstrap_are_recomputed(self):
        data=fixture();status={'state':'complete','claims_allowed':True}
        schedule={'eval_generation_seeds':data['paired_yield']['paired_generation_seeds']}
        bad=deepcopy(data);bad['evaluation_cells']['base']['programs'].reverse()
        with self.assertRaisesRegex(ValueError,'identity/order'):plot.final_data(bad,status,schedule)
        bad=deepcopy(data);bad['paired_yield']['paired_yield_differences'][0]=1.
        with self.assertRaisesRegex(ValueError,'differences'):plot.final_data(bad,status,schedule)
        bad=deepcopy(data);bad['paired_yield']['descriptive_bootstrap_95_interval'][0]=999.
        with self.assertRaisesRegex(ValueError,'bootstrap'):plot.final_data(bad,status,schedule)
        bad=deepcopy(schedule);bad['eval_generation_seeds'].reverse()
        with self.assertRaisesRegex(ValueError,'seeds'):plot.final_data(data,status,bad)

    def test_training_progress_omits_uncommitted_without_inspecting_eval(self):
        data=fixture();data['planned_study_complete']=False
        data['evaluation_cells']='must never be inspected by progress plotting'
        data['training_groups']=data['training_groups'][:3]
        data['training_groups'][2]['checkpoint_committed']=False
        data['genuine_optimizer_steps']=2
        result=plot.progress_data(data)
        self.assertEqual(result['committed_groups'],2)
        self.assertEqual(result['scored_uncommitted_cycles'],[3])
        self.assertFalse(result['complete'])
        bad=deepcopy(data);bad['training_groups'][0]['checkpoint_committed']=False
        with self.assertRaisesRegex(ValueError,'gap'):plot.progress_data(bad)
        bad=deepcopy(data);bad['tensor_reports']['cycle2']['actual_optimizer_step']=4
        with self.assertRaisesRegex(ValueError,'lineage'):plot.progress_data(bad)

    def test_exact_audit_status_and_archive_hash_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            audit=write_fixture(directory)
            *_,bindings=plot.bind_inputs(audit)
            plot.verify_bindings(bindings)
            audit.write_text(audit.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'audit changed'):plot.verify_bindings(bindings)
            status=Path(directory)/'status.json';status.write_text('{"state":"running"}')
            with self.assertRaisesRegex(ValueError,'status binding'):plot.bind_inputs(audit)

    def test_no_observations_means_no_training_plot(self):
        data=fixture();data['training_groups']=[]
        with self.assertRaisesRegex(ValueError,'No committed'):plot.progress_data(data)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Optional isolated reporting runtime')
    def test_render_binds_outputs_refuses_overwrite_and_catches_midrender_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            audit=write_fixture(directory);prefix=Path(directory)/'figure.v1'
            outputs=plot.render(audit,prefix)
            self.assertNotEqual(outputs['progress']['.png'],outputs['final']['.png'])
            for kind in outputs:
                evidence=json.loads(Path(outputs[kind]['.json']).read_text())
                self.assertTrue(evidence['test_fixture'])
                for target,digest in evidence['outputs'].items():self.assertEqual(plot.sha(target),digest)
            with self.assertRaises(FileExistsError):plot.render(audit,prefix)
            original=plot.progress_figure
            def altered(*args,**kwargs):
                fig=original(*args,**kwargs)
                audit.write_text(audit.read_text()+'\n')
                return fig
            with patch.object(plot,'progress_figure',altered),self.assertRaisesRegex(ValueError,'audit changed'):
                plot.render(audit,Path(directory)/'changed',mode='progress')
            self.assertFalse(list(Path(directory).glob('changed-progress.*')))


if __name__=='__main__':unittest.main()
