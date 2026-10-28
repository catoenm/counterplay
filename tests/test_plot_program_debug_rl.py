"""Pure reporting fixtures; no figures saved and no live study outputs read."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import plot_program_debug_rl as plot
from tests.test_plot_cardinal_gpu import fixture as gpu_fixture


def fixture(endpoint=25):
    data=gpu_fixture(endpoint)
    total=8*endpoint+64
    data.update(kind='independent_cpu_program_debug_rl_audit',new_supervised_optimizer_steps=0,
        genuine_RL_optimizer_steps=data['genuine_optimizer_steps'],committed_trajectories=total,
        committed_raw_sampled_turns=2*total,raw_turns_pending_trajectory_materialization=0)
    data.pop('committed_generation_draws')
    schedule={'eval_generation_seeds':data['paired_yield']['paired_generation_seeds']}
    labels=('base',f'cycle{endpoint}')
    generations={f'generations/eval-{label}-{i}.json':hashlib.sha256(f'{label}-{i}'.encode()).hexdigest() for label in labels for i in range(32)}
    checkpoints={label:hashlib.sha256(label.encode()).hexdigest() for label in labels}
    endpointsha=hashlib.sha256(b'endpoint').hexdigest()
    status={'state':'complete','claims_allowed':True,'attempted_groups':endpoint,
        'actual_optimizer_steps':data['genuine_RL_optimizer_steps'],
        'artifacts':{**generations,**{f'checkpoints/{label}/metadata.json':digest for label,digest in checkpoints.items()},
                     'endpoint-selection.json':endpointsha}}
    seal={'draws':64,'sampled_turns':128,'scoring_allowed':True,'generations':generations,'checkpoints':checkpoints,
          'endpoint_selection_sha256':endpointsha}
    return data,status,schedule,seal


def write_fixture(directory):
    run=Path(directory);data,status,schedule,seal=fixture()
    (run/'schedule.json').write_text(json.dumps(schedule))
    source=run/'source/scripts/analyze_program_debug_rl.py';source.parent.mkdir(parents=True);source.write_text('fixture auditor source')
    manifest={'kind':'thinking_debug_league_rl','config':{'thinking':True,'authoring_calls':2,'eval_samples':32,'group_size':8,'endpoint_choices':[25,50,100]},
        'source_sha256':{'scripts/analyze_program_debug_rl.py':plot.sha(source)},'input_sha256':{'schedule.json':plot.sha(run/'schedule.json')}}
    (run/'manifest.json').write_text(json.dumps(manifest))
    sealpath=run/'evaluation/sealed.json';sealpath.parent.mkdir();sealpath.write_text(json.dumps(seal))
    status['artifacts']['evaluation/sealed.json']=plot.sha(sealpath);status['manifest_sha256']=plot.sha(run/'manifest.json')
    (run/'status.json').write_text(json.dumps(status))
    data.update(run=str(run.resolve()),manifest_sha256=plot.sha(run/'manifest.json'),status_sha256=plot.sha(run/'status.json'),
                auditor_sha256=plot.sha(source))
    path=run/'audit.json';path.write_text(json.dumps(data));return path


class DebugPlotTests(unittest.TestCase):
    def test_complete_twentyfive_group_denominator_and_skips(self):
        audit,status,schedule,seal=fixture();data=plot.final_data(audit,status,schedule,seal)
        self.assertEqual(data['final']['program_cluster_n'],32)
        self.assertEqual(data['final']['actual_optimizer_steps'],17)
        self.assertEqual(data['final']['skipped_groups'],8)
        self.assertEqual(data['final']['cells']['base']['counts'],dict(zip(plot.previous.KINDS,[8,8,8,8])))
        self.assertEqual(data['final']['mean_difference'],.25)
        self.assertFalse(data['training']['groups'][2]['update_committed'])
        self.assertEqual(data['final']['authoring_calls_per_trajectory'],2)
        for n in (50,100):
            self.assertEqual(plot.final_data(*fixture(n))['final']['endpoint'],n)

    def test_partial_audit_or_unsealed_missing_turns_never_scores(self):
        original=fixture()
        for key in ('integrity_passed','planned_study_complete','planned_comparisons_available'):
            args=deepcopy(original);args[0][key]=False;args[0]['evaluation_cells']='must not be read'
            with self.assertRaisesRegex(ValueError,'complete'):plot.final_data(*args)
        for change in [{'committed_trajectories':263},{'committed_raw_sampled_turns':527},
                       {'raw_turns_pending_trajectory_materialization':1}]:
            args=deepcopy(original);args[0].update(change)
            with self.assertRaisesRegex(ValueError,'64 final'):plot.final_data(*args)
        args=deepcopy(original);args[3]['sampled_turns']=127
        with self.assertRaisesRegex(ValueError,'sealed'):plot.final_data(*args)
        args=deepcopy(original);args[3]['generations'].pop(next(iter(args[3]['generations'])))
        with self.assertRaisesRegex(ValueError,'sealed'):plot.final_data(*args)
        args=deepcopy(original);args[3]['endpoint_selection_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'endpoint binding'):plot.final_data(*args)

    def test_invalid_zero_valid_zero_runtime_fault_and_missing_are_distinct(self):
        original=fixture();data=plot.final_data(*original)['final']
        rows=data['cells']['base']['programs']
        self.assertEqual([r['category'] for r in rows[:4]],list(plot.previous.KINDS))
        self.assertEqual(rows[0]['games'],0);self.assertEqual(rows[2]['games'],48)
        args=deepcopy(original);args[0]['evaluation_cells']['base']['programs'][0]['games']=48
        with self.assertRaisesRegex(ValueError,'fabricated'):plot.final_data(*args)
        args=deepcopy(original);args[0]['evaluation_cells']['base']['programs'][2]['banked_point_yield']=None
        with self.assertRaisesRegex(ValueError,'not zero'):plot.final_data(*args)
        args=deepcopy(original);args[0]['evaluation_cells']['base']['programs'].pop()
        with self.assertRaisesRegex(ValueError,'identity/order'):plot.final_data(*args)

    def test_pair_order_bootstrap_actual_steps_and_no_new_sft(self):
        original=fixture()
        args=deepcopy(original);args[0]['evaluation_cells']['base']['programs'].reverse()
        with self.assertRaisesRegex(ValueError,'identity/order'):plot.final_data(*args)
        args=deepcopy(original);args[0]['paired_yield']['descriptive_bootstrap_95_interval'][0]=100.
        with self.assertRaisesRegex(ValueError,'bootstrap'):plot.final_data(*args)
        args=deepcopy(original);args[0]['tensor_reports']['cycle3']['actual_optimizer_step']=3
        with self.assertRaisesRegex(ValueError,'Skipped'):plot.final_data(*args)
        args=deepcopy(original);args[0]['new_supervised_optimizer_steps']=1
        with self.assertRaisesRegex(ValueError,'unexpected new SFT'):plot.final_data(*args)
        args=deepcopy(original);args[1]['actual_optimizer_steps']+=1
        with self.assertRaisesRegex(ValueError,'Terminal optimizer'):plot.final_data(*args)

    def test_exact_parsed_audit_status_seal_and_sources_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            path=write_fixture(directory);*_,bindings=plot.bind_inputs(path)
            plot.previous.verify_bindings(bindings)
            path.write_text(path.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'audit changed'):plot.previous.verify_bindings(bindings)
            *_,bindings=plot.bind_inputs(path)
            seal=Path(directory)/'evaluation/sealed.json';seal.write_text('{}')
            with self.assertRaisesRegex(ValueError,'seal changed'):plot.bind_inputs(path)
            with self.assertRaisesRegex(ValueError,'evaluation_seal changed'):plot.previous.verify_bindings(bindings)
        with tempfile.TemporaryDirectory() as directory:
            path=write_fixture(directory);(Path(directory)/'status.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'status binding'):plot.bind_inputs(path)
        with tempfile.TemporaryDirectory() as directory:
            path=write_fixture(directory);(Path(directory)/'source/scripts/analyze_program_debug_rl.py').write_text('changed')
            with self.assertRaisesRegex(ValueError,'archive changed'):plot.bind_inputs(path)

    def test_synthetic_figure_export_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path=write_fixture(directory);prefix=Path(directory)/'must-not-exist'
            with self.assertRaisesRegex(ValueError,'Synthetic fixtures'):plot.render(path,prefix)
            self.assertEqual(list(Path(directory).glob('must-not-exist*')),[])

    def test_final_labels_compare_the_same_harness_before_and_after_RL(self):
        class Text:
            def __init__(self,value):self.value=value
            def get_text(self):return self.value
            def set_text(self,value):self.value=value
        data=plot.final_data(*fixture())['final']
        texts=[Text('Cardinal controller RL: held-out program yield'),
            Text('Qwen3.8-27B · base versus cycle 25 · 17 actual optimizer steps · all 64 evaluation outputs'),
            Text('Mean over all 32 outputs: base 0 → final 1'),Text('Evidence is specific to this fixed opponent pool.')]
        legends=[Text('Fresh GPU base'),Text('Cycle 25')];labels=[]
        axis=SimpleNamespace(set_xlabel=labels.append,set_ylabel=labels.append,set_title=lambda *a,**k:None)
        fig=SimpleNamespace(texts=texts,legends=[SimpleNamespace(get_texts=lambda:legends)],axes=[axis,axis])
        with patch.object(plot.previous,'final_figure',return_value=fig):plot.final_figure(data,None)
        self.assertIn('Same two-call harness before/after RL',texts[1].value)
        self.assertIn('17 actual updates · 8 skipped groups',texts[1].value)
        self.assertEqual(legends[0].value,'Before RL · imported SFT weights')
        self.assertNotIn('Fresh GPU base',' '.join(t.value for t in texts+legends))
        self.assertIn('not a teacher-performance comparison',texts[3].value)
        self.assertIn('After RL − before RL banked points',labels)


if __name__=='__main__':unittest.main()
