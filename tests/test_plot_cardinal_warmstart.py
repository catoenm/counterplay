"""Reporting data fixtures only; no figures or purported research results saved."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import plot_cardinal_warmstart as plot
from tests.test_plot_cardinal_gpu import fixture as gpu_fixture


def fixture():
    data=gpu_fixture()
    data.update(kind='independent_cpu_cardinal_teacher_warmstart_audit',genuine_supervised_optimizer_steps=32,
        genuine_RL_optimizer_steps=data['genuine_optimizer_steps'],optimizer_reset_verified=True)
    data['supervised_updates']=[{'step':i,'mean_program_token_CE':1/i,'completion_tokens':100,
        'adapter_l2_delta':.01,'changed_parameters':10} for i in range(1,33)]
    data['tensor_reports'].update({f'sft_step{i}':{'actual_optimizer_step':i,'actual_learning_rate':5e-5} for i in range(33)})
    return data


def write_fixture(directory):
    root=Path(directory);data=fixture()
    schedule={'eval_generation_seeds':data['paired_yield']['paired_generation_seeds']}
    (root/'schedule.json').write_text(json.dumps(schedule))
    manifest={'kind':'cardinal_teacher_warmstart_league_rl','source_sha256':{},'schedule_sha256':plot.sha(root/'schedule.json')}
    (root/'manifest.json').write_text(json.dumps(manifest))
    (root/'status.json').write_text(json.dumps({'manifest_sha256':plot.sha(root/'manifest.json'),'state':'complete','claims_allowed':True}))
    data.update(run=str(root),manifest_sha256=plot.sha(root/'manifest.json'),status_sha256=plot.sha(root/'status.json'))
    path=root/'audit.json';path.write_text(json.dumps(data));return path


class WarmstartPlotTests(unittest.TestCase):
    def test_separate_supervised_and_RL_counts_full_invalid_denominators(self):
        data=fixture();progress=plot.progress_data(data)
        self.assertEqual(progress['supervised']['genuine_supervised_optimizer_steps'],32)
        self.assertEqual(progress['genuine_RL_optimizer_steps'],34)
        self.assertEqual(progress['RL']['committed_groups'],50)
        final=plot.final_data(data,{'state':'complete','claims_allowed':True},
            {'eval_generation_seeds':data['paired_yield']['paired_generation_seeds']})
        self.assertEqual(final['program_cluster_n'],32)
        self.assertEqual(final['cells']['base']['counts']['invalid_authorship'],8)
        self.assertEqual(final['mean_difference'],.25)
        self.assertEqual(final['supervised_optimizer_steps'],32)
        self.assertEqual(final['RL_optimizer_steps'],34)

    def test_partial_SFT_progress_does_not_inspect_evaluation_or_invent_RL(self):
        data=fixture();data.update(supervised_updates=data['supervised_updates'][:5],genuine_supervised_optimizer_steps=5,
            genuine_RL_optimizer_steps=0,genuine_optimizer_steps=0,optimizer_reset_verified=False,
            training_groups=[],planned_study_complete=False,evaluation_cells='unread partial data')
        progress=plot.progress_data(data)
        self.assertEqual(progress['supervised']['genuine_supervised_optimizer_steps'],5)
        self.assertIsNone(progress['RL'])
        self.assertFalse(progress['complete'])

    def test_final_rejects_missing_draws_unfinished_SFT_and_partial_audit(self):
        status={'state':'complete','claims_allowed':True};data=fixture()
        schedule={'eval_generation_seeds':data['paired_yield']['paired_generation_seeds']}
        for field in ('planned_study_complete','planned_comparisons_available','optimizer_reset_verified'):
            bad=deepcopy(data);bad[field]=False
            with self.assertRaises(ValueError):plot.final_data(bad,status,schedule)
        bad=deepcopy(data);bad['evaluation_cells']['base']['programs'].pop()
        with self.assertRaisesRegex(ValueError,'identity/order'):plot.final_data(bad,status,schedule)
        bad=deepcopy(data);bad['supervised_updates'].pop();bad['genuine_supervised_optimizer_steps']=31
        with self.assertRaisesRegex(ValueError,'32 committed'):plot.final_data(bad,status,schedule)

    def test_loss_and_actual_supervised_step_identity_required(self):
        data=fixture();data['supervised_updates'][0]['mean_program_token_CE']=float('nan')
        with self.assertRaisesRegex(ValueError,'loss'):plot.sft_data(data)
        data=fixture();data['tensor_reports']['sft_step7']['actual_optimizer_step']=8
        with self.assertRaisesRegex(ValueError,'lineage'):plot.sft_data(data)
        data=fixture();data['genuine_RL_optimizer_steps']=66
        with self.assertRaisesRegex(ValueError,'actual-step'):plot.progress_data(data)

    def test_exact_audit_and_status_bytes_are_bound_and_rechecked(self):
        with tempfile.TemporaryDirectory() as directory:
            path=write_fixture(directory)
            *_,bindings=plot.bind_inputs(path)
            plot.previous.verify_bindings(bindings)
            path.write_text(path.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'audit changed'):plot.previous.verify_bindings(bindings)
            (Path(directory)/'status.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'status identity'):plot.bind_inputs(path)

    def test_synthetic_figures_cannot_be_rendered(self):
        with tempfile.TemporaryDirectory() as directory:
            path=write_fixture(directory);prefix=Path(directory)/'do-not-create'
            with self.assertRaisesRegex(ValueError,'Synthetic fixtures'):plot.render(path,prefix)
            self.assertFalse(list(Path(directory).glob('do-not-create*')))

    def test_final_labels_always_name_SFT_baseline(self):
        class Text:
            def __init__(self,value):self.value=value
            def get_text(self):return self.value
            def set_text(self,value):self.value=value
        data=fixture();value=plot.final_data(data,{'state':'complete','claims_allowed':True},
            {'eval_generation_seeds':data['paired_yield']['paired_generation_seeds']})
        texts=[Text('Cardinal controller RL: held-out program yield'),
            Text('Qwen3.8-27B · base versus cycle 50 · 34 actual optimizer steps · all 64 evaluation outputs'),
            Text('Base: sample.   Final: sample.')]
        legends=[Text('Fresh GPU base'),Text('Cycle 50')];labels=[]
        fig=SimpleNamespace(texts=texts,legends=[SimpleNamespace(get_texts=lambda:legends)],
            axes=[None,SimpleNamespace(set_xlabel=labels.append)])
        with patch.object(plot.previous,'final_figure',return_value=fig):plot.final_figure(value,None)
        self.assertIn('32 supervised + 34 RL steps',texts[1].value)
        self.assertEqual(legends[0].value,'SFT baseline · 32 supervised steps')
        self.assertNotIn('Fresh GPU base',' '.join(t.value for t in texts+legends))
        self.assertEqual(labels,['RL final − SFT baseline banked points'])


if __name__=='__main__':unittest.main()
