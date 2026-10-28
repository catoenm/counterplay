"""CPU-only independent checkpoint audit tests with real tiny safetensors."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from safetensors.numpy import load_file,save_file

from scripts import analyze_arena_v2 as audit
from selfplay import arena_v2 as v2,league


class V2AuditTests(unittest.TestCase):
    def fixture(self,root):
        root=Path(root);run=root/'v2';(run/'checkpoints').mkdir(parents=True)
        model={'model':'fake-model','revision':'a'*40}
        manifest={'model':model,'inherited_checkpoints':{},'policy_arithmetic':v2.ARITHMETIC,
                  'policy_arithmetic_sha256':v2.ARITHMETIC_SHA,'minibatch_size':4,'likelihood_tolerance':.002}
        stages={};previous=None
        for index,name in enumerate(('base','cycle1','cycle2','cycle3')):
            directory=(root/'parent' if index<2 else run)/'checkpoints'/name
            directory.mkdir(parents=True)
            arrays={'lora':np.full((2,2),index*.1,dtype=np.float32)}
            save_file(arrays,str(directory/'adapters.safetensors'))
            metadata={**model,'checkpoint':name,'labels':['A','B'],'label_token_ids':[1,2],
                      'trainable_parameters':4,'adapter_sha256':league.sha(directory/'adapters.safetensors')}
            if index>=2:
                metadata.update(policy_arithmetic=v2.ARITHMETIC,policy_arithmetic_sha256=v2.ARITHMETIC_SHA,
                    arithmetic_verification={'integer_weights_exactly_unchanged':True,'all_floating_parameters_float32':True})
            league.atomic_json(directory/'metadata.json',metadata)
            if index:
                source='base' if index==1 else f'cycle{index-1}'
                delta=float(np.sqrt(np.square(arrays['lora'].astype(float)-previous).sum()))
                league.atomic_json(directory/'summary.json',{'complete':True,'source':source,'target':name,
                    'parameter_delta_l2':delta,'checkpoint':metadata,'transitions':1,'episodes':1,'updates':1})
            record=league.checkpoint_record(directory,name,model)[0]
            if index<2:manifest['inherited_checkpoints'][name]=record
            else:
                episode={'episode_id':f'train-{index}','transitions':[{'turn':0,'player':0}]}
                league.atomic_rows(run/f'train-rollouts-{index}.jsonl',[episode])
                league.atomic_rows(directory/'metrics.jsonl',[{'update':1,'kl_to_rollout':0.,'entropy':.5,
                                                               'clip_fraction':0.,'gradient_norm':1.}])
                gate={'complete':True,'checkpoint':source,'batch_size':4,'expected_rows':1,
                      'rows':[{'episode_id':episode['episode_id'],'turn':0,'player':0,'full_vector_error':1e-5}],
                      'max_full_vector_error':1e-5}
                gate_path=run/f'likelihood-cycle{index}.json';league.atomic_json(gate_path,gate)
                stages[f'likelihood-cycle{index}']={'sha256':league.sha(gate_path)}
                stages[f'train-cycle{index}']={'checkpoint':record,
                    'rollout_sha256':league.sha(run/f'train-rollouts-{index}.jsonl')}
            previous=arrays['lora'].astype(float)
        league.atomic_json(run/'status.json',{'stages':stages})
        return run,manifest

    def test_audit_checks_deltas_updates_likelihoods_and_inherited_identity(self):
        with tempfile.TemporaryDirectory() as root:
            run,manifest=self.fixture(root)
            reports,metadata=audit.audit_checkpoints(run,manifest)
            self.assertEqual(set(reports),{'base','cycle1','cycle2','cycle3'})
            self.assertAlmostEqual(reports['cycle2']['independently_measured_delta_l2'],.2,places=6)
            self.assertEqual(reports['cycle3']['likelihood_gate']['rows'],1)
            self.assertTrue(reports['cycle1']['inherited'])

    def test_completed_checkpoint_metadata_rewrite_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            run,manifest=self.fixture(root)
            directory=run/'checkpoints/cycle2'
            metadata=json.loads((directory/'metadata.json').read_text());metadata['extra']='changed'
            league.atomic_json(directory/'metadata.json',metadata)
            with self.assertRaises(ValueError):audit.audit_checkpoints(run,manifest)

    def test_training_file_bytes_are_bound_even_when_gate_identities_still_match(self):
        with tempfile.TemporaryDirectory() as root:
            run,manifest=self.fixture(root)
            path=run/'train-rollouts-2.jsonl'
            episodes=league.complete_jsonl_rows(path)
            episodes[0]['transitions'][0]['extra']='changed without changing identity or row count'
            league.atomic_rows(path,episodes)
            with self.assertRaisesRegex(ValueError,'rollout bytes differ'):
                audit.audit_checkpoints(run,manifest)

    def test_gate_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            run,manifest=self.fixture(root)
            gate_path=run/'likelihood-cycle2.json';gate=json.loads(gate_path.read_text())
            gate['rows'][0]['full_vector_error']=.1;gate['max_full_vector_error']=.1
            league.atomic_json(gate_path,gate)
            with self.assertRaises(ValueError):audit.audit_checkpoints(run,manifest)

    def test_zero_measured_delta_is_rejected_despite_tiny_positive_reported_delta(self):
        with tempfile.TemporaryDirectory() as root:
            run,manifest=self.fixture(root)
            directory=run/'checkpoints/cycle2'
            prior=Path(manifest['inherited_checkpoints']['cycle1']['path'])/'adapters.safetensors'
            save_file(load_file(str(prior)),str(directory/'adapters.safetensors'))
            metadata=json.loads((directory/'metadata.json').read_text())
            metadata['adapter_sha256']=league.sha(directory/'adapters.safetensors')
            league.atomic_json(directory/'metadata.json',metadata)
            summary=json.loads((directory/'summary.json').read_text())
            summary.update(parameter_delta_l2=1e-10,checkpoint=metadata)
            league.atomic_json(directory/'summary.json',summary)
            status=json.loads((run/'status.json').read_text())
            status['stages']['train-cycle2']['checkpoint']=league.checkpoint_record(directory,'cycle2',manifest['model'])[0]
            league.atomic_json(run/'status.json',status)
            with self.assertRaisesRegex(ValueError,'nonzero measured parameter update'):
                audit.audit_checkpoints(run,manifest)


if __name__=='__main__':unittest.main()
