"""Original tokenizer plus uniform CPU fixture policy; no pretrained weights."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest

import torch

from scripts.arena_mixed_action_audit import get_tokenizer, verify_action_rows
from selfplay.arena_mixed import cuda_policy, envs, runner
from tests.test_arena_mixed_review import spec

ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT/'runs/arena-mixed-preflight-v1'
WEIGHTS = {'market':.24,'maze':.24,'chess':.24,'planning':.24,'coding':.04}


class ActionAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory(); run=Path(cls.folder.name)
        (run/'inputs').mkdir()
        identities=json.loads((PRE/'model-file-sha256.json').read_text())
        cls.config=cuda_policy.default_config(WEIGHTS,model_index_sha256=identities['model.safetensors.index.json'],
            tokenizer_files_sha256=cuda_policy.digest({k:identities[k] for k in ('tokenizer.json','tokenizer_config.json')}))
        (run/'manifest.json').write_text(json.dumps({'config':{'policy':cls.config}}))
        (run/'inputs/model-file-sha256.json').write_text(json.dumps(identities))
        cls.tokenizer=get_tokenizer(run,PRE/'tokenizer')
        cls.labels,cls.label_ids=cuda_policy.select_labels(cls.tokenizer)
        cls.base={'checkpoint_id':'base','version':0,'adapter_sha256':'a'*64,
            'base_identity':{**{k:cls.config[k] for k in ('model','revision','model_index_sha256','tokenizer_files_sha256')},
                             'quantization':'nf4-double-fp32','frozen_state_sha256':'b'*64}}
        cls.journal=runner.Journal(run,'c'*64,time.time()+120)
        cls.requests=[spec('planning',955),spec('coding',955),spec('market',955,[0],'base')]
        runner.collect_cohort(cls.choose,cls.requests,cls.base,cls.journal,'fixtures',
                              opponent_decide=cls.choose,opponent_binding=cls.base)
        cls.episodes=runner.load_sealed_cohort(run/'fixtures')[1]

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    @classmethod
    def choose(cls, requests, binding):
        result=[]
        for request in requests:
            row=deepcopy(request);actions=row['actions']
            prompt=cuda_policy.action_prompt(row['observation'],actions,cls.labels)
            rendered=cls.tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,
                          add_generation_prompt=True,enable_thinking=False)
            ids=cls.tokenizer.encode(rendered,add_special_tokens=False)
            vector=torch.zeros(len(actions),dtype=torch.float32).log_softmax(0)
            probs=torch.softmax(vector.double(),0)
            pick=int(torch.multinomial(probs,1,generator=torch.Generator().manual_seed(row['sample_seed'])))
            row.update(prompt=prompt,rendered_prompt=rendered,input_ids=ids,
                input_ids_sha256=cuda_policy.digest(ids),prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                menu_sha256=cuda_policy.digest(actions),labels_sha256=cuda_policy.digest([cls.labels,cls.label_ids]),
                actor_config_sha256=cuda_policy.digest(cls.config),base_identity=deepcopy(binding['base_identity']),
                action_index=pick,action=deepcopy(actions[pick]),action_label=cls.labels[pick],
                action_token_id=cls.label_ids[pick],old_log_prob=float(vector[pick]),old_log_probs=vector.tolist(),
                entropy=float(-(probs*probs.log()).sum()),checkpoint=deepcopy(binding),
                policy_version=binding['version'],actor_snapshot_sha256=binding['adapter_sha256'],thinking=False)
            result.append(row)
        return result

    def verify(self, episodes):
        return verify_action_rows(episodes,self.config,self.tokenizer,opponent_binding=self.base)

    def test_original_native_prompts_all_roles_and_categorical_draws_verify(self):
        report=self.verify(self.episodes)
        self.assertTrue(report['verified'])
        self.assertEqual(report['actions'],sum(len(e['transitions'])+len(e['opponent_transitions']) for e in self.episodes))
        self.assertEqual(report['roles']['opponent'],16)
        self.assertLess(report['maximum_prompt_tokens'],4096)

    def test_prompt_token_label_and_native_mode_mutations_fail(self):
        for key,value in (('prompt','secret answer'),('rendered_prompt','<think>open'),
            ('input_ids',[1,2]),('labels_sha256','0'*64),('action_token_id',0),
            ('thinking',True),('actor_config_sha256','0'*64)):
            data=deepcopy(self.episodes);data[0]['transitions'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.verify(data)

    def test_invalid_vector_selected_logprob_entropy_and_seed_fail(self):
        first=self.episodes[0]['transitions'][0]
        for key,value in (('old_log_probs',[0.]*len(first['actions'])),('old_log_prob',42.),
            ('entropy',42.),('sample_seed',first['sample_seed']+1),('menu_seed',0)):
            data=deepcopy(self.episodes);data[0]['transitions'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.verify(data)

    def test_wrong_checkpoint_and_missing_real_opponent_coverage_fail(self):
        data=deepcopy(self.episodes);data[-1]['opponent_transitions'][0]['checkpoint']['adapter_sha256']='f'*64
        with self.assertRaises(ValueError):self.verify(data)
        data=deepcopy(self.episodes);data[-1]['opponent_transitions'].pop()
        with self.assertRaises(ValueError):self.verify(data)
        with self.assertRaisesRegex(ValueError,'Explicit'):
            verify_action_rows(self.episodes,self.config,self.tokenizer)

    def test_original_tokenizer_hash_tampering_rejected_before_loading(self):
        run=Path(self.folder.name)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            for name in ('config.json','tokenizer.json','tokenizer_config.json'):
                (folder/name).write_bytes((PRE/'tokenizer'/name).read_bytes())
            with (folder/'tokenizer_config.json').open('ab') as f:f.write(b' ')
            with self.assertRaisesRegex(ValueError,'bytes'):
                get_tokenizer(run,folder)


if __name__=='__main__':unittest.main()
