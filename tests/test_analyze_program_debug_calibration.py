"""CPU-only calibration audit fixtures; synthetic samples are never evidence."""
import copy
import math
from pathlib import Path
import tempfile
import unittest
from scripts import analyze_program_debug_calibration as audit
from selfplay import program_debug_calibration as runner
from selfplay import program_debug_calibration_harness as harness
from tests.test_program_cardinal_gpu import deployment


def dep():
    value=deployment();value.update(maximum_hours=3,reservation_usd=25,estimate_usd=20,
        deadline_unix=int(value['deadline_unix']))
    return value


class Backend:
    def __init__(self):
        from transformers import AutoTokenizer
        self.tokenizer=AutoTokenizer.from_pretrained(str(runner.TOKENIZER_INPUT),local_files_only=True,trust_remote_code=False)
        self.calls=0
    def load_checkpoint(self,folder,hashes):
        import torch
        from safetensors.torch import load_file
        self.payload=torch.load(folder/'optimizer.pt',weights_only=True,map_location='cpu')
        self.parameters=load_file(str(folder/'adapter.safetensors'))
    def save_checkpoint(self,folder):
        import torch
        from safetensors.torch import save_file
        folder.mkdir(parents=True);save_file(self.parameters,str(folder/'adapter.safetensors'))
        torch.save(self.payload,folder/'optimizer.pt')
        return {n:audit.sha(folder/n) for n in ('adapter.safetensors','optimizer.pt')}
    def sample_messages_batch(self,contexts,seeds):
        self.calls+=1;records=[]
        for i,(messages,seed) in enumerate(zip(contexts,seeds)):
            text="r</think>def act():\n    return 'rest'" if len(messages)>1 and i==0 else 'r</think>invalid fixture'
            template=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,
                enable_thinking=True,reasoning_effort='low')
            prefix=self.tokenizer.encode(template,add_special_tokens=False)
            ids=self.tokenizer.encode(text,add_special_tokens=False)+[248046]
            records.append({'text':self.tokenizer.decode(ids,skip_special_tokens=True),'raw_text':self.tokenizer.decode(ids,skip_special_tokens=False),
                'messages':copy.deepcopy(messages),'seed':seed,'finish_reason':'eos','prompt_template':template,
                'prompt_ids':prefix,'completion_ids':ids,'sampled_token_logprobs':[-.1]*len(ids),'input_tokens':len(prefix),
                'output_tokens':len(ids),'eos_token_ids':[248044,248046],'sampling_batch_size':2,'sampling_batch_row':i,
                'thinking_enabled':True,'reasoning_effort':'low','completion_token_limit':4096,'calibration_only':True,'elapsed_seconds':.2,'fixture_only':True})
        width=max(r['input_tokens'] for r in records)
        for r in records:r['left_padding_tokens']=width-r['input_tokens']
        lengths=[r['output_tokens'] for r in records];counts=[sum(n>i for n in lengths) for i in range(max(lengths))]
        return {'records':records,'timing':{'batch_elapsed_seconds':.2,'total_output_tokens':sum(lengths),
            'constant_cache_batch_size':2,'active_sequences':2,'active_sequences_by_step':counts,'batch_steps':len(counts),'finished_eos':2}}
    def likelihood_gate(self,records):
        return {'passed':True,'records':[{'tokens':len(r['completion_ids']),'recorded_sequence_logprob':math.fsum(r['sampled_token_logprobs']),
            'max_token_error':0.,'sequence_error':0.,'passed':True} for r in records]}
    def update(self,*args):raise AssertionError('No calibration optimizer calls')


def prepared(folder):
    root=Path(folder);runner.common.atomic_json(root/'deployment.json',dep())
    run=root/'run';runner.prepare(run,root/'deployment.json',runner.ROOT/'runs/cardinal-gpu-preflight/model-info.json')
    return run


def initialized(run):
    backend=Backend();engine=runner.Calibration(run,backend)
    budget,_=runner.journal.execution_budget(engine.config);engine.status['execution_budget']=budget;engine.state()
    runtime=audit.read(runner.INITIAL/'runtime.json');runtime.update(thinking=True,authoring_calls=2,reasoning_effort='low',calibration_only=True)
    engine.commit('runtime.json',runtime);engine.initialize()
    contexts=[harness.initial_messages(),harness.initial_messages()+[{'role':'assistant','content':'fixture'},{'role':'user','content':'Check.'}]]
    records=backend.sample_messages_batch(contexts,[60000000,60000001])['records']
    for r in records:r['completion_token_limit']=32
    engine.commit('smoke-fixture.json',{'passed':True,'training_use':False,'optimizer_steps':0,'source_checkpoint':'base',
        'source_metadata_sha256':engine.status['artifacts']['checkpoints/base/metadata.json'],
        'adapter_unchanged':True,'optimizer_unchanged':True,'rng_restored':True,'config_restored':True,
        'optimizer_state_before_sha256':'fixture','optimizer_state_after_sha256':'fixture','batch':{'records':records},
        'eval_likelihood_gate':backend.likelihood_gate(records),'train_likelihood_gate':backend.likelihood_gate(records),
        'derivative':{'finite_nonzero':True,'gradient_l2_norm':.1,'missing_gradient_names':[]},
        'mixed_length_native_chat_prompts':True,'fixture_only':True})
    return engine,backend


class CalibrationAuditTests(unittest.TestCase):
    def test_schedule_context_and_denominator(self):
        self.assertEqual(audit.registered_schedule(),runner.schedule())
        cfg=runner.config(dep());self.assertEqual(cfg['iterations'],0);self.assertEqual(cfg['batch_size'],2)
        seeds=cfg['capability_samples'];self.assertEqual(seeds,8)
        with self.assertRaisesRegex(ValueError,'eight'):
            audit.cardinal.capability_decision([])
        prior={'warmstart':runner.common.load(runner.INITIAL/'map-fingerprints.json'),
            'debug':runner.common.load(runner.PRIOR_DEBUG/'map-fingerprints.json')}
        maps=runner.fingerprints(runner.schedule(),prior)
        self.assertEqual(sum(map(len,maps['families'].values())),22)

    def test_prepared_actual_identity_and_low_tokenization_tamper(self):
        with tempfile.TemporaryDirectory() as folder:
            run=prepared(folder);result=audit.audit(run)
            self.assertTrue(result['integrity_passed']);self.assertFalse(result['calibration_complete'])
            backend=Backend();row=backend.sample_messages_batch([harness.initial_messages()]*2,[1,2])['records'][0]
            audit.validate_tokenization(row,backend.tokenizer,[248044,248046])
            for change in [{'text':'changed'}, {'reasoning_effort':'xhigh'}, {'prompt_ids':[1]}, {'eos_token_ids':[1]}]:
                with self.assertRaises(ValueError):audit.validate_tokenization({**row,**change},backend.tokenizer,[248044,248046])
            path=run/'inputs/prior-debug-map-fingerprints.json';path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'Artifact'):audit.audit(run)

    def test_complete_no_update_audits_all_draws_and_rejects_missing_or_substituted(self):
        with tempfile.TemporaryDirectory() as folder:
            run=prepared(folder);engine,backend=initialized(run);engine.run_all()
            self.assertEqual(backend.calls,9) # unscored smoke fixture plus eight fixed batches
            result=audit.audit(run)
            self.assertTrue(result['integrity_passed']);self.assertTrue(result['calibration_complete'])
            self.assertFalse(result['qualification_passed']);self.assertFalse(result['learning_comparison_available'])
            self.assertEqual(result['committed_trajectories'],8);self.assertEqual(result['committed_raw_sampled_turns'],16)
            self.assertEqual(result['genuine_RL_optimizer_steps'],0)
            self.assertTrue(result['tensor_reports']['unchanged-final']['unchanged_adapter_adam_rng'])
            self.assertEqual(result['qualification']['summary']['prescribed_program_clusters'],8)
            resumed=runner.Calibration(run,backend);resumed.run_all();self.assertEqual(backend.calls,9)
            p=run/'generations/calibration-0.json';row=runner.common.load(p);row['text']='substituted final';runner.common.atomic_json(p,row)
            with self.assertRaisesRegex(ValueError,'Artifact'):audit.audit(run)


if __name__=='__main__':unittest.main()
