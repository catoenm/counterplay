"""CPU-only revision-audit fixtures; synthetic samples are never run evidence."""
import copy
import unittest
from pathlib import Path
from scripts import analyze_program_debug_revision_calibration as audit

CONFIG={'max_new_tokens':4096,'max_context_tokens':16384,'token_logprob_tolerance':.002,'sequence_logprob_tolerance':.02}

class RevisionAuditUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tokenizer=AutoTokenizer.from_pretrained('runs/cardinal-debug-preflight-v1/tokenizer',local_files_only=True,trust_remote_code=False)
    def record(self,stage='draft',seed=1):
        thinking=stage=='draft';messages=[{'role':'user','content':'Write a controller.'}]
        prompt=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,
            enable_thinking=thinking,reasoning_effort='low' if thinking else None)
        text="r</think>def act():\n    return 'rest'" if thinking else "def act():\n    return 'rest'"
        ids=self.tokenizer.encode(text,add_special_tokens=False)+[248046]
        prefix=self.tokenizer.encode(prompt,add_special_tokens=False)
        return {'authoring_stage':stage,'thinking_enabled':thinking,'reasoning_effort':'low' if thinking else None,
            'reasoning_mode':'native_low' if thinking else 'native_disabled','calibration_only':True,
            'messages':messages,'prompt_template':prompt,'prompt_ids':prefix,'input_tokens':len(prefix),
            'completion_ids':ids,'output_tokens':len(ids),'sampled_token_logprobs':[-.2]*len(ids),
            'text':self.tokenizer.decode(ids,skip_special_tokens=True),'raw_text':self.tokenizer.decode(ids,skip_special_tokens=False),
            'seed':seed,'finish_reason':'eos','eos_token_ids':[248044,248046],'left_padding_tokens':0,'elapsed_seconds':.2}
    def gate(self,records):
        rows=[]
        for index,r in enumerate(records):
            lp=r['sampled_token_logprobs'];check=audit.vector_check(lp,lp,CONFIG)
            rows.append({'index':index,'record_sha256':audit.value_sha(r),'authoring_stage':r['authoring_stage'],
                'seed':r['seed'],'completion_tokens':len(lp),'sampled_token_logprobs':lp[:],
                'eval_token_logprobs':lp[:],'train_token_logprobs':lp[:],
                'eval_check':copy.deepcopy(check),'train_check':copy.deepcopy(check),
                'eval_all_finite':True,'train_all_finite':True,'eval_grad_enabled':False,'train_grad_enabled':True,'passed':True})
        return {'passed':True,'records':rows}
    def test_actual_native_stage_templates_and_flags(self):
        for stage in ('draft','revision'):
            record=self.record(stage)
            audit.validate_turn(record,CONFIG,stage=stage)
            audit.validate_tokenization(record,self.tokenizer,[248044,248046])
            for change in [{'thinking_enabled':stage!='draft'},{'authoring_stage':'other'},
                           {'reasoning_effort':'xhigh'},{'prompt_ids':[1]},{'text':'changed'}]:
                with self.assertRaises(ValueError):audit.validate_tokenization({**record,**change},self.tokenizer,[248044,248046])
        revision=self.record('revision')
        self.assertTrue(revision['prompt_template'].endswith('<think>\n\n</think>\n\n'))
        self.assertNotIn('</think>',revision['text'])
    def test_full16_vector_identity_coverage_and_statistics(self):
        records=[self.record(stage,i*2+j) for i in range(8) for j,stage in enumerate(('draft','revision'))]
        gate=self.gate(records);self.assertTrue(audit.validate_full_vectors(gate,records,CONFIG)['passed'])
        mutations=[lambda g:g['records'].pop(),lambda g:g['records'][0].update(seed=99),
            lambda g:g['records'][0].update(record_sha256='0'*64),
            lambda g:g['records'][0]['train_token_logprobs'].pop(),
            lambda g:g['records'][0]['eval_token_logprobs'].__setitem__(0,-.4),
            lambda g:g['records'][0]['train_check'].update(sequence_error=.01),
            lambda g:g['records'][0].update(train_all_finite=False)]
        for mutate in mutations:
            bad=copy.deepcopy(gate);mutate(bad)
            with self.assertRaises(ValueError):audit.validate_full_vectors(bad,records,CONFIG)
    def test_numerical_failure_remains_failure_without_relaxed_tolerance(self):
        records=[self.record('revision')];gate=self.gate(records)
        gate['records'][0]['train_token_logprobs'][0]-=.003
        gate['records'][0]['train_check']=audit.vector_check(records[0]['sampled_token_logprobs'],gate['records'][0]['train_token_logprobs'],CONFIG)
        gate['records'][0]['passed']=gate['passed']=False
        self.assertFalse(audit.validate_full_vectors(gate,records,CONFIG)['passed'])
        with self.assertRaises(ValueError):audit.vector_check([-.1],['nan'],CONFIG)
    def test_first_eos_excludes_any_continuation_from_scored_tokens(self):
        record=self.record('revision');record['completion_ids']+=[123]
        record['sampled_token_logprobs']+=[-.2];record['output_tokens']+=1
        with self.assertRaisesRegex(ValueError,'EOS'):audit.validate_turn(record,CONFIG)

    def test_failed_vector_diagnostics_are_preserved_without_passing(self):
        records=[self.record('revision')]
        for vector in (None,['nan']*len(records[0]['completion_ids'])):
            gate=self.gate(records);row=gate['records'][0]
            row['train_token_logprobs']=vector;row['train_all_finite']=False
            row['train_check']={'passed':False,'error_type':'ValueError','reason':'fixture numerical failure'}
            row['passed']=gate['passed']=False
            self.assertFalse(audit.validate_full_vectors(gate,records,CONFIG)['passed'])
            gate['passed']=True
            with self.assertRaises(ValueError):audit.validate_full_vectors(gate,records,CONFIG)

    def test_state_restoration_binds_actual_baseline_not_just_self_equal_hashes(self):
        base={'adapter_tensors':{'weight':{'sha256':'a'*64}},'state_sha256':'b'*64,
            'rng':{'cpu':{'sha256':'c'*64},'cuda':[]},'trainable_names':['weight'],'frozen_state_sha256':'d'*64}
        before={'adapter_sha256':audit.value_sha(base['adapter_tensors']),
            'optimizer_state_sha256':base['state_sha256'],'rng':base['rng'],
            'gradients_sha256':audit.value_sha({'weight':None}),'config_sha256':audit.value_sha(CONFIG),
            'model_training':False,'frozen_state_sha256':base['frozen_state_sha256']}
        flags={'adapter_unchanged':'adapter_sha256','optimizer_unchanged':'optimizer_state_sha256',
            'rng_restored':'rng','gradients_restored':'gradients_sha256','config_restored':'config_sha256',
            'model_mode_restored':'model_training','frozen_state_unchanged':'frozen_state_sha256'}
        report={'passed':True,'state_before':copy.deepcopy(before),'state_after':copy.deepcopy(before),
            **{k:True for k in flags}}
        self.assertTrue(audit.validate_restoration(report,base,CONFIG))
        for field in ('adapter_sha256','optimizer_state_sha256','config_sha256'):
            bad=copy.deepcopy(report);bad['state_before'][field]=bad['state_after'][field]='f'*64
            with self.assertRaises(ValueError):audit.validate_restoration(bad,base,CONFIG)
        bad=copy.deepcopy(report);bad['state_after']['optimizer_state_sha256']='f'*64
        with self.assertRaises(ValueError):audit.validate_restoration(bad,base,CONFIG)
        bad.update(passed=False,optimizer_unchanged=False)
        self.assertFalse(audit.validate_restoration(bad,base,CONFIG))




# Full fixture uses real pinned tokenizer/checkpoint files and archived game
# replays; only sampled neural values and derivative checks are synthetic.
from tests.test_analyze_program_debug_calibration import Backend as OriginalBackend, dep
from selfplay import program_debug_revision_calibration as runner
from selfplay import program_debug_revision_harness as harness
import tempfile
from unittest.mock import patch


class Backend(OriginalBackend):
    def __init__(self):
        super().__init__();self.gate_calls=0;self.fail_gate=False
    def load_checkpoint(self,folder,hashes):
        super().load_checkpoint(folder,hashes)
        self.identity=audit.read(folder/'metadata.json')['optimizer_identity']
    def state_identity(self):
        x=self.identity
        return {'adapter_sha256':audit.value_sha(x['adapter_tensors']),
            'optimizer_state_sha256':x['state_sha256'],'rng':copy.deepcopy(x['rng']),
            'gradients_sha256':audit.value_sha({n:None for n in x['trainable_names']}),
            'config_sha256':audit.value_sha(self.config),'model_training':False,
            'frozen_state_sha256':x['frozen_state_sha256']}
    def sample_messages_batch(self,contexts,seeds,authoring_stage='draft',stages=None,maximum=4096):
        self.calls+=1;stages=stages or [authoring_stage]*2;records=[]
        for i,(messages,seed,stage) in enumerate(zip(contexts,seeds,stages)):
            thinking=stage=='draft'
            code="def act():\n    return 'rest'" if thinking or i==0 else 'invalid fixture'
            if seed==201000007:code="def act():\n    v_bad = 1 // 0\n    return 'rest'"
            text=('r</think>' if thinking else '')+code
            prompt=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,
                enable_thinking=thinking,reasoning_effort='low' if thinking else None)
            prefix=self.tokenizer.encode(prompt,add_special_tokens=False)
            ids=self.tokenizer.encode(text,add_special_tokens=False)+[248046]
            records.append({'text':self.tokenizer.decode(ids,skip_special_tokens=True),'raw_text':self.tokenizer.decode(ids,skip_special_tokens=False),
                'messages':copy.deepcopy(messages),'seed':seed,'finish_reason':'eos','prompt_template':prompt,
                'prompt_ids':prefix,'completion_ids':ids,'sampled_token_logprobs':[-.1]*len(ids),
                'input_tokens':len(prefix),'output_tokens':len(ids),'eos_token_ids':[248044,248046],
                'sampling_batch_size':2,'sampling_batch_row':i,**runner.stage_metadata(stage),
                'completion_token_limit':maximum,'calibration_only':True,'elapsed_seconds':.2,'fixture_only':True})
        width=max(r['input_tokens'] for r in records);lengths=[r['output_tokens'] for r in records]
        counts=[sum(n>i for n in lengths) for i in range(max(lengths))]
        for r in records:
            r['left_padding_tokens']=width-r['input_tokens'];r['post_eos_cache_forwards']=len(counts)-r['output_tokens']
        return {'records':records,'timing':{'batch_elapsed_seconds':.2,'total_output_tokens':sum(lengths),
            'constant_cache_batch_size':2,'active_sequences':2,'active_sequences_by_step':counts,
            'batch_steps':len(counts),'finished_eos':2,'authoring_stages':stages}}
    def vectors(self,records):
        rows=[]
        for i,r in enumerate(records):
            lp=r['sampled_token_logprobs'];ev=lp[:];tr=lp[:]
            if self.fail_gate and i==0:tr[0]-=.1
            checks={mode:runner.common.check_likelihoods(lp,vector,.002,.02) for mode,vector in [('eval',ev),('train',tr)]}
            rows.append({'index':i,'record_sha256':audit.value_sha(r),'authoring_stage':r['authoring_stage'],
                'seed':r['seed'],'completion_tokens':len(lp),'sampled_token_logprobs':lp[:],
                'eval_token_logprobs':ev,'train_token_logprobs':tr,'eval_check':checks['eval'],'train_check':checks['train'],
                'eval_all_finite':True,'train_all_finite':True,'eval_grad_enabled':False,'train_grad_enabled':True,
                'passed':all(v['passed'] for v in checks.values())})
        before=self.state_identity()
        return {'passed':all(r['passed'] for r in rows),'records':rows,'state_before':before,'state_after':copy.deepcopy(before),
            **{k:True for k in ('adapter_unchanged','optimizer_unchanged','rng_restored','gradients_restored',
                              'config_restored','model_mode_restored','frozen_state_unchanged')},
            'training_use':False,'optimizer_steps':0,'token_tolerance':.002,'sequence_tolerance':.02}
    def full_completion_likelihood_gate(self,records):
        self.gate_calls+=1
        return {**self.vectors(records),'kind':'revision_full_completion_likelihood_admission','expected_records':16,
            'sampled_tokens':sum(len(r['completion_ids']) for r in records)}


def prepared(folder):
    root=Path(folder);runner.common.atomic_json(root/'deployment.json',dep())
    run=root/'run';runner.prepare(run,root/'deployment.json',runner.ROOT/'runs/cardinal-gpu-preflight/model-info.json')
    return run


def initialized(run):
    backend=Backend();engine=runner.Calibration(run,backend);backend.config=engine.config
    budget,_=runner.journal.execution_budget(engine.config);engine.status['execution_budget']=budget;engine.state()
    runtime=audit.read(runner.INITIAL/'runtime.json');runtime.update(thinking=True,authoring_calls=2,
        reasoning_effort='low',calibration_only=True,turn_native_modes=engine.config['turn_native_modes'])
    engine.commit('runtime.json',runtime);engine.initialize()
    contexts=[harness.initial_messages(),harness.initial_messages()+[{'role':'assistant','content':'fixture'},
              {'role':'user','content':'Check the fixture.'}]]
    seeds=[70000000+engine.config['seed']+i for i in range(2)]
    batch=backend.sample_messages_batch(contexts,seeds,stages=['draft','revision'],maximum=32)
    engine.commit('smoke-fixture.json',{**backend.vectors(batch['records']),
        'kind':'revision_mixed_native_derivative_admission','batch_size':2,'max_new_tokens':32,
        'authoring_stages':['draft','revision'],'seeds':seeds,'source_checkpoint':'base',
        'source_metadata_sha256':engine.status['artifacts']['checkpoints/base/metadata.json'],'batch':batch,
        'derivative':{'finite_nonzero':True,'gradient_l2_norm':.1,'missing_gradient_names':[]},'fixture_only':True})
    return engine,backend


def rewrite_artifact(engine,path,value):
    runner.common.atomic_json(engine.run/path,value);engine.status['artifacts'][path]=audit.sha(engine.run/path)
    runner.common.atomic_json(engine.run/'status.json',engine.status)


class RevisionAuditIntegrationTests(unittest.TestCase):
    def test_real_prepared_sources_baseline_and_disjoint_maps(self):
        with tempfile.TemporaryDirectory() as folder:
            run=prepared(folder);result=audit.audit(run)
            self.assertTrue(result['integrity_passed']);self.assertFalse(result['calibration_complete'])
            self.assertEqual(result['committed_raw_sampled_turns'],0)
            self.assertEqual(audit.registered_schedule(),runner.schedule())
            maps=audit.read(run/'map-fingerprints.json');self.assertEqual(sum(map(len,maps['families'].values())),22)
            path=run/'inputs/previous-calibration-map-fingerprints.json';path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'Artifact'):audit.audit(run)

    def test_complete_all16_full_gate_replays_and_unchanged_state(self):
        with tempfile.TemporaryDirectory() as folder:
            run=prepared(folder);engine,backend=initialized(run);engine.run_all()
            result=audit.audit(run)
            self.assertTrue(result['calibration_complete']);self.assertFalse(result['qualification_passed'])
            self.assertEqual(result['committed_trajectories'],8);self.assertEqual(result['committed_raw_sampled_turns'],16)
            self.assertEqual(result['full_likelihood_admission']['verified_turns'],16)
            self.assertGreater(result['full_likelihood_admission']['post_eos_cache_forwards'],0)
            self.assertEqual(result['genuine_RL_optimizer_steps'],0)
            self.assertTrue(result['tensor_reports']['unchanged-final']['unchanged_adapter_adam_rng'])
            scores=audit.read(run/'calibration/scores.json')['rows']
            self.assertEqual(len(scores),8);self.assertEqual(sum(r['authorship_valid'] for r in scores),5)
            self.assertEqual(sum(len(r['matches']) for r in scores),120)
            self.assertGreater(sum(r['candidate_program_faults'] for r in scores),0)
            self.assertTrue(all(not r['matches'] for r in scores if not r['authorship_valid']))
            resumed=runner.Calibration(run,backend);resumed.run_all()
            self.assertEqual(backend.calls,9);self.assertEqual(backend.gate_calls,1)
            gate=audit.read(run/'calibration/full-likelihood-gate.json');gate['records'][0]['train_token_logprobs'][0]-=.01
            rewrite_artifact(engine,'calibration/full-likelihood-gate.json',gate)
            with self.assertRaisesRegex(ValueError,'likelihood evidence'):audit.audit(run)

    def test_failed_and_interrupted_full_gate_preserved_with_no_qualification(self):
        for mode in ('failed','interrupted'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as folder:
                run=prepared(folder);engine,backend=initialized(run)
                if mode=='failed':backend.fail_gate=True;engine.run_all()
                else:
                    with patch.object(backend,'full_completion_likelihood_gate',side_effect=RuntimeError('fixture interrupted')):
                        with self.assertRaises(RuntimeError):engine.run_all()
                    runner.Calibration(run,backend).run_all()
                result=audit.audit(run)
                self.assertTrue(result['integrity_passed']);self.assertFalse(result['calibration_complete'])
                self.assertIsNone(result['qualification_passed']);self.assertFalse(result['full_likelihood_admission']['passed'])
                self.assertEqual(result['recorded_state'],'numerical_failed')
                self.assertNotIn('calibration/scores.json',audit.read(run/'status.json')['artifacts'])

if __name__=='__main__':unittest.main()
