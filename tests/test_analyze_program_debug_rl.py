"""CPU-only audit fixtures. No generated fixture is scientific evidence."""
import copy
from datetime import datetime,timezone
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import analyze_program_debug_rl as audit
from selfplay import program_debug as harness
from selfplay import program_debug_rl as runner
from selfplay.program_debug_cuda import flatten_trajectories
from tests.test_program_cardinal_gpu import deployment


def configuration():
    dep=deployment();dep.update(maximum_hours=11.5,reservation_usd=75,deadline_unix=int(dep['deadline_unix']))
    return runner.config(dep)


def turn(seed,messages,text='reasoning\n</think>\ndef act():\n    return \'rest\'\n'):
    return {'text':text,'raw_text':text+'<eos>','messages':copy.deepcopy(messages),'seed':seed,
        'finish_reason':'eos','prompt_ids':[1,2,3],'completion_ids':[10,11,99],
        'sampled_token_logprobs':[-.4,-.7,-.9],'input_tokens':3,'output_tokens':3,'eos_token_ids':[99],
        'sampling_batch_size':8,'sampling_batch_row':0,'thinking_enabled':True,
        'left_padding_tokens':0,'elapsed_seconds':.2,'prompt_template':'fixture\n<think>\n'}


def trajectory(i=0,valid_draft=False):
    seed=161000100+2*i;initial=harness.initial_messages()
    draft=turn(seed,initial,text='truncated fixture reasoning' if not valid_draft else 'r</think>def act():\n    return \'rest\'')
    feedback=harness.practice_feedback(draft,[151000010,151000011])
    revision=harness.revision_messages(initial,draft,feedback)
    final=turn(seed+1,revision,text='r</think>invalid final fixture')
    row={**final,'name':f'train-cycle1-{i}','checkpoint':'base','source_metadata_sha256':'a'*64,
        'model_repo':configuration()['model_repo'],'model_revision':configuration()['model_revision'],
        'prompt_condition':'thinking_debug','authorship':'fresh_learner_two_call_trajectory','api_training_data':False,
        'trajectory_seed':seed,'turns':[draft,final],'turn_sha256':[audit.value_sha(draft),audit.value_sha(final)],
        'final_turn_index':1,'training_tokens':6,'practice_seeds':[151000010,151000011]}
    return row,feedback,revision


def checked(row,feedback,revision):
    return audit.validate_two_turn_trajectory(row,config=configuration(),name=row['name'],seed=row['trajectory_seed'],
        checkpoint='base',initial_messages=harness.initial_messages(),feedback=feedback,revision_messages=revision,
        practice_seeds=[151000010,151000011])


class DebugAuditTests(unittest.TestCase):
    def test_two_calls_all_tokens_feedback_context_and_final_alias(self):
        row,feedback,revision=trajectory(valid_draft=True)
        self.assertEqual(len(feedback['matches']),8)
        self.assertEqual(checked(row,feedback,revision),{'turns':2,'sampled_tokens':6,'tool_feedback_target_tokens':0})
        self.assertNotIn('r</think>',revision[1]['content'])
        bad=copy.deepcopy(row);bad['text']=bad['turns'][0]['text']
        with self.assertRaisesRegex(ValueError,'Final alias'):checked(bad,feedback,revision)
        bad=copy.deepcopy(row);bad['turns']=bad['turns'][1:]
        with self.assertRaisesRegex(ValueError,'Exactly two'):checked(bad,feedback,revision)
        bad=copy.deepcopy(row);bad['api_training_data']=True
        with self.assertRaisesRegex(ValueError,'Off-policy'):checked(bad,feedback,revision)

    def test_context_change_seed_and_omitted_reasoning_tokens_rejected(self):
        row,feedback,revision=trajectory()
        for mutate in [lambda r:r['turns'][1]['messages'][2].update(content='altered feedback'),
                       lambda r:r['turns'][1].update(seed=r['trajectory_seed']),
                       lambda r:r.update(training_tokens=3),
                       lambda r:r['turns'][0].update(sampled_token_logprobs=[-.4])]:
            bad=copy.deepcopy(row);mutate(bad)
            with self.assertRaises(ValueError):checked(bad,feedback,revision)

    def test_eos_length_context_and_padding_checks(self):
        row,_,_=trajectory();sample=row['turns'][0]
        audit.validate_turn(sample,configuration())
        for change in [{'finish_reason':'length'}, {'completion_ids':[99,10,99]}, {'thinking_enabled':False},
                       {'left_padding_tokens':-1}, {'input_tokens':2}, {'prompt_template':'not thinking'}]:
            bad={**sample,**change}
            with self.assertRaises(ValueError):audit.validate_turn(bad,configuration())
        cfg=configuration();cfg['max_context_tokens']=100
        with self.assertRaisesRegex(ValueError,'reserve'):audit.validate_turn(sample,cfg)

    def test_rloo_independent_trajectory_divisor_and_zero_signal(self):
        rows=[trajectory(i)[0] for i in range(8)];rewards=[-1.]*7+[.2];advantages=audit.previous.advantages(rewards)
        result=audit.validate_group(rows,rewards,advantages)
        self.assertFalse(result['zero_signal']);self.assertEqual(result['turn_count'],16)
        flat,weights=flatten_trajectories(rows,advantages)
        expected=math.fsum(-a*math.fsum(t['sampled_token_logprobs'])/8 for r,a in zip(rows,advantages) for t in r['turns'])
        actual=math.fsum(-a*math.fsum(t['sampled_token_logprobs'])/16 for t,a in zip(flat,weights))
        self.assertAlmostEqual(actual,expected)
        bad=copy.deepcopy(rows);bad[1]['turns'][0]['seed']=bad[0]['turns'][0]['seed']
        with self.assertRaisesRegex(ValueError,'collision'):audit.validate_group(bad,rewards,advantages)
        update={'objective_divisor':8,'training_tokens':48,'optimizer_steps':0}
        self.assertTrue(audit.validate_group(rows,[-1.]*8,[0.]*8,update)['zero_signal'])
        update['optimizer_steps']=1
        with self.assertRaisesRegex(ValueError,'Zero-signal'):audit.validate_group(rows,[-1.]*8,[0.]*8,update)

    def test_gate_covers_sixteen_turns_and_retains_failed_checks(self):
        rows=[trajectory(i)[0] for i in range(8)];turns=[t for r in rows for t in r['turns']]
        gate={'passed':True,'records':[{'tokens':3,'recorded_sequence_logprob':-2.,'max_token_error':0.,'sequence_error':0.,'passed':True} for _ in turns]}
        audit.validate_turn_gate(gate,turns,configuration(),True)
        bad=copy.deepcopy(gate);bad['records']=bad['records'][:8]
        with self.assertRaisesRegex(ValueError,'every sampled turn'):audit.validate_turn_gate(bad,turns,configuration())
        bad=copy.deepcopy(gate);bad['records'][0].update(max_token_error=.003,passed=False);bad['passed']=False
        audit.validate_turn_gate(bad,turns,configuration())
        with self.assertRaisesRegex(ValueError,'Failed likelihood'):audit.validate_turn_gate(bad,turns,configuration(),True)

    def test_paired_summary_invalid_zero_retained_missing_rejected(self):
        base=[{'generation':f'base-{i}','authorship_valid':False,'runtime_valid':False,'banked_point_yield':0.,'games':0} for i in range(32)]
        final=[{'generation':f'final-{i}','authorship_valid':True,'runtime_valid':i%2==0,'banked_point_yield':float(i),'games':48} for i in range(32)]
        summary=audit.paired_summary(base,final,list(range(32)),100)
        self.assertEqual(summary['program_cluster_n'],32);self.assertEqual(summary['mean_yield_difference'],15.5)
        with self.assertRaisesRegex(ValueError,'All 32'):audit.paired_summary(base[:-1],final,list(range(32)),100)
        bad=copy.deepcopy(base);bad[0]['games']=48
        with self.assertRaisesRegex(ValueError,'fabricated'):audit.paired_summary(bad,final,list(range(32)),100)

    def test_schedule_disjoint_sampling_turns_and_map_replacement(self):
        self.assertEqual(audit.registered_schedule(),runner.schedule())
        schedule=audit.registered_schedule();seeds=[s for values in schedule['train_generation_seeds'].values() for s in values]+schedule['eval_generation_seeds']
        expanded=[x for s in seeds for x in (s,s+1)]
        self.assertEqual(len(expanded),len(set(expanded)))
        self.assertEqual(schedule['train_maps']['22'],[159000000,152000221])

    @unittest.skipUnless((runner.INITIAL/'final-audit.json').exists(),'Real preserved parent evidence unavailable')
    def test_actual_prepared_run_audit_and_input_tamper(self):
        try:import torch
        except ImportError:self.skipTest('CPU Torch runtime required for exact inherited Adam inspection')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);dep=deployment();dep.update(maximum_hours=11.5,reservation_usd=75,deadline_unix=int(dep['deadline_unix']))
            runner.common.atomic_json(root/'deployment.json',dep)
            run=root/'run'
            runner.prepare(run,root/'deployment.json',runner.ROOT/'runs/cardinal-gpu-preflight/model-info.json')
            inspection=audit.Audit(run);result=inspection.execute()
            self.assertTrue(result['integrity_passed']);self.assertFalse(result['planned_comparisons_available'])
            tokenizer=inspection.tokenizer
            row=turn(1,harness.initial_messages())
            template=tokenizer.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
            ids=tokenizer.encode(row['text'],add_special_tokens=False)+[248046]
            row.update(prompt_template=template,prompt_ids=tokenizer.encode(template,add_special_tokens=False),completion_ids=ids,
                text=tokenizer.decode(ids,skip_special_tokens=True),raw_text=tokenizer.decode(ids,skip_special_tokens=False),eos_token_ids=inspection.eos)
            audit.validate_tokenization(row,tokenizer,inspection.eos)
            for change in [{'text':'substituted controller'}, {'prompt_ids':[1,2,3]}, {'raw_text':'dropped reasoning'}, {'eos_token_ids':[99]}]:
                with self.assertRaises(ValueError):audit.validate_tokenization({**row,**change},tokenizer,inspection.eos)
            self.assertEqual(result['genuine_RL_optimizer_steps'],0)
            path=run/'prompts/debug.txt';path.write_text(path.read_text()+'tampered')
            with self.assertRaisesRegex(ValueError,'Artifact differs'):audit.audit(run)


class TensorJournalIntegration(unittest.TestCase):
    @unittest.skipUnless((runner.INITIAL/'final-audit.json').exists(),'Preserved parent fixture absent')
    def test_actual_tensor_update_two_skips_and_all_raw_turns_audited(self):
        try:
            import torch
            from safetensors.torch import load_file,save_file
        except ImportError:self.skipTest('CPU Torch/safetensors required')
        from selfplay.program_cardinal_cuda import checkpoint_state
        class Backend:
            def __init__(self):
                from transformers import AutoTokenizer
                self.use_valid=True
                self.tokenizer=AutoTokenizer.from_pretrained(str(runner.TOKENIZER_INPUT),local_files_only=True,trust_remote_code=False)
            def load_checkpoint(self,folder,hashes):
                self.payload=torch.load(folder/'optimizer.pt',weights_only=True,map_location='cpu')
                values=load_file(str(folder/'adapter.safetensors'))
                self.parameters={r['name']:torch.nn.Parameter(values[r['name']].clone()) for r in self.payload['cardinal_state']['parameter_schema']}
                self.optimizer=torch.optim.AdamW(self.parameters.values(),lr=1e-5,weight_decay=0)
                self.optimizer.load_state_dict(self.payload['optimizer'])
            def save_checkpoint(self,folder):
                folder.mkdir(parents=True)
                save_file({n:p.detach().contiguous() for n,p in self.parameters.items()},str(folder/'adapter.safetensors'))
                self.payload['optimizer']=self.optimizer.state_dict();torch.save(self.payload,folder/'optimizer.pt')
                return {name:audit.sha(folder/name) for name in ('adapter.safetensors','optimizer.pt')}
            def sample_messages_batch(self,contexts,seeds):
                rows=[]
                for i,(messages,seed) in enumerate(zip(contexts,seeds)):
                    valid=len(messages)>1 and self.use_valid and i==0
                    text="r</think>def act():\n    return 'rest'" if valid else 'r</think>invalid fixture'
                    row=turn(seed,messages,text=text);row['sampling_batch_row']=i
                    template=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True)
                    ids=self.tokenizer.encode(text,add_special_tokens=False)+[248046]
                    prefix=self.tokenizer.encode(template,add_special_tokens=False)
                    row.update(prompt_template=template,prompt_ids=prefix,input_tokens=len(prefix),completion_ids=ids,
                        output_tokens=len(ids),sampled_token_logprobs=[-.1]*len(ids),eos_token_ids=[248044,248046],
                        text=self.tokenizer.decode(ids,skip_special_tokens=True),raw_text=self.tokenizer.decode(ids,skip_special_tokens=False))
                    rows.append(row)
                width=max(r['input_tokens'] for r in rows)
                for r in rows:r['left_padding_tokens']=width-r['input_tokens']
                lengths=[r['output_tokens'] for r in rows];counts=[sum(n>i for n in lengths) for i in range(max(lengths))]
                return {'records':rows,'timing':{'batch_elapsed_seconds':.2,'total_output_tokens':sum(lengths),
                    'constant_cache_batch_size':8,'active_sequences':8,'active_sequences_by_step':counts,'batch_steps':len(counts),'finished_eos':8}}
            def likelihood_gate(self,records):
                return {'passed':True,'records':[{'tokens':len(r['completion_ids']),'recorded_sequence_logprob':math.fsum(r['sampled_token_logprobs']),
                    'max_token_error':0.,'sequence_error':0.,'passed':True} for r in records]}
            def update_trajectories(self,records,advantages):
                from selfplay.program_cardinal_cuda import optimizer_identity
                schema=self.payload['cardinal_state']['parameter_schema']
                before=optimizer_identity(self.optimizer.state_dict(),schema,1e-5)
                old={n:p.detach().clone() for n,p in self.parameters.items()}
                self.optimizer.zero_grad();sum(p.square().mean() for p in self.parameters.values()).backward();self.optimizer.step()
                delta=math.sqrt(sum(float((p.detach().double()-old[n].double()).square().sum()) for n,p in self.parameters.items()))
                after=optimizer_identity(self.optimizer.state_dict(),schema,1e-5)
                flat=[t for r in records for t in r['turns']]
                losses=[-advantages[i//2]*math.fsum(t['sampled_token_logprobs'])/8 for i,t in enumerate(flat)]
                return {'optimizer_steps':1,'adapter_l2_delta':delta,'gradient_likelihood_gate':self.likelihood_gate(flat),
                    'loss':math.fsum(losses),'microbatch_losses':losses,'gradient_norm_before_clip':.1,
                    'trajectory_count':8,'sampled_turn_count':16,'tool_feedback_target_tokens':0,
                    'adam_step_before':before['step'],'adam_step_after':after['step'],
                    'optimizer_state_sha256':after['state_sha256'],'frozen_state_sha256':self.payload['cardinal_state']['frozen_state_sha256']}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);dep=deployment();dep.update(maximum_hours=11.5,reservation_usd=75,deadline_unix=int(dep['deadline_unix']))
            runner.common.atomic_json(root/'dep.json',dep);run=root/'run'
            runner.prepare(run,root/'dep.json',runner.ROOT/'runs/cardinal-gpu-preflight/model-info.json')
            backend=Backend();engine=runner.DebugCycle(run,backend)
            budget,_=runner.journal.execution_budget(engine.config);engine.status['execution_budget']=budget;engine.state()
            runtime=audit.read(runner.INITIAL/'runtime.json');runtime.update(thinking=True,authoring_calls=2)
            engine.commit('runtime.json',runtime);engine.initialize()
            contexts=[harness.initial_messages() for _ in range(8)]
            for i in range(1,8):contexts[i]+=[{'role':'assistant','content':'fixture'}, {'role':'user','content':'Check. '*i}]
            smoke_records=backend.sample_messages_batch(contexts,list(range(8)))['records']
            smoke={'passed':True,'training_use':False,'optimizer_steps':0,'source_checkpoint':'base',
                'source_metadata_sha256':engine.status['artifacts']['checkpoints/base/metadata.json'],
                'adapter_unchanged':True,'optimizer_unchanged':True,'rng_restored':True,'config_restored':True,
                'optimizer_state_before_sha256':'fixture','optimizer_state_after_sha256':'fixture',
                'batch':{'records':smoke_records},'eval_likelihood_gate':backend.likelihood_gate(smoke_records),
                'train_likelihood_gate':backend.likelihood_gate(smoke_records),
                'derivative':{'finite_nonzero':True,'gradient_l2_norm':.1,'missing_gradient_names':[]},
                'mixed_length_native_chat_prompts':True,'fixture_only':True}
            engine.commit('smoke-fixture.json',smoke)
            pool=engine.existing('opponents.json')
            engine.group(1,engine.roster(1,pool));backend.use_valid=False
            engine.group(2,engine.roster(2,pool));engine.group(3,engine.roster(3,pool))
            _,remaining=runner.journal.execution_budget(engine.config,engine.status['execution_budget'])
            decision=runner.endpoint_decision([engine.existing(f'groups/{i}/timing.json') for i in range(1,4)],remaining,engine.config)
            decision['profile_timing_sha256']={str(i):engine.status['artifacts'][f'groups/{i}/timing.json'] for i in range(1,4)}
            engine.commit('endpoint-selection.json',decision)
            result=audit.audit(run)
            self.assertTrue(result['integrity_passed']);self.assertFalse(result['planned_study_complete'])
            self.assertEqual(result['genuine_RL_optimizer_steps'],1)
            self.assertEqual(result['committed_trajectories'],24);self.assertEqual(result['committed_raw_sampled_turns'],48)
            self.assertEqual(result['tensor_reports']['cycle3']['actual_optimizer_step'],1)
            self.assertGreater(result['verified_unique_game_replays'],0)
            self.assertEqual(result['training_groups'][1]['gradient_signal'],'no_update')


if __name__=='__main__':unittest.main()
