"""Independent CPU-only audit of fixed native nonthinking-revision calibration; no model calls."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta
import importlib
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import analyze_program_debug_rl as debug
from scripts import analyze_program_cardinal_gpu as gpu
from scripts import analyze_program_cardinal_cycle as cardinal
from scripts import analyze_program_study as previous
read,sha,same,require,safe_path=previous.read,previous.sha,previous.same,previous.require,previous.safe_path
value_sha=previous.value_sha


def registered_schedule():
    return {'practice_maps':{str(i):[191000000+i*10+j for j in range(2)] for i in range(8)},
        'scoring_maps':list(range(192000000,192000006)),
        'generation_seeds':[201000000+i*2 for i in range(8)],
        'revision_seed_rule':'draft_seed+1; paired consecutive seeds, no retry'}


def validate_tokenization(record,tokenizer,eos):
    stage=record['authoring_stage']
    require(stage in ('draft','revision'),'Unknown authoring stage')
    thinking=stage=='draft'
    rendered=tokenizer.apply_chat_template(record['messages'],tokenize=False,add_generation_prompt=True,
        enable_thinking=thinking,reasoning_effort='low' if thinking else None)
    same(record['prompt_template'],rendered,'Native stage-specific template differs')
    require(record['thinking_enabled'] is thinking,'Native thinking flag differs')
    same(record['reasoning_effort'],'low' if thinking else None,'Reported reasoning effort differs')
    same(record['reasoning_mode'],'native_low' if thinking else 'native_disabled','Reported reasoning mode differs')
    if thinking:
        require(rendered.startswith('<|im_start|>system\nReasoning effort is set to low.'),
                'Native draft effort is not low')
    else:
        require(rendered.endswith('<think>\n\n</think>\n\n'), 'Native revision prefix is not closed in the prompt')
    same(record['prompt_ids'],tokenizer.encode(rendered,add_special_tokens=False),'Prompt tokens differ')
    same(record['text'],tokenizer.decode(record['completion_ids'],skip_special_tokens=True),'Scored output decode differs')
    same(record['raw_text'],tokenizer.decode(record['completion_ids'],skip_special_tokens=False),'Raw output decode differs')
    same(record['eos_token_ids'],eos,'Native EOS differs')
    require(record['calibration_only'] is True,'Calibration sampling mode differs')


def validate_turn(record,config,*,stage=None,seed=None,messages=None,maximum=None):
    """Validate the actual native flags; never relabel nonthinking as thinking."""
    stage=record.get('authoring_stage') if stage is None else stage
    require(stage in ('draft','revision') and record.get('authoring_stage')==stage,'Turn stage differs')
    thinking=stage=='draft';limit=config['max_new_tokens'] if maximum is None else maximum
    prefix,ids,lps=record.get('prompt_ids'),record.get('completion_ids'),record.get('sampled_token_logprobs')
    require(isinstance(prefix,list) and prefix and isinstance(ids,list) and ids and isinstance(lps,list),
            'Missing sampled token provenance')
    require(all(type(x) is int and x>=0 for x in prefix+ids),'Invalid sampled token IDs')
    require(len(ids)==len(lps)==record['output_tokens']<=limit and len(prefix)==record['input_tokens'],
            'Sampled token/logprob coverage differs')
    require(len(prefix)+limit<=config['max_context_tokens'],'Full completion reserve exceeds context')
    require(all(type(x) in (int,float) and math.isfinite(x) and x<=1e-6 for x in lps),'Invalid sampled logprob')
    require(record.get('thinking_enabled') is thinking and isinstance(record.get('messages'),list),
            'Native thinking context differs')
    template=record.get('prompt_template')
    require(isinstance(template,str) and (template.rstrip().endswith('<think>') if thinking
            else template.endswith('<think>\n\n</think>\n\n')),'Native generation prefix differs')
    require(type(record.get('left_padding_tokens')) is int and record['left_padding_tokens']>=0,'Invalid left padding')
    require(isinstance(record.get('text'),str) and isinstance(record.get('raw_text'),str),'Decoded output missing')
    eos=record.get('eos_token_ids');finish=record.get('finish_reason')
    require(isinstance(eos,list) and eos and all(type(x) is int and x>=0 for x in eos),'Invalid EOS identity')
    require(finish in ('eos','length','time') and not set(ids[:-1])&set(eos)
            and (ids[-1] in eos)==(finish=='eos'),'EOS/truncation boundary differs')
    if finish=='length':require(len(ids)==limit,'Length stop before token cap')
    require(type(record.get('elapsed_seconds')) in (int,float) and math.isfinite(record['elapsed_seconds'])
            and record['elapsed_seconds']>0,'Invalid sampling time')
    if seed is not None:same(record['seed'],seed,'Sample seed differs')
    if messages is not None:same(record['messages'],messages,'Conversation context differs')


def validate_trajectory(row,*,config,name,seed,checkpoint,initial_messages,feedback,revision_messages,practice_seeds):
    require(row['name']==name and row['trajectory_seed']==seed and row['checkpoint']==checkpoint,'Trajectory identity differs')
    require(row['authorship']=='fresh_learner_two_call_trajectory' and row['api_training_data'] is False,
            'Teacher output cannot be a fresh learner trajectory')
    same((row['model_repo'],row['model_revision']),(config['model_repo'],config['model_revision']),'Model identity differs')
    require(row['prompt_condition']=='native_nonthinking_revision_calibration' and row['final_turn_index']==1,
            'Final revision selection differs')
    turns=row['turns'];require(isinstance(turns,list) and len(turns)==2,'Exactly two sampled turns required')
    validate_turn(turns[0],config,stage='draft',seed=seed,messages=initial_messages)
    validate_turn(turns[1],config,stage='revision',seed=seed+1,messages=revision_messages)
    same(row['turn_sha256'],[value_sha(t) for t in turns],'Turn hashes differ')
    same(row['practice_seeds'],practice_seeds,'Practice map schedule differs')
    require(feedback['record']==turns[0] and feedback['record_sha256']==value_sha(turns[0]),'Feedback is not bound to draft')
    require(feedback['authoring_model_calls']==2 and feedback['final_attempt_index']==1
            and feedback['generated_python_executed'] is False,'Practice harness contract differs')
    require(row['training_tokens']==sum(len(t['completion_ids']) for t in turns),'Recorded completion coverage differs')
    for key,value in turns[1].items():same(row[key],value,'Mandatory revision alias differs: '+key)


def vector_check(sampled,recomputed,config):
    """Recompute every recorded likelihood statistic without invoking a model."""
    require(isinstance(sampled,list) and sampled and isinstance(recomputed,list)
            and len(sampled)==len(recomputed),'Full likelihood vector coverage differs')
    require(all(type(v) in (int,float) and math.isfinite(v) and v<=1e-6 for v in sampled+recomputed),
            'Nonfinite or invalid full likelihood vector')
    token=max(abs(a-b) for a,b in zip(sampled,recomputed))
    sequence=abs(math.fsum(sampled)-math.fsum(recomputed))
    return {'tokens':len(sampled),'max_token_error':token,'sequence_error':sequence,
            'recorded_sequence_logprob':math.fsum(sampled),
            'passed':token<=config['token_logprob_tolerance'] and sequence<=config['sequence_logprob_tolerance']}


def validate_full_vectors(gate,records,config):
    """The schema is fixed by the new backend, independently recomputed here."""
    require(len(gate['records'])==len(records),'Full likelihood gate dropped sampled turns')
    passed=[]
    for index,(check,record) in enumerate(zip(gate['records'],records)):
        require(check['index']==index and check['seed']==record['seed']
                and check['completion_tokens']==len(record['completion_ids']),'Full likelihood row order/coverage differs')
        same(check['record_sha256'],value_sha(record),'Full likelihood record identity differs')
        same(check['authoring_stage'],record['authoring_stage'],'Full likelihood turn stage differs')
        same(check['sampled_token_logprobs'],record['sampled_token_logprobs'],'Cached sampling likelihood changed')
        flags=[]
        for mode in ('eval','train'):
            vector,reported=check[mode+'_token_logprobs'],check[mode+'_check']
            if mode+'_grad_enabled' in check:
                require(check[mode+'_grad_enabled'] is (mode=='train'),'Likelihood replay gradient mode differs')
            else:
                require(vector is None and reported['passed'] is False and isinstance(reported.get('error_type'),str),
                        'Missing actual gradient mode without explicit pre-forward failure')
            finite=(isinstance(vector,list) and all(type(v) in (int,float) and math.isfinite(v) for v in vector))
            require(check[mode+'_all_finite'] is finite,'Full likelihood finite flag differs')
            valid=(finite and len(vector)==len(record['completion_ids']) and all(v<=1e-6 for v in vector))
            if valid:
                result=vector_check(record['sampled_token_logprobs'],vector,config)
                same(reported,result,'Recomputed '+mode+' likelihood evidence differs')
                flags.append(result['passed'])
            else:
                require(vector is None or (isinstance(vector,list) and all(
                    (type(v) in (int,float) and math.isfinite(v)) or v in ('nan','inf','-inf')
                    for v in vector)),'Unknown likelihood failure encoding')
                require(reported['passed'] is False and isinstance(reported.get('error_type'),str)
                        and isinstance(reported.get('reason'),str),'Unexplained full likelihood failure')
                flags.append(False)
        require(check['passed'] is all(flags),'Full likelihood row outcome differs')
        passed.append(all(flags))
    require(type(gate['passed']) is bool and (not gate['passed'] or all(passed)),
            'Full likelihood aggregate outcome differs')
    return {'turns':len(records),'tokens':sum(len(r['completion_ids']) for r in records),
            'passed':all(passed)}


def validate_restoration(report,base_identity,config):
    """Bind the before-state to actual checkpoint tensors, then compare after."""
    before=report['state_before']
    expected={'adapter_sha256':value_sha(base_identity['adapter_tensors']),
        'optimizer_state_sha256':base_identity['state_sha256'],'rng':base_identity['rng'],
        'gradients_sha256':value_sha({n:None for n in base_identity['trainable_names']}),
        'config_sha256':value_sha(config),'model_training':False,
        'frozen_state_sha256':base_identity['frozen_state_sha256']}
    same(before,expected,'Likelihood admission before-state differs from unchanged base')
    after=report.get('state_after')
    if after is None:
        require(report['passed'] is False and isinstance(report.get('restoration_error_type'),str),
                'Unexplained missing state restoration evidence')
        return False
    require(isinstance(after,dict) and after.keys()==before.keys(),'State restoration schema differs')
    flags={'adapter_unchanged':'adapter_sha256','optimizer_unchanged':'optimizer_state_sha256',
        'rng_restored':'rng','gradients_restored':'gradients_sha256','config_restored':'config_sha256',
        'model_mode_restored':'model_training','frozen_state_unchanged':'frozen_state_sha256'}
    for flag,key in flags.items():require(report[flag] is (before[key]==after[key]),'State restoration flag differs: '+flag)
    return all(report[flag] for flag in flags)


def validate_cache_timing(batch,stages):
    records,timing=batch['records'],batch['timing']
    require(len(records)==len(stages)==2,'Exactly two constant-cache rows required')
    require(type(timing['batch_elapsed_seconds']) in (int,float) and math.isfinite(timing['batch_elapsed_seconds'])
            and timing['batch_elapsed_seconds']>0,'Invalid batch wall time')
    lengths=[len(r['completion_ids']) for r in records]
    counts=[sum(n>i for n in lengths) for i in range(max(lengths))]
    same(timing['active_sequences_by_step'],counts,'Decode live-row coverage differs')
    same(timing['authoring_stages'],stages,'Batch native stage differs')
    require(timing['batch_steps']==len(counts) and timing['total_output_tokens']==sum(lengths)
            and timing['constant_cache_batch_size']==timing['active_sequences']==2
            and timing['finished_eos']==sum(r['finish_reason']=='eos' for r in records),'Batch timing/termination totals differ')
    width=max(len(r['prompt_ids']) for r in records)
    for i,row in enumerate(records):
        require(row['sampling_batch_row']==i and row['sampling_batch_size']==2
                and row['left_padding_tokens']==width-len(row['prompt_ids']),'Batch row/padding differs')
        expected=len(counts)-lengths[i] if row['finish_reason']=='eos' else 0
        require(type(row['post_eos_cache_forwards']) is int and row['post_eos_cache_forwards']==expected,
                'Post-EOS fixed-cache continuation count differs')


@contextmanager
def archived_runtime(run,manifest):
    with cardinal.archived_runtime(run,manifest) as runtime:
        prefix=runtime.game.__module__.split('.')[0]
        runtime.debug=importlib.import_module(prefix+'.program_debug_revision_harness')
        yield runtime


class Audit(debug.Audit):
    def frozen(self):
        m,cfg=self.manifest,self.config
        require(m['format_version']==1 and m['kind']=='native_nonthinking_revision_calibration','Unknown debug protocol')
        require(self.start_hashes['manifest.json']==self.status['manifest_sha256'],'Manifest differs')
        for p,digest in {**m['input_sha256'],**self.artifacts}.items():require(sha(safe_path(self.run,p))==digest,'Artifact differs: '+p)
        for p,digest in m['source_sha256'].items():require(sha(safe_path(self.run/'source',p))==digest,'Source differs: '+p)
        for p in ('scripts/analyze_program_study.py','scripts/analyze_program_cardinal_cycle.py','scripts/analyze_program_cardinal_gpu.py',
                  'scripts/cloud/train_program_rl.py','selfplay/program_cardinal_cuda.py','scripts/analyze_program_debug_rl.py',
                  'scripts/analyze_program_debug_revision_calibration.py'):
            require(m['source_sha256'].get(p)==sha(ROOT/p),'Executed CPU helper differs: '+p)
        require({'selfplay/program_debug.py','selfplay/program_debug_revision_calibration.py','selfplay/program_debug_revision_cuda.py',
                 'selfplay/program_debug_revision_harness.py'}<=set(m['source_sha256']),
                'Debug source closure incomplete')
        same(self.plan['config'],cfg,'Plan config differs');same(self.plan['schedule'],self.schedule,'Plan schedule differs')
        same(self.schedule,registered_schedule(),'Registered debug seed schedule differs')
        constants={'model_repo':'Qwen/Qwen3.8-27B','model_revision':'1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
            'backend':'cuda','loader':'image_text_to_text','condition':'debug-revision-calibration-v1','algorithm':'none_calibration_only',
            'group_size':8,'batch_size':2,'iterations':0,'endpoint_choices':[],'learning_rate':1e-5,
            'max_new_tokens':4096,'max_context_tokens':16384,'sample_seconds':900,'worker_seconds':9000,
            'max_rounds':48,'thinking':True,'reasoning_effort':'low','seed':325401,
            'turn_native_modes':{
                'draft':{'authoring_stage':'draft','thinking_enabled':True,'reasoning_effort':'low','reasoning_mode':'native_low'},
                'revision':{'authoring_stage':'revision','thinking_enabled':False,'reasoning_effort':None,'reasoning_mode':'native_disabled'}},
            'temperature':1.,'top_p':1.,'top_k':0,'compute_dtype':'float32','attention':'eager','lora_rank':8,'lora_alpha':16,
            'lora_dropout':0.,'lora_targets':['q_proj','v_proj'],'gradient_clip':.5,'token_logprob_tolerance':.002,
            'sequence_logprob_tolerance':.02,'eval_samples':0,'capability_samples':8,'minimum_productive_programs':4,
            'minimum_productive_distinct_ir':2,'minimum_positive_calibration_games':12,'opponent_attempt_cap':0,
            'profile_attempt_cap':0,'authoring_calls':2,'practice_maps_per_trajectory':2}
        for k,v in constants.items():same(cfg[k],v,'Registered setting differs: '+k)
        dep=read(self.run/'deployment.json')
        require(dep['reservation_usd']==25 and dep['maximum_hours']==3 and dep['maximum_gpu_rate_usd_hour']<=5.29,'New rental bounds differ')
        require(cfg['worker_seconds']==9000 and datetime.fromisoformat(cfg['deadline']).timestamp()==dep['deadline_unix'],
                'Rental time binding differs')
        budget=self.status.get('execution_budget')
        if budget:
            first,stop=datetime.fromisoformat(budget['first_started']),datetime.fromisoformat(budget['hard_stop'])
            require(stop==min(first+timedelta(seconds=cfg['worker_seconds']),datetime.fromisoformat(cfg['deadline']))
                and budget['maximum_seconds']==cfg['worker_seconds'] and budget['resume_counts_wall_time'] is True,'Persistent budget differs')
            if self.status['state']=='complete':require(datetime.fromisoformat(self.status['updated'])<=stop,'Completion after hard deadline')
        else:require(self.status['state']=='prepared','Execution budget missing')
        parent=read(self.run/'inputs/initial-manifest.json');status=read(self.run/'inputs/initial-status.json');audit=read(self.run/'inputs/initial-final-audit.json')
        require(audit['integrity_passed'] is True and audit['genuine_supervised_optimizer_steps']==32 and audit['genuine_RL_optimizer_steps']==0,
                'Inherited baseline lacks verified 32-SFT/zero-RL provenance')
        require(audit['manifest_sha256']==sha(self.run/'inputs/initial-manifest.json')==status['manifest_sha256']
                and audit['status_sha256']==sha(self.run/'inputs/initial-status.json') and status['state']=='capability_failed',
                'Inherited final audit binding differs')
        require(sha(self.run/'inputs/initial-map-fingerprints.json')==parent['map_fingerprints_sha256'], 'Inherited map binding differs')
        prior_debug=read(self.run/'inputs/prior-debug-manifest.json')
        require(sha(self.run/'inputs/prior-debug-map-fingerprints.json')==prior_debug['input_sha256']['map-fingerprints.json'], 'Prior debug map binding differs')
        preceding=read(self.run/'inputs/previous-calibration-manifest.json')
        preceding_status=read(self.run/'inputs/previous-calibration-status.json')
        preceding_audit=read(self.run/'inputs/previous-calibration-final-audit.json')
        require(preceding_audit['integrity_passed'] is True and preceding_audit['calibration_complete'] is True
                and preceding_audit['qualification_passed'] is False and preceding_status['qualification_passed'] is False
                and preceding_status['state']=='complete' and preceding_status['actual_optimizer_steps']==0
                and preceding_audit['new_supervised_optimizer_steps']==preceding_audit['genuine_RL_optimizer_steps']==0,
                'Previous calibration is not completed, unchanged, and below its gate')
        require(preceding_audit['manifest_sha256']==sha(self.run/'inputs/previous-calibration-manifest.json')==preceding_status['manifest_sha256']
                and preceding_audit['status_sha256']==sha(self.run/'inputs/previous-calibration-status.json'),
                'Previous calibration audit identity differs')
        require(sha(self.run/'inputs/previous-calibration-map-fingerprints.json')==preceding['input_sha256']['map-fingerprints.json'],
                'Previous calibration map identity differs')
        require(m['input_sha256']['prompts/debug.txt']==preceding['input_sha256']['prompts/debug.txt'],
                'Initial authoring instructions changed from prior calibration')
        require(self.plan['design_plan_sha256']==sha(self.run/'inputs/design-plan.json'), 'Design plan binding differs')
        design=read(self.run/'inputs/design-plan.json')
        for key,name in [('manifest_sha256','manifest.json'),('status_sha256','status.json'),
                         ('final_audit_sha256','final-audit.json'),('map_fingerprints_sha256','map-fingerprints.json')]:
            require(design['predecessor'][key]==sha(self.run/('inputs/previous-calibration-'+name)),
                    'Authoritative design predecessor binding differs: '+key)
        require(design['supersedes_numerical_requirements_of']['sha256']==sha(self.run/'inputs/design-plan-v1.json'),
                'Preserved design predecessor differs')
        for name in ('context-stress-review.json','context-stress-review-v2.json'):
            require(sha(self.run/'inputs'/name)==preceding['input_sha256']['inputs/'+name], 'Historical context receipt changed')
        stress=read(self.run/'inputs/revision-context-stress-review.json')
        for path in ('selfplay/program_debug_revision_harness.py','selfplay/program_debug_revision_cuda.py'):
            require(stress['source_sha256'][path]==m['source_sha256'][path], 'Final context-stress source differs')
        require(stress['kind']=='cpu_original_tokenizer_revision_context_stress' and stress['fixture_only'] is True
                and stress['pretrained_model_loaded'] is False and stress['network_calls']==0
                and stress['draft_native_reasoning_effort']=='low' and stress['revision_native_enable_thinking'] is False,
                'Context stress scope/native mode differs')
        require(stress['max_new_tokens']==4096 and stress['max_context_tokens']==16384 and stress['all_cases_fit'] is True
                and stress['max_revision_context_required']<=16384,'Context stress bound differs')
        for name,digest in stress['tokenizer_sha256'].items():
            require(digest==m['input_sha256']['inputs/tokenizer/'+name],'Stress tokenizer identity differs')
        metadata=read(self.run/'inputs/initial-checkpoint/metadata.json')
        require(sha(self.run/'inputs/initial-checkpoint/metadata.json')==status['artifacts']['checkpoints/base/metadata.json'],'Inherited checkpoint metadata differs')
        for filename,digest in metadata['files'].items():
            require(sha(self.run/'inputs/initial-checkpoint'/filename)==digest==status['artifacts']['checkpoints/base/'+filename], 'Inherited weights/Adam differ')
        same(m['initial_checkpoint_files'],metadata['files'],'Initial file declaration differs')
        module=importlib.import_module('selfplay.program_cardinal_cuda');self.cuda_inspector=module.checkpoint_state
        identity=self.cuda_inspector(self.run/'inputs/initial-checkpoint',expected_learning_rate=1e-5)
        same(identity,metadata['optimizer_identity'],'Inherited serialized optimizer differs');same(identity,m['initial_checkpoint_identity'],'Initial identity differs')
        same(identity,preceding['initial_checkpoint_identity'],'Baseline changed between calibration conditions')
        require(identity['step']==0 and not identity['moments'],'Inherited baseline is not zero-RL-step Adam')
        self.tokenizer,self.eos=debug.load_original_tokenizer(self.run,m)
        info=read(self.run/'inputs/model-info.json')
        require((info['id'],info['sha'])==(cfg['model_repo'],cfg['model_revision']),'Pinned model identity differs')
        return {family:([s for values in self.schedule[family].values() for s in values] if isinstance(self.schedule[family],dict)
                        else self.schedule[family]) for family in ('practice_maps','scoring_maps')}

    def maps(self,runtime,families):
        document=read(self.run/'map-fingerprints.json');prior={'warmstart':read(self.run/'inputs/initial-map-fingerprints.json'),
            'debug':read(self.run/'inputs/prior-debug-map-fingerprints.json'),
            'calibration':read(self.run/'inputs/previous-calibration-map-fingerprints.json')}
        def layout(seed):
            g=runtime.game(seed,48)
            return {'grid_size':g.size,'capacity':g.capacity,'max_energy':g.max_energy,'max_rounds':g.max_rounds,
                    'sites':g.sites,'homes':[[r['home_x'],r['home_y']] for r in g.robots]}
        expected={k:[{'seed':s,'layout':layout(s),'sha256':value_sha(layout(s))} for s in seeds] for k,seeds in families.items()}
        same(document['families'],expected,'Observable debug map differs');require(document['prior_evidence_sha256']==value_sha(prior),'Prior map chain differs')
        historical=[]
        def collect(v):
            if isinstance(v,dict):
                if {'seed','layout','sha256'}<=set(v):
                    require(value_sha(layout(v['seed']))==v['sha256'],'Historical map differs');historical.append(v['sha256'])
                for s,d in v.get('prior_seed_fingerprints',{}).items():
                    require(value_sha(layout(int(s)))==d,'Prior map fingerprint differs');historical.append(d)
                for child in v.values():collect(child)
            elif isinstance(v,list):
                for child in v:collect(child)
        collect(prior)
        previous.check_fingerprint_overlap({k:[r['sha256'] for r in rows] for k,rows in expected.items()},historical)

    def checkpoint(self,label):
        if label=='base':return super().checkpoint(label)
        if label in self.checkpoints:return self.checkpoints[label]
        require(label=='unchanged-final','Calibration cannot create training checkpoints')
        path=f'checkpoints/{label}/metadata.json';meta=self.artifact(path,False)
        if meta is None:return None
        base=self.checkpoint('base')
        require(base is not None and meta['label']==label and meta['parent']=='base' and meta['complete'] is True
            and meta['manifest_sha256']==self.status['manifest_sha256'], 'Final identity snapshot differs')
        same(meta['report'],{'optimizer_steps':0,'outcome':'calibration_no_training'},'Final snapshot report differs')
        for name,digest in meta['files'].items():
            key=f'checkpoints/{label}/{name}';self.allowed_artifacts.add(key)
            require(self.artifacts.get(key)==digest==sha(self.run/key),'Final snapshot file differs')
        identity=self.cuda_inspector(self.run/'checkpoints'/label,expected_learning_rate=1e-5)
        same(identity,meta['optimizer_identity'],'Final actual state differs')
        same({k:v for k,v in identity.items() if k!='files'},
             {k:v for k,v in base['optimizer_identity'].items() if k!='files'},'Calibration changed model/Adam/RNG state')
        require(meta['files']['adapter.safetensors']==base['files']['adapter.safetensors'],'Calibration changed adapter bytes')
        self.tensor_reports[label]={'actual_optimizer_step':0,'actual_learning_rate':1e-5,'unchanged_adapter_adam_rng':True}
        self.checkpoints[label]=meta
        return meta

    def initial_pool(self,runtime):
        expected={'selected':[{'id':n,'origin':'scripted','authorship':'scripted_anchor',
            'compiled_ir':runtime.reference_programs()[n]} for n in ('nearest','value')],'frozen':True}
        same(read(self.run/'initial-opponents.json'),expected,'Calibration opponents differ')
        self.pool=self.artifact('opponents.json',False)
        if self.pool is not None:same(self.pool,expected,'Scoring opponent snapshot differs')
        return self.pool

    def batch_specs(self):
        for offset in range(0,8,2):
            yield f'calibration-{offset//2}',[f'calibration-{i}' for i in range(offset,offset+2)],                self.schedule['generation_seeds'][offset:offset+2],'base',                [self.schedule['practice_maps'][str(i)] for i in range(offset,offset+2)]

    def all_batches(self):
        initial=self.runtime.debug.initial_messages()
        for batchid,names,seeds,label,maps in self.batch_specs():
            for turn in ('draft','revision'):
                path=f'batches/{batchid}-{turn}.json';intentpath=f'batches/{batchid}-{turn}-intent.json'
                batch=self.artifact(path,False);intent=self.artifact(intentpath,False)
                if batch is None and intent is None:continue
                require(self.checkpoint(label) is not None,'Sampling before committed source checkpoint')
                if batchid.startswith('eval-'):require(self.checkpoint(f'cycle{self.endpoint}') is not None,'Evaluation before fixed endpoint')
                if turn=='draft':contexts=[initial]*2;turnseeds=seeds
                else:
                    draftpath=f'batches/{batchid}-draft.json';drafts=self.artifact(draftpath)
                    contexts=[]
                    for i,draft in enumerate(drafts['records']):
                        feedback=self.feedback(f'feedback/{names[i]}.json',draft,maps[i],draftpath,i)
                        contexts.append(self.runtime.debug.revision_messages(initial,draft,feedback))
                    turnseeds=[s+1 for s in seeds]
                identity={'checkpoint':label,'source_metadata_sha256':self.artifacts[f'checkpoints/{label}/metadata.json'],
                          'contexts_sha256':value_sha(contexts),'seeds':turnseeds,'authoring_stage':turn}
                require(intent is not None,'Sampled batch lacks durable no-redraw intent')
                require(isinstance(intent['started'],str),'Intent timestamp missing')
                for key,value in identity.items():same(intent[key],value,'Intent differs: '+key)
                if batch is None:continue
                for key,value in identity.items():same(batch[key],value,'Raw batch identity differs: '+key)
                require(len(batch['records'])==2,'Partial committed sampling batch')
                width=max(len(row['prompt_ids']) for row in batch['records'])
                for i,row in enumerate(batch['records']):
                    validate_turn(row,self.config,stage=turn,seed=turnseeds[i],messages=contexts[i])
                    validate_tokenization(row,self.tokenizer,self.eos)
                    self.actual_context(path+'/row-'+str(i),row)
                    require(row['completion_token_limit']==4096,'Full calibration token cap differs')
                    require(row['sampling_batch_row']==i and row['sampling_batch_size']==2
                        and row['left_padding_tokens']==width-len(row['prompt_ids']),'Batch row/padding differs')
                    self.raw_turns[(names[i],turn)]=path
                timing=batch['timing']
                require(type(timing['batch_elapsed_seconds']) in (float,int) and math.isfinite(timing['batch_elapsed_seconds'])
                        and timing['batch_elapsed_seconds']>0,'Invalid batch wall time')
                require(timing['total_output_tokens']==sum(len(r['completion_ids']) for r in batch['records'])
                        and timing['constant_cache_batch_size']==2 and timing['active_sequences']==2,'Batch usage differs')
                counts=timing['active_sequences_by_step'];lengths=[len(r['completion_ids']) for r in batch['records']]
                same(counts,[sum(length>i for length in lengths) for i in range(max(lengths))],'Decode live-row coverage differs')
                require(timing['batch_steps']==len(counts) and timing['finished_eos']==sum(r['finish_reason']=='eos' for r in batch['records']),
                        'Decode termination totals differ')
                same(timing['authoring_stages'],[turn,turn],'Batch native stage differs')
                for row in batch['records']:
                    expected=len(counts)-len(row['completion_ids']) if row['finish_reason']=='eos' else 0
                    require(type(row['post_eos_cache_forwards']) is int and row['post_eos_cache_forwards']==expected,
                            'Post-EOS fixed-cache continuation count differs')
        # A crash may leave complete draft feedback before the revision batch intent.
        expected={name:(batchid,i,maps[i]) for batchid,names,seeds,label,maps in self.batch_specs() for i,name in enumerate(names)}
        for path in self.artifacts:
            if path.startswith('feedback/'):
                name=Path(path).stem;require(name in expected,'Unregistered practice feedback')
                batchid,index,seeds=expected[name];draftpath=f'batches/{batchid}-draft.json'
                draft=self.artifact(draftpath)['records'][index]
                self.feedback(path,draft,seeds,draftpath,index)

    def generation(self,name,seed,label,condition='native_nonthinking_revision_calibration'):
        if name in self.generations:return self.generations[name]
        record=self.artifact(f'generations/{name}.json');meta=self.checkpoint(label)
        require(meta is not None and record['source_metadata_sha256']==self.artifacts[f'checkpoints/{label}/metadata.json'], 'Trajectory source differs')
        index=record['batch_index'];require(type(index) is int and 0<=index<2,'Invalid trajectory batch index')
        drafts=self.artifact(record['draft_batch_path']);revisions=self.artifact(record['revision_batch_path'])
        require(record['draft_batch_sha256']==self.artifacts[record['draft_batch_path']] and
                record['revision_batch_sha256']==self.artifacts[record['revision_batch_path']],'Trajectory batch hash differs')
        same(record['turns'],[drafts['records'][index],revisions['records'][index]],'Trajectory substituted sampled output')
        require(record['feedback_path']==f'feedback/{name}.json' and record['feedback_sha256']==self.artifacts[record['feedback_path']],
                'Trajectory feedback hash differs')
        seeds=self.schedule['practice_maps'][name.rsplit('-',1)[1]]
        feedback=self.feedback(record['feedback_path'],record['turns'][0],seeds,record['draft_batch_path'],index)
        initial=self.runtime.debug.initial_messages();revision=self.runtime.debug.revision_messages(initial,record['turns'][0],feedback)
        require(record['prompt_condition']=='native_nonthinking_revision_calibration','Wrong calibration condition')
        validate_trajectory(record,config=self.config,name=name,seed=seed,checkpoint=label,
            initial_messages=initial,feedback=feedback,revision_messages=revision,practice_seeds=seeds)
        require(record['history_transform']==self.config['history_transform'],'History transform differs')
        self.generations[name]=record
        return record

    def all_generations(self,runtime):
        for batchid,names,seeds,label,maps in self.batch_specs():
            for name,seed in zip(names,seeds):
                if f'generations/{name}.json' in self.artifacts:self.generation(name,seed,label)

    def scored_row(self,row,generation,opponents,seeds,runtime):
        classification,program=runtime.debug.classify_revision(generation['text'],generation['finish_reason'])
        same(row['debug_classification'],classification,'Whole-completion revision classification differs')
        inner={'name':generation['name'],'text':classification['final_answer'] if program else '',
               'finish_reason':'eos' if program else 'length'}
        return cardinal.Audit.scored_row(self,row,inner,opponents,seeds,runtime)
    def diagnostics(self):
        runtime=self.artifact('runtime.json',False)
        if runtime:
            require(runtime['cuda']=='12.8' and runtime['loader']=='image_text_to_text','CUDA/runtime loader differs')
            require(runtime['arithmetic']=='NF4 frozen base; FP32 compute/LoRA; TF32 off; eager attention','Recorded arithmetic differs')
            same(runtime['turn_native_modes'],self.config['turn_native_modes'],'Runtime native modes differ')
            require(runtime['calibration_only'] is True and runtime['authoring_calls']==2,'Runtime sampling scope differs')
            schema=runtime['parameter_schema']
            same([r['name'] for r in schema],runtime['trainable_names'],'Runtime parameter order differs')
            require(all(r['dtype']=='torch.float32' and r['numel']==math.prod(r['shape']) and r['numel']>0 for r in schema)
                and sum(r['numel'] for r in schema)==runtime['trainable_parameters'],'Runtime trainable schema/count differs')
            require(not set(runtime['trainable_names'])&set(runtime['frozen_state']['tensors']),'Frozen/trainable identities overlap')
            for name,version in self.manifest['expected_installed_versions'].items():
                require(runtime['versions'][name].split('+')[0]==version,'Installed version differs: '+name)
            index=runtime['model_index'];require(index['repo']==self.config['model_repo'] and index['revision']==self.config['model_revision']
                and index['authoritative_metadata_sha256']==self.manifest['input_sha256']['inputs/model-info.json'],'Model provenance differs')
            info=read(self.run/'inputs/model-info.json')
            for row in info['siblings']:
                if row['rfilename'].endswith('.safetensors'):require(index['files'].get(row['rfilename'])==row['lfs']['sha256'],'Pinned model weight hash differs')
            frozen=runtime['frozen_state'];require(previous.value_sha(frozen['tensors'])==frozen['sha256'],'Frozen tensor fingerprint digest differs')
            quant={n:v for n,v in frozen['tensors'].items() if n in frozen['quantization_state_tensors']}
            require(previous.value_sha(quant)==frozen['quantization_state_sha256'],'Quantization fingerprint differs')
        passed=False
        for path in self.artifacts:
            if not path.startswith('smoke-'):continue
            smoke=self.artifact(path)
            require(smoke['training_use'] is False and smoke['optimizer_steps']==0 and smoke['source_checkpoint']=='base'
                and smoke['source_metadata_sha256']==self.artifacts['checkpoints/base/metadata.json'],'Smoke provenance differs')
            require(smoke['kind']=='revision_mixed_native_derivative_admission' and smoke['batch_size']==2
                and smoke['max_new_tokens']==32 and smoke['authoring_stages']==['draft','revision'],'Mixed native smoke scope differs')
            same(smoke['seeds'],[70000000+self.config['seed']+i for i in range(2)],'Smoke sampling seeds differ')
            same(smoke['token_tolerance'],self.config['token_logprob_tolerance'],'Smoke token tolerance differs')
            same(smoke['sequence_tolerance'],self.config['sequence_logprob_tolerance'],'Smoke sequence tolerance differs')
            base=self.checkpoint('base');require(base is not None,'Smoke precedes base checkpoint')
            restored=validate_restoration(smoke,base['optimizer_identity'],self.config)
            vectors=False
            if 'batch' in smoke:
                records=smoke['batch']['records']
                require(len(records)==2 and len({r['input_tokens'] for r in records})>1,'Smoke lacks mixed-length rows')
                validate_cache_timing(smoke['batch'],['draft','revision'])
                for i,record in enumerate(records):
                    validate_turn(record,self.config,stage=('draft','revision')[i],seed=smoke['seeds'][i],maximum=32)
                    validate_tokenization(record,self.tokenizer,self.eos)
                    require(record['completion_token_limit']==32,'Smoke token cap differs')
                    self.actual_context(f'{path}/row-{i}',record)
                if 'records' in smoke:vectors=validate_full_vectors(smoke,records,self.config)['passed']
            derivative=smoke.get('derivative',{})
            norm=derivative.get('gradient_l2_norm')
            finite=(type(norm) in (int,float) and math.isfinite(norm) and norm>0
                    and derivative.get('missing_gradient_names')==[] and derivative.get('finite_nonzero') is True)
            if smoke['passed']:
                require(restored and vectors and finite,'Smoke lacks full vectors, finite derivative or state restoration')
                passed=True
            else:require(not (restored and vectors and finite),'Successful smoke mislabeled as failure')
        if any(p.startswith(('generations/','batches/')) for p in self.artifacts):
            require(passed,'Authoring began without successful mixed-native numerical admission')

    def full_likelihood(self):
        gate=self.artifact('calibration/full-likelihood-gate.json',False)
        intent=self.artifact('calibration/full-likelihood-gate-intent.json',False)
        if gate is None and intent is None:return None
        seal=self.artifact('calibration/sealed.json')
        require(len(self.generations)==8,'Full likelihood gate before all eight trajectories')
        paths=[f'generations/calibration-{i}.json' for i in range(8)]
        records=[turn for i in range(8) for turn in self.generations[f'calibration-{i}']['turns']]
        require(intent is not None and isinstance(intent['started'],str),'Missing durable full likelihood intent')
        binding={'source_checkpoint':'base','source_metadata_sha256':self.artifacts['checkpoints/base/metadata.json'],
            'seal_sha256':self.artifacts['calibration/sealed.json'],
            'generation_sha256':{p:self.artifacts[p] for p in paths},'turn_sha256':[value_sha(t) for t in records]}
        for key,value in binding.items():same(intent[key],value,'Full likelihood intent binding differs: '+key)
        if gate is None:
            return {'passed':False,'interrupted_intent':True,'verified_turns':0,'verified_tokens':0,'state_restored':None}
        same(gate['generation_sha256'],{p:self.artifacts[p] for p in paths},'Full likelihood generation binding differs')
        same(gate['turn_sha256'],[value_sha(t) for t in records],'Full likelihood ordered turn binding differs')
        require(gate['source_checkpoint']=='base'
                and gate['source_metadata_sha256']==self.artifacts['checkpoints/base/metadata.json']
                and gate['seal_sha256']==self.artifacts['calibration/sealed.json'],'Full likelihood source/seal differs')
        require(gate['kind']=='revision_full_completion_likelihood_admission' and gate['expected_records']==16
                and gate['optimizer_steps']==0 and gate['training_use'] is False,'Full likelihood scope differs')
        same(gate['token_tolerance'],self.config['token_logprob_tolerance'],'Full likelihood token tolerance differs')
        same(gate['sequence_tolerance'],self.config['sequence_logprob_tolerance'],'Full likelihood sequence tolerance differs')
        restored=validate_restoration(gate,self.checkpoint('base')['optimizer_identity'],self.config)
        if 'records' in gate:
            vectors=validate_full_vectors(gate,records,self.config)
            require(gate['sampled_tokens']==vectors['tokens'],'Full likelihood sampled-token count differs')
            require(gate['passed'] is (restored and vectors['passed']),'Full likelihood aggregate/state outcome differs')
        else:
            require(gate['passed'] is False and isinstance(gate.get('error_type'),str),'Missing unexplained full likelihood records')
            vectors={'turns':0,'tokens':0,'passed':False}
        return {'passed':gate['passed'],'artifact_sha256':self.artifacts['calibration/full-likelihood-gate.json'],
                'verified_turns':vectors['turns'],'verified_tokens':vectors['tokens'],'state_restored':restored,
                'post_eos_cache_forwards':sum(t['post_eos_cache_forwards'] for t in records)}

    def qualification(self,runtime,pool):
        seal=self.artifact('calibration/sealed.json',False)
        paths=[f'generations/calibration-{i}.json' for i in range(8)]
        if seal:
            require(len(self.generations)==8,'Missing outputs cannot be sealed')
            same(seal,{'generations':{p:self.artifacts[p] for p in paths},'draws':8,'sampled_turns':16,
                'checkpoint_sha256':self.artifacts['checkpoints/base/metadata.json'],
                'scoring_allowed':False,'requires_full_likelihood_gate':True},'Calibration seal differs')
        scores=self.artifact('calibration/scores.json',False);decision=self.artifact('calibration/decision.json',False)
        if scores is None:
            require(decision is None,'Decision before scores');return None
        require(seal and pool and scores['seal_sha256']==self.artifacts['calibration/sealed.json']
            and scores['opponents_sha256']==self.artifacts['opponents.json'] and scores['training_use'] is False,
            'Scoring provenance differs')
        require(self.full_gate and self.full_gate['passed'] and
                scores['full_likelihood_gate_sha256']==self.artifacts['calibration/full-likelihood-gate.json'],
                'Qualification precedes successful full likelihood admission')
        same(scores['seeds'],self.schedule['scoring_maps'],'Scoring maps differ')
        require(scores['prescribed_program_draws']==len(scores['rows'])==8,'Qualification denominator differs')
        for i,row in enumerate(scores['rows']):
            name=f'calibration-{i}'
            require(row['generation']==name and row['generation_sha256']==self.artifacts[f'generations/{name}.json'],'Score generation binding differs')
            self.scored_row(row,self.generations[name],pool['selected'],scores['seeds'],runtime)
        calculated=cardinal.capability_decision(scores['rows'])
        if decision:
            for key in ('passed','productive_indices','selected_indices'):same(decision[key],calculated[key],'Qualification decision differs')
            require(decision['distinct_productive_ir']==len(calculated['distinct_productive_indices'])
                and decision['scores_sha256']==self.artifacts['calibration/scores.json'] and decision['optimizer_steps']==0
                and decision['all_draws_retained'] is True and decision['training_permitted_in_this_job'] is False,
                'Qualification metadata differs')
            require(decision['full_likelihood_gate_sha256']==self.artifacts['calibration/full-likelihood-gate.json'],
                    'Decision full likelihood binding differs')
        return {'decision':decision,'summary':previous.summarize_programs(scores['rows'],8)}

    def execute(self):
        families=self.frozen()
        with archived_runtime(self.run,self.manifest) as runtime:
            self.runtime=runtime
            require((self.run/'prompts/debug.txt').read_text()==runtime.debug.debug_model_spec(),'Calibration prompt differs')
            require('<think>' not in runtime.debug.debug_model_spec() and '</think>' not in runtime.debug.debug_model_spec(), 'Prompt exposes delimiter literal')
            self.maps(runtime,families);self.diagnostics();self.checkpoint('base')
            pool=self.initial_pool(runtime);self.all_batches();self.all_generations(runtime)
            self.full_gate=self.full_likelihood()
            qualification=self.qualification(runtime,pool);final=self.checkpoint('unchanged-final')
        measured=bool(qualification and qualification['decision'] and final and len(self.generations)==8 and len(self.raw_turns)==16)
        complete=measured and self.status['state']=='complete'
        require(self.status['claims_allowed'] is False,'Calibration must never claim learned improvement')
        if self.status['state']=='complete':
            require(complete and self.status['calibration_complete'] is True and self.status['actual_optimizer_steps']==0
                and self.status['learning_comparison_available'] is False,'Premature/incorrect completion')
            same(self.status['qualification_passed'],qualification['decision']['passed'],'Terminal gate differs')
        unknown=set(self.artifacts)-self.allowed_artifacts
        require(not unknown,'Unrecognized artifacts: '+str(sorted(unknown)))
        for p,d in self.start_hashes.items():require(sha(self.run/p)==d,'Run changed during audit')
        return {'kind':'independent_cpu_debug_revision_calibration_audit','run':str(self.run),'integrity_passed':True,
            'manifest_sha256':self.start_hashes['manifest.json'],'status_sha256':self.start_hashes['status.json'],
            'auditor_sha256':sha(Path(__file__)),'recorded_state':self.status['state'],'calibration_complete':complete,
            'qualification_passed':qualification['decision']['passed'] if complete else None,
            'learning_comparison_available':False,'new_supervised_optimizer_steps':0,'genuine_RL_optimizer_steps':0,
            'committed_trajectories':len(self.generations),'committed_raw_sampled_turns':len(self.raw_turns),
            'verified_unique_game_replays':len(self.verified_replays),'practice_feedback_records':len(self.feedback_checked),
            'tensor_reports':self.tensor_reports,'qualification':qualification,'context_admission':self.context_evidence,
            'full_likelihood_admission':self.full_gate,
            'limits':['Fixed n8 descriptive capability gate, not a learning or causal comparison.',
                'Only the revision uses native-disabled thinking; whole sampled completion is graded without boundary repair.',
                'Prior-condition outcomes use different seeds/maps; this is not a paired causal effect estimate.',
                'Full eval/train forward likelihood replay plus short32-token derivatives do not prove full-length backward/update capacity.',
                'CPU replays games and original tokenization; recorded neural values are not independently recomputed.']}


def audit(run):return Audit(run).execute()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=audit(args.run);text=previous.canonical(result)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as stream:stream.write(text)
    else:print(text,end='')


if __name__=='__main__':main()
