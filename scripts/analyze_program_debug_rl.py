"""Independent CPU audit of two-call thinking/debug policy-gradient trajectories.

Generated Python is never executed. Archived bounded game/compiler code replays
practice and scored games. Recorded neural checks are consistency evidence, not
an independent CPU recomputation of model likelihoods or gradients.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta
import importlib
import importlib.metadata
import hashlib
import importlib.util
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_program_cardinal_gpu as gpu
from scripts import analyze_program_cardinal_cycle as cardinal
from scripts import analyze_program_study as previous

read, sha, same, require, safe_path = previous.read, previous.sha, previous.same, previous.require, previous.safe_path
value_sha = previous.value_sha


def registered_schedule():
    return {
        'train_practice_maps': {str(i): [151000000+i*10+j for j in range(2)] for i in range(1,101)},
        'train_maps': {str(i): ([159000000,152000221] if i==22 else [152000000+i*10+j for j in range(2)]) for i in range(1,101)},
        'eval_practice_maps': {str(i): [153000000+i*10+j for j in range(2)] for i in range(32)},
        'holdout_maps': list(range(154000000,154000006)),
        'train_generation_seeds': {str(i): [161000000+i*100+j*2 for j in range(8)] for i in range(1,101)},
        'eval_generation_seeds': [162000000+i*2 for i in range(32)],
        'revision_seed_rule': 'draft_seed+1; disjoint even/odd seeds within each independent trajectory'}


def validate_tokenization(record,tokenizer,eos):
    """Recompute original native chat tokens and both exact decode presentations."""
    rendered=tokenizer.apply_chat_template(record['messages'],tokenize=False,
        add_generation_prompt=True,enable_thinking=True)
    require(rendered==record['prompt_template'],'Native chat template differs from recorded prompt')
    require(tokenizer.encode(rendered,add_special_tokens=False)==record['prompt_ids'],'Native prompt token IDs differ')
    require(tokenizer.decode(record['completion_ids'],skip_special_tokens=True)==record['text'], 'Decoded scored output differs from sampled tokens')
    require(tokenizer.decode(record['completion_ids'],skip_special_tokens=False)==record['raw_text'],'Raw decoded output differs from sampled tokens')
    same(record['eos_token_ids'],eos,'Actual model generation EOS identities differ')


def load_original_tokenizer(run,manifest):
    expected={r['rfilename']:r for r in read(run/'inputs/model-info.json')['siblings']}
    folder=run/'inputs/tokenizer'
    names=('config.json','generation_config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja')
    for name in names:
        path=folder/name;raw=path.read_bytes();row=expected[name]
        require(manifest['input_sha256'].get('inputs/tokenizer/'+name)==hashlib.sha256(raw).hexdigest(), 'Tokenizer input unbound: '+name)
        require(len(raw)==row['size'],'Original tokenizer metadata size differs: '+name)
        if row.get('lfs'):require(hashlib.sha256(raw).hexdigest()==row['lfs']['sha256'],'Original tokenizer LFS hash differs: '+name)
        else:require(hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==row['blobId'],'Original tokenizer Git blob differs: '+name)
    for package,version in [('transformers','5.20.2'),('tokenizers','0.23.3')]:
        require(importlib.metadata.version(package)==version,'CPU tokenizer dependency differs: '+package)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(folder),local_files_only=True,trust_remote_code=False)
    eos=read(folder/'generation_config.json')['eos_token_id']
    return tokenizer,sorted(set(eos if isinstance(eos,list) else [eos]))


def validate_turn(record, config, *, seed=None, messages=None, maximum=None):
    """Completion-only targets include the first actual EOS, reasoning and code."""
    limit = config['max_new_tokens'] if maximum is None else maximum
    prefix, ids, logps = record.get('prompt_ids'), record.get('completion_ids'), record.get('sampled_token_logprobs')
    require(isinstance(prefix,list) and prefix and isinstance(ids,list) and ids and isinstance(logps,list),
            'Missing sampled token provenance')
    require(all(type(x) is int and x>=0 for x in prefix+ids), 'Invalid sampled token IDs')
    require(len(ids)==len(logps)==record['output_tokens']<=limit and len(prefix)==record['input_tokens'],
            'Sampled token/logprob coverage differs')
    require(len(prefix)+limit<=config['max_context_tokens'], 'Full completion reserve exceeds context')
    require(all(type(x) in (float,int) and math.isfinite(x) and x<=1e-6 for x in logps), 'Invalid sampled logprob')
    require(record.get('thinking_enabled') is True and isinstance(record.get('messages'),list), 'Thinking context missing')
    require(isinstance(record.get('prompt_template'),str) and record['prompt_template'].rstrip().endswith('<think>'),
            'Thinking-enabled generation prefix missing')
    require(type(record.get('left_padding_tokens')) is int and record['left_padding_tokens']>=0, 'Invalid left-padding count')
    require(isinstance(record.get('text'),str) and isinstance(record.get('raw_text'),str), 'Decoded output missing')
    eos=record.get('eos_token_ids');finish=record.get('finish_reason')
    require(isinstance(eos,list) and eos and all(type(x) is int and x>=0 for x in eos), 'Invalid EOS identity')
    require(finish in ('eos','length','time') and not set(ids[:-1])&set(eos)
            and (ids[-1] in eos)==(finish=='eos'), 'EOS/truncation boundary differs')
    if finish=='length':require(len(ids)==limit,'Length stop before token cap')
    require(type(record.get('elapsed_seconds')) in (float,int) and math.isfinite(record['elapsed_seconds'])
            and record['elapsed_seconds']>0,'Invalid sampling time')
    if seed is not None:require(record['seed']==seed,'Sample seed differs')
    if messages is not None:same(record['messages'],messages,'Conversation context differs')
    return {'sampled_tokens':len(ids),'prompt_tokens':len(prefix),'sampled_sequence_logprob':math.fsum(logps)}


def validate_two_turn_trajectory(row, *, config, name, seed, checkpoint, initial_messages,
                                 feedback, revision_messages, practice_seeds):
    """Pure integrity check; caller supplies independently replayed feedback."""
    require(row.get('name')==name and row.get('trajectory_seed')==seed and row.get('checkpoint')==checkpoint,
            'Trajectory identity differs')
    require(row.get('authorship')=='fresh_learner_two_call_trajectory' and row.get('api_training_data') is False,
            'Off-policy/teacher output cannot be learner gradient data')
    require((row['model_repo'],row['model_revision'])==(config['model_repo'],config['model_revision']),
            'Trajectory model identity differs')
    require(row.get('prompt_condition')=='thinking_debug' and row.get('final_turn_index')==1,'Final revision selection differs')
    turns=row.get('turns');require(isinstance(turns,list) and len(turns)==2,'Exactly two sampled turns required')
    validate_turn(turns[0],config,seed=seed,messages=initial_messages)
    validate_turn(turns[1],config,seed=seed+1,messages=revision_messages)
    same(row['turn_sha256'],[value_sha(t) for t in turns],'Turn hashes differ')
    same(row['practice_seeds'],practice_seeds,'Practice map schedule differs')
    require(feedback['record']==turns[0] and feedback['record_sha256']==value_sha(turns[0]),'Practice evidence not bound to draft')
    require(feedback['authoring_model_calls']==2 and feedback['final_attempt_index']==1
            and feedback['generated_python_executed'] is False,'Practice harness changed authoring contract')
    require(row['training_tokens']==sum(len(t['completion_ids']) for t in turns),'Not all sampled turn tokens are targets')
    for key,value in turns[1].items():same(row[key],value,'Final alias differs from mandatory revision: '+key)
    return {'turns':2,'sampled_tokens':row['training_tokens'],'tool_feedback_target_tokens':0}


def validate_group(trajectories,rewards,advantages,update=None):
    require(len(trajectories)==len(rewards)==len(advantages)==8,'Eight complete independent trajectories required')
    require(len({r['name'] for r in trajectories})==8,'Repeated trajectory')
    require(len({r['checkpoint'] for r in trajectories})==1,'Group mixes source policies')
    seeds=[t['seed'] for r in trajectories for t in r['turns']]
    require(len(seeds)==len(set(seeds))==16,'Turn sampling seed collision')
    require(all(len(r['turns'])==2 for r in trajectories),'Dropped a sampled turn')
    expected=previous.advantages(rewards)
    zero=max(rewards)-min(rewards)<=1e-12
    if zero:expected=[0.]*8
    same(advantages,expected,'Trajectory leave-one-out advantage differs')
    if update is not None:
        require(update['objective_divisor']==8,'Objective is not averaged over eight trajectories')
        require(update['training_tokens']==sum(r['training_tokens'] for r in trajectories),'Training token count differs')
        require(update['optimizer_steps']==int(not zero),'Zero-signal/actual optimizer step differs')
        if not zero:
            require(update['trajectory_count']==8 and update['sampled_turn_count']==16
                    and update['tool_feedback_target_tokens']==0,'Trajectory target coverage differs')
    return {'zero_signal':zero,'advantages':expected,'turn_count':16,
            'sampled_tokens':sum(r['training_tokens'] for r in trajectories)}


def validate_turn_gate(gate,turns,config,require_pass=False):
    require(len(gate['records'])==len(turns),'Likelihood gate does not cover every sampled turn')
    passes=[]
    for check,turn in zip(gate['records'],turns):
        require(check['tokens']==len(turn['completion_ids']),'Likelihood token coverage differs')
        same(check['recorded_sequence_logprob'],math.fsum(turn['sampled_token_logprobs']), 'Gate sampling likelihood differs')
        errors=[]
        for key,limit in [('max_token_error',config['token_logprob_tolerance']),('sequence_error',config['sequence_logprob_tolerance'])]:
            value=check[key];require(type(value) in (float,int) and math.isfinite(value) and value>=0,'Nonfinite gate error')
            errors.append(value<=limit)
        passed=all(errors);require(check['passed'] is passed,'Recorded gate outcome differs');passes.append(passed)
    require(gate['passed'] is all(passes),'Aggregate likelihood gate differs')
    if require_pass:require(gate['passed'],'Failed likelihood gate supports an update')


def paired_summary(base,final,seeds,bootstrap_draws=10000):
    """Invalid final authoring is zero yield; missing authoring cannot enter."""
    require(len(base)==len(final)==len(seeds)==32,'All 32 paired final revisions required')
    for rows in (base,final):
        require(len({r['generation'] for r in rows})==32,'Repeated heldout trajectory')
        for r in rows:
            require(type(r['authorship_valid']) is bool and type(r['runtime_valid']) is bool,'Missing authoring classification')
            require(type(r['banked_point_yield']) in (float,int) and math.isfinite(r['banked_point_yield'])
                    and r['banked_point_yield']>=0,'Missing/nonfinite yield')
            if not r['authorship_valid']:
                require(r['banked_point_yield']==0 and r['games']==0 and not r['runtime_valid'],'Invalid output fabricated games/yield')
            else:require(r['games']==48,'Compiled heldout revision lacks fixed 48 games')
    result=gpu.paired_yield(base,final,seeds,bootstrap_draws)
    result['sampling_unit']='Paired two-call authoring trajectory, never its two turns or 48 correlated games'
    return result


@contextmanager
def archived_runtime(run,manifest):
    with cardinal.archived_runtime(run,manifest) as runtime:
        relative='selfplay/program_debug.py'
        require(sha(Path(run)/'source'/relative)==manifest['source_sha256'][relative],'Archived debug harness differs')
        prefix=runtime.game.__module__.split('.')[0]
        harness=importlib.import_module(prefix+'.program_debug')
        runtime.debug=harness
        yield runtime


class Audit(gpu.Audit):
    def __init__(self,run):
        super().__init__(run)
        self.feedback_checked={}
        self.raw_turns={}
        self.tokenizer=None

    def frozen(self):
        m,cfg=self.manifest,self.config
        require(m['format_version']==1 and m['kind']=='thinking_debug_league_rl','Unknown debug protocol')
        require(self.start_hashes['manifest.json']==self.status['manifest_sha256'],'Manifest differs')
        for p,digest in {**m['input_sha256'],**self.artifacts}.items():require(sha(safe_path(self.run,p))==digest,'Artifact differs: '+p)
        for p,digest in m['source_sha256'].items():require(sha(safe_path(self.run/'source',p))==digest,'Source differs: '+p)
        for p in ('scripts/analyze_program_study.py','scripts/analyze_program_cardinal_cycle.py','scripts/analyze_program_cardinal_gpu.py',
                  'scripts/cloud/train_program_rl.py','selfplay/program_cardinal_cuda.py'):
            require(m['source_sha256'].get(p)==sha(ROOT/p),'Executed CPU helper differs: '+p)
        require({'selfplay/program_debug.py','selfplay/program_debug_rl.py','selfplay/program_debug_cuda.py'}<=set(m['source_sha256']),
                'Debug source closure incomplete')
        same(self.plan['config'],cfg,'Plan config differs');same(self.plan['schedule'],self.schedule,'Plan schedule differs')
        same(self.schedule,registered_schedule(),'Registered debug seed schedule differs')
        constants={'model_repo':'Qwen/Qwen3.8-27B','model_revision':'1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
            'backend':'cuda','loader':'image_text_to_text','condition':'two-call-practice-debug-v1','algorithm':'full_completion_RLOO',
            'group_size':8,'batch_size':8,'iterations':100,'endpoint_choices':[25,50,100],'learning_rate':1e-5,
            'max_new_tokens':1536,'max_context_tokens':8192,'sample_seconds':360,'max_rounds':48,'thinking':True,
            'temperature':1.,'top_p':1.,'top_k':0,'compute_dtype':'float32','attention':'eager','lora_rank':8,'lora_alpha':16,
            'lora_dropout':0.,'lora_targets':['q_proj','v_proj'],'gradient_clip':.5,'token_logprob_tolerance':.002,
            'sequence_logprob_tolerance':.02,'eval_samples':32,'capability_samples':0,'minimum_productive_programs':0,
            'minimum_productive_distinct_ir':0,'opponent_attempt_cap':0,'map_seeds_per_group':2,'maximum_model_opponents':6,
            'league_refresh_every':5,'profile_attempt_cap':3,'timing_safety_factor':1.5,'audit_export_reserve_seconds':1200,
            'evaluation_batch_reserve_seconds':6360,'authoring_calls':2,'practice_maps_per_trajectory':2}
        for k,v in constants.items():same(cfg[k],v,'Registered setting differs: '+k)
        dep=read(self.run/'deployment.json')
        require(0<dep['reservation_usd']<=75 and 0<dep['maximum_hours']<=11.5,'New rental bounds differ')
        require(cfg['worker_seconds']==int(dep['maximum_hours']*3600) and datetime.fromisoformat(cfg['deadline']).timestamp()==dep['deadline_unix'],
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
        corpus=self.run/'inputs/teacher-corpus.json'
        require(sha(corpus)==parent['input_sha256']['inputs/teacher/corpus.json']==audit['teacher_corpus_sha256'],'Teacher opponent corpus identity differs')
        metadata=read(self.run/'inputs/initial-checkpoint/metadata.json')
        require(sha(self.run/'inputs/initial-checkpoint/metadata.json')==status['artifacts']['checkpoints/base/metadata.json'],'Inherited checkpoint metadata differs')
        for filename,digest in metadata['files'].items():
            require(sha(self.run/'inputs/initial-checkpoint'/filename)==digest==status['artifacts']['checkpoints/base/'+filename], 'Inherited weights/Adam differ')
        same(m['initial_checkpoint_files'],metadata['files'],'Initial file declaration differs')
        module=importlib.import_module('selfplay.program_cardinal_cuda');self.cuda_inspector=module.checkpoint_state
        identity=self.cuda_inspector(self.run/'inputs/initial-checkpoint',expected_learning_rate=1e-5)
        same(identity,metadata['optimizer_identity'],'Inherited serialized optimizer differs');same(identity,m['initial_checkpoint_identity'],'Initial identity differs')
        require(identity['step']==0 and not identity['moments'],'Inherited baseline is not zero-RL-step Adam')
        self.tokenizer,self.eos=load_original_tokenizer(self.run,m)
        info=read(self.run/'inputs/model-info.json')
        require((info['id'],info['sha'])==(cfg['model_repo'],cfg['model_revision']),'Pinned model identity differs')
        return {family:([s for values in self.schedule[family].values() for s in values] if isinstance(self.schedule[family],dict)
                        else self.schedule[family]) for family in ('train_practice_maps','train_maps','eval_practice_maps','holdout_maps')}

    def maps(self,runtime,families):
        document=read(self.run/'map-fingerprints.json');prior=read(self.run/'inputs/initial-map-fingerprints.json')
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
        meta=super().checkpoint(label)
        if label=='base' and meta is not None:
            same(meta['files'],self.manifest['initial_checkpoint_files'],'Baseline differs from prior SFT adapter/zero-step Adam')
            same(meta['optimizer_identity'],self.manifest['initial_checkpoint_identity'],'Baseline optimizer changed on import')
        return meta

    def initial_pool(self,runtime):
        corpus=read(self.run/'inputs/teacher-corpus.json');selected=[]
        for name in ('nearest','value'):
            selected.append({'id':name,'origin':'scripted','authorship':'scripted_anchor','compiled_ir':runtime.reference_programs()[name]})
        seen=set()
        for demo in corpus['demonstrations']:
            ir=demo['program']['compiled_ir'];digest=runtime.compile_ir(ir).digest
            if digest in seen:continue
            seen.add(digest)
            selected.append({'id':'teacher-'+str(demo['index']),'origin':'model',
                'authorship':'external_teacher_off_policy_opponent','compiled_ir':ir,'ir_sha256':digest,
                'generation_sha256':demo['generation_sha256'],'source_checkpoint':'external_teacher','source_path':'inputs/teacher-corpus.json'})
            if len(selected)==4:break
        require(len(selected)==4,'Missing distinct verified teacher opponents')
        expected={'selected':selected,'frozen':True,'external_teacher_opponents_are_not_RL_samples':True}
        same(read(self.run/'initial-opponents.json'),expected,'Initial opponent selection differs')
        self.pool=self.artifact('opponents.json',False)
        if self.pool is not None:same(self.pool,expected,'Initial opponent snapshot differs')
        return self.pool

    def feedback(self,path,draft,seeds,batchpath,index):
        if path in self.feedback_checked:return self.feedback_checked[path]
        record=self.artifact(path)
        require(record['draft_batch_sha256']==self.artifacts[batchpath] and record['draft_batch_index']==index
                and record['practice_seeds']==seeds,'Draft/practice binding differs')
        actual=self.runtime.debug.practice_feedback(draft,seeds,max_rounds=48)
        for key,value in actual.items():same(record[key],value,'Replayed practice feedback differs: '+key)
        for match in actual['matches']:self.verified_replays.add(value_sha(match['replay']))
        self.feedback_checked[path]=record
        return record

    def batch_specs(self):
        for n in range(1,101):
            source='base' if n==1 else f'cycle{n-1}'
            yield f'train-{n}',[f'train-cycle{n}-{i}' for i in range(8)],self.schedule['train_generation_seeds'][str(n)],source,[self.schedule['train_practice_maps'][str(n)]]*8
        if self.endpoint:
            for label in ('base',f'cycle{self.endpoint}'):
                for offset in range(0,32,8):
                    yield f'eval-{label}-{offset//8}',[f'eval-{label}-{i}' for i in range(offset,offset+8)],self.schedule['eval_generation_seeds'][offset:offset+8],label,[self.schedule['eval_practice_maps'][str(i)] for i in range(offset,offset+8)]

    def all_batches(self):
        initial=self.runtime.debug.initial_messages()
        for batchid,names,seeds,label,maps in self.batch_specs():
            for turn in ('draft','revision'):
                path=f'batches/{batchid}-{turn}.json';intentpath=f'batches/{batchid}-{turn}-intent.json'
                batch=self.artifact(path,False);intent=self.artifact(intentpath,False)
                if batch is None and intent is None:continue
                require(self.checkpoint(label) is not None,'Sampling before committed source checkpoint')
                if batchid.startswith('eval-'):require(self.checkpoint(f'cycle{self.endpoint}') is not None,'Evaluation before fixed endpoint')
                if turn=='draft':contexts=[initial]*8;turnseeds=seeds
                else:
                    draftpath=f'batches/{batchid}-draft.json';drafts=self.artifact(draftpath)
                    contexts=[]
                    for i,draft in enumerate(drafts['records']):
                        feedback=self.feedback(f'feedback/{names[i]}.json',draft,maps[i],draftpath,i)
                        contexts.append(self.runtime.debug.revision_messages(initial,draft,feedback))
                    turnseeds=[s+1 for s in seeds]
                identity={'checkpoint':label,'source_metadata_sha256':self.artifacts[f'checkpoints/{label}/metadata.json'],
                          'contexts_sha256':value_sha(contexts),'seeds':turnseeds}
                require(intent is not None,'Sampled batch lacks durable no-redraw intent')
                require(isinstance(intent['started'],str),'Intent timestamp missing')
                for key,value in identity.items():same(intent[key],value,'Intent differs: '+key)
                if batch is None:continue
                for key,value in identity.items():same(batch[key],value,'Raw batch identity differs: '+key)
                require(len(batch['records'])==8,'Partial committed sampling batch')
                width=max(len(row['prompt_ids']) for row in batch['records'])
                for i,row in enumerate(batch['records']):
                    validate_turn(row,self.config,seed=turnseeds[i],messages=contexts[i])
                    validate_tokenization(row,self.tokenizer,self.eos)
                    self.actual_context(path+'/row-'+str(i),row)
                    require(row['sampling_batch_row']==i and row['sampling_batch_size']==8
                        and row['left_padding_tokens']==width-len(row['prompt_ids']),'Batch row/padding differs')
                    self.raw_turns[(names[i],turn)]=path
                timing=batch['timing']
                require(type(timing['batch_elapsed_seconds']) in (float,int) and math.isfinite(timing['batch_elapsed_seconds'])
                        and timing['batch_elapsed_seconds']>0,'Invalid batch wall time')
                require(timing['total_output_tokens']==sum(len(r['completion_ids']) for r in batch['records'])
                        and timing['constant_cache_batch_size']==8 and timing['active_sequences']==8,'Batch usage differs')
                counts=timing['active_sequences_by_step'];lengths=[len(r['completion_ids']) for r in batch['records']]
                same(counts,[sum(length>i for length in lengths) for i in range(max(lengths))],'Decode live-row coverage differs')
                require(timing['batch_steps']==len(counts) and timing['finished_eos']==sum(r['finish_reason']=='eos' for r in batch['records']),
                        'Decode termination totals differ')
        # A crash may leave complete draft feedback before the revision batch intent.
        expected={name:(batchid,i,maps[i]) for batchid,names,seeds,label,maps in self.batch_specs() for i,name in enumerate(names)}
        for path in self.artifacts:
            if path.startswith('feedback/'):
                name=Path(path).stem;require(name in expected,'Unregistered practice feedback')
                batchid,index,seeds=expected[name];draftpath=f'batches/{batchid}-draft.json'
                draft=self.artifact(draftpath)['records'][index]
                self.feedback(path,draft,seeds,draftpath,index)

    def generation(self,name,seed,label,condition='thinking_debug'):
        if name in self.generations:return self.generations[name]
        record=self.artifact(f'generations/{name}.json');meta=self.checkpoint(label)
        require(meta is not None and record['source_metadata_sha256']==self.artifacts[f'checkpoints/{label}/metadata.json'], 'Trajectory source differs')
        index=record['batch_index'];require(type(index) is int and 0<=index<8,'Invalid trajectory batch index')
        drafts=self.artifact(record['draft_batch_path']);revisions=self.artifact(record['revision_batch_path'])
        require(record['draft_batch_sha256']==self.artifacts[record['draft_batch_path']] and
                record['revision_batch_sha256']==self.artifacts[record['revision_batch_path']],'Trajectory batch hash differs')
        same(record['turns'],[drafts['records'][index],revisions['records'][index]],'Trajectory substituted sampled output')
        require(record['feedback_path']==f'feedback/{name}.json' and record['feedback_sha256']==self.artifacts[record['feedback_path']],
                'Trajectory feedback hash differs')
        if name.startswith('train-'):
            number=int(name.split('-')[1][5:]);seeds=self.schedule['train_practice_maps'][str(number)]
        else:seeds=self.schedule['eval_practice_maps'][name.rsplit('-',1)[1]]
        feedback=self.feedback(record['feedback_path'],record['turns'][0],seeds,record['draft_batch_path'],index)
        initial=self.runtime.debug.initial_messages();revision=self.runtime.debug.revision_messages(initial,record['turns'][0],feedback)
        validate_two_turn_trajectory(record,config=self.config,name=name,seed=seed,checkpoint=label,
            initial_messages=initial,feedback=feedback,revision_messages=revision,practice_seeds=seeds)
        require(record['history_transform']==self.config['history_transform'],'History transform differs')
        self.generations[name]=record
        return record

    def all_generations(self,runtime):
        for batchid,names,seeds,label,maps in self.batch_specs():
            for name,seed in zip(names,seeds):
                if f'generations/{name}.json' in self.artifacts:self.generation(name,seed,label)

    def scored_row(self,row,generation,opponents,seeds,runtime):
        classification,program=runtime.debug.classify_debug(generation['text'],generation['finish_reason'])
        same(row['debug_classification'],classification,'Final thinking-boundary classification differs')
        inner={'name':generation['name'],'text':classification['final_answer'] if program else '',
               'finish_reason':'eos' if program else 'length'}
        return cardinal.Audit.scored_row(self,row,inner,opponents,seeds,runtime)
    def groups(self,runtime,pool):
        initial=[r for r in pool['selected'] if r['origin']=='model'] if pool else []
        models=initial;last_complete=0
        for n in range(1,101):
            folder=f'groups/{n}';source='base' if n==1 else f'cycle{n-1}'
            snapshot=self.artifact(folder+'/opponents.json',False)
            scores=self.artifact(folder+'/scores.json',False);meta=self.checkpoint(f'cycle{n}')
            if snapshot is None:
                require(scores is None and meta is None,'Group skipped opponent snapshot');continue
            require(pool is not None and n<=last_complete+1,'Group skipped prior committed checkpoint')
            anchors=[r for r in pool['selected'] if r['origin']=='scripted'];older=models[:-1]
            expected=anchors+[models[-1],older[(n-1)%len(older)]]
            same(snapshot['selected'],expected,'Dynamic matchup snapshot differs')
            latest=(n-1)//5*5;population_path=f'league/after-{latest}.json' if latest else 'opponents.json'
            require(snapshot['cycle']==n and snapshot['population_sha256']==self.artifacts[population_path]
                and snapshot['frozen_before_generation'] is True,'Population snapshot binding differs')
            start=self.artifact(folder+'/timing-start.json',False)
            if scores is None:
                require(meta is None,'Checkpoint without scores');continue
            require(start is not None and start['cycle']==n,'Missing persistent group timer')
            require(scores['cycle']==n and scores['source_checkpoint']==source and scores['source_metadata_sha256']==
                self.artifacts[f'checkpoints/{source}/metadata.json'] and scores['opponents_sha256']==self.artifacts[folder+'/opponents.json'],
                'On-policy scoring source differs')
            same(scores['seeds'],self.schedule['train_maps'][str(n)],'Training maps differ')
            require(len(scores['rows'])==8,'Training lost invalid outputs')
            records=[];hashes={}
            for i,row in enumerate(scores['rows']):
                name=f'train-cycle{n}-{i}';record=self.generation(name,self.schedule['train_generation_seeds'][str(n)][i],source)
                require(row['generation']==name and row['generation_sha256']==self.artifacts[f'generations/{name}.json'],'Score generation differs')
                self.scored_row(row,record,expected,scores['seeds'],runtime);records.append(record);hashes[name]=row['generation_sha256']
            rewards=[r['reward'] for r in scores['rows']];zero=max(rewards)-min(rewards)<=1e-12
            adv=[0.]*8 if zero else previous.advantages(rewards)
            same(scores['rewards'],rewards,'Rewards differ');same(scores['advantages'],adv,'RLOO differs');require(scores['zero_signal'] is zero,'Zero signal differs')
            for path in self.artifacts:
                if path.startswith(folder+'/gate-'):
                    gate=self.artifact(path);same(gate['generation_sha256'],hashes,'Gate generation hashes differ')
                    require(gate['source_checkpoint']==source and gate['source_metadata_sha256']==scores['source_metadata_sha256'],'Gate source differs')
                    same(gate['turn_sha256'],{r['name']:r['turn_sha256'] for r in records},'Gate turn hashes differ')
                    validate_turn_gate(gate,[t for r in records for t in r['turns']],self.config)
            valid_returns=[r['game_return'] for r in scores['rows'] if r['runtime_valid']]
            self.group_reports.append({'cycle':n,'checkpoint_committed':meta is not None,
                'gradient_signal':'no_update' if zero else 'validity_only_signal' if len(set(valid_returns))<=1 else 'gameplay_and_validity_signal',
                'rewards':rewards,'advantages':adv,**previous.summarize_programs(scores['rows'],8)})
            self.training_rows+=scores['rows']
            if meta is None:continue
            update=meta['report'];last_complete=n
            for key,value in {'source_checkpoint':source,'source_metadata_sha256':scores['source_metadata_sha256'],
                'scores_sha256':self.artifacts[folder+'/scores.json'],'generation_sha256':hashes,'rewards':rewards,'advantages':adv,
                'training_tokens':sum(r['training_tokens'] for r in records),'objective_divisor':8,
                'turn_sha256':{r['name']:r['turn_sha256'] for r in records},'all_runtime_invalid':not any(r['runtime_valid'] for r in scores['rows'])}.items():
                same(update[key],value,'Update lineage/content differs: '+key)
            validate_group(records,rewards,adv,update)
            require(update['optimizer_steps']==int(not zero) and update['outcome']==('skipped_zero_variance' if zero else 'updated'),'Update/skip differs')
            if zero:require(update['likelihood_gate_path'] is None and update['likelihood_gate_sha256'] is None,'Skip claims gradient gate')
            else:
                gatepath=update['likelihood_gate_path'];require(gatepath.startswith(folder+'/gate-') and
                    update['likelihood_gate_sha256']==self.artifacts[gatepath],'Successful gate unbound to update')
                validate_turn_gate(self.artifact(gatepath),[t for r in records for t in r['turns']],self.config,True)
                validate_turn_gate(update['gradient_likelihood_gate'],[t for r in records for t in r['turns']],self.config,True)
                require(math.isfinite(update['gradient_norm_before_clip']) and update['gradient_norm_before_clip']>0,'Invalid gradient norm')
                require(len(update['microbatch_losses'])==16 and all(math.isfinite(x) for x in update['microbatch_losses']),'Loss coverage differs')
                same(math.fsum(update['microbatch_losses']),update['loss'],'Loss sum differs')
                checks=update['gradient_likelihood_gate']['records']
                for i,(loss,check) in enumerate(zip(update['microbatch_losses'],checks)):
                    # CPU cannot rerun the model, but each loss must match its recorded
                    # recomputed full-completion likelihood and trajectory divisor.
                    estimated=-adv[i//2]*check['recorded_sequence_logprob']/8
                    error_bound=abs(adv[i//2])*check['sequence_error']/8+2e-5*max(1.,abs(estimated))
                    require(abs(loss-estimated)<=error_bound,'Per-turn objective normalization differs')
                parent_step=self.checkpoints[source]['optimizer_identity']['step']
                require(update['adam_step_before']==parent_step and update['adam_step_after']==parent_step+1,
                        'Reported actual Adam transition differs')
                require(update['optimizer_state_sha256']==meta['optimizer_identity']['state_sha256']
                        and update['frozen_state_sha256']==meta['optimizer_identity']['frozen_state_sha256'],
                        'Update optimizer/base state hash differs')
            timing=self.artifact(folder+'/timing.json',False)
            if timing:
                require(timing['cycle']==n and timing['update_sha256']==self.artifacts[folder+'/update.json'] and
                    timing['optimizer_steps']==update['optimizer_steps'] and math.isfinite(timing['elapsed_seconds']) and timing['elapsed_seconds']>0
                    and timing['includes_resume_downtime'] is True,'Profiling timer differs')
            if n%5==0:
                league=self.artifact(f'league/after-{n}.json',False)
                if league is None:continue
                seen={r['ir_sha256'] for r in models};added=[]
                for row in scores['rows']:
                    if not row['runtime_valid'] or not row['matches'] or row['positive_banking_games']*2<len(row['matches']):continue
                    p=row['classification']['program'];fingerprint=p['ir_sha256']
                    if fingerprint in seen:continue
                    seen.add(fingerprint);name=row['generation']
                    added.append({'id':name,'origin':'model','compiled_ir':p['compiled_ir'],'ir_sha256':fingerprint,
                        'generation_sha256':hashes[name],'source_path':f'generations/{name}.json','source_checkpoint':source,
                        'source_metadata_sha256':scores['source_metadata_sha256'],'admitted_after_cycle':n})
                    if len(added)==2:break
                fixed_ids={r['id'] for r in initial};models=initial+[r for r in models+added if r['id'] not in fixed_ids][-4:]
                same(league['models'],models,'League admission/pruning differs');same(league['added_ids'],[r['id'] for r in added],'League added order differs')
                require(league['after_cycle']==n and league['scores_sha256']==self.artifacts[folder+'/scores.json'] and
                    league['retained_initial_two'] is True and league['maximum_models']==6,'League provenance differs')
                self.leagues[n]=league


    def selection(self):
        selection=self.artifact('endpoint-selection.json',False)
        if selection is None:return
        require(selection['profile_groups']==3 and set(selection['profile_timing_sha256'])=={'1','2','3'},'Exactly three timing groups required')
        timings=[]
        for n in range(1,4):
            timing=self.artifact(f'groups/{n}/timing.json');timings.append(timing)
            require(selection['profile_timing_sha256'][str(n)]==self.artifacts[f'groups/{n}/timing.json']
                    and self.checkpoint(f'cycle{n}') is not None,'Endpoint choice lacks committed profiling groups')
        maximum=max(r['elapsed_seconds'] for r in timings)
        require(math.isfinite(maximum) and maximum>0,'Invalid throughput measurement')
        budget=self.status['execution_budget']
        available=(datetime.fromisoformat(budget['hard_stop'])-datetime.fromisoformat(budget['first_started'])).total_seconds()
        remaining=selection['remaining_seconds']
        require(type(remaining) in (float,int) and math.isfinite(remaining) and 0<=remaining<=available-sum(r['elapsed_seconds'] for r in timings)+1e-6,
                'Remaining budget exceeds durable timing bound')
        reserves=self.config['audit_export_reserve_seconds']+self.config['evaluation_batch_reserve_seconds']
        projections={str(n):maximum*1.5*(n-3)+reserves for n in (25,50,100)}
        eligible=[n for n in (25,50,100) if projections[str(n)]<=remaining];endpoint=max(eligible) if eligible else None
        require(selection['endpoint']==endpoint and selection['passed']==(endpoint is not None),'Timing-only endpoint choice differs')
        same(selection['projected_remaining_seconds'],projections,'Projection differs')
        same(selection['maximum_measured_group_seconds'],maximum,'Measured maximum differs')
        require(selection['safety_factor']==1.5 and selection['reserved_final_and_export_seconds']==reserves,'Timing reserve differs')
        self.endpoint=endpoint

    def actual_context(self,reference,record):
        check=gpu.actual_context_admission(record,self.config,(self.run/'prompts/debug.txt').read_text())
        self.context_evidence[reference]=check
        return check

    def diagnostics(self):
        # GPU utility validates immutable model/quantization/schema provenance and
        # both numerical modes. Our override permits distinct native chat lengths.
        super().diagnostics()
        runtime=self.artifact('runtime.json',False)
        if runtime:require(runtime['thinking'] is True and runtime['authoring_calls']==2,'Runtime authoring contract differs')
        for path in self.artifacts:
            if path.startswith('smoke-'):
                smoke=self.artifact(path)
                require(smoke['mixed_length_native_chat_prompts'] is True,'New padding path lacks fresh smoke')
                records=smoke.get('batch',{}).get('records',[])
                if smoke['passed']:
                    require(len({r['input_tokens'] for r in records})>1,'Padding smoke used only equal prompt lengths')
                    for r in records:
                        validate_turn(r,self.config,maximum=32)
                        validate_tokenization(r,self.tokenizer,self.eos)

    def evaluation(self,runtime,pool):
        if self.endpoint is None:
            require(not any(p.startswith('evaluation/') for p in self.artifacts),'Evaluation before timing selection');return
        endpoints=('base',f'cycle{self.endpoint}');seal=self.artifact('evaluation/sealed.json',False)
        if seal:
            require(all(self.checkpoint(e) for e in endpoints),'Premature heldout authoring')
            paths=[f'generations/eval-{e}-{i}.json' for e in endpoints for i in range(32)]
            expected={'checkpoints':{e:self.artifacts[f'checkpoints/{e}/metadata.json'] for e in endpoints},
                'generations':{p:self.artifacts.get(p) for p in paths},'draws':64,'sampled_turns':128,'scoring_allowed':True,
                'endpoint_selection_sha256':self.artifacts['endpoint-selection.json']}
            require(all(expected['generations'].values()),'Heldout seal missing trajectories');same(seal,expected,'64-trajectory seal differs')
        for label in endpoints:
            result=self.artifact(f'evaluation/{label}/holdout.json',False)
            if result is None:continue
            require(seal and result['checkpoint']==label and result['prompt_condition']=='thinking_debug' and result['training_use'] is False
                and result['split']=='holdout' and result['prescribed_program_draws']==len(result['scores'])==32,'Heldout denominator differs')
            require(result['opponents_sha256']==self.artifacts['opponents.json'] and result['seal_sha256']==self.artifacts['evaluation/sealed.json'],
                'Heldout opponent/seal binding differs')
            same(result['seeds'],self.schedule['holdout_maps'],'Holdout map schedule differs');same(result['opponents'],pool['selected'],'Final opponents changed')
            for i,row in enumerate(result['scores']):
                name=f'eval-{label}-{i}'
                require(row['generation']==name and row['generation_sha256']==self.artifacts[f'generations/{name}.json'],'Heldout trajectory binding differs')
                self.scored_row(row,self.generations[name],pool['selected'],result['seeds'],runtime)
            self.evaluations[label]=previous.summarize_programs(result['scores'],32)

    def execute(self):
        families=self.frozen()
        with archived_runtime(self.run,self.manifest) as runtime:
            self.runtime=runtime
            require((self.run/'prompts/debug.txt').read_text()==runtime.debug.debug_model_spec(),'Frozen debug prompt differs')
            self.maps(runtime,families);self.diagnostics();self.checkpoint('base')
            pool=self.initial_pool(runtime)
            self.groups(runtime,pool)
            self.selection()
            self.all_batches();self.all_generations(runtime);self.evaluation(runtime,pool)
        complete=(self.endpoint in (25,50,100) and all(f'cycle{i}' in self.checkpoints for i in range(1,self.endpoint+1))
            and len(self.evaluations)==2 and len(self.generations)==8*self.endpoint+64)
        require(not self.status['claims_allowed'] or complete,'Partial run permits comparisons')
        require(self.status['state']!='complete' or complete,'Premature complete state')
        actual=sum(m['report']['optimizer_steps'] for label,m in self.checkpoints.items() if label!='base')
        if self.status['state']=='complete':
            require(self.status['attempted_groups']==self.endpoint and self.status['actual_optimizer_steps']==actual,'Final step count differs')
            require(set(self.leagues)==set(range(5,self.endpoint+1,5)),'Missing prescribed league refresh')
            require(all(f'groups/{i}/timing.json' in self.artifacts for i in range(1,self.endpoint+1)),'Missing group timings')
        if self.endpoint:
            require(not any(int(label[5:])>self.endpoint for label in self.checkpoints if label.startswith('cycle')),'Trained beyond selected endpoint')
        if self.status['state']=='throughput_failed':require(self.endpoint is None and 'endpoint-selection.json' in self.artifacts,'Throughput failure lacks timing decision')
        unknown=set(self.artifacts)-self.allowed_artifacts
        require(not unknown,'Unknown committed artifacts: '+str(sorted(unknown)))
        for path,digest in self.start_hashes.items():require(sha(self.run/path)==digest,'Run changed while auditing: '+path)
        paired=paired_summary(self.evaluations['base']['programs'],self.evaluations[f'cycle{self.endpoint}']['programs'],
                              self.schedule['eval_generation_seeds']) if complete else None
        return {'format_version':1,'kind':'independent_cpu_program_debug_rl_audit','run':str(self.run),
            'manifest_sha256':self.start_hashes['manifest.json'],'status_sha256':self.start_hashes['status.json'],
            'auditor_sha256':sha(Path(__file__)),'integrity_passed':True,'recorded_state':self.status['state'],
            'planned_study_complete':complete,'planned_comparisons_available':complete,'selected_endpoint':self.endpoint,
            'baseline':'Imported exact previous 32-SFT-step adapter; no new SFT; zero-RL-step Adam',
            'new_supervised_optimizer_steps':0,'genuine_RL_optimizer_steps':actual,'genuine_optimizer_steps':actual,
            'committed_trajectories':len(self.generations),'committed_raw_sampled_turns':len(self.raw_turns),
            'raw_turns_pending_trajectory_materialization':len([key for key in self.raw_turns if key[0] not in self.generations]),
            'verified_unique_game_replays':len(self.verified_replays),'practice_feedback_records':len(self.feedback_checked),
            'tensor_reports':self.tensor_reports,'training_groups':self.group_reports,'evaluation_cells':self.evaluations,
            'paired_yield':paired,'context_admission':self.context_evidence,
            'limits':['One training seed; exploratory paired authoring-trajectory intervals, not independent-game intervals.',
                'Recorded neural likelihoods/derivatives are consistency evidence, not independently recomputed by this CPU audit.',
                'Native prompt IDs and both decoded completion strings are independently recomputed with the original pinned HF tokenizer on CPU.',
                'Initial teacher opponents are off-policy historical programs, never learner gradient targets.',
                'Practice testing, thinking and instructions change together; pre/post comparison isolates training under the same new harness.',
                'Endpoint choice is timing-only and may permit zero actual updates; actual Adam transitions are reported separately.']}


def audit(run):return Audit(run).execute()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=audit(args.run);text=previous.canonical(result)+'\n'
    if args.output:
        require(not args.output.exists(),'Audit path exists; choose a new output')
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    else:print(text,end='')


if __name__=='__main__':main()
