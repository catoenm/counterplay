"""CPU audit of verified teacher SFT and subsequent on-policy cardinal RL.

Teacher replays, actual adapter/Adam tensors and every saved likelihood vector
are inspected. Neural model computations are never rerun by this audit.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta
import importlib.util
import hashlib
import math
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import analyze_program_cardinal_gpu as gpu
from scripts import analyze_program_cardinal_cycle as cardinal
from scripts import analyze_program_study as previous
from scripts.collect_cardinal_teacher_named import verify_corpus as teacher_verify
from scripts.cloud.train_program_rl import check_likelihoods

read,sha,same,require,safe_path=previous.read,previous.sha,previous.same,previous.require,previous.safe_path


def registered_schedule():
    value={'teacher_maps':list(range(91000000,91000006)),
        'calibration_maps':list(range(92000000,92000006)),
        'train_maps':{str(i):[93000000+i*100+j for j in range(2)] for i in range(1,101)},
        'holdout_maps':list(range(94000000,94000006)),
        'capability_generation_seeds':list(range(112000000,112000008)),
        'train_generation_seeds':{str(i):[113000000+i*100+j for j in range(8)] for i in range(1,101)},
        'eval_generation_seeds':list(range(114000000,114000032)),
        'map_seed_replacements':[{'family':'train_maps','group':45,'index':0,
            'proposed_seed':93004500,'replacement_seed':93004502,'prior_equal_layout_seed':8203401,
            'reason':'Exact observable-map collision detected before teacher collection or new neural work.'}]}
    value['train_maps']['45'][0]=93004502
    return value


def registered_sft_schedule(indices):
    require(len(indices)==len(set(indices))==16 and indices==sorted(indices),'Expected first16 unique teacher indices')
    result=[]
    for epoch in range(8):
        order=list(indices);random.Random(115000000+epoch).shuffle(order)
        for offset in range(0,16,4):
            result.append({'step':len(result)+1,'epoch':epoch+1,'shuffle_seed':115000000+epoch,
                'teacher_indices':order[offset:offset+4]})
    return result


def validate_supervised_report(report, targets, before, after, delta, cfg):
    require(report['kind']=='external_teacher_supervised_update' and report['on_policy'] is False
        and report['objective']=='mean_over_programs(mean_completion_token_negative_log_likelihood_including_eos)',
        'Supervised update mislabeled as RL or changed objective')
    require(report['batch_size']==len(targets)==4 and report['optimizer_steps']==1,'Supervised batch/update differs')
    require(report['adam_step_before']==before['step'] and report['adam_step_after']==after['step']==before['step']+1
        and report['learning_rate']==before['learning_rate']==after['learning_rate']==5e-5,'Supervised Adam lineage/rate differs')
    same(report['teacher_record_sha256'],[r['teacher_record_sha256'] for r in targets],'Supervised target identity differs')
    counts=[len(r['completion_ids']) for r in targets]
    same(report['completion_tokens'],counts,'Supervised completion token counts differ')
    require(report['total_completion_tokens']==sum(counts),'Supervised total tokens differ')
    evals, trains=report['teacher_forced_eval_logprobs'],report['teacher_forced_train_logprobs']
    require(len(evals)==len(trains)==4,'Incomplete teacher likelihood vectors')
    checks=[];losses=[]
    for n, a, b in zip(counts,evals,trains):
        require(len(a)==len(b)==n,'Teacher likelihood vectors omit completion/EOS')
        check=check_likelihoods(a,b,cfg['token_logprob_tolerance'],cfg['sequence_logprob_tolerance'])
        require(check['passed'],'Saved teacher eval/train likelihood gate failed')
        checks.append(check);losses.append(-math.fsum(b)/n/4)
    same(report['teacher_forced_eval_train_gate'],{'passed':True,'records':checks},'Teacher consistency gate differs')
    require(len(report['microbatch_losses'])==4,'Missing supervised losses')
    for actual,expected in zip(report['microbatch_losses'],losses):
        require(math.isfinite(actual) and math.isclose(actual,expected,rel_tol=2e-6,abs_tol=2e-6),'Per-program token-mean CE differs')
    require(math.isclose(report['loss'],math.fsum(losses),rel_tol=2e-6,abs_tol=2e-6),'Mean of four program losses differs')
    require(not report['missing_gradient_names'] and math.isfinite(report['gradient_norm_before_clip'])
        and report['gradient_norm_before_clip']>0 and math.isfinite(report['gradient_norm_after_clip'])
        and 0<report['gradient_norm_after_clip']<=.500001,'Supervised gradients/clipping differ')
    require(delta['adapter_l2_delta']>0 and delta['changed_parameters']>0,'Claimed supervised step changed no tensor')
    same(report['adapter_l2_delta'],delta['adapter_l2_delta'],'Actual supervised adapter delta differs')
    require(report['adapter_before_sha256']==previous.value_sha(before['adapter_tensors'])
        and report['adapter_after_sha256']==previous.value_sha(after['adapter_tensors']),'Supervised adapter tensor hashes differ')
    require(report['optimizer_state_before_sha256']==before['state_sha256'] and report['optimizer_state_sha256']==after['state_sha256']
        and report['frozen_state_sha256']==before['frozen_state_sha256']==after['frozen_state_sha256'],'Supervised optimizer/frozen hashes differ')
    return {'mean_program_token_CE':math.fsum(losses),'completion_tokens':sum(counts),
        'maximum_token_gate_error':max(c['max_token_error'] for c in checks),
        'maximum_sequence_gate_error':max(c['sequence_error'] for c in checks),**delta}


class Audit(gpu.Audit):
    def __init__(self,run):
        super().__init__(run)
        self.teacher=None;self.warm_checkpoints={};self.supervised_reports=[];self.targets={}
        self.smoke_phases={'sft-admission':[],'rl-admission':[]}

    def frozen(self):
        m, cfg = self.manifest, self.config
        require(m['format_version'] == 1 and m['kind'] == 'cardinal_teacher_warmstart_league_rl', 'Unknown GPU experiment format')
        require(self.start_hashes['manifest.json'] == self.status['manifest_sha256'], 'Manifest hash differs')
        files = {'plan.json':m['plan_sha256'],'schedule.json':m['schedule_sha256'],
            'map-fingerprints.json':m['map_fingerprints_sha256'], **m['input_sha256'], **self.artifacts}
        for relative,digest in files.items():
            require(sha(safe_path(self.run,relative)) == digest, 'Artifact hash differs: '+relative)
        for relative,digest in m['source_sha256'].items():
            require(sha(safe_path(self.run/'source',relative)) == digest, 'Archived source differs: '+relative)
        # These CPU utility implementations must match the archived versions.
        for relative in ('scripts/analyze_program_study.py','scripts/analyze_program_cardinal_cycle.py',
                         'scripts/cloud/train_program_rl.py','scripts/analyze_program_cardinal_gpu.py',
                         'scripts/collect_cardinal_teacher.py','scripts/collect_cardinal_teacher_low.py',
                         'scripts/collect_cardinal_teacher_none.py','scripts/collect_cardinal_teacher_named.py'):
            require(m['source_sha256'].get(relative) == sha(ROOT/relative), 'Executed CPU helper differs from archive: '+relative)
        required = {'selfplay/program_cardinal.py','selfplay/program_cardinal_cuda.py','selfplay/program_cardinal_gpu.py',
            'scripts/cloud/bootstrap_cardinal.py','scripts/cloud/cardinal-requirements.txt',
            'selfplay/program_cardinal_sft.py','selfplay/program_cardinal_warmstart.py',
            'scripts/cloud/cardinal-warmstart-requirements.txt'}
        require(required <= set(m['source_sha256']), 'Missing required frozen GPU source')
        same(self.plan['config'],cfg,'Plan configuration differs');same(self.plan['schedule'],self.schedule,'Plan schedule differs')
        same(self.schedule,registered_schedule(),'Registered seed schedule differs')
        constants = {'model_repo':'Qwen/Qwen3.8-27B','model_revision':'1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
            'kind':'cardinal_teacher_warmstart_league_rl','backend':'cuda','loader':'image_text_to_text','condition':'verified-teacher-named-cardinal-gpu-v1',
            'group_size':8,'batch_size':8,'iterations':100,'endpoint_choices':[50,100],'learning_rate':1e-5,
            'max_new_tokens':2048,'max_context_tokens':4096,'sample_seconds':300,'max_rounds':48,
            'thinking':False,'temperature':1.,'top_p':1.,'top_k':0,'compute_dtype':'float32','attention':'eager',
            'lora_rank':8,'lora_alpha':16,'lora_dropout':0.,'lora_targets':['q_proj','v_proj'],'gradient_clip':.5,
            'token_logprob_tolerance':.002,'sequence_logprob_tolerance':.02,'eval_samples':32,
            'capability_samples':8,'minimum_productive_programs':4,'minimum_productive_distinct_ir':2,
            'minimum_positive_calibration_games':12,'map_seeds_per_group':2,'maximum_model_opponents':6,
            'league_refresh_every':5,'profile_attempt_cap':3,'timing_safety_factor':1.5,
            'audit_export_reserve_seconds':1800,'evaluation_batch_reserve_seconds':2400,
            'sft_learning_rate':5e-5,'sft_updates':32,'sft_batch_size':4,'sft_epochs':8,
            'sft_completion_cap':700,'sft_shuffle_seed':115000000}
        for key,value in constants.items():same(cfg[key],value,'Registered setting differs: '+key)
        dep=read(self.run/'deployment.json')
        require(dep['project']=='cardinal-gpu' and 0 < dep['reservation_usd'] <= 80 and 0 < dep['maximum_hours'] <= 17,
            'Rental scope differs')
        require(cfg['worker_seconds']==int(dep['maximum_hours']*3600) and
            datetime.fromisoformat(cfg['deadline']).timestamp()==dep['deadline_unix'],'Rental deadline differs')
        budget=self.status.get('execution_budget')
        if budget:
            first,stop=datetime.fromisoformat(budget['first_started']),datetime.fromisoformat(budget['hard_stop'])
            require(stop==min(first+timedelta(seconds=cfg['worker_seconds']),datetime.fromisoformat(cfg['deadline']))
                and budget['maximum_seconds']==cfg['worker_seconds'] and budget['resume_counts_wall_time'] is True,
                'Persistent time allowance changed')
            if self.status['state']=='complete':require(datetime.fromisoformat(self.status['updated'])<=stop,'Completion exceeded hard stop')
        else:require(self.status['state']=='prepared','Execution lacks persistent budget')
        info=read(self.run/'inputs/model-info.json')
        require((info['id'],info['sha'])==(cfg['model_repo'],cfg['model_revision']),'Pinned HF metadata differs')
        path=self.run/'source/selfplay/program_cardinal_cuda.py'
        spec=importlib.util.spec_from_file_location('_cardinal_audit_cuda',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.cuda_inspector=module.checkpoint_state
        self.teacher = teacher_verify(self.run/'inputs/teacher', replay=True, require_active=False)
        require(self.teacher['corpus_sha256']==m['teacher_corpus_sha256']==m['input_sha256']['inputs/teacher/corpus.json'],
            'Teacher corpus is not byte-bound to the learner experiment')
        same(read(self.run/'sft-schedule.json'),registered_sft_schedule(self.teacher['selected_indices']), 'Fixed supervised schedule differs')
        same(read(self.run/'map-fingerprints.json'),read(self.run/'inputs/teacher/map-fingerprints.json'), 'Teacher/learner map evidence differs')
        require(self.teacher['clean_prompt_sha256']==m['prompt_sha256']['cardinal'], 'Teacher clean prompt differs')
        return {'teacher_maps':self.schedule['teacher_maps'], 'capability_maps':self.schedule['calibration_maps'],
            'train_maps':[s for row in self.schedule['train_maps'].values() for s in row], 'holdout_maps':self.schedule['holdout_maps']}

    def maps(self,runtime,families):
        document=read(self.run/'map-fingerprints.json')
        def layout(seed):
            game=runtime.game(seed,48)
            return {'grid_size':game.size,'capacity':game.capacity,'max_energy':game.max_energy,
                'max_rounds':game.max_rounds,'sites':game.sites,
                'homes':[[r['home_x'],r['home_y']] for r in game.robots]}
        expected={key:[{'seed':s,'layout':layout(s),'sha256':previous.value_sha(layout(s))} for s in seeds]
            for key,seeds in families.items()}
        same(document['families'],expected,'Warmstart observable maps differ')
        prior=document['prior_seed_fingerprints']
        for seed,digest in prior.items():
            require(previous.value_sha(layout(int(seed)))==digest,'Prior map layout differs')
        previous.check_fingerprint_overlap({key:[r['sha256'] for r in rows] for key,rows in expected.items()},list(prior.values()))
        require(not {s for rows in families.values() for s in rows}&{int(s) for s in prior},'New map seeds overlap prior families')
        teacher_plan=read(self.run/'inputs/teacher/plan.json')
        merged={}
        for relative in teacher_plan['input_sha256']:
            if relative.startswith('inputs/prior-maps-'):
                old=read(self.run/'inputs/teacher'/relative)
                merged.update(old['prior_seed_fingerprints'])
                for rows in old['families'].values():merged.update({str(r['seed']):r['sha256'] for r in rows})
        same(prior,merged,'Historical map evidence omitted or replaced')

    def target(self,index):
        if index in self.targets:return self.targets[index]
        record=self.artifact(f'warmstart/targets/{index}.json',False)
        if record is None:return None
        demo=next((r for r in self.teacher['demonstrations'] if r['index']==index),None)
        require(demo is not None,'Supervised target not selected in frozen corpus')
        prompt,text=[message['content'] for message in demo['messages']]
        require(record['teacher_index']==index and record['corpus_sha256']==self.manifest['teacher_corpus_sha256']
            and record['teacher_generation_sha256']==demo['generation_sha256'],'Supervised target provenance differs')
        same(record['teacher_program'],demo['program'],'Supervised source/IR provenance differs')
        require(record['prompt']==prompt and record['text']==text and record['authorship']=='external_teacher_distillation'
            and record['kind']=='supervised_teacher_demo' and record['on_policy'] is False
            and record['sampled'] is False and record['supervised'] is True,'Teacher target ownership differs')
        require(not {'seed','sampled_token_logprobs','advantages','reward'}&set(record),'Supervised targets masquerade as sampled RL')
        same(record['completion_ids'],demo['completion_ids'],'Supervised target/EOS tokens differ')
        require(record['output_tokens']==len(record['completion_ids']) and 2<=record['output_tokens']<=700
            and record['completion_token_cap']==700,'Supervised completion bound differs')
        eos=read(self.run/'inputs/teacher/plan.json')['learner']['eos_token_id']
        same(record['eos_token_ids'],self.artifact('runtime.json')['eos_token_ids'],
            'Supervised EOS set differs from actual model stopping tokens')
        require(record['completion_ids'][-1]==record['target_eos_token_id']==eos
            and eos in record['eos_token_ids'] and not set(record['completion_ids'][:-1])&set(record['eos_token_ids']),
            'Supervised target has missing/premature EOS')
        require(record['teacher_text_sha256']==hashlib.sha256(text.encode()).hexdigest() and record['prompt_sha256']==self.manifest['prompt_sha256']['cardinal'],
            'Supervised target text/prompt hash differs')
        require(record['finish_reason']=='teacher_target_eos'
            and record['loss_mask']=='completion tokens only, including actual tokenizer EOS','Supervised loss-mask declaration differs')
        expected={key:value for key,value in record.items() if key not in
            ('teacher_record_sha256','teacher_index','corpus_sha256','teacher_generation_sha256','teacher_program')}
        require(previous.value_sha(expected)==record['teacher_record_sha256'],'Supervised record digest differs')
        measure=self.artifact('prompt-measurement.json')
        require(record['prompt_ids']==measure['prompt_ids'] and record['prompt_template']==measure['prompt_template'],
            'Supervised prompt tokens differ from sampled learner prompt')
        gpu.actual_context_admission(record,self.config,prompt)
        self.targets[index]=record
        return record

    def warm_checkpoint(self,step):
        if step in self.warm_checkpoints:return self.warm_checkpoints[step]
        prefix=f'warmstart/checkpoints/step{step}'
        meta=self.artifact(prefix+'/metadata.json',False)
        if meta is None:return None
        require(meta['step']==step and meta['complete'] is True and meta['manifest_sha256']==self.status['manifest_sha256']
            and meta['teacher_corpus_sha256']==self.manifest['teacher_corpus_sha256'],'Supervised checkpoint provenance differs')
        require((meta['model_repo'],meta['model_revision'])==(self.config['model_repo'],self.config['model_revision']),
            'Supervised model identity differs')
        require(set(meta['files'])=={'adapter.safetensors','optimizer.pt'},'Supervised checkpoint files differ')
        for name,digest in meta['files'].items():
            self.allowed_artifacts.add(prefix+'/'+name)
            require(self.artifacts.get(prefix+'/'+name)==digest,'Supervised checkpoint files not committed atomically')
        folder=self.run/prefix
        identity=self.cuda_inspector(folder,expected_learning_rate=5e-5)
        same(meta['optimizer_identity'],identity,'Actual supervised Adam/checkpoint identity differs')
        require(identity['step']==step,'Actual supervised Adam counter differs')
        runtime=self.artifact('runtime.json')
        same(identity['parameter_schema'],runtime['parameter_schema'],'Supervised parameter schema differs')
        require(identity['frozen_state_sha256']==runtime['frozen_state']['sha256']
            and identity['quantization_state_sha256']==runtime['frozen_state']['quantization_state_sha256'],'Supervised frozen tensors changed')
        if step==0:
            require(meta['parent_step'] is None and meta['parent_metadata_sha256'] is None and meta['report'] is None
                and not identity['moments'],'Initial supervised optimizer is not fresh')
        else:
            parent=self.warm_checkpoint(step-1)
            require(parent is not None and meta['parent_step']==step-1
                and meta['parent_metadata_sha256']==self.artifacts[f'warmstart/checkpoints/step{step-1}/metadata.json'],
                'Supervised parent lineage differs')
            report=self.artifact(f'warmstart/updates/{step}.json')
            same(meta['report'],report,'Supervised checkpoint/update transaction differs')
            expected=registered_sft_schedule(self.teacher['selected_indices'])[step-1]
            same(report['schedule'],expected,'Supervised batch schedule differs')
            require(report['step']==step and report['source_metadata_sha256']==meta['parent_metadata_sha256']
                and report['teacher_corpus_sha256']==self.manifest['teacher_corpus_sha256'],'Supervised update provenance differs')
            targets=[self.target(i) for i in expected['teacher_indices']]
            require(all(targets),'Supervised update missing a target')
            same(report['target_artifact_sha256'],{str(i):self.artifacts[f'warmstart/targets/{i}.json'] for i in expected['teacher_indices']},
                'Supervised target hashes differ')
            delta=gpu.checkpoint_delta(folder,self.run/f'warmstart/checkpoints/step{step-1}')
            summary=validate_supervised_report(report,targets,parent['optimizer_identity'],identity,delta,self.config)
            self.supervised_reports.append({'step':step,**summary})
        self.tensor_reports[f'sft_step{step}']={'actual_optimizer_step':identity['step'],
            'actual_learning_rate':identity['learning_rate'],'trainable_parameters':sum(r['numel'] for r in identity['parameter_schema'])}
        self.warm_checkpoints[step]=meta
        return meta

    def checkpoint(self,label):
        if label!='base':return super().checkpoint(label)
        if label in self.checkpoints:return self.checkpoints[label]
        meta=self.artifact('checkpoints/base/metadata.json',False)
        if meta is None:return None
        require(meta['label']=='base' and meta['parent'] is None and meta['complete'] is True
            and meta['manifest_sha256']==self.status['manifest_sha256'],'RL base checkpoint metadata differs')
        require((meta['model_repo'],meta['model_revision'])==(self.config['model_repo'],self.config['model_revision']),
            'RL baseline model differs')
        require(set(meta['files'])=={'adapter.safetensors','optimizer.pt'},'RL baseline checkpoint files differ')
        for name,digest in meta['files'].items():
            self.allowed_artifacts.add('checkpoints/base/'+name)
            require(self.artifacts.get('checkpoints/base/'+name)==digest,'RL baseline files not atomically committed')
        identity=self.cuda_inspector(self.run/'checkpoints/base',expected_learning_rate=1e-5)
        same(meta['optimizer_identity'],identity,'Actual RL baseline Adam differs')
        final=self.warm_checkpoint(32)
        require(final is not None,'RL base exists without32 verified SFT updates')
        old=final['optimizer_identity']
        require(identity['step']==0 and not identity['moments'] and old['step']==32 and old['learning_rate']==5e-5,
            'Supervised Adam was not reset to a fresh RL optimizer')
        require(identity['files']['adapter.safetensors']==old['files']['adapter.safetensors'],
            'SFT→RL transition changed adapter file bytes')
        for key in ('adapter_tensors','parameter_schema','trainable_names','rng','frozen_state_sha256','quantization_state_sha256'):
            same(identity[key],old[key],'SFT→RL reset changed '+key)
        transition=self.artifact('warmstart/optimizer-transition.json')
        same(meta['report'],transition,'RL base/reset transaction differs')
        require(transition['kind']=='supervised_to_rl_optimizer_reset' and transition['source_checkpoint']=='warmstart/step32'
            and transition['source_metadata_sha256']==self.artifacts['warmstart/checkpoints/step32/metadata.json'],
            'RL reset provenance differs')
        require(transition['learning_rate_before']==5e-5 and transition['learning_rate_after']==1e-5
            and transition['sft_adam_step']==32 and transition['rl_adam_step']==transition['optimizer_steps']==0,
            'RL reset rate/counter report differs')
        require(all(transition[k] is True for k in ('adapter_tensors_unchanged','rl_moments_empty','rng_unchanged','adapter_file_equality_verified'))
            and transition['adapter_file_equality_still_required'] is False,'RL reset evidence incomplete')
        same(transition['sft_optimizer_identity'],old,'Reset source optimizer differs')
        same(transition['rl_optimizer_identity'],identity,'Reset target optimizer differs')
        require(transition['optimizer_state_before_sha256']==old['state_sha256']
            and transition['optimizer_state_after_sha256']==identity['state_sha256']
            and transition['adapter_tensor_sha256']==previous.value_sha(old['adapter_tensors'])
            and transition['adapter_file_sha256']==old['files']['adapter.safetensors']
            and transition['frozen_state_sha256']==old['frozen_state_sha256'],'Reset exact tensor hashes differ')
        self.tensor_reports['base']={'actual_optimizer_step':0,'actual_learning_rate':1e-5,
            'trainable_parameters':sum(r['numel'] for r in identity['parameter_schema']),
            'same_adapter_as_sft_step32':True}
        self.checkpoints['base']=meta
        return meta

    def diagnostics(self):
        runtime=self.artifact('runtime.json',False)
        measurement=self.artifact('prompt-measurement.json',False)
        if runtime:
            require(runtime['cuda']=='12.8' and runtime['loader']=='image_text_to_text','CUDA/runtime differs')
            require(runtime['arithmetic']=='NF4 frozen base; FP32 compute/LoRA; TF32 off; eager attention','Arithmetic differs')
            schema=runtime['parameter_schema']
            same([r['name'] for r in schema],runtime['trainable_names'],'Runtime parameter order differs')
            require(all(r['dtype']=='torch.float32' and r['numel']==math.prod(r['shape']) and r['numel']>0 for r in schema)
                and sum(r['numel'] for r in schema)==runtime['trainable_parameters'],'Runtime trainable count differs')
            require(not set(runtime['trainable_names'])&set(runtime['frozen_state']['tensors']),'Frozen and trainable identities overlap')
            for name,version in self.manifest['expected_installed_versions'].items():
                require(runtime['versions'][name].split('+')[0]==version,'Installed version differs: '+name)
            index=runtime['model_index']
            require((index['repo'],index['revision'])==(self.config['model_repo'],self.config['model_revision'])
                and index['authoritative_metadata_sha256']==self.manifest['input_sha256']['inputs/model-info.json'],'Model provenance differs')
            info=read(self.run/'inputs/model-info.json')
            for row in info['siblings']:
                if row['rfilename'].endswith('.safetensors'):
                    require(index['files'].get(row['rfilename'])==row['lfs']['sha256'],'Original model shard hash differs')
            teacher_plan=read(self.run/'inputs/teacher/plan.json')
            require(index['files']['tokenizer.json']==teacher_plan['learner']['tokenizer_sha256'],
                'Learner original tokenizer differs from admitted teacher target tokenizer')
            frozen=runtime['frozen_state']
            require(previous.value_sha(frozen['tensors'])==frozen['sha256'],'Frozen tensor hash differs')
            quant={name:value for name,value in frozen['tensors'].items() if name in frozen['quantization_state_tensors']}
            require(previous.value_sha(quant)==frozen['quantization_state_sha256'],'Frozen quantization identity differs')
        if measurement:
            require(measurement['prompt_tokens']==len(measurement['prompt_ids'])>0
                and measurement['completion_cap']==2048 and measurement['context_cap']==4096
                and measurement['admitted']==(measurement['prompt_tokens']+2048<=4096),'Actual prompt measurement differs')
            require((self.run/'prompts/cardinal.txt').read_text().strip() in measurement['prompt_template'],'Measured prompt omits clean specification')
            from tokenizers import Tokenizer
            tokenizer=Tokenizer.from_file(str(self.run/'inputs/teacher/inputs/tokenizer.json'))
            same(tokenizer.encode(measurement['prompt_template'],add_special_tokens=False).ids,
                measurement['prompt_ids'],'Pinned tokenizer recomputation of rendered prompt IDs differs')
        for i in self.teacher['selected_indices']:self.target(i)
        for step in range(33):self.warm_checkpoint(step)
        order=list(self.artifacts)
        for path in self.artifacts:
            phase=next((p for p in self.smoke_phases if path.startswith(p+'/smoke-')),None)
            if phase is None:continue
            smoke=self.artifact(path)
            source=smoke['source_checkpoint']
            require(smoke['phase']==phase and smoke['training_use'] is False and smoke['optimizer_steps']==0
                and smoke['source_metadata_sha256']==self.artifacts[source],'Admission smoke provenance differs')
            if phase=='sft-admission':
                require(source.startswith('warmstart/checkpoints/step') and source.endswith('/metadata.json'), 'SFT smoke source differs')
            else:require(source=='checkpoints/base/metadata.json','RL admission must use fixed SFT baseline')
            require(order.index(source)<order.index(path),'Smoke source was not committed first')
            if not smoke['passed']:continue
            require(all(smoke[k] is True for k in ('adapter_unchanged','optimizer_unchanged','rng_restored','config_restored')),
                'Smoke changed policy/optimizer/RNG')
            require(smoke['optimizer_state_before_sha256']==smoke['optimizer_state_after_sha256'],'Smoke changed Adam')
            source_meta=self.artifact(source)
            require(smoke['optimizer_state_before_sha256']==source_meta['optimizer_identity']['state_sha256'],
                'Smoke actual optimizer differs from source checkpoint')
            require(smoke['frozen_state_sha256']==source_meta['optimizer_identity']['frozen_state_sha256'],'Smoke frozen state differs')
            records=smoke['batch']['records']
            require(len(records)==8 and all(0<len(r['completion_ids'])<=32 for r in records),'Smoke draw scope differs')
            for i,row in enumerate(records):self.actual_context(path+f'/row-{i}',row)
            for name in ('eval_likelihood_gate','train_likelihood_gate'):
                gate=dict(smoke[name]);gate['generation_names']=[]
                previous.validate_gate(gate,records,[],self.config)
            require(smoke['derivative']['finite_nonzero'] is True and math.isfinite(smoke['derivative']['gradient_l2_norm'])
                and smoke['derivative']['gradient_l2_norm']>0 and not smoke['derivative']['missing_gradient_names'],'Smoke derivative invalid')
            self.smoke_phases[phase].append(path)
        if any(step>0 for step in self.warm_checkpoints):
            first=order.index('warmstart/updates/1.json')
            require(any(order.index(path)<first for path in self.smoke_phases['sft-admission']),
                'SFT began without prior successful numerical admission')
        batches=[order.index(p) for p in self.artifacts if p.startswith('batches/')]
        if batches:
            require(any(order.index(path)<min(batches) for path in self.smoke_phases['rl-admission']),
                'Learner authoring began without successful post-SFT numerical admission')

    def execute(self):
        result=super().execute()
        result.update(kind='independent_cpu_cardinal_teacher_warmstart_audit',auditor_sha256=sha(Path(__file__)),
            teacher_corpus_sha256=self.manifest['teacher_corpus_sha256'],
            teacher_attempts=32,verified_supervised_demonstrations=len(self.teacher['demonstrations']),
            teacher_replays_verified=sum(len(read(self.run/f'inputs/teacher/evaluation/teacher-{i:02d}.json')['matches']) for i in range(32)),
            genuine_supervised_optimizer_steps=len(self.supervised_reports),
            genuine_RL_optimizer_steps=result['genuine_optimizer_steps'],
            supervised_updates=sorted(self.supervised_reports,key=lambda r:r['step']),
            optimizer_reset_verified='base' in self.checkpoints,
            numerical_admission_phases=self.smoke_phases,
            teacher_condition='Reasoning-disabled requested; distinct fresh32 cohort with uniform teacher-only local-variable naming clarification, following the evaluated insufficient none cohort',
            teacher_delivery_gate='Prescribed draw0 nonempty stop before remaining31 calls; no strategic selection or replacement',
            map_reuse_provenance=read(self.run/'inputs/teacher/plan.json')['map_reuse'],
            map_reuse_interpretation='Teacher qualification maps were previously observed; learner calibration, training and heldout map families were unobserved before learner training.',
            baseline='Fixed teacher-SFT adapter with fresh RL Adam; original untrained model is not an evaluation endpoint')
        result['context_admission']['precheck_authoritative']=True
        result['context_admission']['tokenization_verification']='CPU recomputes rendered prompt token IDs with the pinned original tokenizer. Chat-template rendering itself is recorded backend evidence bound to frozen source.'
        result['limits'].append('Supervised teacher targets are explicit off-policy distillation; only subsequent fresh learner samples drive RL.')
        return result


def audit(run):return Audit(run).execute()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=audit(args.run);text=previous.canonical(result)+'\n'
    if args.output:
        require(not args.output.exists(),'Audit output already exists; choose a new immutable path')
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    else:print(text,end='')


if __name__=='__main__':main()

