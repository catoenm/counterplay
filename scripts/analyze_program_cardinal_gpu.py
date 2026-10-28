"""Independent CPU audit of the CUDA cardinal league; never loads a neural model.

Archived compiler/game code replays actual games. Saved neural likelihood checks
are validated as recorded evidence; they cannot be independently recomputed on
CPU without loading the model. Missing evaluation outputs never become zeros.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta
import importlib.util
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_program_cardinal_cycle as cardinal
from scripts import analyze_program_study as previous

read, sha, same, require, safe_path = previous.read, previous.sha, previous.same, previous.require, previous.safe_path


def registered_schedule():
    return {'calibration_maps': list(range(8100000, 8100006)),
        'train_maps': {str(i): [8200000+i*100+j for j in range(2)] for i in range(1, 101)},
        'holdout_maps': list(range(8300000, 8300006)),
        'capability_generation_seeds': list(range(10100000, 10100008)),
        'train_generation_seeds': {str(i): [10200000+i*100+j for j in range(8)] for i in range(1, 101)},
        'eval_generation_seeds': list(range(10300000, 10300032))}


def paired_yield(base, final, seeds, bootstrap_draws=10000):
    require(len(base) == len(final) == len(seeds) == len(set(seeds)) == 32, 'All32 paired controller draws required')
    delta = [b['banked_point_yield']-a['banked_point_yield'] for a,b in zip(base,final)]
    require(all(math.isfinite(x) for x in delta), 'Nonfinite paired yield')
    rng = random.Random(20261024)
    distribution = sorted(math.fsum(delta[rng.randrange(32)] for _ in range(32))/32 for _ in range(bootstrap_draws))
    return {'program_cluster_n': 32, 'paired_generation_seeds': seeds,
        'base_mean_yield': math.fsum(r['banked_point_yield'] for r in base)/32,
        'final_mean_yield': math.fsum(r['banked_point_yield'] for r in final)/32,
        'mean_yield_difference': math.fsum(delta)/32, 'paired_yield_differences': delta,
        'descriptive_bootstrap_95_interval': [distribution[int(.025*(bootstrap_draws-1))], distribution[int(.975*(bootstrap_draws-1))]],
        'bootstrap_draws': bootstrap_draws, 'bootstrap_seed': 20261024,
        'caution': 'Descriptive paired-generation bootstrap, one training seed. Correlated replay games are not independent samples. Invalid authoring remains zero yield; missing authoring never does.'}


def actual_context_admission(record, config, prompt_text=None):
    ids=record.get('prompt_ids');count=record.get('input_tokens')
    require(isinstance(ids,list) and ids and all(type(x) is int and x>=0 for x in ids),
        'Actual prompt token IDs are missing/invalid')
    require(type(count) is int and count==len(ids),'Actual input-token count differs from prompt IDs')
    maximum=count+config['max_new_tokens']
    require(maximum<=config['max_context_tokens'],'Actual prompt plus reserved completion exceeds context cap')
    if prompt_text is not None:
        require(isinstance(record.get('prompt_template'),str) and prompt_text.strip() in record['prompt_template'],
            'Rendered backend template omits the frozen prompt')
    return {'actual_prompt_tokens':count,'reserved_completion_tokens':config['max_new_tokens'],
        'maximum_total_tokens':maximum,'context_cap':config['max_context_tokens'],'admitted':True,
        'prompt_ids_sha256':previous.value_sha(ids)}


def checkpoint_delta(folder, parent):
    import torch
    from safetensors.torch import load_file
    current, old = load_file(str(folder/'adapter.safetensors'), device='cpu'), load_file(str(parent/'adapter.safetensors'), device='cpu')
    require(current.keys() == old.keys(), 'Adapter tensor names changed')
    squared, changed, count = 0., 0, 0
    for name, value in current.items():
        before = old[name]
        require(value.dtype == before.dtype == torch.float32 and value.shape == before.shape
            and bool(torch.isfinite(value).all()) and bool(torch.isfinite(before).all()), 'Invalid adapter tensors')
        squared += float((value.double()-before.double()).square().sum())
        changed += int((value != before).sum()); count += value.numel()
    return {'adapter_l2_delta': math.sqrt(squared), 'changed_parameters': changed, 'trainable_parameters': count}


def independently_select_endpoint(timings, remaining, cfg):
    if not timings or not any(r['optimizer_steps'] == 1 for r in timings):
        return None
    per_group = max(r['elapsed_seconds'] for r in timings)
    require(math.isfinite(per_group) and per_group > 0 and math.isfinite(remaining) and remaining >= 0, 'Invalid profiling time')
    reserves = cfg['evaluation_batch_reserve_seconds']+cfg['audit_export_reserve_seconds']
    choices = [n for n in (50,100) if 1.5*per_group*(n-len(timings))+reserves <= remaining]
    return max(choices) if choices else None


class Audit(cardinal.Audit):
    def __init__(self, run):
        super().__init__(run)
        self.endpoint = None
        self.batch_names = set()
        self.raw_batch_draws = {}
        self.leagues = {}
        self.cuda_inspector = None
        self.context_evidence = {}

    def frozen(self):
        m, cfg = self.manifest, self.config
        require(m['format_version'] == 1 and m['kind'] == 'cardinal_gpu_league_rl', 'Unknown GPU experiment format')
        require(self.start_hashes['manifest.json'] == self.status['manifest_sha256'], 'Manifest hash differs')
        files = {'plan.json':m['plan_sha256'],'schedule.json':m['schedule_sha256'],
            'map-fingerprints.json':m['map_fingerprints_sha256'], **m['input_sha256'], **self.artifacts}
        for relative,digest in files.items():
            require(sha(safe_path(self.run,relative)) == digest, 'Artifact hash differs: '+relative)
        for relative,digest in m['source_sha256'].items():
            require(sha(safe_path(self.run/'source',relative)) == digest, 'Archived source differs: '+relative)
        # These CPU utility implementations must match the archived versions.
        for relative in ('scripts/analyze_program_study.py','scripts/analyze_program_cardinal_cycle.py',
                         'scripts/cloud/train_program_rl.py'):
            require(m['source_sha256'].get(relative) == sha(ROOT/relative), 'Executed CPU helper differs from archive: '+relative)
        required = {'selfplay/program_cardinal.py','selfplay/program_cardinal_cuda.py','selfplay/program_cardinal_gpu.py',
            'scripts/cloud/bootstrap_cardinal.py','scripts/cloud/cardinal-requirements.txt'}
        require(required <= set(m['source_sha256']), 'Missing required frozen GPU source')
        same(self.plan['config'],cfg,'Plan configuration differs');same(self.plan['schedule'],self.schedule,'Plan schedule differs')
        same(self.schedule,registered_schedule(),'Registered seed schedule differs')
        constants = {'model_repo':'Qwen/Qwen3.8-27B','model_revision':'1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
            'kind':'cardinal_gpu_league_rl','backend':'cuda','loader':'image_text_to_text','condition':'clean-cardinal-gpu-v1',
            'group_size':8,'batch_size':8,'iterations':100,'endpoint_choices':[50,100],'learning_rate':1e-5,
            'max_new_tokens':2048,'max_context_tokens':4096,'sample_seconds':300,'max_rounds':48,
            'thinking':False,'temperature':1.,'top_p':1.,'top_k':0,'compute_dtype':'float32','attention':'eager',
            'lora_rank':8,'lora_alpha':16,'lora_dropout':0.,'lora_targets':['q_proj','v_proj'],'gradient_clip':.5,
            'token_logprob_tolerance':.002,'sequence_logprob_tolerance':.02,'eval_samples':32,
            'capability_samples':8,'minimum_productive_programs':4,'minimum_productive_distinct_ir':2,
            'minimum_positive_calibration_games':12,'map_seeds_per_group':2,'maximum_model_opponents':6,
            'league_refresh_every':5,'profile_attempt_cap':3,'timing_safety_factor':1.5,
            'audit_export_reserve_seconds':1800,'evaluation_batch_reserve_seconds':2400}
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
        return {'calibration':self.schedule['calibration_maps'],
            'train':[s for row in self.schedule['train_maps'].values() for s in row], 'holdout':self.schedule['holdout_maps']}

    def maps(self,runtime,families):
        document=read(self.run/'map-fingerprints.json')
        def layout(seed):
            g=runtime.game(seed,48)
            return {'grid_size':g.size,'capacity':g.capacity,'max_energy':g.max_energy,'max_rounds':g.max_rounds,
                'sites':g.sites,'homes':[[r['home_x'],r['home_y']] for r in g.robots]}
        expected={k:[{'seed':s,'layout':layout(s),'sha256':previous.value_sha(layout(s))} for s in seeds] for k,seeds in families.items()}
        same(document['families'],expected,'Observable map differs')
        old=read(self.run/'inputs/prior-map-fingerprints.json');prior=dict(old['prior_seed_fingerprints'])
        for rows in old['families'].values():prior.update({str(r['seed']):r['sha256'] for r in rows})
        same(document['prior_seed_fingerprints'],prior,'Prior map identity differs')
        for seed,digest in prior.items():require(previous.value_sha(layout(int(seed)))==digest,'Historical map identity differs')
        previous.check_fingerprint_overlap({k:[r['sha256'] for r in rows] for k,rows in expected.items()},list(prior.values()))
        require(not {s for seeds in families.values() for s in seeds}&{int(s) for s in prior},'Map seeds overlap historical runs')

    def checkpoint(self,label):
        if label in self.checkpoints:return self.checkpoints[label]
        meta=self.artifact(f'checkpoints/{label}/metadata.json',False)
        if meta is None:return None
        require(label=='base' or label.startswith('cycle') and label[5:].isdigit() and 1<=int(label[5:])<=100,'Unknown checkpoint')
        require(meta['label']==label and meta['complete'] is True and meta['manifest_sha256']==self.status['manifest_sha256'],
            'Checkpoint metadata differs')
        require((meta['model_repo'],meta['model_revision'])==(self.config['model_repo'],self.config['model_revision']), 'Checkpoint model differs')
        require(set(meta['files'])=={'adapter.safetensors','optimizer.pt'},'CUDA checkpoint files differ')
        for name,digest in meta['files'].items():
            relative=f'checkpoints/{label}/{name}';self.allowed_artifacts.add(relative)
            require(self.artifacts.get(relative)==digest,'Checkpoint files not atomically committed')
        folder=self.run/'checkpoints'/label
        identity=self.cuda_inspector(folder,expected_learning_rate=1e-5)
        same(meta['optimizer_identity'],identity,'Actual serialized Adam/LR/adapter identity differs')
        runtime=self.artifact('runtime.json')
        same(identity['trainable_names'],runtime['trainable_names'],'Checkpoint trainable order differs')
        require(identity['frozen_state_sha256']==runtime['frozen_state']['sha256'] and
            identity['quantization_state_sha256']==runtime['frozen_state']['quantization_state_sha256'],'Frozen base/quantization identity differs')
        report={'actual_optimizer_step':identity['step'],'actual_learning_rate':identity['learning_rate'],
            'trainable_parameters':sum(r['numel'] for r in identity['parameter_schema'])}
        require(report['trainable_parameters']==runtime['trainable_parameters'],'Runtime trainable count differs')
        if label=='base':
            require(meta['parent'] is None and meta['report'] is None and identity['step']==0 and not identity['moments'], 'Base is not fresh zero-step Adam')
        else:
            n=int(label[5:]);source='base' if n==1 else f'cycle{n-1}';parent=self.checkpoint(source)
            require(parent is not None and meta['parent']==source,'Checkpoint lineage differs')
            update=self.artifact(f'groups/{n}/update.json');same(meta['report'],update,'Checkpoint/update atomic commit differs')
            require(update['optimizer_steps'] in (0,1) and type(update['optimizer_steps']) is int,'Invalid genuine step count')
            require(identity['step']==parent['optimizer_identity']['step']+update['optimizer_steps'],'Actual Adam step differs')
            delta=checkpoint_delta(folder,self.run/'checkpoints'/source);report.update(delta)
            require(report['trainable_parameters']==runtime['trainable_parameters'],'Trainable parameter count differs')
            same(delta['adapter_l2_delta'],update['adapter_l2_delta'],'Measured adapter delta differs')
            if update['optimizer_steps']==0: same(meta['files'],parent['files'],'Zero-signal checkpoint changed bytes')
            else:require(delta['changed_parameters']>0 and delta['adapter_l2_delta']>0,'Claimed update changed no parameter')
        self.tensor_reports[label]=report;self.checkpoints[label]=meta
        return meta

    def actual_context(self, reference, record):
        check=actual_context_admission(record,self.config,(self.run/'prompts/cardinal.txt').read_text())
        if self.context_evidence:
            same(check,next(iter(self.context_evidence.values())),'Backend prompt tokens changed between recorded samples')
        self.context_evidence[reference]=check
        return check

    def generation(self,name,seed,label,condition='cardinal'):
        record=previous.Audit.generation(self,name,seed,label,condition)
        require(record['completion_ids'],'Empty authoring trajectory')
        self.actual_context(name,record)
        path=record['batch_path'];batch=self.artifact(path)
        require(path.startswith('batches/') and self.artifacts[path]==record['batch_sha256'],'Generation batch binding differs')
        i=record['batch_index'];require(type(i) is int and 0<=i<8,'Invalid batch row')
        require(len(batch['records'])==len(batch['names'])==len(batch['seeds'])==8 and batch['names'][i]==name
            and batch['seeds'][i]==seed and batch['checkpoint']==label,'Batch request order/source differs')
        require(batch['source_metadata_sha256']==record['source_metadata_sha256'] and
            batch['prompt_sha256']==record['prompt_sha256'],'Batch source/prompt differs')
        raw=batch['records'][i]
        require(raw['seed']==seed,'Raw backend seed differs')
        for key,value in raw.items():same(record[key],value,'Generation differs from committed raw batch row: '+key)
        require(raw.get('sampling_batch_row',i)==i and raw.get('sampling_batch_size',8)==8,'Backend row permutation differs')
        timing=batch['timing'];require(math.isfinite(timing['batch_elapsed_seconds']) and timing['batch_elapsed_seconds']>0,'Invalid batch elapsed time')
        require(timing['total_output_tokens']==sum(len(r['completion_ids']) for r in batch['records']),'Batch token count differs')
        same(record['prompt_ids'],batch['records'][0]['prompt_ids'],'Shared batch prompt differs')
        self.batch_names.add(path)
        return record

    def all_generations(self,runtime):
        expected={f'capability-{i}':(s,'base') for i,s in enumerate(self.schedule['capability_generation_seeds'])}
        for n in range(1,101):
            source='base' if n==1 else f'cycle{n-1}'
            expected.update({f'train-cycle{n}-{i}':(s,source) for i,s in enumerate(self.schedule['train_generation_seeds'][str(n)])})
        if self.endpoint:
            for label in ('base',f'cycle{self.endpoint}'):
                expected.update({f'eval-{label}-{i}':(s,label) for i,s in enumerate(self.schedule['eval_generation_seeds'])})
        for relative in self.artifacts:
            if relative.startswith('generations/'):
                name=Path(relative).stem;require(name in expected and relative==f'generations/{name}.json','Unexpected extra controller draw')
                seed,label=expected[name]
                if name.startswith('train-'):require(self.pool is not None,'Training before admission')
                if name.startswith('eval-'):require(self.checkpoint(f'cycle{self.endpoint}') is not None,'Evaluation before fixed endpoint')
                record=self.generation(name,seed,label)
                self.classifications[name]=runtime.classify(record['text'],record['finish_reason'])[0]
        # A committed batch may precede materialization of its per-record journals.
        for relative in self.artifacts:
            if relative.startswith('batches/'):
                batch=self.artifact(relative);require(len(batch['records'])==len(batch['names'])==len(batch['seeds'])==8,'Partial batch')
                batch_id=Path(relative).stem
                require(batch['batch_id']==batch_id and self.checkpoint(batch['checkpoint']) is not None,'Batch source identity differs')
                require(batch['source_metadata_sha256']==self.artifacts[f"checkpoints/{batch['checkpoint']}/metadata.json"]
                    and batch['prompt_condition']=='cardinal' and batch['prompt_sha256']==self.manifest['prompt_sha256']['cardinal'],'Batch context differs')
                if batch_id=='capability':names=[f'capability-{i}' for i in range(8)]
                elif batch_id.startswith('train-'):
                    n=int(batch_id[6:]);names=[f'train-cycle{n}-{i}' for i in range(8)]
                else:
                    require(batch_id.startswith('eval-'),'Unknown batch stage')
                    prefix,block=batch_id.rsplit('-',1);block=int(block);require(0<=block<4,'Unknown eval batch block')
                    names=[f'{prefix}-{i}' for i in range(block*8,block*8+8)]
                same(batch['names'],names,'Batch fixed request order differs')
                for name in names:
                    require(name not in self.raw_batch_draws,'One controller draw appears in two committed batches')
                    self.raw_batch_draws[name]=relative
                for i,(name,seed,raw) in enumerate(zip(batch['names'],batch['seeds'],batch['records'])):
                    require(name in expected and expected[name]==(seed,batch['checkpoint']) and raw['seed']==seed,'Unregistered raw batch draw')
                    require(raw.get('sampling_batch_row',i)==i and raw.get('sampling_batch_size',8)==8,'Raw batch row index differs')
                    wrapped={**raw,'name':name,'checkpoint':batch['checkpoint'],'model_repo':self.config['model_repo'],
                        'model_revision':self.config['model_revision'],'prompt_sha256':batch['prompt_sha256']}
                    previous.validate_generation(wrapped,name=name,seed=seed,checkpoint=batch['checkpoint'],
                        prompt_sha256=self.manifest['prompt_sha256']['cardinal'],config=self.config)
                    same(raw['prompt_ids'],batch['records'][0]['prompt_ids'],'Raw batch prompt differs')
                    require(raw['completion_ids'],'Empty committed raw completion')
                    self.actual_context(name,raw)

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
                    cardinal.validate_recorded_gate(gate,records,list(hashes),self.config)
            valid_returns=[r['game_return'] for r in scores['rows'] if r['runtime_valid']]
            self.group_reports.append({'cycle':n,'checkpoint_committed':meta is not None,
                'gradient_signal':'no_update' if zero else 'validity_only_signal' if len(set(valid_returns))<=1 else 'gameplay_and_validity_signal',
                'rewards':rewards,'advantages':adv,**previous.summarize_programs(scores['rows'],8)})
            self.training_rows+=scores['rows']
            if meta is None:continue
            update=meta['report'];last_complete=n
            for key,value in {'source_checkpoint':source,'source_metadata_sha256':scores['source_metadata_sha256'],
                'scores_sha256':self.artifacts[folder+'/scores.json'],'generation_sha256':hashes,'rewards':rewards,'advantages':adv,
                'training_tokens':sum(len(r['completion_ids']) for r in records),'all_runtime_invalid':not any(r['runtime_valid'] for r in scores['rows'])}.items():
                same(update[key],value,'Update lineage/content differs: '+key)
            require(update['optimizer_steps']==int(not zero) and update['outcome']==('skipped_zero_variance' if zero else 'updated'),'Update/skip differs')
            if zero:require(update['likelihood_gate_path'] is None and update['likelihood_gate_sha256'] is None,'Skip claims gradient gate')
            else:
                gatepath=update['likelihood_gate_path'];require(gatepath.startswith(folder+'/gate-') and
                    update['likelihood_gate_sha256']==self.artifacts[gatepath],'Successful gate unbound to update')
                previous.validate_gate(self.artifact(gatepath),records,list(hashes),self.config)
                previous.validate_gate(update['gradient_likelihood_gate'],records,list(hashes),self.config)
                require(math.isfinite(update['gradient_norm_before_clip']) and update['gradient_norm_before_clip']>0,'Invalid gradient norm')
                require(len(update['microbatch_losses'])==8 and all(math.isfinite(x) for x in update['microbatch_losses']),'Loss coverage differs')
                same(math.fsum(update['microbatch_losses']),update['loss'],'Loss sum differs')
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
        hashes=selection['profile_update_sha256'];count=len(hashes)
        require(list(hashes)==[str(i) for i in range(1,count+1)] and 1<=count<=3,'Profiling prefix differs')
        timings=[]
        for i in range(1,count+1):
            require(hashes[str(i)]==self.artifacts[f'groups/{i}/update.json'] and
                selection['profile_timing_sha256'][str(i)]==self.artifacts[f'groups/{i}/timing.json'],'Timing selection binding differs')
            timings.append(self.artifact(f'groups/{i}/timing.json'))
        actual=[r['optimizer_steps'] for r in timings]
        require(not any(actual[:-1]) and (actual[-1]==1 or count==3),'Profile did not stop at first true update/three attempts')
        if not any(actual):
            require(not selection['passed'] and selection['endpoint'] is None,'No-update profiling admitted production');return
        budget=self.status['execution_budget']
        admitted=(datetime.fromisoformat(budget['hard_stop'])-datetime.fromisoformat(budget['first_started'])).total_seconds()
        require(0<=selection['remaining_seconds']<=min(self.config['worker_seconds'],admitted)-sum(r['elapsed_seconds'] for r in timings)+1e-6,
            'Claimed remaining projection exceeds durable elapsed-time bound')
        require(selection['profile_groups']==count and selection['safety_factor']==1.5 and selection['reserved_final_and_export_seconds']==4200,
            'Profiling summary/reserve differs')
        endpoint=independently_select_endpoint(timings,selection['remaining_seconds'],self.config)
        require(selection['endpoint']==endpoint and selection['passed']==(endpoint is not None),'Timing-only endpoint choice differs')
        same(selection['maximum_measured_group_seconds'],max(r['elapsed_seconds'] for r in timings),'Measured throughput differs')
        for n in (50,100):
            same(selection['projected_remaining_seconds'][str(n)],1.5*max(r['elapsed_seconds'] for r in timings)*(n-count)+4200,'Time projection differs')
        self.endpoint=endpoint

    def diagnostics(self):
        runtime=self.artifact('runtime.json',False);measurement=self.artifact('prompt-measurement.json',False)
        if runtime:
            require(runtime['cuda']=='12.8' and runtime['loader']=='image_text_to_text','CUDA/runtime loader differs')
            require(runtime['arithmetic']=='NF4 frozen base; FP32 compute/LoRA; TF32 off; eager attention','Recorded arithmetic differs')
            schema=runtime['parameter_schema']
            same([r['name'] for r in schema],runtime['trainable_names'],'Runtime parameter order differs')
            require(all(r['dtype']=='torch.float32' and r['numel']==math.prod(r['shape']) and r['numel']>0 for r in schema)
                and sum(r['numel'] for r in schema)==runtime['trainable_parameters'],'Runtime trainable schema/count differs')
            require(not set(runtime['trainable_names'])&set(runtime['frozen_state']['tensors']),'Frozen/trainable identities overlap')
            for name,version in self.manifest['expected_installed_versions'].items():
                if name=='safetensors':continue
                require(runtime['versions'][name].split('+')[0]==version,'Installed version differs: '+name)
            index=runtime['model_index'];require(index['repo']==self.config['model_repo'] and index['revision']==self.config['model_revision']
                and index['authoritative_metadata_sha256']==self.manifest['input_sha256']['inputs/model-info.json'],'Model provenance differs')
            info=read(self.run/'inputs/model-info.json')
            for row in info['siblings']:
                if row['rfilename'].endswith('.safetensors'):require(index['files'].get(row['rfilename'])==row['lfs']['sha256'],'Pinned model weight hash differs')
            frozen=runtime['frozen_state'];require(previous.value_sha(frozen['tensors'])==frozen['sha256'],'Frozen tensor fingerprint digest differs')
            quant={n:v for n,v in frozen['tensors'].items() if n in frozen['quantization_state_tensors']}
            require(previous.value_sha(quant)==frozen['quantization_state_sha256'],'Quantization fingerprint differs')
        if measurement:
            require(measurement['prompt_tokens']>0 and measurement['completion_cap']==2048 and measurement['context_cap']==4096
                and measurement['admitted']==(measurement['prompt_tokens']+2048<=4096),'Prompt admission differs')
        passed=False
        for path in self.artifacts:
            if path.startswith('smoke-'):
                smoke=self.artifact(path)
                require(smoke['training_use'] is False and smoke['optimizer_steps']==0 and smoke['source_checkpoint']=='base' and
                    smoke['source_metadata_sha256']==self.artifacts['checkpoints/base/metadata.json'],'Smoke provenance differs')
                if not smoke['passed']:continue
                require(all(smoke[k] is True for k in ('adapter_unchanged','optimizer_unchanged','rng_restored','config_restored')),'Smoke altered policy/optimizer')
                require(smoke['optimizer_state_before_sha256']==smoke['optimizer_state_after_sha256'],'Smoke changed Adam')
                records=smoke['batch']['records'];require(len(records)==8 and all(0<len(r['completion_ids'])<=32 for r in records),'Smoke sample scope differs')
                for i,record in enumerate(records):self.actual_context(f'{path}/row-{i}',record)
                for key in ('eval_likelihood_gate','train_likelihood_gate'):
                    gate=dict(smoke[key]);gate['generation_names']=[];previous.validate_gate(gate,records,[],self.config)
                require(smoke['derivative']['finite_nonzero'] is True and math.isfinite(smoke['derivative']['gradient_l2_norm'])
                    and smoke['derivative']['gradient_l2_norm']>0 and not smoke['derivative']['missing_gradient_names'],'Smoke derivative differs')
                passed=True
        if any(p.startswith('generations/') for p in self.artifacts):require(passed,'Authoring began without successful numerical admission')

    def evaluation(self,runtime,pool):
        if self.endpoint is None:
            require(not any(p.startswith('evaluation/') for p in self.artifacts),'Evaluation without timing-selected endpoint');return
        endpoints=('base',f'cycle{self.endpoint}');seal=self.artifact('evaluation/sealed.json',False)
        if seal:
            require(all(self.checkpoint(e) for e in endpoints),'Premature final evaluation')
            paths=[f'generations/eval-{e}-{i}.json' for e in endpoints for i in range(32)]
            expected={'checkpoints':{e:self.artifacts[f'checkpoints/{e}/metadata.json'] for e in endpoints},
                'generations':{p:self.artifacts.get(p) for p in paths},'draws':64,'scoring_allowed':True,
                'endpoint_selection_sha256':self.artifacts['endpoint-selection.json']}
            require(all(expected['generations'].values()),'Evaluation seal missing outputs');same(seal,expected,'Final64 seal differs')
        for label in endpoints:
            result=self.artifact(f'evaluation/{label}/holdout.json',False)
            if result is None:continue
            require(seal and result['checkpoint']==label and result['prompt_condition']=='cardinal' and result['training_use'] is False
                and result['split']=='holdout' and result['prescribed_program_draws']==len(result['scores'])==32,'Evaluation denominator differs')
            require(result['opponents_sha256']==self.artifacts['opponents.json'] and result['seal_sha256']==self.artifacts['evaluation/sealed.json'],'Holdout provenance differs')
            same(result['seeds'],self.schedule['holdout_maps'],'Holdout maps differ');same(result['opponents'],pool['selected'],'Heldout fixed opponent roster differs')
            for i,row in enumerate(result['scores']):
                name=f'eval-{label}-{i}';require(row['generation']==name and row['generation_sha256']==self.artifacts[f'generations/{name}.json'],'Holdout generation differs')
                self.scored_row(row,self.generations[name],pool['selected'],result['seeds'],runtime)
            self.evaluations[label]=previous.summarize_programs(result['scores'],32)

    def execute(self):
        families=self.frozen()
        with cardinal.archived_runtime(self.run,self.manifest) as runtime:
            require((self.run/'prompts/cardinal.txt').read_text()==runtime.model_spec(),'Clean cardinal prompt differs')
            self.maps(runtime,families);self.diagnostics();self.checkpoint('base')
            pool=self.capability_and_pool(runtime);self.groups(runtime,pool);self.selection()
            self.all_generations(runtime);self.evaluation(runtime,pool)
        complete=(self.endpoint in (50,100) and all(f'cycle{i}' in self.checkpoints for i in range(1,self.endpoint+1))
            and len(self.evaluations)==2 and len(self.generations)==8+8*self.endpoint+64)
        require(not self.status['claims_allowed'] or complete,'Partial run permits comparisons')
        require(self.status['state']!='complete' or complete,'Premature complete state')
        actual_steps=sum(m['report']['optimizer_steps'] for label,m in self.checkpoints.items() if label!='base')
        if self.status['state']=='complete':
            require(self.status['attempted_groups']==self.endpoint and self.status['actual_optimizer_steps']==actual_steps,'Final actual-step summary differs')
            require(set(self.leagues)==set(range(5,self.endpoint+1,5)),'Complete run omitted a scheduled league refresh')
            require(all(f'groups/{i}/timing.json' in self.artifacts for i in range(1,self.endpoint+1)),'Complete run omitted committed group timing')
        if self.status['state']=='capability_failed':
            require(self.capability is not None and not self.capability['passed'],'Capability-failed state lacks failed gate')
        if self.endpoint:
            require(not any(int(label[5:])>self.endpoint for label in self.checkpoints if label.startswith('cycle')),'Training beyond selected endpoint')
        unknown=set(self.artifacts)-self.allowed_artifacts;require(not unknown,'Unrecognized committed artifacts: '+str(sorted(unknown)))
        for path,digest in self.start_hashes.items():require(sha(self.run/path)==digest,'Run changed during audit: '+path)
        contrast=paired_yield(self.evaluations['base']['programs'],self.evaluations[f'cycle{self.endpoint}']['programs'],self.schedule['eval_generation_seeds']) if complete else None
        measurement=self.artifact('prompt-measurement.json',False)
        actual=next(iter(self.context_evidence.values())) if self.context_evidence else None
        errata=[]
        if actual is not None and measurement is not None and measurement['prompt_tokens']!=actual['actual_prompt_tokens']:
            errata.append({'kind':'non_authoritative_prompt_precheck_count','recorded_precheck_prompt_tokens':measurement['prompt_tokens'],
                'authoritative_prompt_tokens':actual['actual_prompt_tokens'],
                'reason':'Frozen precheck applies len() to apply_chat_template(tokenize=True); Transformers5.20.2 defaults return_dict=True, so this counts mapping keys. Backend render-then-encode records actual token IDs.',
                'impact':'Reporting/precheck metadata error. Each observed backend prompt plus the full2048-token reserve is independently checked; neural likelihood and gradient evidence is assessed separately.',
                'frozen_artifacts_modified':False})
        return {'format_version':1,'kind':'independent_cpu_cardinal_gpu_audit','run':str(self.run),
            'manifest_sha256':self.start_hashes['manifest.json'],'status_sha256':self.start_hashes['status.json'],
            'auditor_sha256':sha(Path(__file__)),'integrity_passed':True,'planned_study_complete':complete,
            'planned_comparisons_available':complete,'recorded_state':self.status['state'],'selected_endpoint':self.endpoint,
            'committed_generation_draws':len(self.generations),'committed_raw_batch_draws':len(self.raw_batch_draws),
            'batch_draws_pending_materialization':len(set(self.raw_batch_draws)-set(self.generations)),'verified_unique_game_replays':len(self.verified_replays),
            'genuine_optimizer_steps':actual_steps,
            'tensor_reports':self.tensor_reports,'training_groups':self.group_reports,'evaluation_cells':self.evaluations,
            'paired_yield':contrast,'context_admission':{'authoritative_record_count':len(self.context_evidence),
                'authoritative_backend_measurement':actual,'recorded_precheck':measurement,
                'evidence':self.context_evidence,'precheck_authoritative':False},'metadata_errata':errata,'capability':None if self.capability is None else {k:self.capability[k] for k in
                ('passed','productive_indices','distinct_productive_ir','selected_indices')},
            'limits':['One training seed; no guaranteed continual improvement.','Neural likelihoods are recorded-gate evidence, not independently rerun by this CPU audit.',
                'Policy baseline/quantization/adapter coverage differ from earlier local experiments.',
                'Endpoint selection records remaining allowance but no exact selection timestamp; budget consistency is bounded, not an independently reconstructed selection clock.']}


def audit(run):return Audit(run).execute()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=audit(args.run);text=previous.canonical(result)+'\n'
    if args.output:
        require(not args.output.exists(),'Audit output already exists; use a new path')
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    else:print(text,end='')


if __name__=='__main__':main()
