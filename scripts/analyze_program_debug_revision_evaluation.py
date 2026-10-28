"""Independent CPU audit of the separate base-versus-cycle3 evaluation closure.

No language model is loaded. Games, tokenization, tensor/Adam/RNG identities and
all paired denominators are independently checked. Recorded neural checks are
validated as evidence, not recomputed with a neural forward pass.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timedelta
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import analyze_program_debug_revision_rl as prior
from scripts import analyze_program_debug_revision_calibration as calibration
from scripts import analyze_program_debug_rl as debug
from scripts import analyze_program_study as previous
from selfplay.program_cardinal_cuda import checkpoint_state

read,sha,same,require,safe_path,value_sha=prior.read,prior.sha,prior.same,prior.require,prior.safe_path,prior.value_sha
ENDPOINTS=('base','cycle3')


def semantic(identity):return {k:v for k,v in identity.items() if k!='files'}


def validate_unchanged(proof,source,actual):
    same(proof['actual_identity'],actual,'Saved final-state identity differs from proof')
    equal=semantic(actual)==semantic(source)
    require(proof['unchanged_adapter_Adam_RNG'] is equal and proof['new_optimizer_steps']==0,
        'Unchanged endpoint proof mislabeled state transition')
    return equal


class Audit(prior.Audit):
    def __init__(self,run):
        super().__init__(run);self.endpoint=3;self.smokes={};self.state_proofs={}

    def frozen(self):
        m,cfg=self.manifest,self.config
        require(m['format_version']==1 and m['kind']=='mixed_native_revision_evaluation_only'
            and m['phase']=='coding_evaluation','Unknown evaluation-only protocol')
        require(self.start_hashes['manifest.json']==self.status['manifest_sha256'],'Manifest differs')
        for p,d in {**m['input_sha256'],**self.artifacts}.items():
            require(sha(safe_path(self.run,p))==d,'Artifact hash differs: '+p)
        for p,d in m['source_sha256'].items():
            require(sha(safe_path(self.run/'source',p))==d,'Archived source differs: '+p)
        helpers=('scripts/analyze_program_study.py','scripts/analyze_program_cardinal_cycle.py',
            'scripts/analyze_program_cardinal_gpu.py','scripts/analyze_program_debug_rl.py',
            'scripts/analyze_program_debug_revision_calibration.py','scripts/analyze_program_debug_revision_rl.py',
            'scripts/analyze_program_debug_revision_evaluation.py','selfplay/program_cardinal_cuda.py')
        for p in helpers:require(m['source_sha256'].get(p)==sha(ROOT/p),'Executed CPU helper differs: '+p)
        require({'selfplay/program_debug_revision_evaluation.py','scripts/cloud/run_revision_evaluation_job.py',
            'scripts/cloud/bootstrap_revision_evaluation.py','scripts/revision_evaluation_supervisor.py',
            'scripts/runpod_arena_mixed.py'}<=set(m['source_sha256']),'Evaluation source closure incomplete')
        parent=self.run/'parent';pm,ps,pa=read(parent/'manifest.json'),read(parent/'status.json'),read(parent/'final-audit.json')
        for name,key in [('manifest.json','parent_manifest_sha256'),('status.json','parent_status_sha256'),('final-audit.json','parent_audit_sha256')]:
            require(sha(parent/name)==m[key]==self.plan[key],'Parent evidence binding differs')
        # Revalidate the complete preserved scientific parent, including its
        # numerical reports, actual Adam transitions, maps and replay evidence.
        parent_result=prior.audit(parent)
        require(parent_result['integrity_passed'] and pa['integrity_passed'] and ps['state']=='throughput_failed'
            and parent_result['recorded_state']=='throughput_failed'
            and parent_result['planned_study_complete'] is False
            and parent_result['planned_comparisons_available'] is False
            and parent_result['genuine_RL_optimizer_steps']==pa['genuine_RL_optimizer_steps']==3
            and parent_result['selected_endpoint'] is None,'Parent is not the audited terminal three-step study')
        require(pa['manifest_sha256']==sha(parent/'manifest.json') and pa['status_sha256']==sha(parent/'status.json'),
            'Saved parent audit is stale')
        require(not any(p.startswith(('evaluation/','generations/eval-','batches/eval-','feedback/eval-')) for p in ps['artifacts']),
            'Parent evaluation was already observed')
        require(not any((parent/'evaluation').glob('**/*')) and
            not any(any((parent/f).glob('eval-*')) for f in ('generations','batches','feedback')),
            'Uncommitted parent evaluation data exists')
        same(self.schedule,read(parent/'schedule.json'),'Original unseen seed schedule changed')
        same(self.schedule,prior.registered_schedule(),'Registered schedule differs')
        for path in ('prompts/debug.txt','heldout-opponents.json','inputs/model-info.json'):
            require(sha(self.run/path)==sha(parent/path),'Original evaluation input changed: '+path)
        same(self.plan['config'],cfg,'Evaluation plan/config differs')
        overrides={'kind':'mixed_native_revision_evaluation_only','condition':'cycle3_evaluation_only_v1',
            'worker_seconds':35100,'evaluation_only':True,'training_allowed':False,
            'fixed_evaluation_checkpoints':['base','cycle3'],'new_optimizer_steps':0,
            'scope':'Separate paired evaluation of three existing RL updates beyond SFT32; original run stays incomplete'}
        for k,v in pm['config'].items():
            if k not in overrides and k!='deadline':same(cfg[k],v,'Parent scientific setting changed: '+k)
        for k,v in overrides.items():same(cfg[k],v,'Evaluation setting differs: '+k)
        require(set(cfg)==set(pm['config'])|set(overrides),'Unexpected evaluation config key')
        dep=read(self.run/'deployment.json')
        require(dep['phase']=='coding_evaluation' and dep['maximum_hours']==10 and dep['reservation_usd']==65
            and dep['combined_funding_usd']==240 and dep['intent']==dep['coding_evaluation_intent']
            and 0<dep['estimate_usd']<=65 and 0<dep['maximum_gpu_rate_usd_hour']<=5.29
            and 0<dep['maximum_cpu_rate_usd_hour']<=.14,'Evaluation lease/funding differs')
        require(datetime.fromisoformat(cfg['deadline']).timestamp()==dep['deadline_unix'],'Deadline differs')
        budget=self.status.get('execution_budget')
        if budget:
            first,stop=datetime.fromisoformat(budget['first_started']),datetime.fromisoformat(budget['hard_stop'])
            require(stop==min(first+timedelta(seconds=35100),datetime.fromisoformat(cfg['deadline']))
                and budget['maximum_seconds']==35100 and budget['resume_counts_wall_time'] is True,'Persistent worker budget differs')
            if self.status['state']=='complete':require(datetime.fromisoformat(self.status['updated'])<=stop,'Completion after deadline')
        else:require(self.status['state']=='prepared','Execution budget missing')
        for k,v in {'fixed_endpoints':['base','cycle3'],'paired_authoring_seeds':16,'trajectories':32,
            'sampled_turns':64,'matches_per_compiled_final':48,'heldout_maps':6,'opponents':4,'seats':2}.items():
            same(self.plan[k],v,'Frozen evaluation denominator differs')
        require(m['evaluation_plan_sha256']==sha(self.run/'plan.json') and m['new_optimizer_steps']==0
            and m['parent_training_complete'] is False,'Evaluation scope changed')
        self.parent_manifest,self.parent_status,self.parent_result=pm,ps,parent_result
        self.tokenizer,self.eos=debug.load_original_tokenizer(self.run,m)
        self.heldout_pool=read(self.run/'heldout-opponents.json')
        return parent_result

    def checkpoint(self,label):
        require(label in ENDPOINTS,'Unregistered evaluation checkpoint')
        if label in self.checkpoints:return self.checkpoints[label]
        meta=self.artifact(f'checkpoints/{label}/metadata.json')
        binding=self.manifest['inherited_checkpoints'][label]
        require(meta['manifest_sha256']==binding['origin_manifest_sha256']==self.manifest['parent_manifest_sha256']
            and meta['label']==label and meta['complete'] is True,'Inherited origin metadata differs')
        require(self.artifacts[f'checkpoints/{label}/metadata.json']==binding['metadata_sha256']
            ==self.parent_status['artifacts'][f'checkpoints/{label}/metadata.json'],'Inherited metadata bytes differ')
        same(meta,read(self.run/f'parent/checkpoints/{label}/metadata.json'),'Inherited metadata rewritten')
        for name,d in meta['files'].items():
            p=f'checkpoints/{label}/{name}';self.allowed_artifacts.add(p)
            require(self.artifacts[p]==d==sha(self.run/p)==sha(self.run/'parent'/p),'Inherited checkpoint bytes differ')
        identity=checkpoint_state(self.run/'checkpoints'/label,1e-5)
        same(identity,meta['optimizer_identity'],'Actual inherited checkpoint identity differs')
        same(identity,binding['optimizer_identity'],'Manifest inherited identity differs')
        require(identity['step']==(0 if label=='base' else 3),'Inherited actual Adam step differs')
        self.checkpoints[label]=meta;self.tensor_reports[label]={'actual_optimizer_step':identity['step'],
            'actual_learning_rate':identity['learning_rate'],'origin_manifest_sha256':meta['manifest_sha256'],
            'unchanged_import':True}
        return meta

    def batch_specs(self):
        for label in ENDPOINTS:
            for offset in range(0,16,2):
                yield f'eval-{label}-{offset//2}',[f'eval-{label}-{i}' for i in range(offset,offset+2)], \
                    self.schedule['eval_generation_seeds'][offset:offset+2],label, \
                    [self.schedule['eval_practice_maps'][str(i)] for i in range(offset,offset+2)]

    def diagnostics(self):
        runtime=self.artifact('runtime.json',False)
        if runtime:
            require(runtime['evaluation_only'] is True and runtime['training_allowed'] is False and runtime['new_optimizer_steps']==0,
                'Runtime permits training')
            parent_runtime=read(self.run/'parent/runtime.json')
            for key in ('cuda','loader','arithmetic','turn_native_modes','trainable_names','parameter_schema',
                'trainable_parameters','frozen_state','eos_token_ids','model_index'):
                same(runtime[key],parent_runtime[key],'Original runtime/model identity changed: '+key)
            require(runtime['authoring_calls']==2 and runtime['calibration_only'] is False,'Runtime harness differs')
            for name,version in self.manifest['expected_installed_versions'].items():
                require(runtime['versions'][name].split('+')[0]==version,'Installed numerical version differs')
        for path in self.artifacts:
            if not path.startswith('smoke/'):continue
            row=self.artifact(path);label=row['source_checkpoint']
            require(label in ENDPOINTS and runtime is not None,'Unregistered/no-runtime endpoint smoke')
            require(row['kind']=='revision_mixed_native_derivative_admission' and row['evaluation_only'] is True
                and row['training_use'] is False and row['optimizer_steps']==0 and row['batch_size']==2
                and row['max_new_tokens']==32 and row['authoring_stages']==['draft','revision']
                and row['source_metadata_sha256']==self.artifacts[f'checkpoints/{label}/metadata.json'],
                'Mixed-native numerical admission scope differs')
            same(row['seeds'],[70000000+self.config['seed']+i for i in range(2)],'Smoke seeds differ')
            same(row['token_tolerance'],.002,'Smoke token tolerance differs');same(row['sequence_tolerance'],.02,'Smoke sequence tolerance differs')
            identity=self.checkpoint(label)['optimizer_identity']
            restored=calibration.validate_restoration(row,identity,self.config)
            vectors=False
            if 'batch' in row:
                records=row['batch']['records']
                require(len(records)==2 and len({r['input_tokens'] for r in records})>1,'Mixed-length smoke coverage differs')
                calibration.validate_cache_timing(row['batch'],['draft','revision'])
                for i,r in enumerate(records):
                    calibration.validate_turn(r,self.config,stage=('draft','revision')[i],seed=row['seeds'][i],maximum=32)
                    prior.validate_tokenization(r,self.tokenizer,self.eos,smoke=True)
                    self.actual_context(path+'/'+str(i),r)
                if 'records' in row:vectors=calibration.validate_full_vectors(row,records,self.config)['passed']
            derivative=row.get('derivative',{});norm=derivative.get('gradient_l2_norm')
            finite=type(norm) in (int,float) and math.isfinite(norm) and norm>0 and derivative.get('finite_nonzero') is True and derivative.get('missing_gradient_names')==[]
            require(row['passed'] is (restored and vectors and finite),'Numerical smoke result differs')
            self.smokes[path]={'passed':row['passed'],'checkpoint':label}
        for label in ENDPOINTS:
            if any(p.startswith(f'batches/eval-{label}-') for p in self.artifacts):
                require(any(v['passed'] and v['checkpoint']==label for v in self.smokes.values()),'Endpoint sampled without passed fresh smoke')

    def endpoint_states(self):
        for label in ENDPOINTS:
            path=f'endpoint-state/{label}/proof.json';row=self.artifact(path,False)
            if row is None:continue
            require(row['checkpoint']==label and row['source_metadata_sha256']==self.artifacts[f'checkpoints/{label}/metadata.json'],
                'Endpoint unchanged proof source differs')
            candidates=[p for p in self.smokes if self.artifacts[p]==row['smoke_sha256']]
            require(len(candidates)==1 and self.smokes[candidates[0]]=={'passed':True,'checkpoint':label},'Endpoint proof lacks its passed smoke')
            expected={f'generations/eval-{label}-{i}.json':self.artifacts.get(f'generations/eval-{label}-{i}.json') for i in range(16)}
            require(all(expected.values()),'Missing authoring cannot become endpoint state proof')
            same(row['generations'],expected,'Endpoint state proof draw binding differs')
            folder=self.run/'endpoint-state'/label/'snapshot'
            for name,d in row['files'].items():
                relative=f'endpoint-state/{label}/snapshot/{name}';self.allowed_artifacts.add(relative)
                require(self.artifacts[relative]==d==sha(folder/name),'Saved final state bytes differ')
            actual=checkpoint_state(folder,1e-5)
            valid=validate_unchanged(row,self.checkpoint(label)['optimizer_identity'],actual)
            self.state_proofs[label]={'unchanged_adapter_adam_rng':valid,'actual_optimizer_step':actual['step']}

    def admissions(self):
        known={f'batches/eval-{label}-{i}-{stage}.json' for label in ENDPOINTS for i in range(8) for stage in ('draft','revision')}
        for p in self.artifacts:
            if not p.startswith('admissions/'):continue
            row=self.artifact(p);paths=row['pending_batch_paths']
            require(len(paths)==len(set(paths)) and set(paths)<=known,'Unknown or duplicate remaining sampling batch')
            remaining=prior.remaining_at(self.status,row['checked_at'])
            same(row['remaining_seconds'],remaining,'Admission remaining time differs')
            required=len(paths)*900+3600
            require(row['sampling_seconds_per_batch']==900 and row['scoring_and_export_reserve_seconds']==3600
                and row['required_seconds']==required and row['passed'] is (remaining>=required),'Remaining fixed evaluation reserve differs')

    def evaluation(self):
        seal=self.artifact('evaluation/sealed.json',False)
        if seal:
            paths=[f'generations/eval-{label}-{i}.json' for label in ENDPOINTS for i in range(16)]
            require(len(self.generations)==32 and len(self.raw_turns)==64,'Missing heldout outputs cannot become zero outcomes')
            require(all(self.state_proofs.get(e,{}).get('unchanged_adapter_adam_rng') for e in ENDPOINTS),'Endpoint state changed')
            expected={'checkpoints':{e:self.artifacts[f'checkpoints/{e}/metadata.json'] for e in ENDPOINTS},
                'generations':{p:self.artifacts[p] for p in paths},'draws':32,'sampled_turns':64,
                'endpoint_state_proofs':{e:self.artifacts[f'endpoint-state/{e}/proof.json'] for e in ENDPOINTS},
                'evaluation_plan_sha256':self.manifest['evaluation_plan_sha256'],'scoring_allowed':True,'new_optimizer_steps':0}
            same(seal,expected,'All32 authoring seal differs')
        for label in ENDPOINTS:
            result=self.artifact(f'evaluation/{label}/holdout.json',False)
            if result is None:continue
            require(seal and result['checkpoint']==label and result['prompt_condition']=='native_nonthinking_revision_rl'
                and result['training_use'] is False and result['split']=='holdout'
                and result['prescribed_program_draws']==len(result['scores'])==16,'Heldout denominator differs')
            require(result['opponents_sha256']==self.manifest['input_sha256']['heldout-opponents.json']
                and result['seal_sha256']==self.artifacts['evaluation/sealed.json'],'Evaluation source binding differs')
            same(result['seeds'],self.schedule['holdout_maps'],'Heldout map schedule changed')
            same(result['opponents'],self.heldout_pool['selected'],'Fixed heldout opponents changed')
            for i,row in enumerate(result['scores']):
                name=f'eval-{label}-{i}'
                require(row['generation']==name and row['generation_sha256']==self.artifacts[f'generations/{name}.json'],'Scored output differs')
                self.scored_row(row,self.generations[name],self.heldout_pool['selected'],result['seeds'],self.runtime)
            self.evaluations[label]=previous.summarize_programs(result['scores'],16)

    def execute(self):
        self.frozen()
        for label in ENDPOINTS:self.checkpoint(label)
        with calibration.archived_runtime(self.run,self.manifest) as runtime:
            self.runtime=runtime
            require((self.run/'prompts/debug.txt').read_text()==runtime.debug.debug_model_spec(),'Frozen native prompt changed')
            same(self.config['history_transform'],runtime.debug.REVISION_CONTEXT_POLICY,'History transform changed')
            self.diagnostics();self.all_batches();self.all_generations(runtime);self.endpoint_states();self.admissions();self.evaluation()
        complete=self.status['state']=='complete' and len(self.evaluations)==2 and len(self.generations)==32 and len(self.raw_turns)==64
        require(not self.status['claims_allowed'] or complete,'Incomplete measurement permits comparison')
        require(self.status['state']!='complete' or complete,'Premature complete state')
        require(self.status['new_optimizer_steps']==0 and self.status['inherited_optimizer_steps']==3,'Optimizer scope mislabeled')
        if complete:
            require(self.status['parent_training_complete'] is False and self.status['final_checkpoint']=='cycle3'
                and self.status['learning_comparison_available'] is True,'Terminal comparison scope differs')
        unknown=set(self.artifacts)-self.allowed_artifacts
        require(not unknown,'Unrecognized committed artifacts: '+str(sorted(unknown)))
        for p,d in self.start_hashes.items():require(sha(self.run/p)==d,'Run changed during audit: '+p)
        paired=prior.paired_summary(self.evaluations['base']['programs'],self.evaluations['cycle3']['programs'],
            self.schedule['eval_generation_seeds']) if complete else None
        return {'format_version':1,'kind':'independent_cpu_revision_evaluation_audit','run':str(self.run),
            'auditor_sha256':sha(Path(__file__)),'manifest_sha256':self.start_hashes['manifest.json'],
            'status_sha256':self.start_hashes['status.json'],'integrity_passed':True,'recorded_state':self.status['state'],
            'planned_study_complete':complete,'planned_comparisons_available':complete,
            'parent_training_complete':False,'parent_manifest_sha256':self.manifest['parent_manifest_sha256'],
            'inherited_RL_optimizer_steps':3,'new_supervised_optimizer_steps':0,'new_RL_optimizer_steps':0,
            'fixed_final_checkpoint':'cycle3','committed_trajectories':len(self.generations),
            'committed_raw_sampled_turns':len(self.raw_turns),'raw_turns_pending_trajectory_materialization':sum(k[0] not in self.generations for k in self.raw_turns),
            'verified_unique_game_replays':len(self.verified_replays),'practice_feedback_records':len(self.feedback_checked),
            'tensor_reports':self.tensor_reports,'endpoint_state_proofs':self.state_proofs,'numerical_smokes':self.smokes,
            'evaluation_cells':self.evaluations,'paired_yield':paired,'context_admission':self.context_evidence,
            'limits':['Separate measurement of three inherited RL updates, not completion of the original timing-selected study.',
                'One training seed and16paired authoring clusters; turns and replaygames are dependent.',
                'Invalid final authorship retains zero yield and no invented games; missing authoring is incomplete.',
                'CPU independently checks tokenization/replays/tensors and consistency of recorded neural gates; no neural forward or derivative is rerun.',
                'Original evaluation uses fresh mixed-native32-token smokes, not full64actual-turn likelihood replay.',
                'Two frozen teacher opponents were prior SFT examples, reserved from RL opponents but not unseen to weights.']}


def audit(run):return Audit(run).execute()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path)
    args=p.parse_args();text=previous.canonical(audit(args.run))+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as f:f.write(text)
    else:print(text,end='')


if __name__=='__main__':main()
