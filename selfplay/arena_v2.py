"""Fresh FP32-arithmetic self-play after the preserved arena-v1 numerical stop.

The quantized integer weights and inherited adapter bytes are unchanged. All
floating model parameters use float32; fresh rollouts and baseline reevaluation
separate this condition from arena-v1's BF16 arithmetic. No external API is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import shutil
import uuid

import numpy as np

from . import league


ARITHMETIC={'floating_model_parameters':'float32','activation_arithmetic':'float32',
            'quantized_integer_storage':'unchanged uint32 packed 4-bit weights',
            'adapter_parameters':'float32','action_policy':'one-token legal categorical distribution',
            'temperature':1.0,'thinking':False}
ARITHMETIC_SHA=hashlib.sha256(league.normalized(ARITHMETIC).encode()).hexdigest()
COMPATIBILITY='arena-v2-fp32'
TOLERANCE=0.002
SOURCE_KEY='selfplay/arena_v2.py'


def validate_diagnostic(report,tolerance=TOLERANCE):
    league.same(report.get('checkpoint'),'cycle1','Precision diagnostic must use inherited cycle1')
    league.same(report.get('floating_parameter_dtype'),'float32','Diagnostic is not FP32')
    if report.get('quantized_integer_weights_unchanged') is not True:
        raise ValueError('Diagnostic must retain integer quantized storage')
    required={('eval',8),('eval',4),('eval',1),('train',4)}
    observed=set()
    for test in report.get('tests',[]):
        identity=(test.get('mode'),test.get('batch_size'))
        if identity not in required or identity in observed:
            raise ValueError('Unexpected or duplicate precision diagnostic condition')
        observed.add(identity)
        errors=test.get('errors',[])
        if test.get('rows')!=32 or len(errors)!=32 or any(not math.isfinite(x) or x<0 for x in errors):
            raise ValueError('Precision diagnostic must complete all 32 rows per condition')
        league.same(max(errors),test.get('max_full_vector_error'),'Diagnostic maximum mismatch')
        if max(errors)>tolerance:
            raise ValueError('FP32 diagnostic exceeds the unchanged likelihood tolerance')
    if observed!=required:
        raise ValueError('Precision diagnostic has not completed every required condition')
    return {'conditions':len(observed),'rows_per_condition':32,
            'max_full_vector_error':max(t['max_full_vector_error'] for t in report['tests'])}


def tensor_digest(value):
    array=np.array(value)
    hasher=hashlib.sha256()
    hasher.update(str(array.dtype).encode());hasher.update(str(array.shape).encode())
    hasher.update(array.tobytes())
    return hasher.hexdigest()


def verify_precision_parameters(before_integer,after,floating_keys):
    if set(after)-set(floating_keys)!=set(before_integer):
        raise ValueError('Integer tensor names changed during precision conversion')
    if any(str(after[key].dtype).split('.')[-1]!='float32' for key in floating_keys):
        raise ValueError('A floating model tensor remains outside float32')
    after_integer={key:tensor_digest(after[key]) for key in before_integer}
    league.same(after_integer,before_integer,'Integer quantized tensors changed during float conversion')
    return {'floating_tensor_count':len(floating_keys),'integer_tensor_count':len(before_integer),
            'integer_tensor_hashes_sha256':hashlib.sha256(league.normalized(after_integer).encode()).hexdigest(),
            'all_floating_parameters_float32':True,'integer_weights_exactly_unchanged':True}


def precision_policy(policy_module,model_config):
    """Live-runtime factory. Tests never invoke this model-loading function."""
    from mlx.utils import tree_flatten
    policy=policy_module.ActionPolicy(model_config)
    original=dict(tree_flatten(policy.model.parameters()))
    floating={key for key,value in original.items() if policy.mx.issubdtype(value.dtype,policy.mx.floating)}
    integer={key:tensor_digest(value) for key,value in original.items() if key not in floating}
    policy.model.set_dtype(policy.mx.float32)
    policy.mx.eval(policy.model.parameters())
    verification=verify_precision_parameters(integer,dict(tree_flatten(policy.model.parameters())),floating)
    policy.save_memory('base');policy.active='base'
    policy.metadata.update(policy_arithmetic=ARITHMETIC,policy_arithmetic_sha256=ARITHMETIC_SHA,
                           arithmetic_verification=verification,rating_compatibility=COMPATIBILITY)
    return policy


def make_schedule(environments):
    schedule,rosters={},{}
    def evaluation(candidate,opponent,iteration):
        specs=[]
        for game_index,game in enumerate(environments):
            for i in range(3):
                for seat in (0,1):
                    names=[opponent,opponent];names[seat]=candidate
                    specs.append({'environment':game,'seed':2900000+game_index*1000+i*2,
                        'regime':'eval','policies':names,'candidate':candidate,'opponent':opponent,
                        'candidate_seat':seat,'iteration':iteration,
                        'episode_id':f'arena-v2-eval-{candidate}-{opponent}-{game}-{i}-{seat}'})
        schedule[f'eval-{candidate}-{opponent}']=specs
    initial=[]
    for candidate in ('base','cycle1'):
        for opponent in ('heuristic','base'):
            evaluation(candidate,opponent,0 if candidate=='base' else 1)
            initial.append(f'eval-{candidate}-{opponent}')
    available=['base','cycle1']
    for cycle in (2,3):
        source,candidate=f'cycle{cycle-1}',f'cycle{cycle}'
        train=[]
        for game_index,game in enumerate(environments):
            for i in range(8):
                train.append({'environment':game,'seed':2800000+cycle*10000+game_index*1000+i*2,
                    'regime':'train','policies':[source,source],
                    'episode_id':f'arena-v2-train-{cycle}-{game}-{i}'})
        schedule[f'train-rollouts-{cycle}']=train
        rosters[candidate]=league.evaluation_roster(cycle,available)
        for opponent in rosters[candidate]:evaluation(candidate,opponent,cycle)
        available.append(candidate)
    return schedule,rosters,initial


def prepare(parent,out,diagnostic,deadline=league.DEFAULT_DEADLINE):
    parent,out,diagnostic=Path(parent).resolve(),Path(out).resolve(),Path(diagnostic).resolve()
    if datetime.fromisoformat(deadline).tzinfo is None:
        raise ValueError('Deadline must include a timezone')
    if out.exists():
        manifest=json.loads((out/'manifest.json').read_text())
        league.same(manifest['parent_run'],str(parent),'V2 parent changed')
        league.same(manifest['deadline'],deadline,'V2 deadline changed')
        league.same(manifest['policy_arithmetic'],ARITHMETIC,'V2 arithmetic changed')
        league.same(manifest['policy_arithmetic_sha256'],ARITHMETIC_SHA,'V2 arithmetic hash changed')
        for original,expected in manifest['source_sha256'].items():
            league.same(league.sha(out/'source'/Path(original).name),expected,'V2 archive changed: '+original)
        for original,path in ((SOURCE_KEY,Path(__file__)),('selfplay/league.py',Path(league.__file__))):
            league.same(league.sha(path),manifest['source_sha256'][original],'V2 runner changed; use frozen code')
        league.same(league.sha(parent/'manifest.json'),manifest['parent_manifest_sha256'],'V1 manifest changed')
        league.same(league.sha(out/'parent-analysis.json'),manifest['parent_analysis_sha256'],
                    'V2 archived parent analysis changed')
        league.same(league.sha(out/'precision-diagnostic.json'),manifest['precision_diagnostic_sha256'],'V2 diagnostic changed')
        validate_diagnostic(json.loads((out/'precision-diagnostic.json').read_text()))
        for name,expected in manifest['inherited_checkpoints'].items():
            actual,_=league.checkpoint_record(expected['path'],name,manifest['model'])
            league.same(actual,expected,'V2 inherited checkpoint changed: '+name)
        schedule,rosters,initial=make_schedule(manifest['environments'])
        league.same(schedule,json.loads((out/'schedule.json').read_text()),'V2 schedule changed')
        league.same(rosters,manifest['evaluation_rosters'],'V2 roster changed')
        league.same(initial,manifest['initial_evaluation_stages'],'V2 baseline stages changed')
        return manifest,schedule
    original=json.loads((parent/'manifest.json').read_text())
    audit=json.loads((parent/'analysis.json').read_text())
    if audit.get('all_completed_episodes_verified') is not True:
        raise ValueError('Parent needs a successful independent partial replay audit')
    league.same(audit['manifest_sha256'],league.sha(parent/'manifest.json'),'Parent audit manifest differs')
    inherited={name:league.checkpoint_record(parent/'checkpoints'/name,name,original['model'])[0]
               for name in ('base','cycle1')}
    for name,record in inherited.items():
        league.same(audit['checkpoint_verification'][name]['adapter_sha256'],record['adapter_sha256'],
                    'Parent adapter differs from partial audit: '+name)
    report=json.loads(diagnostic.read_text());diagnostic_summary=validate_diagnostic(report)
    sources={**original['source_sha256'],'selfplay/league.py':league.sha(league.__file__),SOURCE_KEY:league.sha(__file__)}
    schedule,rosters,initial=make_schedule(original['environments'])
    manifest={**original,'experiment':'arena-v2','rating_compatibility':COMPATIBILITY,
        'policy_arithmetic':ARITHMETIC,'policy_arithmetic_sha256':ARITHMETIC_SHA,
        'parent_run':str(parent),'parent_manifest_sha256':league.sha(parent/'manifest.json'),
        'parent_analysis_sha256':league.sha(parent/'analysis.json'),'parent_complete':audit.get('complete',False),
        'inherited_checkpoints':inherited,'continuation_cycles':[2,3],
        'initial_evaluation_stages':initial,'evaluation_rosters':rosters,'maximum_roster':8,
        'deadline':deadline,'source_sha256':sources,'rollout_chunk_episodes':6,
        'likelihood_tolerance':TOLERANCE,'likelihood_check':'Every legal-action probability in every training row before each update',
        'precision_diagnostic_sha256':league.sha(diagnostic),'precision_diagnostic':diagnostic_summary,
        'minibatch_size':4,'inference_batch_size':8,'external_spend_usd':0,
        'created_at':league.timestamp(),'training':'Fresh FP32 current-self rollouts; historical snapshots are evaluation only',
        'deviation':'V1 stopped before its second update because BF16 likelihoods depended on batch geometry. '
                    'V1 remains incomplete. V2 converts floating model tensors to FP32, reevaluates inherited '
                    'base/cycle1, and starts two fresh training cycles. No V1 rollout trains a V2 update.'}
    manifest['versions']={**original['versions'],'numpy':np.__version__}
    temporary=out.parent/('.'+out.name+'.preparing-'+uuid.uuid4().hex)
    temporary.mkdir(parents=True);(temporary/'source').mkdir();(temporary/'checkpoints').mkdir();(temporary/'rollout-parts').mkdir()
    for name,expected in sources.items():
        path=Path(__file__) if name==SOURCE_KEY else Path(league.__file__) if name=='selfplay/league.py' else parent/'source'/Path(name).name
        league.same(league.sha(path),expected,'V2 source differs: '+name)
        shutil.copyfile(path,temporary/'source'/Path(name).name)
    shutil.copyfile(diagnostic,temporary/'precision-diagnostic.json')
    shutil.copyfile(parent/'analysis.json',temporary/'parent-analysis.json')
    league.atomic_json(temporary/'manifest.json',manifest);league.atomic_json(temporary/'schedule.json',schedule)
    league.atomic_json(temporary/'status.json',{'status':'prepared','stages':{},'external_spend_usd':0})
    temporary.rename(out)
    return manifest,schedule


def probability_error(actual,old,pick):
    actual,old=np.array(actual,dtype=np.float64),np.array(old,dtype=np.float64)
    if actual.shape!=old.shape or not np.isfinite(actual).all() or not np.isfinite(old).all():
        raise ValueError('Invalid full categorical probability vectors')
    delta=actual-old
    return {'full_vector_error':float(np.max(np.abs(delta))),'selected_error':float(abs(delta[pick])),
            'kl_old_to_current':float(np.sum(np.exp(old)*(old-actual))),
            'selected_ratio':float(np.exp(delta[pick]))}


def full_likelihood_gate(policy,episodes,source,batch_size,seed,path,tolerance=TOLERANCE):
    policy.active=None;policy.activate(source);policy.model.train()
    rows=[dict(r,input_ids=policy.tokens(r['prompt']),episode_id=e['episode_id'])
          for e in episodes for r in e['transitions']]
    if not rows or any(r['checkpoint']!=source for r in rows):
        raise ValueError('Gate needs nonempty on-policy source rows')
    random.Random(seed).shuffle(rows)
    report={'complete':False,'checkpoint':source,'batch_size':batch_size,'seed':seed,
            'tolerance':tolerance,'expected_rows':len(rows),'rows':[],'policy_arithmetic_sha256':ARITHMETIC_SHA}
    for offset in range(0,len(rows),batch_size):
        batch=rows[offset:offset+batch_size]
        logps=policy.log_probs(policy.model,*policy.arrays(batch));policy.mx.eval(logps)
        for i,row in enumerate(batch):
            errors=probability_error(logps[i,:len(row['actions'])],row['old_log_probs'],row['action_index'])
            report['rows'].append({'episode_id':row['episode_id'],'turn':row['turn'],'player':row['player'],**errors})
        league.atomic_json(path,report)
        if any(row['full_vector_error']>tolerance for row in report['rows'][-len(batch):]):
            raise ValueError('Full-vector old-policy likelihood gate failed; no update permitted')
        if offset%(batch_size*8)==0:
            print(json.dumps({'likelihood_gate':source,'checked_rows':len(report['rows']),'expected_rows':len(rows)}),flush=True)
    report.update(complete=True,max_full_vector_error=max(r['full_vector_error'] for r in report['rows']),
                  max_selected_error=max(r['selected_error'] for r in report['rows']))
    league.atomic_json(path,report)
    return report


class PrecisionContinuation(league.Continuation):
    def _load_checkpoint(self,policy,name,directory):
        metadata=json.loads((Path(directory)/'metadata.json').read_text())
        if name not in self.manifest['inherited_checkpoints']:
            league.same(metadata['policy_arithmetic_sha256'],ARITHMETIC_SHA,'Checkpoint arithmetic identity differs')
            league.same(metadata['policy_arithmetic'],ARITHMETIC,'Checkpoint arithmetic profile differs')
        return super()._load_checkpoint(policy,name,directory)

    def training_stage(self,cycle,episodes):
        directory=self.out/'checkpoints'/f'cycle{cycle}'
        if league.has_complete_summary(directory):
            path=self.out/f'likelihood-cycle{cycle}.json'
            report=json.loads(path.read_text())
            if report.get('complete') is not True:
                raise ValueError('Completed V2 update lacks its completed likelihood gate')
            recorded=self.status['stages'][f'likelihood-cycle{cycle}']['sha256']
            league.same(league.sha(path),recorded,'Completed V2 likelihood evidence changed')
        else:
            if not self.admitted(f'likelihood-cycle{cycle}'):
                return False
            policy=self.get_policy()
            if not self.admitted(f'likelihood-cycle{cycle}'):
                return False
            path=self.out/f'likelihood-cycle{cycle}.json'
            self.record('checking_likelihood',f'likelihood-cycle{cycle}',expected_rows=sum(len(e['transitions']) for e in episodes))
            report=full_likelihood_gate(policy,episodes,f'cycle{cycle-1}',self.manifest['minibatch_size'],
                                        72020+cycle,path,self.manifest['likelihood_tolerance'])
            self.record('complete',f'likelihood-cycle{cycle}',sha256=league.sha(path),
                        checked_rows=len(report['rows']),max_full_vector_error=report['max_full_vector_error'])
        result=super().training_stage(cycle,episodes)
        if result:
            summary=json.loads((directory/'summary.json').read_text())
            expected=math.ceil(sum(len(e['transitions']) for e in episodes)/self.manifest['minibatch_size'])
            league.same(summary['updates'],expected,'V2 optimizer update count differs')
        return result

    def run(self):
        try:
            for stage in self.manifest['initial_evaluation_stages']:
                if self.rollout_stage(stage) is None:return self.status
            for cycle in self.manifest['continuation_cycles']:
                episodes=self.rollout_stage(f'train-rollouts-{cycle}')
                if episodes is None or not self.training_stage(cycle,episodes):return self.status
                for opponent in self.manifest['evaluation_rosters'][f'cycle{cycle}']:
                    if self.rollout_stage(f'eval-cycle{cycle}-{opponent}') is None:return self.status
            evaluations=[row for stage in self.schedule if stage.startswith('eval-')
                         for row in league.complete_jsonl_rows(self.out/(stage+'.jsonl'))]
            evaluation=self.experiment.summarize_eval(evaluations)
            league.atomic_json(self.out/'evaluation.json',evaluation)
            summary={'complete':True,'experiment':'arena-v2','training_cycles':[2,3],
                'parent_run':self.manifest['parent_run'],'parent_remains_incomplete':True,
                'rating_compatibility':COMPATIBILITY,'policy_arithmetic':ARITHMETIC,
                'policy_arithmetic_sha256':ARITHMETIC_SHA,'evaluation':evaluation,
                'training':[json.loads((self.out/'checkpoints'/f'cycle{c}'/'summary.json').read_text()) for c in (2,3)],
                'stage_records':self.status['stages'],'external_spend_usd':0,'finished_at':league.timestamp(),
                'limitation':'Small action-policy mechanism pilot; fresh FP32 condition, not a completed original BF16 experiment.'}
            league.atomic_json(self.out/'summary.json',summary)
            self.record('complete',None,training_cycles=[2,3])
            return self.status
        except Exception as error:
            self.record('failed',self.status.get('current_stage'),error=f'{type(error).__name__}: {error}')
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,default=Path('runs/arena-v1'))
    parser.add_argument('--out',type=Path,default=Path('runs/arena-v2'))
    parser.add_argument('--diagnostic',type=Path,default=Path('runs/arena-v1/recovery/diagnostic-precision.json'))
    parser.add_argument('--deadline',default=league.DEFAULT_DEADLINE)
    args=parser.parse_args()
    global_lock_target=Path(__file__).resolve().parents[1]/'runs/local-neural'
    with league.process_lock(global_lock_target),league.process_lock(args.out):
        manifest,schedule=prepare(args.parent,args.out,args.diagnostic,args.deadline)
        for name,expected in manifest['versions'].items():
            league.same(importlib.metadata.version(name),expected,'V2 runtime version changed: '+name)
        arena,policy,experiment=league.archived_arena(args.out,manifest)
        runner=PrecisionContinuation(args.out,manifest,schedule,arena,policy,experiment,
                                     policy_factory=lambda:precision_policy(policy,manifest['model']))
        status=runner.run()
    print(json.dumps({'status':status['status'],'run':str(args.out.resolve()),'external_spend_usd':0}),flush=True)


if __name__=='__main__':main()
