#!/usr/bin/env python3
"""Independent CPU replay, provenance, and checkpoint audit for FP32 arena-v2."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from selfplay import arena_v2,league
from scripts import analyze_arena as audit


def audit_checkpoints(run,manifest):
    from safetensors.numpy import load_file
    reports,metadata,weights={},{},{}
    for name in ('base','cycle1','cycle2','cycle3'):
        inherited=manifest['inherited_checkpoints'].get(name)
        directory=Path(inherited['path']) if inherited else run/'checkpoints'/name
        if not inherited and not league.has_complete_summary(directory):
            continue
        record,meta=league.checkpoint_record(directory,name,manifest['model'])
        if inherited:audit.equal(record,inherited,'Inherited policy identity changed: '+name)
        else:
            recorded=json.loads((run/'status.json').read_text())['stages'][f'train-{name}']['checkpoint']
            audit.equal(record,recorded,'Previously completed V2 checkpoint identity changed')
            audit.equal(meta['policy_arithmetic'],manifest['policy_arithmetic'],'New checkpoint arithmetic differs')
            audit.equal(meta['policy_arithmetic_sha256'],manifest['policy_arithmetic_sha256'],'Arithmetic fingerprint differs')
            verification=meta['arithmetic_verification']
            if not verification['integer_weights_exactly_unchanged'] or not verification['all_floating_parameters_float32']:
                raise ValueError('Runtime precision conversion was not verified')
        arrays=load_file(str(directory/'adapters.safetensors'))
        if not arrays or any(v.dtype!=np.float32 or not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError('Adapter tensors must be finite FP32')
        metadata[name],weights[name]=meta,arrays
        if metadata.get('base'):
            for key in ('labels','label_token_ids','trainable_parameters'):
                audit.equal(meta[key],metadata['base'][key],'Checkpoint interface changed: '+key)
        reports[name]={**record,'inherited':bool(inherited),'finite_float32_tensors':True,
            'tensor_count':len(arrays),'trainable_parameters':sum(v.size for v in arrays.values()),
            'policy_arithmetic_sha256':manifest['policy_arithmetic_sha256']}
        audit.equal(reports[name]['trainable_parameters'],meta['trainable_parameters'],'Adapter size differs')
        if name!='base':
            source='base' if name=='cycle1' else f'cycle{int(name[5:])-1}'
            prior=weights[source]
            audit.equal(sorted(arrays),sorted(prior),'Adapter tensor keys differ')
            if any(arrays[key].shape!=prior[key].shape for key in arrays):raise ValueError('Adapter shapes differ')
            delta=math.sqrt(sum(float(np.square(value.astype(np.float64)-prior[key]).sum()) for key,value in arrays.items()))
            if not math.isfinite(delta) or delta<=0:
                raise ValueError('No finite nonzero measured parameter update: '+name)
            summary=json.loads((directory/'summary.json').read_text())
            audit.close(summary['parameter_delta_l2'],delta,'Independent parameter delta differs',max(1e-8,delta*1e-5))
            reports[name].update(source_checkpoint=source,independently_measured_delta_l2=delta,training_summary=summary)
            if not inherited:
                cycle=int(name[5:]);rollout_path=run/f'train-rollouts-{cycle}.jsonl'
                training_stage=json.loads((run/'status.json').read_text())['stages'][f'train-{name}']
                audit.equal(training_stage['rollout_sha256'],audit.sha(rollout_path),
                            'Training rollout bytes differ from the completed update')
                episodes=league.complete_jsonl_rows(rollout_path)
                transitions=sum(len(e['transitions']) for e in episodes)
                audit.equal(summary['transitions'],transitions,'Training transition count differs')
                audit.equal(summary['episodes'],len(episodes),'Training episode count differs')
                audit.equal(summary['updates'],math.ceil(transitions/manifest['minibatch_size']),'Optimizer update count differs')
                metrics=league.complete_jsonl_rows(directory/'metrics.jsonl')
                audit.equal([r['update'] for r in metrics],list(range(1,summary['updates']+1)),'Optimizer metric sequence differs')
                if any(not math.isfinite(float(v)) for r in metrics for v in r.values()):raise ValueError('Nonfinite training metric')
                gate_path=run/f'likelihood-cycle{cycle}.json';gate=json.loads(gate_path.read_text())
                if gate.get('complete') is not True:raise ValueError('Completed update lacks full likelihood gate')
                audit.equal(gate['checkpoint'],source,'Gate used wrong source checkpoint')
                audit.equal(gate['batch_size'],manifest['minibatch_size'],'Gate batch size differs')
                audit.equal(gate['expected_rows'],transitions,'Gate transition count differs')
                expected={(e['episode_id'],r['turn'],r['player']) for e in episodes for r in e['transitions']}
                actual={(r['episode_id'],r['turn'],r['player']) for r in gate['rows']}
                audit.equal(len(gate['rows']),transitions,'Gate must contain every row once')
                audit.equal(sorted(actual),sorted(expected),'Gate identities differ from training data')
                if any(not math.isfinite(r['full_vector_error']) or r['full_vector_error']<0 or
                       r['full_vector_error']>manifest['likelihood_tolerance'] for r in gate['rows']):
                    raise ValueError('Completed update has a failed likelihood check')
                audit.close(gate['max_full_vector_error'],max(r['full_vector_error'] for r in gate['rows']),'Gate maximum differs',0)
                stage=json.loads((run/'status.json').read_text())['stages'][f'likelihood-cycle{cycle}']
                audit.equal(stage['sha256'],audit.sha(gate_path),'Likelihood evidence changed')
                reports[name]['likelihood_gate']={'rows':transitions,'max_full_vector_error':gate['max_full_vector_error'],
                                                   'sha256':audit.sha(gate_path)}
                reports[name]['optimization']={key:{'mean':audit.mean([r[key] for r in metrics]),'max':max(r[key] for r in metrics)}
                    for key in ('kl_to_rollout','entropy','clip_fraction','gradient_norm')}
    return reports,metadata


def analyze(run,allow_partial=False):
    run=Path(run).resolve();manifest=json.loads((run/'manifest.json').read_text())
    package,policy,experiment,sources=audit.load_archive(run,manifest,Path(__file__).resolve().parents[1])
    archived_protocol=importlib.import_module(package.__name__+'.arena_v2')
    audit.equal(manifest['rating_compatibility'],archived_protocol.COMPATIBILITY,'Unexpected V2 compatibility profile')
    audit.equal(manifest['policy_arithmetic'],archived_protocol.ARITHMETIC,'Unexpected V2 arithmetic')
    audit.equal(manifest['policy_arithmetic_sha256'],archived_protocol.ARITHMETIC_SHA,'V2 arithmetic hash differs')
    audit.equal(audit.sha(run/'precision-diagnostic.json'),manifest['precision_diagnostic_sha256'],'Precision diagnostic changed')
    archived_protocol.validate_diagnostic(json.loads((run/'precision-diagnostic.json').read_text()))
    audit.equal(audit.sha(run/'parent-analysis.json'),manifest['parent_analysis_sha256'],'Parent partial audit changed')
    schedule,rosters,initial=archived_protocol.make_schedule(manifest['environments'])
    audit.equal(json.loads((run/'schedule.json').read_text()),schedule,'V2 schedule changed')
    audit.equal(manifest['evaluation_rosters'],rosters,'V2 evaluation roster changed')
    audit.equal(manifest['initial_evaluation_stages'],initial,'V2 initial evaluation changed')
    expected={s['episode_id']:(stage,s) for stage,specs in schedule.items() for s in specs}
    if len(expected)!=sum(map(len,schedule.values())):raise ValueError('Duplicate scheduled identity')
    training_families={(s['environment'],s['seed']//2 if s['environment']=='chess' else s['seed'])
                      for stage,s in expected.values() if s['regime']=='train'}
    evaluation_families={(s['environment'],s['seed']//2 if s['environment']=='chess' else s['seed'])
                        for stage,s in expected.values() if s['regime']=='eval'}
    if training_families&evaluation_families:raise ValueError('V2 training/evaluation family overlap')
    checkpoints,metadata=audit_checkpoints(run,manifest)
    tokenizer=audit.load_tokenizer(manifest)
    for label,token in zip(metadata['base']['labels'],metadata['base']['label_token_ids'],strict=True):
        audit.equal(tokenizer.encode(label,add_special_tokens=False),[token],'Action label changed')
    for name in ('chess','python-chess'):
        audit.equal(importlib.metadata.version(name),manifest['versions'][name],'Simulator package version changed')
    rows=[];seen=set();transitions=0;files={}
    for stage in schedule:
        path=run/(stage+'.jsonl')
        if not path.exists():continue
        data=path.read_bytes()
        if data and not data.endswith(b'\n'):raise ValueError('Canonical rollout file has an incomplete row')
        stage_rows=[json.loads(line) for line in data.splitlines()]
        for row in stage_rows:
            identity=row['episode_id']
            if identity in seen or identity not in expected:raise ValueError('Unknown/duplicate episode')
            expected_stage,spec=expected[identity]
            audit.equal(stage,expected_stage,'Episode is in wrong stage file')
            for key,value in spec.items():audit.equal(row.get(key),value,'Episode differs from frozen schedule')
            transitions+=audit.replay(row,package,policy.action_prompt,metadata,tokenizer,manifest)
            rows.append(row);seen.add(identity)
        files[path.name]={'sha256':hashlib.sha256(data).hexdigest(),'episodes':len(stage_rows)}
    missing=sorted(set(expected)-seen)
    summary=json.loads((run/'summary.json').read_text()) if (run/'summary.json').exists() else None
    complete=not missing and summary is not None and summary.get('complete') is True and {'cycle2','cycle3'}<=set(checkpoints)
    if summary:
        audit.equal(summary['policy_arithmetic_sha256'],manifest['policy_arithmetic_sha256'],'Final arithmetic changed')
    if not allow_partial and not complete:raise ValueError(f'V2 remains incomplete; {len(missing)} scheduled episodes missing')
    evaluation,changes,macro=audit.evaluations(rows)
    return {'complete':complete,'all_completed_episodes_verified':True,'episodes_verified':len(rows),
        'policy_transitions_verified':transitions,'scheduled_episodes':len(expected),'missing_episode_ids':missing,
        'manifest_sha256':audit.sha(run/'manifest.json'),'source_verification':sources,'checkpoint_verification':checkpoints,
        'rollout_files':files,'train_eval_families_disjoint':True,'policy_arithmetic_sha256':manifest['policy_arithmetic_sha256'],
        'rating_compatibility':manifest['rating_compatibility'],'evaluation':evaluation,'paired_checkpoint_changes':changes,
        'macro_fixed_opponent_return':macro,'training_signal':audit.training_signals(rows),
        'warnings':['Fresh FP32 condition inherits base/cycle1 adapter bytes from an incomplete BF16 pilot; compare within V2.',
                    'Three independent scenario clusters per opponent/game; descriptive estimates have very low power.',
                    'Capped distance/material rewards are adjudications, not necessarily goal completions/checkmates.',
                    'CPU audit replays saved probabilities; independently hashed live all-row likelihood gates precede training.',
                    'Two cycles of action-policy RL do not establish unlimited continual improvement or general reasoning transfer.'],
        'analysis_source_sha256':audit.sha(__file__),'shared_analysis_source_sha256':audit.sha(audit.__file__)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('runs/arena-v2'))
    parser.add_argument('--out',type=Path)
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args();result=analyze(args.run,args.allow_partial)
    destination=args.out or args.run/'analysis.json';league.atomic_json(destination,result)
    print(json.dumps({'complete':result['complete'],'episodes_verified':result['episodes_verified'],
                      'policy_transitions_verified':result['policy_transitions_verified'],'out':str(destination.resolve())}))


if __name__=='__main__':main()
