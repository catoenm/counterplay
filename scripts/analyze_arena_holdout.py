"""CPU replay audit and paired comparisons for the fixed final holdout."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from selfplay import league
from scripts import analyze_arena as audit


def canonical_rows(run, stage, specs, status, complete=False):
    """Verify canonical ordering and the exact completed-stage byte commitment."""
    path = run/(stage+'.jsonl')
    evidence = status.get('stages',{}).get(stage,{})
    if not path.exists():
        if complete or evidence.get('status') == 'complete':
            raise ValueError('Completed holdout stage is missing its canonical file')
        return [],None
    data = path.read_bytes()
    if data and not data.endswith(b'\n'):
        raise ValueError('Canonical holdout file has incomplete trailing row')
    rows = [json.loads(line) for line in data.splitlines()]
    ids = [row['episode_id'] for row in rows]
    seen = set(ids)
    if len(ids) != len(seen) or ids != [s['episode_id'] for s in specs if s['episode_id'] in seen]:
        raise ValueError('Canonical holdout rows are unknown, duplicated, or out of schedule order')
    if complete or evidence.get('status') == 'complete':
        audit.equal(evidence.get('status'),'complete','Completed holdout lacks completed stage evidence')
        audit.equal(evidence.get('sha256'),audit.sha(path),'Completed holdout rollout bytes changed')
        audit.equal(evidence.get('completed_episodes'),len(specs),'Completed holdout stage count differs')
        audit.equal(len(rows),len(specs),'Completed holdout file is missing episodes')
    return rows,{'sha256':audit.sha(path),'episodes':len(rows)}


def analyze(run, allow_partial=False):
    run = Path(run).resolve()
    manifest = json.loads((run/'manifest.json').read_text())
    arena, policy, _, sources = audit.load_archive(run,manifest,Path(__file__).resolve().parents[1])
    protocol = importlib.import_module(arena.__name__+'.arena_holdout')
    for key,path in (('scripts/analyze_arena_holdout.py',Path(__file__)),
                     ('scripts/analyze_arena.py',Path(audit.__file__))):
        audit.equal(audit.sha(path),manifest['source_sha256'][key],'Live audit source differs from preregistered archive')
    audit.equal(manifest['holdout_protocol'],protocol.PROTOCOL,'Holdout protocol differs')
    audit.equal(manifest['continuation_cycles'],[],'Holdout must not train')
    audit.equal(manifest['rollout_chunk_episodes'],4,'Holdout candidate pairing chunk differs')
    schedule = protocol.make_schedule(manifest['environments'])
    audit.equal(json.loads((run/'schedule.json').read_text()),schedule,'Holdout schedule differs')
    audit.equal(audit.sha(run/'parent-analysis.json'),manifest['parent_analysis_sha256'],'Parent audit changed')
    parent_audit = json.loads((run/'parent-analysis.json').read_text())
    if not parent_audit['complete'] or not parent_audit['all_completed_episodes_verified']:
        raise ValueError('Final holdout requires an audited complete training run')
    audit.equal(parent_audit['manifest_sha256'],manifest['parent_manifest_sha256'],'Archived audit identifies another parent')
    parent = Path(manifest['parent_run'])
    audit.equal(audit.sha(parent/'manifest.json'),manifest['parent_manifest_sha256'],'Parent manifest changed')
    parent_manifest = json.loads((parent/'manifest.json').read_text())
    audit.equal(audit.sha(parent/'analysis.json'),manifest['parent_analysis_sha256'],'Parent audit changed after preparation')
    protocol.validate_parent(parent,parent_manifest)
    registration = protocol.read_registration(run/'preregistration.json')
    audit.equal(audit.sha(run/'preregistration.json'),manifest['preregistration_sha256'],'Archived preregistration changed')
    protocol.validate_registration(registration,parent,parent_manifest,manifest['deadline'])
    audit.equal(manifest['source_sha256'],registration['source_sha256'],'Source set differs from preregistration')
    for key in ('model','environments','horizons','versions','rating_compatibility','policy_arithmetic',
                'policy_arithmetic_sha256','inference_batch_size'):
        audit.equal(manifest[key],parent_manifest[key],'Holdout runtime differs from audited parent: '+key)
    if protocol.scenario_families(schedule) & protocol.scenario_families(json.loads((parent/'schedule.json').read_text())):
        raise ValueError('Final scenarios overlap parent data')
    metadata,checkpoints = {},{}
    for name,expected in manifest['inherited_checkpoints'].items():
        actual,meta = league.checkpoint_record(expected['path'],name,manifest['model'])
        audit.equal(actual,expected,'Evaluated checkpoint changed')
        protocol.validate_checkpoint(name,actual,meta,manifest,parent_audit)
        metadata[name],checkpoints[name] = meta,actual
    audit.equal(sorted(metadata),sorted(protocol.PROTOCOL['candidates']),'Wrong candidates evaluated')
    tokenizer = audit.load_tokenizer(manifest)
    for label,token in zip(metadata['base']['labels'],metadata['base']['label_token_ids'],strict=True):
        audit.equal(tokenizer.encode(label,add_special_tokens=False),[token],'Action label tokenization changed')
    for name in ('chess','python-chess'):
        audit.equal(importlib.metadata.version(name),manifest['versions'][name],'Simulator version changed')
    rows,seen,files = [],set(),{}
    transitions = 0
    status = json.loads((run/'status.json').read_text())
    summary = json.loads((run/'summary.json').read_text()) if (run/'summary.json').exists() else {}
    if summary.get('complete') is True:
        audit.equal(summary['training'],False,'Holdout summary reports training')
        audit.equal(summary['protocol'],protocol.PROTOCOL,'Holdout summary protocol differs')
        audit.equal(summary['episodes'],sum(map(len,schedule.values())),'Holdout summary episode count differs')
        audit.equal(summary['stage_records'],status['stages'],'Completed holdout stage evidence differs')
    for stage,specs in schedule.items():
        path = run/(stage+'.jsonl')
        saved,evidence = canonical_rows(run,stage,specs,status,complete=summary.get('complete') is True)
        expected = {s['episode_id']:s for s in specs}
        for row in saved:
            identity = row['episode_id']
            if identity not in expected or identity in seen:
                raise ValueError('Unknown or duplicate holdout episode')
            for key,value in expected[identity].items():
                audit.equal(row.get(key),value,'Episode differs from holdout schedule')
            transitions += audit.replay(row,arena,policy.action_prompt,metadata,tokenizer,manifest)
            rows.append(row);seen.add(identity)
        if evidence is not None:
            files[path.name] = evidence
    missing = sorted({s['episode_id'] for group in schedule.values() for s in group}-seen)
    complete = not missing and summary.get('complete') is True
    if not allow_partial and not complete:
        raise ValueError('Final holdout remains incomplete')
    evaluation,changes,macro = audit.evaluations(rows)
    return {'complete':complete,'all_completed_episodes_verified':True,'episodes_verified':len(rows),
        'policy_transitions_verified':transitions,'scheduled_episodes':sum(map(len,schedule.values())),
        'missing_episode_ids':missing,'manifest_sha256':audit.sha(run/'manifest.json'),
        'checkpoint_verification':checkpoints,'source_verification':sources,'rollout_files':files,
        'evaluation':evaluation,'paired_checkpoint_changes':changes,'macro_fixed_opponent_return':macro,
        'policy_arithmetic_sha256':manifest['policy_arithmetic_sha256'],
        'preregistration_sha256':manifest['preregistration_sha256'],
        'training':False,'holdout_protocol':protocol.PROTOCOL,
        'warnings':['Twelve independent paired scenario clusters per environment/opponent, one training seed.',
            'Final checkpoint fixed before results; no outcome-based checkpoint selection.',
            'Distance/material capped rewards are adjudications, not goal completion or checkmate.',
            'General reasoning transfer and indefinite continual improvement remain untested.',
            'Partial evaluations are descriptive only; compare completed paired scenarios.'],
        'analysis_source_sha256':audit.sha(__file__),'shared_analysis_source_sha256':audit.sha(audit.__file__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('runs/arena-final-holdout-v1'))
    parser.add_argument('--allow-partial',action='store_true')
    args = parser.parse_args();result = analyze(args.run,args.allow_partial)
    league.atomic_json(args.run/'analysis.json',result)
    print(json.dumps({k:result[k] for k in ('complete','episodes_verified','policy_transitions_verified')}))


if __name__=='__main__':main()
