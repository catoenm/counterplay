"""CPU-only terminal teacher audit; permits a scientifically valid shortfall.

No API/model work. Replays all frozen games even if fewer than16 demonstrations
qualify. A failed corpus remains insufficient; the audit never relaxes selection.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import collect_cardinal_teacher_none as collector
from scripts.collect_cardinal_teacher_none import (read,sha,require,verify_plan,runtime,generation,
    delivery_gate,EOS,qualifies,select)

def verify_terminal_result(run, replay=True, require_active=True):
    """Self-contained CPU admission for SFT; requires exactly16 verified rows.

No global ledger, API credentials, model weights, or unarchived teacher output
is needed. Returned demonstrations contain original source and exact EOS IDs.
"""
    from tokenizers import Tokenizer
    replay_check = replay
    run = Path(run).resolve(); plan = verify_plan(run, require_active); corpus = read(run/'corpus.json')
    require(corpus['kind'] == 'verified_cardinal_teacher_corpus_result'
        and corpus['state'] in ('ready_for_sft','insufficient_verified_demonstrations') and corpus['all_attempts'] == 32
        and corpus['required_demonstrations'] == 16, 'Teacher corpus did not qualify')
    require(corpus['plan_sha256'] == sha(run/'plan.json')
        and corpus['clean_prompt_sha256'] == sha(run/'prompts/learner.txt')
        and corpus['teacher_ledger_sha256'] == sha(run/'teacher-ledger-final.json')
        and corpus['delivery_gate_sha256'] == sha(run/'delivery-gate.json'), 'Corpus provenance differs')
    require(importlib.metadata.version('tokenizers') == plan['learner']['tokenizers_version'], 'Tokenizer library differs')
    tokenizer = Tokenizer.from_file(str(run/'inputs/tokenizer.json'))
    require(tokenizer.token_to_id('<|im_end|>') == EOS, 'EOS differs')
    rows = []
    with runtime(run, plan['source_sha256']) as rt:
        require((run/'prompts/learner.txt').read_text() == rt.spec(), 'Clean learner prompt differs')
        refs = {key: rt.arena.compile_program(value) for key, value in rt.references().items()}
        for index in range(32):
            record, digest = generation(run, plan, index, run/'teacher-ledger-final.json', rt.teacher)
            if index == 0: require(delivery_gate(run, record, client=rt.teacher)['passed'], 'Delivery gate failed')
            path = run/'evaluation'/f'teacher-{index:02d}.json'; row = read(path)
            require(digest == corpus['generation_sha256'][str(index)] == row['generation_sha256']
                and sha(path) == corpus['evaluation_sha256'][str(index)] and row['index'] == index, 'Teacher grade provenance differs')
            classification, compiled = rt.classify(record['text'], 'eos' if record['finish_reason'] == 'stop' else record['finish_reason'])
            require(classification == row['classification'], 'Teacher reclassification differs')
            if not compiled:
                require(not row['matches'] and not row['completion_ids'] and row['learner_tokens_including_eos'] is None
                    and not row['qualified'] and all(row[k] == 0 for k in ('candidate_program_faults', 'opponent_program_faults', 'positive_banking_games', 'banked_points_sum')), 'Invalid authoring has fabricated games')
            else:
                ids = tokenizer.encode(classification['program']['python_source'], add_special_tokens=False).ids + [EOS]
                require(row['completion_ids'] == ids and row['learner_tokens_including_eos'] == len(ids), 'Teacher tokenization differs')
                expected = [(o, seed, seat) for o in ('nearest', 'value') for seed in plan['schedule']['teacher_maps'] for seat in (0, 1)]
                require([(m['opponent_id'], m['replay']['seed'], m['candidate_seat']) for m in row['matches']] == expected, 'Teacher gameplay schedule differs')
                faults = opponent_faults = points = positive = 0
                for match in row['matches']:
                    replay, seat = match['replay'], match['candidate_seat']
                    require(replay['programs'][seat]['digest'] == compiled.digest
                        and replay['programs'][1-seat]['digest'] == refs[match['opponent_id']].digest
                        and replay['origins'] == (['model', 'scripted'] if seat == 0 else ['scripted', 'model'])
                        and replay['max_rounds'] == 48, 'Teacher executed program/rules differ')
                    if replay_check: rt.arena.verify_replay(replay)
                    metrics = replay['metrics']; faults += metrics['program_faults'][seat]
                    opponent_faults += metrics['program_faults'][1-seat]
                    points += metrics['scores'][seat]; positive += int(metrics['scores'][seat] > 0)
                require([faults, opponent_faults, points, positive] == [row[k] for k in
                    ('candidate_program_faults', 'opponent_program_faults', 'banked_points_sum', 'positive_banking_games')], 'Teacher qualification metrics differ')
                require(row['qualified'] == qualifies(row), 'Teacher qualification differs')
            rows.append(row)
    chosen = select(rows)
    require(chosen == corpus['selected_indices'] and len(chosen) in (0,16)
        and corpus['state'] == ('ready_for_sft' if len(chosen) == 16 else 'insufficient_verified_demonstrations')
        and corpus['qualified_attempts'] == [r['index'] for r in rows if r['qualified']], 'Teacher selection changed')
    expected_demos = [{'index': i, 'messages': [{'role': 'user', 'content': (run/'prompts/learner.txt').read_text()},
        {'role': 'assistant', 'content': rows[i]['classification']['program']['python_source']}],
        'completion_ids': rows[i]['completion_ids'], 'generation_sha256': rows[i]['generation_sha256'],
        'program': rows[i]['classification']['program']} for i in chosen]
    require(corpus['demonstrations'] == expected_demos, 'SFT demonstrations differ from selected teacher sources')
    return {**corpus, 'corpus_sha256': sha(run/'corpus.json'), 'source_sha256': plan['source_sha256'],
        'input_sha256': plan['input_sha256'], 'replays_independently_verified': replay_check}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve();output=args.output.resolve()
    require(not output.exists(),'Refusing to overwrite independent audit')
    # No partial output may be scored or interpreted here.
    require(len(list((run/'generations').glob('teacher-*.json')))==32
        and len(list((run/'evaluation').glob('teacher-*.json')))==32
        and (run/'corpus.json').exists(),'Complete32-response terminal corpus required')
    paths=['plan.json','corpus.json','delivery-gate.json','teacher-ledger-final.json']
    paths += [f'{folder}/teacher-{i:02d}.json' for folder in ('generations','evaluation') for i in range(32)]
    before={name:sha(run/name) for name in paths}
    if read(run/'corpus.json')['state']=='ready_for_sft':
        result=collector.verify_corpus(run,replay=True,require_active=False)
    else:
        result=verify_terminal_result(run,replay=True,require_active=False)
    rows=[read(run/'evaluation'/f'teacher-{i:02d}.json') for i in range(32)]
    records=[read(run/'generations'/f'teacher-{i:02d}.json') for i in range(32)]
    actual=sum(r['cost_usd'] for r in records);upper=sum(r['cost_upper_usd'] for r in records)
    require(actual==result['total_actual_api_cost_usd'] and upper==result['total_conservative_api_cost_usd'], 'Corpus cost summary differs')
    require(before=={name:sha(run/name) for name in paths},'Corpus changed during independent audit')
    def usage_state(record):
        count=record['reported_reasoning_tokens']
        return 'reported_zero' if type(count) is int and count==0 else 'reported_positive' if type(count) is int and count>0 else 'unknown'
    report={'format_version':1,'kind':'independent_terminal_teacher_corpus_audit',
        'created_at':datetime.now(timezone.utc).isoformat(),'integrity_passed':True,
        'cohort_state':result['state'],'sft_admitted':result['state']=='ready_for_sft',
        'plan_sha256':before['plan.json'],'corpus_sha256':before['corpus.json'],
        'status_sha256':None,'status_source':'corpus.json terminal state; collector has no status.json journal',
        'artifact_sha256':before,'source_sha256':result['source_sha256'],
        'input_sha256':result['input_sha256'],'audit_source_sha256':sha(__file__),
        'attempts':32,'compiled_attempts':sum(r['classification']['valid'] for r in rows),
        'qualified_attempts':result['qualified_attempts'],'selected_indices':result['selected_indices'],
        'selected_count':len(result['demonstrations']),'recorded_games':sum(len(r['matches']) for r in rows),
        'all_recorded_games_independently_replayed':True,
        'candidate_program_faults':sum(r['candidate_program_faults'] for r in rows),
        'opponent_program_faults':sum(r['opponent_program_faults'] for r in rows),
        'positive_banking_games':sum(r['positive_banking_games'] for r in rows),
        'banked_points_sum':sum(r['banked_points_sum'] for r in rows),
        'actual_api_cost_usd':actual,'conservative_api_cost_usd':upper,
        'reasoning_condition':'reasoning-disabled requested',
        'reported_reasoning_status_counts':dict(Counter(usage_state(r) for r in records)),
        'attempt_summaries':[{k:r[k] for k in ('index','classification','learner_tokens_including_eos',
            'candidate_program_faults','opponent_program_faults','positive_banking_games','banked_points_sum','qualified')}
            for r in rows],
        'scope':'Teacher corpus verification only; no SFT/RL update or improvement claim.'}
    # Keep diagnostics compact; exact code remains in its bound original grade.
    for row in report['attempt_summaries']:
        row['classification']={k:v for k,v in row['classification'].items() if k!='program'}
    collector.write_new(output,report)
    print(json.dumps({k:report[k] for k in ('integrity_passed','cohort_state','compiled_attempts','selected_count',
        'qualified_attempts','recorded_games','positive_banking_games','actual_api_cost_usd','reported_reasoning_status_counts')}))
    print(json.dumps({'output':str(output),'audit_sha256':sha(output)}))

if __name__=='__main__':main()
