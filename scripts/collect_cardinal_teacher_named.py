"""Freeze, collect, and mechanically qualify a bounded teacher corpus.

prepare/evaluate are CPU-only. advance makes exactly one explicitly requested
teacher call through the separately budgeted client. No generated Python runs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from selfplay import program_cardinal_teacher_named as teacher

GAME_SOURCES = ['selfplay/program_arena/' + name + '.py' for name in
    ('__init__', 'dsl', 'game', 'runner', 'examples', 'spec')]
RUNTIME_SOURCES = GAME_SOURCES + ['selfplay/program_python.py', 'selfplay/program_cardinal.py']
LOCAL_SOURCES = ['selfplay/program_cardinal_teacher_named.py', 'selfplay/program_model_screen.py',
    'scripts/collect_cardinal_teacher_named.py']
TOKENIZER_SHA256 = '0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3'
MODEL = 'Qwen/Qwen3.8-27B'
REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
EOS = 248046
DESIGNS = (
    'Prefer nearby available resources and keep decisions simple.',
    'Balance site value against travel distance.',
    'Favor short complete collection-and-deposit trips.',
    'Balance resource density against total round-trip travel.',
    'Conserve energy while maintaining useful collection throughput.',
    'Favor reliable banking when the remaining horizon is short.',
    'Choose resource targets using a compact integer utility.',
    'Avoid repeatedly pursuing a resource already depleted by the opponent.',
    'Favor valuable cargo while keeping return trips short.',
    'Consider opponent proximity when choosing among useful resources.',
    'Use deterministic tie-breaking that works from either home.',
    'Favor nearby sites that can fill cargo efficiently.',
    'Use a modest amount of persistent memory only if it improves reliability.',
    'Use a stateless policy that recomputes decisions from the public sensors.',
    'Balance frequent deposits with the cost of extra homeward travel.',
    'Optimize a simple robust strategy over both player seats.',
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return teacher.sha(Path(path).read_bytes())


def value_sha(value):
    return teacher.sha(teacher.canonical(value).encode())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def teacher_prompt(spec, index):
    require(type(index) is int and 0 <= index < 32, 'Fixed index0..31 required')
    return ('Design one compact, reliable controller for the exact game and syntax below. '
        'Use private reasoning to check the complete collection, return, and deposit cycle, '
        'energy recovery, depleted sites, both home positions, and the final rounds. '
        'Do not include reasoning in your answer. Return only the complete def act(): source. '
        'Aim for at most35 lines and fewer than700 Qwen learner tokens including the final EOS. '
        'Avoid comments, docstrings, needless repeated branches, and complicated state. '
        'The output must satisfy every syntax limit; there are no extra helpers or implicit actions.\n\n'
        'Naming rule: every sensor listed in the specification is read-only, including score, energy, carrying, '
        'self_x, self_y, home_x, home_y, capacity, turn, horizon, and opponent sensors. '
        'Never assign to a sensor name. Use distinct local names such as best_value, candidate_value, '
        'target_x, and target_y for your own calculations.\n\n'
        f'Independent design {index+1}/32. Design objective: {DESIGNS[index % 16]} '
        f'Replicate {index // 16 + 1}; produce a self-contained policy, not a patch.\n\n' + spec)


def schedule():
    value = {'teacher_maps': list(range(91000000, 91000006)),
        'capability_maps': list(range(92000000, 92000006)),
        'train_maps': {str(i): [93000000+i*100+j for j in range(2)] for i in range(1, 101)},
        'holdout_maps': list(range(94000000, 94000006)),
        'teacher_generation_seeds': list(range(141000000, 141000032)),
        'capability_generation_seeds': list(range(112000000, 112000008)),
        'train_generation_seeds': {str(i): [113000000+i*100+j for j in range(8)] for i in range(1, 101)},
        'eval_generation_seeds': list(range(114000000, 114000032)),
        'map_seed_replacements': [{'family': 'train_maps', 'group': 45, 'index': 0,
            'proposed_seed': 93004500, 'replacement_seed': 93004502, 'prior_equal_layout_seed': 8203401,
            'reason': 'Exact observable-map collision detected before teacher collection or new neural work.'}]}
    value['train_maps']['45'][0] = 93004502
    return value


@contextmanager
def runtime(run, source_hashes):
    prefix = '_cardinal_teacher_' + uuid.uuid4().hex
    root = Path(run) / 'source'
    package = ModuleType(prefix); package.__path__ = [str(root / 'selfplay')]
    sys.modules[prefix] = package
    try:
        for relative in RUNTIME_SOURCES:
            require(sha(root / relative) == source_hashes[relative], 'Archived runtime changed: ' + relative)
        def load(name, relative, package=False):
            path = root / relative
            spec = importlib.util.spec_from_file_location(prefix+'.'+name, path,
                submodule_search_locations=[str(path.parent)] if package else None)
            module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
        arena = load('program_arena', 'selfplay/program_arena/__init__.py', True)
        load('program_python', 'selfplay/program_python.py')
        front = load('program_cardinal', 'selfplay/program_cardinal.py')
        examples = load('program_arena.examples', 'selfplay/program_arena/examples.py')
        for relative in LOCAL_SOURCES[:2]:
            require(sha(root/relative) == source_hashes[relative], 'Archived teacher client changed')
        load('program_model_screen', 'selfplay/program_model_screen.py')
        archived_teacher = load('program_cardinal_teacher_named', 'selfplay/program_cardinal_teacher_named.py')
        yield SimpleNamespace(arena=arena, classify=front.classify_cardinal,
            spec=front.cardinal_model_spec, references=examples.reference_programs, teacher=archived_teacher)
    finally:
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix+'.'): del sys.modules[name]


def layout(game):
    return {'grid_size': game.size, 'capacity': game.capacity, 'max_energy': game.max_energy,
        'max_rounds': game.max_rounds, 'sites': game.sites,
        'homes': [[r['home_x'], r['home_y']] for r in game.robots]}


def map_fingerprints(arena, timetable, prior_documents):
    prior = {}
    for document in prior_documents:
        prior.update(document['prior_seed_fingerprints'])
        for rows in document['families'].values():
            for row in rows:
                require(value_sha(row['layout']) == row['sha256'], 'Prior layout fingerprint differs')
                key = str(row['seed'])
                require(key not in prior or prior[key] == row['sha256'], 'Prior seed has inconsistent layout')
                prior[key] = row['sha256']
    used = set(prior.values()); result = {}
    families = {k: timetable[k] for k in ('teacher_maps', 'capability_maps', 'holdout_maps')}
    families['train_maps'] = [s for row in timetable['train_maps'].values() for s in row]
    seeds_seen = set(prior)
    for family, seeds in families.items():
        result[family] = []
        for seed in seeds:
            state = layout(arena.RobotArena(seed, 48)); fingerprint = value_sha(state)
            require(str(seed) not in seeds_seen and fingerprint not in used, 'New map overlaps a prior/new family')
            seeds_seen.add(str(seed)); used.add(fingerprint)
            result[family].append({'seed': seed, 'sha256': fingerprint, 'layout': state})
    return {'families': result, 'prior_seed_fingerprints': prior,
        'definition': 'Initial observable sites in order, homes, size/capacity/energy/horizon; exclude seed and hidden priority.'}


HIGH_LOCAL_SOURCES = ['selfplay/program_cardinal_teacher.py', 'selfplay/program_model_screen.py',
    'scripts/collect_cardinal_teacher.py']
HIGH_EVIDENCE_FILES = ['plan.json', 'map-fingerprints.json', 'administrative-stop-intent.json',
    'administrative-stop-budget-applied.json', 'administrative-stop-final.json', 'teacher-ledger-stopped.json']


@contextmanager
def high_client(folder, client_name='program_cardinal_teacher'):
    prefix = '_stopped_high_teacher_' + uuid.uuid4().hex
    source = Path(folder)/'source/selfplay'
    package = ModuleType(prefix); package.__path__ = [str(source)]; sys.modules[prefix] = package
    try:
        result = None
        for name in ('program_model_screen', client_name):
            spec = importlib.util.spec_from_file_location(prefix+'.'+name, source/(name+'.py'))
            result = importlib.util.module_from_spec(spec); sys.modules[spec.name] = result
            spec.loader.exec_module(result)
        yield result
    finally:
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix+'.'): del sys.modules[name]


def high_inventory(plan, stop):
    return sorted(set(HIGH_EVIDENCE_FILES + ['source/'+p for p in plan['source_sha256']]
        + list(plan['input_sha256']) + [v['path'] for v in plan['teacher_prompts'].values()]
        + [f"generations/teacher-{r['index']:02d}.json" for r in stop['generations']]))


def verify_high(folder, ledger_path=None):
    """Offline receipt chain for the stopped, ungraded high-reasoning cohort."""
    folder = Path(folder)
    plan, stop = read(folder/'plan.json'), read(folder/'administrative-stop-final.json')
    require(not (folder/'corpus.json').exists() and not (folder/'evaluation').exists(),
        'High cohort has gameplay grades or a corpus; map reuse is not admitted')
    require(stop['kind'] == 'administrative_stop_final' and stop['collector_stopped'] is True
        and stop['no_gameplay_scoring'] is True and stop['no_training'] is True
        and stop['original_cohort_state'] == 'incomplete' and stop['unknown_requests'] == 0
        and 0 < stop['completed_attempts'] < 32 and stop['prescribed_attempts'] == 32
        and stop['unattempted'] == 32-stop['completed_attempts'] and stop['retries'] == 0,
        'High cohort is not a terminal, ungraded administrative stop')
    require(sha(folder/'plan.json') == stop['plan_sha256']
        and sha(folder/'administrative-stop-intent.json') == stop['intent_sha256']
        and sha(folder/'administrative-stop-budget-applied.json') == stop['applied_stop_sha256'],
        'High stop evidence differs')
    applied = read(folder/'administrative-stop-budget-applied.json')
    require(applied['new_phase_cap_usd'] == 0 and applied['existing_commitments_retained'] is True,
        'High cohort budget stop was not applied')
    require(set(plan['source_sha256']) == set(RUNTIME_SOURCES + HIGH_LOCAL_SOURCES), 'High source inventory differs')
    for relative, digest in plan['source_sha256'].items():
        require(sha(folder/'source'/relative) == digest, 'High frozen source differs')
    for relative, digest in plan['input_sha256'].items():
        require(sha(folder/relative) == digest, 'High input differs')
    for index, entry in plan['teacher_prompts'].items():
        require(entry['path'] == f'prompts/teacher-{int(index):02d}.txt'
            and sha(folder/entry['path']) == entry['sha256'], 'High prompt differs')
    require(sha(folder/'map-fingerprints.json') == plan['map_fingerprints_sha256'], 'High map fingerprint differs')
    expected_schedule = schedule(); expected_schedule['teacher_generation_seeds'] = list(range(111000000,111000032))
    require(plan['schedule'] == expected_schedule, 'Reused high maps/schedule differ')
    ledger_path = folder/'teacher-ledger-stopped.json' if ledger_path is None else Path(ledger_path)
    require(sha(ledger_path) == stop['ledger_sha256'], 'Stopped high ledger differs')
    with high_client(folder) as client:
        require(plan['configuration'] == client.configuration(), 'High client configuration differs')
        ledger = client.checked_ledger(sha(folder/'plan.json'), ledger_path)
        entries = list(ledger['requests'].values())
        require(len(entries) == stop['completed_attempts'] and all(e['state'] == 'completed' for e in entries)
            and sorted(e['index'] for e in entries) == list(range(len(entries))), 'Stopped high journal is incomplete/uncertain')
        require([r['index'] for r in stop['generations']] == list(range(len(entries))), 'High retained-output inventory differs')
        for row in stop['generations']:
            index = row['index']; path = folder/'generations'/f'teacher-{index:02d}.json'; record = read(path)
            prompt = (folder/plan['teacher_prompts'][str(index)]['path']).read_text()
            client.verify_completed_response(prompt, index, sha(folder/'plan.json'), record, ledger_path)
            require(sha(path) == row['generation_sha256'] and record['finish_reason'] == row['finish_reason']
                and len(record['text']) == row['answer_characters'], 'High output receipt differs')
        first = next(e['response'] for e in entries if e['index'] == 0)
        require(first['finish_reason'] == 'length' and not first['text'].strip(),
            'High administrative delivery diagnosis differs')
    return plan, stop


LOW_LOCAL_SOURCES = ['selfplay/program_cardinal_teacher_low.py', 'selfplay/program_model_screen.py',
    'scripts/collect_cardinal_teacher_low.py']
LOW_EVIDENCE_FILES = ['plan.json', 'map-fingerprints.json', 'delivery-gate.json',
    'independent-delivery-audit.json', 'teacher-ledger-stopped.json', 'generations/teacher-00.json']
REUSE_REASON = ('The high and low reasoning conditions stopped for answer-delivery exhaustion '
    'before gameplay or scoring; maps remain unobserved.')


def low_inventory(plan):
    return sorted(set(LOW_EVIDENCE_FILES + ['source/'+p for p in plan['source_sha256']]
        + list(plan['input_sha256']) + [v['path'] for v in plan['teacher_prompts'].values()]))


def verify_low(folder, ledger_path=None):
    """Verify both failed conditions from their portable, ungraded receipts."""
    folder = Path(folder)
    plan, audit, gate = (read(folder/p) for p in ('plan.json','independent-delivery-audit.json','delivery-gate.json'))
    require(not (folder/'corpus.json').exists() and not (folder/'evaluation').exists(),
        'Low cohort has gameplay grades or a corpus; map reuse is not admitted')
    require(plan['kind'] == 'verified_cardinal_teacher_low_corpus'
        and audit['receipt_integrity_passed'] is True and audit['cohort_state'] == 'incomplete_delivery_failed'
        and audit['completed_attempts'] == 1 and audit['unattempted'] == 31 and audit['unknown_requests'] == 0
        and audit['prescribed_attempts'] == 32 and audit['automatic_retries'] == 0
        and audit['recorded_games'] == 0 and audit['evaluation_exists'] is False and audit['corpus_exists'] is False,
        'Low cohort is not an ungraded failed delivery gate')
    require(sha(folder/'plan.json') == audit['plan_sha256']
        and sha(folder/'delivery-gate.json') == audit['delivery_gate_sha256']
        and plan['source_sha256'] == audit['source_sha256'], 'Low failure audit binding differs')
    require(set(plan['source_sha256']) == set(RUNTIME_SOURCES + LOW_LOCAL_SOURCES), 'Low source inventory differs')
    for relative, digest in plan['source_sha256'].items():
        require(sha(folder/'source'/relative) == digest, 'Low frozen source differs')
    for relative, digest in plan['input_sha256'].items():
        require(sha(folder/relative) == digest, 'Low input differs')
    for index, entry in plan['teacher_prompts'].items():
        require(entry['path'] == f'prompts/teacher-{int(index):02d}.txt'
            and sha(folder/entry['path']) == entry['sha256'], 'Low prompt differs')
    require(sha(folder/'map-fingerprints.json') == plan['map_fingerprints_sha256'], 'Low map fingerprint differs')
    expected_schedule = schedule(); expected_schedule['teacher_generation_seeds'] = list(range(121000000,121000032))
    require(plan['schedule'] == expected_schedule, 'Reused low maps/schedule differ')
    high_plan, _ = verify_high(folder/'inputs/high-cohort')
    require(plan['map_reuse']['source_plan_sha256'] == sha(folder/'inputs/high-cohort/plan.json')
        and plan['map_fingerprints_sha256'] == high_plan['map_fingerprints_sha256']
        and all(plan['source_sha256'][p] == high_plan['source_sha256'][p] for p in RUNTIME_SOURCES),
        'High-to-low game or map chain differs')
    ledger_path = folder/'teacher-ledger-stopped.json' if ledger_path is None else Path(ledger_path)
    require(sha(ledger_path) == audit['ledger_sha256'], 'Stopped low ledger differs')
    with high_client(folder, 'program_cardinal_teacher_low') as client:
        require(plan['configuration'] == client.configuration(), 'Low client configuration differs')
        ledger = client.checked_ledger(sha(folder/'plan.json'), ledger_path)
        entries = list(ledger['requests'].values())
        require(len(entries) == 1 and entries[0]['index'] == 0 and entries[0]['state'] == 'completed',
            'Low journal is not a single completed delivery attempt')
        record = read(folder/'generations/teacher-00.json')
        prompt = (folder/plan['teacher_prompts']['0']['path']).read_text()
        client.verify_completed_response(prompt, 0, sha(folder/'plan.json'), record, ledger_path)
        require(sha(folder/'generations/teacher-00.json') == audit['generation_sha256'] == gate['generation_sha256']
            and gate['index'] == 0 and gate['passed'] is False
            and gate['cohort_state'] == 'incomplete_delivery_failed' and not client.delivery_passed(record)
            and record['finish_reason'] == audit['finish_reason'] == 'length'
            and len(record['text']) == audit['answer_characters'] == 0,
            'Low completed response or failed delivery gate differs')
    return plan, audit


NONE_LOCAL_SOURCES = ['selfplay/program_cardinal_teacher_none.py', 'selfplay/program_model_screen.py',
    'scripts/collect_cardinal_teacher_none.py']
NONE_EVIDENCE_FILES = ['plan.json','map-fingerprints.json','delivery-gate.json','corpus.json',
    'teacher-ledger-final.json','independent-audit.json','audit-source.py']
NAMED_REUSE_REASON = ('Only the teacher qualification maps were observed in the prior none-requested cohort. '
    'They are deliberately reused for teacher-only qualification under a uniform naming clarification. '
    'Learner capability, RL-training, and final-holdout maps have not been scored.')
NAMING_CHANGE = 'Uniform teacher-only read-only sensor naming clarification; no original-output repair or learner-prompt change.'


def none_inventory(plan):
    return sorted(set(NONE_EVIDENCE_FILES + ['source/'+p for p in plan['source_sha256']]
        + list(plan['input_sha256']) + [v['path'] for v in plan['teacher_prompts'].values()]
        + [f'{folder}/teacher-{i:02d}.json' for folder in ('generations','evaluation') for i in range(32)]))


def verify_none(folder, audit_source=None):
    """Bind all prior32 authored outputs/grades without admitting them to the new corpus."""
    folder=Path(folder)
    plan,corpus,audit=(read(folder/name) for name in ('plan.json','corpus.json','independent-audit.json'))
    require(plan['kind']=='verified_cardinal_teacher_none_corpus'
        and corpus['state']=='insufficient_verified_demonstrations' and corpus['selected_indices']==[]
        and audit['integrity_passed'] is True and audit['cohort_state']=='insufficient_verified_demonstrations'
        and audit['sft_admitted'] is False and audit['attempts']==32 and audit['selected_count']==0,
        'Prior none cohort is not the preserved, independently audited insufficient corpus')
    require(sha(folder/'plan.json')==audit['plan_sha256']==corpus['plan_sha256']
        and sha(folder/'corpus.json')==audit['corpus_sha256']
        and audit['source_sha256']==plan['source_sha256'], 'Prior none audit chain differs')
    require(set(plan['source_sha256'])==set(RUNTIME_SOURCES+NONE_LOCAL_SOURCES),'Prior none source inventory differs')
    for relative,digest in plan['source_sha256'].items():
        require(sha(folder/'source'/relative)==digest,'Prior none source changed')
    for relative,digest in plan['input_sha256'].items():
        require(sha(folder/relative)==digest,'Prior none input changed')
    for relative,digest in audit['artifact_sha256'].items():
        require(sha(folder/relative)==digest,'Prior none audited artifact changed')
    audit_source=folder/'audit-source.py' if audit_source is None else Path(audit_source)
    require(sha(audit_source)==audit['audit_source_sha256'],'Prior none audit source differs')
    low_plan,_=verify_low(folder/'inputs/low-cohort')
    high_plan,_=verify_high(folder/'inputs/high-cohort')
    require(plan['map_fingerprints_sha256']==low_plan['map_fingerprints_sha256']==high_plan['map_fingerprints_sha256']
        ==sha(folder/'map-fingerprints.json'),'Prior none map chain differs')
    expected_schedule=schedule();expected_schedule['teacher_generation_seeds']=list(range(131000000,131000032))
    require(plan['schedule']==expected_schedule,'Prior none schedule differs')
    with high_client(folder,'program_cardinal_teacher_none') as client:
        require(plan['configuration']==client.configuration(),'Prior none client configuration differs')
        ledger=client.checked_ledger(sha(folder/'plan.json'),folder/'teacher-ledger-final.json')
        entries=list(ledger['requests'].values())
        require(len(entries)==32 and all(e['state']=='completed' for e in entries)
            and sorted(e['index'] for e in entries)==list(range(32)),'Prior none completed journal differs')
        rows=[]
        for i in range(32):
            record=read(folder/'generations'/f'teacher-{i:02d}.json')
            prompt=(folder/plan['teacher_prompts'][str(i)]['path']).read_text()
            require(sha(folder/plan['teacher_prompts'][str(i)]['path'])==plan['teacher_prompts'][str(i)]['sha256'],
                'Prior none teacher prompt changed')
            client.verify_completed_response(prompt,i,sha(folder/'plan.json'),record,folder/'teacher-ledger-final.json')
            row=read(folder/'evaluation'/f'teacher-{i:02d}.json');rows.append(row)
            require(row['index']==i and row['generation_sha256']==corpus['generation_sha256'][str(i)]
                ==sha(folder/'generations'/f'teacher-{i:02d}.json')
                and sha(folder/'evaluation'/f'teacher-{i:02d}.json')==corpus['evaluation_sha256'][str(i)],
                'Prior none output or grade differs')
            expected=[(op,seed,seat) for op in ('nearest','value') for seed in plan['schedule']['teacher_maps'] for seat in (0,1)]
            actual=[(m['opponent_id'],m['replay']['seed'],m['candidate_seat']) for m in row['matches']]
            require(actual==(expected if row['classification']['valid'] else []),
                'Prior none grades used maps outside teacher qualification')
        require(select(rows)==[] and corpus['qualified_attempts']==[r['index'] for r in rows if qualifies(r)]
            ==audit['qualified_attempts'],'Prior none selection/evidence changed')
    return plan,corpus,audit


def reuse_binding(run):
    run=Path(run)
    return {'source_condition':'high_low_delivery_failures_and_none_graded_insufficient',
        'source_plan_sha256':sha(run/'inputs/high-cohort/plan.json'),
        'source_map_fingerprints_sha256':sha(run/'inputs/high-cohort/map-fingerprints.json'),
        'administrative_stop_sha256':sha(run/'inputs/high-cohort/administrative-stop-final.json'),
        'low_plan_sha256':sha(run/'inputs/low-cohort/plan.json'),
        'low_delivery_audit_sha256':sha(run/'inputs/low-cohort/independent-delivery-audit.json'),
        'none_plan_sha256':sha(run/'inputs/none-cohort/plan.json'),
        'none_corpus_sha256':sha(run/'inputs/none-cohort/corpus.json'),
        'none_independent_audit_sha256':sha(run/'inputs/none-cohort/independent-audit.json'),
        'teacher_maps_previously_graded':True,'learner_maps_previously_scored':False,
        'prior_outputs_eligible_for_new_selection':False,'reason':NAMED_REUSE_REASON}


def prepare(run,previous,audit_source):
    run,previous,audit_source=Path(run).resolve(),Path(previous).resolve(),Path(audit_source).resolve()
    require(not run.exists(),'Refusing to overwrite naming-clarified teacher corpus')
    none_plan,_,_=verify_none(previous,audit_source)
    previous_high=previous/'inputs/high-cohort';previous_low=previous/'inputs/low-cohort'
    high_plan,stop=verify_high(previous_high);low_plan,_=verify_low(previous_low)
    require(sha(previous/'inputs/tokenizer.json')==TOKENIZER_SHA256,'Wrong learner tokenizer')
    run.mkdir(parents=True);(run/'inputs').mkdir();(run/'prompts').mkdir()
    sources={}
    for relative in RUNTIME_SOURCES+LOCAL_SOURCES:
        path=previous/'source'/relative if relative in RUNTIME_SOURCES else ROOT/relative
        target=run/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target);sources[relative]=sha(target)
    inputs={}
    def copy(source,relative):
        target=run/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target);inputs[relative]=sha(target)
    for relative in high_inventory(high_plan,stop):copy(previous_high/relative,'inputs/high-cohort/'+relative)
    for relative in low_inventory(low_plan):copy(previous_low/relative,'inputs/low-cohort/'+relative)
    for relative in none_inventory(none_plan):
        copy(audit_source if relative=='audit-source.py' else previous/relative,'inputs/none-cohort/'+relative)
    for relative in high_plan['input_sha256']:
        if relative!='inputs/endpoint-quote.json':copy(previous_high/relative,relative)
    shutil.copyfile(previous/'map-fingerprints.json',run/'map-fingerprints.json')
    timetable=schedule()
    with runtime(run,sources) as rt:
        spec=rt.spec();priors=[read(run/p) for p in sorted(inputs) if p.startswith('inputs/prior-maps-')]
        require(map_fingerprints(rt.arena,timetable,priors)==read(run/'map-fingerprints.json'),
            'Reused map layouts differ from original archived families')
    prompts={}
    for i in range(32):
        path=run/'prompts'/f'teacher-{i:02d}.txt';path.write_text(teacher_prompt(spec,i))
        prompts[str(i)]={'path':str(path.relative_to(run)),'sha256':sha(path)}
    quote=teacher.verify_endpoint()
    write_new(run/'inputs/endpoint-quote.json',quote);inputs['inputs/endpoint-quote.json']=sha(run/'inputs/endpoint-quote.json')
    plan={**high_plan,'kind':'verified_cardinal_teacher_named_corpus','configuration':teacher.configuration(),
        'source_sha256':sources,'input_sha256':inputs,'teacher_prompts':prompts,'teacher_prompt_change':NAMING_CHANGE,
        'schedule':timetable,'map_reuse':reuse_binding(run),
        'maximum_initial_request_reservations_usd':sum(teacher.reservation(teacher_prompt(spec,i),i) for i in range(32))}
    require(plan['maximum_initial_request_reservations_usd']<=teacher.PHASE_CAP,'Fixed schedule exceeds phase cap')
    write_new(run/'plan.json',plan);verify_plan(run)
    return {'run':str(run),'plan_sha256':sha(run/'plan.json'),'paid_calls':0,
        'maximum_initial_request_reservations_usd':plan['maximum_initial_request_reservations_usd']}


def verify_plan(run, require_active=True):
    run = Path(run).resolve(); plan = read(run/'plan.json')
    require(plan['kind'] == 'verified_cardinal_teacher_named_corpus' and plan['format_version'] == 1, 'Protocol differs')
    require(set(plan['source_sha256']) == set(RUNTIME_SOURCES + LOCAL_SOURCES), 'Frozen source inventory differs')
    require(set(plan['teacher_prompts']) == {str(i) for i in range(32)}, 'All32 frozen teacher prompts required')
    required_inputs = {'inputs/parent-manifest.json', 'inputs/parent-status.json', 'inputs/parent-runtime.json',
        'inputs/tokenizer.json', 'inputs/endpoint-quote.json', 'prompts/learner.txt'}
    high_plan, stop = verify_high(run/'inputs/high-cohort')
    required_inputs |= {'inputs/high-cohort/'+p for p in high_inventory(high_plan, stop)}
    low_plan, _ = verify_low(run/'inputs/low-cohort')
    required_inputs |= {'inputs/low-cohort/'+p for p in low_inventory(low_plan)}
    none_plan, _, _ = verify_none(run/'inputs/none-cohort')
    required_inputs |= {'inputs/none-cohort/'+p for p in none_inventory(none_plan)}
    require(plan['teacher_prompt_change']==NAMING_CHANGE,'Teacher naming clarification changed')
    require(plan['map_reuse'] == reuse_binding(run), 'Map-reuse provenance differs')
    require(low_plan['map_reuse']['source_plan_sha256'] == sha(run/'inputs/high-cohort/plan.json'), 'Duplicated high evidence differs')
    require(plan['map_fingerprints_sha256'] == high_plan['map_fingerprints_sha256'], 'Map reuse changed layouts')
    require(all(plan['source_sha256'][p] == high_plan['source_sha256'][p] for p in RUNTIME_SOURCES), 'Reused game/frontend changed')
    extras = set(plan['input_sha256']) - required_inputs
    require(required_inputs <= set(plan['input_sha256']) and all(
        k.startswith('inputs/prior-maps-') and k.endswith('.json') and k[18:-5].isdigit() for k in extras), 'Frozen input inventory differs')
    require(plan['learner'] == {'model': MODEL, 'revision': REVISION, 'tokenizer_sha256': TOKENIZER_SHA256,
        'eos_token_id': EOS, 'maximum_completion_tokens_including_eos': 700,
        'tokenizers_version': plan['learner']['tokenizers_version'], 'prompt_path': 'prompts/learner.txt'}, 'Learner contract differs')
    require(plan['selection'] == {'attempts': 32, 'required_demonstrations': 16,
        'rule': 'First16 qualifying distinctIR in attempt order; retain all32 attempts and grades; no replacement or repairs.',
        'qualification': {'maximum_learner_tokens_including_eos': 700, 'games': 24,
            'candidate_interpreter_faults': 0, 'minimum_positive_banking_games': 12},
        'insufficient': 'Stop before SFT; do not relax criteria or generate replacements.'}, 'Teacher admission contract differs')
    require(plan['evaluation'] == {'opponents': ['nearest', 'value'], 'seats': [0, 1], 'max_rounds': 48,
        'all32_responses_before_scoring': True, 'holdout_used': False}, 'Teacher evaluation contract differs')
    require(plan['schedule'] == schedule(), 'Fixed seed schedule differs')
    for relative, digest in plan['source_sha256'].items():
        require(sha(run/'source'/relative) == digest, 'Archived corpus source changed')
        if require_active and relative in LOCAL_SOURCES: require(sha(ROOT/relative) == digest, 'Executed corpus source differs from archive')
    for relative, digest in plan['input_sha256'].items():
        require(sha(run/relative) == digest, 'Frozen input changed: '+relative)
    require(sha(run/'map-fingerprints.json') == plan['map_fingerprints_sha256'], 'Frozen map families changed')
    spec = (run/'prompts/learner.txt').read_text()
    for index, entry in plan['teacher_prompts'].items():
        require(entry['path'] == f'prompts/teacher-{int(index):02d}.txt', 'Teacher prompt path differs')
        require(sha(run/entry['path']) == entry['sha256'], 'Teacher prompt changed')
        require((run/entry['path']).read_text() == teacher_prompt(spec, int(index)), 'Teacher prompt does not match frozen design')
    with runtime(run, plan['source_sha256']) as rt:
        require(plan['configuration'] == rt.teacher.configuration(), 'Archived client configuration differs')
        require(spec == rt.spec(), 'Learner prompt differs from archived specification')
        expected = sum(rt.teacher.reservation(teacher_prompt(spec, i), i) for i in range(32))
        require(plan['maximum_initial_request_reservations_usd'] == expected and expected <= 2, 'Fixed schedule budget estimate differs')
    return plan


def generation(run, plan, index, ledger_path=None, client=teacher):
    path = Path(run)/'generations'/f'teacher-{index:02d}.json'; record = read(path)
    prompt = (Path(run)/plan['teacher_prompts'][str(index)]['path']).read_text()
    client.verify_completed_response(prompt, index, sha(Path(run)/'plan.json'), record, ledger_path)
    return record, sha(path)


def delivery_gate(run, record, allow_create=False, client=teacher):
    run = Path(run)
    expected = {'index': 0, 'generation_sha256': sha(run/'generations/teacher-00.json'),
        'passed': client.delivery_passed(record),
        'criterion': 'Nonempty final text and finish_reason stop; no gameplay or compilation selection.',
        'cohort_state': 'collecting' if client.delivery_passed(record) else 'incomplete_delivery_failed'}
    path = run/'delivery-gate.json'
    if path.exists(): require(read(path) == expected, 'Saved delivery gate differs from first response')
    elif allow_create: write_new(path, expected)
    else: raise ValueError('Saved delivery gate missing')
    return expected


def advance(run):
    run = Path(run).resolve(); plan = verify_plan(run)
    for index in range(32):
        path = run/'generations'/f'teacher-{index:02d}.json'
        if path.exists():
            record, _ = generation(run, plan, index)
            if index == 0:
                gate = delivery_gate(run, record, allow_create=True)
                require(gate['passed'], 'First delivery gate failed; cohort incomplete, no later requests')
            continue
        prompt = (run/plan['teacher_prompts'][str(index)]['path']).read_text()
        record = teacher.complete(prompt, index, sha(run/'plan.json'))
        write_new(path, record)
        if index == 0: delivery_gate(run, record, allow_create=True)
        return {'generation': str(path), 'finish_reason': record['finish_reason'], 'cost_usd': record['cost_usd'],
            'state': 'incomplete_delivery_failed' if index == 0 and not teacher.delivery_passed(record) else 'collecting'}
    return {'state': 'all32_collected', 'paid_calls': 0}


def qualifies(row):
    return (row['classification']['valid'] and row['learner_tokens_including_eos'] <= 700
        and row['candidate_program_faults'] == 0 and row['positive_banking_games'] >= 12 and len(row['matches']) == 24)


def select(rows):
    require(len(rows) == 32 and [r['index'] for r in rows] == list(range(32)), 'All32 attempts in order required')
    seen, selected = set(), []
    for row in rows:
        if qualifies(row):
            fingerprint = row['classification']['program']['ir_sha256']
            if fingerprint not in seen:
                seen.add(fingerprint)
                if len(selected) < 16: selected.append(row['index'])
    return selected if len(selected) == 16 else []


def evaluate(run):
    from tokenizers import Tokenizer
    run = Path(run).resolve(); plan = verify_plan(run)
    records = [generation(run, plan, i) for i in range(32)]
    require(delivery_gate(run, records[0][0])['passed'], 'Delivery gate failed; no gameplay scoring admitted')
    require(importlib.metadata.version('tokenizers') == plan['learner']['tokenizers_version'], 'Tokenizer library version differs')
    tokenizer = Tokenizer.from_file(str(run/'inputs/tokenizer.json'))
    require(tokenizer.token_to_id('<|im_end|>') == EOS, 'Learner EOS identity differs')
    rows = []
    with runtime(run, plan['source_sha256']) as rt:
        refs = rt.references()
        require((run/'prompts/learner.txt').read_text() == rt.spec(), 'Learner specification changed')
        for index, (record, digest) in enumerate(records):
            classification, compiled = rt.classify(record['text'], 'eos' if record['finish_reason'] == 'stop' else record['finish_reason'])
            row = {'index': index, 'generation_sha256': digest, 'classification': classification,
                'completion_ids': [], 'learner_tokens_including_eos': None, 'matches': [],
                'candidate_program_faults': 0, 'opponent_program_faults': 0,
                'positive_banking_games': 0, 'banked_points_sum': 0, 'qualified': False}
            if compiled:
                source = classification['program']['python_source']
                row['completion_ids'] = tokenizer.encode(source, add_special_tokens=False).ids + [EOS]
                row['learner_tokens_including_eos'] = len(row['completion_ids'])
                for opponent in ('nearest', 'value'):
                    for seed in plan['schedule']['teacher_maps']:
                        for seat in (0, 1):
                            programs = [refs[opponent], refs[opponent]]; programs[seat] = compiled
                            origins = ['scripted', 'scripted']; origins[seat] = 'model'
                            replay = rt.arena.play(programs, seed, 48, origins=origins); rt.arena.verify_replay(replay)
                            metrics = replay['metrics']
                            row['candidate_program_faults'] += metrics['program_faults'][seat]
                            row['opponent_program_faults'] += metrics['program_faults'][1-seat]
                            row['positive_banking_games'] += int(metrics['scores'][seat] > 0)
                            row['banked_points_sum'] += metrics['scores'][seat]
                            row['matches'].append({'opponent_id': opponent, 'candidate_seat': seat, 'replay': replay})
                row['qualified'] = qualifies(row)
            path = run/'evaluation'/f'teacher-{index:02d}.json'
            if path.exists(): require(read(path) == row, 'Existing teacher grade differs')
            else: write_new(path, row)
            rows.append(row)
    chosen = select(rows)
    ledger_path = run/'teacher-ledger-final.json'
    ledger = teacher.checked_ledger(sha(run/'plan.json'))
    require(len(ledger['requests']) == 32 and all(e['state'] == 'completed' for e in ledger['requests'].values()), 'Full completed teacher journal required')
    if ledger_path.exists(): require(read(ledger_path) == ledger, 'Final teacher journal changed')
    else: write_new(ledger_path, ledger)
    result = {'format_version': 1, 'kind': 'verified_cardinal_teacher_corpus_result',
        'plan_sha256': sha(run/'plan.json'), 'all_attempts': 32, 'required_demonstrations': 16,
        'clean_prompt_sha256': sha(run/'prompts/learner.txt'), 'teacher_ledger_sha256': sha(ledger_path),
        'delivery_gate_sha256': sha(run/'delivery-gate.json'),
        'state': 'ready_for_sft' if chosen else 'insufficient_verified_demonstrations',
        'selected_indices': chosen, 'qualified_attempts': [r['index'] for r in rows if r['qualified']],
        'generation_sha256': {str(i): digest for i, (_, digest) in enumerate(records)},
        'evaluation_sha256': {str(i): sha(run/'evaluation'/f'teacher-{i:02d}.json') for i in range(32)},
        'demonstrations': [{'index': i, 'messages': [{'role': 'user', 'content': (run/'prompts/learner.txt').read_text()},
            {'role': 'assistant', 'content': rows[i]['classification']['program']['python_source']}],
            'completion_ids': rows[i]['completion_ids'], 'generation_sha256': rows[i]['generation_sha256'],
            'program': rows[i]['classification']['program']} for i in chosen],
        'total_actual_api_cost_usd': sum(r['cost_usd'] for r, _ in records),
        'total_conservative_api_cost_usd': sum(r['cost_upper_usd'] for r, _ in records),
        'no_generated_python_execution': True, 'no_gradient_training': True}
    if (run/'corpus.json').exists(): require(read(run/'corpus.json') == result, 'Existing corpus result differs')
    else: write_new(run/'corpus.json', result)
    return {k: result[k] for k in ('state', 'selected_indices', 'qualified_attempts', 'total_actual_api_cost_usd')}


def verify_corpus(run, replay=True, require_active=True):
    """Self-contained CPU admission for SFT; requires exactly16 verified rows.

No global ledger, API credentials, model weights, or unarchived teacher output
is needed. Returned demonstrations contain original source and exact EOS IDs.
"""
    from tokenizers import Tokenizer
    replay_check = replay
    run = Path(run).resolve(); plan = verify_plan(run, require_active); corpus = read(run/'corpus.json')
    require(corpus['kind'] == 'verified_cardinal_teacher_corpus_result'
        and corpus['state'] == 'ready_for_sft' and corpus['all_attempts'] == 32
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
    require(chosen == corpus['selected_indices'] and len(chosen) == 16
        and corpus['qualified_attempts'] == [r['index'] for r in rows if r['qualified']], 'Teacher selection changed')
    expected_demos = [{'index': i, 'messages': [{'role': 'user', 'content': (run/'prompts/learner.txt').read_text()},
        {'role': 'assistant', 'content': rows[i]['classification']['program']['python_source']}],
        'completion_ids': rows[i]['completion_ids'], 'generation_sha256': rows[i]['generation_sha256'],
        'program': rows[i]['classification']['program']} for i in chosen]
    require(corpus['demonstrations'] == expected_demos, 'SFT demonstrations differ from selected teacher sources')
    return {**corpus, 'corpus_sha256': sha(run/'corpus.json'), 'source_sha256': plan['source_sha256'],
        'input_sha256': plan['input_sha256'], 'replays_independently_verified': replay_check}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'advance', 'evaluate'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--previous', type=Path, default=ROOT/'runs/cardinal-teacher-none-v1')
    parser.add_argument('--audit-source', type=Path, default=ROOT/'scripts/audit_cardinal_teacher_none.py')
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.run, args.previous, args.audit_source)
    elif args.command == 'advance': result = advance(args.run)
    else: result = evaluate(args.run)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
