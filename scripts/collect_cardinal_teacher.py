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
from selfplay import program_cardinal_teacher as teacher

GAME_SOURCES = ['selfplay/program_arena/' + name + '.py' for name in
    ('__init__', 'dsl', 'game', 'runner', 'examples', 'spec')]
RUNTIME_SOURCES = GAME_SOURCES + ['selfplay/program_python.py', 'selfplay/program_cardinal.py']
LOCAL_SOURCES = ['selfplay/program_cardinal_teacher.py', 'selfplay/program_model_screen.py',
    'scripts/collect_cardinal_teacher.py']
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
        f'Independent design {index+1}/32. Design objective: {DESIGNS[index % 16]} '
        f'Replicate {index // 16 + 1}; produce a self-contained policy, not a patch.\n\n' + spec)


def schedule():
    value = {'teacher_maps': list(range(91000000, 91000006)),
        'capability_maps': list(range(92000000, 92000006)),
        'train_maps': {str(i): [93000000+i*100+j for j in range(2)] for i in range(1, 101)},
        'holdout_maps': list(range(94000000, 94000006)),
        'teacher_generation_seeds': list(range(111000000, 111000032)),
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
        archived_teacher = load('program_cardinal_teacher', 'selfplay/program_cardinal_teacher.py')
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


def prepare(run, parent, tokenizer):
    run, parent, tokenizer = Path(run).resolve(), Path(parent).resolve(), Path(tokenizer).resolve()
    require(not run.exists(), 'Refusing to overwrite teacher corpus')
    pm, ps, pr = (read(parent / name) for name in ('manifest.json', 'status.json', 'runtime.json'))
    require(sha(parent/'manifest.json') == ps['manifest_sha256'], 'Parent manifest differs')
    require(sha(parent/'runtime.json') == ps['artifacts']['runtime.json'], 'Parent runtime differs')
    require(pr['model_index']['repo'] == MODEL and pr['model_index']['revision'] == REVISION,
        'Parent model identity differs')
    require(sha(tokenizer) == TOKENIZER_SHA256 == pr['model_index']['files']['tokenizer.json'], 'Wrong learner tokenizer')
    run.mkdir(parents=True); (run/'inputs').mkdir(); (run/'prompts').mkdir()
    sources = {}
    for relative in RUNTIME_SOURCES + LOCAL_SOURCES:
        path = parent/'source'/relative if relative in RUNTIME_SOURCES else ROOT/relative
        if relative in RUNTIME_SOURCES:
            require(sha(path) == pm['source_sha256'][relative], 'Parent source changed')
        target = run/'source'/relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target); sources[relative] = sha(target)
    inputs = {}
    for name in ('manifest.json', 'status.json', 'runtime.json'):
        target = run/'inputs'/('parent-'+name); shutil.copyfile(parent/name, target)
        inputs[str(target.relative_to(run))] = sha(target)
    shutil.copyfile(tokenizer, run/'inputs/tokenizer.json'); inputs['inputs/tokenizer.json'] = TOKENIZER_SHA256
    priors = []
    for index, path in enumerate(sorted((ROOT/'runs').glob('*/map-fingerprints.json'))):
        source_manifest = read(path.parent/'manifest.json')
        require(sha(path) == source_manifest['map_fingerprints_sha256'], 'Prior map manifest differs')
        target = run/'inputs'/f'prior-maps-{index}.json'; shutil.copyfile(path, target)
        inputs[str(target.relative_to(run))] = sha(target); priors.append(read(target))
    timetable = schedule()
    with runtime(run, sources) as rt:
        spec = rt.spec(); fingerprints = map_fingerprints(rt.arena, timetable, priors)
    (run/'prompts/learner.txt').write_text(spec); inputs['prompts/learner.txt'] = sha(run/'prompts/learner.txt')
    prompts = {}
    for i in range(32):
        path = run/'prompts'/f'teacher-{i:02d}.txt'; path.write_text(teacher_prompt(spec, i))
        prompts[str(i)] = {'path': str(path.relative_to(run)), 'sha256': sha(path)}
    write_new(run/'map-fingerprints.json', fingerprints)
    quote = teacher.verify_endpoint()
    write_new(run/'inputs/endpoint-quote.json', quote); inputs['inputs/endpoint-quote.json'] = sha(run/'inputs/endpoint-quote.json')
    plan = {'format_version': 1, 'kind': 'verified_cardinal_teacher_corpus', 'configuration': teacher.configuration(),
        'source_sha256': sources, 'input_sha256': inputs, 'teacher_prompts': prompts,
        'map_fingerprints_sha256': sha(run/'map-fingerprints.json'), 'schedule': timetable,
        'learner': {'model': MODEL, 'revision': REVISION, 'tokenizer_sha256': TOKENIZER_SHA256,
            'eos_token_id': EOS, 'maximum_completion_tokens_including_eos': 700,
            'tokenizers_version': importlib.metadata.version('tokenizers'), 'prompt_path': 'prompts/learner.txt'},
        'selection': {'attempts': 32, 'required_demonstrations': 16, 'rule': 'First16 qualifying distinctIR in attempt order; retain all32 attempts and grades; no replacement or repairs.',
            'qualification': {'maximum_learner_tokens_including_eos': 700, 'games': 24,
                'candidate_interpreter_faults': 0, 'minimum_positive_banking_games': 12},
            'insufficient': 'Stop before SFT; do not relax criteria or generate replacements.'},
        'evaluation': {'opponents': ['nearest', 'value'], 'seats': [0, 1], 'max_rounds': 48,
            'all32_responses_before_scoring': True, 'holdout_used': False},
        'maximum_initial_request_reservations_usd': sum(teacher.reservation(teacher_prompt(spec, i), i) for i in range(32)),
        'scope': 'Teacher distillation corpus only; no self-play or learned-improvement claim; teacher hints are never learner prompt content.'}
    require(plan['maximum_initial_request_reservations_usd'] <= teacher.PHASE_CAP, 'Full fixed schedule cannot fit phase cap')
    write_new(run/'plan.json', plan)
    return {'run': str(run), 'plan_sha256': sha(run/'plan.json'), 'paid_calls': 0,
        'maximum_initial_request_reservations_usd': plan['maximum_initial_request_reservations_usd']}


def verify_plan(run, require_active=True):
    run = Path(run).resolve(); plan = read(run/'plan.json')
    require(plan['kind'] == 'verified_cardinal_teacher_corpus' and plan['format_version'] == 1, 'Protocol differs')
    require(set(plan['source_sha256']) == set(RUNTIME_SOURCES + LOCAL_SOURCES), 'Frozen source inventory differs')
    require(set(plan['teacher_prompts']) == {str(i) for i in range(32)}, 'All32 frozen teacher prompts required')
    required_inputs = {'inputs/parent-manifest.json', 'inputs/parent-status.json', 'inputs/parent-runtime.json',
        'inputs/tokenizer.json', 'inputs/endpoint-quote.json', 'prompts/learner.txt'}
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


def advance(run):
    run = Path(run).resolve(); plan = verify_plan(run)
    for index in range(32):
        path = run/'generations'/f'teacher-{index:02d}.json'
        if path.exists(): generation(run, plan, index); continue
        prompt = (run/plan['teacher_prompts'][str(index)]['path']).read_text()
        record = teacher.complete(prompt, index, sha(run/'plan.json'))
        write_new(path, record)
        return {'generation': str(path), 'finish_reason': record['finish_reason'], 'cost_usd': record['cost_usd']}
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
        and corpus['teacher_ledger_sha256'] == sha(run/'teacher-ledger-final.json'), 'Corpus provenance differs')
    require(importlib.metadata.version('tokenizers') == plan['learner']['tokenizers_version'], 'Tokenizer library differs')
    tokenizer = Tokenizer.from_file(str(run/'inputs/tokenizer.json'))
    require(tokenizer.token_to_id('<|im_end|>') == EOS, 'EOS differs')
    rows = []
    with runtime(run, plan['source_sha256']) as rt:
        require((run/'prompts/learner.txt').read_text() == rt.spec(), 'Clean learner prompt differs')
        refs = {key: rt.arena.compile_program(value) for key, value in rt.references().items()}
        for index in range(32):
            record, digest = generation(run, plan, index, run/'teacher-ledger-final.json', rt.teacher)
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
    parser.add_argument('--parent', type=Path, default=ROOT/'runs/cardinal-gpu-v2')
    parser.add_argument('--tokenizer', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        require(args.tokenizer is not None, '--tokenizer is required for CPU preparation')
        result = prepare(args.run, args.parent, args.tokenizer)
    elif args.command == 'advance': result = advance(args.run)
    else: result = evaluate(args.run)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
