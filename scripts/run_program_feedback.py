"""Frozen, CPU-only practice feedback; paid authoring only with --advance.

Generated Python is parsed by the archived bounded frontend, never executed.
Prepare freezes originals, sources, seeds, selection rule and prompt formatting.
Four sequential revisions precede any untouched holdout evaluation.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from selfplay import program_feedback as client

PRACTICE_SEEDS = list(range(730000, 730006))
HOLDOUT_SEEDS = list(range(740000, 740006))
DIAGNOSTIC_SOURCES = {
    'runner.py': ROOT / 'scripts/run_program_feedback.py',
    'screen_evaluator.py': ROOT / 'scripts/evaluate_program_screen.py',
    'program_feedback.py': ROOT / 'selfplay/program_feedback.py',
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def read(path):
    return json.loads(path.read_bytes())


def publish(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f'Refusing incompatible existing artifact: {path}')
        return
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    try:
        with temp.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def save(path, value):
    publish(path, encoded(value))


def import_bytes(name, path, expected):
    raw = path.read_bytes()
    if sha(raw) != expected:
        raise ValueError(f'Archived source hash differs: {path}')
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def validate_plan(plan):
    if (plan['practice_seeds'] != PRACTICE_SEEDS or plan['holdout_seeds'] != HOLDOUT_SEEDS
            or set(plan['practice_seeds']) & set(plan['holdout_seeds'])
            or set(plan['original_validation_seeds']) & (set(PRACTICE_SEEDS) | set(HOLDOUT_SEEDS))
            or plan['lineages'] != list(client.LINEAGES) or plan['revisions_per_lineage'] != 2
            or plan['max_paid_calls'] != 4 or plan['subcap_usd'] != .1
            or plan['selection_rule'] != 'round_2_even_if_invalid'
            or plan['fixed_opponents'] != ['nearest', 'value'] or plan['max_rounds'] != 48
            or plan['feedback_trace_rule'] != 'first_practice_seed_nearest_both_seats_all_turns'):
        raise ValueError('Frozen feedback protocol differs')
    if plan['client_configuration'] != client.configuration():
        raise ValueError('Feedback client configuration differs')


@contextmanager
def runtime_for(run):
    plan = read(run / 'plan.json')
    validate_plan(plan)
    for name, expected in plan['diagnostic_source_sha256'].items():
        if sha((run / 'diagnostic_source' / name).read_bytes()) != expected:
            raise ValueError(f'Diagnostic source archive differs: {name}')
    if sha(Path(__file__).read_bytes()) != plan['diagnostic_source_sha256']['runner.py']:
        raise ValueError('Live runner differs from frozen diagnostic')
    if sha(Path(client.__file__).read_bytes()) != plan['diagnostic_source_sha256']['program_feedback.py']:
        raise ValueError('Live client differs from frozen diagnostic')
    for name, expected in plan['input_sha256'].items():
        if sha((run / name).read_bytes()) != expected:
            raise ValueError(f'Frozen diagnostic input differs: {name}')
    evaluator = import_bytes('_feedback_evaluator_' + uuid.uuid4().hex,
                             run / 'diagnostic_source/screen_evaluator.py',
                             plan['diagnostic_source_sha256']['screen_evaluator.py'])
    base_plan = read(run / 'original-plan.json')
    with evaluator.archived_runtime(run, base_plan) as runtime:
        yield plan, evaluator, runtime


def prepare(run, original):
    run, original = Path(run).resolve(), Path(original).resolve()
    source_plan = read(original / 'plan.json')
    if source_plan['validation_seeds'] != list(range(720000, 720006)):
        raise ValueError('Expected the exact original Qwen screen seeds')
    if source_plan['model'] != client.PROFILES['qwen38-27b']['model']:
        raise ValueError('Original screen model differs')
    run.mkdir(parents=True, exist_ok=True)
    for rel in ('plan.json', 'prompt.txt', 'generation-1.json', 'generation-2.json'):
        dest = {'plan.json': 'original-plan.json', 'prompt.txt': 'original-prompt.txt'}.get(rel, 'originals/' + rel)
        publish(run / dest, (original / rel).read_bytes())
    # Copy only the frozen sources/configuration, retaining the exact base plan.
    names = {Path(p).name for p in source_plan['source_sha256']}
    names |= {'response_normalization.py', 'model-screen-profiles.json'}
    for name in sorted(names):
        publish(run / 'source' / name, (original / 'source' / name).read_bytes())
    for name, path in DIAGNOSTIC_SOURCES.items():
        publish(run / 'diagnostic_source' / name, path.read_bytes())
    inputs = ['original-plan.json', 'original-prompt.txt', 'originals/generation-1.json', 'originals/generation-2.json']
    plan = {
        'format_version': 1, 'diagnostic': 'two_round_practice_feedback', 'original_run': str(original),
        'lineages': list(client.LINEAGES), 'revisions_per_lineage': 2, 'max_paid_calls': 4, 'subcap_usd': .1,
        'practice_seeds': PRACTICE_SEEDS, 'holdout_seeds': HOLDOUT_SEEDS,
        'original_validation_seeds': source_plan['validation_seeds'],
        'fixed_opponents': ['nearest', 'value'], 'paired_seats': True, 'max_rounds': 48,
        'selection_rule': 'round_2_even_if_invalid', 'training': False,
        'feedback_trace_rule': 'first_practice_seed_nearest_both_seats_all_turns',
        'holdout_rule': 'Only originals and final round-2 versions; unlock after all four generations and practice evaluations',
        'client_configuration': client.configuration(),
        'input_sha256': {name: sha((run / name).read_bytes()) for name in inputs},
        'diagnostic_source_sha256': {name: sha(path.read_bytes()) for name, path in DIAGNOSTIC_SOURCES.items()},
        'interpretation': 'Adaptive authoring diagnostic, not gradient RL. Two lineages and a reused practice set; no outcome-based version selection. Holdout is not sent to the model. No hand repairs or reference code in prompts.',
    }
    save(run / 'plan.json', plan)
    with runtime_for(run) as (plan, evaluator, runtime):
        originals = original_generations(run, plan, evaluator)
        for lineage, generation in originals.items():
            evaluate_version(run, lineage, 'practice', generation, plan, evaluator, runtime)
            prompt_for(run, lineage, 1, generation, plan)
    return {'status': 'prepared_no_paid_calls', 'run': str(run), 'plan_sha256': sha((run / 'plan.json').read_bytes())}


def original_generations(run, plan, evaluator):
    base_plan = read(run / 'original-plan.json')
    # Existing evaluator provenance validation expects generation-N at its run root.
    result = {}
    for index, lineage in enumerate(plan['lineages'], 1):
        generation = read(run / f'originals/generation-{index}.json')
        if (generation['sample_id'] != lineage or generation['returned_model'] != base_plan['model']
                or generation['prompt_sha256'] != base_plan['prompt_sha256']
                or sha(generation['prompt'].encode()) != base_plan['prompt_sha256']
                or generation['client_source_sha256'] != base_plan['source_sha256']['selfplay/program_model_screen.py']
                or generation['generation_client_profile'] != base_plan['generation_client_profile']
                or generation['generation_client_configuration_sha256'] != base_plan['generation_client_configuration_sha256']
                or generation['returned_provider'] != 'Chutes'):
            raise ValueError('Original generation provenance differs')
        result[lineage] = generation
    return result


def schedule(identity, seeds):
    return [{'episode_id': f'{identity}-{seed}-{opponent}-seat{seat}',
             'players': [identity, opponent] if seat == 0 else [opponent, identity],
             'seed': seed, 'max_rounds': 48, 'match_kind': 'fixed_reference'}
            for opponent in ('nearest', 'value') for seed in seeds for seat in (0, 1)]


def feedback_from(wrappers, identity, classification, evaluator):
    """All game totals/failure turns, plus two traces chosen before any outcomes."""
    feedback = {'valid': classification['valid'], 'failure': classification['failure'],
                'compiler_error': classification.get('error'), 'normalization': classification['normalization'],
                'match_summaries': [], 'traces': [],
                'trace_columns': ['turn', 'x', 'y', 'energy', 'cargo_units', 'action', 'outcome', 'program_fault', 'banked_points', 'opponent_banked_points']}
    for wrapper in wrappers:
        seat = wrapper['players'].index(identity)
        opponent = wrapper['players'][1-seat]
        record = wrapper['record']
        failed = {}
        trace = []
        for event in record['events']:
            decision, obs = event['decisions'][seat], event['observations'][seat]
            outcome = event['state']['history'][-1]['outcomes'][str(seat)]
            scores = event['state']['metrics']['scores']
            row = [event['turn'], obs['self_x'], obs['self_y'], obs['energy'], obs['carrying'],
                   decision['action'], outcome, decision['fault'], scores[seat], scores[1-seat]]
            trace.append(row)
            if outcome not in {'moved', 'harvested', 'deposited', 'rested'} or decision['fault'] is not None:
                key = json.dumps([decision['action'], outcome, decision['fault'], scores[seat], scores[1-seat]], sort_keys=True)
                failed.setdefault(key, []).append(event['turn'])
        feedback['match_summaries'].append({
            'seed': record['seed'], 'seat': seat, 'opponent': opponent,
            'banked_points': record['metrics']['scores'][seat], 'opponent_banked_points': record['metrics']['scores'][1-seat],
            'return': record['returns'][seat], 'program_faults': record['metrics']['program_faults'][seat],
            'failed_actions': record['metrics']['failed_actions'][seat],
            'failed_turn_groups': [{'turns': turns, 'action_outcome_fault_points_opponent_points': json.loads(key)}
                                   for key, turns in failed.items()]})
        if record['seed'] == PRACTICE_SEEDS[0] and opponent == 'nearest':
            feedback['traces'].append({'seed': record['seed'], 'seat': seat, 'opponent': opponent, 'rows': trace})
    feedback['overall'] = evaluator.aggregate([(w['record'], w['players'].index(identity)) for w in wrappers])
    return feedback


def evaluate_version(run, identity, split, generation, plan, evaluator, runtime):
    if split not in ('practice', 'holdout'):
        raise ValueError('Unknown evaluation split')
    directory = run / split / identity
    directory.mkdir(parents=True, exist_ok=True)
    classification, program = evaluator.classify_generation(generation, runtime)
    generation_hash = sha(encoded(generation))
    specs = schedule(identity, plan[split + '_seeds']) if program else []
    manifest = {'plan_sha256': sha((run / 'plan.json').read_bytes()), 'identity': identity, 'split': split,
                'generation_content_sha256': generation_hash, 'classification': classification, 'schedule': specs}
    manifest_path, matches_path = directory / 'evaluation.json', directory / 'matches.jsonl'
    if not manifest_path.exists() and any((directory / n).exists() for n in ('matches.jsonl', 'summary.json', 'feedback.json')):
        raise ValueError('Existing results have no evaluation provenance')
    save(manifest_path, manifest)
    programs = {name: runtime.compile_program(source) for name, source in runtime.reference_programs().items() if name in plan['fixed_opponents']}
    python_programs = {identity: program} if program else {}
    if program:
        programs[identity] = program.compiled
    content = matches_path.read_bytes() if matches_path.exists() else b''
    if content and not content.endswith(b'\n'):
        raise ValueError('Partial match line retained; no automatic replacement')
    wrappers = [json.loads(line) for line in content.splitlines()]
    if len(wrappers) > len(specs):
        raise ValueError('Extra games beyond frozen schedule')
    if (directory / 'summary.json').exists() and len(wrappers) != len(specs):
        raise ValueError('Completed summary has an incomplete match archive')
    for spec, wrapper in zip(specs, wrappers):
        evaluator.verify_wrapper(wrapper, spec, runtime, programs, python_programs)
    with matches_path.open('ab') as stream:
        for spec in specs[len(wrappers):]:
            names = spec['players']
            record = runtime.play([programs[name] for name in names], spec['seed'], 48,
                                  origins=['model' if name == identity else 'scripted' for name in names])
            wrapper = {'episode_id': spec['episode_id'], 'players': names, 'match_kind': spec['match_kind'],
                       'record': record, 'python_programs': {identity: program.to_dict()}}
            evaluator.verify_wrapper(wrapper, spec, runtime, programs, python_programs)
            stream.write((json.dumps(wrapper, separators=(',', ':'), allow_nan=False) + '\n').encode())
            stream.flush()
            os.fsync(stream.fileno())
            wrappers.append(wrapper)
    summary = {'complete': True, 'identity': identity, 'split': split, 'classification': classification,
               'matches': len(wrappers), 'generation_content_sha256': generation_hash,
               'evaluation_sha256': sha(manifest_path.read_bytes()), 'matches_sha256': sha(matches_path.read_bytes()),
               'all_replays_verified': True, 'scores': evaluator.aggregate([(w['record'], w['players'].index(identity)) for w in wrappers])}
    save(directory / 'summary.json', summary)
    if split == 'practice':
        save(directory / 'feedback.json', feedback_from(wrappers, identity, classification, evaluator))
    return summary


def version_id(lineage, round_index):
    return lineage if round_index == 0 else f'{lineage}-r{round_index}'


def prompt_for(run, lineage, round_index, previous, plan):
    identity = version_id(lineage, round_index)
    prior_id = version_id(lineage, round_index - 1)
    feedback_raw = (run / 'practice' / prior_id / 'feedback.json').read_bytes()
    feedback = json.loads(feedback_raw)
    prompt = ((run / 'original-prompt.txt').read_text() + '\n\n'
              'PRACTICE REVISION\nRevise your own previous controller using only the factual practice results below. '
              'Return one complete replacement controller in the same required language. This is revision '
              + str(round_index) + ' of 2. No reference-controller source is provided. '
              'Turn indices are zero-based. The two full traces are fixed in advance: the first practice seed, '
              'nearest opponent, both seats. Match summaries include every practice game and all failure turns.\n'
              + 'YOUR PREVIOUS FINAL RESPONSE:\n' + (previous.get('text') or '') + '\n'
              + 'FACTUAL PRACTICE FEEDBACK (JSON):\n' + json.dumps(feedback, separators=(',', ':'), allow_nan=False))
    body = client.body_for(prompt, 'qwen38-27b')
    if len(client.canonical(body).encode()) > client.MAX_REQUEST_BYTES:
        raise ValueError('Frozen feedback exceeds request bound; retained practice data, no adaptive truncation or call')
    context = {'plan_sha256': sha((run / 'plan.json').read_bytes()), 'lineage': lineage, 'round_index': round_index,
               'previous_generation_sha256': sha(encoded(previous)), 'feedback_sha256': sha(feedback_raw),
               'original_prompt_sha256': sha((run / 'original-prompt.txt').read_bytes())}
    publish(run / 'prompts' / (identity + '.txt'), prompt.encode())
    save(run / 'prompts' / (identity + '.context.json'), context)
    return prompt, context


def verify_revision(run, generation, prompt, context, plan):
    profile = {'name': 'qwen38-27b', **client.PROFILES['qwen38-27b']}
    if (generation.get('feedback_context') != context or generation.get('prompt') != prompt
            or generation.get('prompt_sha256') != sha(prompt.encode())
            or generation.get('client_source_sha256') != plan['diagnostic_source_sha256']['program_feedback.py']
            or generation.get('generation_client_configuration_sha256') != sha(client.canonical(client.configuration()).encode())
            or generation.get('generation_client_profile') != profile
            or generation.get('generation_client') != 'program_feedback'
            or generation.get('sample_id') != version_id(context['lineage'], context['round_index'])
            or generation.get('returned_model') != profile['model'] or generation.get('returned_provider') != profile['returned_provider']
            or generation.get('request_parameters') != {k:v for k,v in client.body_for(prompt, 'qwen38-27b').items() if k != 'messages'}):
        raise ValueError('Feedback generation provenance differs')
    client.verify_completed_response(prompt, generation['sample_id'], context, generation)
    # Preserve provider-condition failures separately from compilation/game skill.
    details = generation.get('usage', {}).get('completion_tokens_details', {})
    if generation.get('reasoning_condition_status') != 'confirmed_off' or details.get('reasoning_tokens') != 0 or type(details.get('reasoning_tokens')) is not int:
        save(run / 'provider-condition-failure.json', {'sample_id': generation['sample_id'],
            'status': generation.get('reasoning_condition_status'), 'reported_reasoning_tokens': details.get('reasoning_tokens'),
            'interpretation': 'Provider did not verify the requested reasoning-off condition. Stop further calls; this is not a compile/game capability failure.'})
        raise ValueError('Provider reasoning-off condition failed or unknown; further calls stopped')


def advance(run):
    with runtime_for(run) as (plan, evaluator, runtime):
        current = original_generations(run, plan, evaluator)
        for round_index in (1, 2):
            for lineage in plan['lineages']:
                previous = current[lineage]
                evaluate_version(run, version_id(lineage, round_index-1), 'practice', previous, plan, evaluator, runtime)
                prompt, context = prompt_for(run, lineage, round_index, previous, plan)
                identity = version_id(lineage, round_index)
                path = run / 'generations' / (identity + '.json')
                fresh = not path.exists()
                if fresh:
                    generation = client.complete(prompt, identity, context)
                    save(path, generation)
                else:
                    generation = read(path)
                verify_revision(run, generation, prompt, context, plan)
                current[lineage] = generation
                evaluate_version(run, identity, 'practice', generation, plan, evaluator, runtime)
                if fresh:
                    return {'status': 'one_revision_completed', 'sample_id': identity, 'cost_usd': generation['cost_usd']}
        return {'status': 'all_four_revisions_complete_ready_for_holdout'}


def holdout(run):
    with runtime_for(run) as (plan, evaluator, runtime):
        originals = original_generations(run, plan, evaluator)
        current, all_generations = dict(originals), []
        # Entire chain must be present and practice-evaluated before any holdout.
        for round_index in (1, 2):
            for lineage in plan['lineages']:
                evaluate_version(run, version_id(lineage, round_index-1), 'practice', current[lineage], plan, evaluator, runtime)
                prompt, context = prompt_for(run, lineage, round_index, current[lineage], plan)
                generation = read(run / 'generations' / (version_id(lineage, round_index) + '.json'))
                verify_revision(run, generation, prompt, context, plan)
                evaluate_version(run, generation['sample_id'], 'practice', generation, plan, evaluator, runtime)
                current[lineage] = generation
                all_generations.append(generation)
        summaries = {}
        for generation in list(originals.values()) + list(current.values()):
            identity = generation['sample_id']
            summaries[identity] = evaluate_version(run, identity, 'holdout', generation, plan, evaluator, runtime)
        summary = {'complete': True, 'plan_sha256': sha((run / 'plan.json').read_bytes()),
                   'training': False, 'selection_rule': plan['selection_rule'], 'holdout_seeds': HOLDOUT_SEEDS,
                   'holdout': summaries, 'paid_revision_calls': len(all_generations),
                   'new_cost_usd': sum(g['cost_usd'] for g in all_generations),
                   'new_cost_upper_usd': sum(g['cost_upper_usd'] for g in all_generations),
                   'generation_sha256': {g['sample_id']: sha((run / 'generations' / (g['sample_id'] + '.json')).read_bytes()) for g in all_generations},
                   'interpretation': plan['interpretation'], 'scoring': 'Win/draw/loss is actual banked-point difference at the fixed game end, not shaped adjudication.'}
        save(run / 'summary.json', summary)
        return {'status': 'complete', 'new_cost_usd': summary['new_cost_usd'],
                'holdout': {name: entry['scores'] for name, entry in summaries.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare-from', type=Path)
    group.add_argument('--advance', action='store_true', help='At most one new paid request, then CPU practice')
    group.add_argument('--holdout', action='store_true', help='No paid calls; requires all four revisions')
    args = parser.parse_args()
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    with (run / '.diagnostic.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = prepare(run, args.prepare_from) if args.prepare_from else advance(run) if args.advance else holdout(run)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
