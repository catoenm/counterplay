"""CPU-only, two-call author/test/revise harness over the frozen cardinal game.

The caller generates and journals both model responses. These pure functions
extract a final answer, execute bounded IR on practice maps, and construct the
second conversation. They never execute Python, repair code, choose a best
attempt, call a model, or touch training/evaluation maps.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math

from .program_arena import RobotArena, play, verify_replay
from .program_arena.examples import reference_programs
from .program_cardinal import classify_cardinal, cardinal_model_spec


DEBUG_VERSION = 1
MODEL_CALLS = 2
PRACTICE_MAP_COUNT = 2
PRACTICE_OPPONENTS = ('nearest', 'value')
TRACE_TURNS = (0, 1, 2, 3, 4, 5, 6, 7, 23, 47)
REVISION_CONTEXT_POLICY = 'exact_compiled_source_else_exact_final_suffix_else_raw_completion; prior_reasoning_archived_not_replayed'
CHECKLIST = (
    'Review the compiler result and observed actions. Check: read-only sensor names; '
    'one literal action per call; coordinate bounds; energy before moving or harvesting; '
    'cargo capacity; reaching your own home and depositing nonempty cargo; and a legal '
    'return on every path. Revise your complete controller. This is the second and final '
    'model call; the final revision is used even if the draft performed better. '
    'Finish reasoning with the normal </think> boundary, then output only the complete '
    'controller. Do not describe a patch or provide multiple alternatives.'
)


def _sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _copy(value):
    # Artifacts must be portable JSON. No custom objects or nonfinite numbers.
    return json.loads(_canonical(value))


def debug_model_spec():
    """Lifecycle clarification, without a worked controller or repaired policy."""
    original = cardinal_model_spec()
    _, remaining = original.split('\n', 1)
    return (
        'Design one complete controller: def act(): . Think through the rules first. '
        'Finish reasoning with the model\'s normal </think> boundary. Only the final '
        'answer after that boundary is tested; put the complete Python source there. '
        'Keep reasoning concise: check the initially empty home, the return with cargo, '
        'and energy. Aim for a simple controller of about 40 lines or fewer.\n\n'
        'LIFECYCLE: A new game starts with your robot AT ITS OWN HOME, empty cargo, '
        'zero banked points, and full energy. The environment calls act() exactly once '
        'per round with the current sensors. Its first return selects ONE action for '
        'that round; it does not finish the game. The same controller is called again '
        'on the next round with updated sensors. Locals reset each call; only m0..m15 '
        'persist within that game, starting at zero. New games reset everything.\n'
        'OBJECTIVE: Bank resources, not merely stand at home or collect cargo. You '
        'must move to a resource, harvest, move back to YOUR home, and deposit carried '
        'cargo to earn points. Deposit with empty cargo earns zero. There is no '
        'automatic navigation, harvesting, return home, or deposit. A controller must '
        'handle both home positions, changed resource availability, and low energy.\n'
        'AUTHORING PROCESS: There are exactly two model calls: this draft, then one '
        'mandatory final revision after a deterministic practice-test report. The '
        'second response is used even if the draft was better. '
        'The test uses two practice maps, both seats, and two fixed reference bots. '
        'Only practice feedback is provided; no training-reward or heldout maps are '
        'shown. All drafts and reasoning are retained. No code is repaired by the harness.\n'
        'NAMING: score, energy, carrying, turn, horizon and all other sensors are '
        'read-only. Use separate local names such as best_value, candidate_value, '
        'target_x and target_y. Never assign to a sensor.\n\n' + remaining
    )


def initial_messages():
    return [{'role': 'user', 'content': debug_model_spec()}]


def classify_debug(text, finish_reason):
    """Classify only the exact suffix after one unambiguous reasoning boundary.

    A native chat template can open <think> before the sampled completion, so
    an opening tag is optional here. If present, it must be the first non-space
    text and occur once. All boundaries and original bytes remain auditable.
    """
    if not isinstance(text, str):
        return {'valid': False, 'reason': 'invalid_response_type', 'debug_version': DEBUG_VERSION}, None
    try:
        raw_hash = _sha(text)
    except UnicodeError:
        return {'valid': False, 'reason': 'invalid_response_encoding', 'debug_version': DEBUG_VERSION}, None
    base = {'debug_version': DEBUG_VERSION, 'original_response_sha256': raw_hash, 'finish_reason': finish_reason,
            'final_answer': None, 'final_answer_sha256': None}
    if text.count('</think>') != 1:
        return {**base, 'valid': False, 'reason': 'missing_thinking_boundary' if '</think>' not in text
                else 'ambiguous_thinking_boundary'}, None
    close = text.index('</think>')
    before, final = text[:close], text[close + len('</think>'):]
    opening = text.count('<think>')
    if opening > 1 or (opening == 1 and not before.lstrip().startswith('<think>')):
        return {**base, 'valid': False, 'reason': 'ambiguous_thinking_opening'}, None
    start = before.index('<think>') + len('<think>') if opening else 0
    extraction = {'mode': 'suffix_after_single_thinking_boundary', 'opening_tag_in_completion': bool(opening),
                  'reasoning_start': start, 'reasoning_end': close, 'final_start': close + len('</think>'),
                  'reasoning_sha256': _sha(text[start:close]), 'offset_units': 'Unicode code points'}
    cardinal, compiled = classify_cardinal(final, finish_reason)
    result = {**cardinal, **base, 'valid': cardinal['valid'], 'final_answer': final,
              'final_answer_sha256': _sha(final), 'debug_extraction': extraction,
              'cardinal_classification': cardinal}
    return result, compiled


def _seed_list(seeds, *, exact=None):
    if not isinstance(seeds, (list, tuple)) or any(type(s) is not int for s in seeds):
        raise ValueError('Seeds must be a list of integers')
    if len(set(seeds)) != len(seeds) or (exact is not None and len(seeds) != exact):
        raise ValueError('Expected distinct seeds with the prescribed count')
    return list(seeds)


def _map(seed, max_rounds=48):
    game = RobotArena(seed, max_rounds)
    layout = {'grid_size': game.size, 'capacity': game.capacity, 'max_energy': game.max_energy,
              'max_rounds': game.max_rounds, 'sites': game.sites,
              'homes': [[r['home_x'], r['home_y']] for r in game.robots]}
    return {'seed': seed, 'sha256': _sha(_canonical(layout)), 'layout': layout}


def assert_map_separation(practice_seeds, reserved_seeds):
    """Preparation guard: callers supply every train/evaluation map to exclude.

    Site order is observable and kept; hidden harvest priority and seed labels
    are excluded from the layout fingerprint. Reserved maps never enter prompts.
    """
    practice = _seed_list(practice_seeds, exact=PRACTICE_MAP_COUNT)
    reserved = _seed_list(reserved_seeds)
    if set(practice) & set(reserved):
        raise ValueError('Practice and reserved seed overlap')
    maps, others = [_map(s) for s in practice], [_map(s) for s in reserved]
    fingerprints = [m['sha256'] for m in maps]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError('Practice maps have identical observable layouts')
    if set(fingerprints) & {m['sha256'] for m in others}:
        raise ValueError('Practice and reserved observable layouts overlap')
    return {'definition': 'Initial observable layout, including site order; seed and hidden priority excluded',
            'practice': maps, 'reserved': [{'seed': m['seed'], 'sha256': m['sha256']} for m in others]}


def _action_name(action):
    if action['action'] != 'move':
        return action['action']
    return {(0, -1): 'north', (0, 1): 'south', (1, 0): 'east', (-1, 0): 'west'}[(action['dx'], action['dy'])]


def _summarize(classification, matches):
    totals, outcomes, faults = Counter(), Counter(), Counter()
    other_faults, points, returns, summaries, traces = 0, [], [], [], []
    for index, match in enumerate(matches):
        record, seat = match['replay'], match['candidate_seat']
        metrics = record['metrics']
        points.append(metrics['scores'][seat])
        returns.append(record['returns'][seat])
        other_faults += metrics['program_faults'][1-seat]
        totals.update({k: metrics[k][seat] for k in ('scores', 'banked_units', 'harvested_units', 'carried_value',
                       'moves', 'rests', 'deposits', 'failed_actions', 'program_faults')})
        summaries.append({'seed': record['seed'], 'opponent': match['opponent_id'], 'seat': seat,
                          'banked_points': points[-1], 'opponent_banked_points': metrics['scores'][1-seat],
                          'return': returns[-1], 'candidate_faults': metrics['program_faults'][seat],
                          'failed_actions': metrics['failed_actions'][seat]})
        trace = []
        for event in record['events']:
            decision, obs = event['decisions'][seat], event['observations'][seat]
            outcome = event['state']['history'][-1]['outcomes'][str(seat)]
            outcomes[outcome] += 1
            if decision['fault']:
                faults[decision['fault']] += 1
            if index < 2 and event['turn'] in TRACE_TURNS:
                after = event['state']['robots'][seat]
                trace.append([event['turn'], obs['self_x'], obs['self_y'], obs['energy'], obs['carrying'],
                              _action_name(decision['action']), outcome, decision['fault'],
                              after['x'], after['y'], after['energy'], len(after['cargo']), after['score']])
        if index < 2:
            traces.append({'seed': record['seed'], 'opponent': match['opponent_id'], 'seat': seat, 'rows': trace})
    valid = bool(classification['valid'])
    count = len(matches)
    summary = {'authorship_valid': valid, 'runtime_valid': valid and totals['program_faults'] == 0,
               'games_played': count, 'prescribed_games_if_compiled': 8,
               'candidate_program_faults': totals['program_faults'], 'opponent_program_faults': other_faults,
               'positive_banking_games': sum(p > 0 for p in points),
               'banked_point_yield': math.fsum(points)/count if count else 0.0,
               'game_return': math.fsum(returns)/count if count else None,
               'candidate_totals': dict(totals), 'candidate_outcomes': dict(sorted(outcomes.items())),
               'candidate_fault_reasons': dict(sorted(faults.items()))}
    return summary, summaries, traces


def _previous_content(classification, record):
    if classification['valid']:
        return classification['program']['python_source']
    final = classification.get('final_answer')
    return record['text'] if final is None else final


def practice_feedback(record, seeds, max_rounds=48):
    """Run a fixed practice tool and return full evidence plus compact feedback.

    ``record`` is the entire original backend generation dict (including decoded
    reasoning, tokens, likelihoods and provenance when supplied). Only its text
    and finish_reason fields affect compilation. Evidence is returned unchanged
    for the caller's append-only journal; no files or model calls occur here.
    """
    if not isinstance(record, dict) or not isinstance(record.get('text'), str) or not isinstance(record.get('finish_reason'), str):
        raise ValueError('Generation record requires text and finish_reason strings')
    if type(max_rounds) is not int or max_rounds != 48:
        raise ValueError('Practice uses the unchanged 48-round game')
    original = _copy(record)
    seeds = _seed_list(seeds, exact=PRACTICE_MAP_COUNT)
    maps = assert_map_separation(seeds, [])['practice']
    classification, candidate = classify_debug(record['text'], record['finish_reason'])
    matches = []
    references = reference_programs()
    if candidate is not None:
        for opponent in PRACTICE_OPPONENTS:
            for seed in seeds:
                for seat in (0, 1):
                    players = [candidate, references[opponent]] if seat == 0 else [references[opponent], candidate]
                    origins = ['model', 'scripted'] if seat == 0 else ['scripted', 'model']
                    replay = play(players, seed, max_rounds, origins=origins)
                    verify_replay(replay)
                    matches.append({'opponent_id': opponent, 'candidate_seat': seat, 'replay': replay})
    summary, match_summaries, traces = _summarize(classification, matches)
    feedback = {'tool': 'test_cardinal_controller', 'debug_version': DEBUG_VERSION,
                'scope': 'practice only', 'classification': {k: classification[k] for k in
                    ('valid', 'reason', 'detail', 'normalization') if k in classification},
                'summary': summary, 'match_summaries': match_summaries, 'practice_maps': maps,
                'trace_selection': 'First practice map versus nearest, both seats; fixed zero-based turns 0..7,23,47',
                'trace_columns': ['turn', 'x_before', 'y_before', 'energy_before', 'cargo_before',
                                  'action', 'outcome', 'program_fault', 'x_after', 'y_after',
                                  'energy_after', 'cargo_after', 'banked_points_after'],
                'traces': traces,
                'interpretation': 'Counts and traces refer to the candidate seat. Ordinary failed actions consume a round but are not interpreter faults. Invalid authorship has zero played games, zero yield and no game return.'}
    feedback_text = 'PRACTICE TEST RESULT\n' + _canonical(feedback) + '\n\n' + CHECKLIST
    return {'debug_version': DEBUG_VERSION, 'record': original, 'record_sha256': _sha(_canonical(original)),
            'classification': classification, 'matches': matches, 'maps': maps, 'summary': summary,
            'feedback': feedback, 'feedback_text': feedback_text, 'feedback_sha256': _sha(feedback_text),
            'revision_context': {'policy': REVISION_CONTEXT_POLICY,
                                 'assistant_content_sha256': _sha(_previous_content(classification, original))},
            'authoring_model_calls': MODEL_CALLS, 'final_attempt_index': 1, 'generated_python_executed': False}


def revision_messages(initial, draft_record, feedback):
    """Create the sole revision context, with no historical-reasoning invention.

    Prior reasoning is retained in feedback.record, not fed back as invented
    assistant reasoning. The canonical assistant prefix is exact extracted code
    when compilation succeeded, exact final-answer suffix when invalid, or the
    full raw completion when no unambiguous final boundary exists.
    """
    if initial != initial_messages():
        raise ValueError('Initial messages differ from the fixed debug specification')
    if feedback.get('debug_version') != DEBUG_VERSION or feedback.get('record') != draft_record:
        raise ValueError('Feedback is not bound to this draft')
    if feedback.get('record_sha256') != _sha(_canonical(draft_record)):
        raise ValueError('Draft record hash mismatch')
    if feedback.get('feedback_sha256') != _sha(feedback.get('feedback_text', '')):
        raise ValueError('Feedback text hash mismatch')
    expected = 'PRACTICE TEST RESULT\n' + _canonical(feedback['feedback']) + '\n\n' + CHECKLIST
    if expected != feedback['feedback_text']:
        raise ValueError('Feedback differs from canonical tool text')
    classification, _ = classify_debug(draft_record['text'], draft_record['finish_reason'])
    if classification != feedback['classification']:
        raise ValueError('Draft classification differs from original response')
    previous = _previous_content(classification, draft_record)
    if feedback.get('revision_context') != {'policy': REVISION_CONTEXT_POLICY, 'assistant_content_sha256': _sha(previous)}:
        raise ValueError('Revision assistant-prefix provenance differs')
    return _copy(initial) + [{'role': 'assistant', 'content': previous},
                             {'role': 'user', 'content': feedback['feedback_text']}]
