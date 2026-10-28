"""Separate low-effort delivery calibration over the unchanged cardinal game.

The frozen two-call game's compiler, final-answer extraction, practice seeds,
seat attribution and bounded replay engine are reused without semantic repairs.
Only the authoring instructions and factual delivery feedback are amended.
"""
from __future__ import annotations

from . import program_debug as previous

HARNESS_VERSION = 1
MODEL_CALLS = 2
PRACTICE_MAP_COUNT = 2
REASONING_EFFORT = 'low'
REVISION_CONTEXT_POLICY = previous.REVISION_CONTEXT_POLICY
CHECKLIST = (
    'Review the delivery status, compiler result and observed actions. Check: '
    'read-only sensor names; one literal action per call; coordinate bounds; '
    'energy before moving or harvesting; cargo capacity; reaching your own home '
    'and depositing nonempty cargo; and a legal return on every path. '
    'Keep analysis brief and put only your complete controller in the final answer. '
    'This is the second and final model call; its revision is used even if the '
    'draft performed better. Do not describe a patch or provide alternatives.'
)


def debug_model_spec():
    # Preserve the complete lifecycle/rules, omitting the previous introductory
    # instruction that spelled out internal chat delimiters in the user prompt.
    _, rules = previous.debug_model_spec().split('LIFECYCLE:', 1)
    value = (
        'Write one complete Python controller beginning with def act():. '
        'Keep analysis brief and focused. Put only the complete controller in '
        'your final answer, without commentary or a patch. Aim for a simple '
        'controller of about 40 lines or fewer.\n\nLIFECYCLE:' + rules
    )
    if '<think>' in value or '</think>' in value:
        raise ValueError('Internal chat delimiters must not appear in authoring instructions')
    return value


def initial_messages():
    return [{'role': 'user', 'content': debug_model_spec()}]


def classify_debug(text, finish_reason):
    """Use the exact frozen one-boundary extraction and cardinal compiler."""
    return previous.classify_debug(text, finish_reason)


assert_map_separation = previous.assert_map_separation


def _feedback_text(feedback):
    return 'PRACTICE TEST RESULT\n' + previous._canonical(feedback) + '\n\n' + CHECKLIST


def practice_feedback(record, seeds, max_rounds=48):
    evidence = previous.practice_feedback(record, seeds, max_rounds=max_rounds)
    # Delivery status is factual metadata. Missing counters remain unknown.
    evidence['calibration_harness_version'] = HARNESS_VERSION
    evidence['feedback']['calibration_harness_version'] = HARNESS_VERSION
    evidence['feedback']['delivery'] = {
        'finish_reason': record['finish_reason'],
        'output_tokens': record.get('output_tokens'),
        'completion_token_limit': record.get('completion_token_limit'),
        'reasoning_effort_requested': record.get('reasoning_effort'),
        'token_limit_reached': record['finish_reason'] == 'length',
    }
    evidence['feedback_text'] = _feedback_text(evidence['feedback'])
    evidence['feedback_sha256'] = previous._sha(evidence['feedback_text'])
    return evidence


def revision_messages(initial, draft_record, evidence):
    if initial != initial_messages():
        raise ValueError('Initial messages differ from the calibration specification')
    if (evidence.get('calibration_harness_version') != HARNESS_VERSION
            or evidence.get('record') != draft_record
            or evidence.get('record_sha256') != previous._sha(previous._canonical(draft_record))):
        raise ValueError('Practice evidence is not bound to this exact draft')
    if (evidence.get('feedback_sha256') != previous._sha(evidence.get('feedback_text', ''))
            or evidence.get('feedback_text') != _feedback_text(evidence['feedback'])):
        raise ValueError('Practice feedback differs from canonical tool text')
    expected_delivery = {
        'finish_reason': draft_record['finish_reason'],
        'output_tokens': draft_record.get('output_tokens'),
        'completion_token_limit': draft_record.get('completion_token_limit'),
        'reasoning_effort_requested': draft_record.get('reasoning_effort'),
        'token_limit_reached': draft_record['finish_reason'] == 'length',
    }
    if evidence['feedback'].get('delivery') != expected_delivery:
        raise ValueError('Delivery feedback differs from the actual sampled record')
    classification, _ = classify_debug(draft_record['text'], draft_record['finish_reason'])
    if evidence['classification'] != classification:
        raise ValueError('Draft classification differs from original response')
    content = previous._previous_content(classification, draft_record)
    if evidence.get('revision_context') != {
            'policy': REVISION_CONTEXT_POLICY,
            'assistant_content_sha256': previous._sha(content)}:
        raise ValueError('Revision assistant-prefix provenance differs')
    return previous._copy(initial) + [{'role': 'assistant', 'content': content},
                                     {'role': 'user', 'content': evidence['feedback_text']}]
