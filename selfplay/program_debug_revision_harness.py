"""Distinct native-nonthinking revision condition; unchanged draft practice.

No generated source is repaired and no boundary is injected into completions.
All text prompts and factual practice feedback reuse the preceding frozen low-
effort calibration. Only final-turn native mode and its whole-answer parser vary.
"""
from __future__ import annotations

from . import program_debug_calibration_harness as previous
from .program_cardinal import classify_cardinal

HARNESS_VERSION = 1
MODEL_CALLS = previous.MODEL_CALLS
PRACTICE_MAP_COUNT = previous.PRACTICE_MAP_COUNT
REASONING_EFFORT = previous.REASONING_EFFORT
REVISION_CONTEXT_POLICY = previous.REVISION_CONTEXT_POLICY
CHECKLIST = previous.CHECKLIST

# These aliases deliberately preserve exact model-visible bytes and provenance.
debug_model_spec = previous.debug_model_spec
initial_messages = previous.initial_messages
classify_debug = previous.classify_debug
assert_map_separation = previous.assert_map_separation
practice_feedback = previous.practice_feedback
revision_messages = previous.revision_messages


def classify_revision(text, finish_reason):
    """Grade the entire native-disabled completion, never a synthesized suffix.

    The original cardinal parser rejects sampled thinking tags, truncation and
    ambiguous packaging. Its only normalization is the existing documented
    single Python fence normalization, with original source provenance retained.
    """
    classification, program = classify_cardinal(text, finish_reason)
    classification = {**classification, 'revision_harness_version': HARNESS_VERSION,
        'authoring_stage': 'revision', 'thinking_enabled': False,
        'reasoning_mode': 'native_disabled', 'finish_reason': finish_reason,
        'final_answer': text,
        'revision_extraction': {'mode': 'whole_native_nonthinking_completion',
            'start': 0, 'end': len(text) if isinstance(text, str) else None,
            'offset_units': 'Unicode code points', 'injected_boundaries': 0}}
    return classification, program
