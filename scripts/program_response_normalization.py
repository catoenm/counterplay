"""Versioned, deterministic response packaging policy for the v4 diagnostic.

This only removes packaging around one unambiguous Python code fence. The
enclosed bytes represented by the response string are returned unchanged.
It never parses, repairs, combines, or executes candidate program code.
"""
import re


NORMALIZATION_VERSION = 2
NORMALIZATION_MODE = "single_python_fence_with_optional_prose"


def normalize_final(text):
    if text.count("```") != 2 or "~~~" in text:
        return text, "none"
    # Both delimiters must occupy their own lines. An opening delimiter is
    # exactly ```python; closing it must not open another labelled block.
    matches = list(re.finditer(
        r"^[ \t]*```python[ \t]*\r?\n(?P<source>.*?)^[ \t]*```[ \t]*(?:\r?\n|\Z)",
        text, flags=re.M | re.S))
    if len(matches) != 1:
        return text, "none"
    match = matches[0]
    prose = bool(text[:match.start()].strip() or text[match.end():].strip())
    kind = "extracted_single_python_fence_with_prose" if prose else "stripped_single_wrapping_python_fence"
    return match["source"], kind
