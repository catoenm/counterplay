"""Literal cardinal-action frontend over the unchanged bounded Python compiler.

Only return syntax is translated. There is no controller execution, navigation,
state-dependent action correction, or change to the game or IR interpreter.
The original source and the exact translated legacy source remain auditable.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import re

from .program_arena.dsl import CompiledProgram, MAX_BYTES
from .program_python import (
    FRONTEND_VERSION as PYTHON_FRONTEND_VERSION,
    MAX_PYTHON_AST_NODES,
    MAX_PYTHON_DEPTH,
    PythonProgramValidationError,
    compile_python_program,
)


CARDINAL_FRONTEND_VERSION = 1
INTERFACE = "literal_cardinal_returns_v1"
ACTION_NAMES = ("north", "south", "east", "west", "rest", "harvest", "deposit")
NORMALIZATION_VERSION = 2


class CardinalProgramValidationError(PythonProgramValidationError):
    pass


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CardinalProgram:
    source: str
    source_sha256: str
    lowered_python_source: str
    lowered_python_sha256: str
    compiled: CompiledProgram

    def run(self, observation, memory=None):
        return self.compiled.run(observation, memory)

    def to_dict(self):
        return {
            "source_language": "bounded_cardinal_python_syntax",
            "interface": INTERFACE,
            "frontend_version": CARDINAL_FRONTEND_VERSION,
            "python_frontend_version": PYTHON_FRONTEND_VERSION,
            "python_source": self.source,
            "python_sha256": self.source_sha256,
            "lowered_python_source": self.lowered_python_source,
            "lowered_python_sha256": self.lowered_python_sha256,
            "compiled_ir": json.loads(self.compiled.source),
            "ir_sha256": self.compiled.digest,
        }


class _LiteralReturns(ast.NodeTransformer):
    def visit_Return(self, node):
        value = node.value
        if (not isinstance(value, ast.Constant) or type(value.value) is not str
                or value.value not in ACTION_NAMES):
            raise CardinalProgramValidationError(
                f"line {node.lineno}: return one literal string: " + ", ".join(ACTION_NAMES))
        name = value.value
        vectors = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
        if name in vectors:
            dx, dy = vectors[name]
            action = {"action": "move", "dx": dx, "dy": dy}
        else:
            action = {"action": name}
        translated = ast.Dict(keys=[ast.Constant(k) for k in action],
                              values=[ast.Constant(v) for v in action.values()])
        return ast.copy_location(ast.Return(value=translated), node)


def compile_cardinal_program(source: str) -> CardinalProgram:
    """Parse literal return actions, then use the unchanged legacy compiler.

    Original source limits are checked before transforming. The legacy compiler
    also checks the translated Python and its IR, so this interface cannot widen
    any existing execution limit. Source is never passed to exec/eval/compile.
    """
    try:
        if not isinstance(source, str) or len(source.encode("utf-8")) > MAX_BYTES:
            raise CardinalProgramValidationError("source must be text of at most 16384 UTF-8 bytes")
        tree = ast.parse(source, mode="exec")
        if sum(1 for _ in ast.walk(tree)) > MAX_PYTHON_AST_NODES:
            raise CardinalProgramValidationError("Python AST exceeds 2048 nodes")
        frontier = [(tree, 0)]
        while frontier:
            node, depth = frontier.pop()
            if depth > MAX_PYTHON_DEPTH:
                raise CardinalProgramValidationError("Python AST exceeds nesting depth 24")
            frontier.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
        lowered_tree = ast.fix_missing_locations(_LiteralReturns().visit(tree))
        lowered = ast.unparse(lowered_tree) + "\n"
        legacy = compile_python_program(lowered)
        return CardinalProgram(source, _sha(source), lowered, _sha(lowered), legacy.compiled)
    except CardinalProgramValidationError:
        raise
    except PythonProgramValidationError as error:
        raise CardinalProgramValidationError(str(error)) from error
    except (SyntaxError, ValueError, TypeError, RecursionError, OverflowError, UnicodeError) as error:
        raise CardinalProgramValidationError("invalid cardinal Python-syntax controller: " + str(error)) from error


def _normalize_final(text):
    """Existing v2 single-fence packaging policy, embedded for archive isolation.

    Only one unambiguous Python fence may be removed; its enclosed text is kept
    exactly. No syntax or strategy is repaired. Other packaging is left intact
    for the compiler to reject. The full original response remains hash-bound.
    """
    if text.count("```") != 2 or "~~~" in text:
        return text, "none"
    matches = list(re.finditer(
        r"^[ \t]*```python[ \t]*\r?\n(?P<source>.*?)^[ \t]*```[ \t]*(?:\r?\n|\Z)",
        text, flags=re.M | re.S))
    if len(matches) != 1:
        return text, "none"
    match = matches[0]
    prose = bool(text[:match.start()].strip() or text[match.end():].strip())
    kind = "extracted_single_python_fence_with_prose" if prose else "stripped_single_wrapping_python_fence"
    return match["source"], kind


def classify_cardinal(text, finish):
    """Return (classification, inner CompiledProgram or None), without gameplay."""
    if not isinstance(text, str):
        return {"valid": False, "reason": "invalid_response_type"}, None
    try:
        provenance = {"original_response_sha256": _sha(text), "normalization_version": NORMALIZATION_VERSION}
    except UnicodeError:
        return {"valid": False, "reason": "invalid_response_encoding"}, None
    if finish != "eos":
        return {**provenance, "valid": False, "reason": "generation_" + str(finish)}, None
    if "<think>" in text or "</think>" in text:
        return {**provenance, "valid": False, "reason": "unexpected_thinking"}, None
    source, packaging = _normalize_final(text)
    try:
        program = compile_cardinal_program(source)
    except CardinalProgramValidationError as error:
        return {**provenance, "valid": False, "reason": "invalid_cardinal_python", "detail": str(error),
                "normalization": packaging}, None
    return {**provenance, "valid": True, "normalization": packaging,
            "program": program.to_dict()}, program.compiled


def cardinal_model_spec():
    return '''Write one complete controller: def act(): . Output Python source only, without explanation or markdown.
This restricted syntax is parsed into a bounded interpreter; Python itself is never executed.

GAME: Two robots compete for shared resources on a 7x7 grid for 48 rounds. Coordinates x and y
are 0..6. Your home is (0,3) or (6,3). Six initially mirror-symmetric sites contain finite units,
each worth its site's value. Harvest takes one unit at your current cell. Cargo holds three units.
Only depositing at your own home banks carried points. Cargo still carried at the end scores zero.
Maximize your banked points relative to the other robot. Both act on the same pre-round state.
Robots can share cells; seeded random priority resolves simultaneous harvests of a last unit.

ACTIONS: Each return must be one literal string: 'north', 'south', 'east', 'west', 'rest',
'harvest', or 'deposit'. North changes y by -1; south y by +1; east x by +1; west x by -1.
Move or harvest costs one energy. Energy starts at ten; rest restores three, capped at ten.
Deposit costs no energy and banks all cargo. Out-of-grid moves, moves/harvests without energy,
harvests with full cargo or no resource, and deposits away from home fail and consume the round.
No movement is automatic. The first return ends the turn. Use if/elif to choose between returns;
do not return a variable, expression, dictionary, movement vector, or helper call.

READ-ONLY INTEGER SENSORS (bare names): player, turn, horizon, grid_size, self_x, self_y,
home_x, home_y, energy, max_energy, carrying, capacity, score, opponent_x, opponent_y,
opponent_energy, opponent_carrying, opponent_score, site_count. carrying counts cargo units.
Read one of six sites with site(i, 'x'), site(i, 'y'), site(i, 'units'), site(i, 'value'), i=0..5.
An out-of-range site index reads zero. Signals reveal no future random priorities.

SYNTAX: scalar integer assignment; += -= *= //= %=; if/elif/else; pass;
for i in range(6): or another literal count 0..16; the seven literal string returns.
Integer expressions support + - * // %, < <= > >= == !=, and/or/not, abs(x), min(x,y),
max(x,y), chained comparisons, and numeric conditional expressions. Logic short-circuits.
Use 0/1 flags, not True/False. No imports, helpers, parameters, decorators, annotations, while,
break, continue, loop else, recursion, attributes, subscripts, lists, strings as expressions,
comprehensions, lambdas, /, exponentiation, keyword arguments, or other calls.
Strings are allowed only in literal returns, site field names, and an optional function docstring.
Local names start with an ASCII letter, use letters/digits/underscores, and contain no double
underscore. Sensors and function names cannot be assigned. Locals start at zero each turn.
m0..m15 are persistent integer memory, initially zero; read and assign them directly.

LIMITS: original and mechanically translated Python each <=16384 UTF-8 bytes, <=2048 AST nodes
and nesting <=24. Lowered IR <=512 nodes, nesting <=20, and <=32 locals including compiler
temporaries. Each turn has <=2048 interpreter operations; expressions/statements/loop iterations
cost operations. Integer literals and arithmetic are bounded at +/-1000000; arithmetic saturates.
Division/modulo by zero, exhausted operations, or no return produce a recorded runtime fault,
rest, and discard that turn's memory writes. Ordinary failed game actions are not runtime faults.
'''
