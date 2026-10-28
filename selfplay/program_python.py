"""Python-syntax frontend for the bounded policy-program interpreter.

This module uses ast.parse solely as a parser. It never executes Python source,
generates Python bytecode, resolves attributes, or calls a model-provided function.
Validated syntax lowers into the unchanged selfplay.program_arena JSON language.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import re

from .program_arena.dsl import CompiledProgram, ProgramValidationError, compile_program, SENSORS, SITE_FIELDS, MAX_BYTES


FRONTEND_VERSION = 2
MAX_PYTHON_AST_NODES = 2048
MAX_PYTHON_DEPTH = 24
BINOPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.FloorDiv: "//", ast.Mod: "%"}
COMPARISONS = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}
RESERVED = frozenset({"act", "site", "abs", "min", "max", "range"})


class PythonProgramValidationError(ProgramValidationError):
    pass


def _v(name):
    return {"var": name}


def _op(name, *args):
    return {"op": name, "args": list(args)}


def _set(name, value):
    return {"set": name, "value": value}


def _noop():
    return [_set("v_noop", 0)]


def _branch(condition, yes, no=None):
    return {"if": condition, "then": yes or _noop(), "else": no or _noop()}


@dataclass(frozen=True)
class PythonProgram:
    source: str
    source_sha256: str
    compiled: CompiledProgram

    def run(self, observation, memory=None):
        return self.compiled.run(observation, memory)

    def to_dict(self):
        return {"source_language": "bounded_python_syntax", "frontend_version": FRONTEND_VERSION,
                "python_source": self.source, "python_sha256": self.source_sha256,
                "compiled_ir": json.loads(self.compiled.source), "ir_sha256": self.compiled.digest}


class _Lowerer:
    def __init__(self):
        self.locals = {}
        self.temporaries = 0

    def error(self, node, message):
        raise PythonProgramValidationError(f"line {getattr(node, 'lineno', '?')}: {message}")

    def temp(self):
        name = f"v_t{self.temporaries}"
        self.temporaries += 1
        return name

    def name(self, node, writable=False, loop=False):
        if not isinstance(node, ast.Name):
            self.error(node, "only bare variables are supported")
        name = node.id
        if (not name.isascii() or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,39}", name) is None
                or "__" in name or name in RESERVED):
            self.error(node, "invalid or reserved variable name")
        if name in SENSORS:
            if writable:
                self.error(node, "sensors are read-only")
            return name
        if re.fullmatch(r"m[0-9]+", name):
            if name not in {f"m{i}" for i in range(16)} or loop:
                self.error(node, "persistent memory is m0..m15 and cannot be a loop index")
            return name
        if name not in self.locals:
            self.locals[name] = f"v_l{len(self.locals)}"
        return self.locals[name]

    def expr(self, node):
        """Return statements to evaluate first, then a side-effect-free IR value."""
        if isinstance(node, ast.Constant):
            if type(node.value) is not int or not -1_000_000 <= node.value <= 1_000_000:
                self.error(node, "only bounded integer expression literals are supported")
            return [], node.value
        if isinstance(node, ast.Name):
            return [], _v(self.name(node))
        if isinstance(node, ast.UnaryOp):
            before, value = self.expr(node.operand)
            if isinstance(node.op, ast.UAdd):
                return before, value
            op = {ast.USub: "neg", ast.Not: "not"}.get(type(node.op))
            if op is None:
                self.error(node, "unsupported unary operator")
            return before, _op(op, value)
        if isinstance(node, ast.BinOp):
            op = BINOPS.get(type(node.op))
            if op is None:
                self.error(node, "allowed arithmetic is + - * // %")
            first, left = self.expr(node.left)
            second, right = self.expr(node.right)
            return first + second, _op(op, left, right)
        if isinstance(node, ast.BoolOp):
            if not isinstance(node.op, (ast.And, ast.Or)):
                self.error(node, "unsupported boolean operator")
            before, first = self.expr(node.values[0])
            temporary = self.temp()
            before.append(_set(temporary, first))
            for operand in node.values[1:]:
                following, value = self.expr(operand)
                condition = _v(temporary) if isinstance(node.op, ast.And) else _op("not", _v(temporary))
                before.append(_branch(condition, following + [_set(temporary, value)]))
            # Python's and/or return a selected operand, not necessarily 0 or 1.
            return before, _v(temporary)
        if isinstance(node, ast.Compare):
            if any(type(op) not in COMPARISONS for op in node.ops):
                self.error(node, "only numeric comparisons are supported")
            before, left = self.expr(node.left)
            next_before, right = self.expr(node.comparators[0])
            before += next_before
            first = _op(COMPARISONS[type(node.ops[0])], left, right)
            if len(node.ops) == 1:
                return before, first
            temporary = self.temp()
            before.append(_set(temporary, first))
            previous = right
            for operator, operand in zip(node.ops[1:], node.comparators[1:], strict=True):
                following, value = self.expr(operand)
                comparison = _op(COMPARISONS[type(operator)], previous, value)
                before.append(_branch(_v(temporary), following + [_set(temporary, comparison)]))
                previous = value
            return before, _v(temporary)
        if isinstance(node, ast.IfExp):
            before, condition = self.expr(node.test)
            yes_before, yes = self.expr(node.body)
            no_before, no = self.expr(node.orelse)
            temporary = self.temp()
            before.append(_branch(condition, yes_before + [_set(temporary, yes)], no_before + [_set(temporary, no)]))
            return before, _v(temporary)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords:
                self.error(node, "calls must be whitelisted bare functions with positional arguments")
            name = node.func.id
            if name == "site":
                if (len(node.args) != 2 or not isinstance(node.args[1], ast.Constant)
                        or type(node.args[1].value) is not str or node.args[1].value not in SITE_FIELDS):
                    self.error(node, "site requires index and literal field x/y/units/value")
                before, index = self.expr(node.args[0])
                return before, {"site": {"index": index, "field": node.args[1].value}}
            arity = {"abs": 1, "min": 2, "max": 2}.get(name)
            if arity is None or len(node.args) != arity:
                self.error(node, "allowed expression calls: site(i,field), abs(x), min(x,y), max(x,y)")
            before, arguments = [], []
            for argument in node.args:
                preceding, value = self.expr(argument)
                before.extend(preceding)
                arguments.append(value)
            return before, _op(name, *arguments)
        self.error(node, "unsupported expression (no attributes, subscripts, containers or arbitrary calls)")

    def block(self, statements):
        lowered = []
        for node in statements:
            if isinstance(node, ast.Pass):
                lowered.extend(_noop())
            elif isinstance(node, ast.Assign):
                if len(node.targets) != 1:
                    self.error(node, "assign exactly one scalar variable at a time")
                target = self.name(node.targets[0], writable=True)
                before, value = self.expr(node.value)
                lowered.extend(before + [_set(target, value)])
            elif isinstance(node, ast.AugAssign):
                target = self.name(node.target, writable=True)
                operator = BINOPS.get(type(node.op))
                if operator is None:
                    self.error(node, "unsupported augmented assignment")
                before, value = self.expr(node.value)
                lowered.extend(before + [_set(target, _op(operator, _v(target), value))])
            elif isinstance(node, ast.If):
                before, condition = self.expr(node.test)
                lowered.extend(before + [_branch(condition, self.block(node.body), self.block(node.orelse))])
            elif isinstance(node, ast.For):
                iterator = node.iter
                if (not isinstance(iterator, ast.Call) or not isinstance(iterator.func, ast.Name)
                        or iterator.func.id != "range" or iterator.keywords or len(iterator.args) != 1
                        or not isinstance(iterator.args[0], ast.Constant) or type(iterator.args[0].value) is not int
                        or not 0 <= iterator.args[0].value <= 16 or node.orelse or node.type_comment):
                    self.error(node, "for requires range(literal integer 0..16), without else")
                target = self.name(node.target, writable=True, loop=True)
                lowered.append({"repeat": iterator.args[0].value, "index": target,
                                "body": self.block(node.body) or _noop()})
            elif isinstance(node, ast.Return):
                lowered.extend(self.action(node))
            else:
                self.error(node, "unsupported statement (no imports, helpers, while, break, continue or side effects)")
        return lowered

    def action(self, node):
        value = node.value
        if not isinstance(value, ast.Dict) or any(not isinstance(k, ast.Constant) or type(k.value) is not str for k in value.keys):
            self.error(node, "return a literal action dictionary with string keys")
        keys = [k.value for k in value.keys]
        if len(set(keys)) != len(keys):
            self.error(node, "duplicate return dictionary key")
        fields = dict(zip(keys, value.values, strict=True))
        action = fields.get("action")
        if not isinstance(action, ast.Constant) or type(action.value) is not str:
            self.error(node, "action must be a literal action name")
        if action.value in {"rest", "harvest", "deposit"} and set(fields) == {"action"}:
            return [{"emit": {"action": action.value}}]
        if action.value == "move" and set(fields) == {"action", "dx", "dy"}:
            before_x, dx = self.expr(fields["dx"])
            before_y, dy = self.expr(fields["dy"])
            return before_x + before_y + [{"emit": {"action": "move", "dx": dx, "dy": dy}}]
        self.error(node, "return only rest/harvest/deposit, or move with dx and dy")


def compile_python_program(source: str) -> PythonProgram:
    try:
        if not isinstance(source, str) or len(source.encode("utf-8")) > MAX_BYTES:
            raise PythonProgramValidationError("Python source must be text of at most 16384 UTF-8 bytes")
        tree = ast.parse(source, mode="exec")
        nodes = list(ast.walk(tree))
        if len(nodes) > MAX_PYTHON_AST_NODES:
            raise PythonProgramValidationError("Python AST exceeds 2048 nodes")
        frontier = [(tree, 0)]
        while frontier:
            node, depth = frontier.pop()
            if depth > MAX_PYTHON_DEPTH:
                raise PythonProgramValidationError("Python AST exceeds nesting depth 24")
            frontier.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise PythonProgramValidationError("provide exactly one function: def act():")
        function = tree.body[0]
        arguments = function.args
        if (function.name != "act" or function.decorator_list or function.returns or function.type_comment
                or getattr(function, "type_params", ()) or arguments.posonlyargs or arguments.args
                or arguments.kwonlyargs or arguments.vararg or arguments.kwarg or arguments.defaults or arguments.kw_defaults):
            raise PythonProgramValidationError("act must have no parameters, decorators, type parameters or annotations")
        body = list(function.body)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and type(body[0].value.value) is str:
            body = body[1:]  # A function docstring is inert text, not a statement.
        ir = {"version": 1, "body": _Lowerer().block(body) or _noop()}
        compiled = compile_program(ir)
        return PythonProgram(source, hashlib.sha256(source.encode()).hexdigest(), compiled)
    except PythonProgramValidationError:
        raise
    except ProgramValidationError as error:
        raise PythonProgramValidationError("lowered program exceeds bounded interpreter contract: " + str(error)) from error
    except (SyntaxError, ValueError, TypeError, RecursionError, OverflowError) as error:
        raise PythonProgramValidationError("invalid Python-syntax controller: " + str(error)) from error


def python_model_spec():
    return '''Write a complete controller using the restricted Python syntax below. Output only Python source, no markdown.
Exactly one function: def act(): . This function is parsed, checked and interpreted; Python itself is never executed.

GAME: Two robots compete on a 7x7 grid, for 48 rounds. Homes are (0,3) and (6,3). Six initially
mirror-symmetric resource sites are shared: either robot can harvest any site's finite units.
Each unit has a point value. Move one cardinal cell or harvest one unit at your current cell, each
costing one energy. Rest restores three energy up to ten. Cargo holds three units. Deposit at your
own home banks all carried points. ONLY BANKED POINTS count when time expires. Maximize your points
relative to the rival. Robots can share cells; random priority settles contested last units.
Each round both controllers see the same pre-round state. Failed intents consume the round.

READ-ONLY INTEGER SENSORS, used directly as bare variable names:
player, turn, horizon, grid_size, self_x, self_y, home_x, home_y, energy, max_energy,
carrying, capacity, score, opponent_x, opponent_y, opponent_energy, opponent_carrying,
opponent_score, site_count. carrying is a unit count. site_count is always six here.
Read a site using site(i, 'x'), site(i, 'y'), site(i, 'units'), site(i, 'value'). Indices are 0..5.

LANGUAGE: scalar integer assignments; += -= *= //= %=; if/elif/else (else optional);
for i in range(6): or another literal count 0..16; pass; return action dictionaries.
Arithmetic + - * // %; comparisons < <= > >= == !=; and/or/not; abs(x), min(x,y), max(x,y).
Conditional expressions (a if condition else b) and chained comparisons are supported.
and/or and chained comparisons short-circuit. Loops cannot use break, continue or else.
No while, imports, helpers, recursion, attributes, subscripts, lists, comprehensions, lambdas, strings
as code, exponentiation, true division (/), keyword arguments, or other function calls.
Strings are permitted only for action dictionary keys/names, site field names, and a function docstring.
Use integer flags 0/1 instead of True/False. Local names start with a letter, contain only ASCII
letters/digits/underscores, and contain no double underscore. Sensors and function names are reserved.
Locals start at zero every turn. The special variables m0..m15 are persistent integer memory, initially
zero once per game. Read and assign them directly without global/nonlocal declarations.

Return exactly one of:
{'action':'move', 'dx':DX, 'dy':DY} with abs(DX)+abs(DY)==1;
{'action':'harvest'}; {'action':'deposit'}; {'action':'rest'}.
The first return ends the turn. Out-of-grid moves, empty harvests, and away-from-home deposits fail.

BOUNDS: <=16384 UTF-8 source bytes, <=2048 Python AST nodes and <=512 lowered IR nodes,
<=32 lowered local names including compiler temporaries, nesting <=20 in lowered IR,
<=2048 interpreter operations per turn, and sixteen memory slots. Arithmetic saturates at +/-1000000.
Division by zero, operation exhaustion, invalid movement vectors or no return produce rest and discard
that turn's memory writes. Faults are recorded. Keep code compact and avoid deeply nested conditions.
Each variable, expression, statement and loop iteration costs interpreter operations.

Minimal syntax-only example (not a useful strategy):
def act():
    return {'action': 'rest'}
'''
