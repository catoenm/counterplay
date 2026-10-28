"""A deliberately small interpreted language; generated code is never executed.

JSON objects compile to immutable instruction tuples. There are no Python ASTs,
attribute access, imports, strings-as-code, host callbacks, filesystem or network
operations. A runtime operation counter also bounds nested loops and expressions.
This is an interpreter, not a sandbox for arbitrary Python.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re


MAX_BYTES = 16_384
MAX_NODES = 512
MAX_DEPTH = 20
MAX_STEPS = 2_048
MAX_LOOP = 16
MAX_LOCALS = 32
MEMORY_SLOTS = 16
INTEGER_BOUND = 1_000_000
SENSORS = frozenset({"player", "turn", "horizon", "grid_size", "self_x", "self_y", "home_x", "home_y",
                     "energy", "max_energy", "carrying", "capacity", "score", "opponent_x", "opponent_y",
                     "opponent_energy", "opponent_carrying", "opponent_score", "site_count"})
SITE_FIELDS = frozenset({"x", "y", "units", "value"})
OPERATORS = {"+": 2, "-": 2, "*": 2, "//": 2, "%": 2, "min": 2, "max": 2,
             "<": 2, "<=": 2, ">": 2, ">=": 2, "==": 2, "!=": 2,
             "and": 2, "or": 2, "not": 1, "abs": 1, "neg": 1}


class ProgramValidationError(ValueError):
    pass


class _RuntimeFault(Exception):
    pass


def _integer(value):
    return type(value) is int and -INTEGER_BOUND <= value <= INTEGER_BOUND


def _clamp(value):
    return max(-INTEGER_BOUND, min(INTEGER_BOUND, int(value)))


def _memory_name(name):
    return isinstance(name, str) and name in {f"m{i}" for i in range(MEMORY_SLOTS)}


def _local_name(name):
    return isinstance(name, str) and re.fullmatch(r"v_[a-z][a-z0-9_]{0,19}", name, flags=re.ASCII) is not None


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProgramValidationError("duplicate JSON object key")
        result[key] = value
    return result


class _Compiler:
    def __init__(self):
        self.nodes = 0
        self.locals = set()

    def visit(self, depth):
        self.nodes += 1
        if self.nodes > MAX_NODES or depth > MAX_DEPTH:
            raise ProgramValidationError("program exceeds node or nesting limit")

    def name(self, name, writable=False, loop=False):
        if not isinstance(name, str):
            raise ProgramValidationError("variable names must be strings")
        if _local_name(name):
            self.locals.add(name)
            if len(self.locals) > MAX_LOCALS:
                raise ProgramValidationError("too many local variables")
        elif not loop and _memory_name(name):
            pass
        elif not writable and not loop and name in SENSORS:
            pass
        else:
            raise ProgramValidationError("unknown or unwritable variable")
        return name

    def expr(self, expression, depth):
        self.visit(depth)
        if _integer(expression):
            return ("literal", expression)
        if not isinstance(expression, dict):
            raise ProgramValidationError("expressions must be bounded integers or expression objects")
        keys = set(expression)
        if keys == {"var"}:
            return ("var", self.name(expression["var"]))
        if keys == {"site"}:
            field = expression["site"]
            if not isinstance(field, dict) or set(field) != {"index", "field"}:
                raise ProgramValidationError("site expression requires index and field")
            if field["field"] not in SITE_FIELDS:
                raise ProgramValidationError("unknown site field")
            return ("site", self.expr(field["index"], depth + 1), field["field"])
        if keys == {"op", "args"}:
            op, args = expression["op"], expression["args"]
            if (not isinstance(op, str) or op not in OPERATORS or not isinstance(args, list)
                    or len(args) != OPERATORS[op]):
                raise ProgramValidationError("unknown operator or incorrect arity")
            return ("op", op, tuple(self.expr(arg, depth + 1) for arg in args))
        raise ProgramValidationError("unknown expression syntax")

    def block(self, statements, depth):
        if not isinstance(statements, list) or not statements:
            raise ProgramValidationError("a statement block must be a nonempty list")
        result = []
        for statement in statements:
            self.visit(depth)
            if not isinstance(statement, dict):
                raise ProgramValidationError("statements must be objects")
            keys = set(statement)
            if keys == {"set", "value"}:
                result.append(("set", self.name(statement["set"], writable=True),
                               self.expr(statement["value"], depth + 1)))
            elif keys == {"if", "then", "else"}:
                result.append(("if", self.expr(statement["if"], depth + 1),
                               self.block(statement["then"], depth + 1),
                               self.block(statement["else"], depth + 1)))
            elif keys == {"repeat", "index", "body"}:
                count = statement["repeat"]
                if type(count) is not int or not 0 <= count <= MAX_LOOP:
                    raise ProgramValidationError("repeat count must be a literal integer from 0 through 16")
                result.append(("repeat", count, self.name(statement["index"], writable=True, loop=True),
                               self.block(statement["body"], depth + 1)))
            elif keys == {"emit"}:
                action = statement["emit"]
                if not isinstance(action, dict) or not isinstance(action.get("action"), str):
                    raise ProgramValidationError("emit requires an action object")
                if action["action"] == "move" and set(action) == {"action", "dx", "dy"}:
                    result.append(("emit_move", self.expr(action["dx"], depth + 1), self.expr(action["dy"], depth + 1)))
                elif action["action"] in {"harvest", "deposit", "rest"} and set(action) == {"action"}:
                    result.append(("emit", action["action"]))
                else:
                    raise ProgramValidationError("unknown action or extra action fields")
            else:
                raise ProgramValidationError("unknown statement syntax")
        return tuple(result)


@dataclass(frozen=True)
class Decision:
    action: dict
    memory: tuple
    steps: int
    fault: str | None

    def to_dict(self):
        return {"action": dict(self.action), "memory": list(self.memory), "steps": self.steps, "fault": self.fault}


@dataclass(frozen=True)
class CompiledProgram:
    source: str
    digest: str
    instructions: tuple
    node_count: int

    def run(self, observation, memory=None):
        memory = tuple([0] * MEMORY_SLOTS if memory is None else memory)
        if len(memory) != MEMORY_SLOTS or not all(_integer(v) for v in memory):
            raise ValueError("program memory must contain 16 bounded integers")
        machine = _Machine(observation, memory)
        try:
            action = machine.block(self.instructions)
            if action is None:
                raise _RuntimeFault("no_action")
            return Decision(action, tuple(machine.memory), machine.steps, None)
        except _RuntimeFault as error:
            # Faults cost a turn but cannot partly update persistent memory.
            return Decision({"action": "rest"}, memory, machine.steps, str(error))


def compile_program(source: str | dict) -> CompiledProgram:
    """Validate text or a JSON-compatible program and freeze its instruction tree."""
    try:
        text = json.dumps(source, separators=(",", ":")) if isinstance(source, dict) else source
        if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_BYTES:
            raise ProgramValidationError("program text exceeds byte limit or is not text")
        parsed = json.loads(text, object_pairs_hook=_no_duplicate_keys)
        if not isinstance(parsed, dict) or set(parsed) != {"version", "body"} or type(parsed["version"]) is not int or parsed["version"] != 1:
            raise ProgramValidationError("program requires version 1 and body")
        compiler = _Compiler()
        instructions = compiler.block(parsed["body"], 0)
        canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
        return CompiledProgram(canonical, hashlib.sha256(canonical.encode()).hexdigest(), instructions, compiler.nodes)
    except ProgramValidationError:
        raise
    except (ValueError, TypeError, KeyError, RecursionError, OverflowError) as error:
        raise ProgramValidationError("invalid bounded program") from error


class _Machine:
    def __init__(self, observation, memory):
        if (not isinstance(observation, dict) or any(not _integer(observation.get(k)) for k in SENSORS)
                or not isinstance(observation.get("sites"), list) or len(observation["sites"]) > MAX_LOOP
                or observation["site_count"] != len(observation["sites"])
                or any(not isinstance(site, dict) or any(not _integer(site.get(k)) for k in SITE_FIELDS)
                       for site in observation["sites"])):
            raise ValueError("invalid trusted observation")
        self.observation = observation
        self.memory = list(memory)
        self.locals = {}
        self.steps = 0

    def tick(self):
        if self.steps >= MAX_STEPS:
            raise _RuntimeFault("step_limit")
        self.steps += 1

    def expr(self, expression):
        self.tick()
        kind = expression[0]
        if kind == "literal":
            return expression[1]
        if kind == "var":
            name = expression[1]
            if _memory_name(name):
                return self.memory[int(name[1:])]
            return self.observation[name] if name in SENSORS else self.locals.get(name, 0)
        if kind == "site":
            index = self.expr(expression[1])
            return self.observation["sites"][index][expression[2]] if 0 <= index < len(self.observation["sites"]) else 0
        op = expression[1]
        args = [self.expr(arg) for arg in expression[2]]
        a, b = args[0], args[-1]
        if op in {"//", "%"} and b == 0:
            raise _RuntimeFault("division_by_zero")
        if op == "+": result = a + b
        elif op == "-": result = a - b
        elif op == "*": result = a * b
        elif op == "//": result = a // b
        elif op == "%": result = a % b
        elif op == "min": result = min(a, b)
        elif op == "max": result = max(a, b)
        elif op == "<": result = a < b
        elif op == "<=": result = a <= b
        elif op == ">": result = a > b
        elif op == ">=": result = a >= b
        elif op == "==": result = a == b
        elif op == "!=": result = a != b
        elif op == "and": result = bool(a) and bool(b)
        elif op == "or": result = bool(a) or bool(b)
        elif op == "not": result = not a
        elif op == "abs": result = abs(a)
        elif op == "neg": result = -a
        else: raise _RuntimeFault("invalid_compiled_operator")
        return _clamp(result)

    def assign(self, name, value):
        if _memory_name(name):
            self.memory[int(name[1:])] = value
        else:
            self.locals[name] = value

    def block(self, block):
        for instruction in block:
            self.tick()
            kind = instruction[0]
            if kind == "set":
                self.assign(instruction[1], self.expr(instruction[2]))
            elif kind == "if":
                result = self.block(instruction[2] if self.expr(instruction[1]) else instruction[3])
                if result is not None: return result
            elif kind == "repeat":
                for index in range(instruction[1]):
                    self.tick()
                    self.assign(instruction[2], index)
                    result = self.block(instruction[3])
                    if result is not None: return result
            elif kind == "emit":
                return {"action": instruction[1]}
            elif kind == "emit_move":
                dx, dy = self.expr(instruction[1]), self.expr(instruction[2])
                if abs(dx) + abs(dy) != 1:
                    raise _RuntimeFault("invalid_move")
                return {"action": "move", "dx": dx, "dy": dy}
        return None
