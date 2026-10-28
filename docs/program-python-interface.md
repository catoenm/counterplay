# Python-syntax coding-arena interface amendment

The first controller screen asked a model to emit a complete JSON AST. A separate
Python-syntax frontend now permits familiar code while retaining exactly the same
bounded interpreter and robot game. This is an interface change, not new game
mechanics or a training update. Preserve the original generation attempts and
identify Python-source experiments separately, for example `program-screen-v2`.

`selfplay/program_python.py` parses source with `ast.parse`, validates a whitelist,
and lowers the permitted nodes to the existing JSON program language. It never
executes the function, generates Python bytecode, imports model-specified modules,
or evaluates source text. All existing interpreter limits, fault rollback,
simulator accounting, and replay checks remain in effect. The frozen
`selfplay/program_arena/*.py` implementation is unchanged.

## API

```python
from selfplay.program_python import compile_python_program, python_model_spec
from selfplay.program_arena import play, verify_replay

instructions = python_model_spec()
program = compile_python_program(model_source)
provenance = program.to_dict()
record = play([program.compiled, opponent], seed=42000,
              origins=["model", "scripted"])
verify_replay(record)
```

The provenance dictionary preserves `python_source`, `python_sha256`,
`frontend_version`, `compiled_ir`, and `ir_sha256`. Store it alongside the unchanged
match record, and freeze the frontend file hash with the screen configuration.
`PythonProgramValidationError` subclasses the existing `ProgramValidationError`.
The complete model-facing rules are returned by `python_model_spec()`, which
contains only this syntax example:

```python
def act():
    return {'action': 'rest'}
```

## Supported syntax and intentional limits

The source contains exactly one parameterless `def act():`. Allowed statements
are scalar assignments, arithmetic augmented assignments, `if`/`elif`/optional
`else`, `for i in range(N)` with literal `N` from zero through sixteen, `pass`, and
returning a literal action dictionary. Conditions support normal comparisons,
short-circuit `and`/`or`, chained comparisons, `not`, and conditional expressions.
Allowed calls are `site(i, 'x'/'y'/'units'/'value')`, `abs(x)`, `min(x,y)`, and
`max(x,y)`. Arithmetic permits `+`, `-`, `*`, `//`, `%`, and unary signs.

Sensors use the same bare names as the interpreter. Ordinary local variables are
mapped to safe internal names and start at zero each turn. The special variables
`m0` through `m15` preserve integer values between turns. Their special persistence
means this is a Python-syntax language, not a promise to reproduce unrestricted
CPython variable-binding semantics. Arithmetic retains the interpreter's integer
saturation. Short-circuiting and selected-operand values of `and`/`or` are
preserved, including avoiding a guarded division by zero.

Unsupported constructs are rejected: imports, attributes, subscripts, containers
outside return dictionaries, helper functions, recursion, lambdas, comprehensions,
decorators, annotations, globals/nonlocals, while, break, continue, loop-else,
dynamic range bounds, arbitrary calls, keyword arguments, exponentiation, and true
division. Use integer flags instead of boolean literals. A function docstring is
permitted as inert text; other standalone expression statements are rejected.

Frontend version 2 limits are 16 KiB source, 2,048 Python AST nodes, and depth 24. Lowered code
must also fit the existing 512-node/20-depth/32-local/2,048-operation limits.
Compiler temporaries for conditionals and short-circuiting count toward these
limits. Empty branches lower to a bounded no-op assignment. Missing returns remain
visible runtime faults, rather than an invented successful action.

## Parser-cap amendment and retained failures

Frontend version 1 capped the Python parser AST at 512 nodes. The first frozen
Python screen rejected a 543-node generated source before lowering; the same
source produces 342 IR nodes within the existing interpreter budget. Version 2
raises only this parser-node cap to 2,048. Source size, nesting limits, IR limits,
runtime limits, instruction semantics, the robot game, and replay are unchanged.

Re-evaluation of existing responses under this amendment must be labeled
separately, for example `program-screen-v3`, and retain the exact source responses
and the original failed screen. This is post-hoc interface diagnosis using the
same generations, not a fresh independently generated success. Both versions and
their source hashes remain part of the experimental record.

## Validation and scientific interpretation

Six tests cover semantic equivalence with independently authored IR, short-circuit
division guards and operand values, source/IR preservation, rejected escapes and
unbounded constructs, runtime limit rollback, and exact simulator replay. They do
not execute generated Python as a test oracle.

Run: `.venv/bin/python -m unittest tests.test_program_python -v`.

A successful Python-syntax generation would establish controller-writing ability
under this interface. It would not retroactively validate failed JSON-AST attempts,
show that one language is generally easier, or demonstrate gradient learning.
Model identity, thinking mode, output budget, repair attempts, compiler validity,
runtime faults, deposit scores, and paired match outcomes should all remain
separately identifiable in the screen results.
