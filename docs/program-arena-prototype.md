# Executable policy-program arena prototype

This is a new branch alongside the frozen action-policy RL pilot. It does not
modify `selfplay/arena/` or its protocol. Its purpose is to let an LLM write a
complete algorithm which then controls a robot throughout a competitive game.

The implementation and initial benchmark controllers are hand authored. The
scripted demo demonstrates executable programs, competition, and replay. It does
**not** demonstrate model-generated program quality, algorithm discovery,
distillation, gradient training, or continual improvement. The runner accepts
explicit per-program provenance so a later model screen can be labeled honestly.

**October 22 follow-up:** the initial capability screens and feedback-authoring
experiments are preserved separately. A local Qwen3.8-27B experiment subsequently
completed one genuine RLOO update on 3,191 newly generated completion tokens in
`runs/program-local-27b-compact-v1`; its source, reward, likelihood and checkpoint
audit passed. Before/after held-out controllers were token-identical and banked
zero points in six games each. The plain-prompt condition stopped at its
capability gate with zero updates. Thus later work establishes executable
model-generated controllers and a working gradient-training mechanism, but no
measured improvement from that update. Current state is recorded in
`runs/overnight-progress.json`; the original prototype description below remains
historical.

## Resource competition

Two robots start at opposite homes on a 7 by 7 grid. Six resource sites occur in
three mirror-symmetric pairs, with finite supplies and integer values. Either
robot can harvest any site, so both algorithms affect the resource opportunities
available to their opponent. A round asks both programs for one action from the
same pre-round state. Actions are move one cardinal cell, harvest, deposit, or
rest. Robots may share cells; a pre-sampled random priority resolves simultaneous
harvests of limited supply. This priority is independent of program behavior and
is hidden until its round resolves.

Moving and harvesting cost one energy. Robots have ten maximum energy, regain
three by resting, and can carry three resource units. Depositing at their own
home converts carried resource values to points. Only deposited points count at
the fixed 48-round limit. Units left in transit do not score. Initial resource
units and value are exactly conserved between sites, cargo, and banked points.

The relative terminal reward is `(points_0 - points_1) / initial_total_value`
and its negative. Raw points, deposited units, failed actions, interpreter faults,
and leftover cargo remain available separately. A positive return need not mean
efficient collection. Evaluation should pair seeds and swap program seats.

## Programs rather than action menus

A controller is a JSON abstract syntax tree. It can compute integer expressions,
scan resource sites using bounded loops, branch on observations, and preserve
sixteen integer memory slots between turns. Every controller uses the same
interpreter; it is not restricted to selecting a supplied strategy. New programs
can implement different route choice, collection thresholds, energy management,
opponent responses, and memory-dependent behavior.

The exact language, sensor names, action rules, and limits are returned by
`selfplay.program_arena.model_spec()`. That model-facing text includes only an
idle syntax example, not a working solution. A standalone minimal program is
`selfplay/program_arena/examples/idle.json`. The hand-authored scanning algorithms
in `examples.py` are benchmark opponents, not prompts for unaided generation.

```python
from selfplay.program_arena import compile_program, play, verify_replay, model_spec
from selfplay.program_arena.examples import reference_programs

instructions = model_spec()
candidate = compile_program(model_response_text)
opponent = compile_program(reference_programs()["nearest"])
record = play([candidate, opponent], seed=41000, origins=["model", "scripted"])
verify_replay(record)
```

The model response should be plain JSON. Invalid generation must be counted and
stored before any repair. If a later screen allows feedback or repair, its
attempt limit and separate success rates need to be recorded. Seeds used during
program revision should be separated from held-out evaluation seeds.

## Interpreter boundaries

Generated text is parsed as JSON and validated into immutable instruction tuples.
The implementation never calls `eval`, `exec`, a compiler, a subprocess, an import
from model output, or generated Python. Programs have no attribute access, host
object access, filenames, network operations, external functions, or recursion.
This is a restricted programming language, not a claim to sandbox arbitrary code.

Limits are 16 KiB UTF-8 source, 512 AST nodes, nesting depth 20, 32 local names,
16 persistent integer slots, and 2,048 interpreter operations per turn. A repeat
count must be a literal from zero through sixteen. Every expression and statement
costs an operation; iterations cost another. Nested loops cannot exceed the
runtime operation limit. Arithmetic saturates at plus or minus one million, so
intermediate multiplication is also bounded. Unknown syntax and duplicate JSON
keys are rejected. Read-only sensors cannot be overwritten.

A runtime fault, invalid movement vector, or missing emitted action produces a
rest action and rolls back that turn's memory writes. The fault is recorded.
Movement outside the grid, harvesting unavailable resources, and depositing away
from home are valid intents that fail in the simulator and consume a round.
Rest on a fault means faults are not automatically losses; comparisons must
report both return and fault rate. Programs share no memory or locals.

## Records and tests

Matches save exact source/digests, provenance, the initial simulator state,
per-turn observations, actions, persistent memory, operation counts, faults,
complete replay states, and terminal metrics. `verify_replay` recompiles the saved
programs, reruns them, and requires an exact JSON-equivalent match. Simulator
snapshots include hidden future event priorities for replay and must never be
passed to a policy. Policy observations contain current public coordinates,
resources, inventory counts, energy, and scores; they contain no route oracle.

Tests cover rejected host-escape syntax, duplicate keys, type violations,
oversized/deep programs, resource-exhaustion loops, arithmetic limits, fault
rollback, resource/value conservation, simultaneous contest priority, symmetric
maps, atomic action validation, effective algorithms, and tamper-detecting replay.

Run the scripted prototype:

```sh
.venv/bin/python -m unittest tests.test_program_arena -v
.venv/bin/python -m selfplay.program_arena.demo --out runs/program-arena-prototype
```

The first intended model experiment is a capability screen: independently ask
models to generate controllers from the rules, compile them, and evaluate against
the frozen bots and each other with saved attempts and paired held-out seeds.
Winning generated programs are not themselves RL updates. Training the language
model on program-generation rollouts would need generation token log-probabilities,
outcome-to-program credit assignment, and a separately frozen training protocol.

This small finite game can saturate. Additional resource geometries, obstacles,
multiple robots, hidden information, or newly generated rules would each require
separate tested mechanics. The existing prototype does not promise an unlimited
curriculum or general programming improvement.
