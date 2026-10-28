"""Deterministic program-versus-program matches and executable replay checks."""
from __future__ import annotations

import json

from .dsl import CompiledProgram, compile_program
from .game import RobotArena


def play(programs, seed: int, max_rounds: int = 48, origins=None):
    if len(programs) != 2:
        raise ValueError("exactly two programs required")
    programs = [p if isinstance(p, CompiledProgram) else compile_program(p) for p in programs]
    origins = ["unspecified", "unspecified"] if origins is None else list(origins)
    if len(origins) != 2 or any(o not in {"scripted", "model", "unspecified"} for o in origins):
        raise ValueError("each program origin must be scripted, model, or unspecified")
    env = RobotArena(seed, max_rounds)
    memory = [None, None]
    events = []
    faults = [0, 0]
    initial = env.snapshot()
    while not env.done:
        observations = [env.observation(p) for p in (0, 1)]
        decisions = [programs[p].run(observations[p], memory[p]) for p in (0, 1)]
        for p in (0, 1):
            memory[p] = decisions[p].memory
            faults[p] += decisions[p].fault is not None
        env.step({p: decisions[p].action for p in (0, 1)})
        events.append({"turn": env.turn - 1, "observations": observations,
                       "decisions": [d.to_dict() for d in decisions], "state": env.snapshot()})
    # Normalize integer dictionary keys now so round-tripping through JSON is exact.
    record = {"format_version": 1, "kind": "program_match", "seed": seed,
              "max_rounds": max_rounds, "programs": [{"digest": p.digest, "source": json.loads(p.source)} for p in programs],
              "initial_state": initial, "events": events, "returns": env.returns(),
              "metrics": {**env.metrics(), "program_faults": faults},
              "origins": origins, "model_generated": "model" in origins, "gradient_training": False}
    return json.loads(json.dumps(record))


def verify_replay(record):
    if record.get("format_version") != 1:
        raise ValueError("unsupported replay format")
    programs = [compile_program(p["source"]) for p in record["programs"]]
    if [p.digest for p in programs] != [p["digest"] for p in record["programs"]]:
        raise ValueError("program digest mismatch")
    replayed = play(programs, record["seed"], record["max_rounds"], origins=record["origins"])
    if replayed != record:
        raise ValueError("replayed program execution or game state differs from record")
    return True
