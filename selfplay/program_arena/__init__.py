"""Executable bounded policy programs competing in a robot resource arena."""

from .dsl import CompiledProgram, ProgramValidationError, compile_program
from .game import RobotArena
from .runner import play, verify_replay
from .spec import model_spec

__all__ = ["CompiledProgram", "ProgramValidationError", "compile_program", "RobotArena", "play", "verify_replay", "model_spec"]
