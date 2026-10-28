import copy
import json
import random
import unittest
from collections import deque

import chess

from selfplay.arena.chess_game import ChessEnv
from selfplay.arena.maze import MazeEnv


def reachable(start, goal, walls, size):
    """Independent flood fill, deliberately not the environment's BFS."""
    seen = {tuple(start)}
    frontier = deque(seen)
    while frontier:
        r, c = frontier.popleft()
        for cell in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if (0 <= cell[0] < size and 0 <= cell[1] < size
                    and cell not in walls and cell not in seen):
                seen.add(cell)
                frontier.append(cell)
    return tuple(goal) in seen


class MazeTests(unittest.TestCase):
    def test_deterministic_reset_public_observation_and_regimes(self):
        for regime in ("train", "eval", "ood"):
            env = MazeEnv(317, regime=regime)
            self.assertEqual(env.snapshot(), MazeEnv(317, regime=regime).snapshot())
            self.assertEqual(env.positions[0], env.positions[1])
            self.assertEqual(env.walls[0], env.walls[1])
            self.assertLessEqual(len(env.legal_actions(0)), 256)
            other = env.observation(1)
            self.assertEqual(other["player"], 1)
            self.assertEqual(other["positions"], env.observation(0)["positions"])
            self.assertNotIn("shortest_distances", other)
            self.assertIn("shortest_distances", env.snapshot()["metrics"])
            json.dumps(env.snapshot())
        self.assertEqual(MazeEnv(317, regime="ood").size, 7)
        self.assertGreater(len({json.dumps(MazeEnv(i).snapshot()["terrain"]) for i in range(20)}), 10)

    def test_illegal_batch_is_atomic(self):
        env = MazeEnv(4)
        before = copy.deepcopy(env.snapshot())
        with self.assertRaises(ValueError):
            env.step({0: env.heuristic_action(0), 1: {"type": "move", "direction": "invalid"}})
        self.assertEqual(before, env.snapshot())
        with self.assertRaises(ValueError):
            env.step({0: env.heuristic_action(0)})
        with self.assertRaises(ValueError):
            env.step({False: env.heuristic_action(0), True: env.heuristic_action(1)})
        with self.assertRaises(RuntimeError):
            env.returns()

    def test_last_path_and_special_tiles_cannot_be_blocked(self):
        env = MazeEnv(10)
        # One horizontal corridor gives an adversary an obvious disconnection exploit.
        corridor = {(2, c) for c in range(5)}
        env.walls = [set(), {(r, c) for r in range(5) for c in range(5)} - corridor]
        blocks = [a for a in env.legal_actions(0) if a["type"] == "block"]
        self.assertEqual(blocks, [])
        with self.assertRaises(ValueError):
            env.step({0: {"type": "block", "row": 2, "col": 2},
                      1: {"type": "move", "direction": "E"}})
        self.assertTrue(reachable(env.positions[1], env.goal, env.walls[1], env.size))

    def test_simultaneous_movement_has_no_submission_order_bias(self):
        env = MazeEnv(1)
        env.walls = [set(), set()]
        other = copy.deepcopy(env)
        actions = {0: {"type": "move", "direction": "E"},
                   1: {"type": "block", "row": 2, "col": 1}}
        env.step(actions)
        other.step(dict(reversed(list(actions.items()))))
        self.assertEqual(env.snapshot(), other.snapshot())
        self.assertEqual(env.positions[0], (2, 1))
        self.assertNotIn((2, 1), env.walls[0])
        self.assertEqual(env.blocks_remaining[1], 2)
        self.assertEqual(env.blocks_cancelled[1], 1)

    def test_block_budget_and_reachability_across_random_rollouts(self):
        for regime in ("train", "ood"):
            for seed in range(12):
                env, rng = MazeEnv(seed, regime=regime), random.Random(seed + 611)
                while not env.done:
                    actions = {}
                    for player in (0, 1):
                        legal = env.legal_actions(player)
                        if env.blocks_remaining[player] == 0:
                            self.assertTrue(all(a["type"] == "move" for a in legal))
                        actions[player] = rng.choice(legal)
                    env.step(actions)
                    for player in (0, 1):
                        self.assertTrue(reachable(env.positions[player], env.goal,
                                                  env.walls[player], env.size))
                        self.assertNotIn(env.positions[player], env.walls[player])
                        self.assertEqual(env.blocks_remaining[player] + env.blocks_placed[player], 2)
                self.assertLessEqual(env.turn, env.max_rounds)
                self.assertAlmostEqual(sum(env.returns()), 0)
                self.assertTrue(all(-1 <= r <= 1 for r in env.returns()))
                with self.assertRaises(RuntimeError):
                    env.step({})

    def test_goal_draw_and_explicit_distance_adjudication(self):
        env = MazeEnv(1)
        env.positions = [(2, 3), (2, 3)]
        env.walls = [set(), set()]
        env.step({p: {"type": "move", "direction": "E"} for p in (0, 1)})
        self.assertEqual(env.returns(), [0, 0])
        self.assertEqual(env.finish_reason, "simultaneous_finish")
        env = MazeEnv(1, max_rounds=1)
        env.walls = [set(), set()]
        env.step({0: {"type": "move", "direction": "E"},
                  1: {"type": "move", "direction": "N"}})
        self.assertEqual(env.finish_reason, "distance_adjudication_at_cap")
        self.assertEqual(env.metrics()["shortest_distances"], [3, 5])
        self.assertAlmostEqual(env.returns()[0], 0.2)
        env = MazeEnv(1)
        env.positions = [(2, 3), (2, 0)]
        env.walls = [set(), set()]
        env.step({p: {"type": "move", "direction": "E"} for p in (0, 1)})
        self.assertEqual(env.returns(), [1, -1])


class ChessTests(unittest.TestCase):
    @staticmethod
    def set_board(env, fen):
        env.board = chess.Board(fen)
        env.initial_fen = fen
        env.initial_material = env._material()
        env._check_terminal()

    @staticmethod
    def play(env, uci):
        env.step({env.active_players()[0]: {"move": uci}})

    def test_determinism_color_pairs_and_legal_endgame_curriculum(self):
        for regime in ("train", "eval", "ood"):
            for seed in range(10):
                env = ChessEnv(seed, regime)
                self.assertEqual(env.snapshot(), ChessEnv(seed, regime).snapshot())
                self.assertTrue(env.board.is_valid())
                self.assertEqual(env._material()[0], env._material()[1])
                self.assertFalse(env.done)
                self.assertLessEqual(len(env.legal_actions(env.active_players()[0])), 256)
                if seed % 2 == 0:
                    self.assertEqual(env.board.mirror().fen(), ChessEnv(seed + 1, regime).board.fen())
                json.dumps(env.snapshot())
        standard = ChessEnv(100, "standard")
        self.assertEqual(standard.board.fen(), chess.STARTING_FEN)
        self.assertEqual(len(standard.legal_actions(0)), 20)
        self.assertEqual(standard.legal_actions(1), [])

    def test_invalid_actions_do_not_change_game_and_history_is_public(self):
        env = ChessEnv(5, "standard")
        before = env.snapshot()
        for actions in ({1: {"move": "e7e5"}}, {0: {"move": "e2e5"}},
                        {False: {"move": "e2e4"}},
                        {0: {"move": "e2e4"}, 1: {"move": "e7e5"}},
                        {0: {"claim_draw": True}}):
            with self.assertRaises(ValueError):
                env.step(actions)
            self.assertEqual(env.snapshot(), before)
        self.play(env, "e2e4")
        self.assertEqual(env.observation(1)["move_history"], ["e2e4"])
        self.assertEqual(env.observation(1)["your_color"], "black")

    def test_checkmate_is_actual_terminal_win(self):
        env = ChessEnv(1, "standard")
        for move in ("f2f3", "e7e5", "g2g4", "d8h4"):
            self.play(env, move)
        self.assertTrue(env.board.is_checkmate())
        self.assertEqual(env.finish_reason, "checkmate")
        self.assertEqual(env.returns(), [-1, 1])
        self.assertEqual(env.active_players(), [])
        with self.assertRaises(RuntimeError):
            env.step({0: {"move": "a2a3"}})

    def test_stalemate_insufficient_material_and_claimed_repetition(self):
        for fen, reason in (("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1", "stalemate"),
                            ("7k/8/8/8/8/8/8/K7 w - - 0 1", "insufficient_material")):
            env = ChessEnv(1)
            self.set_board(env, fen)
            self.assertTrue(env.done)
            self.assertEqual(env.finish_reason, reason)
            self.assertEqual(env.returns(), [0, 0])
        env = ChessEnv(1, "standard", max_rounds=100)
        for move in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
            self.play(env, move)
        self.assertIn({"claim_draw": True}, env.legal_actions(0))
        env.step({0: {"claim_draw": True}})
        self.assertEqual(env.finish_reason, "claimed_threefold_repetition")
        self.assertEqual(env.returns(), [0, 0])

    def test_cap_is_material_adjudication_not_checkmate(self):
        env = ChessEnv(1, max_rounds=1)
        self.set_board(env, "7k/8/8/8/8/8/8/KQ6 w - - 0 1")
        self.play(env, "b1b2")
        self.assertTrue(env.done)
        self.assertFalse(env.metrics()["checkmate"])
        self.assertEqual(env.finish_reason, "material_adjudication_at_cap")
        self.assertEqual(env.metrics()["material_advantage_white"], 9)
        self.assertEqual(env.returns(), [0.9, -0.9])

    def test_special_moves_and_capture_accounting(self):
        env = ChessEnv(1, "standard")
        for move in ("e2e4", "a7a6", "e4e5", "d7d5", "e5d6"):
            self.play(env, move)
        self.assertEqual(env.captures, [1, 0])
        self.assertIsNone(env.board.piece_at(chess.D5))
        self.assertEqual(env.board.piece_at(chess.D6).piece_type, chess.PAWN)
        env = ChessEnv(1, "standard")
        for move in ("g1f3", "g8f6", "g2g3", "g7g6", "f1g2", "f8g7", "e1g1"):
            self.play(env, move)
        self.assertEqual(env.board.piece_at(chess.G1).piece_type, chess.KING)
        self.assertEqual(env.board.piece_at(chess.F1).piece_type, chess.ROOK)
        env = ChessEnv(1)
        self.set_board(env, "7k/P7/8/8/8/8/8/7K w - - 0 1")
        self.assertTrue(all({"move": "a7a8" + p} in env.legal_actions(0) for p in "qrbn"))
        self.play(env, "a7a8q")
        self.assertEqual(env.promotions, [1, 0])
        self.assertEqual(env._material(), [9, 0])

    def test_heuristic_is_legal_and_games_terminate(self):
        for regime in ("train", "ood", "standard"):
            for seed in range(4):
                env = ChessEnv(seed, regime)
                while not env.done:
                    player = env.active_players()[0]
                    action = env.heuristic_action(player)
                    self.assertIn(action, env.legal_actions(player))
                    env.step({player: action})
                    self.assertTrue(env.board.is_valid())
                self.assertLessEqual(env.turn, env.max_rounds)
                self.assertAlmostEqual(sum(env.returns()), 0)
                self.assertTrue(all(-1 <= reward <= 1 for reward in env.returns()))


if __name__ == "__main__":
    unittest.main()
