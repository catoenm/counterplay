"""Legal chess with short endgame curricula and explicit capped adjudication.

Dependency: python-chess==1.999 (chess==1.11.2). Player 0 is White and player 1
is Black. Train/eval use equal-material KRP vs KRP endgames; OOD uses KQPP vs
KQPP. Paired seeds 2*k and 2*k+1 mirror the same position and exchange colors,
including which side moves first. regime='standard' starts ordinary chess.
"""
from __future__ import annotations

import random
import chess


PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


class ChessEnv:
    name = "chess"

    def __init__(self, seed: int, regime: str = "train", max_rounds: int = 24):
        if regime not in {"train", "eval", "ood", "standard"}:
            raise ValueError("chess regime must be train, eval, ood, or standard")
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        self.seed, self.regime = int(seed), regime
        # Chess uses one ply per environment step, not one move by both sides.
        self.max_rounds = int(max_rounds)
        self.board = chess.Board() if regime == "standard" else self._endgame()
        self.initial_fen = self.board.fen()
        self.turn, self.done = 0, False
        self.finish_reason = None
        self._result = [0.0, 0.0]
        self.captures = [0, 0]
        self.promotions = [0, 0]
        self.initial_material = self._material()
        self._check_terminal()

    def _endgame(self):
        rng = random.Random(self.seed // 2)
        for _ in range(10000):
            board = chess.Board(None)
            piece_types = [chess.KING, chess.QUEEN if self.regime == "ood" else chess.ROOK,
                           chess.PAWN]
            if self.regime == "ood":
                piece_types.append(chess.PAWN)
            available = list(chess.SQUARES)
            for color in (chess.WHITE, chess.BLACK):
                for piece_type in piece_types:
                    choices = [s for s in available if piece_type != chess.PAWN
                               or chess.square_rank(s) not in {0, 7}]
                    square = rng.choice(choices)
                    available.remove(square)
                    board.set_piece_at(square, chess.Piece(piece_type, color))
            board.turn = chess.WHITE
            # Avoid illegal boards and any checked king at the initial state.
            if not board.is_valid() or board.is_check() or board.is_game_over():
                continue
            opponent_board = board.copy(stack=False)
            opponent_board.turn = chess.BLACK
            if not opponent_board.is_valid() or opponent_board.is_check():
                continue
            # Short episodes should contain a real tactical choice. This is a
            # declared endgame sampling condition, not a general chess benchmark.
            if not (any(board.is_capture(m) for m in board.legal_moves)
                    or any(opponent_board.is_capture(m) for m in opponent_board.legal_moves)):
                continue
            return board.mirror() if self.seed % 2 else board
        raise RuntimeError("unable to sample a valid chess endgame")

    @staticmethod
    def _player(color):
        return 0 if color == chess.WHITE else 1

    def active_players(self):
        return [] if self.done else [self._player(self.board.turn)]

    def legal_actions(self, player):
        if type(player) is not int or player not in {0, 1}:
            raise ValueError("player must be 0 or 1")
        if self.done or player != self._player(self.board.turn):
            return []
        actions = [{"move": move.uci()} for move in sorted(self.board.legal_moves, key=lambda m: m.uci())]
        if self.board.can_claim_draw():
            actions.append({"claim_draw": True})
        return actions

    def observation(self, player):
        if type(player) is not int or player not in {0, 1}:
            raise ValueError("player must be 0 or 1")
        return {
            "environment": self.name, "player": player,
            "your_color": "white" if player == 0 else "black",
            "side_to_move": "white" if self.board.turn == chess.WHITE else "black",
            "ply": self.turn, "ply_limit": self.max_rounds,
            "rules": (
                "Ordinary legal chess: move in UCI notation (e2e4, a7a8q for promotion). "
                "Player 0 is White; player 1 is Black. Checkmate wins +1 and loses -1. "
                "Stalemate, insufficient material, fivefold repetition, and 75-move draws "
                "end automatically. A listed claim_draw action claims a legal threefold "
                "or 50-move draw, including a claim possible by an intended next move. "
                "At the ply limit only, White reward = (White material - Black material)/10 "
                "clipped to [-0.9,0.9]; Black gets its negative. Piece values: "
                "pawn 1, knight/bishop 3, rook 5, queen 9. This is capped material "
                "adjudication, not checkmate. Capital letters are White; lowercase Black."
            ),
            "sampling": "standard starting position" if self.regime == "standard" else
                        "equal-material tactical endgame; paired seeds swap colors",
            "fen": self.board.fen(), "initial_fen": self.initial_fen,
            "board": str(self.board), "board_rows": "8 through 1, files a through h",
            "move_history": [move.uci() for move in self.board.move_stack],
            "in_check": self.board.is_check(), "material": self._material(),
            "can_claim_draw": self.board.can_claim_draw(), "done": self.done,
        }

    def _material(self):
        return [sum(PIECE_VALUES[t] * len(self.board.pieces(t, color)) for t in PIECE_VALUES)
                for color in (chess.WHITE, chess.BLACK)]

    def _check_terminal(self):
        outcome = self.board.outcome(claim_draw=False)
        if outcome is not None:
            self.done = True
            self.finish_reason = outcome.termination.name.lower()
            reward = 0.0 if outcome.winner is None else (1.0 if outcome.winner else -1.0)
            self._result = [reward, -reward]
        elif self.turn >= self.max_rounds:
            self.done, self.finish_reason = True, "material_adjudication_at_cap"
            material = self._material()
            reward = max(-0.9, min(0.9, (material[0] - material[1]) / 10.0))
            self._result = [reward, -reward]

    def step(self, actions):
        if self.done:
            raise RuntimeError("episode already finished")
        player = self._player(self.board.turn)
        if (not isinstance(actions, dict) or any(type(p) is not int for p in actions)
                or set(actions) != {player} or not isinstance(actions[player], dict)
                or ("claim_draw" in actions[player] and actions[player]["claim_draw"] is not True)
                or actions[player] not in self.legal_actions(player)):
            raise ValueError("submit exactly one legal action for the side to move")
        action = actions[player]
        if action.get("claim_draw"):
            self.done = True
            self.turn += 1
            self.finish_reason = ("claimed_fifty_moves" if self.board.can_claim_fifty_moves()
                                  else "claimed_threefold_repetition")
            self._result = [0.0, 0.0]
            return
        move = chess.Move.from_uci(action["move"])
        if self.board.is_capture(move):
            self.captures[player] += 1
        if move.promotion:
            self.promotions[player] += 1
        self.board.push(move)
        self.turn += 1
        self._check_terminal()

    def returns(self):
        if not self.done:
            raise RuntimeError("returns are only defined after termination")
        return list(self._result)

    def metrics(self):
        material = self._material()
        return {"plies": self.turn, "finish_reason": self.finish_reason,
                "material": material, "initial_material": list(self.initial_material),
                "material_advantage_white": material[0] - material[1],
                "captures": list(self.captures), "promotions": list(self.promotions),
                "checkmate": self.board.is_checkmate(), "stalemate": self.board.is_stalemate(),
                "returns": list(self._result)}

    def snapshot(self):
        return {**self.observation(0), "seed": self.seed, "regime": self.regime,
                "metrics": self.metrics(), "python_chess_version": chess.__version__}

    def heuristic_action(self, player):
        """Fixed public-state one-ply opponent with a capture-reply penalty."""
        actions = self.legal_actions(player)
        if not actions:
            raise RuntimeError("no active action")
        ranked = []
        for index, action in enumerate(actions):
            if action.get("claim_draw"):
                score = 0.0
            else:
                board = self.board.copy(stack=True)
                move = chess.Move.from_uci(action["move"])
                captured = (1 if board.is_en_passant(move) else
                            PIECE_VALUES.get(board.piece_type_at(move.to_square), 0))
                promotion_gain = (PIECE_VALUES[move.promotion] - 1) if move.promotion else 0
                board.push(move)
                if board.is_checkmate():
                    score = 1000.0
                elif board.outcome(claim_draw=False) is not None:
                    score = 0.0
                else:
                    opponent_capture = max((1 if board.is_en_passant(reply) else
                        PIECE_VALUES.get(board.piece_type_at(reply.to_square), 0)
                        for reply in board.legal_moves if board.is_capture(reply)), default=0)
                    score = captured + promotion_gain - opponent_capture
                    score += 0.05 if board.is_check() else 0.0
            ranked.append((score, -index, action))
        return dict(max(ranked, key=lambda item: item[:2])[2])
