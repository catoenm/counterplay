"""A simultaneous two-player maze race with connectivity-preserving sabotage.

Each player has an identical, separate grid. A turn spends one action on moving
or on adding a permanent roadblock to the other grid. All observations are
public; a roadblock can lengthen a route but can never remove every route.
"""
from __future__ import annotations

from collections import deque
import random


DIRECTIONS = (("N", -1, 0), ("E", 0, 1), ("S", 1, 0), ("W", 0, -1))


class MazeEnv:
    name = "maze"

    def __init__(self, seed: int, regime: str = "train", max_rounds: int = 10):
        if regime not in {"train", "eval", "ood"}:
            raise ValueError("maze regime must be train, eval, or ood")
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        self.seed, self.regime = int(seed), regime
        self.max_rounds = int(max_rounds)
        self.size = 7 if regime == "ood" else 5
        self.start = (self.size // 2, 0)
        self.goal = (self.size // 2, self.size - 1)
        rng = random.Random(self.seed)
        terrain = set()
        candidates = [(r, c) for r in range(self.size) for c in range(self.size)
                      if (r, c) not in {self.start, self.goal}]
        rng.shuffle(candidates)
        wall_target = int(self.size ** 2 * (0.20 if regime == "ood" else 0.16))
        for cell in candidates:
            proposed = terrain | {cell}
            if self._distance(self.start, proposed) is not None:
                terrain = proposed
            if len(terrain) == wall_target:
                break
        self.terrain = frozenset(terrain)
        self.walls = [set(terrain), set(terrain)]
        self.positions = [self.start, self.start]
        self.blocks_remaining = [2, 2]
        self.turn, self.done = 0, False
        self.finish_reason = None
        self._result = [0.0, 0.0]
        self.history = []
        self.initial_distance = self._distance(self.start, terrain)
        self.blocks_placed = [0, 0]
        self.blocks_cancelled = [0, 0]
        self.move_counts = [0, 0]

    def _distance(self, start, walls):
        """Shortest public path length, or None if disconnected."""
        if start in walls or self.goal in walls:
            return None
        queue, seen = deque([(start, 0)]), {start}
        while queue:
            (r, c), distance = queue.popleft()
            if (r, c) == self.goal:
                return distance
            for _, dr, dc in DIRECTIONS:
                cell = (r + dr, c + dc)
                if (0 <= cell[0] < self.size and 0 <= cell[1] < self.size
                        and cell not in walls and cell not in seen):
                    seen.add(cell)
                    queue.append((cell, distance + 1))
        return None

    def active_players(self):
        return [] if self.done else [0, 1]

    def _valid_block(self, target, cell):
        return (0 <= cell[0] < self.size and 0 <= cell[1] < self.size
                and cell not in self.walls[target]
                and cell not in {self.positions[target], self.goal}
                and self._distance(self.positions[target], self.walls[target] | {cell}) is not None)

    def legal_actions(self, player):
        if type(player) is not int or player not in {0, 1}:
            raise ValueError("player must be 0 or 1")
        if self.done:
            return []
        r, c = self.positions[player]
        actions = []
        for direction, dr, dc in DIRECTIONS:
            cell = (r + dr, c + dc)
            if (0 <= cell[0] < self.size and 0 <= cell[1] < self.size
                    and cell not in self.walls[player]):
                actions.append({"type": "move", "direction": direction})
        if self.blocks_remaining[player]:
            for row in range(self.size):
                for col in range(self.size):
                    if self._valid_block(1 - player, (row, col)):
                        actions.append({"type": "block", "row": row, "col": col})
        return actions

    def observation(self, player):
        if type(player) is not int or player not in {0, 1}:
            raise ValueError("player must be 0 or 1")
        return {
            "environment": self.name, "player": player, "round": self.turn,
            "round_limit": self.max_rounds, "size": self.size,
            "rules": (
                "Race to the goal on your own grid. Both players choose simultaneously. "
                "Use one action to move N/E/S/W or place a permanent block on the opponent's grid. "
                "Each player has two blocks. Never block a current position, goal, existing wall, "
                "or the last remaining route. Moves happen first; a block that then covers the "
                "opponent or disconnects them is cancelled without using its allowance. "
                "One finisher wins (+1/-1); simultaneous finish is a draw. At the round limit, "
                "reward is (opponent distance - own distance)/(2*size), clipped to [-0.5,0.5]. "
                "Rows increase south; columns increase east. All state is public."
            ),
            "goal": list(self.goal),
            "positions": [list(x) for x in self.positions],
            "walls": [[list(x) for x in sorted(w)] for w in self.walls],
            "blocks_remaining": list(self.blocks_remaining),
            "done": self.done,
        }

    def _distances(self):
        return [self._distance(self.positions[p], self.walls[p]) for p in (0, 1)]

    def step(self, actions):
        if self.done:
            raise RuntimeError("episode already finished")
        if (not isinstance(actions, dict) or any(type(p) is not int for p in actions)
                or set(actions) != {0, 1}):
            raise ValueError("submit exactly one action for each player")
        # Validate the complete simultaneous batch before changing any state.
        for player in (0, 1):
            action = actions[player]
            if (not isinstance(action, dict)
                    or (action.get("type") == "block"
                        and (type(action.get("row")) is not int or type(action.get("col")) is not int))
                    or action not in self.legal_actions(player)):
                raise ValueError(f"illegal maze action for player {player}")
        for player in (0, 1):
            action = actions[player]
            if action["type"] == "move":
                _, dr, dc = next(d for d in DIRECTIONS if d[0] == action["direction"])
                r, c = self.positions[player]
                self.positions[player] = (r + dr, c + dc)
                self.move_counts[player] += 1
        cancelled = []
        for player in (0, 1):
            action = actions[player]
            if action["type"] == "block":
                cell = (action["row"], action["col"])
                if self._valid_block(1 - player, cell):
                    self.walls[1 - player].add(cell)
                    self.blocks_remaining[player] -= 1
                    self.blocks_placed[player] += 1
                else:
                    self.blocks_cancelled[player] += 1
                    cancelled.append(player)
        self.turn += 1
        reached = [position == self.goal for position in self.positions]
        if any(reached):
            self.done = True
            self.finish_reason = "simultaneous_finish" if all(reached) else "goal"
            reward = float(reached[0]) - float(reached[1])
            self._result = [reward, -reward]
        elif self.turn >= self.max_rounds:
            self.done, self.finish_reason = True, "distance_adjudication_at_cap"
            distances = self._distances()
            reward = max(-0.5, min(0.5, (distances[1] - distances[0]) / (2 * self.size)))
            self._result = [reward, -reward]
        self.history.append({"round": self.turn,
                             "actions": {p: dict(actions[p]) for p in (0, 1)},
                             "positions": [list(x) for x in self.positions],
                             "cancelled_blocks": cancelled})

    def returns(self):
        if not self.done:
            raise RuntimeError("returns are only defined after termination")
        return list(self._result)

    def metrics(self):
        distances = self._distances()
        return {"rounds": self.turn, "finish_reason": self.finish_reason,
                "shortest_distances": distances,
                "distance_advantage_player0": distances[1] - distances[0],
                "initial_shortest_distance": self.initial_distance,
                "blocks_placed": list(self.blocks_placed),
                "blocks_cancelled": list(self.blocks_cancelled),
                "moves": list(self.move_counts), "returns": list(self._result)}

    def snapshot(self):
        return {**self.observation(0), "seed": self.seed, "regime": self.regime,
                "terrain": [list(x) for x in sorted(self.terrain)],
                "start": list(self.start), "metrics": self.metrics(),
                "history": [{**item, "actions": {p: dict(a) for p, a in item["actions"].items()},
                             "positions": [list(x) for x in item["positions"]],
                             "cancelled_blocks": list(item["cancelled_blocks"])} for item in self.history]}

    def heuristic_action(self, player):
        actions = self.legal_actions(player)
        if not actions:
            raise RuntimeError("no active action")
        current_distances = self._distances()
        candidates = []
        for index, action in enumerate(actions):
            if action["type"] == "move":
                _, dr, dc = next(d for d in DIRECTIONS if d[0] == action["direction"])
                r, c = self.positions[player]
                new_distance = self._distance((r + dr, c + dc), self.walls[player])
                score = current_distances[player] - new_distance
                if new_distance == 0:
                    score += 100
            else:
                target, cell = 1 - player, (action["row"], action["col"])
                new_distance = self._distance(self.positions[target], self.walls[target] | {cell})
                score = new_distance - current_distances[target] - 0.25
            candidates.append((score, -index, action))
        return dict(max(candidates, key=lambda item: item[:2])[2])
