"""Two programmable robots competing over finite shared resources."""
from __future__ import annotations

import copy
import random


class RobotArena:
    """Symmetric 7x7 resource race; programs choose simultaneous bounded actions.

    Robots start at opposite home cells. Moving and harvesting each cost one
    energy. Rest restores three energy up to ten. Capacity is three resource
    units. Points are earned only by returning resources to one's own home and
    depositing. Robots may share cells; simultaneous harvests resolve using a
    pre-sampled priority independent of program execution and random draws.
    """

    size = 7
    capacity = 3
    max_energy = 10

    def __init__(self, seed: int, max_rounds: int = 48):
        if type(seed) is not int or type(max_rounds) is not int or not 1 <= max_rounds <= 256:
            raise ValueError("integer seed and horizon 1..256 required")
        self.seed, self.max_rounds = seed, max_rounds
        self.turn, self.done = 0, False
        self.robots = [{"x": x, "y": 3, "home_x": x, "home_y": 3,
                        "energy": self.max_energy, "cargo": [], "score": 0,
                        "banked_units": 0, "harvested_units": 0, "moves": 0,
                        "rests": 0, "deposits": 0, "failed_actions": 0}
                       for x in (0, self.size - 1)]
        rng = random.Random(seed)
        candidates = [(x, y) for x in (1, 2) for y in range(1, 6)]
        rng.shuffle(candidates)
        self.sites = []
        for x, y in candidates[:3]:
            units, value = rng.randint(2, 3), rng.randint(1, 4)
            for sx in (x, self.size - 1 - x):
                self.sites.append({"x": sx, "y": y, "units": units, "value": value})
        self.total_value = sum(s["units"] * s["value"] for s in self.sites)
        self.total_units = sum(s["units"] for s in self.sites)
        self.priority = [rng.sample([0, 1], 2) for _ in range(max_rounds)]
        self.history = []
        self._assert_invariants()

    def observation(self, player: int) -> dict:
        if type(player) is not int or player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        own, other = self.robots[player], self.robots[1 - player]
        return {"player": player, "turn": self.turn, "horizon": self.max_rounds, "grid_size": self.size,
                "self_x": own["x"], "self_y": own["y"], "home_x": own["home_x"], "home_y": own["home_y"],
                "energy": own["energy"], "max_energy": self.max_energy,
                "carrying": len(own["cargo"]), "capacity": self.capacity, "score": own["score"],
                "opponent_x": other["x"], "opponent_y": other["y"], "opponent_energy": other["energy"],
                "opponent_carrying": len(other["cargo"]), "opponent_score": other["score"],
                "site_count": len(self.sites), "sites": copy.deepcopy(self.sites)}

    @staticmethod
    def _validate_action(action):
        if not isinstance(action, dict) or not isinstance(action.get("action"), str):
            raise ValueError("action must be a structured action object")
        name = action["action"]
        if name in {"rest", "harvest", "deposit"} and set(action) == {"action"}:
            return
        if (name == "move" and set(action) == {"action", "dx", "dy"}
                and type(action["dx"]) is int and type(action["dy"]) is int
                and abs(action["dx"]) + abs(action["dy"]) == 1):
            return
        raise ValueError("invalid action structure")

    def step(self, actions: dict) -> None:
        if self.done:
            raise ValueError("game already finished")
        if not isinstance(actions, dict) or set(actions) != {0, 1} or any(type(p) is not int for p in actions):
            raise ValueError("both simultaneous actions are required")
        for action in actions.values():
            self._validate_action(action)
        outcomes = {}
        # Move intents are resolved simultaneously; sharing a cell is legal.
        for player in (0, 1):
            robot, action = self.robots[player], actions[player]
            if action["action"] == "move":
                x, y = robot["x"] + action["dx"], robot["y"] + action["dy"]
                if not robot["energy"]:
                    outcomes[player] = "no_energy"
                elif not 0 <= x < self.size or not 0 <= y < self.size:
                    outcomes[player] = "outside_grid"
                else:
                    robot.update(x=x, y=y, energy=robot["energy"] - 1, moves=robot["moves"] + 1)
                    outcomes[player] = "moved"
        for player in self.priority[self.turn]:
            robot, action = self.robots[player], actions[player]
            name = action["action"]
            if name == "harvest":
                site = next((s for s in self.sites if s["x"] == robot["x"] and s["y"] == robot["y"]), None)
                if not robot["energy"]:
                    outcomes[player] = "no_energy"
                elif len(robot["cargo"]) == self.capacity:
                    outcomes[player] = "cargo_full"
                elif site is None or not site["units"]:
                    outcomes[player] = "no_resource"
                else:
                    site["units"] -= 1
                    robot["cargo"].append(site["value"])
                    robot["energy"] -= 1
                    robot["harvested_units"] += 1
                    outcomes[player] = "harvested"
            elif name == "deposit":
                if (robot["x"], robot["y"]) != (robot["home_x"], robot["home_y"]):
                    outcomes[player] = "not_at_home"
                else:
                    robot["score"] += sum(robot["cargo"])
                    robot["banked_units"] += len(robot["cargo"])
                    robot["cargo"] = []
                    robot["deposits"] += 1
                    outcomes[player] = "deposited"
            elif name == "rest":
                robot["energy"] = min(self.max_energy, robot["energy"] + 3)
                robot["rests"] += 1
                outcomes[player] = "rested"
        success = {"moved", "harvested", "deposited", "rested"}
        for player, outcome in outcomes.items():
            self.robots[player]["failed_actions"] += outcome not in success
        self.history.append({"turn": self.turn, "actions": copy.deepcopy(actions),
                             "priority": list(self.priority[self.turn]), "outcomes": outcomes})
        self.turn += 1
        self.done = self.turn == self.max_rounds
        self._assert_invariants()

    def returns(self):
        if not self.done:
            raise ValueError("terminal rewards require a finished game")
        score = (self.robots[0]["score"] - self.robots[1]["score"]) / self.total_value
        return [score, -score]

    def metrics(self):
        return {"turn": self.turn, "terminal": self.done,
                "scores": [r["score"] for r in self.robots],
                "banked_units": [r["banked_units"] for r in self.robots],
                "harvested_units": [r["harvested_units"] for r in self.robots],
                "carried_value": [sum(r["cargo"]) for r in self.robots],
                "failed_actions": [r["failed_actions"] for r in self.robots],
                "deposits": [r["deposits"] for r in self.robots],
                "moves": [r["moves"] for r in self.robots], "rests": [r["rests"] for r in self.robots],
                "remaining_value": sum(s["units"] * s["value"] for s in self.sites),
                "total_value": self.total_value,
                "returns": self.returns() if self.done else None}

    def snapshot(self):
        return {"seed": self.seed, "turn": self.turn, "done": self.done, "max_rounds": self.max_rounds,
                "robots": copy.deepcopy(self.robots), "sites": copy.deepcopy(self.sites),
                "priority": copy.deepcopy(self.priority), "history": copy.deepcopy(self.history),
                "metrics": self.metrics()}

    def _assert_invariants(self):
        assert sum(s["units"] * s["value"] for s in self.sites) + sum(sum(r["cargo"]) + r["score"] for r in self.robots) == self.total_value
        assert sum(s["units"] for s in self.sites) + sum(len(r["cargo"]) + r["banked_units"] for r in self.robots) == self.total_units
        assert all(s["units"] >= 0 for s in self.sites)
        for r in self.robots:
            assert 0 <= r["energy"] <= self.max_energy and len(r["cargo"]) <= self.capacity
            assert 0 <= r["x"] < self.size and 0 <= r["y"] < self.size
