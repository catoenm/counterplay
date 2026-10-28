"""Finite planning tasks with withheld witnesses and an exact reference solver.

The model supplies data, never executable code. All arithmetic is modulo m.
This is a deliberately small diagnostic environment, not a novel RL algorithm.
"""
import hashlib
import json
import random
from collections import deque
from dataclasses import asdict, dataclass


RULES = {
    "inc_x": "x = (x + 1) mod m; y is unchanged",
    "inc_y": "y = (y + 1) mod m; x is unchanged",
    "swap": "swap x and y",
    "sum_x": "x = (x + y) mod m; y is unchanged",
    "sum_y": "y = (x + y) mod m; x is unchanged",
    "double_x": "x = (2 * x) mod m; y is unchanged",
    "double_y": "y = (2 * y) mod m; x is unchanged",
}


def step(state, action, modulus):
    x, y = state
    if action == "inc_x": x += 1
    elif action == "inc_y": y += 1
    elif action == "swap": x, y = y, x
    elif action == "sum_x": x += y
    elif action == "sum_y": y += x
    elif action == "double_x": x *= 2
    elif action == "double_y": y *= 2
    else: raise ValueError("Unknown action")
    return x % modulus, y % modulus


@dataclass(frozen=True)
class Task:
    modulus: int
    start: tuple
    goal: tuple
    actions: tuple
    max_steps: int
    witness: tuple

    def __post_init__(self):
        if type(self.modulus) is not int or not 3 <= self.modulus <= 31:
            raise ValueError("modulus must be an integer in [3, 31]")
        for state in (self.start, self.goal):
            if len(state) != 2 or any(type(v) is not int or not 0 <= v < self.modulus for v in state):
                raise ValueError("Invalid state")
        if not 2 <= len(self.actions) <= 7 or any(type(a) is not str or a not in RULES for a in self.actions):
            raise ValueError("Invalid action set")
        if len(set(self.actions)) != len(self.actions):
            raise ValueError("Duplicate actions")
        if type(self.max_steps) is not int or not 1 <= self.max_steps <= 12:
            raise ValueError("max_steps must be an integer in [1, 12]")
        if not self.accepts(self.witness):
            raise ValueError("Proposer witness does not solve task within budget")
        if self.start == self.goal:
            raise ValueError("Trivial task")

    def accepts(self, plan):
        if not isinstance(plan, (tuple, list)) or len(plan) > self.max_steps:
            return False
        state = self.start
        for a in plan:
            if type(a) is not str or a not in self.actions:
                return False
            state = step(state, a, self.modulus)
        return state == self.goal

    def shortest_plan(self):
        queue = deque([(self.start, ())])
        seen = {self.start}
        while queue:
            state, path = queue.popleft()
            if state == self.goal:
                return path
            if len(path) == self.max_steps:
                continue
            for a in sorted(self.actions):
                nxt = step(state, a, self.modulus)
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, path + (a,)))
        return None

    @property
    def identity(self):
        # Witnesses and action ordering do not define a different problem.
        public = {k: v for k, v in asdict(self).items() if k != "witness"}
        public["actions"] = sorted(self.actions)
        return hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_dict(cls, d):
        if not isinstance(d, dict) or set(d) != set(cls.__dataclass_fields__):
            raise ValueError("Task must have exactly the documented fields")
        for key in ("start", "goal", "actions", "witness"):
            if not isinstance(d[key], (tuple, list)):
                raise ValueError("Expected array for " + key)
        return cls(**{k: tuple(v) if k in ("start", "goal", "actions", "witness") else v for k, v in d.items()})


def generate_tasks(count=24, seed=17, depths=(1, 2, 3), modulus=7):
    rng = random.Random(seed)
    tasks, seen = [], set()
    for _ in range(max(1000, count * 1000)):
        depth = depths[len(tasks) % len(depths)]
        actions = tuple(sorted(rng.sample(list(RULES), 4)))
        start = (rng.randrange(modulus), rng.randrange(modulus))
        path = tuple(rng.choice(actions) for _ in range(depth))
        goal = start
        for a in path:
            goal = step(goal, a, modulus)
        if goal == start:
            continue
        task = Task(modulus, start, goal, actions, depth, path)
        if len(task.shortest_plan()) != depth or task.identity in seen:
            continue
        seen.add(task.identity)
        tasks.append(task)
        if len(tasks) == count:
            return tasks
    raise ValueError("Could not generate enough distinct tasks at requested depths")


def construct_task(d):
    """Alice supplies a path; the environment derives Bob's reachable target.

    This is a different proposal protocol from checking a claimed goal. It must
    be logged separately, never used to silently repair an invalid claimed task.
    """
    if not isinstance(d, dict) or set(d) != set(Task.__dataclass_fields__) - {"goal"}:
        raise ValueError("Constructive proposals must omit goal")
    # Validate structure and types before executing any transitions.
    if type(d["modulus"]) is not int or not 3 <= d["modulus"] <= 31:
        raise ValueError("Invalid modulus")
    if not isinstance(d["start"], list) or len(d["start"]) != 2 or any(type(v) is not int or not 0 <= v < d["modulus"] for v in d["start"]):
        raise ValueError("Invalid start")
    if not isinstance(d["witness"], list) or len(d["witness"]) > 12:
        raise ValueError("Invalid witness")
    goal = tuple(d["start"])
    for a in d["witness"]:
        if type(a) is not str or a not in RULES:
            raise ValueError("Invalid witness action")
        goal = step(goal, a, d["modulus"])
    return Task.from_dict({**d, "goal": list(goal)})


def render(task, variant="canonical", seed=0, prompt_version="v1"):
    if variant not in ("canonical", "reordered", "renamed", "renamed_reordered"):
        raise ValueError("Unknown presentation")
    actions = sorted(task.actions)
    labels = ["tool_" + str(i) for i in range(len(actions))]
    if "renamed" in variant:
        labels = random.Random(seed).sample(["dax", "wug", "fep", "zup", "kiv", "bex", "lom"], len(actions))
    mapping = dict(zip(labels, actions))
    entries = list(mapping.items())
    if "reordered" in variant:
        random.Random(seed + 1).shuffle(entries)
    if prompt_version not in ("v1", "v2"):
        raise ValueError("Unknown prompt version")
    answer_instruction = (
        "You may repeat tools. Return only a JSON object with key plan and an array of tool names, "
        'for example {"plan":["tool_0"]}. Do not include explanations.\n'
    ) if prompt_version == "v1" else (
        "You may repeat tools. Your final answer must be only a JSON object with key plan "
        "and an array of tool names from the list below.\n"
    )
    prompt = (
        "Find a sequence of tools that transforms the starting state into the target state. "
        "All calculations use the state before that tool call. Each tool costs one step. "
        + answer_instruction +
        f"m = {task.modulus}\nStart (x,y) = {task.start}\nTarget (x,y) = {task.goal}\n"
        f"Maximum steps = {task.max_steps}\nTools:\n"
        + "\n".join(f"{label}: {RULES[action]}" for label, action in entries)
    )
    return prompt, mapping


def parse_object(text):
    # Accept a final JSON code fence, but never fish a plan out of reasoning text.
    text = text.rsplit("</think>", 1)[-1].strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value


def grade(task, text, mapping):
    try:
        obj = parse_object(text)
        if set(obj) != {"plan"} or not isinstance(obj["plan"], list):
            return {"success": False, "status": "format_error"}
        plan = obj["plan"]
        if any(type(a) is not str or a not in mapping for a in plan):
            return {"success": False, "status": "illegal_action"}
        if len(plan) > task.max_steps:
            return {"success": False, "status": "over_budget"}
        ok = task.accepts([mapping[a] for a in plan])
        return {"success": ok, "status": "solved" if ok else "wrong_goal"}
    except (ValueError, TypeError):
        return {"success": False, "status": "format_error"}
