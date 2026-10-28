"""Stepwise, single-player wrapper around the existing exact planning tasks.

Only public transition rules, state and goal enter observations. Witnesses and
reference search are restricted to validation, replay and the named oracle bot.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json

from ..environment import RULES, Task, generate_tasks, step as transition


def task_family_payload(task):
    """Same x/y symmetry and budget exclusion as selfplay.depth.family."""
    fields = {k: v for k, v in asdict(task).items() if k not in ('witness', 'max_steps')}
    fields['actions'] = sorted(fields['actions'])
    dual_names = {'inc_x':'inc_y', 'inc_y':'inc_x', 'sum_x':'sum_y', 'sum_y':'sum_x',
                  'double_x':'double_y', 'double_y':'double_x', 'swap':'swap'}
    dual = {**fields, 'start':list(reversed(task.start)), 'goal':list(reversed(task.goal)),
            'actions':sorted(dual_names[a] for a in task.actions)}
    return json.loads(min(json.dumps(fields, sort_keys=True), json.dumps(dual, sort_keys=True)))


class PlanningEnv:
    name = 'planning'
    role_count = 1

    def __init__(self, seed, regime='train', max_rounds=12):
        if type(seed) is not int or seed < 0 or regime not in ('train', 'eval', 'ood'):
            raise ValueError('Invalid planning seed/regime')
        if type(max_rounds) is not int or not 6 <= max_rounds <= 12:
            raise ValueError('Planning horizon must be in6..12')
        self.seed, self.regime, self.max_rounds = seed, regime, max_rounds
        depth = (3, 4, 5, 6)[seed % 4]
        modulus = 13 if regime == 'ood' else 11
        original = generate_tasks(1, seed=seed, depths=(depth,), modulus=modulus)[0]
        self._task = Task(original.modulus, original.start, original.goal, original.actions,
                          max_rounds, original.witness)
        self._initial_shortest_depth = depth
        self.state, self.turn, self.done = self._task.start, 0, False
        self.history, self.finish_reason = [], None

    def active_players(self):
        return [] if self.done else [0]

    @staticmethod
    def _player(player):
        if type(player) is not int or player != 0:
            raise ValueError('Planning has exactly one active role:0')

    def legal_actions(self, player):
        self._player(player)
        return [] if self.done else [{'tool':a} for a in sorted(self._task.actions)]

    def observation(self, player):
        self._player(player)
        return {'environment':self.name, 'player':0, 'role_count':1, 'turn':self.turn,
                'round_limit':self.max_rounds, 'modulus':self._task.modulus,
                'start':list(self._task.start), 'state':list(self.state), 'goal':list(self._task.goal),
                'tools':{a:RULES[a] for a in sorted(self._task.actions)},
                'history':deepcopy(self.history), 'done':self.done,
                'rules':'Single-player verified planning. Choose one listed tool per turn. All arithmetic '
                        'uses the state before that tool and is modulo m. Tools may repeat. Reach the exact '
                        'goal within the turn limit for reward1; otherwise reward0. No opponent role exists.'}

    def step(self, actions):
        if self.done:
            raise RuntimeError('Episode already terminal')
        if (not isinstance(actions, dict) or list(actions) != [0]
                or type(next(iter(actions))) is not int or actions[0] not in self.legal_actions(0)):
            raise ValueError('Submit exactly one legal role0 action')
        before = self.state
        self.state = transition(self.state, actions[0]['tool'], self._task.modulus)
        self.turn += 1
        self.history.append({'turn':self.turn, 'action':deepcopy(actions[0]),
                             'before':list(before), 'after':list(self.state)})
        if self.state == self._task.goal:
            self.done, self.finish_reason = True, 'verified_goal'
        elif self.turn >= self.max_rounds:
            self.done, self.finish_reason = True, 'planning_step_cap'

    def returns(self):
        if not self.done:
            raise RuntimeError('Returns only exist at termination')
        return [float(self.state == self._task.goal), 0.0]

    def metrics(self):
        return {'turns':self.turn, 'finish_reason':self.finish_reason,
                'verified_goal':self.state == self._task.goal,
                'returns':self.returns() if self.done else None, 'role_count':1}

    def fingerprint_payload(self):
        return {'environment':self.name, 'canonical_task':task_family_payload(self._task)}

    def snapshot(self):
        return {**self.observation(0), 'seed':self.seed, 'regime':self.regime,
                'metrics':self.metrics(), 'scenario_identity':self.fingerprint_payload(),
                'audit_only':{'task':asdict(self._task), 'initial_shortest_depth':self._initial_shortest_depth}}

    def heuristic_action(self, player):
        """Named exact-solver benchmark; solver output never enters policy input."""
        self._player(player)
        from collections import deque
        if self.done:
            raise RuntimeError('No active action')
        queue, seen = deque([(self.state, ())]), {self.state}
        while queue:
            state, path = queue.popleft()
            if state == self._task.goal and path:
                return {'tool':path[0]}
            if len(path) >= self.max_rounds - self.turn:
                continue
            for action in sorted(self._task.actions):
                following = transition(state, action, self._task.modulus)
                if following not in seen:
                    seen.add(following); queue.append((following, path + (action,)))
        return self.legal_actions(0)[0]

    def assert_private_boundary(self):
        before = (self.observation(0), self.legal_actions(0))
        other = deepcopy(self)
        # Bypass construction only for a counterfactual non-policy audit field.
        object.__setattr__(other._task, 'witness', ('withheld',))
        other._initial_shortest_depth += 1000
        if before != (other.observation(0), other.legal_actions(0)):
            raise ValueError('Planning witness/solver information leaked')
        return {'checked':True, 'roles':[0], 'private_fields':['witness', 'initial_shortest_depth']}
