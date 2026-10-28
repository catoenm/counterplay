"""Count exact shortest distances in the finite proposal grammar."""
import argparse
from collections import Counter, deque
from itertools import combinations, product
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selfplay.environment import RULES, step


def enumerate_space(modulus=7):
    counts = Counter()
    unreachable = 0
    states = list(product(range(modulus), repeat=2))
    action_sets = list(combinations(sorted(RULES), 4))
    for actions in action_sets:
        for start in states:
            distances = {start: 0}
            queue = deque([start])
            while queue:
                state = queue.popleft()
                for action in actions:
                    target = step(state, action, modulus)
                    if target not in distances:
                        distances[target] = distances[state] + 1
                        queue.append(target)
            counts.update(d for d in distances.values() if d > 0)
            unreachable += len(states) - len(distances)
    nontrivial_pairs = len(action_sets) * len(states) * (len(states) - 1)
    assert sum(counts.values()) + unreachable == nontrivial_pairs
    return {
        "modulus": modulus, "states": len(states), "four_action_sets": len(action_sets),
        "nontrivial_start_goal_action_set_combinations": nontrivial_pairs,
        "reachable_by_shortest_depth": dict(sorted(counts.items())),
        "unreachable": unreachable,
        "within_proposal_depth_range_2_to_4": sum(counts[d] for d in (2, 3, 4)),
        "interpretation": "Counts distinct start/goal/action-set combinations, not model difficulty or semantically independent tasks; isomorphic problems are not collapsed. Each can be given its shortest path as witness and exact depth as step budget. Longer budgets create additional task identities not counted here.",
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = enumerate_space()
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
