#!/usr/bin/env python3
"""Describe saved scalar-policy behavior without executing models or games.

Reads exactly twenty production cohort episode files and the two completed
evaluation episode files. Prints JSON to stdout; never writes artifacts.
Terminal-return contrasts are derived from the saved two-episode cohorts,
not from gradients. Exact-reference matches are retrospective audit evidence,
not information that was available to the policy.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


ENVIRONMENTS = ("market", "maze", "chess", "planning", "coding")
FOCUS = ENVIRONMENTS[1:]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def counts(values):
    return dict(sorted(Counter(str(value) for value in values).items()))


def average(values):
    return math.fsum(values) / len(values)


def action_name(action):
    return action.get("type", action.get("tool", "claim_draw" if "claim_draw" in action else "move"))


def goal_count(episode):
    return sum(episode["metrics"]["shortest_distances"][p] == 0 for p in episode["learner_seats"])


def validate_episode(episode):
    require(episode["environment"] in ENVIRONMENTS, "Unknown environment")
    roles = episode["learner_seats"]
    require(roles and len(set(roles)) == len(roles) and all(type(p) is int and p in (0, 1) for p in roles),
            "Invalid learner roles")
    require(len(episode["returns"]) == 2 and all(type(x) in (int, float) and math.isfinite(x)
            and -1 <= x <= 1 for x in episode["returns"]), "Invalid terminal returns")
    # A fixed opponent can mate before the candidate receives its first turn.
    # Preserve that real loss in the episode denominator rather than dropping it.
    if not episode["transitions"]:
        require(episode["events"] and len(roles) == 1
                and all(not set(event["active_players"]) & set(roles) for event in episode["events"]),
                "Empty candidate trace without an opponent-only terminal game")
    seen = set()
    for row in episode["transitions"]:
        key = (row["player"], row["turn"])
        require(key not in seen and row["player"] in roles, "Duplicate or nonlearner decision")
        seen.add(key)
        require(row["episode_id"] == episode["episode_id"]
                and row["environment"] == episode["environment"]
                and row["terminal_return"] == episode["returns"][row["player"]],
                "Decision/episode/return mismatch")


def summarize(episodes, environment):
    decisions = [row for episode in episodes for row in episode["transitions"]]
    result = {
        "episodes": len(episodes), "learner_roles": sum(len(e["learner_seats"]) for e in episodes),
        "zero_learner_action_episodes": [e["episode_id"] for e in episodes if not e["transitions"]],
        "actions": len(decisions), "finish_reasons": counts(e["metrics"]["finish_reason"] for e in episodes),
        "terminal_reward_counts": counts(0.0 if e["returns"][p] == 0 else e["returns"][p]
                                         for e in episodes for p in e["learner_seats"]),
        "action_types": counts(action_name(row["action"]) for row in decisions),
    }
    if environment == "maze":
        moves = reversals = 0
        reverse = {"N": "S", "S": "N", "E": "W", "W": "E"}
        for episode in episodes:
            for player in episode["learner_seats"]:
                previous = None
                rows = sorted((r for r in episode["transitions"] if r["player"] == player), key=lambda r: r["turn"])
                for row in rows:
                    action = row["action"]
                    if action["type"] == "move":
                        moves += 1
                        reversals += bool(previous and previous["type"] == "move"
                                          and action["direction"] == reverse[previous["direction"]])
                    previous = action
        result.update(candidate_goal_count=sum(goal_count(e) for e in episodes),
                      grid_sizes=counts(e["initial_state"]["size"] for e in episodes),
                      initial_shortest_distances=counts(e["metrics"]["initial_shortest_distance"] for e in episodes),
                      move_actions=moves, immediate_reversal_moves=reversals,
                      immediate_reversal_fraction=reversals / moves if moves else None,
                      total_block_actions=sum(r["action"]["type"] == "block" for r in decisions),
                      block_actions_first_two_turns=sum(r["action"]["type"] == "block" and r["turn"] < 2 for r in decisions))
    elif environment == "chess":
        result.update(checkmate_games=sum(e["metrics"]["checkmate"] for e in episodes),
                      candidate_checkmate_wins=sum(e["metrics"]["checkmate"] and e["returns"][p] > 0
                                                  for e in episodes for p in e["learner_seats"]),
                      candidate_checkmate_losses=sum(e["metrics"]["checkmate"] and e["returns"][p] < 0
                                                    for e in episodes for p in e["learner_seats"]),
                      draw_claim_actions=sum("claim_draw" in r["action"] for r in decisions),
                      draw_claim_available_decisions=sum(any("claim_draw" in a for a in r["actions"]) for r in decisions),
                      mean_plies=average([e["metrics"]["plies"] for e in episodes]))
    elif environment == "planning":
        repeats = [len(e["transitions"]) - len({tuple(r["observation"]["state"]) for r in e["transitions"]})
                   for e in episodes]
        result.update(verified_goals=sum(e["metrics"]["verified_goal"] for e in episodes),
                      initial_shortest_depths=counts(e["initial_state"]["audit_only"]["initial_shortest_depth"] for e in episodes),
                      repeated_observed_state_decisions=sum(repeats), episodes_revisiting_state=sum(n > 0 for n in repeats),
                      solved_episode_ids=[e["episode_id"] for e in episodes if e["metrics"]["verified_goal"]])
    elif environment == "coding":
        reference_cases = []
        public_pass_events = []
        for episode in episodes:
            reference = episode["initial_state"]["audit_only"]["reference_solution"]
            hits = [r for r in episode["transitions"] if r["observation"]["program"] == reference]
            if hits:
                first = hits[0]
                reference_cases.append({"episode_id": episode["episode_id"], "turns_observed": [r["turn"] for r in hits],
                    "first_action_while_exact": first["action"], "final_hidden_fraction": episode["metrics"]["hidden_pass_fraction"],
                    "final_fully_solved": episode["metrics"]["hidden_all_passed"],
                    "reference_source_sha256": hashlib.sha256(reference.encode()).hexdigest()})
            for row in episode["transitions"]:
                feedback = row["observation"]["public_feedback"]
                if feedback and all(case["passed"] for case in feedback):
                    public_pass_events.append({"episode_id": episode["episode_id"], "turn": row["turn"],
                                               "passed": len(feedback), "next_action": row["action"]})
        test_events = [event for e in episodes for event in e["events"]
                       if list(event["actions"].values())[0].get("type") == "run_public_tests"]
        result.update(task_families=counts(e["initial_state"]["scenario_identity"]["task_family"] for e in episodes),
                      fully_solved=sum(e["metrics"]["hidden_all_passed"] for e in episodes),
                      mean_hidden_fraction=average([e["metrics"]["hidden_pass_fraction"] for e in episodes]),
                      edits=sum(e["metrics"]["edits"] for e in episodes),
                      public_test_runs=sum(e["metrics"]["public_test_runs"] for e in episodes),
                      episodes_without_public_test=sum(e["metrics"]["public_test_runs"] == 0 for e in episodes),
                      public_test_events_all_passed=sum(bool(event["state"]["public_feedback"])
                          and all(r["passed"] for r in event["state"]["public_feedback"]) for event in test_events),
                      episodes_with_repeated_observed_program=sum(len({r["observation"]["program"] for r in e["transitions"]})
                          < len(e["transitions"]) for e in episodes),
                      exact_reference_observed_before_action=reference_cases,
                      actions_with_all_public_tests_passed=public_pass_events)
        require(len(test_events) == result["public_test_runs"], "Public test event count differs")
    return result


def analyze(training_run, evaluation_run):
    inputs = {}
    def read(path):
        path = path.resolve()
        raw = path.read_bytes()
        inputs[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    datasets = {"train": [], "base": [], "final": []}
    signals = {e: {"cohorts_with_reward_contrast": [], "nonzero_terminal_advantage_actions": 0,
                   "total_actions": 0} for e in FOCUS}
    identifiers = set()
    for cohort in range(1, 21):
        episodes = read(training_run / f"cohorts/{cohort}/episodes.json")
        require(Counter(e["environment"] for e in episodes) == {e: 2 for e in ENVIRONMENTS},
                "Each production cohort must contain two episodes per environment")
        for episode in episodes:
            validate_episode(episode)
            require(episode["episode_id"] not in identifiers, "Duplicate training episode")
            identifiers.add(episode["episode_id"])
            require(episode["learner_seats"] == ([0, 1] if episode["environment"] in ENVIRONMENTS[:3] else [0]),
                    "Training roles differ from frozen protocol")
            require(episode["transitions"], "A production episode must have learner decisions")
        datasets["train"].extend(episodes)
        for environment in FOCUS:
            pair = [e for e in episodes if e["environment"] == environment]
            signal = False
            for episode in pair:
                other = pair[1] if episode is pair[0] else pair[0]
                for row in episode["transitions"]:
                    advantage = episode["returns"][row["player"]] - other["returns"][row["player"]]
                    nonzero = abs(advantage) > 1e-12
                    signals[environment]["total_actions"] += 1
                    signals[environment]["nonzero_terminal_advantage_actions"] += nonzero
                    signal |= nonzero
            if signal:
                signals[environment]["cohorts_with_reward_contrast"].append(cohort)
    for endpoint in ("base", "final"):
        episodes = read(evaluation_run / f"evaluation/{endpoint}/episodes.json")
        require(Counter(e["environment"] for e in episodes) == dict(zip(ENVIRONMENTS, (64, 64, 64, 16, 16))),
                "Evaluation must contain224 episodes per endpoint")
        for episode in episodes:
            validate_episode(episode)
            require(len(episode["learner_seats"]) == 1 and episode["episode_id"] not in identifiers,
                    "Evaluation learner role or episode identity differs")
            identifiers.add(episode["episode_id"])
        datasets[endpoint] = episodes

    result = {split: {e: summarize([episode for episode in episodes if episode["environment"] == e], e)
                      for e in FOCUS} for split, episodes in datasets.items()}
    paired = {}
    final = {e["schedule"]["paired_seed_key"]: e for e in datasets["final"]}
    require(len(final) == 224, "Duplicate final paired key")
    for environment in FOCUS:
        rows = [e for e in datasets["base"] if e["environment"] == environment]
        same_trace = same_return = 0
        for baseline in rows:
            candidate = final[baseline["schedule"]["paired_seed_key"]]
            require({k:v for k,v in baseline["schedule"].items() if k != "episode_id"}
                    == {k:v for k,v in candidate["schedule"].items() if k != "episode_id"}, "Unpaired evaluation scenario")
            trace = lambda e: [(r["turn"], r["player"], r["action"]) for r in e["transitions"]]
            same_trace += trace(baseline) == trace(candidate)
            seat = baseline["learner_seats"][0]
            same_return += baseline["returns"][seat] == candidate["returns"][seat]
        paired[environment] = {"episode_pairs": len(rows), "identical_learner_action_sequences": same_trace,
                               "identical_candidate_returns": same_return}
    opponents = {}
    for environment in ("maze", "chess"):
        opponents[environment] = {}
        for endpoint in ("base", "final"):
            groups = defaultdict(list)
            for episode in datasets[endpoint]:
                if episode["environment"] == environment:
                    groups[episode["schedule"]["opponent"]].append(episode)
            opponents[environment][endpoint] = {}
            for opponent, episodes in groups.items():
                rewards = [e["returns"][e["learner_seats"][0]] for e in episodes]
                opponents[environment][endpoint][opponent] = {
                    "episodes": len(episodes), "mean_return": average(rewards),
                    "positive": sum(x > 0 for x in rewards), "zero": sum(x == 0 for x in rewards),
                    "negative": sum(x < 0 for x in rewards),
                    "actual_goal_or_checkmate_wins": sum(goal_count(e) if environment == "maze" else
                        bool(e["metrics"]["checkmate"] and e["returns"][e["learner_seats"][0]] > 0) for e in episodes)}
    for signal in signals.values():
        signal["cohorts_without_reward_contrast"] = 20 - len(signal["cohorts_with_reward_contrast"])
    require(len(inputs) == 22 and len(datasets["train"]) == 200, "Expected22 source files and200 production episodes")
    # Detect concurrent replacement of any input without running a scientific audit.
    for path, expected in inputs.items():
        require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, "Input changed during read: " + path)
    return {"kind": "saved_scalar_behavior_description_v1", "input_file_sha256": inputs,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "Saved production and completed evaluation JSON only; no game/model/optimizer execution or audit rerun.",
            "episode_counts": {k:len(v) for k,v in datasets.items()}, "behaviors": result,
            "derived_terminal_return_signals": signals, "paired_evaluation_behavior": paired,
            "evaluation_opponent_breakdown": opponents,
            "definitions_and_limits": [
                "Terminal advantages are reconstructed as each role's return minus its counterpart's return in the other task in the two-episode cohort; these are not measured gradients.",
                "An immediate maze reversal is opposite-direction movement on consecutive actions by the same player; it need not always be strategically wrong.",
                "Repeated observed program counts also include test actions that leave code unchanged; they are not a count of destructive edits.",
                "Exact reference equality is string equality in saved pre-action observations. Equivalent correct programs may not match, and reference source was audit-only.",
                "Positive bounded return is distinct from an actual goal or checkmate; capped adjudications remain explicitly labeled.",
                "Different training tasks and self-play opponents prevent interpreting aggregate chronological changes as causal learning trends."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, default=Path("runs/arena-mixed-scalar-training-gpu-v1"))
    parser.add_argument("--evaluation-run", type=Path, default=Path("runs/arena-mixed-scalar-evaluation-gpu-v1"))
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    print(json.dumps(analyze(args.training_run, args.evaluation_run), sort_keys=True,
                     indent=2 if args.pretty else None, separators=None if args.pretty else (",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
