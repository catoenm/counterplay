"""Describe preserved scalar market traces; stdlib only, stdout only.

This is an offline descriptive calculation, not a simulator replay or integrity
audit. It does not load models, import experiment code, or alter saved evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = "eval-market-4-1-heuristic"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def mean(values):
    return math.fsum(values) / len(values)


def read_markets(path, hashes, expected, seats):
    path = Path(path).resolve()
    raw = path.read_bytes()
    hashes[str(path)] = hashlib.sha256(raw).hexdigest()
    episodes = [e for e in json.loads(raw) if e["environment"] == "market"]
    require(len(episodes) == expected, "Unexpected market episode denominator: " + str(path))
    require(len({e["episode_id"] for e in episodes}) == expected, "Duplicate market episode ID")
    for e in episodes:
        players = e["learner_seats"]
        require(players == [0, 1] if seats == 2 else players in ([0], [1]), "Wrong learner roles")
        require(e["metrics"]["terminal"] and e["metrics"]["rounds"] == 16, "Incomplete market episode")
        require(len(e["transitions"]) == 16 * seats, "Incomplete learner action history")
        require({(r["player"], r["turn"]) for r in e["transitions"]}
                == {(p, t) for p in players for t in range(16)}, "Duplicate/missing player-turn")
        require(all(math.isfinite(v) for v in e["returns"]), "Nonfinite terminal return")
        require(e["returns"] == e["metrics"]["returns"], "Saved terminal return fields differ")
        require(all(type(v) is int for v in e["metrics"]["absolute_pnl"]), "Expected integer native P&L")
    return episodes


def summarize(episodes):
    rows = [r for e in episodes for r in e["transitions"]]
    pnls = [e["metrics"]["absolute_pnl"][p] for e in episodes for p in e["learner_seats"]]
    returns = [e["returns"][p] for e in episodes for p in e["learner_seats"]]
    actions = Counter(r["action"].get("side", r["action"]["type"]) for r in rows)
    limits = [r for r in rows if r["action"]["type"] == "limit"]
    negative_signal = sum((r["observation"]["private_signal"] - r["action"]["price"])
                          * (1 if r["action"]["side"] == "buy" else -1) < 0 for r in limits)
    return {
        "episodes": len(episodes), "learner_trajectories": len(pnls), "learner_decisions": len(rows),
        "absolute_pnl": {"mean": mean(pnls), "total": sum(pnls), "min": min(pnls), "max": max(pnls),
                         "positive": sum(v > 0 for v in pnls), "zero": pnls.count(0),
                         "negative": sum(v < 0 for v in pnls)},
        "relative_return": {"mean": mean(returns), "mean_absolute": mean([abs(v) for v in returns]),
                            "min": min(returns), "max": max(returns)},
        "both_agents_lose_episodes": sum(max(e["metrics"]["absolute_pnl"]) < 0 for e in episodes),
        "relative_winner_with_negative_absolute_pnl": sum(
            e["returns"][p] > 0 and e["metrics"]["absolute_pnl"][p] < 0
            for e in episodes for p in e["learner_seats"]),
        "action_counts": dict(sorted(actions.items())), "limit_orders": len(limits),
        "limit_orders_worse_than_own_private_signal": negative_signal,
        "signal_edge_caveat": "Limit price versus noisy private signal; not realized fill profitability.",
    }


def trading(episodes):
    units = Counter(); pnl = Counter(); participant_units = 0
    for e in episodes:
        seats = e["learner_seats"]
        value = e["metrics"]["terminal_value"]
        by_player = Counter(); filled = Counter()
        for trade in e["events"][-1]["state"]["trades"]:
            buyer, seller, quantity = trade["buyer"], trade["seller"], trade["quantity"]
            if not (set(seats) & {buyer, seller}):
                continue
            category = "between_agents" if buyer in (0, 1) and seller in (0, 1) else "agent_background"
            units[category] += quantity  # Each contract once, even when both agents are learners.
            for player in seats:
                if player not in (buyer, seller):
                    continue
                profit = quantity * (value - trade["price"]) * (1 if player == buyer else -1)
                pnl[category] += profit
                by_player[player] += profit
                filled[player] += quantity
                participant_units += quantity
        for player in seats:
            require(by_player[player] - e["metrics"]["fees"][player] == e["metrics"]["absolute_pnl"][player],
                    "Complete saved trades do not reconcile with native P&L")
            require(filled[player] == e["metrics"]["filled_units"][player], "Filled volume differs")
    return {"contract_units": dict(sorted(units.items())), "learner_filled_units": participant_units,
            "learner_gross_pnl_by_counterparty": dict(sorted(pnl.items())),
            "learner_fees": sum(e["metrics"]["fees"][p] for e in episodes for p in e["learner_seats"]),
            "volume_convention": "Contracts count once; a learner-to-learner contract contributes two learner-filled units."}


def analyze(training_run, evaluation_run):
    hashes = {}; cohorts = []; train = []
    for i in range(1, 21):
        episodes = read_markets(Path(training_run) / f"cohorts/{i}/episodes.json", hashes, 2, 2)
        cohorts.append(episodes); train.extend(episodes)
    advantage_pairs = [(e["returns"][p] - other["returns"][p], e["metrics"]["absolute_pnl"][p])
                       for pair in cohorts for e, other in (pair, pair[::-1]) for p in (0, 1)]
    magnitudes = [abs(pair[0]["returns"][0] - pair[1]["returns"][0]) for pair in cohorts]
    base = read_markets(Path(evaluation_run) / "evaluation/base/episodes.json", hashes, 64, 1)
    final = read_markets(Path(evaluation_run) / "evaluation/final/episodes.json", hashes, 64, 1)
    bmap = {e["schedule"]["paired_seed_key"]: e for e in base}
    require(len(bmap) == 64 and set(bmap) == {e["schedule"]["paired_seed_key"] for e in final},
            "Evaluation endpoints lack exact paired keys")
    counts = Counter(); matched_tv = []; matched_kl = []; changed = []; example = None
    for f in final:
        b = bmap[f["schedule"]["paired_seed_key"]]; player = f["learner_seats"][0]
        require(b["seed"] == f["seed"] and b["learner_seats"] == f["learner_seats"]
                and b["schedule"]["opponent"] == f["schedule"]["opponent"], "Paired scenario/role differs")
        require(b["scenario_family_id"] == f["scenario_family_id"], "Paired family differs")
        pairs = list(zip(b["transitions"], f["transitions"]))
        difference = sum(x["action"] != y["action"] for x, y in pairs)
        counts["unchanged_action_sequences" if not difference else "changed_action_sequences"] += 1
        counts["changed_action_decisions"] += difference
        delta = f["returns"][player] - b["returns"][player]
        counts["return_improved" if delta > 0 else "return_worsened" if delta < 0 else "return_unchanged"] += 1
        if difference:
            changed.append(f["episode_id"])
        for x, y in pairs:
            require(x["player"] == y["player"] and x["turn"] == y["turn"], "Paired action ordering differs")
            if x["observation"] == y["observation"] and x["actions"] == y["actions"]:
                require(x["sample_seed"] == y["sample_seed"], "Common-state sampling seeds differ")
                a, z = x["old_log_probs"], y["old_log_probs"]
                require(len(a) == len(z) == len(x["actions"])
                        and all(math.isfinite(v) for v in a + z), "Malformed full action vector")
                matched_tv.append(math.fsum(abs(math.exp(u) - math.exp(v)) for u, v in zip(a, z)) / 2)
                matched_kl.append(math.fsum(math.exp(u) * (u - v) for u, v in zip(a, z)))
        if f["episode_id"] == "final-" + EXAMPLE:
            x, y = next((x, y) for x, y in pairs if x["action"] != y["action"])
            example = {"base_id": b["episode_id"], "final_id": f["episode_id"], "seed": f["seed"],
                       "learner_seat": player, "opponent": f["schedule"]["opponent"], "first_changed_turn": x["turn"],
                       "same_observation": x["observation"] == y["observation"],
                       "private_signal": x["observation"]["private_signal"], "terminal_value": f["metrics"]["terminal_value"],
                       "base_action": x["action"], "final_action": y["action"],
                       "absolute_pnl": {"base": b["metrics"]["absolute_pnl"][player], "final": f["metrics"]["absolute_pnl"][player]},
                       "relative_return": {"base": b["returns"][player], "final": f["returns"][player]},
                       "final_action_probability": {"base": math.exp(x["old_log_probs"][x["actions"].index(y["action"])]),
                                                    "final": math.exp(y["old_log_prob"])}}
    require(example is not None, "Prespecified illustrative example is missing")
    return {"kind": "saved_scalar_market_descriptive_analysis_v1", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "input_sha256": hashes, "training": {**summarize(train), "trading": trading(train),
                "cohorts": 20, "market_episodes_per_cohort": 2,
                "loo_advantage": {"nonzero_cohorts": sum(v != 0 for v in magnitudes), "min_absolute": min(magnitudes),
                    "median_absolute": statistics.median(magnitudes), "max_absolute": max(magnitudes),
                    "positive_trajectories": sum(a > 0 for a, _ in advantage_pairs),
                    "positive_but_negative_absolute_pnl": sum(a > 0 and p < 0 for a, p in advantage_pairs),
                    "definition": "Same-seat terminal return minus the other market episode's return; shared by all 16 trajectory actions."}},
            "evaluation": {label: {**summarize(es), "trading": trading(es),
                "by_opponent": {op: summarize([e for e in es if e["schedule"]["opponent"] == op]) for op in ("base", "heuristic")}}
                for label, es in (("base", base), ("final", final))},
            "paired_evaluation": {**dict(counts), "episodes": 64, "changed_sequence_ids": changed,
                "identical_state_menu_rows": len(matched_tv), "mean_total_variation": mean(matched_tv),
                "max_total_variation": max(matched_tv), "mean_kl_base_to_final": mean(matched_kl), "common_sampling_seeds_verified": True},
            "illustrative_example": example,
            "limitations": ["Descriptive saved-trace analysis, not an audit rerun or causal experiment.",
                "All-player self-play mean relative return is zero by construction; absolute P&L is separate.",
                "Paired episodes are clustered within 16 scenarios; episode counts are not 64 independent scenarios.",
                "Example selected after inspecting outcomes; it is illustrative, not an independent test."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, default=ROOT / "runs/arena-mixed-scalar-training-gpu-v1")
    parser.add_argument("--evaluation-run", type=Path, default=ROOT / "runs/arena-mixed-scalar-evaluation-gpu-v1")
    args = parser.parse_args()
    print(json.dumps(analyze(args.training_run, args.evaluation_run), sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
