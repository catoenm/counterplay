"""Offline diagnostics from committed scalar traces; never loads a model or runs an audit.

Usage: PYTHONDONTWRITEBYTECODE=1 python3 scripts/analyze_saved_scalar_training.py
Prints JSON to stdout. Redirect only to a new analysis artifact, never into a frozen run.
"""
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "runs/arena-mixed-scalar-training-gpu-v1"
EVAL = ROOT / "runs/arena-mixed-scalar-evaluation-gpu-v1"
sources = {}


def read(path):
    data = path.read_bytes()
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def describe(values):
    return {"min": min(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "max": max(values)}


manifest = read(TRAIN / "manifest.json")
config = manifest["config"]["policy"]
environments = config["environment_weights"]
per_env = {env: {"episodes": 0, "actions": 0, "nonzero_advantage_actions": 0,
                 "cohorts_with_reward_contrast": [], "action_weight_mass": 0.0,
                 "weighted_abs_advantage": 0.0, "weighted_entropy_sum": 0.0,
                 "learner_role_returns": [], "advantages": [], "entropy": [],
                 "normalized_entropy": [], "episode_actions": [],
                 "action_logit_gradient_reward_squared": 0.0,
                 "action_logit_gradient_entropy_squared": 0.0,
                 "action_logit_gradient_kl_squared": 0.0}
           for env in environments}
cohorts, ratios, tvs, kls = [], [], [], []
for cohort in range(1, 21):
    folder = TRAIN / "cohorts" / str(cohort)
    episodes = read(folder / "episodes.json")
    report = read(folder / "update.json")["report"]
    raw = [row for episode in episodes for row in episode["transitions"]]
    assert len(raw) == len(report["learner_records"])
    returns = defaultdict(list)
    counts = Counter(episode["environment"] for episode in episodes)
    for episode in episodes:
        for seat in episode["learner_seats"]:
            returns[episode["environment"], seat].append((episode["episode_id"], episode["returns"][seat]))
    offset = 0
    contrasts = set()
    for episode in episodes:
        env = episode["environment"]
        dst = per_env[env]
        dst["episodes"] += 1
        dst["episode_actions"].append(len(episode["transitions"]))
        for seat in episode["learner_seats"]:
            dst["learner_role_returns"].append(episode["returns"][seat])
        for row in episode["transitions"]:
            record = report["learner_records"][offset]
            offset += 1
            peers = [value for eid, value in returns[env, row["player"]] if eid != episode["episode_id"]]
            advantage = episode["returns"][row["player"]] - statistics.mean(peers)
            weight = environments[env] / counts[env] / len(episode["learner_seats"])
            assert abs(advantage - record["advantage"]) < 1e-12
            assert abs(weight - record["weight"]) < 1e-12
            old_logp, logp = row["old_log_probs"], record["learner_log_probs"]
            old_p, p = [math.exp(x) for x in old_logp], [math.exp(x) for x in logp]
            action = row["action_index"]
            ratio = math.exp(logp[action] - old_logp[action])
            entropy = -sum(prob * log for prob, log in zip(p, logp))
            kl = sum(prob * (old - new) for prob, old, new in zip(old_p, old_logp, logp))
            tv = sum(abs(a-b) for a, b in zip(p, old_p)) / 2
            ratios.append(ratio); tvs.append(tv); kls.append(kl)
            dst["actions"] += 1
            dst["nonzero_advantage_actions"] += int(abs(advantage) > 1e-12)
            if abs(advantage) > 1e-12:
                contrasts.add(env)
            dst["advantages"].append(advantage)
            dst["action_weight_mass"] += weight
            dst["weighted_abs_advantage"] += weight * abs(advantage)
            dst["weighted_entropy_sum"] += weight * entropy
            dst["entropy"].append(entropy)
            dst["normalized_entropy"].append(entropy / math.log(len(p)) if len(p) > 1 else 0.0)
            # Local derivatives w.r.t. independent action logits, NOT shared model gradients.
            clipped = (advantage > 0 and ratio > 1 + config["clip_ratio"]) or (advantage < 0 and ratio < 1 - config["clip_ratio"])
            for j, prob in enumerate(p):
                reward_grad = 0.0 if clipped else -weight * advantage * ratio * ((j == action) - prob)
                entropy_grad = weight * config["entropy_coefficient"] * prob * (logp[j] + entropy)
                kl_grad = weight * config["kl_coefficient"] * (prob - old_p[j])
                dst["action_logit_gradient_reward_squared"] += reward_grad ** 2
                dst["action_logit_gradient_entropy_squared"] += entropy_grad ** 2
                dst["action_logit_gradient_kl_squared"] += kl_grad ** 2
    for env in contrasts:
        per_env[env]["cohorts_with_reward_contrast"].append(cohort)
    keys = ["transitions", "nonzero_advantage_actions", "policy_loss", "entropy_loss", "kl_loss",
            "gradient_norm_before_clip", "parameter_delta_l2", "elapsed_seconds", "policy_lag",
            "weighted_entropy", "weighted_kl", "weighted_clip_fraction", "optimizer_steps"]
    cohorts.append({"cohort": cohort, **{key: report[key] for key in keys}})

total_weight = sum(dst["action_weight_mass"] for dst in per_env.values())
for dst in per_env.values():
    dst["zero_advantage_actions"] = dst["actions"] - dst["nonzero_advantage_actions"]
    dst["nonzero_advantage_action_fraction"] = dst["nonzero_advantage_actions"] / dst["actions"]
    dst["cohorts_without_reward_contrast"] = 20 - len(dst["cohorts_with_reward_contrast"])
    dst["action_weight_mass_share"] = dst["action_weight_mass"] / total_weight
    for key in ["learner_role_returns", "advantages", "entropy", "normalized_entropy", "episode_actions"]:
        values = dst.pop(key)
        dst[key + "_summary"] = describe(values)
        if key == "learner_role_returns":
            dst["learner_role_return_counts"] = dict(Counter(map(str, values)))
    for term in ["reward", "entropy", "kl"]:
        dst["action_logit_gradient_" + term + "_l2"] = math.sqrt(dst.pop("action_logit_gradient_" + term + "_squared"))

summary = read(EVAL / "scientific-summary-v1.json")
read(TRAIN / "training-complete.json")
read(TRAIN / "remote-exit.json")
read(EVAL / "remote-exit.json")
budget = read(ROOT / "budget.json")
for path in sorted((ROOT / "runs").glob("*ledger.json")):
    read(path)
for path in [TRAIN / "source/selfplay/arena_mixed/cuda_policy.py", TRAIN / "source/selfplay/arena_mixed_scalar/cuda_policy.py"]:
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()

output = {
    "kind": "offline_saved_scalar_training_diagnosis_v1",
    "scope": "Twenty committed production cohorts only; calibration excluded. No new forward, optimizer, simulator, provider, or final-audit call.",
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "sources_sha256": sources,
    "policy_config": config,
    "totals": {
        "episodes": sum(dst["episodes"] for dst in per_env.values()),
        "actions": sum(dst["actions"] for dst in per_env.values()),
        "nonzero_advantage_actions": sum(dst["nonzero_advantage_actions"] for dst in per_env.values()),
        "optimizer_steps": sum(row["optimizer_steps"] for row in cohorts),
        "gradient_norm_exceeded_clip": sum(row["gradient_norm_before_clip"] > config["gradient_clip"] for row in cohorts),
        "update_seconds_sum": sum(row["elapsed_seconds"] for row in cohorts),
        "gradient_norm_before_clip": describe([row["gradient_norm_before_clip"] for row in cohorts]),
        "parameter_delta_l2_per_step": describe([row["parameter_delta_l2"] for row in cohorts]),
        "action_weight_mass": total_weight,
    },
    "policy_motion_at_saved_preupdate_rows": {
        "selected_action_ratio": describe(ratios), "total_variation": describe(tvs),
        "kl_old_to_learner": describe(kls),
        "ratio_outside_0_8_to_1_2": sum(abs(ratio - 1) > 0.2 for ratio in ratios),
        "interpretation": "Learner versus rollout behavior on that cohort, usually one version apart; not base-to-final movement. Tiny negative KL values can arise from saved float32 normalization.",
    },
    "per_environment": per_env,
    "cohorts": cohorts,
    "limitations": [
        "This is descriptive diagnosis, not a rerun of the terminal audit or causal attribution.",
        "Zero advantage means zero reward-surrogate derivative for those rows, not no entropy/KL/shared-model update.",
        "Action weight mass and independent action-logit derivatives are not shares or norms of model-parameter gradients. Jacobians, cancellation and Adam matter.",
        "Summing per-action terms is the declared episodic objective; its duration dependence is not itself proof of a bug.",
        "The scalar policy-loss value may cancel at ratio one despite nonzero policy gradients; loss magnitude does not establish entropy domination.",
        "A baseline from another scenario adds difficulty variation; saved data do not establish how much variance it caused.",
        "No new experiment is authorized or launched by this analysis.",
    ],
}
print(json.dumps(output, indent=2, sort_keys=True))
