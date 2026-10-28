"""Scientific figure from an already-complete independent holdout audit.

This reporting script never loads a policy, runs games or reruns the audit.
Use an isolated plotting runtime if Matplotlib is absent from the training env.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

GAMES = ("market", "maze", "chess")
OPPONENTS = ("heuristic", "base")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_complete(analysis):
    if not (analysis.get("complete") is True and analysis.get("all_completed_episodes_verified") is True
            and analysis.get("episodes_verified") == analysis.get("scheduled_episodes") == 288
            and not analysis.get("missing_episode_ids")):
        raise ValueError("Plot requires the complete independent 288-game holdout audit")
    cells = {}
    for row in analysis["paired_checkpoint_changes"]:
        if (row["before"], row["after"]) != ("base", "cycle3"):
            continue
        key = (row["environment"], row["opponent"])
        if key in cells:
            raise ValueError("Duplicate environment/opponent cell")
        pairs = sorted(row["paired_scenario_changes"], key=lambda x: x["seed"])
        if row["scenario_clusters"] != 12 or len(pairs) != 12 or len({p["seed"] for p in pairs}) != 12:
            raise ValueError("Each cell requires twelve complete paired scenarios")
        values = [p["mean_return_change"] for p in pairs]
        interval = row["descriptive_bootstrap_95_interval"]
        if not np.isfinite(values).all() or not np.isfinite(interval).all() or len(interval) != 2:
            raise ValueError("Nonfinite/missing audit statistics")
        if not np.isclose(np.mean(values), row["mean_return_change"], atol=1e-12, rtol=0):
            raise ValueError("Audit mean and scenario data disagree")
        cells[key] = {"environment": key[0], "opponent": key[1], "scenario_clusters": 12,
                      "mean": row["mean_return_change"], "interval_95": interval,
                      "paired_scenario_changes": pairs,
                      "interval_source": "Independent audit; 10,000 paired-scenario bootstrap resamples"}
    if set(cells) != {(game, opponent) for game in GAMES for opponent in OPPONENTS}:
        raise ValueError("Missing or unexpected environment/opponent cells")
    return cells


def macro_summary(cells, draws=20000, seed=20261022):
    """Resample twelve scenarios per fixed environment, pairing its opponents.

    The environments use distinct scenario families. Both opponent cells within
    one environment reuse the same scenarios and must move together. Environments
    are fixed strata, not three sampled observations from a larger population.
    """
    rng = np.random.default_rng(seed)
    sampled = np.zeros(draws)
    values_by_game, cluster_mapping = {}, {}
    for game in GAMES:
        by_opponent = {opponent: {p["seed"]: p["mean_return_change"] for p in
                          cells[(game, opponent)]["paired_scenario_changes"]} for opponent in OPPONENTS}
        raw_seeds = sorted(by_opponent["base"])
        if raw_seeds != sorted(by_opponent["heuristic"]):
            raise ValueError("Opponent cells do not share the same scenario clusters")
        identities = [s // 2 if game == "chess" else s for s in raw_seeds]
        if len(set(identities)) != 12:
            raise ValueError("Raw seeds collapse onto fewer than twelve actual scenario families")
        # Average opponents BEFORE resampling so their shared maps stay paired.
        values = np.array([np.mean([by_opponent[o][s] for o in OPPONENTS]) for s in raw_seeds])
        sampled += rng.choice(values, size=(draws, 12), replace=True).mean(axis=1) / len(GAMES)
        values_by_game[game] = values.tolist()
        cluster_mapping[game] = [{"raw_seed": s, "scenario_family": identity,
                                  "opponents_kept_together": list(OPPONENTS)}
                                 for s, identity in zip(raw_seeds, identities)]
    return {"mean": float(np.mean([cell["mean"] for cell in cells.values()])),
            "interval_95": np.percentile(sampled, [2.5, 97.5]).tolist(),
            "scenario_clusters": 36, "clusters_per_environment": 12,
            "bootstrap_draws": draws, "bootstrap_seed": seed,
            "method": "Stratified paired-scenario bootstrap; opponents averaged within each scenario, independent resampling within each of three fixed environments, equal environment weight",
            "opponent_averaged_scenario_changes": values_by_game, "cluster_mapping": cluster_mapping}


def render(analysis_path, prefix):
    analysis_path, prefix = Path(analysis_path), Path(prefix)
    analysis = json.loads(analysis_path.read_text())
    cells = extract_complete(analysis)
    manifest = analysis_path.parent / "manifest.json"
    if not manifest.exists() or sha(manifest) != analysis["manifest_sha256"]:
        raise ValueError("Analysis is not bound to this run's manifest")
    macro = macro_summary(cells)
    report_macro = analysis["macro_fixed_opponent_return"]
    expected = report_macro["cycle3"]["mean_return"] - report_macro["base"]["mean_return"]
    if not np.isclose(macro["mean"], expected, atol=1e-12, rtol=0):
        raise ValueError("Equal-cell macro disagrees with the independent audit")
    targets = {suffix: prefix.with_suffix(suffix) for suffix in (".png", ".svg", ".json")}
    if any(path.exists() for path in targets.values()):
        raise FileExistsError("Use a new reporting prefix; existing figures are retained")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "axes.unicode_minus": True})
    colors = {"market": "#2869A9", "maze": "#B9781D", "chess": "#247E75"}
    fig, ax = plt.subplots(figsize=(11.5, 7.7), facecolor="white")
    fig.subplots_adjust(left=.265, right=.87, bottom=.28, top=.805)
    ys = [7.0, 6.0, 4.6, 3.6, 2.2, 1.2]
    labels = []
    extent = []
    for y, key in zip(ys, [(g, o) for g in GAMES for o in OPPONENTS]):
        cell = cells[key]
        values = np.array([p["mean_return_change"] for p in cell["paired_scenario_changes"]])
        # Deterministic vertical offsets expose individual clusters without
        # suggesting another random variable or resampling data for display.
        offsets = np.linspace(-.15, .15, len(values))
        ax.scatter(values, y + offsets, s=28, color=colors[key[0]], alpha=.30,
                   edgecolors="none", zorder=2)
        lo, hi = cell["interval_95"]
        ax.plot([lo, hi], [y, y], color=colors[key[0]], linewidth=2.7, zorder=4)
        ax.plot([lo, lo], [y-.065, y+.065], color=colors[key[0]], linewidth=1.7, zorder=4)
        ax.plot([hi, hi], [y-.065, y+.065], color=colors[key[0]], linewidth=1.7, zorder=4)
        ax.scatter([cell["mean"]], [y], marker="o", s=65, color=colors[key[0]],
                   edgecolors="white", linewidths=.8, zorder=5)
        opponent = "heuristic" if key[1] == "heuristic" else "frozen base"
        labels.append(key[0].title() + " / " + opponent)
        ax.text(1.035, y, f"{cell['mean']:+.3f}", transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=11, color="#20252B")
        extent.extend([*values, lo, hi])
    macro_y = -.45
    ax.plot(macro["interval_95"], [macro_y, macro_y], color="#20252B", linewidth=2.7, zorder=4)
    ax.scatter([macro["mean"]], [macro_y], marker="D", s=65, color="#20252B", zorder=5)
    ax.text(1.035, macro_y, f"{macro['mean']:+.3f}", transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=11, fontweight="bold")
    extent.extend(macro["interval_95"])
    radius = max(.2, max(abs(x) for x in extent) * 1.13)
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-1.1, 7.65)
    ax.axvline(0, color="#56616C", linewidth=1, linestyle=(0, (3, 3)), zorder=1)
    for y in (5.3, 2.9, .35):
        ax.axhline(y, color="#E3E8ED", linewidth=.8, zorder=0)
    ax.set_yticks(ys + [macro_y], labels + ["Macro / equal cell weights"])
    ax.tick_params(axis="y", length=0, pad=13)
    ax.tick_params(axis="x", labelsize=10, colors="#4A535D")
    ax.set_xlabel("Change in environment return (cycle3 − base)", fontsize=11, labelpad=10)
    ax.text(1.035, 1.045, "Mean Δ", transform=ax.transAxes, fontsize=10, color="#4A535D")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#BAC3CC")
    ax.grid(axis="x", color="#E9EDF1", linewidth=.65, zorder=0)

    fig.text(.065, .943, "Self-play RL: final held-out evaluation", fontsize=21,
             fontweight="bold", color="#17212B")
    fig.text(.065, .897, "Qwen3-8B · fixed final cycle3 versus base · 288 verified games",
             fontsize=12, color="#4A535D")
    handles = [Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#6C8295",
                      markeredgecolor="none", markersize=5, alpha=.45,
                      label="12 paired scenario means per cell"),
               Line2D([0], [0], marker="o", color="#33475A", markersize=5, linewidth=2,
                      label="Mean and descriptive 95% bootstrap interval")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.059, .866),
               frameon=False, ncol=2, fontsize=10, handlelength=2.4, columnspacing=2.0)
    fig.text(.065, .145,
             "Each dot averages the two seat swaps. Cell intervals come from the completed independent audit.",
             fontsize=9.5, color="#4A535D")
    fig.text(.065, .113,
             "Macro resampling keeps both opponents paired within each scenario; environments are fixed strata.",
             fontsize=9.5, color="#4A535D")
    fig.text(.065, .081,
             "One training seed; six exploratory cells; no multiplicity adjustment or significance claim.",
             fontsize=9.5, color="#4A535D")
    fig.text(.065, .049,
             "Return scales differ by game. Capped maze/chess rewards use distance/material adjudication.",
             fontsize=9.5, color="#4A535D")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(targets[".png"], dpi=220, facecolor="white")
    fig.savefig(targets[".svg"], facecolor="white", metadata={"Title": "Self-play RL final holdout",
                "Description": "Paired final-checkpoint minus base return changes; exploratory cluster intervals."})
    plt.close(fig)
    output = {"kind": "completed_holdout_reporting_figure", "analysis": str(analysis_path.resolve()),
              "analysis_sha256": sha(analysis_path), "manifest_sha256": analysis["manifest_sha256"],
              "reporting_source_sha256": sha(__file__), "data_source_complete": True,
              "versions": {"matplotlib": matplotlib.__version__, "numpy": np.__version__},
              "cells": list(cells.values()), "macro": macro,
              "interpretation": "Descriptive exploratory uncertainty conditional on one training run; not a significance claim or evidence of general reasoning transfer.",
              "outputs": {str(path.resolve()): sha(path) for suffix, path in targets.items() if suffix != ".json"}}
    targets[".json"].write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    return {"png": str(targets[".png"].resolve()), "svg": str(targets[".svg"].resolve()),
            "provenance": str(targets[".json"].resolve()), "macro_mean": macro["mean"],
            "macro_descriptive_interval_95": macro["interval_95"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.analysis, args.out_prefix), indent=2))


if __name__ == "__main__":
    main()
