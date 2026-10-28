"""Plot only a completed independent matched-program-study audit.

This reporting module reads audited aggregates. It never loads a model, executes
a controller, scores an incomplete evaluation, or changes experimental files.
Matplotlib is imported only for rendering; use the isolated reporting runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile

ENDPOINTS = ("base", "game_rl-round3", "validity_rl-round3")
CONDITIONS = {"compact": 8, "plain": 4}
LABELS = {"base": "Frozen base", "game_rl-round3": "Game RL / round 3",
          "validity_rl-round3": "Validity RL / round 3"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bind_audit_inputs(analysis_path):
    """Bind the exact parsed audit bytes and the run state that it verified."""
    analysis_path = Path(analysis_path)
    audit_bytes = analysis_path.read_bytes()
    analysis = json.loads(audit_bytes)
    run = Path(analysis["run"])
    bindings = {"analysis": {"path": str(analysis_path.resolve()),
                             "sha256": hashlib.sha256(audit_bytes).hexdigest()}}
    for name in ("manifest", "status"):
        path = run / f"{name}.json"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == analysis[f"{name}_sha256"], f"Audit {name} binding differs")
        bindings[name] = {"path": str(path.resolve()), "sha256": actual}
    return analysis, bindings


def verify_bound_inputs(bindings):
    for name, record in bindings.items():
        require(sha(record["path"]) == record["sha256"],
                f"Bound {name} changed during rendering; no reporting output committed")


def checked_cell(cell, endpoint, condition, split, n):
    require(cell.get("complete") is True and cell.get("prescribed_program_clusters") == n
            and cell.get("observed_program_clusters") == n and cell.get("missing_program_clusters") == 0,
            "Missing program draws are not zero yield; require the full prescribed denominator")
    programs = cell.get("programs", [])
    names = [f"eval-{endpoint}-{condition}-{i}" for i in range(n)]
    require(len(programs) == n and [p["generation"] for p in programs] == names,
            "Program identity/order does not match the paired generation schedule")
    result = []
    expected_games = 48 if split == "holdout" else 32
    for p in programs:
        value = p.get("banked_point_yield")
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                "Missing/nonfinite banked-point yield is not a zero")
        authored, valid = p.get("authorship_valid"), p.get("runtime_valid")
        require(type(authored) is bool and type(valid) is bool, "Missing program validity classification")
        faults = p.get("candidate_program_faults")
        require(type(faults) is int and faults >= 0, "Missing candidate fault count")
        if not authored:
            require(not valid and value == 0 and p.get("games") == 0 and faults == 0,
                    "Invalid authoring must retain zero yield with no fabricated game")
            category = "invalid_authorship"
        else:
            require(p.get("games") == expected_games and valid == (faults == 0),
                    "Compiled program lacks the complete game grid or consistent runtime validity")
            category = "runtime_fault" if not valid else "valid_zero" if value == 0 else "valid_positive"
        result.append({"generation": p["generation"], "banked_point_yield": value,
                       "category": category, "authorship_valid": authored, "runtime_valid": valid,
                       "candidate_program_faults": faults, "games": p["games"]})
    mean = math.fsum(p["banked_point_yield"] for p in result) / n
    require(math.isclose(mean, cell["primary_mean_banked_point_yield"], abs_tol=1e-12, rel_tol=0),
            "Audited mean excludes or misweights a prescribed draw")
    counts = {category: sum(p["category"] == category for p in result)
              for category in ("invalid_authorship", "runtime_fault", "valid_zero", "valid_positive")}
    require(cell["compiled_programs"] == n - counts["invalid_authorship"]
            and cell["invalid_no_game_programs"] == counts["invalid_authorship"]
            and cell["fault_free_programs"] == counts["valid_zero"] + counts["valid_positive"],
            "Audited validity counts disagree with individual program records")
    return {"endpoint": endpoint, "condition": condition, "programs": result, "n": n,
            "mean": mean, "counts": counts,
            "runtime_valid_rate": (counts["valid_zero"] + counts["valid_positive"]) / n}


def extract_complete(analysis):
    require(analysis.get("kind") == "independent_cpu_matched_program_study_audit"
            and analysis.get("integrity_passed") is True
            and analysis.get("planned_study_complete") is True
            and analysis.get("planned_comparisons_available") is True
            and analysis.get("evaluation_draws") == analysis.get("required_evaluation_draws") == 36,
            "Rendering requires the complete independent audit and all 36 evaluation outputs")
    evaluation = analysis["evaluation"]
    expected = {f"{e}/{c}/{s}" for e in ENDPOINTS for c in CONDITIONS for s in ("validation", "holdout")}
    require(set(evaluation) == expected, "All twelve completed evaluation cells are required")
    cells, seed_sets = {}, {}
    for condition, n in CONDITIONS.items():
        for endpoint in ENDPOINTS:
            # Both splits must be complete even though only holdout is plotted.
            for split in ("validation", "holdout"):
                checked = checked_cell(evaluation[f"{endpoint}/{condition}/{split}"],
                                       endpoint, condition, split, n)
                if split == "holdout":
                    cells[(condition, endpoint)] = checked
        for reference in ("base", "validity_rl-round3"):
            contrast = analysis["contrasts"][f"game_rl-minus-{reference}/{condition}"]
            seeds = contrast["paired_generation_seeds"]
            require(contrast["program_cluster_n"] == n and len(seeds) == len(set(seeds)) == n
                    and all(type(s) is int for s in seeds), "Invalid paired generation seeds")
            if condition in seed_sets:
                require(seed_sets[condition] == seeds, "Contrasts use different paired generation seeds")
            seed_sets[condition] = seeds
            game, control = cells[(condition, "game_rl-round3")], cells[(condition, reference)]
            diffs = [a["banked_point_yield"] - b["banked_point_yield"]
                     for a, b in zip(game["programs"], control["programs"])]
            recorded = contrast["raw_paired_banked_yield_differences"]
            require(len(recorded) == n and all(math.isclose(a, b, abs_tol=1e-12, rel_tol=0)
                    for a, b in zip(diffs, recorded)), "Paired differences do not match program identities")
    require(not set(seed_sets["compact"]) & set(seed_sets["plain"]), "Prompt conditions reuse generation seeds")
    for (condition, _), cell in cells.items():
        for p, seed in zip(cell["programs"], seed_sets[condition]):
            p["generation_seed"] = seed
    return cells


def render(analysis_path, prefix):
    analysis_path, prefix = Path(analysis_path), Path(prefix)
    analysis, bindings = bind_audit_inputs(analysis_path)
    cells = extract_complete(analysis)
    targets = {suffix: prefix.with_suffix(suffix) for suffix in (".png", ".svg", ".json")}
    if any(p.exists() for p in targets.values()):
        raise FileExistsError("Existing reporting artifacts are retained; choose a new prefix")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    colors = {"base": "#657482", "game_rl-round3": "#197DA1", "validity_rl-round3": "#8B5795"}
    validity_colors = {"invalid_authorship": "#E7A3A3", "runtime_fault": "#D9B36A", "runtime_valid": "#69A793"}
    offsets = (-.19, 0, .19)
    maximum = max(1.0, max(p["banked_point_yield"] for c in cells.values() for p in c["programs"]))
    fig = plt.figure(figsize=(14, 10.5), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=(3.2, 1), left=.125, right=.96,
                           top=.80, bottom=.22, hspace=.40, wspace=.40)
    for column, (condition, n) in enumerate(CONDITIONS.items()):
        ax = fig.add_subplot(grid[0, column])
        seeds = [p["generation_seed"] for p in cells[(condition, "base")]["programs"]]
        for i in range(n):
            y = n - 1 - i
            xs = [cells[(condition, e)]["programs"][i]["banked_point_yield"] for e in ENDPOINTS]
            ax.plot(xs, [y + o for o in offsets], color="#CBD2D8", linewidth=.8, zorder=1)
            for endpoint, offset in zip(ENDPOINTS, offsets):
                p = cells[(condition, endpoint)]["programs"][i]
                category = p["category"]
                marker = "X" if category == "invalid_authorship" else "^" if category == "runtime_fault" else "o"
                face = "white" if category == "valid_zero" else colors[endpoint]
                ax.scatter([p["banked_point_yield"]], [y + offset], marker=marker, s=48,
                           facecolors=face, edgecolors=colors[endpoint], linewidths=1.2, zorder=3)
        mean_y = -1.25
        ax.axhline(-.65, color="#DAE1E7", linewidth=.8)
        for endpoint, offset in zip(ENDPOINTS, offsets):
            ax.scatter([cells[(condition, endpoint)]["mean"]], [mean_y + offset], marker="D", s=45,
                       color=colors[endpoint], zorder=3)
        title = f"Compact prompt · primary (n = {n})" if condition == "compact" else f"Plain prompt · secondary (n = {n})"
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=14)
        ax.set_yticks([n - 1 - i for i in range(n)] + [mean_y],
                     [str(seed) for seed in seeds] + [f"Mean / all {n} draws"])
        ax.set_ylim(-1.8, n - .4)
        ax.set_xlim(-maximum * .045, maximum * 1.09)
        ax.set_xlabel("Mean banked points per held-out game", labelpad=10)
        if all(p["banked_point_yield"] == 0 for e in ENDPOINTS
               for p in cells[(condition, e)]["programs"]):
            ax.text(.55, .54, f"All {n} prescribed draws yielded zero\nfor every checkpoint",
                    transform=ax.transAxes, ha="center", va="center", fontsize=11,
                    color="#657482", linespacing=1.5)
        if column == 0:
            ax.set_ylabel("Paired program-generation seed", labelpad=9)
        ax.grid(axis="x", color="#E8EDF1", linewidth=.7, zorder=0)
        ax.tick_params(axis="y", length=0, labelsize=9)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#B4BFC8")

        bars = fig.add_subplot(grid[1, column])
        for j, endpoint in enumerate(ENDPOINTS):
            cell = cells[(condition, endpoint)]
            counts = cell["counts"]
            components = [("invalid_authorship", counts["invalid_authorship"]),
                          ("runtime_fault", counts["runtime_fault"]),
                          ("runtime_valid", counts["valid_zero"] + counts["valid_positive"])]
            left = 0
            for category, count in components:
                bars.barh(2-j, count, left=left, height=.55, color=validity_colors[category])
                if count:
                    bars.text(left + count/2, 2-j, str(count), ha="center", va="center", fontsize=9)
                left += count
            bars.text(n + .18, 2-j, f"{components[-1][1]}/{n} fault-free", va="center", fontsize=9)
        bars.set_yticks((2, 1, 0), [LABELS[e] for e in ENDPOINTS], fontsize=9)
        bars.set_xlim(0, n * 1.40)
        bars.set_xticks((0, n/2, n), ("0", str(n//2), str(n)))
        bars.set_title("Program validity over the complete held-out grid", loc="left", fontsize=10, pad=9)
        bars.set_xlabel("Generated programs (all prescribed draws)", fontsize=9)
        bars.tick_params(axis="y", length=0)
        for side in ("top", "right", "left"):
            bars.spines[side].set_visible(False)
        bars.spines["bottom"].set_color("#B4BFC8")

    fig.text(.055, .956, "Coding RL: final held-out program yield", fontsize=22, fontweight="bold", color="#17212B")
    fig.text(.055, .916, "Qwen3.8-27B · fixed round 3 endpoints · all 36 evaluation draws · one training seed per arm",
             fontsize=11.5, color="#4A535D")
    endpoint_handles = [Line2D([0], [0], marker="o", linestyle="none", color=colors[e], label=LABELS[e]) for e in ENDPOINTS]
    fig.legend(handles=endpoint_handles, loc="upper left", bbox_to_anchor=(.05, .897), ncol=3, frameon=False,
               fontsize=10, columnspacing=2)
    status_handles = [Line2D([0], [0], marker="X", linestyle="none", color="#526372", label="Invalid authoring: zero yield, no game"),
                      Line2D([0], [0], marker="^", linestyle="none", color="#526372", label="Compiled, runtime faults"),
                      Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white", color="#526372", label="Fault-free, zero yield"),
                      Line2D([0], [0], marker="o", linestyle="none", color="#526372", label="Fault-free, positive yield")]
    fig.legend(handles=status_handles, loc="upper left", bbox_to_anchor=(.05, .866), ncol=4, frameon=False,
               fontsize=8.8, columnspacing=1.8)
    fig.legend(handles=[Patch(facecolor=validity_colors[k], label=label) for k, label in
                        (("invalid_authorship", "Invalid authoring"), ("runtime_fault", "Runtime faults"), ("runtime_valid", "Fault-free"))],
               loc="upper left", bbox_to_anchor=(.05, .172), ncol=3, frameon=False, fontsize=9)
    fig.text(.055, .128, "Each compiled program is measured on 6 held-out maps × 4 frozen opponents × 2 seats (48 games).",
             fontsize=9.5, color="#4A535D")
    fig.text(.055, .101, "Lines join the same generation seed across checkpoints. Diamonds are means over every prescribed draw, including invalid outputs.",
             fontsize=9.5, color="#4A535D")
    fig.text(.055, .074, "Missing outputs are never plotted as zeros: this figure requires both round 3 endpoints and the complete independent audit.",
             fontsize=9.5, color="#4A535D")
    fig.text(.055, .047, "Descriptive results only; no significance claim. Fault-free or all-rest behavior alone does not establish strategic improvement.",
             fontsize=9.5, color="#4A535D")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".program-figure-", dir=prefix.parent) as temporary:
        staged = {suffix: Path(temporary) / ("figure" + suffix) for suffix in targets}
        try:
            fig.savefig(staged[".png"], dpi=220, facecolor="white")
            fig.savefig(staged[".svg"], facecolor="white", metadata={"Title": "Coding RL final held-out program yield"})
        finally:
            plt.close(fig)
        provenance = {"kind": "completed_program_study_reporting_figure", "analysis": str(analysis_path.resolve()),
                      "analysis_sha256": bindings["analysis"]["sha256"],
                      "manifest_sha256": bindings["manifest"]["sha256"],
                      "status_sha256": bindings["status"]["sha256"], "bound_inputs": bindings,
                      "reporting_source_sha256": sha(__file__), "matplotlib_version": matplotlib.__version__,
                      "data_source_complete": True, "evaluation_draws": 36, "cells": list(cells.values()),
                      "interpretation": "Descriptive paired-generation-seed display; no significance or general transfer claim.",
                      "outputs": {str(p.resolve()): sha(staged[suffix]) for suffix, p in targets.items() if suffix != ".json"}}
        staged[".json"].write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
        # No destination files exist until both the rendered inputs and the
        # independently audited state have been rechecked. Commit JSON last.
        verify_bound_inputs(bindings)
        if any(p.exists() for p in targets.values()):
            raise FileExistsError("Reporting target appeared during rendering")
        for suffix in (".png", ".svg", ".json"):
            staged[suffix].rename(targets[suffix])
    return {suffix.lstrip("."): str(path.resolve()) for suffix, path in targets.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.analysis, args.out_prefix), indent=2))


if __name__ == "__main__":
    main()
