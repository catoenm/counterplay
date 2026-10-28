"""Reporting-only figures from an independent cardinal GPU audit.

Never imports a neural runtime or scores a game. Progress uses only committed
training groups. Final comparisons require the complete fixed endpoint, all 64
evaluation draws, and the current independently audited completion state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile

KINDS = ("invalid_authorship", "runtime_fault", "valid_zero", "valid_positive")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def close(a, b):
    return type(a) in (int, float) and type(b) in (int, float) and math.isclose(a, b, rel_tol=0, abs_tol=1e-12)


def bind_inputs(audit_path):
    path = Path(audit_path).resolve()
    raw = path.read_bytes()
    audit = json.loads(raw)
    require(audit.get("kind") == "independent_cpu_cardinal_gpu_audit" and audit.get("integrity_passed") is True,
            "A passing independent cardinal GPU audit is required")
    run = Path(audit["run"]).resolve()
    bindings = {"audit": {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()},
                "reporter": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)}}
    for name in ("manifest", "status"):
        target = run / (name + ".json")
        digest = sha(target)
        require(digest == audit[name + "_sha256"], "Audit " + name + " binding differs")
        bindings[name] = {"path": str(target), "sha256": digest}
    manifest, status = json.loads((run / "manifest.json").read_text()), json.loads((run / "status.json").read_text())
    require(status.get("manifest_sha256") == bindings["manifest"]["sha256"], "Status/manifest identity differs")
    schedule = run / "schedule.json"
    require(sha(schedule) == manifest["schedule_sha256"], "Frozen schedule changed")
    bindings["schedule"] = {"path": str(schedule), "sha256": sha(schedule)}
    for relative, expected in manifest["source_sha256"].items():
        part = Path(relative)
        require(not part.is_absolute() and ".." not in part.parts, "Unsafe archived source path")
        source = run / "source" / part
        require(sha(source) == expected, "Audited source archive changed: " + relative)
        bindings["source/" + relative] = {"path": str(source), "sha256": expected}
    return audit, manifest, status, json.loads(schedule.read_text()), bindings


def verify_bindings(bindings):
    for name, row in bindings.items():
        require(sha(row["path"]) == row["sha256"], "Bound " + name + " changed during rendering")


def checked_cell(cell, names, games_per_program):
    n = len(names)
    require(cell.get("complete") is True and cell.get("prescribed_program_clusters") == n
            and cell.get("observed_program_clusters") == n and cell.get("missing_program_clusters") == 0,
            "Missing draws cannot become zero yield; full prescribed denominator required")
    rows = cell.get("programs", [])
    require(len(rows) == n and [p.get("generation") for p in rows] == names, "Program identity/order differs")
    programs = []
    for row in rows:
        value, authored, valid, faults = (row.get(k) for k in
            ("banked_point_yield", "authorship_valid", "runtime_valid", "candidate_program_faults"))
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "Missing/nonfinite yield is not zero")
        require(type(authored) is bool and type(valid) is bool and type(faults) is int and faults >= 0,
                "Missing authoring/runtime classification")
        if not authored:
            require(not valid and value == 0 and faults == 0 and row.get("games") == 0,
                    "Invalid authoring must have zero yield and no fabricated game")
            category = "invalid_authorship"
        else:
            require(row.get("games") == games_per_program and valid == (faults == 0),
                    "Compiled program is missing games or has inconsistent runtime faults")
            category = "runtime_fault" if faults else "valid_zero" if value == 0 else "valid_positive"
        programs.append({"generation": row["generation"], "banked_point_yield": value,
                         "category": category, "games": row["games"], "candidate_program_faults": faults})
    mean = math.fsum(row["banked_point_yield"] for row in programs) / n
    counts = {kind: sum(row["category"] == kind for row in programs) for kind in KINDS}
    require(close(mean, cell.get("primary_mean_banked_point_yield")), "Mean excludes or misweights prescribed outputs")
    require(cell.get("compiled_programs") == n-counts["invalid_authorship"]
            and cell.get("invalid_no_game_programs") == counts["invalid_authorship"]
            and cell.get("fault_free_programs") == counts["valid_zero"]+counts["valid_positive"]
            and cell.get("played_games") == sum(row["games"] for row in programs), "Summary denominators differ")
    return {"n": n, "mean": mean, "programs": programs, "counts": counts}


def progress_data(audit):
    require(audit.get("integrity_passed") is True, "Independent integrity audit required")
    groups = audit.get("training_groups", [])
    require([g["cycle"] for g in groups] == list(range(1, len(groups)+1)), "Training groups are not consecutive")
    require(all(type(g.get("checkpoint_committed")) is bool for g in groups), "Missing checkpoint commit status")
    committed = [g for g in groups if g["checkpoint_committed"]]
    require([g["cycle"] for g in committed] == list(range(1, len(committed)+1)), "Committed training prefix has a gap")
    require(committed, "No committed training groups to plot")
    tensors = audit["tensor_reports"]
    require(tensors["base"]["actual_optimizer_step"] == 0, "Base is not a zero-step checkpoint")
    rows, previous_step = [], 0
    for group in committed:
        n = group["cycle"]
        cell = checked_cell(group, [f"train-cycle{n}-{i}" for i in range(8)], 16)
        rewards = group.get("rewards")
        require(isinstance(rewards, list) and len(rewards) == 8
                and all(type(r) in (int, float) and math.isfinite(r) and -1 <= r <= 1 for r in rewards),
                "Training reward denominator differs")
        require(close(math.fsum(rewards)/8, group.get("mean_penalized_task_reward")), "Training reward mean differs")
        step = tensors[f"cycle{n}"]["actual_optimizer_step"]
        require(type(step) is int and step-previous_step in (0, 1), "Actual Adam step lineage differs")
        signal = group["gradient_signal"]
        require(signal in ("no_update", "validity_only_signal", "gameplay_and_validity_signal"), "Unknown gradient signal")
        require((step == previous_step) == (signal == "no_update"), "Skipped group/actual optimizer step disagree")
        rows.append({"cycle": n, "mean_reward": math.fsum(rewards)/8, "mean_yield": cell["mean"],
                     "compiled_fraction": (8-cell["counts"]["invalid_authorship"])/8,
                     "fault_free_fraction": (cell["counts"]["valid_zero"]+cell["counts"]["valid_positive"])/8,
                     "positive_yield_fraction": sum(p["banked_point_yield"] > 0 for p in cell["programs"])/8,
                     "actual_optimizer_step": step, "update_committed": step > previous_step,
                     "gradient_signal": signal, "counts": cell["counts"]})
        previous_step = step
    require(previous_step == audit["genuine_optimizer_steps"], "Total genuine optimizer steps differs")
    return {"groups": rows, "committed_groups": len(rows), "genuine_optimizer_steps": previous_step,
            "scored_uncommitted_cycles": [g["cycle"] for g in groups if not g["checkpoint_committed"]],
            "selected_endpoint": audit.get("selected_endpoint"),
            "complete": audit.get("planned_study_complete") is True,
            "interpretation": "Training diagnostics only. Maps and opponents change; these curves do not establish held-out improvement."}


def bootstrap_interval(values, draws, seed):
    require(type(draws) is int and 100 <= draws <= 1_000_000 and type(seed) is int, "Invalid bootstrap protocol")
    rng, n = random.Random(seed), len(values)
    samples = sorted(math.fsum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(draws))
    return [samples[int(.025*(draws-1))], samples[int(.975*(draws-1))]]


def final_data(audit, status, schedule):
    require(audit.get("planned_study_complete") is True and audit.get("planned_comparisons_available") is True
            and audit.get("integrity_passed") is True and status.get("state") == "complete"
            and status.get("claims_allowed") is True and audit.get("recorded_state") == "complete",
            "Final comparison requires complete status and a passing complete independent audit")
    endpoint = audit.get("selected_endpoint")
    require(type(endpoint) is int and endpoint in (50, 100), "Unexpected registered endpoint")
    require(audit.get("committed_generation_draws") == 8+8*endpoint+64, "All 64 evaluation draws are required")
    labels = ("base", f"cycle{endpoint}")
    require(set(audit["evaluation_cells"]) == set(labels), "Both fixed evaluation cells required")
    cells = {label: checked_cell(audit["evaluation_cells"][label], [f"eval-{label}-{i}" for i in range(32)], 48)
             for label in labels}
    contrast = audit["paired_yield"]
    seeds = schedule["eval_generation_seeds"]
    require(len(seeds) == len(set(seeds)) == 32 and all(type(s) is int for s in seeds)
            and contrast.get("paired_generation_seeds") == seeds and contrast.get("program_cluster_n") == 32,
            "Paired generation seeds differ")
    delta = [b["banked_point_yield"]-a["banked_point_yield"]
             for a, b in zip(cells["base"]["programs"], cells[labels[1]]["programs"])]
    require(len(contrast.get("paired_yield_differences", [])) == 32
            and all(close(a, b) for a, b in zip(delta, contrast["paired_yield_differences"])), "Paired yield differences disagree")
    require(close(cells["base"]["mean"], contrast.get("base_mean_yield"))
            and close(cells[labels[1]]["mean"], contrast.get("final_mean_yield"))
            and close(math.fsum(delta)/32, contrast.get("mean_yield_difference")), "Paired means disagree")
    interval = bootstrap_interval(delta, contrast["bootstrap_draws"], contrast["bootstrap_seed"])
    require(len(contrast.get("descriptive_bootstrap_95_interval", [])) == 2
            and all(close(a, b) for a, b in zip(interval, contrast["descriptive_bootstrap_95_interval"])),
            "Audited paired-cluster bootstrap interval differs")
    progress = progress_data(audit)
    require(progress["committed_groups"] == endpoint, "Fixed training endpoint is incomplete")
    return {"endpoint": endpoint, "labels": labels, "cells": cells, "generation_seeds": seeds,
            "paired_differences": delta, "mean_difference": math.fsum(delta)/32,
            "descriptive_bootstrap_95_interval": interval, "program_cluster_n": 32,
            "actual_optimizer_steps": audit["genuine_optimizer_steps"],
            "interpretation": "One training seed. Paired authoring-seed clusters; replay games are correlated. No continual-improvement or general-transfer claim."}


def style_axis(ax):
    ax.grid(axis="y", color="#E8EDF1", linewidth=.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#BAC5CE")
    ax.tick_params(colors="#405364")


def progress_figure(data, plt, fixture=False):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.8))
    fig.subplots_adjust(left=.09, right=.96, top=.82, bottom=.20, hspace=.40, wspace=.28)
    rows = data["groups"]; x = [r["cycle"] for r in rows]
    axes[0, 0].plot(x, [r["mean_reward"] for r in rows], color="#267AA1", linewidth=1.4, marker="o", markersize=3)
    axes[0, 0].set(title="Training reward · mean of all 8 outputs", ylabel="Mean penalized reward", ylim=(-1.05, 1.05))
    axes[0, 1].plot(x, [r["mean_yield"] for r in rows], color="#388872", linewidth=1.4, marker="o", markersize=3)
    axes[0, 1].set(title="Banked points · mean of all 8 outputs", ylabel="Mean banked-point yield")
    axes[0, 1].set_ylim(bottom=0)
    for key, label, color in (("compiled_fraction", "Compiled", "#657482"),
            ("fault_free_fraction", "Fault-free", "#267AA1"), ("positive_yield_fraction", "Positive yield", "#388872")):
        axes[1, 0].plot(x, [r[key] for r in rows], label=label, color=color, linewidth=1.2)
    axes[1, 0].set(title="Validity and useful play", ylabel="Fraction of all 8 outputs", ylim=(-.03, 1.08))
    axes[1, 0].legend(loc="lower left", fontsize=9, frameon=False)
    axes[1, 1].step([0]+x, [0]+[r["actual_optimizer_step"] for r in rows], where="post", color="#725A98", linewidth=1.6)
    skipped = [r for r in rows if not r["update_committed"]]
    if skipped:
        axes[1, 1].scatter([r["cycle"] for r in skipped], [r["actual_optimizer_step"] for r in skipped],
                           marker="x", color="#C27D32", s=28, label="Zero variance; no update")
        axes[1, 1].legend(frameon=False, fontsize=9)
    axes[1, 1].set(title="Actual Adam steps", ylabel="Cumulative committed optimizer steps")
    for ax in axes.flat:
        ax.set_xlabel("Attempted training group")
        ax.set_xlim(.5 if len(x)>1 else 0, max(x)+.5)
        style_axis(ax)
    prefix = "TEST FIXTURE · " if fixture else ""
    fig.text(.075, .948, prefix+"Cardinal controller RL: training progress", fontsize=20, weight="bold", color="#17212B")
    state = "Completed study" if data["complete"] else "Partial audited snapshot"
    fig.text(.075, .905, f"{state} · {len(rows)} committed groups · {data['genuine_optimizer_steps']} genuine optimizer steps · 8 outputs per group",
             fontsize=11, color="#4A535D")
    fig.text(.075, .105, "Training diagnostics only: maps and historical opponents change between groups; these curves are not held-out improvement.", fontsize=9.5, color="#4A535D")
    fig.text(.075, .073, "Each compiled controller plays 2 maps × 4 opponents × 2 seats (16 games). Yield is averaged over every output, including invalid outputs as zero.", fontsize=9, color="#4A535D")
    fig.text(.075, .043, "Runtime faults remain visible in reward and validity; missing or uncommitted training groups are omitted.", fontsize=9, color="#4A535D")
    return fig


def final_figure(data, plt, fixture=False):
    from matplotlib.lines import Line2D
    labels = data["labels"]; colors = ("#657482", "#197DA1")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 12.5), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.subplots_adjust(left=.12, right=.96, top=.82, bottom=.23, wspace=.30)
    maximum = max(1., max(p["banked_point_yield"] for c in data["cells"].values() for p in c["programs"]))
    for i in range(32):
        y = 31-i
        xs = [data["cells"][label]["programs"][i]["banked_point_yield"] for label in labels]
        axes[0].plot(xs, [y-.13, y+.13], color="#CBD2D8", linewidth=.8, zorder=1)
        for j, label in enumerate(labels):
            row = data["cells"][label]["programs"][i]
            marker = "X" if row["category"] == "invalid_authorship" else "^" if row["category"] == "runtime_fault" else "o"
            face = "white" if row["category"] == "valid_zero" else colors[j]
            axes[0].scatter([row["banked_point_yield"]], [y+(-.13 if j==0 else .13)], marker=marker,
                            facecolors=face, edgecolors=colors[j], s=34, linewidths=1, zorder=3)
        axes[1].scatter([data["paired_differences"][i]], [y], marker="D", color="#197DA1", s=24, zorder=3)
    axes[0].set_yticks(list(range(31, -1, -1)), [str(s) for s in data["generation_seeds"]], fontsize=8)
    axes[0].set_ylabel("Paired program-generation seed")
    axes[1].set_yticks([])
    axes[0].set_xlim(-maximum*.045, maximum*1.08)
    axes[0].set_xlabel("Mean banked points per held-out game")
    axes[1].set_xlabel("Final − base banked points")
    axes[0].set_title("All 32 prescribed programs per checkpoint", loc="left", fontsize=12, pad=12)
    axes[1].set_title("Paired change for each generation seed", loc="left", fontsize=12, pad=12)
    axes[1].axvline(0, color="#A5AFB8", linewidth=1)
    span = max(1., max(abs(x) for x in data["paired_differences"]))
    axes[1].set_xlim(-1.12*span, 1.12*span)
    for ax in axes:
        ax.set_ylim(-1, 32)
        style_axis(ax)
        ax.grid(axis="x", color="#E8EDF1", linewidth=.7)
        ax.tick_params(axis="y", length=0)
    if all(p["banked_point_yield"] == 0 for c in data["cells"].values() for p in c["programs"]):
        axes[0].text(.60, .52, "Every prescribed output\nyielded zero at both endpoints", transform=axes[0].transAxes,
                     ha="center", color="#657482", fontsize=11, linespacing=1.5)
    prefix = "TEST FIXTURE · " if fixture else ""
    fig.text(.07, .958, prefix+"Cardinal controller RL: held-out program yield", fontsize=20, weight="bold", color="#17212B")
    fig.text(.07, .925, f"Qwen3.8-27B · base versus cycle {data['endpoint']} · {data['actual_optimizer_steps']} actual optimizer steps · all 64 evaluation outputs", fontsize=11, color="#4A535D")
    fig.legend(handles=[Line2D([0], [0], color=c, marker="o", linestyle="none", label=label)
                        for c, label in zip(colors, ("Fresh GPU base", f"Cycle {data['endpoint']}"))],
               loc="upper left", bbox_to_anchor=(.064, .91), ncol=2, frameon=False)
    fig.legend(handles=[Line2D([0], [0], color="#526372", marker=m, linestyle="none",
        markerfacecolor="white" if kind=="valid_zero" else "#526372", label=title)
        for kind, m, title in (("invalid_authorship","X","Invalid authoring; no game"),
            ("runtime_fault","^","Compiled with runtime faults"), ("valid_zero","o","Fault-free, zero yield"),
            ("valid_positive","o","Fault-free, positive yield"))],
        loc="upper left", bbox_to_anchor=(.064, .882), ncol=4, frameon=False, fontsize=8.8)
    base, final = (data["cells"][label] for label in labels)
    lo, hi = data["descriptive_bootstrap_95_interval"]
    fig.text(.07, .18, f"Mean over all 32 outputs: base {base['mean']:.3f}  →  final {final['mean']:.3f} points/game.  Paired change {data['mean_difference']:+.3f} [{lo:+.3f}, {hi:+.3f}].",
             fontsize=11, weight="bold", color="#243E50")
    def line(cell):
        c=cell["counts"]
        return f"{c['invalid_authorship']} invalid · {c['runtime_fault']} with faults · {c['valid_zero']} fault-free zero · {c['valid_positive']} fault-free positive"
    fig.text(.07, .148, "Base: "+line(base)+".   Final: "+line(final)+".", fontsize=9.5, color="#4A535D")
    fig.text(.07, .115, "Each compiled controller: 6 held-out maps × 4 frozen opponents × 2 seats = 48 correlated games. Invalid authoring has zero yield and no game.", fontsize=9, color="#4A535D")
    fig.text(.07, .087, "Interval: descriptive 95% paired authoring-seed bootstrap (n = 32); one training seed. Missing outputs are never treated as zero.", fontsize=9, color="#4A535D")
    fig.text(.07, .059, "Evidence is specific to this fixed opponent pool. It does not establish general transfer or sustained continual improvement.", fontsize=9, color="#4A535D")
    return fig


def render(audit_path, prefix, mode="both"):
    require(mode in ("progress", "final", "both"), "Unknown plot mode")
    audit, manifest, status, schedule, bindings = bind_inputs(audit_path)
    data = {}
    if mode in ("progress", "both"):
        data["progress"] = progress_data(audit)
    if mode in ("final", "both"):
        data["final"] = final_data(audit, status, schedule)
    prefix = Path(prefix)
    targets = {(kind, suffix): Path(str(prefix)+"-"+kind+suffix)
               for kind in data for suffix in (".png", ".svg", ".json")}
    if any(path.exists() for path in targets.values()):
        raise FileExistsError("Existing figures are retained; choose a new output prefix")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "svg.fonttype":"none"})
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cardinal-report-", dir=prefix.parent) as temp:
        staged = {}
        for kind, value in data.items():
            fig = (progress_figure if kind=="progress" else final_figure)(value, plt, audit.get("test_fixture") is True)
            try:
                for suffix in (".png", ".svg"):
                    path = Path(temp)/(kind+suffix)
                    fig.savefig(path, dpi=200, facecolor="white")
                    staged[(kind, suffix)] = path
            finally:
                plt.close(fig)
            provenance = {"kind":"cardinal_gpu_reporting_figure", "plot":kind,
                "source_audit_complete":audit.get("planned_study_complete") is True,
                "test_fixture":audit.get("test_fixture") is True, "bound_inputs":bindings,
                "reporting_source_sha256":bindings["reporter"]["sha256"], "matplotlib_version":matplotlib.__version__,
                "data":value, "outputs":{str(targets[(kind,s)].resolve()):sha(staged[(kind,s)]) for s in (".png",".svg")}}
            path = Path(temp)/(kind+".json")
            path.write_text(json.dumps(provenance, indent=2, allow_nan=False)+"\n")
            staged[(kind,".json")] = path
        verify_bindings(bindings)
        # Exclusive creation avoids overwriting another reporting invocation.
        for key, path in targets.items():
            with path.open("xb") as stream:
                stream.write(staged[key].read_bytes())
    return {kind:{suffix:str(targets[(kind,suffix)].resolve()) for suffix in (".png",".svg",".json")} for kind in data}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--mode", choices=("progress","final","both"), default="both")
    args=parser.parse_args()
    print(json.dumps(render(args.audit,args.output_prefix,args.mode),indent=2))


if __name__=="__main__":
    main()
