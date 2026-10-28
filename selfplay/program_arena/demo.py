"""Run a reproducible scripted program league; no model calls or training.

python -m selfplay.program_arena.demo --out runs/program-arena-prototype
"""
import argparse
import hashlib
import json
from pathlib import Path

from .dsl import compile_program
from .examples import reference_programs
from .runner import play, verify_replay
from .spec import model_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/program-arena-prototype"))
    parser.add_argument("--seeds", type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.seeds <= 100:
        raise ValueError("scripted demo supports 1..100 seeds")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "programs").mkdir()
    sources = reference_programs()
    programs = {name: compile_program(source) for name, source in sources.items()}
    for name, program in programs.items():
        (args.out / "programs" / f"{name}.json").write_text(json.dumps(json.loads(program.source), indent=2))
    (args.out / "model-instructions.txt").write_text(model_spec())
    comparisons = [("nearest", "idle"), ("nearest", "value"), ("value", "avoid_rival")]
    source_dir = Path(__file__).parent
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_dir.glob("*.py"))}
    rows, representative = [], None
    for a, b in comparisons:
        for seed in range(args.seeds):
            for seat in (0, 1):
                names = [a, b] if seat == 0 else [b, a]
                record = play([programs[n] for n in names], seed, origins=["scripted", "scripted"])
                verify_replay(record)
                row = {"program_a": a, "program_b": b, "seed": seed, "a_seat": seat,
                       "a_return": record["returns"][seat],
                       "a_score": record["metrics"]["scores"][seat],
                       "b_score": record["metrics"]["scores"][1-seat],
                       "faults": record["metrics"]["program_faults"], "replay_verified": True}
                rows.append(row)
                if representative is None or (a == "nearest" and b == "value" and seed == 1 and seat == 0):
                    representative = record
    summary = {"kind": "scripted prototype validation", "model_generated": False, "gradient_training": False,
               "external_spend_usd": 0, "games": len(rows), "seeds_per_pair": args.seeds,
               "seat_swaps": True, "source_sha256": hashes, "all_replays_verified": True,
               "program_digests": {name: p.digest for name, p in programs.items()}, "pairs": {}}
    for a, b in comparisons:
        group = [r for r in rows if r["program_a"] == a and r["program_b"] == b]
        summary["pairs"][f"{a} vs {b}"] = {"games": len(group),
            "a_wins": sum(r["a_return"] > 0 for r in group), "ties": sum(r["a_return"] == 0 for r in group),
            "a_mean_return": sum(r["a_return"] for r in group) / len(group),
            "a_mean_score": sum(r["a_score"] for r in group) / len(group),
            "b_mean_score": sum(r["b_score"] for r in group) / len(group),
            "program_faults": sum(sum(r["faults"]) for r in group)}
    (args.out / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (args.out / "representative-replay.json").write_text(json.dumps(representative))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
