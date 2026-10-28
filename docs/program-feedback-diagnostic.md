# Practice-feedback controller diagnostic

`selfplay/program_feedback.py` is a separate gateway phase. It permits four requests, two sequential revisions of each of the original `qwen38-27b-1` and `qwen38-27b-2` controllers, with a $0.10 phase cap and the existing $100 cumulative pilot cap. It counts prior API ledger commitments, cloud charges and cloud reservations under `runs/external-budget.lock`. Unknown outcomes retain their reservation and are never retried. A completed response can be recovered from its journal without another request. The original model-screen client and records remain unchanged.

The provider is pinned to Qwen3.8-27B on Chutes FP8, without fallback, at 4,096 output tokens, temperature 0.6, top-p 0.95 and top-k 20. Each response must report zero reasoning tokens for the requested reasoning-off condition. Missing or nonzero reasoning usage stops the diagnostic and is recorded separately from compilation or game performance.

`scripts/run_program_feedback.py` freezes the simulator, Python frontend, response normalizer, evaluator, new runner/client, original outputs and protocol. Model Python is parsed and lowered into the bounded interpreter; it is never executed as Python. Exact saved revision responses must equal their completed API ledger entries. Source, response, Python and IR hashes accompany the records.

The six practice seeds are 730000–730005. Every valid version plays both fixed references, `nearest` and `value`, in both seats: 24 games per version. Prompts contain the unchanged original specification, only that lineage's immediate previous response, all practice game point totals and failure turns, and full traces for the first practice seed against `nearest` in both seats. These trace choices are fixed before outcomes. No reference code or human patch instructions are supplied. Invalid or truncated responses receive their diagnostic feedback; they receive no invented games or losses.

Exactly two revisions are attempted for each lineage. Round 2 is the final version even if invalid or worse than round 1. There is no best-of selection. Only after all four responses and their practice evaluations are saved may the originals and final versions play the six holdout seeds 740000–740005. Intermediate revisions are never evaluated on holdout, and holdout results never enter prompts. The original screening seeds, 720000–720005, are distinct from both sets.

Commands from the repository root:

```sh
.venv/bin/python scripts/run_program_feedback.py --run runs/program-feedback-v1 --prepare-from runs/program-qwen38-27b-v1
.venv/bin/python scripts/run_program_feedback.py --run runs/program-feedback-v1 --advance
```

The first command performs only CPU practice and writes the frozen plan. Each `--advance` invocation admits at most one new paid request, saves it, and evaluates that revision on practice. Repeat `--advance` four times in total, then run:

```sh
.venv/bin/python scripts/run_program_feedback.py --run runs/program-feedback-v1 --holdout
```

Resumption verifies archived sources, response journal equality and every existing game by deterministic replay. Incompatible artifacts, partial JSONL lines or incomplete match archives with completed summaries are retained and rejected. Files are not silently overwritten.

This is an adaptive authoring diagnostic with two lineages and one reused practice set. It measures whether tool feedback helps a model write useful controllers. It does not perform gradient RL and does not establish continuing self-play improvement. Win/draw/loss uses the sign of actual final banked points; unbanked cargo has no value. Reference code is available to the evaluator but absent from model prompts.
