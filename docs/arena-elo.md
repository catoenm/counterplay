# Per-environment league Elo

[`scripts/arena_elo.py`](../scripts/arena_elo.py) creates an exploratory rating history from completed evaluation and league episodes. It performs CPU-only reconstruction using each run's hash-verified archived simulators, checks every recorded state and terminal result, and never invokes the language model. Training self-play is excluded.

```sh
.venv/bin/python scripts/arena_elo.py --run runs/arena-v1
# Once the continuation contains completed matches:
.venv/bin/python scripts/arena_elo.py \
  --run runs/arena-v1 --run runs/league-v1 \
  --out runs/league-v1/elo.json
```

Each run must include `manifest.json`, its archived `source/` files, and completed match files named `eval-*.jsonl`, `league-*.jsonl`, or `matches-*.jsonl`. The existing episode schema is required. A `schedule.json` in the arena stage-map format is checked when present. An incomplete trailing JSONL write is omitted and reported. Duplicate episode IDs are errors, including duplicates across runs.

Continuation manifests must declare a matching `rating_compatibility`. Original manifests without a declaration are treated as legacy `arena-v1`. The script independently compares factory and game-source hashes, horizon dictionaries, chess-library versions, rating-pool declarations, and policy arithmetic; it refuses incompatible pools. If an environment, adjudication horizon, or effective arithmetic changes, start a separate rating pool. Preserve dependency versions recorded by the run.

The FP32 run declares `rating_compatibility: "arena-v2-fp32"`, a `policy_arithmetic` object, and its `policy_arithmetic_sha256` (SHA-256 of sorted, compact canonical JSON). The hash is verified rather than trusted. BF16 and FP32 can produce different action distributions from identical adapter bytes, so their records **must not be pooled**. Analyze `runs/arena-v2` separately from `runs/arena-v1`; old records are never relabeled as FP32. Same arithmetic with a different pool declaration is also rejected.

Local checkpoint metadata and adapter bytes are hash-verified. New local checkpoints must record the same arithmetic profile, hash, and pool declaration as their run. A continuation can declare `inherited_checkpoints` keyed by policy name, with each entry containing its checkpoint directory `path`, `adapter_sha256`, and `metadata_sha256`. These inherited files are verified too. For inherited weights re-evaluated with FP32, the original metadata remains intact; the new run's manifest defines their effective arithmetic. Effective policy fingerprints include model revision, adapter hash, arithmetic hash, and pool declaration. Reusing a persistent name with different weights or effective arithmetic is rejected.

## Definition and interpretation

Each persistent checkpoint and each game's fixed heuristic enters that game's pool at **1500**, at its first completed match against a distinct policy. Checkpoints have separate identities (`base`, `cycle1`, `cycle2`, and so on); each historical checkpoint retains its own rating. A positive terminal return scores **1**, exactly zero scores **0.5**, and a negative return scores **0**. Fractional reward magnitude is deliberately discarded. With `K = 24`, the expected score is `1 / (1 + 10 ** ((opponent_rating - rating) / 400))`, and the update is `K * (observed_score - expected_score)`. Both players receive equal and opposite changes.

These are **adjudicated-outcome ratings**. A positive maze distance score at timeout or positive chess material score at the ply cap counts as a rating win but does not become a goal finish or checkmate. The JSON and visual retain termination counts. Market settlement outcomes reflect relative simulated profit; they are not financial performance ratings. Pools are independent across environments and are not comparable to public chess Elo.

Matches are processed in a declared deterministic order: training iteration, opponent phase (heuristic, base, then other), source filename, scenario seed, candidate seat, and episode ID. Batched completion order does not change a rating. Iteration defaults to the largest `cycleN` number among the two policies; a nonnegative integer `iteration` field can explicitly place historical-only exhibition matches at the iteration when they were played. Identical-policy matches are reported but excluded: two seats using `base` cannot both receive different persistent ratings. Distinct names with the same verified model revision, adapter bytes, and effective arithmetic are treated as the same policy.

The visual plots only observed stages. Newly observed checkpoints appear at their first completed rating stage. A historical model's rating can move when a new opponent plays it, even though its weights are unchanged; if it does not play during an observed game stage, its rating is carried forward with `new_games: 0`. Missing future stages are not extrapolated. Stage coverage labels incomplete scheduled evaluation. Hover values between stage points are explicitly labeled interpolations; they do not represent extra matches.

The output `elo.json` preserves the match-by-match update trail, exclusions, stage coverage, rule compatibility, source/file hashes, and missing scheduled matches. The companion `arena-rating-history.html` contains compact stage aggregates for an interactive three-panel view. Rerunning the command incorporates newly completed records and deterministically recomputes ratings. No ratings or match outcomes are fabricated.

## Limits and historical roster

Small samples, fixed opponent matching, seat effects, cyclic strategies, and the chosen processing order all influence Elo. A checkpoint's first rating stage is based on fewer games than an older checkpoint's later rating. Use the fixed-opponent paired-return analysis and raw outcome counts alongside ratings; Elo alone is not evidence of reliable improvement or general transfer.

For a bounded ongoing league, retain a fixed anchor and recent predecessors, then a deterministic rotating sample of older checkpoints. Every active historical opponent should face the new checkpoint each iteration with swapped seats; old-versus-old exhibitions may be added with explicit iteration IDs. Pruning removes checkpoints from the active match schedule, **not from the checkpoint archive, existing match records, or rating history**. Keep an oldest/base anchor and preserve distinct counterstrategies rather than pruning solely on a noisy Elo threshold. Changing the active roster changes the matchup distribution and must be recorded. Historical-opponent evaluation does not imply historical-opponent training.

Validation: `tests/test_arena_elo.py` checks rating conservation, equal-rating draws, a known update, strict reward-sign scoring, deterministic ordering, persistent identities, independent environment pools, identical-effective-policy exclusion, merged schedule counts excluding both training-ID formats, and absence of unobserved stages. It also tests arithmetic hash integrity, cross-precision and pool-tag rejection, local checkpoint metadata matching, and inherited metadata preservation. No GPU is needed.
