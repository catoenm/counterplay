# Reasoning calibration results

20 October 2026. Reasoning mode produced usable verified solutions on this small planning environment. This is an inference configuration result, not self-play training. The checkpoint remained frozen throughout.

## Complete paired comparison

Both arms used the same six fresh development tasks, two attempts per presentation, canonical and renamed/reordered tools, prompt v2, temperature 0.6, top-p 0.95, top-k 20 and a 4096-token cap. The checkpoint was `mlx-community/Qwen3-4B-4bit` at `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`. Qwen documents the thinking switch in its [official model card](https://huggingface.co/Qwen/Qwen3-4B). Only that generation switch differed between arms; it also changes the model's chat template.

| Measurement | Reasoning enabled | Reasoning disabled |
|---|---:|---:|
| Verified solutions | 22/24 (91.7%) | 3/24 (12.5%) |
| Distinct tasks solved at least once | 6/6 | 1/6 |
| Canonical presentation | 11/12 | 2/12 |
| Renamed/reordered presentation | 11/12 | 1/12 |
| Output tokens, including reasoning | 42,933 | 444 |
| Input tokens across calls | 4,108 | 4,204 |
| Correct responses per 1,000 output tokens | 0.512 | 6.757 |
| Elapsed inference-run time | 430.2 seconds | 10.3 seconds |
| Truncated responses | 1/24 | 0/24 |

The 79.2 percentage-point accuracy difference is descriptive. A task-cluster bootstrap gives an exploratory interval of 54.2–100 percentage points, but only six task clusters were sampled, and the larger budget was chosen adaptively. This interval does not establish broad generalization.

The same maximum output cap did **not** match compute. Reasoning used about 96.7 times as many output tokens and 41.8 times the observed run time. The disabled arm generated short plans rather than spending its available budget. Its higher correctness-per-token ratio is not proof that repeated short samples would solve the same range of tasks: all three successes were on one task. A token-matched repeated-sampling comparison remains unrun. Wall times include generation and run bookkeeping and should not be treated as a hardware benchmark.

Reasoning solved 8/8 depth-one, 8/8 depth-two and 6/8 depth-three attempts; disabled reasoning solved 3/8, 0/8 and 0/8 respectively. The two reasoning failures were one unfinished, token-limited response and one illegal tool name. One of six tasks had mixed canonical outcomes. The equal aggregate accuracy across presentations does not establish invariance on unseen tasks.

## Earlier interrupted diagnostic

The first 1536-token reasoning run was stopped early because of truncation. The final saved record contains nine completed calls: three solved and six truncated; a tenth call was interrupted. The stop decision was made after eight responses, and one more finished before the interrupt took effect. This explains the budget change but is not a completed paired experiment. See the [protocol and adaptive change](reasoning-calibration.md).

## Decision

Use reasoning mode when testing whether the proposer–solver harness can produce verified trajectories. Do not start adapter training on these six development tasks. The original calibration gates were not demonstrated: the follow-up uses a different sample and only one task has mixed canonical outcomes, below the original four-task threshold. The present pool is mostly too easy with reasoning enabled.

The subsequent diagnostic is an interleaved constructive proposer–solver loop, with and without explicit outcome feedback. Its [protocol](feedback-protocol.md) states the limitations of two-attempt rewards, and its [completed results](feedback-results.md) expose a tool-label failure that receives maximum empirical difficulty reward. Weight training requires a larger, separately generated pool, independently scored difficulty, and untouched evaluation tasks.

## Reproducible records

* `runs/reasoning-on-dev`: interrupted 1536-token diagnostic, retained as incomplete.
* `runs/reasoning-on-4096-dev`: complete reasoning arm and source snapshot.
* `runs/reasoning-off-4096-dev`: complete disabled arm and source snapshot.
* `runs/reasoning-comparison.json`: paired results, checked for matching tasks, prompts and settings.

Recompute the comparison:

```sh
.venv/bin/python scripts/compare_reasoning.py \
  runs/reasoning-on-4096-dev runs/reasoning-off-4096-dev \
  --out runs/reasoning-comparison.json
```

All inference used local Apple Silicon. Cloud resources created: zero. Cloud spending: $0 of the approved $100 pilot cap.
