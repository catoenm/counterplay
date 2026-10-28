# Depth-controlled training pilot: results

20 October 2026. **Actual training completed, but neither curriculum demonstrated a planning improvement.** Canonical held-out accuracy was 9.77% for the frozen base, 9.38% after uniform filtered self-training, and 8.20% after difficulty-ranked filtered self-training. Canonical transfer accuracy fell from 9.38% to 3.12% and 2.34%, respectively. These results do not justify scaling this configuration on rented GPUs.

The working deliverable is a reproducible pipeline with depth-controlled tasks, strict legal actions, independently verified supervision, real QLoRA updates, and an untouched held-out comparison. This is one round of procedural challenge generation and filtered self-training, not coevolving proposer–solver RL or a novel training method.

## Data and controls

The [pre-training protocol](depth-training-protocol.md) fixed the model, splits, training settings and final-checkpoint evaluation. The model is `mlx-community/Qwen3-4B-4bit`, snapshot `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`. Thinking is disabled throughout this comparison. Earlier reasoning-enabled results are a different protocol and are not the baseline here.

Each task has four actions on a two-coordinate modular state. Exact shortest depths are 2–5; the public horizon equals that depth. The decoder permits every legal action sequence at that horizon, including incorrect sequences. It receives no target or reference solution. Task families ignore witness and budget and collapse the exact x/y coordinate-swap symmetry. Candidate, validation and test families are disjoint; this is not a guarantee against every semantic similarity.

Collection produced **425 verified solutions in 4,096 attempts** on 256 candidates: depth 2, 256/1,024; depth 3, 40/1,024; depth 4, 92/1,024; depth 5, 37/1,024. Every response was legal JSON with valid actions and the required length. All failures reached the wrong goal. The paused run resumed from 208 verified saved responses and finished with 4,096 unique task/attempt identities.

Eight attempts per candidate estimated difficulty using `4*p*(1-p)`; eight separate attempts supplied possible training answers. No oracle plan, repaired answer or reasoning trace entered training. There were 43 tasks with a successful supervision response and 22 also having a positive selection-stage frontier score. The predeclared fallback selected **20 tasks per arm**, distributed as 7 at depth 2, 3 at depth 3, 6 at depth 4, and 4 at depth 5. Each task supplies one answer. Twelve tasks overlap between arms.

| Selected curriculum | Selection-half success rate | Separate supervision-half success rate |
|---|---:|---:|
| uniform | 51.88% | 59.38% |
| frontier | 48.75% | 59.38% |

These rates are conditional on obtaining successful supervision; they are not unbiased pool-wide difficulty estimates. Both sets are highly selected. Only four of 32 validation tasks yielded any successful answer (34 successes across 512 attempts); their depths were 2, 3, 3 and 5. This shared, success-filtered validation set is too narrow to establish generalization.

## Training actually performed

Both arms used the same 80 updates, batch size 4, Adam learning rate 1e-4, rank-8/scale-16 QLoRA on the last eight layers, no dropout, seed 71 and gradient checkpointing. Each updated 3.67 million parameters and processed 5,568 supervised answer tokens, including EOS, over 16 dataset passes. Prompt and padding tokens are excluded from loss. Input tokens per pass differed slightly: 4,239 versus 4,228. The final update-80 checkpoint was used without test-based selection.

| Arm | Full validation loss before | Full validation loss after | Training seconds | Updated adapter tensors |
|---|---:|---:|---:|---:|
| uniform | 1.2128 | 0.1299 | 108.7 | 112 |
| frontier | 1.2128 | 0.1028 | 106.0 | 112 |

Both saved adapters contain finite tensors and nonzero B weights, which were initialized to zero. Their evaluated checkpoint hashes match the completed training records. Reduced answer cross-entropy did not predict improved held-out planning.

## Held-out accuracy

The 64 in-domain tasks and 32 transfer tasks were frozen before data collection. Each model received four samples per task in canonical and renamed/reordered presentations: **2,304 evaluation calls total**. All three models used the same prompts, per-request seeds, temperature 0.6, top-p 0.95, top-k 20, 128-token cap and batch size 16. All outputs were independently regraded. All evaluation failures were wrong goals, with no syntax or illegal-action failures.

Pass@1 below is the fraction of the four sampled attempts that succeeded, averaged over tasks. Pass@4 is the fraction of tasks solved at least once. ID cells use 256 attempts; transfer cells use 128. Parentheses contain successful attempt counts.

| Split and presentation | Base pass@1 | Uniform pass@1 | Frontier pass@1 |
|---|---:|---:|---:|
| In-domain (mod 7), canonical | 9.77% (25) | 9.38% (24) | 8.20% (21) |
| In-domain (mod 7), renamed_reordered | 5.47% (14) | 4.69% (12) | 5.86% (15) |
| Transfer (mod 11), canonical | 9.38% (12) | 3.12% (4) | 2.34% (3) |
| Transfer (mod 11), renamed_reordered | 1.56% (2) | 3.91% (5) | 1.56% (2) |

| Split and presentation | Base pass@4 | Uniform pass@4 | Frontier pass@4 |
|---|---:|---:|---:|
| In-domain (mod 7), canonical | 12.50% | 9.38% | 12.50% |
| In-domain (mod 7), renamed_reordered | 6.25% | 7.81% | 12.50% |
| Transfer (mod 11), canonical | 12.50% | 3.12% | 3.12% |
| Transfer (mod 11), renamed_reordered | 3.12% | 6.25% | 3.12% |

The exact expected success rate of uniformly sampling a legal fixed-length plan is 5.20% on the ID tasks and 3.97% on transfer tasks. This is an exhaustive combinatorial reference, not an LLM baseline or a significance test. Models used exactly 4,608 canonical and 3,712 renamed/reordered output tokens per ID evaluation; transfer totals were 2,304 and 1,856. Counts include EOS, and are identical across model arms within each presentation.

### Paired uncertainty

Differences are percentage points. Intervals resample whole tasks, retaining their four paired attempts, using 5,000 bootstrap samples. They measure task-sample uncertainty only, not uncertainty across training seeds. The transfer and presentation comparisons are secondary; no multiple-comparison correction is applied. Sparse, identical paired outcomes can produce a degenerate interval and do not imply population equivalence.

| Split / presentation | Difference | Estimate (pp) | Task bootstrap 95% interval (pp) |
|---|---|---:|---:|
| In-domain (mod 7) / canonical | uniform-base | -0.39 | [-4.30, +3.91] |
| In-domain (mod 7) / canonical | frontier-base | -1.56 | [-6.25, +3.12] |
| In-domain (mod 7) / canonical | frontier-uniform | -1.17 | [-4.69, +1.56] |
| In-domain (mod 7) / renamed_reordered | uniform-base | -0.78 | [-3.52, +1.56] |
| In-domain (mod 7) / renamed_reordered | frontier-base | +0.39 | [-3.12, +3.52] |
| In-domain (mod 7) / renamed_reordered | frontier-uniform | +1.17 | [-1.17, +3.52] |
| Transfer (mod 11) / canonical | uniform-base | -6.25 | [-14.84, +0.00] |
| Transfer (mod 11) / canonical | frontier-base | -7.03 | [-15.62, -0.78] |
| Transfer (mod 11) / canonical | frontier-uniform | -0.78 | [-2.34, +0.00] |
| Transfer (mod 11) / renamed_reordered | uniform-base | +2.34 | [+0.00, +6.25] |
| Transfer (mod 11) / renamed_reordered | frontier-base | +0.00 | [+0.00, +0.00] |
| Transfer (mod 11) / renamed_reordered | frontier-uniform | -2.34 | [-6.25, +0.00] |

All primary ID canonical intervals include zero. The frontier adapter's canonical transfer decline has a negative task-bootstrap interval, but this single-seed secondary result is not a general conclusion about frontier curricula.

### Depth breakdown

Each ID depth/presentation cell has 64 attempts on 16 tasks; each transfer cell has 32 attempts on eight tasks.

| Split / presentation | Depth | Base | Uniform | Frontier |
|---|---:|---:|---:|---:|
| In-domain (mod 7) / canonical | 2 | 18.75% | 25.00% | 21.88% |
| In-domain (mod 7) / canonical | 3 | 6.25% | 6.25% | 3.12% |
| In-domain (mod 7) / canonical | 4 | 9.38% | 0.00% | 3.12% |
| In-domain (mod 7) / canonical | 5 | 4.69% | 6.25% | 4.69% |
| In-domain (mod 7) / renamed_reordered | 2 | 9.38% | 9.38% | 9.38% |
| In-domain (mod 7) / renamed_reordered | 3 | 12.50% | 9.38% | 9.38% |
| In-domain (mod 7) / renamed_reordered | 4 | 0.00% | 0.00% | 0.00% |
| In-domain (mod 7) / renamed_reordered | 5 | 0.00% | 0.00% | 4.69% |
| Transfer (mod 11) / canonical | 2 | 12.50% | 0.00% | 0.00% |
| Transfer (mod 11) / canonical | 3 | 12.50% | 12.50% | 9.38% |
| Transfer (mod 11) / canonical | 4 | 3.12% | 0.00% | 0.00% |
| Transfer (mod 11) / canonical | 5 | 9.38% | 0.00% | 0.00% |
| Transfer (mod 11) / renamed_reordered | 2 | 0.00% | 0.00% | 0.00% |
| Transfer (mod 11) / renamed_reordered | 3 | 6.25% | 12.50% | 6.25% |
| Transfer (mod 11) / renamed_reordered | 4 | 0.00% | 3.12% | 0.00% |
| Transfer (mod 11) / renamed_reordered | 5 | 0.00% | 0.00% | 0.00% |

## Interpretation and next experiment

Strict actions fixed the earlier interface-error reward artifact. Depth control produced usable verified supervision across all four depths, and actual local training is inexpensive. **Curriculum usefulness as a learning method remains unproven.** The uniform adapter's mean number of distinct canonical ID plans fell from 1.89 to 1.41 per four attempts; frontier was 1.81. Uniform canonical pass@4 also fell from 12.5% to 9.4%. Reduced diversity is a plausible contributor, not an established causal explanation. Renamed-prompt diversity increased, so this is not a universal output collapse.

The tiny dataset, 60% arm overlap, single training seed, canonical-only supervision, no thinking traces, and success-filtered validation all limit interpretation. Sixteen passes over 20 answers can fit local answer patterns without learning transferable planning. We have not isolated whether data coverage, the final-answer-only objective, optimization strength, or another factor caused the weak result. The task family itself is a small synthetic diagnostic; no general reasoning or market-performance claim follows.

The next experiment should change the data before renting more compute: collect a substantially larger set of distinct verified tasks, mix tool-label/order presentations, and compare final-answer supervision with verified intermediate-state trajectories at matched compute. Use fresh held-out families after treating this test set as observed development evidence. Repeat promising settings across training seeds before testing a coevolving proposer or policy-gradient RL objective. The market environment remains a separate research design, not an implemented or evaluated simulator.

## Artifacts and cost

Cloud spending remains **$0 of the approved $100**. No RunPod or Fly resource was provisioned; both trainings and all inference ran locally. Twenty repository tests pass. Both training jobs and all six evaluation jobs completed.

- [Frozen split manifest](../experiments/depth-v1/manifest.json)
- [Curriculum selection, hashes and overlap](../datasets/depth-v1/manifest.json)
- [Candidate rollouts](../runs/depth-candidates/rollouts.jsonl) and [validation rollouts](../runs/depth-validation/rollouts.jsonl)
- [Uniform adapter](../adapters/depth-uniform/adapters.safetensors) and [training summary](../adapters/depth-uniform/summary.json)
- [Frontier adapter](../adapters/depth-frontier/adapters.safetensors) and [training summary](../adapters/depth-frontier/summary.json)
- [Full in-domain comparison](../runs/depth-training-comparison.json) and [transfer comparison](../runs/depth-training-ood-comparison.json)
- [Verifier and comparison script](../scripts/analyze_depth.py), [training runner](../selfplay/train.py), and [dependency lock](../requirements-local.lock.txt)

Evaluation manifests record source and checkpoint hashes; training directories retain source and protocol snapshots. Raw test outputs are in `runs/depth-eval-{base,uniform,frontier}` and `runs/depth-eval-ood-{base,uniform,frontier}`. Recompute the ID comparison with:

```sh
.venv/bin/python scripts/analyze_depth.py \
  --base runs/depth-eval-base \
  --uniform runs/depth-eval-uniform \
  --frontier runs/depth-eval-frontier \
  --out runs/depth-training-comparison.json
```

For transfer, use the corresponding `depth-eval-ood-*` directories. Do not train on these observed final-test tasks and reuse them as an untouched evaluation.
