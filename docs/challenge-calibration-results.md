# Thinking enabled challenge calibration results

21 October 2026. Qwen3 4B solved substantially more of the fresh procedural puzzles with thinking enabled, but generated too few valid challenges for a curriculum. The local calibration is complete. The frontier comparison has not run because API credentials are not configured. No training ran and external spending remains $0 of the approved $100 pilot cap.

## Paired procedural comparison

Both configurations use the same pinned 4-bit Qwen3-4B checkpoint, six previously unused development tasks, identical public prompts, and a 4096-output-token cap. Tasks have exact shortest depths 2, 4 and 6, with two tasks per depth, modulus 11, and four actions. We excluded 2,311 existing task families, including the unfinished training study's held-out tasks. Each task received four canonical attempts and two independently sampled renamed/reordered attempts. The final-answer grammar only receives legal labels and the horizon; it cannot supply a solution.

| Measurement | Thinking enabled | Thinking disabled |
|---|---:|---:|
| Canonical verified solutions | 18/24 (75.0%) | 0/24 (0%) |
| Renamed/reordered verified solutions | 10/12 (83.3%) | 2/12 (16.7%) |
| All procedural attempts | 28/36 (77.8%) | 2/36 (5.6%) |
| Canonical tasks solved at least once | 5/6 | 0/6 |
| Reasoning/output truncations | 8/36 | 0/36 |
| Completed legal wrong-goal plans | 0/36 | 34/36 |
| Format or illegal-action errors | 0/36 | 0/36 |
| Actual output tokens, including reasoning | 84,758 | 672 |
| Amortized generation time | 508.45 seconds | 9.12 seconds |

The canonical improvement is 75 percentage points; an exploratory bootstrap over six task clusters gives a 95% interval of 41.7–100 percentage points. Six tasks are too few to establish broad generalization. Four attempts estimate task difficulty very noisily, and the two presentation variants do not provide independent task clusters.

Equal output caps did not match actual compute. Thinking consumed 126.1 times as many output tokens. The disabled model finishes with short plans, while the thinking model spends most of its generation on reasoning. Per-group times apportion batch duration equally across its requests; they are not independent per-request latency measurements or a hardware benchmark. Thinking also changes the model's chat template. This comparison does not isolate parameter count, prove learning, or show that thinking is more efficient per token.

## Agent generated challenges

Qwen attempted six proposals with thinking enabled, requesting depths 2, 4, 6, 2, 4 and 6. Unlike the earlier constructive diagnostic, this protocol requires a claimed goal as well as a witness. The arbiter checks the claim by execution without repairing it. No solver outcomes are fed back during this screening; accepted prior proposals are supplied only to encourage diversity.

One of six proposals was accepted, with verified shortest depth two. The other five exhausted 4096 tokens without a final answer. They are truncation failures, not examples of completed but mathematically invalid proposals. The proposal calls generated 24,043 tokens in total.

The accepted proposal starts at `(1,0)` with target `(2,1)` modulo 11 and permits `double_x`, `inc_y`, `swap` and `sum_x`. Its witness is `double_x → inc_y`. The solver never receives that witness. It solved the puzzle in all four canonical and both renamed/reordered attempts, including an alternative valid solution `inc_y → sum_x`. The six solver attempts generated 14,338 tokens. A uniformly random legal two-action plan solves this task with probability 3/16; the exact reference is enumerated separately from model inference.

The screening gate was not met: only one distinct valid proposal was produced, with no depth diversity, and it did not challenge the solver in these six attempts. Across the procedural controls, one task had mixed canonical outcomes, but its failures were truncations. No task met the preset criterion of mixed successes and completed legal wrong-goal failures. This criterion deliberately separates a planning curriculum from one driven by output-budget exhaustion; it does not prove that truncation is unrelated to task difficulty.

## Frontier comparison and next decision

The planned frontier configuration is `gpt-6-astra`, medium reasoning, standard service, no tools, separate contexts and a 4096-output-token cap. Its client and budget ledger have unit coverage using mocked network calls; account availability and live API compatibility remain untested. No API request has been sent. Configure `OPENAI_API_KEY` locally using the ignored project `.env` or environment, or select another provider before running that arm. Never place a secret in the run records or chat.

The API client reserves at most $35 of the $100 cumulative pilot budget, reserves a conservative per-call cost before sending, records provider usage, retains an upper-bound charge for unknown outcomes, and does not automatically retry. The protocol records the checked pricing and assumptions. Actual account billing remains authoritative.

Keep training paused. The next intended comparison is frontier-generated tasks solved by both models, and the frontier model solving Qwen's task and the procedural controls. If the stronger model reaches a ceiling, the environment needs a separately recorded difficulty expansion. If proposal generation is the limiting stage, a separately recorded larger reasoning budget or constructive proposal protocol can distinguish budget exhaustion from an inability to construct valid tasks. The present data do not justify a claim that small-model self-play fails, or that this curriculum improves a model.

## Verification and reproduction

All 78 solver responses were independently regraded from saved task definitions and prompts. Proposal witnesses and all seven visualized trajectories were replayed. The suite passes 29 tests, including incorrect-goal rejection, witness exclusion from solver prompts, truncated-result exclusion from curriculum scoring, budget refusal before network activity, retained reservations after timeout, and reuse of journaled successful responses. The interactive replay passes browser interaction and overflow checks at 320, 360 and 736 pixels.

The [frozen protocol](challenge-calibration-protocol.md) describes sampling, admission gates, pricing and limitations. Raw records are in `runs/challenge-proposals-qwen`, `runs/challenge-solve-qwen` and `runs/challenge-solve-qwen-off`. The [comparison JSON](../runs/challenge-comparison.json) includes per-source results, the paired bootstrap and enumerated uniform-random-plan references. Recompute it with:

```sh
.venv/bin/python scripts/analyze_challenges.py --runs runs/challenge-solve-qwen runs/challenge-solve-qwen-off --out runs/challenge-comparison.json
```

The earlier larger trajectory study remains incomplete: the plan-only adapter is saved, but the mixed-training process exited without a completed checkpoint. It has not been restarted or evaluated as part of this calibration.
