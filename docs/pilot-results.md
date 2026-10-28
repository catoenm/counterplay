# Initial local self play results

20 October 2026. The project now has a literature review, a verified planning environment, two frozen-model calibration runs, a constructive-proposal follow-up, and a design for market self-play. Cloud spend is **$0 of the approved $100 pilot cap**. No weights were trained and no market simulation was run.

The immediate finding is that this short, non-thinking model setup supplies too few successful trajectories for the proposed curriculum comparison. Checking that a task is solvable is necessary, but does not ensure the solver can learn from it. These runs do not establish self-improvement or a significant interface-robustness effect.

## Comparable frozen-model audit

The same 24 procedural tasks were evaluated with two attempts under four presentations. There were eight tasks at each shortest-path length of 1, 2 and 3. Both Qwen models used community 4-bit MLX checkpoints, temperature 0.7, top-p 0.95, a 256-token output cap and thinking disabled. Solver prompts requested only a final JSON plan. This is a restrictive calibration setting, not a full reasoning-capability benchmark.

| Policy | Canonical | Reordered | Renamed | Renamed and reordered | Total correct |
|---|---:|---:|---:|---:|---:|
| Exact search reference | 48/48 | 48/48 | 48/48 | 48/48 | 192/192 |
| Random legal plans, sampled | 4/48 | 4/48 | 4/48 | 4/48 | 16/192 |
| Qwen3 0.6B | 2/48 | 5/48 | 0/48 | 1/48 | 8/192 |
| Qwen3 4B | 7/48 | 8/48 | 2/48 | 4/48 | 21/192 |

The random policy samples a length uniformly from one through the allowed maximum, then samples legal actions uniformly. The same semantic random plan is used across equivalent presentations. Exhaustive enumeration gives an expected success rate of **10.39%** on these tasks; its observed sample rate was 8.33%. Qwen3 4B's aggregate observed rate was 10.94%, so these results do not show a convincing advantage over random planning. Repeated presentations are correlated observations, not 192 independent tasks.

For Qwen3 4B, the paired change from canonical to renamed-and-reordered was −6.25 percentage points; the exploratory task-bootstrap 95% interval was approximately **−18.75 to +4.17 points**. The interval includes zero. The initial hypothesis remains unconfirmed. The example in the prompt uses a canonical tool name even when labels are renamed; a future balanced design must remove or counterbalance this anchoring cue before interpreting any difference specifically as planning failure.

The strict robust-frontier heuristic assigned a positive score to **zero of 24 tasks** for either LLM. With only two attempts per presentation and low success, taking the minimum across presentations is too restrictive for a training curriculum. This is evidence against using this particular configuration immediately; it is not evidence against all robustness-aware curriculum methods.

## Proposal feasibility

In the original protocol, each model proposed eight tasks containing both a claimed target and a solution witness. **All 16 witnesses failed replay.** These proposals were rejected and none entered the solver pool. The harness did not accept a model's claim to know the answer.

A separate constructive protocol then asked Qwen3 4B for a start, permitted tools and action sequence, and had the environment compute the target. This follows the logic of action-based task construction in asymmetric self-play: the executed trajectory establishes reachability. It is a different protocol, not a silent repair of rejected tasks. [Asymmetric self-play precedent](https://arxiv.org/abs/1703.05407).

The constructive run accepted **3 distinct tasks from 8 proposals**. Five proposals were duplicates. All three accepted tasks had checked witnesses and shortest paths of at least two steps. The solver got **0/12 attempts** correct on these generated tasks across two presentations. The six procedural calibration tasks included in that run yielded 1/24 correct. Because this run uses a smaller task set and different protocol, its pooled accuracy should not be compared directly to the larger audit.

This small sample reveals repeated proposals, not training-induced diversity collapse: the weights never changed. A concrete next intervention is to improve exploration and proposal diversity while preserving exact verification.

## Decision

Do not launch curriculum training with these defaults. First compare explicit reasoning or thinking-enabled inference at matched total generation budgets, test a simpler initial difficulty distribution, and increase independent solver samples. Keep the constructive proposer as one baseline. Consider ordinary presentation augmentation and an average frontier score before a minimum-based reward. Make these comparisons on fresh development tasks; the current tasks have now been inspected and are development data.

Only advance to the $100 cloud training pilot after a meaningful pool of distinct tasks produces verified successful solver trajectories. This prevents spending the training budget on a reward function that has no useful variation. The frozen-proposer adapter comparison and its controls are specified in [the experiment plan](research-plan.md).

The market proposal remains a second track. It has direct environment precedents in ABIDES and ABIDES-MARL, but requires careful wealth accounting and evaluation against independent opponents. Its initial reward and comparison design is in [the market track](market-track.md).

## Reproduction and verification

Six test cases pass, including exhaustive transition checks against an independent affine specification, witness rejection, shortest-path checks, equivalent-presentation checks over short plans, malformed-output handling and constructive target generation. The exact solver passed all 192 presentation checks.

There were **444 local model calls** across the two audits and the constructive follow-up, including 24 proposer calls. All were inference only. The source, model snapshots, arguments and raw outputs are retained in the workspace.

| Record | Location |
|---|---|
| Original 0.6B audit | [Summary](../runs/qwen06b-audit/summary.json), [manifest](../runs/qwen06b-audit/manifest.json), [rollouts](../runs/qwen06b-audit/rollouts.jsonl) |
| Original 4B audit | [Summary](../runs/qwen4b-audit/summary.json), [manifest](../runs/qwen4b-audit/manifest.json), [rollouts](../runs/qwen4b-audit/rollouts.jsonl) |
| Constructive 4B follow-up | [Summary](../runs/qwen4b-constructive/summary.json), [proposals](../runs/qwen4b-constructive/proposals.jsonl), [tasks](../runs/qwen4b-constructive/tasks.jsonl) |
| Comparable statistics and exact random expectation | [comparison.json](../runs/comparison.json) |

Run `.venv/bin/python scripts/analyze_runs.py` from the project root to regenerate the comparison from raw rollouts. Run `.venv/bin/python -m unittest discover -s tests -v` to rerun verifier tests. To reproduce the constructive protocol, add `--proposal-mode construct` to a model pilot command. Existing result directories are protected against overwrite.
