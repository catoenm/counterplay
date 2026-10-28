# Follow-up: a working loop and a misleading difficulty reward

20 October 2026. The interleaved proposer–solver harness works, but this small frozen-model experiment did not demonstrate that outcome feedback produces useful training challenges. Its clearest observation is that a tool-label typo can earn the maximum empirical difficulty reward.

## What ran

Both roles used the same pinned Qwen3 4B quantized checkpoint, separate contexts, reasoning enabled and a 4096-token cap. Each arm attempted eight constructive proposals. The environment derived the target from a submitted path, checked feasibility and shortest path length, and withheld the witness from the solver. Each accepted task received two solver attempts.

In the feedback arm, verified outcomes and rewards entered the next proposer prompt. The control withheld explicit feedback while retaining accepted paths for diversity. The first proposer prompt and generated response were identical across arms. Full details, including the exploratory nature of this control, are in the [protocol](feedback-protocol.md).

| Measurement | Outcome feedback | Control |
|---|---:|---:|
| Proposal attempts | 8 | 8 |
| Accepted distinct tasks within arm | 5 | 7 |
| Truncated proposals | 2 | 1 |
| Rejected because shortest solution was too easy | 1 | 0 |
| Verified solver successes | 10/10 | 13/14 |
| Tasks with one success and one failure | 0 | 1 |
| Two-step / three-step shortest solutions | 5 / 0 | 6 / 1 |
| Distinct action sets | 2 | 5 |
| Output tokens across both roles | 42,448 | 58,185 |
| Input tokens across both roles | 7,465 | 6,143 |
| Total model calls | 18 | 22 |
| Run time | 476.0 seconds | 677.4 seconds |

These are counts from one adaptive sequence per arm. Rounds are dependent, tasks diverge, and actual compute differs despite matching proposal counts and per-call caps. They do not establish that feedback helps or hurts. The two arms share their first task, so there are eleven distinct accepted task identities across both arms, not twelve independent tasks. Two solver attempts per task cannot reliably estimate its underlying success probability.

## The reward failure

The control's fourth proposal produced one correct response and one response with this final answer:

```json
{"plan":["tool2","tool0"]}
```

The interface defines `tool_2` and `tool_0`. The exact grader correctly rejected the original answer as an illegal action. With one success and one failure, the proposed `4*p*(1-p)` reward equals **1**, its maximum.

A post-hoc diagnostic inserted only the missing underscores. That action sequence reaches the target within budget. The original grade remains unchanged in every primary result. This isolates an interface-compliance failure in this example; it does not justify silently repairing outputs or globally relaxing the verifier.

Thus, the one positive difficulty reward in these two arms did not require a more difficult planning problem. A future curriculum could select such failures unless evaluation distinguishes protocol compliance from planning. The proposer did not demonstrably discover or exploit this intentionally, and the example is not evidence of a general feedback effect. It is a concrete mechanism to test in the proposed interface-robustness study.

## The generator has room to choose harder tasks

An exact enumeration of modulus 7, all 35 four-action subsets and all nonidentical start/goal states found 13,771 two-step, 22,758 three-step and 21,312 four-step combinations. All fit the existing proposal grammar when supplied with an exact shortest witness. These counts do not collapse isomorphic problems and do not measure LLM difficulty.

The feedback arm's five accepted tasks were all at the minimum permitted depth. Four of its five used the same action set. The scarcity of deeper proposals therefore does not reflect an absence of deeper tasks in the grammar. It suggests a procedural generator with controlled depth is a necessary baseline before investing in learned task generation. The control's one three-step task was solved twice, so increasing depth alone is not yet a demonstrated fix.

## What this changes about the next experiment

1. Use a common candidate pool balanced by **verified shortest depth**, including deeper tasks, and report step-budget slack separately. Compare procedural and LLM-generated tasks at matched difficulty and generation cost.
2. Independently rescore candidates with more attempts. Keep selection samples separate from evaluation samples and separate incorrect plans, illegal actions and truncation. The present two-sample rewards are insufficient for curriculum selection.
3. Test a prospectively specified action-format control, such as constrained final-action output, alongside the current strict interface. Preserve exact transition checking. Do not treat the post-hoc underscore edit as an experimental treatment already run.
4. Once both uniform and difficulty-selected pools contain enough independent verified examples, compare small adapters under matched training tokens and updates. Freeze untouched evaluation tasks first and test transfer outside this toy domain separately.

No optimizer or weight update ran in this follow-up. The separate [reasoning comparison](reasoning-results.md) found 22/24 successes with reasoning versus 3/24 without it, but used approximately 97 times as many output tokens. That improved inference configuration made this loop possible; it is not a self-play learning result. The original calibration gate requiring several mixed-outcome tasks remains unmet.

## Records and verification

The source snapshots, prompts, responses and outcomes are in `runs/reasoning-feedback` and `runs/reasoning-no-feedback`. The main comparison is `runs/feedback-comparison.json`; the finite-space enumeration is `runs/task-space.json`. Successful responses were independently regraded from saved prompts and task specifications, and archived source hashes were checked. All 13 unit tests passed, including feedback delivery and feedback withholding.

```sh
.venv/bin/python scripts/compare_feedback.py \
  runs/reasoning-feedback runs/reasoning-no-feedback \
  --out runs/feedback-comparison.json
.venv/bin/python scripts/task_space.py --out runs/task-space.json
```

All runs used local hardware. Cloud spending remains **$0 of the approved $100 pilot cap**, and no cloud resources were created. The market branch remains a [research design](market-track.md), with no completed trading experiment.
