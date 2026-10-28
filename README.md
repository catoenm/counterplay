# Verified language model self play

A small independent research project on whether competition between language model agents produces useful training experience. The pilot has an approved cumulative $100 cap. Experiments began locally on Apple Silicon and now include separately recorded GPU runs.

## Current thinking and debugging experiment

`runs/program-debug-gpu-v1` launched on October 22, 2026 using the pinned original `Qwen/Qwen3.8-27B` on one H200. It starts from the previous audited 32-step SFT checkpoint with zero RL optimizer steps. Each controller trajectory contains a thinking-enabled draft, deterministic practice-game feedback, and one mandatory thinking-enabled revision. The instructions explain the complete collect → return → deposit loop. Training includes both sampled model turns; practice feedback is context rather than a target. All drafts, revisions, failures and skipped zero-signal groups are preserved.

The frozen [protocol](runs/program-debug-gpu-v1/plan.json) selects 25, 50 or 100 attempted groups of eight trajectories using only the first three groups' timing and the remaining fixed budget. Actual optimizer steps are counted separately. Evaluation compares 32 paired trajectories from each checkpoint under the same debugging harness on separate held-out maps. The run passed its [independent CPU preflight](runs/program-debug-gpu-v1/preflight-audit.json); GPU numerical checks and final evaluation must pass before any improvement claim. Follow [current progress](runs/cardinal-cycle-progress.json) and the [deployment review](runs/cardinal-debug-deployment-v1/launch-audit.json). Conservative commitments at launch were $93.576147678, including prior pending liabilities; this is not settled spend. Independent cleanup is bounded by 01:52:15 Pacific on October 23.

The preceding `runs/cardinal-warmstart-gpu-v2` completed 32 supervised updates but stopped at its useful-play gate with zero productive sampled controllers and zero RL updates. That failure remains separate and preserved in its [final audit](runs/cardinal-warmstart-gpu-v2/final-audit.json).

## Competitive self-play RL suite

The experiment suite uses actual policy-gradient training across three games, with a separate programmable-controller environment. See the [arena protocol](docs/arena-rl-protocol.md) and [guide to adding environments](docs/adding-arena-environments.md) for the shared contract, replay checks, and extension points. The earlier modular-puzzle calibration is stopped; its results remain historical.

* **Market:** simultaneous decisions in a limit-order book, private price signals, constrained capital and inventory, background liquidity, and exogenous settlement. Record absolute P&L alongside zero-sum relative returns.
* **Maze:** race on separate grids with limited roadblocks on the opponent's grid; every legal roadblock preserves a path. Reaching the goal ends the game; timeouts use explicitly labeled distance adjudication.
* **Chess:** full python-chess move legality and game-ending rules. This first short-horizon pilot uses varied legal endgames; a standard starting position is supported by the environment. Capped games use explicitly labeled material adjudication, not a claimed checkmate.
* **Programmable resource game:** models write bounded controllers that move, harvest, rest and bank resources on a shared map. A Python subset is compiled to a custom interpreter; generated Python is never executed. API authoring and feedback revisions are separate from model-weight RL. See the [controller environment](docs/program-arena-prototype.md), [Python interface](docs/program-python-interface.md), and [feedback diagnostic](docs/program-feedback-diagnostic.md).

The learner is pinned `mlx-community/Qwen3-8B-4bit` with 851,968 trainable LoRA parameters. It observes the current game state and samples a single-token label from a shuffled legal-action menu. Training uses both seats' on-policy trajectories, terminal returns, leave-one-episode-out role/environment baselines, a clipped likelihood-ratio objective, entropy regularization, and a KL penalty to the rollout policy. This trains action selection; it does not train free-form reasoning or use winner-only supervised fine-tuning.

The original `runs/arena-v1` remains **incomplete**. It finished one actual RL update and replay, then stopped before the second update because BF16 action probabilities changed with batch geometry. Its outputs and failed attempt remain intact. `runs/arena-v2` is a separately named FP32 condition: unchanged packed four-bit base weights, FP32 floating tensors, fresh baseline evaluations, fresh rollouts, and two new training cycles. Every legal-action probability in every training row is checked before each cycle. Do not combine its ratings with the original BF16 run.

**October 22 result:** the FP32 run completed all 246 games and passed the [independent audit](runs/arena-v2/analysis.json), including 2,997 policy transitions and both checkpoint deltas. Each new cycle made 88 optimizer updates on 352 action transitions from 24 self-play games. Mean returns against the same frozen base and heuristic opponents were −0.252171 for base, −0.303590 for cycle1, −0.296915 for cycle2, and −0.257130 for cycle3. These are descriptive results from three paired scenarios per cell; no overall training improvement is established. The separately preregistered 288-game base/cycle3 [holdout audit](runs/arena-final-holdout-v1/analysis.json) also passed: mean return was −0.298178 for base and −0.297624 for cycle3, a change of +0.000553 with descriptive paired-scenario 95% bootstrap interval [−0.041921, +0.036646]. No reliable gain is established. The [comparison figure](runs/arena-final-holdout-v1/figures/paired-return-change.png) shows all paired scenarios; the [quality summary](runs/arena-final-holdout-v1/research-quality-summary.json) distinguishes goal/checkmate outcomes from capped rewards. The local 27B plain coding condition stopped without training; the compact feedback condition completed one verified RLOO update with unchanged held-out behavior, as detailed below. The matched coding study, `runs/program-study-v1`, also completed and passed its [independent audit](runs/program-study-v1/final-audit.json): six real updates on 39,480 completion tokens, followed by all 36 fixed evaluation outputs. Gameplay RL, validity-only RL and base all had zero held-out banked yield; every paired evaluation completion was token-identical across endpoints despite different learned weights. The [overnight results](docs/overnight-results-2026-10-22.md) collect methods, figures, limitations and costs. Current lifecycle state is in `runs/overnight-progress.json`.

The [research-quality audit](runs/arena-v2/research-quality-audit.json) also separates actual finishes from proxy rewards: all 16 maze training games ended at the distance cap, and 15 of 16 chess training games used material adjudication (the other was a repetition draw). Market games settled normally; seven relative winners across the full run nevertheless lost absolute money. These reward and opponent-quality limitations matter when designing a stronger next experiment.

The local Qwen3.8-27B coding smoke also ran. The plain condition produced no usable opponent in four attempts and stopped without training. The separately frozen feedback condition completed one RLOO optimizer update over 3,191 fresh completion tokens, with verified adapter change and likelihood checks ([audit](runs/program-local-27b-compact-v1-independent-audit.json)). Its one sampled held-out controller was token-identical before and after, banked zero points, and averaged −0.519608 over six paired games. This demonstrates the code-generation training mechanism, not improved coding-game ability. The actual evaluation is one program draw and one map against three opponents per checkpoint; inherited metadata wording that says two draws/three maps is incorrect. The failed original controller and factual practice feedback were context; successful API answers were never gradient targets.

This workspace's resumable command and independent audit are:

```sh
.venv/bin/python -m selfplay.arena_v2
.venv/bin/python scripts/analyze_arena_v2.py --run runs/arena-v2
# Inspect a still-running experiment without claiming completion:
.venv/bin/python scripts/analyze_arena_v2.py --run runs/arena-v2 --allow-partial
```

The run saves complete trajectories, exact action probabilities, source/schedule manifests and checkpoints. Its process lock rejects duplicate jobs; completed stages are verified and reused. Both training seats use the current checkpoint; historical checkpoints are evaluation opponents only. The bounded roster retains anchors and recent/older snapshots without deleting research evidence. [Per-environment Elo](docs/arena-elo.md) is exploratory and separate from paired fixed-opponent evaluation. Local execution adds no external inference charges. Small samples measure pipeline operation and game performance, not guaranteed continual improvement or general reasoning transfer.

## Research

* [Literature review](docs/literature-review.md): proposer–solver methods, games, formal verification, recent curriculum research, and market simulation.
* [Experiment plan](docs/research-plan.md): hypotheses, controls, evaluation gates and budget.
* [Market track](docs/market-track.md): order-book self-play, reward accounting and opponent evaluation.
* [Reasoning calibration](docs/reasoning-calibration.md): controlled thinking-mode and token-budget follow-up.
* [Reasoning results](docs/reasoning-results.md): paired accuracy, actual token cost and limitations.
* [Feedback diagnostic](docs/feedback-protocol.md): interleaved constructive self-play with an outcome-feedback control.
* [Completed feedback results](docs/feedback-results.md): proposal quality, the control comparison and an interface error rewarded as difficulty.
* [Depth-controlled training protocol](docs/depth-training-protocol.md): frozen splits, strict action decoding, independent curriculum scoring and a matched QLoRA comparison.
* [Completed training results](docs/depth-training-results.md): two trained adapters, 2,304 held-out evaluations, no demonstrated canonical accuracy improvement, and $0 cloud spend.
* [Larger trajectory protocol](docs/trajectory-training-protocol.md): 128 parent tasks, four presentations, and a plan-only versus auxiliary state-trajectory comparison on fresh test families.
* [Thinking enabled challenge calibration](docs/challenge-calibration-protocol.md): frozen Qwen/frontier proposer–solver comparison with exact verification and separate truncation accounting. Further training remains paused.
* [Completed local challenge results](docs/challenge-calibration-results.md): thinking solved 28/36 procedural attempts versus 2/36 without thinking; only one of six proposals completed and it was solved 6/6 times.
* [OpenRouter amendment](docs/openrouter-calibration-protocol.md): DeepSeek V4 Pro 0813 through pinned DeepInfra FP8, high reasoning, strict solver JSON, and a $10 inference cap within the $100 pilot budget.

## Historical puzzle experiments

The initial pilot was a **frozen-model audit**. It included finite planning tasks with verified solution witnesses, an exact solver, four equivalent tool presentations, procedural and LLM proposals, failure/frontier/robust-frontier scores, and saved raw rollouts. The subsequent local QLoRA experiment completed 80 updates on each of two matched 20-task curricula. Canonical held-out accuracy was 9.77% for the base, 9.38% for uniform selection and 8.20% for difficulty ranking; neither showed a clear gain. Those experiments used filtered self-training. The competitive suite above adds the actual policy-gradient trainer and executable market environment.

## Historical challenge calibration

The frozen development pool is `experiments/challenge-v1`. Generate six proposals per model, then evaluate each accepted proposal pool and the procedural controls. Runs reject changed configuration, inputs or source. The original local runs below are already complete and preserve their original source snapshots; do not rerun them with the modified client source. Use fresh output directories for a new study. Do not run multiple local model processes simultaneously.

```sh
.venv/bin/python -m selfplay.challenges propose --backend qwen --out runs/challenge-proposals-qwen
.venv/bin/python -m selfplay.challenges solve --backend qwen --tasks experiments/challenge-v1/procedural.jsonl runs/challenge-proposals-qwen/tasks.jsonl --out runs/challenge-solve-qwen
.venv/bin/python -m selfplay.challenges solve --backend qwen-off --tasks experiments/challenge-v1/procedural.jsonl --out runs/challenge-solve-qwen-off
```

The stopped API comparison used `OPENROUTER_API_KEY` from the environment or ignored owner-only `.env`. Authentication and a structured-output plumbing call succeeded. The client pins `deepseek/deepseek-v4-pro-0813` to `deepinfra/fp8`, disables fallback, and reserves spending before each request in `runs/openrouter-api-ledger.json`. It stops at a $10 inference subcap, within the original $100 pilot cap. Unknown request outcomes halt subsequent calls and are not retried. Gateway credit-purchase fees are separate from inference costs. The previous unstarted Astra configuration remains historical and is not used for this arm.

```sh
.venv/bin/python -m selfplay.challenges propose --backend openrouter --out runs/challenge-proposals-deepseek
.venv/bin/python -m selfplay.challenges solve --backend openrouter --tasks experiments/challenge-v1/procedural.jsonl runs/challenge-proposals-qwen/tasks.jsonl runs/challenge-proposals-deepseek/tasks.jsonl --out runs/challenge-solve-deepseek --batch-size 1
.venv/bin/python -m selfplay.challenges solve --backend qwen --tasks runs/challenge-proposals-deepseek/tasks.jsonl --out runs/challenge-solve-qwen-deepseek-tasks
.venv/bin/python scripts/analyze_challenges.py --runs runs/challenge-solve-qwen runs/challenge-solve-qwen-off runs/challenge-solve-deepseek runs/challenge-solve-qwen-deepseek-tasks --out runs/challenge-deepseek-comparison.json
```

The solver sees only a public task in a fresh context. The arbiter executes plans, accepts alternative valid solutions, and checks proposal witnesses independently. The local final-answer grammar only receives action labels and horizon. Proposal validity, shortest-depth compliance, solver errors and reasoning truncation are distinct metrics. This is a frozen capability screen, not training or evidence of improvement.

Use the project virtual environment and recorded dependencies for the full suite. The original puzzle smoke tests can also run without model inference. Run from this directory:

```sh
.venv/bin/python -m unittest discover -s tests -v
python3 -m selfplay.pilot --backend oracle --out runs/new-oracle-run
python3 -m selfplay.pilot --backend random --out runs/new-random-run
```

Every output directory must be new so results cannot be overwritten accidentally.

For local Apple Silicon inference, use Python 3.12 and the recorded dependency lock:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-local.lock.txt
.venv/bin/python -m selfplay.pilot \
  --backend mlx --model mlx-community/Qwen3-4B-4bit \
  --tasks 24 --proposals 8 --attempts 2 \
  --max-tokens 256 --max-calls 400 --max-minutes 20 \
  --out runs/new-qwen4b-audit
```

MLX downloads model weights into the Hugging Face cache. Runs record the resolved model snapshot; pass `--revision` with that snapshot hash to reproduce a checkpoint. Thinking mode defaults to disabled; add `--thinking` to enable it. For the new calibration use `--prompt-version v2 --top-k 20 --temperature 0.6 --max-tokens 4096`; v2 removes the example-label anchoring cue. Reasoning tokens count toward the output cap. The current workspace already has a `.venv` containing the local dependencies.

An OpenAI-compatible chat-completions endpoint can be used with `--backend endpoint --model MODEL_ID --base-url URL`. Set `MODEL_API_KEY` in the environment if required; it is never written into run logs. This endpoint adapter has not been validated against a cloud service. Token limits and call-admission limits are not dollar caps and do not terminate rented hardware.

## Run records

Each run saves:

* `manifest.json`: arguments, model snapshot, source hashes, seeds and training status.
* `tasks.jsonl`: task specification, withheld witness, origin and canonical identity.
* `proposals.jsonl`: proposer output and validity/rejection result.
* `rollouts.jsonl`: solver prompts, outputs, correctness, failure type, token counts and latency.
* `summary.json`: presentation success rates, task-level reward scores and an exploratory paired bootstrap interval.

The model never receives the witness in a solver prompt. Proposed tasks require verified feasibility; they can still be too difficult for learning. Two samples per presentation are a calibration setting and are too few for reliable curriculum selection.

`--proposal-mode claim` checks a model's claimed goal against its witness. `--proposal-mode construct` instead derives the goal by executing the proposed path; these are separately logged protocols. The latter still rejects duplicates and trivial shortest paths. Recompute original audit statistics with `.venv/bin/python scripts/analyze_runs.py`; it excludes depth-collection runs, which use `scripts/analyze_depth.py`.

Results from initial experiments are summarized in [the pilot report](docs/pilot-results.md). Additional model calibration and actual training must be evaluated separately; changing prompts, checkpoints or budgets does not itself establish self-improvement.

## Verified dataset export

`python3 -m selfplay.dataset --run RUN_DIRECTORY --out NEW_DATASET_DIRECTORY` replays the verifier against stored model rollouts and exports one verified final answer per task. It splits by task identity, rejects oracle/random baselines as self-training sources, and records the selected identities. Use `--exclude-tasks FILE` for held-out task manifests. `--selection frontier` is a screening option; scores reuse rollout samples and therefore need independent rescoring for a fair curriculum study. The exporter does not train weights. Do not use the inspected development runs as a final evaluation set.

## Interleaved feedback

```sh
.venv/bin/python -m selfplay.feedback \
  --out runs/new-feedback-run \
  --revision 4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25 \
  --proposals 8 --attempts 2 --max-tokens 4096 --seed 3059
```

This validates each proposal, immediately obtains solver outcomes, and supplies those outcomes to the next proposal. It adapts the proposer context; weights remain frozen. Add `--feedback-mode none` for the control, which retains accepted paths but withholds explicit outcome and rejection feedback. Both arms save all outcomes for analysis. Candidate rewards based on two solver attempts require independent rescoring before curriculum selection.

## Depth-controlled training pipeline

`selfplay.depth prepare` freezes disjoint task families at exact depths 2–5. `selfplay.depth collect` can constrain plans to the public fixed horizon and legal action labels; its grammar does not use the target or a solution. Eight candidate attempts score difficulty and eight separate attempts supply supervision. `selfplay.curriculum` independently regrades responses and selects uniform and difficulty-ranked sets with matching depth counts. `selfplay.train` updates local QLoRA parameters and saves checkpoint hashes and losses.

The frozen `experiments/depth-v1` manifests are already created. Example collection command, using a **new** output directory:

```sh
.venv/bin/python -m selfplay.depth collect \
  --tasks experiments/depth-v1/candidates.jsonl \
  --out runs/new-depth-candidates --attempts 16 --batch-size 16
```

To continue an interrupted depth collection, repeat its exact configuration with `--resume`. Completed responses are reverified and skipped. A changed task manifest or inference implementation blocks resume. Training, validation and final evaluation are separate stages; see the protocol before running a comparison. The local training process uses the GPU and should be stopped before gaming if resource contention matters.

## Larger verified trajectories

`selfplay.trajectories freeze` creates fresh splits excluding previously used task families. `selfplay.trajectories build` rechecks model-generated successes, remaps them across canonical, reordered, renamed, and renamed/reordered presentations, and executes each plan to label its intermediate states. Intermediate-state labels come from the environment; they are not model-generated reasoning. The plan-only and mixed datasets share parent tasks and example/update counts, but have different supervised token counts.

The frozen `experiments/trajectory-v1` study uses `datasets/trajectory-v1`. Its training configuration is:

```sh
.venv/bin/python -m selfplay.train \
  --data datasets/trajectory-v1/plans --out adapters/NEW-plans \
  --iters 256 --seed 173 --protocol docs/trajectory-training-protocol.md
```

Use `mixed` for the auxiliary-trajectory arm and another new output directory. `scripts/evaluate_trajectories.py` runs the prescribed base/plans/mixed comparisons sequentially against the frozen splits, with checkpoint and source verification. Its output directories must be new; do not rerun completed evaluations to select a favorable result. The original pilot remains separately recorded.
