# Initial experiments and budget

The first experiment measures whether equivalent planning tasks change difficulty when tool names and ordering change. It is intended to validate a research instrument before spending the approved $100 cloud pilot budget. The local audit uses frozen weights and does not demonstrate learning.

## Hypotheses

H1: A proposer's reward against a single presentation can be materially affected by interface details that do not change the underlying task.

H2: Selecting tasks that remain learnable across multiple presentations improves performance on held-out presentations more than selecting tasks using one presentation, at matched inference and training cost.

H3, reserved for the market branch: a mixture of past opponents improves performance against an independent trader population more than training only against the latest self. General reasoning transfer requires a separate evaluation.

These are hypotheses, not findings. A null or negative result is useful if the environment, controls and compute accounting are trustworthy.

## Experiment zero implemented now

The planning world contains two integers and a small set of explicit arithmetic tools. Each operation updates one or both registers modulo a specified integer. A task supplies a start, target and step budget. Its proposer must also supply a solution sequence. The harness replays that witness, withholds it from the solver, and uses breadth-first search to measure the shortest solution.

Every solver response is an action plan. The engine checks the actual terminal state, legal action names and step budget. No model-generated code is executed and no LLM judge decides correctness. The same task is presented in four forms: canonical, reordered tools, renamed tools, and both. The public renderer excludes witnesses. A stable task ID ignores witness choice and tool ordering.

The initial run uses 24 procedurally generated tasks, balanced across shortest paths of 1, 2 and 3 steps, two attempts per presentation, and eight LLM proposal attempts. Valid proposals are added to the audit; invalid, duplicate or too-easy proposals are recorded, not silently repaired. The same frozen 0.6B model plays proposer and solver in separate contexts. This is a model/environment calibration run, not a co-evolution experiment.

The completed kickoff also evaluates a 4B model on the same tasks and a separate constructive-proposal protocol that derives the goal from the proposed action sequence. See [the pilot results](pilot-results.md): both models are near the performance floor under these short, non-thinking prompts, and the proposed minimum-based reward currently yields no retained procedural tasks. The next gate is improved calibration before paid training.

The [reasoning follow-up](reasoning-results.md) changes that calibration: thinking enabled with a larger output budget solves 22/24 attempts on six fresh tasks, versus 3/24 with thinking disabled. Actual token usage differs substantially. Most of these tasks are now at the performance ceiling, so this result does not yet supply a useful training curriculum. The [interleaved feedback diagnostic](feedback-protocol.md) tests constructive proposals under this configuration.

The implementation stores task specifications, source hashes, model snapshot, seeds, prompts, outputs, token counts, failures and elapsed time. Randomized job order reduces temporal confounding. Exact reference solutions across all presentations and an independent affine transition specification check the environment. A seed is recorded for reproducibility, but stochastic behavior may still vary across devices or backend versions.

## Rewards to compare

For empirical success rate p on a valid task, compare:

* Failure reward: 1 − p. This is the deliberately naive control and rewards tasks the solver never solves.
* Frontier reward: 4p(1 − p). This gives zero to always-solved and never-solved tasks, peaking at intermediate success.
* Robust frontier reward: the minimum frontier reward across presentations. This proposed conservative heuristic favors tasks that appear learnable across interfaces. It is not a known unbiased estimate of learning progress and may over-penalize useful tasks.

All policies require a valid witness and identical deduplication. Include a uniform selection baseline. Two attempts provide only a smoke estimate: expand to at least eight attempts in a serious selection study, use disjoint attempts for selection and reward evaluation, and report sensitivity to sampling noise. Minimum-based rewards have downward estimation bias. They should be tested against simple average rewards and ordinary presentation augmentation.

## Training experiment after calibration

Start with a 1.5B–3B open model and short contexts; the small local model is an instrumentation check. First verify model licensing, checkpoint revision, tokenizer, action-format competence and nontrivial initial success on the target domain. Choose the actual GPU from available RunPod inventory after a throughput and memory smoke run.

Stage one uses a frozen proposer and solver to produce a common candidate pool. Train separate LoRA adapters using verified successful solver responses selected uniformly, by frontier reward, or by robust frontier reward. Keep retained training tokens, optimizer steps, base checkpoint, generation budget and presentation augmentation matched. Include a fixed procedural curriculum and an untrained checkpoint. This is filtered self-training or expert iteration, not policy-gradient RL and not a novel algorithm. Do not substitute oracle trajectories in one arm unless the same supervision is supplied to all arms and explicitly labeled.

If that comparison shows a reproducible advantage, stage two lets the proposer and solver co-evolve and uses an existing RL implementation. GRPO or REINFORCE is a practical candidate; this repository does not yet implement an RL trainer. Reward baselines must be role-conditioned when roles have different return distributions. Record invalid generations and zero-variance groups, because excluding these can make results look artificially strong.

## Evaluation and decision gates

Separate development tasks, selection tasks, training tasks and final evaluation before training. Use canonical hashes to prevent exact overlap; also hold out operation combinations and longer planning horizons because different seeds alone do not guarantee meaningful generalization. Hold out tool-name vocabularies and presentation templates. Do not tune rewards or thresholds on the final test set.

Report verified pass@1, pass@k at a fixed sampling budget, invalid-action and format rates, per-depth accuracy, task-family coverage, duplicated tasks, and success per generated token. Estimate uncertainty by bootstrapping tasks, not individual responses. For actual training, run three seeds if the measured throughput permits it; otherwise label the run exploratory and reduce the number of arms before pretending a single seed is definitive.

Use separate reasoning and tool-use evaluations to support any claim beyond this domain. A small public benchmark subset can screen regressions but cannot establish broad capability gains; pretraining contamination remains possible. Improvement on the training game alone does not support a general reasoning claim.

Proceed to training only if enough valid, distinct tasks produce mixed success rather than floor or ceiling effects. If changing presentation only causes JSON-format failures, investigate parsing and action specification before attributing the effect to planning. The initial v1 renderer retains a canonical-label example in every presentation. The reasoning follow-up uses v2, which removes that example; comparisons across these prompt versions must remain separately labeled.

## Approved pilot allocation

| Component | Spending ceiling | Gate |
|---|---:|---|
| GPU setup and throughput measurement | $10 | Check memory, throughput and checkpoint storage first |
| Expanded curriculum audit and baseline generation | $25 | Enough valid tasks with nontrivial solver success |
| Small adapter comparison | $40 | Sufficient retained training trajectories in every arm |
| Independent evaluation | $15 | Frozen protocol and checkpoints |
| Storage and contingency | $10 | Total remains at or below $100 |

These are allocations, not performance or runtime estimates. RunPod's official cloud-GPU page currently lists examples around $0.59/hour for A40/A6000 and $1.09/hour for L40S, but other official pages show differing snapshots. Use the live account quote at provisioning. At an illustrative $1/hour, 40 GPU-hours cost $40 before storage; this does not predict how much training fits in 40 hours. [Official cloud GPU page](https://www.runpod.io/product/cloud-gpus), [pricing](https://www.runpod.io/pricing).

Use one GPU and sequential generation/training initially. A command's token or wall-time cap does not stop a rented pod from billing. Before any paid launch, set provider-side termination or an independently running watchdog and account for storage that persists after shutdown. Reserve funds before provisioning and terminate resources after copying results. No cloud resources have been provisioned by the initial local audit.

Fly credits can later support a lightweight coordinator or CPU environment service; a local process is sufficient for the first run. Introducing remote orchestration now would add variables without testing the research hypothesis. RunPod access is not configured in the current shell, so a cloud training launch still requires a usable authenticated account connection.

## Larger project allocation

For an illustrative total of $2,000: $100 pilot, $400 to reproduce the strongest small-model result, $900 for a focused multi-seed ablation study, $400 for independent evaluations, and $200 reserve. Choose either the curriculum or market branch after the pilot rather than fragmenting the budget across several underpowered studies. No portion beyond the $100 pilot has been committed.
