# Competitive environment suite: RL pilot protocol

Design date: 2026-10-21. This document specifies the scientific intent and safeguards for a two-cycle pilot. The executable run manifest must freeze the final numerical settings, seeds, model revision, source hashes, and checkpoint identities before scored rollouts begin. Unfilled numerical choices here are not a preregistered configuration.

## Question and scope

Can a shared, open-weight language-model policy learn from competitive interaction across a small suite, then generate new training experience with its updated weights? The immediate milestone is to complete **play → policy-gradient update → replay → another update → replay** with reproducible trajectories and independently measured performance.

This pilot tests the mechanism and searches for a learning signal. Two training cycles cannot establish continual improvement, general reasoning transfer, superiority over fixed-opponent training, or a novel algorithm. Improvements against a fixed reference population and on held-out seeds would justify a larger controlled experiment. A negative result must be reported without changing the test set or selecting an attractive checkpoint afterward.

Self-play rewards are relative. When both sides use the same policy in a symmetric zero-sum game, their average reward is zero even if both become stronger. Conversely, beating the immediate predecessor can reflect a narrow counterstrategy. The winner of one noisy game is not necessarily the stronger policy.

## Relationship to prior work

Original **AlphaGo** combined supervised learning from expert games, reinforcement learning from self-play, policy/value networks, and tree search. **AlphaZero** is the closer conceptual comparison: repeatedly generate games with the current system and improve its network using self-play outcomes and search-derived policies. Our pilot uses a pretrained LLM and policy gradients; it does not implement AlphaZero's search or value-learning procedure. AlphaZero trained a separate network for each game, whereas this experiment updates one shared model across environments. [AlphaGo](https://www.nature.com/articles/nature16961), [AlphaZero](https://arxiv.org/abs/1712.01815)

**SPIRAL** directly studies multi-turn LLM self-play with policy-gradient updates across games. Its game/role-conditioned return baselines are particularly relevant when first-move advantage and private information differ between players. The paper reports transfer from small competitive games, but also reports plateaus and substantial compute requirements. Our short, constrained-action pilot does not replicate its free-form reasoning setup or its transfer results. [SPIRAL](https://arxiv.org/abs/2506.24119)

**TextArena** supplies precedent for a common interface over many competitive text games. **Policy-Space Response Oracles** studies generalization against populations of policies rather than a single co-adapting opponent. These motivate an extensible suite and a saved checkpoint league. [TextArena](https://arxiv.org/abs/2504.11442), [PSRO](https://arxiv.org/abs/1711.00832)

Fixed game rules with changing opponents can generate a curriculum, but an unbounded curriculum is a separate claim. **POET** explicitly co-evolves environments and solutions; adding environment generation later would be an extension rather than something this pilot already achieves. [POET](https://arxiv.org/abs/1901.01753)

## Learner and action interface

Use **Qwen3-8B**, a dense 8.2B-parameter Apache-2.0 open-weight model, with the existing MLX stack and a pinned four-bit conversion. It is a practical upgrade over the earlier 4B learner while retaining an architecture already exercised locally. Local viability must be established by a forward/backward smoke check and measured memory use; it is not a guarantee of strong game play. Train LoRA adapters while retaining frozen quantized base weights. [Official Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B), [MLX conversion](https://huggingface.co/mlx-community/Qwen3-8B-4bit)

For this pilot, present each observation as text and enumerate **all legal actions** using single-token labels. Compute a categorical policy by masking to those labels and applying softmax. The environment determines legality, not strategic quality: it must not rank actions by hidden information, search value, shortest path, or reward. Do not silently prune chess moves to fit the label vocabulary; use enough verified single-token labels or explicitly change the interface.

Record the exact prompt tokens, label-to-action mapping, allowed token IDs, selected action, masked sampling log-probability, temperature, checkpoint identity, role, and environment state needed for replay. The training distribution must use the same mask and temperature as the rollout distribution. No greedy sampling, top-k filtering, or probability clipping may be introduced in one side only.

The model chooses an action directly; it does not generate a reasoning trace. This is **LLM action-policy RL**, not evidence that reasoning chains improve. A later reasoning-enabled experiment would require generated-token log-probabilities, separate reasoning budgets, and its own frozen protocol.

## Initial suite

| Environment | Competition and information | Scoring concerns |
| --- | --- | --- |
| Simulated market | Two learning traders share a market, budgets, and inventory constraints, with background order flow and private signals where specified. | Match orders reproducibly; account for cash and inventory; settle positions using an exogenous rule that cannot be inflated by the last trade. Report raw economic outcomes separately from any bounded relative RL reward. |
| Maze race with roadblocks | Each player traverses a grid and can spend a limited resource to impede the opponent. | Keep both goals reachable; specify turn order, placement costs, terminal outcomes, and horizon ties. Do not expose a shortest-path answer in the observation. |
| Chess | Standard legal moves and rules enforced by a chess library, with paired colors and a documented start-position distribution. | Distinguish real checkmate/draw from a horizon cutoff. If truncated games use material adjudication, call the metric material-adjudicated performance rather than chess win rate. |

These are compact research environments, not claims of realistic financial execution or unrestricted game mastery. The market is a simulator; learned profit there is not evidence of real-world trading skill. Agent-based market simulation has established precedent, but realism requires substantially more validation than this pilot. [ABIDES](https://doi.org/10.1145/3384441.3395986)

For future suite expansion, candidate families include sealed-bid auctions, hidden-information card games, resource allocation, negotiation, capture-the-flag, and build–break–repair programs. Each must first define an executable outcome, information boundary, and baseline opponents. Adding many nearly identical games is less informative than adding a few genuinely distinct information and planning demands.

## Frozen two-cycle experiment

Before the run, save a manifest with model repository/revision, tokenizer, runtime versions, source hashes, base checkpoint hash, LoRA layout, optimizer, learning rate, clipping, entropy or KL coefficients, temperature, batch schedule, episode limits, reward transforms, opponent mixture, and separate training/evaluation seed lists. Include the exact game counts, starting-position distributions, and checkpoint promotion rules.

1. Save the untrained policy as checkpoint 0. Run its evaluation against the frozen reference population on held-out seeds.
2. Generate the first training batch with checkpoint 0 in both seats. Keep the generating policy unchanged for the entire batch.
3. Apply the prescribed policy-gradient updates using those rollouts and save checkpoint 1. Evaluate it on the same fixed held-out protocol.
4. Generate fresh training rollouts using checkpoint 1 in both seats. This first pilot trains against the current self; archived checkpoints form an evaluation league. Historical-opponent training is a later controlled extension, and old-opponent actions in such an extension must not be treated as on-policy actions from the learner.
5. Apply the next prescribed update schedule, save checkpoint 2, and evaluate it under the same held-out protocol. Retain all checkpoints and reports regardless of outcome.

The proposed small run uses eight training episodes per environment per cycle. Both seats contribute current-policy trajectories. Evaluation compares each checkpoint against checkpoint 0 and one fixed environment heuristic using three held-out scenario seeds with swapped seats, plus a final checkpoint 1 versus checkpoint 2 comparison. These numbers are deliberately a mechanism pilot: three scenario clusters per opponent/game cannot yield a precise estimate of general improvement. The executable manifest is authoritative if the final configuration differs.

Keep distinct PRNG streams for environment generation, exogenous market events, actor sampling, action-menu permutations if used, and optimizer minibatches. An action consuming an extra random draw must not alter future private signals or environment seeds. A paired evaluation swaps model seats on the same underlying scenario; both runs count as one scenario cluster for uncertainty estimates.

The frozen reference population should include the base model and simple deterministic or seeded baseline agents appropriate to each environment. Evaluate all checkpoints against the same population, not only the newest opponent. A checkpoint cross-play matrix can expose cyclic strategies. Aggregate performance with equal weight per environment, and publish each game's results as well as the macro average.

Evaluation data must not enter policy updates, baseline fitting, prompt selection, demonstrations, or early stopping. If debugging reveals a correctness bug after evaluation, preserve the affected run, document the fix, and start a new version rather than overwriting its evidence.

## Policy-gradient update

Use the actual sampled action likelihood under the frozen behavior checkpoint. A suitable pilot objective is a clipped likelihood-ratio policy-gradient surrogate:

`ratio = exp(log_pi_current(action | observation) - log_pi_behavior(action | observation))`

`loss = -mean(min(ratio * advantage, clip(ratio, 1-epsilon, 1+epsilon) * advantage))`

Any entropy bonus or reference KL term must be recorded separately. This is an actor-only clipped policy-gradient implementation, not full actor–critic PPO or GRPO unless those additional components are actually implemented. The clipping construction follows PPO. [PPO](https://arxiv.org/abs/1707.06347)

Use terminal returns or documented discounted returns, with a separate baseline for each environment and role. Compute baselines from prior batches, an independent estimate, or leave-one-out episode returns. For leave-one-out estimation, exclude the entire current episode: the opponent's outcome from that same game is not an independent baseline. Do not treat repeated turns from one game as independent returns when fitting a baseline.

Assign equal coefficients to environments, episodes, and seats, then sum the action terms within each seat's trajectory. This is the ordinary score-function gradient of the terminal-return objective: longer trajectories legitimately contain more decision terms. Do not divide by realized trajectory length, which changes the objective when the policy can affect termination. Fixed per-environment horizon scaling would be a possible later reweighting, but this pilot does not apply it.

Include losing actions with negative advantages and winning actions with positive advantages where the baseline supports that sign. Filtering to winning trajectories and maximizing their likelihood would be a different, supervised method. Draws are legitimate data; if all returns in a batch are identical and the advantages vanish, report no learning signal rather than manufacturing a winner.

Verify before real updates that recomputed behavior log-probabilities match the rollout values; new/old ratios are one before optimization; all probabilities and gradients are finite; positive and negative advantages change selected action probabilities in the expected direction; and checkpoint save/reload preserves predictions. Stop or bound updates if KL, clipping fraction, gradient norm, or entropy shows policy collapse. Bounds must be declared before examining scored outcomes.

## Measurement and reporting

Save full simulator trajectories, observations, action mappings, behavior probabilities, outcome reasons, and model/checkpoint identifiers. Replay logs through the environment to verify recorded transitions and rewards. Simulator tests should cover conservation/accounting, information isolation, legal moves, terminal rewards, seed determinism, and role symmetry where intended.

Report per environment: outcome counts, horizon/cutoff rates, mean reward, standard uncertainty intervals, performance versus each fixed opponent, action entropy, meaningful action diversity, and training time/memory. For market games also report trades, cash, inventory, and settled wealth; for maze games report goal completion; for chess report checkmate/draw/cutoff separately. Bootstrap paired scenario clusters rather than individual turns. A tiny pilot may have wide intervals: the result can be inconclusive even when the loop works.

Do not interpret falling policy loss, rising training reward, or a successful adapter write as evidence of increased capability. A credible positive signal is a held-out gain against frozen references with no major regressions elsewhere. Transfer beyond the suite requires a separate untouched evaluation; it cannot be inferred from game rewards.

## Budget and next decision

The user-authorized external pilot ceiling remains **$100 total**, including earlier API commitments. This local MLX phase incurs no cloud rental. Existing API spending and unknown interrupted-call reservations remain in the project ledger; do not reset them. Strong API opponents or demonstrations are optional extensions and must be separately identified so self-play-only and externally assisted conditions are not conflated.

After the two cycles, decide using the saved evidence: extend the current setup if it produces diverse valid behavior and a measurable learning signal; address reward/interface bottlenecks if it does not; or scale to a GPU only after identifying a concrete throughput or model-capacity constraint. A later scientific comparison should add matched-compute fixed-opponent and no-update controls plus multiple independent training seeds.
