# Self play for language models

Research snapshot: 20 October 2026. The project asks whether competing language model agents can generate useful training experience, with a first cloud pilot capped at $100 and a larger research budget of roughly $2,000.

The direction is well supported, but the broad ideas already exist: a model proposing challenges for another model, verified self-generated problems, competitive language games, and model-generated executable environments. A useful independent contribution will need a narrower hypothesis, strong controls, and a result that survives changes of opponent or interface. Our proposed starting point is to measure whether curriculum rewards select transferable planning difficulty or incidental interface difficulty.

This is a targeted scoping review, not an exhaustive systematic review. Searches covered proposer–solver training, asymmetric self-play, games, environment generation, diversity collapse, verification, tool learning, and simulated markets. Claims below use primary papers or official proceedings. Most entries were screened at abstract level; reward and compute details for Absolute Zero and SPIRAL and validity details for R-Quest were checked in the full text. Reported gains are the authors' results, not independent replications. Dates refer to initial public release unless stated otherwise.

## What self play means here

There are several different mechanisms hiding under the same name:

* **Competitive environment training:** agents receive rewards from a game engine and update their policies. Winning supplies a grounded signal, but game performance and general reasoning are separate outcomes.
* **Asymmetric curriculum generation:** a proposer creates tasks and a solver attempts them. The proposer needs a reward for useful difficulty, and the solver needs a trustworthy correctness signal.
* **Preference self-play:** previous model outputs supply comparisons for an alignment objective, usually with external prompts, reference answers, or a preference evaluator.
* **Inference-time interaction:** debate, search, or multiple agents improve an answer without changing weights. This can improve a system but is not evidence that the underlying model learned.

An LLM starting from pretrained weights is not learning from a blank slate. “Zero data” in this literature generally concerns the post-training task data, with knowledge already present in the pretrained model and assumptions supplied by the environment.

## Foundations and automatic curricula

| Work | Mechanism and relevance | What it does not establish |
|---|---|---|
| [POWERPLAY, 2011](https://arxiv.org/abs/1112.5309) | Searches for new tasks and solver modifications that extend the repertoire while retaining previous skills. Early precedent for inventing solvable challenges. | A cheap, practical LLM training recipe. |
| [Asymmetric self-play, 2017](https://arxiv.org/abs/1703.05407) | Alice performs actions and Bob repeats or reverses the resulting task. Construction supplies a feasibility witness. | General language reasoning transfer. |
| [AlphaZero, 2017](https://arxiv.org/abs/1712.01815) | Self-play and search yield strong policies under exact game rules. | That arbitrary language competition has an equally reliable reward or converges similarly. |
| [POET, 2019](https://arxiv.org/abs/1901.01753) | Co-evolves environments and agents, transferring solutions between environment–agent pairs. | That a single adversarial proposer will maintain an open-ended curriculum. |
| [PAIRED, 2020](https://arxiv.org/abs/2012.02096) | Uses regret between protagonist and antagonist to generate solvable, challenging environments; directly addresses unsolvable worst-case curricula. | That raw solver failure is sufficient as a proposer reward. |
| [Robotic asymmetric self-play, 2021](https://arxiv.org/abs/2101.04882) | Generates manipulation goals and learns from relabeled Alice trajectories. | That expensive embodied interaction is necessary for our pilot. |

The lesson for our setup is to separate **feasibility** from **learnability**. A proposer knowing one solution proves that the problem has a solution under the verifier. It does not prove that the current solver has enough exploration probability to learn it.

## Direct LLM precedents

| Work | Main mechanism | Implication for this project |
|---|---|---|
| [SPIN, 2024](https://arxiv.org/abs/2401.01335) | Iteratively distinguishes reference responses from the model's own responses. | Useful alignment precedent, but not autonomous generation of new, externally verified problems. |
| [SPPO, 2024](https://arxiv.org/abs/2405.00675) | Self-play preference optimization for alignment. | Keep preference evaluators separate from executable correctness. |
| [SPAG, 2024, NeurIPS](https://arxiv.org/abs/2404.10642) | Trains through Adversarial Taboo, with attacker and defender roles. Reports reasoning gains. | Language games can provide training signals, but hidden-word competition is already prior art. |
| [Non-zero-sum self-play, 2024](https://arxiv.org/abs/2406.18872) | Filtered behavior cloning in Deal or No Deal under different competitive/cooperative objectives; evaluates human partners too. | Cross-play with independent partners matters more than self-play score alone. |
| [Prover–Verifier Games, 2024](https://arxiv.org/abs/2407.13692) | Helpful and deceptive provers train alongside a smaller correctness verifier, targeting checkability. | An adversarial verifier game is distinct from a task-generation curriculum; a learned judge is not an exact oracle. |
| [STP, January 2025](https://arxiv.org/abs/2502.00212) | Iterative conjecturing and proving in Lean and Isabelle, using barely provable conjectures. | Strong verifier precedent. Its reported Lean run generated 51.3B tokens, far outside this pilot's intended scale. |
| [Sol-Ver, February 2025](https://arxiv.org/abs/2502.14948) | Co-improves code and test generation. | Self-generated tests can contribute, but tests alone do not prove a general program specification. |
| [Absolute Zero, May 2025](https://arxiv.org/abs/2505.03335) | One model proposes and solves code-reasoning tasks, with executable validation and RL updates. | Closest predecessor to the original idea; the generic proposer–solver loop is not novel. |
| [SPIRAL, June 2025; ICLR 2026](https://arxiv.org/abs/2506.24119) | Multi-turn, shared-policy self-play in zero-sum games; role-conditioned advantage estimation. | Direct evidence for the game track. The paper studies TicTacToe, Kuhn Poker, and negotiation, and reports transfer to reasoning benchmarks. |
| [Self-Questioning Language Models, August 2025](https://arxiv.org/abs/2508.03682) | Asymmetric question generation with intermediate-difficulty rewards; majority-vote solver rewards or coding tests. | Closely matches “ask something the other cannot solve.” Agreement is a proxy for truth. |
| [R-Zero, August 2025](https://arxiv.org/abs/2508.05004) | Separately optimized Challenger and Solver co-evolve from the same base model. | Separate roles and capability-frontier rewards are established. |
| [Propose Solve Verify, December 2025; ICML 2026](https://proceedings.mlr.press/v306/wilf26a.html) | Formal verification supports proposal and solver expert iteration in verified code generation. | Strong evidence for reliable verification and difficulty-aware proposals. Specialized proof environments carry setup and capability costs. |

Absolute Zero's proposer reward is explicitly zero when none of its sampled solver attempts succeeds, and otherwise equals one minus the empirical success rate. This differs from rewarding failure unconditionally. Its executor grounds code tasks, while the exact correctness guarantees still depend on the task type and verification procedure. [Full-text reward definition](https://arxiv.org/html/2505.03335v3).

SPIRAL is encouraging evidence for transfer, not a guarantee that any competitive game will improve any LLM. Its reproducibility statement specifies an eight-H100 cluster. A smaller LoRA experiment would be a reduced experiment inspired by the paper, not a faithful compute-scale reproduction. [Full text](https://arxiv.org/html/2506.24119v3).

## Recent work that constrains novelty

| Work | Finding or method | Consequence |
|---|---|---|
| [Towards Understanding Self-play for LLM Reasoning, October 2025](https://arxiv.org/abs/2510.27072) | Analyzes AZR training dynamics, entropy, parameter updates and alternative rewards against other post-training methods. | Measure pass@k and collapse, not only average single-answer accuracy. |
| [R-Diverse, February 2026](https://arxiv.org/abs/2602.13103) | Identifies superficial diversity and cross-round cycling; adds persistent memory and skill-aware measurement. | “Add a novelty reward” is already covered by close prior art. |
| [Tool-R0, February 2026](https://arxiv.org/abs/2602.21320) | Co-evolves a task generator and tool-using solver. | Self-play for tool use is also established. |
| [Vocabulary Dropout, April 2026; COLM 2026](https://arxiv.org/abs/2604.03472) | Masks proposer output logits to sustain curriculum diversity. | Include simple diversity interventions before claiming a complex method is necessary. |
| [Skill Self-Play, July 2026](https://arxiv.org/abs/2607.22529) | Uses a dynamic skill controller alongside proposer and solver. | Routing across verified skills is existing work. |
| [SPADE, August 2026](https://arxiv.org/abs/2608.19197) | An LLM writes stateful executable environments and learns to act in them; hinted versus unhinted agent performance estimates regret. | “Let the model invent its own executable games” is not an unexplored contribution. Its document grounding also differs from a strict zero-external-data claim. |
| [R-Quest, October 2026](https://arxiv.org/abs/2610.04299) | Studies invalid and mathematically repetitive questions, adding validity and novelty feedback. | Even validity plus diversity has very recent direct predecessors. Learned validity judgments retain error. |

R-Quest is particularly relevant to the original “confirmation that they know the solution” requirement. It reports that answer-consistency filtering can increase the proportion of invalid questions. Its validity-aware initialization uses labeled examples, and its later model judgments should not be confused with formal verification. The paper's evaluation also uses a stronger external model for some audits. [Full text](https://arxiv.org/html/2610.04299v2).

## Practical environments

| Resource | Best use here | Limitation |
|---|---|---|
| [TextArena](https://arxiv.org/abs/2504.11442) | Reuse text-based competitive games for a SPIRAL-inspired branch. | Each game's observations, action parsing and reward need inspection. |
| [Reasoning Gym](https://arxiv.org/abs/2505.24760) | Procedural generators and verifiers; a strong random-curriculum baseline and separate evaluation families. | Procedural generation alone is not self-play. |
| Our finite planning diagnostic | Cheap exact feasibility checks, controlled shortest path length, equivalent interfaces. | A tiny hand-designed domain with no established general-reasoning transfer. |

## The simulated market idea

The strongest direct foundation is multi-agent market microstructure simulation, rather than real-stock prompting demos.

| Work | What it establishes | Relationship to our idea |
|---|---|---|
| [Market Making via RL, 2018](https://arxiv.org/abs/1804.04216) | RL market making in a limit-order-book simulator with inventory-sensitive rewards. | Supports inventory and cost-aware reward design. |
| [ABIDES, 2019](https://arxiv.org/abs/1904.12066) | Discrete-event markets with many agents, exchange messages and configurable latencies. | Candidate simulation foundation for interacting traders. |
| [Multi-Agent RL in a Realistic LOB, 2020](https://arxiv.org/abs/2006.05574) | Learns execution policies in an ABIDES-based multi-agent environment. | Strong precedent for studying actions that affect other market participants; not LLM general reasoning evidence. |
| [TradingAgents, 2024](https://arxiv.org/abs/2412.20138) | Specialized LLM roles collaborate on trading decisions. | Related agent orchestration, but does not by itself establish weight-learning through adversarial order-book self-play. |
| [ABIDES-MARL, 2025, revised August 2026](https://arxiv.org/abs/2511.02016) | Multiple learning agents and endogenous liquidity; evaluates how adaptive market makers change execution outcomes. | Closest market-environment precedent. Fixed passive liquidity can give misleading evaluations. |
| [MarketBench, April 2026](https://arxiv.org/abs/2604.23897) | Auctions based on LLM estimates of task success and token cost expose calibration problems. | Relevant to resource-allocation games; it is not a securities order-book benchmark. |

A synthetic market is attractive for partial information, budget constraints, delayed payoff, and adaptation. It is a harder first causal experiment than finite planning: luck, inventory exposure, market regimes and opponent weaknesses all affect reward. We should treat market profit and transfer to independent reasoning tasks as separate hypotheses. Details are in [the market experiment design](market-track.md).

## Recommended research question

**Does scoring proposed tasks across equivalent agent interfaces produce a curriculum that transfers better than scoring difficulty against one interface?**

This is a candidate empirical question, not an established novelty claim. Interface randomization, robust RL, domain randomization and population-based self-play provide adjacent precedents; the literature search has not established priority for the exact combination. Before a paper claim, extend citation tracing around Tool-R0, SPADE, R-Diverse and interface-robustness work.

The contribution could be an inexpensive diagnostic benchmark and a careful negative or positive result. A new optimizer is unnecessary unless an optimizer change is what the evidence actually supports. The first steps are an exact-verifier audit, a frozen-model measurement, and then a matched-budget training comparison.
