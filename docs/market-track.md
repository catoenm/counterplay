# Market self play experiment

The proposed market is a synthetic strategy environment for learning under partial information, limited capital and competing agents. Two trainable LLM policies interact through a shared order book alongside scripted or frozen traders. This directly extends the user's idea of awarding positive and negative trajectory rewards based on economic outcomes.

ABIDES supplies a well-established event-driven simulation foundation, while ABIDES-MARL explicitly studies several learning agents and endogenous liquidity. These support the environment choice. They do not establish that market training improves general LLM reasoning. [ABIDES](https://arxiv.org/abs/1904.12066), [ABIDES-MARL](https://arxiv.org/abs/2511.02016).

## Smallest informative version

Use one synthetic asset, integer price ticks, a 20–50-step episode and equal initial cash and inventory for the two learners. Start without borrowing or shorting. Actions are submit limit buy, submit limit sell, cancel own order, and hold, with bounded quantities. Process crossing orders using price–time priority, and randomize seat assignment and event ordering across paired episodes.

The book must reserve cash for outstanding buys and inventory for outstanding sells; otherwise an agent can overspend by submitting several individually affordable orders. Canceling releases only the remaining reservation. Support partial fills, reject self-trades, and settle all fees consistently. Match both agents' observation windows, action budgets and latency assumptions.

Background traders should include a fixed spread market maker, noisy liquidity traders and traders responding to noisy private signals of a synthetic terminal asset value. That terminal value is sampled independently of the learners' final trade and revealed only for settlement. Different trader mixes and signal qualities form held-out regimes. No external market feed is needed.

Before LLM rollouts, test cash-plus-fee conservation, inventory conservation, reservation accounting, cancel ownership, partial fills, price–time priority, nonnegative resources and observation boundaries. Compare known scripted strategies, including no-trade. An always-profitable random trader or a strategy exploiting its own last price is a simulator diagnosis, not a learning success.

## Reward and credit assignment

Let cash C already include transaction fees, inventory be q, and common terminal value be V. Define excess terminal wealth relative to holding initial inventory as:

`R_i = [C_i,T + q_i,T * V_T - (C_i,0 + q_i,0 * V_T)] / W_i,0`

where `W_i,0 = C_i,0 + q_i,0 * V_0` is a positive normalization. This removes passive gains from the initial endowment. Do not subtract fees twice. Report both this return and any inventory-risk-shaped training reward separately. Value remaining inventory at the common exogenous terminal value, not an easily manipulated last transaction price. This is a proposed synthetic-game objective, not a claim about real-market profitability.

For an isolated transfer between two traders at a common settlement value, gains and losses offset before fees. With other traders, aggregate learner P&L need not be zero-sum: both learners can profit from background traders or both can lose. If strict competition is desired, use the relative score `(R_A − R_B)/2` and its negative, while retaining absolute returns in evaluation. Relative score alone can reward losing less badly.

The winner's trajectory can receive positive return and the loser's negative return, as proposed. Policy-gradient training should use an advantage such as `A_i = R_i − b(role, regime)`; a positive raw reward can still have negative advantage if it was worse than the baseline. A zero terminal reward can have positive advantage. At first use a role-conditioned return baseline, then assess whether a learned value function is worth its complexity.

Do not pair the winner's and loser's transcripts as a DPO preference merely because one made money: they may have seen different private information and action opportunities. Comparisons need the same decision context or a defensible counterfactual protocol. Also, a good trade can lose due to stochastic price movement; evaluate over many paired seeds.

## Experimental comparisons

Compare a frozen model, training against scripted traders, latest-opponent self-play, and a mixture of frozen historical opponents. Hold capital, token budget, market seeds and updates constant. Evaluate each policy against an independent fixed league and under swapped seats. Then test held-out volatility, fees, liquidity and signal quality. Run separate episodes for each evaluation pairing so policies are compared on common random scenarios without sharing hidden information.

Primary outcomes are excess terminal wealth, drawdown, inventory exposure, fees, fill rate, illegal actions and performance by opponent and regime. Self-play win rate alone is insufficient: two equally strong or equally weak policies can both yield 50% in symmetric play. Retain absolute scores against fixed bots and record payoff matrices to detect cyclic strategies.

For the language-learning question, freeze a separate set of planning, probability and resource-allocation problems before training. Improvement on those held-out tasks is required to support transfer. A successful market policy with no external gain is still useful game research, but it answers a narrower question.

## Place in the project

This track is currently a design, not an implemented exchange or completed trading experiment. Reuse an audited simulator when possible; do not spend the initial budget building exchange infrastructure unless existing engines cannot support the chosen test. Complete the inexpensive planning audit first, then decide whether the market branch or proposer–solver branch offers the clearest measurable result.
