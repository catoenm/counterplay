# Adding an arena environment

The suite shares one action-selection policy across games. A game supplies observations, a complete legal-action menu, deterministic transitions, and terminal returns. The runner and trainer are in [`selfplay/arena/experiment.py`](../selfplay/arena/experiment.py) and [`policy.py`](../selfplay/arena/policy.py); the current registry is [`selfplay/arena/__init__.py`](../selfplay/arena/__init__.py).

## Implemented contract

There is no required base class. Implement these members on a two-player environment class:

| Member | Required behavior |
| --- | --- |
| `__init__(seed: int, regime='train', max_rounds=...)` | Start a nonterminal episode. Use a local seeded random generator. Define the horizon's unit explicitly: market/maze rounds or chess plies. `train` and `eval` currently share a distribution with disjoint seeds; `ood` changes the distribution. |
| `name`, `done`, `turn` | A stable game name, terminal boolean, and integer step count starting at zero. Increment `turn` once per accepted action batch. |
| `active_players() -> list[int]` | Exactly the players who must act now, using IDs `0` and `1`. Return `[0, 1]` for simultaneous decisions or one ID for alternating turns. Return `[]` after termination. |
| `observation(player) -> dict` | JSON-serializable rules and the state this player may see. Include enough history for the intended information structure. Do not return an internal mutable state object. |
| `legal_actions(player) -> list[dict]` | All legal choices for an active player, with deterministic ordering and no duplicates. The current policy supports **1–256 actions**; inactive or terminal players receive `[]`. Each action must be JSON-serializable. |
| `step(actions: dict[int, dict]) -> None` | Require exactly one legal action from every active player. Validate the entire batch before any mutation. Reject invalid batches and calls after termination. Define simultaneous resolution, conflicts, and ties explicitly. |
| `returns() -> list[float]` | Available only after termination: two finite values in `[-1, 1]`, summing to zero. The current implementations raise before termination. Separate true terminal outcomes from horizon adjudication. |
| `metrics() -> dict` | JSON-serializable diagnostics and raw scores that explain the bounded return. Include a termination reason where applicable. Do not describe positive shaped returns as game wins. |
| `snapshot() -> dict` | A JSON-serializable replay state, including the horizon and relevant hidden state/history. It must support exact comparison with a reconstruction from the constructor and recorded actions. No `restore()` method is currently required. |
| `heuristic_action(player) -> dict` | A reproducible, legal fixed evaluation opponent. Derive its action only from information available to that player; never read hidden future outcomes. |

The runner shuffles each legal menu with a separate deterministic random stream, maps it to verified single-token labels, and samples one label. Legality must not encode strategic advice through ranked/pruned actions or hidden-state-dependent labels. `ActionPolicy.tokens()` currently rejects prompts over **3,072 tokens**, and the runner rejects nonterminal episodes exceeding **100 steps**. A game with an enormous action space, longer horizon, or free-form code actions needs an explicit interface/protocol change before use.

## Replay and information boundaries

An episode records its specification and `initial_state`, then each step's actions and resulting snapshot in `events`. Model `transitions` separately record observations, prompts, shuffled menus, checkpoint identity, chosen actions, and rollout log probabilities. Final returns and metrics close the record. `play()` writes one completed episode per JSONL line and refuses an existing output file.

Only `observation(player)` and `legal_actions(player)` enter the model prompt. Snapshots and evaluator metrics are not policy inputs. For example, market snapshots contain both private signals, settlement value, and the exogenous event schedule; the trader sees only its own signal and public market information. Maze shortest-path distances belong in evaluator metrics, not its observation. A replay may expose hidden state to a human, but must label that view and never feed it back to the policy.

Replay must reproduce every state from `(seed, regime, horizon, actions)`, independently of batching, player-dictionary insertion order, or which checkpoint chose those actions. Preserve random-event schedules across counterfactual policies when possible. Never use global unseeded randomness, wall-clock time, or Python's process-dependent string hash to determine transitions. JSON round-trips convert integer dictionary keys to strings; normalize them during comparison and convert action player IDs back to integers before stepping.

## Register a fourth game

Do this for a **new experiment**, without changing sources or rules during a frozen run.

1. Add `selfplay/arena/<game>.py` and focused tests. Keep simulator imports independent of model loading and paid services.
2. Append its stable name to `ENVIRONMENTS`, then add a `make_env()` branch with the chosen horizon. Registry order currently contributes to scenario seeds; reordering existing names changes their seed allocation.
3. Update the experiment's frozen `horizons`, protocol, dependencies, and any new regime/sampling settings. `game_specs()` automatically schedules eight training episodes per registered game per cycle and paired-seat evaluation, so a fourth game increases work and changes the equal-game training objective. Keep training/evaluation scenario families disjoint, including symmetries or mirrored variants. OOD regimes exist but are not scheduled by the current pilot.
4. Ensure `freeze()` archives every source and asset needed for reconstruction. It currently copies top-level `selfplay/arena/*.py` and the protocol; nested modules, generated maps, or external assets need explicit archival and hashes. Use a fresh output directory.
5. Extend [`scripts/analyze_arena.py`](../scripts/analyze_arena.py) for the game's raw metrics, privacy checks, and scenario-family identity. Review assumptions such as the current six environment/opponent evaluation cells. Its core transition replay uses the archived factory and otherwise follows the common contract.
6. Extend [`scripts/export_arena_visual.py`](../scripts/export_arena_visual.py): register the class and horizon field in `verify_and_compact()`, then add a compact state projection. In the `selfplay-arena.html` fragment identified by that script's `DEFAULT_FRAGMENT`, add `gameNames`, a renderer branch in `draw()`, and action/result labels as needed. The current fallback renderer is market-specific; a fourth game will not render correctly without this change. Preserve source-derived states and the 1 MB embedded-data cap.

## Validation before model rollouts

Use the existing [`test_arena_games.py`](../tests/test_arena_games.py) and [`test_arena_market.py`](../tests/test_arena_market.py) as examples. Check deterministic resets and full replay; all action batches either succeed completely or leave the state unchanged; every active state has legal actions; random and heuristic episodes terminate; rewards are finite, bounded, and zero-sum; snapshots/observations survive JSON round-trips; and a player's observation and legal menu do not change when only inaccessible information changes. Verify actual game invariants independently where possible, such as cash/inventory conservation or maze reachability. Include outcome, draw, timeout, and adversarial invalid-action cases.

Also inspect prompt lengths and the legal-menu maximum before loading the learner. A game that always draws or gives identical returns cannot supply the current trainer's required nonzero advantage signal. Check signal availability without tuning against held-out evaluation seeds.

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_arena_*.py' -v
.venv/bin/python scripts/analyze_arena.py --run runs/NEW-arena-run
.venv/bin/python scripts/export_arena_visual.py --run runs/NEW-arena-run
```

The audit requires a completed run by default; `--allow-partial` explicitly audits available records without claiming completion. The visual exporter rechecks exact states using the current simulator classes and therefore refuses incompatible historical records. The analysis script uses hash-verified archived simulator sources. Test the replay controls and layout at 736, 360, and 320 pixels after adding a renderer.

## What this pilot does—and what a population league would add

The current run performs two cycles. In each cycle, **both training seats use the same current checkpoint**, then its on-policy rollouts update LoRA weights. Evaluation compares each checkpoint with the frozen base policy and fixed heuristics; a final comparison faces checkpoint 2 against checkpoint 1. Historical checkpoints are evaluation opponents, not a sampled training population. The learner selects a legal action directly; it does not generate or train a reasoning trace.

A later population-training experiment needs an explicit opponent mixture, matchmaking and checkpoint-retention rules, frozen evaluation, and a treatment of opponent-generated actions: those actions must not be trained as if sampled from the current learner. Adding games does not itself create an open-ended curriculum, establish continual improvement, or demonstrate transfer outside the suite. Keep those hypotheses separate from the completed play/update/replay mechanism.
