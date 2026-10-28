# Larger verified trajectory training protocol

20 October 2026. Follow-up to the negative 20-task pilot. Test whether a larger set of independently verified model plans, augmented tool presentations, and auxiliary state-trajectory prediction improves held-out planning. All work is local under the existing $100 cloud cap; no cloud resource is needed.

## Frozen data and collection

Use the same pinned Qwen3 4B 4-bit model, disabled thinking, prompt v2, exact shortest horizons 2–5, legal-action decoder, temperature 0.6, top-p 0.95 and top-k 20. Freeze 1,536 candidate tasks, 128 validation tasks, 128 in-domain test tasks (modulus 7) and 64 transfer test tasks (modulus 11). All splits are balanced by shortest depth. Exclude all task families appearing in any earlier experiment or run. Family identity ignores budget/witness and collapses x/y coordinate exchange. Training-derived prefixes must also be checked against held-out families; this pilot uses full parent trajectories, not new planning subproblems.

Collect two canonical base-model attempts per candidate and validation task. Use one earliest verified successful response per task. Do not use a reference solver's plan as supervision, repair failed outputs, or train on inspected old test tasks. No difficulty ranking in this experiment: the previous study did not establish its benefit. Uniformly sample within each depth using seed 173. Allocate up to 128 tasks across depths with a round-robin capacity limit, requiring at least eight per depth. If fewer than 128 are available, accept at least 96 (a multiple of four). If the feasibility gate fails, explicitly expand collection to four attempts before training, preserving the first run. Require at least eight separately solved validation tasks, otherwise expand validation to four attempts. Any further adaptation must be recorded before training.

## Presentation and trajectory supervision

Map each selected model plan back to environment actions, replay it, and verify its goal and horizon. Render four equivalent presentations: canonical, reordered, renamed, renamed/reordered. Remap action labels exactly and independently verify every augmented answer. All variants of a parent stay in one split. The canonical plan is model-generated; the other presentations are deterministic verified augmentation, not additional independent model discoveries.

For an auxiliary example, supply the starting state, action definitions and that successful proposed plan, and ask for the state after each action. Omit the target state from this simulation prompt. Execute each action to label the complete state trajectory. These are environment-generated intermediate labels, not a claim that the model produced correct reasoning. Record parent task, source rollout and presentation for every example.

Compare two adapters starting from the same frozen base:

- **Plans:** two copies of each of the four presentation-specific plan examples per parent.
- **Mixed trajectories:** one copy of each plan example plus one corresponding full state-trajectory example.

Both sets contain eight examples per parent and exactly the same distinct parent tasks and plan variants. The mixed arm spends half its example budget on state simulation instead of repeating a plan. This tests that allocation under a fixed update budget; it does not isolate auxiliary supervision from reduced direct-plan repetition. Answer-token counts and sequence lengths will differ, so report them and runtime rather than claim equal FLOPs or token budgets.

## Training

Use rank-8, scale-16 QLoRA on the last eight layers, Adam 1e-4, batch size 4, no dropout, gradient checkpointing, max sequence length 512, seed 173, and masked assistant cross-entropy through EOS. One full pass over each dataset: 256 updates if 128 parent tasks, or twice the parent count otherwise. Use the same shared direct-plan validation set for both arms, with final full validation loss at batch size one. Record unique parents separately from augmented example counts, target/input tokens, exact source and dataset hashes, finite losses and nonzero updated weights. Reject truncation. Use only the final checkpoint; no final-test-based checkpoint choice or hyperparameter changes.

## Evaluation and interpretation

Evaluate the base and both final adapters on the fresh ID and transfer tasks, four attempts per task, in canonical and renamed/reordered presentations, identical settings and paired seeds. Direct-plan inference is unchanged in all arms: auxiliary trajectories are a training task, not extra inference-time reasoning. Primary outcome is canonical ID verified pass@1 averaged over the four attempts. Report pass@4, each depth, presentation robustness, modulus transfer, diversity, actual output tokens and paired task-bootstrap intervals. All outputs must be independently regraded and checkpoint hashes checked.

This is a single-training-seed screening experiment. The larger-data arms use more total updates than the previous pilot; compare against the frozen base on the new shared test set and do not interpret a change from the old test's percentages as an isolated effect. Apparent gains require replication across seeds. Neither a positive result nor verified trajectory generation would establish coevolving self-play RL or broad reasoning improvement. A null result is a completed experiment; preserve it and do not spend the cloud budget trying to tune against this test set.
