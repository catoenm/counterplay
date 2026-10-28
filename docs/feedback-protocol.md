# Frozen proposer–solver feedback diagnostic

20 October 2026. This tests whether the constructive task generator and exact verifier support an interleaved two-role loop. Both roles use the pinned Qwen3 4B 4-bit checkpoint with reasoning enabled, temperature 0.6, top-p 0.95, top-k 20 and a 4096-token cap. No weights change.

Each arm attempts eight proposals. A proposal supplies a starting state, four allowed actions and a witness path; the environment computes its target. We require modulus 7, a two-to-four-step limit, an exact shortest solution of at least two steps, and a previously unseen task identity. Every accepted task receives two solver attempts. The solver sees the target and tool rules but never the witness. Worst-case model calls are 24 per arm.

The feedback arm places the verified outcomes and `4*p*(1-p)` score into the next proposer context. Invalid proposals receive zero and a rejection explanation. The control supplies an empty feedback list. Both retain earlier accepted paths for diversity, so the control does receive implicit information about validity. This comparison withholds explicit outcome/rejection feedback; it does not remove all proposer memory. Common sampling settings, seed 3059 and proposal counts are held fixed. Tasks and token consumption can diverge between arms.

The control was added while the feedback run was underway, before inspecting its round outcomes. This is an exploratory development comparison, not a preregistered study. Report accepted distinct tasks, shortest depths, solver successes, mixed-outcome tasks, rejection reasons and actual tokens. Eight proposals and two attempts per task cannot reliably estimate learning-frontier difficulty or demonstrate a feedback benefit. One early sample can steer the whole trajectory; repeated independent runs are needed.

A task with one success and one failure earns the maximum empirical score, but this does not imply a true 50% success rate. Raw empirical frontier rewards are biased with two samples. Independently rescore candidates before using the reward for training-data selection. Distinguish failures caused by truncation or formatting from incorrect plans.

The archived source for the feedback arm predates the optional control flag. Its prompt is exactly the default `outcomes` branch of the subsequent implementation. Each run saves its own source snapshot and hashes. The first proposer prompt and first solver results should coincide under the shared seed; subsequent rounds need not.

Passing this diagnostic means the loop produces verified challenges and trajectories. Establishing self-play learning requires parameter updates, matched procedural/static-curriculum training controls, and evaluation on untouched tasks and external environments. None of those claims follows from a functioning frozen loop.
