# Reasoning calibration protocol

20 October 2026. This follow-up tests whether enabling Qwen3's reasoning mode supplies usable verified trajectories before weight training. The previous frozen-model audit is development evidence, not a training baseline under the new prompt.

## Fixed comparison

Use the same pinned Qwen3 4B 4-bit checkpoint for both arms, the same 12 fresh tasks generated with seed 2029, four attempts per task and two presentations (canonical and renamed/reordered). Use temperature 0.6, top-p 0.95, top-k 20 and a 1536-token output cap in both arms. The only between-arm generation change is the thinking switch. Both arms use prompt v2, which removes the canonical-label example and states the final-answer format without forbidding intermediate reasoning.

This matches the maximum per-call output budget, not actual token consumption. Report actual completion tokens and success per generated token alongside accuracy. Qwen's model card documents the thinking switch and recommends non-greedy sampling for reasoning. [Official model card](https://huggingface.co/Qwen/Qwen3-4B).

The exact checker grades only the final answer after the closing reasoning delimiter. Truncated outputs and unfinished reasoning receive separate failure categories; no plan is extracted from unfinished reasoning. Save the source snapshot with each run as well as its hashes. Reuse paired seeds, but do not assume that token sampling is identical across different templates.

## Gates fixed before reading results

The calibration passes only if the reasoning-enabled arm has verified success on at least six distinct tasks, at least four tasks have mixed successes and failures in the canonical presentation, and fewer than 25% of responses are truncated. These are pragmatic pilot gates, not statistical significance criteria.

If truncation dominates, inspect a small development sample and test a separately labeled larger budget. If almost all tasks are solved, increase depth. If both modes remain near chance, compare simpler tasks or explicit execution feedback before weight updates. Do not interpret inference-time improvement as self-play training.

Once calibration passes, measure the validity and diversity of constructive LLM proposals with reasoning enabled. Generate a separate training pool and a frozen evaluation set; reserve all already-inspected tasks for development. A small adapter-training experiment requires enough independent verified tasks and a matched supervision baseline. No cloud spending or weight update is triggered merely by a higher calibration score.

## Adaptive budget diagnostic

The decision to stop the initial 1536-token run was made after eight completed responses, six of which were truncated and two solved. One more successful response completed before the interrupt took effect. The final saved record therefore contains nine completed responses (three solved, six truncated) and a tenth interrupted call. This incomplete, adaptively stopped sample is not the planned 96-response comparison and will not be used to estimate a population accuracy difference.

The next complete comparison uses six fresh tasks with seed 2031, two attempts and the same two presentations (24 responses per mode), with a 4096-token cap in both modes. Other sampling settings and the prompt remain unchanged. This is a small budget diagnostic, not a replacement claim of having met the original gates. Actual token consumption and truncation remain explicit outcomes. Reasoning and non-reasoning share the same pinned checkpoint; neither mode trains weights.
