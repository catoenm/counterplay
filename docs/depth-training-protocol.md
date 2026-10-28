# Depth-controlled constrained-plan training pilot

20 October 2026. The user authorized depth-controlled tasks, strict action formatting and actual training under the existing $100 cloud cap. Use local Apple Silicon first. This protocol precedes training and final-test evaluation; calibration and candidate statistics may inform feasibility decisions, which must be recorded.

## Task and inference controls

Use the pinned Qwen3 4B 4-bit snapshot from the previous experiments. Thinking is disabled in **all** main comparison arms. This first training pilot targets the inexpensive direct-plan policy; it does not compare its accuracy to the earlier reasoning-enabled runs. Sample at temperature 0.6, top-p 0.95, top-k 20 with a 128-token cap. Keep prompt v2.

The final strict decoder allows precisely compact JSON containing `plan`, legal tool names, and exactly the public fixed horizon. It does not receive the target, starting state, witness or reference solution. It permits every action combination of that length, including wrong plans, and the ordinary verifier still decides whether a plan succeeds. Fixed length is part of this action-space constraint. Both formats append an explicit fixed-horizon instruction to prompt v2. The free-format calibration control has the same prompt, seeds and token cap, with no logit mask.

Freeze 16 calibration, 256 candidate, 32 validation, 64 in-domain test and 32 out-of-domain test tasks in `experiments/depth-v1` before collecting data. Balance exact shortest depths 2, 3, 4 and 5. Set each task's step budget equal to its shortest depth, eliminating budget slack as a confound. In-domain tasks use modulus 7; out-of-domain tasks use modulus 11. Exclude all prior development-task families. Split families ignore witness and budget and collapse the exact x/y coordinate-swap symmetry. This blocks those overlaps, not every possible semantic similarity. Record manifest hashes.

## Collection and curriculum

First calibrate strict versus free formatting with four attempts on each of sixteen tasks. Keep any implementation-aborted run explicitly incomplete; do not pool it into the comparison.

Calibration adaptation before candidate collection: the first implementation run stopped on an EOS look-ahead integration error, which was fixed and tested. The completed variable-length calibration (`depth-format-calibration-v2`) produced empty plans on all 64 strict calls. Its free-format arm solved 1/64. Because every generated task has shortest length equal to its public budget, the next calibration explicitly requires that exact length in both prompts and constrains it in the strict decoder. This is a revised action protocol, not a hidden repair of answers; the earlier run remains separate. Training and final evaluation will use only the final fixed-horizon protocol.

Collect sixteen strict canonical attempts per candidate: attempts 0–7 estimate empirical difficulty, and 8–15 independently supply possible verified training responses. Do not extract an oracle plan or repair model outputs. Rescore retained tasks using those disjoint attempts and report selection optimism. An empirical frontier score is `4*p*(1-p)`; its sample uncertainty remains material even with eight samples.

Build two small training sets from tasks with an independently verified successful response: uniform sampling and difficulty-ranked selection. Match the number of tasks at each shortest depth across arms, using the same candidate pool and one successful response per selected task. Prefer up to 32 tasks per arm, with at least 16 distinct tasks across multiple depths as the minimum training feasibility gate. The ranked arm requires positive selection-stage difficulty. If the pool is too small, expand generation or revise the experiment explicitly; do not fabricate supervision. Overlap between the two training arms is permitted and must be reported.

Generate validation responses on the frozen validation split separately. Use the same verified validation examples for both adapters. Selection and supervision samples are separated, but conditioning on successful supervision still selects solvable tasks; this is not an unbiased estimate of all task difficulty.

## Actual parameter training

Use MLX-LM QLoRA on the existing quantized model: rank 8, scale 16, dropout 0, last eight transformer layers, Adam learning rate 0.0001, batch size 4, 80 updates, seed 71, sequence cap 512, and gradient checkpointing. Use masked final-answer cross-entropy. Construct the exact non-thinking prompt tokens used at inference; do not accidentally switch the training chat template back to thinking mode. Record losses, target-token counts, adapter tensors and checkpoint hashes. Abort on non-finite loss or changed task/data manifests. Choose the final 80-update checkpoint in both arms, without choosing by final-test results.

Implementation details fixed before training: use 32 tasks per arm if feasible, otherwise 20 or 16, so each arm completes an integer number of dataset passes in 80 batches of four. Match depth counts to match supervised target-token counts; report actual input tokens separately. The loss mask includes assistant tokens through EOS and excludes padding. This uses a custom loss callback with the installed MLX trainer. Full validation loss before and after training uses batch size one to include every validation example.

Collection and evaluation use batches of sixteen requests with per-request keyed sampling, so random state is not shared between tasks. Sample only the grammar-permitted token branches after applying the same temperature/top-p/top-k rules. This avoids sorting the entire vocabulary when only a few legal tokens remain. The calibration's initial sequential sampler has the same distributional settings, but its random draws are not bit-for-bit identical to the keyed batch sampler. A local pause saved 208 candidate responses; continuation reverified them and skipped completed job identities. No paused wall time is counted as inference time.

This is verified filtered self-training / expert iteration using procedurally generated challenges. It updates real parameters but is not policy-gradient RL, adversarial proposer training, or evidence of a novel method. See the [MLX-LM fine-tuning documentation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md). The [Qwen model card](https://huggingface.co/Qwen/Qwen3-4B) documents thinking control and the Apache 2.0 license. The local installed library is the implementation used; dependencies are locked in the repository.

## Evaluation fixed before training

Evaluate the frozen base and both final adapters on the same untouched test tasks, strict decoding, four independent samples per task, and canonical plus renamed/reordered presentation. Use the same seeds across model conditions. Primary outcome: in-domain canonical verified pass@1 estimated by the mean of four samples per task. Report all depths, renamed/reordered robustness and modulus-11 transfer separately. Report task-cluster paired bootstrap intervals, pass@4, output-token usage and the number of unique tasks solved. A small single-training-seed pilot cannot establish a reliable curriculum advantage or general LLM reasoning transfer.

Also report training and validation behavior without treating low cross-entropy as evidence of task success. No retry or checkpoint selection based on final evaluation. A null or negative result is a completed experiment. Cloud spending remains bounded by the approved $100 cap; no paid resources are needed for the planned local run.
