"""Pure CPU, fixed signal and resource admission for the proposed study.

These functions do not create reservations, select replacement samples, run a
model, or change a deadline. A passed screen alone never authorizes training.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import math

from .cuda_policy import digest

SIGNAL_CONTRACT = {"screen_episodes": 64, "hard_episodes": 32, "easy_episodes": 32,
    "easy_success_min": 4, "easy_success_max": 28, "easy_pairs": 16,
    "mixed_pair_min": 4, "pairing": "eight preassigned pairs within each easy depth"}
COST_CONTRACT = {"training_updates": 80, "training_episodes": 160, "evaluation_episodes": 320,
    "decisions_per_cohort": 24, "decisions_per_episode": 12, "safety_factor": 1.5,
    "aggregate_seconds": 43200, "protected_seconds": 3600,
    "total_usd": 90., "admission_usd": 15., "main_usd": 65., "reserve_usd": 10.}


def build_context_bound(tokenizer_directory):
    """CPU-only finite-schema token upper bound for the unchanged public view.

    The envelope maximizes serialized byte length; it is deliberately not a
    valid task trajectory. For this exact ASCII + byte-level-BPE tokenizer,
    token count cannot exceed the number of rendered UTF-8 bytes. The envelope's
    *actual* token count is descriptive and is never used as the upper bound.
    The returned proof hash is an integrity identifier, not peer-review approval.
    The worker must freeze this receipt and its independent review before draws.
    """
    from copy import deepcopy
    import hashlib
    import inspect
    import json
    from pathlib import Path
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import AutoTokenizer
    from selfplay import environment
    from selfplay.arena_mixed import planning, cuda_policy as renderer
    from . import envs

    directory = Path(tokenizer_directory)
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.iterdir()) if p.is_file()}
    for name in ("tokenizer.json", "tokenizer_config.json", "config.json"):
        if name not in files:
            raise ValueError("Original pinned tokenizer files required")
    raw = json.loads((directory/"tokenizer.json").read_text())
    model, pre = raw.get("model", {}), raw.get("pre_tokenizer", {})
    components = pre.get("pretokenizers", [])
    if (raw.get("normalizer") != {"type": "NFC"} or model.get("type") != "BPE"
            or model.get("dropout") is not None or model.get("unk_token") is not None
            or model.get("continuing_subword_prefix", "") or model.get("end_of_word_suffix", "")
            or model.get("byte_fallback") is not False
            or not set(ByteLevel.alphabet()) <= set(model.get("vocab", {}))
            or pre.get("type") != "Sequence" or len(components) != 2
            or components[0].get("type") != "Split" or components[0].get("behavior") != "Isolated"
            or components[0].get("invert") is not False
            or components[1] != {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}
            or raw.get("post_processor") != {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}):
        raise ValueError("Tokenizer no longer satisfies the byte-count upper-bound proof")
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    labels = list("ABCD")
    if any(len(tokenizer.encode(x, add_special_tokens=False)) != 1 for x in labels):
        raise ValueError("Expected original first four one-token labels")
    protocol = envs.protocol_config()
    if (envs.PlanningEnv.observation is not planning.PlanningEnv.observation
            or envs.PlanningEnv.legal_actions is not planning.PlanningEnv.legal_actions
            or envs.PlanningEnv.step is not planning.PlanningEnv.step
            or protocol["modulus"] != 11 or protocol["horizons"] != {"planning": 12}
            or protocol["roles"] != {"planning": [0]}):
        raise ValueError("New planning wrapper changed the bounded public schema")
    # All generated tasks use four tools; coordinates are modulo 11. Source
    # identity plus tests bind these finite domain invariants before execution.
    tool_names = sorted(environment.RULES,
        key=lambda name: (len(renderer.canonical({name: environment.RULES[name]})), name), reverse=True)[:4]
    tool_names.sort()
    longest_action = max(tool_names, key=lambda name: len(renderer.canonical({"tool": name})))
    example = planning.PlanningEnv(0, "train", 12)
    obs = example.observation(0)
    obs.update(turn=11, modulus=11, round_limit=12, start=[10, 10], state=[10, 10], goal=[10, 10],
        tools={name: environment.RULES[name] for name in tool_names}, done=False,
        history=[{"turn": 11, "action": {"tool": longest_action}, "before": [10, 10], "after": [10, 10]}
            for _ in range(11)])
    actions = [{"tool": name} for name in tool_names]
    prompt = renderer.action_prompt(obs, actions, labels)
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    sentinel = "PLANNING_BOUND_SENTINEL"
    shell = tokenizer.apply_chat_template([{"role": "user", "content": sentinel}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if not rendered.isascii() or shell.count(sentinel) != 1:
        raise ValueError("Expected one ASCII prompt insertion in native template")
    prefix, suffix = shell.split(sentinel)
    if rendered != prefix+prompt+suffix:
        raise ValueError("Template content transformation is outside the proof")
    maximum = len(rendered.encode("utf-8"))
    actual = len(tokenizer.encode(rendered, add_special_tokens=False))
    if actual > maximum or maximum > 4096:
        raise ValueError("Conservative context envelope exceeds declared context capacity")
    # The menu's four names appear once each, while the same four rules appear
    # once each. Maximize their *joint* contribution, not just rule text alone.
    from itertools import combinations
    candidates = []
    for names in combinations(sorted(environment.RULES), 4):
        candidate = deepcopy(obs)
        candidate["tools"] = {name: environment.RULES[name] for name in names}
        longest = max(names, key=lambda name: len(renderer.canonical({"tool": name})))
        candidate["history"] = [{**item, "action": {"tool": longest}} for item in obs["history"]]
        text = renderer.action_prompt(candidate, [{"tool": name} for name in names], labels)
        rendered_candidate = prefix+text+suffix
        candidates.append((len(rendered_candidate.encode()), rendered_candidate, candidate, names))
    maximum, rendered, obs, names = max(candidates, key=lambda row: (row[0], row[1]))
    actual = len(tokenizer.encode(rendered, add_special_tokens=False))
    if not rendered.isascii() or actual > maximum or maximum > 4096:
        raise ValueError("Final finite-domain context bound failed")
    source_files = {"selfplay/environment.py": Path(inspect.getfile(environment)),
        "selfplay/arena_mixed/planning.py": Path(inspect.getfile(planning)),
        "selfplay/arena_mixed/cuda_policy.py": Path(inspect.getfile(renderer)),
        "selfplay/planning_curriculum/envs.py": Path(inspect.getfile(envs)),
        "selfplay/planning_curriculum/admission.py": Path(__file__)}
    proof = {"kind": "planning_curriculum_public_context_bound_v1",
        "maximum_input_tokens": maximum, "token_upper_bound": "rendered ASCII byte length; BPE merges cannot increase byte symbols",
        "envelope_actual_tokens_descriptive_only": actual,
        "envelope_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "envelope_observation": obs, "envelope_menu_tools": list(names),
        "domain": {"modulus": 11, "coordinate_min": 0, "coordinate_max": 10,
            "maximum_live_turn": 11, "maximum_history_rows": 11, "tools_per_task": 4,
            "tool_subsets_checked": len(candidates), "labels": labels,
            "synthetic_envelope_is_an_episode": False},
        "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source_files.items()},
        "tokenizer_files_sha256": files,
        "renderer_sha256": hashlib.sha256(Path(inspect.getfile(renderer)).read_bytes()).hexdigest(),
        "tokenizer_sha256": digest(files),
        "method": "quadratic_context_upper_envelope",
        "justification": "All observations/actions are finite ASCII JSON: at most 11 history rows, coordinates 0..10, four of seven fixed tools. Enumerate all 35 tool subsets at maximum scalar widths, then add exact unchanged native-template overhead. NFC preserves ASCII and byte-level BPE with a complete byte alphabet, no inserted prefix, and no special postprocessor cannot create more tokens than input bytes. Scale observed per-decision work quadratically to this conservative token bound; retain fixed overhead and 1.5x margin. This is an engineering cost envelope, not a universal GPU wall-time guarantee."}
    proof["proof_sha256"] = digest(proof)
    return proof


def _positive(value, name, *, zero=False):
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError("Invalid finite cost/timing: " + name)
    return float(value)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(x in "0123456789abcdef" for x in value)


def _instant(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timezone-aware aggregate clock required")
    return result


def signal_gate(episodes, screen_specs):
    """Validate the complete fixed screen, then evaluate its predeclared gate.

    Episode replay/transition correctness is separately verified by the worker
    and independent auditor. This function binds the decision to those complete
    saved episode objects and the exact frozen specification list.
    """
    if len(episodes) != 64 or len(screen_specs) != 64:
        raise ValueError("All 64 unchanged-base screen episodes are required")
    ids = [s["episode_id"] for s in screen_specs]
    if len(set(ids)) != 64 or [e["episode_id"] for e in episodes] != ids:
        raise ValueError("Missing, reordered, duplicate or replacement screen episode")
    if len({s["scenario_family_id"] for s in screen_specs}) != 64:
        raise ValueError("Screen families must be unique")
    hard = [s for s in screen_specs if s.get("screen_group") == "hard"]
    easy = [s for s in screen_specs if s.get("screen_group") == "easy"]
    if (Counter(s["depth"] for s in hard) != {3: 8, 4: 8, 5: 8, 6: 8}
            or Counter(s["depth"] for s in easy) != {1: 16, 2: 16}
            or screen_specs != hard+easy or any(s.get("pair_id") is not None for s in hard)):
        raise ValueError("Frozen hard/easy depth balance/order differs")
    expected_pair_ids = {f"easy-depth{d}-pair{i:02d}" for d in (1, 2) for i in range(8)}
    pairs = {}
    base = episodes[0].get("behavior_binding")
    if not isinstance(base, dict) or base.get("version") != 0 or base.get("checkpoint_id") != "base":
        raise ValueError("Screen must use the unchanged base")
    outcomes = {}
    for episode, spec in zip(episodes, screen_specs, strict=True):
        if (spec.get("environment") != "planning" or spec.get("learner_seats") != [0]
                or spec.get("split") != "screen" or spec.get("opponent") != "none"
                or episode.get("schedule") != spec
                or episode.get("environment") != "planning" or episode.get("depth") != spec["depth"]
                or episode.get("scenario_family_id") != spec["scenario_family_id"]
                or episode.get("scenario_sha256") != spec["scenario_sha256"]
                or episode.get("learner_seats") != [0] or episode.get("behavior_binding") != base
                or episode.get("returns") not in ([0., 0.], [1., 0.])
                or not 1 <= len(episode.get("transitions", [])) <= 12):
            raise ValueError("Screen episode/spec/base binding differs")
        outcomes[spec["episode_id"]] = int(episode["returns"][0])
        if spec["screen_group"] == "easy":
            pairs.setdefault(spec.get("pair_id"), []).append(spec)
    if set(pairs) != expected_pair_ids:
        raise ValueError("Missing fixed easy pair")
    pair_records = []
    for depth in (1, 2):
        depth_specs = [s for s in easy if s["depth"] == depth]
        for index in range(8):
            pair_id = f"easy-depth{depth}-pair{index:02d}"
            selected = pairs[pair_id]
            if selected != depth_specs[2*index:2*index+2] or any(s["depth"] != depth for s in selected):
                raise ValueError("Easy pairs must be fixed consecutive within-depth pairs")
            rewards = [outcomes[s["episode_id"]] for s in selected]
            pair_records.append({"pair_id": pair_id, "depth": depth,
                "episode_ids": [s["episode_id"] for s in selected], "returns": rewards,
                "reward_contrast": rewards[0] != rewards[1]})
    successes = sum(outcomes[s["episode_id"]] for s in easy)
    contrast = sum(p["reward_contrast"] for p in pair_records)
    passed = 4 <= successes <= 28 and contrast >= 4
    return {"kind": "planning_curriculum_signal_gate_v1", "passed": passed,
        "contract": SIGNAL_CONTRACT, "episodes_sha256": digest(episodes),
        "screen_specs_sha256": digest(screen_specs), "base_binding": base,
        "easy_successes": successes, "easy_episodes": 32, "mixed_easy_pairs": contrast,
        "easy_pairs": pair_records,
        "hard_successes_descriptive_only": sum(outcomes[s["episode_id"]] for s in hard),
        "hard_episodes": 32,
        "timing_probe_episode_ids": [s["episode_id"] for s in hard if s["depth"] == 6][:2],
        "failure_reasons": ([] if passed else
            (["easy_success_outside_4_to_28"] if not 4 <= successes <= 28 else [])+
            (["fewer_than_four_mixed_pairs"] if contrast < 4 else [])),
        "training_authorized": False}


def timing_gate(profile, sampling_measurements, context_bound, budget):
    """Price all fixed work, giving no discount for early success or overlap.

    ``sampling_measurements`` are raw choose_profiled results (rows+timing).
    ``context_bound`` is a pre-reviewed CPU renderer/tokenizer receipt containing
    maximum_input_tokens, method='quadratic_context_upper_envelope', justification,
    and renderer_sha256/tokenizer_sha256/review_sha256. It must be frozen before
    inference. Quadratic scaling is an explicit conservative engineering model,
    not a proof of a wall-time upper bound; the independent hard stop still binds.

    ``budget`` carries aware aggregate_started_at/checked_at/hard_stop, actual
    admission_spent_usd, GPU/guard/storage hourly rates, other_reserved_cost_usd,
    and fixed_main_overhead_seconds (all model loads/switches/checkpoint reloads,
    CPU replay, IPC, status publication). The latter is reviewed prelaunch,
    bound by overhead_review_sha256; zero/omitted overhead is rejected.
    """
    if (profile.get("kind") != "planning_curriculum_restored_full_step_probe_v1"
            or profile.get("passed") is not True or profile.get("training_use") is not False
            or profile.get("retained_optimizer_steps") != 0 or profile.get("temporary_optimizer_steps") != 1
            or profile.get("state_restored") is not True or profile.get("state_before") != profile.get("state_after")
            or profile.get("temporary_checkpoint", {}).get("optimizer_step") != 1
            or profile.get("gate", {}).get("passed") is not True):
        raise ValueError("One successful fully restored real-step probe required")
    if not 2 <= profile.get("decisions", 0) <= 24:
        raise ValueError("Probe action coverage differs")
    mature = profile.get("mature_state_roundtrip", {})
    if (mature.get("passed") is not True or mature.get("state_before") != mature.get("state_after")
            or mature.get("state_after", {}).get("optimizer", {}).get("step") != 1
            or mature.get("optimizer_steps") != 0 or mature.get("model_forwards") != 0):
        raise ValueError("Exact mature-Adam state-only roundtrip is required for pricing")
    _positive(mature.get("seconds"), "mature state roundtrip")
    if (context_bound.get("method") != "quadratic_context_upper_envelope"
            or not isinstance(context_bound.get("justification"), str) or len(context_bound["justification"]) < 20
            or any(not _sha(context_bound.get(k)) for k in ("renderer_sha256", "tokenizer_sha256", "review_sha256"))
            or type(context_bound.get("maximum_input_tokens")) is not int
            or not 1 <= context_bound["maximum_input_tokens"] <= 4096):
        raise ValueError("A frozen reviewed longest-prompt cost receipt is required")
    maximum_tokens = context_bound["maximum_input_tokens"]

    def scaled(seconds, tokens):
        _positive(seconds, "physical elapsed")
        if type(tokens) is not int or not 1 <= tokens <= maximum_tokens:
            raise ValueError("Observed context is outside reviewed maximum")
        return seconds * max(1., (maximum_tokens/tokens)**2)

    sampling_units, sampling_overheads, sampled_rows = [], [], 0
    screen_decisions = {}
    for raw in sampling_measurements:
        rows, measurement = raw["rows"], raw["timing"]
        forwards = measurement["forward_timings"]
        if (measurement.get("kind") != "planning_screen_sampling_timing_v1" or not rows
                or len(forwards) != len(rows)
                or measurement.get("row_sha256") != [digest(r) for r in rows]):
            raise ValueError("Screen timing must bind every actual sampled row")
        elapsed = _positive(measurement["elapsed_seconds"], "sampling batch")
        overhead = _positive(measurement["nonforward_seconds"], "sampling overhead", zero=True)
        total_forward = sum(_positive(x["seconds"], "sampling forward") for x in forwards)
        if not math.isclose(elapsed, overhead+total_forward, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError("Sampling phase timings do not cover elapsed time")
        for row, forward in zip(rows, forwards, strict=True):
            if len(row["input_ids"]) != forward["input_tokens"] or forward["training"] or forward["grad_enabled"]:
                raise ValueError("Sampling timing shape/mode differs")
            if row.get("environment") != "planning" or row.get("player") != 0:
                raise ValueError("Screen timings must describe planning learner decisions")
            key = (row["episode_id"], row["turn"])
            if key in screen_decisions or type(row["turn"]) is not int or not 0 <= row["turn"] < 12:
                raise ValueError("Duplicate or out-of-cap screen decision timing")
            screen_decisions[key] = row
            sampling_units.append(scaled(forward["seconds"], forward["input_tokens"]))
        # Entire logical-batch nonforward overhead is charged to each future
        # decision, rather than assuming the same batching efficiency persists.
        sampling_overheads.append(overhead)
        sampled_rows += len(rows)
    episode_ids = {key[0] for key in screen_decisions}
    if len(episode_ids) != 64 or any(
            sorted(turn for episode, turn in screen_decisions if episode == name) !=
            list(range(sum(episode == name for episode, turn in screen_decisions))) for name in episode_ids):
        raise ValueError("Complete contiguous sampling timings for all 64 screen episodes required")
    sample_unit = max(sampling_units)+max(sampling_overheads)
    gate_units = {False: [], True: []}
    gate_forwards = profile.get("gate_forward_timings", [])
    if len(gate_forwards) != 2*profile["decisions"]:
        raise ValueError("Full two-mode likelihood timing coverage required")
    for row in gate_forwards:
        if type(row.get("training")) is not bool or row.get("grad_enabled") != row["training"]:
            raise ValueError("Likelihood timing modes differ")
        gate_units[row["training"]].append(scaled(row["seconds"], row["input_tokens"]))
    if any(len(v) != profile["decisions"] for v in gate_units.values()):
        raise ValueError("Both gate modes must cover every decision")
    gate_elapsed = _positive(profile["gate_seconds"], "whole gate")
    gate_overhead = max(0., gate_elapsed-sum(row["seconds"] for row in gate_forwards))
    update = profile["temporary_update"]
    backward_units, covered = [], []
    for batch in update["microbatch_timings"]:
        if len(batch["row_indices"]) != len(batch["input_tokens"]) or not 1 <= len(batch["row_indices"]) <= 2:
            raise ValueError("Physical/logical backward timing coverage differs")
        covered.extend(batch["row_indices"])
        # A complete <=2-row microbatch cost is charged to one future action.
        # No optimistic average over the pair or shorter realized episode.
        backward_units.append(scaled(batch["forward_backward_seconds"], min(batch["input_tokens"])))
    if covered != list(range(profile["decisions"])):
        raise ValueError("Full backward timing must cover the complete probe")
    fixed_step = _positive(update["optimizer_verification_seconds"], "Adam and verification")
    checkpoint = _positive(profile["checkpoint_seconds"], "checkpoint export")
    # Include any measured update work outside the timed microbatches and step.
    update_elapsed = _positive(update["elapsed_seconds"], "whole update")
    update_overhead = max(0., update_elapsed-fixed_step-sum(x["forward_backward_seconds"] for x in update["microbatch_timings"]))
    learner_unit = max(backward_units)+max(gate_units[False])+max(gate_units[True])
    cohort_seconds = 24*(sample_unit+learner_unit)+fixed_step+checkpoint+gate_overhead+update_overhead
    evaluation_seconds = 12*sample_unit
    fixed_main = _positive(budget["fixed_main_overhead_seconds"], "reviewed full main overhead")
    if not _sha(budget.get("overhead_review_sha256")):
        raise ValueError("Frozen model-load/CPU/reload overhead review is required")
    main_seconds = 1.5*(80*cohort_seconds+320*evaluation_seconds+fixed_main)
    required_remaining = main_seconds+3600
    start, checked, stop = (_instant(budget[k]) for k in ("aggregate_started_at", "checked_at", "hard_stop"))
    elapsed, remaining, allocation = (checked-start).total_seconds(), (stop-checked).total_seconds(), (stop-start).total_seconds()
    if elapsed < 0 or allocation <= 0 or allocation > 43200:
        raise ValueError("Aggregate twelve-hour clock differs or was reset")
    rates = {k: _positive(budget[k], k, zero=k=="storage_hourly_rate")
        for k in ("gpu_hourly_rate", "guard_hourly_rate", "storage_hourly_rate")}
    hourly = sum(rates.values())
    spent = _positive(budget["admission_spent_usd"], "actual admission cost", zero=True)
    other = _positive(budget["other_reserved_cost_usd"], "other reserved liability", zero=True)
    main_usd = main_seconds/3600*hourly
    reserve_usd = hourly+other
    total_usd = spent+main_usd+reserve_usd
    failures = []
    for condition, name in ((spent <= 15, "admission_dollar_cap"), (main_usd <= 65, "main_dollar_cap"),
            (reserve_usd <= 10, "protected_dollar_cap"), (total_usd <= 90, "aggregate_dollar_cap"),
            (required_remaining <= remaining, "immutable_aggregate_deadline")):
        if not condition:
            failures.append(name)
    return {"kind": "planning_curriculum_timing_gate_v1", "passed": not failures,
        "contract": COST_CONTRACT, "profile_sha256": digest(profile),
        "sampling_measurements_sha256": digest(sampling_measurements),
        "context_bound_sha256": digest(context_bound), "budget_sha256": digest(budget),
        "timing_model": "Conservative observed full-work/context envelope; no overlap or zero-signal discount; hard deadline remains authoritative",
        "sampled_screen_decisions": sampled_rows, "sample_decision_seconds": sample_unit,
        "learner_decision_seconds": learner_unit, "full_cohort_seconds_before_margin": cohort_seconds,
        "full_eval_episode_seconds_before_margin": evaluation_seconds,
        "fixed_main_overhead_seconds_before_margin": fixed_main,
        "main_seconds_with_margin": main_seconds, "protected_seconds": 3600,
        "required_remaining_seconds": required_remaining, "remaining_seconds": remaining,
        "elapsed_seconds": elapsed, "checked_at": budget["checked_at"], "hard_stop": budget["hard_stop"],
        "admission_spent_usd": spent, "projected_main_usd": main_usd,
        "protected_and_other_usd": reserve_usd, "projected_total_usd": total_usd,
        "failure_reasons": failures, "training_authorized": False}
