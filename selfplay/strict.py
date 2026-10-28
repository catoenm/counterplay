"""Syntax-only constrained decoding. The grammar never receives a task target."""
from itertools import product
import json

from .pilot import MLXBackend


class PlanGrammar:
    def __init__(self, encode, labels, max_steps, eos_ids, exact_steps=False):
        if not 1 <= max_steps <= 6 or len(set(labels)) != len(labels):
            raise ValueError("Unsupported grammar")
        self.root = {}
        self.eos_ids = tuple(eos_ids)
        self.plans = 0
        for length in ([max_steps] if exact_steps else range(max_steps + 1)):
            for plan in product(labels, repeat=length):
                ids = encode(json.dumps({"plan": plan}, separators=(",", ":")))
                node = self.root
                for token in ids:
                    node = node.setdefault(token, {})
                for eos in self.eos_ids:
                    node.setdefault(eos, {})
                self.plans += 1

    def allowed(self, prefix):
        node = self.root
        for token in prefix:
            if token not in node:
                raise ValueError("Generated tokens violate grammar")
            node = node[token]
        # MLX computes one step ahead before stream_generate consumes EOS.
        return tuple(node) if node else self.eos_ids


class StrictBackend(MLXBackend):
    def __init__(self, model, revision, adapter_path=None):
        super().__init__(model, revision, thinking=False, top_k=20)
        if adapter_path:
            from mlx_lm.tuner.utils import load_adapters
            load_adapters(self.model, adapter_path)
        self.metadata.update(adapter_path=adapter_path, protocol="fixed-horizon-plan-v1")
        self.grammars = {}

    def formatted(self, prompt):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )

    def complete_plan(self, prompt, labels, max_steps, seed, constrained=True):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        mx = self.mx
        mx.random.seed(seed)
        prompt_ids = self.tokenizer.encode(self.formatted(prompt), add_special_tokens=False)
        processors = []
        if constrained:
            key = (tuple(labels), max_steps)
            if key not in self.grammars:
                self.grammars[key] = PlanGrammar(
                    lambda s: self.tokenizer.encode(s, add_special_tokens=False),
                    labels, max_steps, self.tokenizer.eos_token_ids, exact_steps=True,
                )
            grammar = self.grammars[key]

            def constrain(tokens, logits):
                allowed = grammar.allowed(tokens[len(prompt_ids):].tolist())
                if not allowed:
                    raise ValueError("Grammar has no continuation")
                mask = mx.full(logits.shape, -float("inf"), dtype=logits.dtype)
                mask[:, mx.array(allowed)] = 0
                return logits + mask

            processors.append(constrain)
        chunks, last = [], None
        for last in stream_generate(
            self.model, self.tokenizer, prompt_ids, max_tokens=128,
            sampler=make_sampler(temp=0.6, top_p=0.95, top_k=20),
            logits_processors=processors,
        ):
            chunks.append(last.text)
        return {"text": "".join(chunks), "prompt_tokens": last.prompt_tokens,
                "completion_tokens": last.generation_tokens, "finish_reason": last.finish_reason}

    def complete_batch(self, requests):
        from mlx_lm.generate import BatchGenerator
        from mlx_lm.sample_utils import apply_top_k, apply_top_p
        mx = self.mx
        prompts, processors, samplers = [], [], []

        def keyed_sampler(seed, state):
            key = mx.random.key(seed)
            def sample(logprobs):
                nonlocal key
                key, draw_key = mx.random.split(key)
                allowed = state.get("allowed")
                if allowed is not None:
                    # Syntax mask leaves at most a few branches. Avoid sorting
                    # the entire vocabulary when sampling those same branches.
                    ids = mx.array(allowed)
                    filtered = apply_top_p(logprobs[:, ids], 0.95)
                    if len(allowed) > 20:
                        filtered = apply_top_k(filtered, 20)
                    return ids[mx.random.categorical(filtered / 0.6, key=draw_key)]
                filtered = apply_top_k(apply_top_p(logprobs, 0.95), 20)
                return mx.random.categorical(filtered / 0.6, key=draw_key)
            return sample

        def processor(grammar, prompt_length, state):
            def constrain(tokens, logits):
                allowed = grammar.allowed(tokens[prompt_length:].tolist())
                state["allowed"] = allowed
                mask = mx.full(logits.shape, -float("inf"), dtype=logits.dtype)
                mask[:, mx.array(allowed)] = 0
                return logits + mask
            return constrain

        for prompt, labels, horizon, seed, constrained in requests:
            ids = self.tokenizer.encode(self.formatted(prompt), add_special_tokens=False)
            prompts.append(ids)
            state = {}
            samplers.append(keyed_sampler(seed, state))
            if constrained:
                key = (tuple(labels), horizon)
                if key not in self.grammars:
                    self.grammars[key] = PlanGrammar(
                        lambda s: self.tokenizer.encode(s, add_special_tokens=False),
                        labels, horizon, self.tokenizer.eos_token_ids, exact_steps=True,
                    )
                processors.append([processor(self.grammars[key], len(ids), state)])
            else:
                processors.append([])
        generator = BatchGenerator(self.model, max_tokens=128,
                                   stop_tokens=[[x] for x in self.tokenizer.eos_token_ids],
                                   completion_batch_size=len(requests), prefill_batch_size=len(requests))
        uids = generator.insert(prompts, samplers=samplers, logits_processors=processors)
        outputs = {uid: [] for uid in uids}
        finished = {}
        try:
            while responses := generator.next_generated():
                for r in responses:
                    if r.finish_reason != "stop":
                        outputs[r.uid].append(r.token)
                    if r.finish_reason:
                        finished[r.uid] = r.finish_reason
        finally:
            generator.close()
        if len(finished) != len(uids):
            raise RuntimeError("Batch did not complete")
        return [{"text": self.tokenizer.decode(outputs[uid]), "prompt_tokens": len(prompts[i]),
                 "completion_tokens": len(outputs[uid]) + int(finished[uid] == "stop"),
                 "finish_reason": finished[uid]} for i, uid in enumerate(uids)]
