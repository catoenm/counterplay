"""Thinking-enabled, two-turn policy sampling with exact token provenance.

No pretrained model is imported until explicit CUDA policy construction. Tool
messages are input context only; both sampled completions receive the trajectory
advantage. Frozen cardinal checkpoint identities and numerical tolerances apply.
"""
from __future__ import annotations

import math
import time

from selfplay.program_cardinal_cuda import CardinalCudaPolicy
from scripts.cloud import train_program_rl as common


def flatten_trajectories(trajectories, advantages):
    if (not 1 <= len(trajectories) <= 8 or len(trajectories) != len(advantages)
            or any(type(a) not in (int, float) or not math.isfinite(a) for a in advantages)):
        raise ValueError('One finite advantage per independent trajectory required')
    records, scaled = [], []
    for trajectory, advantage in zip(trajectories, advantages):
        turns = trajectory.get('turns')
        if not isinstance(turns, list) or len(turns) != 2:
            raise ValueError('Exactly two unmodified sampled turns per trajectory required')
        records.extend(turns)
        # The inherited accumulation divides by number of records. Multiplying
        # by two makes this sum(log p(draft)+log p(revision))/N trajectories.
        scaled.extend([2 * float(advantage)] * 2)
    return records, scaled


class DebugCudaPolicy(CardinalCudaPolicy):
    def template(self, messages):
        if isinstance(messages, str):
            messages = [{'role': 'user', 'content': messages}]
        if (not isinstance(messages, list) or not messages
                or any(set(m) != {'role', 'content'} or m['role'] not in ('user', 'assistant')
                       or not isinstance(m['content'], str) for m in messages)):
            raise ValueError('Explicit canonical user/assistant messages required')
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=True)
        if not rendered.rstrip().endswith('<think>'):
            raise ValueError('Tokenizer did not open the thinking-enabled generation prefix')
        return rendered, self.tokenizer.encode(rendered, add_special_tokens=False)

    def sample_batch(self, prompt, seeds):
        contexts = prompt['batch_messages'] if isinstance(prompt, dict) else [prompt] * len(seeds)
        return self.sample_messages_batch(contexts, seeds)

    def sample_messages_batch(self, messages_batch, seeds):
        torch, cfg = self.torch, self.config
        if (not isinstance(seeds, (tuple, list)) or not 1 <= len(seeds) <= 8
                or len(messages_batch) != len(seeds)
                or any(type(s) is not int or not 0 <= s < 2**63 for s in seeds)):
            raise ValueError('One to eight explicit contexts and independent seeds required')
        templates = [self.template(messages) for messages in messages_batch]
        if any(not ids or len(ids) + cfg['max_new_tokens'] > cfg['max_context_tokens'] for _, ids in templates):
            raise ValueError('Prompt plus complete response reserve exceeds context')
        device = getattr(self, '_sample_device', 'cuda')
        width = max(len(ids) for _, ids in templates)
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.eos[0]
        ids = torch.tensor([[pad] * (width-len(row)) + row for _, row in templates],
                           dtype=torch.long, device=device)
        mask = torch.tensor([[0] * (width-len(row)) + [1] * len(row) for _, row in templates],
                            dtype=torch.long, device=device)
        generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
        sampled, logps = [[] for _ in seeds], [[] for _ in seeds]
        finish, elapsed = ['length'] * len(seeds), [None] * len(seeds)
        active, counts, cache = list(range(len(seeds))), [], None
        started = time.monotonic()
        self.model.eval()
        with torch.no_grad():
            for _ in range(cfg['max_new_tokens']):
                if time.monotonic()-started >= cfg['sample_seconds']:
                    for i in active:
                        finish[i] = 'time'
                    break
                counts.append(len(active))
                positions = (mask.cumsum(-1)-1).masked_fill(mask == 0, 0)[:, -ids.shape[1]:]
                output = self.model(input_ids=ids, attention_mask=mask, position_ids=positions,
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
                lp = torch.log_softmax(output.logits[:, -1].float(), dim=-1)
                if not bool(torch.isfinite(lp[active]).all()):
                    raise ValueError('Nonfinite sampled distribution')
                chosen = torch.full((len(seeds),), self.eos[0], dtype=torch.long, device=device)
                for i in active:
                    chosen[i] = torch.multinomial(lp[i].exp(), 1, generator=generators[i])[0]
                values = lp.gather(1, chosen[:, None]).squeeze(1).cpu().tolist()
                tokens = chosen.cpu().tolist()
                cache = output.past_key_values
                del output, lp
                remaining = []
                for i in active:
                    sampled[i].append(tokens[i]); logps[i].append(float(values[i]))
                    if tokens[i] in self.eos:
                        finish[i], elapsed[i] = 'eos', time.monotonic()-started
                    else:
                        remaining.append(i)
                active = remaining
                if not active:
                    break
                ids = chosen[:, None]
                mask = torch.cat([mask, torch.ones((len(seeds), 1), dtype=mask.dtype, device=device)], dim=1)
        del cache
        duration = time.monotonic()-started
        records = []
        for i, seed in enumerate(seeds):
            rendered, prefix = templates[i]
            records.append({'text': self.tokenizer.decode(sampled[i], skip_special_tokens=True),
                'raw_text': self.tokenizer.decode(sampled[i], skip_special_tokens=False),
                'messages': messages_batch[i], 'prompt_template': rendered, 'prompt_ids': prefix,
                'completion_ids': sampled[i], 'sampled_token_logprobs': logps[i],
                'seed': seed, 'finish_reason': finish[i], 'input_tokens': len(prefix),
                'output_tokens': len(sampled[i]), 'elapsed_seconds': elapsed[i] or duration,
                'eos_token_ids': list(self.eos), 'sampling_batch_size': len(seeds),
                'sampling_batch_row': i, 'thinking_enabled': True,
                'left_padding_tokens': width-len(prefix)})
        return {'records': records, 'timing': {'batch_elapsed_seconds': duration,
            'total_output_tokens': sum(map(len, sampled)), 'active_sequences': len(seeds),
            'active_sequences_by_step': counts, 'batch_steps': len(counts),
            'finished_eos': finish.count('eos'), 'constant_cache_batch_size': len(seeds)}}

    def update_trajectories(self, trajectories, advantages):
        records, scaled = flatten_trajectories(trajectories, advantages)
        before = self._optimizer_identity()
        self.verify_frozen()
        report = common.CudaPolicy.update(self, records, scaled)
        after = self._optimizer_identity()
        if after['step'] != before['step'] + 1:
            raise ValueError('Trajectory optimizer did not advance exactly one step')
        return {**report, 'adam_step_before': before['step'], 'adam_step_after': after['step'],
            'optimizer_state_sha256': after['state_sha256'], 'frozen_state_sha256': self.verify_frozen(),
            'trajectory_count': len(trajectories), 'sampled_turn_count': len(records),
            'objective': 'Mean trajectory advantage times sum of both sampled completion log probabilities',
            'tool_feedback_target_tokens': 0}
