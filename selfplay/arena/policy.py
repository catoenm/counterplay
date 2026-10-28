"""A trainable language-model policy over a game's public legal-action menu.

Only the output token representing a selected action is sampled and optimized.
No heuristic, future state, terminal reward or hidden opponent observation enters
the policy. This pilot does not train free-form reasoning or tool-generation.
"""
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np


def action_prompt(observation, actions, labels):
    return ('You are playing a competitive game. Maximize your final game reward. '
            'Use only your observation; the opponent chooses its own actions. '
            'Choose one action from the legal menu. Reply with only its exact label.\n'
            'Observation: ' + json.dumps(observation, separators=(',', ':')) + '\nLegal actions:\n' +
            '\n'.join(f'{labels[i]}: ' + json.dumps(a, separators=(',', ':')) for i, a in enumerate(actions)))


def episode_advantages(episodes):
    """Leave-one-episode-out baseline, conditioned on environment and player.

    Independent episode returns, not repeated transitions, define each baseline.
    Each trajectory retains the Monte Carlo terminal return (gamma=1).
    """
    groups = {}
    for e in episodes:
        for player in (0, 1):
            groups.setdefault((e['environment'], player), []).append((e['episode_id'], e['returns'][player]))
    result = {}
    for key, values in groups.items():
        for identity, reward in values:
            others = [r for i, r in values if i != identity]
            baseline = sum(others) / len(others) if others else 0.0
            result[(identity, key[1])] = reward - baseline
    return result


def clipped_objective(logp, old_logp, advantages, weights, entropy, old_kl,
                      clip=0.2, entropy_coef=0.01, kl_coef=0.02):
    import mlx.core as mx
    ratio = mx.exp(logp - old_logp)
    surrogate = mx.minimum(ratio * advantages, mx.clip(ratio, 1-clip, 1+clip) * advantages)
    return mx.mean(weights * (-surrogate - entropy_coef * entropy + kl_coef * old_kl))


class ActionPolicy:
    def __init__(self, model_config, seed=71021):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.tuner.utils import linear_to_lora_layers
        from mlx_lm.tuner.trainer import grad_checkpoint
        from huggingface_hub import snapshot_download
        self.mx = mx
        mx.random.seed(seed)
        self.path = snapshot_download(model_config['model'], revision=model_config['revision'])
        self.model, self.tokenizer = load(self.path)
        self.model.freeze()
        linear_to_lora_layers(self.model, 8, {'rank': 8, 'scale': 16.0, 'dropout': 0.0,
                                           'keys': ['self_attn.q_proj', 'self_attn.v_proj']})
        grad_checkpoint(self.model.layers[0])
        self.model.eval()
        import string
        candidates = list(string.ascii_uppercase + string.ascii_lowercase)
        candidates += sorted((s for s in self.tokenizer.get_vocab() if s.isascii() and s.isalpha()
                              and 2 <= len(s) <= 6), key=lambda s:(len(s), s))
        self.labels, self.label_ids = [], []
        for label in candidates:
            token = self.tokenizer.encode(label, add_special_tokens=False)
            if len(token) == 1 and token[0] not in self.label_ids and self.tokenizer.decode(token) == label:
                self.labels.append(label); self.label_ids.append(token[0])
            if len(self.labels) == 256: break
        if len(set(self.label_ids)) != 256:
            raise ValueError('Action labels must be distinct tokens')
        self.checkpoints = {}
        self.active = None
        self.save_memory('base')
        self.active = 'base'
        from mlx.utils import tree_flatten
        self.metadata = {**model_config, 'method': 'clipped Monte Carlo policy gradient with LoRA',
                         'action_policy': 'one-token categorical distribution over legal action labels',
                         'labels': self.labels, 'label_token_ids': self.label_ids,
                         'thinking': False, 'temperature': 1.0, 'lora_rank': 8, 'lora_scale': 16,
                         'lora_last_layers': 8, 'lora_keys': ['q_proj', 'v_proj'],
                         'trainable_parameters': sum(v.size for _,v in tree_flatten(self.model.trainable_parameters()))}

    def save_memory(self, name):
        from mlx.utils import tree_flatten
        self.mx.eval(self.model.trainable_parameters())
        self.checkpoints[name] = [(k, self.mx.array(v)) for k, v in tree_flatten(self.model.trainable_parameters())]

    def activate(self, name):
        if self.active != name:
            if name not in self.checkpoints:
                raise ValueError('Unknown checkpoint ' + name)
            self.model.load_weights(self.checkpoints[name], strict=False)
            self.mx.eval(self.model.trainable_parameters())
            self.active = name

    def tokens(self, prompt):
        formatted = self.tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = self.tokenizer.encode(formatted, add_special_tokens=False)
        if len(ids) > 3072:
            raise ValueError(f'Observation exceeds pilot context cap: {len(ids)} tokens')
        return ids

    def arrays(self, rows):
        mx = self.mx
        inputs = [r.get('input_ids') or self.tokens(r['prompt']) for r in rows]
        width = max(map(len, inputs))
        pad = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        tokens = mx.array([x + [pad]*(width-len(x)) for x in inputs])
        lengths = mx.array([len(x)-1 for x in inputs])
        counts = mx.array([len(r['actions']) for r in rows])
        return tokens, lengths, counts

    def log_probs(self, model, tokens, lengths, counts):
        mx = self.mx
        hidden = model.model(tokens)
        last = hidden[mx.arange(tokens.shape[0]), lengths]
        logits = model.model.embed_tokens.as_linear(last) if model.args.tie_word_embeddings else model.lm_head(last)
        logits = logits[:, mx.array(self.label_ids)].astype(mx.float32)
        mask = mx.arange(256)[None, :] < counts[:, None]
        logits = mx.where(mask, logits, -1e9)
        return logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    def choose(self, requests, checkpoint):
        self.activate(checkpoint)
        self.model.eval()
        rows = [{**r, 'input_ids': self.tokens(r['prompt'])} for r in requests]
        arrays = self.arrays(rows)
        lp = self.log_probs(self.model, *arrays)
        self.mx.eval(lp)
        result = []
        for i, row in enumerate(rows):
            logp = np.array(lp[i, :len(row['actions'])])
            probs = np.exp(logp.astype(np.float64)); probs /= probs.sum()
            pick = int(np.random.default_rng(row['sample_seed']).choice(len(probs), p=probs))
            result.append({'action_index': pick, 'action': row['actions'][pick],
                           'action_label': self.labels[pick],
                           'old_log_prob': float(logp[pick]), 'old_log_probs': logp.tolist(),
                           'entropy': float(-(probs * np.log(probs)).sum()),
                           'prompt_tokens': len(row['input_ids']), 'checkpoint': checkpoint})
        return result

    def save(self, directory, name):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        self.save_memory(name); self.active = name
        weights = dict(self.checkpoints[name])
        self.mx.save_safetensors(str(directory / 'adapters.safetensors'), weights)
        meta = {**self.metadata, 'checkpoint': name,
                'adapter_sha256': hashlib.sha256((directory / 'adapters.safetensors').read_bytes()).hexdigest()}
        (directory / 'metadata.json').write_text(json.dumps(meta, indent=2))
        return meta

    def train(self, episodes, source, target, out, seed, batch_size=4, learning_rate=2e-5):
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten
        mx = self.mx
        self.activate(source)
        self.model.train()
        before = {k: np.array(v) for k,v in tree_flatten(self.model.trainable_parameters())}
        adv = episode_advantages(episodes)
        rows = []
        counts = {}
        for e in episodes:
            counts[e['environment']] = counts.get(e['environment'], 0) + 1
        for e in episodes:
            for r in e['transitions']:
                if r['checkpoint'] != source:
                    raise ValueError('Training data is not sampled from the rollout policy')
                rows.append({**r, 'advantage': adv[(e['episode_id'], r['player'])],
                             'episode_weight': 1 / (len(counts) * counts[e['environment']] * 2),
                             'input_ids': self.tokens(r['prompt'])})
        if not rows or not any(abs(r['advantage']) > 1e-8 for r in rows):
            raise ValueError('No nonzero on-policy learning signal')
        # Importance weighting estimates equal environment/episode/seat objectives:
        # the score-function gradient sums over actions; no length-dependent baseline.
        for r in rows: r['weight'] = r['episode_weight'] * len(rows)
        random.Random(seed).shuffle(rows)
        optimizer = optim.Adam(learning_rate=learning_rate)
        out = Path(out); out.mkdir(parents=True, exist_ok=False)
        metrics = []

        def loss_fn(model, tokens, lengths, counts, picked, old, advantage, weights, old_all):
            logps = self.log_probs(model, tokens, lengths, counts)
            selected = logps[mx.arange(tokens.shape[0]), picked]
            mask = mx.arange(256)[None, :] < counts[:, None]
            probs = mx.exp(logps)
            entropy = -mx.sum(mx.where(mask, probs * logps, 0), axis=-1)
            old_probs = mx.exp(old_all)
            kl = mx.sum(mx.where(mask, old_probs * (old_all-logps), 0), axis=-1)
            loss = clipped_objective(selected, old, advantage, weights, entropy, kl)
            ratio = mx.exp(selected-old)
            return loss, (mx.mean(kl), mx.mean(entropy), mx.mean(mx.abs(ratio-1) > .2))

        value_grad = nn.value_and_grad(self.model, loss_fn)
        # Ensure recomputation matches rollout probabilities before any updates.
        for offset in range(0, min(len(rows), 16), batch_size):
            batch = rows[offset:offset+batch_size]
            lp = self.log_probs(self.model, *self.arrays(batch)); mx.eval(lp)
            error = max(abs(float(lp[i, r['action_index']])-r['old_log_prob']) for i,r in enumerate(batch))
            if error > 2e-3:
                raise ValueError(f'Old-policy likelihood mismatch: {error}')
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset+batch_size]
            args = self.arrays(batch) + (mx.array([r['action_index'] for r in batch]),
                    mx.array([r['old_log_prob'] for r in batch]), mx.array([r['advantage'] for r in batch]),
                    mx.array([r['weight'] for r in batch]),
                    mx.array([r['old_log_probs'] + [-1e9]*(256-len(r['old_log_probs'])) for r in batch]))
            (loss, stats), grads = value_grad(self.model, *args)
            grads, norm = optim.clip_grad_norm(grads, 1.0)
            mx.eval(loss, stats, norm)
            if not all(math.isfinite(float(v)) for v in [loss,norm,*stats]):
                raise FloatingPointError('Nonfinite RL loss or gradient')
            optimizer.update(self.model, grads)
            mx.eval(self.model.trainable_parameters(), optimizer.state)
            item = {'update': len(metrics)+1, 'loss': float(loss), 'kl_to_rollout': float(stats[0]),
                    'entropy': float(stats[1]), 'clip_fraction': float(stats[2]), 'gradient_norm': float(norm)}
            metrics.append(item)
            with (out / 'metrics.jsonl').open('a') as f: f.write(json.dumps(item)+'\n')
            if len(metrics) % 10 == 0: print(json.dumps({'training':target,**item}),flush=True)
        self.model.eval()
        delta = sum(float(np.square(np.array(v)-before[k]).sum()) for k,v in tree_flatten(self.model.trainable_parameters()))**.5
        if not delta > 0: raise ValueError('RL did not change model parameters')
        checkpoint = self.save(out, target)
        summary = {'complete': True, 'source': source, 'target': target, 'episodes': len(episodes),
                   'transitions': len(rows), 'updates': len(metrics), 'epochs': 1,
                   'positive_advantage_actions': sum(r['advantage']>0 for r in rows),
                   'negative_advantage_actions': sum(r['advantage']<0 for r in rows),
                   'parameter_delta_l2':delta, 'peak_memory_gb':mx.get_peak_memory()/1e9,
                   'checkpoint':checkpoint, 'learning_rate':learning_rate}
        (out / 'summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary),flush=True)
        return summary
