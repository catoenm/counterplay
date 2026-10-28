"""Low-effort, at-most-two-row calibration sampler; no optimizer updates.

Uses the archived FP32/NF4 numerical path and exact original native template.
A two-row, 32-token derivative smoke checks new prefixes/batching and restores
all adapter/Adam/RNG/gradient state. No pretrained model is loaded on import.
"""
from __future__ import annotations

import math

from selfplay.program_debug_cuda import DebugCudaPolicy
from selfplay.program_cardinal_cuda import tensor_identity
from scripts.cloud import train_program_rl as common

REASONING_EFFORT = 'low'
MAX_BATCH_SIZE = 2


class CalibrationCudaPolicy(DebugCudaPolicy):
    def template(self, messages):
        if isinstance(messages, str):
            messages = [{'role': 'user', 'content': messages}]
        if (not isinstance(messages, list) or not messages
                or any(set(m) != {'role', 'content'} or m['role'] not in ('user', 'assistant')
                       or not isinstance(m['content'], str) for m in messages)):
            raise ValueError('Explicit canonical user/assistant messages required')
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=True, reasoning_effort=REASONING_EFFORT)
        if (not rendered.rstrip().endswith('<think>')
                or not rendered.startswith('<|im_start|>system\nReasoning effort is set to low.')):
            raise ValueError('Native template did not enable explicit low reasoning')
        return rendered, self.tokenizer.encode(rendered, add_special_tokens=False)

    def sample_messages_batch(self, contexts, seeds):
        if (not isinstance(seeds, (tuple, list)) or not 1 <= len(seeds) <= MAX_BATCH_SIZE
                or len(contexts) != len(seeds)):
            raise ValueError('Calibration requires one or two fixed independent rows')
        result = super().sample_messages_batch(contexts, seeds)
        for record in result['records']:
            record.update(reasoning_effort=REASONING_EFFORT,
                          completion_token_limit=self.config['max_new_tokens'],
                          calibration_only=True)
        result['timing']['reasoning_effort'] = REASONING_EFFORT
        return result

    def update(self, *args, **kwargs):
        raise RuntimeError('Calibration never performs optimizer updates')

    def update_trajectories(self, *args, **kwargs):
        raise RuntimeError('Calibration never performs optimizer updates')

    def mechanics_smoke(self, prompt):
        torch, original_config = self.torch, self.config
        contexts = prompt['batch_messages'] if isinstance(prompt, dict) else [prompt] * MAX_BATCH_SIZE
        if len(contexts) != MAX_BATCH_SIZE:
            raise ValueError('Calibration smoke requires exactly two mixed-length contexts')
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = [value.clone() for value in torch.cuda.get_rng_state_all()]
        previous_mode = self.model.training
        previous_grads = {name: None if p.grad is None else p.grad.detach().clone()
                          for name, p in self.parameters.items()}
        adapter_before = {name: tensor_identity(p) for name, p in self.parameters.items()}
        optimizer_before = self._optimizer_identity()
        report = {'passed': False, 'kind': 'calibration_two_row_derivative_admission',
            'training_use': False, 'optimizer_steps': 0, 'batch_size': MAX_BATCH_SIZE,
            'max_new_tokens': min(32, original_config['max_new_tokens']),
            'reasoning_effort': REASONING_EFFORT,
            'seeds': [60_000_000 + original_config['seed'] + i for i in range(MAX_BATCH_SIZE)],
            'optimizer_state_before_sha256': optimizer_before['state_sha256']}
        try:
            self.verify_frozen()
            self.config = {**original_config, 'max_new_tokens': report['max_new_tokens']}
            batch = self.sample_messages_batch(contexts, report['seeds'])
            report['batch'] = batch
            records = batch['records']
            if len(records) != MAX_BATCH_SIZE or any(not r['completion_ids'] for r in records):
                raise ValueError('Smoke did not produce both nonempty sampled records')
            report['eval_likelihood_gate'] = self.likelihood_gate(records)
            if not report['eval_likelihood_gate']['passed']:
                raise ValueError('Smoke cached-batch versus teacher-forced eval gate failed')
            self.optimizer.zero_grad(set_to_none=True)
            self.model.train()
            checks = []
            for record in records:
                logps = self.token_logps(record)
                check = common.check_likelihoods(record['sampled_token_logprobs'],
                    logps.detach().cpu().tolist(), original_config['token_logprob_tolerance'],
                    original_config['sequence_logprob_tolerance'])
                checks.append(check)
                report['train_likelihood_gate'] = {'passed': all(r['passed'] for r in checks), 'records': checks}
                if not check['passed']:
                    raise ValueError('Smoke cached-batch versus gradient-mode gate failed')
                (-logps.sum() / len(records)).backward()
                del logps
            squared, missing = 0.0, []
            for name, value in self.parameters.items():
                if value.grad is None:
                    missing.append(name)
                elif not bool(torch.isfinite(value.grad).all()):
                    raise ValueError('Nonfinite smoke derivative')
                else:
                    squared += float(value.grad.detach().double().square().sum().cpu())
            report['derivative'] = {'gradient_l2_norm': math.sqrt(squared),
                'missing_gradient_names': missing,
                'finite_nonzero': math.isfinite(squared) and squared > 0 and not missing}
            if not report['derivative']['finite_nonzero']:
                raise ValueError('Smoke derivative is zero, incomplete, or nonfinite')
            report['passed'] = True
        except Exception as error:
            report.update(passed=False, error_type=type(error).__name__, reason=str(error)[:2000])
        finally:
            self.config = original_config
            for name, value in self.parameters.items():
                value.grad = previous_grads[name]
            self.model.train(previous_mode)
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(cuda_rng)
        adapter_after = {name: tensor_identity(p) for name, p in self.parameters.items()}
        optimizer_after = self._optimizer_identity()
        report.update(optimizer_state_after_sha256=optimizer_after['state_sha256'],
            adapter_unchanged=adapter_before == adapter_after,
            optimizer_unchanged=optimizer_before == optimizer_after,
            rng_restored=(torch.equal(torch.get_rng_state(), cpu_rng)
                and len(torch.cuda.get_rng_state_all()) == len(cuda_rng)
                and all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), cuda_rng))),
            config_restored=self.config is original_config,
            model_mode_restored=self.model.training == previous_mode)
        try:
            report['frozen_state_sha256'] = self.verify_frozen()
        except ValueError as error:
            report.update(passed=False, error_type=type(error).__name__, reason=str(error))
        if not all(report[k] for k in ('adapter_unchanged', 'optimizer_unchanged', 'rng_restored',
                                      'config_restored', 'model_mode_restored')):
            report.update(passed=False, reason='Smoke mutated state')
        return report

    numerical_smoke = mechanics_smoke
