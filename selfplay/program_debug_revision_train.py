"""Full-completion RLOO for the frozen mixed-native two-call controller harness.

Importing this module never imports a pretrained model. Sampling and likelihood
arithmetic are inherited unchanged. The runner owns durable intents and atomic
checkpoint commits; a failed/ambiguous optimizer report must never be retried.
"""
from __future__ import annotations

import math
import time

from .program_debug_revision_cuda import RevisionCudaPolicy, object_hash
from .program_cardinal_cuda import tensor_identity
from scripts.cloud import train_program_rl as common

GROUP_SIZE = 8
TURN_COUNT = 16
PROBE_COEFFICIENTS = (1.0,) * TURN_COUNT


def flatten_training_trajectories(trajectories, advantages):
    """Keep the actual two sampled turns; no feedback/prompt target is created."""
    if (not isinstance(trajectories, list) or len(trajectories) != GROUP_SIZE
            or not isinstance(advantages, (list, tuple)) or len(advantages) != GROUP_SIZE
            or any(type(a) not in (int, float) or not math.isfinite(a) for a in advantages)
            or abs(math.fsum(advantages)) > 1e-10):
        raise ValueError('Exactly eight trajectories and finite zero-sum RLOO advantages required')
    records, coefficients = [], []
    for trajectory, advantage in zip(trajectories, advantages):
        turns = trajectory.get('turns')
        if (not isinstance(turns, list) or len(turns) != 2
                or [r.get('authoring_stage') for r in turns] != ['draft', 'revision']):
            raise ValueError('Each trajectory must retain its ordered draft and mandatory revision')
        records.extend(turns)
        coefficients.extend([float(advantage)] * 2)
    return records, coefficients


def validate_checkpoint_transition(report, source, target):
    """Bind a successful report to independently inspected saved checkpoints.

    This is a pre-commit assertion, not a commit or recovery procedure. Container
    byte hashes may differ after serialization; tensor, Adam and RNG state may
    not differ except for the declared single genuine update.
    """
    if report.get('passed') is not True or report.get('outcome') not in ('updated', 'skipped_zero_variance'):
        raise ValueError('Failed or ambiguous optimizer report cannot be committed')
    for label, identity in [('before', source), ('after', target)]:
        state = report['state_' + label]
        expected = {'adapter_sha256': object_hash(identity['adapter_tensors']),
            'optimizer_state_sha256': identity['state_sha256'], 'rng': identity['rng'],
            'frozen_state_sha256': identity['frozen_state_sha256']}
        if any(state.get(k) != v for k, v in expected.items()):
            raise ValueError('Saved checkpoint differs from optimizer report: ' + label)
        if state['gradients_sha256'] != object_hash({n: None for n in identity['trainable_names']}):
            raise ValueError('Checkpoint boundary has uncommitted gradients')
    for key in ('learning_rate', 'trainable_names', 'parameter_schema', 'parameter_groups',
                'frozen_state_sha256', 'quantization_state_sha256', 'rng'):
        if source[key] != target[key]:
            raise ValueError('Checkpoint invariant changed: ' + key)
    if report['state_before']['config_sha256'] != report['state_after']['config_sha256']:
        raise ValueError('Update changed configuration')
    if report['state_before']['model_training'] or report['state_after']['model_training']:
        raise ValueError('Checkpoint boundary must be in eval mode')
    if (report['adam_step_before'], report['adam_step_after']) != (source['step'], target['step']):
        raise ValueError('Reported Adam step differs from saved state')
    if report['outcome'] == 'updated':
        if (report['optimizer_steps'] != 1 or report['optimizer_step_attempted'] is not True
                or report['optimizer_step_returned'] is not True or target['step'] != source['step'] + 1
                or report['adapter_l2_delta'] <= 0 or source['adapter_tensors'] == target['adapter_tensors']):
            raise ValueError('No proven single genuine optimizer update')
    else:
        if (report['optimizer_steps'] != 0 or report['optimizer_step_attempted'] is not False
                or {k:v for k,v in source.items() if k != 'files'} != {k:v for k,v in target.items() if k != 'files'}):
            raise ValueError('Zero-variance skip changed checkpoint state')
    return True


class TrainRevisionCudaPolicy(RevisionCudaPolicy):
    def __init__(self, model_dir, config):
        super().__init__(model_dir, config)
        self.runtime.update(calibration_only=False, training_capable=True,
            training_objective='RLOO mean over eight trajectories; sum both full sampled completions')

    def sample_messages_batch(self, contexts, seeds, authoring_stage='draft'):
        result = super().sample_messages_batch(contexts, seeds, authoring_stage=authoring_stage)
        for record in result['records']:
            record.update(calibration_only=False, training_capable=True)
        return result

    def update(self, *args, **kwargs):
        raise RuntimeError('Use the complete eight-trajectory update_trajectories interface')

    def _synchronize(self):
        if any(p.device.type == 'cuda' for p in self.parameters.values()):
            self.torch.cuda.synchronize()

    def _admit(self):
        if 'deadline' in self.config:
            common.admission(self.config)

    def _validate_group_records(self, records):
        if (not isinstance(records, list) or len(records) != TURN_COUNT
                or [r.get('authoring_stage') for r in records] != ['draft', 'revision'] * GROUP_SIZE):
            raise ValueError('All sixteen original turns in trajectory order are required')
        for record in records:
            self.validate_record(record)
            if record.get('calibration_only') is not False or record.get('training_capable') is not True:
                raise ValueError('Calibration samples cannot be relabeled as fresh training trajectories')
        if len({r['seed'] for r in records}) != TURN_COUNT:
            raise ValueError('Independent registered turn seeds are required')
        for draft, revision in zip(records[::2], records[1::2]):
            if revision['seed'] != draft['seed'] + 1:
                raise ValueError('Revision seed must follow its original draft seed')

    def _gradient_summary(self):
        squared, missing, nonfinite, identities = 0.0, [], [], {}
        for name, parameter in self.parameters.items():
            gradient = parameter.grad
            if gradient is None:
                missing.append(name)
                continue
            if not bool(self.torch.isfinite(gradient).all()):
                nonfinite.append(name)
                continue
            value = float(gradient.detach().double().square().sum().cpu())
            squared += value
            identities[name] = {**tensor_identity(gradient), 'squared_l2_norm': value}
        norm = math.sqrt(squared) if math.isfinite(squared) else None
        return {'gradient_l2_norm': norm, 'missing_gradient_names': missing,
            'nonfinite_gradient_names': nonfinite, 'gradient_tensors': identities,
            'finite_nonzero': not missing and not nonfinite and norm is not None and norm > 0}

    def _backward_rows(self, records, coefficients, divisor, report, include_eval):
        """One unpadded full-completion graph at a time; retain failed evidence."""
        torch = self.torch
        for index, (record, coefficient) in enumerate(zip(records, coefficients)):
            self._admit()
            entry = {'index': index, 'record_sha256': object_hash(record),
                'authoring_stage': record['authoring_stage'], 'seed': record['seed'],
                'completion_tokens': len(record['completion_ids']),
                'sampled_token_logprobs': list(record['sampled_token_logprobs']),
                'coefficient': coefficient, 'objective_divisor': divisor,
                'backward_completed': False, 'passed': False, 'timing': {}}
            report['records'].append(entry)
            modes = ('eval', 'train') if include_eval else ('train',)
            for mode in modes:
                values = loss = None
                started = time.monotonic()
                try:
                    self._admit()
                    self.validate_record(record)
                    self.model.train(mode == 'train')
                    with torch.enable_grad() if mode == 'train' else torch.no_grad():
                        entry[mode + '_grad_enabled'] = torch.is_grad_enabled()
                        values = self.token_logps(record)
                        vector = values.detach().cpu().tolist()
                        entry[mode + '_token_logprobs'] = [v if math.isfinite(v) else str(v) for v in vector]
                        entry[mode + '_all_finite'] = all(math.isfinite(v) for v in vector)
                        check = common.check_likelihoods(record['sampled_token_logprobs'], vector,
                            self.config['token_logprob_tolerance'], self.config['sequence_logprob_tolerance'])
                        entry[mode + '_check'] = check
                        if not include_eval:
                            entry.update(check)  # Preserve the previous check schema as well.
                        entry['timing'][mode + '_forward_seconds'] = time.monotonic() - started
                        if not check['passed']:
                            raise ValueError(mode + '-mode likelihood gate failed before optimizer step')
                        if mode == 'train':
                            loss = -float(coefficient) * values.sum() / divisor
                            loss_value = float(loss.detach().cpu())
                            if not math.isfinite(loss_value):
                                raise ValueError('Nonfinite full-completion objective')
                            entry['train_loss'] = loss_value
                            backward_started = time.monotonic()
                            loss.backward()
                            self._synchronize()
                            entry['timing']['backward_seconds'] = time.monotonic() - backward_started
                            entry['backward_completed'] = True
                            report['microbatch_losses'].append(loss_value)
                except Exception as error:
                    entry.setdefault(mode + '_token_logprobs', None)
                    entry.setdefault(mode + '_all_finite', False)
                    entry.setdefault(mode + '_check', {'passed': False,
                        'error_type': type(error).__name__, 'reason': str(error)[:2000]})
                    entry.update(passed=False, error_type=type(error).__name__, reason=str(error)[:2000])
                    entry['timing'][mode + '_failed_elapsed_seconds'] = time.monotonic() - started
                    raise
                finally:
                    del loss, values
            entry['passed'] = entry['backward_completed'] and all(entry[m + '_check']['passed'] for m in modes)

    def full_backward_probe(self, records):
        """Numerical-only full-length backward, even when genuine rewards tie."""
        started = time.monotonic()
        saved = self._capture()
        report = {'passed': False, 'kind': 'revision_full_backward_probe', 'training_use': False,
            'optimizer_steps': 0, 'expected_records': TURN_COUNT,
            'coefficients': list(PROBE_COEFFICIENTS), 'objective_divisor': TURN_COUNT,
            'objective': 'Negative mean of sixteen full completion logprob sums; fixed coefficients, not rewards',
            'token_tolerance': self.config['token_logprob_tolerance'],
            'sequence_tolerance': self.config['sequence_logprob_tolerance'],
            'records': [], 'microbatch_losses': [], 'timing': {}}
        try:
            self._validate_group_records(records)
            report['sampled_tokens'] = sum(len(r['completion_ids']) for r in records)
            self.optimizer.zero_grad(set_to_none=True)
            backward_started = time.monotonic()
            self._backward_rows(records, PROBE_COEFFICIENTS, TURN_COUNT, report, include_eval=True)
            report['timing']['full_replay_backward_seconds'] = time.monotonic() - backward_started
            report['loss'] = math.fsum(report['microbatch_losses'])
            report['derivative'] = self._gradient_summary()
            if not report['derivative']['finite_nonzero']:
                raise ValueError('Full-length probe derivative is zero, incomplete or nonfinite')
            report['passed'] = True
        except Exception as error:
            report.update(passed=False, error_type=type(error).__name__, reason=str(error)[:2000])
            report['loss'] = math.fsum(report['microbatch_losses'])
        finally:
            self._restore(saved, report)
            report['timing']['total_seconds'] = time.monotonic() - started
        return report

    def update_trajectories(self, trajectories, advantages):
        """Exactly one genuine RLOO step, or an explicit immutable zero skip.

        The runner must first journal a passing full eval/train likelihood gate
        for these same original sixteen records. Here every train-mode vector is
        checked again before any Adam step. Failures are returned for journaling;
        they never trigger retries, alternate precision, repaired draws or masks.
        """
        started = time.monotonic()
        saved = self._capture()
        before_optimizer = self._optimizer_identity()
        before_adapter = {n: p.detach().cpu().clone() for n, p in self.parameters.items()}
        report = {'passed': False, 'outcome': 'failed', 'kind': 'mixed_native_full_completion_rloo',
            'training_use': True, 'optimizer_steps': 0, 'optimizer_step_attempted': False,
            'optimizer_step_returned': False, 'trajectory_count': GROUP_SIZE, 'sampled_turn_count': TURN_COUNT,
            'objective_divisor': GROUP_SIZE, 'objective': '-sum_i A_i*(sum draft logp + sum revision logp)/8',
            'tool_feedback_target_tokens': 0, 'prompt_target_tokens': 0,
            'token_tolerance': self.config['token_logprob_tolerance'],
            'sequence_tolerance': self.config['sequence_logprob_tolerance'],
            'records': [], 'microbatch_losses': [], 'timing': {},
            'adam_step_before': before_optimizer['step'], 'optimizer_identity_before': before_optimizer,
            'state_before': saved['identity']}
        try:
            records, coefficients = flatten_training_trajectories(trajectories, advantages)
            self._validate_group_records(records)
            report.update(advantages=list(map(float, advantages)), coefficients=coefficients,
                sampled_tokens=sum(len(r['completion_ids']) for r in records),
                turn_sha256=[object_hash(r) for r in records])
            if any(p.grad is not None for p in self.parameters.values()) or self.model.training:
                raise ValueError('Actual update must start from a clean eval-mode checkpoint boundary')
            if not any(coefficients):
                report.update(passed=True, outcome='skipped_zero_variance', adapter_l2_delta=0., loss=0.)
            else:
                self._admit()
                self.optimizer.zero_grad(set_to_none=True)
                backward_started = time.monotonic()
                self._backward_rows(records, coefficients, GROUP_SIZE, report, include_eval=False)
                report['timing']['full_train_forward_backward_seconds'] = time.monotonic() - backward_started
                report['loss'] = math.fsum(report['microbatch_losses'])
                report['derivative'] = self._gradient_summary()
                if not report['derivative']['finite_nonzero']:
                    raise ValueError('Zero, incomplete or nonfinite genuine policy gradient; no update')
                norm = self.torch.nn.utils.clip_grad_norm_(list(self.parameters.values()),
                    self.config['gradient_clip'], error_if_nonfinite=True)
                report['gradient_norm_before_clip'] = float(norm)
                report['derivative_after_clip'] = self._gradient_summary()
                if (not report['derivative_after_clip']['finite_nonzero']
                        or not math.isfinite(float(norm)) or float(norm) <= 0):
                    raise ValueError('Clipped policy gradient is zero or nonfinite')
                self._admit()
                report['optimizer_step_attempted'] = True
                report['optimizer_steps'] = None  # A throwing step may have partially mutated Adam.
                step_started = time.monotonic()
                self.optimizer.step()
                self._synchronize()
                report['timing']['adam_step_seconds'] = time.monotonic() - step_started
                report['optimizer_step_returned'] = True
                after_optimizer = self._optimizer_identity()
                report['optimizer_steps'] = after_optimizer['step'] - before_optimizer['step']
                if report['optimizer_steps'] != 1:
                    raise ValueError('Adam did not advance exactly one step')
                deltas = {}
                for name, parameter in self.parameters.items():
                    if not bool(self.torch.isfinite(parameter).all()):
                        raise ValueError('Nonfinite updated adapter')
                    deltas[name] = float((parameter.detach().cpu().double() - before_adapter[name].double()).square().sum())
                report['adapter_squared_deltas'] = deltas
                report['adapter_l2_delta'] = math.sqrt(math.fsum(deltas.values()))
                if not math.isfinite(report['adapter_l2_delta']) or report['adapter_l2_delta'] <= 0:
                    raise ValueError('Optimizer did not change adapter weights')
                report.update(passed=True, outcome='updated')
        except Exception as error:
            report.update(passed=False, outcome='failed', error_type=type(error).__name__, reason=str(error)[:2000])
            report.setdefault('loss', math.fsum(report['microbatch_losses']))
        finally:
            # Failed Adam mutations are deliberately NOT rolled back or retried.
            # Clear transient graphs, restore non-training context, then report
            # whatever actual adapter/Adam state remains for the runner to save.
            self.optimizer.zero_grad(set_to_none=True)
            self.config = saved['config']
            self.model.train(saved['mode'])
            self.torch.set_rng_state(saved['cpu_rng'])
            self.torch.cuda.set_rng_state_all(saved['cuda_rng'])
            report['gradient_likelihood_gate'] = {'passed': len(report['records']) == TURN_COUNT
                and all(r['passed'] for r in report['records']), 'records': report['records']}
            try:
                after = self._state_identity(); report['state_after'] = after
                identity = self._optimizer_identity()
                report.update(adam_step_after=identity['step'], optimizer_identity_after=identity,
                    optimizer_state_sha256=identity['state_sha256'], frozen_state_sha256=after['frozen_state_sha256'])
                flags = {'rng_restored': 'rng', 'config_restored': 'config_sha256',
                         'model_mode_restored': 'model_training', 'frozen_state_unchanged': 'frozen_state_sha256'}
                for name, key in flags.items(): report[name] = saved['identity'][key] == after[key]
                report['gradients_cleared'] = after['gradients_sha256'] == object_hash({n: None for n in self.parameters})
                if not all(report[k] for k in [*flags, 'gradients_cleared']):
                    raise ValueError('Post-update context or frozen-state invariant failed')
                if not report['optimizer_step_attempted']:
                    if any(after[k] != saved['identity'][k] for k in ('adapter_sha256', 'optimizer_state_sha256')):
                        raise ValueError('Adapter/Adam mutated without an optimizer step')
            except Exception as error:
                report.update(passed=False, outcome='failed', state_inspection_error_type=type(error).__name__,
                    state_inspection_error=str(error)[:2000])
            report['timing']['total_seconds'] = time.monotonic() - started
        return report
