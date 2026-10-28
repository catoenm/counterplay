"""Once-only RPCs for a single bounded planning comparison allocation."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import time
import traceback

from selfplay.arena_mixed_scalar.processes import ProcessClient as ScalarClient, checkpoint_reference
from .runner import Journal, collect_cohort, digest, load_sealed_cohort, now


def _dispatch(policy, command, parent_journal):
    method, data = command['method'], command['data']
    stage_deadline = data['stage_deadline_unix']
    if not time.time() < stage_deadline <= parent_journal.deadline:
        raise TimeoutError('Stage deadline is exhausted or escapes aggregate deadline')
    journal = Journal(parent_journal.folder, parent_journal.run_sha256, stage_deadline)
    if method == 'publish_base':
        if policy.role != 'learner' or policy.binding is not None:
            raise ValueError('Only fresh learner may publish shared base')
        folder = journal.path('checkpoints/base')
        result = checkpoint_reference(folder, policy.save_checkpoint(folder))
        policy._study_base_snapshot = policy._capture()
        policy._study_base_identity = policy._state_identity()
        return result
    if method == 'load':
        result = checkpoint_reference(data['folder'], policy.load_checkpoint(data['folder'], data['files']))
        if not hasattr(policy, '_study_base_snapshot'):
            if result['binding']['version'] != 0 or result['optimizer_step'] != 0:
                raise ValueError('Initial checkpoint must be exact shared zero-step base')
            policy._study_base_snapshot = policy._capture()
            policy._study_base_identity = policy._state_identity()
        return result
    if method == 'reset_base':
        if policy.poisoned or not hasattr(policy, '_study_base_snapshot'):
            raise ValueError('Cannot restore ambiguous or uncaptured study base')
        before = policy._state_identity()
        policy._restore(deepcopy(policy._study_base_snapshot))
        policy.verify_frozen()
        after = policy._state_identity()
        if after != policy._study_base_identity or policy.binding != data['binding']:
            raise ValueError('Shared adapter/zero-step Adam/RNG restoration differs')
        return {'passed': True, 'before': before, 'after': after,
                'shared_base_state': policy._study_base_identity, 'optimizer_steps': 0}
    if method == 'smoke':
        if getattr(policy, '_study_smoke_attempted', False):
            raise ValueError('Numerical smoke already attempted')
        policy._study_smoke_attempted = True
        return policy.numerical_smoke(data['requests'])
    if method in ('collect', 'collect_profiled'):
        if policy.role != 'actor':
            raise ValueError('Separate immutable actor required for rollout')
        reference = data['checkpoint']
        policy.load_checkpoint(reference['folder'], reference['files'])
        profiled = method == 'collect_profiled'
        return collect_cohort(policy.choose_profiled if profiled else policy.choose,
            data['specs'], reference['binding'], journal, data['prefix'],
            batch_size=policy.config['inference_batch_size'], profiled=profiled)
    if method == 'timing_probe':
        if policy.role != 'learner':
            raise ValueError('Learner alone may probe optimizer')
        _, episodes = load_sealed_cohort(data['cohort'])
        by_id = {e['episode_id']: e for e in episodes}
        selected = [by_id[name] for name in data['episode_ids']]
        journal.intent('admission/probe-optimizer-intent.json', 'planning_temporary_optimizer_probe',
            {'episode_ids': data['episode_ids'], 'episodes_sha256': digest(selected)})
        result = policy.timing_probe(selected, data['episode_ids'], journal.path('admission/probe-checkpoint'))
        journal.write('admission/probe.json', result)
        return result
    if method == 'update':
        if policy.role != 'learner':
            raise ValueError('Actor cannot update')
        seal, episodes = load_sealed_cohort(data['cohort'])
        current, behavior, prefix = data['current'], data['behavior'], data['prefix']
        if policy.binding != current['binding']:
            raise ValueError('Learner source checkpoint differs')
        journal.intent(prefix+'/gate-intent.json', 'planning_behavior_gate',
            {'behavior': behavior, 'learner': current, 'seal_sha256': digest(seal)})
        gate = policy.gate_at_checkpoint([r for e in episodes for r in e['transitions']],
            behavior['folder'], behavior['files'])
        journal.write(prefix+'/gate.json', gate)
        if not gate['passed']:
            raise ValueError('Full behavior likelihood gate failed; no retry')
        journal.intent(prefix+'/optimizer-intent.json', 'planning_retained_optimizer_update',
            {'behavior': behavior, 'learner': current, 'gate_sha256': digest(gate)})
        report = policy.train_cohort(episodes, behavior['binding'], gate)
        journal.write(prefix+'/optimizer-result.json', report)
        folder = journal.path(data['checkpoint_relative'])
        reference = checkpoint_reference(folder, policy.save_checkpoint(folder, version=data['index']))
        result = {'report': report, 'checkpoint': reference, 'committed_at': now()}
        journal.write(prefix+'/update.json', result)
        return result
    raise ValueError('Unknown planning operation')


def worker(connection, role, model_dir, config, folder, run_sha256, deadline):
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    journal = Journal(folder, run_sha256, deadline)
    try:
        from .cuda_policy import CudaActionPolicy
        policy = CudaActionPolicy(model_dir, config, role=role)
        connection.send({'ready': True, 'runtime': policy.runtime, 'base_identity': policy.base_identity})
        while True:
            if not connection.poll(max(0., deadline-time.time())):
                raise TimeoutError('Immutable aggregate worker deadline')
            command = connection.recv()
            if command['method'] == 'stop':
                return
            journal.admit()
            result = _dispatch(policy, command, journal)
            connection.send({'id': command['id'], 'passed': True, 'result': result})
    except BaseException as error:
        failure = {'passed': False, 'error_type': type(error).__name__, 'reason': str(error),
            'traceback': traceback.format_exc(), 'role': role, 'at': now()}
        if 'policy' in locals() and hasattr(policy, 'last_failure'):
            failure['neural_failure'] = policy.last_failure
        try:
            journal.write(f'processes/{role}-failure.json', failure)
            connection.send(failure)
        finally:
            connection.close()
        raise


class ProcessClient(ScalarClient):
    def __init__(self, role, model_dir, config, journal, *, target=worker):
        super().__init__(role, model_dir, config, journal, target=target)

    def request(self, method, data):
        return super().request(method, {**deepcopy(data), 'stage_deadline_unix': self.journal.deadline})


class ArmActor:
    def __init__(self, client, prefix):
        self.client, self.prefix = client, prefix

    def submit(self, specs, checkpoint, prefix):
        return self.client.submit(specs, checkpoint, self.prefix+'/'+prefix)


class ArmLearner:
    def __init__(self, client, prefix, base):
        self.client, self.prefix, self.checkpoints = client, prefix, {0: base}

    def update(self, cohort, current, index):
        seal, _ = load_sealed_cohort(cohort)
        behavior = self.checkpoints[seal['binding']['version']]
        result = self.client.call('update', {'cohort': str(cohort), 'current': current,
            'behavior': behavior, 'index': index, 'prefix': f'{self.prefix}/cohorts/{index}',
            'checkpoint_relative': f'{self.prefix}/checkpoints/cycle{index}'})
        self.checkpoints[index] = result['checkpoint']
        return result
