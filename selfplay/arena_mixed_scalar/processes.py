"""Versioned scalar actor/learner processes; immutable one-call queues."""
from __future__ import annotations

from copy import deepcopy
import multiprocessing
import os
from pathlib import Path
import time
import traceback
import subprocess
import threading

from .runner import Journal, collect_cohort, digest, load_sealed_cohort, now


def checkpoint_reference(folder, state):
    return {'folder': str(Path(folder).resolve()), 'files': state['files'],
            'binding': state['policy']['binding'], 'optimizer_step': state['step']}


def _dispatch(policy, command, journal):
    method, data = command['method'], command['data']
    if method == 'publish_base':
        if policy.role != 'learner' or policy.binding is not None:
            raise ValueError('Only a fresh learner may publish base')
        folder = journal.path('checkpoints/base')
        return checkpoint_reference(folder, policy.save_checkpoint(folder))
    if method == 'load':
        result = policy.load_checkpoint(data['folder'], data['files'])
        return checkpoint_reference(data['folder'], result)
    if method == 'execution_diagnostic':
        return policy.execution_diagnostic(data['fixture'])
    if method == 'smoke':
        return policy.numerical_smoke(data['requests'])
    if method == 'choose':
        return policy.choose(data['requests'], data['binding'])
    if method == 'collect':
        if policy.role != 'actor':
            raise ValueError('Training rollouts require immutable separate actor')
        checkpoint = data['checkpoint']
        policy.load_checkpoint(checkpoint['folder'], checkpoint['files'])
        return collect_cohort(policy.choose, data['specs'], checkpoint['binding'],
            journal, data['prefix'], batch_size=policy.config['inference_batch_size'])
    if method in ('update', 'profile'):
        if policy.role != 'learner':
            raise ValueError('Optimizer only runs in learner process')
        seal, episodes = load_sealed_cohort(data['cohort'])
        current, behavior = data['current'], data['behavior']
        if policy.binding != current['binding']:
            raise ValueError('Learner does not hold exact published source')
        prefix = data['prefix']
        journal.intent(prefix+'/gate-intent.json', 'mixed_behavior_gate',
            {'behavior': behavior, 'learner': current, 'seal_sha256': digest(seal)})
        gate = policy.gate_at_checkpoint([r for e in episodes for r in e['transitions']],
            behavior['folder'], behavior['files'])
        journal.write(prefix+'/gate.json', gate)
        if not gate['passed']:
            raise ValueError('Full behavior likelihood gate failed; no retry')
        journal.intent(prefix+'/optimizer-intent.json', 'mixed_optimizer_probe' if method=='profile' else 'mixed_optimizer_update',
            {'behavior': behavior, 'learner': current, 'gate_sha256': digest(gate)})
        if method == 'profile':
            report = policy.profile_cohort(episodes, behavior['binding'], gate,
                checkpoint_folder=journal.path(prefix+'/profile-checkpoint'))
            journal.write(prefix+'/profile.json', report)
            return report
        report = policy.train_cohort(episodes, behavior['binding'], gate)
        journal.write(prefix+'/optimizer-result.json', report)
        folder = journal.path(f"checkpoints/cycle{data['index']}")
        state = policy.save_checkpoint(folder, version=data['index'])
        reference = checkpoint_reference(folder, state)
        result = {'report': report, 'checkpoint': reference, 'committed_at': now()}
        journal.write(prefix+'/update.json', result)
        return result
    raise ValueError('Unknown worker operation')


def worker(connection, role, model_dir, config, folder, run_sha256, deadline):
    # spawn starts a clean interpreter: no inherited CUDA context or live model.
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    journal = Journal(folder, run_sha256, deadline)
    try:
        from .cuda_policy import CudaActionPolicy
        policy = CudaActionPolicy(model_dir, config, role=role)
        connection.send({'ready': True, 'runtime': policy.runtime, 'base_identity': policy.base_identity})
        while True:
            if not connection.poll(max(0., deadline-time.time())):
                raise TimeoutError('Worker reached immutable deadline')
            command = connection.recv()
            if command['method'] == 'stop':
                return
            journal.admit()
            result = _dispatch(policy, command, journal)
            connection.send({'id': command['id'], 'passed': True, 'result': result})
    except BaseException as error:
        failure = {'passed': False, 'error_type': type(error).__name__, 'reason': str(error),
                   'traceback': traceback.format_exc(), 'role': role, 'at': now()}
        if 'policy' in locals() and hasattr(policy,'last_failure'):
            failure['neural_failure'] = policy.last_failure
        try:
            journal.write(f'processes/{role}-failure.json', failure)
            connection.send(failure)
        finally:
            connection.close()
        raise


class Pending:
    def __init__(self, client, request_id):
        self.client, self.request_id, self.resolved = client, request_id, False

    def result(self):
        if self.resolved:
            raise ValueError('Future already consumed')
        result = self.client._receive(self.request_id)
        self.resolved = True
        return result


class ProcessClient:
    def __init__(self, role, model_dir, config, journal, *, target=worker):
        self.role, self.journal, self.counter, self.pending = role, journal, 0, None
        context = multiprocessing.get_context('spawn')
        parent, child = context.Pipe(duplex=True)
        self.connection = parent
        self.process = context.Process(target=target, args=(child, role, str(model_dir), config,
            str(journal.folder), journal.run_sha256, journal.deadline), name=f'mixed-{role}')
        journal.intent(f'processes/{role}-start-intent.json', 'mixed_process_launch',
            {'role': role, 'config_sha256': digest(config)})
        self.process.start()
        child.close()

    def ready(self):
        response = self._wait()
        if response.get('ready') is not True:
            raise RuntimeError('CUDA process initialization failed: '+str(response))
        self.journal.write(f'processes/{self.role}-ready.json', response)
        return response

    def _wait(self):
        while True:
            self.journal.admit()
            if self.connection.poll(min(1., max(.01, self.journal.deadline-time.time()))):
                return self.connection.recv()
            if not self.process.is_alive():
                raise RuntimeError(f'{self.role} process exited without a bound result')

    def request(self, method, data):
        if self.pending is not None:
            raise ValueError('Queue capacity one; prior operation is unresolved')
        self.journal.admit()
        request_id = f'{self.role}-{self.counter:06d}'
        self.counter += 1
        self.journal.intent(f'processes/{request_id}-intent.json', 'mixed_rpc',
            {'request_id': request_id, 'method': method, 'data_sha256': digest(data)})
        self.pending = request_id
        self.connection.send({'id': request_id, 'method': method, 'data': deepcopy(data)})
        return Pending(self, request_id)

    def _receive(self, request_id):
        if self.pending != request_id:
            raise ValueError('RPC identity differs')
        response = self._wait()
        self.journal.write(f'processes/{request_id}-result.json', response)
        if response.get('id') != request_id or response.get('passed') is not True:
            raise RuntimeError('Neural RPC failed or ambiguous; no retry: '+str(response))
        self.pending = None
        return response['result']

    def call(self, method, data):
        return self.request(method, data).result()

    def submit(self, specs, checkpoint, prefix):
        return self.request('collect', {'specs': specs, 'checkpoint': checkpoint, 'prefix': prefix})

    def choose(self, requests, binding):
        return self.call('choose', {'requests': requests, 'binding': binding})

    def stop(self):
        try:
            if self.process.is_alive() and self.pending is None:
                try:
                    self.connection.send({'method': 'stop'})
                    self.process.join(timeout=10)
                except (BrokenPipeError, EOFError, OSError):
                    # A failed/closing worker may have already closed its pipe.
                    # Continue bounded process cleanup before any artifact scan.
                    pass
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=10)
            self.process.join(timeout=0)
            if self.process.is_alive():
                raise RuntimeError(f'{self.role} process survived bounded shutdown')
        finally:
            self.connection.close()


class LearnerClient:
    def __init__(self, client, checkpoints):
        self.client, self.checkpoints = client, checkpoints

    def update(self, cohort, current, index):
        seal, _ = load_sealed_cohort(cohort)
        behavior = self.checkpoints[seal['binding']['version']]
        result = self.client.call('update', {'cohort': str(cohort), 'current': current,
            'behavior': behavior, 'index': index, 'prefix': f'cohorts/{index}'})
        self.checkpoints[index] = result['checkpoint']
        return result


class GPUMonitor:
    """Read-only process-independent memory/utilization sampling during overlap."""
    def __init__(self):
        self.samples=[];self.errors=[];self.event=threading.Event()
        self.thread=threading.Thread(target=self._loop,daemon=True)

    def _loop(self):
        while not self.event.is_set():
            try:
                output=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=5).stdout
                rows=[line for line in output.splitlines() if line.strip()]
                if len(rows)!=1:raise ValueError('Expected exactly one visible GPU')
                used,total,utilization=[float(v.strip()) for v in rows[0].split(',')]
                self.samples.append({'at':now(),'memory_used_mib':used,'memory_total_mib':total,'utilization_percent':utilization})
            except Exception as error:
                self.errors.append({'at':now(),'error_type':type(error).__name__})
            self.event.wait(2)

    def start(self):
        self.thread.start();return self

    def stop(self):
        self.event.set();self.thread.join(timeout=7)
        return {'samples':self.samples,'errors':self.errors,
            'peak_observed_memory_mib':max((s['memory_used_mib'] for s in self.samples),default=None),
            'memory_measurement':'Sampled whole-device usage; may miss between-sample peaks.'}
