"""CPU-only journal mutations; synthetic payloads are not neural evidence."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts import analyze_planning_curriculum as audit
from selfplay.planning_curriculum.runner import Journal, digest, file_sha


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


class RpcAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        started = time.time()
        plan = {'created_at': datetime.fromtimestamp(started, timezone.utc).isoformat(),
            'deadline_unix': started + 43200, 'gpu_hourly_rate': 5.29,
            'guard_hourly_rate': .14, 'storage_hourly_rate': .02, 'other_reserved_cost_usd': 1.}
        self.manifest = {'deployment': plan, 'config': {'policy': {'learning_rate': 1e-5}}}
        write(self.run/'manifest.json', self.manifest)
        self.deadline = audit.lease_clocks(plan)['admission_stop']
        self.journal = Journal(self.run, file_sha(self.run/'manifest.json'), started+43200)
        self.base = {'folder': '/workspace/fixture/checkpoints/base', 'files': {'fixture': 'a'*64},
            'binding': {'version': 0, 'checkpoint_id': 'base'}, 'optimizer_step': 0}
        self.actual = {'files': self.base['files'], 'policy': {'binding': self.base['binding']}}
        self.spec = {'seed': 100, 'regime': 'train', 'depth': 3, 'scenario_family_id': 'family'}
        self.specs = {'screen-hard': [self.spec], 'screen-easy': []}
        write(self.run/'inputs/schedule.json', self.specs)
        self.view = {'observation': {'fixture': True}, 'legal_actions': [{'tool': 'inc_x'}]}
        self.request = {'episode_id': 'numerical-planning', 'environment': 'planning',
            'scenario_family_id': 'family', 'turn': 0, 'player': 0,
            'observation': self.view['observation'], 'actions': self.view['legal_actions'], 'sample_seed': 731991}
        self.counts = {'actor': 0, 'learner': 0}
        for target, value in ((audit.policy, 'checkpoint_state'), (audit.envs, 'make_env'), (audit.envs, 'policy_view')):
            result = self.actual if value == 'checkpoint_state' else self.view if value == 'policy_view' else object()
            mocked = patch.object(target, value, return_value=result)
            mocked.start(); self.addCleanup(mocked.stop)

    def rpc(self, role, method, data, result, target=None, *, reply=True, deadline=None):
        name = f'{role}-{self.counts[role]:06d}'
        self.counts[role] += 1
        self.journal.intent(f'processes/{name}-intent.json', 'mixed_rpc', {
            'request_id': name, 'method': method,
            'data_sha256': digest({**deepcopy(data), 'stage_deadline_unix': self.deadline if deadline is None else deadline})})
        if target: write(self.run/target, result)
        if reply: self.journal.write(f'processes/{name}-result.json', {'id': name, 'passed': True, 'result': result})
        return name

    def admission(self, *, probe=True):
        self.rpc('learner', 'publish_base', {}, self.base)
        self.rpc('actor', 'load', self.base, self.base)
        for role in ('actor', 'learner'):
            self.rpc(role, 'smoke', {'requests': [self.request]}, {'fixture_smoke': role}, f'admission/{role}-smoke.json')
        self.rpc('actor', 'collect_profiled', {'specs': [self.spec], 'checkpoint': self.base, 'prefix': 'admission/screen'},
            {'fixture_seal': True}, 'admission/screen/seal.json')
        write(self.run/'admission/signal-gate.json', {'passed': probe, 'timing_probe_episode_ids': ['a', 'b']})
        if probe:
            self.rpc('learner', 'timing_probe', {'cohort': '/workspace/fixture/admission/screen', 'episode_ids': ['a', 'b']},
                {'fixture_probe': True}, 'admission/probe.json')

    def check(self): return audit.verify_rpc_journal(self.run, self.manifest)

    def test_publish_load_smoke_profiled_collection_probe_publications_bound(self):
        self.admission()
        self.assertEqual(self.check(), {'verified_successful_rpcs': 6, 'unresolved_or_failed_rpcs': [], 'base_publication_available': True})

    def test_changed_command_digest_rejected(self):
        self.admission()
        path = self.run/'processes/learner-000002-intent.json'
        row = audit.load(path); row['data_sha256'] = '0'*64; write(path, row)
        with self.assertRaisesRegex(ValueError, 'RPC command'): self.check()

    def test_stage_deadline_cannot_be_extended_or_replaced(self):
        self.admission()
        path = self.run/'processes/actor-000002-intent.json'
        row = audit.load(path)
        data = {'specs': [self.spec], 'checkpoint': self.base, 'prefix': 'admission/screen', 'stage_deadline_unix': self.deadline+1}
        row['data_sha256'] = digest(data); write(path, row)
        with self.assertRaisesRegex(ValueError, 'RPC command'): self.check()

    def test_changed_successful_reply_and_publication_rejected(self):
        for target in ('processes/actor-000002-result.json', 'admission/screen/seal.json'):
            with self.subTest(target=target):
                if not self.counts['actor']: self.admission()
                path = self.run/target; original = path.read_bytes(); value = audit.load(path)
                if target.startswith('processes/'): value['result']['fixture_seal'] = False
                else: value['fixture_seal'] = False
                write(path, value)
                with self.assertRaisesRegex(ValueError, 'durable publication'): self.check()
                path.write_bytes(original)

    def test_wrong_result_id_and_base_checkpoint_rejected(self):
        self.admission()
        path = self.run/'processes/actor-000000-result.json'; value = audit.load(path)
        value['id'] = 'learner-000000'; write(path, value)
        with self.assertRaisesRegex(ValueError, 'RPC result identity'): self.check()
        value['id'] = 'actor-000000'; write(path, value)
        self.actual['files'] = {'fixture': 'changed'}
        with self.assertRaisesRegex(ValueError, 'Base RPC'): self.check()

    def test_missing_final_reply_is_reported_without_claiming_success(self):
        self.admission()
        (self.run/'processes/learner-000002-result.json').unlink()
        result = self.check()
        self.assertEqual(result['verified_successful_rpcs'], 5)
        self.assertEqual(result['unresolved_or_failed_rpcs'], ['learner-000002'])

    def test_failed_final_reply_preserved_as_unknown(self):
        self.admission()
        write(self.run/'processes/learner-000002-result.json', {'passed': False, 'error_type': 'FixtureFailure'})
        self.assertEqual(self.check()['unresolved_or_failed_rpcs'], ['learner-000002'])

    def test_rpc_sequence_has_no_retries_or_unregistered_extra_commands(self):
        self.admission(probe=False)
        self.rpc('learner', 'timing_probe', {}, {'fabricated': True})
        with self.assertRaisesRegex(ValueError, 'More RPCs'): self.check()

    def test_initial_missing_base_reply_retains_exact_pending_intent(self):
        self.rpc('learner', 'publish_base', {}, self.base, reply=False)
        result = self.check()
        self.assertFalse(result['base_publication_available'])
        self.assertEqual(result['unresolved_or_failed_rpcs'], ['learner-000000'])
        path = self.run/'processes/learner-000000-intent.json'; row = audit.load(path)
        row['data_sha256'] = '0'*64; write(path, row)
        with self.assertRaisesRegex(ValueError, 'Initial publication'): self.check()

    def test_later_dispatch_after_missing_or_failed_reply_rejected(self):
        self.admission()
        path = self.run/'processes/actor-000001-result.json'; original = path.read_bytes()
        for failure in (None, {'passed': False, 'error_type': 'FixtureFailure'}):
            if failure is None: path.unlink()
            else: write(path, failure)
            with self.subTest(failure=failure), self.assertRaisesRegex(ValueError, 'unresolved|failed|Later'):
                self.check()
            path.write_bytes(original)
        (self.run/'processes/learner-000000-result.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Later RPC'): self.check()

    def test_orphan_result_and_noncanonical_rpc_filename_rejected(self):
        self.admission()
        path = self.run/'processes/actor-000001-intent.json'; original = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'intent|orphan'): self.check()
        path.write_bytes(original)
        path = self.run/'processes/actor-1-result.json'; write(path, {'passed': True})
        with self.assertRaisesRegex(ValueError, 'canonical|filename|RPC'): self.check()

    def test_comparison_reset_collect_and_update_bind_current_stage(self):
        self.admission()
        started = datetime.fromisoformat(self.manifest['deployment']['created_at']).timestamp()+10
        stop = audit.lease_clocks(self.manifest['deployment'], comparison_started=started)['comparison_stop']
        write(self.run/'admission/timing-gate.json', {'passed': True})
        write(self.run/'comparison-admission.json', {'kind': 'planning_comparison_admitted',
            'run_sha256': file_sha(self.run/'manifest.json'), 'comparison_started_unix': started,
            'comparison_deadline_unix': stop, 'signal_sha256': file_sha(self.run/'admission/signal-gate.json'),
            'timing_sha256': file_sha(self.run/'admission/timing-gate.json'), 'retained_updates': 80,
            'evaluation_episodes': 320, 'admission_spent_upper_usd': audit.lease_clocks(self.manifest['deployment'], checked=started)['admission_spent']})
        label = audit.schedule.ENDPOINTS[1]; prefix = 'arms/'+label
        for i in (1, 2): self.specs[f'train-{label}-{i}'] = [{'fixture_cohort': i}]
        write(self.run/'inputs/schedule.json', self.specs)
        for role in ('actor', 'learner'):
            self.rpc(role, 'reset_base', {'binding': self.base['binding']}, {'fixture_reset': role}, prefix+'/'+role+'-base-restoration.json', deadline=stop)
        self.rpc('actor', 'collect', {'specs': self.specs[f'train-{label}-1'], 'checkpoint': self.base,
            'prefix': prefix+'/cohorts/1'}, {'fixture_seal': 1}, prefix+'/cohorts/1/seal.json', deadline=stop)
        data = {'cohort': '/workspace/fixture/'+prefix+'/cohorts/1', 'current': self.base,
            'behavior': self.base, 'index': 1, 'prefix': prefix+'/cohorts/1', 'checkpoint_relative': prefix+'/checkpoints/cycle1'}
        self.rpc('learner', 'update', data, {'fixture_update': 1}, prefix+'/cohorts/1/update.json', deadline=stop)
        self.assertEqual(self.check()['verified_successful_rpcs'], 10)
        # A matching successful reply cannot authorize a different publication.
        path = self.run/(prefix+'/cohorts/1/update.json'); write(path, {'fixture_update': 2})
        with self.assertRaisesRegex(ValueError, 'durable publication'): self.check()


if __name__ == '__main__': unittest.main()
