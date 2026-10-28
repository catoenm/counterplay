"""Independent CPU integration/mutation fixtures; no neural or study results."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import unittest

from selfplay.arena_mixed import envs, runner


def binding(version=0):
    return {'checkpoint_id': 'base' if version == 0 else f'cycle{version}',
            'version': version, 'adapter_sha256': str(version % 10) * 64,
            'base_identity': {'fixture': True}}


def spec(name, seed, seats=None, opponent='self'):
    fingerprint = envs.scenario_fingerprint(name, seed)
    return {'episode_id': f'fixture-{name}-{seed}', 'environment': name, 'seed': seed,
            'regime': 'train', 'split': 'train', 'scenario_sha256': fingerprint,
            'scenario_family_id': name + ':' + fingerprint,
            'learner_seats': envs.protocol_config()['roles'][name] if seats is None else seats,
            'opponent': opponent}


def decide(requests, identity):
    # Deterministic fixture: choose a visible action, with no strategy access.
    return [{**deepcopy(row), 'checkpoint': deepcopy(identity),
             'action': deepcopy(row['actions'][0]), 'fixture_draw': i}
            for i, row in enumerate(requests)]


def read(path):
    return json.loads(Path(path).read_text())


def replace_json(path, value):
    # Deliberately mutate fixture evidence to test independent validation.
    Path(path).write_text(runner.canonical(value) + '\n')


def reseal(folder):
    seal = read(folder/'seal.json')
    for relative in seal['draw_files']:
        seal['draw_files'][relative] = runner.file_sha(folder/relative)
    seal['episodes_sha256'] = runner.file_sha(folder/'episodes.json')
    seal['replay_sha256'] = runner.file_sha(folder/'replay-verification.json')
    replace_json(folder/'seal.json', seal)


class IndependentRolloutReview(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.journal = runner.Journal(self.folder, 'a'*64, time.time()+60)

    def collect(self, specs=None, **kwargs):
        runner.collect_cohort(decide, specs or [spec('market', 17)], binding(),
                              self.journal, 'cohort', **kwargs)
        return self.folder/'cohort'

    def test_all_five_real_replays_have_exact_raw_action_coverage(self):
        folder = self.collect([spec(name, 19) for name in envs.ENVIRONMENTS])
        seal, episodes = runner.load_sealed_cohort(folder)
        self.assertEqual(seal['episodes'], 5)
        for episode in episodes:
            self.assertTrue(envs.verify_episode(episode)['verified'])
            expected = sum(len(event['active_players']) for event in episode['events'])
            self.assertEqual(len(episode['transitions']), expected)
            for row in episode['transitions']:
                self.assertEqual(row['scenario_family_id'], episode['scenario_family_id'])
                self.assertNotIn('initial_state', row)
                self.assertNotIn('audit_only', row['observation'])
        solo = [e for e in episodes if e['role_count'] == 1]
        self.assertTrue(all(e['learner_seats'] == [0] and e['returns'][1] == 0 for e in solo))

    def test_existing_intent_prevents_another_sampler_invocation(self):
        calls = []
        def counted(*args):
            calls.append(True); return decide(*args)
        specs = [spec('planning', 17)]
        runner.collect_cohort(counted, specs, binding(), self.journal, 'cohort')
        before = len(calls)
        with self.assertRaises(FileExistsError):
            runner.collect_cohort(counted, specs, binding(), self.journal, 'cohort')
        self.assertEqual(len(calls), before)

    def test_resealed_episode_cannot_change_original_sample_metadata(self):
        folder = self.collect()
        episodes = read(folder/'episodes.json')
        episodes[0]['transitions'][0]['fixture_draw'] = 999
        replace_json(folder/'episodes.json', episodes); reseal(folder)
        with self.assertRaisesRegex(ValueError, 'original|draw|request'):
            runner.load_sealed_cohort(folder)

    def test_resealed_episode_cannot_reclassify_a_real_learner_seat(self):
        folder = self.collect()
        episodes = read(folder/'episodes.json'); episode = episodes[0]
        episode['learner_seats'] = [0]; episode['schedule']['learner_seats'] = [0]
        episode['schedule']['opponent'] = 'heuristic'
        episode['opponent_transitions'] = [r for r in episode['transitions'] if r['player'] == 1]
        episode['transitions'] = [r for r in episode['transitions'] if r['player'] == 0]
        replace_json(folder/'episodes.json', episodes); reseal(folder)
        with self.assertRaises(ValueError):
            runner.load_sealed_cohort(folder)

    def test_base_opponent_missing_decisions_fail_even_after_resealing(self):
        folder = self.collect([spec('market', 17, [0], 'base')],
                              opponent_decide=decide, opponent_binding=binding())
        runner.load_sealed_cohort(folder)
        episodes = read(folder/'episodes.json')
        episodes[0]['opponent_transitions'].pop()
        replace_json(folder/'episodes.json', episodes); reseal(folder)
        with self.assertRaises(ValueError):
            runner.load_sealed_cohort(folder)

    def test_wrong_model_response_is_retained_before_validation_failure(self):
        def wrong(requests, identity):
            rows = decide(requests, identity)
            rows[0]['observation']['undeclared_private_value'] = 123
            return rows
        with self.assertRaisesRegex(ValueError, 'provenance'):
            runner.collect_cohort(wrong, [spec('planning', 3)], binding(), self.journal, 'cohort')
        self.assertTrue((self.folder/'cohort/batches/000000.json').is_file())
        self.assertFalse((self.folder/'cohort/seal.json').exists())

    def test_short_batch_response_is_retained_without_retry(self):
        calls = []
        def short(requests, identity):
            calls.append(True)
            return []
        with self.assertRaises(ValueError):
            runner.collect_cohort(short, [spec('planning', 3)], binding(), self.journal, 'cohort')
        self.assertEqual(calls, [True])
        self.assertTrue((self.folder/'cohort/batches/000000.json').is_file())
        self.assertEqual(read(self.folder/'cohort/batches/000000.json')['rows'], [])
        self.assertFalse((self.folder/'cohort/seal.json').exists())

    def test_pipeline_submits_next_actor_before_prior_update_with_actual_lag_one(self):
        events = []
        class Future:
            def __init__(self, number, checkpoint): self.number, self.checkpoint = number, checkpoint
            def result(self):
                events.append(('resolved', self.number))
                return {'binding': self.checkpoint['binding']}
        class Actor:
            def submit(self, specs, checkpoint, prefix):
                number = int(prefix.rsplit('/', 1)[1])
                events.append(('submitted', number, checkpoint['binding']['version']))
                return Future(number, checkpoint)
        class Learner:
            def update(self, cohort, checkpoint, index):
                events.append(('updated', index, checkpoint['binding']['version']))
                return {'checkpoint': {'binding': binding(index)},
                        'report': {'optimizer_steps': int(index != 2)}}
        result = runner.run_pipeline(Actor(), Learner(), self.journal,
                    {f'train-{i}': [] for i in range(1, 5)}, 4, {'binding': binding()})
        self.assertEqual([e for e in events if e[0] == 'submitted'],
                         [('submitted', 1, 0), ('submitted', 2, 0),
                          ('submitted', 3, 1), ('submitted', 4, 2)])
        for i in range(1, 4):
            self.assertLess(events.index(('submitted', i+1, i-1)), events.index(('updated', i, i-1)))
        self.assertEqual(result['actual_optimizer_steps'], 3)
        self.assertEqual(result['endpoint']['binding']['version'], 4)


if __name__ == '__main__':
    unittest.main()
