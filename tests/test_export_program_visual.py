"""Verify multi-source replay provenance; no network or model calls."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.export_program_visual import compact, digest, load_sources, select_representatives, peer_first_seed
from selfplay.program_arena import play
from selfplay.program_python import compile_python_program


class ProgramVisualTests(unittest.TestCase):
    def setUp(self):
        self.program = compile_python_program("def act():\n    return {'action': 'rest'}\n")
        self.record = play([self.program.compiled] * 2, seed=7, max_rounds=2,
                           origins=['model', 'scripted'])
        self.wrapper = {'episode_id': 'same-id', 'players': ['controller-1', 'idle'],
                        'record': self.record, 'python_programs': {'controller-1': self.program.to_dict()}}

    def source(self, root, name, wrapper=None):
        directory = root / name
        directory.mkdir()
        (directory / 'plan.json').write_text(json.dumps({'model': 'test/model'}))
        path = directory / 'matches.jsonl'
        path.write_text(json.dumps(wrapper or self.wrapper) + '\n')
        return path

    def test_prefixes_only_outer_ids_and_preserves_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [self.source(Path(directory), name) for name in ('one', 'two')]
            before = [p.read_bytes() for p in paths]
            rows, sources, manifest = load_sources(paths)
            self.assertEqual([r['episode_id'] for r in rows], ['one:same-id', 'two:same-id'])
            for row, provenance in zip(rows, manifest):
                self.assertEqual(row['record'], self.record)
                self.assertEqual(provenance['record_sha256'], digest(self.record))
            self.assertEqual([p.read_bytes() for p in paths], before)
            self.assertEqual([s['total'] for s in sources], [1, 1])
            shown = compact(rows[0])
            self.assertEqual(shown['players'], ['test/model · P1 · original', 'idle bot'])
            self.assertEqual(shown['returns'], [0.0, 0.0])
            self.assertNotIn('priority', json.dumps(shown))
            self.assertNotIn('python_source', shown)

    def test_provenance_tampering_and_partial_line_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            altered = deepcopy(self.wrapper)
            altered['python_programs']['controller-1']['python_source'] += '# changed'
            path = self.source(Path(directory), 'one', altered)
            with self.assertRaisesRegex(ValueError, 'provenance'):
                load_sources([path])
            path.write_text(json.dumps(self.wrapper))
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                load_sources([path])

    def test_selection_covers_sources_pairings_without_using_outcomes(self):
        rows = [dict(self.wrapper, episode_id=f'{source}-{i}', visual_source=source)
                for source in ('one', 'two') for i in range(3)]
        selected = select_representatives(rows, 4)
        self.assertEqual([r['episode_id'] for r in selected], ['one-0', 'two-0', 'one-1', 'two-1'])

    def test_tournament_roster_names_versions_and_program_hashes_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapper=deepcopy(self.wrapper)
            wrapper['players']=['controller-1-r2','idle']
            wrapper['python_programs']={'controller-1-r2':self.program.to_dict()}
            path=self.source(Path(directory),'tournament',wrapper)
            roster={'controller-1-r2':{'authoring_model':'test/revised-model','authoring_version':'feedback_round_2',
                                     'ir_sha256':self.program.compiled.digest,'python_sha256':self.program.to_dict()['python_sha256'],
                                     'python_program':self.program.to_dict()},
                    'idle':{'authoring_model':None,'authoring_version':'scripted_anchor',
                            'ir_sha256':self.program.compiled.digest,'python_sha256':None}}
            (path.parent/'plan.json').write_text(json.dumps({'roster':roster}))
            rows,_,_=load_sources([path])
            self.assertEqual(compact(rows[0])['players'][0],'test/revised-model · P1 · feedback 2')
            roster['controller-1-r2']['ir_sha256']='changed'
            (path.parent/'plan.json').write_text(json.dumps({'roster':roster}))
            with self.assertRaisesRegex(ValueError,'frozen tournament roster'):
                load_sources([path])

    def test_peer_selection_uses_seed_and_origin_without_reward_filtering(self):
        peer=deepcopy(self.wrapper);peer['record']['origins']=['model','model']
        later=deepcopy(peer);later['record']['seed']=8
        reverse=deepcopy(peer);reverse['players'].reverse()
        self.assertEqual(peer_first_seed([self.wrapper,peer,later,reverse]),[peer,reverse])


if __name__ == '__main__':
    unittest.main()
