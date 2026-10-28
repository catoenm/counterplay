import importlib.util
from pathlib import Path
import unittest

from selfplay.program_arena import compile_program, play


spec = importlib.util.spec_from_file_location('rl_visual', Path(__file__).parents[1]/'scripts/export_program_rl_visual.py')
visual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visual)


class ReplayIdentityTests(unittest.TestCase):
    def setUp(self):
        self.program = compile_program({'version':1,'body': [{'emit': {'action': 'rest'}}]})
        self.candidate = {'id':'sample-after','label':'Step 1 · evaluation draw 1',
                          'ir_sha256':self.program.digest,'checkpoint':'step1','model':'test-model'}
        self.opponent = dict(self.candidate, id='frozen-opponent',label='Base · frozen opponent draw 2',checkpoint='base')
        self.row = {'classification':{'valid':True},'generation':'sample-after','reward':0.0}

    def test_same_program_keeps_distinct_checkpoint_and_sample_identity(self):
        record = play([self.program,self.program],seed=7,max_rounds=2,origins=['model','model'])
        for seat in (0,1):
            episode=visual.compact_match(self.row,{'replay':record,'candidate_seat':seat},
                                        self.candidate,self.opponent,'evaluation/step1.json','after',seat)
            self.assertEqual(episode['identities'][seat],'sample-after')
            self.assertEqual(episode['checkpoint_labels'][seat],'step1')
            self.assertEqual(episode['checkpoint_labels'][1-seat],'base')
            self.assertEqual(episode['record_sha256'],visual.digest(record))
            self.assertFalse(record['gradient_training'])  # Inner simulator record stays unchanged.

    def test_false_controller_identity_rejected(self):
        record=play([self.program,self.program],seed=7,max_rounds=2,origins=['model','model'])
        with self.assertRaisesRegex(ValueError,'identity differs'):
            visual.compact_match(self.row,{'replay':record,'candidate_seat':0},
                                 dict(self.candidate,ir_sha256='wrong'),self.opponent,'test','training',0)

    def test_invalid_authorship_cannot_acquire_a_game(self):
        row={'classification':{'valid':False},'matches':[],'game_return':None,'reward_kind':'invalid_authorship'}
        visual.validate_invalid(row)
        with self.assertRaisesRegex(ValueError,'no game'):
            visual.validate_invalid(dict(row,matches=[{'fabricated':True}]))
        with self.assertRaisesRegex(ValueError,'cannot become'):
            visual.compact_match(row,{},self.candidate,self.opponent,'test','training',0)


if __name__ == '__main__':
    unittest.main()
