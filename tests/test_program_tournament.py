"""CPU-only tournament provenance/schedule/rating checks; no API or inference."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('controller_tournament_test',ROOT/'scripts/run_program_tournament.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)


class TournamentTests(unittest.TestCase):
    def test_roster_dedup_schedule_pairing_and_new_seeds(self):
        roster={f'model{i}':{'ir_sha256':str(i),'origin':'model'} for i in range(5)}
        roster.update({name:{'ir_sha256':name,'origin':'scripted'} for name in ['nearest','value']})
        schedule,aliases=t.schedule_for(roster,t.SEEDS)
        self.assertFalse(aliases);self.assertEqual(len(schedule),252)
        self.assertEqual(sum(s['match_kind']=='peer_crossplay' for s in schedule),120)
        self.assertEqual(len({s['episode_id'] for s in schedule}),252)
        for identity in roster:
            selected=[s for s in schedule if identity in s['players']]
            self.assertEqual(len(selected),72)
            self.assertEqual(sum(s['players'][0]==identity for s in selected),36)
        self.assertFalse(set(t.SEEDS)&set(range(720000,740006)))
        roster['r1_alias']={'ir_sha256':'0','origin':'model'}
        dedup,aliases=t.schedule_for(roster,t.SEEDS)
        self.assertEqual(dedup,schedule);self.assertEqual(aliases,{'r1_alias':'model0'})

    def test_elo_conservation_equal_draw_and_determinism(self):
        self.assertEqual(t.elo_update(1500,1500,.5),(1500,1500))
        self.assertEqual(t.elo_update(1500,1500,1.),(1512,1488))
        a,b=t.elo_update(1700,1300,0.)
        self.assertAlmostEqual(a+b,3000)
        wrappers=[{'episode_id':'one','players':['a','b'],'record':{'metrics':{'scores':[8,2]}}},
                  {'episode_id':'two','players':['b','a'],'record':{'metrics':{'scores':[3,3]}}}]
        first=t.ratings_for(wrappers,['a','b'])
        self.assertEqual(first,t.ratings_for(copy.deepcopy(wrappers),['a','b']))
        self.assertAlmostEqual(sum(first['ratings'].values()),3000)
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            t.ratings_for(wrappers+[wrappers[0]],['a','b'])
        self.assertIn('not RL checkpoints',first['interpretation'])

    def test_real_retained_provenance_frozen_before_games_and_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)
            result=t.prepare(run)
            self.assertEqual(result['scheduled_games'],252)
            self.assertFalse((run/'matches.jsonl').exists())
            self.assertEqual(result,t.prepare(run))
            with t.runtime_for(run) as (plan,evaluator,runtime,programs,python_programs):
                self.assertEqual(len(programs),7);self.assertEqual(len(python_programs),5)
                self.assertEqual(plan['roster']['controller-python-1']['authoring_model'],'deepseek/deepseek-v4-pro-0813')
                first=plan['schedule'][0]
                names=first['players']
                record=runtime.play([programs[n] for n in names],first['seed'],48,
                                    origins=[plan['roster'][n]['origin'] for n in names])
                wrapper={'episode_id':first['episode_id'],'players':names,'match_kind':first['match_kind'],
                         'record':record,'python_programs':{n:python_programs[n].to_dict() for n in names if n in python_programs}}
                evaluator.verify_wrapper(wrapper,first,runtime,programs,python_programs)
                altered=copy.deepcopy(wrapper);altered['record']['events'][0]['decisions'][0]['action']={'action':'rest'}
                with self.assertRaises(ValueError):
                    evaluator.verify_wrapper(altered,first,runtime,programs,python_programs)
            path=run/'inputs/qwen38-27b-1/generation.json'
            raw=path.read_bytes();value=json.loads(raw);value['text']='def act():\n return {"action":"rest"}'
            path.write_bytes(t.encoded(value))
            with self.assertRaisesRegex(ValueError,'provenance input'):
                with t.runtime_for(run):pass
            path.write_bytes(raw)
            plan=t.read(run/'plan.json');plan['roster']['qwen38-27b-1']['authoring_model']='other-model'
            (run/'plan.json').write_bytes(t.encoded(plan))
            with self.assertRaisesRegex(ValueError,'Authoring model'):
                with t.runtime_for(run):pass


if __name__=='__main__':unittest.main()
