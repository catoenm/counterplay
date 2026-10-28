"""Failed-terminal replay coverage with real CPU fixtures, never neural data."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts import analyze_arena_mixed_scalar as audit
from selfplay.arena_mixed_scalar import cuda_policy, runner
from tests.test_arena_mixed_scalar_review import binding, decide, spec, read, replace_json, reseal


class TerminalSealedIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.fixture_dir=Path(cls.temp.name)
        cls.config=cuda_policy.default_config({'market':.24,'maze':.24,'chess':.24,'planning':.24,'coding':.04})
        cls.journal=runner.Journal(cls.fixture_dir,'a'*64,time.time()+180)
        cls.production=[spec(name,seed) for name in ('market','maze','chess','planning','coding') for seed in (19,21)]
        runner.collect_cohort(decide,cls.production,binding(),cls.journal,'cohorts/1')
        # Fixture-only 224 task episodes exercise the terminal helper's full
        # endpoint denominator; production schedule shape is tested separately.
        cls.evaluation=[{**spec('planning',1000+i,opponent='none'),
                         'episode_id':'fixture-eval-'+str(i),'split':'eval'} for i in range(224)]
        runner.collect_cohort(decide,cls.evaluation,binding(),cls.journal,'evaluation/base')
        cls.state={'policy':{'config':cls.config,'binding':binding()},
                   'learning_rate':cls.config['learning_rate'],'files':{'fixture_checkpoint':'b'*64}}

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def verify(self,run,phase,state,schedule,checkpoint=None):
        with patch.object(audit,'checkpoint_state',return_value=checkpoint or deepcopy(self.state)) as checkpoint_read, \
             patch.object(audit,'verify_native_actions',return_value={'verified':True,'fixture_token_checker':True}) as actions:
            result=audit.verify_sealed_cohort_integrity(run,{'phase':phase},{'state':state},schedule,self.config,None)
        return result,checkpoint_read,actions

    def test_failed_production_seal_is_verified_without_pipeline_publication(self):
        # The sibling evaluation folder is intentionally omitted from this
        # isolated phase fixture, just as actual training has no eval artifacts.
        with tempfile.TemporaryDirectory() as tmp:
            import shutil
            run=Path(tmp);shutil.copytree(self.fixture_dir/'cohorts',run/'cohorts')
            self.assertFalse((run/'cohorts/1/pipeline-commit.json').exists())
            result,checkpoint,actions=self.verify(run,'training','failed',{'train-1':self.production})
            item=result['verified']['cohorts/1']
            self.assertEqual(item['episodes'],10);self.assertTrue(item['replay_verified'])
            self.assertFalse(item['optimizer_publication_claim']);self.assertFalse(item['comparison_scores_emitted'])
            self.assertEqual(checkpoint.call_count,1);self.assertEqual(actions.call_count,1)

    def test_terminal_partial_evaluation_seal_is_audited_without_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            import shutil
            run=Path(tmp);shutil.copytree(self.fixture_dir/'evaluation',run/'evaluation')
            self.assertFalse((run/'evaluation-seal.json').exists())
            result,checkpoint,actions=self.verify(run,'evaluation','failed',{'eval-base':self.evaluation})
            item=result['verified']['evaluation/base']
            self.assertEqual(item['episodes'],224);self.assertFalse(item['comparison_scores_emitted'])
            self.assertFalse(item['optimizer_publication_claim'])
            self.assertEqual(actions.call_args.kwargs['opponent_binding'],binding())

    def test_active_partial_evaluation_is_not_read_or_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);folder=run/'evaluation/base';folder.mkdir(parents=True)
            (folder/'seal.json').write_text('Intentionally unread invalid fixture bytes')
            with patch.object(audit,'load_sealed_cohort',side_effect=AssertionError('Must not read active partial eval')):
                result,checkpoint,actions=self.verify(run,'evaluation','running',{})
            self.assertEqual(result['verified'],{})
            self.assertEqual(result['active_evaluation_seals_not_read'],['evaluation/base'])
            checkpoint.assert_not_called();actions.assert_not_called()

    def test_schedule_or_saved_checkpoint_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            import shutil
            run=Path(tmp);shutil.copytree(self.fixture_dir/'cohorts',run/'cohorts')
            schedule=deepcopy(self.production);schedule[0]['seed']+=1
            with self.assertRaisesRegex(ValueError,'frozen schedule'):
                self.verify(run,'training','failed',{'train-1':schedule})
            bad=deepcopy(self.state);bad['policy']['binding']=binding(1)
            with self.assertRaisesRegex(ValueError,'actual saved checkpoint'):
                self.verify(run,'training','failed',{'train-1':self.production},bad)

    def test_failed_seal_still_rejects_terminal_score_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            import shutil
            run=Path(tmp);shutil.copytree(self.fixture_dir/'cohorts',run/'cohorts')
            folder=run/'cohorts/1';episodes=read(folder/'episodes.json')
            episodes[0]['returns'][0]+=0.00001
            replace_json(folder/'episodes.json',episodes);reseal(folder)
            with self.assertRaisesRegex(ValueError,'returns|reward|Terminal'):
                self.verify(run,'training','failed',{'train-1':self.production})


if __name__=='__main__':unittest.main()
