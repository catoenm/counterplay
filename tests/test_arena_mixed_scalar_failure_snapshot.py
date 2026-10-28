"""Terminal retrieval binds draws published during child shutdown."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from selfplay.arena_mixed_scalar import experiment,processes

class FinalFailureSnapshot(unittest.TestCase):
    def test_shutdown_writes_are_bound_before_failed_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);(run/'inputs').mkdir()
            manifest={'phase':'training','config':{'policy':{},'deadline_unix':time.time()+600}}
            (run/'manifest.json').write_text(json.dumps(manifest))
            (run/'inputs/model-file-sha256.json').write_text('{}')
            children=[]
            class Client:
                def __init__(self,role,*args):
                    self.role=role;self.alive=True;self.process=self;children.append(self)
                def ready(self):raise ValueError('fixture failed before result')
                def is_alive(self):return self.alive
                def stop(self):
                    self.alive=False
                    (run/(self.role+'-late-draw.json')).write_text(json.dumps({'raw':'retained'}))
            with patch.object(experiment,'verify_run',return_value=(manifest,{'state':'prepared'})),patch.object(experiment.importlib.metadata,'version',side_effect=lambda name:experiment.VERSIONS[name]),patch.object(experiment,'index_model',return_value={'files':{}}),patch.object(processes,'ProcessClient',Client):
                with self.assertRaisesRegex(ValueError,'fixture failed'):
                    experiment.run_experiment(run,run,'training')
            status=json.loads((run/'status.json').read_text())
            failure=json.loads((run/'failure.json').read_text())
            self.assertEqual(status['state'],'failed')
            self.assertTrue(failure['workers_stopped_before_artifact_scan'])
            self.assertEqual(len(children),2)
            self.assertTrue(all(not c.alive for c in children))
            for role in ('actor','learner'):
                self.assertEqual(status['artifacts'][role+'-late-draw.json'],experiment.file_sha(run/(role+'-late-draw.json')))
if __name__=='__main__':unittest.main()
