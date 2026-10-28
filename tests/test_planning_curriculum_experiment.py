"""Full CPU worker wiring with a fake neural backend and real game/replay IO.

No pretrained weights, CUDA, subprocess, provider or network calls are made.
Synthetic checkpoint JSON is deliberately not a scientific tensor fixture.
"""
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from selfplay.environment import step
from selfplay.planning_curriculum import experiment, processes, runner, schedule


def json_write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True))


class FakePolicy:
    """Deterministic public-state solver/failure fixtures, not a model."""
    def __init__(self,role,config,signal_pass):
        self.role,self.config,self.signal_pass=role,config,signal_pass
        self.binding=None;self.value=0;self.step=0;self.rng='fixture-shared-rng';self.poisoned=False
    def _capture(self):return deepcopy({'binding':self.binding,'value':self.value,'step':self.step,'rng':self.rng})
    def _restore(self,state):
        self.binding=state['binding'];self.value=state['value'];self.step=state['step'];self.rng=state['rng']
    def _state_identity(self):return self._capture()
    def verify_frozen(self):return True
    def save_checkpoint(self,folder,version=0):
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        self.binding={'checkpoint_id':'base' if version==0 else 'cycle'+str(version),
            'version':version,'adapter_sha256':runner.digest(self.value),'fixture_only':True}
        json_write(folder/'adapter.safetensors',{'value':self.value})
        json_write(folder/'optimizer.pt',{'step':self.step,'rng':self.rng})
        json_write(folder/'policy.json',{'binding':self.binding})
        return {'files':{name:runner.file_sha(folder/name) for name in ('adapter.safetensors','optimizer.pt','policy.json')},
                'policy':{'binding':deepcopy(self.binding)},'step':self.step}
    def load_checkpoint(self,folder,files):
        folder=Path(folder)
        if any(runner.file_sha(folder/name)!=sha for name,sha in files.items()):raise ValueError('Fixture checksum')
        self.value=json.loads((folder/'adapter.safetensors').read_text())['value']
        state=json.loads((folder/'optimizer.pt').read_text());self.step=state['step'];self.rng=state['rng']
        self.binding=json.loads((folder/'policy.json').read_text())['binding']
        return {'files':files,'policy':{'binding':deepcopy(self.binding)},'step':self.step}
    def numerical_smoke(self,requests):return {'passed':True,'state_restored':True,'fixture_only':True,'requests':requests}
    def choose(self,requests,binding):
        assert binding==self.binding
        rows=[]
        for request in requests:
            obs=request['observation'];state=tuple(obs['state']);goal=tuple(obs['goal']);modulus=obs['modulus']
            success=(not request['episode_id'].startswith('screen-') or
                self.signal_pass and request['episode_id'] in {f'screen-easy-depth1-{i:02d}' for i in (0,2,4,6)})
            if success:
                queue=deque([(state,())]);seen={state};path=None
                while queue:
                    current,plan=queue.popleft()
                    if current==goal:path=plan;break
                    for action in request['actions']:
                        following=step(current,action['tool'],modulus)
                        if following not in seen:seen.add(following);queue.append((following,plan+(action['tool'],)))
                action={'tool':path[0]}
            else:
                action=next(x for x in request['actions'] if step(state,x['tool'],modulus)!=goal)
            rows.append({**deepcopy(request),'checkpoint':deepcopy(binding),'action':deepcopy(action),
                'action_index':request['actions'].index(action),'input_ids':[1]*100,
                'old_log_probs':[-1.3862943611198906]*4,'old_log_prob':-1.3862943611198906,'entropy':1.3862943611198906})
        return rows
    def choose_profiled(self,requests,binding):
        rows=self.choose(requests,binding)
        return {'rows':rows,'timing':{'kind':'planning_screen_sampling_timing_v1','elapsed_seconds':float(len(rows)),
            'nonforward_seconds':0.,'row_sha256':[runner.digest(row) for row in rows],
            'forward_timings':[{'row_sha256':runner.digest(row),'input_tokens':100,'training':False,
                'grad_enabled':False,'seconds':1.} for row in rows]}}
    def gate_at_checkpoint(self,rows,folder,files):
        return {'passed':True,'state_restored':True,'fixture_only':True,'rows':len(rows)}
    def timing_probe(self,episodes,ids,folder):
        assert self.step==0 and [e['episode_id'] for e in episodes]==ids
        return {'passed':True,'fixture_only':True,'elapsed_seconds':3.,'gate_seconds':1.,
            'temporary_update':{'elapsed_seconds':1.},'checkpoint_seconds':.5,
            'mature_state_roundtrip':{'passed':True,'state_before':{'fixture_adam_step':1},
                'state_after':{'fixture_adam_step':1},'seconds':.25,'optimizer_steps':0,'model_forwards':0}}
    def train_cohort(self,episodes,binding,gate):
        assert len(episodes)==2 and gate['passed']
        self.value+=1;self.step+=1
        return {'optimizer_steps':1,'fixture_only':True,'source_step':self.step-1,'target_step':self.step}


class FakeFuture:
    def __init__(self,client,command):self.client,self.command=client,command
    def result(self):
        client=self.client;command=self.command
        if client.pending is not self:raise ValueError('Fixture future reused')
        try:
            if client.fail_prefix and command['data'].get('prefix')==client.fail_prefix:
                raise TimeoutError('Fixture deadline during an unresolved draw')
            value=processes._dispatch(client.policy,command,client.parent_journal)
            client.journal.write('processes/'+command['id']+'-result.json',{'id':command['id'],'passed':True,'result':value})
            client.pending=None
            return value
        except BaseException:
            client.policy.poisoned=True
            raise


class FakeClient:
    instances=[];events=[];signal_pass=True;fail_prefix=None
    def __init__(self,role,model_dir,config,journal):
        self.role,self.journal,self.parent_journal=role,journal,journal
        self.counter=0;self.pending=None;self.alive=True
        self.policy=FakePolicy(role,config,self.signal_pass)
        self.process=SimpleNamespace(is_alive=lambda:self.alive)
        self.instances.append(self)
    def ready(self):return {'ready':True,'base_identity':{'fixture':'same-original-weights'}}
    def request(self,method,data):
        if self.pending is not None:raise ValueError('Fixture queue already occupied')
        self.journal.admit();data={**deepcopy(data),'stage_deadline_unix':self.journal.deadline}
        identifier=f'{self.role}-{self.counter:06d}';self.counter+=1
        self.events.append({'role':self.role,'method':method,'data':deepcopy(data),'state':self.policy._state_identity()})
        self.journal.intent('processes/'+identifier+'-intent.json','fixture_rpc',{'method':method,'data_sha256':runner.digest(data)})
        self.pending=FakeFuture(self,{'id':identifier,'method':method,'data':data})
        return self.pending
    def call(self,method,data):return self.request(method,data).result()
    def submit(self,specs,checkpoint,prefix):return self.request('collect',{'specs':specs,'checkpoint':checkpoint,'prefix':prefix})
    def stop(self):self.alive=False


class PlanningExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.stages=schedule.make_schedules([])[0]
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.run=Path(self.temp.name)
        FakeClient.instances=[];FakeClient.events=[];FakeClient.signal_pass=True;FakeClient.fail_prefix=None
        now=time.time();plan={'created_at':datetime.fromtimestamp(now-1,timezone.utc).isoformat(),
            'deadline_unix':now+43199,'gpu_hourly_rate':5.29,'guard_hourly_rate':.14,
            'storage_hourly_rate':.02,'other_reserved_cost_usd':0.}
        self.manifest={'kind':'planning_curriculum_study_v1','deployment':plan,'config':{'policy':{'inference_batch_size':8}}}
        for name,value in [('manifest.json',self.manifest),('inputs/schedule.json',self.stages),
            ('inputs/model-info.json',{}),('inputs/model-file-sha256.json',{}),('inputs/context-bound.json',{}),
            ('inputs/implementation-review.json',{'fixture_only':True})]:json_write(self.run/name,value)
        json_write(self.run/'status.json',{'state':'prepared','manifest_sha256':runner.file_sha(self.run/'manifest.json')})
    def execute(self,timing_pass=True):
        self.commit_states=[]
        original_commit=experiment.commit_status
        def commit(run,manifest,state,stage):
            if state in experiment.TERMINAL:self.assertTrue(all(not c.alive for c in FakeClient.instances))
            self.commit_states.append((state,stage));return original_commit(run,manifest,state,stage)
        with patch.object(processes,'ProcessClient',FakeClient), \
             patch.object(experiment,'verify_run',return_value=(self.manifest,{'state':'prepared'})), \
             patch.object(experiment,'VERSIONS',{}),patch.object(experiment,'index_model',return_value={'files':{}}), \
             patch.object(experiment,'compare_actor_reports',return_value={'passed':True}), \
             patch.object(experiment,'timing_gate',return_value={'passed':timing_pass,'fixture_only':True}), \
             patch.object(experiment,'commit_status',side_effect=commit):
            return experiment.run_experiment(self.run,self.run/'unused-model')
    def methods(self):return [e['method'] for e in FakeClient.events]
    def test_failed_signal_stops_without_probe_or_training(self):
        FakeClient.signal_pass=False;self.execute()
        self.assertEqual(json.loads((self.run/'status.json').read_text())['state'],'admission_signal_failed')
        self.assertNotIn('timing_probe',self.methods());self.assertNotIn('update',self.methods())
        _,episodes=runner.load_sealed_cohort(self.run/'admission/screen');self.assertEqual(len(episodes),64)
        self.assertFalse((self.run/'comparison-admission.json').exists())
    def test_failed_timing_retains_probe_but_no_retained_update(self):
        self.execute(timing_pass=False)
        self.assertEqual(json.loads((self.run/'status.json').read_text())['state'],'admission_timing_failed')
        self.assertEqual(self.methods().count('timing_probe'),1);self.assertNotIn('update',self.methods())
        self.assertTrue((self.run/'admission/probe-optimizer-intent.json').exists())
        self.assertTrue(all(c.policy.step==0 for c in FakeClient.instances))
    def test_pass_executes_exact_four_arms_and_five_matched_evaluations(self):
        self.execute()
        self.assertEqual(json.loads((self.run/'status.json').read_text())['state'],'complete')
        self.assertEqual(self.methods().count('timing_probe'),1)
        self.assertEqual(self.methods().count('update'),80)
        resets=[e for e in FakeClient.events if e['method']=='reset_base'];self.assertEqual(len(resets),8)
        for seed in schedule.TRAINING_SEEDS:
            for arm in schedule.ARMS:
                prefix=f'arms/{seed}-{arm}'
                proof=json.loads((self.run/prefix/'learner-base-restoration.json').read_text())
                self.assertEqual(proof['after']['step'],0);self.assertEqual(proof['after']['value'],0)
                completion=json.loads((self.run/prefix/'training-complete.json').read_text())
                self.assertEqual(completion['attempted_cohorts'],20);self.assertEqual(completion['actual_optimizer_steps'],20)
                self.assertEqual(completion['endpoint']['optimizer_step'],20)
                for i in range(1,21):
                    seal,episodes=runner.load_sealed_cohort(self.run/prefix/f'cohorts/{i}')
                    self.assertEqual(len(episodes),2);self.assertEqual(seal['binding']['version'],max(0,i-2))
                    gate=json.loads((self.run/prefix/f'cohorts/{i}/gate-intent.json').read_text())
                    self.assertEqual(gate['learner']['binding']['version'],i-1)
        for label in schedule.ENDPOINTS:
            _,episodes=runner.load_sealed_cohort(self.run/'evaluation'/label)
            self.assertEqual(len(episodes),64)
            self.assertEqual([e['schedule']['paired_seed_key'] for e in episodes],
                             [s['paired_seed_key'] for s in self.stages['eval-base']])
        self.assertEqual(len(list((self.run/'arms').glob('*/cohorts/*/pipeline-commit.json'))),80)
        self.assertEqual(len(list((self.run/'evaluation').glob('*/seal.json'))),5)
    def test_timed_out_draw_remains_unresolved_and_children_stop_before_terminal(self):
        FakeClient.fail_prefix='arms/742101-control/cohorts/2'
        with self.assertRaises(TimeoutError):self.execute()
        self.assertEqual(json.loads((self.run/'status.json').read_text())['state'],'failed')
        actor=next(c for c in FakeClient.instances if c.role=='actor')
        self.assertIsNotNone(actor.pending);self.assertTrue(actor.policy.poisoned)
        self.assertEqual(self.methods().count('update'),1)
        self.assertFalse((self.run/'comparison-complete.json').exists())
        self.assertFalse((self.run/'evaluation').exists())
        self.assertTrue(all(not c.alive for c in FakeClient.instances))
    def test_expired_or_extended_stage_rejected_before_dispatch(self):
        journal=runner.Journal(self.run,'a'*64,time.time()+60)
        for deadline in (time.time()-1,journal.deadline+1):
            policy=SimpleNamespace()
            with self.assertRaises(TimeoutError):
                processes._dispatch(policy,{'method':'publish_base','data':{'stage_deadline_unix':deadline}},journal)
        self.assertFalse((self.run/'checkpoints').exists())
    def test_actual_rpc_timeout_keeps_pending_for_no_repeat(self):
        client=object.__new__(processes.ProcessClient);client.role='actor';client.pending='actor-000000'
        client.journal=runner.Journal(self.run,'a'*64,time.time()-1)
        with self.assertRaises(TimeoutError):client._receive(client.pending)
        self.assertEqual(client.pending,'actor-000000')
        with self.assertRaisesRegex(ValueError,'Queue capacity one'):client.request('collect',{})

    def test_one_shutdown_exception_does_not_skip_other_child_or_publish_terminal(self):
        FakeClient.signal_pass=False
        attempts=[]
        def fail_one(client):
            attempts.append(client.role)
            if client.role=='learner':raise RuntimeError('Fixture child survives shutdown')
            client.alive=False
        with patch.object(FakeClient,'stop',fail_one):
            with self.assertRaises(RuntimeError):self.execute()
        self.assertEqual(attempts,['learner','actor'])
        self.assertEqual(json.loads((self.run/'status.json').read_text())['state'],'running')
        self.assertFalse(next(c for c in FakeClient.instances if c.role=='actor').alive)

    def test_real_temporary_prepare_and_public_schema_roundtrip(self):
        from scripts import runpod_planning_curriculum as lifecycle
        from selfplay.planning_curriculum.admission import build_context_bound
        root=self.run/'fixture-root';(root/'scripts/cloud').mkdir(parents=True)
        shutil.copyfile(experiment.ROOT/'scripts/cloud/cardinal_guard.py',root/'scripts/cloud/cardinal_guard.py')
        key=root/'public.pub';key.write_text('ssh-ed25519 fixture-public-only')
        deployment=root/'lease'
        with patch.object(lifecycle,'ROOT',root):lifecycle.prepare(deployment,key)
        public=experiment.public_deployment(json.loads((deployment/'plan.json').read_text()))
        self.assertEqual(experiment.public_deployment(public),public)
        self.assertNotIn('guard_secret_path',public);self.assertNotIn('ssh_public_key',public)
        model_inputs=experiment.ROOT/'runs/arena-mixed-preflight-v1'
        preflight=root/'preflight';preflight.mkdir()
        # Production preflight has already snapshotted history before writing
        # this study's schedule. Reuse that snapshot; do not exclude the new
        # study from itself by rescanning its now-present preflight schedule.
        prior_path=experiment.ROOT/'runs/planning-curriculum-preflight-v1/prior-scenario-inventory.json'
        inventory=json.loads(prior_path.read_text()) if prior_path.is_file() else schedule.historical_inventory(experiment.ROOT)
        stages,selection,verification=schedule.make_schedules(inventory['fingerprints'])
        proof=build_context_bound(model_inputs/'tokenizer')
        json_write(preflight/'context-proof.json',proof)
        json_write(preflight/'context-review.json',{'passed':True,'fixture_only':True,'proof_sha256':proof['proof_sha256'],
            'proof_file_sha256':runner.file_sha(preflight/'context-proof.json')})
        context={**proof,'review_sha256':runner.file_sha(preflight/'context-review.json')}
        values={'context-bound.json':context,'overhead-contract.json':experiment.OVERHEAD_CONTRACT,
            'prior-scenario-inventory.json':inventory,'schedule.json':stages,'schedule-verification.json':verification,
            'scenario-selections.json':selection,'implementation-review.json':{'passed':True,'fixture_only':True,
                'sources':experiment.sources(),'overhead_contract_sha256':runner.digest(experiment.OVERHEAD_CONTRACT)}}
        for name,value in values.items():json_write(preflight/name,value)
        prepared=root/'prepared'
        json_write(preflight/'prior-scenario-inventory.json',{**inventory,'verified':False})
        with self.assertRaisesRegex(ValueError,'Verified nonempty historical'):
            experiment.prepare(prepared,deployment/'plan.json',preflight,model_inputs)
        self.assertFalse(prepared.exists())
        json_write(preflight/'prior-scenario-inventory.json',inventory)
        manifest=experiment.prepare(prepared,deployment/'plan.json',preflight,model_inputs)
        checked,status=experiment.verify_run(prepared)
        self.assertEqual(manifest,checked);self.assertEqual(status['state'],'prepared')
        self.assertEqual(manifest['config']['maximum_policy_episodes'],544)
        self.assertEqual(manifest['config']['attempted_retained_updates'],80)
        self.assertEqual(manifest['deployment'],public)
        self.assertFalse((prepared/'execution-intent.json').exists())
        with self.assertRaises(FileExistsError):experiment.prepare(prepared,deployment/'plan.json',preflight,model_inputs)
        (prepared/'inputs/schedule.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'Committed input'):experiment.verify_run(prepared)

    def test_overhead_rejects_nonfinite_or_unrestored_mature_measurement(self):
        profile=FakePolicy('learner',{},True).timing_probe([],[],None)
        self.assertGreater(experiment.main_overhead(profile,.1)['fixed_main_overhead_seconds'],1800)
        for value in (float('nan'),float('inf'),-1,0,True):
            bad=deepcopy(profile);bad['mature_state_roundtrip']['seconds']=value
            with self.assertRaisesRegex(ValueError,'Finite positive'):experiment.main_overhead(bad,.1)
        bad=deepcopy(profile);bad['mature_state_roundtrip']['state_after']={'fixture_adam_step':2}
        with self.assertRaisesRegex(ValueError,'restoration'):experiment.main_overhead(bad,.1)


if __name__=='__main__':unittest.main()
