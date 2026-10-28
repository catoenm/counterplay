"""No SSH, provider, model, allocation, or real child processes in these tests."""
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import planning_curriculum_dispatch as dispatch


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.deployment=self.root/'runs/deployment';self.deployment.mkdir(parents=True)
        self.run=self.root/'runs/study';self.run.mkdir()
        self.root_patch=patch.object(dispatch,'ROOT',self.root);self.root_patch.start();self.addCleanup(self.root_patch.stop)
        source=self.root/'scripts/fixture.py';source.parent.mkdir();source.write_text('value = 1\n')
        archived=self.run/'source/scripts/fixture.py';archived.parent.mkdir(parents=True);archived.write_bytes(source.read_bytes())
        task=self.run/'inputs/task.json';task.parent.mkdir();task.write_text('{"fixture": true}\n')
        self.plan={'intent':'a'*32,'project':'cardinal-gpu','deadline_unix':9999999999,'phase':'planning_curriculum'}
        self.manifest={'sources':{'scripts/fixture.py':dispatch.sha(source)},'inputs':{'inputs/task.json':dispatch.sha(task)},'deployment':self.plan}
        dispatch.write(self.run/'manifest.json',self.manifest)
        dispatch.write(self.run/'status.json',{'state':'prepared','artifacts':{}})
        dispatch.write(self.run/'deployment.json',self.plan)
        dispatch.write(self.deployment/'plan.json',self.plan)
        self.state={'roles':{},'reservation_applied':False}
        self.prepare_patch=patch.object(dispatch,'prepared',side_effect=lambda *a:(deepcopy(self.plan),deepcopy(self.state),deepcopy(self.manifest)))
        self.prepare_patch.start();self.addCleanup(self.prepare_patch.stop)

    def package(self):
        return dispatch.package(self.deployment,self.run)

    def active(self):
        self.state={'reservation_applied':True,'roles':{'gpu':{'pod_id':'fixture-gpu'},'guard':{'pod_id':'fixture-guard'}}}
        key=self.root/'private-key';key.write_text('fixture only')
        self.key=key
        known=self.deployment/'known_hosts';known.write_text('fixture host key\n')
        self.hardware={'host':'127.0.0.1','port':2222,'user':'root','key':str(key),'known_hosts':str(known),
            'known_hosts_sha256':dispatch.sha(known),'intent':self.plan['intent'],'pod_id':'fixture-gpu'}
        self.live={'at':'fixture','resources':{'gpu':{'id':'fixture-gpu','ssh':{'direct':{'host':'127.0.0.1','port':2222,'username':'root'}}}}}

    def stage_ready(self):
        inventory=self.package();self.active()
        dispatch.write(self.deployment/'hardware-check.json',self.hardware)
        dispatch.write(self.deployment/'remote-package-readiness.json',{
            'manifest_sha256':inventory['manifest_sha256'],'archive_sha256':inventory['archive_sha256'],
            'verified_members':len(inventory['files']),'venv_ready':True,
            'hardware_sha256':dispatch.sha(self.deployment/'hardware-check.json')})
        return inventory

    def supervisor(self,command,**kwargs):
        origin={'manifest_sha256':dispatch.sha(self.run/'manifest.json'),'run':str(self.run),
            'remote_run':dispatch.REMOTE_ROOT+'/'+dispatch.relative(self.run),'deployment':self.plan,
            'host':'127.0.0.1','port':2222,'user':'root'}
        dispatch.write(self.deployment/'supervisor-origin.json',origin)
        return SimpleNamespace(pid=4321,poll=lambda:None)

    def test_import_has_no_actions(self):
        spec=importlib.util.spec_from_file_location('dispatch_import_fixture',Path(dispatch.__file__))
        module=importlib.util.module_from_spec(spec)
        with patch.object(subprocess,'run',side_effect=AssertionError('subprocess on import')),patch.object(subprocess,'Popen',side_effect=AssertionError('spawn on import')):
            spec.loader.exec_module(module)

    def test_package_exact_allowlist_and_exclusive_attempt(self):
        secret=self.root/'.secrets/key';secret.parent.mkdir();secret.write_text('DO NOT PACKAGE')
        (self.run/'unlisted.json').write_text('not in package')
        inventory=self.package()
        self.assertEqual(len(inventory['files']),6)
        self.assertNotIn('.secrets/key',inventory['files'])
        self.assertNotIn('runs/study/unlisted.json',inventory['files'])
        with tarfile.open(self.deployment/'source-package.tar.gz') as tar:
            self.assertEqual(sorted(m.name for m in tar),sorted(inventory['files']))
        with self.assertRaises(FileExistsError):self.package()
        self.assertEqual(dispatch.inventory_for(self.deployment,self.run,self.manifest),inventory)

    def test_changed_source_input_and_unsafe_paths_rejected(self):
        self.package();(self.run/'inputs/task.json').write_text('{}')
        with self.assertRaises(ValueError):dispatch.inventory_for(self.deployment,self.run,self.manifest)
        for name in ('../escape','/absolute','a/../b','.secrets/key','a\\b','a//b'):
            with self.assertRaises(ValueError):dispatch.safe_member(name)
        target=self.root/'scripts/link.py';target.symlink_to(self.root/'scripts/fixture.py')
        with self.assertRaises(ValueError):dispatch.regular(target)

    def test_safe_remote_extraction_and_plain_import_before_pip(self):
        inventory=self.package();destination=self.root/'extracted';venv=self.root/'fixture-venv'
        with patch.object(dispatch,'REMOTE_ROOT',str(destination)),patch.object(dispatch,'REMOTE_ARCHIVE',str(self.deployment/'source-package.tar.gz')),patch.object(dispatch,'REMOTE_VENV',str(venv)):
            code=dispatch.extraction_code(inventory)
        namespace={'Path':Path,'hashlib':hashlib,'json':json,'subprocess':subprocess}
        with patch.object(subprocess,'run',return_value=SimpleNamespace(returncode=0)) as calls,redirect_stdout(io.StringIO()) as output:
            exec(compile(code,'<fixture-extraction>','exec'),namespace)
        self.assertEqual(calls.call_count,2)
        self.assertIn('--system-site-packages',calls.call_args_list[0].args[0])
        self.assertIn('import selfplay.planning_curriculum.experiment',calls.call_args_list[1].args[0][-1])
        self.assertTrue(json.loads(output.getvalue())['imports_before_pip'])
        for name,digest in inventory['files'].items():self.assertEqual(dispatch.sha(destination/name),digest)
        with self.assertRaises(AssertionError):exec(compile(code,'<repeat-extraction>','exec'),namespace)

    def test_stage_failure_preserved_and_cannot_retry(self):
        self.package();self.active();Path(self.hardware['known_hosts']).unlink()
        blob=b'\x00\x00\x00\x0bssh-ed25519'+b'\x00\x00\x00\x20'+b'x'*32
        scanned='[127.0.0.1]:2222 ssh-ed25519 '+base64.b64encode(blob).decode()+'\n'
        with patch.object(dispatch,'remaining',return_value=1000),patch.object(dispatch,'live_pair',return_value=self.live),patch.object(dispatch,'run_checked',return_value=scanned),patch.object(dispatch,'remote_header',return_value=''),patch.object(dispatch,'remote_json',side_effect=TimeoutError('fixture timeout')) as remote:
            with self.assertRaises(TimeoutError):dispatch.stage(self.deployment,self.run,self.key)
            with self.assertRaises(FileExistsError):dispatch.stage(self.deployment,self.run,self.key)
        self.assertEqual(remote.call_count,1)
        self.assertFalse(load(self.deployment/'stage-failure.json')['retry_allowed'])
        self.assertTrue((self.deployment/'stage-intent.json').is_file())

    def test_stage_success_binds_pin_hardware_and_exact_package(self):
        inventory=self.package();self.active();Path(self.hardware['known_hosts']).unlink()
        blob=b'\x00\x00\x00\x0bssh-ed25519'+b'\x00\x00\x00\x20'+b'x'*32
        scanned='[127.0.0.1]:2222 ssh-ed25519 '+base64.b64encode(blob).decode()+'\n'
        ready={'manifest_sha256':inventory['manifest_sha256'],'archive_sha256':inventory['archive_sha256'],
            'verified_members':len(inventory['files']),'venv_ready':True,'imports_before_pip':True}
        with patch.object(dispatch,'remaining',return_value=1000),patch.object(dispatch,'live_pair',return_value=self.live),patch.object(dispatch,'run_checked',side_effect=[scanned,'']) as calls,patch.object(dispatch,'remote_header',return_value=''),patch.object(dispatch,'remote_json',side_effect=[{'pid1_identity_verified':True},ready]):
            result=dispatch.stage(self.deployment,self.run,self.key)
        self.assertEqual(result['hardware_sha256'],dispatch.sha(self.deployment/'hardware-check.json'))
        scp=calls.call_args_list[1].args[0]
        self.assertEqual(scp[:5],['scp','-i',str(self.key),'-P','2222'])
        self.assertIn('StrictHostKeyChecking=yes',scp)

    def test_host_pin_endpoint_and_keyscan_type_fail_closed(self):
        self.package();self.active()
        self.hardware['host']='-oProxyCommand=oops'
        with self.assertRaises(ValueError):dispatch.ssh_arguments(self.hardware)
        self.hardware['host']='127.0.0.1';Path(self.hardware['known_hosts']).write_text('changed')
        with self.assertRaises(ValueError):dispatch.ssh_arguments(self.hardware)
        Path(self.hardware['known_hosts']).unlink()
        with patch.object(dispatch,'remaining',return_value=1000),patch.object(dispatch,'live_pair',return_value=self.live),patch.object(dispatch,'run_checked',return_value='[127.0.0.1]:2222 ssh-rsa fixture'):
            with self.assertRaises(ValueError):dispatch.stage(self.deployment,self.run,self.key)
        self.assertFalse((self.deployment/'known_hosts').exists())

    def test_launch_origin_precedes_sole_dispatch_and_unknown_never_retried(self):
        self.stage_ready()
        def unknown(*args,**kwargs):
            self.assertTrue((self.deployment/'supervisor-origin.json').exists())
            self.assertTrue((self.deployment/'launch-intent.json').exists())
            raise TimeoutError('unknown remote result')
        with patch.object(dispatch,'remaining',return_value=1000),patch.object(dispatch,'live_pair',return_value=self.live),patch.object(dispatch,'remote_header',return_value=''),patch.object(subprocess,'Popen',side_effect=self.supervisor) as popen,patch.object(dispatch,'remote_json',side_effect=unknown) as remote:
            with self.assertRaises(TimeoutError):dispatch.launch(self.deployment,self.run,self.key)
            with self.assertRaises(FileExistsError):dispatch.launch(self.deployment,self.run,self.key)
        self.assertEqual(remote.call_count,1);self.assertEqual(popen.call_count,1)
        self.assertEqual(load(self.deployment/'launch-failure.json')['outcome'],'launch_attempt_failed_or_unknown')
        self.assertFalse((self.deployment/'remote-launch.json').exists())

    def test_remote_launch_intent_exclusive_before_process_creation(self):
        inventory=self.package();destination=self.root/'remote';destination.mkdir()
        for name in inventory['files']:
            target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((self.root/name).read_bytes())
        remote_run=destination/'runs/study';command=['fixture-python','wrapper']
        with patch.object(dispatch,'REMOTE_ROOT',str(destination)):code=dispatch.launch_code(str(remote_run),inventory,command)
        namespace={'Path':Path,'os':os,'hashlib':hashlib,'json':json,'subprocess':subprocess,
            'datetime':datetime,'timezone':timezone,'write':dispatch.write}
        old=Path.cwd()
        def spawn(*args,**kwargs):
            self.assertTrue((remote_run/'remote-dispatch-intent.json').exists())
            return SimpleNamespace(pid=2222)
        try:
            with patch.object(subprocess,'Popen',side_effect=spawn) as calls,redirect_stdout(io.StringIO()):
                exec(compile(code,'<fixture-launch>','exec'),namespace)
                with self.assertRaises(AssertionError):exec(compile(code,'<repeat-launch>','exec'),namespace)
            self.assertEqual(calls.call_count,1)
        finally:os.chdir(old)

    def test_live_protection_check_is_get_only_and_rejects_stale_or_duplicate(self):
        from scripts import runpod_planning_curriculum as lifecycle
        self.active()
        secret=self.root/'guard.json';dispatch.write(secret,{'token':'fixture-private-token'})
        plan={**self.plan,'guard_source_sha256':'b'*64,'guard_secret_path':str(secret)}
        for row in self.state['roles'].values():
            row.update(state='identified',rate_verified=True,rate_within_plan=True)
        pods=[{'id':'fixture-'+role,'name':f"selfplay-cardinal-{plan['intent']}-{role}",
            'cost':1,'env':{'SELFPLAY_PROJECT':plan['project'],'SELFPLAY_INTENT':plan['intent'],'SELFPLAY_ROLE':role}}
            for role in ('gpu','guard')]
        health={'ready':True,'intent':plan['intent'],'deadline_unix':plan['deadline_unix'],
            'source_sha256':'b'*64,'guard_pod_id':'fixture-guard','last_inventory_unix':time.time()}
        calls=[]
        def call(method,path):
            calls.append((method,path));self.assertEqual(method,'GET')
            return next(p for p in pods if path=='/pods/'+p['id'])
        client=SimpleNamespace(pods=lambda:deepcopy(pods),call=call)
        controller=SimpleNamespace(client=client,check_reservation=lambda:None,rate=lambda *a:{'within_plan':True})
        class Opener:
            def open(self,request,timeout):
                self_outer.assertEqual(request.get_method(),'GET')
                return io.StringIO(json.dumps(health))
        self_outer=self
        with patch.object(lifecycle,'Client',return_value=client),patch.object(lifecycle,'Deployment',return_value=controller),patch.object(dispatch,'remaining',return_value=1000),patch.object(dispatch.urllib.request,'build_opener',return_value=Opener()):
            result=dispatch.live_pair(self.deployment,plan,self.state)
            self.assertNotIn('fixture-private-token',json.dumps(result))
            self.assertEqual(len(calls),2)
            health['last_inventory_unix']=time.time()-61
            with self.assertRaisesRegex(ValueError,'guard readiness'):dispatch.live_pair(self.deployment,plan,self.state)
            health['last_inventory_unix']=time.time();pods.append({**pods[0],'id':'duplicate'})
            with self.assertRaisesRegex(ValueError,'Unexpected project'):dispatch.live_pair(self.deployment,plan,self.state)


def load(path):return json.loads(Path(path).read_text())


if __name__=='__main__':unittest.main()
