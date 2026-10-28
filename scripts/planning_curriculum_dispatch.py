"""Explicit package, stage and once-only launch; importing performs no actions.

Run only after the separately reviewed scientific preparation and provider
allocation. This module never creates/deletes pods, releases funds, retries an
unknown dispatch, or changes a deployment deadline. A failed attempt remains
durable for human inspection.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = '/workspace/selfplay'
REMOTE_ARCHIVE = '/workspace/planning-curriculum-package.tar.gz'
REMOTE_VENV = '/workspace/planning-curriculum-venv'
REMOTE_STAGE_INTENT = '/workspace/planning-curriculum-stage-intent.json'
WRAPPER = 'scripts/cloud/run_planning_curriculum_job.py'
SUPERVISOR = 'scripts/planning_curriculum_supervisor.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def relative(path):
    path = Path(path).absolute()
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
        raise ValueError('Only ordinary workspace paths are permitted')
    value = path.resolve().relative_to(ROOT).as_posix()
    if not value.startswith('runs/') or not re.fullmatch(r'[A-Za-z0-9_./-]+', value):
        raise ValueError('Run/deployment must be a safe runs/ path')
    return value


def safe_member(name):
    p = PurePosixPath(name)
    if (not isinstance(name, str) or not name or p.is_absolute() or '..' in p.parts
            or p.as_posix() != name or any(x in ('.secrets', '.git') for x in p.parts)
            or '\\' in name):
        raise ValueError('Unsafe package member')
    return p


def regular(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(ROOT):
        raise ValueError('Package members must be regular workspace files')
    # Disallow symlinked parent directories as well.
    if any(p.is_symlink() for p in (path, *path.parents) if p != ROOT.parent):
        raise ValueError('Symlinked package path')


def prepared(deployment, run):
    from selfplay.planning_curriculum.experiment import verify_run, public_deployment
    from scripts.runpod_planning_curriculum import validate
    deployment, run = Path(deployment).resolve(), Path(run).resolve()
    relative(deployment); relative(run)
    plan, state = load(deployment/'plan.json'), load(deployment/'state.json')
    validate(plan)
    if state['plan_sha256'] != sha(deployment/'plan.json'):
        raise ValueError('Deployment plan binding differs')
    manifest, status = verify_run(run, require_active=True)
    if (status['state'] != 'prepared' or status['artifacts']
            or manifest['deployment'] != public_deployment(plan)):
        raise ValueError('Exact new prepared run/deployment required')
    if manifest['sources'].get('scripts/planning_curriculum_dispatch.py') != sha(__file__):
        raise ValueError('Dispatch source must be included in the frozen manifest')
    return plan, state, manifest


def expected_files(run, manifest):
    prefix = relative(run)
    files = dict(manifest['sources'])
    files.update({prefix+'/source/'+name: digest for name, digest in manifest['sources'].items()})
    files.update({prefix+'/'+name: digest for name, digest in manifest['inputs'].items()})
    for name in ('manifest.json', 'status.json', 'deployment.json'):
        files[prefix+'/'+name] = sha(Path(run)/name)
    for name, digest in files.items():
        safe_member(name); regular(ROOT/name)
        if sha(ROOT/name) != digest:
            raise ValueError('Package input changed: '+name)
    return dict(sorted(files.items()))


def check_archive(archive, inventory):
    if (sha(archive) != inventory['archive_sha256']
            or Path(archive).stat().st_size != inventory['archive_bytes']):
        raise ValueError('Package archive bytes differ')
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        if len(members) != len(inventory['files']) or len({m.name for m in members}) != len(members):
            raise ValueError('Duplicate or missing package member')
        for member in members:
            safe_member(member.name)
            if not member.isfile() or member.name not in inventory['files']:
                raise ValueError('Nonregular/unlisted package member')
            if hashlib.sha256(tar.extractfile(member).read()).hexdigest() != inventory['files'][member.name]:
                raise ValueError('Package member hash differs')


def package(deployment, run):
    deployment, run = Path(deployment).resolve(), Path(run).resolve()
    plan, state, manifest = prepared(deployment, run)
    if state.get('roles'):
        raise ValueError('Package must be fixed before provider creation')
    files = expected_files(run, manifest)
    write(deployment/'package-intent.json', {'at':now(), 'intent':plan['intent'],
        'manifest_sha256':sha(run/'manifest.json'), 'files':files, 'one_attempt_only':True})
    archive = deployment/'source-package.tar.gz'
    with archive.open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w') as tar:
                for name, digest in files.items():
                    data = (ROOT/name).read_bytes()
                    if hashlib.sha256(data).hexdigest() != digest:
                        raise ValueError('Input changed during package creation')
                    info = tarfile.TarInfo(name); info.size=len(data); info.mode=0o644
                    tar.addfile(info, io.BytesIO(data))
        raw.flush(); os.fsync(raw.fileno())
    inventory = {'kind':'planning_curriculum_package_v1', 'intent':plan['intent'],
        'run':relative(run), 'manifest_sha256':sha(run/'manifest.json'),
        'plan_sha256':sha(deployment/'plan.json'), 'files':files,
        'archive_sha256':sha(archive), 'archive_bytes':archive.stat().st_size}
    check_archive(archive, inventory)
    if files != expected_files(run, manifest):
        raise ValueError('Inputs changed before package commit')
    write(deployment/'package-inventory.json', inventory)
    return inventory


def inventory_for(deployment, run, manifest):
    inventory = load(deployment/'package-inventory.json')
    if (inventory['files'] != expected_files(run, manifest)
            or inventory['manifest_sha256'] != sha(run/'manifest.json')
            or inventory['plan_sha256'] != sha(deployment/'plan.json')
            or inventory['run'] != relative(run)):
        raise ValueError('Current prepared package identity differs')
    check_archive(deployment/'source-package.tar.gz', inventory)
    return inventory


def remaining(plan):
    from scripts.cloud.bootstrap_planning_curriculum import stage_deadline
    value = stage_deadline(plan)-time.time()
    if value <= 0:
        raise TimeoutError('Original admission/deployment clock exhausted')
    return value


def live_pair(deployment, plan, state):
    """Fresh GET-only protection check. Never calls reconcile/delete/create."""
    from scripts.runpod_planning_curriculum import Deployment, Client, matches, identity, NoRedirect
    d = Deployment(deployment, Client())
    d.check_reservation()
    if not state.get('reservation_applied') or state.get('cleanup_started') or state.get('creation_closed'):
        raise ValueError('Funded active deployment required')
    expected = {}
    for role in ('gpu', 'guard'):
        row = state['roles'].get(role, {})
        if (row.get('state') != 'identified' or row.get('duplicate_intent_anomaly')
                or not row.get('rate_verified') or not row.get('rate_within_plan')):
            raise ValueError('Unique identified and rate-verified pair required')
        expected[role] = row['pod_id']
    pods = d.client.pods()
    project = [p for p in pods if p.get('env', {}).get('SELFPLAY_PROJECT') == plan['project']]
    if {p.get('id') for p in project} != set(expected.values()) or len(project) != 2:
        raise ValueError('Unexpected project resource or duplicate pair')
    public = {}
    for role, identifier in expected.items():
        pod = d.client.call('GET', '/pods/'+identifier)
        if pod.get('id') != identifier or not matches(pod, plan['intent'], role) or not d.rate(pod, role)['within_plan']:
            raise ValueError('Fresh provider identity/rate differs')
        public[role] = identity(pod)
    request = urllib.request.Request('https://'+expected['guard']+'-8080.proxy.runpod.net/health',
        headers={'X-Guard-Token':load(plan['guard_secret_path'])['token'], 'User-Agent':'Mozilla/5.0'})
    with urllib.request.build_opener(NoRedirect).open(request, timeout=min(20,remaining(plan))) as response:
        health = json.load(response)
    if (health.get('ready') is not True or health.get('intent') != plan['intent']
            or health.get('deadline_unix') != plan['deadline_unix']
            or health.get('source_sha256') != plan['guard_source_sha256']
            or health.get('guard_pod_id') != expected['guard']
            or any(health.get(k) for k in ('finish_requested','self_delete_requested','creation_closed','duplicate_intent_anomaly'))
            or type(health.get('last_inventory_unix')) not in (int,float)
            or not 0 <= time.time()-health['last_inventory_unix'] < 60):
        raise ValueError('Fresh exact independent guard readiness failed')
    return {'at':now(), 'intent':plan['intent'], 'deadline_unix':plan['deadline_unix'],
        'resources':public, 'guard':{k:health.get(k) for k in ('ready','intent','deadline_unix',
            'source_sha256','guard_pod_id','last_inventory_unix','duplicate_intent_anomaly')}}


def ssh_arguments(hardware):
    host, port = hardware['host'], hardware['port']
    if (not isinstance(host,str) or not re.fullmatch(r'[A-Za-z0-9.-]+',host)
            or host.startswith('-') or type(port) is not int or not 0 < port < 65536
            or hardware['user'] != 'root'):
        raise ValueError('Unsafe direct SSH endpoint')
    if sha(hardware['known_hosts']) != hardware['known_hosts_sha256']:
        raise ValueError('Pinned host key changed')
    return ['ssh','-i',hardware['key'],'-p',str(port),'-o','BatchMode=yes',
        '-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',
        '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+hardware['known_hosts'], 'root@'+host]


def run_checked(command, timeout):
    response = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if response.returncode:
        # Do not echo remote stderr: it may contain sensitive environment data.
        raise RuntimeError('Subprocess failed with return code '+str(response.returncode))
    return response.stdout


def remote_json(hardware, code, plan, maximum=60):
    output = run_checked(ssh_arguments(hardware)+[shlex.join(['python3','-B','-c',code])],
        min(maximum,remaining(plan)))
    return json.loads(output)


def remote_header(plan, identifier):
    from scripts.cloud.bootstrap_planning_curriculum import stage_deadline
    expected = {'SELFPLAY_PROJECT':plan['project'], 'SELFPLAY_INTENT':plan['intent'],
        'SELFPLAY_ROLE':'gpu', 'RUNPOD_POD_ID':identifier}
    return """from pathlib import Path
import os,json,hashlib,subprocess,sys,time
from datetime import datetime,timezone
env=dict(x.split('=',1) for x in Path('/proc/1/environ').read_bytes().decode().split('\\0') if '=' in x)
assert all(env.get(k)==v for k,v in EXPECTED.items()), 'PID1 identity differs'
assert time.time()<DEADLINE, 'Admission deadline exhausted'
def write(path,value):
 with Path(path).open('x') as f:
  json.dump(value,f,sort_keys=True);f.write('\\n');f.flush();os.fsync(f.fileno())
""".replace('EXPECTED',repr(expected)).replace('DEADLINE',repr(stage_deadline(plan)))


def extraction_code(inventory):
    # Validate the entire archive before creating any output file.
    return """import tarfile
inv=INVENTORY
archive=Path(ARCHIVE);root=Path(ROOT);venv=Path(VENV)
assert not root.exists() and not venv.exists()
assert hashlib.sha256(archive.read_bytes()).hexdigest()==inv['archive_sha256']
assert archive.stat().st_size==inv['archive_bytes']
with tarfile.open(archive,'r:gz') as tar:
 members=tar.getmembers();assert len(members)==len(inv['files']) and len({m.name for m in members})==len(members)
 for m in members:
  p=Path(m.name)
  assert m.isfile() and not p.is_absolute() and '..' not in p.parts and p.as_posix()==m.name and m.name in inv['files']
  assert hashlib.sha256(tar.extractfile(m).read()).hexdigest()==inv['files'][m.name]
 for m in members:
  target=root/m.name;target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as f:f.write(tar.extractfile(m).read())
subprocess.run(['python3','-B','-m','venv','--system-site-packages',str(venv)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
imports='import selfplay.planning_curriculum.experiment; import scripts.cloud.bootstrap_planning_curriculum; import scripts.cloud.run_planning_curriculum_job'
subprocess.run([str(venv/'bin/python'),'-B','-c',imports],cwd=root,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
print(json.dumps({'verified_members':len(members),'manifest_sha256':inv['manifest_sha256'],'archive_sha256':inv['archive_sha256'],'imports_before_pip':True,'venv_ready':True,'new_model_executions':0}))
""".replace('INVENTORY',repr(inventory)).replace('ARCHIVE',repr(REMOTE_ARCHIVE)).replace('ROOT',repr(REMOTE_ROOT)).replace('VENV',repr(REMOTE_VENV))


def stage(deployment, run, key):
    deployment, run, key = Path(deployment).resolve(), Path(run).resolve(), Path(key).resolve()
    plan, state, manifest = prepared(deployment, run)
    inventory = inventory_for(deployment,run,manifest); remaining(plan)
    if not key.is_file(): raise ValueError('Existing private SSH key required')
    write(deployment/'stage-intent.json', {'at':now(),'intent':plan['intent'],
        'manifest_sha256':inventory['manifest_sha256'],'archive_sha256':inventory['archive_sha256'],'one_attempt_only':True})
    try:
        check=live_pair(deployment,plan,state); write(deployment/'pre-stage-check.json',check)
        direct=check['resources']['gpu']['ssh']['direct']
        host,port,user=direct['host'],direct['port'],direct['username']
        known=deployment/'known_hosts'
        provisional={'host':host,'port':port,'user':user,'key':str(key),'known_hosts':str(known)}
        # Validate the endpoint before passing it to ssh-keyscan.
        if not isinstance(host,str) or not re.fullmatch(r'[A-Za-z0-9.-]+',host) or host.startswith('-') or type(port) is not int or not 0<port<65536 or user!='root':
            raise ValueError('Unsafe direct SSH endpoint')
        scan=run_checked(['ssh-keyscan','-T','15','-p',str(port),'-t','ed25519',host],min(25,remaining(plan)))
        lines=[line for line in scan.splitlines() if line and not line.startswith('#')]
        fields=lines[0].split() if len(lines)==1 else []
        expected_host=host if port==22 else '['+host+']:'+str(port)
        if len(fields)!=3 or fields[:2]!=[expected_host,'ssh-ed25519']:
            raise ValueError('Expected exactly one endpoint-bound Ed25519 host key')
        decoded=base64.b64decode(fields[2],validate=True)
        if len(decoded)!=51 or decoded[:15]!=b'\x00\x00\x00\x0bssh-ed25519':
            raise ValueError('Malformed Ed25519 host key')
        with known.open('x') as stream:stream.write(lines[0]+'\n');stream.flush();os.fsync(stream.fileno())
        hardware={**provisional,'known_hosts_sha256':sha(known),'intent':plan['intent'],
            'pod_id':check['resources']['gpu']['id'],'project':plan['project'],'role':'gpu'}
        code=remote_header(plan,hardware['pod_id'])+"""import shutil,torch
assert not Path(STAGE).exists() and not Path(ARCHIVE).exists() and not Path(ROOT).exists() and not Path(VENV).exists()
write(STAGE,{'at':datetime.now(timezone.utc).isoformat(),'manifest_sha256':MANIFEST,'one_attempt_only':True})
assert torch.__version__.split('+')[0]=='2.8.0' and torch.version.cuda=='12.8' and torch.cuda.device_count()==1
assert 'H200' in torch.cuda.get_device_name(0)
mem=torch.cuda.get_device_properties(0).total_memory//(1024**2);assert mem>=130000
disk=shutil.disk_usage('/workspace');assert disk.free>50*1024**3
print(json.dumps({'pid1_identity_verified':True,'gpu':torch.cuda.get_device_name(0),'gpu_count':1,'gpu_memory_mib':mem,'torch':torch.__version__,'cuda':torch.version.cuda,'disk_free_bytes':disk.free,'disk_total_bytes':disk.total}))
""".replace('STAGE',repr(REMOTE_STAGE_INTENT)).replace('ARCHIVE',repr(REMOTE_ARCHIVE)).replace('ROOT',repr(REMOTE_ROOT)).replace('VENV',repr(REMOTE_VENV)).replace('MANIFEST',repr(inventory['manifest_sha256']))
        hardware.update(remote_json(hardware,code,plan,45));hardware['at']=now()
        write(deployment/'hardware-check.json',hardware)
        ssh=ssh_arguments(hardware)
        scp=['scp','-i',str(key),'-P',str(port),*ssh[5:-1],str(deployment/'source-package.tar.gz'),'root@'+host+':'+REMOTE_ARCHIVE]
        run_checked(scp,min(600,remaining(plan)))
        ready=remote_json(hardware,remote_header(plan,hardware['pod_id'])+extraction_code(inventory),plan,120)
        if ready.get('manifest_sha256')!=inventory['manifest_sha256'] or ready.get('verified_members')!=len(inventory['files']):
            raise ValueError('Remote extraction identity differs')
        ready.update(at=now(),intent=plan['intent'],pod_id=hardware['pod_id'],hardware_sha256=sha(deployment/'hardware-check.json'))
        write(deployment/'remote-package-readiness.json',ready)
        return ready
    except BaseException as error:
        write(deployment/'stage-failure.json',{'at':now(),'error_type':type(error).__name__,
            'outcome':'stage_attempt_failed_or_unknown','retry_allowed':False})
        raise


def launch_code(run_path, inventory, command):
    return """run=Path(RUN);inv=INVENTORY;command=COMMAND
os.chdir(ROOT)
for name,digest in inv['files'].items():
 p=Path(ROOT)/name;assert p.is_file() and not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==digest
assert not any((run/n).exists() for n in ['remote-dispatch-intent.json','remote-dispatch.json','remote-job-start.json','remote-exit.json','remote-launch-failure.json'])
write(run/'remote-dispatch-intent.json',{'at':datetime.now(timezone.utc).isoformat(),'manifest_sha256':inv['manifest_sha256'],'command':command,'one_attempt_only':True})
with (run/'remote.log').open('ab',buffering=0) as log:
 p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PIP_NO_CACHE_DIR':'1'})
value={'at':datetime.now(timezone.utc).isoformat(),'pid':p.pid,'command':command,'manifest_sha256':inv['manifest_sha256']}
write(run/'remote-dispatch.json',value);print(json.dumps(value),flush=True)
""".replace('RUN',repr(run_path)).replace('INVENTORY',repr(inventory)).replace('COMMAND',repr(command)).replace('ROOT',repr(REMOTE_ROOT))


def launch(deployment, run, supervisor_python):
    deployment,run=Path(deployment).resolve(),Path(run).resolve()
    plan,state,manifest=prepared(deployment,run);inventory=inventory_for(deployment,run,manifest)
    hardware,ready=load(deployment/'hardware-check.json'),load(deployment/'remote-package-readiness.json')
    if (hardware['intent']!=plan['intent'] or hardware['pod_id']!=state['roles']['gpu']['pod_id']
            or ready.get('hardware_sha256')!=sha(deployment/'hardware-check.json')
            or ready.get('archive_sha256')!=inventory['archive_sha256']
            or ready.get('manifest_sha256')!=inventory['manifest_sha256']
            or ready.get('verified_members')!=len(inventory['files']) or ready.get('venv_ready') is not True):
        raise ValueError('Staging evidence differs')
    ssh_arguments(hardware);remaining(plan)
    if not Path(supervisor_python).is_file():raise ValueError('Existing local supervisor Python required')
    write(deployment/'launch-intent.json',{'at':now(),'manifest_sha256':inventory['manifest_sha256'],
        'intent':plan['intent'],'one_attempt_only':True})
    try:
        check=live_pair(deployment,plan,state);write(deployment/'pre-launch-check.json',check)
        direct=check['resources']['gpu']['ssh']['direct']
        if (direct['host'],direct['port'],direct['username'])!=(hardware['host'],hardware['port'],hardware['user']):
            raise ValueError('Pinned direct SSH endpoint changed')
        remote_run=REMOTE_ROOT+'/'+relative(run)
        command=[str(Path(supervisor_python).resolve()),'-B','-u',SUPERVISOR,
            '--deployment',str(deployment),'--run',str(run),'--remote-run',remote_run,
            '--host',hardware['host'],'--port',str(hardware['port']),'--user',hardware['user'],
            '--key',hardware['key'],'--known-hosts',hardware['known_hosts']]
        write(deployment/'supervisor-intent.json',{'at':now(),'command':command,'one_attempt_only':True})
        with (deployment/'supervisor.log').open('ab',buffering=0) as log:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,close_fds=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
        write(deployment/'supervisor-launch.json',{'at':now(),'pid':child.pid,'command':command})
        wait_until=time.monotonic()+min(30,remaining(plan))
        while not (deployment/'supervisor-origin.json').exists():
            if child.poll() is not None or time.monotonic()>=wait_until:
                raise RuntimeError('Supervisor origin unavailable; wrapper not dispatched')
            time.sleep(.25)
        origin=load(deployment/'supervisor-origin.json')
        if (origin['manifest_sha256']!=inventory['manifest_sha256'] or origin['run']!=str(run)
                or origin['remote_run']!=remote_run or origin['deployment']!=manifest['deployment']
                or (origin['host'],origin['port'],origin['user'])!=(hardware['host'],hardware['port'],hardware['user'])
                or child.poll() is not None):
            raise ValueError('Supervisor did not bind the exact original run before launch')
        wrapper=[REMOTE_VENV+'/bin/python','-B','-u',WRAPPER,'--run',remote_run,
            '--model-dir','/workspace/models/qwen3-8b','--expected-manifest-sha256',inventory['manifest_sha256']]
        result=remote_json(hardware,remote_header(plan,hardware['pod_id'])+launch_code(remote_run,inventory,wrapper),plan,60)
        if (result.get('manifest_sha256')!=inventory['manifest_sha256'] or result.get('command')!=wrapper
                or type(result.get('pid')) is not int or result['pid']<=1):
            raise ValueError('Unknown remote dispatch response; do not retry')
        write(deployment/'remote-launch.json',result)
        return {'supervisor_pid':child.pid,'remote_wrapper':result}
    except BaseException as error:
        write(deployment/'launch-failure.json',{'at':now(),'error_type':type(error).__name__,
            'outcome':'launch_attempt_failed_or_unknown','retry_allowed':False,
            'supervisor_may_be_running':(deployment/'supervisor-launch.json').exists()})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('package','stage','launch'))
    parser.add_argument('--deployment',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--key',type=Path)
    parser.add_argument('--supervisor-python',type=Path)
    args=parser.parse_args()
    if args.command=='package':result=package(args.deployment,args.run)
    elif args.command=='stage':
        if args.key is None:parser.error('stage requires --key')
        result=stage(args.deployment,args.run,args.key)
    else:
        if args.supervisor_python is None:parser.error('launch requires --supervisor-python')
        result=launch(args.deployment,args.run,args.supervisor_python)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':
    sys.path.insert(0,str(ROOT));main()
