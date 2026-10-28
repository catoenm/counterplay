"""Local artifact retrieval and early cleanup for one explicitly launched GPU job.

Never provisions, starts, or retries training. The remote CPU deadline guard and
separate local rental watcher remain responsible if this process fails.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.runpod_cardinal import Deployment, lock, now
from scripts.cloud.train_program_rl import atomic_json, load, digest, process_lock
from selfplay.program_cardinal_gpu import public_deployment

def verify_run(*args, **kwargs):
    from selfplay.program_cardinal_warmstart import verify_run as verify
    return verify(*args, **kwargs)


def pull(args):
    ssh = ['ssh', '-i', str(args.key), '-p', str(args.port),
           '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
           '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
           '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile='+str(args.known_hosts)]
    remote = f'{args.user}@{args.host}:{args.remote_run.rstrip("/")}/'
    args.run.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(['rsync', '-az', '--partial', '--timeout=90', '-e', shlex.join(ssh),
                             remote, str(args.run)+'/'], capture_output=True, text=True, timeout=180)
    return result.returncode


def capture_origin(args):
    """Bind locally once before rsync; resumed cleanup uses this original identity."""
    deployment=public_deployment(load(args.deployment/'plan.json'))
    fixed={'deployment':deployment,'run':str(args.run.resolve()),'remote_run':args.remote_run,
           'host':args.host,'port':args.port,'user':args.user}
    target=args.deployment/'supervisor-origin.json'
    if target.exists():
        saved=load(target)
        if {k:saved.get(k) for k in fixed}!=fixed:
            raise ValueError('Supervisor origin cannot change on restart')
        return saved
    if public_deployment(load(args.run/'deployment.json'))!=deployment:
        raise ValueError('Prepared run belongs to another deployment')
    manifest,status=verify_run(args.run,require_active=False)
    value={'manifest_sha256':digest(args.run/'manifest.json'),**fixed}
    if status['manifest_sha256']!=value['manifest_sha256']: raise ValueError('Prepared manifest differs')
    atomic_json(target,value)
    return value


def verify_receipt(args,origin):
    if public_deployment(load(args.deployment/'plan.json'))!=origin['deployment'] or public_deployment(load(args.run/'deployment.json'))!=origin['deployment']:
        raise ValueError('Downloaded run belongs to another deployment')
    paths=[args.run/n for n in ('remote-exit.json','remote-launch-failure.json') if (args.run/n).exists()]
    if not paths: return None
    if len(paths)!=1: raise ValueError('Conflicting remote receipts')
    receipt=load(paths[0]);start=load(args.run/'remote-job-start.json')
    validated=paths[0].name=='remote-exit.json'
    kind='cardinal_gpu_job_exit' if validated else 'cardinal_gpu_launch_failure'
    if (receipt.get('kind')!=kind or type(receipt.get('exit_code')) is not int or
            receipt.get('verification_completed') is not validated or
            receipt.get('manifest_sha256')!=origin['manifest_sha256'] or
            start.get('manifest_sha256')!=origin['manifest_sha256'] or
            receipt.get('attempt_id')!=start.get('attempt_id') or not start.get('attempt_id') or
            receipt.get('start_sha256')!=digest(args.run/'remote-job-start.json')):
        raise ValueError('Remote receipt/start/expected manifest differs')
    observed=digest(args.run/'manifest.json') if (args.run/'manifest.json').is_file() else None
    if receipt.get('observed_manifest_sha256')!=observed: raise ValueError('Observed launch manifest differs')
    if validated:
        if observed!=origin['manifest_sha256']: raise ValueError('Remote manifest replaced prepared local identity')
        manifest,status=verify_run(args.run,require_active=False)
        if status['manifest_sha256']!=origin['manifest_sha256']: raise ValueError('Exit journal differs')
        return {'remote_exit':receipt,'study_state':status['state'],
                'verified_artifacts':len(status['artifacts']),'scientific_artifacts_verified':True}
    if receipt['exit_code']==0: raise ValueError('Launch failure cannot claim success')
    return {'remote_exit':receipt,'study_state':'launch_failed_before_validation',
            'verified_artifacts':0,'scientific_artifacts_verified':False}


def resume_cleanup(args,origin):
    """A durable verified retrieval survives GPU deletion and transient API errors."""
    proof=load(args.deployment/'supervisor-cleanup-intent.json')
    if proof['origin']!=origin or proof['origin_sha256']!=digest(args.deployment/'supervisor-origin.json'):
        raise ValueError('Cleanup intent belongs to another captured run')
    state={**proof['result'],'updated_at':now(),'cleanup_requested':True,'reservation_retained':True,
           'expected_manifest_sha256':origin['manifest_sha256']}
    atomic_json(args.deployment/'supervisor-status.json',state)
    with lock(args.deployment/'deployment.lock'):
        d=Deployment(args.deployment)
        if public_deployment(d.plan)!=origin['deployment']:
            raise ValueError('Cleanup deployment differs from verified receipt')
        # Receipt is proof the one attempted remote job ended. No later create
        # is permitted. Ambiguous creation elsewhere is never inferred closed.
        d.state.update(creation_closed=True,cleanup_started=True)
        d.save();d.reconcile()
        if not d.gpu_resolved(): d.delete('gpu')
        if not d.gpu_resolved(): raise RuntimeError('GPU deletion not yet confirmed')
        guard=d.state['roles'].get('guard',{})
        if guard.get('state')!='terminated_confirmed': d.delete('guard')
        if d.state['roles'].get('guard',{}).get('state')!='terminated_confirmed':
            raise RuntimeError('Guard deletion not yet confirmed')
    state.update(gpu_404_confirmed=True,guard_404_confirmed=True)
    atomic_json(args.deployment/'supervisor-status.json',state)
    return state


def supervisor_step(args,origin):
    proof=args.deployment/'supervisor-cleanup-intent.json'
    if proof.exists(): return resume_cleanup(args,origin)
    code=pull(args)
    state={'updated_at':now(),'pull_returncode':code,'cleanup_requested':False,'reservation_retained':True,
           'expected_manifest_sha256':origin['manifest_sha256']}
    result=verify_receipt(args,origin) if code==0 else None
    if result is not None:
        state.update(result)
        # Commit retrieval/verification before the first destructive operation.
        atomic_json(proof,{'origin':origin,'origin_sha256':digest(args.deployment/'supervisor-origin.json'),
                           'result':state,'verified_at':now()})
        return resume_cleanup(args,origin)
    atomic_json(args.deployment/'supervisor-status.json',state)
    return None


def supervise(args,origin):
    while True:
        try:
            complete=supervisor_step(args,origin)
            if complete is not None:
                print(json.dumps(complete),flush=True)
                return
        except Exception as error:
            atomic_json(args.deployment/'supervisor-status.json',{'updated_at':now(),'error_type':type(error).__name__,
                'reservation_retained':True,'expected_manifest_sha256':origin['manifest_sha256'],
                'cleanup_intent_persisted':(args.deployment/'supervisor-cleanup-intent.json').exists()})
        time.sleep(60)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deployment',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--remote-run',required=True)
    p.add_argument('--host',required=True)
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--user',default='root')
    p.add_argument('--key',type=Path,required=True)
    p.add_argument('--known-hosts',type=Path,required=True)
    args=p.parse_args()
    with process_lock(args.deployment/'supervisor.lock'):
        origin=capture_origin(args)
        supervise(args,origin)


if __name__=='__main__':main()
