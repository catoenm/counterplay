"""Run one prepared job once; early launch failures retain a separate receipt."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exclusive_json(path, value):
    """Atomic publication without replacing any earlier attempt or receipt."""
    temporary = path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    try:
        with temporary.open('x') as f:
            json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.link(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def stop_child(child):
    if child is None or child.poll() is not None: return 'already_exited'
    try: os.killpg(child.pid,signal.SIGINT)
    except ProcessLookupError: return 'already_exited'
    try:
        child.wait(timeout=30)
        return 'sigint_parent_cleanup_completed'
    except subprocess.TimeoutExpired:
        try: os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError: pass
        child.wait(timeout=10)
        return 'forced_kill_external_guard_still_required'


def execute(args, verifier=None):
    run=Path(args.run).resolve();model=Path(args.model_dir).resolve()
    expected=args.expected_manifest_sha256
    if re.fullmatch('[0-9a-f]{64}',expected) is None: raise ValueError('Expected manifest SHA256 required')
    with (run/'remote-job.lock').open('a+') as lock:
        # An accidental second invocation must not fabricate a failure receipt
        # that would cause the supervisor to delete an already-running job.
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if any((run/n).exists() for n in ('remote-job-start.json','remote-exit.json','remote-launch-failure.json')):
            raise RuntimeError('Job already attempted; no automatic retry or receipt replacement')
        attempt=uuid.uuid4().hex
        start={'kind':'cardinal_gpu_job_attempt','attempt_id':attempt,'pid':os.getpid(),
               'manifest_sha256':expected,'started_at':datetime.now(timezone.utc).isoformat()}
        exclusive_json(run/'remote-job-start.json',start)
        child=None;code=1;stage='launch_validation';error_type=None;validated=False;cleanup=None
        try:
            if digest(run/'manifest.json')!=expected: raise ValueError('Local expected manifest differs')
            if verifier is None:
                from selfplay.program_debug_revision_rl import verify_run
                verifier=verify_run
            manifest,status=verifier(run)
            if status['state']!='prepared' or status['manifest_sha256']!=expected:
                raise ValueError('New prepared run with expected identity required')
            deadline=json.loads((run/'deployment.json').read_text())['deadline_unix']
            validated=True
            commands=[('staging',[sys.executable,'-u','scripts/cloud/bootstrap_revision_rl.py',
                       '--run',str(run),'--model-dir',str(model)]),
                      ('training',[sys.executable,'-u','-m','selfplay.program_debug_revision_rl','run',
                       '--run',str(run),'--model-dir',str(model)])]
            for stage,command in commands:
                remaining=deadline-time.time()-60
                if remaining<=0: raise TimeoutError('Immutable deployment deadline exhausted')
                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PIP_NO_CACHE_DIR':'1'}
                child=subprocess.Popen(command,cwd=ROOT,start_new_session=True,env=env)
                child.wait(timeout=remaining)
                if child.returncode:
                    code=child.returncode
                    raise RuntimeError('Remote phase failed; no automatic retry')
                child=None
            code=0
        except BaseException as error:
            error_type=type(error).__name__
            cleanup=stop_child(child)
        finally:
            name='remote-exit.json' if validated else 'remote-launch-failure.json'
            exclusive_json(run/name,{'kind':'cardinal_gpu_job_exit' if validated else 'cardinal_gpu_launch_failure',
                'manifest_sha256':expected,'attempt_id':attempt,'start_sha256':digest(run/'remote-job-start.json'),
                'verification_completed':validated,'observed_manifest_sha256':digest(run/'manifest.json') if (run/'manifest.json').is_file() else None,
                'exit_code':code,'last_stage':stage,'error_type':error_type,'child_cleanup':cleanup,
                'finished_at':datetime.now(timezone.utc).isoformat()})
        return code


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--expected-manifest-sha256',required=True)
    sys.exit(execute(p.parse_args()))


if __name__=='__main__':main()
