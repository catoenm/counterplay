"""Bounded, local base-model controller generation; no training or API calls.

Run only while older arena/league GPU jobs are idle. Each sample runs in its own
subprocess so a parent can enforce the wall-clock limit even during Metal work.
The shared process lock serializes this screen with the FP32 arena runner.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = ROOT/'experiments/arena-v1/model.json'
LOCK = ROOT/'runs/local-neural.lock'
MAX_SECONDS = 600.0
MAX_OUTPUT_TOKENS = 16384
SAMPLING = {'temperature':0.6,'top_p':0.95,'top_k':20}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    with temporary.open('w') as handle:
        handle.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
        handle.flush();os.fsync(handle.fileno())
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def process_lock(path=LOCK):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another local neural experiment owns '+str(path)) from error
        handle.seek(0);handle.truncate()
        handle.write(json.dumps({'pid':os.getpid(),'started_at':now()})+'\n')
        handle.flush();os.fsync(handle.fileno())
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def format_prompt(tokenizer, prompt, thinking):
    """Keep identical sampling/prompt text across arms; only toggle the template."""
    formatted=tokenizer.apply_chat_template([{'role':'user','content':prompt}],
        tokenize=False,add_generation_prompt=True,enable_thinking=thinking)
    tokens=tokenizer.encode(formatted,add_special_tokens=False)
    if not tokens or len(tokens)>8192:
        raise ValueError('Formatted prompt must contain 1–8192 tokens')
    return formatted,tokens


def extract_final_answer(generated_text, thinking):
    """Never return unclosed reasoning as a candidate controller."""
    opened='<think>' in generated_text
    closed='</think>' in generated_text
    if (thinking or opened) and not closed:
        return {'final_answer':'','reasoning':generated_text,'reasoning_closed':False,
                'extraction_status':'unfinished_reasoning'}
    if closed:
        if not thinking and not opened:
            return {'final_answer':'','reasoning':None,'reasoning_closed':False,
                    'extraction_status':'unexpected_reasoning_delimiter'}
        reasoning,answer=generated_text.split('</think>',1)
        if '<think>' in answer or '</think>' in answer:
            return {'final_answer':'','reasoning':reasoning,'reasoning_closed':False,
                    'extraction_status':'multiple_reasoning_sections'}
        reasoning=reasoning.removeprefix('<think>').strip()
    else:
        answer,reasoning=generated_text,None
    answer=answer.strip()
    return {'final_answer':answer,'reasoning':reasoning,'reasoning_closed':closed,
            'extraction_status':'complete' if answer else 'empty_final_answer'}


def complete_token_rows(path):
    if not Path(path).exists():
        return []
    lines=Path(path).read_bytes().splitlines(keepends=True)
    if lines and not lines[-1].endswith(b'\n'):
        lines.pop()
    return [json.loads(line) for line in lines]


def response_record(manifest,index,rows,context,finish_reason,elapsed_seconds,returncode,
                    error=None,decode=None):
    raw=''.join(row['text'] for row in rows)
    ids=[row['token'] for row in rows]
    final_stream=bool(rows and rows[-1].get('finish_reason') in ('stop','length'))
    # A killed stream may have an unflushed detokenizer segment. Decode saved
    # token IDs with the CPU tokenizer when available; preserve the IDs either way.
    if not final_stream and ids and decode is not None:
        raw=decode(ids)
    extraction=extract_final_answer(raw,manifest['thinking'])
    usable=finish_reason=='stop' and extraction['extraction_status']=='complete'
    count=rows[-1]['generation_tokens'] if rows else 0
    if count!=len(rows) or count>manifest['max_tokens']:
        raise ValueError('Token records violate generation-count/cap invariant')
    prompt_tokens=context.get('prompt_tokens',rows[-1]['prompt_tokens'] if rows else None)
    return {'sample_id':f'generation-{index}','seed':manifest['seeds'][index-1],
        'model':manifest['model']['model'],'revision':manifest['model']['revision'],
        'backend':'mlx','adapter_path':None,'training':False,'thinking':manifest['thinking'],
        'reasoning_mode':'on' if manifest['thinking'] else 'off',
        'prompt':manifest['prompt'],'prompt_sha256':manifest['prompt_sha256'],
        'formatted_prompt':context.get('formatted_prompt'),
        'prompt_token_ids':context.get('prompt_token_ids'),
        'chat_template':context.get('chat_template'),
        'model_snapshot_path':context.get('model_snapshot_path'),
        'generated_text':raw,'generated_token_ids':ids,
        # Only a completed final answer is compatible with the API screen's
        # text field. Partial answers remain available separately for diagnosis.
        'text':extraction['final_answer'] if usable else '',**extraction,
        'usable_final':usable,'finish_reason':finish_reason,
        'thinking_truncated':extraction['extraction_status']=='unfinished_reasoning' and finish_reason!='stop',
        'generation_finished':finish_reason in ('stop','length'),
        'prompt_tokens':prompt_tokens,'completion_tokens':count,
        'usage':{'prompt_tokens':prompt_tokens,'completion_tokens':count,
                 'total_tokens':prompt_tokens+count if prompt_tokens is not None else None},
        'token_count_note':'Includes EOS when stop; counts durably observed stream tokens if interrupted.',
        'stream_finished':final_stream,'elapsed_seconds':elapsed_seconds,'returncode':returncode,
        'error':error,'max_tokens':manifest['max_tokens'],'max_seconds':manifest['max_seconds'],
        'sampling':manifest['sampling'],'client_source_sha256':manifest['source_sha256'],
        'external_spend_usd':0,'completed_at':now()}


def worker(manifest_path,index):
    """The only function that imports MLX or loads model weights."""
    manifest_path=Path(manifest_path);out=manifest_path.parent
    manifest=json.loads(manifest_path.read_text())
    if digest(__file__)!=manifest['source_sha256']:
        raise ValueError('Worker source differs from the frozen screen source')
    if not 1<=index<=manifest['samples']:
        raise ValueError('Unknown sample index')
    prefix=out/f'generation-{index}'
    try:
        from huggingface_hub import snapshot_download
        path=Path(snapshot_download(manifest['model']['model'],revision=manifest['model']['revision'],
                                    local_files_only=True))
        if path.name!=manifest['model']['revision']:
            raise ValueError('Cached model snapshot is not the pinned revision')
        import mlx.core as mx
        from mlx_lm import load,stream_generate
        from mlx_lm.sample_utils import make_sampler
        model,tokenizer=load(str(path),adapter_path=None,trust_remote_code=False)
        model.eval()
        formatted,tokens=format_prompt(tokenizer,manifest['prompt'],manifest['thinking'])
        context={'formatted_prompt':formatted,'prompt_token_ids':tokens,'prompt_tokens':len(tokens),
                 'chat_template':tokenizer.chat_template,'model_snapshot_path':str(path),
                 'eos_token_ids':sorted(tokenizer.eos_token_ids),'ready_at':now()}
        atomic_json(prefix.with_name(prefix.name+'-context.json'),context)
        mx.random.seed(manifest['seeds'][index-1])
        sampler=make_sampler(temp=manifest['sampling']['temperature'],
            top_p=manifest['sampling']['top_p'],top_k=manifest['sampling']['top_k'])
        with prefix.with_name(prefix.name+'-tokens.jsonl').open('x') as handle:
            for result in stream_generate(model,tokenizer,tokens,
                    max_tokens=manifest['max_tokens'],sampler=sampler):
                row={'text':result.text,'token':int(result.token),'prompt_tokens':int(result.prompt_tokens),
                     'generation_tokens':int(result.generation_tokens),'finish_reason':result.finish_reason}
                handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush()
                if result.generation_tokens%32==0 or result.finish_reason:
                    os.fsync(handle.fileno())
        return 0
    except Exception as error:
        atomic_json(prefix.with_name(prefix.name+'-error.json'),
                    {'type':type(error).__name__,'message':str(error),'at':now()})
        raise


def stop_worker(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:
            pass  # The worker exited between poll() and killpg().
    process.wait()


def run_sample(out,manifest,index,launch=subprocess.Popen):
    out=Path(out);prefix=out/f'generation-{index}'
    started=time.monotonic();timed_out=False;interrupted=False
    env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'}
    with prefix.with_name(prefix.name+'-worker.log').open('x') as log:
        process=launch([sys.executable,str(out/'source/program_local.py'),'--worker',
                        str(out/'manifest.json'),'--sample-index',str(index)],
                       stdout=log,stderr=log,env=env,start_new_session=True)
        try:
            process.wait(timeout=manifest['max_seconds'])
        except subprocess.TimeoutExpired:
            timed_out=True;stop_worker(process)
        except BaseException:
            interrupted=True;stop_worker(process)
    context_path=prefix.with_name(prefix.name+'-context.json')
    context=json.loads(context_path.read_text()) if context_path.exists() else {}
    rows=complete_token_rows(prefix.with_name(prefix.name+'-tokens.jsonl'))
    finish='time' if timed_out else 'interrupted' if interrupted else (
        rows[-1].get('finish_reason') if rows and process.returncode==0 else 'error')
    if finish not in ('time','interrupted','stop','length'):
        finish='error'
    error_path=prefix.with_name(prefix.name+'-error.json')
    error=json.loads(error_path.read_text()) if error_path.exists() else None
    decode=None
    if rows and not rows[-1].get('finish_reason') and context.get('model_snapshot_path'):
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(context['model_snapshot_path'],
                                                local_files_only=True,trust_remote_code=False)
        eos=set(context['eos_token_ids'])
        decode=lambda ids:tokenizer.decode([token for token in ids if token not in eos],
                                           skip_special_tokens=False)
    response=response_record(manifest,index,rows,context,finish,time.monotonic()-started,
                             process.returncode,error,decode)
    atomic_json(prefix.with_suffix('.json'),response)
    return response


def prepare(prompt_path,out,thinking,samples=2,max_tokens=4096,max_seconds=MAX_SECONDS,
            seed=202610220,model_config=MODEL_CONFIG):
    prompt_path,out,model_config=Path(prompt_path),Path(out).resolve(),Path(model_config)
    if out.exists():
        raise ValueError('Refusing to overwrite an existing local screen directory')
    if thinking not in ('on','off') or not 1<=samples<=2 or not 1<=max_tokens<=MAX_OUTPUT_TOKENS:
        raise ValueError('Use thinking on/off, 1–2 samples, and 1–16384 output tokens')
    if not math.isfinite(max_seconds) or not 0<max_seconds<=MAX_SECONDS:
        raise ValueError('Each sample must have a wall-clock limit of at most 600 seconds')
    if not 0<=seed<=2**32-samples:
        raise ValueError('Sampling seeds must fit unsigned 32-bit integers')
    raw_prompt=prompt_path.read_bytes();prompt=raw_prompt.decode('utf-8')
    if not prompt.strip() or len(raw_prompt)>40000:
        raise ValueError('Prompt must contain nonempty UTF-8 text of at most 40 KB')
    model=json.loads(model_config.read_text())
    if set(model)!={'model','revision'} or not re.fullmatch(r'[0-9a-f]{40}',model['revision']):
        raise ValueError('Model configuration must pin an exact commit revision')
    manifest={'purpose':'local generated-controller usability diagnostic',
        'training':False,'adapter_path':None,'model':model,'model_config_sha256':digest(model_config),
        'prompt':prompt,'prompt_sha256':hashlib.sha256(raw_prompt).hexdigest(),
        'prompt_path':str(prompt_path.resolve()),'thinking':thinking=='on','samples':samples,
        'max_tokens':max_tokens,'max_seconds':max_seconds,'seeds':[seed+i for i in range(samples)],
        'sampling':dict(SAMPLING),'source_sha256':digest(__file__),
        'versions':{name:importlib.metadata.version(name) for name in ('mlx','mlx-lm','transformers','huggingface-hub')},
        'timing':'Hard subprocess wall limit per sample, including model load and prompt prefill.',
        'token_usage':'Output token cap includes reasoning and final answer together.',
        'gpu_concurrency':'Shared local-neural.lock serializes this screen with arena-v2; older runners must be idle.',
        'external_spend_usd':0,'created_at':now()}
    out.mkdir(parents=True,exist_ok=False);(out/'source').mkdir()
    shutil.copyfile(__file__,out/'source/program_local.py')
    (out/'prompt.txt').write_bytes(raw_prompt);shutil.copyfile(model_config,out/'model.json')
    atomic_json(out/'manifest.json',manifest)
    atomic_json(out/'status.json',{'status':'prepared','completed_samples':0,'external_spend_usd':0})
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--thinking',choices=('on','off'),default='on')
    parser.add_argument('--samples',type=int,default=2)
    parser.add_argument('--max-tokens',type=int,default=4096)
    parser.add_argument('--max-seconds',type=float,default=MAX_SECONDS)
    parser.add_argument('--seed',type=int,default=202610220)
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--sample-index',type=int,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:
        return worker(args.worker,args.sample_index)
    if not args.prompt or not args.out:
        parser.error('--prompt and --out are required')
    print('Local GPU screen: arena/league jobs must be idle. This command does not train.',file=sys.stderr,flush=True)
    with process_lock():
        manifest=prepare(args.prompt,args.out,args.thinking,args.samples,args.max_tokens,args.max_seconds,args.seed)
        for index in range(1,args.samples+1):
            atomic_json(args.out/'status.json',{'status':'generating','sample':index,
                        'completed_samples':index-1,'external_spend_usd':0})
            response=run_sample(args.out.resolve(),manifest,index)
            print(json.dumps({'sample':index,'finish_reason':response['finish_reason'],
                'extraction_status':response['extraction_status'],'completion_tokens':response['completion_tokens'],
                'usable_final':response['usable_final'],'out':str((args.out/f'generation-{index}.json').resolve())}),flush=True)
            state='failed' if response['finish_reason']=='error' else 'interrupted' if response['finish_reason']=='interrupted' else 'running'
            atomic_json(args.out/'status.json',{'status':state,'completed_samples':index,
                        'external_spend_usd':0,'last_finish_reason':response['finish_reason']})
            if state in ('failed','interrupted'):
                return 1
        atomic_json(args.out/'status.json',{'status':'complete','completed_samples':args.samples,'external_spend_usd':0})
    return 0


if __name__=='__main__':
    raise SystemExit(main())
