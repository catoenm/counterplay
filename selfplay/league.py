"""Bounded, resumable local self-play continuation with an evaluation league.

The behavior policy plays both seats during training. Historical checkpoints
are evaluation opponents only. Archived arena code supplies unchanged games,
rollouts, and the actor-only RL update. No paid service is called here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
import fcntl
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import uuid
from zoneinfo import ZoneInfo

import numpy as np


DEFAULT_DEADLINE = '2026-10-22T08:00:00-07:00'
SOURCE_KEY = 'selfplay/league.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(value):
    return json.dumps(json.loads(json.dumps(value)), sort_keys=True, separators=(',', ':'), allow_nan=False)


def same(actual, expected, message):
    if normalized(actual) != normalized(expected):
        raise ValueError(message)


def timestamp():
    return datetime.now(ZoneInfo('America/Los_Angeles')).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    with temporary.open('w') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False)+'\n')
        handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_rows(path, rows):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    with temporary.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False)+'\n')
        handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)


@contextmanager
def process_lock(out):
    """An OS lock, not a possibly stale PID file, owns the continuation."""
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    path = out.parent/(out.name+'.lock')
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another process owns {path}; no duplicate job started') from error
        handle.seek(0); handle.truncate()
        handle.write(json.dumps({'pid':os.getpid(),'acquired_at':timestamp(),'run':str(out)})+'\n')
        handle.flush(); os.fsync(handle.fileno())
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def evaluation_roster(iteration, available, maximum=8):
    """At most maximum identities INCLUDING candidate, heuristic, and base."""
    if maximum < 5:
        raise ValueError('Roster must fit candidate, base, heuristic, and two predecessors')
    candidate = f'cycle{iteration}'
    available = set(available)-{candidate}
    recent = [f'cycle{i}' for i in (iteration-1, iteration-2) if f'cycle{i}' in available]
    opponents = ['heuristic','base']+recent
    older = [n for n in available if n.startswith('cycle') and n not in recent]
    def milestone_key(name):
        number = int(name[5:])
        milestone = number == 1 or number & (number-1) == 0
        reservoir = hashlib.sha256(('league-v1-history:'+name).encode()).hexdigest()
        return (not milestone, reservoir, number)
    for name in sorted(older,key=milestone_key):
        if len(opponents)+1 >= maximum:
            break
        opponents.append(name)
    if len(set(opponents)) != len(opponents) or len(opponents)+1 > maximum:
        raise ValueError('Invalid bounded evaluation roster')
    return opponents


def make_schedule(environments, cycles=(3,4), maximum=8):
    schedule, rosters = {}, {}
    available = ['base','cycle1','cycle2']
    for cycle in cycles:
        candidate = f'cycle{cycle}'
        source = f'cycle{cycle-1}'
        train = []
        for index, game in enumerate(environments):
            for i in range(8):
                train.append({'environment':game,'seed':1800000+cycle*10000+index*1000+i*2,
                    'regime':'train','policies':[source,source],
                    'episode_id':f'league-train-{cycle}-{game}-{i}'})
        schedule[f'train-rollouts-{cycle}'] = train
        rosters[candidate] = evaluation_roster(cycle,available,maximum)
        for opponent in rosters[candidate]:
            specs = []
            for index, game in enumerate(environments):
                for i in range(3):
                    for seat in (0,1):
                        names = [opponent,opponent]; names[seat] = candidate
                        specs.append({'environment':game,'seed':1900000+index*1000+i*2,
                            'regime':'eval','policies':names,'candidate':candidate,
                            'candidate_seat':seat,'opponent':opponent,'iteration':cycle,
                            'episode_id':f'league-eval-{candidate}-{opponent}-{game}-{i}-{seat}'})
            schedule[f'eval-{candidate}-{opponent}'] = specs
        available.append(candidate)
    return schedule,rosters


def checkpoint_record(directory, name, model, require_training=True):
    directory = Path(directory).resolve()
    metadata = json.loads((directory/'metadata.json').read_text())
    same(metadata['checkpoint'],name,f'Checkpoint name mismatch at {directory}')
    same({k:metadata[k] for k in ('model','revision')},model,f'Checkpoint model mismatch at {directory}')
    actual = sha(directory/'adapters.safetensors')
    same(actual,metadata['adapter_sha256'],f'Checkpoint tensor hash mismatch at {directory}')
    record = {'name':name,'path':str(directory),'adapter_sha256':actual,
              'metadata_sha256':sha(directory/'metadata.json')}
    if name != 'base' and require_training:
        summary = json.loads((directory/'summary.json').read_text())
        if summary.get('complete') is not True:
            raise ValueError(f'Checkpoint training is not complete: {directory}')
        number = int(name[5:])
        same(summary['target'],name,f'Training target mismatch: {directory}')
        same(summary['source'],'base' if number==1 else f'cycle{number-1}',f'Training source mismatch: {directory}')
        same(summary['checkpoint']['adapter_sha256'],actual,f'Training summary hash mismatch: {directory}')
        if not math.isfinite(summary['parameter_delta_l2']) or summary['parameter_delta_l2'] <= 0:
            raise ValueError(f'Checkpoint has no finite parameter update: {directory}')
        record['training_summary_sha256'] = sha(directory/'summary.json')
    return record,metadata


def has_complete_summary(directory):
    path=Path(directory)/'summary.json'
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get('complete') is True
    except json.JSONDecodeError:
        # The old trainer writes this final marker last, without an atomic
        # rename. Preserve an interrupted write as an incomplete update.
        return False


def validate_adapter_tensors(weights, expected):
    """Reject partial/malformed adapters before a non-strict base-model load."""
    actual_keys,expected_keys=set(weights),set(expected)
    if actual_keys != expected_keys:
        raise ValueError('Adapter tensor names differ from trainable parameters: '
                         f'missing={sorted(expected_keys-actual_keys)}, '
                         f'unexpected={sorted(actual_keys-expected_keys)}')
    snapshot={}
    for key,value in weights.items():
        array=np.array(value)
        if tuple(array.shape) != tuple(expected[key].shape):
            raise ValueError(f'Adapter tensor shape mismatch: {key}')
        if not np.isfinite(array).all():
            raise ValueError(f'Adapter tensor contains non-finite values: {key}')
        snapshot[key]=array.copy()
    return snapshot


def verify_loaded_adapter_tensors(saved, actual):
    loaded=validate_adapter_tensors(actual,saved)
    for key,array in saved.items():
        if not np.array_equal(array,loaded[key]):
            raise ValueError(f'Loaded adapter differs from saved checkpoint: {key}')


def prepare(parent, out, deadline=DEFAULT_DEADLINE):
    parent,out = Path(parent).resolve(),Path(out).resolve()
    limit = datetime.fromisoformat(deadline)
    if limit.tzinfo is None:
        raise ValueError('Deadline must include a timezone')
    if out.exists():
        manifest = json.loads((out/'manifest.json').read_text())
        same(manifest['parent_run'],str(parent),'Existing run has a different parent')
        same(manifest['deadline'],deadline,'Existing run has a different fixed deadline')
        same(manifest['source_sha256'][SOURCE_KEY],sha(__file__),'Runner code changed; resume with the archived runner')
        for original,expected in manifest['source_sha256'].items():
            same(sha(out/'source'/Path(original).name),expected,f'Archive changed: {original}')
        schedule,rosters = make_schedule(manifest['environments'],manifest['continuation_cycles'],manifest['maximum_roster'])
        same(json.loads((out/'schedule.json').read_text()),schedule,'Frozen continuation schedule changed')
        same(manifest['evaluation_rosters'],rosters,'Frozen evaluation roster changed')
        for name,expected in manifest['inherited_checkpoints'].items():
            actual,_ = checkpoint_record(expected['path'],name,manifest['model'])
            same(actual,expected,f'Inherited checkpoint changed: {name}')
        return manifest,schedule
    initial = json.loads((parent/'manifest.json').read_text())
    summary = json.loads((parent/'summary.json').read_text())
    audit = json.loads((parent/'analysis.json').read_text())
    if summary.get('complete') is not True or audit.get('complete') is not True or not audit.get('all_completed_episodes_verified'):
        raise ValueError('Parent run must finish and pass independent complete replay analysis first')
    same(audit['manifest_sha256'],sha(parent/'manifest.json'),'Parent analysis is for another manifest')
    if audit.get('missing_episode_ids'):
        raise ValueError('Parent audit has missing episodes')
    inherited = {name:checkpoint_record(parent/'checkpoints'/name,name,initial['model'])[0]
                 for name in ('base','cycle1','cycle2')}
    for name, record in inherited.items():
        same(audit['checkpoint_verification'][name]['adapter_sha256'],record['adapter_sha256'],
             f'Parent checkpoint differs from audited data: {name}')
    sources = dict(initial['source_sha256']); sources[SOURCE_KEY] = sha(__file__)
    schedule,rosters = make_schedule(initial['environments'])
    manifest = {**initial,'experiment':'league-v1','rating_compatibility':'arena-v1',
        'parent_run':str(parent),'parent_manifest_sha256':sha(parent/'manifest.json'),
        'parent_analysis_sha256':sha(parent/'analysis.json'),'continuation_cycles':[3,4],
        'deadline':deadline,'maximum_roster':8,'evaluation_rosters':rosters,
        'inherited_checkpoints':inherited,'source_sha256':sources,'rollout_chunk_episodes':6,
        'training':'same current checkpoint in both seats; historical league is evaluation only',
        'external_spend_usd':0,'resume':'verify completed episodes; restart incomplete update from its original checkpoint and frozen rollouts',
        'created_at':timestamp()}
    manifest['versions']={**initial['versions'],'numpy':np.__version__}
    temporary = out.parent/('.'+out.name+'.preparing-'+uuid.uuid4().hex)
    temporary.mkdir(parents=True)
    (temporary/'source').mkdir(); (temporary/'checkpoints').mkdir(); (temporary/'rollout-parts').mkdir()
    for original,expected in sources.items():
        source = Path(__file__) if original == SOURCE_KEY else parent/'source'/Path(original).name
        same(sha(source),expected,f'Parent source archive changed: {original}')
        shutil.copyfile(source,temporary/'source'/Path(original).name)
    atomic_json(temporary/'manifest.json',manifest)
    atomic_json(temporary/'schedule.json',schedule)
    atomic_json(temporary/'status.json',{'status':'prepared','stages':{},'updated_at':timestamp(),'external_spend_usd':0})
    temporary.rename(out)
    return manifest,schedule


def archived_arena(out, manifest):
    name = '_league_arena_'+hashlib.sha256(str(Path(out).resolve()).encode()).hexdigest()[:12]
    source = Path(out)/'source'
    for original,expected in manifest['source_sha256'].items():
        same(sha(source/Path(original).name),expected,f'Archive changed: {original}')
    spec = importlib.util.spec_from_file_location(name,source/'__init__.py',submodule_search_locations=[str(source)])
    module = importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    policy = importlib.import_module(name+'.policy')
    experiment = importlib.import_module(name+'.experiment')
    return module,policy,experiment


def verify_episode(row, spec, factory, labels, prompt_function):
    for key,value in spec.items():
        same(row.get(key),value,f'Episode schedule mismatch: {spec["episode_id"]}/{key}')
    env = factory(spec['environment'],spec['seed'],spec['regime'])
    same(row['initial_state'],env.snapshot(),'Replay initial state mismatch')
    transitions = {}
    for transition in row['transitions']:
        key = (transition['turn'],transition['player'])
        if key in transitions:
            raise ValueError('Duplicate policy transition')
        transitions[key]=transition
    for event in row['events']:
        if env.done:
            raise ValueError('Recorded event after terminal state')
        same(event['turn'],env.turn,'Replay turn mismatch')
        actions = {int(k):v for k,v in event['actions'].items()}
        same(sorted(actions),sorted(env.active_players()),'Replay active players mismatch')
        for player in env.active_players():
            identity = spec['policies'][player]
            if identity == 'heuristic':
                same(actions[player],env.heuristic_action(player),'Replay heuristic mismatch')
                continue
            transition = transitions.pop((env.turn,player))
            seed = spec['seed']*100000+env.turn*37+player*11
            menu = deepcopy(env.legal_actions(player));random.Random(seed+907).shuffle(menu)
            observation = env.observation(player)
            same(transition['checkpoint'],identity,'Replay behavior policy mismatch')
            same(transition['observation'],observation,'Replay private/public observation mismatch')
            same(transition['actions'],menu,'Replay action menu mismatch')
            same(transition['sample_seed'],seed,'Replay sampling seed mismatch')
            same(transition['prompt'],prompt_function(observation,menu,labels),'Replay prompt mismatch')
            pick = transition['action_index']
            if type(pick) is not int or not 0 <= pick < len(menu):
                raise ValueError('Replay action index is invalid')
            same(transition['action_label'],labels[pick],'Replay action label mismatch')
            same(transition['action'],menu[pick],'Replay selected action mismatch')
            same(actions[player],menu[pick],'Replay executed action mismatch')
            lp = np.asarray(transition['old_log_probs'],dtype=np.float64)
            if len(lp)!=len(menu) or not np.isfinite(lp).all() or abs(np.exp(lp).sum()-1)>2e-5:
                raise ValueError('Replay behavior probabilities invalid')
            if abs(transition['old_log_prob']-float(lp[pick]))>1e-7:
                raise ValueError('Replay chosen behavior probability mismatch')
            probabilities=np.exp(lp);probabilities/=probabilities.sum()
            same(int(np.random.default_rng(seed).choice(len(menu),p=probabilities)),pick,'Replay random sample mismatch')
            same(transition['terminal_return'],row['returns'][player],'Replay attributed reward mismatch')
        env.step(actions)
        same(event['state'],env.snapshot(),'Replay post-action state mismatch')
    if transitions or not env.done:
        raise ValueError('Incomplete episode or unconsumed transitions')
    same(row['returns'],env.returns(),'Replay terminal rewards mismatch')
    same(row['metrics'],env.metrics(),'Replay terminal metrics mismatch')


def complete_jsonl_rows(path):
    """A crash may leave one incomplete trailing write; preserve it in the part."""
    if not Path(path).exists():
        return []
    lines=Path(path).read_bytes().splitlines(keepends=True)
    if lines and not lines[-1].endswith(b'\n'):
        lines=lines[:-1]
    return [json.loads(line) for line in lines]


class Continuation:
    def __init__(self,out,manifest,schedule,arena,policy_module,experiment,policy_factory=None,
                 checkpoint_loader=None,now=None):
        self.out,self.manifest,self.schedule=Path(out),manifest,schedule
        self.arena,self.policy_module,self.experiment=arena,policy_module,experiment
        self.policy_factory=policy_factory or (lambda:policy_module.ActionPolicy(manifest['model']))
        self.checkpoint_loader=checkpoint_loader or self._load_checkpoint
        self.now=now or (lambda:datetime.now(ZoneInfo('America/Los_Angeles')))
        self.deadline=datetime.fromisoformat(manifest['deadline'])
        self.policy=None
        self.status=json.loads((self.out/'status.json').read_text())
        self.labels=json.loads((Path(manifest['inherited_checkpoints']['base']['path'])/'metadata.json').read_text())['labels']

    def record(self,kind,stage=None,**fields):
        event={'kind':kind,'stage':stage,'at':timestamp(),**fields}
        with (self.out/'journal.jsonl').open('a') as handle:
            handle.write(json.dumps(event,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno())
        overall='running' if kind=='complete' and stage else kind
        self.status.update(status=overall,current_stage=stage,updated_at=event['at'],external_spend_usd=0)
        if stage:
            self.status.setdefault('stages',{}).setdefault(stage,{}).update(status=kind,**fields)
        atomic_json(self.out/'status.json',self.status)

    def admitted(self,stage):
        if self.now() >= self.deadline:
            self.record('deadline_reached',stage,deadline=self.manifest['deadline'])
            return False
        return True

    def _load_checkpoint(self,policy,name,directory):
        from mlx.utils import tree_flatten
        record,metadata=checkpoint_record(directory,name,self.manifest['model'])
        same(metadata['labels'],self.labels,f'Checkpoint label interface changed: {name}')
        same(policy.labels,self.labels,f'Runtime label interface differs from inherited checkpoint: {name}')
        same(metadata['label_token_ids'],policy.label_ids,f'Runtime action token IDs changed: {name}')
        weights=policy.mx.load(str(Path(directory)/'adapters.safetensors'))
        saved=validate_adapter_tensors(weights,dict(tree_flatten(policy.model.trainable_parameters())))
        policy.model.load_weights(list(weights.items()),strict=False)
        policy.mx.eval(policy.model.trainable_parameters())
        verify_loaded_adapter_tensors(saved,dict(tree_flatten(policy.model.trainable_parameters())))
        policy.save_memory(name);policy.active=name
        return record

    def get_policy(self):
        if self.policy is None:
            self.policy=self.policy_factory()
            for name,record in self.manifest['inherited_checkpoints'].items():
                self.checkpoint_loader(self.policy,name,record['path'])
            for cycle in self.manifest['continuation_cycles']:
                path=self.out/'checkpoints'/f'cycle{cycle}'
                if has_complete_summary(path):
                    self.checkpoint_loader(self.policy,f'cycle{cycle}',path)
        return self.policy

    def recovered_rollouts(self,stage):
        specs=self.schedule[stage];expected={s['episode_id']:s for s in specs};rows={}
        parts=self.out/'rollout-parts'/stage
        paths=[self.out/(stage+'.jsonl')]+sorted(parts.glob('*.jsonl'))
        for path in paths:
            for row in complete_jsonl_rows(path):
                identity=row['episode_id']
                if identity not in expected:
                    raise ValueError(f'Unknown episode in stage {stage}: {identity}')
                if identity in rows:
                    same(row,rows[identity],f'Conflicting duplicate episode: {identity}')
                    continue
                verify_episode(row,expected[identity],self.arena.make_env,self.labels,self.policy_module.action_prompt)
                rows[identity]=row
        ordered=[rows[s['episode_id']] for s in specs if s['episode_id'] in rows]
        if ordered:
            atomic_rows(self.out/(stage+'.jsonl'),ordered)
        return ordered

    def rollout_stage(self,stage):
        rows=self.recovered_rollouts(stage)
        done={r['episode_id'] for r in rows}
        missing=[s for s in self.schedule[stage] if s['episode_id'] not in done]
        while missing:
            if not self.admitted(stage):
                return None
            chunk=missing[:self.manifest['rollout_chunk_episodes']]
            parts=self.out/'rollout-parts'/stage;parts.mkdir(exist_ok=True)
            path=parts/('attempt-'+uuid.uuid4().hex+'.jsonl')
            self.record('rollout_running',stage,completed_episodes=len(rows),expected_episodes=len(self.schedule[stage]),part=str(path))
            policy=self.get_policy()
            if not self.admitted(stage):
                return None
            self.experiment.play(policy,chunk,path,batch_size=self.manifest['inference_batch_size'])
            rows=self.recovered_rollouts(stage)
            done={r['episode_id'] for r in rows}
            missing=[s for s in self.schedule[stage] if s['episode_id'] not in done]
        self.record('complete',stage,completed_episodes=len(rows),sha256=sha(self.out/(stage+'.jsonl')))
        return rows

    def training_stage(self,cycle,episodes):
        name,source=f'cycle{cycle}',f'cycle{cycle-1}'
        stage='train-'+name;directory=self.out/'checkpoints'/name
        rollout_hash=sha(self.out/f'train-rollouts-{cycle}.jsonl')
        prior=self.status.get('stages',{}).get(stage,{})
        if prior.get('rollout_sha256') and prior['rollout_sha256']!=rollout_hash:
            raise ValueError(f'Training rollouts changed before resume: {stage}')
        if has_complete_summary(directory):
            record,_=checkpoint_record(directory,name,self.manifest['model'])
            if prior.get('checkpoint'):
                same(record,prior['checkpoint'],f'Previously completed checkpoint changed: {name}')
            if self.policy is not None and name not in self.policy.checkpoints:
                self.checkpoint_loader(self.policy,name,directory)
            self.record('complete',stage,checkpoint=record,rollout_sha256=rollout_hash,reused=True)
            return True
        if prior.get('checkpoint'):
            raise ValueError(f'Previously completed checkpoint is now missing/incomplete: {name}')
        if not self.admitted(stage):
            return False
        policy=self.get_policy()
        if not self.admitted(stage):
            return False
        if directory.exists():
            preserved=directory.with_name(name+'.incomplete-'+uuid.uuid4().hex)
            directory.rename(preserved)
            self.record('incomplete_update_preserved',stage,preserved_directory=str(preserved),rollout_sha256=rollout_hash)
        # Force reloading the original behavior snapshot even after a failed
        # in-process update; never continue from half-updated in-memory weights.
        policy.active=None;policy.activate(source)
        self.record('training_running',stage,source=source,target=name,rollout_sha256=rollout_hash,seed=72020+cycle)
        summary=policy.train(episodes,source,name,directory,seed=72020+cycle,
                             batch_size=self.manifest['minibatch_size'],learning_rate=self.manifest['learning_rate'])
        if summary.get('complete') is not True:
            raise ValueError('Training returned without a complete checkpoint')
        record,_=checkpoint_record(directory,name,self.manifest['model'])
        self.record('complete',stage,checkpoint=record,rollout_sha256=rollout_hash,reused=False)
        return True

    def run(self):
        try:
            for cycle in self.manifest['continuation_cycles']:
                episodes=self.rollout_stage(f'train-rollouts-{cycle}')
                if episodes is None or not self.training_stage(cycle,episodes):
                    return self.status
                candidate=f'cycle{cycle}'
                for opponent in self.manifest['evaluation_rosters'][candidate]:
                    if self.rollout_stage(f'eval-{candidate}-{opponent}') is None:
                        return self.status
            result={'complete':True,'continuation_cycles':self.manifest['continuation_cycles'],
                'parent_run':self.manifest['parent_run'],'rating_compatibility':'arena-v1',
                'external_spend_usd':0,'finished_at':timestamp(),
                'evaluation_rosters':self.manifest['evaluation_rosters'],
                'training':'Current self-play; historical policies are evaluation opponents only',
                'stage_records':self.status['stages']}
            atomic_json(self.out/'summary.json',result)
            self.record('complete',None,continuation_cycles=self.manifest['continuation_cycles'])
            return self.status
        except Exception as error:
            self.record('failed',self.status.get('current_stage'),error=f'{type(error).__name__}: {error}')
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,default=Path('runs/arena-v1'))
    parser.add_argument('--out',type=Path,default=Path('runs/league-v1'))
    parser.add_argument('--deadline',default=DEFAULT_DEADLINE)
    args=parser.parse_args()
    with process_lock(args.out):
        manifest,schedule=prepare(args.parent,args.out,args.deadline)
        for name,expected in manifest['versions'].items():
            same(importlib.metadata.version(name),expected,f'Runtime version changed: {name}')
        arena,policy,experiment=archived_arena(args.out,manifest)
        result=Continuation(args.out,manifest,schedule,arena,policy,experiment).run()
    print(json.dumps({'status':result['status'],'run':str(args.out.resolve()),'external_spend_usd':0}))


if __name__=='__main__':
    main()
