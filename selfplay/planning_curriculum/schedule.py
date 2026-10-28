"""Pre-model planning schedules and canonical historical family exclusion."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from . import envs
from ..environment import Task
from ..arena_mixed.planning import task_family_payload

TRAINING_SEEDS = (742101, 742102)
ARMS = ('control', 'curriculum')
ENDPOINTS = ('base',) + tuple(f'{seed}-{arm}' for seed in TRAINING_SEEDS for arm in ARMS)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def historical_inventory(root):
    """Read all saved/prepared planning schedules plus legacy task JSONL files.

    No trajectory is played. Scheduled tasks are reconstructed at reset and
    checked against any declared canonical fingerprint and manifest file hash.
    Recursion includes preserved parent evidence, prepared runs and nested runs.
    """
    root = Path(root).resolve()
    inputs, fingerprints, rows = {}, set(), []

    def read(path):
        raw = path.read_bytes()
        inputs[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        return raw

    def specs(value):
        if isinstance(value, dict):
            if value.get('environment') == 'planning' and 'seed' in value:
                yield value
            else:
                for child in value.values():
                    yield from specs(child)
        elif isinstance(value, list):
            for child in value:
                yield from specs(child)

    for path in sorted((root/'runs').rglob('schedule.json')):
        raw = read(path)
        values = list(specs(json.loads(raw)))
        if not values:
            continue
        folder = path.parent.parent if path.parent.name == 'inputs' else path.parent
        manifest_path = folder/'manifest.json'
        if manifest_path.is_file():
            manifest = json.loads(read(manifest_path))
            expected = manifest.get('inputs', {}).get(str(path.relative_to(folder)))
            if expected is not None:
                _require(expected == hashlib.sha256(raw).hexdigest(), 'Historical schedule manifest hash differs')
        seen_specs = set()
        for spec in values:
            depth = spec.get('depth', (3,4,5,6)[spec['seed'] % 4])
            regime = spec.get('regime', 'train')
            # OOD historical modulus13 tasks must retain their original identity.
            if regime == 'ood' and 'depth' not in spec:
                old = envs.HistoricalPlanningEnv(spec['seed'], regime=regime, max_rounds=12)
                fingerprint = envs.scenario_fingerprint(initial_state=old.snapshot())
            else:
                fingerprint = envs.scenario_fingerprint(seed=spec['seed'], regime=regime, depth=depth)
            if 'scenario_sha256' in spec:
                _require(spec['scenario_sha256'] == fingerprint, 'Historical planning family reconstruction differs')
            fingerprints.add(fingerprint)
            key = (spec['seed'], regime, depth, fingerprint)
            if key not in seen_specs:
                rows.append({'path':str(path.relative_to(root)), 'seed':spec['seed'], 'regime':regime,
                             'depth':depth, 'scenario_sha256':fingerprint})
                seen_specs.add(key)
    task_fields = set(Task.__dataclass_fields__)
    for path in sorted((root/'runs').rglob('tasks.jsonl')):
        for line, raw in enumerate(read(path).decode().splitlines(), 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            value = value.get('task', value)
            task = Task.from_dict({name:value[name] for name in task_fields})
            state = {'turn':0, 'scenario_identity':{'environment':'planning', 'canonical_task':task_family_payload(task)}}
            fingerprint = envs.scenario_fingerprint(initial_state=state)
            fingerprints.add(fingerprint)
            rows.append({'path':str(path.relative_to(root)), 'line':line, 'scenario_sha256':fingerprint})
    for relative in ('selfplay/environment.py', 'selfplay/arena_mixed/planning.py',
                     'selfplay/planning_curriculum/envs.py', 'selfplay/planning_curriculum/schedule.py'):
        path = root/relative
        if path.is_file():
            read(path)
    for relative, expected in inputs.items():
        _require(hashlib.sha256((root/relative).read_bytes()).hexdigest() == expected,
                 'Historical inventory input changed during read')
    return {'kind':'planning_curriculum_historical_inventory_v1', 'verified':True,
            'fingerprints':sorted(fingerprints), 'source_sha256':inputs, 'rows':rows,
            'distinct_planning_families':len(fingerprints),
            'method':'All runs/**/schedule.json including unplayed prepared schedules, and runs/**/tasks.jsonl; exact canonical x/y-symmetric task identity; no outcomes or model calls.'}


def make_schedules(prior_fingerprints):
    excluded = set(prior_fingerprints)
    _require(all(isinstance(x,str) and len(x)==64 and set(x)<=set('0123456789abcdef') for x in excluded),
             'Historical fingerprints must be SHA256 strings')
    used, selected, rejected = set(excluded), [], []
    next_seed = {depth: 2742100 + depth*100000 for depth in range(1,7)}

    def allocate(depth):
        for _ in range(100000):
            seed = next_seed[depth]
            next_seed[depth] += 1
            fingerprint = envs.scenario_fingerprint(seed=seed, depth=depth)
            item = {'environment':'planning','seed':seed,'regime':'train','depth':depth,
                    'scenario_sha256':fingerprint,'scenario_family_id':'planning:'+fingerprint}
            if fingerprint in used:
                rejected.append({**item,'reason':'historical_or_preassigned_canonical_family'})
                continue
            used.add(fingerprint)
            selected.append(item)
            return item
        raise ValueError('Unable to allocate required independent families within fixed search bound')

    def spec(item, episode_id, split, paired_key, **extra):
        return {**deepcopy(item),'episode_id':episode_id,'split':split,'learner_seats':[0],
                'opponent':'none','paired_seed_key':paired_key,**extra}

    schedules = {'screen-hard':[], 'screen-easy':[]}
    for group, depths, count in [('hard',(3,4,5,6),8),('easy',(1,2),16)]:
        for depth in depths:
            for index in range(count):
                identifier = f'screen-{group}-depth{depth}-{index:02d}'
                pair = f'easy-depth{depth}-pair{index//2:02d}' if group=='easy' else None
                schedules['screen-'+group].append(spec(allocate(depth),identifier,'screen',identifier,
                                                       screen_group=group,pair_id=pair))
    for seed in TRAINING_SEEDS:
        control = {}
        for cohort in range(1,21):
            depths = (3,4) if cohort % 2 else (5,6)
            control[cohort] = [allocate(depth) for depth in depths]
        for arm in ARMS:
            for cohort in range(1,21):
                items = [allocate(1),allocate(2)] if arm=='curriculum' and cohort<=10 else control[cohort]
                stage = f'train-{seed}-{arm}-{cohort}'
                schedules[stage] = [spec(item,f'{stage}-{slot}','train',f'train-{seed}-{cohort}-{slot}',
                                         training_seed=seed,arm=arm,cohort=cohort)
                                    for slot,item in enumerate(items)]
    evaluation = [allocate(depth) for depth in (3,4,5,6) for _ in range(16)]
    for endpoint in ENDPOINTS:
        schedules['eval-'+endpoint] = [spec(item,f'eval-{endpoint}-{i:02d}','eval',f'eval-planning-{i:02d}',endpoint=endpoint)
                                      for i,item in enumerate(evaluation)]
    verification = validate_schedules(schedules, prior_fingerprints)
    receipt = {'selected':selected,'rejected':rejected,
               'selection':'First unused exact initial task families at each declared depth; before any policy draw.',
               'timing_probe_episode_ids':[x['episode_id'] for x in schedules['screen-hard'] if x['depth']==6][:2],
               'training_seeds':list(TRAINING_SEEDS),'endpoints':list(ENDPOINTS)}
    return schedules, receipt, verification


def training_stages(schedules, seed, arm):
    _require(seed in TRAINING_SEEDS and arm in ARMS,'Unknown fixed training arm/seed')
    return {f'train-{i}':deepcopy(schedules[f'train-{seed}-{arm}-{i}']) for i in range(1,21)}


def validate_schedules(schedules, prior_fingerprints=()):
    expected = {'screen-hard','screen-easy'} | {f'train-{seed}-{arm}-{i}' for seed in TRAINING_SEEDS for arm in ARMS for i in range(1,21)} | {'eval-'+e for e in ENDPOINTS}
    _require(set(schedules)==expected,'All fixed screen/training/evaluation stages required')
    seen_ids, origins, prior = set(), {}, set(prior_fingerprints)
    counts = Counter()
    for stage, rows in schedules.items():
        _require(isinstance(rows,list) and len(rows)==(32 if stage.startswith('screen-') else 64 if stage.startswith('eval-') else 2),'Stage episode count differs')
        for offset,row in enumerate(rows):
            _require(row['episode_id'] not in seen_ids,'Duplicate episode ID')
            seen_ids.add(row['episode_id'])
            _require(row['environment']=='planning' and row['regime']=='train' and row['learner_seats']==[0]
                     and type(row['learner_seats'][0]) is int and row['opponent']=='none','Planning role/rules differ')
            fingerprint=envs.scenario_fingerprint(seed=row['seed'],regime=row['regime'],depth=row['depth'])
            _require(row['scenario_sha256']==fingerprint and row['scenario_family_id']=='planning:'+fingerprint,'Task depth/family binding differs')
            _require(fingerprint not in prior,'Historical family reused')
            if stage.startswith('screen-'):
                group=stage.removeprefix('screen-')
                depth=(3+offset//8) if group=='hard' else (1+offset//16)
                index=offset%8 if group=='hard' else offset%16
                identifier=f'screen-{group}-depth{depth}-{index:02d}'
                pair=f'easy-depth{depth}-pair{index//2:02d}' if group=='easy' else None
                _require(row['split']=='screen' and row['screen_group']==group and row['depth']==depth
                         and row['pair_id']==pair and row['episode_id']==identifier and row['paired_seed_key']==identifier,'Screen order/depth/pair differs')
                origin=('screen',identifier)
            elif stage.startswith('train-'):
                _,seed_text,arm,index_text=stage.split('-');seed,index=int(seed_text),int(index_text)
                depth=(1,2)[offset] if arm=='curriculum' and index<=10 else ((3,4) if index%2 else (5,6))[offset]
                _require(row['split']=='train' and row['training_seed']==seed and row['arm']==arm and row['cohort']==index
                         and row['depth']==depth and row['episode_id']==f'{stage}-{offset}'
                         and row['paired_seed_key']==f'train-{seed}-{index}-{offset}','Training treatment/seed schedule differs')
                origin=('train',seed,'shared-late' if index>10 else arm,index,offset)
            else:
                endpoint=stage.removeprefix('eval-')
                _require(row['split']=='eval' and row['depth']==3+offset//16 and row['endpoint']==endpoint
                         and row['episode_id']==f'eval-{endpoint}-{offset:02d}' and row['paired_seed_key']==f'eval-planning-{offset:02d}',
                         'Shared final evaluation order differs')
                origin=('eval',offset)
            _require(fingerprint not in origins or origins[fingerprint]==origin,'Unapproved family reuse across splits/arms/seeds')
            origins[fingerprint]=origin;counts[row['split']]+=1
    for seed in TRAINING_SEEDS:
        for index in range(11,21):
            a=schedules[f'train-{seed}-control-{index}'];b=schedules[f'train-{seed}-curriculum-{index}']
            for left,right in zip(a,b,strict=True):
                _require({k:v for k,v in left.items() if k not in ('episode_id','arm')}=={k:v for k,v in right.items() if k not in ('episode_id','arm')},'Late paired training scenarios/seeds must match exactly')
    baseline=schedules['eval-base']
    for endpoint in ENDPOINTS[1:]:
        for left,right in zip(baseline,schedules['eval-'+endpoint],strict=True):
            _require({k:v for k,v in left.items() if k not in ('episode_id','endpoint')}=={k:v for k,v in right.items() if k not in ('episode_id','endpoint')},'All five endpoints require exact shared evaluation scenarios/seeds')
    _require(dict(counts)=={'screen':64,'train':160,'eval':320} and len(origins)==248,'Frozen complete budget/unique family count differs')
    return {'verified':True,'episodes':len(seen_ids),'episodes_by_split':dict(counts),'distinct_scenarios':len(origins),
            'scenario_sha256':sorted(origins),'schedule_sha256':envs.digest(schedules),
            'environment_protocol_sha256':envs.digest(envs.protocol_config()),'historical_exclusions':len(prior)}
