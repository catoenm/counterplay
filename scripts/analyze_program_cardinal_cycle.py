"""Independent CPU audit of the bounded cardinal-controller RL cycle.

No model is loaded and generated Python is never executed. Frozen interpreters
recompute games; saved numerical gates are checked, not rerun on a neural model.
This is a prompt/interface/token-budget/learning-rate bundle, not an isolated
interface ablation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
from types import ModuleType, SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_program_study as previous

canonical = previous.canonical
sha = previous.sha
value_sha = previous.value_sha
read = previous.read
require = previous.require
same = previous.same
safe_path = previous.safe_path
ENDPOINTS = ("base", "cycle3")


@contextmanager
def archived_runtime(run, manifest):
    """Only load trusted game/compiler sources whose archived bytes are pinned."""
    run = Path(run)
    prefix = "_cardinal_cycle_audit_" + uuid.uuid4().hex
    package = ModuleType(prefix)
    package.__path__ = [str(run / "source/selfplay")]
    sys.modules[prefix] = package
    try:
        required = ["selfplay/program_arena/" + name + ".py" for name in
                    ("__init__", "dsl", "game", "runner", "examples", "spec")]
        required += ["selfplay/program_python.py", "selfplay/program_cardinal.py"]
        for relative in required:
            require(relative in manifest["source_sha256"], "Missing frozen source: " + relative)
            require(sha(run / "source" / relative) == manifest["source_sha256"][relative],
                    "Archived runtime source changed: " + relative)

        def module(name, relative, is_package=False):
            path = run / "source" / relative
            spec = importlib.util.spec_from_file_location(prefix + "." + name, path,
                submodule_search_locations=[str(path.parent)] if is_package else None)
            obj = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = obj
            spec.loader.exec_module(obj)
            return obj

        arena = module("program_arena", "selfplay/program_arena/__init__.py", True)
        module("program_python", "selfplay/program_python.py")
        frontend = module("program_cardinal", "selfplay/program_cardinal.py")
        examples = module("program_arena.examples", "selfplay/program_arena/examples.py")
        game = sys.modules[prefix + ".program_arena.game"]
        yield SimpleNamespace(classify=frontend.classify_cardinal,
            compile_cardinal=frontend.compile_cardinal_program, compile_ir=arena.compile_program,
            verify_replay=arena.verify_replay, game=game.RobotArena,
            reference_programs=examples.reference_programs, model_spec=frontend.cardinal_model_spec)
    finally:
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                del sys.modules[name]


def score_from_matches(matches, valid):
    """Raw yield keeps every draw; only candidate faults trigger its penalty."""
    return previous.score_from_matches(matches, valid, "game_rl")


def productivity(row):
    """Admission is over the complete 24-game calibration grid, never a subset."""
    if not row["authorship_valid"]:
        require(not row["matches"], "Invalid authoring has invented calibration games")
        return False
    require(len(row["matches"]) == 24, "Productivity requires all 24 calibration games")
    positive = sum(m["replay"]["metrics"]["scores"][m["candidate_seat"]] > 0
                   for m in row["matches"])
    return row["runtime_valid"] and positive >= 12


def capability_decision(rows):
    require(len(rows) == 8, "Capability gate requires all eight prescribed draws")
    productive, distinct = [], []
    seen = set()
    for index, row in enumerate(rows):
        if productivity(row):
            fingerprint = row["classification"]["program"]["ir_sha256"]
            productive.append(index)
            if fingerprint not in seen:
                seen.add(fingerprint)
                distinct.append(index)
    return {"productive_indices": productive, "distinct_productive_indices": distinct,
            "passed": len(productive) >= 4 and len(distinct) >= 2,
            "selected_indices": distinct[:2] if len(productive) >= 4 and len(distinct) >= 2 else []}


def validate_optimizer_rate(folder, learning_rate=1e-5):
    import numpy as np
    from safetensors.numpy import load_file
    values = load_file(str(Path(folder) / "optimizer.safetensors"))
    require("learning_rate" in values, "Actual optimizer learning rate is absent")
    value = values["learning_rate"]
    require(value.size == 1 and value.dtype == np.float32 and np.isfinite(value).all()
            and np.array_equal(value, np.full_like(value, learning_rate)),
            "Actual saved optimizer learning rate differs from registered rate")
    return float(value.reshape(-1)[0])


def validate_optimizer_initialization(original, initialized, learning_rate=1e-5):
    """The new zero-step base changes only Adam's LR scalar, preserving moments."""
    import numpy as np
    from safetensors.numpy import load_file
    original, initialized = Path(original), Path(initialized)
    require(sha(original / "adapter.safetensors") == sha(initialized / "adapter.safetensors"),
            "New base changed the untrained adapter")
    before = load_file(str(original / "optimizer.safetensors"))
    after = load_file(str(initialized / "optimizer.safetensors"))
    require(before.keys() == after.keys() and "step" in before and "learning_rate" in before,
            "Optimizer initialization schema changed")
    require(before["step"].size == 1 and before["step"].dtype.kind in "iu" and not np.any(before["step"])
            and after["step"].size == 1 and after["step"].dtype.kind in "iu" and not np.any(after["step"]),
            "Initialized optimizer is not zero-step")
    for name, value in before.items():
        require(value.shape == after[name].shape and value.dtype == after[name].dtype
                and np.isfinite(value).all() and np.isfinite(after[name]).all(),
                "Optimizer tensor schema/finiteness changed")
        if name != "learning_rate":
            require(np.array_equal(value, after[name]), "Initialization changed optimizer moments/state")
            if name.endswith(".m") or name.endswith(".v"):
                require(not np.any(value), "Initial Adam moments are nonzero")
    validate_optimizer_rate(original, 1e-6)
    return validate_optimizer_rate(initialized, learning_rate)


def paired_summary(base, final, seeds, draws=20000, bootstrap_seed=20261023):
    """Bootstrap independent authoring-seed clusters, not correlated replay games."""
    n = 12
    require(type(draws) is int and draws >= 100, "Bootstrap needs at least 100 resamples")
    require(len(seeds) == len(set(seeds)) == n and all(type(s) is int for s in seeds),
            "Twelve unique paired generation seeds required")
    require(len(base) == len(final) == n, "Incomplete endpoint draws cannot become zero outcomes")
    expected_base = [f"eval-base-{i}" for i in range(n)]
    expected_final = [f"eval-cycle3-{i}" for i in range(n)]
    require([r["generation"] for r in base] == expected_base
            and [r["generation"] for r in final] == expected_final,
            "Paired generation identity/order differs")
    a = [r["banked_point_yield"] for r in base]
    b = [r["banked_point_yield"] for r in final]
    require(all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in a+b),
            "Missing or nonfinite yield cannot become zero")
    delta = [y-x for x,y in zip(a,b)]
    rng = random.Random(bootstrap_seed)
    samples = sorted(math.fsum(delta[rng.randrange(n)] for _ in range(n))/n for _ in range(draws))
    def quantile(q):
        index = q*(draws-1); low = int(index); high = min(low+1, draws-1)
        return samples[low]+(samples[high]-samples[low])*(index-low)
    return {"program_cluster_n": n, "paired_generation_seeds": seeds,
            "base_mean_yield": math.fsum(a)/n, "final_mean_yield": math.fsum(b)/n,
            "raw_paired_yield_differences": delta, "mean_yield_difference": math.fsum(delta)/n,
            "descriptive_bootstrap_95_interval": [quantile(.025), quantile(.975)],
            "bootstrap_draws": draws, "bootstrap_seed": bootstrap_seed,
            "training_seeds": 1,
            "caution": "Descriptive paired-program bootstrap; twelve generation seeds, one training run. "
                       "Invalid draws remain zero yield; missing draws do not. No equivalence, "
                       "isolated interface effect or continual improvement claim."}


def validate_execution_budget(config, status, runtime=None):
    budget = status.get("execution_budget")
    if budget is None:
        require(runtime is None and status["state"] == "prepared", "Missing durable execution budget")
        return
    first = datetime.fromisoformat(budget["first_started"])
    stop = datetime.fromisoformat(budget["hard_stop"])
    deadline = datetime.fromisoformat(config["deadline"])
    require(first.tzinfo is not None and stop.tzinfo is not None and deadline.tzinfo is not None,
            "Execution time lacks timezone")
    require(config["worker_seconds"] == budget["maximum_seconds"] == 21600
            and budget["resume_counts_wall_time"] is True
            and stop == min(first+timedelta(seconds=21600), deadline) and stop > first,
            "Six-hour whole-cycle budget reset or changed")
    if runtime is not None:
        same(runtime["execution_budget"], budget, "Worker execution budget differs")
    if status["state"] == "complete":
        require(datetime.fromisoformat(status["updated"]) <= stop, "Completion exceeded hard stop")


def optimizer_identity(folder, rate):
    import numpy as np
    from safetensors.numpy import load_file
    folder = Path(folder)
    adapter = load_file(str(folder / 'adapter.safetensors'))
    values = load_file(str(folder / 'optimizer.safetensors'))
    names = {f'self_attn.{p}.lora_{s}' for p in ('q_proj', 'v_proj') for s in ('a', 'b')}
    require(set(adapter) == names and set(values) == {'step','learning_rate'} |
            {n+s for n in names for s in ('.m','.v')}, 'Checkpoint tensor schema differs')
    require(values['step'].shape == () and values['step'].dtype == np.uint64,
            'Actual optimizer step is not a uint64 scalar')
    require(values['learning_rate'].shape == (), 'Learning rate is not scalar')
    for name, value in adapter.items():
        require(value.dtype == np.float32 and np.isfinite(value).all(), 'Invalid adapter arithmetic')
        for suffix in ('.m','.v'):
            moment = values[name+suffix]
            require(moment.dtype == np.float32 and moment.shape == value.shape and np.isfinite(moment).all(),
                    'Adam moment schema/finiteness differs')
            if suffix == '.v': require(np.all(moment >= 0), 'Negative second moment')
    return {'learning_rate': validate_optimizer_rate(folder, rate), 'step': int(values['step'].item()),
            'tensor_shapes': {k:list(v.shape) for k,v in values.items()},
            'adapter_sha256':sha(folder/'adapter.safetensors'),
            'optimizer_sha256':sha(folder/'optimizer.safetensors')}


def registered_schedule():
    return {'calibration_maps':list(range(7100000,7100006)),
            'train_maps':{str(i):[7200000+i*100+j for j in range(2)] for i in range(1,4)},
            'holdout_maps':list(range(7300000,7300006)),
            'capability_generation_seeds':list(range(9100000,9100008)),
            'train_generation_seeds':{str(i):[9200000+i*100+j for j in range(8)] for i in range(1,4)},
            'eval_generation_seeds':list(range(9300000,9300012))}


class Audit(previous.Audit):
    def __init__(self, run):
        super().__init__(run)
        self.start_hashes = {p:sha(self.run/p) for p in ('manifest.json','status.json')}
        self.capability = None
        self.pool = None
        self.classifications = {}
        self.all_rows = []
        self.allowed_artifacts = set()

    def artifact(self, relative, required=True):
        self.allowed_artifacts.add(relative)
        return super().artifact(relative, required)

    def frozen(self):
        m, cfg = self.manifest, self.config
        require(m['format_version'] == 1 and m['kind'] == 'cardinal_program_rl_cycle', 'Unknown cycle format')
        require(self.start_hashes['manifest.json'] == self.status['manifest_sha256'], 'Manifest identity differs')
        files = {'plan.json':m['plan_sha256'],'schedule.json':m['schedule_sha256'],
                 'map-fingerprints.json':m['map_fingerprints_sha256'], **m['input_sha256'], **self.artifacts}
        for relative,digest in files.items():
            require(sha(safe_path(self.run,relative)) == digest, 'Artifact hash differs: '+relative)
        for relative,digest in m['source_sha256'].items():
            require(sha(safe_path(self.run/'source',relative)) == digest, 'Frozen source changed: '+relative)
        for relative in ('scripts/analyze_program_study.py','scripts/analyze_program_cardinal_cycle.py'):
            require(m['source_sha256'].get(relative) == sha(ROOT/relative), 'Executed auditor differs from frozen source')
        required_sources = {'scripts/cloud/train_program_rl.py','scripts/program_response_normalization.py',
            'selfplay/program_python.py','selfplay/program_local_rl.py','selfplay/program_study.py',
            'scripts/analyze_program_study.py','selfplay/program_cardinal.py','selfplay/program_cardinal_cycle.py',
            'scripts/analyze_program_cardinal_cycle.py'} | {'selfplay/program_arena/'+n+'.py' for n in
            ('__init__','dsl','game','runner','spec','examples')}
        require(set(m['source_sha256']) == required_sources, 'Frozen source coverage differs')
        parent, ps = read(self.run/'inputs/parent-manifest.json'), read(self.run/'inputs/parent-status.json')
        require(sha(self.run/'inputs/parent-manifest.json') == ps['manifest_sha256'] and ps['state'] == 'complete',
                'Parent completion/manifest provenance differs')
        for relative,digest in parent['source_sha256'].items():
            require(m['source_sha256'].get(relative) == digest, 'Previously frozen source changed: '+relative)
        prior = read(self.run/'inputs/prior-study-manifest.json')
        for relative,digest in prior['source_sha256'].items():
            require(m['source_sha256'].get(relative) == digest, 'Prior study source changed: '+relative)
        same(m['library_source_sha256'],parent['library_source_sha256'],'Library identities changed')
        same(m['expected_installed_versions'],parent['expected_installed_versions'],'Library versions changed')
        same(self.plan['config'],cfg,'Plan configuration differs')
        same(self.plan['schedule'],self.schedule,'Plan schedule differs')
        same(self.schedule,registered_schedule(),'Registered seed schedule differs')
        constants = {'group_size':8,'iterations':3,'max_rounds':48,'max_new_tokens':2048,
            'max_context_tokens':4096,'thinking':False,'worker_seconds':21600,'learning_rate':1e-5,
            'temperature':1.0,'top_p':1.0,'top_k':0,'lora_rank':8,'lora_alpha':16,'lora_dropout':0.0,
            'lora_targets':['q_proj','v_proj'],'lora_last_layers':1,'lora_scale':16.0,
            'gradient_clip':0.5,'compute_dtype':'float32','sample_seconds':300,
            'token_logprob_tolerance':0.002,'sequence_logprob_tolerance':0.02,
            'deadline':'2026-10-22T16:30:00-07:00','memory_limit_bytes':28*1024**3,
            'capability_samples':8,'minimum_productive_programs':4,'minimum_productive_distinct_ir':2,
            'minimum_positive_calibration_games':12,'opponent_count':2,'opponent_attempt_cap':8,
            'eval_samples':12,'map_seeds_per_group':2,'algorithm':'full_completion_RLOO',
            'condition':'clean-cardinal-v1','backend':'mlx','seed':124301,
            'model_repo':'mlx-community/Qwen3.8-27B-4bit',
            'model_revision':'10c35caafbb80f7dc6a7a432cdd11af10a6d4818'}
        for key,value in constants.items(): same(cfg[key],value,'Registered configuration differs: '+key)
        require(m['external_spend_usd'] == 0 and m['claims_require_complete'] is True, 'Resource/claim scope differs')
        require(self.plan['capability_gate']['required_productive'] == 4 and
                self.plan['capability_gate']['required_distinct_productive_ir'] == 2 and
                self.plan['capability_gate']['games_per_compiled_program'] == 24 and
                self.plan['capability_gate']['draws'] == 8 and self.plan['capability_gate']['stop_on_failure'] is True,
                'Registered admission rule differs')
        require(self.plan['evaluation']['endpoints'] == list(ENDPOINTS) and
                self.plan['evaluation']['draws_per_endpoint'] == 12 and
                self.plan['evaluation']['games_per_compiled_program'] == 48 and
                self.plan['resources']['maximum_authoring_draws'] == 56, 'Evaluation/draw budget differs')
        original = self.run/'inputs/parent-base'
        meta = read(original/'metadata.json')
        require(sha(original/'metadata.json') == ps['artifacts']['checkpoints/base/metadata.json'],
                'Original zero-step base metadata unbound to parent')
        require(meta['report'] is None and meta['parent'] is None and meta['complete'] is True, 'Parent base is trained')
        for name,digest in meta['files'].items():
            require(sha(original/name) == digest == ps['artifacts']['checkpoints/base/'+name], 'Parent base bytes changed')
        validate_optimizer_initialization(original,self.run/'inputs/base')
        initialization = read(self.run/'optimizer-initialization.json')
        same(initialization['before'],optimizer_identity(original,1e-6),'Initial optimizer identity differs')
        same(initialization['after'],optimizer_identity(self.run/'inputs/base',1e-5),'Remapped optimizer identity differs')
        same(initialization['changed_tensor_names'],['learning_rate'],'Initialization changed more than LR')
        updated = read(self.run/'inputs/base/metadata.json')
        expected = {**meta,'files':{n:sha(self.run/'inputs/base'/n) for n in meta['files']},
                    'optimizer_initialization_sha256':sha(self.run/'optimizer-initialization.json')}
        same(updated,expected,'New base metadata differs from LR-only amendment')
        self.model_index = read(self.run/'inputs/model-index.json')
        require(sha(self.run/'inputs/model-index.json') == m['model_index_sha256'], 'Model index changed')
        require((self.model_index['repo'],self.model_index['revision']) == (cfg['model_repo'],cfg['model_revision']),
                'Pinned model index identity differs')
        require(sha(self.run/'inputs/parent-runtime.json') == ps['artifacts']['runtime.json'], 'Parent runtime changed')
        same(self.model_index,{k:v for k,v in read(self.run/'inputs/parent-runtime.json')['model_index'].items()
                              if k != 'index_sha256'},'Parent model bytes/index differ')
        return {'calibration':self.schedule['calibration_maps'],
                'train':[s for row in self.schedule['train_maps'].values() for s in row],
                'holdout':self.schedule['holdout_maps']}

    def maps(self,runtime,families):
        document = read(self.run/'map-fingerprints.json')
        def layout(seed):
            g=runtime.game(seed,48)
            return {'grid_size':g.size,'capacity':g.capacity,'max_energy':g.max_energy,
                    'max_rounds':g.max_rounds,'sites':g.sites,
                    'homes':[[r['home_x'],r['home_y']] for r in g.robots]}
        expected={name:[{'seed':s,'layout':layout(s),'sha256':value_sha(layout(s))} for s in seeds]
                  for name,seeds in families.items()}
        same(document['families'],expected,'Public map fingerprint differs')
        old=read(self.run/'inputs/prior-map-fingerprints.json')
        require(sha(self.run/'inputs/prior-map-fingerprints.json') ==
                read(self.run/'inputs/prior-study-manifest.json')['map_fingerprints_sha256'], 'Prior map evidence changed')
        prior=dict(old['prior_seed_fingerprints'])
        for rows in old['families'].values(): prior.update({str(r['seed']):r['sha256'] for r in rows})
        same(document['prior_seed_fingerprints'],prior,'Historical map provenance incomplete')
        same(document['prior_evidence_sha256'],{'inputs/prior-map-fingerprints.json':sha(self.run/'inputs/prior-map-fingerprints.json')},
             'Historical evidence binding differs')
        for s,digest in prior.items(): require(value_sha(layout(int(s))) == digest,'Historical layout changed')
        newseeds=[s for seeds in families.values() for s in seeds]
        require(len(set(newseeds)) == len(newseeds) and not set(newseeds)&{int(s) for s in prior},'Map seeds overlap')
        previous.check_fingerprint_overlap({k:[r['sha256'] for r in rows] for k,rows in expected.items()},list(prior.values()))

    def checkpoint(self,label):
        if label in self.checkpoints: return self.checkpoints[label]
        metadata=self.artifact(f'checkpoints/{label}/metadata.json',False)
        if metadata is None: return None
        require(label in ('base','cycle1','cycle2','cycle3') and metadata['label'] == label and
                metadata['complete'] is True,'Checkpoint identity differs')
        require(metadata['manifest_sha256'] == self.status['manifest_sha256'] and
                (metadata['model_repo'],metadata['model_revision']) == (self.config['model_repo'],self.config['model_revision']),
                'Checkpoint manifest/model differs')
        require(set(metadata['files']) == {'adapter.safetensors','optimizer.safetensors'},'Checkpoint files differ')
        for name,digest in metadata['files'].items():
            path=f'checkpoints/{label}/{name}'
            self.allowed_artifacts.add(path)
            require(self.artifacts.get(path) == digest,'Checkpoint files not atomically committed')
        folder=self.run/'checkpoints'/label
        identity=optimizer_identity(folder,1e-5)
        same(metadata['optimizer_identity'],identity,'Actual optimizer identity differs')
        if label == 'base':
            require(metadata['parent'] is None and metadata['report'] is None,'Base already trained')
            same(metadata['files'],read(self.run/'inputs/base/metadata.json')['files'],'Base initialization differs')
            tensors=previous.validate_checkpoint_tensors(folder,metadata,expected_steps=0)
        else:
            n=int(label[-1]);source='base' if n==1 else f'cycle{n-1}'
            parent=self.checkpoint(source)
            require(parent is not None and metadata['parent'] == source,'Checkpoint lineage differs')
            report=metadata['report']
            same(self.artifact(f'groups/{n}/update.json'),report,'Checkpoint/update atomic transaction differs')
            require(report['source_checkpoint'] == source and report['source_metadata_sha256'] ==
                    self.artifacts[f'checkpoints/{source}/metadata.json'],'Update source identity differs')
            require(type(report['optimizer_steps']) is int and report['optimizer_steps'] in (0,1),'Invalid update count')
            tensors=previous.validate_checkpoint_tensors(folder,metadata,self.run/'checkpoints'/source,
                        self.tensor_reports[source]['optimizer_steps']+report['optimizer_steps'])
        require(tensors['trainable_parameters'] == 188416,'Trainable parameter count differs')
        self.tensor_reports[label]={**tensors,'actual_learning_rate':identity['learning_rate']}
        self.checkpoints[label]=metadata
        return metadata

    def generation(self,name,seed,label,condition='cardinal'):
        record=super().generation(name,seed,label,condition)
        measurement=self.artifact('prompt-measurement.json')
        same(record['prompt_ids'],measurement['prompt_ids'],'Generation used different prompt tokens')
        require(record['completion_ids'],'Empty authoring trajectory')
        return record

    def scored_row(self,row,generation,opponents,seeds,runtime):
        classification,program=runtime.classify(generation['text'],generation['finish_reason'])
        same(row['classification'],classification,'Compilation/source provenance differs')
        self.classifications[generation['name']]=classification
        expected=[(o,s,p) for o in opponents for s in seeds for p in (0,1)] if program else []
        require(len(row['matches']) == len(expected),'Played-match denominator differs')
        for match,(opponent,seed,seat) in zip(row['matches'],expected):
            r=match['replay'];rival=runtime.compile_ir(opponent['compiled_ir'])
            require(match['opponent_id'] == opponent['id'] and type(match['candidate_seat']) is int and
                    match['candidate_seat'] == seat and r['seed'] == seed and r['max_rounds'] == 48,
                    'Match seat/map/opponent differs')
            require(r['programs'][seat]['digest'] == program.digest and r['programs'][1-seat]['digest'] == rival.digest,
                    'Replay controller identity differs')
            require(r['origins'][seat] == 'model' and r['origins'][1-seat] == opponent['origin'],'Replay origin differs')
            fingerprint=value_sha(r)
            if fingerprint not in self.verified_replays:
                runtime.verify_replay(r);self.verified_replays.add(fingerprint)
        calculated=score_from_matches(row['matches'],classification['valid'])
        calculated['positive_banking_games']=sum(m['replay']['metrics']['scores'][m['candidate_seat']] > 0 for m in row['matches'])
        for key,value in calculated.items(): same(row[key],value,'Recomputed score differs: '+key)
        self.all_rows.append(row)
        return calculated

    def capability_and_pool(self,runtime):
        cap=self.artifact('capability.json',False)
        pool=self.artifact('opponents.json',False)
        if cap is None:
            require(pool is None,'Opponent pool committed before capability gate')
            return None
        require(cap['source_checkpoint'] == 'base' and cap['source_metadata_sha256'] ==
                self.artifacts['checkpoints/base/metadata.json'],'Capability source differs')
        seeds=self.schedule['calibration_maps']
        same(cap['seeds'],seeds,'Capability maps differ')
        scripted=[{'id':n,'origin':'scripted','compiled_ir':runtime.reference_programs()[n]} for n in ('nearest','value')]
        same(cap['opponents'],scripted,'Capability anchors differ')
        require(len(cap['rows']) == 8,'Capability lost prescribed draws')
        for i,row in enumerate(cap['rows']):
            name=f'capability-{i}'
            record=self.generation(name,self.schedule['capability_generation_seeds'][i],'base')
            require(row['generation'] == name and row['generation_sha256'] == self.artifacts[f'generations/{name}.json'],
                    'Capability generation provenance differs')
            self.scored_row(row,record,scripted,seeds,runtime)
        decision=capability_decision(cap['rows'])
        for key in ('passed','productive_indices','selected_indices'): same(cap[key],decision[key],'Capability rule differs: '+key)
        require(cap['distinct_productive_ir'] == len(decision['distinct_productive_indices']),'Distinct productive IR count differs')
        self.capability=cap
        if pool is not None:
            require(cap['passed'] and pool['frozen'] is True and pool['capability_sha256'] == self.artifacts['capability.json'],
                    'Opponent gate/provenance differs')
            selected=list(scripted)
            for i in decision['selected_indices']:
                row=cap['rows'][i];p=row['classification']['program'];name=row['generation']
                selected.append({'id':name,'origin':'model','compiled_ir':p['compiled_ir'],'ir_sha256':p['ir_sha256'],
                    'generation_sha256':row['generation_sha256'],'source_path':f'generations/{name}.json',
                    'source_checkpoint':'base','source_metadata_sha256':cap['source_metadata_sha256']})
            same(pool['selected'],selected,'Pool is not first two qualifying distinct controllers')
        self.pool=pool
        return pool

    def groups(self,runtime,pool):
        missing=False;streak=0
        for n in range(1,4):
            folder=f'groups/{n}';label=f'cycle{n}';source='base' if n==1 else f'cycle{n-1}'
            scores=self.artifact(folder+'/scores.json',False);metadata=self.checkpoint(label)
            if scores is None:
                require(metadata is None,'Checkpoint committed without all scores');missing=True;continue
            require(not missing and pool is not None and streak<2,'Training skipped gate/group or capability stop')
            require(scores['cycle'] == n and scores['source_checkpoint'] == source and
                    scores['source_metadata_sha256'] == self.artifacts[f'checkpoints/{source}/metadata.json'] and
                    scores['opponents_sha256'] == self.artifacts['opponents.json'],'Training source provenance differs')
            same(scores['seeds'],self.schedule['train_maps'][str(n)],'Training maps differ')
            require(len(scores['rows']) == 8,'Training lost invalid draws')
            records=[];hashes={}
            for i,row in enumerate(scores['rows']):
                name=f'train-cycle{n}-{i}'
                record=self.generation(name,self.schedule['train_generation_seeds'][str(n)][i],source)
                require(row['generation'] == name and row['generation_sha256'] == self.artifacts[f'generations/{name}.json'],
                        'Scored training generation differs')
                self.scored_row(row,record,pool['selected'],scores['seeds'],runtime)
                records.append(record);hashes[name]=row['generation_sha256']
            rewards=[r['reward'] for r in scores['rows']]
            zero=max(rewards)-min(rewards)<=1e-12
            adv=[0.]*8 if zero else previous.advantages(rewards)
            same(scores['rewards'],rewards,'Training rewards differ');same(scores['advantages'],adv,'RLOO advantages differ')
            require(scores['zero_signal'] is zero,'Zero-signal decision differs')
            invalid=not any(r['runtime_valid'] for r in scores['rows']);streak=streak+1 if invalid else 0
            tokens=sum(len(r['completion_ids']) for r in records)
            valid_returns=[r['game_return'] for r in scores['rows'] if r['runtime_valid']]
            signal='no_update' if zero else 'validity_only_signal' if len(set(valid_returns))<=1 else 'gameplay_and_validity_signal'
            self.group_reports.append({'cycle':n,'checkpoint_committed':metadata is not None,'gradient_signal':signal,
                'rewards':rewards,'advantages':adv,'collected_completion_tokens':tokens,
                'committed_gradient_tokens':tokens if metadata and not zero else 0,
                **previous.summarize_programs(scores['rows'],8)})
            self.training_rows+=scores['rows']
            gates=[p for p in self.artifacts if p.startswith(folder+'/gate-') and p.endswith('.json')]
            for path in gates:
                gate=self.artifact(path)
                same(gate['generation_sha256'],hashes,'Gate generation hashes differ')
                require(gate['source_checkpoint'] == source and gate['source_metadata_sha256'] ==
                        scores['source_metadata_sha256'],'Gate source weights differ')
                validate_recorded_gate(gate,records,list(hashes),self.config)
            if metadata is None: missing=True;continue
            report=metadata['report']
            for key,value in {'cycle':n,'rewards':rewards,'advantages':adv,'generation_sha256':hashes,
                'training_tokens':tokens,'scores_sha256':self.artifacts[folder+'/scores.json'],'all_runtime_invalid':invalid}.items():
                same(report[key],value,'Update report differs: '+key)
            require(report['outcome'] == ('skipped_zero_variance' if zero else 'updated') and
                    report['optimizer_steps'] == (0 if zero else 1),'Update/skip differs')
            if zero:
                require(report['likelihood_gate_path'] is None and report['likelihood_gate_sha256'] is None,'Skip claims gate')
            else:
                path=report['likelihood_gate_path']
                require(path in gates and report['likelihood_gate_sha256'] == self.artifacts[path],'Update gate binding differs')
                previous.validate_gate(self.artifact(path),records,list(hashes),self.config)
                previous.validate_gate(report['gradient_likelihood_gate'],records,list(hashes),self.config)
                require(all(type(report[k]) in (int,float) and math.isfinite(report[k]) for k in
                    ('loss','gradient_norm_before_clip','peak_memory_bytes')) and report['gradient_norm_before_clip'] > 0,
                    'Nonfinite/zero gradient update')
                require(len(report['microbatch_losses']) == 8 and all(math.isfinite(x) for x in report['microbatch_losses']),
                        'Gradient microbatch loss coverage differs')
                same(math.fsum(report['microbatch_losses']),report['loss'],'Gradient loss sum differs')
        self.invalid_streak=streak

    def all_generations(self,runtime):
        expected={f'capability-{i}':(s,'base') for i,s in enumerate(self.schedule['capability_generation_seeds'])}
        for n in range(1,4):
            source='base' if n==1 else f'cycle{n-1}'
            expected.update({f'train-cycle{n}-{i}':(s,source) for i,s in enumerate(self.schedule['train_generation_seeds'][str(n)])})
        expected.update({f'eval-{e}-{i}':(s,e) for e in ENDPOINTS for i,s in enumerate(self.schedule['eval_generation_seeds'])})
        for relative in self.artifacts:
            if relative.startswith('generations/'):
                name=Path(relative).stem
                require(relative == f'generations/{name}.json' and name in expected,'Unexpected extra authoring draw')
                seed,label=expected[name]
                if name.startswith('train-'): require(self.pool is not None,'Training before capability admission')
                if name.startswith('eval-'): require(self.checkpoint('cycle3') is not None,'Evaluation before fixed final checkpoint')
                record=self.generation(name,seed,label)
                self.classifications[name]=runtime.classify(record['text'],record['finish_reason'])[0]

    def evaluation(self,runtime,pool):
        seal=self.artifact('evaluation/sealed.json',False)
        if seal is not None:
            require(pool is not None and all(self.checkpoint(e) is not None for e in ENDPOINTS),'Premature evaluation seal')
            paths=[f'generations/eval-{e}-{i}.json' for e in ENDPOINTS for i in range(12)]
            expected={'checkpoints':{e:self.artifacts[f'checkpoints/{e}/metadata.json'] for e in ENDPOINTS},
                'generations':{p:self.artifacts.get(p) for p in paths},'draws':24,'scoring_allowed':True}
            require(all(expected['generations'].values()),'Evaluation sealed with missing draws')
            same(seal,expected,'Evaluation seal differs')
        for endpoint in ENDPOINTS:
            saved=self.artifact(f'evaluation/{endpoint}/holdout.json',False)
            if saved is None:continue
            require(seal is not None,'Evaluation scored before all24 draws sealed')
            require(saved['checkpoint'] == endpoint and saved['prompt_condition'] == 'cardinal' and
                    saved['split'] == 'holdout' and saved['training_use'] is False and
                    saved['prescribed_program_draws'] == 12 == len(saved['scores']),'Evaluation denominator/condition differs')
            same(saved['seeds'],self.schedule['holdout_maps'],'Holdout maps differ')
            same(saved['opponents'],pool['selected'],'Evaluation opponent pool differs')
            require(saved['opponents_sha256'] == self.artifacts['opponents.json'] and
                    saved['seal_sha256'] == self.artifacts['evaluation/sealed.json'],'Evaluation provenance differs')
            for i,row in enumerate(saved['scores']):
                name=f'eval-{endpoint}-{i}'
                require(row['generation'] == name and row['generation_sha256'] == self.artifacts[f'generations/{name}.json'],
                        'Evaluated authoring output differs')
                self.scored_row(row,self.generations[name],pool['selected'],saved['seeds'],runtime)
            self.evaluations[endpoint]=previous.summarize_programs(saved['scores'],12)

    def diagnostics(self):
        measurement=self.artifact('prompt-measurement.json',False)
        if measurement is not None:
            require(measurement['condition'] == 'clean-cardinal-v1' and measurement['truncation'] is False and
                    measurement['reserved_completion_tokens'] == 2048 and measurement['context_cap'] == 4096 and
                    measurement['prompt_tokens'] == len(measurement['prompt_ids']) > 0 and
                    measurement['maximum_total_tokens'] == len(measurement['prompt_ids'])+2048 and
                    measurement['admitted'] == (measurement['maximum_total_tokens']<=4096),'Prompt admission differs')
            require((self.run/'prompts/cardinal.txt').read_text() in measurement['prompt_template'],
                    'Tokenized template omits full registered prompt')
        for path in self.artifacts:
            if path.startswith('smoke-') and path.endswith('.json'):
                smoke=self.artifact(path);sample=smoke['sample']
                require(smoke['passed'] is True and smoke['optimizer_steps'] == 0 and
                        0 < len(sample['completion_ids']) <= 32 and len(sample['completion_ids']) == len(sample['sampled_token_logprobs']),
                        'Derivative diagnostic scope differs')
                same(sample['prompt_ids'],measurement['prompt_ids'],'Smoke prompt differs')
                previous.validate_gate(smoke['gate'],[sample],[],self.config)
                previous.validate_gate({'passed':True,'records':[smoke['gradient_likelihood_check']]},[sample],[],self.config)
                require(math.isfinite(smoke['loss']) and math.isfinite(smoke['gradient_norm']) and smoke['gradient_norm']>0,
                        'Invalid derivative smoke')

    def finish(self):
        complete=(all(e in self.checkpoints for e in ENDPOINTS) and len(self.group_reports)==3
                  and len(self.evaluations)==2 and len(self.generations)==56)
        require(self.status['claims_allowed'] is False or complete,'Incomplete run allows learning claims')
        require(self.status['state'] != 'complete' or complete,'Premature completed state')
        if self.status['state'] == 'capability_failed':
            require(self.capability is not None and (not self.capability['passed'] or self.invalid_streak==2),
                    'Capability failure lacks backing evidence')
        unknown=set(self.artifacts)-self.allowed_artifacts
        require(not unknown,'Unrecognized committed artifacts: '+str(sorted(unknown)))
        for p,digest in self.start_hashes.items():require(sha(self.run/p)==digest,'Run changed during CPU audit: '+p)
        contrast=paired_summary(self.evaluations['base']['programs'],self.evaluations['cycle3']['programs'],
                                self.schedule['eval_generation_seeds']) if complete else None
        categories={}
        for row in self.classifications.values():
            key='compiled' if row['valid'] else row['reason'];categories[key]=categories.get(key,0)+1
        return {'format_version':1,'kind':'independent_cpu_cardinal_cycle_audit','run':str(self.run),
            'manifest_sha256':self.start_hashes['manifest.json'],'status_sha256':self.start_hashes['status.json'],
            'integrity_passed':True,'planned_cycle_complete':complete,'planned_comparisons_available':complete,
            'recorded_state':self.status['state'],'committed_fresh_authoring_draws':len(self.generations),
            'required_maximum_fresh_draws':56,'generation_classifications':categories,
            'evaluation_draws':sum(n.startswith('eval-') for n in self.generations),'required_evaluation_draws':24,
            'capability_screen_complete':self.capability is not None,
            'capability_passed':self.capability['passed'] if self.capability else None,
            'capability_productive_indices':self.capability['productive_indices'] if self.capability else [],
            'capability_distinct_productive_ir':self.capability['distinct_productive_ir'] if self.capability else None,
            'actual_recorded_game_occurrences':sum(len(r['matches']) for r in self.all_rows),
            'verified_distinct_replays':len(self.verified_replays),
            'actual_training_games':sum(len(r['matches']) for r in self.training_rows),
            'actual_evaluation_games':sum(r['played_games'] for r in self.evaluations.values()),
            'checkpoint_tensors':self.tensor_reports,'training_groups':self.group_reports,'evaluation':self.evaluations,
            'paired_yield_contrast':contrast,'external_spend_usd':0,
            'limits':['One training seed and twelve paired generated-controller clusters; individual games are correlated measurements.',
                'Prompt, cardinal interface, token cap and learning rate changed together; not an isolated interface ablation.',
                'Fixed productive base opponent pool, seen controller families on fresh maps; no population refresh or guarantee of continual improvement.',
                'Invalid authored draws have zero yield and no invented games; missing draws are not zeros.',
                'CPU audit verifies saved full-token likelihood and gradient-gate summaries, not a new independent neural likelihood computation.',
                'Packed model identity is bound to the frozen model-file index and executed runtime; only final-block q/v LoRA is trained.']}

    def execute(self):
        families=self.frozen()
        runtime_record=self.artifact('runtime.json',False)
        validate_execution_budget(self.config,self.status,runtime_record)
        if runtime_record is not None:
            previous.validate_runtime(runtime_record,self.model_index,self.manifest)
            parent=read(self.run/'inputs/parent-runtime.json')
            for key in ('model_index','versions','arithmetic','prefix_mode','memory_limit_bytes','eos_token_ids',
                        'trainable_tensor_names','trainable_parameters','packed_integer_tensors'):
                same(runtime_record[key],parent[key],'Executed parent/model identity differs: '+key)
        self.diagnostics()
        with archived_runtime(self.run,self.manifest) as runtime:
            prompt=(self.run/'prompts/cardinal.txt').read_text()
            require(prompt == runtime.model_spec() and self.manifest['prompt_sha256'] ==
                    {'cardinal':sha(self.run/'prompts/cardinal.txt')},'Clean compiler specification/prompt differs')
            self.maps(runtime,families);self.checkpoint('base')
            pool=self.capability_and_pool(runtime)
            self.groups(runtime,pool);self.all_generations(runtime);self.evaluation(runtime,pool)
        return self.finish()


def validate_recorded_gate(gate,records,names,config):
    """Failed attempts remain inspectable, but only a passed gate can back an update."""
    same(gate['generation_names'],names,'Gate generation order differs')
    require(len(gate['records']) == len(records),'Gate coverage differs')
    passes=[]
    for check,row in zip(gate['records'],records):
        require(check['tokens'] == len(row['completion_ids']),'Gate token coverage differs')
        same(check['recorded_sequence_logprob'],math.fsum(row['sampled_token_logprobs']),'Gate sampled likelihood differs')
        for key in ('max_token_error','sequence_error'):
            require(type(check[key]) in (int,float) and math.isfinite(check[key]) and check[key]>=0,'Nonfinite gate error')
        passed=check['max_token_error']<=config['token_logprob_tolerance'] and check['sequence_error']<=config['sequence_logprob_tolerance']
        require(check['passed'] is passed,'Gate outcome disagrees with measured errors');passes.append(passed)
    require(gate['passed'] is all(passes),'Aggregate gate outcome differs')


def audit(run):
    return Audit(run).execute()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=audit(args.run)
    text=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.output:
        if args.output.exists(): require(args.output.read_text()==text,'Refusing to overwrite an earlier audit')
        else:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            with args.output.open('x') as stream:stream.write(text)
    print(text)


if __name__ == '__main__':
    main()
