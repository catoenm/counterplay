"""Independent CPU audit of mixed-native controller RL; never loads a model.

Recorded neural vectors/gradients are checked as evidence. Original tokenization,
bounded game replays, serialized tensors, schedules and all denominators are
independently recomputed. Missing authoring is incomplete, never a zero outcome.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta
import importlib
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_program_debug_revision_calibration as calibration
from scripts import analyze_program_debug_rl as debug
from scripts import analyze_program_cardinal_gpu as gpu
from scripts import analyze_program_cardinal_cycle as cardinal
from scripts import analyze_program_study as previous

read, sha, same, require, safe_path = previous.read, previous.sha, previous.same, previous.require, previous.safe_path
value_sha = previous.value_sha
DESIGN_SHA256 = 'cb660f63a0b17450916350f4e65b57c52487924ba828ca93611410b523ba37e7'
ENDPOINTS = (10, 15, 25)
EVALUATION_RESERVE = 28800 + 1800 + 1800


def finite_positive(value, where):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0, where)
    return value


def paired_summary(base, final, seeds, bootstrap_draws=10000):
    """One cluster is one paired two-call program draw, not one replay game."""
    require(len(base) == len(final) == len(seeds) == len(set(seeds)) == 16,
            'All sixteen paired authoring trajectories are required')
    require(type(bootstrap_draws) is int and bootstrap_draws >= 100, 'Invalid bootstrap size')
    for rows in (base, final):
        require(len({r['generation'] for r in rows}) == 16, 'Repeated heldout trajectory')
        for row in rows:
            require(type(row['authorship_valid']) is bool and type(row['runtime_valid']) is bool,
                    'Missing authoring classification')
            value = row['banked_point_yield']
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                    'Invalid heldout banked yield')
            if row['authorship_valid']:
                require(row['games'] == 48, 'Compiled heldout controller lacks the fixed 48 games')
            else:
                require(row['games'] == 0 and value == 0 and not row['runtime_valid'],
                        'Invalid output fabricated games or yield')
    delta = [b['banked_point_yield'] - a['banked_point_yield'] for a, b in zip(base, final)]
    rng = random.Random(20261024)
    distribution = sorted(math.fsum(delta[rng.randrange(16)] for _ in range(16)) / 16
                          for _ in range(bootstrap_draws))
    return {'program_cluster_n': 16, 'paired_generation_seeds': seeds,
        'base_mean_yield': math.fsum(r['banked_point_yield'] for r in base) / 16,
        'final_mean_yield': math.fsum(r['banked_point_yield'] for r in final) / 16,
        'mean_yield_difference': math.fsum(delta) / 16, 'paired_yield_differences': delta,
        'descriptive_bootstrap_95_interval': [distribution[int(.025 * (bootstrap_draws - 1))],
                                             distribution[int(.975 * (bootstrap_draws - 1))]],
        'bootstrap_draws': bootstrap_draws, 'bootstrap_seed': 20261024,
        'sampling_unit': 'Paired two-call authoring trajectory; its turns and 48 games are dependent',
        'caution': 'One training seed; descriptive paired-generation bootstrap. Invalid authorship stays zero yield; missing authoring never does.'}


def select_endpoint(full_group_costs, remaining):
    """No reward or successful-update count enters the frozen timing rule."""
    require(len(full_group_costs) == 3, 'Exactly three full-priced profiling groups required')
    maximum = max(finite_positive(t, 'Invalid full-priced group cost') for t in full_group_costs)
    require(type(remaining) in (int, float) and math.isfinite(remaining) and remaining >= 0,
            'Invalid remaining budget')
    projections = {str(n): 1.5 * maximum * (n - 3) + EVALUATION_RESERVE for n in ENDPOINTS}
    eligible = [n for n in ENDPOINTS if projections[str(n)] <= remaining]
    return {'endpoint': max(eligible) if eligible else None,
            'maximum_full_group_seconds': maximum, 'projected_remaining_seconds': projections}


def validate_probe(report, turns, identity, config):
    require(len(turns) == 16, 'Full backward probe must cover all sixteen sampled turns')
    require(report['kind'] == 'revision_full_backward_probe' and report['expected_records'] == 16
            and report['optimizer_steps'] == 0 and report['training_use'] is False,
            'Full backward probe scope differs')
    same(report['token_tolerance'], config['token_logprob_tolerance'], 'Probe token tolerance differs')
    same(report['sequence_tolerance'], config['sequence_logprob_tolerance'], 'Probe sequence tolerance differs')
    restored = calibration.validate_restoration(report, identity, config)
    rows=report.get('records', [])
    require(isinstance(rows,list) and len(rows)<=16,'Probe contains extra backward rows')
    completed=[]
    for i,row in enumerate(rows):
        require(row['index']==i and row['record_sha256']==value_sha(turns[i]),'Probe row identity differs')
        require(row['coefficient']==1. and row['objective_divisor']==16,'Probe per-row coefficient differs')
        if row.get('backward_completed'):
            completed.append(row)
        else:
            require(i==len(rows)-1 and report['passed'] is False and isinstance(row.get('error_type'),str),
                    'Incomplete probe backward lacks a terminal diagnostic')
    normalized=copy.deepcopy(report); normalized['records']=completed; normalized['passed']=False
    vectors=calibration.validate_full_vectors(normalized,turns[:len(completed)],config)
    if 'sampled_tokens' in report:
        require(report['sampled_tokens']==sum(len(t['completion_ids']) for t in turns),'Probe omitted sampled tokens')
    # The exact probe coefficient/loss schema is fixed by the new training backend.
    coefficients = report['coefficients']
    same(coefficients, [1.] * 16, 'Probe coefficients were adapted to rewards')
    require(report['objective_divisor'] == 16, 'Probe normalization differs')
    losses = report.get('microbatch_losses', [])
    require(len(losses)==len(completed),'Probe loss/backward count differs')
    if losses:
        for loss, row in zip(losses, completed):
            require(type(loss) in (int, float) and math.isfinite(loss), 'Nonfinite probe loss')
            expected = -math.fsum(row['train_token_logprobs']) / 16
            require(abs(loss - expected) <= 2e-5 * max(1., abs(expected)), 'Probe objective differs')
        same(math.fsum(losses), report['loss'], 'Probe loss sum differs')
    else:
        require(report['passed'] is False, 'Successful probe dropped backward microbatches')
    derivative = report.get('derivative', {})
    norm = derivative.get('gradient_l2_norm')
    finite = (type(norm) in (int, float) and math.isfinite(norm) and norm > 0
              and derivative.get('missing_gradient_names') == []
              and derivative.get('nonfinite_gradient_names') == []
              and derivative.get('finite_nonzero') is True)
    if finite:
        validate_derivative(derivative,identity)
    passed = restored and vectors['passed'] and finite and len(losses) == 16
    require(report['passed'] is passed, 'Full backward probe outcome differs')
    return {'passed': passed, 'verified_turns': len(completed), 'verified_tokens': vectors['tokens'],
            'state_restored': restored, 'optimizer_steps': 0}


def validate_objective(update, trajectories, rewards, advantages, config):
    group = debug.validate_group(trajectories, rewards, advantages, update)
    if group['zero_signal']:
        require(update['outcome'] == 'skipped_zero_variance' and update['optimizer_steps'] == 0,
                'Zero-signal group claims an update')
        return group
    require(update['outcome'] == 'updated' and update['optimizer_steps'] == 1,
            'Nonzero-signal group lacks exactly one update')
    turns = [t for r in trajectories for t in r['turns']]
    gate = update['gradient_likelihood_gate']
    debug.validate_turn_gate(gate, turns, config, require_pass=True)
    same(update['coefficients'],[float(a) for a in advantages for _ in range(2)],'Actual per-turn advantages differ')
    for i,(row,turn) in enumerate(zip(gate['records'],turns)):
        require(row['index']==i and row['record_sha256']==value_sha(turn) and row['seed']==turn['seed']
            and row['authoring_stage']==turn['authoring_stage'] and row['train_grad_enabled'] is True
            and row['backward_completed'] is True,'Actual backward row provenance differs')
        same(row['sampled_token_logprobs'],turn['sampled_token_logprobs'],'Actual gradient sampling likelihood differs')
        check=calibration.vector_check(turn['sampled_token_logprobs'],row['train_token_logprobs'],config)
        same(row['train_check'],check,'Full actual gradient-forward vector differs')
        require(check['passed'] and row['train_all_finite'] is True,'Actual backward likelihood failed')
        same(row['coefficient'],advantages[i//2],'Backward coefficient differs')
        require(row['objective_divisor']==8,'Backward divisor differs')
    losses = update['microbatch_losses']
    require(len(losses) == 16 and all(type(x) in (int, float) and math.isfinite(x) for x in losses),
            'Actual update dropped or invalidated per-turn losses')
    for i, (loss, row) in enumerate(zip(losses, gate['records'])):
        expected = -advantages[i // 2] * row['recorded_sequence_logprob'] / 8
        tolerance = abs(advantages[i // 2]) * row['sequence_error'] / 8 + 2e-5 * max(1., abs(expected))
        require(abs(loss - expected) <= tolerance,
                'Actual RLOO objective is not the sum of both sampled turns divided by eight trajectories')
    same(math.fsum(losses), update['loss'], 'Accumulated RLOO loss differs')
    finite_positive(update['gradient_norm_before_clip'], 'Actual gradient norm is not finite and positive')
    require(update['adam_step_after'] == update['adam_step_before'] + 1, 'Actual Adam did not advance once')
    return group


def validate_tokenization(record, tokenizer, eos, *, smoke=False):
    require(record.get('calibration_only') is smoke, 'Sample calibration/training scope differs')
    if not smoke: require(record.get('training_capable') is True, 'RL sample training capability missing')
    # Reuse every native template/token/decode check, replacing only its old
    # calibration-scope assertion after checking the new scope above.
    calibration.validate_tokenization({**record, 'calibration_only': True}, tokenizer, eos)


validate_turn = calibration.validate_turn
validate_full_vectors = calibration.validate_full_vectors
validate_restoration = calibration.validate_restoration
validate_cache_timing = calibration.validate_cache_timing


class Audit(debug.Audit):
    checkpoint = debug.Audit.checkpoint
    scored_row = calibration.Audit.scored_row

    def all_batches(self):
            initial=self.runtime.debug.initial_messages()
            for batchid,names,seeds,label,maps in self.batch_specs():
                for turn in ('draft','revision'):
                    path=f'batches/{batchid}-{turn}.json';intentpath=f'batches/{batchid}-{turn}-intent.json'
                    batch=self.artifact(path,False);intent=self.artifact(intentpath,False)
                    if batch is None and intent is None:continue
                    require(self.checkpoint(label) is not None,'Sampling before committed source checkpoint')
                    if batchid.startswith('eval-'):require(self.checkpoint(f'cycle{self.endpoint}') is not None,'Evaluation before fixed endpoint')
                    if turn=='draft':contexts=[initial]*2;turnseeds=seeds
                    else:
                        draftpath=f'batches/{batchid}-draft.json';drafts=self.artifact(draftpath)
                        contexts=[]
                        for i,draft in enumerate(drafts['records']):
                            feedback=self.feedback(f'feedback/{names[i]}.json',draft,maps[i],draftpath,i)
                            contexts.append(self.runtime.debug.revision_messages(initial,draft,feedback))
                        turnseeds=[s+1 for s in seeds]
                    identity={'checkpoint':label,'source_metadata_sha256':self.artifacts[f'checkpoints/{label}/metadata.json'],
                              'contexts_sha256':value_sha(contexts),'seeds':turnseeds,'authoring_stage':turn}
                    require(intent is not None,'Sampled batch lacks durable no-redraw intent')
                    require(isinstance(intent['started'],str),'Intent timestamp missing')
                    for key,value in identity.items():same(intent[key],value,'Intent differs: '+key)
                    if batch is None:continue
                    for key,value in identity.items():same(batch[key],value,'Raw batch identity differs: '+key)
                    require(len(batch['records'])==2,'Partial committed sampling batch')
                    width=max(len(row['prompt_ids']) for row in batch['records'])
                    for i,row in enumerate(batch['records']):
                        validate_turn(row,self.config,stage=turn,seed=turnseeds[i],messages=contexts[i])
                        validate_tokenization(row,self.tokenizer,self.eos)
                        self.actual_context(path+'/row-'+str(i),row)
                        require(row['completion_token_limit']==4096,'Full calibration token cap differs')
                        require(row['sampling_batch_row']==i and row['sampling_batch_size']==2
                            and row['left_padding_tokens']==width-len(row['prompt_ids']),'Batch row/padding differs')
                        self.raw_turns[(names[i],turn)]=path
                    timing=batch['timing']
                    require(type(timing['batch_elapsed_seconds']) in (float,int) and math.isfinite(timing['batch_elapsed_seconds'])
                            and timing['batch_elapsed_seconds']>0,'Invalid batch wall time')
                    require(timing['total_output_tokens']==sum(len(r['completion_ids']) for r in batch['records'])
                            and timing['constant_cache_batch_size']==2 and timing['active_sequences']==2,'Batch usage differs')
                    counts=timing['active_sequences_by_step'];lengths=[len(r['completion_ids']) for r in batch['records']]
                    same(counts,[sum(length>i for length in lengths) for i in range(max(lengths))],'Decode live-row coverage differs')
                    require(timing['batch_steps']==len(counts) and timing['finished_eos']==sum(r['finish_reason']=='eos' for r in batch['records']),
                            'Decode termination totals differ')
                    same(timing['authoring_stages'],[turn,turn],'Batch native stage differs')
                    for row in batch['records']:
                        expected=len(counts)-len(row['completion_ids']) if row['finish_reason']=='eos' else 0
                        require(type(row['post_eos_cache_forwards']) is int and row['post_eos_cache_forwards']==expected,
                                'Post-EOS fixed-cache continuation count differs')
            # A crash may leave complete draft feedback before the revision batch intent.
            expected={name:(batchid,i,maps[i]) for batchid,names,seeds,label,maps in self.batch_specs() for i,name in enumerate(names)}
            for path in self.artifacts:
                if path.startswith('feedback/'):
                    name=Path(path).stem;require(name in expected,'Unregistered practice feedback')
                    batchid,index,seeds=expected[name];draftpath=f'batches/{batchid}-draft.json'
                    draft=self.artifact(draftpath)['records'][index]
                    self.feedback(path,draft,seeds,draftpath,index)

    def frozen(self):
        m,cfg=self.manifest,self.config
        require(m['format_version']==1 and m['kind']=='native_nonthinking_revision_rl','Unknown RL protocol')
        require(self.start_hashes['manifest.json']==self.status['manifest_sha256'],'Manifest differs')
        for p,d in {**m['input_sha256'],**self.artifacts}.items():require(sha(safe_path(self.run,p))==d,'Artifact differs: '+p)
        for p,d in m['source_sha256'].items():require(sha(safe_path(self.run/'source',p))==d,'Archived source differs: '+p)
        helpers=('scripts/analyze_program_study.py','scripts/analyze_program_cardinal_cycle.py',
            'scripts/analyze_program_cardinal_gpu.py','scripts/analyze_program_debug_rl.py',
            'scripts/analyze_program_debug_revision_calibration.py','scripts/analyze_program_debug_revision_rl.py',
            'scripts/cloud/train_program_rl.py','selfplay/program_cardinal_cuda.py')
        for p in helpers:require(m['source_sha256'].get(p)==sha(ROOT/p),'Executed helper differs from archive: '+p)
        require({'selfplay/program_debug_revision_rl.py','selfplay/program_debug_revision_train.py',
            'selfplay/program_debug_revision_harness.py','selfplay/program_debug_revision_cuda.py'}<=set(m['source_sha256']),
            'New RL source closure incomplete')
        same(self.plan['config'],cfg,'Plan config differs');same(self.plan['schedule'],self.schedule,'Plan schedule differs')
        same(self.schedule,registered_schedule(),'Registered authoring/map schedule differs')
        constants={'kind':'native_nonthinking_revision_rl','model_repo':'Qwen/Qwen3.8-27B',
            'model_revision':'1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0','backend':'cuda','loader':'image_text_to_text',
            'condition':'mixed-native-revision-rl-v1','algorithm':'full_completion_RLOO','seed':425401,
            'thinking':True,'reasoning_effort':'low','group_size':8,'batch_size':2,'iterations':25,
            'endpoint_choices':[10,15,25],'profile_attempt_cap':3,'authoring_calls':2,
            'practice_maps_per_trajectory':2,'eval_samples':16,'capability_samples':0,'opponent_attempt_cap':0,
            'maximum_model_opponents':2,'map_seeds_per_group':2,'league_refresh_every':5,
            'learning_rate':1e-5,'max_new_tokens':4096,'max_context_tokens':16384,'sample_seconds':900,
            'worker_seconds':86400,'max_rounds':48,'temperature':1.,'top_p':1.,'top_k':0,
            'compute_dtype':'float32','attention':'eager','lora_rank':8,'lora_alpha':16,'lora_dropout':0.,
            'lora_targets':['q_proj','v_proj'],'gradient_clip':.5,'token_logprob_tolerance':.002,
            'sequence_logprob_tolerance':.02,'timing_safety_factor':1.5,
            'evaluation_generation_reserve_seconds':28800,'evaluation_cpu_reserve_seconds':1800,
            'audit_export_reserve_seconds':1800,'evaluation_total_reserve_seconds':32400,
            'full_update_checkpoint_allowance_seconds':120}
        for k,v in constants.items():same(cfg[k],v,'Registered setting differs: '+k)
        modes={s:{'authoring_stage':s,'thinking_enabled':s=='draft','reasoning_effort':'low' if s=='draft' else None,
                  'reasoning_mode':'native_low' if s=='draft' else 'native_disabled'} for s in ('draft','revision')}
        same(cfg['turn_native_modes'],modes,'Native modes changed')
        dep=read(self.run/'deployment.json')
        require(dep['project']=='cardinal-gpu' and dep['maximum_hours']==25 and dep['reservation_usd']==145
            and 0<dep['maximum_gpu_rate_usd_hour']<=5.29 and 0<dep['maximum_cpu_rate_usd_hour']<=.14,
            'Resource envelope differs')
        require(datetime.fromisoformat(cfg['deadline']).timestamp()==dep['deadline_unix'],'Immutable deadline differs')
        budget=self.status.get('execution_budget')
        if budget:
            first,stop=datetime.fromisoformat(budget['first_started']),datetime.fromisoformat(budget['hard_stop'])
            require(stop==min(first+timedelta(seconds=86400),datetime.fromisoformat(cfg['deadline']))
                and budget['maximum_seconds']==86400 and budget['resume_counts_wall_time'] is True,'Persistent worker budget differs')
            if self.status['state']=='complete':require(datetime.fromisoformat(self.status['updated'])<=stop,'Completed after deadline')
        else:require(self.status['state']=='prepared','Execution budget missing')
        require(sha(self.run/'inputs/design-plan.json')==DESIGN_SHA256==self.plan['design_sha256'],'Accepted design differs')
        acceptance=read(self.run/'inputs/root-acceptance.json')
        require(acceptance['root_accepted'] is True and acceptance['design_sha256']==DESIGN_SHA256
            and sha(self.run/'inputs/root-acceptance.json')==self.plan['root_acceptance_sha256'],'Root acceptance differs')
        accounting=read(self.run/'inputs/accounting-review.json')
        parent=read(self.run/'inputs/initial-manifest.json');parent_status=read(self.run/'inputs/initial-status.json')
        parent_audit=read(self.run/'inputs/initial-final-audit.json')
        require(parent_audit['integrity_passed'] is True and parent_audit['genuine_supervised_optimizer_steps']==32
            and parent_audit['genuine_RL_optimizer_steps']==0 and parent_audit['optimizer_reset_verified'] is True,
            'Initial SFT32/zero-RL baseline lacks audited provenance')
        require(parent_audit['manifest_sha256']==sha(self.run/'inputs/initial-manifest.json')==parent_status['manifest_sha256']
            and parent_audit['status_sha256']==sha(self.run/'inputs/initial-status.json'),'Initial audit binding differs')
        prior=read(self.run/'inputs/calibration-manifest.json');prior_status=read(self.run/'inputs/calibration-status.json')
        prior_audit=read(self.run/'inputs/calibration-final-audit.json')
        require(prior_audit['integrity_passed'] is True and prior_audit['calibration_complete'] is True
            and prior_audit['qualification_passed'] is True and prior_status['state']=='complete'
            and prior_status['qualification_passed'] is True and prior_status['actual_optimizer_steps']==0
            and prior_audit['genuine_RL_optimizer_steps']==prior_audit['new_supervised_optimizer_steps']==0,
            'Prior calibration did not pass unchanged')
        require(prior_audit['manifest_sha256']==sha(self.run/'inputs/calibration-manifest.json')==prior_status['manifest_sha256']
            and prior_audit['status_sha256']==sha(self.run/'inputs/calibration-status.json'),'Passing calibration audit binding differs')
        require(prior_audit['full_likelihood_admission']['passed'] is True
            and prior_audit['full_likelihood_admission']['verified_turns']==16,'Prior full-vector admission missing')
        for name in ('scores','decision','sealed','full-likelihood-gate'):
            require(sha(self.run/f'inputs/calibration-{name}.json')==prior_status['artifacts'][f'calibration/{name}.json'],
                'Calibration evidence differs: '+name)
        require(sha(self.run/'inputs/calibration-map-fingerprints.json')==prior['input_sha256']['map-fingerprints.json'],
            'Passing calibration maps differ')
        for p,d in prior['input_sha256'].items():
            if p.startswith('inputs/') and p not in ('inputs/design-plan.json','inputs/design-plan-v1.json'):
                require(m['input_sha256'].get(p)==d,'Earlier input chain changed: '+p)
        require(sha(self.run/'inputs/teacher-corpus.json')==parent['input_sha256']['inputs/teacher/corpus.json']
            ==parent_audit['teacher_corpus_sha256'],'Teacher anchor corpus differs')
        metadata=read(self.run/'inputs/initial-checkpoint/metadata.json')
        require(sha(self.run/'inputs/initial-checkpoint/metadata.json')==parent_status['artifacts']['checkpoints/base/metadata.json'],
            'Initial metadata changed')
        for filename,d in metadata['files'].items():require(sha(self.run/'inputs/initial-checkpoint'/filename)==d
            ==parent_status['artifacts']['checkpoints/base/'+filename],'Initial tensor bytes differ')
        self.cuda_inspector=importlib.import_module('selfplay.program_cardinal_cuda').checkpoint_state
        identity=self.cuda_inspector(self.run/'inputs/initial-checkpoint',expected_learning_rate=1e-5)
        for expected in (metadata['optimizer_identity'],m['initial_checkpoint_identity'],prior['initial_checkpoint_identity']):
            same(identity,expected,'Initial actual Adam/adapter/RNG differs')
        same(m['initial_checkpoint_files'],metadata['files'],'Initial file declaration differs')
        require(identity['step']==0 and not identity['moments'],'Initial optimizer is not zero-step Adam')
        self.tokenizer,self.eos=debug.load_original_tokenizer(self.run,m)
        info=read(self.run/'inputs/model-info.json')
        same((info['id'],info['sha']),(cfg['model_repo'],cfg['model_revision']),'Pinned model differs')
        self.prior_manifest,self.prior_status,self.prior_audit=prior,prior_status,prior_audit
        return {k:[s for row in v.values() for s in row] if isinstance(v,dict) else v
            for k,v in self.schedule.items() if k in ('train_practice_maps','train_maps','eval_practice_maps','holdout_maps')}

    def maps(self,runtime,families):
        prior={key:read(self.run/path) for key,path in {
            'warmstart':'inputs/initial-map-fingerprints.json','debug':'inputs/prior-debug-map-fingerprints.json',
            'low_calibration':'inputs/previous-calibration-map-fingerprints.json',
            'mixed_calibration':'inputs/calibration-map-fingerprints.json'}.items()}
        document=read(self.run/'map-fingerprints.json')
        def layout(seed):
            g=runtime.game(seed,48)
            return {'grid_size':g.size,'capacity':g.capacity,'max_energy':g.max_energy,'max_rounds':g.max_rounds,
                'sites':g.sites,'homes':[[r['home_x'],r['home_y']] for r in g.robots]}
        expected={k:[{'seed':s,'sha256':value_sha(layout(s)),'layout':layout(s)} for s in seeds] for k,seeds in families.items()}
        same(document['families'],expected,'Observable map differs')
        require(document['prior_evidence_sha256']==value_sha(prior),'Prior map evidence chain differs')
        historical=[]
        def collect(value):
            if isinstance(value,dict):
                if {'seed','layout','sha256'}<=set(value):
                    require(value_sha(layout(value['seed']))==value['sha256'],'Historical map changed');historical.append(value['sha256'])
                for s,d in value.get('prior_seed_fingerprints',{}).items():
                    require(value_sha(layout(int(s)))==d,'Historical layout changed');historical.append(d)
                for child in value.values():collect(child)
            elif isinstance(value,list):
                for child in value:collect(child)
        collect(prior)
        previous.check_fingerprint_overlap({k:[r['sha256'] for r in v] for k,v in expected.items()},historical)

    def batch_specs(self):
        for n in range(1,26):
            source='base' if n==1 else f'cycle{n-1}'
            for offset in range(0,8,2):
                yield f'train-{n}-{offset//2}',[f'train-{n}-{i}' for i in range(offset,offset+2)], \
                    self.schedule['train_generation_seeds'][str(n)][offset:offset+2],source, \
                    [self.schedule['train_practice_maps'][str(n)][i*2:i*2+2] for i in range(offset,offset+2)]
        if self.endpoint:
            for label in ('base',f'cycle{self.endpoint}'):
                for offset in range(0,16,2):
                    yield f'eval-{label}-{offset//2}',[f'eval-{label}-{i}' for i in range(offset,offset+2)], \
                        self.schedule['eval_generation_seeds'][offset:offset+2],label, \
                        [self.schedule['eval_practice_maps'][str(i)] for i in range(offset,offset+2)]

    def generation(self,name,seed,label,condition='native_nonthinking_revision_rl'):
        if name in self.generations:return self.generations[name]
        row=self.artifact(f'generations/{name}.json');meta=self.checkpoint(label)
        require(meta is not None and row['source_metadata_sha256']==self.artifacts[f'checkpoints/{label}/metadata.json'],'Trajectory source differs')
        specs={n:(batch,i,maps[i]) for batch,names,seeds,checkpoint,maps in self.batch_specs() for i,n in enumerate(names)}
        require(name in specs,'Unregistered trajectory');batch,index,maps=specs[name]
        require(row['batch_index']==index and row['draft_batch_path']==f'batches/{batch}-draft.json'
            and row['revision_batch_path']==f'batches/{batch}-revision.json','Trajectory batch/index differs')
        drafts=self.artifact(row['draft_batch_path']);revisions=self.artifact(row['revision_batch_path'])
        for stage in ('draft','revision'):
            require(row[stage+'_batch_sha256']==self.artifacts[row[stage+'_batch_path']],'Trajectory raw batch hash differs')
        same(row['turns'],[drafts['records'][index],revisions['records'][index]],'Trajectory substituted original output')
        require(row['feedback_path']==f'feedback/{name}.json' and row['feedback_sha256']==self.artifacts[row['feedback_path']],
            'Trajectory feedback binding differs')
        feedback=self.feedback(row['feedback_path'],row['turns'][0],maps,row['draft_batch_path'],index)
        initial=self.runtime.debug.initial_messages();revision=self.runtime.debug.revision_messages(initial,row['turns'][0],feedback)
        require(row['prompt_condition']==condition,'Trajectory RL condition differs')
        calibration.validate_trajectory({**row,'prompt_condition':'native_nonthinking_revision_calibration'},
            config=self.config,name=name,seed=seed,checkpoint=label,initial_messages=initial,feedback=feedback,
            revision_messages=revision,practice_seeds=maps)
        same(row['history_transform'],self.runtime.debug.REVISION_CONTEXT_POLICY,'History transform differs')
        self.generations[name]=row;return row

    all_generations=calibration.Audit.all_generations
    def initial_pool(self,runtime):
        scores=read(self.run/'inputs/calibration-scores.json');decision=read(self.run/'inputs/calibration-decision.json')
        calculated=cardinal.capability_decision(scores['rows'])
        require(calculated['passed'] is True,'Inherited capability gate does not pass')
        same(decision['selected_indices'],calculated['selected_indices'],'Initial controller selection differs')
        same(decision['productive_indices'],calculated['productive_indices'],'Inherited productive indices differ')
        require(decision['passed'] is True and decision['scores_sha256']==sha(self.run/'inputs/calibration-scores.json'),
            'Inherited capability score binding differs')
        anchors=[{'id':n,'origin':'scripted','authorship':'scripted_anchor','compiled_ir':runtime.reference_programs()[n]}
                 for n in ('nearest','value')]
        models=[]
        for i in calculated['selected_indices']:
            row=scores['rows'][i];name=f'calibration-{i}';path=f'inputs/calibration-generations/{name}.json'
            generation=read(self.run/path)
            require(sha(self.run/path)==row['generation_sha256']==self.prior_status['artifacts'][f'generations/{name}.json'],
                'Inherited controller generation differs')
            self.scored_row(row,generation,anchors,scores['seeds'],runtime)
            program=row['classification']['program'];ir=runtime.compile_ir(program['compiled_ir']).digest
            same(program['ir_sha256'],ir,'Inherited controller IR differs')
            models.append({'id':name,'origin':'model','authorship':'inherited_SFT32_controller',
                'source_checkpoint':'SFT32-base','compiled_ir':program['compiled_ir'],'ir_sha256':ir,
                'generation_sha256':row['generation_sha256'],'source_path':path})
        require(len(models)==2 and len({r['ir_sha256'] for r in models})==2,'Initial training controllers are not distinct')
        corpus=read(self.run/'inputs/teacher-corpus.json');seen={r['ir_sha256'] for r in models};held=[]
        for demo in sorted(corpus['demonstrations'],key=lambda r:r['index']):
            program=demo['program'];ir=runtime.compile_ir(program['compiled_ir']).digest
            same(program['ir_sha256'],ir,'Teacher anchor IR differs')
            if ir in seen:continue
            seen.add(ir);held.append({'id':'teacher-'+str(demo['index']),'origin':'model',
                'authorship':'external_teacher_off_policy_anchor','was_prior_SFT_example':True,
                'source_checkpoint':'external_teacher','compiled_ir':program['compiled_ir'],'ir_sha256':ir,
                'generation_sha256':demo['generation_sha256'],'source_path':'inputs/teacher-corpus.json'})
            if len(held)==2:break
        require([r['id'] for r in held]==['teacher-0','teacher-1'],'Predetermined evaluation anchors changed')
        expected={'selected':anchors+models,'frozen':True}
        evaluation={'selected':anchors+held,'frozen':True,'teacher_anchors_reserved_from_RL_roster_not_unseen_to_weights':True}
        same(read(self.run/'initial-opponents.json'),expected,'Initial RL roster differs')
        same(read(self.run/'evaluation-opponents.json'),evaluation,'Fixed evaluation roster differs')
        self.pool=self.artifact('opponents.json',False)
        self.heldout_pool=self.artifact('heldout-opponents.json',False)
        if self.pool is not None:same(self.pool,expected,'Committed training roster differs')
        if self.heldout_pool is not None:same(self.heldout_pool,evaluation,'Committed holdout roster differs')
        return self.pool
    def diagnostics(self):
            runtime=self.artifact('runtime.json',False)
            if runtime:
                require(runtime['cuda']=='12.8' and runtime['loader']=='image_text_to_text','CUDA/runtime loader differs')
                require(runtime['arithmetic']=='NF4 frozen base; FP32 compute/LoRA; TF32 off; eager attention','Recorded arithmetic differs')
                same(runtime['turn_native_modes'],self.config['turn_native_modes'],'Runtime native modes differ')
                require(runtime['calibration_only'] is False and runtime['training_capable'] is True and runtime['authoring_calls']==2,'Runtime sampling scope differs')
                schema=runtime['parameter_schema']
                same([r['name'] for r in schema],runtime['trainable_names'],'Runtime parameter order differs')
                require(all(r['dtype']=='torch.float32' and r['numel']==math.prod(r['shape']) and r['numel']>0 for r in schema)
                    and sum(r['numel'] for r in schema)==runtime['trainable_parameters'],'Runtime trainable schema/count differs')
                require(not set(runtime['trainable_names'])&set(runtime['frozen_state']['tensors']),'Frozen/trainable identities overlap')
                for name,version in self.manifest['expected_installed_versions'].items():
                    require(runtime['versions'][name].split('+')[0]==version,'Installed version differs: '+name)
                index=runtime['model_index'];require(index['repo']==self.config['model_repo'] and index['revision']==self.config['model_revision']
                    and index['authoritative_metadata_sha256']==self.manifest['input_sha256']['inputs/model-info.json'],'Model provenance differs')
                info=read(self.run/'inputs/model-info.json')
                for row in info['siblings']:
                    if row['rfilename'].endswith('.safetensors'):require(index['files'].get(row['rfilename'])==row['lfs']['sha256'],'Pinned model weight hash differs')
                frozen=runtime['frozen_state'];require(previous.value_sha(frozen['tensors'])==frozen['sha256'],'Frozen tensor fingerprint digest differs')
                quant={n:v for n,v in frozen['tensors'].items() if n in frozen['quantization_state_tensors']}
                require(previous.value_sha(quant)==frozen['quantization_state_sha256'],'Quantization fingerprint differs')
            passed=False
            for path in self.artifacts:
                if not path.startswith('smoke-'):continue
                smoke=self.artifact(path)
                require(smoke['training_use'] is False and smoke['optimizer_steps']==0 and smoke['source_checkpoint']=='base'
                    and smoke['source_metadata_sha256']==self.artifacts['checkpoints/base/metadata.json'],'Smoke provenance differs')
                require(smoke['kind']=='revision_mixed_native_derivative_admission' and smoke['batch_size']==2
                    and smoke['max_new_tokens']==32 and smoke['authoring_stages']==['draft','revision'],'Mixed native smoke scope differs')
                same(smoke['seeds'],[70000000+self.config['seed']+i for i in range(2)],'Smoke sampling seeds differ')
                same(smoke['token_tolerance'],self.config['token_logprob_tolerance'],'Smoke token tolerance differs')
                same(smoke['sequence_tolerance'],self.config['sequence_logprob_tolerance'],'Smoke sequence tolerance differs')
                base=self.checkpoint('base');require(base is not None,'Smoke precedes base checkpoint')
                restored=validate_restoration(smoke,base['optimizer_identity'],self.config)
                vectors=False
                if 'batch' in smoke:
                    records=smoke['batch']['records']
                    require(len(records)==2 and len({r['input_tokens'] for r in records})>1,'Smoke lacks mixed-length rows')
                    validate_cache_timing(smoke['batch'],['draft','revision'])
                    for i,record in enumerate(records):
                        validate_turn(record,self.config,stage=('draft','revision')[i],seed=smoke['seeds'][i],maximum=32)
                        validate_tokenization(record,self.tokenizer,self.eos,smoke=True)
                        require(record['completion_token_limit']==32,'Smoke token cap differs')
                        self.actual_context(f'{path}/row-{i}',record)
                    if 'records' in smoke:vectors=validate_full_vectors(smoke,records,self.config)['passed']
                derivative=smoke.get('derivative',{})
                norm=derivative.get('gradient_l2_norm')
                finite=(type(norm) in (int,float) and math.isfinite(norm) and norm>0
                        and derivative.get('missing_gradient_names')==[] and derivative.get('finite_nonzero') is True)
                if smoke['passed']:
                    require(restored and vectors and finite,'Smoke lacks full vectors, finite derivative or state restoration')
                    passed=True
                else:require(not (restored and vectors and finite),'Successful smoke mislabeled as failure')
            if any(p.startswith(('generations/','batches/')) for p in self.artifacts):
                require(passed,'Authoring began without successful mixed-native numerical admission')

    def numerical(self,folder,name,seal,records,source):
        path=f'{folder}/{name}.json';report=self.artifact(path,False);intent=self.artifact(f'{folder}/{name}-intent.json',False)
        if report is None and intent is None:return None
        binding={**seal,'seal_sha256':self.artifacts[folder+'/seal.json']}
        require(intent is not None and isinstance(intent['started'],str),'Numerical gate lacks durable intent')
        for k,v in binding.items():same(intent[k],v,'Numerical intent differs: '+k)
        if report is None:return {'passed':False,'interrupted_intent':True}
        for k,v in binding.items():same(report[k],v,'Numerical report binding differs: '+k)
        finite_positive(report['elapsed_seconds'],'Numerical elapsed time missing')
        identity=self.checkpoint(source)['optimizer_identity']
        if name=='probe':return validate_probe(report,records,identity,self.config)
        require(report['kind']=='revision_full_completion_likelihood_admission' and report['expected_records']==16
            and report['optimizer_steps']==0 and report['training_use'] is False,'Full likelihood scope differs')
        same(report['token_tolerance'],.002,'Gate token tolerance differs');same(report['sequence_tolerance'],.02,'Gate sequence tolerance differs')
        restored=validate_restoration(report,identity,self.config)
        vectors=validate_full_vectors(report,records,self.config)
        require(report['sampled_tokens']==vectors['tokens'] and report['passed'] is (restored and vectors['passed']),
            'Full likelihood aggregate differs')
        return {'passed':report['passed'],'verified_turns':vectors['turns'],'verified_tokens':vectors['tokens'],'state_restored':restored}

    def groups(self,runtime,pool):
        current=pool['selected'] if pool else None;last=0;self.probe=None;self.timings={}
        reserved={r['ir_sha256'] for r in self.heldout_pool['selected'] if r['origin']=='model'} if self.heldout_pool else set()
        for n in range(1,26):
            folder=f'groups/{n}';source='base' if n==1 else f'cycle{n-1}'
            start=self.artifact(folder+'/timing-start.json',False)
            opponents=self.artifact(folder+'/opponents.json',False)
            seal=self.artifact(folder+'/seal.json',False);scores=self.artifact(folder+'/scores.json',False)
            meta=self.checkpoint(f'cycle{n}')
            if start is None and opponents is None:
                require(seal is None and scores is None and meta is None,'Group lacks durable start/roster');continue
            require(start and start['cycle']==n and n==last+1,'Group skipped a source checkpoint')
            if opponents is None:
                require(seal is None and scores is None and meta is None,'Group lacks opponent snapshot');continue
            require(current is not None and self.heldout_pool is not None,'Group has no immutable opponent pools')
            same(opponents['selected'],current,'League snapshot differs')
            population=f'league/after-{(n-1)//5*5}.json' if n>5 else 'opponents.json'
            require(opponents['cycle']==n and opponents['population_sha256']==self.artifacts[population]
                and opponents['frozen_before_generation'] is True,'Population source binding differs')
            model_irs=[r['ir_sha256'] for r in current if r['origin']=='model']
            require(len(model_irs)==len(set(model_irs))==2 and not set(model_irs)&reserved,'Training/evaluation model roster overlaps')
            if seal is None:
                require(scores is None and meta is None,'Scores before complete group seal');continue
            records=[self.generation(f'train-{n}-{i}',self.schedule['train_generation_seeds'][str(n)][i],source) for i in range(8)]
            flat=[t for r in records for t in r['turns']]
            hashes={f"generations/{r['name']}.json":self.artifacts[f"generations/{r['name']}.json"] for r in records}
            expected={'cycle':n,'source_checkpoint':source,'source_metadata_sha256':self.artifacts[f'checkpoints/{source}/metadata.json'],
                'generation_sha256':hashes,'turn_sha256':[value_sha(t) for t in flat],
                'opponents_sha256':self.artifacts[folder+'/opponents.json'],'draws':8,'sampled_turns':16}
            same(seal,expected,'Group seal differs')
            gate=self.numerical(folder,'gate',seal,flat,source)
            if n==1:self.probe=self.numerical(folder,'probe',seal,flat,source)
            if scores is None:
                require(meta is None,'Checkpoint precedes scored rollouts');continue
            require(gate and gate['passed'] and self.probe and self.probe['passed'],'Scoring before full forward/backward admission')
            for k,v in {'cycle':n,'source_checkpoint':source,'source_metadata_sha256':seal['source_metadata_sha256'],
                'seal_sha256':self.artifacts[folder+'/seal.json'],'opponents_sha256':seal['opponents_sha256'],
                'gate_sha256':self.artifacts[folder+'/gate.json'],'probe_sha256':self.artifacts['groups/1/probe.json']}.items():
                same(scores[k],v,'Scoring lineage differs: '+k)
            same(scores['seeds'],self.schedule['train_maps'][str(n)],'Training maps differ')
            require(len(scores['rows'])==8,'Training dropped invalid authorship')
            for row,record in zip(scores['rows'],records):
                require(row['generation']==record['name'] and row['generation_sha256']==hashes[f"generations/{record['name']}.json"],
                    'Scored generation identity differs')
                self.scored_row(row,record,current,scores['seeds'],runtime)
            rewards=[r['reward'] for r in scores['rows']];advantages=previous.advantages(rewards);zero=not any(advantages)
            same(scores['rewards'],rewards,'Rewards differ');same(scores['advantages'],advantages,'LOO advantages differ')
            require(scores['zero_signal'] is zero,'Zero-signal flag differs')
            self.training_rows+=scores['rows']
            self.group_reports.append({'cycle':n,'checkpoint_committed':meta is not None,'rewards':rewards,'advantages':advantages,
                **previous.summarize_programs(scores['rows'],8)})
            intent=self.artifact(folder+'/optimizer-intent.json',False);result=self.artifact(folder+'/optimizer-result.json',False)
            skip=self.artifact(folder+'/skip-result.json',False)
            if intent:
                for k,v in {'source_checkpoint':source,'source_metadata_sha256':seal['source_metadata_sha256'],
                    'scores_sha256':self.artifacts[folder+'/scores.json'],'gate_sha256':self.artifacts[folder+'/gate.json'],
                    'probe_sha256':self.artifacts['groups/1/probe.json']}.items():same(intent[k],v,'Optimizer intent differs: '+k)
                require(not zero and isinstance(intent['started'],str),'Optimizer intent on a zero-signal group')
            if result is not None:require(intent is not None,'Optimizer result lacks no-retry intent')
            if skip is not None:require(zero and result is None and intent is None,'Nonzero or ambiguous skip evidence')
            if meta is None:continue
            update=meta['report'];last=n
            expected_update={'cycle':n,'source_checkpoint':source,'source_metadata_sha256':seal['source_metadata_sha256'],
                'scores_sha256':self.artifacts[folder+'/scores.json'],'generation_sha256':{r['name']:hashes[f"generations/{r['name']}.json"] for r in records},
                'rewards':rewards,'advantages':advantages,'objective_divisor':8,'trajectory_count':8,'sampled_turn_count':16,
                'training_tokens':sum(r['training_tokens'] for r in records),'tool_feedback_target_tokens':0,
                'likelihood_gate_path':folder+'/gate.json','likelihood_gate_sha256':self.artifacts[folder+'/gate.json'],
                'full_backward_probe_sha256':self.artifacts['groups/1/probe.json']}
            for k,v in expected_update.items():same(update[k],v,'Update lineage differs: '+k)
            same(update['trajectory_turn_sha256'],{r['name']:r['turn_sha256'] for r in records},'Trajectory turn lineage differs')
            same(update['turn_sha256'],[value_sha(t) for t in flat],'Ordered turn lineage differs')
            validate_objective(update,records,rewards,advantages,self.config)
            backend_result=skip if zero else result
            result_path=folder+('/skip-result.json' if zero else '/optimizer-result.json')
            require(backend_result is not None and update['backend_result_path']==result_path
                and update['backend_result_sha256']==self.artifacts[result_path],'Backend result hash differs')
            for k,v in backend_result.items():same(update[k],v,'Committed backend result differs: '+k)
            validate_checkpoint_update(backend_result,self.checkpoints[source]['optimizer_identity'],meta['optimizer_identity'],self.config)
            if zero:
                require(intent is None and result is None,'Skipped group sampled an optimizer update')
                require(update['adam_step_before']==update['adam_step_after']==meta['optimizer_identity']['step'],
                    'Skipped Adam transition differs')
            else:
                require(result is not None and intent is not None,'Actual update lacks durable result/intent')
            timer=self.artifact(folder+'/timing.json',False);checkpoint_time=self.artifact(folder+'/checkpoint-timing.json',False)
            if timer:
                validate_timing(timer,start,update,checkpoint_time,self.artifacts,self.artifact('groups/1/probe.json'),n)
                self.timings[n]=timer
            if n%5==0:
                league=self.artifact(f'league/after-{n}.json',False)
                if league is not None:
                    candidates=[r for r in current if r['origin']=='model'];admissions=[];evidence={}
                    for group in range(n-4,n+1):
                        score_path=f'groups/{group}/scores.json';old_scores=self.artifact(score_path);evidence[score_path]=self.artifacts[score_path]
                        for i,row in enumerate(old_scores['rows']):
                            if (not row['runtime_valid'] or not row['matches'] or row['candidate_program_faults']
                                or row['positive_banking_games']*2<len(row['matches'])):continue
                            program=row['classification']['program'];ir=program['ir_sha256']
                            if ir in reserved:continue
                            item={'id':row['generation'],'origin':'model','authorship':'fresh_learner_final_controller',
                                'source_checkpoint':old_scores['source_checkpoint'],'compiled_ir':program['compiled_ir'],
                                'ir_sha256':ir,'generation_sha256':row['generation_sha256'],
                                'source_path':f"generations/{row['generation']}.json",'admitted_after_group':n}
                            candidates=[r for r in candidates if r['ir_sha256']!=ir]+[item]
                            admissions.append({'group':group,'candidate_index':i,'controller':item})
                    current=[r for r in current if r['origin']=='scripted']+candidates[-2:]
                    same(league['selected'],current,'Deterministic league refresh differs')
                    same(league['admissions'],admissions,'League candidate admission history differs')
                    same(league['scores_sha256'],evidence,'League scoring provenance differs')
                    require(league['after_group']==n and league['reserved_model_ir_sha256']==sorted(reserved),
                        'League heldout exclusion differs')
                    self.leagues[n]=league
    def selection(self):
        row=self.artifact('endpoint-selection.json',False)
        if row is None:return
        require(row['profile_groups']==3 and set(row['timing_sha256'])=={'1','2','3'},'Endpoint was not chosen after exactly three groups')
        cost=full_cost([self.timings[i] for i in (1,2,3)],self.artifact('groups/1/probe.json'),self.config)
        for k in ('full_update_price_seconds','maximum_full_group_seconds'):same(row[k],cost[k],'Full-priced endpoint timing differs')
        for i in (1,2,3):require(row['timing_sha256'][str(i)]==self.artifacts[f'groups/{i}/timing.json'],'Profile timing hash differs')
        require(row['probe_sha256']==self.artifacts['groups/1/probe.json'],'Profile full-backward probe differs')
        remaining=remaining_at(self.status,row['selected_at'])
        same(row['remaining_seconds'],remaining,'Endpoint remaining time differs')
        expected=select_endpoint([cost['maximum_full_group_seconds']]*3,remaining)
        same(row['projected_remaining_seconds'],expected['projected_remaining_seconds'],'Endpoint projection differs')
        require(row['endpoint']==expected['endpoint'] and row['passed'] is (expected['endpoint'] is not None),
            'Endpoint choice is not the fixed timing-only maximum')
        require(row['safety_factor']==1.5 and row['reserved_final_and_export_seconds']==32400,'Evaluation reserve changed or counted twice')
        self.endpoint=expected['endpoint']
        for n in range(4,26):
            path=f'admissions/before-group-{n}.json';admission=self.artifact(path,False)
            if admission is None:
                require(f'groups/{n}/timing-start.json' not in self.artifacts,'Post-profile group lacks budget admission');continue
            require(self.endpoint is not None and n<=self.endpoint,'Admission after selected endpoint')
            require(set(admission['timing_sha256'])=={str(i) for i in range(1,n)},'Later admission omitted a slower group')
            for i in range(1,n):require(admission['timing_sha256'][str(i)]==self.artifacts[f'groups/{i}/timing.json'],'Admission timing hash differs')
            updated=full_cost([self.timings[i] for i in range(1,n)],self.artifact('groups/1/probe.json'),self.config)
            for k in ('full_update_price_seconds','maximum_full_group_seconds'):same(admission[k],updated[k],'Later full update price differs')
            same(admission['remaining_seconds'],remaining_at(self.status,admission['checked_at']),'Admission remaining time differs')
            required=1.5*updated['maximum_full_group_seconds']+32400
            same(admission['required_seconds'],required,'Later full update allowance differs')
            require(admission['passed'] is (admission['remaining_seconds']>=required),'Later admission outcome differs')
            require(admission['endpoint_selection_sha256']==self.artifacts['endpoint-selection.json']
                and admission['probe_sha256']==self.artifacts['groups/1/probe.json'],'Later admission lineage differs')
            if f'groups/{n}/timing-start.json' in self.artifacts:require(admission['passed'],'Group started after failed timing admission')
    def evaluation(self,runtime,pool):
        if self.endpoint is None:
            require(not any(p.startswith('evaluation/') for p in self.artifacts),'Evaluation before fixed timing endpoint');return
        endpoints=('base',f'cycle{self.endpoint}');seal=self.artifact('evaluation/sealed.json',False)
        if seal:
            require(all(self.checkpoint(e) for e in endpoints),'Evaluation precedes final checkpoint')
            paths=[f'generations/eval-{e}-{i}.json' for e in endpoints for i in range(16)]
            expected={'checkpoints':{e:self.artifacts[f'checkpoints/{e}/metadata.json'] for e in endpoints},
                'generations':{p:self.artifacts.get(p) for p in paths},'draws':32,'sampled_turns':64,
                'endpoint_selection_sha256':self.artifacts['endpoint-selection.json'],'scoring_allowed':True}
            require(all(expected['generations'].values()),'Missing endpoint authoring is not a zero outcome')
            same(seal,expected,'Fixed all32 evaluation seal differs')
        for label in endpoints:
            result=self.artifact(f'evaluation/{label}/holdout.json',False)
            if result is None:continue
            require(seal and result['checkpoint']==label and result['prompt_condition']=='native_nonthinking_revision_rl'
                and result['training_use'] is False and result['split']=='holdout'
                and result['prescribed_program_draws']==len(result['scores'])==16,'Heldout denominator differs')
            require(result['opponents_sha256']==self.artifacts['heldout-opponents.json']
                and result['seal_sha256']==self.artifacts['evaluation/sealed.json'],'Evaluation roster/seal differs')
            same(result['seeds'],self.schedule['holdout_maps'],'Heldout maps differ')
            same(result['opponents'],self.heldout_pool['selected'],'Fixed evaluation opponents changed')
            for i,row in enumerate(result['scores']):
                name=f'eval-{label}-{i}'
                require(row['generation']==name and row['generation_sha256']==self.artifacts[f'generations/{name}.json'],'Heldout generation differs')
                self.scored_row(row,self.generations[name],self.heldout_pool['selected'],result['seeds'],runtime)
            self.evaluations[label]=previous.summarize_programs(result['scores'],16)

    def execute(self):
        families=self.frozen()
        with calibration.archived_runtime(self.run,self.manifest) as runtime:
            self.runtime=runtime
            require((self.run/'prompts/debug.txt').read_text()==runtime.debug.debug_model_spec(),'Frozen initial messages changed')
            same(self.config['history_transform'],runtime.debug.REVISION_CONTEXT_POLICY,'Practice history transform changed')
            self.maps(runtime,families);self.diagnostics();self.checkpoint('base')
            pool=self.initial_pool(runtime);self.groups(runtime,pool);self.selection()
            self.all_batches();self.all_generations(runtime);self.evaluation(runtime,pool)
        complete=bool(self.status['state']=='complete' and self.endpoint in ENDPOINTS
            and all(f'cycle{i}' in self.checkpoints for i in range(1,self.endpoint+1))
            and len(self.evaluations)==2 and len(self.generations)==8*self.endpoint+32
            and len(self.raw_turns)==16*self.endpoint+64)
        require(not self.status['claims_allowed'] or complete,'Incomplete run permits comparison')
        require(self.status['state']!='complete' or complete,'Premature complete state')
        actual=sum(m['report']['optimizer_steps'] for label,m in self.checkpoints.items() if label!='base')
        if complete:
            require(self.status['attempted_groups']==self.endpoint and self.status['actual_optimizer_steps']==actual
                and self.status['final_checkpoint']==f'cycle{self.endpoint}' and self.status['learning_comparison_available'] is True,
                'Terminal endpoint/update counts differ')
            require(set(self.leagues)==set(range(5,self.endpoint+1,5)),'Missing prescribed league refresh')
            require(set(self.timings)==set(range(1,self.endpoint+1)),'Missing complete group timing')
        if self.endpoint:
            require(not any(int(label[5:])>self.endpoint for label in self.checkpoints if label.startswith('cycle')),
                'Training continued beyond the selected endpoint')
        if self.status['state']=='throughput_failed':require(self.endpoint is None and 'endpoint-selection.json' in self.artifacts,
            'Throughput failure lacks fixed timing decision')
        unknown=set(self.artifacts)-self.allowed_artifacts
        require(not unknown,'Unrecognized committed artifacts: '+str(sorted(unknown)))
        for p,d in self.start_hashes.items():require(sha(self.run/p)==d,'Run changed during audit: '+p)
        paired=paired_summary(self.evaluations['base']['programs'],self.evaluations[f'cycle{self.endpoint}']['programs'],
            self.schedule['eval_generation_seeds']) if complete else None
        return {'format_version':1,'kind':'independent_cpu_debug_revision_rl_audit','run':str(self.run),
            'manifest_sha256':self.start_hashes['manifest.json'],'status_sha256':self.start_hashes['status.json'],
            'auditor_sha256':sha(Path(__file__)),'integrity_passed':True,'recorded_state':self.status['state'],
            'planned_study_complete':complete,'planned_comparisons_available':complete,'selected_endpoint':self.endpoint,
            'baseline':'Exact audited SFT32 adapter plus zero-step RL Adam under the same mixed-native harness',
            'new_supervised_optimizer_steps':0,'genuine_RL_optimizer_steps':actual,'genuine_optimizer_steps':actual,
            'committed_trajectories':len(self.generations),'committed_raw_sampled_turns':len(self.raw_turns),
            'raw_turns_pending_trajectory_materialization':sum(k[0] not in self.generations for k in self.raw_turns),
            'verified_unique_game_replays':len(self.verified_replays),'practice_feedback_records':len(self.feedback_checked),
            'tensor_reports':self.tensor_reports,'training_groups':self.group_reports,'evaluation_cells':self.evaluations,
            'paired_yield':paired,'full_backward_admission':self.probe,'context_admission':self.context_evidence,
            'limits':['One training seed and sixteen paired program clusters; games and turns are not independent model draws.',
                'CPU validates recorded neural evidence; it does not independently recompute language-model likelihoods or derivatives.',
                'Full backward feasibility covers the actual first group, not every possible later sequence; numerical failures remain terminal.',
                'Fixed teacher evaluation anchors were prior SFT examples; reserved from this RL roster, not unseen to the model.',
                'Historical-controller league is not simultaneous fresh neural self-play or evidence of perpetual improvement.',
                'Timing-only endpoint admission permits zero actual updates; attempted groups and genuine Adam steps are separate.']}


def registered_schedule():
    result={'train_practice_maps':{str(g):[211000000+g*100+i*2+j for i in range(8) for j in range(2)] for g in range(1,26)},
        'train_maps':{str(g):[212000000+g*10+j for j in range(2)] for g in range(1,26)},
        'eval_practice_maps':{str(i):[231000000+i*2+j for j in range(2)] for i in range(16)},
        'holdout_maps':list(range(232000000,232000006)),
        'train_generation_seeds':{str(g):[221000000+g*100+i*2 for i in range(8)] for g in range(1,26)},
        'eval_generation_seeds':[241000000+i*2 for i in range(16)],
        'revision_seed_rule':'draft_seed+1; fixed disjoint authoring families; no retries',
        'map_seed_replacements':[{'family':'train_practice_maps','group':'8','index':11,
            'rejected_seed':211000811,'collision_with_seed':151000251,'replacement_seed':219000000}]}
    result['train_practice_maps']['8'][11]=219000000
    return result


def validate_derivative(derivative,identity):
    require(derivative['finite_nonzero'] is True
        and derivative['missing_gradient_names']==derivative['nonfinite_gradient_names']==[],'Incomplete/nonfinite gradient')
    require(set(derivative['gradient_tensors'])==set(identity['trainable_names']),'Gradient tensor coverage differs')
    squared=[]
    for name,row in derivative['gradient_tensors'].items():
        schema=next(r for r in identity['parameter_schema'] if r['name']==name)
        require(row['shape']==schema['shape'] and row['dtype']==schema['dtype'],'Gradient tensor schema differs')
        value=row['squared_l2_norm'];require(type(value) in (int,float) and math.isfinite(value) and value>=0,'Invalid gradient squared norm')
        squared.append(value)
    same(derivative['gradient_l2_norm'],math.sqrt(math.fsum(squared)),'Gradient norm evidence differs')


def validate_checkpoint_update(report,source,target,config):
    for label,identity in (('before',source),('after',target)):
        expected={'adapter_sha256':value_sha(identity['adapter_tensors']),'optimizer_state_sha256':identity['state_sha256'],
            'rng':identity['rng'],'gradients_sha256':value_sha({n:None for n in identity['trainable_names']}),
            'config_sha256':value_sha(config),'model_training':False,'frozen_state_sha256':identity['frozen_state_sha256']}
        same(report['state_'+label],expected,'Actual optimizer checkpoint state differs: '+label)
        for key,value in report['optimizer_identity_'+label].items():same(identity[key],value,'Reported optimizer tensor state differs: '+key)
    for key in ('learning_rate','trainable_names','parameter_schema','parameter_groups','frozen_state_sha256','quantization_state_sha256','rng'):
        same(source[key],target[key],'Update changed invariant: '+key)
    require(report['passed'] is True and report['adam_step_before']==source['step'] and report['adam_step_after']==target['step'],
        'Successful update has incorrect actual Adam steps')
    for flag in ('rng_restored','config_restored','model_mode_restored','frozen_state_unchanged','gradients_cleared'):
        require(report[flag] is True,'Optimizer did not restore checkpoint boundary: '+flag)
    if report['optimizer_steps']==0:
        require(report['optimizer_step_attempted'] is False and report['optimizer_step_returned'] is False
            and source['step']==target['step'],'Zero-signal report attempted a step')
        same({k:v for k,v in source.items() if k!='files'},{k:v for k,v in target.items() if k!='files'},'Skip changed optimizer/model state')
        return
    require(report['optimizer_steps']==1 and report['optimizer_step_attempted'] is True
        and report['optimizer_step_returned'] is True and target['step']==source['step']+1,'Unproven Adam transition')
    same(report['optimizer_state_sha256'],target['state_sha256'],'Actual target optimizer hash differs')
    same(report['frozen_state_sha256'],target['frozen_state_sha256'],'Actual frozen state differs')
    for label in ('derivative','derivative_after_clip'):
        validate_derivative(report[label],source)
    require(math.isclose(report['gradient_norm_before_clip'],report['derivative']['gradient_l2_norm'],rel_tol=2e-5,abs_tol=1e-7),
        'Clipping input norm differs')
    require(report['derivative_after_clip']['gradient_l2_norm']<=config['gradient_clip']+1e-6,'Gradient clip bound exceeded')
    require(set(report['adapter_squared_deltas'])==set(source['trainable_names']),'Adapter delta coverage differs')
    require(all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in report['adapter_squared_deltas'].values()),'Nonfinite adapter delta')
    same(report['adapter_l2_delta'],math.sqrt(math.fsum(report['adapter_squared_deltas'].values())),'Adapter norm delta differs')


def validate_timing(row,start,update,checkpoint,artifacts,probe,number):
    require(row['cycle']==number and row['optimizer_steps']==update['optimizer_steps'] and row['includes_resume_downtime'] is True,
        'Group timer metadata differs')
    require(row['update_sha256']==artifacts[f'groups/{number}/update.json']
        and row['timing_start_sha256']==artifacts[f'groups/{number}/timing-start.json'],'Persistent timing source differs')
    elapsed=(datetime.fromisoformat(row['finished_at'])-datetime.fromisoformat(start['started'])).total_seconds()
    finite_positive(elapsed,'Invalid whole group duration');same(row['elapsed_seconds'],elapsed,'Whole group walltime differs')
    same(row['update_compute_seconds'],update['update_compute_seconds'],'Update compute time differs')
    require(type(row['update_compute_seconds']) in (int,float) and math.isfinite(row['update_compute_seconds']) and row['update_compute_seconds']>=0,
        'Invalid update compute time')
    cp=checkpoint['elapsed_seconds'] if checkpoint else elapsed
    if checkpoint:require(checkpoint['checkpoint_metadata_sha256']==artifacts[f'checkpoints/cycle{number}/metadata.json'],'Checkpoint timing identity differs')
    same(row['checkpoint_seconds'],cp,'Checkpoint timing differs')
    require(row['checkpoint_timing_recovered_upper_bound'] is (checkpoint is None),'Checkpoint recovery timing differs')
    actual=(update['update_compute_seconds']+cp) if update['optimizer_steps'] else 0.
    same(row['actual_update_stage_seconds'],actual,'Actual update-stage price differs')
    probe_time=probe['elapsed_seconds'] if number==1 else 0.
    same(row['probe_elapsed_seconds'],probe_time,'Probe timing differs')
    same(row['non_update_seconds'],max(.000001,elapsed-probe_time-actual),'Non-update phase accounting differs')


def full_cost(timings,probe,config):
    require(timings,'No timing evidence')
    price=max(finite_positive(probe['elapsed_seconds'],'Full backward timing missing')+120,
        *(r['actual_update_stage_seconds'] for r in timings))
    return {'full_update_price_seconds':price,
        'maximum_full_group_seconds':max(r['non_update_seconds']+price for r in timings)}


def remaining_at(status,timestamp):
    budget=status['execution_budget'];at=datetime.fromisoformat(timestamp)
    require(at>=datetime.fromisoformat(budget['first_started']),'Timing decision predates execution')
    remaining=(datetime.fromisoformat(budget['hard_stop'])-at).total_seconds()
    require(remaining>=0,'Timing decision after immutable stop');return remaining


def audit(run):return Audit(run).execute()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path);args=parser.parse_args()
    text=previous.canonical(audit(args.run))+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as stream:stream.write(text)
    else:print(text,end='')


if __name__=='__main__':main()
