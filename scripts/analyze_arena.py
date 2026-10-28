#!/usr/bin/env python3
"""CPU-only audit of frozen arena episodes, checkpoints, and paired evaluation.

No model forward passes, GPU work, API calls, or source edits are performed.
Replay uses the verified archived simulator sources, not the current checkout.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import math
from pathlib import Path
import random
import sys

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    # Serialized simulator dictionaries have string keys; normalize both sides.
    return json.dumps(json.loads(json.dumps(value)), sort_keys=True, separators=(',', ':'), allow_nan=False)


def equal(actual, expected, description):
    if canonical(actual) != canonical(expected):
        raise ValueError(description)


def close(actual, expected, description, tolerance=1e-7):
    if not math.isfinite(float(actual)) or abs(float(actual)-float(expected)) > tolerance:
        raise ValueError(f'{description}: {actual} != {expected}')


def mean(values):
    return float(np.mean(values)) if values else None


def bootstrap_description(values):
    """Descriptive resampling of independent scenario clusters, never turns."""
    n = len(values)
    if n < 2:
        return {'scenario_clusters': n, 'descriptive_bootstrap_95_interval': None}
    if n <= 7:
        samples = [sum(v)/n for v in itertools.product(values, repeat=n)]
    else:
        rng = np.random.default_rng(212026)
        samples = np.mean(rng.choice(values, size=(10000, n), replace=True), axis=1)
    return {'scenario_clusters': n,
            'descriptive_bootstrap_95_interval': np.percentile(samples, [2.5, 97.5]).tolist(),
            'caution': 'Very small scenario sample; a descriptive bootstrap interval does not establish reliable improvement.'}


def load_archive(run, manifest, project):
    verified, current = {}, {}
    for original, expected in manifest['source_sha256'].items():
        archive = run/'source'/Path(original).name
        if not archive.is_file() or sha(archive) != expected:
            raise ValueError(f'Frozen source archive mismatch: {archive}')
        verified[original] = expected
        live = project/original
        current[original] = live.is_file() and sha(live) == expected
    name = '_arena_audit_' + sha(run/'manifest.json')[:12]
    spec = importlib.util.spec_from_file_location(name, run/'source/__init__.py',
                                                 submodule_search_locations=[str(run/'source')])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    policy = importlib.import_module(name+'.policy')
    experiment = importlib.import_module(name+'.experiment')
    return package, policy, experiment, {'archives_verified': verified, 'live_source_matches': current,
        'live_source_unchanged': all(current.values()),
        'note': 'Archived source hashes are mandatory. Live-source drift is reported but does not invalidate archived replay.'}


def frozen_schedule(experiment, manifest):
    schedule = {}
    for cycle in range(manifest['cycles']):
        schedule[f'train-{cycle}'] = experiment.game_specs('train', cycle=cycle,
                                      candidate='base' if cycle == 0 else f'cycle{cycle}')
    for checkpoint in ['base']+[f'cycle{i+1}' for i in range(manifest['cycles'])]:
        for opponent in manifest['evaluation_opponents']:
            schedule[f'eval-{checkpoint}-{opponent}'] = experiment.game_specs(
                'eval', candidate=checkpoint, opponent=opponent,
                seeds=manifest['evaluation_scenario_seeds_per_game_opponent'])
    schedule['league-cycle2-cycle1'] = experiment.game_specs('league', candidate='cycle2', opponent='cycle1',
                                  seeds=manifest['evaluation_scenario_seeds_per_game_opponent'])
    return schedule


def load_tokenizer(manifest):
    # Explicitly forbid downloads/network. Only tokenizer files in the already
    # cached, pinned model snapshot are used; model tensors are never loaded.
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    path = snapshot_download(manifest['model']['model'], revision=manifest['model']['revision'],
                             local_files_only=True)
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def checkpoint_audit(run, manifest):
    from safetensors.numpy import load_file
    reports, metadata, arrays = {}, {}, {}
    for name in ['base']+[f'cycle{i+1}' for i in range(manifest['cycles'])]:
        directory = run/'checkpoints'/name
        if not (directory/'metadata.json').exists():
            continue
        meta = json.loads((directory/'metadata.json').read_text())
        actual_hash = sha(directory/'adapters.safetensors')
        if actual_hash != meta['adapter_sha256']:
            raise ValueError(f'Checkpoint hash mismatch: {name}')
        equal({k: meta[k] for k in ('model', 'revision')}, manifest['model'], f'Model identity differs: {name}')
        equal(meta['checkpoint'], name, f'Checkpoint name differs: {name}')
        if metadata:
            for key in ('labels', 'label_token_ids', 'trainable_parameters'):
                equal(meta[key], metadata['base'][key], f'Checkpoint interface changed: {name}/{key}')
        weights = load_file(str(directory/'adapters.safetensors'))
        if not weights or any(not np.isfinite(v).all() for v in weights.values()):
            raise ValueError(f'Non-finite or empty checkpoint: {name}')
        metadata[name], arrays[name] = meta, weights
        reports[name] = {'adapter_sha256': actual_hash, 'hash_verified': True,
                         'tensor_count': len(weights), 'finite_tensors': True,
                         'trainable_parameters': sum(v.size for v in weights.values())}
        if reports[name]['trainable_parameters'] != meta['trainable_parameters']:
            raise ValueError(f'Trainable tensor count mismatch: {name}')
        if name != 'base':
            source = 'base' if name == 'cycle1' else f'cycle{int(name[5:])-1}'
            prior = arrays[source]
            equal(sorted(weights), sorted(prior), f'Checkpoint tensor keys changed: {name}')
            if any(weights[k].shape != prior[k].shape for k in weights):
                raise ValueError(f'Checkpoint shapes changed: {name}')
            delta = math.sqrt(sum(float(np.square(weights[k].astype(np.float64)-prior[k]).sum())
                                  for k in weights))
            if not delta > 0:
                raise ValueError(f'No model update in checkpoint {name}')
            reports[name].update(source_checkpoint=source, independently_measured_delta_l2=delta)
            summary = json.loads((directory/'summary.json').read_text())
            equal(summary['source'], source, f'Training source mismatch: {name}')
            equal(summary['target'], name, f'Training target mismatch: {name}')
            close(summary['parameter_delta_l2'], delta, f'Parameter delta mismatch: {name}',
                  max(1e-8, delta*1e-5))
            equal(summary['checkpoint']['adapter_sha256'], actual_hash, f'Summary hash mismatch: {name}')
            reports[name]['training_summary'] = summary
            metrics = [json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
            if any(not all(math.isfinite(float(v)) for v in row.values()) for row in metrics):
                raise ValueError(f'Nonfinite training metrics: {name}')
            reports[name]['optimization'] = {key: {'mean': mean([r[key] for r in metrics]),
                                                   'max': max(r[key] for r in metrics)}
                for key in ('kl_to_rollout', 'entropy', 'clip_fraction', 'gradient_norm')}
    return reports, metadata


def replay(row, package, prompt_function, metadata, tokenizer, manifest):
    environment = package.make_env(row['environment'], row['seed'], row['regime'])
    equal(environment.max_rounds, manifest['horizons'][row['environment']], 'Environment horizon differs')
    equal(row['initial_state'], environment.snapshot(), 'Initial snapshot differs')
    transitions = {}
    for transition in row['transitions']:
        key = (transition['turn'], transition['player'])
        if key in transitions:
            raise ValueError(f'Duplicate transition at {key}')
        transitions[key] = transition
    checked = 0
    for event in row['events']:
        if environment.done:
            raise ValueError('Event after terminal state')
        equal(event['turn'], environment.turn, 'Event turn differs')
        actions = {int(k): v for k, v in event['actions'].items()}
        equal(sorted(actions), sorted(environment.active_players()), 'Active players differ')
        for player in environment.active_players():
            checkpoint = row['policies'][player]
            if checkpoint == 'heuristic':
                equal(actions[player], environment.heuristic_action(player), 'Heuristic action differs')
                continue
            transition = transitions.pop((environment.turn, player))
            labels = metadata[checkpoint]['labels']
            sample_seed = row['seed']*100000 + environment.turn*37 + player*11
            menu = deepcopy(environment.legal_actions(player))
            random.Random(sample_seed+907).shuffle(menu)
            observation = environment.observation(player)
            equal(transition['sample_seed'], sample_seed, 'Policy sample seed differs')
            equal(transition['checkpoint'], checkpoint, 'Behavior checkpoint differs')
            equal(transition['actions'], menu, 'Action menu differs')
            equal(transition['observation'], observation, 'Policy observation differs/contains extra information')
            equal(transition['prompt'], prompt_function(observation, menu, labels), 'Policy prompt differs')
            if row['environment'] == 'market':
                forbidden = {'terminal_value', 'private_signals', 'schedule', 'absolute_pnl', 'accounts'}
                if forbidden.intersection(observation):
                    raise ValueError('Hidden market information appears in policy observation')
                equal(observation['private_signal'], environment.snapshot()['private_signals'][player],
                      'Private signal belongs to wrong player')
            formatted = tokenizer.apply_chat_template([{'role':'user','content':transition['prompt']}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tokenizer.encode(formatted, add_special_tokens=False)
            equal(transition['prompt_tokens'], len(ids), 'Tokenized prompt length differs')
            if len(ids) > 3072:
                raise ValueError('Prompt exceeds frozen context cap')
            pick = transition['action_index']
            if type(pick) is not int or not 0 <= pick < len(menu):
                raise ValueError('Invalid sampled action index')
            equal(transition['action'], menu[pick], 'Selected action and menu differ')
            equal(transition['action_label'], labels[pick], 'Action label differs')
            equal(transition['action'], actions[player], 'Recorded action differs from executed action')
            logps = np.array(transition['old_log_probs'], dtype=np.float64)
            if len(logps) != len(menu) or not np.isfinite(logps).all() or np.max(logps) > 2e-5:
                raise ValueError('Invalid behavior probability vector')
            close(np.exp(logps).sum(), 1., 'Behavior probabilities not normalized', 2e-5)
            close(transition['old_log_prob'], logps[pick], 'Selected behavior log-probability differs')
            probabilities = np.exp(logps); probabilities /= probabilities.sum()
            sampled = int(np.random.default_rng(sample_seed).choice(len(menu), p=probabilities))
            equal(pick, sampled, 'Action cannot be reproduced from recorded probabilities and seed')
            entropy = -sum(p*math.log(p) for p in probabilities if p)
            close(transition['entropy'], entropy, 'Behavior entropy differs', 2e-6)
            close(transition['terminal_return'], row['returns'][player], 'Attributed return differs')
            checked += 1
        environment.step(actions)
        equal(event['state'], environment.snapshot(), 'Post-action snapshot differs')
    if transitions:
        raise ValueError('Unused/extra policy transitions remain')
    if not environment.done:
        raise ValueError('Incomplete episode recorded as completed')
    equal(row['returns'], environment.returns(), 'Terminal returns differ')
    equal(row['metrics'], environment.metrics(), 'Terminal metrics differ')
    close(sum(row['returns']), 0., 'Competitive returns not zero-sum')
    return checked


def game_outcomes(rows):
    values = [r['returns'][r['candidate_seat']] for r in rows]
    reasons = Counter(r['metrics'].get('finish_reason', 'settlement') for r in rows)
    paired = defaultdict(list)
    for row in rows:
        paired[row['seed']].append(row['returns'][row['candidate_seat']])
    pairs = [{'seed': seed, 'games': len(scores), 'mean_return': mean(scores)}
             for seed, scores in sorted(paired.items())]
    result = {'games': len(rows), 'mean_return': mean(values),
              'positive_outcomes': sum(x > 1e-8 for x in values),
              'zero_outcomes': sum(abs(x) <= 1e-8 for x in values),
              'negative_outcomes': sum(x < -1e-8 for x in values),
              'termination_reasons': dict(reasons), 'paired_scenario_scores': pairs,
              **bootstrap_description([p['mean_return'] for p in pairs if p['games'] == 2])}
    game = rows[0]['environment']
    if game == 'market':
        pnl = [r['metrics']['absolute_pnl'][r['candidate_seat']] for r in rows]
        result['market'] = {'mean_absolute_pnl': mean(pnl), 'positive_pnl_games': sum(v > 0 for v in pnl),
            'negative_pnl_games': sum(v < 0 for v in pnl), 'absolute_pnl_by_game': pnl,
            'mean_filled_units': mean([r['metrics']['filled_units'][r['candidate_seat']] for r in rows]),
            'mean_trade_count': mean([r['metrics']['trade_count'] for r in rows]),
            'mean_final_cash': mean([r['metrics']['cash'][r['candidate_seat']] for r in rows]),
            'mean_final_inventory': mean([r['metrics']['inventory'][r['candidate_seat']] for r in rows])}
    elif game == 'maze':
        result['maze'] = {'candidate_goal_completions': sum(
            r['events'][-1]['state']['positions'][r['candidate_seat']] == r['events'][-1]['state']['goal'] for r in rows),
            'actual_goal_wins': sum(r['metrics']['finish_reason']=='goal' and r['returns'][r['candidate_seat']]>0 for r in rows),
            'distance_adjudications': reasons['distance_adjudication_at_cap'],
            'mean_blocks_placed': mean([r['metrics']['blocks_placed'][r['candidate_seat']] for r in rows])}
    elif game == 'chess':
        result['chess'] = {'checkmate_wins': sum(r['metrics']['checkmate'] and r['returns'][r['candidate_seat']]>0 for r in rows),
            'checkmate_losses': sum(r['metrics']['checkmate'] and r['returns'][r['candidate_seat']]<0 for r in rows),
            'material_adjudications': reasons['material_adjudication_at_cap'],
            'mean_candidate_material_advantage': mean([
                r['metrics']['material_advantage_white'] * (1 if r['candidate_seat']==0 else -1) for r in rows])}
    return result


def evaluations(rows):
    grouped = defaultdict(list)
    for row in rows:
        if 'candidate' in row:
            grouped[(row['candidate'], row['opponent'], row['environment'])].append(row)
    reports = {' / '.join(key): game_outcomes(group) for key, group in sorted(grouped.items())}
    deltas = []
    candidates=sorted({key[0] for key in grouped},key=lambda name:0 if name=='base' else int(name[5:]))
    for before, after in itertools.combinations(candidates,2):
        for opponent, game in sorted({key[1:] for key in grouped if key[1] in ('base','heuristic')}):
            left = {(r['seed'],r['candidate_seat']):r['returns'][r['candidate_seat']]
                    for r in grouped.get((before,opponent,game),[])}
            right = {(r['seed'],r['candidate_seat']):r['returns'][r['candidate_seat']]
                     for r in grouped.get((after,opponent,game),[])}
            paired = defaultdict(list)
            for seed, seat in sorted(left.keys() & right.keys()):
                paired[seed].append(right[(seed,seat)]-left[(seed,seat)])
            complete = [{'seed': seed, 'mean_return_change':mean(scores)}
                        for seed, scores in sorted(paired.items()) if len(scores)==2]
            if complete:
                differences = [p['mean_return_change'] for p in complete]
                deltas.append({'before':before,'after':after,'opponent':opponent,'environment':game,
                    'mean_return_change':mean(differences),'paired_scenario_changes':complete,
                    'positive_scenario_changes':sum(x>1e-8 for x in differences),
                    'negative_scenario_changes':sum(x<-1e-8 for x in differences),
                    **bootstrap_description(differences)})
    macro = {}
    for checkpoint in candidates:
        cells = [value['mean_return'] for key,value in reports.items()
                 if key.split(' / ')[0] == checkpoint and key.split(' / ')[1] in ('base','heuristic')]
        if cells:
            macro[checkpoint] = {'mean_return':mean(cells), 'available_environment_opponent_cells':len(cells),
                'note':'Equal weight for each environment and fixed opponent; compare only complete six-cell checkpoints.'}
    return reports, deltas, macro


def training_signals(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row.get('regime')=='train':
            grouped[(row['policies'][0],row['environment'])].append(row)
    reports = {}
    for (checkpoint,game), episodes in sorted(grouped.items()):
        per_role = {}
        for player in (0,1):
            returns = [e['returns'][player] for e in episodes]
            advantages = [reward-mean([other for j,other in enumerate(returns) if i!=j])
                          if len(returns)>1 else reward for i,reward in enumerate(returns)]
            per_role[str(player)] = {'episode_returns':returns,'mean_return':mean(returns),
                'return_std':float(np.std(returns)), 'loo_advantages':advantages,
                'positive_advantage_episodes':sum(x>1e-8 for x in advantages),
                'negative_advantage_episodes':sum(x<-1e-8 for x in advantages),
                'nonzero_advantage_actions':sum(sum(t['player']==player for t in episode['transitions'])
                    for episode,advantage in zip(episodes,advantages) if abs(advantage)>1e-8)}
        transitions = [t for e in episodes for t in e['transitions']]
        reports[f'{checkpoint} / {game}'] = {'episodes':len(episodes),'transitions':len(transitions),
            'roles':per_role,'mean_action_entropy':mean([t['entropy'] for t in transitions]),
            'mean_prompt_tokens':mean([t['prompt_tokens'] for t in transitions]),
            'action_types':dict(Counter(t['action'].get('type','claim_draw' if t['action'].get('claim_draw') else 'chess_move')
                                       for t in transitions)),
            'termination_reasons':dict(Counter(e['metrics'].get('finish_reason','settlement') for e in episodes))}
    return reports


def analyze(run, allow_partial=False):
    run = Path(run).resolve()
    project = Path(__file__).resolve().parents[1]
    manifest = json.loads((run/'manifest.json').read_text())
    package, policy, experiment, sources = load_archive(run, manifest, project)
    schedule = frozen_schedule(experiment, manifest)
    equal(json.loads((run/'schedule.json').read_text()), schedule, 'Schedule differs from frozen generator')
    expected = {s['episode_id']:s for entries in schedule.values() for s in entries}
    if len(expected) != sum(map(len,schedule.values())):
        raise ValueError('Duplicate scheduled episode IDs')
    expected_file = {s['episode_id']:(f'train-rollouts-{int(stage.split("-")[1])+1}.jsonl'
                                    if stage.startswith('train-') else stage+'.jsonl')
                     for stage, entries in schedule.items() for s in entries}
    train_families = {(s['environment'],s['seed']//2 if s['environment']=='chess' else s['seed'])
                      for key,s in expected.items() if key.startswith('train-')}
    eval_families = {(s['environment'],s['seed']//2 if s['environment']=='chess' else s['seed'])
                     for key,s in expected.items() if not key.startswith('train-')}
    if train_families & eval_families:
        raise ValueError('Training and evaluation scenario families overlap')
    checkpoints, metadata = checkpoint_audit(run, manifest)
    tokenizer = load_tokenizer(manifest)
    for label, token_id in zip(metadata['base']['labels'], metadata['base']['label_token_ids'], strict=True):
        equal(tokenizer.encode(label,add_special_tokens=False), [token_id], 'Action label tokenization differs')
    if importlib.metadata.version('chess') != manifest['versions']['chess']:
        raise ValueError('Install the frozen python-chess/chess version before replay')
    rows, checked, seen, files, ignored = [], 0, set(), {}, []
    for path in sorted(run.glob('*.jsonl')):
        data = path.read_bytes()
        lines = data.splitlines(keepends=True)
        if lines and not lines[-1].endswith(b'\n'):
            if not allow_partial:
                raise ValueError(f'Incomplete JSONL write: {path.name}')
            ignored.append(path.name+': trailing incomplete line'); lines = lines[:-1]
        files[path.name] = {'sha256_of_read_snapshot':hashlib.sha256(data).hexdigest(), 'completed_rows':len(lines)}
        for number,line in enumerate(lines,1):
            row = json.loads(line)
            identity = row['episode_id']
            if identity not in expected or identity in seen:
                raise ValueError(f'Unexpected or duplicate episode {identity}')
            if path.name != expected_file[identity]:
                raise ValueError(f'Episode {identity} in wrong rollout file')
            for key,value in expected[identity].items():
                equal(row.get(key),value,f'Schedule mismatch {identity}/{key}')
            try:
                checked += replay(row,package,policy.action_prompt,metadata,tokenizer,manifest)
            except Exception as error:
                raise ValueError(f'{path.name}:{number} {identity}: {error}') from error
            rows.append(row); seen.add(identity)
    missing = sorted(set(expected)-seen)
    summary = json.loads((run/'summary.json').read_text()) if (run/'summary.json').exists() else None
    complete = not missing and summary is not None and summary.get('complete') is True
    if not allow_partial and not complete:
        raise ValueError(f'Run incomplete: {len(missing)} scheduled episodes missing or final summary absent')
    evaluation, changes, macro = evaluations(rows)
    warnings = ['Three scenario clusters per opponent/game give very low statistical power.',
        'Positive fractional rewards include capped material/distance adjudications, not necessarily actual wins.',
        'Action probabilities are checked for consistency and sampling replay; no neural forward pass is recomputed.',
        'One training seed, two cycles, action-only policy: no evidence of unbounded continual improvement or general reasoning transfer.']
    if not sources['live_source_unchanged']:
        warnings.append('Current source differs from the run archive; analysis used hash-verified archived sources.')
    return {'complete':complete,'all_completed_episodes_verified':True,'episodes_verified':len(rows),
        'policy_transitions_verified':checked,'scheduled_episodes':len(expected),'missing_episode_ids':missing,
        'source_verification':sources,'checkpoint_verification':checkpoints,'rollout_files':files,
        'manifest_sha256':sha(run/'manifest.json'),'schedule_sha256':sha(run/'schedule.json'),
        'train_eval_families_disjoint':True,'ignored_partial_writes':ignored,
        'evaluation':evaluation,'paired_checkpoint_changes':changes,'macro_fixed_opponent_return':macro,
        'training_signal':training_signals(rows),'warnings':warnings,
        'analysis_source_sha256':sha(__file__),'runtime':{'numpy':np.__version__,'chess':importlib.metadata.version('chess')},
        'audit_scope':'CPU-only archived simulator replay, visible observations, exact prompts/token counts, legal actions, seeded sampling, snapshots, returns, source and adapter integrity.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('runs/arena-v1'))
    parser.add_argument('--out',type=Path)
    parser.add_argument('--allow-partial',action='store_true')
    args = parser.parse_args()
    result = analyze(args.run,args.allow_partial)
    destination = args.out or args.run/'analysis.json'
    temporary = destination.with_suffix(destination.suffix+'.tmp')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    temporary.replace(destination)
    print(json.dumps({'complete':result['complete'],'episodes_verified':result['episodes_verified'],
        'policy_transitions_verified':result['policy_transitions_verified'],'output':str(destination.resolve())}))


if __name__ == '__main__':
    main()
