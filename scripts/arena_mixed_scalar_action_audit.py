"""Independent CPU provenance for legal-action draws; never loads model weights."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from selfplay.arena_mixed_scalar import cuda_policy, envs


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get_tokenizer(run, tokenizer_dir=None):
    """Load only locally hash-verified original tokenizer/config, never weights."""
    from transformers import AutoTokenizer
    run = Path(run)
    manifest = json.loads((run/'manifest.json').read_text())
    config = manifest['config']['policy']
    require(config['model'] == cuda_policy.MODEL and config['revision'] == cuda_policy.REVISION,
            'Original pinned model/revision required')
    expected = json.loads((run/'inputs/model-file-sha256.json').read_text())
    folder = Path(tokenizer_dir) if tokenizer_dir is not None else run/'inputs/tokenizer'
    files = {name:file_sha(folder/name) for name in ('config.json','tokenizer.json','tokenizer_config.json')}
    require(all(files[name] == expected[name] for name in files), 'Local tokenizer bytes differ from pinned inputs')
    require(config['tokenizer_files_sha256'] == cuda_policy.digest({name:files[name]
        for name in ('tokenizer.json','tokenizer_config.json')}), 'Policy tokenizer digest differs')
    require(config['model_index_sha256'] == expected['model.safetensors.index.json'], 'Policy model-index digest differs')
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    tokenizer._mixed_action_audit_identity = {'files':files, 'tokenizer_files_sha256':config['tokenizer_files_sha256'],
        'model_index_sha256':config['model_index_sha256'], 'model':config['model'], 'revision':config['revision']}
    return tokenizer


def same(left, right, message):
    require(envs.canonical(left) == envs.canonical(right), message)


def _binding(binding, config):
    cuda_policy.validate_binding(binding)
    base = binding['base_identity']
    for key in ('model','revision','model_index_sha256','tokenizer_files_sha256'):
        require(base.get(key) == config[key], 'Binding original model/tokenizer identity differs: '+key)
    require(base.get('quantization') == 'nf4-double-fp32', 'Behavior arithmetic identity differs')
    frozen = base.get('frozen_state_sha256')
    require(isinstance(frozen, str) and len(frozen) == 64 and set(frozen) <= set('0123456789abcdef'),
            'Missing frozen-state fingerprint')


def verify_action_rows(episodes, config, tokenizer, *, opponent_binding=None):
    """Check every real learner/base-opponent draw including the sampled label.

    Replay and checkpoint-tensor verification remain separate required audits.
    For a fixed base model opponent, callers must supply its verified checkpoint
    binding instead of trusting a binding repeated by the sampled output.
    """
    import torch
    cuda_policy.validate_config(config)
    identity = getattr(tokenizer, '_mixed_action_audit_identity', None)
    require(identity is not None, 'Use get_tokenizer to bind original tokenizer bytes first')
    for key in ('model','revision','model_index_sha256','tokenizer_files_sha256'):
        require(identity[key] == config[key], 'Verified tokenizer/config identity differs')
    labels, label_ids = cuda_policy.select_labels(tokenizer)
    labels_sha = cuda_policy.digest([labels,label_ids])
    if opponent_binding is not None:
        _binding(opponent_binding,config)
        require(opponent_binding['checkpoint_id']=='base' and opponent_binding['version']==0,
                'Fixed opponent must be exact base checkpoint')
    summaries, role_counts, maximum = [], {'learner':0,'opponent':0}, 0
    for episode in episodes:
        expected = {}
        for event in episode['events']:
            normalized = json.loads(envs.canonical(event))
            for player in event['active_players']:
                expected[(event['turn'],player)] = normalized
        _binding(episode['behavior_binding'],config)
        all_seen = set()
        for field, role in (('transitions','learner'),('opponent_transitions','opponent')):
            if role == 'opponent':
                require(not episode[field] or episode['schedule']['opponent']=='base',
                        'Heuristic/absent opponent cannot have neural draws')
                require(not episode[field] or opponent_binding is not None,
                        'Explicit audited base-opponent binding required')
            source = episode['behavior_binding'] if role=='learner' else opponent_binding
            for row in episode[field]:
                player, turn = row['player'], row['turn']
                require(type(player) is int and type(turn) is int, 'Action role/turn must be exact integers')
                key = (turn,player)
                require(key in expected and key not in all_seen, 'Duplicate/nonexistent policy decision')
                all_seen.add(key)
                require((player in episode['learner_seats']) == (role=='learner'), 'Neural action has wrong real role')
                event = expected[key]
                same(row['observation'],event['observations'][str(player)],'Prompt observation differs from replay')
                same(row['actions'],event['legal_actions'][str(player)],'Prompt menu differs from replay')
                same(row['action'],event['actions'][str(player)],'Selected action differs from replay')
                require(row['episode_id']==episode['episode_id'] and row['environment']==episode['environment']
                        and row['scenario_family_id']==episode['scenario_family_id'], 'Action scenario provenance differs')
                require(row['terminal_return']==episode['returns'][player], 'Action terminal return differs')
                same(row['checkpoint'],source,'Action checkpoint differs from authoritative binding')
                require(row['policy_version']==source['version'] and row['actor_snapshot_sha256']==source['adapter_sha256']
                        and row['thinking'] is False, 'Action policy version/arithmetic metadata differs')
                same(row['base_identity'],source['base_identity'],'Action base identity differs')
                require(row['actor_config_sha256']==cuda_policy.digest(config), 'Action sampler configuration differs')
                require(row.get('physical_execution_contract_sha256')==cuda_policy.digest(config['physical_execution_contract']), 'Physical action execution contract differs')
                actions = row['actions']
                require(1<=len(actions)<=len(labels), 'Action menu size differs')
                prompt = cuda_policy.action_prompt(event['observations'][str(player)],actions,labels)
                rendered = tokenizer.apply_chat_template([{'role':'user','content':prompt}],
                    tokenize=False,add_generation_prompt=True,enable_thinking=False)
                ids = tokenizer.encode(rendered,add_special_tokens=False)
                require(0<len(ids)<=config['max_context_tokens'], 'Sampled prompt exceeds frozen context limit')
                require(row['prompt']==prompt and row['rendered_prompt']==rendered and row['input_ids']==ids,
                        'Original native action prompt/token IDs differ')
                require(row['prompt_sha256']==hashlib.sha256(prompt.encode()).hexdigest()
                        and row['input_ids_sha256']==cuda_policy.digest(ids)
                        and row['menu_sha256']==cuda_policy.digest(actions)
                        and row['labels_sha256']==labels_sha, 'Action prompt/menu/label digest differs')
                values, pick = row['old_log_probs'], row['action_index']
                require(isinstance(values,list) and len(values)==len(actions)
                        and all(type(v) in (int,float) and math.isfinite(v) for v in values),
                        'Missing/nonfinite full legal categorical vector')
                require(all(v<=1e-6 for v in values) and abs(math.fsum(math.exp(v) for v in values)-1)<1e-5,
                        'Behavior log probabilities are not normalized')
                require(type(pick) is int and 0<=pick<len(actions)
                        and row['old_log_prob']==values[pick] and row['action_label']==labels[pick]
                        and row['action_token_id']==label_ids[pick], 'Selected label/token/log probability differs')
                same(row['action'],actions[pick],'Selected label does not identify played action')
                seed_key = episode['schedule'].get('paired_seed_key',episode['episode_id'])
                seed = int(envs.digest(['action',seed_key,turn,player])[:15],16)
                menu_seed = int(envs.digest(['menu',seed_key,turn,player])[:15],16)
                require(type(row['sample_seed']) is int and row['sample_seed']==seed
                        and type(row['menu_seed']) is int and row['menu_seed']==menu_seed,
                        'Original scenario-derived action/menu seed differs')
                generator = torch.Generator(device='cpu').manual_seed(seed)
                probs = torch.softmax(torch.tensor(values,dtype=torch.float64),-1)
                actual = int(torch.multinomial(probs,1,generator=generator))
                require(actual==pick,'Seeded CPU categorical does not reproduce sampled action')
                entropy = float(-(probs*probs.log()).sum())
                require(type(row['entropy']) in (int,float) and math.isfinite(row['entropy'])
                        and math.isclose(row['entropy'],entropy,rel_tol=1e-12,abs_tol=1e-12),
                        'Recorded sampled entropy differs')
                summaries.append({'episode_id':episode['episode_id'],'turn':turn,'player':player,
                                  'role':role,'row_sha256':cuda_policy.digest(row),'input_tokens':len(ids)})
                role_counts[role] += 1; maximum=max(maximum,len(ids))
        expected_neural = {key for key in expected if key[1] in episode['learner_seats']
                          or episode['schedule']['opponent']=='base'}
        require(all_seen==expected_neural,'Missing exact neural decision coverage')
    return {'verified':True,'kind':'mixed_arena_cpu_native_action_provenance',
        'model':config['model'],'revision':config['revision'],'tokenizer_files_sha256':identity['files'],
        'labels_sha256':labels_sha,'roles':role_counts,'actions':len(summaries),
        'maximum_prompt_tokens':maximum,'rows':summaries,
        'scope':'Exact CPU tokenizer, labels, full categorical distribution and seeded choice; no model forward or gradient replay.'}
