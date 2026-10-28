"""CPU tokenizer/context fixture, not model inference or evaluation evidence."""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from selfplay.arena_mixed import envs, cuda_policy


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stress(tokenizer_path, identities, context_limit=4096):
    from transformers import AutoTokenizer
    import importlib.metadata
    tokenizer_path = Path(tokenizer_path)
    expected = json.loads(Path(identities).read_text())
    token_files = {name:sha(tokenizer_path/name)
                   for name in ('tokenizer.json','tokenizer_config.json','config.json')}
    if any(value != expected[name] for name,value in token_files.items()):
        raise ValueError('Tokenizer is not the pinned original model tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
    labels, label_ids = cuda_policy.select_labels(tokenizer)
    maxima, counts, episodes, violations = {}, Counter(), [], []
    for game_index, name in enumerate(envs.ENVIRONMENTS):
        choices = ('heuristic','random','public_tests_repeat') if name=='coding' else ('heuristic','random')
        for offset in range(8):
            seed = 955000 + game_index*1000 + offset*2
            for chooser in choices:
                env = envs.make_env(name, seed)
                rng = random.Random(seed+7)
                while not env.done:
                    selected = {}
                    for player in env.active_players():
                        view = envs.policy_view(env, player)
                        prompt = cuda_policy.action_prompt(view['observation'],view['legal_actions'],labels)
                        messages = [{'role':'user','content':prompt}]
                        rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                            add_generation_prompt=True, enable_thinking=False)
                        ids = tokenizer.encode(rendered, add_special_tokens=False)
                        direct = tokenizer.apply_chat_template(messages, tokenize=True,
                            add_generation_prompt=True, enable_thinking=False)
                        if hasattr(direct, 'keys'):
                            direct = direct['input_ids']
                        if ids != direct:
                            raise ValueError('Native rendering and tokenization disagree')
                        if not rendered.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n'):
                            raise ValueError('Expected native thinking-disabled assistant prefix')
                        row = {'environment':name,'seed':seed,'chooser':chooser,'turn':env.turn,
                            'player':player,'input_tokens':len(ids),'legal_actions':len(view['legal_actions']),
                            'policy_view':view,'prompt':prompt,'rendered_prompt':rendered,
                            'input_ids':ids,'input_ids_sha256':envs.digest(ids)}
                        counts[name] += 1
                        if name not in maxima or len(ids) > maxima[name]['input_tokens']:
                            maxima[name] = row
                        if len(ids) > context_limit:
                            violations.append({k:row[k] for k in ('environment','seed','chooser','turn','player','input_tokens')})
                        selected[player] = ({'type':'run_public_tests'} if chooser=='public_tests_repeat' else
                            env.heuristic_action(player) if chooser=='heuristic' else rng.choice(view['legal_actions']))
                    envs.record_step(env, selected)
                episodes.append({'environment':name,'seed':seed,'chooser':chooser,'turns':env.turn})
    sources = envs.source_hashes()
    sources.update({p:sha(ROOT/p) for p in ('selfplay/arena_mixed/cuda_policy.py',
                                          'scripts/stress_arena_mixed_context.py')})
    return {'kind':'mixed_arena_original_tokenizer_cpu_stress','passed':not violations,
        'context_limit':context_limit,'model':cuda_policy.MODEL,'revision':cuda_policy.REVISION,
        'tokenizer_files_sha256':token_files,'metadata_sha256':sha(identities),
        'library_versions':{p:importlib.metadata.version(p) for p in ('transformers','tokenizers','chess','python-chess')},
        'source_sha256':sources,'labels':labels,'label_token_ids':label_ids,
        'thinking':False,'native_rendering_matches_direct_template_tokens':True,
        'fixture_episodes':len(episodes),'decision_views_by_environment':dict(counts),
        'maxima':maxima,'violations':violations,'fixtures':episodes,
        'scope':'Deterministic CPU fixture trajectories only: eight seeds/game, heuristic and random; coding also repeats public tests through the cap. No weights, inference, neural outcomes, API or GPU used. Empirical coverage, not a proof for every possible trajectory.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer',type=Path,required=True)
    parser.add_argument('--identities',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();value=stress(args.tokenizer,args.identities)
    with args.output.open('x') as f:
        f.write(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'path':str(args.output),'sha256':sha(args.output),'passed':value['passed'],
        'fixture_episodes':value['fixture_episodes'],'decision_views':sum(value['decision_views_by_environment'].values()),
        'max_input_tokens':{k:v['input_tokens'] for k,v in value['maxima'].items()}}))


if __name__=='__main__':main()
