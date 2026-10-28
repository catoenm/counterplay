"""Embed verified real program matches in the robot replay visualization.

Select representatives by round-robin across sources and ordered player pairings,
without looking at outcomes. Never include program source in the inline data.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selfplay.program_arena import verify_replay


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRAGMENT = Path('/Users/mitchellcatoen/.codex/visualizations/2026/10/20/01a15b4f-05a3-7053-9034-4d1bc9f8d726/selfplay-program-arena.html')


def player_name(identity):
    if isinstance(identity, str):
        return identity
    if isinstance(identity, dict):
        for key in ('name', 'id', 'label', 'identity', 'model'):
            if isinstance(identity.get(key), str):
                return identity[key]
    return json.dumps(identity, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def load_sources(paths):
    """Keep inner records unchanged, verify them, and bind all source provenance."""
    wrappers, sources, manifest = [], [], []
    names = set()
    for path in paths:
        path = Path(path)
        source_id = path.parent.name
        if source_id in names:
            raise ValueError('Source directory names must be unique')
        names.add(source_id)
        content = path.read_bytes()
        if content and not content.endswith(b'\n'):
            raise ValueError('Incomplete final JSONL record')
        plan_path = path.parent / 'plan.json'
        plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
        provenance_paths = [plan_path] + sorted(path.parent.glob('generation-*.json'))
        source = {'id': source_id, 'path': str(path.resolve()),
                  'sha256': hashlib.sha256(content).hexdigest(),
                  'model': plan.get('model'),
                  'provenance_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in provenance_paths if p.exists()},
                  'total': 0}
        roster = plan.get('roster', {})
        if roster:
            source['roster'] = {identity:{key:entry.get(key) for key in
                                ('authoring_model','authoring_version','ir_sha256','python_sha256',
                                 'generation_sha256','source_run','source_generation')}
                                for identity,entry in roster.items()}
        seen = set()
        for raw in content.splitlines():
            original = json.loads(raw)
            original_id = original['episode_id']
            if not isinstance(original_id, str) or original_id in seen:
                raise ValueError('Duplicate or invalid source episode ID')
            seen.add(original_id)
            record = original['record']
            verify_replay(record)
            models,versions = [],[]
            # Model-source metadata must agree with the exact executed IR.
            for i, identity in enumerate(original['players']):
                identity = player_name(identity)
                entry = roster.get(identity, {})
                model = entry.get('authoring_model') if roster else source['model']
                if record['origins'][i] == 'model' and not model:
                    raise ValueError('Model episode lacks a source model identity')
                if roster and (not entry or entry['ir_sha256'] != record['programs'][i]['digest']):
                    raise ValueError('Executed program differs from frozen tournament roster')
                models.append(model)
                versions.append(entry.get('authoring_version', 'original_retained'))
                program = original.get('python_programs', {}).get(identity)
                if program:
                    if (hashlib.sha256(program['python_source'].encode()).hexdigest() != program['python_sha256']
                        or program['compiled_ir'] != record['programs'][i]['source']
                        or program['ir_sha256'] != record['programs'][i]['digest']):
                        raise ValueError('Python program provenance differs from executed program')
                    if roster and (entry['python_sha256'] != program['python_sha256']
                                   or entry.get('python_program') != program):
                        raise ValueError('Python program differs from tournament authoring provenance')
            outer_id = source_id + ':' + original_id
            row = dict(original, episode_id=outer_id, visual_source=source_id,
                       visual_model=source['model'], visual_models=models, visual_versions=versions,
                       original_episode_id=original_id)
            wrappers.append(row)
            manifest.append({'episode_id': outer_id, 'original_episode_id': original_id,
                             'source': source_id, 'record_sha256': digest(record),
                             'wrapper_sha256': hashlib.sha256(raw).hexdigest()})
        source['total'] = len(seen)
        sources.append(source)
    return wrappers, sources, manifest


def display_player(wrapper, index):
    identity = player_name(wrapper['players'][index])
    if wrapper['record']['origins'][index] != 'model':
        return identity + ' bot'
    model = wrapper.get('visual_models', [wrapper.get('visual_model')]*2)[index] or 'Model'
    model = {'deepseek/deepseek-v4-pro-0813': 'DeepSeek V4 Pro 0813',
             'qwen/qwen3.8-27b': 'Qwen3.8-27B'}.get(model, model)
    # Sample numbers are authoring identities, not training checkpoints.
    sample = re.search(r'-(\d+)(?:-r(\d+))?$', identity)
    version = wrapper.get('visual_versions', ['original_retained']*2)[index]
    suffix = 'feedback '+version.removeprefix('feedback_round_') if version.startswith('feedback_round_') else 'original'
    return model + (' · P' + sample[1] if sample else ' · ' + identity) + ' · ' + suffix


def compact_state(state):
    return {'robots': [{k: r[k] for k in ('x', 'y', 'home_x', 'home_y', 'energy', 'cargo', 'score')}
                       for r in state['robots']],
            'sites': state['sites']}


def compact(wrapper):
    record = wrapper['record']
    verify_replay(record)
    events = record['events']
    return {'id': wrapper['episode_id'], 'seed': record['seed'],
            'source': wrapper.get('visual_source'), 'model': wrapper.get('visual_model'),
            'models':wrapper.get('visual_models'), 'authoring_versions':wrapper.get('visual_versions'),
            'match_kind': wrapper.get('match_kind', 'scripted'),
            'record_sha256': digest(record),
            'players': [display_player(wrapper, p) for p in (0, 1)],
            'identities': [player_name(p) for p in wrapper['players']],
            'origins': record['origins'], 'digests': [p['digest'] for p in record['programs']],
            'returns': record['returns'],
            'states': [compact_state(record['initial_state'])] + [compact_state(e['state']) for e in events],
            'decisions': [[{k: d[k] for k in ('action', 'steps', 'fault')} for d in e['decisions']] for e in events],
            'outcomes': [[e['state']['history'][-1]['outcomes'][str(p)] for p in (0, 1)] for e in events]}


def select_representatives(wrappers, maximum):
    groups = OrderedDict()
    for wrapper in wrappers:
        key = (wrapper.get('visual_source'), *[player_name(p) for p in wrapper['players']])
        groups.setdefault(key, []).append(wrapper)
    selected = []
    while groups and len(selected) < maximum:
        for key in list(groups):
            selected.append(groups[key].pop(0))
            if not groups[key]:
                del groups[key]
            if len(selected) == maximum:
                break
    return selected


def peer_first_seed(wrappers):
    """Predefined map selection across every model pair; never inspect scores."""
    first = {}
    for row in wrappers:
        source = row.get('visual_source')
        first[source] = min(first.get(source,row['record']['seed']),row['record']['seed'])
    return [row for row in wrappers if row['record']['origins'] == ['model','model']
            and row['record']['seed'] == first[row.get('visual_source')]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matches', type=Path, nargs='+',
                        default=[ROOT / 'runs/program-screen-v1/matches.jsonl'])
    parser.add_argument('--fragment', type=Path, default=DEFAULT_FRAGMENT)
    parser.add_argument('--max-episodes', type=int, default=18)
    parser.add_argument('--scripted-fallback', action='store_true')
    parser.add_argument('--peer-first-seed',action='store_true',
                        help='All model pairs on each source first seed, ordered seats; no outcome selection')
    args = parser.parse_args()
    if not 1 <= args.max_episodes <= 30:
        raise ValueError('max episodes must be 1..30')
    present = [p for p in args.matches if p.exists()]
    if len(present) != len(args.matches) and not args.scripted_fallback:
        raise ValueError('Requested source file is missing')
    wrappers, sources, manifest = load_sources(present)
    mode = 'Model program screen'
    if not wrappers and args.scripted_fallback:
        fallback = ROOT / 'runs/program-arena-prototype/representative-replay.json'
        if fallback.exists():
            record = json.loads(fallback.read_text())
            wrappers = [{'episode_id': 'scripted-nearest-value-1', 'players': ['nearest', 'value'], 'record': record}]
            sources = [{'id': 'scripted-prototype', 'path': str(fallback),
                        'sha256': hashlib.sha256(fallback.read_bytes()).hexdigest(), 'total': 1}]
            mode = 'Scripted prototype only'
    elif wrappers and not any('model' in row['record']['origins'] for row in wrappers):
        mode = 'Scripted matches only'
    ids = [row['episode_id'] for row in wrappers]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate episode IDs')
    eligible = peer_first_seed(wrappers) if args.peer_first_seed else wrappers
    selected = select_representatives(eligible, args.max_episodes)
    episodes = [compact(row) for row in selected]
    for source in sources:
        source['shown'] = sum(row.get('visual_source') == source['id'] for row in selected)
    payload = {'revision': digest(sources)[:16],
               'total': len(wrappers), 'shown': len(episodes), 'mode': mode,
               'sources': sources, 'records_manifest': manifest,
               'all_source_replays_verified': True,
               'selection': ('model pairs on each source first seed, both seats; ' if args.peer_first_seed else '')
                            +'round-robin by source and ordered pairing, input order, no outcome selection',
               'selection_label':'First seed · model pairs · both seats' if args.peer_first_seed else 'Representative ordered pairings',
               'episodes': episodes}
    encoded = json.dumps(payload, separators=(',', ':')).replace('<', '\\u003c')
    text, count = re.subn(r'(<script id="pa-data" type="application/json">).*?(</script>)',
                          lambda m: m[1] + encoded + m[2], args.fragment.read_text(), flags=re.S)
    if count != 1:
        raise ValueError('expected exactly one replay-data block')
    if len(text.encode()) > 1_000_000:
        raise ValueError('visualization exceeds 1 MB; reduce max episodes')
    args.fragment.write_text(text)
    print(json.dumps({'shown': len(episodes), 'total': len(wrappers), 'mode': mode,
                      'bytes': len(text.encode()), 'fragment': str(args.fragment)}))


if __name__ == '__main__':
    main()
