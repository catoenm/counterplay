"""CPU-only, provenance-checked export of the completed local coding-RL pilot.

The simulator's immutable inner records describe execution only. Outer metadata
binds their sampled programs to actual model checkpoints and the RLOO update.
No calibration games or invalid-program stand-ins are displayed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from cloud.train_program_rl import audit, classify, digest as file_digest, load, verify_sources
from export_program_visual import compact, digest

DEFAULT_RUN = ROOT / 'runs/program-local-27b-compact-v1'
DEFAULT_FRAGMENT = Path('/Users/mitchellcatoen/.codex/visualizations/2026/10/20/'
                        '01a15b4f-05a3-7053-9034-4d1bc9f8d726/coding-rl-replay.html')


def validate_invalid(row):
    if not row['classification']['valid']:
        if row['matches'] or row['game_return'] is not None or row['reward_kind'] != 'invalid_authorship':
            raise ValueError('Invalid authorship must have no game')


def compact_match(row, match, candidate, rival, relative, phase, index):
    validate_invalid(row)
    if not row['classification']['valid']:
        raise ValueError('Invalid authorship cannot become a replay')
    record, seat = match['replay'], match['candidate_seat']
    if type(seat) is not int or seat not in (0, 1):
        raise ValueError('Invalid candidate seat')
    participants = [candidate, rival] if seat == 0 else [rival, candidate]
    for p, controller in enumerate(participants):
        if controller['ir_sha256'] != record['programs'][p]['digest']:
            raise ValueError('Controller identity differs from executed program')
    identity = f'{phase}:{row["generation"]}:{index}'
    wrapper = {'episode_id': identity, 'players': [p['id'] for p in participants],
               'record': record, 'visual_source': relative,
               'visual_models': [p.get('model') for p in participants], 'match_kind': phase}
    result = compact(wrapper)  # Replays every event before compacting.
    result.update(players=[p['label'] for p in participants],
                  checkpoint_labels=[p['checkpoint'] for p in participants],
                  phase=phase, candidate_seat=seat, candidate=row['generation'],
                  candidate_paired_reward=row['reward'])
    return result


def build_payload(run, independent_audit):
    run, independent_audit = Path(run), Path(independent_audit)
    manifest, status = verify_sources(run)
    result = audit(run)
    if not result['complete'] or result != load(independent_audit):
        raise ValueError('Independent audit does not match current complete artifacts')
    config = manifest['config']
    if config['iterations'] != 1 or config['condition'] != 'feedback-compact-r1-v1':
        raise ValueError('This visual requires the completed single-update compact-context pilot')
    if status.get('final_checkpoint') != 'step1':
        raise ValueError('Unexpected final checkpoint')
    files = {str(run / name): value for name, value in status['artifacts'].items()}
    files.update({str(run / 'manifest.json'): file_digest(run / 'manifest.json'),
                  str(run / 'status.json'): file_digest(run / 'status.json'),
                  str(run / 'prompt.txt'): file_digest(run / 'prompt.txt'),
                  str(independent_audit): file_digest(independent_audit)})

    def bound(relative):
        path = run / relative
        if relative not in status['artifacts'] or file_digest(path) != status['artifacts'][relative]:
            raise ValueError('Uncommitted or changed artifact: ' + relative)
        return load(path)

    checkpoints = {label: bound(f'checkpoints/{label}/metadata.json') for label in ('base', 'step1')}
    update = checkpoints['step1']['report']
    controllers = {}

    def authored(name, checkpoint, label):
        generation = bound(f'generations/{name}.json')
        if generation['checkpoint'] != checkpoints[checkpoint]:
            raise ValueError('Generation checkpoint differs from advertised checkpoint')
        classification, program = classify(generation['text'], generation['finish_reason'])
        if not classification['valid']:
            raise ValueError('A displayed controller did not compile')
        data = classification['program']
        entry = {'id': name, 'label': label, 'checkpoint': checkpoint,
                 'model': config['model_repo'], 'model_revision': config['model_revision'],
                 'generation_sha256': status['artifacts'][f'generations/{name}.json'],
                 'adapter_sha256': checkpoints[checkpoint]['files']['adapter.safetensors'],
                 'python_sha256': data['python_sha256'], 'ir_sha256': data['ir_sha256'],
                 'python_source': data['python_source'], 'origin': 'model'}
        controllers[name] = entry
        return entry

    opponent_file = bound('groups/1/opponents.json')
    selected = opponent_file['selected']
    if not opponent_file['passed'] or len(selected) != config['opponent_count']:
        raise ValueError('Selected current-model opponent gate did not pass')
    for opponent in selected:
        name = opponent['id']
        entry = authored(name, 'base', 'Base · frozen opponent draw ' + str(int(name.rsplit('-', 1)[1]) + 1))
        if (opponent['source_checkpoint'] != checkpoints['base']
            or opponent['generation_sha256'] != entry['generation_sha256']
            or digest(opponent['compiled_ir']) != entry['ir_sha256']):
            raise ValueError('Selected opponent provenance differs')

    episodes, record_manifest, training_rows = [], [], []
    evaluations = {}
    for relative, phase, key, checkpoint in [
        ('groups/1/scores.json', 'training', 'rows', 'base'),
        ('evaluation/base.json', 'before', 'scores', 'base'),
        ('evaluation/step1.json', 'after', 'scores', 'step1')]:
        container = bound(relative)
        expected_checkpoint = container.get('source', container.get('checkpoint'))
        if expected_checkpoint != checkpoints[checkpoint]:
            raise ValueError('Scored container has another source checkpoint')
        if phase != 'training' and container['training_use'] is not False:
            raise ValueError('Evaluation marked for training')
        for opponent in container['opponents']:
            if opponent['origin'] == 'model':
                if opponent not in selected:
                    raise ValueError('Evaluation opponent was not frozen before training')
            else:
                controllers[opponent['id']] = {'id': opponent['id'], 'label': opponent['id'].title()+' bot',
                    'checkpoint': None, 'model': None, 'origin': 'scripted',
                    'ir_sha256': digest(opponent['compiled_ir']), 'python_source': None}
        for index, row in enumerate(container[key]):
            validate_invalid(row)
            generation = bound(f'generations/{row["generation"]}.json')
            if generation['checkpoint'] != checkpoints[checkpoint]:
                raise ValueError('Scored draw differs from source checkpoint')
            if phase == 'training':
                training_rows.append({'id': row['generation'], 'draw': index+1,
                    'valid': row['classification']['valid'], 'reward': row['reward'],
                    'advantage': update['advantages'][index], 'tokens': len(generation['completion_ids']),
                    'reason': row['classification'].get('reason'), 'matches': len(row['matches'])})
            if not row['classification']['valid']:
                continue
            label = ('Base' if checkpoint == 'base' else 'Step 1')+' · '+('training' if phase == 'training' else 'evaluation')+f' draw {index+1}'
            candidate = authored(row['generation'], checkpoint, label)
            for match_index, match in enumerate(row['matches']):
                episode = compact_match(row, match, candidate, controllers[match['opponent_id']], relative, phase, match_index)
                episodes.append(episode)
                record_manifest.append({'episode_id': episode['id'], 'record_sha256': episode['record_sha256'],
                    'source_path': relative, 'source_sha256': status['artifacts'][relative],
                    'generation': row['generation'], 'match_index': match_index})
        if phase != 'training':
            matches = [match for row in container[key] for match in row['matches']]
            evaluations[checkpoint] = {'program_draws': len(container[key]), 'maps': len(container['seeds']),
                'opponents': len(container['opponents']), 'games': len(matches),
                'mean_reward': sum(row['reward'] for row in container[key])/len(container[key]),
                'banked_points': sum(match['replay']['metrics']['scores'][match['candidate_seat']] for match in matches),
                'program_digests': [row['classification']['program']['ir_sha256'] for row in container[key]
                                    if row['classification']['valid']]}
    if len(episodes) != result['replayed_matches'] or len({e['id'] for e in episodes}) != len(episodes):
        raise ValueError('Actual training/evaluation match coverage differs')
    if sum(row['tokens'] for row in training_rows) != update['training_tokens']:
        raise ValueError('Displayed training tokens differ from update')
    if {e['seed'] for e in episodes if e['phase']=='training'} & {e['seed'] for e in episodes if e['phase']!='training'}:
        raise ValueError('Training and evaluation maps overlap')
    # Fail if any source changed while the export was being assembled.
    for path, expected in files.items():
        if file_digest(path) != expected:
            raise ValueError('Artifact changed during export: '+path)
    return {'revision': digest(files)[:16], 'run': run.name,
            'total': len(episodes), 'shown': len(episodes), 'episodes': episodes,
            'model': config['model_repo'], 'condition': config['condition'],
            'prompt_sha256': manifest['prompt_sha256'], 'source_files_sha256': files,
            'records_manifest': record_manifest, 'controllers': controllers,
            'checkpoints': {k:{'adapter_sha256':v['files']['adapter.safetensors']} for k,v in checkpoints.items()},
            'optimizer_steps': update['optimizer_steps'], 'training_tokens': update['training_tokens'],
            'training_rows': training_rows, 'evaluations': evaluations,
            'same_evaluation_program': evaluations['base']['program_digests']==evaluations['step1']['program_digests'],
            'selection': 'All completed training and held-out evaluation games; no opponent-calibration games; no outcome selection',
            'source_archive_verified': result['source_archive_verified'],
            'all_selected_replays_verified': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--audit', type=Path, default=ROOT/'runs/program-local-27b-compact-v1-independent-audit.json')
    parser.add_argument('--fragment', type=Path, default=DEFAULT_FRAGMENT)
    parser.add_argument('--out', type=Path, default=ROOT/'runs/program-local-27b-compact-visual/payload.json')
    args = parser.parse_args()
    payload = build_payload(args.run, args.audit)
    encoded = json.dumps(payload, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    text, count = re.subn(r'(<script id="cr-data" type="application/json">).*?(</script>)',
                         lambda m:m[1]+encoded+m[2], args.fragment.read_text(), flags=re.S)
    if count != 1 or len(text.encode()) >= 1_000_000:
        raise ValueError('Expected one data block and fragment under 1 MB')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    args.fragment.write_text(text)
    print(json.dumps({'episodes':len(payload['episodes']), 'states':sum(len(e['states']) for e in payload['episodes']),
                      'bytes':len(text.encode()), 'fragment':str(args.fragment)}))


if __name__ == '__main__':
    main()
