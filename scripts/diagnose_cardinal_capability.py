"""CPU-only behavioral audit and deterministic replay export of a failed gate.

Recompile with the hash-verified archived frontend; execute only bounded IR.
The selected replay is the first compiled draw, first scheduled map/opponent,
both seats. No reward/outcome selection and no generated Python execution.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_program_cardinal_cycle import archived_runtime
from scripts.export_program_visual import compact


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def write_new(path, text):
    with Path(path).open('x') as stream:
        stream.write(text)


def literal_lowering_check(program):
    original = ast.parse(program['python_source'])
    lowered = ast.parse(program['lowered_python_source'])
    mapping = {'north': {'action': 'move', 'dx': 0, 'dy': -1},
        'south': {'action': 'move', 'dx': 0, 'dy': 1},
        'east': {'action': 'move', 'dx': 1, 'dy': 0},
        'west': {'action': 'move', 'dx': -1, 'dy': 0},
        **{name: {'action': name} for name in ('rest', 'harvest', 'deposit')}}
    for tree in (original, lowered):
        for node in ast.walk(tree):
            if isinstance(node, ast.Return):
                action = mapping[ast.literal_eval(node.value)] if tree is original else ast.literal_eval(node.value)
                # Canonicalize only literal return payloads; all control flow
                # and expressions must remain structurally identical.
                node.value = ast.Constant(json.dumps(action, sort_keys=True))
    require(ast.dump(original) == ast.dump(lowered), 'Translation changed more than literal returns')


def diagnose(run):
    run = Path(run).resolve()
    manifest, status, capability = (read(run / name) for name in ('manifest.json', 'status.json', 'capability.json'))
    require(status['state'] == 'capability_failed', 'Only terminal failed capability gates are supported')
    require(sha(run / 'manifest.json') == status['manifest_sha256'], 'Manifest journal mismatch')
    inputs = {name: sha(run / name) for name in ('manifest.json', 'status.json', 'capability.json')}
    require(inputs['capability.json'] == status['artifacts']['capability.json'], 'Capability journal mismatch')
    rows, selected, total = [], [], 0
    with archived_runtime(run, manifest) as runtime:
        for row in capability['rows']:
            name = row['generation']; relative = 'generations/' + name + '.json'
            inputs[relative] = sha(run / relative)
            require(inputs[relative] == row['generation_sha256'] == status['artifacts'][relative], 'Generation journal mismatch')
            generation = read(run / relative)
            classification, compiled = runtime.classify(generation['text'], generation['finish_reason'])
            require(classification == row['classification'], 'Archived reclassification mismatch')
            summary = {'generation': name, 'authorship_valid': classification['valid'],
                'generation_sha256': inputs[relative], 'classification': {k: v for k, v in classification.items() if k != 'program'}}
            if not compiled:
                require(not row['matches'], 'Invalid authoring has recorded games')
                rows.append(summary)
                continue
            literal_lowering_check(classification['program'])
            totals, outcomes, actions = Counter(), Counter(), Counter()
            peaks, positions, harvest_games = [], [], []
            for match in row['matches']:
                record, seat = match['replay'], match['candidate_seat']
                runtime.verify_replay(record)
                require(record['programs'][seat]['digest'] == compiled.digest, 'Candidate IR mismatch')
                metrics = record['metrics']
                totals.update({k: metrics[k][seat] for k in ('scores', 'banked_units', 'harvested_units',
                    'carried_value', 'failed_actions', 'deposits', 'moves', 'rests', 'program_faults')})
                robots = [state['robots'][seat] for state in [record['initial_state']] + [e['state'] for e in record['events']]]
                peaks.append(max(len(r['cargo']) for r in robots))
                positions.append(len({(r['x'], r['y']) for r in robots}))
                if metrics['harvested_units'][seat]:
                    harvest_games.append({'seed': record['seed'], 'opponent': match['opponent_id'], 'seat': seat,
                        'harvested_units': metrics['harvested_units'][seat], 'unbanked_value': metrics['carried_value'][seat]})
                for event in record['events']:
                    outcomes[event['state']['history'][-1]['outcomes'][str(seat)]] += 1
                    actions[event['decisions'][seat]['action']['action']] += 1
            summary.update(games=len(row['matches']), totals=dict(totals), outcomes=dict(outcomes),
                actions=dict(actions), peak_cargo=max(peaks), harvest_games=harvest_games,
                unique_positions_minmax=[min(positions), max(positions)],
                python_sha256=classification['program']['python_sha256'], ir_sha256=compiled.digest,
                literal_action_translation_verified=True)
            total += len(row['matches']); rows.append(summary)
            if not selected:
                first = row['matches'][0]
                selected = [(name, m) for m in row['matches'] if m['replay']['seed'] == first['replay']['seed']
                    and m['opponent_id'] == first['opponent_id']]
    require(len(selected) == 2 and {m['candidate_seat'] for _, m in selected} == {0, 1}, 'Expected both selected seats')
    require(sha(run / 'status.json') == inputs['status.json'], 'Journal changed during audit')
    return {'kind': 'cardinal_capability_behavior_diagnostic', 'format_version': 1, 'run': str(run),
        'scope': 'terminal capability only; no training or performance-comparison claim',
        'method': 'Reclassify original outputs using hash-verified archived frontend; independently compare ASTs outside literal-return lowering; verify every original bounded-IR replay; count candidate-seat outcomes only.',
        'verified_games': total, 'prescribed_authorships': len(rows), 'rows': rows,
        'input_sha256': inputs, 'archived_source_sha256': manifest['source_sha256'],
        'diagnostic_script_sha256': sha(__file__), 'generated_python_executed': False}, selected


def export(run, report, selected, template, destination):
    manifest = read(Path(run) / 'manifest.json')
    # The shared compact renderer verifies with the active game. Bind it to
    # the identical archived game before using that convenience routine.
    for relative, digest in manifest['source_sha256'].items():
        if relative.startswith('selfplay/program_arena/'):
            require(sha(ROOT / relative) == digest, 'Renderer game differs from archived game')
    episodes = []
    for name, match in selected:
        seat = match['candidate_seat']; players = [match['opponent_id']] * 2; players[seat] = name
        wrapper = {'episode_id': f"{name}:{match['replay']['seed']}:{match['opponent_id']}:seat{seat}",
            'players': players, 'record': match['replay'], 'visual_source': Path(run).name,
            'visual_model': manifest['config']['model_repo'],
            'visual_models': [manifest['config']['model_repo'] if p == seat else None for p in (0, 1)]}
        episode = compact(wrapper)
        episode['players'] = [f'Qwen3.8-27B · {name} · base' if p == seat else match['opponent_id'] + ' bot' for p in (0, 1)]
        episodes.append(episode)
    payload = {'revision': report['input_sha256']['capability.json'][:16], 'total': report['verified_games'],
        'shown': 2, 'mode': 'Failed capability gate', 'all_source_replays_verified': True,
        'selection': 'First compiled capability draw, first recorded calibration map and scripted opponent, both seats; no outcome selection.',
        'selection_label': 'Capability 0 · First map · Both seats · Gate: 0/8 productive',
        'source_sha256': report['input_sha256'], 'episodes': episodes}
    text = Path(template).read_text().replace('program-arena-replay', 'cardinal-capability-replay')
    text = text.replace('Programs competing for resources', 'A controller that never leaves home')
    encoded = json.dumps(payload, separators=(',', ':')).replace('<', '\\u003c')
    text, count = re.subn(r'(<script id="pa-data" type="application/json">).*?(</script>)',
        lambda m: m[1] + encoded + m[2], text, flags=re.S)
    require(count == 1 and len(text.encode()) < 1_000_000, 'Invalid replay fragment')
    write_new(destination, text)
    return {'fragment': str(Path(destination).resolve()), 'sha256': sha(destination),
        'template_sha256': sha(template), 'compact_exporter_sha256': sha(ROOT / 'scripts/export_program_visual.py')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--template', type=Path)
    parser.add_argument('--fragment', type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing to overwrite diagnostic')
    require(bool(args.template) == bool(args.fragment), 'Both template and fragment are required for export')
    report, selected = diagnose(args.run)
    if args.fragment:
        report['visual'] = export(args.run, report, selected, args.template, args.fragment)
    write_new(args.output, json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'verified_games': report['verified_games'],
        'sha256': sha(args.output), 'visual': report.get('visual')}))


if __name__ == '__main__':
    main()
