"""Reporting-only export of an independently audited failed warmstart gate.

Recompile original sampled controllers and replay all calibration records with
archived bounded interpreters. Select each compiled sample's first map/reference
in both seats, in original sample order, without selecting on game outcomes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_cardinal_capability import diagnose, read, require, sha, write_new
from scripts.export_program_visual import compact


def export(run, audit_path, template, fragment, output):
    run = Path(run).resolve()
    audit_path, template, fragment, output = map(Path, (audit_path, template, fragment, output))
    require(not fragment.exists() and not output.exists(), 'Refusing to overwrite reporting artifacts')
    audit = read(audit_path)
    require(audit['integrity_passed'] is True, 'Independent audit must pass')
    require(audit['recorded_state'] == 'capability_failed', 'Expected failed capability gate')
    require(not audit['planned_study_complete'] and not audit['planned_comparisons_available'],
            'This export is calibration only')
    require(audit['genuine_supervised_optimizer_steps'] == 32 and audit['genuine_RL_optimizer_steps'] == 0,
            'Unexpected training lineage')
    require(audit['manifest_sha256'] == sha(run / 'manifest.json')
            and audit['status_sha256'] == sha(run / 'status.json'), 'Stale independent audit')
    report, _ = diagnose(run)
    report.update(kind='warmstart_capability_replay_reporting',
                  scope='Terminal calibration after 32 supervised updates and zero RL updates; no heldout comparison',
                  independent_audit_sha256=sha(audit_path), reporting_script_sha256=sha(__file__))
    capability, manifest = read(run / 'capability.json'), read(run / 'manifest.json')
    require(capability['passed'] is False and capability['productive_indices'] == [], 'Unexpected gate outcome')
    require(len(capability['rows']) == 8 and report['verified_games'] == audit['verified_unique_game_replays'] == 120,
            'Unexpected calibration denominator')
    valid = [row for row in capability['rows'] if row['classification']['valid']]
    require([row['generation'] for row in valid] == ['capability-0', 'capability-1', 'capability-5',
                                                   'capability-6', 'capability-7'], 'Sample order changed')
    require(all(row['banked_point_yield'] == 0 and row['positive_banking_games'] == 0
                for row in capability['rows']), 'Nonzero candidate banking')
    for relative, digest in manifest['source_sha256'].items():
        if relative.startswith('selfplay/program_arena/'):
            require(sha(ROOT / relative) == digest, 'Compact renderer differs from archived game')
    episodes, selected = [], []
    for row in valid:
        first = row['matches'][0]
        matches = [m for m in row['matches'] if m['replay']['seed'] == first['replay']['seed']
                   and m['opponent_id'] == first['opponent_id']]
        require(len(matches) == 2 and [m['candidate_seat'] for m in matches] == [0, 1],
                'First recorded map must have both ordered seats')
        for match in matches:
            name, seat = row['generation'], match['candidate_seat']
            players = [match['opponent_id']] * 2
            players[seat] = name
            wrapper = {'episode_id': f"{name}:{match['replay']['seed']}:{match['opponent_id']}:seat{seat}",
                       'players': players, 'record': match['replay'], 'visual_source': run.name,
                       'visual_model': manifest['config']['model_repo'],
                       'visual_models': [manifest['config']['model_repo'] if p == seat else None for p in (0, 1)],
                       'visual_versions': ['after_32_supervised_updates' if p == seat else 'scripted' for p in (0, 1)]}
            episode = compact(wrapper)
            number = int(name.split('-')[-1])
            episode['players'] = [f'Qwen3.8-27B · sample {number}' if p == seat else match['opponent_id'] + ' bot'
                                  for p in (0, 1)]
            episode['candidate_seat'] = seat
            episode['sample_index'] = number
            episode['option_label'] = f'Sample {number} · model {"A" if seat == 0 else "B"} · nearest bot'
            episodes.append(episode)
            selected.append({'id': episode['id'], 'generation': name, 'candidate_seat': seat,
                             'record_sha256': episode['record_sha256'], 'python_sha256': row['classification']['program']['python_sha256'],
                             'ir_sha256': row['classification']['program']['ir_sha256']})
    payload = {'revision': sha(audit_path)[:16], 'total': 120, 'shown': len(episodes),
               'mode': '32 supervised updates · 0 RL updates · Calibration only, no heldout comparison',
               'coverage_label': '0/8 productive · 5 compiled / 3 invalid · Model banks 0 points in 120 games',
               'selection_label': '10 recorded matches · All 5 compiled samples · First map, both seats',
               'selection': 'Original sample order, first recorded calibration map and reference opponent, both seats. All 120 source games independently replayed. Invalid samples 2, 3, 4 have no games. No heldout comparison.',
               'all_source_replays_verified': True, 'independent_audit_sha256': sha(audit_path),
               'source_sha256': report['input_sha256'], 'episodes': episodes}
    text = template.read_text().replace('cardinal-teacher-replay', 'cardinal-warmstart-capability-replay')
    text = text.replace('A teacher policy collecting and banking', 'After supervised training: still no banking')
    text = text.replace('`${data.shown} of ${data.total} matches · all source replays verified · no gradient training`',
                        'data.coverage_label')
    text = text.replace("data.mode+' · Scripted reference'", 'data.mode')
    text = text.replace(' · no gradient training`', ' · calibration, no heldout comparison`')
    text = text.replace('option.textContent=`${e.players[0]} vs ${e.players[1]} · seed ${e.seed}`;',
                        'option.textContent=e.option_label;')
    encoded = json.dumps(payload, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    text, count = re.subn(r'(<script id="pa-data" type="application/json">).*?(</script>)',
                         lambda m: m[1] + encoded + m[2], text, flags=re.S)
    require(count == 1 and len(text.encode()) < 1_000_000, 'Invalid fragment size or data block')
    require('no gradient training' not in text and '<html' not in text.lower(), 'Stale labels or document wrapper')
    require(sha(run / 'status.json') == audit['status_sha256'] and sha(audit_path) == payload['independent_audit_sha256'],
            'Source changed during reporting')
    write_new(fragment, text)
    report['visual'] = {'fragment': str(fragment.resolve()), 'sha256': sha(fragment), 'shown': len(episodes),
                       'selection': payload['selection'], 'selected': selected, 'template_sha256': sha(template),
                       'compact_exporter_sha256': sha(ROOT / 'scripts/export_program_visual.py')}
    write_new(output, json.dumps(report, indent=2, allow_nan=False) + '\n')
    return {'fragment': str(fragment.resolve()), 'report': str(output.resolve()), 'sha256': sha(fragment),
            'verified_source_games': 120, 'shown_games': len(episodes)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'audit', 'template', 'fragment', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(export(args.run, args.audit, args.template, args.fragment, args.output)))


if __name__ == '__main__':
    main()
