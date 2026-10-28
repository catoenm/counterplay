"""Run the frozen trajectory-pilot evaluations sequentially, then compare."""
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selfplay.curriculum import check_splits
from scripts.analyze_depth import compare


def main():
    split_dir=Path('experiments/trajectory-v1')
    check_splits(split_dir)
    dataset=json.loads(Path('datasets/trajectory-v1/manifest.json').read_text())
    updates=dataset['updates']
    for arm in ('plans','mixed'):
        summary=json.loads(Path(f'adapters/trajectory-{arm}/summary.json').read_text())
        if not summary['complete'] or summary['updates']!=updates:
            raise ValueError('Training has not completed at the declared update count')
    for split,suffix,seed in [('test','',1210013),('test_ood','-ood',1310013)]:
        paths=[]
        for arm in ('base','plans','mixed'):
            out=Path(f'runs/trajectory-eval{suffix}-{arm}')
            paths.append(out)
            # Start with new directories; a partial run must be explicitly resumed.
            command=[sys.executable,'-m','selfplay.depth','collect',
                     '--tasks',str(split_dir/(split+'.jsonl')),'--out',str(out),
                     '--attempts','4','--formats','strict','--variants','canonical,renamed_reordered',
                     '--batch-size','16','--max-minutes','20','--seed',str(seed)]
            if arm!='base':
                command+=['--adapter',f'adapters/trajectory-{arm}']
            print('Starting '+str(out),flush=True)
            subprocess.run(command,check=True)
        result=compare(*paths,arm_names=('plans','mixed'),expected_updates=updates)
        Path(f'runs/trajectory{suffix}-comparison.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({'split':split,'accuracy':{
            arm:{v:stats['pass_at_1'] for v,stats in groups.items()}
            for arm,groups in result['models'].items()}},indent=2),flush=True)


if __name__=='__main__':
    main()
