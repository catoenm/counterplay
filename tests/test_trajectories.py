from dataclasses import asdict
from collections import Counter
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from selfplay.depth import MODEL, REVISION, family, render_depth
from selfplay.environment import generate_tasks, parse_object, step
from selfplay.pilot import grade_response
from selfplay.trajectories import augment, build


class TrajectoryTests(unittest.TestCase):
    def test_build_keeps_parents_together_and_matches_example_budget(self):
        unique={family(t):t for t in generate_tasks(180,seed=3301,depths=(2,3,4,5))}
        tasks=list(unique.values())
        selected={'candidates':tasks[:128],'validation':tasks[128:136],
                  'test':tasks[136:140],'test_ood':tasks[140:144]}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);splits=root/'splits';splits.mkdir()
            split_manifest={'split_files':{}}
            for name,group in selected.items():
                specs=[{**asdict(t),'task_id':t.identity,'family_id':family(t),
                        'depth':t.max_steps,'render_seed':i} for i,t in enumerate(group)]
                text=''.join(json.dumps(r)+'\n' for r in specs)
                (splits/(name+'.jsonl')).write_text(text)
                split_manifest['split_files'][name]={'sha256':hashlib.sha256(text.encode()).hexdigest()}
                if name not in ('candidates','validation'):
                    continue
                run=root/name;run.mkdir();(run/'tasks.jsonl').write_text(text)
                (run/'manifest.json').write_text(json.dumps({'config':{'attempts':2},'model':{
                    'backend':'mlx','model':MODEL,'snapshot':REVISION,'thinking':False,'adapter_path':None}}))
                (run/'summary.json').write_text(json.dumps({'complete':True}))
                rows=[]
                # Synthetic fixture responses, never used as experiment data.
                for spec,task in zip(specs,group):
                    prompt,mapping=render_depth(spec,task)
                    inverse={a:l for l,a in mapping.items()}
                    for attempt in (0,1):
                        rows.append({'task_id':task.identity,'attempt':attempt,'format':'strict',
                                     'variant':'canonical','prompt':prompt,'finish_reason':'stop','success':True,
                                     'text':json.dumps({'plan':[inverse[a] for a in task.witness]})})
                (run/'rollouts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (splits/'manifest.json').write_text(json.dumps(split_manifest))
            args=SimpleNamespace(splits=splits,candidates=root/'candidates',validation=root/'validation',out=root/'data')
            with contextlib.redirect_stdout(io.StringIO()):
                build(args)
            datasets={arm:[json.loads(x) for x in (args.out/arm/'train.jsonl').read_text().splitlines()]
                      for arm in ('plans','mixed')}
            parents={t.identity for t in selected['candidates']}
            for arm,rows in datasets.items():
                self.assertEqual(len(rows),1024)
                self.assertEqual({r['task_id'] for r in rows},parents)
                self.assertEqual(set(Counter(r['task_id'] for r in rows).values()),{8})
            self.assertEqual(Counter(r['kind'] for r in datasets['mixed']),{'plan':512,'state_trajectory':512})
            self.assertEqual((args.out/'plans/valid.jsonl').read_bytes(),(args.out/'mixed/valid.jsonl').read_bytes())
            rollout=args.candidates/'rollouts.jsonl'
            rollout.write_text('\n'.join(rollout.read_text().splitlines()[:-1])+'\n')
            args.out=root/'bad-data'
            with self.assertRaisesRegex(ValueError,'Missing collection attempts'):
                build(args)

    def test_all_presentations_preserve_plan_and_each_transition(self):
        for task in generate_tasks(8, seed=2901, depths=(2,3,4,5)):
            spec={**asdict(task),'task_id':task.identity,'family_id':family(task),
                  'depth':task.max_steps,'render_seed':203}
            prompt,mapping=render_depth(spec,task)
            inverse={a:l for l,a in mapping.items()}
            response={'prompt':prompt,'text':json.dumps({'plan':[inverse[a] for a in task.witness]}),
                      'finish_reason':'stop','attempt':0}
            for variant in ('canonical','reordered','renamed','renamed_reordered'):
                plan,trace=augment(spec,task,response,variant)
                _,renamed=render_depth(spec,task,variant)
                self.assertTrue(grade_response(task,{'text':plan['answer'],'finish_reason':'stop'},renamed)['success'])
                self.assertNotIn('Target',trace['prompt'])
                self.assertEqual(trace['task_id'],plan['task_id'])
                self.assertEqual(len(parse_object(trace['answer'])['states']),task.max_steps)
                state=task.start
                for action,expected in zip(trace['environment_actions'],trace['verified_states']):
                    state=step(state,action,task.modulus)
                    self.assertEqual(list(state),expected)
                self.assertEqual(state,task.goal)

    def test_wrong_plan_and_changed_prompt_are_rejected(self):
        task=generate_tasks(1,seed=1001,depths=(3,))[0]
        spec={**asdict(task),'task_id':task.identity,'family_id':family(task),
              'depth':3,'render_seed':17}
        prompt,_=render_depth(spec,task)
        response={'prompt':prompt,'text':'{"plan":[]}','finish_reason':'stop','attempt':0}
        with self.assertRaisesRegex(ValueError,'Unverified'):
            augment(spec,task,response,'renamed')
        response['prompt']='different puzzle'
        with self.assertRaisesRegex(ValueError,'Unverified'):
            augment(spec,task,response,'canonical')


if __name__=='__main__':
    unittest.main()
