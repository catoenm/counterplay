import contextlib
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from selfplay.curriculum import prepare
from selfplay.depth import MODEL, REVISION, family, render_depth
from selfplay.environment import generate_tasks


class CurriculumTests(unittest.TestCase):
    def test_independent_answers_and_disjoint_depth_matched_sets(self):
        tasks, seen = [], set()
        for task in generate_tasks(80, seed=1967, depths=(2, 3, 4, 5)):
            if family(task) not in seen:
                seen.add(family(task)); tasks.append(task)
        specs = {"candidates": tasks[:32], "validation": tasks[32:40],
                 "test": tasks[40:44], "test_ood": tasks[44:48]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits = root / "splits"; splits.mkdir()
            manifest = {"split_files": {}}
            for name, selected in specs.items():
                rows = [{**asdict(t), "task_id": t.identity, "family_id": family(t),
                         "depth": len(t.shortest_plan()), "render_seed": i} for i, t in enumerate(selected)]
                text = "".join(json.dumps(r) + "\n" for r in rows)
                (splits / (name + ".jsonl")).write_text(text)
                manifest["split_files"][name] = {"sha256": hashlib.sha256(text.encode()).hexdigest()}
                if name not in ("candidates", "validation"):
                    continue
                run = root / name; run.mkdir()
                (run / "tasks.jsonl").write_text(text)
                (run / "summary.json").write_text(json.dumps({"complete": True}))
                (run / "manifest.json").write_text(json.dumps({"model": {
                    "model": MODEL, "snapshot": REVISION, "backend": "mlx", "thinking": False, "adapter_path": None}}))
                rollouts = []
                for i, (spec, task) in enumerate(zip(rows, selected)):
                    prompt, mapping = render_depth(spec, task)
                    inverse = {action: label for label, action in mapping.items()}
                    answer = {"plan": [inverse[a] for a in task.witness]}
                    for attempt in range(16):
                        # First four candidates solve only in the score stage:
                        # they must not provide training supervision.
                        solved = attempt == 0 or (attempt == 8 and (name == "validation" or i >= 4))
                        rollouts.append({"task_id": task.identity, "attempt": attempt, "format": "strict",
                                         "variant": "canonical", "prompt": prompt, "finish_reason": "stop",
                                         "text": json.dumps(answer if solved else {"plan": []}), "success": solved})
                (run / "rollouts.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rollouts))
            (splits / "manifest.json").write_text(json.dumps(manifest))
            out = root / "data"
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(SimpleNamespace(candidates=root/"candidates", validation=root/"validation", splits=splits, out=out))
            result = json.loads((out / "manifest.json").read_text())
            self.assertEqual(result["per_arm_tasks"], 20)
            forbidden = {t.identity for t in tasks[:4]}
            depths = []
            for arm in ("uniform", "frontier"):
                rows = [json.loads(x) for x in (out / arm / "train.jsonl").read_text().splitlines()]
                self.assertFalse(forbidden & {r["task_id"] for r in rows})
                self.assertEqual(len({r["family_id"] for r in rows}), 20)
                depths.append(sorted(r["depth"] for r in rows))
            self.assertEqual(depths[0], depths[1])


if __name__ == "__main__":
    unittest.main()
