import unittest
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from selfplay.dataset import load_verified, split_ids
from selfplay.environment import generate_tasks, render


class DatasetTests(unittest.TestCase):
    def test_grouped_split(self):
        train, valid = split_ids([str(i // 4) for i in range(40)], 71)
        self.assertFalse(set(train) & set(valid))
        self.assertEqual(len(train) + len(valid), 10)
        self.assertEqual((train, valid), split_ids([str(i) for i in range(10)], 71))

    def test_too_few_tasks(self):
        with self.assertRaises(ValueError): split_ids(["a", "b"], 1)

    def test_reject_oracle_supervision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(json.dumps({"model": {"backend": "oracle"}}))
            with self.assertRaises(ValueError): load_verified(root)

    def test_rechecks_grade_and_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = generate_tasks(1)[0]
            prompt, mapping = render(task, seed=17)
            inverse = {a: label for label, a in mapping.items()}
            answer = json.dumps({"plan": [inverse[a] for a in task.witness]})
            (root / "manifest.json").write_text(json.dumps({"model": {"backend": "mlx", "thinking": False}, "config": {"seed": 17}}))
            (root / "tasks.jsonl").write_text(json.dumps({**asdict(task), "task_id": task.identity}) + "\n")
            row = {"task_id": task.identity, "variant": "canonical", "prompt": prompt, "text": answer, "success": True}
            path = root / "rollouts.jsonl"
            path.write_text(json.dumps(row) + "\n")
            examples, _ = load_verified(root)
            self.assertEqual(len(examples), 1)
            path.write_text(json.dumps({**row, "success": False}) + "\n")
            with self.assertRaises(ValueError): load_verified(root)
            path.write_text(json.dumps({**row, "prompt": "different task"}) + "\n")
            with self.assertRaises(ValueError): load_verified(root)


if __name__ == "__main__":
    unittest.main()
