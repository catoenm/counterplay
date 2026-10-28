"""Reporting-only fixtures; these are not experimental observations."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.plot_program_study import (CONDITIONS, ENDPOINTS, bind_audit_inputs,
                                       extract_complete, verify_bound_inputs)


def fixture():
    result = {"kind": "independent_cpu_matched_program_study_audit", "integrity_passed": True,
              "planned_study_complete": True, "planned_comparisons_available": True,
              "evaluation_draws": 36, "required_evaluation_draws": 36,
              "evaluation": {}, "contrasts": {}}
    for condition, n in CONDITIONS.items():
        for endpoint in ENDPOINTS:
            for split in ("validation", "holdout"):
                rows = []
                for i in range(n):
                    authored = i >= 1
                    valid = i >= 2
                    rows.append({"generation": f"eval-{endpoint}-{condition}-{i}",
                                 "banked_point_yield": 8.0 if i == n-1 else 0.0,
                                 "authorship_valid": authored, "runtime_valid": valid,
                                 "candidate_program_faults": 1 if i == 1 else 0,
                                 "games": (48 if split == "holdout" else 32) if authored else 0})
                result["evaluation"][f"{endpoint}/{condition}/{split}"] = {
                    "complete": True, "prescribed_program_clusters": n, "observed_program_clusters": n,
                    "missing_program_clusters": 0, "programs": rows,
                    "primary_mean_banked_point_yield": 8/n, "compiled_programs": n-1,
                    "invalid_no_game_programs": 1, "fault_free_programs": n-2}
        for reference in ("base", "validity_rl-round3"):
            result["contrasts"][f"game_rl-minus-{reference}/{condition}"] = {
                "program_cluster_n": n,
                "paired_generation_seeds": list(range(8300000 if condition == "compact" else 8400000,
                                                       (8300000 if condition == "compact" else 8400000)+n)),
                "raw_paired_banked_yield_differences": [0.] * n}
    return result


class PlotProgramStudyTests(unittest.TestCase):
    def test_full_denominators_keep_invalid_and_valid_zero_distinct(self):
        cells = extract_complete(fixture())
        compact = cells[("compact", "base")]
        self.assertEqual(compact["n"], 8)
        self.assertEqual(compact["mean"], 1.0)  # Not 8/7 conditional on authorship.
        self.assertEqual(compact["counts"], {"invalid_authorship": 1, "runtime_fault": 1,
                                             "valid_zero": 5, "valid_positive": 1})
        self.assertEqual(compact["runtime_valid_rate"], 6/8)
        self.assertEqual(cells[("plain", "base")]["mean"], 2.0)
        for endpoint in ENDPOINTS:
            self.assertEqual([p["generation_seed"] for p in cells[("compact", endpoint)]["programs"]],
                             list(range(8300000, 8300008)))

    def test_partial_or_missing_draw_never_becomes_zero(self):
        for field in ("planned_study_complete", "planned_comparisons_available", "integrity_passed"):
            data = fixture(); data[field] = False
            with self.assertRaisesRegex(ValueError, "complete independent"):
                extract_complete(data)
        data = fixture(); data["evaluation_draws"] = 35
        with self.assertRaisesRegex(ValueError, "36 evaluation"):
            extract_complete(data)
        data = fixture(); cell = data["evaluation"]["base/compact/holdout"]
        cell["programs"].pop()
        with self.assertRaisesRegex(ValueError, "identity/order"):
            extract_complete(data)
        data = fixture(); data["evaluation"]["base/compact/holdout"]["programs"][0]["banked_point_yield"] = None
        with self.assertRaisesRegex(ValueError, "not a zero"):
            extract_complete(data)

    def test_missing_validation_and_wrong_paired_order_are_rejected(self):
        data = fixture(); del data["evaluation"]["base/plain/validation"]
        with self.assertRaisesRegex(ValueError, "twelve"):
            extract_complete(data)
        data = fixture(); rows = data["evaluation"]["base/compact/holdout"]["programs"]
        rows[0], rows[1] = rows[1], rows[0]
        with self.assertRaisesRegex(ValueError, "identity/order"):
            extract_complete(data)

    def test_invalid_authoring_cannot_be_a_fabricated_game(self):
        data = fixture(); data["evaluation"]["base/plain/holdout"]["programs"][0]["games"] = 48
        with self.assertRaisesRegex(ValueError, "fabricated game"):
            extract_complete(data)

    def test_exact_parsed_audit_bytes_and_current_state_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest, status = b'{"frozen": true}\n', b'{"state": "complete"}\n'
            (run / "manifest.json").write_bytes(manifest)
            (run / "status.json").write_bytes(status)
            evidence = fixture()
            evidence.update(run=str(run), manifest_sha256=hashlib.sha256(manifest).hexdigest(),
                            status_sha256=hashlib.sha256(status).hexdigest())
            exact_bytes = (json.dumps(evidence, indent=3) + "\n\n").encode()
            audit_path = run / "audit.json"
            audit_path.write_bytes(exact_bytes)
            parsed, bindings = bind_audit_inputs(audit_path)
            self.assertEqual(bindings["analysis"]["sha256"], hashlib.sha256(exact_bytes).hexdigest())
            self.assertEqual(parsed, evidence)
            verify_bound_inputs(bindings)
            # Formatting-only replacement still changes the exact parsed artifact.
            audit_path.write_text(json.dumps(evidence))
            with self.assertRaisesRegex(ValueError, "analysis changed"):
                verify_bound_inputs(bindings)
            audit_path.write_bytes(exact_bytes)
            (run / "status.json").write_text('{"state": "changed"}')
            with self.assertRaisesRegex(ValueError, "status binding differs"):
                bind_audit_inputs(audit_path)
            with self.assertRaisesRegex(ValueError, "status changed"):
                verify_bound_inputs(bindings)
            (run / "status.json").write_bytes(status)
            (run / "manifest.json").write_text('{"frozen": false}')
            with self.assertRaisesRegex(ValueError, "manifest changed"):
                verify_bound_inputs(bindings)


if __name__ == "__main__":
    unittest.main()
