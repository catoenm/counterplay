"""CPU-only fixtures for the independent cardinal-cycle auditor; no observations."""
import copy
from datetime import datetime, timezone, timedelta
from pathlib import Path
import tempfile
import unittest

import numpy as np
from safetensors.numpy import load_file, save_file

from scripts import analyze_program_cardinal_cycle as audit


def calibration_row(index, positive=12, faults=0, fingerprint=None):
    matches=[]
    for i in range(24):
        matches.append({"candidate_seat": i % 2, "replay": {"returns": [0.,0.],
            "metrics": {"scores": ([1,0] if i%2==0 else [0,1]) if i<positive else [0,0],
                        "program_faults": [faults if i==0 else 0, 0]}}})
    return {"generation":f"pool-{index}", "classification":{"program":{"ir_sha256":fingerprint or str(index)}},
            "matches":matches, **audit.score_from_matches(matches,True)}


class CardinalAuditTests(unittest.TestCase):
    def test_gate_requires_all_eight_four_productive_and_two_distinct(self):
        rows=[calibration_row(i,positive=12 if i<4 else 0) for i in range(8)]
        decision=audit.capability_decision(rows)
        self.assertTrue(decision["passed"])
        self.assertEqual(decision["selected_indices"],[0,1])
        with self.assertRaisesRegex(ValueError,"eight"):
            audit.capability_decision(rows[:7])
        duplicate=copy.deepcopy(rows)
        for row in duplicate:row["classification"]["program"]["ir_sha256"]="same"
        self.assertFalse(audit.capability_decision(duplicate)["passed"])
        rows[3]=calibration_row(3,positive=11)
        self.assertFalse(audit.capability_decision(rows)["passed"])

    def test_gate_faults_and_full_grid_not_first_successful_subset(self):
        row=calibration_row(0,positive=24,faults=1)
        self.assertFalse(audit.productivity(row))
        row=calibration_row(0);row["matches"].pop()
        with self.assertRaisesRegex(ValueError,"24"):
            audit.productivity(row)
        invalid={"authorship_valid":False,"matches":[]}
        self.assertFalse(audit.productivity(invalid))
        invalid["matches"]=[{}]
        with self.assertRaisesRegex(ValueError,"invented"):
            audit.productivity(invalid)

    def test_opponent_faults_do_not_penalize_candidate(self):
        row=calibration_row(0)
        for match in row["matches"]:
            seat=match["candidate_seat"]
            match["replay"]["metrics"]["program_faults"][1-seat]=48
        result=audit.score_from_matches(row["matches"],True)
        self.assertTrue(result["runtime_valid"])
        self.assertEqual(result["reward"],0)
        self.assertEqual(result["candidate_program_faults"],0)
        self.assertEqual(result["opponent_program_faults"],1152)

    def test_new_optimizer_changes_only_lr_and_preserves_zero_moments(self):
        with tempfile.TemporaryDirectory() as directory:
            original=Path(directory)/"old";initialized=Path(directory)/"new"
            original.mkdir();initialized.mkdir()
            arrays={"step":np.array(0,dtype=np.uint64),"learning_rate":np.array(1e-6,dtype=np.float32),
                    "q.m":np.zeros((2,2),np.float32),"q.v":np.zeros((2,2),np.float32)}
            for p in (original,initialized):save_file({"q":np.zeros((2,2),np.float32)},str(p/"adapter.safetensors"))
            save_file(arrays,str(original/"optimizer.safetensors"))
            newer={**arrays,"learning_rate":np.array(1e-5,dtype=np.float32)}
            save_file(newer,str(initialized/"optimizer.safetensors"))
            self.assertAlmostEqual(audit.validate_optimizer_initialization(original,initialized),1e-5)
            save_file(arrays,str(initialized/"optimizer.safetensors"))
            with self.assertRaisesRegex(ValueError,"learning rate"):
                audit.validate_optimizer_initialization(original,initialized)
            newer["q.m"]=np.ones((2,2),np.float32)
            save_file(newer,str(initialized/"optimizer.safetensors"))
            with self.assertRaisesRegex(ValueError,"moments/state"):
                audit.validate_optimizer_initialization(original,initialized)

    def test_paired_bootstrap_retains_zeros_and_rejects_missing_draws(self):
        base=[{"generation":f"eval-base-{i}","banked_point_yield":0.} for i in range(12)]
        final=[{"generation":f"eval-cycle3-{i}","banked_point_yield":12. if i==0 else 0.} for i in range(12)]
        result=audit.paired_summary(base,final,list(range(12)),draws=1000)
        self.assertEqual(result["mean_yield_difference"],1.)
        self.assertEqual(result,audit.paired_summary(base,final,list(range(12)),draws=1000))
        with self.assertRaisesRegex(ValueError,"Incomplete"):
            audit.paired_summary(base,final[:11],list(range(12)))
        final[0]["banked_point_yield"]=None
        with self.assertRaisesRegex(ValueError,"Missing"):
            audit.paired_summary(base,final,list(range(12)))

    def test_budget_resumption_never_renews_six_hours(self):
        start=datetime(2026,10,22,15,tzinfo=timezone.utc)
        cfg={"deadline":"2026-10-22T16:30:00-07:00","worker_seconds":21600}
        budget={"first_started":start.isoformat(),"hard_stop":(start+timedelta(hours=6)).isoformat(),
                "maximum_seconds":21600,"resume_counts_wall_time":True}
        state={"state":"running","execution_budget":budget}
        audit.validate_execution_budget(cfg,state,{"execution_budget":budget})
        state["execution_budget"]["hard_stop"]=(start+timedelta(hours=7)).isoformat()
        with self.assertRaisesRegex(ValueError,"reset"):
            audit.validate_execution_budget(cfg,state)


if __name__=="__main__":
    unittest.main()
