import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts import runpod_cardinal as rental


class FakeProvider:
    key = "unused-fake-key"

    def __init__(self):
        self.rows = []
        self.calls = []
        self.lose_create_response = False
        self.next_cost = .06
        self.delayed_delete = set()

    def pods(self):
        return self.rows.copy()

    def call(self, method, path, body=None):
        self.calls.append((method, path))
        if method == "POST":
            pod = {**body, "id": "testpod", "status": "RUNNING", "cost": self.next_cost}
            self.rows.append(pod)
            if self.lose_create_response:
                raise TimeoutError("simulated lost response")
            return pod
        pod = next((r for r in self.rows if r["id"] == path.rsplit("/", 1)[-1]), None)
        if pod is None:
            raise rental.ProviderError(404)
        if method == "DELETE":
            if pod["id"] not in self.delayed_delete:
                self.rows.remove(pod)
            return None
        return pod


class RentalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        (self.root / "runs").mkdir()
        self.folder = self.root / "deployment"
        self.folder.mkdir()
        self.plan = {"intent": "a" * 32, "deadline_unix": time.time()+7200,
            "maximum_gpu_rate_usd_hour": 3.19, "maximum_cpu_rate_usd_hour": .06,
            "gpu_disk_gb": 160, "guard_disk_gb": 10}
        rental.atomic_json(self.folder / "plan.json", self.plan)
        rental.atomic_json(self.folder / "state.json", {"plan_sha256": rental.digest(self.folder / "plan.json"),
            "reservation_applied": False, "roles": {}, "events": []})
        rental.atomic_json(self.root / "budget.json", {"approved_pilot_cap": 100, "cloud_spend_to_date": 0})
        self.provider = FakeProvider()
        self.d = rental.Deployment(self.folder, self.provider)
        self.root_patch = patch.object(rental, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def body(self, role):
        return {"name": f"selfplay-cardinal-{self.plan['intent']}-{role}",
            "env": {"SELFPLAY_PROJECT": rental.PROJECT, "SELFPLAY_INTENT": self.plan["intent"],
                    "SELFPLAY_ROLE": role, "RUNPOD_API_KEY": "secret-must-not-be-journaled"}}

    def test_budget_is_reserved_once_and_existing_ledgers_count(self):
        self.d.reserve(); self.d.reserve()
        self.assertEqual(rental.load(self.root / "budget.json")["cloud_reserved_usd"], 80)
        self.assertTrue(self.d.state["reservation_applied"])
        # An unrelated previous commitment prevents a fresh reservation.
        other = "b" * 32
        self.d.plan["intent"] = other
        with self.assertRaisesRegex(ValueError, "cap"):
            self.d.reserve()

    def test_ambiguous_create_is_never_reposted_and_can_be_reconciled(self):
        self.d.reserve()
        self.provider.lose_create_response = True
        with patch.object(self.d, "body", side_effect=self.body):
            with self.assertRaisesRegex(RuntimeError, "read-only"):
                self.d.create("guard")
            with self.assertRaisesRegex(ValueError, "already attempted"):
                self.d.create("guard")
        self.assertEqual(self.provider.calls.count(("POST", "/pods")), 1)
        self.d.reconcile()
        self.assertEqual(self.d.state["roles"]["guard"]["pod_id"], "testpod")
        self.assertNotIn("secret-must-not-be-journaled", (self.folder / "state.json").read_text())

    def test_unready_guard_prevents_gpu_post(self):
        self.d.reserve()
        self.d.state["roles"]["guard"] = {"pod_id": "testguard"}
        with patch.object(self.d, "guard_health", return_value={"ready": False}):
            with self.assertRaisesRegex(ValueError, "not freshly armed"):
                self.d.create("gpu")
        self.assertFalse(any(method == "POST" for method, _ in self.provider.calls))
        self.assertNotIn("gpu", self.d.state["roles"])

    def test_delete_rechecks_identity_and_requires_404_confirmation(self):
        self.d.reserve()
        with patch.object(self.d, "body", side_effect=self.body):
            self.d.create("guard")
        self.provider.rows[0]["name"] = "unrelated"
        with self.assertRaisesRegex(ValueError, "identity"):
            self.d.delete("guard")
        self.assertFalse(any(method == "DELETE" for method, _ in self.provider.calls))
        self.provider.rows[0]["name"] = self.body("guard")["name"]
        self.d.delete("guard")
        self.assertEqual(self.d.state["roles"]["guard"]["state"], "terminated_confirmed")
        self.assertEqual(rental.load(self.root / "budget.json")["cloud_reserved_usd"], 80)

    def test_multiple_exact_intent_matches_are_retained_as_ambiguous(self):
        self.d.state["roles"]["gpu"] = {"state": "create_unknown"}
        self.provider.rows = [{**self.body("gpu"), "id": "one"}, {**self.body("gpu"), "id": "two"}]
        self.d.reconcile()
        self.assertEqual(self.d.state["roles"]["gpu"]["state"], "ambiguous_multiple")
        self.assertEqual(self.d.state["roles"]["gpu"]["matching_ids"], ["one", "two"])
        self.d.delete("gpu")
        row = self.d.state["roles"]["gpu"]
        self.assertEqual(row["state"], "terminated_confirmed")
        self.assertTrue(row["duplicate_intent_anomaly"])
        self.assertEqual(row["confirmed_404_ids"], ["one", "two"])
        self.assertEqual([p for m,p in self.provider.calls if m == "DELETE"], ["/pods/one", "/pods/two"])


    def add_guard(self, cost=.06):
        self.d.state["roles"]["guard"] = {"state":"identified", "pod_id":"guard"}
        self.provider.rows.append({**self.body("guard"), "id":"guard", "cost":cost})
        self.d.save()
        return {"ready":True,"guard_pod_id":"guard","last_inventory_unix":time.time(),
                "finish_requested":False,"self_delete_requested":False,"creation_closed":False,
                "duplicate_intent_anomaly":False}

    def test_manual_guard_delete_preserves_live_or_unknown_gpu_protection(self):
        self.add_guard()
        self.d.state["roles"]["gpu"] = {"state":"create_unknown"}
        with self.assertRaisesRegex(ValueError,"protection required"):
            self.d.delete("guard")
        self.provider.rows.append({**self.body("gpu"),"id":"gpu","cost":3.19})
        with self.assertRaisesRegex(ValueError,"protection required"):
            self.d.delete("guard")
        self.assertFalse(any(m == "DELETE" for m,p in self.provider.calls))
        self.d.delete("gpu"); self.d.delete("guard")
        self.assertEqual(self.d.state["roles"]["guard"]["state"],"terminated_confirmed")

    def test_explicit_creation_closure_resolves_never_seen_unknown(self):
        self.add_guard()
        self.d.state["roles"]["gpu"]={"state":"create_unknown"}
        self.d.state["creation_closed"]=True
        self.d.delete("guard")
        self.assertEqual(self.d.state["roles"]["guard"]["state"],"terminated_confirmed")
        self.assertEqual(self.d.state["roles"]["gpu"]["state"],"creation_closed_absent")

    def test_rate_missing_nonfinite_negative_or_non_numeric_is_unverified(self):
        for value in (None, float("nan"), float("inf"), -1, "3.19", False):
            with self.subTest(value=value):
                self.assertEqual(self.d.rate({"cost":value},"gpu"),{"verified":False,"within_plan":False})
        self.assertTrue(self.d.rate({"cost":3.19},"gpu")["within_plan"])
        self.assertFalse(self.d.rate({"cost":5.29},"gpu")["within_plan"])

    def test_unverified_guard_rate_blocks_gpu_and_cleans_guard(self):
        self.d.reserve();health=self.add_guard(cost=None)
        with patch.object(self.d,"guard_health",return_value=health):
            with self.assertRaisesRegex(ValueError,"cost or unique identity"):
                self.d.create("gpu")
        self.assertFalse(any(m == "POST" for m,p in self.provider.calls))
        self.assertEqual(self.d.state["roles"]["guard"]["state"],"terminated_confirmed")
        self.assertTrue(self.d.state["cleanup_started"])

    def test_new_gpu_overrate_or_missing_rate_triggers_immediate_scoped_cleanup(self):
        for rate in (None, 9.0):
            with self.subTest(rate=rate):
                self.provider.rows=[];self.provider.calls=[]
                self.d.state["roles"]={};self.d.state.pop("cleanup_started",None)
                self.d.reserve();health=self.add_guard();self.provider.next_cost=rate
                with patch.object(self.d,"guard_health",return_value=health),patch.object(self.d,"body",side_effect=self.body):
                    self.d.create("gpu")
                self.assertEqual(self.d.state["roles"]["gpu"]["state"],"terminated_confirmed")
                self.assertEqual(self.d.state["roles"]["guard"]["state"],"terminated_confirmed")
                self.assertEqual([p for m,p in self.provider.calls if m == "DELETE"],["/pods/testpod","/pods/guard"])
                self.assertEqual(self.provider.calls.count(("POST","/pods")),1)

    def test_delayed_deletion_keeps_guard_and_retains_duplicate_anomaly(self):
        self.add_guard()
        self.d.state["roles"]["gpu"]={"state":"create_unknown"}
        self.provider.rows += [{**self.body("gpu"),"id":n,"cost":3.19} for n in ("one","two")]
        self.provider.delayed_delete.add("two")
        self.d.reconcile();self.d.cleanup_if_needed()
        self.assertTrue(self.d.state["roles"]["gpu"]["duplicate_intent_anomaly"])
        self.assertEqual(self.d.state["roles"]["gpu"]["confirmed_404_ids"],["one"])
        self.assertNotIn(("DELETE","/pods/guard"),self.provider.calls)
        self.provider.delayed_delete.clear();self.d.reconcile();self.d.cleanup_if_needed()
        self.assertEqual(self.d.state["roles"]["guard"]["state"],"terminated_confirmed")

    def test_watch_cleanup_retains_guard_on_unknown_creation(self):
        self.d.reserve()
        self.add_guard()
        self.d.state["roles"]["gpu"]={"state":"create_unknown"}
        self.d.plan["deadline_unix"]=0
        self.d.reconcile();self.d.cleanup_if_needed()
        self.assertNotIn(("DELETE","/pods/guard"),self.provider.calls)
        self.assertTrue(self.d.state["cleanup_started"])
        with self.assertRaisesRegex(ValueError,"cleanup started"):
            self.d.create("guard")

    def test_cleanup_continues_if_gpu_disappears_between_watch_ticks(self):
        self.add_guard()
        self.provider.rows.append({**self.body("gpu"),"id":"gpu","cost":3.19})
        self.d.state["roles"]["gpu"]={"state":"identified","pod_id":"gpu"}
        self.d.reconcile();self.d.state["cleanup_started"]=True
        self.provider.rows=[p for p in self.provider.rows if p["id"]!="gpu"]
        self.d.reconcile();self.d.cleanup_if_needed()
        self.assertEqual(self.d.state["roles"]["guard"]["state"],"terminated_confirmed")

    def test_h200_twelve_hours_fits_but_seventeen_exceeds_reservation(self):
        (self.root/"scripts/cloud").mkdir(parents=True)
        (self.root/"scripts/cloud/cardinal_guard.py").write_text("fixture only")
        key=self.root/"public.pub";key.write_text("ssh-ed25519 fixture-only")
        result=rental.prepare(self.root/"h200", "NVIDIA H200", 5.29, key, hours=12)
        self.assertLess(result["estimate_usd"]+10,80)
        self.assertEqual(rental.load(self.root/"h200/plan.json")["maximum_hours"],12)
        with self.assertRaisesRegex(ValueError,"contingency"):
            rental.prepare(self.root/"too-long", "NVIDIA H200", 5.29, key, hours=17)
        with self.assertRaisesRegex(ValueError,"rate/duration"):
            rental.prepare(self.root/"too-costly", "NVIDIA H200", 5.30, key, hours=12)

    def test_near_match_duplicate_is_never_deleted(self):
        self.d.state["roles"]["gpu"]={"state":"create_unknown"}
        self.provider.rows=[{**self.body("gpu"),"id":"one","cost":3.19},
                            {**self.body("gpu"),"id":"other","cost":3.19}]
        self.provider.rows[1]["env"]={**self.provider.rows[1]["env"],"SELFPLAY_INTENT":"different"}
        self.d.delete("gpu")
        self.assertEqual([p for m,p in self.provider.calls if m == "DELETE"],["/pods/one"])
        self.assertEqual([p["id"] for p in self.provider.rows],["other"])


if __name__ == "__main__":
    unittest.main()
