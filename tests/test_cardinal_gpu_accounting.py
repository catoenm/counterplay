import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from scripts import cardinal_gpu_accounting as accounting


AT = datetime(2026, 10, 22, 18, 30, tzinfo=timezone.utc)
INTENT = "a" * 32


def response(query, amount=2.0):
    sums = {"totalAmount": amount, "gpuAmount": amount - .1, "cpuAmount": 0., "diskAmount": .1}
    row = {"podId": query["podId"], "startTime": "2026-10-22T17:00:00Z", "endTime": "2026-10-22T18:00:00Z", **sums}
    return {"records": [row], "metadata": {"query": query, "recordCount": 1, "uniquePodCount": 1, "totals": sums}}


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / "runs/deployment"
        self.folder.mkdir(parents=True)
        (self.folder / "deployment.lock").touch()
        (self.root / "runs/external-budget.lock").touch()
        self.plan = {"project": "cardinal-gpu", "intent": INTENT, "created_at": "2026-10-22T17:01:00Z", "reservation_usd": 80}
        self.save(self.folder / "plan.json", self.plan)
        self.state = {"plan_sha256": accounting.sha((self.folder / "plan.json").read_bytes()), "roles": {}}
        for role in ("guard", "gpu"):
            self.state["roles"][role] = {"state": "identified", "known_ids": [role+"-1"], "matching_ids": [role+"-1"],
                "latest": {"id": role+"-1", "name": f"selfplay-cardinal-{INTENT}-{role}",
                    "tags": {"SELFPLAY_PROJECT": "cardinal-gpu", "SELFPLAY_INTENT": INTENT, "SELFPLAY_ROLE": role}}}
        self.save(self.folder / "state.json", self.state)
        self.budget = {"approved_pilot_cap": 100, "cloud_spend_to_date": 0, "cloud_reserved_usd": 80,
            "cloud_reservations": {INTENT: {"state": "reserved", "amount_usd": 80}}, "cloud_resources_created": []}
        self.save(self.root / "budget.json", self.budget)
        self.save(self.root / "runs/api-ledger.json", {"requests": {"done": {"state": "completed", "charged_upper_usd": .2},
            "unknown": {"state": "unknown", "reserved_usd": .05}}})

    def save(self, path, value):
        path.write_text(json.dumps(value))

    def snap(self):
        return accounting.snapshot(self.root, self.folder)

    def capture(self, snap=None, amount=2):
        return accounting.capture(snap or self.snap(), AT, lambda q: response(q, amount))

    def rows(self, value, path="capture.json"):
        return [(self.root / path, json.dumps(value).encode())]

    def test_preview_offline_retains_all_and_does_not_mutate(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*.json")}
        result = accounting.preview(self.snap(), [], AT)
        self.assertEqual(result["budget"]["conservative_commitment_usd"], 80.25)
        self.assertEqual(result["budget"]["uncommitted_headroom_usd"], 19.75)
        self.assertEqual(result["provider_reported_pod_charges_usd"]["totalAmount"], 0)
        self.assertTrue(all(r["state"] == "no_valid_billing_capture" for r in result["pod_billing"]))
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*.json")})

    def test_exact_ids_disk_component_and_no_double_count(self):
        snap = self.snap()
        captured = self.capture(snap)
        self.assertEqual({x["query"]["podId"] for x in captured["entries"]}, {"guard-1", "gpu-1"})
        result = accounting.preview(snap, self.rows(captured), AT)
        self.assertEqual(result["provider_reported_pod_charges_usd"]["diskAmount"], .2)
        self.assertEqual(result["provider_reported_pod_charges_usd"]["totalAmount"], 4)
        self.assertEqual(result["budget"]["conservative_commitment_usd"], 80.25)
        self.assertFalse(result["settlement_confirmed"])

    def test_duplicate_captures_are_not_summed_latest_full_query_wins(self):
        captured = self.capture()
        newer = copy.deepcopy(captured)
        newer["captured_at"] = "2026-10-22T18:31:00Z"
        newer["entries"][0]["response"] = response(newer["entries"][0]["query"], 3)
        result = accounting.preview(self.snap(), self.rows(captured)+self.rows(newer, "later.json"), datetime(2026,10,22,19,tzinfo=timezone.utc))
        self.assertEqual(result["provider_reported_pod_charges_usd"]["totalAmount"], 5)

    def test_empty_or_failed_response_and_termination_never_releases(self):
        for row in self.state["roles"].values():
            row.update(state="terminated_confirmed", matching_ids=[], confirmed_404_ids=row["known_ids"])
        self.save(self.folder / "state.json", self.state)
        def empty(q):
            return {"records": [], "metadata": {"query": q, "recordCount": 0, "uniquePodCount": 0,
                "totals": {k: 0 for k in accounting.FIELDS}}}
        snap = self.snap()
        captured = accounting.capture(snap, AT, empty)
        result = accounting.preview(snap, self.rows(captured), AT)
        self.assertTrue(all(r["all_known_ids_confirmed_404"] for r in result["roles"].values()))
        self.assertEqual(result["budget"]["suggested_reservation_release_usd"], 0)
        self.assertEqual(result["budget"]["conservative_commitment_usd"], 80.25)
        def failed(q):
            raise RuntimeError("secret body that must not appear")
        failure = accounting.capture(snap, AT, failed)
        self.assertNotIn("secret body", json.dumps(failure))
        self.assertTrue(all(x["state"] == "unavailable" for x in failure["entries"]))

    def test_overspend_raises_preview_commitment_above_reservation(self):
        result = accounting.preview(self.snap(), self.rows(self.capture(amount=45)), AT)
        self.assertEqual(result["budget"]["conservative_commitment_usd"], 90.25)
        self.assertEqual(result["budget"]["project_reservation_usd"], 80)

    def test_all_historical_exact_intent_ids_are_included(self):
        self.budget["cloud_resources_created"] = [
            {"id": "gpu-duplicate", "role": "gpu", "intent": INTENT, "deployment": "runs/deployment"},
            {"id": "unrelated", "role": "gpu", "intent": "b"*32, "deployment": "elsewhere"}]
        self.save(self.root / "budget.json", self.budget)
        snap = self.snap()
        self.assertEqual(snap["roles"]["gpu"]["ids"], ["gpu-1", "gpu-duplicate"])
        self.assertTrue(snap["roles"]["gpu"]["duplicate_intent_anomaly"])
        self.assertEqual(len(self.capture(snap)["entries"]), 3)

    def test_rejects_wrong_identity_scope_hash_and_reservation(self):
        captured = self.capture()
        for mutate in (
            lambda x: x.update(intent="b"*32),
            lambda x: x.update(plan_sha256="0"*64),
            lambda x: x["entries"][0].update(pod_id="unrelated"),
            lambda x: x["entries"].append(copy.deepcopy(x["entries"][0])),
        ):
            bad = copy.deepcopy(captured); mutate(bad)
            with self.assertRaises(ValueError): accounting.preview(self.snap(), self.rows(bad), AT)
        self.state["roles"]["gpu"]["latest"]["tags"]["SELFPLAY_INTENT"] = "b"*32
        self.save(self.folder / "state.json", self.state)
        with self.assertRaises(ValueError): self.snap()

    def test_billing_shape_amount_interval_and_metadata_validation(self):
        q = self.capture()["entries"][0]["query"]
        equivalent = response(q)
        equivalent["metadata"]["query"] = {**q, "startTime": q["startTime"].replace("Z", ".000+00:00")}
        self.assertEqual(accounting.validate_response(equivalent, q)["totalAmount"], 2)
        for mutate in (
            lambda x: x["records"][0].update(podId="other"),
            lambda x: x["records"][0].update(diskAmount=-.1),
            lambda x: x["records"][0].update(diskAmount=float("nan")),
            lambda x: x["records"][0].update(gpuAmount=True),
            lambda x: x["records"][0].update(totalAmount=99),
            lambda x: x["records"].append(copy.deepcopy(x["records"][0])),
            lambda x: x["records"][0].update(endTime="2026-10-22T20:00:00Z"),
            lambda x: x["metadata"].update(recordCount=2),
            lambda x: x["metadata"]["totals"].update(diskAmount=1),
        ):
            bad = response(q); mutate(bad)
            with self.assertRaises(ValueError): accounting.validate_response(bad, q)

    def test_missing_unknown_creation_and_reservation_discrepancy(self):
        self.state["roles"]["gpu"] = {"state": "create_unknown"}
        self.save(self.folder / "state.json", self.state)
        self.assertTrue(self.snap()["roles"]["gpu"]["ambiguous_unobserved_create"])
        self.budget["cloud_reserved_usd"] = 79
        self.save(self.root / "budget.json", self.budget)
        with self.assertRaises(ValueError): accounting.preview(self.snap(), [], AT)

    def test_immutable_output(self):
        path = self.root / "capture.json"
        accounting.write_new(path, {"a": 1})
        with self.assertRaises(FileExistsError): accounting.write_new(path, {"a": 2})
        self.assertEqual(json.loads(path.read_text()), {"a": 1})


if __name__ == "__main__":
    unittest.main()
