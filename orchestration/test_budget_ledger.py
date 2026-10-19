"""Local dry-run tests for the budget ledger. No paid network calls are made.

Covers the Section 4 required scenarios:
  - two simultaneous reservations near the budget boundary
  - restart recovery
  - unknown-usage timeout handling
  - denial of an unpriced model
"""

import json
import os
import tempfile
import threading
import unittest

from budget_ledger import (
    BudgetLedger,
    Denied,
    STATUS_UNRESOLVED,
)


def _fresh(tmp):
    db = os.path.join(tmp, "ledger.sqlite")
    return BudgetLedger(db_path=db)


class BoundaryRaceTest(unittest.TestCase):
    def test_two_simultaneous_reservations_near_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            led = _fresh(tmp)
            led.config["PROJECT_TOTAL_API_BUDGET_USD"] = 0.50
            led.config["FINAL_REVIEW_AND_HANDOFF_RESERVE_USD"] = 0.0
            led.config["MAX_RESERVED_COST_PER_REQUEST_USD"] = 1.0
            led.config["STOP_NEW_TASKS_FRACTION"] = 1.0
            led.pricing["safety_margin"]["token_estimate_multiplier"] = 1.0

            results = []
            barrier = threading.Barrier(2)

            def worker(i):
                barrier.wait()
                try:
                    r = led.admit(
                        provider="deepseek", model="deepseek-flash",
                        est_input_tokens=1_000_000, max_output_tokens=0,
                        task_id=f"task-{i}",
                    )
                    results.append(("admitted", r["reserved_cost_usd"]))
                except Denied as exc:
                    results.append(("denied", exc.reason))

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            admitted = [r for r in results if r[0] == "admitted"]
            denied = [r for r in results if r[0] == "denied"]
            self.assertEqual(len(admitted), 1, results)
            self.assertEqual(len(denied), 1, results)
            self.assertEqual(denied[0][1], "project_budget_exceeded")
            self.assertAlmostEqual(admitted[0][1], 0.30, places=6)


class RestartRecoveryTest(unittest.TestCase):
    def test_outstanding_survives_restart_and_is_not_refunded(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "ledger.sqlite")
            led = BudgetLedger(db_path=db)
            r = led.admit("deepseek", "deepseek-flash", 1000, 100, "task-restart")
            rid = r["reservation_id"]

            reopened = BudgetLedger(db_path=db)
            outstanding = reopened.recover()
            ids = [o["reservation_id"] for o in outstanding]
            self.assertIn(rid, ids)
            self.assertEqual(reopened.get(rid)["status"], "reserved")

            reopened.recover(max_age_seconds=-1)
            self.assertEqual(reopened.get(rid)["status"], STATUS_UNRESOLVED)
            st = reopened.status()
            self.assertAlmostEqual(st["outstanding"], r["reserved_cost_usd"], places=8)
            self.assertEqual(st["committed"], 0.0)


class UnknownUsageTimeoutTest(unittest.TestCase):
    def test_timeout_retains_full_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            led = _fresh(tmp)
            r = led.admit("deepseek", "deepseek-flash", 4000, 500, "task-timeout")
            rid = r["reservation_id"]
            led.mark_timeout(rid)
            row = led.get(rid)
            self.assertEqual(row["status"], STATUS_UNRESOLVED)
            self.assertIsNone(row["resolved_cost_usd"])
            st = led.status()
            self.assertAlmostEqual(st["outstanding"], r["reserved_cost_usd"], places=8)
            self.assertEqual(st["committed"], 0.0)
            self.assertAlmostEqual(st["total"], r["reserved_cost_usd"], places=8)


class UnpricedModelTest(unittest.TestCase):
    def test_unpriced_model_is_denied(self):
        with tempfile.TemporaryDirectory() as tmp:
            led = _fresh(tmp)
            with self.assertRaises(Denied) as cm:
                led.admit("deepseek", "deepseek-unicorn-99", 1000, 100, "task-x")
            self.assertEqual(cm.exception.reason, "unpriced_model")


class ReconciliationTest(unittest.TestCase):
    def test_reconcile_replaces_reservation_with_calculated_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            led = _fresh(tmp)
            led.pricing["safety_margin"]["token_estimate_multiplier"] = 1.0
            led.config["MAX_RESERVED_COST_PER_REQUEST_USD"] = 2.0
            r = led.admit("deepseek", "deepseek-flash", 1_000_000, 1_000_000, "task-recon")
            rid = r["reservation_id"]
            self.assertAlmostEqual(r["reserved_cost_usd"], 0.30 + 1.20, places=6)
            out = led.reconcile(rid, {
                "prompt_tokens": 1000, "cached_tokens": 500,
                "completion_tokens": 200, "reasoning_tokens": 0,
            })
            self.assertLess(out["calc_cost_usd"], r["reserved_cost_usd"])
            st = led.status()
            self.assertAlmostEqual(st["committed"], out["calc_cost_usd"], places=8)
            self.assertEqual(st["outstanding"], 0.0)


class PerRequestCapTest(unittest.TestCase):
    def test_per_request_cap_denies_oversized_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            led = _fresh(tmp)
            with self.assertRaises(Denied) as cm:
                led.admit("deepseek", "deepseek-v4-pro", 1_000_000, 1_000_000, "task-big")
            self.assertEqual(cm.exception.reason, "per_request_cap_exceeded")


if __name__ == "__main__":
    unittest.main(verbosity=2)
