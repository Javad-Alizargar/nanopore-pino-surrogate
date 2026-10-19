"""Concurrency-safe SQLite budget ledger and admission controller.

Single accounting service for all paid LLM routes. Reserves a conservative
upper-bound cost before every paid attempt and reconciles against reported
usage afterwards. Never stores request/response payloads or secrets.
"""

import json
import os
import sqlite3
import time
import uuid
from contextlib import closing
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "project_config.json")
PRICING_PATH = os.path.join(PROJECT_ROOT, "config", "pricing.json")
MODELS_PATH = os.path.join(PROJECT_ROOT, "config", "models.json")
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "logs", "budget_ledger.sqlite")

STATUS_RESERVED = "reserved"
STATUS_RECONCILED = "reconciled"
STATUS_UNRESOLVED = "unresolved"
STATUS_RELEASED = "released"
STATUS_DENIED = "denied"

OPEN_STATUSES = (STATUS_RESERVED, STATUS_UNRESOLVED)

SCHEMA = """
CREATE TABLE IF NOT EXISTS reservations (
    reservation_id     TEXT PRIMARY KEY,
    request_id         TEXT,
    provider           TEXT NOT NULL,
    model              TEXT NOT NULL,
    endpoint           TEXT,
    task_id            TEXT,
    attempt            INTEGER DEFAULT 1,
    essential          INTEGER DEFAULT 0,
    est_input_tokens   INTEGER NOT NULL,
    max_output_tokens  INTEGER NOT NULL,
    reserved_cost_usd  REAL NOT NULL,
    resolved_cost_usd  REAL,
    calc_cost_usd      REAL,
    billed_cost_usd    REAL,
    provider_usage     TEXT,
    status             TEXT NOT NULL,
    notes              TEXT,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    event_id       TEXT PRIMARY KEY,
    reservation_id TEXT,
    kind           TEXT NOT NULL,
    detail         TEXT,
    created_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE INDEX IF NOT EXISTS idx_res_status ON reservations(status);
CREATE INDEX IF NOT EXISTS idx_res_provider ON reservations(provider);
CREATE INDEX IF NOT EXISTS idx_res_task ON reservations(task_id);
"""


class Denied(Exception):
    def __init__(self, reason, detail=None):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail or {}


def _utcnow():
    return datetime.now(timezone.utc).isoformat()


def _deepseek_tier(when):
    hour = when.weekday() < 5 and (
        (1 <= when.hour < 4) or (6 <= when.hour < 10)
    )
    return "peak" if hour else "off_peak"


class BudgetLedger:
    def __init__(self, db_path=DEFAULT_DB_PATH, config_path=CONFIG_PATH,
                 pricing_path=PRICING_PATH, models_path=MODELS_PATH):
        self.db_path = db_path
        with open(config_path, "r", encoding="utf-8") as fh:
            self.config = json.load(fh)
        with open(pricing_path, "r", encoding="utf-8") as fh:
            self.pricing = json.load(fh)
        with open(models_path, "r", encoding="utf-8") as fh:
            self.models = json.load(fh)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _connect(self):
        con = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=30000")
        con.execute("PRAGMA foreign_keys=ON")
        return con

    def _init_db(self):
        with closing(self._connect()) as con:
            con.executescript(SCHEMA)
            con.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
                ("schema_version", "1.0"),
            )

    def _event(self, con, kind, reservation_id=None, detail=None):
        con.execute(
            "INSERT INTO events(event_id, reservation_id, kind, detail, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                uuid.uuid4().hex,
                reservation_id,
                kind,
                json.dumps(detail) if detail is not None else None,
                _utcnow(),
            ),
        )

    def _rates(self, provider, model):
        if provider == "deepseek":
            entry = self.pricing["deepseek"]["models"].get(model)
            if entry is None:
                raise Denied("unpriced_model", {"provider": provider, "model": model})
            return {
                "input_miss": entry["input_cache_miss"],
                "input_hit": entry["input_cache_hit"],
                "output": entry["output"],
            }
        if provider == "openai":
            entry = self.pricing["openai"]["models"].get(model)
            if entry is None:
                raise Denied("unpriced_model", {"provider": provider, "model": model})
            return {
                "input_miss": {"off_peak": entry["input_standard"], "peak": entry["input_standard"]},
                "input_hit": {"off_peak": entry["input_cached"], "peak": entry["input_cached"]},
                "output": {"off_peak": entry["output"], "peak": entry["output"]},
            }
        raise Denied("unknown_provider", {"provider": provider})

    def _known_model(self, provider, model):
        prov = self.models.get("providers", {}).get(provider)
        if not prov:
            return False
        return any(m["model_id"] == model for m in prov.get("models", []))

    def compute_reservation_cost(self, provider, model, est_input_tokens, max_output_tokens):
        if not self._known_model(provider, model):
            raise Denied("unpriced_model", {"provider": provider, "model": model})
        rates = self._rates(provider, model)
        margin = self.pricing["safety_margin"]["token_estimate_multiplier"]
        in_rate = rates["input_miss"]["peak"]
        out_rate = rates["output"]["peak"]
        in_cost = est_input_tokens * margin / 1e6 * in_rate
        out_cost = max_output_tokens / 1e6 * out_rate
        total = round(in_cost + out_cost, 8)
        return total, {
            "input_cost": round(in_cost, 8),
            "output_cost": round(out_cost, 8),
            "input_rate": in_rate,
            "output_rate": out_rate,
            "margin": margin,
            "billing_tier_reserved": "peak" if provider == "deepseek" else "standard",
        }

    def _aggregate(self, con):
        rows = con.execute(
            "SELECT status, provider, COALESCE(resolved_cost_usd, 0.0),"
            " COALESCE(reserved_cost_usd, 0.0), task_id FROM reservations"
        ).fetchall()
        committed = 0.0
        outstanding = 0.0
        inflight = 0
        by_provider = {}
        by_task = {}
        total_attempts = 0
        for status, provider, resolved, reserved, task_id in rows:
            total_attempts += 1
            if status == STATUS_RECONCILED:
                committed += resolved
                by_provider[provider] = by_provider.get(provider, 0.0) + resolved
                by_task[task_id] = by_task.get(task_id, 0.0) + resolved
            elif status in OPEN_STATUSES:
                outstanding += reserved
                by_provider[provider] = by_provider.get(provider, 0.0) + reserved
                by_task[task_id] = by_task.get(task_id, 0.0) + reserved
                if status == STATUS_RESERVED:
                    inflight += 1
        return {
            "committed": round(committed, 8),
            "outstanding": round(outstanding, 8),
            "total": round(committed + outstanding, 8),
            "inflight": inflight,
            "by_provider": by_provider,
            "by_task": by_task,
            "total_attempts": total_attempts,
        }

    def admit(self, provider, model, est_input_tokens, max_output_tokens,
              task_id, request_id=None, task_budget_usd=None, essential=False,
              attempt=1):
        reserved, detail = self.compute_reservation_cost(
            provider, model, est_input_tokens, max_output_tokens
        )
        cfg = self.config
        project_total = float(cfg["PROJECT_TOTAL_API_BUDGET_USD"])
        reserve = float(cfg["FINAL_REVIEW_AND_HANDOFF_RESERVE_USD"])
        per_req_cap = float(cfg["MAX_RESERVED_COST_PER_REQUEST_USD"])
        stop_frac = float(cfg["STOP_NEW_TASKS_FRACTION"])
        max_conc = int(cfg["MAX_CONCURRENT_PAID_REQUESTS"])
        max_run_attempts = int(cfg["MAX_PAID_REQUEST_ATTEMPTS_PER_RUN"])
        max_task_attempts = int(cfg["MAX_PAID_REQUEST_ATTEMPTS_PER_TASK"])
        oai_sub = float(cfg["OPENAI_PROJECT_SUBBUDGET_USD"])

        if reserved > per_req_cap:
            raise Denied("per_request_cap_exceeded",
                         {"reserved": reserved, "cap": per_req_cap})

        with closing(self._connect()) as con:
            try:
                con.execute("BEGIN IMMEDIATE")
                agg = self._aggregate(con)
                projected = agg["total"] + reserved

                if projected > project_total:
                    raise Denied("project_budget_exceeded",
                                 {"projected": round(projected, 8), "cap": project_total})
                if not essential and projected > (project_total - reserve):
                    raise Denied("final_review_reserve_protected",
                                 {"projected": round(projected, 8),
                                  "usable_cap": round(project_total - reserve, 8)})
                if not essential and agg["total"] >= stop_frac * project_total:
                    raise Denied("stop_new_tasks_fraction_reached",
                                 {"total": agg["total"], "threshold": stop_frac * project_total})
                if provider == "openai":
                    oai_after = agg["by_provider"].get("openai", 0.0) + reserved
                    if oai_after > oai_sub:
                        raise Denied("openai_subbudget_exceeded",
                                     {"projected": round(oai_after, 8), "cap": oai_sub})
                if agg["inflight"] >= max_conc:
                    raise Denied("concurrency_slot_unavailable",
                                 {"inflight": agg["inflight"], "max": max_conc})
                if agg["total_attempts"] >= max_run_attempts:
                    raise Denied("run_attempt_limit_reached",
                                 {"attempts": agg["total_attempts"], "max": max_run_attempts})
                task_used = agg["by_task"].get(task_id, 0.0)
                task_attempts = con.execute(
                    "SELECT COUNT(*) FROM reservations WHERE task_id = ?", (task_id,)
                ).fetchone()[0]
                if task_attempts >= max_task_attempts:
                    raise Denied("task_attempt_limit_reached",
                                 {"attempts": task_attempts, "max": max_task_attempts})
                if task_budget_usd is not None and (task_used + reserved) > float(task_budget_usd):
                    raise Denied("task_allowance_exceeded",
                                 {"projected": round(task_used + reserved, 8),
                                  "allowance": float(task_budget_usd)})

                rid = uuid.uuid4().hex
                now = _utcnow()
                endpoint = (self.models.get("providers", {}).get(provider, {}) or {}).get("endpoint")
                con.execute(
                    "INSERT INTO reservations(reservation_id, request_id, provider, model,"
                    " endpoint, task_id, attempt, essential, est_input_tokens,"
                    " max_output_tokens, reserved_cost_usd, status, notes, created_at, updated_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        rid, request_id, provider, model, endpoint, task_id, int(attempt),
                        1 if essential else 0, int(est_input_tokens), int(max_output_tokens),
                        reserved, STATUS_RESERVED,
                        json.dumps({"reservation_detail": detail}), now, now,
                    ),
                )
                self._event(con, "admitted", rid, {"reserved": reserved, "provider": provider})
                con.execute("COMMIT")
                return {
                    "status": "admitted",
                    "reservation_id": rid,
                    "reserved_cost_usd": reserved,
                    "detail": detail,
                    "aggregate_before": agg,
                }
            except Denied as exc:
                con.execute("ROLLBACK")
                with closing(self._connect()) as con2:
                    con2.execute(
                        "INSERT INTO events(event_id, kind, detail, created_at)"
                        " VALUES (?,?,?,?)",
                        (uuid.uuid4().hex, "denied:" + exc.reason,
                         json.dumps({"provider": provider, "model": model,
                                     "task_id": task_id, **exc.detail}), _utcnow()),
                    )
                raise

    def reconcile(self, reservation_id, usage, billed_cost_usd=None, request_id=None):
        with closing(self._connect()) as con:
            try:
                con.execute("BEGIN IMMEDIATE")
                row = con.execute(
                    "SELECT provider, model, status, created_at FROM reservations"
                    " WHERE reservation_id = ?", (reservation_id,)
                ).fetchone()
                if row is None:
                    raise Denied("unknown_reservation", {"reservation_id": reservation_id})
                provider, model, status, created_at = row
                if status not in OPEN_STATUSES:
                    raise Denied("reservation_not_open",
                                 {"reservation_id": reservation_id, "status": status})
                rates = self._rates(provider, model)
                prompt = int(usage.get("prompt_tokens", 0))
                cached = int(usage.get("cached_tokens", 0))
                completion = int(usage.get("completion_tokens", 0))
                reasoning = int(usage.get("reasoning_tokens", 0))
                billed_output = completion + (reasoning if provider == "openai" else 0)
                when = datetime.fromisoformat(created_at)
                tier = _deepseek_tier(when) if provider == "deepseek" else "off_peak"
                miss = max(prompt - cached, 0)
                in_cost = (miss * rates["input_miss"][tier] + cached * rates["input_hit"][tier]) / 1e6
                out_cost = billed_output * rates["output"][tier] / 1e6
                calc = round(in_cost + out_cost, 8)
                now = _utcnow()
                con.execute(
                    "UPDATE reservations SET status=?, resolved_cost_usd=?, calc_cost_usd=?,"
                    " billed_cost_usd=?, provider_usage=?, request_id=COALESCE(?, request_id),"
                    " updated_at=? WHERE reservation_id=?",
                    (
                        STATUS_RECONCILED, calc, calc, billed_cost_usd,
                        json.dumps(usage), request_id, now, reservation_id,
                    ),
                )
                self._event(con, "reconciled", reservation_id,
                            {"calc_cost_usd": calc, "tier": tier})
                con.execute("COMMIT")
                return {"status": STATUS_RECONCILED, "calc_cost_usd": calc, "billing_tier": tier}
            except Denied:
                con.execute("ROLLBACK")
                raise

    def mark_timeout(self, reservation_id, reason="timeout_or_unknown_usage"):
        with closing(self._connect()) as con:
            con.execute("BEGIN IMMEDIATE")
            con.execute(
                "UPDATE reservations SET status=?, notes=?, updated_at=?"
                " WHERE reservation_id=? AND status=?",
                (STATUS_UNRESOLVED, reason, _utcnow(), reservation_id, STATUS_RESERVED),
            )
            self._event(con, "unresolved", reservation_id, {"reason": reason})
            con.execute("COMMIT")
        return self.get(reservation_id)

    def release(self, reservation_id, reason="cancelled_before_send"):
        with closing(self._connect()) as con:
            con.execute("BEGIN IMMEDIATE")
            con.execute(
                "UPDATE reservations SET status=?, resolved_cost_usd=0.0, notes=?, updated_at=?"
                " WHERE reservation_id=? AND status IN (?, ?)",
                (STATUS_RELEASED, reason, _utcnow(), reservation_id,
                 STATUS_RESERVED, STATUS_UNRESOLVED),
            )
            self._event(con, "released", reservation_id, {"reason": reason})
            con.execute("COMMIT")
        return self.get(reservation_id)

    def get(self, reservation_id):
        with closing(self._connect()) as con:
            row = con.execute(
                "SELECT reservation_id, provider, model, task_id, status,"
                " reserved_cost_usd, resolved_cost_usd, created_at, updated_at"
                " FROM reservations WHERE reservation_id=?", (reservation_id,)
            ).fetchone()
        if row is None:
            return None
        keys = ["reservation_id", "provider", "model", "task_id", "status",
                "reserved_cost_usd", "resolved_cost_usd", "created_at", "updated_at"]
        return dict(zip(keys, row))

    def recover(self, max_age_seconds=None):
        with closing(self._connect()) as con:
            rows = con.execute(
                "SELECT reservation_id, provider, model, reserved_cost_usd, created_at"
                " FROM reservations WHERE status IN (?, ?) ORDER BY created_at",
                OPEN_STATUSES,
            ).fetchall()
        now = time.time()
        out = []
        for rid, provider, model, reserved, created in rows:
            age = None
            try:
                age = now - datetime.fromisoformat(created).timestamp()
            except Exception:
                pass
            out.append({
                "reservation_id": rid, "provider": provider, "model": model,
                "reserved_cost_usd": reserved, "created_at": created,
                "age_seconds": age,
            })
        if max_age_seconds is not None:
            for item in out:
                if item["age_seconds"] is not None and item["age_seconds"] > max_age_seconds:
                    self.mark_timeout(item["reservation_id"], reason="stale_after_restart")
        return out

    def status(self):
        with closing(self._connect()) as con:
            agg = self._aggregate(con)
        cfg = self.config
        total = float(cfg["PROJECT_TOTAL_API_BUDGET_USD"])
        warn = float(cfg["SPEND_WARNING_FRACTION"])
        stop = float(cfg["STOP_NEW_TASKS_FRACTION"])
        remaining = round(total - agg["total"], 8)
        return {
            **agg,
            "project_total_usd": total,
            "remaining_usd": remaining,
            "fraction_used": round(agg["total"] / total, 6),
            "reserved_share_usd": agg["outstanding"],
            "spent_share_usd": agg["committed"],
            "openai_subbudget_used": round(agg["by_provider"].get("openai", 0.0), 8),
            "openai_subbudget_cap": float(cfg["OPENAI_PROJECT_SUBBUDGET_USD"]),
            "warning": agg["total"] >= warn * total,
            "stop_new_tasks": agg["total"] >= stop * total,
            "final_review_reserve_usd": float(cfg["FINAL_REVIEW_AND_HANDOFF_RESERVE_USD"]),
        }

    def events(self, limit=50):
        with closing(self._connect()) as con:
            rows = con.execute(
                "SELECT kind, reservation_id, detail, created_at FROM events"
                " ORDER BY created_at DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [dict(zip(["kind", "reservation_id", "detail", "created_at"], r)) for r in rows]


def _main():
    import argparse
    ap = argparse.ArgumentParser(description="Budget ledger / admission controller")
    ap.add_argument("command", choices=["status", "recover", "events"])
    ap.add_argument("--db", default=DEFAULT_DB_PATH)
    ap.add_argument("--max-age-seconds", type=float, default=None)
    args = ap.parse_args()
    led = BudgetLedger(db_path=args.db)
    if args.command == "status":
        print(json.dumps(led.status(), indent=2))
    elif args.command == "recover":
        print(json.dumps(led.recover(max_age_seconds=args.max_age_seconds), indent=2))
    elif args.command == "events":
        print(json.dumps(led.events(), indent=2))


if __name__ == "__main__":
    _main()
