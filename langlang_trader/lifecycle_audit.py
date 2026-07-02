from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import sqlite3
from pathlib import Path
from typing import Any

from langlang_trader.ledger import Ledger


@dataclass(frozen=True)
class AuditFinding:
    severity: str
    code: str
    bot_id: str
    symbol: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuditReport:
    path: str
    findings: list[AuditFinding]

    @property
    def critical_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity == "critical")

    def to_dict(self, *, critical_only: bool = False) -> dict[str, Any]:
        findings = self.findings
        if critical_only:
            findings = [finding for finding in findings if finding.severity == "critical"]
        return {
            "path": self.path,
            "critical_count": self.critical_count,
            "findings": [finding.__dict__ for finding in findings],
        }


@dataclass(frozen=True)
class RepairResult:
    path: str
    dry_run: bool
    actions: int


def audit_signal_ledger(path: str | Path) -> AuditReport:
    db_path = str(path)
    findings: list[AuditFinding] = []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if not _has_table(conn, "trade_lifecycle"):
            return AuditReport(db_path, findings)
        findings.extend(_duplicate_open_findings(conn))
        findings.extend(_position_mismatch_findings(conn))
        findings.extend(_bad_open_stop_findings(conn))
    return AuditReport(db_path, findings)


def audit_maker_ledger(path: str | Path) -> AuditReport:
    db_path = str(path)
    findings: list[AuditFinding] = []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if not _has_table(conn, "mm_orders"):
            return AuditReport(db_path, findings)
        findings.extend(_maker_reused_order_id_findings(conn))
        findings.extend(_maker_current_order_findings(conn))
        findings.extend(_maker_fill_integrity_findings(conn))
    return AuditReport(db_path, findings)


def audit_ledger(path: str | Path) -> AuditReport:
    db_path = str(path)
    with sqlite3.connect(db_path) as conn:
        if _has_table(conn, "mm_orders"):
            return audit_maker_ledger(db_path)
    return audit_signal_ledger(db_path)


def repair_signal_ledger(path: str | Path, *, dry_run: bool = True) -> RepairResult:
    db_path = str(path)
    actions = 0
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if not _has_table(conn, "trade_lifecycle") or not _has_table(conn, "positions"):
            return RepairResult(db_path, dry_run, 0)
        groups = conn.execute(
            """
            select run_id, bot_id, variant_id, exchange, symbol
            from trade_lifecycle
            where status = 'open'
            group by run_id, bot_id, exchange, symbol
            having count(*) > 1
            union
            select t.run_id, t.bot_id, t.variant_id, t.exchange, t.symbol
            from trade_lifecycle t
            left join positions p
              on p.run_id = t.run_id and p.bot_id = t.bot_id and p.exchange = t.exchange and p.symbol = t.symbol
            where t.status = 'open'
              and (p.symbol is null or p.side != t.side or abs(p.qty - t.open_qty) > max(1e-9, abs(p.qty) * 0.0001))
            """
        ).fetchall()
        for row in groups:
            actions += 1
            if dry_run:
                continue
            ledger = Ledger(
                db_path,
                run_id=row["run_id"],
                bot_id=row["bot_id"],
                variant_id=row["variant_id"],
                exchange=row["exchange"],
            )
            ledger.reconcile_open_trades_with_position(row["symbol"], exchange=row["exchange"])
    return RepairResult(db_path, dry_run, actions)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit or repair LiangH paper lifecycle ledgers.")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--critical-only", action="store_true")
    args = parser.parse_args(argv)

    payload: list[dict[str, Any]] = []
    exit_code = 0
    for path in args.paths:
        if args.repair:
            result = repair_signal_ledger(path, dry_run=not args.apply)
            payload.append(result.__dict__)
        report = audit_ledger(path)
        payload.append(report.to_dict(critical_only=args.critical_only))
        if report.critical_count:
            exit_code = 1
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return exit_code


def _duplicate_open_findings(conn: sqlite3.Connection) -> list[AuditFinding]:
    rows = conn.execute(
        """
        select run_id, bot_id, exchange, symbol, count(*) count, group_concat(side) sides
        from trade_lifecycle
        where status = 'open'
        group by run_id, bot_id, exchange, symbol
        having count(*) > 1
        """
    ).fetchall()
    return [
        AuditFinding(
            "critical",
            "duplicate_open_lifecycle",
            row["bot_id"],
            row["symbol"],
            {"run_id": row["run_id"], "exchange": row["exchange"], "count": row["count"], "sides": row["sides"]},
        )
        for row in rows
    ]


def _position_mismatch_findings(conn: sqlite3.Connection) -> list[AuditFinding]:
    if not _has_table(conn, "positions"):
        return []
    rows = conn.execute(
        """
        select t.run_id, t.bot_id, t.exchange, t.symbol, t.side lifecycle_side, t.open_qty,
               p.side position_side, p.qty position_qty
        from trade_lifecycle t
        left join positions p
          on p.run_id = t.run_id and p.bot_id = t.bot_id and p.exchange = t.exchange and p.symbol = t.symbol
        where t.status = 'open'
          and (p.symbol is null or p.side != t.side or abs(p.qty - t.open_qty) > max(1e-9, abs(p.qty) * 0.0001))
        """
    ).fetchall()
    return [
        AuditFinding(
            "critical",
            "position_lifecycle_mismatch",
            row["bot_id"],
            row["symbol"],
            {
                "run_id": row["run_id"],
                "exchange": row["exchange"],
                "lifecycle_side": row["lifecycle_side"],
                "open_qty": row["open_qty"],
                "position_side": row["position_side"],
                "position_qty": row["position_qty"],
            },
        )
        for row in rows
    ]


def _bad_open_stop_findings(conn: sqlite3.Connection) -> list[AuditFinding]:
    rows = conn.execute(
        """
        select run_id, bot_id, exchange, symbol, side, entry_price, initial_stop_loss
        from trade_lifecycle
        where status = 'open' and initial_stop_loss is not null
          and ((side = 'long' and initial_stop_loss >= entry_price)
            or (side = 'short' and initial_stop_loss <= entry_price))
        """
    ).fetchall()
    return [
        AuditFinding(
            "critical",
            "invalid_initial_stop_side",
            row["bot_id"],
            row["symbol"],
            {
                "run_id": row["run_id"],
                "exchange": row["exchange"],
                "side": row["side"],
                "entry_price": row["entry_price"],
                "initial_stop_loss": row["initial_stop_loss"],
            },
        )
        for row in rows
    ]


def _maker_reused_order_id_findings(conn: sqlite3.Connection) -> list[AuditFinding]:
    rows = conn.execute(
        """
        select bot_id, symbol, order_id, count(distinct order_created_at_ns) created_count,
               min(order_created_at_ns) first_created_at_ns,
               max(order_created_at_ns) last_created_at_ns
        from mm_orders
        group by bot_id, venue, symbol, order_id
        having count(distinct order_created_at_ns) > 1
        """
    ).fetchall()
    return [
        AuditFinding(
            "warning",
            "maker_order_id_reused",
            row["bot_id"],
            row["symbol"],
            {
                "order_id": row["order_id"],
                "created_count": row["created_count"],
                "first_created_at_ns": row["first_created_at_ns"],
                "last_created_at_ns": row["last_created_at_ns"],
            },
        )
        for row in rows
    ]


def _maker_current_order_findings(conn: sqlite3.Connection) -> list[AuditFinding]:
    rows = conn.execute(
        """
        with latest as (
            select run_id, bot_id, venue, symbol, order_id, max(id) as max_id
            from mm_orders
            group by run_id, bot_id, venue, symbol, order_id
        )
        select orders.run_id, orders.bot_id, orders.venue, orders.symbol, orders.order_id,
               orders.status, orders.remaining_qty, orders.expires_at_ns, orders.recorded_at_ns
        from mm_orders orders
        join latest on orders.id = latest.max_id
        where orders.status in ('open', 'partially_filled')
          and (orders.remaining_qty <= 0 or orders.expires_at_ns < orders.recorded_at_ns)
        """
    ).fetchall()
    findings: list[AuditFinding] = []
    for row in rows:
        code = "maker_open_order_remaining_qty_invalid"
        if row["expires_at_ns"] < row["recorded_at_ns"]:
            code = "maker_open_order_expired_in_current_view"
        findings.append(
            AuditFinding(
                "critical",
                code,
                row["bot_id"],
                row["symbol"],
                {
                    "run_id": row["run_id"],
                    "venue": row["venue"],
                    "order_id": row["order_id"],
                    "status": row["status"],
                    "remaining_qty": row["remaining_qty"],
                    "expires_at_ns": row["expires_at_ns"],
                    "recorded_at_ns": row["recorded_at_ns"],
                },
            )
        )
    return findings


def _maker_fill_integrity_findings(conn: sqlite3.Connection) -> list[AuditFinding]:
    if not _has_table(conn, "mm_fills"):
        return []
    rows = conn.execute(
        """
        select fills.run_id, fills.bot_id, fills.venue, fills.symbol, fills.fill_id, fills.order_id
        from mm_fills fills
        left join mm_orders orders
          on orders.run_id = fills.run_id
         and orders.bot_id = fills.bot_id
         and orders.venue = fills.venue
         and orders.symbol = fills.symbol
         and orders.order_id = fills.order_id
        where orders.id is null
        """
    ).fetchall()
    return [
        AuditFinding(
            "warning",
            "maker_fill_missing_order",
            row["bot_id"],
            row["symbol"],
            {
                "run_id": row["run_id"],
                "venue": row["venue"],
                "fill_id": row["fill_id"],
                "order_id": row["order_id"],
            },
        )
        for row in rows
    ]


def _has_table(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute("select 1 from sqlite_master where type = 'table' and name = ?", (table,)).fetchone()
        is not None
    )


if __name__ == "__main__":
    raise SystemExit(main())
