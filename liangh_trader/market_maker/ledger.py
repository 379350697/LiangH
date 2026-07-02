from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from .config import LedgerContext
from .models import FillEvent, InventoryState, LatencySample, LimitOrderState, OrderTruthEvent, QuoteIntent


_SQLITE_BUSY_TIMEOUT_MS = 10_000
_SQLITE_LOCK_RETRY_ATTEMPTS = 8
_SQLITE_LOCK_RETRY_SLEEP_SECONDS = 0.05


class MarketMakerLedger:
    def __init__(self, path: str | Path, context: LedgerContext) -> None:
        self.path = str(path)
        self.context = context
        ledger_path = Path(path)
        if str(path) != ":memory:":
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=_SQLITE_BUSY_TIMEOUT_MS / 1000.0)
        self._conn.row_factory = sqlite3.Row
        self._configure_connection()
        self._ensure_schema()

    def close(self) -> None:
        self._conn.close()

    def list_rows(self, table: str) -> list[sqlite3.Row]:
        if table not in {
            "mm_quotes",
            "mm_orders",
            "mm_fills",
            "mm_inventory_snapshots",
            "mm_latency_events",
            "mm_risk_events",
            "mm_execution_requests",
            "mm_order_truth_events",
        }:
            raise ValueError(f"unsupported ledger table: {table}")
        return list(self._conn.execute(f"SELECT * FROM {table} ORDER BY id"))

    def _configure_connection(self) -> None:
        self._conn.execute(f"PRAGMA busy_timeout = {_SQLITE_BUSY_TIMEOUT_MS}")
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")

    def record_quote(self, quote: QuoteIntent, status: str) -> None:
        self._insert(
            "mm_quotes",
            {
                **self._base_fields(quote.symbol, quote.strategy_version, quote.strategy_tree_variant_id, quote.strategy_tree_parent_id, quote.strategy_tree_path),
                "quote_id": quote.quote_id,
                "side": quote.side,
                "price": quote.price,
                "qty": quote.qty,
                "ttl_ms": quote.ttl_ms,
                "post_only": int(quote.post_only),
                "status": status,
            },
        )

    def record_order_state(self, order: LimitOrderState, reason: str = "") -> None:
        self._insert(
            "mm_orders",
            {
                **self._base_fields(order.symbol, order.strategy_version, order.strategy_tree_variant_id, order.strategy_tree_parent_id, order.strategy_tree_path),
                "order_id": order.order_id,
                "quote_id": order.quote_id,
                "side": order.side,
                "price": order.price,
                "qty": order.qty,
                "remaining_qty": order.remaining_qty,
                "status": order.status,
                "post_only": int(order.post_only),
                "order_created_at_ns": order.created_at_ns,
                "order_updated_at_ns": order.updated_at_ns,
                "expires_at_ns": order.expires_at_ns,
                "reason": reason,
            },
        )

    def max_numeric_suffix(self, table: str, column: str, prefix: str) -> int:
        if table not in {
            "mm_quotes",
            "mm_orders",
            "mm_fills",
            "mm_inventory_snapshots",
            "mm_latency_events",
            "mm_risk_events",
            "mm_execution_requests",
            "mm_order_truth_events",
        }:
            return 0
        rows = self._conn.execute(
            f"select {column} from {table} where {column} like ?",
            (f"{prefix}%",),
        ).fetchall()
        max_value = 0
        for row in rows:
            value = str(row[column])
            suffix = value.removeprefix(prefix)
            if suffix.isdigit():
                max_value = max(max_value, int(suffix))
        return max_value

    def current_order_status_counts(self) -> dict[str, int]:
        rows = self._conn.execute(
            """
            with latest as (
                select run_id, bot_id, venue, symbol, order_id, max(id) as max_id
                from mm_orders
                group by run_id, bot_id, venue, symbol, order_id
            )
            select status, count(*) as count
            from mm_orders orders
            join latest on orders.id = latest.max_id
            group by status
            """
        ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def record_fill(self, fill: FillEvent) -> None:
        self._insert(
            "mm_fills",
            {
                **self._base_fields(fill.symbol, fill.strategy_version, fill.strategy_tree_variant_id, fill.strategy_tree_parent_id, fill.strategy_tree_path),
                "fill_id": fill.fill_id,
                "order_id": fill.order_id,
                "side": fill.side,
                "price": fill.price,
                "qty": fill.qty,
                "fee_usdt": fill.fee_usdt,
                "liquidity": fill.liquidity,
                "trade_id": str(fill.trade_id),
                "event_time_ms": fill.event_time_ms,
                "receive_time_ns": fill.receive_time_ns,
                "inventory_base_qty": fill.inventory_base_qty,
                "inventory_quote_usdt": fill.inventory_quote_usdt,
            },
        )

    def record_inventory(self, inventory: InventoryState, reason: str) -> None:
        self._insert(
            "mm_inventory_snapshots",
            {
                **self._base_fields(
                    inventory.symbol,
                    self.context.strategy_version,
                    self.context.strategy_tree_variant_id,
                    self.context.strategy_tree_parent_id,
                    self.context.strategy_tree_path,
                ),
                "base_qty": inventory.base_qty,
                "quote_usdt": inventory.quote_usdt,
                "avg_price": inventory.avg_price,
                "realized_pnl_usdt": inventory.realized_pnl_usdt,
                "fees_usdt": inventory.fees_usdt,
                "reason": reason,
            },
        )

    def record_latency(self, sample: LatencySample) -> None:
        self._insert(
            "mm_latency_events",
            {
                **self._base_fields(
                    sample.symbol,
                    self.context.strategy_version,
                    self.context.strategy_tree_variant_id,
                    self.context.strategy_tree_parent_id,
                    self.context.strategy_tree_path,
                ),
                "name": sample.name,
                "sample_time_ns": sample.sample_time_ns,
                "latency_ms": sample.latency_ms,
                "payload_json": _json(sample.payload),
            },
        )

    def record_risk_event(self, reason: str, severity: str = "warning", payload: dict[str, Any] | None = None) -> None:
        self._insert(
            "mm_risk_events",
            {
                **self._base_fields(
                    self.context.symbol,
                    self.context.strategy_version,
                    self.context.strategy_tree_variant_id,
                    self.context.strategy_tree_parent_id,
                    self.context.strategy_tree_path,
                ),
                "reason": reason,
                "severity": severity,
                "payload_json": _json(payload or {}),
            },
        )

    def record_execution_request(
        self,
        *,
        gateway: str,
        method: str,
        request_id: str,
        status: str,
        latency_ms: float,
        payload: dict[str, Any] | None = None,
        rate_limit_snapshot: dict[str, Any] | None = None,
    ) -> None:
        self._insert(
            "mm_execution_requests",
            {
                **self._base_fields(
                    self.context.symbol,
                    self.context.strategy_version,
                    self.context.strategy_tree_variant_id,
                    self.context.strategy_tree_parent_id,
                    self.context.strategy_tree_path,
                ),
                "gateway": gateway,
                "method": method,
                "request_id": request_id,
                "status": status,
                "latency_ms": latency_ms,
                "payload_json": _json(payload or {}),
                "rate_limit_snapshot_json": _json(rate_limit_snapshot or {}),
            },
        )

    def record_order_truth_event(self, event: OrderTruthEvent) -> None:
        self._insert(
            "mm_order_truth_events",
            {
                **self._base_fields(
                    event.symbol,
                    event.strategy_version,
                    event.strategy_tree_variant_id,
                    event.strategy_tree_parent_id,
                    event.strategy_tree_path,
                ),
                "order_id": event.order_id,
                "client_order_id": event.client_order_id,
                "event_type": event.event_type,
                "order_status": event.order_status,
                "execution_type": event.execution_type,
                "side": event.side,
                "price": event.price,
                "qty": event.qty,
                "filled_qty": event.filled_qty,
                "last_fill_qty": event.last_fill_qty,
                "last_fill_price": event.last_fill_price,
                "event_time_ms": event.event_time_ms,
                "transaction_time_ms": event.transaction_time_ms,
                "receive_time_ns": event.receive_time_ns,
                "truth_source": event.truth_source,
                "payload_json": _json(event.payload),
            },
        )

    def _ensure_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS mm_quotes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at_ns INTEGER NOT NULL,
                run_id TEXT NOT NULL,
                bot_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                venue TEXT NOT NULL,
                symbol TEXT NOT NULL,
                strategy_version TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                strategy_tree_variant_id TEXT NOT NULL,
                strategy_tree_parent_id TEXT NOT NULL,
                strategy_tree_path TEXT NOT NULL,
                quote_id TEXT,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                qty REAL NOT NULL,
                ttl_ms INTEGER NOT NULL,
                post_only INTEGER NOT NULL,
                status TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS mm_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at_ns INTEGER NOT NULL,
                run_id TEXT NOT NULL,
                bot_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                venue TEXT NOT NULL,
                symbol TEXT NOT NULL,
                strategy_version TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                strategy_tree_variant_id TEXT NOT NULL,
                strategy_tree_parent_id TEXT NOT NULL,
                strategy_tree_path TEXT NOT NULL,
                order_id TEXT NOT NULL,
                quote_id TEXT,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                qty REAL NOT NULL,
                remaining_qty REAL NOT NULL,
                status TEXT NOT NULL,
                post_only INTEGER NOT NULL,
                order_created_at_ns INTEGER NOT NULL,
                order_updated_at_ns INTEGER NOT NULL,
                expires_at_ns INTEGER NOT NULL,
                reason TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS mm_fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at_ns INTEGER NOT NULL,
                run_id TEXT NOT NULL,
                bot_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                venue TEXT NOT NULL,
                symbol TEXT NOT NULL,
                strategy_version TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                strategy_tree_variant_id TEXT NOT NULL,
                strategy_tree_parent_id TEXT NOT NULL,
                strategy_tree_path TEXT NOT NULL,
                fill_id TEXT NOT NULL,
                order_id TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                qty REAL NOT NULL,
                fee_usdt REAL NOT NULL,
                liquidity TEXT NOT NULL,
                trade_id TEXT NOT NULL,
                event_time_ms INTEGER NOT NULL,
                receive_time_ns INTEGER NOT NULL,
                inventory_base_qty REAL NOT NULL,
                inventory_quote_usdt REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS mm_inventory_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at_ns INTEGER NOT NULL,
                run_id TEXT NOT NULL,
                bot_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                venue TEXT NOT NULL,
                symbol TEXT NOT NULL,
                strategy_version TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                strategy_tree_variant_id TEXT NOT NULL,
                strategy_tree_parent_id TEXT NOT NULL,
                strategy_tree_path TEXT NOT NULL,
                base_qty REAL NOT NULL,
                quote_usdt REAL NOT NULL,
                avg_price REAL NOT NULL,
                realized_pnl_usdt REAL NOT NULL,
                fees_usdt REAL NOT NULL,
                reason TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS mm_latency_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at_ns INTEGER NOT NULL,
                run_id TEXT NOT NULL,
                bot_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                venue TEXT NOT NULL,
                symbol TEXT NOT NULL,
                strategy_version TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                strategy_tree_variant_id TEXT NOT NULL,
                strategy_tree_parent_id TEXT NOT NULL,
                strategy_tree_path TEXT NOT NULL,
                name TEXT NOT NULL,
                sample_time_ns INTEGER NOT NULL,
                latency_ms REAL NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS mm_risk_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at_ns INTEGER NOT NULL,
                run_id TEXT NOT NULL,
                bot_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                venue TEXT NOT NULL,
                symbol TEXT NOT NULL,
                strategy_version TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                strategy_tree_variant_id TEXT NOT NULL,
                strategy_tree_parent_id TEXT NOT NULL,
                strategy_tree_path TEXT NOT NULL,
                reason TEXT NOT NULL,
                severity TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS mm_execution_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at_ns INTEGER NOT NULL,
                run_id TEXT NOT NULL,
                bot_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                venue TEXT NOT NULL,
                symbol TEXT NOT NULL,
                strategy_version TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                strategy_tree_variant_id TEXT NOT NULL,
                strategy_tree_parent_id TEXT NOT NULL,
                strategy_tree_path TEXT NOT NULL,
                gateway TEXT NOT NULL,
                method TEXT NOT NULL,
                request_id TEXT NOT NULL,
                status TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                payload_json TEXT NOT NULL,
                rate_limit_snapshot_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS mm_order_truth_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at_ns INTEGER NOT NULL,
                run_id TEXT NOT NULL,
                bot_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                venue TEXT NOT NULL,
                symbol TEXT NOT NULL,
                strategy_version TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                strategy_tree_variant_id TEXT NOT NULL,
                strategy_tree_parent_id TEXT NOT NULL,
                strategy_tree_path TEXT NOT NULL,
                order_id TEXT NOT NULL,
                client_order_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                order_status TEXT NOT NULL,
                execution_type TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                qty REAL NOT NULL,
                filled_qty REAL NOT NULL,
                last_fill_qty REAL NOT NULL,
                last_fill_price REAL NOT NULL,
                event_time_ms INTEGER NOT NULL,
                transaction_time_ms INTEGER NOT NULL,
                receive_time_ns INTEGER NOT NULL,
                truth_source TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    def _base_fields(
        self,
        symbol: str,
        strategy_version: str,
        strategy_tree_variant_id: str,
        strategy_tree_parent_id: str,
        strategy_tree_path: list[str],
    ) -> dict[str, Any]:
        return {
            "recorded_at_ns": time.monotonic_ns(),
            "run_id": self.context.run_id,
            "bot_id": self.context.bot_id,
            "mode": self.context.mode,
            "venue": self.context.venue,
            "symbol": symbol,
            "strategy_version": strategy_version,
            "variant_id": self.context.variant_id,
            "strategy_tree_variant_id": strategy_tree_variant_id,
            "strategy_tree_parent_id": strategy_tree_parent_id,
            "strategy_tree_path": _json(strategy_tree_path),
        }

    def _insert(self, table: str, fields: dict[str, Any]) -> None:
        columns = ", ".join(fields.keys())
        placeholders = ", ".join("?" for _ in fields)
        for attempt in range(_SQLITE_LOCK_RETRY_ATTEMPTS):
            try:
                self._conn.execute(
                    f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
                    list(fields.values()),
                )
                self._conn.commit()
                return
            except sqlite3.OperationalError as exc:
                if "database is locked" not in str(exc).lower() or attempt == _SQLITE_LOCK_RETRY_ATTEMPTS - 1:
                    raise
                self._conn.rollback()
                time.sleep(_SQLITE_LOCK_RETRY_SLEEP_SECONDS)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
