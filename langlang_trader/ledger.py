from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterator

from langlang_trader.models import (
    AccountSnapshot,
    OrderIntent,
    Position,
    Side,
    Signal,
    to_jsonable,
    utc_now_iso,
)


class Ledger:
    """SQLite ledger that mirrors the shape of a real exchange execution trail."""

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str = "default",
        bot_id: str = "default",
        variant_id: str = "rules_v01_default",
        exchange: str = "okx",
    ):
        self.path = Path(path)
        self.run_id = run_id
        self.bot_id = bot_id
        self.variant_id = variant_id
        self.exchange = exchange
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def scoped(
        self,
        *,
        run_id: str,
        bot_id: str,
        variant_id: str,
        exchange: str | None = None,
    ) -> "Ledger":
        return Ledger(
            self.path,
            run_id=run_id,
            bot_id=bot_id,
            variant_id=variant_id,
            exchange=exchange or self.exchange,
        )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                create table if not exists signals (
                    id integer primary key autoincrement,
                    run_id text not null default 'default',
                    bot_id text not null default 'default',
                    variant_id text not null default 'rules_v01_default',
                    created_at text not null,
                    strategy_version text not null,
                    symbol text not null,
                    side text not null,
                    strength real not null,
                    reason_codes_json text not null,
                    features_json text not null,
                    invalidation_price real not null,
                    take_profit_hint real,
                    regime text,
                    setup text,
                    filter_codes_json text not null default '[]',
                    take_profit_plan_json text not null default '{}',
                    hold_plan_json text not null default '{}',
                    decision_trace_json text not null default '{}',
                    historical_match_score real,
                    matched_trade_examples_json text not null default '[]',
                    risk_notes_json text not null default '[]'
                );

                create table if not exists order_intents (
                    id integer primary key autoincrement,
                    run_id text not null default 'default',
                    bot_id text not null default 'default',
                    variant_id text not null default 'rules_v01_default',
                    exchange text not null default 'okx',
                    created_at text not null,
                    signal_id integer,
                    symbol text not null,
                    side text not null,
                    order_type text not null,
                    qty real not null,
                    leverage integer not null,
                    reduce_only integer not null,
                    entry_reason text not null,
                    stop_loss real,
                    max_slippage_bps real not null,
                    strategy_version text,
                    regime text,
                    setup text,
                    exit_reason text,
                    decision_trace_json text not null default '{}',
                    historical_match_score real
                );

                create table if not exists orders (
                    id integer primary key autoincrement,
                    run_id text not null default 'default',
                    bot_id text not null default 'default',
                    variant_id text not null default 'rules_v01_default',
                    exchange text not null default 'okx',
                    created_at text not null,
                    intent_id integer,
                    exchange_order_id text,
                    symbol text not null,
                    side text not null,
                    order_type text not null,
                    qty real not null,
                    leverage integer not null,
                    reduce_only integer not null,
                    status text not null,
                    raw_payload_json text not null,
                    strategy_version text,
                    regime text,
                    setup text,
                    exit_reason text,
                    decision_trace_json text not null default '{}',
                    historical_match_score real
                );

                create table if not exists fills (
                    id integer primary key autoincrement,
                    run_id text not null default 'default',
                    bot_id text not null default 'default',
                    variant_id text not null default 'rules_v01_default',
                    exchange text not null default 'okx',
                    created_at text not null,
                    order_id integer not null,
                    exchange_order_id text not null,
                    symbol text not null,
                    side text not null,
                    qty real not null,
                    price real not null,
                    fee real not null,
                    liquidity text not null,
                    raw_payload_json text not null,
                    strategy_version text,
                    regime text,
                    setup text,
                    exit_reason text,
                    decision_trace_json text not null default '{}',
                    historical_match_score real
                );

                create table if not exists positions (
                    run_id text not null default 'default',
                    bot_id text not null default 'default',
                    variant_id text not null default 'rules_v01_default',
                    exchange text not null default 'okx',
                    symbol text not null,
                    side text not null,
                    qty real not null,
                    avg_price real not null,
                    leverage integer not null,
                    updated_at text not null,
                    strategy_version text,
                    regime text,
                    setup text,
                    primary key (run_id, bot_id, exchange, symbol)
                );

                create table if not exists equity_snapshots (
                    id integer primary key autoincrement,
                    run_id text not null default 'default',
                    bot_id text not null default 'default',
                    variant_id text not null default 'rules_v01_default',
                    exchange text not null default 'okx',
                    created_at text not null,
                    equity_usdt real not null,
                    cash_usdt real not null,
                    margin_used_usdt real not null,
                    realized_pnl_usdt real not null,
                    raw_json text not null,
                    strategy_version text,
                    regime text,
                    setup text,
                    decision_trace_json text not null default '{}'
                );

                create table if not exists risk_events (
                    id integer primary key autoincrement,
                    run_id text not null default 'default',
                    bot_id text not null default 'default',
                    variant_id text not null default 'rules_v01_default',
                    exchange text not null default 'okx',
                    created_at text not null,
                    symbol text,
                    reason text not null,
                    payload_json text not null,
                    strategy_version text,
                    regime text,
                    setup text,
                    exit_reason text,
                    decision_trace_json text not null default '{}',
                    historical_match_score real
                );

                create table if not exists raw_exchange_payloads (
                    id integer primary key autoincrement,
                    created_at text not null,
                    source text not null,
                    payload_json text not null
                );

                create table if not exists position_sizing_decisions (
                    id integer primary key autoincrement,
                    run_id text not null default 'default',
                    bot_id text not null default 'default',
                    variant_id text not null default 'rules_v01_default',
                    exchange text not null default 'okx',
                    created_at text not null,
                    signal_id integer,
                    symbol text not null,
                    side text not null,
                    risk_unit text not null,
                    risk_unit_w_usdt real not null,
                    capital_step_level integer not null,
                    size_multiplier real not null,
                    leverage integer not null,
                    margin_usdt real not null,
                    notional_usdt real not null,
                    capped_by_json text not null default '[]',
                    reason_codes_json text not null default '[]',
                    decision_trace_json text not null default '{}',
                    strategy_version text,
                    regime text,
                    setup text,
                    historical_match_score real
                );

                create table if not exists trade_lifecycle (
                    trade_id text primary key,
                    run_id text not null default 'default',
                    bot_id text not null default 'default',
                    variant_id text not null default 'rules_v01_default',
                    exchange text not null default 'okx',
                    symbol text not null,
                    side text not null,
                    status text not null,
                    opened_at text not null,
                    closed_at text,
                    entry_signal_id integer,
                    entry_intent_id integer,
                    entry_order_id integer,
                    entry_fill_id integer,
                    exit_order_id integer,
                    exit_fill_id integer,
                    entry_price real not null,
                    exit_price real,
                    qty real not null,
                    open_qty real not null default 0.0,
                    leverage integer not null,
                    initial_stop_loss real,
                    current_stop_loss real,
                    initial_risk_usdt real,
                    partial_take_profit_taken integer not null default 0,
                    breakeven_stop_moved_at text,
                    partial_take_profit_taken_at text,
                    entry_fee real not null default 0.0,
                    exit_fee real not null default 0.0,
                    total_fees real not null default 0.0,
                    gross_pnl_usdt real,
                    realized_pnl_usdt real,
                    hold_duration_seconds real,
                    mae_usdt real not null default 0.0,
                    mfe_usdt real not null default 0.0,
                    max_open_drawdown_usdt real not null default 0.0,
                    time_to_mae_seconds real,
                    time_to_mfe_seconds real,
                    r_multiple real,
                    mfe_capture_ratio real,
                    entry_reason_codes_json text not null default '[]',
                    entry_reason_summary text not null default '',
                    entry_decision_trace_json text not null default '{}',
                    entry_feature_snapshot_json text not null default '{}',
                    exit_reason_codes_json text not null default '[]',
                    exit_reason_summary text,
                    exit_decision_trace_json text not null default '{}',
                    exit_feature_snapshot_json text not null default '{}',
                    data_quality_flags_json text not null default '[]',
                    strategy_version text,
                    regime text,
                    setup text,
                    historical_match_score real
                );

                create index if not exists idx_trade_lifecycle_context_symbol
                    on trade_lifecycle (run_id, bot_id, exchange, symbol, status);

                create table if not exists trade_events (
                    id integer primary key autoincrement,
                    trade_id text,
                    run_id text not null default 'default',
                    bot_id text not null default 'default',
                    variant_id text not null default 'rules_v01_default',
                    exchange text not null default 'okx',
                    created_at text not null,
                    event_type text not null,
                    symbol text not null,
                    side text,
                    order_id integer,
                    fill_id integer,
                    price real,
                    qty real,
                    fee real,
                    reason_codes_json text not null default '[]',
                    reason_summary text,
                    decision_trace_json text not null default '{}',
                    feature_snapshot_json text not null default '{}',
                    data_quality_flags_json text not null default '[]',
                    raw_payload_json text not null default '{}'
                );
                """
            )
            self._migrate_context_columns(conn)

    def _migrate_context_columns(self, conn: sqlite3.Connection) -> None:
        context_columns = {
            "run_id": "text not null default 'default'",
            "bot_id": "text not null default 'default'",
            "variant_id": "text not null default 'rules_v01_default'",
        }
        for table in (
            "signals",
            "order_intents",
            "orders",
            "fills",
            "equity_snapshots",
            "risk_events",
            "position_sizing_decisions",
            "trade_lifecycle",
            "trade_events",
        ):
            existing = self._columns(conn, table)
            for column, definition in context_columns.items():
                if column not in existing:
                    conn.execute(f"alter table {table} add column {column} {definition}")
        extra_columns = {
            "signals": {
                "regime": "text",
                "setup": "text",
                "filter_codes_json": "text not null default '[]'",
                "take_profit_plan_json": "text not null default '{}'",
                "hold_plan_json": "text not null default '{}'",
                "decision_trace_json": "text not null default '{}'",
                "historical_match_score": "real",
                "matched_trade_examples_json": "text not null default '[]'",
                "risk_notes_json": "text not null default '[]'",
            },
            "order_intents": {
                "exchange": "text not null default 'okx'",
                "strategy_version": "text",
                "regime": "text",
                "setup": "text",
                "exit_reason": "text",
                "decision_trace_json": "text not null default '{}'",
                "historical_match_score": "real",
            },
            "orders": {
                "exchange": "text not null default 'okx'",
                "strategy_version": "text",
                "regime": "text",
                "setup": "text",
                "exit_reason": "text",
                "decision_trace_json": "text not null default '{}'",
                "historical_match_score": "real",
            },
            "fills": {
                "exchange": "text not null default 'okx'",
                "strategy_version": "text",
                "regime": "text",
                "setup": "text",
                "exit_reason": "text",
                "decision_trace_json": "text not null default '{}'",
                "historical_match_score": "real",
            },
            "equity_snapshots": {
                "exchange": "text not null default 'okx'",
                "strategy_version": "text",
                "regime": "text",
                "setup": "text",
                "decision_trace_json": "text not null default '{}'",
            },
            "risk_events": {
                "exchange": "text not null default 'okx'",
                "strategy_version": "text",
                "regime": "text",
                "setup": "text",
                "exit_reason": "text",
                "decision_trace_json": "text not null default '{}'",
                "historical_match_score": "real",
            },
            "positions": {
                "exchange": "text not null default 'okx'",
                "strategy_version": "text",
                "regime": "text",
                "setup": "text",
            },
            "position_sizing_decisions": {
                "exchange": "text not null default 'okx'",
                "strategy_version": "text",
                "regime": "text",
                "setup": "text",
                "historical_match_score": "real",
            },
            "trade_lifecycle": {
                "exchange": "text not null default 'okx'",
                "open_qty": "real not null default 0.0",
                "current_stop_loss": "real",
                "partial_take_profit_taken": "integer not null default 0",
                "breakeven_stop_moved_at": "text",
                "partial_take_profit_taken_at": "text",
                "strategy_version": "text",
                "regime": "text",
                "setup": "text",
                "historical_match_score": "real",
            },
            "trade_events": {
                "exchange": "text not null default 'okx'",
            },
        }
        for table, columns in extra_columns.items():
            existing = self._columns(conn, table)
            for column, definition in columns.items():
                if column not in existing:
                    conn.execute(f"alter table {table} add column {column} {definition}")

        position_columns = self._columns(conn, "positions")
        position_pk = self._primary_key_columns(conn, "positions")
        expected_position_pk = ["run_id", "bot_id", "exchange", "symbol"]
        if (
            "run_id" not in position_columns
            or "bot_id" not in position_columns
            or "exchange" not in position_columns
            or position_pk != expected_position_pk
        ):
            legacy_name = "positions_legacy_single_bot"
            if "exchange" in position_columns:
                legacy_name = "positions_legacy_exchange_pk"
            conn.execute(f"alter table positions rename to {legacy_name}")
            conn.execute(
                """
                create table positions (
                    run_id text not null default 'default',
                    bot_id text not null default 'default',
                    variant_id text not null default 'rules_v01_default',
                    exchange text not null default 'okx',
                    symbol text not null,
                    side text not null,
                    qty real not null,
                    avg_price real not null,
                    leverage integer not null,
                    updated_at text not null,
                    strategy_version text,
                    regime text,
                    setup text,
                    primary key (run_id, bot_id, exchange, symbol)
                )
                """
            )
            legacy_columns = self._columns(conn, legacy_name)
            run_expr = "run_id" if "run_id" in legacy_columns else "'default'"
            bot_expr = "bot_id" if "bot_id" in legacy_columns else "'default'"
            variant_expr = "variant_id" if "variant_id" in legacy_columns else "'rules_v01_default'"
            exchange_expr = "exchange" if "exchange" in legacy_columns else "'okx'"
            conn.execute(
                f"""
                insert or replace into positions (
                    run_id, bot_id, variant_id, exchange, symbol, side, qty, avg_price, leverage, updated_at
                )
                select {run_expr}, {bot_expr}, {variant_expr}, {exchange_expr}, symbol, side, qty, avg_price, leverage, updated_at
                from {legacy_name}
                """
            )

    @staticmethod
    def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
        return {row["name"] for row in conn.execute(f"pragma table_info({table})").fetchall()}

    @staticmethod
    def _primary_key_columns(conn: sqlite3.Connection, table: str) -> list[str]:
        rows = conn.execute(f"pragma table_info({table})").fetchall()
        return [row["name"] for row in sorted((row for row in rows if row["pk"]), key=lambda row: row["pk"])]

    def _context_tuple(self) -> tuple[str, str, str]:
        return self.run_id, self.bot_id, self.variant_id

    @staticmethod
    def _intent_context_tuple(intent: OrderIntent) -> tuple[Any, Any, Any, Any, str, Any]:
        return (
            getattr(intent, "strategy_version", None),
            _enum_value(getattr(intent, "regime", None)),
            _enum_value(getattr(intent, "setup", None)),
            getattr(intent, "exit_reason", None),
            json.dumps(to_jsonable(getattr(intent, "decision_trace", {}) or {}), ensure_ascii=False, sort_keys=True),
            getattr(intent, "historical_match_score", None),
        )

    def record_signal(self, signal: Signal, strategy_version: str) -> int:
        regime = _enum_value(getattr(signal, "regime", None))
        setup = _enum_value(getattr(signal, "setup", None))
        filter_codes = getattr(signal, "filter_codes", [])
        take_profit_plan = getattr(signal, "take_profit_plan", {})
        hold_plan = getattr(signal, "hold_plan", {})
        decision_trace = getattr(signal, "decision_trace", {})
        historical_match_score = getattr(signal, "historical_match_score", None)
        matched_trade_examples = getattr(signal, "matched_trade_examples", [])
        risk_notes = getattr(signal, "risk_notes", [])
        with self.connect() as conn:
            cur = conn.execute(
                """
                insert into signals (
                    run_id, bot_id, variant_id, created_at, strategy_version, symbol, side, strength,
                    reason_codes_json, features_json, invalidation_price, take_profit_hint,
                    regime, setup, filter_codes_json, take_profit_plan_json, hold_plan_json,
                    decision_trace_json, historical_match_score, matched_trade_examples_json, risk_notes_json
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    *self._context_tuple(),
                    signal.created_at,
                    strategy_version,
                    signal.symbol,
                    signal.side.value,
                    signal.strength,
                    json.dumps(signal.reason_codes, ensure_ascii=False),
                    json.dumps(to_jsonable(signal.features), ensure_ascii=False, sort_keys=True),
                    signal.invalidation_price,
                    signal.take_profit_hint,
                    regime,
                    setup,
                    json.dumps(to_jsonable(filter_codes), ensure_ascii=False, sort_keys=True),
                    json.dumps(to_jsonable(take_profit_plan), ensure_ascii=False, sort_keys=True),
                    json.dumps(to_jsonable(hold_plan), ensure_ascii=False, sort_keys=True),
                    json.dumps(to_jsonable(decision_trace), ensure_ascii=False, sort_keys=True),
                    historical_match_score,
                    json.dumps(to_jsonable(matched_trade_examples), ensure_ascii=False, sort_keys=True),
                    json.dumps(to_jsonable(risk_notes), ensure_ascii=False, sort_keys=True),
                ),
            )
            return int(cur.lastrowid)

    def record_order_intent(self, intent: OrderIntent, signal_id: int | None = None) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                insert into order_intents (
                    run_id, bot_id, variant_id, exchange, created_at, signal_id, symbol, side, order_type, qty,
                    leverage, reduce_only, entry_reason, stop_loss, max_slippage_bps,
                    strategy_version, regime, setup, exit_reason, decision_trace_json, historical_match_score
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    *self._context_tuple(),
                    self.exchange,
                    utc_now_iso(),
                    signal_id,
                    intent.symbol,
                    intent.side.value,
                    intent.order_type,
                    intent.qty,
                    intent.leverage,
                    int(intent.reduce_only),
                    intent.entry_reason,
                    intent.stop_loss,
                    intent.max_slippage_bps,
                    *self._intent_context_tuple(intent),
                ),
            )
            return int(cur.lastrowid)

    def record_position_sizing_decision(self, intent: OrderIntent, signal_id: int | None = None) -> int:
        trace = getattr(intent, "decision_trace", {}) or {}
        with self.connect() as conn:
            cur = conn.execute(
                """
                insert into position_sizing_decisions (
                    run_id, bot_id, variant_id, exchange, created_at, signal_id, symbol, side,
                    risk_unit, risk_unit_w_usdt, capital_step_level, size_multiplier, leverage,
                    margin_usdt, notional_usdt, capped_by_json, reason_codes_json, decision_trace_json,
                    strategy_version, regime, setup, historical_match_score
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    *self._context_tuple(),
                    self.exchange,
                    utc_now_iso(),
                    signal_id,
                    intent.symbol,
                    intent.side.value,
                    str(trace.get("risk_unit", "")),
                    float(trace.get("risk_unit_w_usdt", 0.0)),
                    int(trace.get("capital_step_level", 0)),
                    float(trace.get("position_size_multiplier", 0.0)),
                    intent.leverage,
                    float(trace.get("position_margin_usdt", 0.0)),
                    float(trace.get("position_notional_usdt", 0.0)),
                    json.dumps(to_jsonable(trace.get("position_sizing_capped_by", [])), ensure_ascii=False, sort_keys=True),
                    json.dumps(to_jsonable(trace.get("position_sizing_reason_codes", [])), ensure_ascii=False, sort_keys=True),
                    json.dumps(to_jsonable(trace), ensure_ascii=False, sort_keys=True),
                    *_intent_context_tuple_without_trace(intent),
                ),
            )
            return int(cur.lastrowid)

    def record_order(
        self,
        intent: OrderIntent,
        *,
        status: str,
        exchange_order_id: str | None = None,
        intent_id: int | None = None,
        raw_payload: dict[str, Any] | None = None,
    ) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                insert into orders (
                    run_id, bot_id, variant_id, exchange, created_at, intent_id, exchange_order_id, symbol, side,
                    order_type, qty, leverage, reduce_only, status, raw_payload_json,
                    strategy_version, regime, setup, exit_reason, decision_trace_json, historical_match_score
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    *self._context_tuple(),
                    self.exchange,
                    utc_now_iso(),
                    intent_id,
                    exchange_order_id,
                    intent.symbol,
                    intent.side.value,
                    intent.order_type,
                    intent.qty,
                    intent.leverage,
                    int(intent.reduce_only),
                    status,
                    json.dumps(to_jsonable(raw_payload or {}), ensure_ascii=False, sort_keys=True),
                    *self._intent_context_tuple(intent),
                ),
            )
            return int(cur.lastrowid)

    def record_fill(
        self,
        *,
        order_id: int,
        exchange_order_id: str,
        symbol: str,
        side: Side,
        qty: float,
        price: float,
        fee: float,
        liquidity: str = "taker",
        raw_payload: dict[str, Any] | None = None,
        strategy_version: str | None = None,
        regime: str | None = None,
        setup: str | None = None,
        exit_reason: str | None = None,
        decision_trace: dict[str, Any] | None = None,
        historical_match_score: float | None = None,
    ) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                insert into fills (
                    run_id, bot_id, variant_id, exchange, created_at, order_id, exchange_order_id, symbol, side,
                    qty, price, fee, liquidity, raw_payload_json,
                    strategy_version, regime, setup, exit_reason, decision_trace_json, historical_match_score
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    *self._context_tuple(),
                    self.exchange,
                    utc_now_iso(),
                    order_id,
                    exchange_order_id,
                    symbol,
                    side.value,
                    qty,
                    price,
                    fee,
                    liquidity,
                    json.dumps(to_jsonable(raw_payload or {}), ensure_ascii=False, sort_keys=True),
                    strategy_version,
                    _enum_value(regime),
                    _enum_value(setup),
                    exit_reason,
                    json.dumps(to_jsonable(decision_trace or {}), ensure_ascii=False, sort_keys=True),
                    historical_match_score,
                ),
            )
            return int(cur.lastrowid)

    def latest_order_intent_for(self, intent: OrderIntent) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = self._latest_order_intent_row(conn, intent)
        return None if row is None else dict(row)

    def record_trade_fill(
        self,
        *,
        intent: OrderIntent,
        order_id: int,
        fill_id: int,
        price: float,
        fee: float,
        raw_payload: dict[str, Any] | None = None,
    ) -> str | None:
        now = utc_now_iso()
        if intent.reduce_only:
            return self._record_exit_trade_fill(
                intent=intent,
                order_id=order_id,
                fill_id=fill_id,
                price=price,
                fee=fee,
                raw_payload=raw_payload or {},
                now=now,
            )
        return self._record_entry_trade_fill(
            intent=intent,
            order_id=order_id,
            fill_id=fill_id,
            price=price,
            fee=fee,
            raw_payload=raw_payload or {},
            now=now,
        )

    def record_trade_mark(
        self,
        *,
        symbol: str,
        mark_price: float,
        data_quality_flags: list[str] | None = None,
    ) -> None:
        flags = data_quality_flags or []
        with self.connect() as conn:
            trade = self._open_trade_row(conn, symbol)
            if trade is None:
                return
            self._append_trade_event(
                conn,
                trade_id=trade["trade_id"],
                event_type="path_mark",
                symbol=symbol,
                side=trade["side"],
                price=mark_price,
                qty=trade["qty"],
                reason_codes=[],
                reason_summary="mark_price",
                decision_trace={},
                feature_snapshot={},
                data_quality_flags=flags,
                raw_payload={"mark_price": mark_price},
            )
            if flags:
                self._merge_trade_quality_flags(conn, trade["trade_id"], flags)
                return
            self._update_trade_excursion(conn, trade, mark_price=mark_price, at=utc_now_iso())

    def _record_entry_trade_fill(
        self,
        *,
        intent: OrderIntent,
        order_id: int,
        fill_id: int,
        price: float,
        fee: float,
        raw_payload: dict[str, Any],
        now: str,
    ) -> str:
        with self.connect() as conn:
            existing = self._open_trade_row(conn, intent.symbol)
            if existing is not None:
                trade_id = existing["trade_id"]
                new_qty = float(existing["qty"]) + intent.qty
                new_open_qty = float(existing["open_qty"]) + intent.qty
                new_entry_price = (
                    float(existing["entry_price"]) * float(existing["qty"]) + price * intent.qty
                ) / max(new_qty, 1e-12)
                conn.execute(
                    """
                    update trade_lifecycle
                    set qty = ?, open_qty = ?, entry_price = ?, entry_fee = entry_fee + ?, total_fees = total_fees + ?
                    where trade_id = ?
                    """,
                    (new_qty, new_open_qty, new_entry_price, fee, fee, trade_id),
                )
                self._append_trade_event(
                    conn,
                    trade_id=trade_id,
                    event_type="add_fill",
                    symbol=intent.symbol,
                    side=intent.side.value,
                    order_id=order_id,
                    fill_id=fill_id,
                    price=price,
                    qty=intent.qty,
                    fee=fee,
                    reason_codes=self._entry_reason_codes(intent, None),
                    reason_summary=intent.entry_reason,
                    decision_trace=intent.decision_trace,
                    feature_snapshot={},
                    raw_payload=raw_payload,
                )
                return str(trade_id)

            latest_intent = self._latest_order_intent_row(conn, intent)
            intent_row = None if latest_intent is None else dict(latest_intent)
            signal_row = self._signal_for_intent(conn, intent_row)
            reason_codes = self._entry_reason_codes(intent, signal_row)
            feature_snapshot = self._json_from_row(signal_row, "features_json", {})
            decision_trace = self._json_from_row(signal_row, "decision_trace_json", intent.decision_trace)
            signal_id = None if intent_row is None else intent_row.get("signal_id")
            intent_id = None if intent_row is None else intent_row.get("id")
            trade_id = f"{self.run_id}:{self.bot_id}:{self.exchange}:{intent.symbol}:{fill_id}"
            initial_risk = None
            if intent.stop_loss is not None:
                initial_risk = abs(price - intent.stop_loss) * intent.qty
            conn.execute(
                """
                insert into trade_lifecycle (
                    trade_id, run_id, bot_id, variant_id, exchange, symbol, side, status, opened_at,
                    entry_signal_id, entry_intent_id, entry_order_id, entry_fill_id, entry_price, qty, open_qty,
                    leverage, initial_stop_loss, current_stop_loss, initial_risk_usdt, entry_fee, total_fees,
                    entry_reason_codes_json, entry_reason_summary, entry_decision_trace_json,
                    entry_feature_snapshot_json, strategy_version, regime, setup, historical_match_score
                ) values (?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade_id,
                    *self._context_tuple(),
                    self.exchange,
                    intent.symbol,
                    intent.side.value,
                    now,
                    signal_id,
                    intent_id,
                    order_id,
                    fill_id,
                    price,
                    intent.qty,
                    intent.qty,
                    intent.leverage,
                    intent.stop_loss,
                    intent.stop_loss,
                    initial_risk,
                    fee,
                    fee,
                    json.dumps(to_jsonable(reason_codes), ensure_ascii=False, sort_keys=True),
                    intent.entry_reason,
                    json.dumps(to_jsonable(decision_trace), ensure_ascii=False, sort_keys=True),
                    json.dumps(to_jsonable(feature_snapshot), ensure_ascii=False, sort_keys=True),
                    intent.strategy_version,
                    _enum_value(intent.regime),
                    _enum_value(intent.setup),
                    intent.historical_match_score,
                ),
            )
            self._append_trade_event(
                conn,
                trade_id=trade_id,
                event_type="entry_fill",
                symbol=intent.symbol,
                side=intent.side.value,
                order_id=order_id,
                fill_id=fill_id,
                price=price,
                qty=intent.qty,
                fee=fee,
                reason_codes=reason_codes,
                reason_summary=intent.entry_reason,
                decision_trace=decision_trace,
                feature_snapshot=feature_snapshot,
                raw_payload=raw_payload,
            )
            return trade_id

    def _record_exit_trade_fill(
        self,
        *,
        intent: OrderIntent,
        order_id: int,
        fill_id: int,
        price: float,
        fee: float,
        raw_payload: dict[str, Any],
        now: str,
    ) -> str | None:
        with self.connect() as conn:
            trade = self._open_trade_row(conn, intent.symbol)
            if trade is None:
                self._append_trade_event(
                    conn,
                    trade_id=None,
                    event_type="orphan_close_fill",
                    symbol=intent.symbol,
                    side=intent.side.value,
                    order_id=order_id,
                    fill_id=fill_id,
                    price=price,
                    qty=intent.qty,
                    fee=fee,
                    reason_codes=self._exit_reason_codes(intent.exit_reason),
                    reason_summary=intent.exit_reason,
                    decision_trace=intent.decision_trace,
                    feature_snapshot={},
                    data_quality_flags=["missing_open_trade"],
                    raw_payload=raw_payload,
                )
                return None

            self._update_trade_excursion(conn, trade, mark_price=price, at=now)
            refreshed = self._trade_row(conn, str(trade["trade_id"])) or trade
            open_qty = float(refreshed["open_qty"])
            if open_qty <= 0:
                open_qty = float(refreshed["qty"])
            closed_qty = min(open_qty, intent.qty)
            gross_pnl = (price - float(refreshed["entry_price"])) * closed_qty * Side.from_value(refreshed["side"]).sign
            total_fees = float(refreshed["total_fees"]) + fee
            previous_gross = float(refreshed["gross_pnl_usdt"] or 0.0)
            total_gross = previous_gross + gross_pnl
            realized = total_gross - total_fees
            hold_seconds = _seconds_between(str(refreshed["opened_at"]), now)
            initial_risk = refreshed["initial_risk_usdt"]
            r_multiple = None
            if initial_risk is not None and float(initial_risk) > 0:
                r_multiple = realized / float(initial_risk)
            mfe = float(refreshed["mfe_usdt"])
            mfe_capture = None
            if mfe > 0:
                mfe_capture = max(0.0, realized) / mfe
            event_type = "close_fill" if intent.qty >= open_qty - 1e-12 else "partial_take_profit"
            if event_type == "close_fill":
                conn.execute(
                    """
                    update trade_lifecycle
                    set status = 'closed',
                        closed_at = ?,
                        exit_order_id = ?,
                        exit_fill_id = ?,
                        exit_price = ?,
                        open_qty = 0.0,
                        exit_fee = exit_fee + ?,
                        total_fees = ?,
                        gross_pnl_usdt = ?,
                        realized_pnl_usdt = ?,
                        hold_duration_seconds = ?,
                        r_multiple = ?,
                        mfe_capture_ratio = ?,
                        exit_reason_codes_json = ?,
                        exit_reason_summary = ?,
                        exit_decision_trace_json = ?,
                        exit_feature_snapshot_json = ?
                    where trade_id = ?
                    """,
                    (
                        now,
                        order_id,
                        fill_id,
                        price,
                        fee,
                        total_fees,
                        total_gross,
                        realized,
                        hold_seconds,
                        r_multiple,
                        mfe_capture,
                        json.dumps(to_jsonable(self._exit_reason_codes(intent.exit_reason)), ensure_ascii=False, sort_keys=True),
                        intent.exit_reason,
                        json.dumps(to_jsonable(intent.decision_trace), ensure_ascii=False, sort_keys=True),
                        json.dumps(to_jsonable(intent.decision_trace.get("features", {})), ensure_ascii=False, sort_keys=True),
                        refreshed["trade_id"],
                    ),
                )
            else:
                remaining_qty = max(0.0, open_qty - closed_qty)
                conn.execute(
                    """
                    update trade_lifecycle
                    set open_qty = ?,
                        exit_fee = exit_fee + ?,
                        total_fees = ?,
                        gross_pnl_usdt = ?,
                        realized_pnl_usdt = ?,
                        partial_take_profit_taken = 1,
                        partial_take_profit_taken_at = coalesce(partial_take_profit_taken_at, ?)
                    where trade_id = ?
                    """,
                    (remaining_qty, fee, total_fees, total_gross, realized, now, refreshed["trade_id"]),
                )
            self._append_trade_event(
                conn,
                trade_id=refreshed["trade_id"],
                event_type=event_type,
                symbol=intent.symbol,
                side=intent.side.value,
                order_id=order_id,
                fill_id=fill_id,
                price=price,
                qty=intent.qty,
                fee=fee,
                reason_codes=self._exit_reason_codes(intent.exit_reason),
                reason_summary=intent.exit_reason,
                decision_trace=intent.decision_trace,
                feature_snapshot=intent.decision_trace.get("features", {}),
                raw_payload=raw_payload,
            )
            return str(refreshed["trade_id"])

    def _latest_order_intent_row(self, conn: sqlite3.Connection, intent: OrderIntent) -> sqlite3.Row | None:
        return conn.execute(
            """
            select *
            from order_intents
            where run_id = ?
              and bot_id = ?
              and variant_id = ?
              and exchange = ?
              and symbol = ?
              and side = ?
              and reduce_only = ?
            order by id desc
            limit 1
            """,
            (
                self.run_id,
                self.bot_id,
                self.variant_id,
                self.exchange,
                intent.symbol,
                intent.side.value,
                int(intent.reduce_only),
            ),
        ).fetchone()

    def _open_trade_row(
        self,
        conn: sqlite3.Connection,
        symbol: str,
        *,
        exchange: str | None = None,
    ) -> sqlite3.Row | None:
        selected_exchange = exchange or self.exchange
        return conn.execute(
            """
            select *
            from trade_lifecycle
            where run_id = ? and bot_id = ? and exchange = ? and symbol = ? and status = 'open'
            order by opened_at desc
            limit 1
            """,
            (self.run_id, self.bot_id, selected_exchange, symbol),
        ).fetchone()

    def _trade_row(self, conn: sqlite3.Connection, trade_id: str) -> sqlite3.Row | None:
        return conn.execute("select * from trade_lifecycle where trade_id = ?", (trade_id,)).fetchone()

    def _signal_for_intent(self, conn: sqlite3.Connection, intent_row: dict[str, Any] | None) -> sqlite3.Row | None:
        if intent_row is None or intent_row.get("signal_id") is None:
            return None
        return conn.execute("select * from signals where id = ?", (intent_row["signal_id"],)).fetchone()

    @staticmethod
    def _json_from_row(row: sqlite3.Row | None, column: str, default: Any) -> Any:
        if row is None:
            return default
        try:
            return json.loads(row[column])
        except (IndexError, KeyError, TypeError, json.JSONDecodeError):
            return default

    def _entry_reason_codes(self, intent: OrderIntent, signal_row: sqlite3.Row | None) -> list[str]:
        signal_codes = self._json_from_row(signal_row, "reason_codes_json", [])
        if signal_codes:
            return [str(code) for code in signal_codes]
        trace_codes = intent.decision_trace.get("reason_codes")
        if isinstance(trace_codes, list) and trace_codes:
            return [str(code) for code in trace_codes]
        return [intent.entry_reason] if intent.entry_reason else []

    @staticmethod
    def _exit_reason_codes(exit_reason: str | None) -> list[str]:
        if not exit_reason:
            return []
        if exit_reason.startswith("stop_loss"):
            return ["stop_loss_hit"]
        return [exit_reason]

    def _append_trade_event(
        self,
        conn: sqlite3.Connection,
        *,
        trade_id: str | None,
        event_type: str,
        symbol: str,
        side: str | None = None,
        order_id: int | None = None,
        fill_id: int | None = None,
        price: float | None = None,
        qty: float | None = None,
        fee: float | None = None,
        reason_codes: list[str] | None = None,
        reason_summary: str | None = None,
        decision_trace: dict[str, Any] | None = None,
        feature_snapshot: dict[str, Any] | None = None,
        data_quality_flags: list[str] | None = None,
        raw_payload: dict[str, Any] | None = None,
    ) -> None:
        conn.execute(
            """
            insert into trade_events (
                trade_id, run_id, bot_id, variant_id, exchange, created_at, event_type, symbol, side,
                order_id, fill_id, price, qty, fee, reason_codes_json, reason_summary,
                decision_trace_json, feature_snapshot_json, data_quality_flags_json, raw_payload_json
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade_id,
                *self._context_tuple(),
                self.exchange,
                utc_now_iso(),
                event_type,
                symbol,
                side,
                order_id,
                fill_id,
                price,
                qty,
                fee,
                json.dumps(to_jsonable(reason_codes or []), ensure_ascii=False, sort_keys=True),
                reason_summary,
                json.dumps(to_jsonable(decision_trace or {}), ensure_ascii=False, sort_keys=True),
                json.dumps(to_jsonable(feature_snapshot or {}), ensure_ascii=False, sort_keys=True),
                json.dumps(to_jsonable(data_quality_flags or []), ensure_ascii=False, sort_keys=True),
                json.dumps(to_jsonable(raw_payload or {}), ensure_ascii=False, sort_keys=True),
            ),
        )

    def _merge_trade_quality_flags(
        self,
        conn: sqlite3.Connection,
        trade_id: str,
        flags: list[str],
    ) -> None:
        row = self._trade_row(conn, trade_id)
        if row is None:
            return
        existing = set(self._json_from_row(row, "data_quality_flags_json", []))
        merged = sorted(existing | {str(flag) for flag in flags})
        conn.execute(
            "update trade_lifecycle set data_quality_flags_json = ? where trade_id = ?",
            (json.dumps(merged, ensure_ascii=False, sort_keys=True), trade_id),
        )

    def _update_trade_excursion(
        self,
        conn: sqlite3.Connection,
        trade: sqlite3.Row,
        *,
        mark_price: float,
        at: str,
    ) -> None:
        side = Side.from_value(trade["side"])
        open_qty = float(trade["open_qty"])
        if open_qty <= 0:
            open_qty = float(trade["qty"])
        pnl = (mark_price - float(trade["entry_price"])) * open_qty * side.sign
        mae = min(float(trade["mae_usdt"]), pnl)
        mfe = max(float(trade["mfe_usdt"]), pnl)
        time_to_mae = trade["time_to_mae_seconds"]
        time_to_mfe = trade["time_to_mfe_seconds"]
        elapsed = _seconds_between(str(trade["opened_at"]), at)
        if mae < float(trade["mae_usdt"]):
            time_to_mae = elapsed
        if mfe > float(trade["mfe_usdt"]):
            time_to_mfe = elapsed
        conn.execute(
            """
            update trade_lifecycle
            set mae_usdt = ?,
                mfe_usdt = ?,
                max_open_drawdown_usdt = ?,
                time_to_mae_seconds = ?,
                time_to_mfe_seconds = ?
            where trade_id = ?
            """,
            (mae, mfe, mae, time_to_mae, time_to_mfe, trade["trade_id"]),
        )

    def upsert_position(self, position: Position) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                insert into positions (
                    run_id, bot_id, variant_id, exchange, symbol, side, qty, avg_price, leverage, updated_at,
                    strategy_version, regime, setup
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(run_id, bot_id, exchange, symbol) do update set
                    variant_id=excluded.variant_id,
                    side=excluded.side,
                    qty=excluded.qty,
                    avg_price=excluded.avg_price,
                    leverage=excluded.leverage,
                    updated_at=excluded.updated_at,
                    strategy_version=excluded.strategy_version,
                    regime=excluded.regime,
                    setup=excluded.setup
                """,
                (
                    *self._context_tuple(),
                    self.exchange,
                    position.symbol,
                    position.side.value,
                    position.qty,
                    position.avg_price,
                    position.leverage,
                    utc_now_iso(),
                    position.strategy_version,
                    _enum_value(position.regime),
                    _enum_value(position.setup),
                ),
            )

    def delete_position(self, symbol: str, exchange: str | None = None) -> None:
        selected_exchange = exchange or self.exchange
        with self.connect() as conn:
            conn.execute(
                "delete from positions where run_id = ? and bot_id = ? and exchange = ? and symbol = ?",
                (self.run_id, self.bot_id, selected_exchange, symbol),
            )

    def record_equity_snapshot(
        self,
        snapshot: AccountSnapshot,
        raw: dict[str, Any] | None = None,
        *,
        strategy_version: str | None = None,
        regime: str | None = None,
        setup: str | None = None,
        decision_trace: dict[str, Any] | None = None,
    ) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                insert into equity_snapshots (
                    run_id, bot_id, variant_id, exchange, created_at, equity_usdt, cash_usdt, margin_used_usdt,
                    realized_pnl_usdt, raw_json, strategy_version, regime, setup, decision_trace_json
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    *self._context_tuple(),
                    self.exchange,
                    utc_now_iso(),
                    snapshot.equity_usdt,
                    snapshot.cash_usdt,
                    snapshot.margin_used_usdt,
                    snapshot.realized_pnl_usdt,
                    json.dumps(to_jsonable(raw or {}), ensure_ascii=False, sort_keys=True),
                    strategy_version,
                    _enum_value(regime),
                    _enum_value(setup),
                    json.dumps(to_jsonable(decision_trace or {}), ensure_ascii=False, sort_keys=True),
                ),
            )
            return int(cur.lastrowid)

    def record_risk_event(self, reason: str, payload: dict[str, Any], symbol: str | None = None) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                insert into risk_events (run_id, bot_id, variant_id, exchange, created_at, symbol, reason, payload_json)
                values (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    *self._context_tuple(),
                    self.exchange,
                    utc_now_iso(),
                    symbol,
                    reason,
                    json.dumps(to_jsonable(payload), ensure_ascii=False, sort_keys=True),
                ),
            )
            return int(cur.lastrowid)

    def record_raw_payload(self, source: str, payload: dict[str, Any]) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                insert into raw_exchange_payloads (created_at, source, payload_json)
                values (?, ?, ?)
                """,
                (utc_now_iso(), source, json.dumps(to_jsonable(payload), ensure_ascii=False, sort_keys=True)),
            )
            return int(cur.lastrowid)

    def get_position(self, symbol: str, exchange: str | None = None) -> Position | None:
        selected_exchange = exchange or self.exchange
        with self.connect() as conn:
            row = conn.execute(
                "select * from positions where run_id = ? and bot_id = ? and exchange = ? and symbol = ?",
                (self.run_id, self.bot_id, selected_exchange, symbol),
            ).fetchone()
        if row is None:
            return None
        return Position(
            symbol=row["symbol"],
            side=Side.from_value(row["side"]),
            qty=float(row["qty"]),
            avg_price=float(row["avg_price"]),
            leverage=int(row["leverage"]),
            exchange=row["exchange"],
            strategy_version=row["strategy_version"] if "strategy_version" in row.keys() else None,
            regime=row["regime"] if "regime" in row.keys() else None,
            setup=row["setup"] if "setup" in row.keys() else None,
        )

    def list_positions(self, exchange: str | None = None) -> list[Position]:
        selected_exchange = self.exchange if exchange is None else exchange
        rows = self.list_rows("positions", run_id=self.run_id, bot_id=self.bot_id, exchange=selected_exchange)
        return [
            Position(
                symbol=row["symbol"],
                side=Side.from_value(row["side"]),
                qty=float(row["qty"]),
                avg_price=float(row["avg_price"]),
                leverage=int(row["leverage"]),
                exchange=row["exchange"],
                strategy_version=row.get("strategy_version"),
                regime=row.get("regime"),
                setup=row.get("setup"),
            )
            for row in rows
        ]

    def latest_stop_loss(self, symbol: str, exchange: str | None = None) -> float | None:
        selected_exchange = exchange or self.exchange
        with self.connect() as conn:
            trade = self._open_trade_row(conn, symbol, exchange=selected_exchange)
            if trade is not None and trade["exchange"] == selected_exchange and trade["current_stop_loss"] is not None:
                return float(trade["current_stop_loss"])
            row = conn.execute(
                """
                select stop_loss from order_intents
                where run_id = ?
                  and bot_id = ?
                  and exchange = ?
                  and symbol = ?
                  and reduce_only = 0
                  and stop_loss is not null
                order by id desc
                limit 1
                """,
                (self.run_id, self.bot_id, selected_exchange, symbol),
            ).fetchone()
        if row is None:
            return None
        return float(row["stop_loss"])

    def open_trade_exit_state(self, symbol: str, exchange: str | None = None) -> dict[str, Any] | None:
        selected_exchange = exchange or self.exchange
        with self.connect() as conn:
            trade = self._open_trade_row(conn, symbol, exchange=selected_exchange)
            if trade is None or trade["exchange"] != selected_exchange:
                return None
            take_profit_plan: dict[str, Any] = {}
            entry_intent_id = trade["entry_intent_id"]
            if entry_intent_id is not None:
                intent = conn.execute("select * from order_intents where id = ?", (entry_intent_id,)).fetchone()
                take_profit_plan = self._json_from_row(intent, "take_profit_plan_json", {}) if intent is not None else {}
            current_stop = trade["current_stop_loss"]
            if current_stop is None:
                current_stop = trade["initial_stop_loss"]
            return {
                "trade_id": trade["trade_id"],
                "entry_price": float(trade["entry_price"]),
                "initial_stop_loss": None if trade["initial_stop_loss"] is None else float(trade["initial_stop_loss"]),
                "current_stop_loss": None if current_stop is None else float(current_stop),
                "initial_risk_usdt": None if trade["initial_risk_usdt"] is None else float(trade["initial_risk_usdt"]),
                "mfe_usdt": float(trade["mfe_usdt"]),
                "partial_taken": bool(int(trade["partial_take_profit_taken"] or 0)),
                "take_profit_plan": take_profit_plan,
            }

    def record_stop_moved(
        self,
        *,
        symbol: str,
        new_stop_loss: float,
        reason_codes: list[str],
        reason_summary: str,
        decision_trace: dict[str, Any] | None = None,
        feature_snapshot: dict[str, Any] | None = None,
        exchange: str | None = None,
    ) -> None:
        selected_exchange = exchange or self.exchange
        with self.connect() as conn:
            trade = self._open_trade_row(conn, symbol, exchange=selected_exchange)
            if trade is None or trade["exchange"] != selected_exchange:
                return
            conn.execute(
                """
                update trade_lifecycle
                set current_stop_loss = ?,
                    breakeven_stop_moved_at = coalesce(breakeven_stop_moved_at, ?)
                where trade_id = ?
                """,
                (new_stop_loss, utc_now_iso(), trade["trade_id"]),
            )
            self._append_trade_event(
                conn,
                trade_id=trade["trade_id"],
                event_type="stop_moved",
                symbol=symbol,
                side=trade["side"],
                price=new_stop_loss,
                qty=trade["open_qty"],
                reason_codes=reason_codes,
                reason_summary=reason_summary,
                decision_trace=decision_trace or {},
                feature_snapshot=feature_snapshot or {},
                raw_payload={"new_stop_loss": new_stop_loss},
            )

    def record_open_trade_quality_flags(
        self,
        *,
        symbol: str,
        flags: list[str],
        reason_summary: str = "exit_management_data_quality",
        exchange: str | None = None,
    ) -> None:
        if not flags:
            return
        selected_exchange = exchange or self.exchange
        with self.connect() as conn:
            trade = self._open_trade_row(conn, symbol, exchange=selected_exchange)
            if trade is None or trade["exchange"] != selected_exchange:
                return
            self._merge_trade_quality_flags(conn, trade["trade_id"], flags)
            self._append_trade_event(
                conn,
                trade_id=trade["trade_id"],
                event_type="exit_management_data_quality",
                symbol=symbol,
                side=trade["side"],
                reason_codes=[],
                reason_summary=reason_summary,
                data_quality_flags=flags,
                raw_payload={"flags": flags},
            )

    def list_rows(
        self,
        table: str,
        *,
        run_id: str | None = None,
        bot_id: str | None = None,
        variant_id: str | None = None,
        exchange: str | None = None,
    ) -> list[dict[str, Any]]:
        allowed = {
            "signals",
            "order_intents",
            "orders",
            "fills",
            "positions",
            "equity_snapshots",
            "risk_events",
            "raw_exchange_payloads",
            "position_sizing_decisions",
            "trade_lifecycle",
            "trade_events",
        }
        if table not in allowed:
            raise ValueError(f"unsupported ledger table: {table}")
        where = []
        params: list[Any] = []
        if run_id is not None:
            where.append("run_id = ?")
            params.append(run_id)
        if bot_id is not None:
            where.append("bot_id = ?")
            params.append(bot_id)
        if variant_id is not None:
            where.append("variant_id = ?")
            params.append(variant_id)
        if exchange is not None and table != "raw_exchange_payloads":
            where.append("exchange = ?")
            params.append(exchange)
        query = f"select * from {table}"
        if where:
            query += " where " + " and ".join(where)
        query += " order by rowid"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def _enum_value(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    return value


def _intent_context_tuple_without_trace(intent: OrderIntent) -> tuple[Any, Any, Any, Any]:
    return (
        getattr(intent, "strategy_version", None),
        _enum_value(getattr(intent, "regime", None)),
        _enum_value(getattr(intent, "setup", None)),
        getattr(intent, "historical_match_score", None),
    )


def _seconds_between(start: str, end: str) -> float | None:
    try:
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
    except ValueError:
        return None
    return (end_dt - start_dt).total_seconds()
