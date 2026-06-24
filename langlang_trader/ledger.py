from __future__ import annotations

from contextlib import contextmanager
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
