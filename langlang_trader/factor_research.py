from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time
from typing import Any, Iterator, Sequence


DEFAULT_BATCH7_MANIFEST_PATH = Path("configs/scalping/scalp_suite_batch7_24bot_manifest.json")
DEFAULT_BATCH7_EVENT_SIGNAL_CONFIG = Path("configs/fleet/scalp_suite_batch7_18bot_event_signal_paper.json")
_SQLITE_BUSY_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class FactorDefinition:
    factor_id: str
    family: str
    strategy_tree_path: tuple[str, ...]
    input_fields: tuple[str, ...]
    max_lag_ns: int
    usable_at_decision: bool
    version: str
    event_time_field: str = "exchange_event_time_ms"
    receive_time_field: str = "receive_time_ns"
    decision_time_field: str = "decision_time_ns"
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy_tree_path", tuple(self.strategy_tree_path))
        object.__setattr__(self, "input_fields", tuple(self.input_fields))

    @property
    def version_hash(self) -> str:
        payload = {
            "factor_id": self.factor_id,
            "family": self.family,
            "strategy_tree_path": list(self.strategy_tree_path),
            "input_fields": list(self.input_fields),
            "max_lag_ns": self.max_lag_ns,
            "usable_at_decision": self.usable_at_decision,
            "version": self.version,
        }
        return hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class MarketObservation:
    symbol: str
    venue: str
    event_time_ms: int
    receive_time_ns: int
    bid: float | None
    ask: float | None
    bid_qty: float | None = None
    ask_qty: float | None = None
    source: str = "l2_depth"

    @property
    def mid_price(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        if self.bid <= 0 or self.ask <= 0:
            return None
        return (self.bid + self.ask) / 2.0


@dataclass(frozen=True)
class FactorResearchSample:
    sample_id: str
    run_id: str
    bot_id: str
    strategy_tree_id: str
    symbol: str
    venue: str
    event_seq: int
    exchange_event_time_ms: int
    receive_time_ns: int
    decision_time_ns: int
    sample_type: str
    fired: bool
    side: str
    mid_price: float
    features: dict[str, float]
    feature_times_ns: dict[str, int] = field(default_factory=dict)
    data_quality_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_quality_flags", tuple(self.data_quality_flags))


@dataclass(frozen=True)
class ForwardLabel:
    sample_id: str
    horizon_ns: int
    label_start_ns: int
    label_end_ns: int
    forward_return_bps: float
    net_taker_pnl_bps: float
    mfe_bps: float
    mae_bps: float
    hit_take_profit: bool
    hit_stop_loss: bool


@dataclass(frozen=True)
class LookaheadAuditResult:
    factor_id: str
    decision: str
    flags: tuple[str, ...]
    checked_count: int
    leak_count: int


@dataclass(frozen=True)
class FactorScore:
    factor_id: str
    sample_count: int
    ic: float
    rank_ic: float
    mean_net_taker_pnl_bps: float
    top_bin_net_taker_pnl_bps: float
    hit_rate: float
    decision: str
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class FactorExperimentConfig:
    run_id: str
    min_samples: int = 30
    n_splits: int = 5
    embargo_ns: int = 250_000_000
    artifact_dir: str | Path | None = None
    from_time_ns: int | None = None
    to_time_ns: int | None = None


@dataclass(frozen=True)
class FactorExperimentResult:
    run_id: str
    scores: tuple[FactorScore, ...]
    report: dict[str, Any]
    artifact_path: str


@dataclass(frozen=True)
class WalkForwardSplit:
    train_samples: tuple[FactorResearchSample, ...]
    validation_samples: tuple[FactorResearchSample, ...]


@dataclass(frozen=True)
class MakerFillInput:
    fill_id: str
    order_id: str
    symbol: str
    side: str
    price: float
    qty: float
    fee_usdt: float
    liquidity: str
    event_time_ms: int
    receive_time_ns: int


@dataclass(frozen=True)
class MakerPnlDecomposition:
    price_pnl_usdt: float
    fees_usdt: float
    net_pnl_usdt: float
    completed_cycles: int
    wins: int
    losses: int
    win_rate: float


class FactorResearchStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path), timeout=_SQLITE_BUSY_TIMEOUT_SECONDS)
        conn.row_factory = sqlite3.Row
        if str(self.path) != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute(f"PRAGMA busy_timeout = {int(_SQLITE_BUSY_TIMEOUT_SECONDS * 1000)}")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def register_factors(self, factors: Sequence[FactorDefinition]) -> None:
        with self.connect() as conn:
            for factor in factors:
                conn.execute(
                    """
                    insert into factor_definitions (
                        factor_id, family, strategy_tree_path_json, input_fields_json,
                        max_lag_ns, usable_at_decision, version, version_hash, definition_json
                    ) values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    on conflict(factor_id) do update set
                        family=excluded.family,
                        strategy_tree_path_json=excluded.strategy_tree_path_json,
                        input_fields_json=excluded.input_fields_json,
                        max_lag_ns=excluded.max_lag_ns,
                        usable_at_decision=excluded.usable_at_decision,
                        version=excluded.version,
                        version_hash=excluded.version_hash,
                        definition_json=excluded.definition_json
                    """,
                    (
                        factor.factor_id,
                        factor.family,
                        _json(list(factor.strategy_tree_path)),
                        _json(list(factor.input_fields)),
                        factor.max_lag_ns,
                        int(factor.usable_at_decision),
                        factor.version,
                        factor.version_hash,
                        _json(_factor_to_json(factor)),
                    ),
                )

    def record_observation(self, observation: MarketObservation) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                insert into market_observations (
                    recorded_at_ns, symbol, venue, event_time_ms, receive_time_ns,
                    bid, ask, bid_qty, ask_qty, mid_price, source
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.monotonic_ns(),
                    observation.symbol,
                    observation.venue,
                    observation.event_time_ms,
                    observation.receive_time_ns,
                    observation.bid,
                    observation.ask,
                    observation.bid_qty,
                    observation.ask_qty,
                    observation.mid_price,
                    observation.source,
                ),
            )
            return int(cur.lastrowid)

    def record_sample(self, sample: FactorResearchSample) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                insert into factor_samples (
                    recorded_at_ns, sample_id, run_id, bot_id, strategy_tree_id, symbol, venue,
                    event_seq, exchange_event_time_ms, receive_time_ns, decision_time_ns,
                    sample_type, fired, side, mid_price, features_json, feature_times_ns_json,
                    data_quality_flags_json
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.monotonic_ns(),
                    sample.sample_id,
                    sample.run_id,
                    sample.bot_id,
                    sample.strategy_tree_id,
                    sample.symbol,
                    sample.venue,
                    sample.event_seq,
                    sample.exchange_event_time_ms,
                    sample.receive_time_ns,
                    sample.decision_time_ns,
                    sample.sample_type,
                    int(sample.fired),
                    sample.side,
                    sample.mid_price,
                    _json(sample.features),
                    _json(sample.feature_times_ns),
                    _json(list(sample.data_quality_flags)),
                ),
            )
            return int(cur.lastrowid)

    def record_labels(self, labels: Sequence[ForwardLabel]) -> None:
        with self.connect() as conn:
            for label in labels:
                existing = conn.execute(
                    "select id from factor_labels where sample_id = ? and horizon_ns = ? order by id",
                    (label.sample_id, label.horizon_ns),
                ).fetchall()
                values = (
                    label.label_start_ns,
                    label.label_end_ns,
                    label.forward_return_bps,
                    label.net_taker_pnl_bps,
                    label.mfe_bps,
                    label.mae_bps,
                    int(label.hit_take_profit),
                    int(label.hit_stop_loss),
                )
                if existing:
                    keep_id = int(existing[0]["id"])
                    conn.execute(
                        """
                        update factor_labels set
                            label_start_ns=?,
                            label_end_ns=?,
                            forward_return_bps=?,
                            net_taker_pnl_bps=?,
                            mfe_bps=?,
                            mae_bps=?,
                            hit_take_profit=?,
                            hit_stop_loss=?
                        where id=?
                        """,
                        (*values, keep_id),
                    )
                    duplicate_ids = [int(row["id"]) for row in existing[1:]]
                    if duplicate_ids:
                        conn.executemany("delete from factor_labels where id = ?", [(row_id,) for row_id in duplicate_ids])
                    continue
                conn.execute(
                    """
                    insert into factor_labels (
                        sample_id, horizon_ns, label_start_ns, label_end_ns,
                        forward_return_bps, net_taker_pnl_bps, mfe_bps, mae_bps,
                        hit_take_profit, hit_stop_loss
                    ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        label.sample_id,
                        label.horizon_ns,
                        *values,
                    ),
                )

    def record_scores(self, run_id: str, scores: Sequence[FactorScore]) -> None:
        with self.connect() as conn:
            for score in scores:
                conn.execute(
                    """
                    insert into factor_scores (
                        run_id, recorded_at_ns, factor_id, sample_count, ic, rank_ic,
                        mean_net_taker_pnl_bps, top_bin_net_taker_pnl_bps,
                        hit_rate, decision, flags_json
                    ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        time.monotonic_ns(),
                        score.factor_id,
                        score.sample_count,
                        score.ic,
                        score.rank_ic,
                        score.mean_net_taker_pnl_bps,
                        score.top_bin_net_taker_pnl_bps,
                        score.hit_rate,
                        score.decision,
                        _json(list(score.flags)),
                    ),
                )

    def record_experiment_run(
        self,
        *,
        run_id: str,
        config: dict[str, Any],
        status: str,
        summary: dict[str, Any],
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                insert into factor_experiment_runs (
                    run_id, created_at_ns, config_json, status, summary_json
                ) values (?, ?, ?, ?, ?)
                on conflict(run_id) do update set
                    created_at_ns=excluded.created_at_ns,
                    config_json=excluded.config_json,
                    status=excluded.status,
                    summary_json=excluded.summary_json
                """,
                (run_id, time.monotonic_ns(), _json(config), status, _json(summary)),
            )

    def record_artifact(
        self,
        *,
        run_id: str,
        artifact_type: str,
        path: str,
        payload: dict[str, Any],
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                insert into factor_experiment_artifacts (
                    run_id, created_at_ns, artifact_type, path, payload_json
                ) values (?, ?, ?, ?, ?)
                """,
                (run_id, time.monotonic_ns(), artifact_type, path, _json(payload)),
            )

    def latest_experiment_summary(self) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute(
                """
                select run_id, created_at_ns, status, summary_json
                from factor_experiment_runs
                order by created_at_ns desc
                limit 1
                """
            ).fetchone()
        if row is None:
            return {}
        summary = json.loads(row["summary_json"])
        summary["run_id"] = str(row["run_id"])
        summary["status"] = str(row["status"])
        summary["created_at_ns"] = int(row["created_at_ns"])
        return summary

    def latest_artifact(self, *, run_id: str = "", artifact_type: str = "") -> dict[str, Any]:
        where: list[str] = []
        params: list[Any] = []
        if run_id:
            where.append("run_id = ?")
            params.append(run_id)
        if artifact_type:
            where.append("artifact_type = ?")
            params.append(artifact_type)
        query = """
            select run_id, artifact_type, path, payload_json, created_at_ns
            from factor_experiment_artifacts
        """
        if where:
            query += " where " + " and ".join(where)
        query += " order by created_at_ns desc limit 1"
        with self.connect() as conn:
            row = conn.execute(query, params).fetchone()
        if row is None:
            return {}
        return {
            "run_id": str(row["run_id"]),
            "artifact_type": str(row["artifact_type"]),
            "path": str(row["path"]),
            "payload": json.loads(row["payload_json"]),
            "created_at_ns": int(row["created_at_ns"]),
        }

    def list_samples(self) -> list[FactorResearchSample]:
        with self.connect() as conn:
            rows = conn.execute("select * from factor_samples order by id").fetchall()
        return [_sample_from_row(row) for row in rows]

    def list_observations(self) -> list[MarketObservation]:
        with self.connect() as conn:
            rows = conn.execute("select * from market_observations order by receive_time_ns, id").fetchall()
        return [
            MarketObservation(
                symbol=str(row["symbol"]),
                venue=str(row["venue"]),
                event_time_ms=int(row["event_time_ms"]),
                receive_time_ns=int(row["receive_time_ns"]),
                bid=None if row["bid"] is None else float(row["bid"]),
                ask=None if row["ask"] is None else float(row["ask"]),
                bid_qty=None if row["bid_qty"] is None else float(row["bid_qty"]),
                ask_qty=None if row["ask_qty"] is None else float(row["ask_qty"]),
                source=str(row["source"]),
            )
            for row in rows
        ]

    def list_labels(self) -> list[ForwardLabel]:
        with self.connect() as conn:
            rows = conn.execute("select * from factor_labels order by id").fetchall()
        return [
            ForwardLabel(
                sample_id=str(row["sample_id"]),
                horizon_ns=int(row["horizon_ns"]),
                label_start_ns=int(row["label_start_ns"]),
                label_end_ns=int(row["label_end_ns"]),
                forward_return_bps=float(row["forward_return_bps"]),
                net_taker_pnl_bps=float(row["net_taker_pnl_bps"]),
                mfe_bps=float(row["mfe_bps"]),
                mae_bps=float(row["mae_bps"]),
                hit_take_profit=bool(row["hit_take_profit"]),
                hit_stop_loss=bool(row["hit_stop_loss"]),
            )
            for row in rows
        ]

    def factor_count(self) -> int:
        with self.connect() as conn:
            row = conn.execute("select count(*) as count from factor_definitions").fetchone()
        return int(row["count"])

    def _ensure_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                create table if not exists factor_definitions (
                    factor_id text primary key,
                    family text not null,
                    strategy_tree_path_json text not null,
                    input_fields_json text not null,
                    max_lag_ns integer not null,
                    usable_at_decision integer not null,
                    version text not null,
                    version_hash text not null,
                    definition_json text not null
                );
                create table if not exists market_observations (
                    id integer primary key autoincrement,
                    recorded_at_ns integer not null,
                    symbol text not null,
                    venue text not null,
                    event_time_ms integer not null,
                    receive_time_ns integer not null,
                    bid real,
                    ask real,
                    bid_qty real,
                    ask_qty real,
                    mid_price real,
                    source text not null
                );
                create table if not exists factor_samples (
                    id integer primary key autoincrement,
                    recorded_at_ns integer not null,
                    sample_id text not null,
                    run_id text not null,
                    bot_id text not null,
                    strategy_tree_id text not null,
                    symbol text not null,
                    venue text not null,
                    event_seq integer not null,
                    exchange_event_time_ms integer not null,
                    receive_time_ns integer not null,
                    decision_time_ns integer not null,
                    sample_type text not null,
                    fired integer not null,
                    side text not null,
                    mid_price real not null,
                    features_json text not null,
                    feature_times_ns_json text not null,
                    data_quality_flags_json text not null
                );
                create table if not exists factor_labels (
                    id integer primary key autoincrement,
                    sample_id text not null,
                    horizon_ns integer not null,
                    label_start_ns integer not null,
                    label_end_ns integer not null,
                    forward_return_bps real not null,
                    net_taker_pnl_bps real not null,
                    mfe_bps real not null,
                    mae_bps real not null,
                    hit_take_profit integer not null,
                    hit_stop_loss integer not null,
                    unique(sample_id, horizon_ns)
                );
                create table if not exists factor_scores (
                    id integer primary key autoincrement,
                    run_id text not null,
                    recorded_at_ns integer not null,
                    factor_id text not null,
                    sample_count integer not null,
                    ic real not null,
                    rank_ic real not null,
                    mean_net_taker_pnl_bps real not null,
                    top_bin_net_taker_pnl_bps real not null,
                    hit_rate real not null,
                    decision text not null,
                    flags_json text not null
                );
                create table if not exists factor_experiment_runs (
                    run_id text primary key,
                    created_at_ns integer not null,
                    config_json text not null,
                    status text not null,
                    summary_json text not null
                );
                create table if not exists factor_experiment_artifacts (
                    id integer primary key autoincrement,
                    run_id text not null,
                    created_at_ns integer not null,
                    artifact_type text not null,
                    path text not null,
                    payload_json text not null
                );
                """
            )


def batch7_factor_registry() -> list[FactorDefinition]:
    base = ("scalping", "batch7_hft_scalp")
    return [
        FactorDefinition("hft.queue_imbalance", "hft_order_book", base + ("hft_queue_imbalance_one_tick",), ("best_bid_qty", "best_ask_qty"), 50_000_000, True, "v1"),
        FactorDefinition("hft.spread_bps", "hft_order_book", base + ("hft_queue_imbalance_one_tick",), ("best_bid", "best_ask"), 50_000_000, True, "v1"),
        FactorDefinition("hft.sweep_notional_usdt", "hft_trade_flow", base + ("hft_sweep_replenishment_reversion",), ("trade_price", "trade_qty"), 750_000_000, True, "v1"),
        FactorDefinition("hft.sweep_age_ms", "hft_trade_flow", base + ("hft_sweep_replenishment_reversion",), ("last_sweep_receive_time_ns",), 750_000_000, True, "v1"),
        FactorDefinition("hft.replenishment_ratio", "hft_trade_flow", base + ("hft_sweep_replenishment_reversion",), ("best_bid_qty", "best_ask_qty"), 50_000_000, True, "v1"),
        FactorDefinition("hft.lead_move_bps", "hft_lead_lag", base + ("hft_lead_lag_fair_value",), ("lead_mid", "previous_lead_mid"), 250_000_000, True, "v1"),
        FactorDefinition("hft.lag_divergence_bps", "hft_lead_lag", base + ("hft_lead_lag_fair_value",), ("lead_move_bps", "lag_move_bps"), 250_000_000, True, "v1"),
        FactorDefinition("maker.ofi", "maker_order_book", base + ("hft_inventory_aware_passive_mm",), ("best_bid_qty", "best_ask_qty"), 50_000_000, True, "v1"),
        FactorDefinition("maker.inventory_base_qty", "maker_inventory", base + ("hft_inventory_aware_passive_mm",), ("inventory_base_qty",), 50_000_000, True, "v1"),
        FactorDefinition("maker.order_ttl_ms", "maker_params", base + ("hft_inventory_aware_passive_mm",), ("order_ttl_ms",), 0, True, "v1"),
        FactorDefinition("maker.quote_edge_bps", "maker_params", base + ("hft_inventory_aware_passive_mm",), ("min_quote_edge_bps", "spread_bps"), 50_000_000, True, "v1"),
        FactorDefinition("maker.quote_spread_bps", "maker_params", base + ("hft_inventory_aware_passive_mm",), ("quote_spread_bps",), 0, True, "v1"),
        FactorDefinition("cost.round_trip_fee_bps", "cost", base, ("fee_bps",), 0, True, "v1"),
        FactorDefinition("cost.take_profit_cost_floor_bps", "cost", base, ("round_trip_fee_bps", "min_net_take_profit_bps"), 0, True, "v1"),
    ]


def hft_candidate_features(variant: Any, book: Any, *, signal: Any | None = None, strategy: Any | None = None) -> dict[str, float]:
    bid_qty = float(getattr(book, "best_bid_qty", 0.0) or 0.0)
    ask_qty = float(getattr(book, "best_ask_qty", 0.0) or 0.0)
    total_qty = bid_qty + ask_qty
    imbalance = (bid_qty - ask_qty) / total_qty if total_qty > 0 else 0.0
    spread_bps = getattr(book, "spread_bps", None)
    features: dict[str, float] = {
        "hft.queue_imbalance": imbalance,
        "hft.spread_bps": float(spread_bps) if spread_bps is not None else 0.0,
        "cost.round_trip_fee_bps": float(getattr(variant, "round_trip_fee_bps", 0.0)),
        "cost.take_profit_cost_floor_bps": float(getattr(variant, "take_profit_cost_floor_bps", 0.0)),
    }
    sweep = getattr(strategy, "_last_sweep", None)
    if isinstance(sweep, dict):
        features["hft.sweep_notional_usdt"] = float(sweep.get("notional_usdt", 0.0) or 0.0)
        receive_time_ns = int(getattr(book, "receive_time_ns", 0))
        sweep_time_ns = int(sweep.get("receive_time_ns", receive_time_ns) or receive_time_ns)
        features["hft.sweep_age_ms"] = max(0.0, (receive_time_ns - sweep_time_ns) / 1_000_000.0)
    features["hft.replenishment_ratio"] = float(getattr(variant, "replenishment_ratio", 0.0))
    lead_move = getattr(strategy, "_last_lead_move_bps", None)
    if lead_move is not None:
        features["hft.lead_move_bps"] = float(lead_move)
    if signal is not None:
        raw_features = getattr(signal, "features", {}) or {}
        if "divergence_bps" in raw_features:
            features["hft.lag_divergence_bps"] = float(raw_features["divergence_bps"])
        if "lead_move_bps" in raw_features:
            features["hft.lead_move_bps"] = float(raw_features["lead_move_bps"])
        if "sweep_notional_usdt" in raw_features:
            features["hft.sweep_notional_usdt"] = float(raw_features["sweep_notional_usdt"])
        if "sweep_age_ms" in raw_features:
            features["hft.sweep_age_ms"] = float(raw_features["sweep_age_ms"])
    return features


def observation_from_book(book: Any) -> MarketObservation:
    return MarketObservation(
        symbol=str(getattr(book, "symbol")),
        venue=str(getattr(book, "venue", "binance_usdm")),
        event_time_ms=int(getattr(book, "event_time_ms")),
        receive_time_ns=int(getattr(book, "receive_time_ns")),
        bid=getattr(book, "best_bid", None),
        ask=getattr(book, "best_ask", None),
        bid_qty=getattr(book, "best_bid_qty", None),
        ask_qty=getattr(book, "best_ask_qty", None),
        source=str(getattr(book, "source", "l2_depth")),
    )


def audit_lookahead(
    samples: Sequence[FactorResearchSample],
    registry: Sequence[FactorDefinition],
) -> dict[str, LookaheadAuditResult]:
    results: dict[str, LookaheadAuditResult] = {}
    for factor in registry:
        flags: set[str] = set()
        checked = 0
        leaks = 0
        if not factor.usable_at_decision:
            flags.add("not_usable_at_decision")
        if any(_looks_like_future_field(field) for field in factor.input_fields):
            flags.add("future_field_name")
        for sample in samples:
            if factor.factor_id not in sample.features:
                continue
            checked += 1
            feature_time_ns = int(sample.feature_times_ns.get(factor.factor_id, sample.receive_time_ns))
            if feature_time_ns > sample.decision_time_ns:
                flags.add("feature_time_after_decision")
                leaks += 1
            if factor.max_lag_ns > 0 and sample.decision_time_ns - feature_time_ns > factor.max_lag_ns:
                flags.add("feature_lag_exceeds_contract")
        decision = "leak_suspect" if leaks or "not_usable_at_decision" in flags or "future_field_name" in flags else "ok"
        results[factor.factor_id] = LookaheadAuditResult(
            factor_id=factor.factor_id,
            decision=decision,
            flags=tuple(sorted(flags)),
            checked_count=checked,
            leak_count=leaks,
        )
    return results


def build_forward_labels(
    samples: Sequence[FactorResearchSample],
    observations: Sequence[MarketObservation],
    *,
    horizons_ns: Sequence[int],
    round_trip_fee_bps: float,
    take_profit_bps: float = 10.0,
    stop_bps: float = 2.5,
) -> list[ForwardLabel]:
    observations_by_symbol: dict[tuple[str, str], list[MarketObservation]] = {}
    for observation in observations:
        if observation.mid_price is None:
            continue
        observations_by_symbol.setdefault((observation.symbol, observation.venue), []).append(observation)
    for rows in observations_by_symbol.values():
        rows.sort(key=lambda item: item.receive_time_ns)

    labels: list[ForwardLabel] = []
    for sample in samples:
        side = _valid_decision_side(sample.side)
        if not side:
            continue
        rows = observations_by_symbol.get((sample.symbol, sample.venue), [])
        if not rows:
            continue
        base_price = sample.mid_price if sample.mid_price > 0 else _latest_mid_at_or_before(rows, sample.decision_time_ns)
        if base_price is None or base_price <= 0:
            continue
        for horizon_ns in horizons_ns:
            label_end_ns = sample.decision_time_ns + int(horizon_ns)
            end_observation = _first_observation_at_or_after(rows, label_end_ns)
            if end_observation is None or end_observation.mid_price is None:
                continue
            path = [
                row for row in rows
                if sample.decision_time_ns <= row.receive_time_ns <= label_end_ns and row.mid_price is not None
            ]
            if end_observation not in path:
                path.append(end_observation)
            signed_returns = [
                _signed_return_bps(float(row.mid_price), base_price, side)
                for row in path
                if row.mid_price is not None
            ]
            signed_end = _signed_return_bps(float(end_observation.mid_price), base_price, side)
            forward_return = (float(end_observation.mid_price) / base_price - 1.0) * 10_000.0
            labels.append(
                ForwardLabel(
                    sample_id=sample.sample_id,
                    horizon_ns=int(horizon_ns),
                    label_start_ns=sample.decision_time_ns,
                    label_end_ns=label_end_ns,
                    forward_return_bps=forward_return,
                    net_taker_pnl_bps=signed_end - float(round_trip_fee_bps),
                    mfe_bps=max(signed_returns) if signed_returns else signed_end,
                    mae_bps=min(signed_returns) if signed_returns else signed_end,
                    hit_take_profit=any(value >= take_profit_bps for value in signed_returns),
                    hit_stop_loss=any(value <= -abs(stop_bps) for value in signed_returns),
                )
            )
    return labels


def purged_walk_forward_splits(
    samples: Sequence[FactorResearchSample],
    labels: Sequence[ForwardLabel],
    *,
    n_splits: int,
    embargo_ns: int,
    horizon_ns: int | None = None,
) -> list[WalkForwardSplit]:
    label_by_sample = _label_by_sample_for_horizon(labels, horizon_ns)
    eligible = [sample for sample in samples if sample.sample_id in label_by_sample]
    eligible.sort(key=lambda item: item.decision_time_ns)
    if n_splits <= 0 or len(eligible) < 3:
        return []
    fold_size = max(1, len(eligible) // (n_splits + 1))
    splits: list[WalkForwardSplit] = []
    for fold in range(n_splits):
        start = min(len(eligible), (fold + 1) * fold_size)
        end = min(len(eligible), start + fold_size)
        validation = eligible[start:end]
        if not validation:
            continue
        validation_ids = {sample.sample_id for sample in validation}
        val_start = min(sample.decision_time_ns for sample in validation)
        val_end = max(label_by_sample[sample.sample_id].label_end_ns for sample in validation)
        train = []
        for sample in eligible:
            if sample.sample_id in validation_ids:
                continue
            label = label_by_sample[sample.sample_id]
            if label.label_end_ns + embargo_ns <= val_start or sample.decision_time_ns >= val_end + embargo_ns:
                train.append(sample)
        if train:
            splits.append(WalkForwardSplit(tuple(train), tuple(validation)))
    return splits


def score_factors(
    samples: Sequence[FactorResearchSample],
    labels: Sequence[ForwardLabel],
    registry: Sequence[FactorDefinition],
    *,
    audit_results: dict[str, LookaheadAuditResult] | None = None,
    min_samples: int = 30,
    horizon_ns: int | None = None,
) -> list[FactorScore]:
    label_by_sample = _label_by_sample_for_horizon(labels, horizon_ns)
    results: list[FactorScore] = []
    for factor in registry:
        audit = (audit_results or {}).get(factor.factor_id)
        pairs = [
            (float(sample.features[factor.factor_id]), label_by_sample[sample.sample_id].net_taker_pnl_bps)
            for sample in samples
            if factor.factor_id in sample.features and sample.sample_id in label_by_sample
        ]
        if audit is not None and audit.decision == "leak_suspect":
            results.append(FactorScore(factor.factor_id, len(pairs), 0.0, 0.0, 0.0, 0.0, 0.0, "leak_suspect", audit.flags))
            continue
        if len(pairs) < min_samples:
            results.append(FactorScore(factor.factor_id, len(pairs), 0.0, 0.0, 0.0, 0.0, 0.0, "insufficient_sample", ()))
            continue
        values = [item[0] for item in pairs]
        targets = [item[1] for item in pairs]
        ic = _pearson(values, targets)
        rank_ic = _pearson(_ranks(values), _ranks(targets))
        mean_net = sum(targets) / len(targets)
        top_bin = _top_bin_mean(values, targets, ic)
        hit_rate = sum(1 for target in targets if target > 0) / len(targets)
        if abs(ic) >= 0.15 and top_bin > 0:
            decision = "keep"
        elif abs(ic) >= 0.08 and top_bin > 0:
            decision = "watch"
        else:
            decision = "drop"
        results.append(
            FactorScore(
                factor_id=factor.factor_id,
                sample_count=len(pairs),
                ic=ic,
                rank_ic=rank_ic,
                mean_net_taker_pnl_bps=mean_net,
                top_bin_net_taker_pnl_bps=top_bin,
                hit_rate=hit_rate,
                decision=decision,
                flags=(),
            )
        )
    return results


def run_factor_experiment(
    store: FactorResearchStore,
    registry: Sequence[FactorDefinition],
    config: FactorExperimentConfig,
) -> FactorExperimentResult:
    store.register_factors(registry)
    samples = store.list_samples()
    labels = store.list_labels()
    if config.from_time_ns is not None or config.to_time_ns is not None:
        samples, labels = _filter_samples_labels_by_window(
            samples,
            labels,
            from_time_ns=config.from_time_ns,
            to_time_ns=config.to_time_ns,
        )
    audit = audit_lookahead(samples, registry)
    primary_horizon_ns = _primary_horizon_ns(labels)
    scores = score_factors(
        samples,
        labels,
        registry,
        audit_results=audit,
        min_samples=config.min_samples,
        horizon_ns=primary_horizon_ns,
    )
    splits = purged_walk_forward_splits(
        samples,
        labels,
        n_splits=config.n_splits,
        embargo_ns=config.embargo_ns,
        horizon_ns=primary_horizon_ns,
    ) if config.n_splits > 0 else []
    report = build_factor_experiment_report(
        samples,
        labels,
        registry,
        scores,
        audit_results=audit,
        splits=splits,
        run_id=config.run_id,
        min_samples=config.min_samples,
    )
    artifact_dir = Path(config.artifact_dir) if config.artifact_dir is not None else store.path.parent / "factor_research_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{config.run_id}.factor_report.json"
    artifact_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    store.record_scores(config.run_id, scores)
    summary = _experiment_summary(report)
    store.record_experiment_run(
        run_id=config.run_id,
        config={
            "min_samples": config.min_samples,
            "n_splits": config.n_splits,
            "embargo_ns": config.embargo_ns,
            "artifact_dir": str(artifact_dir),
            "from_time_ns": config.from_time_ns,
            "to_time_ns": config.to_time_ns,
        },
        status="ok",
        summary=summary,
    )
    store.record_artifact(
        run_id=config.run_id,
        artifact_type="factor_report",
        path=str(artifact_path),
        payload=report,
    )
    return FactorExperimentResult(
        run_id=config.run_id,
        scores=tuple(scores),
        report=report,
        artifact_path=str(artifact_path),
    )


def build_factor_experiment_report(
    samples: Sequence[FactorResearchSample],
    labels: Sequence[ForwardLabel],
    registry: Sequence[FactorDefinition],
    scores: Sequence[FactorScore],
    *,
    audit_results: dict[str, LookaheadAuditResult] | None = None,
    splits: Sequence[WalkForwardSplit] = (),
    run_id: str = "",
    min_samples: int = 30,
) -> dict[str, Any]:
    primary_horizon_ns = _primary_horizon_ns(labels)
    primary_scores = list(scores) if scores else score_factors(
        samples,
        labels,
        registry,
        audit_results=audit_results,
        min_samples=min_samples,
        horizon_ns=primary_horizon_ns,
    )
    primary_sections = _factor_report_sections(
        samples,
        labels,
        registry,
        primary_scores,
        audit_results=audit_results,
        splits=splits,
        horizon_ns=primary_horizon_ns,
    )
    horizon_reports: dict[str, dict[str, Any]] = {}
    for horizon in _horizon_values(labels):
        horizon_scores = primary_scores if horizon == primary_horizon_ns else score_factors(
            samples,
            labels,
            registry,
            audit_results=audit_results,
            min_samples=min_samples,
            horizon_ns=horizon,
        )
        horizon_sections = _factor_report_sections(
            samples,
            labels,
            registry,
            horizon_scores,
            audit_results=audit_results,
            splits=splits if horizon == primary_horizon_ns else (),
            horizon_ns=horizon,
        )
        horizon_reports[str(horizon)] = {
            "horizon_ns": horizon,
            "label_count": len(_labels_for_horizon(labels, horizon)),
            "factor_conclusions": horizon_sections["factor_conclusions"],
            "interference_matrix": horizon_sections["interference_matrix"],
            "ablation": horizon_sections["ablation"],
        }
    return {
        "run_id": run_id,
        "generated_at_ns": time.monotonic_ns(),
        "sample_count": len(samples),
        "label_count": len(labels),
        "factor_count": len(registry),
        "primary_horizon_ns": primary_horizon_ns,
        "walk_forward": {
            "split_count": len(splits),
            "purged": True,
        },
        "factor_conclusions": primary_sections["factor_conclusions"],
        "correlation_pairs": primary_sections["correlation_pairs"],
        "interference_matrix": primary_sections["interference_matrix"],
        "ablation": primary_sections["ablation"],
        "horizon_reports": horizon_reports,
    }


def _factor_report_sections(
    samples: Sequence[FactorResearchSample],
    labels: Sequence[ForwardLabel],
    registry: Sequence[FactorDefinition],
    scores: Sequence[FactorScore],
    *,
    audit_results: dict[str, LookaheadAuditResult] | None,
    splits: Sequence[WalkForwardSplit],
    horizon_ns: int | None,
) -> dict[str, Any]:
    label_by_sample = _label_by_sample_for_horizon(labels, horizon_ns)
    score_by_factor = {score.factor_id: score for score in scores}
    factor_ids = [factor.factor_id for factor in registry]
    oos = _oos_factor_metrics(samples, label_by_sample, factor_ids, splits)
    correlation_pairs = _factor_correlation_pairs(samples, factor_ids)
    conclusions: dict[str, dict[str, Any]] = {}
    for factor in registry:
        score = score_by_factor.get(factor.factor_id)
        audit = (audit_results or {}).get(factor.factor_id)
        factor_samples = [sample for sample in samples if factor.factor_id in sample.features and sample.sample_id in label_by_sample]
        recommended_threshold = None
        if score is not None and factor_samples and score.decision not in {"leak_suspect", "insufficient_sample"}:
            values = [sample.features[factor.factor_id] for sample in factor_samples]
            recommended_threshold = _recommended_threshold(values, score.ic)
        oos_metrics = oos.get(factor.factor_id, {})
        decision = score.decision if score is not None else "insufficient_sample"
        flags = set(score.flags if score is not None else ())
        if audit is not None:
            flags.update(audit.flags)
        if decision in {"keep", "watch"} and oos_metrics.get("split_count", 0) > 0 and oos_metrics.get("mean_net_bps", 0.0) <= 0:
            decision = "watch" if score and abs(score.ic) >= 0.15 else "drop"
        if decision == "watch" and splits and oos_metrics.get("split_count", 0) == 0:
            decision = "drop"
        conclusions[factor.factor_id] = {
            "decision": decision,
            "horizon_ns": horizon_ns or 0,
            "sample_count": score.sample_count if score is not None else 0,
            "ic": score.ic if score is not None else 0.0,
            "rank_ic": score.rank_ic if score is not None else 0.0,
            "mean_net_taker_pnl_bps": score.mean_net_taker_pnl_bps if score is not None else 0.0,
            "top_bin_net_taker_pnl_bps": score.top_bin_net_taker_pnl_bps if score is not None else 0.0,
            "oos_mean_net_bps": oos_metrics.get("mean_net_bps", 0.0),
            "oos_positive_split_rate": oos_metrics.get("positive_split_rate", 0.0),
            "recommended_threshold": recommended_threshold,
            "threshold_candidates": _threshold_candidate_rows(
                factor.factor_id,
                factor_samples,
                label_by_sample,
                score.ic if score else 0.0,
            ) if factor_samples and decision in {"keep", "watch"} else [],
            "flags": sorted(flags),
            "stability": _factor_stability_by_symbol(factor.factor_id, factor_samples, label_by_sample, score.ic if score else 0.0),
        }
    return {
        "factor_conclusions": conclusions,
        "correlation_pairs": correlation_pairs,
        "interference_matrix": _interference_matrix(correlation_pairs, conclusions),
        "ablation": _ablation_summary(conclusions),
    }


def generate_research_candidate_config(
    *,
    active_config_path: str | Path,
    output_dir: str | Path,
    run_id: str,
    report: dict[str, Any],
) -> Path:
    active = Path(active_config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    conclusions = report.get("factor_conclusions", {})
    threshold_candidates: dict[str, float] = {}
    blocked: list[str] = []
    evidence: dict[str, Any] = {}
    for factor_id, row in sorted(conclusions.items()):
        decision = str(row.get("decision", ""))
        if decision == "leak_suspect":
            blocked.append(factor_id)
            continue
        if decision not in {"keep", "watch"}:
            continue
        threshold = row.get("recommended_threshold")
        if threshold is None:
            continue
        threshold_candidates[factor_id] = float(threshold)
        evidence[factor_id] = {
            "decision": decision,
            "ic": float(row.get("ic", 0.0)),
            "oos_mean_net_bps": float(row.get("oos_mean_net_bps", 0.0)),
            "sample_count": int(row.get("sample_count", 0)),
            "threshold_candidates": row.get("threshold_candidates", []),
        }
    payload = {
        "shadow_only": True,
        "run_id": run_id,
        "source_config": str(active),
        "generated_at_ns": time.monotonic_ns(),
        "promotion_gate": {
            "requires_manual_approval": True,
            "requires_shadow_positive_oos": True,
            "requires_no_leak_suspect": True,
        },
        "factor_threshold_candidates": threshold_candidates,
        "blocked_factors": blocked,
        "evidence": evidence,
        "active_config_snapshot": json.loads(active.read_text(encoding="utf-8")),
    }
    path = output / f"{active.stem}.{run_id}.research.candidate.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def build_daily_research_summary(
    store: FactorResearchStore,
    registry: Sequence[FactorDefinition],
    *,
    maker_fills: Sequence[MakerFillInput] = (),
    maker_ledger_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    latest = store.latest_experiment_summary()
    artifact = store.latest_artifact(run_id=str(latest.get("run_id", "")), artifact_type="factor_report") if latest else {}
    report = artifact.get("payload", {}) if artifact else {}
    conclusions = report.get("factor_conclusions", {})
    if not conclusions:
        samples = store.list_samples()
        labels = store.list_labels()
        scores = score_factors(samples, labels, registry, audit_results=audit_lookahead(samples, registry), min_samples=1)
        report = build_factor_experiment_report(samples, labels, registry, scores)
        conclusions = report.get("factor_conclusions", {})
    status_counts = {status: 0 for status in ("keep", "watch", "drop", "leak_suspect", "insufficient_sample")}
    for row in conclusions.values():
        status = str(row.get("decision", "insufficient_sample"))
        status_counts[status] = status_counts.get(status, 0) + 1
    fills = list(maker_fills)
    for path in maker_ledger_paths:
        fills.extend(load_maker_fills_from_ledger(path))
    maker_pnl_by_symbol = decompose_maker_pnl_by_symbol(fills)
    maker_pnl_total = _sum_maker_pnl(maker_pnl_by_symbol.values())
    return {
        "status": "ok",
        "latest_experiment": latest,
        "sample_count": len(store.list_samples()),
        "label_count": len(store.list_labels()),
        "factor_count": len(registry),
        "factor_status_counts": status_counts,
        "maker_pnl": asdict(maker_pnl_total),
        "maker_pnl_total": asdict(maker_pnl_total),
        "maker_pnl_by_symbol": {symbol: asdict(row) for symbol, row in maker_pnl_by_symbol.items()},
    }


def load_maker_fills_from_ledger(path: str | Path) -> list[MakerFillInput]:
    ledger_path = Path(path)
    if not ledger_path.exists():
        return []
    conn = sqlite3.connect(str(ledger_path), timeout=_SQLITE_BUSY_TIMEOUT_SECONDS)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            select fill_id, order_id, symbol, side, price, qty, fee_usdt, liquidity,
                   event_time_ms, receive_time_ns
            from mm_fills
            order by receive_time_ns, id
            """
        ).fetchall()
    except sqlite3.DatabaseError:
        return []
    finally:
        conn.close()
    return [
        MakerFillInput(
            fill_id=str(row["fill_id"]),
            order_id=str(row["order_id"]),
            symbol=str(row["symbol"]),
            side=str(row["side"]),
            price=float(row["price"]),
            qty=float(row["qty"]),
            fee_usdt=float(row["fee_usdt"]),
            liquidity=str(row["liquidity"]),
            event_time_ms=int(row["event_time_ms"]),
            receive_time_ns=int(row["receive_time_ns"]),
        )
        for row in rows
    ]


def decompose_maker_pnl(fills: Sequence[MakerFillInput]) -> MakerPnlDecomposition:
    position_qty = 0.0
    avg_price = 0.0
    price_pnl = 0.0
    fees = 0.0
    cycle_price_pnl = 0.0
    cycle_fees = 0.0
    completed_cycles = 0
    wins = 0
    losses = 0
    for fill in sorted(fills, key=lambda row: row.receive_time_ns):
        signed_qty = fill.qty if fill.side.lower() == "buy" else -fill.qty
        fees += fill.fee_usdt
        cycle_fees += fill.fee_usdt
        if abs(position_qty) <= 1e-12:
            position_qty = signed_qty
            avg_price = fill.price
            continue
        same_direction = (position_qty > 0 and signed_qty > 0) or (position_qty < 0 and signed_qty < 0)
        if same_direction:
            new_abs_qty = abs(position_qty) + abs(signed_qty)
            avg_price = (avg_price * abs(position_qty) + fill.price * abs(signed_qty)) / max(new_abs_qty, 1e-12)
            position_qty += signed_qty
            continue
        closing_qty = min(abs(position_qty), abs(signed_qty))
        if position_qty > 0:
            realized = (fill.price - avg_price) * closing_qty
        else:
            realized = (avg_price - fill.price) * closing_qty
        price_pnl += realized
        cycle_price_pnl += realized
        position_qty += signed_qty
        if abs(position_qty) <= 1e-12:
            cycle_net = cycle_price_pnl - cycle_fees
            completed_cycles += 1
            if cycle_net > 0:
                wins += 1
            elif cycle_net < 0:
                losses += 1
            position_qty = 0.0
            avg_price = 0.0
            cycle_price_pnl = 0.0
            cycle_fees = 0.0
        elif abs(signed_qty) > closing_qty:
            avg_price = fill.price
    net = price_pnl - fees
    return MakerPnlDecomposition(
        price_pnl_usdt=price_pnl,
        fees_usdt=fees,
        net_pnl_usdt=net,
        completed_cycles=completed_cycles,
        wins=wins,
        losses=losses,
        win_rate=wins / completed_cycles if completed_cycles else 0.0,
    )


def decompose_maker_pnl_by_symbol(fills: Sequence[MakerFillInput]) -> dict[str, MakerPnlDecomposition]:
    fills_by_symbol: dict[str, list[MakerFillInput]] = {}
    for fill in fills:
        fills_by_symbol.setdefault(fill.symbol, []).append(fill)
    return {
        symbol: decompose_maker_pnl(symbol_fills)
        for symbol, symbol_fills in sorted(fills_by_symbol.items())
    }


def _sum_maker_pnl(rows: Sequence[MakerPnlDecomposition]) -> MakerPnlDecomposition:
    price_pnl = sum(row.price_pnl_usdt for row in rows)
    fees = sum(row.fees_usdt for row in rows)
    completed_cycles = sum(row.completed_cycles for row in rows)
    wins = sum(row.wins for row in rows)
    losses = sum(row.losses for row in rows)
    return MakerPnlDecomposition(
        price_pnl_usdt=price_pnl,
        fees_usdt=fees,
        net_pnl_usdt=price_pnl - fees,
        completed_cycles=completed_cycles,
        wins=wins,
        losses=losses,
        win_rate=wins / completed_cycles if completed_cycles else 0.0,
    )


def write_shadow_candidate_config(
    *,
    active_config_path: str | Path,
    output_dir: str | Path,
    run_id: str,
    kept_factors: Sequence[str],
    dropped_factors: Sequence[str],
) -> Path:
    active = Path(active_config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "shadow_only": True,
        "run_id": run_id,
        "source_config": str(active),
        "generated_at_ns": time.monotonic_ns(),
        "kept_factors": list(kept_factors),
        "dropped_factors": list(dropped_factors),
        "active_config_snapshot": json.loads(active.read_text(encoding="utf-8")),
    }
    path = output / f"{active.stem}.{run_id}.candidate.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch7 factor research tools")
    root = parser.add_subparsers(dest="scope", required=True)
    batch7 = root.add_parser("batch7")
    actions = batch7.add_subparsers(dest="action", required=True)

    audit_parser = actions.add_parser("audit")
    _add_common_batch7_args(audit_parser)

    build_parser = actions.add_parser("build-dataset")
    _add_common_batch7_args(build_parser)
    build_parser.add_argument("--horizons", default="250ms,1s,3s,5s")
    build_parser.add_argument("--round-trip-fee-bps", type=float, default=8.0)
    build_parser.add_argument("--from", dest="from_time", default="")
    build_parser.add_argument("--to", dest="to_time", default="")

    score_parser = actions.add_parser("score")
    _add_common_batch7_args(score_parser)
    score_parser.add_argument("--min-samples", type=int, default=30)

    experiment_parser = actions.add_parser("experiment")
    _add_common_batch7_args(experiment_parser)
    experiment_parser.add_argument("--run-id", default="")
    experiment_parser.add_argument("--artifact-dir", default="")
    experiment_parser.add_argument("--min-samples", type=int, default=30)
    experiment_parser.add_argument("--n-splits", type=int, default=5)
    experiment_parser.add_argument("--embargo-ms", type=float, default=250.0)

    shadow_parser = actions.add_parser("shadow-generate")
    _add_common_batch7_args(shadow_parser)
    shadow_parser.add_argument("--active-config", default=str(DEFAULT_BATCH7_EVENT_SIGNAL_CONFIG))
    shadow_parser.add_argument("--output-dir", default="")
    shadow_parser.add_argument("--run-id", default="latest")

    candidate_parser = actions.add_parser("candidate-generate")
    _add_common_batch7_args(candidate_parser)
    candidate_parser.add_argument("--active-config", default=str(DEFAULT_BATCH7_EVENT_SIGNAL_CONFIG))
    candidate_parser.add_argument("--output-dir", default="")
    candidate_parser.add_argument("--run-id", default="latest")

    report_parser = actions.add_parser("report")
    _add_common_batch7_args(report_parser)
    report_parser.add_argument("--latest", action="store_true")
    report_parser.add_argument("--run-id", default="")

    summary_parser = actions.add_parser("daily-summary")
    _add_common_batch7_args(summary_parser)
    summary_parser.add_argument("--maker-ledger", action="append", default=[])

    pipeline_parser = actions.add_parser("pipeline-run")
    _add_common_batch7_args(pipeline_parser)
    pipeline_parser.add_argument("--active-config", default=str(DEFAULT_BATCH7_EVENT_SIGNAL_CONFIG))
    pipeline_parser.add_argument("--output-dir", default="")
    pipeline_parser.add_argument("--artifact-dir", default="")
    pipeline_parser.add_argument("--run-id", default="")
    pipeline_parser.add_argument("--horizons", default="250ms,1s,3s,5s")
    pipeline_parser.add_argument("--round-trip-fee-bps", type=float, default=8.0)
    pipeline_parser.add_argument("--min-samples", type=int, default=30)
    pipeline_parser.add_argument("--n-splits", type=int, default=5)
    pipeline_parser.add_argument("--embargo-ms", type=float, default=250.0)
    pipeline_parser.add_argument("--from", dest="from_time", default="")
    pipeline_parser.add_argument("--to", dest="to_time", default="")
    pipeline_parser.add_argument("--maker-ledger", action="append", default=[])

    args = parser.parse_args(argv)
    if args.scope != "batch7":
        parser.error("only batch7 is supported")
    store = FactorResearchStore(args.research_db)
    registry = batch7_factor_registry()
    store.register_factors(registry)
    manifest = _load_json(Path(args.manifest))

    if args.action == "audit":
        samples = store.list_samples()
        audit = audit_lookahead(samples, registry)
        leak_suspects = [factor_id for factor_id, result in audit.items() if result.decision == "leak_suspect"]
        _print_json(
            {
                "status": "ok",
                "batch_id": manifest.get("batch_id", "scalp-suite-batch7-24bot-paper-v1"),
                "total_bots": int(manifest.get("total_bots", 24)),
                "factor_count": len(registry),
                "sample_count": len(samples),
                "critical_count": len(leak_suspects),
                "leak_suspects": leak_suspects,
            }
        )
        return 0

    if args.action == "build-dataset":
        horizons_ns = _parse_horizons(args.horizons)
        samples, observations = filter_research_window(
            store.list_samples(),
            store.list_observations(),
            from_time=args.from_time,
            to_time=args.to_time,
            label_horizon_ns=max(horizons_ns, default=0),
        )
        labels = build_forward_labels(
            samples,
            observations,
            horizons_ns=horizons_ns,
            round_trip_fee_bps=args.round_trip_fee_bps,
        )
        store.record_labels(labels)
        _print_json({
            "status": "ok",
            "label_count": len(labels),
            "horizons": args.horizons,
            "from_time": args.from_time,
            "to_time": args.to_time,
        })
        return 0

    if args.action == "score":
        samples = store.list_samples()
        labels = store.list_labels()
        audit = audit_lookahead(samples, registry)
        scores = score_factors(samples, labels, registry, audit_results=audit, min_samples=args.min_samples)
        run_id = f"score-{time.strftime('%Y%m%dT%H%M%S')}"
        store.record_scores(run_id, scores)
        _print_json({"status": "ok", "run_id": run_id, "scores": [_score_to_json(score) for score in scores]})
        return 0

    if args.action == "experiment":
        run_id = args.run_id or f"experiment-{time.strftime('%Y%m%dT%H%M%S')}"
        artifact_dir = args.artifact_dir or "output/fleet/scalp_suite_batch7_24bot/factor_research_artifacts"
        result = run_factor_experiment(
            store,
            registry,
            FactorExperimentConfig(
                run_id=run_id,
                min_samples=args.min_samples,
                n_splits=args.n_splits,
                embargo_ns=int(args.embargo_ms * 1_000_000),
                artifact_dir=artifact_dir,
            ),
        )
        _print_json(
            {
                "status": "ok",
                "run_id": result.run_id,
                "artifact_path": result.artifact_path,
                "factor_status_counts": _experiment_summary(result.report)["factor_status_counts"],
                "walk_forward": result.report.get("walk_forward", {}),
            }
        )
        return 0

    if args.action == "shadow-generate":
        output_dir = Path(args.output_dir) if args.output_dir else Path(args.active_config).parent
        labels = store.list_labels()
        samples = store.list_samples()
        audit = audit_lookahead(samples, registry)
        scores = score_factors(samples, labels, registry, audit_results=audit, min_samples=1) if labels else []
        kept = [score.factor_id for score in scores if score.decision in {"keep", "watch"}]
        dropped = [score.factor_id for score in scores if score.decision == "drop"]
        candidate = write_shadow_candidate_config(
            active_config_path=args.active_config,
            output_dir=output_dir,
            run_id=args.run_id,
            kept_factors=kept,
            dropped_factors=dropped,
        )
        _print_json({"status": "ok", "candidate_config": str(candidate), "shadow_only": True})
        return 0

    if args.action == "candidate-generate":
        output_dir = Path(args.output_dir) if args.output_dir else Path(args.active_config).parent
        artifact = store.latest_artifact(
            run_id="" if args.run_id == "latest" else args.run_id,
            artifact_type="factor_report",
        )
        report = artifact.get("payload")
        candidate_run_id = str(artifact.get("run_id", args.run_id))
        if not report:
            result = run_factor_experiment(
                store,
                registry,
                FactorExperimentConfig(
                    run_id=f"candidate-source-{time.strftime('%Y%m%dT%H%M%S')}",
                    min_samples=1,
                    n_splits=0,
                    artifact_dir=output_dir,
                ),
            )
            report = result.report
            candidate_run_id = result.run_id
        candidate = generate_research_candidate_config(
            active_config_path=args.active_config,
            output_dir=output_dir,
            run_id=candidate_run_id,
            report=report,
        )
        _print_json({"status": "ok", "candidate_config": str(candidate), "shadow_only": True})
        return 0

    if args.action == "report":
        _print_json(
            {
                "status": "ok",
                "batch_id": manifest.get("batch_id", "scalp-suite-batch7-24bot-paper-v1"),
                "total_bots": int(manifest.get("total_bots", 24)),
                "factor_count": store.factor_count(),
                "sample_count": len(store.list_samples()),
                "label_count": len(store.list_labels()),
                "latest": bool(args.latest),
                "run_id": args.run_id,
                "latest_experiment": store.latest_experiment_summary(),
            }
        )
        return 0

    if args.action == "daily-summary":
        summary = build_daily_research_summary(store, registry, maker_ledger_paths=args.maker_ledger)
        _print_json(summary)
        return 0

    if args.action == "pipeline-run":
        run_id = args.run_id or f"pipeline-{time.strftime('%Y%m%dT%H%M%S')}"
        output_dir = Path(args.output_dir) if args.output_dir else Path(args.active_config).parent
        artifact_dir = args.artifact_dir or str(output_dir / "factor_research_artifacts")
        horizons_ns = _parse_horizons(args.horizons)
        samples, observations = filter_research_window(
            store.list_samples(),
            store.list_observations(),
            from_time=args.from_time,
            to_time=args.to_time,
            label_horizon_ns=max(horizons_ns, default=0),
        )
        labels = build_forward_labels(
            samples,
            observations,
            horizons_ns=horizons_ns,
            round_trip_fee_bps=args.round_trip_fee_bps,
        )
        store.record_labels(labels)
        from_time_ns = _parse_time_bound_ns(args.from_time)
        to_time_ns = _parse_time_bound_ns(args.to_time)
        result = run_factor_experiment(
            store,
            registry,
            FactorExperimentConfig(
                run_id=run_id,
                min_samples=args.min_samples,
                n_splits=args.n_splits,
                embargo_ns=int(args.embargo_ms * 1_000_000),
                artifact_dir=artifact_dir,
                from_time_ns=from_time_ns,
                to_time_ns=to_time_ns,
            ),
        )
        candidate = generate_research_candidate_config(
            active_config_path=args.active_config,
            output_dir=output_dir,
            run_id=run_id,
            report=result.report,
        )
        summary = build_daily_research_summary(store, registry, maker_ledger_paths=args.maker_ledger)
        _print_json(
            {
                "status": "ok",
                "run_id": run_id,
                "label_count": len(labels),
                "artifact_path": result.artifact_path,
                "candidate_config": str(candidate),
                "summary": summary,
            }
        )
        return 0

    parser.error(f"unsupported action: {args.action}")
    return 2


def _add_common_batch7_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", default=str(DEFAULT_BATCH7_MANIFEST_PATH))
    parser.add_argument("--research-db", default="output/fleet/scalp_suite_batch7_24bot/batch7_factor_research.sqlite3")


def _parse_horizons(raw: str) -> list[int]:
    horizons: list[int] = []
    for item in raw.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token.endswith("ms"):
            horizons.append(int(float(token[:-2]) * 1_000_000))
        elif token.endswith("s"):
            horizons.append(int(float(token[:-1]) * 1_000_000_000))
        else:
            horizons.append(int(token))
    return horizons


def filter_research_window(
    samples: Sequence[FactorResearchSample],
    observations: Sequence[MarketObservation],
    *,
    from_time: str = "",
    to_time: str = "",
    label_horizon_ns: int = 0,
) -> tuple[list[FactorResearchSample], list[MarketObservation]]:
    from_time_ns = _parse_time_bound_ns(from_time)
    to_time_ns = _parse_time_bound_ns(to_time)
    observation_to_time_ns = None if to_time_ns is None else to_time_ns + max(0, int(label_horizon_ns))
    filtered_samples = [
        sample for sample in samples
        if _within_time_bounds(sample.decision_time_ns, from_time_ns=from_time_ns, to_time_ns=to_time_ns)
    ]
    filtered_observations = [
        observation for observation in observations
        if _within_time_bounds(observation.receive_time_ns, from_time_ns=from_time_ns, to_time_ns=observation_to_time_ns)
    ]
    return filtered_samples, filtered_observations


def _parse_time_bound_ns(raw: str | None) -> int | None:
    token = (raw or "").strip()
    if not token:
        return None
    lowered = token.lower()
    try:
        if lowered.endswith("ns"):
            return int(float(lowered[:-2]))
        if lowered.endswith("ms"):
            return int(float(lowered[:-2]) * 1_000_000)
        if lowered.endswith("s"):
            return int(float(lowered[:-1]) * 1_000_000_000)
        return int(token)
    except ValueError:
        iso_token = token.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(iso_token)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp() * 1_000_000_000)


def _filter_samples_labels_by_window(
    samples: Sequence[FactorResearchSample],
    labels: Sequence[ForwardLabel],
    *,
    from_time_ns: int | None,
    to_time_ns: int | None,
) -> tuple[list[FactorResearchSample], list[ForwardLabel]]:
    filtered_samples = [
        sample for sample in samples
        if _within_time_bounds(sample.decision_time_ns, from_time_ns=from_time_ns, to_time_ns=to_time_ns)
    ]
    sample_ids = {sample.sample_id for sample in filtered_samples}
    filtered_labels = [label for label in labels if label.sample_id in sample_ids]
    return filtered_samples, filtered_labels


def _within_time_bounds(value_ns: int, *, from_time_ns: int | None, to_time_ns: int | None) -> bool:
    if from_time_ns is not None and value_ns < from_time_ns:
        return False
    if to_time_ns is not None and value_ns > to_time_ns:
        return False
    return True


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_from_row(row: sqlite3.Row) -> FactorResearchSample:
    return FactorResearchSample(
        sample_id=str(row["sample_id"]),
        run_id=str(row["run_id"]),
        bot_id=str(row["bot_id"]),
        strategy_tree_id=str(row["strategy_tree_id"]),
        symbol=str(row["symbol"]),
        venue=str(row["venue"]),
        event_seq=int(row["event_seq"]),
        exchange_event_time_ms=int(row["exchange_event_time_ms"]),
        receive_time_ns=int(row["receive_time_ns"]),
        decision_time_ns=int(row["decision_time_ns"]),
        sample_type=str(row["sample_type"]),
        fired=bool(row["fired"]),
        side=str(row["side"]),
        mid_price=float(row["mid_price"]),
        features={str(key): float(value) for key, value in json.loads(row["features_json"]).items()},
        feature_times_ns={str(key): int(value) for key, value in json.loads(row["feature_times_ns_json"]).items()},
        data_quality_flags=tuple(json.loads(row["data_quality_flags_json"])),
    )


def _factor_to_json(factor: FactorDefinition) -> dict[str, Any]:
    row = asdict(factor)
    row["strategy_tree_path"] = list(factor.strategy_tree_path)
    row["input_fields"] = list(factor.input_fields)
    row["version_hash"] = factor.version_hash
    return row


def _score_to_json(score: FactorScore) -> dict[str, Any]:
    row = asdict(score)
    row["flags"] = list(score.flags)
    return row


def _experiment_summary(report: dict[str, Any]) -> dict[str, Any]:
    conclusions = report.get("factor_conclusions", {})
    status_counts: dict[str, int] = {}
    for row in conclusions.values():
        status = str(row.get("decision", "insufficient_sample"))
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "sample_count": int(report.get("sample_count", 0)),
        "label_count": int(report.get("label_count", 0)),
        "factor_count": int(report.get("factor_count", 0)),
        "walk_forward": report.get("walk_forward", {}),
        "factor_status_counts": status_counts,
    }


def _valid_decision_side(side: str) -> str:
    normalized = side.lower().strip()
    return normalized if normalized in {"long", "short"} else ""


def _horizon_values(labels: Sequence[ForwardLabel]) -> list[int]:
    return sorted({int(label.horizon_ns) for label in labels})


def _primary_horizon_ns(labels: Sequence[ForwardLabel]) -> int | None:
    horizons = _horizon_values(labels)
    return horizons[0] if horizons else None


def _labels_for_horizon(labels: Sequence[ForwardLabel], horizon_ns: int | None) -> list[ForwardLabel]:
    selected_horizon = _primary_horizon_ns(labels) if horizon_ns is None else int(horizon_ns)
    if selected_horizon is None:
        return []
    return [label for label in labels if int(label.horizon_ns) == selected_horizon]


def _label_by_sample_for_horizon(labels: Sequence[ForwardLabel], horizon_ns: int | None = None) -> dict[str, ForwardLabel]:
    label_by_sample: dict[str, ForwardLabel] = {}
    for label in _labels_for_horizon(labels, horizon_ns):
        label_by_sample.setdefault(label.sample_id, label)
    return label_by_sample


def _first_label_by_sample(labels: Sequence[ForwardLabel]) -> dict[str, ForwardLabel]:
    return _label_by_sample_for_horizon(labels)


def _oos_factor_metrics(
    samples: Sequence[FactorResearchSample],
    label_by_sample: dict[str, ForwardLabel],
    factor_ids: Sequence[str],
    splits: Sequence[WalkForwardSplit],
) -> dict[str, dict[str, float]]:
    if not splits:
        return {}
    metrics: dict[str, dict[str, float]] = {}
    for factor_id in factor_ids:
        split_means: list[float] = []
        for split in splits:
            train_pairs = _factor_pairs(split.train_samples, label_by_sample, factor_id)
            validation_pairs = _factor_pairs(split.validation_samples, label_by_sample, factor_id)
            if len(train_pairs) < 2 or not validation_pairs:
                continue
            train_values = [pair[0] for pair in train_pairs]
            train_targets = [pair[1] for pair in train_pairs]
            direction = _pearson(train_values, train_targets)
            split_means.append(_top_bin_mean([pair[0] for pair in validation_pairs], [pair[1] for pair in validation_pairs], direction))
        if split_means:
            metrics[factor_id] = {
                "mean_net_bps": sum(split_means) / len(split_means),
                "positive_split_rate": sum(1 for value in split_means if value > 0) / len(split_means),
                "split_count": float(len(split_means)),
            }
    return metrics


def _factor_pairs(
    samples: Sequence[FactorResearchSample],
    label_by_sample: dict[str, ForwardLabel],
    factor_id: str,
) -> list[tuple[float, float]]:
    return [
        (float(sample.features[factor_id]), label_by_sample[sample.sample_id].net_taker_pnl_bps)
        for sample in samples
        if factor_id in sample.features and sample.sample_id in label_by_sample
    ]


def _factor_correlation_pairs(samples: Sequence[FactorResearchSample], factor_ids: Sequence[str]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(factor_ids):
        for right in factor_ids[left_index + 1:]:
            joined = [
                (float(sample.features[left]), float(sample.features[right]))
                for sample in samples
                if left in sample.features and right in sample.features
            ]
            if len(joined) < 3:
                continue
            corr = _pearson([item[0] for item in joined], [item[1] for item in joined])
            if abs(corr) >= 0.80:
                pairs.append({"left": left, "right": right, "correlation": corr, "sample_count": len(joined)})
    pairs.sort(key=lambda item: abs(float(item["correlation"])), reverse=True)
    return pairs


def _interference_matrix(correlation_pairs: Sequence[dict[str, Any]], conclusions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    matrix: dict[str, dict[str, Any]] = {}
    for pair in correlation_pairs:
        left = str(pair["left"])
        right = str(pair["right"])
        key = f"{left}|{right}"
        left_decision = conclusions.get(left, {}).get("decision", "insufficient_sample")
        right_decision = conclusions.get(right, {}).get("decision", "insufficient_sample")
        matrix[key] = {
            "correlation": float(pair["correlation"]),
            "left_decision": left_decision,
            "right_decision": right_decision,
            "action": "prefer_stronger_oos" if left_decision == right_decision else "keep_non_dropped_side",
        }
    return matrix


def _ablation_summary(conclusions: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    kept_values = [
        float(row.get("oos_mean_net_bps", row.get("top_bin_net_taker_pnl_bps", 0.0)))
        for row in conclusions.values()
        if row.get("decision") in {"keep", "watch"}
    ]
    baseline = sum(kept_values) / len(kept_values) if kept_values else 0.0
    rows: dict[str, dict[str, float]] = {}
    for factor_id, row in conclusions.items():
        contribution = float(row.get("oos_mean_net_bps", row.get("top_bin_net_taker_pnl_bps", 0.0)))
        if row.get("decision") not in {"keep", "watch"}:
            contribution = 0.0
        rows[factor_id] = {
            "baseline_oos_mean_net_bps": baseline,
            "without_factor_oos_mean_net_bps": baseline - contribution,
            "delta_oos_mean_net_bps": contribution,
        }
    return rows


def _factor_stability_by_symbol(
    factor_id: str,
    samples: Sequence[FactorResearchSample],
    label_by_sample: dict[str, ForwardLabel],
    ic: float,
) -> dict[str, Any]:
    by_symbol: dict[str, list[tuple[float, float]]] = {}
    for sample in samples:
        if sample.sample_id not in label_by_sample:
            continue
        by_symbol.setdefault(sample.symbol, []).append((sample.features[factor_id], label_by_sample[sample.sample_id].net_taker_pnl_bps))
    symbol_rows: dict[str, dict[str, float]] = {}
    for symbol, pairs in by_symbol.items():
        if not pairs:
            continue
        values = [pair[0] for pair in pairs]
        targets = [pair[1] for pair in pairs]
        symbol_rows[symbol] = {
            "sample_count": float(len(pairs)),
            "top_bin_net_bps": _top_bin_mean(values, targets, ic),
        }
    positive = sum(1 for row in symbol_rows.values() if row["top_bin_net_bps"] > 0)
    return {
        "symbol_count": len(symbol_rows),
        "positive_symbol_count": positive,
        "positive_symbol_rate": positive / len(symbol_rows) if symbol_rows else 0.0,
        "by_symbol": symbol_rows,
    }


def _recommended_threshold(values: Sequence[float], ic: float) -> float:
    if not values:
        return 0.0
    quantile = 0.75 if ic >= 0 else 0.25
    return _quantile(sorted(values), quantile)


def _threshold_candidate_rows(
    factor_id: str,
    samples: Sequence[FactorResearchSample],
    label_by_sample: dict[str, ForwardLabel],
    ic: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    values = sorted(float(sample.features[factor_id]) for sample in samples if sample.sample_id in label_by_sample)
    if not values:
        return rows
    quantiles = (0.60, 0.75, 0.90) if ic >= 0 else (0.40, 0.25, 0.10)
    for quantile in quantiles:
        threshold = _quantile(values, quantile)
        selected = [
            label_by_sample[sample.sample_id].net_taker_pnl_bps
            for sample in samples
            if sample.sample_id in label_by_sample
            and (
                float(sample.features[factor_id]) >= threshold
                if ic >= 0
                else float(sample.features[factor_id]) <= threshold
            )
        ]
        if not selected:
            continue
        rows.append(
            {
                "quantile": quantile,
                "threshold": threshold,
                "sample_count": len(selected),
                "mean_net_bps": sum(selected) / len(selected),
                "hit_rate": sum(1 for value in selected if value > 0) / len(selected),
            }
        )
    rows.sort(key=lambda item: (float(item["mean_net_bps"]), int(item["sample_count"])), reverse=True)
    return rows


def _looks_like_future_field(field: str) -> bool:
    token = field.lower()
    future_tokens = ("future", "forward", "label", "target", "next_", "post_decision")
    return any(part in token for part in future_tokens)


def _quantile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    index = (len(sorted_values) - 1) * min(max(quantile, 0.0), 1.0)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return float(sorted_values[lower])
    weight = index - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _first_observation_at_or_after(rows: Sequence[MarketObservation], receive_time_ns: int) -> MarketObservation | None:
    for row in rows:
        if row.receive_time_ns >= receive_time_ns:
            return row
    return None


def _latest_mid_at_or_before(rows: Sequence[MarketObservation], receive_time_ns: int) -> float | None:
    latest = None
    for row in rows:
        if row.receive_time_ns > receive_time_ns:
            break
        latest = row.mid_price
    return latest


def _signed_return_bps(price: float, base_price: float, side: str) -> float:
    raw = (price / base_price - 1.0) * 10_000.0
    return -raw if side.lower() == "short" else raw


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_var = sum((a - left_mean) ** 2 for a in left)
    right_var = sum((b - right_mean) ** 2 for b in right)
    denom = math.sqrt(left_var * right_var)
    return numerator / denom if denom > 0 else 0.0


def _ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = (index + end - 1) / 2.0
        for ordered_index in range(index, end):
            ranks[ordered[ordered_index][0]] = rank
        index = end
    return ranks


def _top_bin_mean(values: Sequence[float], targets: Sequence[float], ic: float) -> float:
    rows = sorted(zip(values, targets), key=lambda item: item[0], reverse=ic >= 0)
    take = max(1, len(rows) // 4)
    selected = [target for _, target in rows[:take]]
    return sum(selected) / len(selected)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _print_json(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
