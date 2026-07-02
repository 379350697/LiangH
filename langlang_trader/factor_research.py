from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
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
                        label.label_start_ns,
                        label.label_end_ns,
                        label.forward_return_bps,
                        label.net_taker_pnl_bps,
                        label.mfe_bps,
                        label.mae_bps,
                        int(label.hit_take_profit),
                        int(label.hit_stop_loss),
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
                    hit_stop_loss integer not null
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
        decision = "leak_suspect" if leaks or "not_usable_at_decision" in flags else "ok"
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
                _signed_return_bps(float(row.mid_price), base_price, sample.side)
                for row in path
                if row.mid_price is not None
            ]
            signed_end = _signed_return_bps(float(end_observation.mid_price), base_price, sample.side)
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
) -> list[WalkForwardSplit]:
    label_by_sample = {label.sample_id: label for label in labels}
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
) -> list[FactorScore]:
    label_by_sample: dict[str, ForwardLabel] = {}
    for label in labels:
        label_by_sample.setdefault(label.sample_id, label)
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

    shadow_parser = actions.add_parser("shadow-generate")
    _add_common_batch7_args(shadow_parser)
    shadow_parser.add_argument("--active-config", default=str(DEFAULT_BATCH7_EVENT_SIGNAL_CONFIG))
    shadow_parser.add_argument("--output-dir", default="")
    shadow_parser.add_argument("--run-id", default="latest")

    report_parser = actions.add_parser("report")
    _add_common_batch7_args(report_parser)
    report_parser.add_argument("--latest", action="store_true")
    report_parser.add_argument("--run-id", default="")

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
        labels = build_forward_labels(
            store.list_samples(),
            store.list_observations(),
            horizons_ns=_parse_horizons(args.horizons),
            round_trip_fee_bps=args.round_trip_fee_bps,
        )
        store.record_labels(labels)
        _print_json({"status": "ok", "label_count": len(labels), "horizons": args.horizons})
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
