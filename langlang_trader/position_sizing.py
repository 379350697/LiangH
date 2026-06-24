from __future__ import annotations

from dataclasses import dataclass, field
from math import floor
from typing import Any, Protocol

from langlang_trader.config import RiskConfig
from langlang_trader.models import AccountSnapshot, Position, Signal


@dataclass(frozen=True)
class PositionSizeDecision:
    risk_unit: str
    risk_unit_w_usdt: float
    capital_step_level: int
    size_multiplier: float
    leverage: int
    margin_usdt: float
    notional_usdt: float
    capped_by: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    decision_trace: dict[str, Any] = field(default_factory=dict)


class PositionSizer(Protocol):
    def size(
        self,
        *,
        signal: Signal,
        account: AccountSnapshot,
        open_positions: list[Position],
        latest_price: float,
    ) -> PositionSizeDecision | None:
        ...


class LangLangPositionSizer:
    """LangLang W-unit sizing: position margin follows document wave quality."""

    def __init__(self, config: RiskConfig, *, initial_equity_usdt: float):
        self.config = config
        self.initial_equity_usdt = initial_equity_usdt

    def size(
        self,
        *,
        signal: Signal,
        account: AccountSnapshot,
        open_positions: list[Position],
        latest_price: float,
    ) -> PositionSizeDecision | None:
        if latest_price <= 0 or account.equity_usdt <= 0:
            return None
        trace = dict(getattr(signal, "decision_trace", {}) or {})
        features = dict(getattr(signal, "features", {}) or {})
        stop_loss_cluster = int(_number(features.get("stop_loss_cluster_24h"), 0.0))
        if stop_loss_cluster >= 3:
            return None

        capital_step_level = max(1, floor(account.equity_usdt / max(self.initial_equity_usdt, 1e-9)))
        active_capital = self.initial_equity_usdt * self.config.active_capital_fraction * capital_step_level
        risk_unit_w = active_capital / 3.0

        entry_position_id = str(trace.get("entry_position_id") or features.get("entry_position_id") or "")
        base_multiplier, risk_unit, entry_reason = _entry_position_multiplier(entry_position_id, getattr(signal, "setup", None))
        market_season = str(trace.get("market_season") or features.get("market_season") or "").lower()
        season_multiplier = _season_multiplier(market_season)
        signal_multiplier = _number(
            trace.get("position_size_multiplier", features.get("position_size_multiplier")),
            1.0,
        )
        cooldown_multiplier = 0.35 if stop_loss_cluster >= 2 else 1.0

        size_multiplier = base_multiplier * season_multiplier * signal_multiplier * cooldown_multiplier
        leverage = _leverage_for_symbol(str(getattr(signal, "symbol", "")), self.config)
        margin = risk_unit_w * size_multiplier
        notional = margin * leverage
        capped_by: list[str] = []

        if self.config.max_position_usdt is not None and notional > self.config.max_position_usdt:
            notional = self.config.max_position_usdt
            margin = notional / leverage
            capped_by.append("max_position_usdt")

        if self.config.max_total_position_usdt is not None:
            current_notional = sum(abs(position.qty * position.avg_price) for position in open_positions)
            remaining_notional = self.config.max_total_position_usdt - current_notional
            if remaining_notional <= 0:
                return None
            if notional > remaining_notional:
                notional = remaining_notional
                margin = notional / leverage
                capped_by.append("max_total_position_usdt")

        if notional <= 0 or margin <= 0:
            return None

        reason_codes = [
            f"entry_position:{entry_reason}",
            f"market_season:{market_season or 'unknown'}",
            f"leverage:{leverage}x",
        ]
        if stop_loss_cluster >= 2:
            reason_codes.append("stop_loss_cluster_reduce")
        if capped_by:
            reason_codes.extend(f"capped_by:{item}" for item in capped_by)

        banked_profit = max(0.0, (capital_step_level - 1) * self.initial_equity_usdt)
        decision_trace = {
            "risk_unit": risk_unit,
            "risk_unit_w_usdt": risk_unit_w,
            "capital_step_level": capital_step_level,
            "banked_profit_usdt": banked_profit,
            "position_size_multiplier": size_multiplier,
            "position_margin_usdt": margin,
            "position_notional_usdt": notional,
            "position_sizing_reason_codes": reason_codes,
            "position_sizing_capped_by": capped_by,
        }
        return PositionSizeDecision(
            risk_unit=risk_unit,
            risk_unit_w_usdt=risk_unit_w,
            capital_step_level=capital_step_level,
            size_multiplier=size_multiplier,
            leverage=leverage,
            margin_usdt=margin,
            notional_usdt=notional,
            capped_by=capped_by,
            reason_codes=reason_codes,
            decision_trace=decision_trace,
        )


def _entry_position_multiplier(entry_position_id: str, setup: Any) -> tuple[float, str, str]:
    normalized = entry_position_id.lower()
    setup_value = getattr(setup, "value", setup)
    if normalized.startswith("1_"):
        return 1.0, "W", "1_startup_long"
    if normalized.startswith("4_"):
        return 1.0, "W", "4_second_wave_long"
    if normalized.startswith("2_") or "small_divergence" in normalized:
        return 0.6, "0.6W", "2_small_divergence_entry"
    if normalized.startswith("6_") or "box_rebound" in normalized:
        return 0.25, "0.25W", "6_box_rebound_long"
    if normalized.startswith("3_") or normalized.startswith("5_") or "top_short" in normalized:
        return 0.15, "0.15W", normalized or "top_short"
    if "waterfall" in normalized:
        return 0.45, "0.45W", "short_waterfall_continuation"
    if "rebound_failure" in normalized:
        return 0.30, "0.3W", "short_rebound_failure"
    if setup_value == "starter_buy":
        return 1.0, "W", "1_startup_long"
    if setup_value == "post_divergence_rebound":
        return 1.0, "W", "4_second_wave_long"
    return 0.35, "0.35W", normalized or "unclassified_setup"


def _season_multiplier(season: str) -> float:
    return {
        "summer": 1.0,
        "spring": 0.7,
        "autumn": 0.35,
        "winter": 0.15,
    }.get(season, 1.0)


def _leverage_for_symbol(symbol: str, config: RiskConfig) -> int:
    if symbol.startswith("BTC-") or symbol.startswith("ETH-"):
        return config.reference_leverage
    return config.alt_leverage


def _number(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
