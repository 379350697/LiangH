from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langlang_trader.models import Position, Side


class ExitActionType(str, Enum):
    HOLD = "hold"
    MOVE_STOP = "move_stop"
    PARTIAL_TAKE_PROFIT = "partial_take_profit"
    CLOSE_POSITION = "close_position"


@dataclass(frozen=True)
class ExitManagementContext:
    position: Position
    latest_price: float | None
    entry_price: float
    initial_stop_loss: float | None
    current_stop_loss: float | None
    initial_risk_usdt: float | None
    mfe_usdt: float
    partial_taken: bool
    take_profit_plan: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    fee_bps: float = 0.0
    slippage_bps: float = 0.0


@dataclass(frozen=True)
class ExitManagementDecision:
    action: ExitActionType
    reason_codes: list[str] = field(default_factory=list)
    reason_summary: str = ""
    new_stop_loss: float | None = None
    reduce_qty: float | None = None
    data_quality_flags: list[str] = field(default_factory=list)
    decision_trace: dict[str, Any] = field(default_factory=dict)


class ExitManagementEngine:
    breakeven_at_r = 1.0
    mfe_trail_activation_r = 3.0
    normal_giveback_pct = 0.45
    risk_signal_giveback_pct = 0.30

    def evaluate(self, context: ExitManagementContext) -> ExitManagementDecision:
        if context.latest_price is None or context.latest_price <= 0:
            return ExitManagementDecision(
                action=ExitActionType.HOLD,
                reason_codes=["hold_no_exit_trigger"],
                reason_summary="missing realtime exit price",
                data_quality_flags=["missing_realtime_exit_price"],
            )
        if context.initial_stop_loss is None or context.initial_risk_usdt is None or context.initial_risk_usdt <= 0:
            return ExitManagementDecision(
                action=ExitActionType.HOLD,
                reason_codes=["hold_no_exit_trigger"],
                reason_summary="missing initial risk",
                data_quality_flags=["missing_initial_exit_risk"],
            )

        side = context.position.side
        current_pnl = self._open_pnl(context)
        current_r = current_pnl / context.initial_risk_usdt
        trace = {
            "latest_price": context.latest_price,
            "entry_price": context.entry_price,
            "initial_stop_loss": context.initial_stop_loss,
            "current_stop_loss": context.current_stop_loss,
            "initial_risk_usdt": context.initial_risk_usdt,
            "current_pnl_usdt": current_pnl,
            "current_r": current_r,
            "mfe_usdt": context.mfe_usdt,
        }

        partial_r = float(context.take_profit_plan.get("partial_r", 2.0) or 2.0)
        partial_fraction = float(context.take_profit_plan.get("partial_exit_fraction", 0.5) or 0.5)
        partial_fraction = min(1.0, max(0.0, partial_fraction))
        if not context.partial_taken and current_r >= partial_r and partial_fraction > 0:
            reduce_qty = min(context.position.qty, context.position.qty * partial_fraction)
            return ExitManagementDecision(
                action=ExitActionType.PARTIAL_TAKE_PROFIT,
                reason_codes=["partial_take_profit"],
                reason_summary="partial take profit reached",
                reduce_qty=reduce_qty,
                decision_trace={**trace, "partial_r": partial_r, "partial_exit_fraction": partial_fraction},
            )

        trail_decision = self._mfe_trailing_decision(context, current_pnl=current_pnl, trace=trace)
        if trail_decision.action is not ExitActionType.HOLD:
            return trail_decision

        if current_r >= self.breakeven_at_r:
            target_stop = self._fee_buffered_breakeven(context)
            current_stop = context.current_stop_loss
            improves_stop = (
                current_stop is None
                or (side is Side.LONG and target_stop > current_stop)
                or (side is Side.SHORT and target_stop < current_stop)
            )
            if improves_stop:
                return ExitManagementDecision(
                    action=ExitActionType.MOVE_STOP,
                    reason_codes=["breakeven_stop_moved"],
                    reason_summary="move stop to fee buffered breakeven",
                    new_stop_loss=target_stop,
                    decision_trace={**trace, "breakeven_at_r": self.breakeven_at_r},
                )

        return ExitManagementDecision(
            action=ExitActionType.HOLD,
            reason_codes=["hold_no_exit_trigger"],
            reason_summary="no exit trigger",
            decision_trace=trace,
        )

    @staticmethod
    def _open_pnl(context: ExitManagementContext) -> float:
        return (context.latest_price - context.entry_price) * context.position.qty * context.position.side.sign

    def _mfe_trailing_decision(
        self,
        context: ExitManagementContext,
        *,
        current_pnl: float,
        trace: dict[str, Any],
    ) -> ExitManagementDecision:
        if context.mfe_usdt <= 0 or context.initial_risk_usdt is None or context.initial_risk_usdt <= 0:
            return ExitManagementDecision(action=ExitActionType.HOLD)
        mfe_r = context.mfe_usdt / context.initial_risk_usdt
        if mfe_r < self.mfe_trail_activation_r:
            return ExitManagementDecision(action=ExitActionType.HOLD)
        giveback_pct = max(0.0, (context.mfe_usdt - current_pnl) / max(context.mfe_usdt, 1e-12))
        risk_tightened = self._has_exit_risk(context.features)
        threshold = self.risk_signal_giveback_pct if risk_tightened else self.normal_giveback_pct
        if giveback_pct <= threshold:
            return ExitManagementDecision(action=ExitActionType.HOLD)
        reason_codes = ["mfe_trailing_exit"]
        if risk_tightened:
            reason_codes.append("wyckoff_exit_tightened")
        return ExitManagementDecision(
            action=ExitActionType.CLOSE_POSITION,
            reason_codes=reason_codes,
            reason_summary="mfe giveback exit",
            decision_trace={
                **trace,
                "mfe_r": mfe_r,
                "giveback_pct": giveback_pct,
                "giveback_threshold": threshold,
                "risk_tightened": risk_tightened,
            },
        )

    def _fee_buffered_breakeven(self, context: ExitManagementContext) -> float:
        buffer_pct = max(0.0, context.fee_bps + context.slippage_bps) / 10_000
        if context.position.side is Side.LONG:
            return context.entry_price * (1 + buffer_pct)
        return context.entry_price * (1 - buffer_pct)

    @staticmethod
    def _has_exit_risk(features: dict[str, Any]) -> bool:
        risk_tag = str(features.get("risk_pattern_tag") or "")
        if risk_tag in {"five_wave_late_risk", "false_breakout_risk"}:
            return True
        try:
            if float(features.get("wyckoff_exit_score", 0.0) or 0.0) >= 0.70:
                return True
            if float(features.get("wyckoff_risk_score", 0.0) or 0.0) >= 0.70:
                return True
        except (TypeError, ValueError):
            return False
        return False
