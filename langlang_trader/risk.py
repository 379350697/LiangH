from __future__ import annotations

from langlang_trader.config import RiskConfig
from langlang_trader.models import AccountSnapshot, OrderIntent, Position, Signal


class RiskEngine:
    def __init__(self, config: RiskConfig):
        self.config = config

    def intent_from_signal(
        self,
        *,
        signal: Signal,
        account: AccountSnapshot,
        latest_price: float,
        existing_position: Position | None = None,
        open_positions: list[Position] | None = None,
    ) -> OrderIntent | None:
        if signal.strength < self.config.min_signal_strength:
            return None
        if existing_position is not None:
            return None
        open_positions = open_positions or []
        if self.config.max_open_positions is not None and len(open_positions) >= self.config.max_open_positions:
            return None
        if self.config.max_total_position_usdt is not None:
            current_notional = sum(abs(position.qty * position.avg_price) for position in open_positions)
            if current_notional >= self.config.max_total_position_usdt:
                return None
        if account.realized_pnl_usdt <= -abs(self.config.max_daily_loss_usdt):
            return None
        available_notional = max(account.equity_usdt, 0.0) * self.config.default_leverage
        notional = min(self.config.max_position_usdt, available_notional)
        if self.config.max_total_position_usdt is not None:
            remaining_notional = self.config.max_total_position_usdt - sum(
                abs(position.qty * position.avg_price) for position in open_positions
            )
            notional = min(notional, remaining_notional)
        if latest_price <= 0 or notional <= 0:
            return None
        qty = notional / latest_price
        return OrderIntent(
            symbol=signal.symbol,
            side=signal.side,
            order_type="market",
            qty=qty,
            leverage=self.config.default_leverage,
            reduce_only=False,
            entry_reason=",".join(signal.reason_codes),
            stop_loss=signal.invalidation_price,
            max_slippage_bps=self.config.max_slippage_bps,
            strategy_version=getattr(signal, "strategy_version", None),
            regime=_context_value(getattr(signal, "regime", None)),
            setup=_context_value(getattr(signal, "setup", None)),
            decision_trace=getattr(signal, "decision_trace", {}) or {},
            historical_match_score=getattr(signal, "historical_match_score", None),
        )


def _context_value(value):
    return value.value if hasattr(value, "value") else value
