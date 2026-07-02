from __future__ import annotations

from langlang_trader.config import RiskConfig
from langlang_trader.models import AccountSnapshot, OrderIntent, Position, Signal
from langlang_trader.position_sizing import LangLangPositionSizer, PositionSizer


class RiskEngine:
    def __init__(
        self,
        config: RiskConfig,
        *,
        position_sizer: PositionSizer | None = None,
        initial_equity_usdt: float = 10_000.0,
    ):
        self.config = config
        if position_sizer is None and config.position_sizing_mode == "langlang_w_unit":
            position_sizer = LangLangPositionSizer(config, initial_equity_usdt=initial_equity_usdt)
        self.position_sizer = position_sizer
        self.last_rejection_reason: str | None = None
        self.last_rejection_trace: dict[str, object] = {}

    def intent_from_signal(
        self,
        *,
        signal: Signal,
        account: AccountSnapshot,
        latest_price: float,
        existing_position: Position | None = None,
        open_positions: list[Position] | None = None,
    ) -> OrderIntent | None:
        self.last_rejection_reason = None
        self.last_rejection_trace = {}
        if signal.strength < self.config.min_signal_strength:
            self._reject("signal_strength_below_min", strength=signal.strength)
            return None
        if existing_position is not None:
            self._reject("position_already_open")
            return None
        open_positions = open_positions or []
        if self.config.max_open_positions is not None and len(open_positions) >= self.config.max_open_positions:
            self._reject(
                "max_open_positions",
                open_count=len(open_positions),
                max_open_positions=self.config.max_open_positions,
            )
            return None
        if self.config.max_open_symbols is not None:
            open_symbols = {position.symbol for position in open_positions if abs(position.qty) > 0}
            if signal.symbol not in open_symbols and len(open_symbols) >= self.config.max_open_symbols:
                self._reject(
                    "max_open_symbols",
                    open_symbol_count=len(open_symbols),
                    max_open_symbols=self.config.max_open_symbols,
                )
                return None
        if self.config.max_total_position_usdt is not None:
            current_notional = sum(abs(position.qty * position.avg_price) for position in open_positions)
            if current_notional >= self.config.max_total_position_usdt:
                self._reject(
                    "max_total_position_usdt",
                    current_notional=current_notional,
                    max_total_position_usdt=self.config.max_total_position_usdt,
                )
                return None
        if self.config.max_daily_loss_usdt is not None and account.realized_pnl_usdt <= -abs(
            self.config.max_daily_loss_usdt
        ):
            self._reject(
                "max_daily_loss_usdt",
                realized_pnl_usdt=account.realized_pnl_usdt,
                max_daily_loss_usdt=self.config.max_daily_loss_usdt,
            )
            return None
        leverage = self.config.default_leverage
        decision_trace = {
            **_decision_trace_from_signal_features(signal),
            **(getattr(signal, "decision_trace", {}) or {}),
        }
        if self.position_sizer is not None and self.config.position_sizing_mode == "langlang_w_unit":
            size_decision = self.position_sizer.size(
                signal=signal,
                account=account,
                open_positions=open_positions,
                latest_price=latest_price,
            )
            if size_decision is None:
                self._reject("position_sizer_rejected")
                return None
            notional = size_decision.notional_usdt
            leverage = size_decision.leverage
            decision_trace = {**decision_trace, **size_decision.decision_trace}
        else:
            available_notional = max(account.equity_usdt, 0.0) * self.config.default_leverage
            notional = min(self.config.max_position_usdt, available_notional)
            if self.config.max_total_position_usdt is not None:
                remaining_notional = self.config.max_total_position_usdt - sum(
                    abs(position.qty * position.avg_price) for position in open_positions
                )
                notional = min(notional, remaining_notional)
            multiplier = _position_size_multiplier(signal)
            notional *= multiplier
            decision_trace = {
                **decision_trace,
                "position_size_multiplier": multiplier,
                "position_notional_usdt": notional,
            }
        if latest_price <= 0 or notional <= 0:
            self._reject("invalid_price_or_notional", latest_price=latest_price, notional=notional)
            return None
        if not self._valid_stop_side(signal=signal, latest_price=latest_price):
            self._reject(
                "invalid_stop_loss_side",
                side=signal.side.value,
                latest_price=latest_price,
                stop_loss=signal.invalidation_price,
            )
            return None
        qty = notional / latest_price
        return OrderIntent(
            symbol=signal.symbol,
            side=signal.side,
            order_type="market",
            qty=qty,
            leverage=leverage,
            reduce_only=False,
            entry_reason=",".join(signal.reason_codes),
            stop_loss=signal.invalidation_price,
            max_slippage_bps=self.config.max_slippage_bps,
            strategy_version=getattr(signal, "strategy_version", None),
            regime=_context_value(getattr(signal, "regime", None)),
            setup=_context_value(getattr(signal, "setup", None)),
            decision_trace=decision_trace,
            historical_match_score=getattr(signal, "historical_match_score", None),
        )

    def _reject(self, reason: str, **trace: object) -> None:
        self.last_rejection_reason = reason
        self.last_rejection_trace = dict(trace)

    @staticmethod
    def _valid_stop_side(*, signal: Signal, latest_price: float) -> bool:
        stop = float(signal.invalidation_price)
        if stop <= 0:
            return False
        if signal.side.value == "long":
            return stop < latest_price
        return stop > latest_price


def _context_value(value):
    return value.value if hasattr(value, "value") else value


def _position_size_multiplier(signal: Signal) -> float:
    features = getattr(signal, "features", {}) or {}
    raw = features.get("position_size_multiplier", 1.0)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 1.0
    return min(1.0, max(0.0, value))


def _decision_trace_from_signal_features(signal: Signal) -> dict[str, object]:
    features = getattr(signal, "features", {}) or {}
    trace: dict[str, object] = {}
    for key in (
        "strategy_tree_variant_id",
        "strategy_tree_parent_id",
        "strategy_tree_path",
        "time_stop_bars",
        "take_profit_r",
        "strategy_kind",
        "position_size_multiplier",
    ):
        if key in features:
            trace[key] = features[key]
    return trace
