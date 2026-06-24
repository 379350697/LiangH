from __future__ import annotations

from enum import Enum
from typing import Any


class TradeLabel(str, Enum):
    BIG_WIN = "big_win"
    BIG_LOSS = "big_loss"
    RIGHT_TAIL = "right_tail"
    FAST_FAILURE = "fast_failure"
    CHASE_FAILURE = "chase_failure"
    ORDINARY = "ordinary"


class TradeLabeler:
    def label_trades(self, trades: list[dict[str, Any]]) -> dict[str, list[str]]:
        wins = [_float(trade.get("pnl_usdt")) for trade in trades if _float(trade.get("pnl_usdt")) > 0]
        losses = [abs(_float(trade.get("pnl_usdt"))) for trade in trades if _float(trade.get("pnl_usdt")) < 0]
        big_win_floor = _percentile(wins, 0.75)
        right_tail_floor = _percentile(wins, 0.90)
        big_loss_floor = _percentile(losses, 0.75)

        labels: dict[str, list[str]] = {}
        for idx, trade in enumerate(trades):
            trade_id = str(trade.get("trade_id") or idx)
            pnl = _float(trade.get("pnl_usdt"))
            return_rate = _float(trade.get("return_rate"))
            hold_minutes = _float(trade.get("hold_minutes"))
            row_labels: list[str] = []
            if pnl > 0 and pnl >= big_win_floor:
                row_labels.append(TradeLabel.BIG_WIN.value)
            if pnl > 0 and pnl >= right_tail_floor:
                row_labels.append(TradeLabel.RIGHT_TAIL.value)
            if pnl < 0 and abs(pnl) >= big_loss_floor:
                row_labels.append(TradeLabel.BIG_LOSS.value)
            if pnl < 0 and 0 <= hold_minutes <= 30:
                row_labels.append(TradeLabel.FAST_FAILURE.value)
            if pnl < 0 and abs(return_rate) >= 0.15 and 0 <= hold_minutes <= 60:
                row_labels.append(TradeLabel.CHASE_FAILURE.value)
            if not row_labels:
                row_labels.append(TradeLabel.ORDINARY.value)
            labels[trade_id] = row_labels
        return labels


def _percentile(values: list[float], q: float) -> float:
    clean = sorted(value for value in values if value == value)
    if not clean:
        return float("inf")
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(clean) - 1)
    weight = pos - lower
    return clean[lower] * (1 - weight) + clean[upper] * weight


def _float(value: Any) -> float:
    try:
        if value in {None, ""}:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0
