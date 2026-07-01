from __future__ import annotations

from langlang_trader.config import AppConfig
from langlang_trader.execution.base import Executor
from langlang_trader.execution.live_okx import OkxLiveExecutor
from langlang_trader.execution.paper import PaperExecutor
from langlang_trader.features import DailyFeatureBuilder, MultiTimeframeFeatureBuilder
from langlang_trader.ledger import Ledger
from langlang_trader.market_data import MarketData
from langlang_trader.models import Position
from langlang_trader.risk import RiskEngine
from langlang_trader.strategy import RulesLangLangV1Strategy, RulesV01Strategy, strategy_from_version


class TradingRunner:
    def __init__(
        self,
        *,
        config: AppConfig,
        market_data: MarketData,
        ledger: Ledger,
        executor: Executor | None = None,
        strategy: RulesV01Strategy | RulesLangLangV1Strategy | None = None,
        risk_engine: RiskEngine | None = None,
    ):
        self.config = config
        self.market_data = market_data
        self.ledger = ledger
        self.strategy = strategy or strategy_from_version(config.strategy_version)
        self.feature_builder = DailyFeatureBuilder()
        self.multi_feature_builder = MultiTimeframeFeatureBuilder()
        self.risk_engine = risk_engine or RiskEngine(config.risk)
        self.executor = executor or self._build_executor()

    def run_once(self) -> dict[str, int]:
        cycle = {
            "symbols": 0,
            "signals": 0,
            "intents": 0,
            "orders": 0,
            "fills": 0,
            "stop_exits": 0,
            "risk_rejections": 0,
            "errors": 0,
        }
        for symbol in self.config.market_data.symbols:
            cycle["symbols"] += 1
            try:
                if self._close_if_stop_loss_hit(symbol, cycle):
                    continue
                if self.config.strategy_version == RulesLangLangV1Strategy.version:
                    candles_by_bar = {
                        bar: self.market_data.get_candles(
                            symbol,
                            bar=bar,
                            limit=self.config.market_data.candle_limit,
                        )
                        for bar in self.config.market_data.bars
                    }
                    feature_snapshot = self.multi_feature_builder.build(symbol, candles_by_bar)
                else:
                    candles = self.market_data.get_candles(symbol, bar="1D", limit=self.config.market_data.candle_limit)
                    feature_snapshot = self.feature_builder.build(symbol, candles)
                if feature_snapshot is None:
                    continue
                signal = self.strategy.generate_from_features(feature_snapshot)
                if signal is None:
                    continue
                signal_id = self.ledger.record_signal(signal, strategy_version=self.config.strategy_version)
                cycle["signals"] += 1

                latest_price = self.market_data.latest_price(symbol)
                intent = self.risk_engine.intent_from_signal(
                    signal=signal,
                    account=self.executor.get_account(),
                    latest_price=latest_price,
                    existing_position=_position_for_symbol(self.executor.get_positions(), symbol),
                )
                if intent is None:
                    self.ledger.record_risk_event(
                        "intent_rejected",
                        {"signal_id": signal_id, "strength": signal.strength},
                        symbol=symbol,
                    )
                    cycle["risk_rejections"] += 1
                    continue

                self.ledger.record_order_intent(intent, signal_id=signal_id)
                cycle["intents"] += 1
                result = self.executor.place_order(intent)
                if result.status in {"filled", "accepted", "submitted"}:
                    cycle["orders"] += 1
                if result.filled_qty > 0:
                    cycle["fills"] += 1
                if result.status == "rejected":
                    self.ledger.record_risk_event("executor_rejected", result.raw_payload, symbol=symbol)
            except Exception as exc:  # pragma: no cover - operational safety path
                self.ledger.record_risk_event("runner_error", {"error": repr(exc)}, symbol=symbol)
                cycle["errors"] += 1
        return cycle

    def _close_if_stop_loss_hit(self, symbol: str, cycle: dict[str, int]) -> bool:
        position = _position_for_symbol(self.executor.get_positions(), symbol)
        if position is None:
            return False
        stop_loss = self.ledger.latest_stop_loss(symbol)
        if stop_loss is None:
            return False
        latest_price = self.market_data.latest_price(symbol)
        triggered = (
            latest_price <= stop_loss
            if position.side.value == "long"
            else latest_price >= stop_loss
        )
        if not triggered:
            return False
        result = self.executor.close_position(symbol, reason=f"stop_loss:{stop_loss}")
        self.ledger.record_risk_event(
            "stop_loss_exit",
            {
                "latest_price": latest_price,
                "stop_loss": stop_loss,
                "status": result.status,
                "exchange_order_id": result.exchange_order_id,
            },
            symbol=symbol,
        )
        if result.status in {"filled", "accepted", "submitted"}:
            cycle["orders"] += 1
        if result.filled_qty > 0:
            cycle["fills"] += 1
        cycle["stop_exits"] += 1
        return True

    def _build_executor(self) -> Executor:
        if self.config.execution.executor in {"paper_okx", "paper_binance"}:
            return PaperExecutor(
                ledger=self.ledger,
                paper_config=self.config.paper,
                price_provider=self.market_data.latest_price,
                exchange=self.config.execution.exchange,
            )
        if self.config.execution.executor == "live_okx":
            return OkxLiveExecutor(config=self.config.execution)
        raise ValueError(f"unsupported executor: {self.config.execution.executor}")


def _position_for_symbol(positions: list[Position], symbol: str) -> Position | None:
    for position in positions:
        if position.symbol == symbol:
            return position
    return None
