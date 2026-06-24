from __future__ import annotations

from typing import Callable
import uuid

from langlang_trader.config import PaperConfig
from langlang_trader.execution.routing import ExecutionRouter, RoutedOrderIntent
from langlang_trader.ledger import Ledger
from langlang_trader.models import AccountSnapshot, OrderIntent, OrderResult, Position, Side


class PaperExecutor:
    def __init__(
        self,
        *,
        ledger: Ledger,
        paper_config: PaperConfig,
        price_provider: Callable[[str], float],
        exchange: str = "okx",
        order_id_prefix: str | None = None,
        exchange_symbol_mapper: Callable[[str], str] | None = None,
        quote_fallback: Callable[[str], float] | None = None,
    ):
        self.exchange = exchange
        self.ledger = ledger.scoped(
            run_id=ledger.run_id,
            bot_id=ledger.bot_id,
            variant_id=ledger.variant_id,
            exchange=exchange,
        )
        self.paper_config = paper_config
        self.price_provider = price_provider
        self.quote_fallback = quote_fallback
        self.order_id_prefix = order_id_prefix or f"paper-{exchange}"
        self.exchange_symbol_mapper = exchange_symbol_mapper or (lambda symbol: symbol)
        self.cash_usdt = paper_config.initial_equity_usdt
        self.realized_pnl_usdt = 0.0
        self._restore_account_state()

    def get_account(self) -> AccountSnapshot:
        margin_used = 0.0
        unrealized = 0.0
        for position in self.get_positions():
            try:
                mark = self.price_provider(position.symbol)
            except Exception as exc:
                mark = self._recover_mark_price(position, exc)
            notional = position.qty * mark
            margin_used += notional / max(position.leverage, 1)
            unrealized += (mark - position.avg_price) * position.qty * position.side.sign
        return AccountSnapshot(
            equity_usdt=self.cash_usdt + unrealized,
            cash_usdt=self.cash_usdt,
            margin_used_usdt=margin_used,
            realized_pnl_usdt=self.realized_pnl_usdt,
        )

    def _recover_mark_price(self, position: Position, original_error: Exception) -> float:
        if self.quote_fallback is not None:
            try:
                mark = self.quote_fallback(position.symbol)
                self.ledger.record_risk_event(
                    "missing_mark_price_recovered",
                    {
                        "error": repr(original_error),
                        "fallback_price": mark,
                        "source": "quote_fallback",
                    },
                    symbol=position.symbol,
                )
                return mark
            except Exception as fallback_error:
                self.ledger.record_risk_event(
                    "missing_mark_price_quote_fallback_failed",
                    {
                        "error": repr(original_error),
                        "fallback_error": repr(fallback_error),
                    },
                    symbol=position.symbol,
                )
        mark = position.avg_price
        self.ledger.record_risk_event(
            "missing_mark_price_fallback",
            {
                "error": repr(original_error),
                "fallback_price": mark,
                "source": "position_avg_price",
            },
            symbol=position.symbol,
        )
        return mark

    def get_positions(self) -> list[Position]:
        return self.ledger.list_positions()

    def place_order(self, intent: OrderIntent, route: RoutedOrderIntent | None = None) -> OrderResult:
        order_type = intent.order_type.lower()
        route_payload = self._route_payload(intent, route)
        if order_type != "market":
            raw_payload = {**route_payload, "reason": "limit_not_supported_v01"}
            order_id = self.ledger.record_order(intent, status="rejected", raw_payload=raw_payload)
            return OrderResult(
                exchange_order_id=f"{self.order_id_prefix}-{order_id}",
                status="rejected",
                filled_qty=0.0,
                avg_price=None,
                fee=0.0,
                raw_payload=raw_payload,
            )

        base_price = self.price_provider(intent.symbol)
        fill_price = self._apply_slippage(base_price, intent.side)
        filled_qty = intent.qty
        notional = abs(fill_price * filled_qty)
        fee = notional * self.paper_config.fee_bps / 10_000
        exchange_order_id = f"{self.order_id_prefix}-{uuid.uuid4().hex[:12]}"
        order_id = self.ledger.record_order(
            intent,
            status="filled",
            exchange_order_id=exchange_order_id,
            raw_payload={**route_payload, "base_price": base_price, "slippage_bps": self.paper_config.slippage_bps},
        )
        self.ledger.record_fill(
            order_id=order_id,
            exchange_order_id=exchange_order_id,
            symbol=intent.symbol,
            side=intent.side,
            qty=filled_qty,
            price=fill_price,
            fee=fee,
            raw_payload={**route_payload, "reduce_only": intent.reduce_only},
            strategy_version=intent.strategy_version,
            regime=intent.regime,
            setup=intent.setup,
            exit_reason=intent.exit_reason,
            decision_trace=intent.decision_trace,
            historical_match_score=intent.historical_match_score,
        )
        realized_before = self.realized_pnl_usdt
        self.cash_usdt -= fee
        self._apply_fill_to_position(intent=intent, price=fill_price)
        self.cash_usdt += self.realized_pnl_usdt - realized_before
        self.ledger.record_equity_snapshot(
            self.get_account(),
            raw={**route_payload, "source": "paper_place_order", "exchange_order_id": exchange_order_id},
            strategy_version=intent.strategy_version,
            regime=intent.regime,
            setup=intent.setup,
            decision_trace=intent.decision_trace,
        )
        return OrderResult(
            exchange_order_id=exchange_order_id,
            status="filled",
            filled_qty=filled_qty,
            avg_price=fill_price,
            fee=fee,
            raw_payload={**route_payload, "order_id": order_id},
        )

    def cancel_order(self, order_id: str) -> OrderResult:
        return OrderResult(
            exchange_order_id=order_id,
            status="cancel_ignored",
            filled_qty=0.0,
            avg_price=None,
            fee=0.0,
            raw_payload={"paper": True, "paper_exchange": self.exchange, "reason": "market_orders_fill_immediately"},
        )

    def sync_fills(self) -> list[OrderResult]:
        return []

    def close_position(self, symbol: str, reason: str) -> OrderResult:
        position = self.ledger.get_position(symbol)
        if position is None:
            return OrderResult(
                exchange_order_id=f"{self.order_id_prefix}-no-position-{symbol}",
                status="no_position",
                filled_qty=0.0,
                avg_price=None,
                fee=0.0,
                raw_payload={"paper": True, "paper_exchange": self.exchange, "reason": reason},
            )
        close_side = Side.SHORT if position.side is Side.LONG else Side.LONG
        return self.place_order(
            OrderIntent(
                symbol=symbol,
                side=close_side,
                order_type="market",
                qty=position.qty,
                leverage=position.leverage,
                reduce_only=True,
                entry_reason=f"close:{reason}",
                stop_loss=None,
                max_slippage_bps=self.paper_config.slippage_bps,
                strategy_version=position.strategy_version,
                regime=position.regime,
                setup=position.setup,
                exit_reason=reason,
                decision_trace={"close_reason": reason},
            )
        )

    def _apply_slippage(self, base_price: float, side: Side) -> float:
        slippage = self.paper_config.slippage_bps / 10_000
        if side is Side.LONG:
            return base_price * (1 + slippage)
        return base_price * (1 - slippage)

    def _restore_account_state(self) -> None:
        cash, realized = self._account_state_from_fills()
        self.cash_usdt = cash
        self.realized_pnl_usdt = realized

    def _account_state_from_fills(self) -> tuple[float, float]:
        cash = self.paper_config.initial_equity_usdt
        realized = 0.0
        positions: dict[str, tuple[float, float]] = {}
        fills = self.ledger.list_rows(
            "fills",
            run_id=self.ledger.run_id,
            bot_id=self.ledger.bot_id,
            exchange=self.exchange,
        )
        for row in fills:
            side = Side.from_value(row["side"])
            qty = float(row["qty"])
            price = float(row["price"])
            fee = float(row["fee"])
            cash -= fee
            old_signed_qty, avg_price = positions.get(row["symbol"], (0.0, 0.0))
            fill_signed_qty = qty * side.sign
            new_signed_qty = old_signed_qty + fill_signed_qty

            if old_signed_qty == 0 or (old_signed_qty > 0) == (fill_signed_qty > 0):
                total_qty = abs(old_signed_qty) + abs(fill_signed_qty)
                avg = price if total_qty == 0 else (
                    (avg_price * abs(old_signed_qty) + price * abs(fill_signed_qty)) / total_qty
                )
                positions[row["symbol"]] = (new_signed_qty, avg)
                continue

            closed_qty = min(abs(old_signed_qty), abs(fill_signed_qty))
            old_sign = 1 if old_signed_qty > 0 else -1
            realized += (price - avg_price) * closed_qty * old_sign
            if abs(new_signed_qty) < 1e-12:
                positions.pop(row["symbol"], None)
            elif (old_signed_qty > 0) != (new_signed_qty > 0):
                positions[row["symbol"]] = (new_signed_qty, price)
            else:
                positions[row["symbol"]] = (new_signed_qty, avg_price)

        cash += realized
        return cash, realized

    def _apply_fill_to_position(self, *, intent: OrderIntent, price: float) -> None:
        existing = self.ledger.get_position(intent.symbol)
        old_signed_qty = 0.0 if existing is None else existing.qty * existing.side.sign
        fill_signed_qty = intent.qty * intent.side.sign
        new_signed_qty = old_signed_qty + fill_signed_qty

        if abs(new_signed_qty) < 1e-12:
            if existing is not None:
                self.realized_pnl_usdt += (price - existing.avg_price) * existing.qty * existing.side.sign
            self.ledger.delete_position(intent.symbol)
            return

        new_side = Side.LONG if new_signed_qty > 0 else Side.SHORT
        new_qty = abs(new_signed_qty)
        if existing is None or old_signed_qty == 0:
            avg_price = price
        elif abs(new_signed_qty) > abs(old_signed_qty):
            old_notional = existing.avg_price * abs(old_signed_qty)
            added_notional = price * abs(fill_signed_qty)
            avg_price = (old_notional + added_notional) / (abs(old_signed_qty) + abs(fill_signed_qty))
        elif (old_signed_qty > 0) != (new_signed_qty > 0):
            self.realized_pnl_usdt += (price - existing.avg_price) * abs(old_signed_qty) * existing.side.sign
            avg_price = price
        else:
            avg_price = existing.avg_price
            self.realized_pnl_usdt += (price - existing.avg_price) * abs(fill_signed_qty) * existing.side.sign

        self.ledger.upsert_position(
            Position(
                symbol=intent.symbol,
                side=new_side,
                qty=new_qty,
                avg_price=avg_price,
                leverage=intent.leverage,
                exchange=self.exchange,
                strategy_version=intent.strategy_version,
                regime=intent.regime,
                setup=intent.setup,
            )
        )

    def _route_payload(self, intent: OrderIntent, route: RoutedOrderIntent | None) -> dict[str, object]:
        return {
            "paper": True,
            "paper_exchange": self.exchange,
            "exchange_symbol": route.exchange_symbol if route is not None else self.exchange_symbol_mapper(intent.symbol),
            "route_reason": route.route_reason if route is not None else f"{self.exchange}_paper_direct",
        }


def okx_exchange_symbol(symbol: str) -> str:
    return symbol


def binance_exchange_symbol(symbol: str) -> str:
    if symbol.endswith("-USDT-SWAP"):
        return symbol.removesuffix("-USDT-SWAP").replace("-", "") + "USDT"
    return symbol.replace("-", "")


class OkxPaperExecutor(PaperExecutor):
    def __init__(
        self,
        *,
        ledger: Ledger,
        paper_config: PaperConfig,
        price_provider: Callable[[str], float],
        quote_fallback: Callable[[str], float] | None = None,
    ):
        super().__init__(
            ledger=ledger,
            paper_config=paper_config,
            price_provider=price_provider,
            exchange="okx",
            order_id_prefix="paper-okx",
            exchange_symbol_mapper=okx_exchange_symbol,
            quote_fallback=quote_fallback,
        )


class BinancePaperExecutor(PaperExecutor):
    def __init__(
        self,
        *,
        ledger: Ledger,
        paper_config: PaperConfig,
        price_provider: Callable[[str], float],
        quote_fallback: Callable[[str], float] | None = None,
    ):
        super().__init__(
            ledger=ledger,
            paper_config=paper_config,
            price_provider=price_provider,
            exchange="binance",
            order_id_prefix="paper-binance",
            exchange_symbol_mapper=binance_exchange_symbol,
            quote_fallback=quote_fallback,
        )


class MultiExchangePaperExecutor:
    def __init__(
        self,
        *,
        ledger: Ledger,
        paper_config: PaperConfig,
        price_provider: Callable[[str], float],
        router: ExecutionRouter,
        quote_fallback: Callable[[str], float] | None = None,
    ):
        self.ledger = ledger
        self.paper_config = paper_config
        self.price_provider = price_provider
        self.quote_fallback = quote_fallback
        self.router = router
        self.executors = {
            "okx": OkxPaperExecutor(
                ledger=ledger,
                paper_config=paper_config,
                price_provider=price_provider,
                quote_fallback=quote_fallback,
            ),
            "binance": BinancePaperExecutor(
                ledger=ledger,
                paper_config=paper_config,
                price_provider=price_provider,
                quote_fallback=quote_fallback,
            ),
        }

    def get_account(self) -> AccountSnapshot:
        accounts = [executor.get_account() for executor in self.executors.values()]
        duplicate_initial_equity = self.paper_config.initial_equity_usdt * max(0, len(accounts) - 1)
        return AccountSnapshot(
            equity_usdt=sum(account.equity_usdt for account in accounts) - duplicate_initial_equity,
            cash_usdt=sum(account.cash_usdt for account in accounts) - duplicate_initial_equity,
            margin_used_usdt=sum(account.margin_used_usdt for account in accounts),
            realized_pnl_usdt=sum(account.realized_pnl_usdt for account in accounts),
        )

    def get_positions(self) -> list[Position]:
        positions: list[Position] = []
        for executor in self.executors.values():
            positions.extend(executor.get_positions())
        return positions

    def place_order(self, intent: OrderIntent, route: RoutedOrderIntent | None = None) -> OrderResult:
        routed = route or self.router.route(intent)
        if routed is None:
            reason = self.router.rejection_reason(intent)
            self.ledger.record_risk_event(
                "execution_route_rejected",
                {"reason": reason, "symbol": intent.symbol},
                symbol=intent.symbol,
            )
            return OrderResult(
                exchange_order_id=f"paper-route-rejected-{intent.symbol}",
                status="rejected",
                filled_qty=0.0,
                avg_price=None,
                fee=0.0,
                raw_payload={"paper": True, "reason": reason, "symbol": intent.symbol},
            )
        return self.executors[routed.exchange].place_order(intent, route=routed)

    def cancel_order(self, order_id: str) -> OrderResult:
        if order_id.startswith("paper-binance-"):
            return self.executors["binance"].cancel_order(order_id)
        return self.executors["okx"].cancel_order(order_id)

    def sync_fills(self) -> list[OrderResult]:
        return []

    def close_position(self, symbol: str, reason: str) -> OrderResult:
        for exchange, executor in self.executors.items():
            if executor.ledger.get_position(symbol, exchange=exchange) is not None:
                return executor.close_position(symbol, reason)
        return OrderResult(
            exchange_order_id=f"paper-no-position-{symbol}",
            status="no_position",
            filled_qty=0.0,
            avg_price=None,
            fee=0.0,
            raw_payload={"paper": True, "reason": reason},
        )
