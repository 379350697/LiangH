from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

from .binance_execution_ws import BinanceWsApiRequestBuilder, BinanceWsExecutionGateway
from .binance_ws import BinanceUsdmWebSocketMarketData
from .config import MarketMakerConfig
from .config import load_market_maker_config
from .exchange_interfaces import RateLimitBudget
from .hybrid_runtime import HybridMarketMakerRuntime
from .ledger import MarketMakerLedger
from .live_executor import assert_live_orders_enabled
from .models import BookTick, TopBookTick, TradeTick
from .paper_executor import MarketMakerPaperExecutor
from .runner import MarketMakerRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LiangH market maker paper scaffold")
    parser.add_argument("--config", required=True, help="Path to market maker JSON config")
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument("--synthetic", action="store_true", help="Run an offline synthetic replay smoke")
    parser.add_argument("--market-data-smoke", action="store_true", help="Run only market-data capture and ledger latency")
    parser.add_argument("--hybrid-runtime", action="store_true", help="Route synthetic/live events through hybrid runtime")
    return parser


async def run_from_args(args: argparse.Namespace) -> None:
    config = load_market_maker_config(args.config)
    if config.mode == "live":
        assert_live_orders_enabled(config)
    ledger = MarketMakerLedger(config.ledger_path, context=config.ledger_context)
    executor = MarketMakerPaperExecutor(config=config, ledger=ledger)
    runner = MarketMakerRunner(config=config, ledger=ledger, executor=executor)
    if args.synthetic:
        if args.hybrid_runtime:
            await _run_hybrid_synthetic_replay(config=config, ledger=ledger, duration_seconds=args.duration_seconds)
            return
        await runner.run_synthetic_replay(duration_seconds=args.duration_seconds)
        return
    if args.market_data_smoke:
        await _run_market_data_smoke(config, ledger, runner, duration_seconds=args.duration_seconds)
        return

    if args.hybrid_runtime or config.execution.primary_gateway != "paper":
        await _run_hybrid_market_data(config=config, ledger=ledger, duration_seconds=args.duration_seconds)
        return

    market_data = BinanceUsdmWebSocketMarketData(config.symbol)
    deadline_ns = time.monotonic_ns() + int(args.duration_seconds * 1_000_000_000)
    last_loop_ns = time.monotonic_ns()
    async for event in market_data.stream():
        now_ns = time.monotonic_ns()
        loop_lag_ms = max(0.0, (now_ns - last_loop_ns) / 1_000_000.0)
        last_loop_ns = now_ns
        if isinstance(event, TopBookTick):
            runner.record_market_data_latency(event, now_ns=now_ns)
        elif isinstance(event, BookTick):
            runner.on_book(event, now_ns=now_ns, loop_lag_ms=loop_lag_ms)
        else:
            runner.on_trade(event)
        if now_ns >= deadline_ns:
            executor.cancel_all(reason="duration_complete")
            break


async def _run_hybrid_synthetic_replay(config: MarketMakerConfig, ledger: MarketMakerLedger, duration_seconds: float) -> None:
    runtime = _build_hybrid_runtime(config=config, ledger=ledger)
    end_ns = time.monotonic_ns() + int(duration_seconds * 1_000_000_000)
    update_id = 1
    while time.monotonic_ns() < end_ns:
        now_ns = time.monotonic_ns()
        runtime.on_market_event(
            BookTick(
                symbol=config.symbol,
                event_time_ms=int(time.time() * 1000),
                receive_time_ns=now_ns,
                best_bid=99.99,
                best_bid_qty=5.0,
                best_ask=100.01,
                best_ask_qty=5.0,
                update_id=update_id,
                source="l2_depth",
                book_status="hot",
            ),
            now_ns=now_ns,
        )
        update_id += 1
        await asyncio.sleep(config.strategy.quote_interval_ms / 1000.0)


async def _run_hybrid_market_data(config: MarketMakerConfig, ledger: MarketMakerLedger, duration_seconds: float) -> None:
    runtime = _build_hybrid_runtime(config=config, ledger=ledger)
    market_data = BinanceUsdmWebSocketMarketData(config.symbol)
    deadline_ns = time.monotonic_ns() + int(duration_seconds * 1_000_000_000)
    last_loop_ns = time.monotonic_ns()
    async for event in market_data.stream():
        now_ns = time.monotonic_ns()
        loop_lag_ms = max(0.0, (now_ns - last_loop_ns) / 1_000_000.0)
        last_loop_ns = now_ns
        runtime.on_market_event(event, now_ns=now_ns, loop_lag_ms=loop_lag_ms)
        if now_ns >= deadline_ns:
            runtime.execution_gateway.cancel_all(reason="duration_complete", now_ns=now_ns)
            break


def _build_hybrid_runtime(config: MarketMakerConfig, ledger: MarketMakerLedger) -> HybridMarketMakerRuntime:
    budget = RateLimitBudget(
        max_order_ops_per_minute=config.limits.max_order_ops_per_minute,
        max_order_ops_per_10s=config.limits.max_order_ops_per_10s,
    )
    if config.execution.primary_gateway != "binance_ws_api":
        return HybridMarketMakerRuntime(config=config, ledger=ledger, rate_limit_budget=budget)

    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")
    if config.mode == "live" and (not api_key or not api_secret):
        raise RuntimeError("binance_ws_api live mode requires BINANCE_API_KEY and BINANCE_API_SECRET")
    builder = BinanceWsApiRequestBuilder(api_key=api_key or "dry-run-key", api_secret=api_secret or "dry-run-secret")
    gateway = BinanceWsExecutionGateway(
        config=config,
        ledger=ledger,
        builder=builder,
        rate_limit_budget=budget,
        dry_run=config.mode != "live",
    )
    return HybridMarketMakerRuntime(
        config=config,
        ledger=ledger,
        execution_gateway=gateway,
        rate_limit_budget=budget,
    )


async def _run_market_data_smoke(
    config: MarketMakerConfig,
    ledger: MarketMakerLedger,
    runner: MarketMakerRunner,
    duration_seconds: float,
) -> None:
    market_data = BinanceUsdmWebSocketMarketData(config.symbol)
    deadline_ns = time.monotonic_ns() + int(duration_seconds * 1_000_000_000)
    start_ns = time.monotonic_ns()
    initial_counts = _ledger_counts(ledger)
    stats = {
        "book_events": 0,
        "top_book_events": 0,
        "trade_events": 0,
        "l2_depth_events": 0,
        "hot_l2_depth_events": 0,
        "sequence_gap_events": 0,
        "connection_errors": 0,
        "last_error": "",
        "max_nonhot_duration_ms": 0.0,
        "last_book_status": "",
        "resync_count": 0,
        "sequence_gap_count": 0,
    }
    nonhot_since_ns: int | None = None
    book_event_ages_ms: list[float] = []

    try:
        async for event in market_data.stream():
            now_ns = time.monotonic_ns()
            if isinstance(event, TopBookTick):
                stats["top_book_events"] += 1
                runner.record_market_data_latency(event, now_ns=now_ns)
            elif isinstance(event, BookTick):
                stats["book_events"] += 1
                stats["l2_depth_events"] += 1
                stats["last_book_status"] = event.book_status
                stats["resync_count"] = max(int(stats["resync_count"]), event.resync_count)
                stats["sequence_gap_count"] = max(int(stats["sequence_gap_count"]), event.sequence_gap_count)
                if event.event_time_ms > 0:
                    book_event_ages_ms.append(max(0.0, time.time() * 1000.0 - event.event_time_ms))
                if event.sequence_gap:
                    stats["sequence_gap_events"] += 1
                if event.book_status == "hot" and not event.stale and not event.sequence_gap:
                    stats["hot_l2_depth_events"] += 1
                    if nonhot_since_ns is not None:
                        stats["max_nonhot_duration_ms"] = max(
                            float(stats["max_nonhot_duration_ms"]),
                            (now_ns - nonhot_since_ns) / 1_000_000.0,
                        )
                    nonhot_since_ns = None
                else:
                    if nonhot_since_ns is None:
                        nonhot_since_ns = now_ns
                    stats["max_nonhot_duration_ms"] = max(
                        float(stats["max_nonhot_duration_ms"]),
                        (now_ns - nonhot_since_ns) / 1_000_000.0,
                    )
                runner.record_market_data_latency(event, now_ns=now_ns)
            elif isinstance(event, TradeTick):
                stats["trade_events"] += 1
                runner.record_trade_latency(event, now_ns=now_ns)

            if now_ns >= deadline_ns:
                break
    except Exception as exc:
        stats["connection_errors"] += 1
        stats["last_error"] = repr(exc)

    elapsed_seconds = max(0.0, (time.monotonic_ns() - start_ns) / 1_000_000_000.0)
    if nonhot_since_ns is not None:
        stats["max_nonhot_duration_ms"] = max(
            float(stats["max_nonhot_duration_ms"]),
            (time.monotonic_ns() - nonhot_since_ns) / 1_000_000.0,
        )
    hot_coverage = (
        float(stats["hot_l2_depth_events"]) / float(stats["l2_depth_events"])
        if stats["l2_depth_events"]
        else 0.0
    )
    summary = {
        "mode": "market_data_smoke",
        "symbol": config.symbol,
        "venue": config.venue,
        "duration_seconds": elapsed_seconds,
        "connection_errors": stats["connection_errors"],
        "snapshot_errors": market_data.snapshot_error_count,
        "last_error": stats["last_error"] or market_data.last_snapshot_error,
        "book_events": stats["book_events"],
        "top_book_events": stats["top_book_events"],
        "trade_events": stats["trade_events"],
        "hot_coverage": hot_coverage,
        "sequence_gap_events": stats["sequence_gap_events"],
        "sequence_gap_count": stats["sequence_gap_count"],
        "resync_count": stats["resync_count"],
        "unrecovered_sequence_gap": int(stats["last_book_status"] != "hot" and int(stats["sequence_gap_count"]) > 0),
        "max_nonhot_duration_ms": stats["max_nonhot_duration_ms"],
        "book_event_age_p95_ms": _percentile(book_event_ages_ms, 95),
        "book_event_age_p99_ms": _percentile(book_event_ages_ms, 99),
        "quote_rows": len(ledger.list_rows("mm_quotes")) - initial_counts["mm_quotes"],
        "order_rows": len(ledger.list_rows("mm_orders")) - initial_counts["mm_orders"],
        "fill_rows": len(ledger.list_rows("mm_fills")) - initial_counts["mm_fills"],
        "latency_rows": len(ledger.list_rows("mm_latency_events")) - initial_counts["mm_latency_events"],
        "risk_rows": len(ledger.list_rows("mm_risk_events")) - initial_counts["mm_risk_events"],
    }
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((percentile / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def _ledger_counts(ledger: MarketMakerLedger) -> dict[str, int]:
    return {
        table: len(ledger.list_rows(table))
        for table in (
            "mm_quotes",
            "mm_orders",
            "mm_fills",
            "mm_latency_events",
            "mm_risk_events",
        )
    }


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(run_from_args(args))


if __name__ == "__main__":
    main()
