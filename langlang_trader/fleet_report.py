from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import sqlite3
from pathlib import Path
from typing import Any


_HFT_COST_FIELD_NAMES = (
    "take_profit_bps",
    "round_trip_fee_bps",
    "min_net_take_profit_bps",
    "take_profit_cost_floor_bps",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize a LangLang paper fleet ledger")
    parser.add_argument("--ledger")
    parser.add_argument("--run-id")
    parser.add_argument("--manifest")
    parser.add_argument("--out-dir", default="output/fleet/langlang_strategy_forest/reports")
    parser.add_argument("--initial-equity-usdt", type=float, default=10_000.0)
    args = parser.parse_args(argv)
    if args.manifest:
        result = write_scalping_batch_report(
            manifest_path=args.manifest,
            out_dir=args.out_dir,
            initial_equity_usdt=args.initial_equity_usdt,
        )
    else:
        if not args.ledger or not args.run_id:
            parser.error("--ledger and --run-id are required unless --manifest is provided")
        result = write_fleet_report(
            ledger_path=args.ledger,
            run_id=args.run_id,
            out_dir=args.out_dir,
            initial_equity_usdt=args.initial_equity_usdt,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def write_fleet_report(
    *,
    ledger_path: str | Path,
    run_id: str,
    out_dir: str | Path,
    initial_equity_usdt: float = 10_000.0,
) -> dict[str, str]:
    summary = summarize_fleet_ledger(ledger_path, run_id=run_id, initial_equity_usdt=initial_equity_usdt)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    markdown_path = out / f"fleet_report_{run_id}_{stamp}.md"
    json_path = out / f"fleet_report_{run_id}_{stamp}.json"
    latest_markdown_path = out / "latest.md"
    latest_json_path = out / "latest.json"
    markdown = _render_markdown(summary)
    payload = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    markdown_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(payload, encoding="utf-8")
    latest_markdown_path.write_text(markdown, encoding="utf-8")
    latest_json_path.write_text(payload, encoding="utf-8")
    return {
        "markdown_path": str(markdown_path),
        "json_path": str(json_path),
        "latest_markdown_path": str(latest_markdown_path),
        "latest_json_path": str(latest_json_path),
    }


def write_scalping_batch_report(
    *,
    manifest_path: str | Path,
    out_dir: str | Path,
    initial_equity_usdt: float = 10_000.0,
) -> dict[str, str]:
    summary = summarize_scalping_batch_manifest(manifest_path, initial_equity_usdt=initial_equity_usdt)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_id = str(summary["run_id"])
    markdown_path = out / f"batch_report_{batch_id}_{stamp}.md"
    json_path = out / f"batch_report_{batch_id}_{stamp}.json"
    latest_markdown_path = out / "batch_latest.md"
    latest_json_path = out / "batch_latest.json"
    markdown = _render_markdown(summary)
    payload = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    markdown_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(payload, encoding="utf-8")
    latest_markdown_path.write_text(markdown, encoding="utf-8")
    latest_json_path.write_text(payload, encoding="utf-8")
    return {
        "markdown_path": str(markdown_path),
        "json_path": str(json_path),
        "latest_markdown_path": str(latest_markdown_path),
        "latest_json_path": str(latest_json_path),
    }


def summarize_scalping_batch_manifest(
    manifest_path: str | Path,
    *,
    initial_equity_usdt: float = 10_000.0,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    base_dir = manifest_file.parent
    fleet_config_path = _resolve_manifest_path(base_dir, manifest["fleet_config"])
    fleet_config = json.loads(fleet_config_path.read_text(encoding="utf-8"))
    signal_summary = summarize_fleet_ledger(
        _resolve_manifest_path(base_dir, fleet_config["ledger_path"]),
        run_id=str(fleet_config["run_id"]),
        initial_equity_usdt=initial_equity_usdt,
    )
    signal_bots = _include_configured_signal_bots(
        signal_summary["bots"],
        fleet_config=fleet_config,
        initial_equity_usdt=initial_equity_usdt,
    )
    signal_summary = {
        **signal_summary,
        "bots": signal_bots,
        "totals": _totals(signal_bots),
    }
    maker_rows = [
        _maker_bot_row(
            _resolve_manifest_path(base_dir, config_path),
            initial_equity_usdt=initial_equity_usdt,
        )
        for config_path in manifest.get("market_maker_configs", [])
    ]
    bots = list(signal_bots) + maker_rows
    return {
        "run_id": str(manifest.get("batch_id") or fleet_config["run_id"]),
        "ledger_path": str(manifest_file),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bots": bots,
        "risk_rejections": signal_summary.get("risk_rejections", {}),
        "trade_journal": signal_summary.get("trade_journal", _empty_trade_journal()),
        "totals": _totals(bots),
        "signal_summary": signal_summary,
        "maker_bot_count": len(maker_rows),
    }


def _resolve_manifest_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return path


def _include_configured_signal_bots(
    rows: list[dict[str, Any]],
    *,
    fleet_config: dict[str, Any],
    initial_equity_usdt: float,
) -> list[dict[str, Any]]:
    enriched = list(rows)
    seen = {(str(row["bot_id"]), str(row["variant_id"])) for row in rows}
    for bot in _configured_signal_bots(fleet_config):
        bot_id = str(bot["bot_id"])
        variant = dict(bot.get("variant", {}))
        variant_id = str(variant.get("variant_id") or bot.get("variant_id") or bot_id)
        key = (bot_id, variant_id)
        if key in seen:
            continue
        enriched.append(
            _zero_signal_bot_row(
                bot_id,
                variant_id,
                initial_equity_usdt=initial_equity_usdt,
                variant=variant,
            )
        )
        seen.add(key)
    return enriched


def _configured_signal_bots(fleet_config: dict[str, Any]) -> list[dict[str, Any]]:
    bots = fleet_config.get("bots", [])
    if bots:
        return [dict(row) for row in bots]
    matrix = fleet_config.get("bot_matrix", {})
    templates = matrix.get("strategies", []) if isinstance(matrix, dict) else []
    symbols = fleet_config.get("symbol_matrix", [])
    rows: list[dict[str, Any]] = []
    for symbol_row in symbols:
        slug = str(symbol_row["slug"])
        symbol = str(symbol_row["symbol"])
        exchange_symbol = str(symbol_row["exchange_symbol"])
        lead_exchange_symbol = str(symbol_row.get("lead_exchange_symbol", "BTCUSDT"))
        for template in templates:
            strategy_key = str(template["strategy_key"])
            variant_id = f"{template['variant_prefix']}_{slug}_v1"
            variant = {
                "variant_id": variant_id,
                "symbol": symbol,
                "exchange_symbol": exchange_symbol,
                "strategy_kind": strategy_key.removeprefix("hft_"),
                "strategy_tree_parent_id": str(template["strategy_tree_parent_id"]),
                "strategy_tree_variant_id": variant_id,
                "strategy_tree_path": [
                    "scalping",
                    "batch7_hft_scalp",
                    strategy_key,
                    variant_id,
                ],
                **template.get("parameters", {}),
            }
            if strategy_key == "hft_lead_lag_fair_value":
                variant["lead_exchange_symbol"] = lead_exchange_symbol
            rows.append(
                {
                    "bot_id": f"{template['bot_prefix']}_{slug}_paper",
                    "strategy_version": str(template["strategy_version"]),
                    "variant": variant,
                }
            )
    return rows


def _zero_signal_bot_row(
    bot_id: str,
    variant_id: str,
    *,
    initial_equity_usdt: float,
    variant: dict[str, Any] | None = None,
) -> dict[str, Any]:
    funding_basis = _is_funding_basis_bot(bot_id, variant_id)
    return {
        "bot_id": bot_id,
        "variant_id": variant_id,
        "trading_role": "shadow_only" if funding_basis else "paper_trading",
        "paper_trading_bot": not funding_basis,
        "zero_open_reason": _zero_open_reason(
            signal_count=0,
            opened_orders=0,
            shadow_pair_events=0,
            funding_basis=funding_basis,
        ),
        "signals": 0,
        "opened_orders": 0,
        "closed_orders": 0,
        "fills": 0,
        "shadow_pair_events": 0,
        "hft_event_count": 0,
        "average_hold_ms": None,
        "spread_capture_bps": None,
        "adverse_selection_bps": None,
        "fill_ratio": 0.0,
        "stale_or_sequence_gap_guard_count": 0,
        **_hft_cost_fields_from_payload(variant or {}),
        "partial_take_profit_count": 0,
        "take_profit_count": 0,
        "runner_take_profit_count": 0,
        "mfe_trailing_exit_count": 0,
        "stop_loss_count": 0,
        "time_or_guard_exit_count": 0,
        "other_exit_count": 0,
        "fees_paid": 0.0,
        "equity_usdt": round(initial_equity_usdt, 6),
        "equity_pnl_net": 0.0,
        "snapshot_sharpe": None,
        "realized_price_pnl": 0.0,
        "margin_used": 0.0,
        "positions": 0,
        "symbols": 0,
        "long_positions": 0,
        "short_positions": 0,
        "notional": 0.0,
        "latest_snapshot": "",
        "latest_snapshot_source": "",
    }


def summarize_fleet_ledger(
    ledger_path: str | Path,
    *,
    run_id: str,
    initial_equity_usdt: float = 10_000.0,
) -> dict[str, Any]:
    path = Path(ledger_path)
    generated_at = datetime.now(timezone.utc).isoformat()
    if not path.exists():
        return {
            "run_id": run_id,
            "ledger_path": str(path),
            "generated_at": generated_at,
            "bots": [],
            "risk_rejections": {},
            "trade_journal": _empty_trade_journal(),
            "totals": _totals([]),
        }
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        bots = _bot_rows(conn, run_id, initial_equity_usdt)
        risk_rejections = _risk_rejections(conn, run_id)
        trade_journal = _trade_journal(conn, run_id)
    return {
        "run_id": run_id,
        "ledger_path": str(path),
        "generated_at": generated_at,
        "bots": bots,
        "risk_rejections": dict(risk_rejections),
        "trade_journal": trade_journal,
        "totals": _totals(bots),
    }


def _bot_rows(conn: sqlite3.Connection, run_id: str, initial_equity_usdt: float) -> list[dict[str, Any]]:
    latest_multi = _latest_multi_equity(conn, run_id)
    latest_exchange = _latest_exchange_equity(conn, run_id, initial_equity_usdt)
    opened = _count_by_bot(conn, run_id, "orders", "reduce_only = 0")
    closed = _count_by_bot(conn, run_id, "orders", "reduce_only = 1")
    fills = _fills_by_bot(conn, run_id)
    signals = _count_by_bot(conn, run_id, "signals")
    positions = _positions_by_bot(conn, run_id)
    shadow_pair_events = _shadow_pair_events_by_bot(conn, run_id)
    partial_exits = _partial_exit_events_by_bot(conn, run_id)
    final_exits = _final_exit_reasons_by_bot(conn, run_id)
    snapshot_sharpe = _snapshot_sharpe_by_bot(conn, run_id)
    hft_cost_fields = _hft_cost_fields_by_bot(conn, run_id)
    bot_keys = (
        set(latest_multi)
        | set(latest_exchange)
        | set(opened)
        | set(closed)
        | set(fills)
        | set(signals)
        | set(positions)
        | set(shadow_pair_events)
        | set(partial_exits)
        | set(final_exits)
        | set(snapshot_sharpe)
    )
    rows: list[dict[str, Any]] = []
    for key in sorted(bot_keys):
        bot_id, variant_id = key
        equity = _select_latest_account_snapshot(
            multi=latest_multi.get(key, {}),
            reconstructed=latest_exchange.get(key, {}),
        )
        pos = positions.get(key, {})
        fill = fills.get(key, {})
        equity_usdt = _float(equity.get("equity_usdt"), initial_equity_usdt)
        opened_orders = int(opened.get(key, 0))
        signal_count = int(signals.get(key, 0))
        shadow_events = int(shadow_pair_events.get(key, 0))
        shadow_only = opened_orders == 0 and (shadow_events > 0 or _is_funding_basis_bot(bot_id, variant_id))
        partial_counts = partial_exits.get(key, Counter())
        final_counts = final_exits.get(key, Counter())
        known_final_exit_count = sum(
            final_counts.get(reason, 0)
            for reason in (
                "take_profit_exit",
                "runner_take_profit_exit",
                "mfe_trailing_exit",
                "stop_loss_exit",
                "stop_loss_hit",
                "time_or_guard_exit",
            )
        )
        all_final_exit_count = sum(final_counts.values())
        rows.append(
            {
                "bot_id": bot_id,
                "variant_id": variant_id,
                "trading_role": "shadow_only" if shadow_only else "paper_trading",
                "paper_trading_bot": not shadow_only,
                "zero_open_reason": _zero_open_reason(
                    signal_count=signal_count,
                    opened_orders=opened_orders,
                    shadow_pair_events=shadow_events,
                    funding_basis=_is_funding_basis_bot(bot_id, variant_id),
                ),
                "signals": signal_count,
                "opened_orders": opened_orders,
                "closed_orders": int(closed.get(key, 0)),
                "fills": int(fill.get("fills", 0)),
                "shadow_pair_events": shadow_events,
                "hft_event_count": 0,
                "average_hold_ms": None,
                "spread_capture_bps": None,
                "adverse_selection_bps": None,
                "fill_ratio": 0.0,
                "stale_or_sequence_gap_guard_count": 0,
                **hft_cost_fields.get(key, _hft_cost_fields_from_payload({})),
                "partial_take_profit_count": int(partial_counts.get("partial_take_profit", 0)),
                "take_profit_count": int(final_counts.get("take_profit_exit", 0)),
                "runner_take_profit_count": int(final_counts.get("runner_take_profit_exit", 0)),
                "mfe_trailing_exit_count": int(final_counts.get("mfe_trailing_exit", 0)),
                "stop_loss_count": int(final_counts.get("stop_loss_exit", 0) + final_counts.get("stop_loss_hit", 0)),
                "time_or_guard_exit_count": int(final_counts.get("time_or_guard_exit", 0)),
                "other_exit_count": int(max(0, all_final_exit_count - known_final_exit_count)),
                "fees_paid": round(_float(fill.get("fees"), 0.0), 6),
                "equity_usdt": round(equity_usdt, 6),
                "equity_pnl_net": round(equity_usdt - initial_equity_usdt, 6),
                "snapshot_sharpe": snapshot_sharpe.get(key),
                "realized_price_pnl": round(_float(equity.get("realized_pnl_usdt"), 0.0), 6),
                "margin_used": round(_float(equity.get("margin_used_usdt"), 0.0), 6),
                "positions": int(pos.get("positions", 0)),
                "symbols": int(pos.get("symbols", 0)),
                "long_positions": int(pos.get("long_positions", 0)),
                "short_positions": int(pos.get("short_positions", 0)),
                "notional": round(_float(pos.get("notional"), 0.0), 6),
                "latest_snapshot": equity.get("created_at", ""),
                "latest_snapshot_source": equity.get("source", ""),
            }
        )
    return rows


def _latest_multi_equity(conn: sqlite3.Connection, run_id: str) -> dict[tuple[str, str], dict[str, Any]]:
    rows = conn.execute(
        """
        select *
        from (
            select *,
                   row_number() over (
                       partition by run_id, bot_id, variant_id
                       order by id desc
                   ) rn
            from equity_snapshots
            where run_id = ? and exchange = 'multi'
        )
        where rn = 1
        """,
        (run_id,),
    ).fetchall()
    return {(row["bot_id"], row["variant_id"]): dict(row) for row in rows}


def _latest_exchange_equity(
    conn: sqlite3.Connection,
    run_id: str,
    initial_equity_usdt: float,
) -> dict[tuple[str, str], dict[str, Any]]:
    rows = conn.execute(
        """
        select *
        from (
            select *,
                   row_number() over (
                       partition by run_id, bot_id, variant_id, exchange
                       order by id desc
                   ) rn
            from equity_snapshots
            where run_id = ? and exchange != 'multi'
        )
        where rn = 1
        """,
        (run_id,),
    ).fetchall()
    grouped: dict[tuple[str, str], list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault((row["bot_id"], row["variant_id"]), []).append(row)
    reconstructed: dict[tuple[str, str], dict[str, Any]] = {}
    for key, exchange_rows in grouped.items():
        exchange_count = len(exchange_rows)
        reconstructed[key] = {
            "id": max(int(row["id"]) for row in exchange_rows),
            "source": "reconstructed_exchange",
            "created_at": max(str(row["created_at"]) for row in exchange_rows),
            "equity_usdt": sum(float(row["equity_usdt"]) for row in exchange_rows)
            - initial_equity_usdt * max(0, exchange_count - 1),
            "cash_usdt": sum(float(row["cash_usdt"]) for row in exchange_rows)
            - initial_equity_usdt * max(0, exchange_count - 1),
            "margin_used_usdt": sum(float(row["margin_used_usdt"]) for row in exchange_rows),
            "realized_pnl_usdt": sum(float(row["realized_pnl_usdt"]) for row in exchange_rows),
        }
    return reconstructed


def _select_latest_account_snapshot(multi: dict[str, Any], reconstructed: dict[str, Any]) -> dict[str, Any]:
    if not reconstructed:
        if multi:
            return {**multi, "source": "multi"}
        return {}
    if not multi:
        return reconstructed
    if int(reconstructed.get("id", 0)) > int(multi.get("id", 0)):
        return reconstructed
    return {**multi, "source": "multi"}


def _count_by_bot(
    conn: sqlite3.Connection,
    run_id: str,
    table: str,
    where: str | None = None,
) -> dict[tuple[str, str], int]:
    sql = f"select bot_id, variant_id, count(*) count from {table} where run_id = ?"
    if where:
        sql += f" and {where}"
    sql += " group by bot_id, variant_id"
    return {(row["bot_id"], row["variant_id"]): int(row["count"]) for row in conn.execute(sql, (run_id,))}


def _fills_by_bot(conn: sqlite3.Connection, run_id: str) -> dict[tuple[str, str], dict[str, Any]]:
    rows = conn.execute(
        """
        select bot_id, variant_id, count(*) fills, coalesce(sum(fee), 0.0) fees
        from fills
        where run_id = ?
        group by bot_id, variant_id
        """,
        (run_id,),
    ).fetchall()
    return {(row["bot_id"], row["variant_id"]): dict(row) for row in rows}


def _hft_cost_fields_by_bot(conn: sqlite3.Connection, run_id: str) -> dict[tuple[str, str], dict[str, Any]]:
    if not _table_exists(conn, "signals") or not _table_has_columns(conn, "signals", {"features_json"}):
        return {}
    rows = conn.execute(
        """
        select bot_id, variant_id, features_json
        from signals
        where run_id = ?
        order by id
        """,
        (run_id,),
    ).fetchall()
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        fields = _hft_cost_fields_from_payload(_json_dict(row["features_json"]))
        if any(value is not None for value in fields.values()):
            result[(row["bot_id"], row["variant_id"])] = fields
    return result


def _hft_cost_fields_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {name: _maybe_float(payload.get(name)) for name in _HFT_COST_FIELD_NAMES}


def _positions_by_bot(conn: sqlite3.Connection, run_id: str) -> dict[tuple[str, str], dict[str, Any]]:
    rows = conn.execute(
        """
        select bot_id,
               variant_id,
               count(*) positions,
               count(distinct symbol) symbols,
               sum(case when side = 'long' then 1 else 0 end) long_positions,
               sum(case when side = 'short' then 1 else 0 end) short_positions,
               coalesce(sum(abs(qty * avg_price)), 0.0) notional
        from positions
        where run_id = ? and abs(qty) > 0
        group by bot_id, variant_id
        """,
        (run_id,),
    ).fetchall()
    return {(row["bot_id"], row["variant_id"]): dict(row) for row in rows}


def _partial_exit_events_by_bot(conn: sqlite3.Connection, run_id: str) -> dict[tuple[str, str], Counter[str]]:
    if not _table_exists(conn, "trade_events"):
        return {}
    rows = conn.execute(
        """
        select bot_id, variant_id, reason_codes_json, reason_summary
        from trade_events
        where run_id = ? and event_type = 'partial_take_profit'
        """,
        (run_id,),
    ).fetchall()
    counts: dict[tuple[str, str], Counter[str]] = {}
    for row in rows:
        key = (row["bot_id"], row["variant_id"])
        bucket = counts.setdefault(key, Counter())
        codes = _json_list(row["reason_codes_json"])
        if codes:
            for code in codes:
                bucket[code] += 1
        else:
            bucket[str(row["reason_summary"] or "partial_take_profit")] += 1
    return counts


def _final_exit_reasons_by_bot(conn: sqlite3.Connection, run_id: str) -> dict[tuple[str, str], Counter[str]]:
    if not _table_exists(conn, "trade_lifecycle"):
        return {}
    rows = conn.execute(
        """
        select bot_id, variant_id, exit_reason_codes_json, exit_reason_summary
        from trade_lifecycle
        where run_id = ? and status = 'closed'
        """,
        (run_id,),
    ).fetchall()
    counts: dict[tuple[str, str], Counter[str]] = {}
    for row in rows:
        key = (row["bot_id"], row["variant_id"])
        bucket = counts.setdefault(key, Counter())
        codes = _json_list(row["exit_reason_codes_json"])
        if codes:
            for code in codes:
                bucket[_normalize_exit_code(code)] += 1
        else:
            bucket[_normalize_exit_code(str(row["exit_reason_summary"] or "unknown_exit"))] += 1
    return counts


def _snapshot_sharpe_by_bot(conn: sqlite3.Connection, run_id: str) -> dict[tuple[str, str], float | None]:
    rows = conn.execute(
        """
        select bot_id, variant_id, equity_usdt
        from equity_snapshots
        where run_id = ? and exchange = 'multi'
        order by bot_id, variant_id, id
        """,
        (run_id,),
    ).fetchall()
    series: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        series.setdefault((row["bot_id"], row["variant_id"]), []).append(float(row["equity_usdt"]))
    return {key: _snapshot_sharpe(values) for key, values in series.items()}


def _normalize_exit_code(code: str) -> str:
    if code == "stop_loss_hit" or code.startswith("stop_loss:"):
        return "stop_loss_exit"
    return code


def _snapshot_sharpe(values: list[float]) -> float | None:
    returns: list[float] = []
    for previous, current in zip(values, values[1:]):
        if previous <= 0:
            continue
        returns.append((current - previous) / previous)
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / len(returns)
    if variance <= 0:
        return None
    return round(mean / math.sqrt(variance) * math.sqrt(len(returns)), 6)


def _maker_bot_row(config_path: Path, *, initial_equity_usdt: float) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_id = str(config["run_id"])
    bot_id = str(config["bot_id"])
    variant_id = str(config.get("strategy", {}).get("variant_id") or config.get("variant_id") or bot_id)
    ledger_path = _resolve_manifest_path(config_path.parent, config["ledger_path"])
    initial_quote = float(config.get("paper", {}).get("initial_quote_usdt", initial_equity_usdt))
    counts = {
        "orders": 0,
        "closed_orders": 0,
        "fills": 0,
        "fees": 0.0,
        "risk_events": 0,
        "hft_event_count": 0,
        "stop_loss_fills": 0,
        "guard_exit_fills": 0,
        "other_taker_stop_fills": 0,
    }
    inventory: dict[str, Any] = {}
    sharpe = None
    if ledger_path.exists():
        with sqlite3.connect(ledger_path) as conn:
            conn.row_factory = sqlite3.Row
            counts = _maker_counts(conn, run_id)
            inventory = _latest_maker_inventory(conn, run_id)
            sharpe = _maker_snapshot_sharpe(conn, run_id)
    base_qty = _float(inventory.get("base_qty"), 0.0)
    avg_price = _float(inventory.get("avg_price"), 0.0)
    quote_usdt = _float(inventory.get("quote_usdt"), initial_quote)
    equity_usdt = quote_usdt + base_qty * avg_price
    notional = abs(base_qty * avg_price)
    opened_orders = int(counts["orders"])
    fills = int(counts["fills"])
    fill_ratio = fills / opened_orders if opened_orders > 0 else 0.0
    return {
        "bot_id": bot_id,
        "variant_id": variant_id,
        "trading_role": "paper_maker",
        "paper_trading_bot": True,
        "zero_open_reason": _maker_zero_open_reason(opened_orders, fills),
        "signals": opened_orders,
        "opened_orders": opened_orders,
        "closed_orders": int(counts["closed_orders"]),
        "fills": fills,
        "shadow_pair_events": 0,
        "hft_event_count": int(counts["hft_event_count"]),
        "average_hold_ms": None,
        "spread_capture_bps": None,
        "adverse_selection_bps": None,
        "fill_ratio": round(fill_ratio, 6),
        "stale_or_sequence_gap_guard_count": int(counts["risk_events"]),
        "partial_take_profit_count": 0,
        "take_profit_count": 0,
        "runner_take_profit_count": 0,
        "mfe_trailing_exit_count": 0,
        "stop_loss_count": int(counts["stop_loss_fills"]),
        "time_or_guard_exit_count": int(counts["guard_exit_fills"]),
        "other_exit_count": int(counts["other_taker_stop_fills"]),
        "fees_paid": round(float(counts["fees"]), 6),
        "equity_usdt": round(equity_usdt, 6),
        "equity_pnl_net": round(equity_usdt - initial_quote, 6),
        "snapshot_sharpe": sharpe,
        "realized_price_pnl": round(_float(inventory.get("realized_pnl_usdt"), 0.0), 6),
        "margin_used": round(notional, 6),
        "positions": 1 if abs(base_qty) > 0 else 0,
        "symbols": 1 if abs(base_qty) > 0 else 0,
        "long_positions": 1 if base_qty > 0 else 0,
        "short_positions": 1 if base_qty < 0 else 0,
        "notional": round(notional, 6),
        "latest_snapshot": str(inventory.get("id", "")),
        "latest_snapshot_source": "maker_inventory" if inventory else "",
    }


def _maker_counts(conn: sqlite3.Connection, run_id: str) -> dict[str, Any]:
    orders = 0
    closed_orders = 0
    fills = 0
    fees = 0.0
    risk_events = 0
    hft_event_count = 0
    stop_loss_fills = 0
    guard_exit_fills = 0
    other_taker_stop_fills = 0
    if _table_exists(conn, "mm_orders"):
        orders = int(
            conn.execute("select count(distinct order_id) from mm_orders where run_id = ?", (run_id,)).fetchone()[0]
        )
        closed_orders = int(
            conn.execute(
                """
                select count(distinct order_id)
                from mm_orders
                where run_id = ? and status in ('filled', 'canceled', 'cancelled', 'expired', 'rejected')
                """,
                (run_id,),
            ).fetchone()[0]
        )
    if _table_exists(conn, "mm_fills"):
        row = conn.execute(
            "select count(*) fills, coalesce(sum(fee_usdt), 0.0) fees from mm_fills where run_id = ?",
            (run_id,),
        ).fetchone()
        fills = int(row["fills"])
        fees = float(row["fees"])
        if _table_has_columns(conn, "mm_fills", {"liquidity", "trade_id"}):
            exit_row = conn.execute(
                """
                select
                    coalesce(sum(case
                        when liquidity = 'taker_stop' and trade_id = 'inventory_stop_loss' then 1
                        else 0
                    end), 0) stop_loss_fills,
                    coalesce(sum(case
                        when liquidity = 'taker_stop'
                         and trade_id in ('inventory_cap_exceeded', 'notional_cap_exceeded') then 1
                        else 0
                    end), 0) guard_exit_fills,
                    coalesce(sum(case
                        when liquidity = 'taker_stop'
                         and trade_id not in ('inventory_stop_loss', 'inventory_cap_exceeded', 'notional_cap_exceeded') then 1
                        else 0
                    end), 0) other_taker_stop_fills
                from mm_fills
                where run_id = ?
                """,
                (run_id,),
            ).fetchone()
            stop_loss_fills = int(exit_row["stop_loss_fills"])
            guard_exit_fills = int(exit_row["guard_exit_fills"])
            other_taker_stop_fills = int(exit_row["other_taker_stop_fills"])
    if _table_exists(conn, "mm_risk_events"):
        risk_events = int(conn.execute("select count(*) from mm_risk_events where run_id = ?", (run_id,)).fetchone()[0])
    if _table_exists(conn, "mm_latency_events"):
        hft_event_count = int(
            conn.execute("select count(*) from mm_latency_events where run_id = ?", (run_id,)).fetchone()[0]
        )
    return {
        "orders": orders,
        "closed_orders": closed_orders,
        "fills": fills,
        "fees": fees,
        "risk_events": risk_events,
        "hft_event_count": hft_event_count,
        "stop_loss_fills": stop_loss_fills,
        "guard_exit_fills": guard_exit_fills,
        "other_taker_stop_fills": other_taker_stop_fills,
    }


def _latest_maker_inventory(conn: sqlite3.Connection, run_id: str) -> dict[str, Any]:
    if not _table_exists(conn, "mm_inventory_snapshots"):
        return {}
    row = conn.execute(
        """
        select *
        from mm_inventory_snapshots
        where run_id = ?
        order by id desc
        limit 1
        """,
        (run_id,),
    ).fetchone()
    return dict(row) if row is not None else {}


def _maker_snapshot_sharpe(conn: sqlite3.Connection, run_id: str) -> float | None:
    if not _table_exists(conn, "mm_inventory_snapshots"):
        return None
    rows = conn.execute(
        """
        select quote_usdt, base_qty, avg_price
        from mm_inventory_snapshots
        where run_id = ?
        order by id
        """,
        (run_id,),
    ).fetchall()
    values = [float(row["quote_usdt"]) + float(row["base_qty"]) * float(row["avg_price"]) for row in rows]
    return _snapshot_sharpe(values)


def _maker_zero_open_reason(opened_orders: int, fills: int) -> str:
    if fills > 0:
        return ""
    if opened_orders > 0:
        return "no_maker_fill"
    return "no_maker_order"


def _shadow_pair_events_by_bot(conn: sqlite3.Connection, run_id: str) -> dict[tuple[str, str], int]:
    if not _table_exists(conn, "shadow_pair_events"):
        return {}
    rows = conn.execute(
        """
        select bot_id, variant_id, count(*) count
        from shadow_pair_events
        where run_id = ?
        group by bot_id, variant_id
        """,
        (run_id,),
    ).fetchall()
    return {(row["bot_id"], row["variant_id"]): int(row["count"]) for row in rows}


def _is_funding_basis_bot(bot_id: str, variant_id: str) -> bool:
    text = f"{bot_id} {variant_id}".lower()
    return "funding_basis" in text


def _zero_open_reason(
    *,
    signal_count: int,
    opened_orders: int,
    shadow_pair_events: int,
    funding_basis: bool,
) -> str:
    if opened_orders > 0:
        return ""
    if funding_basis and shadow_pair_events > 0:
        return "shadow_only_funding_basis"
    if shadow_pair_events > 0:
        return "shadow_only"
    if funding_basis:
        return "funding_basis_shadow_or_no_trigger"
    if signal_count > 0:
        return "signals_rejected_or_blocked"
    return "no_signal"


def _risk_rejections(conn: sqlite3.Connection, run_id: str) -> Counter[str]:
    rows = conn.execute(
        """
        select payload_json
        from risk_events
        where run_id = ? and reason = 'intent_rejected'
        """,
        (run_id,),
    ).fetchall()
    counts: Counter[str] = Counter()
    for row in rows:
        try:
            reason = json.loads(row["payload_json"]).get("risk_rejection_reason", "unknown")
        except json.JSONDecodeError:
            reason = "invalid_payload"
        counts[str(reason)] += 1
    return counts


def _exit_event_buckets(conn: sqlite3.Connection, run_id: str) -> Counter[str]:
    if not _table_exists(conn, "trade_events"):
        return Counter()
    rows = conn.execute(
        """
        select event_type, reason_codes_json, reason_summary
        from trade_events
        where run_id = ? and event_type in ('partial_take_profit', 'close_fill')
        """,
        (run_id,),
    ).fetchall()
    counts: Counter[str] = Counter()
    for row in rows:
        codes = _json_list(row["reason_codes_json"])
        if codes:
            for code in codes:
                counts[_normalize_exit_code(code)] += 1
            continue
        reason = _normalize_exit_code(str(row["reason_summary"] or row["event_type"] or "unknown_exit_event"))
        counts[reason] += 1
    return counts


def _trade_journal(conn: sqlite3.Connection, run_id: str) -> dict[str, Any]:
    legacy_gap = _legacy_journal_gap(conn, run_id)
    if not _table_exists(conn, "trade_lifecycle"):
        return _empty_trade_journal(**legacy_gap)
    rows = conn.execute(
        """
        select *
        from trade_lifecycle
        where run_id = ?
        order by opened_at, trade_id
        """,
        (run_id,),
    ).fetchall()
    if not rows:
        return _empty_trade_journal(**legacy_gap)

    entry_reasons: Counter[str] = Counter()
    exit_reasons: Counter[str] = Counter()
    exit_events = _exit_event_buckets(conn, run_id)
    strong_patterns: Counter[str] = Counter()
    risk_patterns: Counter[str] = Counter()
    wyckoff_setups: Counter[str] = Counter()
    quality_flags: Counter[str] = Counter()
    realized_values: list[float] = []
    r_values: list[float] = []
    mfe_capture_values: list[float] = []
    closed = 0
    winners = 0
    for row in rows:
        for code in _json_list(row["entry_reason_codes_json"]):
            entry_reasons[code] += 1
        for code in _json_list(row["exit_reason_codes_json"]):
            exit_reasons[_normalize_exit_code(code)] += 1
        features = _json_dict(row["entry_feature_snapshot_json"])
        strong_tag = str(features.get("strong_pattern_tag") or "")
        risk_tag = str(features.get("risk_pattern_tag") or "")
        wyckoff_tag = str(
            features.get("wyckoff_long_setup_tag")
            or features.get("wyckoff_short_setup_tag")
            or features.get("wyckoff_exit_tag")
            or ""
        )
        if strong_tag and strong_tag != "none":
            strong_patterns[strong_tag] += 1
        if risk_tag and risk_tag != "none":
            risk_patterns[risk_tag] += 1
        if wyckoff_tag and wyckoff_tag != "none":
            wyckoff_setups[wyckoff_tag] += 1
        for flag in _json_list(row["data_quality_flags_json"]):
            quality_flags[flag] += 1
        if row["status"] == "closed":
            closed += 1
            realized = _maybe_float(row["realized_pnl_usdt"])
            if realized is not None:
                realized_values.append(realized)
                if realized > 0:
                    winners += 1
            r_multiple = _maybe_float(row["r_multiple"])
            if r_multiple is not None:
                r_values.append(r_multiple)
            capture = _maybe_float(row["mfe_capture_ratio"])
            if capture is not None:
                mfe_capture_values.append(capture)

    return {
        "total_trades": len(rows),
        "closed_trades": closed,
        "open_trades": len(rows) - closed,
        "win_rate": round(winners / closed, 6) if closed else None,
        "realized_pnl_usdt": round(sum(realized_values), 6),
        "avg_realized_pnl_usdt": round(sum(realized_values) / len(realized_values), 6) if realized_values else None,
        "avg_r_multiple": round(sum(r_values) / len(r_values), 6) if r_values else None,
        "avg_mfe_capture_ratio": round(sum(mfe_capture_values) / len(mfe_capture_values), 6)
        if mfe_capture_values
        else None,
        "entry_reason_buckets": dict(entry_reasons),
        "exit_reason_buckets": dict(exit_reasons),
        "exit_event_buckets": dict(exit_events),
        "strong_pattern_buckets": dict(strong_patterns),
        "risk_pattern_buckets": dict(risk_patterns),
        "wyckoff_setup_buckets": dict(wyckoff_setups),
        "data_quality_flags": dict(quality_flags),
        **legacy_gap,
    }


def _empty_trade_journal(
    *,
    legacy_unjournaled_fills: int = 0,
    legacy_unjournaled_closed_orders: int = 0,
    journal_data_quality: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "total_trades": 0,
        "closed_trades": 0,
        "open_trades": 0,
        "win_rate": None,
        "realized_pnl_usdt": 0.0,
        "avg_realized_pnl_usdt": None,
        "avg_r_multiple": None,
        "avg_mfe_capture_ratio": None,
        "entry_reason_buckets": {},
        "exit_reason_buckets": {},
        "exit_event_buckets": {},
        "strong_pattern_buckets": {},
        "risk_pattern_buckets": {},
        "wyckoff_setup_buckets": {},
        "data_quality_flags": {},
        "legacy_unjournaled_fills": legacy_unjournaled_fills,
        "legacy_unjournaled_closed_orders": legacy_unjournaled_closed_orders,
        "journal_data_quality": journal_data_quality or [],
    }


def _legacy_journal_gap(conn: sqlite3.Connection, run_id: str) -> dict[str, Any]:
    if _table_exists(conn, "trade_events"):
        fills = conn.execute(
            """
            select count(*)
            from fills f
            left join trade_events e
              on e.run_id = f.run_id
             and e.bot_id = f.bot_id
             and e.exchange = f.exchange
             and e.fill_id = f.id
            where f.run_id = ? and e.id is null
            """,
            (run_id,),
        ).fetchone()[0]
    else:
        fills = conn.execute("select count(*) from fills where run_id = ?", (run_id,)).fetchone()[0]
    if _table_exists(conn, "trade_lifecycle"):
        closed_orders = conn.execute(
            """
            select count(*)
            from orders o
            left join trade_lifecycle t
              on t.run_id = o.run_id
             and t.bot_id = o.bot_id
             and t.exchange = o.exchange
             and t.exit_order_id = o.id
            where o.run_id = ? and o.reduce_only = 1 and t.trade_id is null
            """,
            (run_id,),
        ).fetchone()[0]
    else:
        closed_orders = conn.execute(
            "select count(*) from orders where run_id = ? and reduce_only = 1",
            (run_id,),
        ).fetchone()[0]
    quality = []
    if fills:
        quality.append("legacy_fills_without_trade_lifecycle")
    if closed_orders:
        quality.append("legacy_closes_without_structured_exit_journal")
    return {
        "legacy_unjournaled_fills": int(fills),
        "legacy_unjournaled_closed_orders": int(closed_orders),
        "journal_data_quality": quality,
    }


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute("select name from sqlite_master where type = 'table' and name = ?", (table,)).fetchone()
    return row is not None


def _table_has_columns(conn: sqlite3.Connection, table: str, columns: set[str]) -> bool:
    existing = {row["name"] for row in conn.execute(f"pragma table_info({table})").fetchall()}
    return columns.issubset(existing)


def _json_list(payload: str | None) -> list[str]:
    if not payload:
        return []
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return ["invalid_json"]
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _json_dict(payload: str | None) -> dict[str, Any]:
    if not payload:
        return {}
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sharpe_values = [float(row["snapshot_sharpe"]) for row in rows if row.get("snapshot_sharpe") is not None]
    return {
        "bot_count": len(rows),
        "paper_trading_bot_count": sum(1 for row in rows if row.get("paper_trading_bot", True)),
        "shadow_only_bot_count": sum(1 for row in rows if not row.get("paper_trading_bot", True)),
        "opened_orders": sum(int(row["opened_orders"]) for row in rows),
        "closed_orders": sum(int(row["closed_orders"]) for row in rows),
        "fills": sum(int(row["fills"]) for row in rows),
        "shadow_pair_events": sum(int(row.get("shadow_pair_events", 0)) for row in rows),
        "hft_event_count": sum(int(row.get("hft_event_count", 0)) for row in rows),
        "partial_take_profit_count": sum(int(row.get("partial_take_profit_count", 0)) for row in rows),
        "take_profit_count": sum(int(row.get("take_profit_count", 0)) for row in rows),
        "runner_take_profit_count": sum(int(row.get("runner_take_profit_count", 0)) for row in rows),
        "mfe_trailing_exit_count": sum(int(row.get("mfe_trailing_exit_count", 0)) for row in rows),
        "stop_loss_count": sum(int(row.get("stop_loss_count", 0)) for row in rows),
        "time_or_guard_exit_count": sum(int(row.get("time_or_guard_exit_count", 0)) for row in rows),
        "other_exit_count": sum(int(row.get("other_exit_count", 0)) for row in rows),
        "positions": sum(int(row["positions"]) for row in rows),
        "symbols": sum(int(row["symbols"]) for row in rows),
        "avg_snapshot_sharpe": round(sum(sharpe_values) / len(sharpe_values), 6) if sharpe_values else None,
        "equity_pnl_net": round(sum(float(row["equity_pnl_net"]) for row in rows), 6),
        "fees_paid": round(sum(float(row["fees_paid"]) for row in rows), 6),
    }


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# LangLang Fleet Report",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- ledger: `{summary['ledger_path']}`",
        f"- generated_at: `{summary['generated_at']}`",
        "",
        "## Totals",
        "",
    ]
    totals = summary["totals"]
    for key in (
        "bot_count",
        "paper_trading_bot_count",
        "shadow_only_bot_count",
        "opened_orders",
        "closed_orders",
        "fills",
        "shadow_pair_events",
        "partial_take_profit_count",
        "take_profit_count",
        "runner_take_profit_count",
        "mfe_trailing_exit_count",
        "stop_loss_count",
        "time_or_guard_exit_count",
        "other_exit_count",
        "positions",
        "symbols",
        "avg_snapshot_sharpe",
        "equity_pnl_net",
        "fees_paid",
    ):
        lines.append(f"- {key}: {totals[key]}")
    journal = summary.get("trade_journal", _empty_trade_journal())
    lines.extend(
        [
            "",
            "## Trade Journal",
            "",
            f"- total_trades: {journal['total_trades']}",
            f"- closed_trades: {journal['closed_trades']}",
            f"- open_trades: {journal['open_trades']}",
            f"- win_rate: {journal['win_rate']}",
            f"- realized_pnl_usdt: {journal['realized_pnl_usdt']}",
            f"- avg_r_multiple: {journal['avg_r_multiple']}",
            f"- avg_mfe_capture_ratio: {journal['avg_mfe_capture_ratio']}",
            f"- legacy_unjournaled_fills: {journal.get('legacy_unjournaled_fills', 0)}",
            f"- legacy_unjournaled_closed_orders: {journal.get('legacy_unjournaled_closed_orders', 0)}",
            "",
            "### Entry Reasons",
            "",
        ]
    )
    if journal["entry_reason_buckets"]:
        for reason, count in sorted(journal["entry_reason_buckets"].items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")
    lines.extend(["", "### Exit Reasons", ""])
    if journal["exit_reason_buckets"]:
        for reason, count in sorted(journal["exit_reason_buckets"].items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")
    lines.extend(["", "### Exit Events", ""])
    if journal.get("exit_event_buckets"):
        for reason, count in sorted(journal["exit_event_buckets"].items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")
    lines.extend(["", "### Strong Pattern Buckets", ""])
    if journal["strong_pattern_buckets"]:
        for tag, count in sorted(journal["strong_pattern_buckets"].items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {tag}: {count}")
    else:
        lines.append("- none")
    lines.extend(["", "### Wyckoff Buckets", ""])
    if journal["wyckoff_setup_buckets"]:
        for tag, count in sorted(journal["wyckoff_setup_buckets"].items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {tag}: {count}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Bots",
            "",
            "| bot | role | zero_open_reason | opened | TP | runnerTP | partialTP | trail | SL | guard | other | fills | hft_events | fill_ratio | shadow_events | sharpe | net_pnl | fees | positions | symbols | long/short | notional |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["bots"]:
        lines.append(
            "| {bot_id} | {trading_role} | {zero_open_reason} | {opened_orders} | {take_profit_count} | "
            "{runner_take_profit_count} | {partial_take_profit_count} | {mfe_trailing_exit_count} | "
            "{stop_loss_count} | {time_or_guard_exit_count} | {other_exit_count} | {fills} | "
            "{hft_event_count} | {fill_ratio} | {shadow_pair_events} | {snapshot_sharpe} | "
            "{equity_pnl_net:.2f} | {fees_paid:.2f} | "
            "{positions} | {symbols} | {long_positions}/{short_positions} | {notional:.2f} |".format(**row)
        )
    lines.extend(["", "## Risk Rejections", ""])
    if summary["risk_rejections"]:
        for reason, count in sorted(summary["risk_rejections"].items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _float(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
