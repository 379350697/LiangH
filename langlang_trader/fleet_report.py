from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import sqlite3
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize a LangLang paper fleet ledger")
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", default="output/fleet/langlang_strategy_forest/reports")
    parser.add_argument("--initial-equity-usdt", type=float, default=10_000.0)
    args = parser.parse_args(argv)
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
    bot_keys = set(latest_multi) | set(latest_exchange) | set(opened) | set(closed) | set(fills) | set(signals) | set(positions)
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
        rows.append(
            {
                "bot_id": bot_id,
                "variant_id": variant_id,
                "signals": int(signals.get(key, 0)),
                "opened_orders": int(opened.get(key, 0)),
                "closed_orders": int(closed.get(key, 0)),
                "fills": int(fill.get("fills", 0)),
                "fees_paid": round(_float(fill.get("fees"), 0.0), 6),
                "equity_usdt": round(equity_usdt, 6),
                "equity_pnl_net": round(equity_usdt - initial_equity_usdt, 6),
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
            exit_reasons[code] += 1
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
    return {
        "bot_count": len(rows),
        "opened_orders": sum(int(row["opened_orders"]) for row in rows),
        "closed_orders": sum(int(row["closed_orders"]) for row in rows),
        "fills": sum(int(row["fills"]) for row in rows),
        "positions": sum(int(row["positions"]) for row in rows),
        "symbols": sum(int(row["symbols"]) for row in rows),
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
    for key in ("bot_count", "opened_orders", "closed_orders", "fills", "positions", "symbols", "equity_pnl_net", "fees_paid"):
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
            "| bot | opened | closed | fills | net_pnl | fees | positions | symbols | long/short | notional |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["bots"]:
        lines.append(
            "| {bot_id} | {opened_orders} | {closed_orders} | {fills} | {equity_pnl_net:.2f} | "
            "{fees_paid:.2f} | {positions} | {symbols} | {long_positions}/{short_positions} | {notional:.2f} |".format(
                **row
            )
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
