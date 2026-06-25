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
            "totals": _totals([]),
        }
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        bots = _bot_rows(conn, run_id, initial_equity_usdt)
        risk_rejections = _risk_rejections(conn, run_id)
    return {
        "run_id": run_id,
        "ledger_path": str(path),
        "generated_at": generated_at,
        "bots": bots,
        "risk_rejections": dict(risk_rejections),
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
