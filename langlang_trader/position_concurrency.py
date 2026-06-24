from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TradeInterval:
    trade_id: str
    symbol: str
    entry_time: datetime
    exit_time: datetime
    return_rate: float
    time_repair_status: str


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Infer LangLang concurrent position limits from trade sheets")
    parser.add_argument("--trades", default="output/langlang_v1_3/excel_digest/trade_sheet_label_matrix.csv")
    parser.add_argument("--out", default="output/langlang_v1_3/position_concurrency")
    args = parser.parse_args(argv)
    result = build_concurrency_report(trades_csv=args.trades, out_dir=args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_concurrency_report(*, trades_csv: str | Path, out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = analyze_trade_concurrency(trades_csv)
    summary_json = out / "position_concurrency_summary.json"
    entry_snapshot_csv = out / "position_concurrency_entry_snapshots.csv"
    report_md = out / "position_concurrency_report.md"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    _write_rows(entry_snapshot_csv, summary["entry_snapshots"])
    report_md.write_text(_report_markdown(summary), encoding="utf-8")
    return {
        "summary_json": str(summary_json),
        "entry_snapshot_csv": str(entry_snapshot_csv),
        "report_md": str(report_md),
    }


def analyze_trade_concurrency(trades_csv: str | Path, *, abnormal_entry_cluster_threshold: int = 10) -> dict[str, Any]:
    trades = _read_trade_intervals(Path(trades_csv))
    snapshots = _entry_snapshots(trades)
    clusters = Counter(row["entry_time"] for row in snapshots)
    abnormal_entry_times = {time for time, count in clusters.items() if count > abnormal_entry_cluster_threshold}
    robust_snapshots = [row for row in snapshots if row["entry_time"] not in abnormal_entry_times]
    position_counts = [int(row["open_positions_after_entry"]) for row in snapshots]
    symbol_counts = [int(row["open_symbols_after_entry"]) for row in snapshots]
    robust_position_counts = [int(row["open_positions_after_entry"]) for row in robust_snapshots]
    robust_symbol_counts = [int(row["open_symbols_after_entry"]) for row in robust_snapshots]
    right_tail = _tail_subset(snapshots, 0.95, high=True)
    big_loss = _tail_subset(snapshots, 0.10, high=False)
    repair_counts = Counter(trade.time_repair_status for trade in trades)
    recommended_positions = _recommended_cap(robust_position_counts)
    recommended_symbols = _recommended_cap(robust_symbol_counts)
    return {
        "source": str(trades_csv),
        "trade_count": len(trades),
        "time_repair_counts": dict(sorted(repair_counts.items())),
        "abnormal_entry_cluster_threshold": abnormal_entry_cluster_threshold,
        "abnormal_entry_clusters": [
            {"entry_time": time, "trade_count": count}
            for time, count in clusters.most_common()
            if time in abnormal_entry_times
        ],
        "entry_concurrency": {
            "max_open_positions": max(position_counts) if position_counts else 0,
            "p50_open_positions": _quantile(position_counts, 0.50),
            "p90_open_positions": _quantile(position_counts, 0.90),
            "p95_open_positions": _quantile(position_counts, 0.95),
            "p99_open_positions": _quantile(position_counts, 0.99),
            "max_open_symbols": max(symbol_counts) if symbol_counts else 0,
            "p50_open_symbols": _quantile(symbol_counts, 0.50),
            "p90_open_symbols": _quantile(symbol_counts, 0.90),
            "p95_open_symbols": _quantile(symbol_counts, 0.95),
            "p99_open_symbols": _quantile(symbol_counts, 0.99),
            "open_position_count_distribution": _count_distribution(position_counts),
            "open_symbol_count_distribution": _count_distribution(symbol_counts),
        },
        "robust_entry_concurrency": {
            "excluded_trade_count": len(snapshots) - len(robust_snapshots),
            "max_open_positions": max(robust_position_counts) if robust_position_counts else 0,
            "p50_open_positions": _quantile(robust_position_counts, 0.50),
            "p90_open_positions": _quantile(robust_position_counts, 0.90),
            "p95_open_positions": _quantile(robust_position_counts, 0.95),
            "p99_open_positions": _quantile(robust_position_counts, 0.99),
            "max_open_symbols": max(robust_symbol_counts) if robust_symbol_counts else 0,
            "p50_open_symbols": _quantile(robust_symbol_counts, 0.50),
            "p90_open_symbols": _quantile(robust_symbol_counts, 0.90),
            "p95_open_symbols": _quantile(robust_symbol_counts, 0.95),
            "p99_open_symbols": _quantile(robust_symbol_counts, 0.99),
        },
        "right_tail_top5pct": _subset_stats(right_tail),
        "big_loss_bottom10pct": _subset_stats(big_loss),
        "recommended_risk": {
            "max_open_positions": recommended_positions,
            "max_open_symbols": recommended_symbols,
            "basis": "robust_entry_time_p95_with_same_timestamp_clusters_excluded",
        },
        "entry_snapshots": snapshots,
    }


def _read_trade_intervals(path: Path) -> list[TradeInterval]:
    rows: list[TradeInterval] = []
    with path.open(encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            entry = _parse_datetime(raw.get("entry_time", ""))
            exit_time = _parse_datetime(raw.get("exit_time", ""))
            if entry is None:
                continue
            hold_minutes = _float(raw.get("hold_minutes"), 0.0)
            repair_status = "as_reported"
            if exit_time is None:
                if hold_minutes > 0:
                    exit_time = entry + timedelta(minutes=hold_minutes)
                    repair_status = "exit_rebuilt_from_hold_minutes"
                else:
                    exit_time = entry + timedelta(minutes=1)
                    repair_status = "missing_exit_min_1m"
            elif exit_time < entry:
                if hold_minutes > 0:
                    exit_time = entry + timedelta(minutes=hold_minutes)
                    repair_status = "exit_rebuilt_from_hold_minutes"
                else:
                    exit_time = entry + timedelta(minutes=1)
                    repair_status = "negative_duration_min_1m"
            elif exit_time == entry:
                exit_time = entry + timedelta(minutes=1)
                repair_status = "zero_duration_min_1m"
            rows.append(
                TradeInterval(
                    trade_id=str(raw.get("trade_id", "")),
                    symbol=str(raw.get("symbol", "")),
                    entry_time=entry,
                    exit_time=exit_time,
                    return_rate=_float(raw.get("return_rate"), 0.0),
                    time_repair_status=repair_status,
                )
            )
    return sorted(rows, key=lambda item: (item.entry_time, item.trade_id))


def _entry_snapshots(trades: list[TradeInterval]) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for trade in trades:
        active = [other for other in trades if other.entry_time <= trade.entry_time < other.exit_time]
        symbols = {item.symbol for item in active}
        snapshots.append(
            {
                "trade_id": trade.trade_id,
                "symbol": trade.symbol,
                "entry_time": trade.entry_time.isoformat(sep=" "),
                "exit_time": trade.exit_time.isoformat(sep=" "),
                "return_rate": trade.return_rate,
                "open_positions_after_entry": len(active),
                "open_symbols_after_entry": len(symbols),
                "time_repair_status": trade.time_repair_status,
            }
        )
    return snapshots


def _subset_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positions = [int(row["open_positions_after_entry"]) for row in rows]
    symbols = [int(row["open_symbols_after_entry"]) for row in rows]
    return {
        "trade_count": len(rows),
        "avg_open_positions": sum(positions) / len(positions) if positions else 0.0,
        "p90_open_positions": _quantile(positions, 0.90),
        "max_open_positions": max(positions) if positions else 0,
        "avg_open_symbols": sum(symbols) / len(symbols) if symbols else 0.0,
        "p90_open_symbols": _quantile(symbols, 0.90),
        "max_open_symbols": max(symbols) if symbols else 0,
    }


def _tail_subset(rows: list[dict[str, Any]], quantile: float, *, high: bool) -> list[dict[str, Any]]:
    if not rows:
        return []
    values = sorted(float(row["return_rate"]) for row in rows)
    cutoff = values[int(quantile * (len(values) - 1))]
    if high:
        return [row for row in rows if float(row["return_rate"]) >= cutoff]
    return [row for row in rows if float(row["return_rate"]) <= cutoff]


def _quantile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[int(q * (len(ordered) - 1))]


def _count_distribution(values: list[int]) -> dict[str, int]:
    return {str(key): count for key, count in sorted(Counter(values).items())}


def _recommended_cap(values: list[int]) -> int:
    if not values:
        return 1
    max_value = max(values)
    if max_value <= 3:
        return max_value
    return max(3, int(_quantile(values, 0.95)))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _report_markdown(summary: dict[str, Any]) -> str:
    entry = summary["entry_concurrency"]
    robust = summary["robust_entry_concurrency"]
    risk = summary["recommended_risk"]
    repairs = ", ".join(f"{key}={value}" for key, value in summary["time_repair_counts"].items())
    clusters = ", ".join(
        f"{item['entry_time']}:{item['trade_count']}" for item in summary["abnormal_entry_clusters"]
    ) or "none"
    return "\n".join(
        [
            "# LangLang Position Concurrency Report",
            "",
            f"- source: `{summary['source']}`",
            f"- trade_count: {summary['trade_count']}",
            f"- time_repairs: {repairs}",
            f"- abnormal_entry_clusters: {clusters}",
            f"- max_open_positions: {entry['max_open_positions']}",
            f"- p95_open_positions: {entry['p95_open_positions']}",
            f"- p99_open_positions: {entry['p99_open_positions']}",
            f"- max_open_symbols: {entry['max_open_symbols']}",
            f"- p95_open_symbols: {entry['p95_open_symbols']}",
            f"- p99_open_symbols: {entry['p99_open_symbols']}",
            f"- robust_excluded_trade_count: {robust['excluded_trade_count']}",
            f"- robust_p95_open_positions: {robust['p95_open_positions']}",
            f"- robust_p95_open_symbols: {robust['p95_open_symbols']}",
            "",
            "## Recommended Risk Caps",
            "",
            f"- max_open_positions: {risk['max_open_positions']}",
            f"- max_open_symbols: {risk['max_open_symbols']}",
            f"- basis: {risk['basis']}",
            "",
        ]
    )


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    for candidate in (text, text.replace("T", " ").replace("+00:00", "")):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return None


def _float(value: Any, default: float) -> float:
    try:
        if value in {None, ""}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    raise SystemExit(main())
