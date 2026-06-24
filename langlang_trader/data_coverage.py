from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MarketDataCoverageLedger:
    cache_dir: str | Path

    def build(self, trades: list[dict[str, Any]], *, bars: list[str]) -> list[dict[str, Any]]:
        cache_root = Path(self.cache_dir)
        rows: list[dict[str, Any]] = []
        for trade in trades:
            symbol = str(trade["symbol"])
            entry_ts = int(float(trade.get("entry_ts") or trade.get("entry_time_ts") or 0))
            for bar in bars:
                files = sorted((cache_root / bar).glob(f"{symbol}_*.csv"))
                candle_count, first_ts, last_ts = _cache_stats(files)
                available = candle_count > 0 and (entry_ts <= 0 or first_ts <= entry_ts)
                rows.append(
                    {
                        "trade_id": trade.get("trade_id", ""),
                        "symbol": symbol,
                        "bar": bar,
                        "required_window": _required_window(entry_ts, bar),
                        "coverage_status": "available" if available else "unavailable",
                        "cached_files": len(files),
                        "candle_count": candle_count,
                        "first_ts": first_ts or "",
                        "last_ts": last_ts or "",
                        "unavailable_reason": "" if available else _unavailable_reason(files, candle_count, entry_ts, first_ts),
                        "evidence": ";".join(str(path) for path in files[:3]) if files else f"no cache file for {symbol} {bar}",
                    }
                )
        return rows

    def write(self, trades: list[dict[str, Any]], out_dir: str | Path, *, bars: list[str]) -> dict[str, str]:
        rows = self.build(trades, bars=bars)
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        csv_path = out_path / "market_data_coverage.csv"
        json_path = out_path / "market_data_coverage.json"
        md_path = out_path / "market_data_coverage.md"
        _write_rows(csv_path, rows)
        json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(_report(rows), encoding="utf-8")
        return {"csv": str(csv_path), "json": str(json_path), "report": str(md_path)}


def _cache_stats(files: list[Path]) -> tuple[int, int, int]:
    count = 0
    first_ts = 0
    last_ts = 0
    for path in files:
        try:
            with path.open(encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    ts = int(float(row.get("ts") or row.get("timestamp") or 0))
                    if ts <= 0:
                        continue
                    count += 1
                    first_ts = ts if first_ts == 0 else min(first_ts, ts)
                    last_ts = max(last_ts, ts)
        except (OSError, ValueError):
            continue
    return count, first_ts, last_ts


def _required_window(entry_ts: int, bar: str) -> str:
    lookbacks = {"1D": 140, "1H": 45, "15m": 7, "5m": 3, "1m": 1}
    unit = "days" if bar in {"1D", "1H", "15m", "5m", "1m"} else "bars"
    return f"entry_ts={entry_ts}; lookback={lookbacks.get(bar, 1)} {unit}"


def _unavailable_reason(files: list[Path], candle_count: int, entry_ts: int, first_ts: int) -> str:
    if not files:
        return "okx_cache_missing_or_unfetched"
    if candle_count <= 0:
        return "cache_files_empty_or_unreadable"
    if entry_ts > 0 and first_ts > entry_ts:
        return "cache_starts_after_trade_entry"
    return "coverage_window_incomplete"


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _report(rows: list[dict[str, Any]]) -> str:
    total = len(rows)
    available = sum(1 for row in rows if row["coverage_status"] == "available")
    unavailable = total - available
    unknown = sum(1 for row in rows if row["coverage_status"] == "unknown")
    reasons: dict[str, int] = {}
    for row in rows:
        reason = row.get("unavailable_reason") or "available"
        reasons[reason] = reasons.get(reason, 0) + 1
    lines = [
        "# Market Data Coverage",
        "",
        f"- rows: {total}",
        f"- available: {available}",
        f"- unavailable: {unavailable}",
        f"- unknown: {unknown}",
        "",
        "## Reasons",
        "",
    ]
    lines.extend(f"- {reason}: {count}" for reason, count in sorted(reasons.items()))
    return "\n".join(lines) + "\n"
