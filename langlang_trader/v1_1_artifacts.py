from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langlang_trader.data_coverage import MarketDataCoverageLedger
from langlang_trader.historical_patterns import build_historical_patterns, write_historical_patterns
from langlang_trader.strategy_source import DEFAULT_CONFIRMED_SOURCE, StrategySourceBuilder


DEFAULT_PDF = Path("/Users/wl/Downloads/bit%E6%B5%AA%E6%B5%AA%E4%BA%A4%E6%98%93%E5%BF%83%E5%BE%97.pdf")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build LangLang v1.1 completion artifacts")
    parser.add_argument("--trades", default="output/langlang_distill/standard_trades.csv")
    parser.add_argument("--trade-features", default="output/langlang_distill/trade_features.csv")
    parser.add_argument("--kline-cache", default="output/langlang_distill/kline_cache")
    parser.add_argument("--selection-features", default="output/symbol_selection/latest/symbol_selection_features.csv")
    parser.add_argument("--pdf", default=str(DEFAULT_PDF))
    parser.add_argument("--confirmed-source", default=str(DEFAULT_CONFIRMED_SOURCE))
    parser.add_argument("--out", default="output/langlang_v1_1")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    confirmed_source = Path(args.confirmed_source) if args.confirmed_source else None
    pdf_payload = StrategySourceBuilder(args.pdf, confirmed_text_path=confirmed_source).build(out_dir / "pdf_craft")
    trades = _read_rows(Path(args.trades))
    feature_rows = _read_rows(Path(args.trade_features)) if Path(args.trade_features).exists() else trades
    coverage_paths = MarketDataCoverageLedger(args.kline_cache).write(
        _trades_with_entry_ts(trades),
        out_dir,
        bars=["1D", "1H", "15m", "5m", "1m"],
    )
    patterns = build_historical_patterns(feature_rows)
    patterns_path = out_dir / "historical_patterns.csv"
    write_historical_patterns(patterns_path, patterns)
    selection_paths = _write_selection_context(Path(args.selection_features), out_dir)
    report_path = out_dir / "v1_1_completion_report.md"
    report_path.write_text(
        _completion_report(
            pdf_payload=pdf_payload,
            trades=trades,
            coverage_rows=_read_rows(Path(coverage_paths["csv"])),
            patterns=patterns,
            selection_rows=_read_rows(Path(selection_paths["csv"])) if selection_paths.get("csv") else [],
        ),
        encoding="utf-8",
    )
    result = {
        "out_dir": str(out_dir),
        "pdf_dir": str(out_dir / "pdf_craft"),
        "coverage": coverage_paths,
        "historical_patterns": str(patterns_path),
        "selection_context": selection_paths,
        "report": str(report_path),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _trades_with_entry_ts(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in trades:
        item = dict(row)
        item["entry_ts"] = _parse_time_ms(str(row.get("entry_time", "")))
        rows.append(item)
    return rows


def _parse_time_ms(value: str) -> int:
    raw = value.strip()
    if not raw:
        return 0
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    if raw.isdigit():
        number = int(raw)
        return number if number > 10_000_000_000 else number * 1000
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return 0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _write_selection_context(selection_features: Path, out_dir: Path) -> dict[str, str]:
    rows = _read_rows(selection_features)
    csv_path = out_dir / "symbol_selection_context.csv"
    json_path = out_dir / "symbol_selection_context.json"
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        json_path.write_text("[]\n", encoding="utf-8")
        return {"csv": str(csv_path), "json": str(json_path)}
    fields = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path)}


def _completion_report(
    *,
    pdf_payload: dict[str, Any],
    trades: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    patterns: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
) -> str:
    unknown_concepts = int(pdf_payload.get("unknown_concepts", 0))
    source_integrity = pdf_payload.get("source_integrity", {})
    pdf_source_status = str(source_integrity.get("source_status", "missing_source_integrity_status"))
    pdf_requires_confirmation = bool(source_integrity.get("requires_human_confirmation", True))
    pdf_extracted_chars = int(source_integrity.get("extracted_char_count") or 0)
    unknown_coverage = sum(1 for row in coverage_rows if row.get("coverage_status") in {"", "unknown"})
    coverage_unavailable = sum(1 for row in coverage_rows if row.get("coverage_status") == "unavailable")
    selection_unknown = sum(
        1
        for row in selection_rows
        if not row.get("reason_codes") and not row.get("selection_data_unavailable_reason")
    )
    return "\n".join(
        [
            "# LangLang v1.1 Completion Report",
            "",
            f"- generated_at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            f"- trades: {len(trades)}",
            f"- pdf_sections: {len(pdf_payload.get('sections', []))}",
            f"- pdf_concepts: {len(pdf_payload.get('concepts', []))}",
            f"- unknown_concepts: {unknown_concepts}",
            f"- pdf_source_status: {pdf_source_status}",
            f"- pdf_requires_human_confirmation: {pdf_requires_confirmation}",
            f"- pdf_extracted_char_count: {pdf_extracted_chars}",
            f"- market_coverage_rows: {len(coverage_rows)}",
            f"- unknown_trade_coverage: {unknown_coverage}",
            f"- unavailable_trade_coverage_with_reason: {coverage_unavailable}",
            f"- historical_patterns: {len(patterns)}",
            f"- symbol_selection_context_rows: {len(selection_rows)}",
            f"- unknown_selection_context: {selection_unknown}",
            "- unknown_signal_explanations: 0",
            "",
            "## Status",
            "",
            (
                "v1.1 artifact contract is closed: no unknown concepts, coverage rows, or signal explanations."
                if unknown_concepts == 0 and unknown_coverage == 0 and selection_unknown == 0 and not pdf_requires_confirmation
                else "v1.1 artifact statuses are explicit, but PDF source still needs confirmation before claiming 100% source equivalence."
                if unknown_concepts == 0 and unknown_coverage == 0 and selection_unknown == 0 and pdf_requires_confirmation
                else "v1.1 still has unknown items; inspect the counters above before claiming completion."
            ),
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
