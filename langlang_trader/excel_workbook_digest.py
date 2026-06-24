from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


PRIMARY_XLSX = Path(
    "/Users/wl/Downloads/%E6%B5%AA%E6%B5%AA%E4%BA%A4%E5%89%B2%E5%8D%95%E6%94%B9%E8%89%AF%E7%89%88.xlsx"
)
SECONDARY_XLSX = Path(
    "/Users/wl/Downloads/2022.5~2024.6%E6%B5%AA%E6%B5%AA%E4%BA%A4%E6%98%93%E4%BA%A4%E5%89%B2%E5%8D%95%E6%95%B0%E6%8D%AE3+(1)+(1).xlsx"
)

TRADE_INDEX_SHEET = "时间排列+去除金额错误单子"
RAW_SOURCE_SHEET = "初始版"
MANUAL_REVIEW_SHEET = "5%波动以上单子"
LARGE_MOVE_SHEETS = {"5%波动以上的单子", "5%波动以上单子"}
REFERENCE_SYMBOLS = ("BTC-USDT-SWAP", "ETH-USDT-SWAP")
TIMEFRAMES = ("1D", "1H", "15m", "5m", "1m")
BAR_MS = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "1H": 60 * 60 * 1000,
    "1D": 24 * 60 * 60 * 1000,
}
STANDARD_WINDOWS = {
    "1D": (180 * BAR_MS["1D"], 30 * BAR_MS["1D"]),
    "1H": (45 * BAR_MS["1D"], 7 * BAR_MS["1D"]),
    "15m": (10 * BAR_MS["1D"], 3 * BAR_MS["1D"]),
    "5m": (3 * BAR_MS["1D"], BAR_MS["1D"]),
    "1m": (BAR_MS["1D"], BAR_MS["1D"]),
}
EXPANDED_WINDOWS = {
    "1D": (365 * BAR_MS["1D"], 90 * BAR_MS["1D"]),
    "1H": (90 * BAR_MS["1D"], 14 * BAR_MS["1D"]),
    "15m": (21 * BAR_MS["1D"], 7 * BAR_MS["1D"]),
    "5m": (7 * BAR_MS["1D"], 3 * BAR_MS["1D"]),
    "1m": (2 * BAR_MS["1D"], 2 * BAR_MS["1D"]),
}


@dataclass(frozen=True)
class SheetData:
    workbook: str
    sheet_name: str
    rows: list[list[Any]]
    formula_cells: int


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Digest all LangLang Excel workbook sheets before strategy distillation")
    parser.add_argument("--primary-xlsx", default=str(PRIMARY_XLSX))
    parser.add_argument("--secondary-xlsx", default=str(SECONDARY_XLSX))
    parser.add_argument("--trades", default="output/langlang_distill/standard_trades.csv")
    parser.add_argument("--kline-cache", default="output/langlang_distill/kline_cache")
    parser.add_argument(
        "--terminal-coverage",
        action="append",
        default=[],
        help="Optional kline_backfill coverage CSV with terminal unavailable evidence. May be passed multiple times.",
    )
    parser.add_argument("--out", default="output/langlang_excel_digest")
    args = parser.parse_args(argv)
    result = build_excel_digest_artifacts(
        workbooks=[args.primary_xlsx, args.secondary_xlsx],
        trades_csv=args.trades,
        kline_cache=args.kline_cache,
        terminal_coverage_csv=args.terminal_coverage or None,
        out_dir=args.out,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_excel_digest_artifacts(
    *,
    workbooks: list[str | Path],
    trades_csv: str | Path,
    out_dir: str | Path,
    kline_cache: str | Path | None = None,
    terminal_coverage_csv: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
) -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    digest = digest_workbooks([Path(path) for path in workbooks], Path(trades_csv))
    windows = build_required_kline_windows_from_digest(digest["trade_matrix"])
    coverage_rows = (
        build_kline_window_coverage(windows, Path(kline_cache), terminal_coverage_csv=terminal_coverage_csv)
        if kline_cache
        else []
    )
    inventory_csv = out_path / "excel_workbook_inventory.csv"
    inventory_json = out_path / "excel_workbook_inventory.json"
    matrix_csv = out_path / "trade_sheet_label_matrix.csv"
    matrix_json = out_path / "trade_sheet_label_matrix.json"
    events_csv = out_path / "excel_evidence_event_dataset.csv"
    events_json = out_path / "excel_evidence_event_dataset.json"
    windows_csv = out_path / "required_kline_windows.csv"
    windows_json = out_path / "required_kline_windows.json"
    coverage_csv = out_path / "kline_window_coverage.csv"
    coverage_json = out_path / "kline_window_coverage.json"
    report_path = out_path / "excel_workbook_digest.md"
    _write_rows(inventory_csv, digest["sheet_inventory"])
    inventory_json.write_text(json.dumps(digest["sheet_inventory"], ensure_ascii=False, indent=2), encoding="utf-8")
    _write_rows(matrix_csv, digest["trade_matrix"])
    matrix_json.write_text(json.dumps(digest["trade_matrix"], ensure_ascii=False, indent=2), encoding="utf-8")
    events = build_excel_evidence_events(digest["sheet_data"], digest["trade_matrix"])
    _write_rows(events_csv, events)
    events_json.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_rows(windows_csv, windows)
    windows_json.write_text(json.dumps(windows, ensure_ascii=False, indent=2), encoding="utf-8")
    if coverage_rows:
        _write_rows(coverage_csv, coverage_rows)
        coverage_json.write_text(json.dumps(coverage_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_report(digest, windows, coverage_rows), encoding="utf-8")
    return {
        "workbook_count": digest["workbook_count"],
        "sheet_count": digest["sheet_count"],
        "trade_rows": len(digest["trade_matrix"]),
        "required_kline_windows": len(windows),
        "kline_window_coverage": len(coverage_rows),
        "unknown_sheet_count": digest["unknown_sheet_count"],
        "inventory_csv": str(inventory_csv),
        "trade_sheet_label_matrix": str(matrix_csv),
        "excel_evidence_event_dataset": str(events_csv),
        "required_kline_windows_csv": str(windows_csv),
        "kline_window_coverage_csv": str(coverage_csv) if coverage_rows else "",
        "report": str(report_path),
    }


def digest_workbooks(workbooks: list[str | Path], trades_csv: str | Path) -> dict[str, Any]:
    sheets: list[SheetData] = []
    for path in workbooks:
        workbook_path = Path(path)
        if workbook_path.exists():
            sheets.extend(_read_xlsx_sheets(workbook_path))
    inventory = [_sheet_inventory_row(sheet) for sheet in sheets]
    trades = _read_standard_trades(Path(trades_csv))
    labels_by_trade = _labels_by_trade_id(sheets)
    symbol_rank = _symbol_profit_rank(sheets)
    matrix = [_trade_matrix_row(trade, labels_by_trade.get(str(trade["trade_id"]), {}), symbol_rank) for trade in trades]
    unknown = sum(1 for row in inventory if row["sheet_role"] == "unknown")
    return {
        "workbook_count": len([path for path in workbooks if Path(path).exists()]),
        "sheet_count": len(inventory),
        "unknown_sheet_count": unknown,
        "sheet_inventory": inventory,
        "trade_matrix": matrix,
        "sheet_data": sheets,
    }


def build_excel_evidence_events(sheets: list[SheetData], trade_matrix: list[dict[str, Any]]) -> list[dict[str, Any]]:
    known_trade_ids = {str(row.get("trade_id") or "") for row in trade_matrix}
    events: list[dict[str, Any]] = []
    for sheet in sheets:
        header_idx, header = _find_header(sheet.rows)
        if header_idx < 0:
            continue
        role = _classify_sheet(
            sheet.sheet_name,
            header,
            sum(1 for row in sheet.rows[header_idx + 1 :] if _row_get(row, header, "交易对") not in {"", None}),
        )
        if role in {"empty_or_formula_only", "unknown"}:
            continue
        workbook = Path(sheet.workbook).name
        for row_offset, row in enumerate(sheet.rows[header_idx + 1 :], start=1):
            trade_id = str(_row_get(row, header, "序号")).strip()
            if trade_id.endswith(".0"):
                trade_id = trade_id[:-2]
            symbol = str(_first_nonempty(row, header, ["交易对", "列B"])).strip()
            if not trade_id and not symbol:
                continue
            side = str(_row_get(row, header, "方向")).strip()
            source_row_number = header_idx + 1 + row_offset + 1
            for col_idx, field in enumerate(header):
                field_name = str(field).strip()
                if not field_name:
                    continue
                value = row[col_idx] if col_idx < len(row) else ""
                if value in {"", None}:
                    continue
                field_value = str(value).strip()
                if not field_value or field_value == "None":
                    continue
                semantic = _event_semantic(field_name, sheet.sheet_name, role)
                event_id = f"{workbook}|{sheet.sheet_name}|{source_row_number}|{field_name}"
                events.append(
                    {
                        "event_id": event_id,
                        "trade_id": trade_id if trade_id in known_trade_ids or trade_id else "",
                        "symbol": symbol,
                        "side": side,
                        "workbook": workbook,
                        "sheet_name": sheet.sheet_name,
                        "sheet_role": role,
                        "source_row_number": source_row_number,
                        "field_name": field_name,
                        "field_value": field_value,
                        "event_semantic": semantic,
                        "evidence_role": role,
                        "evidence_weight": _event_weight(semantic, role),
                    }
                )
    return events


def build_required_kline_windows_from_digest(trade_matrix: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    min_start: int | None = None
    max_end: int | None = None
    for trade in trade_matrix:
        entry_ts = _parse_time_ms(str(trade.get("entry_time") or ""))
        if entry_ts <= 0:
            continue
        exit_ts = _parse_time_ms(str(trade.get("exit_time") or "")) or entry_ts
        expanded = trade.get("large_move_label") not in {"", "not_large_move"} or bool(trade.get("manual_review_text"))
        profile = "expanded_manual_review" if expanded else "standard"
        windows = EXPANDED_WINDOWS if expanded else STANDARD_WINDOWS
        for timeframe in TIMEFRAMES:
            before_ms, after_ms = windows[timeframe]
            start_ms = entry_ts - before_ms
            end_ms = max(exit_ts, entry_ts) + after_ms
            min_start = start_ms if min_start is None else min(min_start, start_ms)
            max_end = end_ms if max_end is None else max(max_end, end_ms)
            rows.append(
                {
                    "trade_id": str(trade.get("trade_id") or ""),
                    "symbol": trade.get("symbol", ""),
                    "timeframe": timeframe,
                    "window_profile": profile,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "before_ms": before_ms,
                    "after_ms": after_ms,
                    "coverage_status": "required",
                    "coverage_reason": "large_move_or_manual_review" if expanded else "standard_trade_window",
                    "source": "excel_digest",
                }
            )
    if min_start is not None and max_end is not None:
        for symbol in REFERENCE_SYMBOLS:
            for timeframe in TIMEFRAMES:
                rows.append(
                    {
                        "trade_id": "market_reference",
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "window_profile": "market_reference",
                        "start_ms": min_start,
                        "end_ms": max_end,
                        "before_ms": "",
                        "after_ms": "",
                        "coverage_status": "required",
                        "coverage_reason": "btc_eth_market_environment",
                        "source": "excel_digest",
                    }
                )
    return rows


def build_kline_window_coverage(
    required_windows: list[dict[str, Any]],
    cache_dir: Path,
    *,
    terminal_coverage_csv: str | Path | None = None,
) -> list[dict[str, Any]]:
    ts_cache: dict[tuple[str, str], list[int]] = {}
    terminal_evidence = _load_terminal_coverage(terminal_coverage_csv) if terminal_coverage_csv else {}
    rows: list[dict[str, Any]] = []
    for window in required_windows:
        symbol = str(window.get("symbol") or "")
        timeframe = str(window.get("timeframe") or "")
        key = (symbol, timeframe)
        if key not in ts_cache:
            ts_cache[key] = _load_cache_timestamps(cache_dir, symbol, timeframe)
        timestamps = ts_cache[key]
        start_ms = int(window.get("start_ms") or 0)
        end_ms = int(window.get("end_ms") or 0)
        status, reason, available_start, available_end, available_rows = _coverage_status(timestamps, start_ms, end_ms)
        evidence = ""
        if status != "available":
            terminal = _matching_terminal_evidence(
                terminal_evidence.get((symbol, timeframe), []),
                start_ms=start_ms,
                end_ms=end_ms,
            )
            if terminal:
                status = str(terminal.get("coverage_status") or status)
                reason = str(terminal.get("unavailable_reason") or reason)
                evidence = str(terminal.get("evidence") or "")
        row = dict(window)
        row.update(
            {
                "coverage_status": status,
                "unavailable_reason": reason,
                "available_start_ms": available_start,
                "available_end_ms": available_end,
                "available_rows": available_rows,
                "evidence": evidence,
            }
        )
        rows.append(row)
    return rows


def _read_xlsx_sheets(path: Path) -> list[SheetData]:
    with zipfile.ZipFile(path) as zf:
        shared_strings = _read_shared_strings(zf)
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = _read_relationships(zf, "xl/_rels/workbook.xml.rels")
        ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main", "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
        sheets: list[SheetData] = []
        for sheet in workbook.findall(".//m:sheet", ns):
            name = str(sheet.attrib.get("name") or "")
            rid = sheet.attrib.get(f"{{{ns['r']}}}id") or ""
            target = rels.get(rid, "")
            if not target:
                continue
            target_path = target if target.startswith("xl/") else f"xl/{target}"
            if target_path not in zf.namelist():
                continue
            rows, formula_cells = _read_sheet_xml(zf.read(target_path), shared_strings)
            sheets.append(SheetData(workbook=str(path), sheet_name=name, rows=rows, formula_cells=formula_cells))
        return sheets


def _read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    values: list[str] = []
    for si in root.findall(".//m:si", ns):
        texts = [node.text or "" for node in si.findall(".//m:t", ns)]
        values.append("".join(texts))
    return values


def _load_cache_timestamps(cache_dir: Path, symbol: str, timeframe: str) -> list[int]:
    candidates = sorted((cache_dir / timeframe).glob(f"{symbol}*.csv"))
    if timeframe == "1D":
        candidates.extend(sorted(cache_dir.glob(f"{symbol}*.csv")))
    candidates = [candidate for candidate in dict.fromkeys(candidates) if candidate.exists()]
    if not candidates:
        return []
    timestamps: list[int] = []
    for path in candidates:
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    ts = int(float(row.get("ts") or row.get("timestamp") or 0))
                except (TypeError, ValueError):
                    continue
                if ts > 0:
                    timestamps.append(ts)
    return sorted(set(timestamps))


def _load_terminal_coverage(paths: str | Path | list[str | Path] | tuple[str | Path, ...]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    terminal_statuses = {
        "instrument_unavailable",
        "delisted_history_unavailable",
        "listing_boundary",
        "exchange_unavailable",
    }
    result: dict[tuple[str, str], list[dict[str, Any]]] = {}
    path_list = list(paths) if isinstance(paths, (list, tuple)) else [paths]
    for raw_path in path_list:
        path = Path(raw_path)
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                status = str(row.get("coverage_status") or "")
                if status not in terminal_statuses:
                    continue
                symbol = str(row.get("symbol") or "")
                timeframe = str(row.get("timeframe") or row.get("bar") or "")
                if not symbol or not timeframe:
                    continue
                try:
                    row["_start_ms"] = int(float(row.get("start_ms") or 0))
                    row["_end_ms"] = int(float(row.get("end_ms") or 0))
                except (TypeError, ValueError):
                    continue
                result.setdefault((symbol, timeframe), []).append(row)
    for rows in result.values():
        rows.sort(key=lambda item: (int(item["_start_ms"]), int(item["_end_ms"])))
    return result


def _matching_terminal_evidence(rows: list[dict[str, Any]], *, start_ms: int, end_ms: int) -> dict[str, Any] | None:
    for row in rows:
        row_start = int(row.get("_start_ms") or 0)
        row_end = int(row.get("_end_ms") or 0)
        if row_start <= start_ms and row_end >= end_ms:
            return row
    for row in rows:
        row_start = int(row.get("_start_ms") or 0)
        row_end = int(row.get("_end_ms") or 0)
        if row_start <= end_ms and row_end >= start_ms:
            return row
    return None


def _coverage_status(timestamps: list[int], start_ms: int, end_ms: int) -> tuple[str, str, Any, Any, int]:
    if not timestamps:
        return "insufficient_history", "cache_file_missing_or_empty", "", "", 0
    available_start = timestamps[0]
    available_end = timestamps[-1]
    rows_in_window = sum(1 for ts in timestamps if start_ms <= ts <= end_ms)
    if available_start <= start_ms and available_end >= end_ms and rows_in_window > 0:
        return "available", "", available_start, available_end, rows_in_window
    if available_start > start_ms:
        return "insufficient_history", "cache_starts_after_required_window", available_start, available_end, rows_in_window
    if available_end < end_ms:
        return "insufficient_history", "cache_ends_before_required_window", available_start, available_end, rows_in_window
    return "insufficient_history", "no_rows_inside_required_window", available_start, available_end, rows_in_window


def _read_relationships(zf: zipfile.ZipFile, path: str) -> dict[str, str]:
    root = ET.fromstring(zf.read(path))
    rels: dict[str, str] = {}
    for node in root:
        rid = node.attrib.get("Id")
        target = node.attrib.get("Target")
        if rid and target:
            rels[rid] = target
    return rels


def _read_sheet_xml(raw: bytes, shared_strings: list[str]) -> tuple[list[list[Any]], int]:
    root = ET.fromstring(raw)
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rows: list[list[Any]] = []
    formula_cells = 0
    for row in root.findall(".//m:row", ns):
        values: dict[int, Any] = {}
        for cell in row.findall("m:c", ns):
            ref = cell.attrib.get("r", "")
            col_idx = _column_index(ref)
            if col_idx <= 0:
                continue
            if cell.find("m:f", ns) is not None:
                formula_cells += 1
            values[col_idx] = _cell_value(cell, shared_strings, ns)
        if values:
            width = max(values)
            rows.append([values.get(idx, "") for idx in range(1, width + 1)])
    return rows, formula_cells


def _cell_value(cell: ET.Element, shared_strings: list[str], ns: dict[str, str]) -> Any:
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        texts = [node.text or "" for node in cell.findall(".//m:t", ns)]
        return "".join(texts)
    value_node = cell.find("m:v", ns)
    raw = value_node.text if value_node is not None else ""
    if cell_type == "s":
        try:
            return shared_strings[int(raw)]
        except (IndexError, TypeError, ValueError):
            return ""
    if cell_type == "str":
        return raw
    if raw == "":
        return ""
    try:
        number = float(raw)
        return int(number) if number.is_integer() else number
    except ValueError:
        return raw


def _sheet_inventory_row(sheet: SheetData) -> dict[str, Any]:
    header_idx, header = _find_header(sheet.rows)
    data_rows = sheet.rows[header_idx + 1 :] if header_idx >= 0 else []
    rows_with_trade_pair = sum(1 for row in data_rows if _row_get(row, header, "交易对") not in {"", None})
    valid_trade_ids = {
        str(_row_get(row, header, "序号")).strip()
        for row in data_rows
        if str(_row_get(row, header, "序号")).strip() not in {"", "None"}
    }
    role = _classify_sheet(sheet.sheet_name, header, rows_with_trade_pair)
    return {
        "workbook": Path(sheet.workbook).name,
        "sheet_name": sheet.sheet_name,
        "sheet_role": role,
        "access_status": "included" if role != "empty_or_formula_only" else "ignored_with_reason",
        "ignored_reason": "" if role != "empty_or_formula_only" else "no_trade_rows_or_no_effective_labels",
        "max_rows_read": len(sheet.rows),
        "nonempty_rows": sum(1 for row in sheet.rows if any(value not in {"", None} for value in row)),
        "header_row": header_idx + 1 if header_idx >= 0 else "",
        "valid_trade_rows": rows_with_trade_pair,
        "unique_trade_ids": len(valid_trade_ids),
        "formula_cells": sheet.formula_cells,
        "columns": "|".join(str(value) for value in header if str(value).strip()),
        "available_label_fields": "|".join(_available_label_fields(header)),
    }


def _classify_sheet(sheet_name: str, header: list[Any], rows_with_trade_pair: int) -> str:
    if sheet_name == TRADE_INDEX_SHEET:
        return "trade_index"
    if sheet_name == RAW_SOURCE_SHEET:
        return "raw_source"
    if sheet_name == "不同币做的收益率":
        return "derived_stats"
    if rows_with_trade_pair <= 0:
        return "empty_or_formula_only"
    if sheet_name == MANUAL_REVIEW_SHEET or any(str(value).strip() in _manual_fields() for value in header):
        return "manual_review"
    if sheet_name in LARGE_MOVE_SHEETS or sheet_name in {"单笔收益率（看止损）", "实际振幅与收益分析（看空间）", "所有单子持单时间（分钟）", "不同币做的收益率"}:
        return "derived_stats"
    return "unknown"


def _labels_by_trade_id(sheets: list[SheetData]) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for sheet in sheets:
        header_idx, header = _find_header(sheet.rows)
        if header_idx < 0:
            continue
        role = _classify_sheet(sheet.sheet_name, header, 1)
        if role in {"trade_index", "raw_source", "empty_or_formula_only", "unknown"}:
            continue
        for row in sheet.rows[header_idx + 1 :]:
            trade_id = str(_row_get(row, header, "序号")).strip()
            if not trade_id or trade_id == "None":
                continue
            label = labels.setdefault(
                trade_id,
                {
                    "analysis_membership": set(),
                    "large_move_label": "",
                    "manual_review_parts": [],
                    "btc_cycle_label": "",
                    "entry_position_label": "",
                    "return_rate": None,
                    "realized_move": None,
                    "hold_minutes": None,
                },
            )
            if sheet.sheet_name in LARGE_MOVE_SHEETS:
                label["large_move_label"] = "large_move_manual_review" if sheet.sheet_name == MANUAL_REVIEW_SHEET else "large_move_filtered"
                label["analysis_membership"].add(sheet.sheet_name)
            if sheet.sheet_name == MANUAL_REVIEW_SHEET:
                label["analysis_membership"].add(sheet.sheet_name)
            btc_cycle = _first_nonempty(row, header, ["大饼大周期判断", "大饼情况"])
            if btc_cycle:
                label["btc_cycle_label"] = btc_cycle
            entry_position = _first_nonempty(row, header, ["大饼周期标底（属于主要开仓位置的那种）", "预测属于开仓位置的那种", "实际属于开仓位置的那种"])
            if entry_position not in {"", None}:
                label["entry_position_label"] = str(entry_position)
            for field in ("复盘操作细节", "操作细节", "操作间节"):
                text = str(_row_get(row, header, field)).strip()
                if text and text != "None":
                    label["manual_review_parts"].append(text)
            label["return_rate"] = _coalesce_number(label.get("return_rate"), _row_get(row, header, "收益率"))
            label["realized_move"] = _coalesce_number(label.get("realized_move"), _row_get(row, header, "收益率/倍数=实际振幅"))
            label["hold_minutes"] = _coalesce_number(label.get("hold_minutes"), _row_get(row, header, "交易时间差（分钟）"))
    return labels


def _trade_matrix_row(trade: dict[str, Any], label: dict[str, Any], symbol_rank: dict[str, int]) -> dict[str, Any]:
    trade_id = str(trade.get("trade_id") or "")
    symbol = str(trade.get("symbol") or "")
    return_rate = _coalesce_number(label.get("return_rate"), trade.get("return_rate"))
    realized_move = _coalesce_number(label.get("realized_move"), trade.get("realized_move"))
    hold_minutes = _coalesce_number(label.get("hold_minutes"), trade.get("hold_minutes"))
    membership = sorted(label.get("analysis_membership", set()))
    return {
        "trade_id": trade_id,
        "symbol": symbol,
        "side": trade.get("side", ""),
        "entry_time": trade.get("entry_time", ""),
        "exit_time": trade.get("exit_time", ""),
        "pnl_usdt": trade.get("pnl_usdt", ""),
        "return_rate": trade.get("return_rate", ""),
        "realized_move": trade.get("realized_move", ""),
        "hold_minutes": trade.get("hold_minutes", ""),
        "sheet_membership": "|".join(membership) if membership else "not_in_analysis_sheet",
        "large_move_label": label.get("large_move_label") or "not_large_move",
        "manual_review_text": " | ".join(label.get("manual_review_parts", [])),
        "btc_cycle_label": label.get("btc_cycle_label", ""),
        "entry_position_label": label.get("entry_position_label", ""),
        "space_bucket": _space_bucket(realized_move),
        "stop_loss_bucket": _return_bucket(return_rate),
        "hold_time_bucket": _hold_bucket(hold_minutes),
        "symbol_profit_rank": symbol_rank.get(symbol, ""),
        "sheet_label_status": "labeled" if label else "not_in_analysis_sheet",
    }


def _symbol_profit_rank(sheets: list[SheetData]) -> dict[str, int]:
    for sheet in sheets:
        if sheet.sheet_name != "不同币做的收益率":
            continue
        header = [str(value).strip() for value in sheet.rows[0]] if sheet.rows else []
        if "列B" not in header:
            continue
        symbol_idx = header.index("列B")
        rows = []
        for row in sheet.rows[1:]:
            symbol = str(row[symbol_idx]).strip() if symbol_idx < len(row) else ""
            if symbol.endswith("-USDT-SWAP"):
                rows.append(symbol)
        return {symbol: idx for idx, symbol in enumerate(rows, start=1)}
    return {}


def _read_standard_trades(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    trades: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        trade = dict(row)
        trade_id = str(trade.get("trade_id") or idx).strip()
        if trade_id.endswith(".0"):
            trade_id = trade_id[:-2]
        trade["trade_id"] = trade_id
        if trade.get("symbol") and trade.get("entry_time"):
            trades.append(trade)
    return trades


def _find_header(rows: list[list[Any]]) -> tuple[int, list[Any]]:
    for idx, row in enumerate(rows):
        normalized = [str(value).strip() for value in row]
        if "交易对" in normalized or "列B" in normalized:
            return idx, normalized
    return -1, []


def _row_get(row: list[Any], header: list[Any], field: str) -> Any:
    if field not in header:
        return ""
    idx = header.index(field)
    return row[idx] if idx < len(row) else ""


def _first_nonempty(row: list[Any], header: list[Any], fields: list[str]) -> Any:
    for field in fields:
        value = _row_get(row, header, field)
        if value not in {"", None}:
            return value
    return ""


def _available_label_fields(header: list[Any]) -> list[str]:
    fields = []
    for field in list(_manual_fields()) + ["大饼情况", "交易时间差（分钟）", "收益率/倍数=实际振幅", "收益率", "列B", "收益率/单数"]:
        if field in header:
            fields.append(field)
    return fields


def _event_semantic(field_name: str, sheet_name: str, role: str) -> str:
    if field_name in {"复盘操作细节", "操作细节", "操作间节"}:
        return "manual_review_text"
    if field_name in {"大饼大周期判断", "大饼情况"}:
        return "btc_cycle"
    if field_name in {"大饼周期标底（属于主要开仓位置的那种）", "预测属于开仓位置的那种", "实际属于开仓位置的那种"}:
        return "entry_position"
    if field_name in {"收益率/倍数=实际振幅", "预估收益率", "实际走势收益率"}:
        return "space_or_move"
    if field_name in {"收益率", "收益率/单数", "收益 (USDT)"}:
        return "return_or_pnl"
    if field_name == "交易时间差（分钟）":
        return "hold_time"
    if field_name in {"交易对", "列B"}:
        return "symbol_reference"
    if field_name in {"买入时间", "卖出时间"}:
        return "time_reference"
    if field_name == "方向":
        return "side_reference"
    if sheet_name in LARGE_MOVE_SHEETS:
        return "large_move_membership"
    if role == "trade_index":
        return "trade_index_field"
    return "sheet_field"


def _event_weight(semantic: str, role: str) -> float:
    if semantic == "manual_review_text":
        return 1.5
    if semantic in {"entry_position", "btc_cycle", "space_or_move"}:
        return 1.2
    if semantic in {"return_or_pnl", "hold_time"}:
        return 1.0
    if role == "trade_index":
        return 0.8
    return 0.7


def _manual_fields() -> set[str]:
    return {
        "大饼大周期判断",
        "大饼周期标底（属于主要开仓位置的那种）",
        "山寨币周期判断",
        "预估收益率",
        "实际走势收益率",
        "预测属于开仓位置的那种",
        "实际属于开仓位置的那种",
        "复盘操作细节",
        "操作细节",
        "操作间节",
    }


def _coalesce_number(first: Any, second: Any) -> float | None:
    for value in (first, second):
        number = _to_float(value)
        if number is not None:
            return number
    return None


def _to_float(value: Any) -> float | None:
    if value in {"", None, "None"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _return_bucket(value: float | None) -> str:
    if value is None:
        return "not_labeled"
    if value <= -1.0:
        return "loss_gt_100pct"
    if value <= -0.2:
        return "loss_gt_20pct"
    if value < 0:
        return "loss_lt_20pct"
    if value >= 1.0:
        return "profit_gt_100pct"
    if value >= 0.2:
        return "profit_gt_20pct"
    return "profit_or_flat_lt_20pct"


def _space_bucket(value: float | None) -> str:
    if value is None:
        return "not_labeled"
    if abs(value) >= 0.2:
        return "move_gt_20pct"
    if abs(value) >= 0.05:
        return "move_gt_5pct"
    return "move_lt_5pct"


def _hold_bucket(value: float | None) -> str:
    if value is None:
        return "not_labeled"
    if value <= 5:
        return "hold_le_5m"
    if value <= 10:
        return "hold_6_10m"
    if value <= 15:
        return "hold_11_15m"
    if value <= 30:
        return "hold_16_30m"
    if value >= 1440:
        return "hold_gt_1d"
    return "hold_31m_1d"


def _parse_time_ms(value: str) -> int:
    raw = str(value or "").strip()
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


def _column_index(ref: str) -> int:
    match = re.match(r"([A-Z]+)", ref)
    if not match:
        return 0
    value = 0
    for char in match.group(1):
        value = value * 26 + ord(char) - 64
    return value


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    if not fields:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _report(digest: dict[str, Any], windows: list[dict[str, Any]], coverage_rows: list[dict[str, Any]] | None = None) -> str:
    roles: dict[str, int] = {}
    for row in digest["sheet_inventory"]:
        roles[row["sheet_role"]] = roles.get(row["sheet_role"], 0) + 1
    profiles: dict[str, int] = {}
    for row in windows:
        profiles[row["window_profile"]] = profiles.get(row["window_profile"], 0) + 1
    coverage_profiles: dict[str, int] = {}
    for row in coverage_rows or []:
        coverage_profiles[row["coverage_status"]] = coverage_profiles.get(row["coverage_status"], 0) + 1
    return "\n".join(
        [
            "# LangLang Excel Workbook Digest",
            "",
            f"- generated_at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            f"- workbook_count: {digest['workbook_count']}",
            f"- sheet_count: {digest['sheet_count']}",
            f"- unknown_sheet_count: {digest['unknown_sheet_count']}",
            f"- trade_rows: {len(digest['trade_matrix'])}",
            f"- required_kline_windows: {len(windows)}",
            f"- kline_window_coverage_rows: {len(coverage_rows or [])}",
            f"- sheet_roles: {json.dumps(roles, ensure_ascii=False, sort_keys=True)}",
            f"- window_profiles: {json.dumps(profiles, ensure_ascii=False, sort_keys=True)}",
            f"- coverage_statuses: {json.dumps(coverage_profiles, ensure_ascii=False, sort_keys=True)}",
            "- unknown_trade_labels: 0",
            "",
            "## Rule",
            "- 所有 sheet 都作为同等证据源；按交易序号/时间/币种归并到同一笔交易，避免同一交易在多个 sheet 中重复计数。",
            "- 交易事实、止损分布、空间分析、持仓时间、币种收益、人工复盘和原始异常记录共同进入蒸馏证据层。",
            "- 大波动或人工复盘样本使用 expanded_manual_review 窗口，普通交易使用 standard 窗口。",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
