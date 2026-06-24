import csv
import tempfile
import unittest
import zipfile
from html import escape
from pathlib import Path

from langlang_trader.excel_workbook_digest import (
    build_excel_digest_artifacts,
    build_excel_evidence_events,
    build_kline_window_coverage,
    build_required_kline_windows_from_digest,
    digest_workbooks,
)
from langlang_trader.v1_3_artifacts import build_v1_3_artifacts


class ExcelWorkbookDigestTest(unittest.TestCase):
    def test_digest_classifies_all_sheets_and_merges_manual_review_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workbook = root / "langlang.xlsx"
            trades = root / "standard_trades.csv"
            _write_workbook(workbook)
            trades.write_text(
                "trade_id,symbol,side,entry_time,exit_time,pnl_usdt,return_rate,hold_minutes,realized_move\n"
                "1,BTC-USDT-SWAP,long,2022-07-03T14:03:00+00:00,2022-07-03T16:00:00+00:00,1005,2.9042,117,0.14521\n"
                "2,APT-USDT-SWAP,long,2023-01-01T00:00:00+00:00,2023-01-02T00:00:00+00:00,24900,9.1755,1440,0.6117\n",
                encoding="utf-8",
            )

            digest = digest_workbooks([workbook], trades)

            self.assertEqual(digest["workbook_count"], 1)
            self.assertEqual(digest["sheet_count"], 9)
            by_sheet = {row["sheet_name"]: row for row in digest["sheet_inventory"]}
            self.assertEqual(by_sheet["时间排列+去除金额错误单子"]["sheet_role"], "trade_index")
            self.assertEqual(by_sheet["初始版"]["sheet_role"], "raw_source")
            self.assertEqual(by_sheet["5%波动以上单子"]["sheet_role"], "manual_review")
            self.assertEqual(by_sheet["不同币做的收益率"]["sheet_role"], "derived_stats")
            self.assertEqual(by_sheet["个人交割单分析"]["sheet_role"], "empty_or_formula_only")
            self.assertEqual(digest["unknown_sheet_count"], 0)

            by_trade = {row["trade_id"]: row for row in digest["trade_matrix"]}
            self.assertIn("5%波动以上单子", by_trade["1"]["sheet_membership"])
            self.assertEqual(by_trade["1"]["large_move_label"], "large_move_manual_review")
            self.assertIn("大跌后的反弹", by_trade["1"]["btc_cycle_label"])
            self.assertEqual(by_trade["1"]["entry_position_label"], "6")
            self.assertIn("箱体震荡", by_trade["1"]["manual_review_text"])
            self.assertEqual(by_trade["2"]["sheet_membership"], "not_in_analysis_sheet")
            self.assertEqual(by_trade["2"]["stop_loss_bucket"], "profit_gt_100pct")
            self.assertEqual(by_trade["2"]["space_bucket"], "move_gt_20pct")
            self.assertEqual(by_trade["2"]["hold_time_bucket"], "hold_gt_1d")

    def test_required_kline_windows_expand_large_move_and_manual_review_trades(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workbook = root / "langlang.xlsx"
            trades = root / "standard_trades.csv"
            _write_workbook(workbook)
            trades.write_text(
                "trade_id,symbol,side,entry_time,exit_time,pnl_usdt,return_rate,hold_minutes,realized_move\n"
                "1,BTC-USDT-SWAP,long,2022-07-03T14:03:00+00:00,2022-07-03T16:00:00+00:00,1005,2.9042,117,0.14521\n"
                "2,APT-USDT-SWAP,long,2023-01-01T00:00:00+00:00,2023-01-01T01:00:00+00:00,10,0.02,60,0.0013\n",
                encoding="utf-8",
            )
            digest = digest_workbooks([workbook], trades)

            windows = build_required_kline_windows_from_digest(digest["trade_matrix"])

            trade1_daily = next(row for row in windows if row["trade_id"] == "1" and row["timeframe"] == "1D")
            trade2_daily = next(row for row in windows if row["trade_id"] == "2" and row["timeframe"] == "1D")
            self.assertEqual(trade1_daily["window_profile"], "expanded_manual_review")
            self.assertEqual(trade2_daily["window_profile"], "standard")
            self.assertGreater(int(trade1_daily["before_ms"]), int(trade2_daily["before_ms"]))
            self.assertIn("BTC-USDT-SWAP", {row["symbol"] for row in windows if row["trade_id"] == "market_reference"})
            self.assertIn("ETH-USDT-SWAP", {row["symbol"] for row in windows if row["trade_id"] == "market_reference"})

    def test_build_artifacts_writes_inventory_matrix_and_window_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workbook = root / "langlang.xlsx"
            trades = root / "standard_trades.csv"
            out = root / "out"
            cache = root / "kline_cache"
            _write_workbook(workbook)
            _write_daily_cache(cache, "BTC-USDT-SWAP")
            trades.write_text(
                "trade_id,symbol,side,entry_time,exit_time,pnl_usdt,return_rate,hold_minutes,realized_move\n"
                "1,BTC-USDT-SWAP,long,2022-07-03T14:03:00+00:00,2022-07-03T16:00:00+00:00,1005,2.9042,117,0.14521\n",
                encoding="utf-8",
            )

            result = build_excel_digest_artifacts(workbooks=[workbook], trades_csv=trades, out_dir=out, kline_cache=cache)

            self.assertEqual(result["unknown_sheet_count"], 0)
            for name in [
                "excel_workbook_inventory.csv",
                "trade_sheet_label_matrix.csv",
                "required_kline_windows.csv",
                "kline_window_coverage.csv",
                "excel_workbook_digest.md",
            ]:
                self.assertTrue((out / name).exists(), name)
            with (out / "required_kline_windows.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertIn("window_profile", rows[0])
            with (out / "kline_window_coverage.csv").open(encoding="utf-8") as handle:
                coverage_rows = list(csv.DictReader(handle))
            self.assertIn("available", {row["coverage_status"] for row in coverage_rows})
            self.assertIn("insufficient_history", {row["coverage_status"] for row in coverage_rows})
            report = (out / "excel_workbook_digest.md").read_text(encoding="utf-8")
            self.assertIn("所有 sheet 都作为同等证据源", report)
            self.assertNotIn("其它 sheet 只作为", report)

    def test_builds_row_and_field_level_excel_evidence_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workbook = root / "langlang.xlsx"
            trades = root / "standard_trades.csv"
            out = root / "out"
            _write_workbook(workbook)
            trades.write_text(
                "trade_id,symbol,side,entry_time,exit_time,pnl_usdt,return_rate,hold_minutes,realized_move\n"
                "1,BTC-USDT-SWAP,long,2022-07-03T14:03:00+00:00,2022-07-03T16:00:00+00:00,1005,2.9042,117,0.14521\n"
                "2,APT-USDT-SWAP,long,2023-01-01T00:00:00+00:00,2023-01-02T00:00:00+00:00,24900,9.1755,1440,0.6117\n",
                encoding="utf-8",
            )

            digest = digest_workbooks([workbook], trades)
            events = build_excel_evidence_events(digest["sheet_data"], digest["trade_matrix"])

            self.assertGreater(len(events), len(digest["trade_matrix"]))
            by_role = {event["evidence_role"] for event in events}
            self.assertIn("trade_index", by_role)
            self.assertIn("manual_review", by_role)
            self.assertIn("derived_stats", by_role)
            manual_events = [
                event for event in events
                if event["trade_id"] == "1" and event["field_name"] == "复盘操作细节"
            ]
            self.assertTrue(manual_events)
            self.assertEqual(manual_events[0]["event_semantic"], "manual_review_text")
            self.assertIn("箱体震荡", manual_events[0]["field_value"])
            derived_events = [
                event for event in events
                if event["trade_id"] == "2" and event["event_semantic"] == "space_or_move"
            ]
            self.assertTrue(derived_events)

            result = build_excel_digest_artifacts(workbooks=[workbook], trades_csv=trades, out_dir=out)

            self.assertTrue((out / "excel_evidence_event_dataset.csv").exists())
            with (out / "excel_evidence_event_dataset.csv").open(encoding="utf-8") as handle:
                event_rows = list(csv.DictReader(handle))
            self.assertEqual(len(event_rows), len(events))
            self.assertIn("event_semantic", event_rows[0])
            self.assertIn("source_row_number", event_rows[0])

    def test_kline_coverage_uses_terminal_backfill_evidence_for_unavailable_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            terminal = root / "terminal.csv"
            terminal.write_text(
                "symbol,bar,start_ms,end_ms,coverage_status,unavailable_reason,evidence\n"
                "GONE-USDT-SWAP,1D,1000,200000,instrument_unavailable,invalid_symbol,okx:51001;binance:-1121\n",
                encoding="utf-8",
            )
            required = [
                {
                    "trade_id": "1",
                    "symbol": "GONE-USDT-SWAP",
                    "timeframe": "1D",
                    "window_profile": "standard",
                    "start_ms": 2_000,
                    "end_ms": 100_000,
                    "coverage_status": "required",
                    "coverage_reason": "standard_trade_window",
                    "source": "excel_digest",
                }
            ]

            rows = build_kline_window_coverage(required, root / "empty_cache", terminal_coverage_csv=terminal)

            self.assertEqual(rows[0]["coverage_status"], "instrument_unavailable")
            self.assertEqual(rows[0]["unavailable_reason"], "invalid_symbol")
            self.assertIn("okx:51001", rows[0]["evidence"])

    def test_kline_coverage_reads_ranged_symbol_cache_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "kline_cache"
            daily = cache / "1D"
            daily.mkdir(parents=True)
            ranged = daily / "ETC-USDT-SWAP_1000_2000.csv"
            with ranged.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["ts", "open", "high", "low", "close", "vol"])
                writer.writeheader()
                writer.writerow({"ts": "1000", "open": "1", "high": "1", "low": "1", "close": "1", "vol": "1"})
                writer.writerow({"ts": str(1000 + 86_400_000), "open": "1", "high": "1", "low": "1", "close": "1", "vol": "1"})
            required = [
                {
                    "trade_id": "1",
                    "symbol": "ETC-USDT-SWAP",
                    "timeframe": "1D",
                    "window_profile": "standard",
                    "start_ms": 1000,
                    "end_ms": 1000 + 86_400_000,
                    "coverage_status": "required",
                    "coverage_reason": "standard_trade_window",
                    "source": "excel_digest",
                }
            ]

            rows = build_kline_window_coverage(required, cache)

            self.assertEqual(rows[0]["coverage_status"], "available")

    def test_v1_3_artifacts_can_include_excel_digest_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workbook = root / "langlang.xlsx"
            trades = root / "standard_trades.csv"
            cache = root / "kline_cache"
            out = root / "out"
            _write_workbook(workbook)
            trades.write_text(
                "trade_id,symbol,side,entry_time,exit_time,pnl_usdt,return_rate,hold_minutes,realized_move\n"
                "1,BTC-USDT-SWAP,long,2022-07-03T14:03:00+00:00,2022-07-03T16:00:00+00:00,1005,2.9042,117,0.14521\n",
                encoding="utf-8",
            )
            _write_daily_cache(cache, "BTC-USDT-SWAP")
            _write_daily_cache(cache, "ETH-USDT-SWAP")

            result = build_v1_3_artifacts(
                trades_csv=trades,
                kline_cache=cache,
                out_dir=out,
                excel_workbooks=[workbook],
            )

            self.assertIn("excel_digest", result)
            self.assertTrue((out / "excel_digest" / "excel_workbook_inventory.csv").exists())
            self.assertTrue((out / "excel_digest" / "required_kline_windows.csv").exists())


def _write_workbook(path: Path) -> None:
    sheets = [
        ("初始版", [["", "交易对", "方向", "杠杆倍数", "开仓均价", "平仓均价", "收益率", "收益 (USDT)", "买入时间"], ["5.1-20", "BTC-USDT-SWAP", "多", 20, 1, 2, 0.1, 10, "2022-07-03 22:03:00"]]),
        ("时间排列+去除金额错误单子", [["序号", "买入时间", "卖出时间", "交易对", "方向", "收益率", "收益 (USDT)", "交易时间差（分钟）", "收益率/倍数=实际振幅"], [1, "2022-07-03 22:03:00", "2022-07-04 00:00:00", "BTC-USDT-SWAP", "多", 2.9042, 1005, 117, 0.14521]]),
        ("5%波动以上的单子", [["序号", "交易对", "方向", "收益率", "收益 (USDT)", "买入时间", "大饼情况", "交易时间差（分钟）", "收益率/倍数=实际振幅"], [1, "BTC-USDT-SWAP", "多", 2.9042, 1005, "2022-07-03 22:03:00", "大跌后的反弹", 117, 0.14521]]),
        ("单笔收益率（看止损）", [["序号", "交易对", "方向", "收益率", "收益 (USDT)", "买入时间"], [2, "APT-USDT-SWAP", "多", 9.1755, 24900, "2023-01-01 08:00:00"]]),
        ("实际振幅与收益分析（看空间）", [["序号", "交易对", "方向", "收益率/倍数=实际振幅", "收益率"], [2, "APT-USDT-SWAP", "多", 0.6117, 9.1755]]),
        ("所有单子持单时间（分钟）", [["序号", "交易对", "方向", "交易时间差（分钟）"], [2, "APT-USDT-SWAP", "多", 1440]]),
        ("不同币做的收益率", [["列B", "计数", "占比", "收益率", "收益率/单数"], ["APT-USDT-SWAP", 1, 0.5, 9.1755, 9.1755]]),
        ("5%波动以上单子", [["序号", "交易对", "方向", "收益率", "收益 (USDT)", "买入时间", "大饼大周期判断", "大饼周期标底（属于主要开仓位置的那种）", "复盘操作细节", "操作细节"], [1, "BTC-USDT-SWAP", "多", 2.9042, 1005, "2022-07-03 22:03:00", "大跌后的反弹", 6, "箱体震荡末期企稳做多", "倍数太高"]]),
        ("个人交割单分析", [["序号", "交易对", "方向", "收益率", "大饼大周期判断", "复盘操作细节"]]),
    ]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _content_types(len(sheets)))
        zf.writestr("_rels/.rels", _root_rels())
        zf.writestr("xl/workbook.xml", _workbook_xml([name for name, _ in sheets]))
        zf.writestr("xl/_rels/workbook.xml.rels", _workbook_rels(len(sheets)))
        zf.writestr("xl/styles.xml", '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"/>')
        for idx, (_, rows) in enumerate(sheets, start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", _sheet_xml(rows))


def _content_types(sheet_count: int) -> str:
    overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for idx in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        f"{overrides}</Types>"
    )


def _root_rels() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )


def _workbook_xml(sheet_names: list[str]) -> str:
    sheets = "".join(
        f'<sheet name="{escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>'
        for idx, name in enumerate(sheet_names, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{sheets}</sheets></workbook>"
    )


def _workbook_rels(sheet_count: int) -> str:
    rels = "".join(
        f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>'
        for idx in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{rels}</Relationships>"
    )


def _sheet_xml(rows: list[list[object]]) -> str:
    row_xml = []
    for ridx, row in enumerate(rows, start=1):
        cells = []
        for cidx, value in enumerate(row, start=1):
            ref = f"{_col(cidx)}{ridx}"
            if isinstance(value, (int, float)):
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>')
        row_xml.append(f'<row r="{ridx}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(row_xml)}</sheetData></worksheet>'
    )


def _col(index: int) -> str:
    chars = []
    while index:
        index, rem = divmod(index - 1, 26)
        chars.append(chr(65 + rem))
    return "".join(reversed(chars))


def _write_daily_cache(cache: Path, symbol: str) -> None:
    daily = cache / "1D"
    daily.mkdir(parents=True, exist_ok=True)
    path = daily / f"{symbol}_merged.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ts", "open", "high", "low", "close", "vol", "vol_ccy", "vol_quote", "confirm"],
        )
        writer.writeheader()
        start_ts = 1_609_459_200_000
        for idx in range(800):
            close = 100 + idx
            writer.writerow(
                {
                    "ts": start_ts + idx * 86_400_000,
                    "open": close - 1,
                    "high": close + 2,
                    "low": close - 2,
                    "close": close,
                    "vol": 1000,
                    "vol_ccy": "",
                    "vol_quote": close * 1000,
                    "confirm": "1",
                }
            )


if __name__ == "__main__":
    unittest.main()
