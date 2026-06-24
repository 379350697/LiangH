import json
import os
import tempfile
import unittest

from langlang_trader.data_coverage import MarketDataCoverageLedger
from langlang_trader.historical_patterns import (
    HistoricalPatternMatcher,
    build_historical_patterns,
    read_historical_patterns,
    write_historical_patterns,
)
from langlang_trader.strategy_source import DEFAULT_CONFIRMED_SOURCE, StrategySourceBuilder
from langlang_trader.v1_1_artifacts import _completion_report


class LangLangV1_1SourceTest(unittest.TestCase):
    def test_strategy_source_outputs_no_unmapped_pdf_concepts(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = StrategySourceBuilder(pdf_path="/tmp/missing.pdf").build(tmp)

            self.assertEqual(result["unknown_concepts"], 0)
            self.assertTrue(os.path.exists(os.path.join(tmp, "strategy_text.md")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "strategy_sections.json")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "visual_qa.md")))
            with open(os.path.join(tmp, "strategy_sections.json"), encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["source_integrity"]["source_status"], "pdf_missing_needs_source_confirmation")
            self.assertTrue(payload["source_integrity"]["requires_human_confirmation"])
            self.assertEqual(
                {section["section"] for section in payload["sections"]},
                {"前言", "市场理解", "交易系统", "交易纪律", "交易心法"},
            )
            statuses = {concept["status"] for concept in payload["concepts"]}
            self.assertNotIn("unmapped", statuses)
            concept_ids = {concept["concept_id"] for concept in payload["concepts"]}
            self.assertIn("entry_second_pressure_retest", concept_ids)
            self.assertIn("risk_w_unit_positioning", concept_ids)

    def test_strategy_source_marks_image_pdf_as_needing_pdf_craft_or_human_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = os.path.join(tmp, "image_style.pdf")
            with open(pdf_path, "wb") as handle:
                handle.write(
                    b"%PDF-1.4\n"
                    b"1 0 obj<<>>endobj\n"
                    b"2 0 obj<< /Type /Catalog /Pages 3 0 R >>endobj\n"
                    b"3 0 obj<< /Type /Pages /Kids [4 0 R] /Count 1 >>endobj\n"
                    b"4 0 obj<< /Type /Page /Parent 3 0 R /MediaBox [0 0 100 100] >>endobj\n"
                    b"xref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000030 00000 n \n"
                    b"0000000079 00000 n \n0000000136 00000 n \ntrailer<< /Root 2 0 R /Size 5 >>\n"
                    b"startxref\n203\n%%EOF\n"
                )

            result = StrategySourceBuilder(pdf_path=pdf_path).build(tmp)

            integrity = result["source_integrity"]
            self.assertEqual(integrity["file_exists"], True)
            self.assertEqual(integrity["extracted_char_count"], 0)
            self.assertIn(
                integrity["source_status"],
                {"image_pdf_needs_ocr_or_pdf_craft_confirmation", "pdf_read_error_needs_source_confirmation"},
            )
            self.assertTrue(integrity["requires_human_confirmation"])
            with open(os.path.join(tmp, "visual_qa.md"), encoding="utf-8") as handle:
                visual_qa = handle.read()
            self.assertIn("requires_human_confirmation: true", visual_qa)

    def test_completion_report_does_not_claim_source_complete_before_pdf_confirmation(self):
        pdf_payload = {
            "sections": [{"section": "前言"}],
            "concepts": [{"concept_id": "regime_main_wave", "status": "implemented"}],
            "unknown_concepts": 0,
            "source_integrity": {
                "source_status": "image_pdf_needs_ocr_or_pdf_craft_confirmation",
                "requires_human_confirmation": True,
                "extracted_char_count": 0,
            },
        }

        report = _completion_report(
            pdf_payload=pdf_payload,
            trades=[{"trade_id": "1"}],
            coverage_rows=[],
            patterns=[],
            selection_rows=[],
        )

        self.assertIn("pdf_source_status: image_pdf_needs_ocr_or_pdf_craft_confirmation", report)
        self.assertIn("pdf_requires_human_confirmation: True", report)
        self.assertIn("v1.1 artifact statuses are explicit, but PDF source still needs confirmation", report)

    def test_confirmed_pdf_source_closes_human_confirmation_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            confirmed = os.path.join(tmp, "confirmed.md")
            with open(confirmed, "w", encoding="utf-8") as handle:
                handle.write("# 前言\n\n用户确认原文\n\n## 市场理解\n\n龙头山寨和大饼共振\n")

            result = StrategySourceBuilder(pdf_path="/tmp/missing.pdf", confirmed_text_path=confirmed).build(tmp)

            integrity = result["source_integrity"]
            self.assertEqual(integrity["source_status"], "user_confirmed_pdf_text")
            self.assertEqual(integrity["extraction_method"], "user_confirmed_pdf_text")
            self.assertFalse(integrity["requires_human_confirmation"])
            self.assertEqual(integrity["confirmed_text_path"], confirmed)
            with open(os.path.join(tmp, "strategy_text.md"), encoding="utf-8") as handle:
                strategy_text = handle.read()
            self.assertIn("## User Confirmed PDF Text", strategy_text)
            self.assertIn("龙头山寨和大饼共振", strategy_text)
            self.assertTrue(os.path.exists(os.path.join(tmp, "source_images", "full_pdf_visual_reference.png")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "source_images", "wave_entry_notes.png")))

    def test_packaged_confirmed_source_contains_core_langlang_pdf_concepts(self):
        text = DEFAULT_CONFIRMED_SOURCE.read_text(encoding="utf-8")

        for phrase in [
            "实盘共盈利 5500 倍",
            "龙头山寨在大饼调整结束后再次拉升时",
            "主要包含六个开仓位置",
            "主升浪不做任何一笔空单",
            "心态波动时，及时调整",
        ]:
            self.assertIn(phrase, text)

    def test_market_coverage_ledger_has_no_unknown_rows(self):
        trades = [
            {"trade_id": "1", "symbol": "BTC-USDT-SWAP", "entry_ts": 1_700_000_000_000},
            {"trade_id": "2", "symbol": "MISSING-USDT-SWAP", "entry_ts": 1_700_086_400_000},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "1D"))
            with open(os.path.join(tmp, "1D", "BTC-USDT-SWAP_cache.csv"), "w", encoding="utf-8") as handle:
                handle.write("ts,open,high,low,close,vol\n1699913600000,1,2,1,2,10\n")

            rows = MarketDataCoverageLedger(tmp).build(trades, bars=["1D", "1H"])

            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["coverage_status"] != "unknown" for row in rows))
            missing = [row for row in rows if row["symbol"] == "MISSING-USDT-SWAP"]
            self.assertTrue(all(row["unavailable_reason"] for row in missing))

    def test_historical_patterns_match_same_side_setup_and_penalize_big_loss(self):
        trades = [
            {
                "trade_id": "win",
                "symbol": "BTC-USDT-SWAP",
                "side": "long",
                "pnl_usdt": 1000,
                "return_rate": 0.8,
                "hold_minutes": 900,
                "regime": "first_divergence",
                "setup": "small_divergence_entry",
                "ret_20d": 0.35,
                "pos_20d": 0.68,
            },
            {
                "trade_id": "loss",
                "symbol": "BTC-USDT-SWAP",
                "side": "long",
                "pnl_usdt": -800,
                "return_rate": -0.35,
                "hold_minutes": 15,
                "regime": "top_divergence",
                "setup": "first_breakout",
                "ret_20d": 0.80,
                "pos_20d": 0.98,
            },
        ]

        patterns = build_historical_patterns(trades)
        match = HistoricalPatternMatcher(patterns).match(
            side="long",
            regime="first_divergence",
            setup="small_divergence_entry",
            features={"ret_20d": 0.33, "pos_20d": 0.66},
        )

        self.assertEqual(match.examples[0]["trade_id"], "win")
        self.assertGreater(match.score, 0.5)
        self.assertEqual(match.big_loss_overlap_count, 0)

    def test_historical_pattern_round_trip_preserves_features_for_fleet_matcher(self):
        trades = [
            {
                "trade_id": "win",
                "symbol": "BTC-USDT-SWAP",
                "side": "long",
                "pnl_usdt": 1000,
                "return_rate": 0.8,
                "hold_minutes": 900,
                "regime": "first_divergence",
                "setup": "small_divergence_entry",
                "ret_20d": 0.35,
                "ret_60d": 0.80,
                "pos_20d": 0.68,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "historical_patterns.csv")
            write_historical_patterns(path, build_historical_patterns(trades))

            loaded = read_historical_patterns(path)
            match = HistoricalPatternMatcher(loaded).match(
                side="long",
                regime="first_divergence",
                setup="small_divergence_entry",
                features={"ret_20d": 0.34, "ret_60d": 0.78, "pos_20d": 0.67},
            )

        self.assertIsInstance(loaded[0]["features"], dict)
        self.assertIn("ret_20d", loaded[0]["features"])
        self.assertEqual(match.examples[0]["trade_id"], "win")
        self.assertGreater(match.score, 0.5)


if __name__ == "__main__":
    unittest.main()
