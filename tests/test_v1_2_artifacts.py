import unittest

from langlang_trader.symbol_selection import SymbolSelectionResult
from langlang_trader.universe import UniverseSnapshot
from langlang_trader.v1_2_artifacts import _context_row, _report


class V12ArtifactReportTest(unittest.TestCase):
    def test_report_separates_universe_size_from_ranked_feature_coverage(self):
        boards = {
            "long_main_wave": [
                SymbolSelectionResult(
                    symbol="WAVE-USDT-SWAP",
                    selected=True,
                    selection_rank=1,
                    selection_score=0.8,
                    selection_bias="long",
                    reason_codes=["main_wave_acceleration"],
                    features={},
                    selection_mode="long_main_wave",
                )
            ],
            "short_waterfall": [],
        }
        universe = UniverseSnapshot(
            mode="okx_binance_usdt_swap_observe",
            generated_at="2024-03-12T00:00:00+00:00",
            symbols=["WAVE-USDT-SWAP"],
            reference_symbols=["BTC-USDT-SWAP", "ETH-USDT-SWAP"],
            rows=[],
            raw_payload={},
            observed_symbols=["BTC-USDT-SWAP", "ETH-USDT-SWAP", "WAVE-USDT-SWAP", "MISSING-USDT-SWAP"],
        )

        report = _report([], boards, universe)

        self.assertIn("ranking_feature_symbols: 1", report)
        self.assertIn("observed_symbols_without_ranking_features: 1", report)
        self.assertIn("ranking_feature_coverage_status: partial_feature_coverage", report)

    def test_context_row_uses_terminal_status_for_missing_daily_snapshot(self):
        row = _context_row(
            {"trade_id": "1", "symbol": "MISS-USDT-SWAP", "side": "long", "entry_time": "2024-01-01"},
            result=None,
            long_result=None,
            short_result=None,
            day_key=1_700_000_000_000,
        )

        self.assertEqual(row["data_status"], "exchange_unavailable")
        self.assertEqual(row["reason_codes"], "selection_data_unavailable_with_evidence")
        self.assertEqual(row["unavailable_reason"], "no_completed_daily_snapshot_before:1700000000000")


if __name__ == "__main__":
    unittest.main()
