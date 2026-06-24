import unittest
import csv
import tempfile
from pathlib import Path

from langlang_trader.symbol_selection import SymbolSelectionResult
from langlang_trader.universe import UniverseSnapshot
from langlang_trader.v1_3_artifacts import _report_v1_3, _trade_feature_matrix, build_v1_3_artifacts


DAY_MS = 86_400_000


class V13ArtifactReportTest(unittest.TestCase):
    def test_report_names_final_strategy_source_and_complete_feature_status(self):
        boards = {
            "long_main_wave": [
                SymbolSelectionResult(
                    symbol="LEADER-USDT-SWAP",
                    selected=True,
                    selection_rank=1,
                    selection_score=0.8,
                    selection_bias="long",
                    reason_codes=["leader_altcoin"],
                    features={"symbol_selection_tag": "leader_altcoin"},
                    selection_mode="long_main_wave",
                )
            ],
            "short_waterfall": [],
        }
        universe = UniverseSnapshot(
            mode="okx_binance_usdt_swap_observe",
            generated_at="2024-03-12T00:00:00+00:00",
            symbols=["LEADER-USDT-SWAP"],
            reference_symbols=["BTC-USDT-SWAP", "ETH-USDT-SWAP"],
            rows=[],
            raw_payload={},
            observed_symbols=["BTC-USDT-SWAP", "ETH-USDT-SWAP", "LEADER-USDT-SWAP"],
        )

        report = _report_v1_3([], boards, universe)

        self.assertIn("strategy_version: rules_langlang_v1_3", report)
        self.assertIn("pdf_source_status: user_confirmed_pdf_text", report)
        self.assertIn("execution_order: all_market_selection -> market_season", report)
        self.assertIn("ranking_feature_coverage_status: complete_feature_coverage", report)
        self.assertIn("unknown_signal_explanations: 0", report)

    def test_report_lists_terminal_ranking_exclusions_without_partial_status(self):
        boards = {"long_main_wave": [], "short_waterfall": []}
        universe = UniverseSnapshot(
            mode="okx_binance_usdt_swap_observe",
            generated_at="2024-03-12T00:00:00+00:00",
            symbols=["MISSING-USDT-SWAP"],
            reference_symbols=["BTC-USDT-SWAP", "ETH-USDT-SWAP"],
            rows=[],
            raw_payload={},
            observed_symbols=["BTC-USDT-SWAP", "ETH-USDT-SWAP", "MISSING-USDT-SWAP"],
        )

        report = _report_v1_3([], boards, universe)

        self.assertIn("ranking_feature_coverage_status: complete_with_terminal_exclusions", report)
        self.assertIn("ranking_unavailable_symbols:", report)
        self.assertIn("MISSING-USDT-SWAP: ranking_feature_unavailable_with_terminal_evidence", report)
        self.assertNotIn("partial_feature_coverage", report)

    def test_build_artifacts_writes_market_feature_layer_and_trade_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "kline_cache"
            trades = root / "standard_trades.csv"
            out = root / "out"
            _write_daily_cache(cache, "BTC-USDT-SWAP", 70, ret_step=0.001)
            _write_daily_cache(cache, "ETH-USDT-SWAP", 70, ret_step=0.001)
            _write_daily_cache(cache, "LEADER-USDT-SWAP", 70, ret_step=0.006)
            trades.write_text(
                "trade_id,symbol,side,entry_time,pnl_usdt,return_rate\n"
                "1,LEADER-USDT-SWAP,long,2024-03-10T00:00:00+00:00,10,0.1\n",
                encoding="utf-8",
            )

            result = build_v1_3_artifacts(trades_csv=trades, kline_cache=cache, out_dir=out)

            self.assertIn("market_features", result)
            self.assertTrue((out / "market_features" / "technical_features_1d.csv").exists())
            self.assertTrue((out / "market_features" / "derivatives_features_1d.csv").exists())
            self.assertTrue((out / "market_features" / "external_market_features_1d.csv").exists())
            self.assertTrue((out / "market_features" / "feature_coverage.csv").exists())
            self.assertTrue((out / "data_unification_audit.md").exists())
            matrix_path = out / "market_features" / "trade_feature_matrix.csv"
            self.assertTrue(matrix_path.exists())
            with matrix_path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["symbol"], "LEADER-USDT-SWAP")
            self.assertIn("turnover_usdt", rows[0])
            self.assertIn("market_cap_status", rows[0])
            self.assertIn("technical_data_status", rows[0])
            self.assertIn("external_data_status", rows[0])

            audit = (out / "data_unification_audit.md").read_text(encoding="utf-8")
            self.assertIn("blank_trade_feature_status_cells: 0", audit)
            self.assertIn("ranking_feature_coverage_status: complete_feature_coverage", audit)

    def test_trade_feature_matrix_fills_terminal_statuses_when_feature_row_missing(self):
        matrix = _trade_feature_matrix(
            [
                {
                    "trade_id": "1",
                    "symbol": "DELISTED-USDT-SWAP",
                    "side": "long",
                    "entry_time": "2022-06-16T00:00:00+00:00",
                    "entry_ts": 1_650_000_000_000,
                }
            ],
            market_feature_rows=[[]],
        )

        self.assertEqual(matrix[0]["feature_data_status"], "exchange_unavailable")
        self.assertEqual(matrix[0]["technical_data_status"], "exchange_unavailable")
        self.assertEqual(matrix[0]["derivatives_data_status"], "exchange_unavailable")
        self.assertEqual(matrix[0]["external_data_status"], "provider_limited")
        self.assertEqual(matrix[0]["funding_rate_status"], "exchange_unavailable")
        self.assertEqual(matrix[0]["open_interest_status"], "exchange_unavailable")
        self.assertEqual(matrix[0]["market_cap_status"], "provider_limited")


def _write_daily_cache(cache: Path, symbol: str, count: int, *, ret_step: float) -> None:
    daily = cache / "1D"
    daily.mkdir(parents=True, exist_ok=True)
    path = daily / f"{symbol}_merged.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ts", "open", "high", "low", "close", "vol", "vol_ccy", "vol_quote", "confirm"],
        )
        writer.writeheader()
        start_ts = 1_700_000_000_000
        for idx in range(count):
            close = 100 * (1 + idx * ret_step)
            writer.writerow(
                {
                    "ts": start_ts + idx * DAY_MS,
                    "open": close * 0.99,
                    "high": close * 1.02,
                    "low": close * 0.98,
                    "close": close,
                    "vol": 1000 + idx,
                    "vol_ccy": "",
                    "vol_quote": close * (1000 + idx),
                    "confirm": "1",
                }
            )


if __name__ == "__main__":
    unittest.main()
