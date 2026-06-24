import unittest

from langlang_trader.config import SymbolSelectionConfig
from langlang_trader.features import FeatureSnapshot
from langlang_trader.symbol_selection import SelectionEngine


def snapshot(symbol, **features):
    base = {
        "ret_3d": 0.0,
        "ret_7d": 0.0,
        "ret_20d": 0.0,
        "ret_60d": 0.0,
        "pos_20d": 0.5,
        "pullback_from_20d_high": -0.03,
        "vol_ratio_20d": 1.0,
        "latest_close": 100.0,
        "high_20d": 110.0,
        "high_60d": 120.0,
        "ma_5": 100.0,
        "ma_20": 99.0,
    }
    base.update(features)
    return FeatureSnapshot(
        symbol=symbol,
        bar="multi",
        last_ts=1_710_000_000_000,
        created_at="2024-03-12T00:00:00+00:00",
        features=base,
    )


class SelectionEngineV12Test(unittest.TestCase):
    def test_rank_all_market_builds_independent_long_and_short_boards(self):
        engine = SelectionEngine(
            SymbolSelectionConfig(enabled=True, style="dual_board", long_top_n=2, short_top_n=1)
        )
        snapshots = {
            "BTC-USDT-SWAP": snapshot("BTC-USDT-SWAP", ret_20d=0.10, ret_60d=0.22, ret_7d=0.03, pos_20d=0.62),
            "ETH-USDT-SWAP": snapshot("ETH-USDT-SWAP", ret_20d=0.06, ret_60d=0.18, ret_7d=0.02, pos_20d=0.56),
            "TURBO-USDT-SWAP": snapshot(
                "TURBO-USDT-SWAP",
                ret_3d=0.12,
                ret_7d=0.30,
                ret_20d=0.86,
                ret_60d=1.55,
                pos_20d=0.88,
                pullback_from_20d_high=-0.045,
                vol_ratio_20d=2.4,
            ),
            "WATER-USDT-SWAP": snapshot(
                "WATER-USDT-SWAP",
                ret_3d=-0.08,
                ret_7d=-0.24,
                ret_20d=-0.46,
                ret_60d=-0.72,
                pos_20d=0.08,
                pullback_from_20d_high=-0.34,
                vol_ratio_20d=2.0,
                ma_5=75.0,
                ma_20=90.0,
                latest_close=70.0,
            ),
            "CHASE-USDT-SWAP": snapshot(
                "CHASE-USDT-SWAP",
                ret_3d=0.26,
                ret_7d=0.80,
                ret_20d=2.40,
                ret_60d=4.20,
                pos_20d=0.99,
                pullback_from_20d_high=-0.002,
                vol_ratio_20d=3.0,
            ),
        }

        boards = engine.rank_all_market(snapshots)
        long_board = boards["long_main_wave"]
        short_board = boards["short_waterfall"]

        self.assertNotIn("BTC-USDT-SWAP", [row.symbol for row in long_board + short_board])
        self.assertEqual(long_board[0].symbol, "TURBO-USDT-SWAP")
        self.assertEqual(short_board[0].symbol, "WATER-USDT-SWAP")
        self.assertEqual(long_board[0].selection_mode, "long_main_wave")
        self.assertEqual(short_board[0].selection_mode, "short_waterfall")
        self.assertTrue(long_board[0].selected)
        self.assertTrue(short_board[0].selected)
        self.assertIn("main_wave_acceleration", long_board[0].reason_codes)
        self.assertIn("relative_to_btc_strength", long_board[0].reason_codes)
        self.assertIn("waterfall_breakdown", short_board[0].reason_codes)
        self.assertIn("relative_to_btc_weakness", short_board[0].reason_codes)

        chase = next(row for row in long_board if row.symbol == "CHASE-USDT-SWAP")
        self.assertIn("chase_overheat", chase.filter_codes)
        self.assertLess(chase.selection_score, long_board[0].selection_score)
        self.assertEqual(long_board[0].data_status, "available")
        self.assertIn("btc_ret_20d", long_board[0].market_env)

    def test_dual_board_only_selects_complete_directional_structures(self):
        engine = SelectionEngine(
            SymbolSelectionConfig(enabled=True, style="dual_board", long_top_n=2, short_top_n=2)
        )
        snapshots = {
            "BTC-USDT-SWAP": snapshot("BTC-USDT-SWAP", ret_20d=-0.20, ret_60d=-0.12, ret_7d=-0.04, pos_20d=0.35),
            "ETH-USDT-SWAP": snapshot("ETH-USDT-SWAP", ret_20d=-0.18, ret_60d=-0.10, ret_7d=-0.03, pos_20d=0.38),
            "REALWAVE-USDT-SWAP": snapshot(
                "REALWAVE-USDT-SWAP",
                ret_3d=0.07,
                ret_7d=0.16,
                ret_20d=0.42,
                ret_60d=0.88,
                pos_20d=0.86,
                pullback_from_20d_high=-0.05,
                vol_ratio_20d=1.6,
            ),
            "RELONLY-USDT-SWAP": snapshot(
                "RELONLY-USDT-SWAP",
                ret_3d=-0.03,
                ret_7d=-0.08,
                ret_20d=-0.05,
                ret_60d=0.40,
                pos_20d=0.34,
                pullback_from_20d_high=-0.24,
                vol_ratio_20d=0.8,
            ),
            "REALFALL-USDT-SWAP": snapshot(
                "REALFALL-USDT-SWAP",
                ret_3d=-0.09,
                ret_7d=-0.22,
                ret_20d=-0.38,
                ret_60d=-0.52,
                pos_20d=0.12,
                pullback_from_20d_high=-0.41,
                vol_ratio_20d=1.8,
                latest_close=60.0,
                ma_5=66.0,
                ma_20=80.0,
            ),
            "SOFTPULL-USDT-SWAP": snapshot(
                "SOFTPULL-USDT-SWAP",
                ret_3d=-0.04,
                ret_7d=-0.09,
                ret_20d=0.18,
                ret_60d=0.62,
                pos_20d=0.58,
                pullback_from_20d_high=-0.14,
                vol_ratio_20d=1.3,
            ),
        }

        boards = engine.rank_all_market(snapshots)
        long_by_symbol = {row.symbol: row for row in boards["long_main_wave"]}
        short_by_symbol = {row.symbol: row for row in boards["short_waterfall"]}

        self.assertTrue(long_by_symbol["REALWAVE-USDT-SWAP"].selected)
        self.assertFalse(long_by_symbol["RELONLY-USDT-SWAP"].selected)
        self.assertIn("relative_to_btc_strength", long_by_symbol["RELONLY-USDT-SWAP"].reason_codes)
        self.assertIn("incomplete_long_main_wave_structure", long_by_symbol["RELONLY-USDT-SWAP"].filter_codes)

        self.assertTrue(short_by_symbol["REALFALL-USDT-SWAP"].selected)
        self.assertFalse(short_by_symbol["SOFTPULL-USDT-SWAP"].selected)
        self.assertIn("incomplete_short_waterfall_structure", short_by_symbol["SOFTPULL-USDT-SWAP"].filter_codes)


if __name__ == "__main__":
    unittest.main()
