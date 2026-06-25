import unittest

from langlang_trader.features import DailyFeatureBuilder, MultiTimeframeFeatureBuilder
from langlang_trader.models import Candle


def candles(symbol, bar, count, start_price=100.0, step=0.01, start_ts=1_700_000_000_000, interval=60_000):
    rows = []
    for idx in range(count):
        close = start_price * (1 + idx * step)
        rows.append(
            Candle(
                symbol=symbol,
                bar=bar,
                ts=start_ts + idx * interval,
                open=close * 0.99,
                high=close * 1.02,
                low=close * 0.98,
                close=close,
                volume=1000 + idx * 10,
            )
        )
    return rows


class MultiTimeframeFeatureBuilderTest(unittest.TestCase):
    def test_daily_snapshot_includes_candlestick_body_shadow_and_volume_tags(self):
        symbol = "BIG-USDT-SWAP"
        rows = [
            Candle(
                symbol=symbol,
                bar="1D",
                ts=1_700_000_000_000 + idx * 86_400_000,
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=1_000.0,
            )
            for idx in range(60)
        ]
        rows.append(
            Candle(
                symbol=symbol,
                bar="1D",
                ts=1_700_000_000_000 + 60 * 86_400_000,
                open=100.0,
                high=113.0,
                low=99.0,
                close=112.0,
                volume=3_000.0,
            )
        )

        snapshot = DailyFeatureBuilder().build(symbol, rows)

        self.assertIsNotNone(snapshot)
        features = snapshot.features
        self.assertEqual(features["latest_candle_direction"], "bullish")
        self.assertAlmostEqual(features["latest_body_pct"], 12.0 / 14.0)
        self.assertAlmostEqual(features["latest_upper_shadow_pct"], 1.0 / 14.0)
        self.assertAlmostEqual(features["latest_lower_shadow_pct"], 1.0 / 14.0)
        self.assertGreater(features["latest_body_atr_ratio"], 2.0)
        self.assertGreater(features["latest_volume_ratio"], 2.0)
        self.assertTrue(features["latest_is_big_bull_candle"])
        self.assertFalse(features["latest_is_big_bear_candle"])
        self.assertTrue(features["latest_is_volume_expansion"])
        self.assertGreaterEqual(features["recent_big_bull_count_20"], 1)

    def test_multitimeframe_snapshot_includes_prefixed_candlestick_tags(self):
        symbol = "BEAR-USDT-SWAP"
        h1_rows = candles(symbol, "1H", 59, step=0.0, interval=3_600_000)
        h1_rows.append(
            Candle(
                symbol=symbol,
                bar="1H",
                ts=1_700_000_000_000 + 59 * 3_600_000,
                open=120.0,
                high=121.0,
                low=99.0,
                close=100.0,
                volume=5_000.0,
            )
        )

        snapshot = MultiTimeframeFeatureBuilder().build(
            symbol,
            {
                "1D": candles(symbol, "1D", 70, step=0.001, interval=86_400_000),
                "1H": h1_rows,
            },
        )

        self.assertIsNotNone(snapshot)
        features = snapshot.features
        self.assertEqual(features["h1_latest_candle_direction"], "bearish")
        self.assertGreater(features["h1_latest_body_atr_ratio"], 2.0)
        self.assertGreater(features["h1_latest_volume_ratio"], 2.0)
        self.assertTrue(features["h1_latest_is_big_bear_candle"])
        self.assertFalse(features["h1_latest_is_big_bull_candle"])
        self.assertTrue(features["h1_latest_is_volume_expansion"])
        self.assertGreaterEqual(features["h1_recent_big_bear_count_20"], 1)

    def test_builds_daily_and_intraday_features_in_one_snapshot(self):
        symbol = "BTC-USDT-SWAP"
        snapshot = MultiTimeframeFeatureBuilder().build(
            symbol,
            {
                "1D": candles(symbol, "1D", 70, step=0.012, interval=86_400_000),
                "1H": candles(symbol, "1H", 60, step=0.002, interval=3_600_000),
                "15m": candles(symbol, "15m", 50, step=0.0015, interval=900_000),
                "5m": candles(symbol, "5m", 40, step=0.001, interval=300_000),
            },
        )

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.bar, "multi")
        self.assertIn("ret_20d", snapshot.features)
        self.assertIn("vol_ratio_20d", snapshot.features)
        self.assertIn("ma_60", snapshot.features)
        self.assertIn("ema_12", snapshot.features)
        self.assertIn("ema_26", snapshot.features)
        self.assertIn("macd_dif", snapshot.features)
        self.assertIn("macd_dea", snapshot.features)
        self.assertIn("macd_hist", snapshot.features)
        self.assertIn("atr_14", snapshot.features)
        self.assertIn("rsi_14", snapshot.features)
        self.assertIn("h1_ret_24", snapshot.features)
        self.assertIn("h1_ema_fast", snapshot.features)
        self.assertIn("h1_macd_hist", snapshot.features)
        self.assertIn("h1_atr_14", snapshot.features)
        self.assertIn("m15_ret_8", snapshot.features)
        self.assertIn("m5_pos_24", snapshot.features)
        self.assertTrue(snapshot.features["m5_data_available"])
        self.assertEqual(snapshot.features["m5_candle_count"], 40)
        self.assertGreater(snapshot.features["ret_20d"], 0)
        self.assertGreater(snapshot.features["h1_ret_24"], 0)
        self.assertGreater(snapshot.features["ema_12"], snapshot.features["ema_26"])
        self.assertGreaterEqual(snapshot.features["rsi_14"], 0)
        self.assertLessEqual(snapshot.features["rsi_14"], 100)

    def test_missing_intraday_bar_is_marked_unavailable(self):
        symbol = "BTC-USDT-SWAP"
        snapshot = MultiTimeframeFeatureBuilder().build(
            symbol,
            {
                "1D": candles(symbol, "1D", 70, step=0.012, interval=86_400_000),
                "15m": candles(symbol, "15m", 50, step=0.0015, interval=900_000),
            },
        )

        self.assertIsNotNone(snapshot)
        self.assertTrue(snapshot.features["m15_data_available"])
        self.assertFalse(snapshot.features["m5_data_available"])
        self.assertEqual(snapshot.features["m5_candle_count"], 0)
        self.assertEqual(snapshot.features["m5_ret_6"], 0.0)


if __name__ == "__main__":
    unittest.main()
