import unittest

from langlang_trader.features import MultiTimeframeFeatureBuilder
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
        self.assertGreater(snapshot.features["ret_20d"], 0)
        self.assertGreater(snapshot.features["h1_ret_24"], 0)
        self.assertGreater(snapshot.features["ema_12"], snapshot.features["ema_26"])
        self.assertGreaterEqual(snapshot.features["rsi_14"], 0)
        self.assertLessEqual(snapshot.features["rsi_14"], 100)


if __name__ == "__main__":
    unittest.main()
