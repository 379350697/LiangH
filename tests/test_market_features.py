import unittest

from langlang_trader.market_features import (
    BinanceDerivativeMetricsClient,
    BinanceSymbolMapper,
    DerivativeMetric,
    HistoricalMarketFeatureBuilder,
    MarketFeatureJoiner,
    OkxDerivativeMetricsClient,
    collect_historical_derivative_metrics,
)
from langlang_trader.features import FeatureSnapshot
from langlang_trader.models import Candle


DAY_MS = 86_400_000


def daily_candles(symbol, count, *, start_ts=1_700_000_000_000, vol_quote_everywhere=True):
    rows = []
    for idx in range(count):
        close = 100 + idx
        rows.append(
            Candle(
                symbol=symbol,
                bar="1D",
                ts=start_ts + idx * DAY_MS,
                open=close - 1,
                high=close + 2,
                low=close - 2,
                close=close,
                volume=10 + idx,
                vol_quote=(close * (10 + idx)) if vol_quote_everywhere else None,
            )
        )
    return rows


class HistoricalMarketFeatureBuilderTest(unittest.TestCase):
    def test_materializes_daily_indicators_without_lookahead(self):
        symbol = "TEST-USDT-SWAP"
        rows = daily_candles(symbol, 70)

        features = HistoricalMarketFeatureBuilder().build_technical_rows({symbol: {"1D": rows}})
        row_by_ts = {row["ts"]: row for row in features}
        target = row_by_ts[rows[60].ts]

        self.assertEqual(target["symbol"], symbol)
        self.assertEqual(target["timeframe"], "1D")
        self.assertEqual(target["data_status"], "available")
        self.assertAlmostEqual(target["ret_20"], rows[60].close / rows[40].close - 1.0)
        self.assertAlmostEqual(target["ma_5"], sum(row.close for row in rows[56:61]) / 5)
        self.assertIn("macd_hist", target)
        self.assertIn("atr_14", target)
        self.assertIn("rsi_14", target)

    def test_indicator_warmup_is_explicit_terminal_status(self):
        symbol = "WARM-USDT-SWAP"
        rows = daily_candles(symbol, 3)

        features = HistoricalMarketFeatureBuilder().build_technical_rows({symbol: {"1D": rows}})

        self.assertTrue(features)
        self.assertEqual({row["data_status"] for row in features}, {"indicator_warmup"})

    def test_turnover_uses_quote_volume_or_marks_estimate(self):
        symbol = "TURN-USDT-SWAP"
        with_quote = daily_candles(symbol, 62, vol_quote_everywhere=True)
        no_quote = daily_candles("EST-USDT-SWAP", 62, vol_quote_everywhere=False)

        rows = HistoricalMarketFeatureBuilder().build_technical_rows(
            {symbol: {"1D": with_quote}, "EST-USDT-SWAP": {"1D": no_quote}}
        )
        latest = {row["symbol"]: row for row in rows if row["ts"] == row["last_ts"]}

        self.assertEqual(latest[symbol]["turnover_status"], "available")
        self.assertEqual(latest["EST-USDT-SWAP"]["turnover_status"], "estimated_turnover")
        self.assertAlmostEqual(
            latest["EST-USDT-SWAP"]["turnover_usdt"],
            no_quote[-1].close * no_quote[-1].volume,
        )

    def test_derivatives_join_latest_value_without_future_leak(self):
        symbol = "OI-USDT-SWAP"
        candles = daily_candles(symbol, 62)
        funding = [
            DerivativeMetric(symbol=symbol, ts=candles[-2].ts + 1_000, metric="funding_rate", value=0.0002, source="binance"),
            DerivativeMetric(symbol=symbol, ts=candles[-1].ts + 1_000, metric="funding_rate", value=0.0099, source="binance"),
        ]
        oi = [
            DerivativeMetric(symbol=symbol, ts=candles[-2].ts, metric="open_interest_usd", value=1000, source="binance"),
            DerivativeMetric(symbol=symbol, ts=candles[-1].ts + 1, metric="open_interest_usd", value=5000, source="binance"),
        ]

        rows = HistoricalMarketFeatureBuilder().build_derivative_rows(
            {symbol: {"1D": candles}},
            funding_metrics=funding,
            open_interest_metrics=oi,
        )
        row = next(item for item in rows if item["ts"] == candles[-1].ts)

        self.assertEqual(row["funding_rate_status"], "available")
        self.assertAlmostEqual(row["funding_rate_last"], 0.0002)
        self.assertEqual(row["open_interest_status"], "available")
        self.assertEqual(row["open_interest_usd"], 1000)
        self.assertNotEqual(row["open_interest_usd"], 5000)

    def test_external_market_cap_is_explicit_provider_limited_by_default(self):
        symbol = "CAP-USDT-SWAP"
        rows = daily_candles(symbol, 62)

        external = HistoricalMarketFeatureBuilder().build_external_market_rows({symbol: {"1D": rows}})

        self.assertTrue(external)
        self.assertEqual(external[-1]["market_cap_status"], "provider_limited")
        self.assertEqual(external[-1]["data_status"], "provider_limited")

    def test_joiner_adds_market_features_to_snapshot(self):
        snapshot = FeatureSnapshot(
            symbol="JOIN-USDT-SWAP",
            bar="multi",
            last_ts=1000,
            created_at="2024-01-01T00:00:00+00:00",
            features={"ret_20d": 0.1},
        )
        joined = MarketFeatureJoiner({("JOIN-USDT-SWAP", "1D", 900): {"turnover_usdt": 123, "funding_rate_last": 0.01}}).join(
            snapshot
        )

        self.assertEqual(joined.features["turnover_usdt"], 123)
        self.assertEqual(joined.features["funding_rate_last"], 0.01)

    def test_binance_symbol_mapper_discovers_multiplier_contracts_from_exchange_info(self):
        mapper = BinanceSymbolMapper.from_exchange_info(
            {
                "symbols": [
                    {"symbol": "BTCUSDT", "contractType": "PERPETUAL", "quoteAsset": "USDT", "status": "TRADING"},
                    {"symbol": "1000PEPEUSDT", "contractType": "PERPETUAL", "quoteAsset": "USDT", "status": "TRADING"},
                ]
            }
        )

        self.assertEqual(mapper.to_exchange_symbol("BTC-USDT-SWAP"), "BTCUSDT")
        self.assertEqual(mapper.to_exchange_symbol("PEPE-USDT-SWAP"), "1000PEPEUSDT")

    def test_derivative_clients_parse_binance_and_okx_payloads(self):
        binance = BinanceDerivativeMetricsClient(
            symbol_mapper=BinanceSymbolMapper({"BTC": "BTCUSDT"}),
            json_getter=lambda url: [
                {"fundingTime": 1000, "fundingRate": "0.0001"},
            ]
            if "fundingRate" in url
            else [{"timestamp": 2000, "sumOpenInterestValue": "123456"}],
        )
        okx = OkxDerivativeMetricsClient(
            json_getter=lambda url: {
                "code": "0",
                "data": [
                    {"fundingTime": "3000", "fundingRate": "0.0002"},
                ],
            }
        )

        funding = binance.fetch_funding_rates("BTC-USDT-SWAP", 0, 10_000)
        oi = binance.fetch_open_interest("BTC-USDT-SWAP", 0, 10_000)
        okx_funding = okx.fetch_funding_rates("BTC-USDT-SWAP", 0, 10_000)

        self.assertEqual(funding[0], DerivativeMetric("BTC-USDT-SWAP", 1000, "funding_rate", 0.0001, "binance"))
        self.assertEqual(oi[0], DerivativeMetric("BTC-USDT-SWAP", 2000, "open_interest_usd", 123456.0, "binance"))
        self.assertEqual(okx_funding[0], DerivativeMetric("BTC-USDT-SWAP", 3000, "funding_rate", 0.0002, "okx"))

    def test_binance_derivative_client_paginates_long_history(self):
        seen_urls = []

        def getter(url):
            seen_urls.append(url)
            if "fundingRate" in url:
                if len([item for item in seen_urls if "fundingRate" in item]) == 1:
                    return [
                        {"fundingTime": 1000, "fundingRate": "0.0001"},
                        {"fundingTime": 2000, "fundingRate": "0.0002"},
                    ]
                return [{"fundingTime": 3000, "fundingRate": "0.0003"}]
            if len([item for item in seen_urls if "openInterestHist" in item]) == 1:
                return [
                    {"timestamp": 1000, "sumOpenInterestValue": "100"},
                    {"timestamp": 2000, "sumOpenInterestValue": "200"},
                ]
            return [{"timestamp": 3000, "sumOpenInterestValue": "300"}]

        binance = BinanceDerivativeMetricsClient(
            symbol_mapper=BinanceSymbolMapper({"BTC": "BTCUSDT"}),
            json_getter=getter,
        )

        funding = binance.fetch_funding_rates("BTC-USDT-SWAP", 0, 10_000, limit=2)
        oi = binance.fetch_open_interest("BTC-USDT-SWAP", 0, 10_000, limit=2)

        self.assertEqual([row.ts for row in funding], [1000, 2000, 3000])
        self.assertEqual([row.ts for row in oi], [1000, 2000, 3000])

    def test_collect_historical_derivative_metrics_fetches_symbol_ranges_and_coverage(self):
        class FakeBinance:
            def __init__(self):
                self.funding_calls = []
                self.oi_calls = []

            def fetch_funding_rates(self, symbol, start_ms, end_ms):
                self.funding_calls.append((symbol, start_ms, end_ms))
                return [DerivativeMetric(symbol, start_ms + 1, "funding_rate", 0.0001, "binance")]

            def fetch_open_interest(self, symbol, start_ms, end_ms):
                self.oi_calls.append((symbol, start_ms, end_ms))
                return [DerivativeMetric(symbol, start_ms + 2, "open_interest_usd", 1000.0, "binance")]

        class FakeOkx:
            def __init__(self):
                self.funding_calls = []

            def fetch_funding_rates(self, symbol, start_ms, end_ms):
                self.funding_calls.append((symbol, start_ms, end_ms))
                return [DerivativeMetric(symbol, start_ms + 3, "funding_rate", 0.0002, "okx")]

        symbol = "DERIV-USDT-SWAP"
        candles = daily_candles(symbol, 2)
        binance = FakeBinance()
        okx = FakeOkx()

        funding, oi, coverage = collect_historical_derivative_metrics(
            {symbol: {"1D": candles}},
            binance_client=binance,
            okx_client=okx,
        )

        self.assertEqual(len(funding), 2)
        self.assertEqual(len(oi), 1)
        self.assertEqual(binance.funding_calls[0], (symbol, candles[0].ts, candles[-1].ts + DAY_MS))
        self.assertEqual(okx.funding_calls[0], (symbol, candles[0].ts, candles[-1].ts + DAY_MS))
        self.assertEqual({row["metric"] for row in coverage}, {"funding_rate", "open_interest_usd"})
        self.assertTrue(all(row["data_status"] == "available" for row in coverage))


if __name__ == "__main__":
    unittest.main()
