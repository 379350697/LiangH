import unittest
from http.client import RemoteDisconnected
from unittest.mock import patch

from langlang_trader.market_data import BinanceRestMarketData, FallbackMarketData, OkxRestMarketData


class StubMarketData:
    def __init__(self, *, candles=None, price=None, error=None):
        self.candles = candles or []
        self.price = price
        self.error = error
        self.calls = []

    def get_candles(self, symbol, bar="1D", limit=120):
        self.calls.append(("candles", symbol, bar, limit))
        if self.error:
            raise self.error
        return self.candles

    def latest_price(self, symbol):
        self.calls.append(("latest_price", symbol))
        if self.error:
            raise self.error
        return self.price

    def get_ticker(self, symbol):
        raise NotImplementedError

    def get_order_book(self, symbol, depth=20):
        raise NotImplementedError


class StubBinanceRestMarketData(BinanceRestMarketData):
    def __init__(self, payloads):
        super().__init__(base_url="https://example.invalid")
        self.payloads = payloads
        self.urls = []

    def _get_json(self, url):
        self.urls.append(url)
        if "klines" in url:
            return self.payloads["klines"]
        if "aggTrades" in url:
            return self.payloads["aggTrades"]
        if "ticker/24hr" in url:
            return self.payloads["ticker"]
        if "depth" in url:
            return self.payloads["depth"]
        if "premiumIndex" in url:
            return self.payloads["premiumIndex"]
        if "openInterest" in url:
            return self.payloads["openInterest"]
        raise AssertionError(url)


class FakeResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self.payload


class BinanceMarketDataTest(unittest.TestCase):
    def test_binance_rest_market_data_maps_canonical_swap_symbols_to_futures_klines(self):
        market = StubBinanceRestMarketData(
            {
                "klines": [
                    [1_700_000_000_000, "100", "110", "90", "105", "1234", 1_700_000_059_999],
                    [1_700_000_060_000, "105", "115", "101", "112", "2345", 1_700_000_119_999],
                ],
                "ticker": {
                    "closeTime": 1_700_000_120_000,
                    "lastPrice": "112",
                    "bidPrice": "111.9",
                    "askPrice": "112.1",
                    "quoteVolume": "123456",
                },
                "depth": {"lastUpdateId": 99, "bids": [["111.9", "10"]], "asks": [["112.1", "11"]]},
            }
        )

        candles = market.get_candles("SOL-USDT-SWAP", bar="1m", limit=2)
        ticker = market.get_ticker("SOL-USDT-SWAP")
        book = market.get_order_book("SOL-USDT-SWAP")

        self.assertIn("symbol=SOLUSDT", market.urls[0])
        self.assertIn("interval=1m", market.urls[0])
        self.assertEqual([candle.close for candle in candles], [105.0, 112.0])
        self.assertEqual(candles[0].symbol, "SOL-USDT-SWAP")
        self.assertEqual(ticker.last, 112.0)
        self.assertEqual(ticker.volume_24h, 123456.0)
        self.assertEqual(book.bids[0].price, 111.9)

    def test_binance_rest_market_data_supports_4h_klines(self):
        market = StubBinanceRestMarketData(
            {
                "klines": [[1_700_000_000_000, "100", "110", "90", "105", "1234", 1_700_000_059_999]],
                "ticker": {},
                "depth": {},
            }
        )

        candles = market.get_candles("SOL-USDT-SWAP", bar="4H", limit=1)

        self.assertEqual(candles[0].bar, "4H")
        self.assertIn("interval=4h", market.urls[0])

    def test_binance_rest_market_data_supports_3m_klines(self):
        market = StubBinanceRestMarketData(
            {
                "klines": [[1_700_000_000_000, "100", "110", "90", "105", "1234", 1_700_000_059_999]],
                "ticker": {},
                "depth": {},
            }
        )

        candles = market.get_candles("BTC-USDT-SWAP", bar="3m", limit=1)

        self.assertEqual(candles[0].bar, "3m")
        self.assertIn("interval=3m", market.urls[0])

    def test_binance_rest_market_data_builds_second_candles_from_agg_trades(self):
        market = StubBinanceRestMarketData(
            {
                "klines": [],
                "aggTrades": [
                    {"T": 1_700_000_000_100, "p": "100.0", "q": "2.0"},
                    {"T": 1_700_000_001_200, "p": "101.0", "q": "1.0"},
                    {"T": 1_700_000_004_900, "p": "99.0", "q": "3.0"},
                    {"T": 1_700_000_005_100, "p": "102.0", "q": "4.0"},
                ],
                "ticker": {},
                "depth": {},
            }
        )

        candles = market.get_candles("BTC-USDT-SWAP", bar="5s", limit=10)

        self.assertIn("aggTrades", market.urls[0])
        self.assertEqual([row.ts for row in candles], [1_700_000_000_000, 1_700_000_005_000])
        self.assertEqual(candles[0].open, 100.0)
        self.assertEqual(candles[0].high, 101.0)
        self.assertEqual(candles[0].low, 99.0)
        self.assertEqual(candles[0].close, 99.0)
        self.assertEqual(candles[0].volume, 6.0)
        self.assertEqual(candles[0].source, "binance_agg_trades")

    def test_binance_market_metrics_include_mark_index_basis_and_next_funding_time(self):
        market = StubBinanceRestMarketData(
            {
                "klines": [],
                "aggTrades": [],
                "ticker": {
                    "closeTime": 1_700_000_120_000,
                    "lastPrice": "101",
                    "bidPrice": "100.9",
                    "askPrice": "101.1",
                    "quoteVolume": "123456",
                },
                "depth": {"lastUpdateId": 99, "bids": [["100.9", "10"]], "asks": [["101.1", "11"]]},
                "premiumIndex": {
                    "markPrice": "101.0",
                    "indexPrice": "100.0",
                    "lastFundingRate": "0.0004",
                    "nextFundingTime": 1_700_028_800_000,
                },
                "openInterest": {"openInterest": "1000"},
            }
        )

        metrics = market.get_market_metrics("BTC-USDT-SWAP")

        self.assertEqual(metrics["mark_price"], 101.0)
        self.assertEqual(metrics["index_price"], 100.0)
        self.assertAlmostEqual(metrics["basis_bps"], 100.0)
        self.assertEqual(metrics["next_funding_time_ms"], 1_700_028_800_000)
        self.assertEqual(metrics["funding_rate_last"], 0.0004)

    def test_fallback_market_data_uses_secondary_source_when_primary_does_not_have_symbol(self):
        primary = StubMarketData(error=RuntimeError("not on okx"))
        fallback = StubMarketData(candles=["fallback-candle"], price=42.0)
        market = FallbackMarketData(primary, fallback)

        self.assertEqual(market.get_candles("NEWCOIN-USDT-SWAP"), ["fallback-candle"])
        self.assertEqual(market.latest_price("NEWCOIN-USDT-SWAP"), 42.0)
        self.assertEqual(primary.calls[0][0], "candles")
        self.assertEqual(fallback.calls[0][0], "candles")

    def test_fallback_market_data_reports_both_errors_when_sources_fail(self):
        primary = StubMarketData(error=RuntimeError("primary ssl eof"))
        fallback = StubMarketData(error=RuntimeError("fallback timeout"))
        market = FallbackMarketData(primary, fallback)

        with self.assertRaisesRegex(RuntimeError, "primary ssl eof.*fallback timeout"):
            market.get_candles("BTC-USDT-SWAP")

    def test_fallback_market_data_treats_empty_primary_candles_as_failure(self):
        primary = StubMarketData(candles=[])
        fallback = StubMarketData(candles=["fallback-candle"])
        market = FallbackMarketData(primary, fallback)

        self.assertEqual(market.get_candles("BTC-USDT-SWAP"), ["fallback-candle"])
        self.assertEqual(primary.calls[0][0], "candles")
        self.assertEqual(fallback.calls[0][0], "candles")

    def test_fallback_market_data_raises_when_both_sources_return_empty_candles(self):
        market = FallbackMarketData(StubMarketData(candles=[]), StubMarketData(candles=[]))

        with self.assertRaisesRegex(RuntimeError, "empty market data response.*empty market data response"):
            market.get_candles("BTC-USDT-SWAP")

    def test_rest_market_data_defaults_to_three_retries(self):
        self.assertEqual(OkxRestMarketData().retries, 3)
        self.assertEqual(BinanceRestMarketData().retries, 3)

    def test_binance_get_json_retries_os_errors(self):
        market = BinanceRestMarketData(base_url="https://example.invalid", retries=3)
        calls = []

        def flaky_urlopen(req, timeout):
            calls.append(req.full_url)
            if len(calls) < 3:
                raise OSError("ssl eof")
            return FakeResponse(b'{"lastPrice":"100","closeTime":1}')

        with patch("langlang_trader.market_data.request.urlopen", side_effect=flaky_urlopen), patch(
            "langlang_trader.market_data.time.sleep"
        ):
            payload = market._get_json("https://example.invalid/fapi/v1/ticker/24hr?symbol=BTCUSDT")

        self.assertEqual(payload["lastPrice"], "100")
        self.assertEqual(len(calls), 3)

    def test_okx_get_json_retries_remote_disconnect(self):
        market = OkxRestMarketData(base_url="https://example.invalid", retries=3)
        calls = []

        def flaky_urlopen(req, timeout):
            calls.append(req.full_url)
            if len(calls) < 3:
                raise RemoteDisconnected("remote closed")
            return FakeResponse(b'{"code":"0","data":[]}')

        with patch("langlang_trader.market_data.request.urlopen", side_effect=flaky_urlopen), patch(
            "langlang_trader.market_data.time.sleep"
        ):
            payload = market._get_json("https://example.invalid/api/v5/market/ticker?instId=BTC-USDT-SWAP")

        self.assertEqual(payload["code"], "0")
        self.assertEqual(len(calls), 3)


if __name__ == "__main__":
    unittest.main()
