import unittest

from langlang_trader.market_data import BinanceRestMarketData, FallbackMarketData


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
        if "ticker/24hr" in url:
            return self.payloads["ticker"]
        if "depth" in url:
            return self.payloads["depth"]
        raise AssertionError(url)


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

    def test_fallback_market_data_uses_secondary_source_when_primary_does_not_have_symbol(self):
        primary = StubMarketData(error=RuntimeError("not on okx"))
        fallback = StubMarketData(candles=["fallback-candle"], price=42.0)
        market = FallbackMarketData(primary, fallback)

        self.assertEqual(market.get_candles("NEWCOIN-USDT-SWAP"), ["fallback-candle"])
        self.assertEqual(market.latest_price("NEWCOIN-USDT-SWAP"), 42.0)
        self.assertEqual(primary.calls[0][0], "candles")
        self.assertEqual(fallback.calls[0][0], "candles")


if __name__ == "__main__":
    unittest.main()
