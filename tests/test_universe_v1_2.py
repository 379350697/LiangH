import json
import os
import tempfile
import unittest

from langlang_trader.universe import BinanceUniverseProvider, OkxBinanceUniverseProvider, OkxUniverseProvider, write_universe_snapshot


class UniverseProviderV12Test(unittest.TestCase):
    def test_okx_universe_keeps_live_usdt_swaps_and_separates_btc_eth_references(self):
        payload = {
            "code": "0",
            "data": [
                {
                    "instId": "BTC-USDT-SWAP",
                    "instType": "SWAP",
                    "state": "live",
                    "settleCcy": "USDT",
                    "baseCcy": "BTC",
                    "quoteCcy": "USDT",
                },
                {
                    "instId": "ETH-USDT-SWAP",
                    "instType": "SWAP",
                    "state": "live",
                    "settleCcy": "USDT",
                    "baseCcy": "ETH",
                    "quoteCcy": "USDT",
                },
                {
                    "instId": "SOL-USDT-SWAP",
                    "instType": "SWAP",
                    "state": "live",
                    "settleCcy": "USDT",
                    "baseCcy": "SOL",
                    "quoteCcy": "USDT",
                },
                {
                    "instId": "DOGE-USDT-SWAP",
                    "instType": "SWAP",
                    "state": "suspend",
                    "settleCcy": "USDT",
                    "baseCcy": "DOGE",
                    "quoteCcy": "USDT",
                },
                {
                    "instId": "BTC-USD-SWAP",
                    "instType": "SWAP",
                    "state": "live",
                    "settleCcy": "USD",
                    "baseCcy": "BTC",
                    "quoteCcy": "USD",
                },
                {
                    "instId": "SOL-USDT",
                    "instType": "SPOT",
                    "state": "live",
                    "settleCcy": "",
                    "baseCcy": "SOL",
                    "quoteCcy": "USDT",
                },
            ],
        }

        snapshot = OkxUniverseProvider.snapshot_from_payload(payload)

        self.assertEqual(snapshot.mode, "okx_all_usdt_swap")
        self.assertEqual(snapshot.symbols, ["SOL-USDT-SWAP"])
        self.assertEqual(snapshot.reference_symbols, ["BTC-USDT-SWAP", "ETH-USDT-SWAP"])
        by_symbol = {row.symbol: row for row in snapshot.rows}
        self.assertTrue(by_symbol["BTC-USDT-SWAP"].is_reference)
        self.assertEqual(by_symbol["BTC-USDT-SWAP"].filter_reason, "reference_market_anchor")
        self.assertEqual(by_symbol["SOL-USDT-SWAP"].filter_reason, "")
        self.assertIn("not_live", by_symbol["DOGE-USDT-SWAP"].filter_reason)
        self.assertIn("not_usdt_settled", by_symbol["BTC-USD-SWAP"].filter_reason)
        self.assertIn("not_swap", by_symbol["SOL-USDT"].filter_reason)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "universe_snapshot.json")
            write_universe_snapshot(path, snapshot)
            with open(path, encoding="utf-8") as handle:
                saved = json.load(handle)

        self.assertEqual(saved["symbols"], ["SOL-USDT-SWAP"])
        self.assertEqual(saved["reference_symbols"], ["BTC-USDT-SWAP", "ETH-USDT-SWAP"])
        self.assertIn("raw_payload", saved)

    def test_binance_universe_maps_trading_usdt_perpetuals_to_canonical_swap_symbols(self):
        payload = {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "pair": "BTCUSDT",
                    "contractType": "PERPETUAL",
                    "status": "TRADING",
                    "baseAsset": "BTC",
                    "quoteAsset": "USDT",
                },
                {
                    "symbol": "SOLUSDT",
                    "pair": "SOLUSDT",
                    "contractType": "PERPETUAL",
                    "status": "TRADING",
                    "baseAsset": "SOL",
                    "quoteAsset": "USDT",
                },
                {
                    "symbol": "DOGEUSDT",
                    "pair": "DOGEUSDT",
                    "contractType": "CURRENT_QUARTER",
                    "status": "TRADING",
                    "baseAsset": "DOGE",
                    "quoteAsset": "USDT",
                },
                {
                    "symbol": "BNBUSDC",
                    "pair": "BNBUSDC",
                    "contractType": "PERPETUAL",
                    "status": "TRADING",
                    "baseAsset": "BNB",
                    "quoteAsset": "USDC",
                },
                {
                    "symbol": "OLDUSDT",
                    "pair": "OLDUSDT",
                    "contractType": "PERPETUAL",
                    "status": "SETTLING",
                    "baseAsset": "OLD",
                    "quoteAsset": "USDT",
                },
            ]
        }

        snapshot = BinanceUniverseProvider.snapshot_from_payload(payload)

        self.assertEqual(snapshot.mode, "binance_usdt_perp_observe")
        self.assertEqual(snapshot.symbols, ["SOL-USDT-SWAP"])
        self.assertEqual(snapshot.reference_symbols, ["BTC-USDT-SWAP"])
        by_symbol = {row.symbol: row for row in snapshot.rows}
        self.assertEqual(by_symbol["SOL-USDT-SWAP"].source_exchange, "binance")
        self.assertEqual(by_symbol["SOL-USDT-SWAP"].exchange_symbol, "SOLUSDT")
        self.assertEqual(by_symbol["SOL-USDT-SWAP"].execution_symbol, "")
        self.assertFalse(by_symbol["SOL-USDT-SWAP"].observed_only)
        self.assertEqual(by_symbol["BTC-USDT-SWAP"].filter_reason, "reference_market_anchor")
        self.assertIn("not_perpetual", by_symbol["DOGE-USDT-SWAP"].filter_reason)
        self.assertIn("not_usdt_quote", by_symbol["BNB-USDC-SWAP"].filter_reason)
        self.assertIn("not_trading", by_symbol["OLD-USDT-SWAP"].filter_reason)

    def test_combined_okx_binance_universe_tracks_binance_only_symbols_without_expanding_okx_execution_pool(self):
        okx_payload = {
            "code": "0",
            "data": [
                {
                    "instId": "BTC-USDT-SWAP",
                    "instType": "SWAP",
                    "state": "live",
                    "settleCcy": "USDT",
                    "baseCcy": "BTC",
                    "quoteCcy": "USDT",
                },
                {
                    "instId": "SOL-USDT-SWAP",
                    "instType": "SWAP",
                    "state": "live",
                    "settleCcy": "USDT",
                    "baseCcy": "SOL",
                    "quoteCcy": "USDT",
                },
            ],
        }
        binance_payload = {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "contractType": "PERPETUAL",
                    "status": "TRADING",
                    "baseAsset": "SOL",
                    "quoteAsset": "USDT",
                },
                {
                    "symbol": "NEWCOINUSDT",
                    "contractType": "PERPETUAL",
                    "status": "TRADING",
                    "baseAsset": "NEWCOIN",
                    "quoteAsset": "USDT",
                },
            ]
        }

        snapshot = OkxBinanceUniverseProvider.snapshot_from_payloads(okx_payload, binance_payload)

        self.assertEqual(snapshot.mode, "okx_binance_usdt_swap_observe")
        self.assertEqual(snapshot.symbols, ["SOL-USDT-SWAP"])
        self.assertIn("NEWCOIN-USDT-SWAP", snapshot.observed_symbols)
        self.assertEqual(snapshot.raw_payload["summary"]["binance_only_observed_count"], 1)
        by_symbol_source = {(row.symbol, row.source_exchange): row for row in snapshot.rows}
        self.assertFalse(by_symbol_source[("SOL-USDT-SWAP", "okx")].observed_only)
        self.assertTrue(by_symbol_source[("NEWCOIN-USDT-SWAP", "binance")].observed_only)
        self.assertFalse(by_symbol_source[("NEWCOIN-USDT-SWAP", "binance")].tradable)
        self.assertEqual(
            by_symbol_source[("NEWCOIN-USDT-SWAP", "binance")].filter_reason,
            "binance_observed_only_not_okx_executable",
        )

    def test_combined_universe_filters_symbols_outside_top_liquidity_rank(self):
        okx_payload = {
            "code": "0",
            "data": [
                {
                    "instId": "BTC-USDT-SWAP",
                    "instType": "SWAP",
                    "state": "live",
                    "settleCcy": "USDT",
                    "baseCcy": "BTC",
                    "quoteCcy": "USDT",
                },
                {
                    "instId": "HIGH-USDT-SWAP",
                    "instType": "SWAP",
                    "state": "live",
                    "settleCcy": "USDT",
                    "baseCcy": "HIGH",
                    "quoteCcy": "USDT",
                },
                {
                    "instId": "LOW-USDT-SWAP",
                    "instType": "SWAP",
                    "state": "live",
                    "settleCcy": "USDT",
                    "baseCcy": "LOW",
                    "quoteCcy": "USDT",
                },
                {
                    "instId": "OLD-USD-SWAP",
                    "instType": "SWAP",
                    "state": "live",
                    "settleCcy": "USD",
                    "baseCcy": "OLD",
                    "quoteCcy": "USD",
                },
            ],
        }
        binance_payload = {
            "symbols": [
                {
                    "symbol": "HIGHUSDT",
                    "contractType": "PERPETUAL",
                    "status": "TRADING",
                    "baseAsset": "HIGH",
                    "quoteAsset": "USDT",
                },
                {
                    "symbol": "LOWUSDT",
                    "contractType": "PERPETUAL",
                    "status": "TRADING",
                    "baseAsset": "LOW",
                    "quoteAsset": "USDT",
                },
                {
                    "symbol": "MIDUSDT",
                    "contractType": "PERPETUAL",
                    "status": "TRADING",
                    "baseAsset": "MID",
                    "quoteAsset": "USDT",
                },
            ]
        }
        snapshot = OkxBinanceUniverseProvider.snapshot_from_payloads(
            okx_payload,
            binance_payload,
            okx_tickers_payload={
                "code": "0",
                "data": [
                    {"instId": "HIGH-USDT-SWAP", "volCcy24h": "1000000"},
                    {"instId": "LOW-USDT-SWAP", "volCcy24h": "100"},
                    {"instId": "OLD-USD-SWAP", "volCcy24h": "50"},
                ],
            },
            binance_tickers_payload=[
                {"symbol": "HIGHUSDT", "quoteVolume": "900000"},
                {"symbol": "MIDUSDT", "quoteVolume": "500000"},
                {"symbol": "LOWUSDT", "quoteVolume": "100"},
            ],
            liquidity_top_n=2,
        )

        self.assertIn("HIGH-USDT-SWAP", snapshot.observed_symbols)
        self.assertIn("MID-USDT-SWAP", snapshot.observed_symbols)
        self.assertNotIn("LOW-USDT-SWAP", snapshot.observed_symbols)
        self.assertNotIn("LOW-USDT-SWAP", snapshot.symbols)
        rows = {(row.symbol, row.source_exchange): row for row in snapshot.rows}
        self.assertEqual(rows[("HIGH-USDT-SWAP", "okx")].liquidity_rank, 1)
        self.assertEqual(rows[("MID-USDT-SWAP", "binance")].liquidity_rank, 2)
        self.assertIn("liquidity_rank_gt_2", rows[("LOW-USDT-SWAP", "okx")].filter_reason)
        self.assertIn("liquidity_rank_gt_2", rows[("LOW-USDT-SWAP", "binance")].filter_reason)
        self.assertIn("not_usdt_settled", rows[("OLD-USD-SWAP", "okx")].filter_reason)
        self.assertNotIn("liquidity_rank_gt_2", rows[("OLD-USD-SWAP", "okx")].filter_reason)
        self.assertEqual(snapshot.raw_payload["summary"]["liquidity_filter_top_n"], 2)
        self.assertEqual(snapshot.raw_payload["summary"]["liquidity_excluded_count"], 1)


if __name__ == "__main__":
    unittest.main()
