from __future__ import annotations

import json
import time
from typing import Protocol
from urllib.error import URLError
from urllib import parse, request

from langlang_trader.models import Candle, OrderBook, OrderBookLevel, Ticker


class MarketData(Protocol):
    def get_candles(self, symbol: str, bar: str = "1D", limit: int = 120) -> list[Candle]:
        ...

    def get_ticker(self, symbol: str) -> Ticker:
        ...

    def get_order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        ...

    def latest_price(self, symbol: str) -> float:
        ...


class StaticMarketData:
    def __init__(self, candles_by_symbol: dict[str, list[Candle]]):
        self.candles_by_symbol = {
            symbol: sorted(rows, key=lambda candle: candle.ts)
            for symbol, rows in candles_by_symbol.items()
        }

    def get_candles(self, symbol: str, bar: str = "1D", limit: int = 120) -> list[Candle]:
        rows = [candle for candle in self.candles_by_symbol.get(symbol, []) if candle.bar == bar]
        if not rows and bar == "1D":
            rows = self.candles_by_symbol.get(symbol, [])
        return rows[-limit:]

    def latest_price(self, symbol: str) -> float:
        rows = self.candles_by_symbol.get(symbol, [])
        if not rows:
            raise ValueError(f"no static candles for {symbol}")
        return rows[-1].close

    def get_ticker(self, symbol: str) -> Ticker:
        rows = self.candles_by_symbol.get(symbol, [])
        if not rows:
            raise ValueError(f"no static candles for {symbol}")
        latest = rows[-1]
        return Ticker(symbol=symbol, ts=latest.ts, last=latest.close)

    def get_order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        latest = self.get_ticker(symbol)
        return OrderBook(
            symbol=symbol,
            ts=latest.ts,
            bids=[OrderBookLevel(price=latest.last, qty=0.0)],
            asks=[OrderBookLevel(price=latest.last, qty=0.0)],
        )


class OkxRestMarketData:
    def __init__(self, *, base_url: str = "https://www.okx.com", timeout_seconds: float = 6.0, retries: int = 2):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.retries = max(1, retries)
        self.headers = {
            "User-Agent": "langlang-trader/0.1",
            "Accept": "application/json",
            "Connection": "close",
        }

    def get_candles(self, symbol: str, bar: str = "1D", limit: int = 120) -> list[Candle]:
        query = parse.urlencode({"instId": symbol, "bar": bar, "limit": str(limit)})
        url = f"{self.base_url}/api/v5/market/history-candles?{query}"
        payload = self._get_json(url)
        if payload.get("code") != "0":
            raise RuntimeError(f"OKX candle request failed: {payload}")
        rows = []
        for item in payload.get("data", []):
            rows.append(
                Candle(
                    symbol=symbol,
                    bar=bar,
                    ts=int(item[0]),
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5]),
                    vol_ccy=_float_or_none(item[6]) if len(item) > 6 else None,
                    vol_quote=_float_or_none(item[7]) if len(item) > 7 else None,
                    source="okx",
                )
            )
        return sorted(rows, key=lambda candle: candle.ts)

    def latest_price(self, symbol: str) -> float:
        return self.get_ticker(symbol).last

    def get_ticker(self, symbol: str) -> Ticker:
        query = parse.urlencode({"instId": symbol})
        url = f"{self.base_url}/api/v5/market/ticker?{query}"
        payload = self._get_json(url)
        if payload.get("code") != "0" or not payload.get("data"):
            raise RuntimeError(f"OKX ticker request failed: {payload}")
        row = payload["data"][0]
        return Ticker(
            symbol=symbol,
            ts=int(row.get("ts") or 0),
            last=float(row["last"]),
            bid=_float_or_none(row.get("bidPx")),
            ask=_float_or_none(row.get("askPx")),
            volume_24h=_float_or_none(row.get("volCcy24h")),
        )

    def get_order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        query = parse.urlencode({"instId": symbol, "sz": str(depth)})
        url = f"{self.base_url}/api/v5/market/books?{query}"
        payload = self._get_json(url)
        if payload.get("code") != "0" or not payload.get("data"):
            raise RuntimeError(f"OKX order book request failed: {payload}")
        row = payload["data"][0]
        return OrderBook(
            symbol=symbol,
            ts=int(row.get("ts") or 0),
            bids=[OrderBookLevel(price=float(level[0]), qty=float(level[1])) for level in row.get("bids", [])],
            asks=[OrderBookLevel(price=float(level[0]), qty=float(level[1])) for level in row.get("asks", [])],
        )

    def _get_json(self, url: str) -> dict:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                with request.urlopen(request.Request(url, headers=self.headers), timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except URLError as exc:
                last_error = exc
                if attempt < self.retries - 1:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"OKX public request failed after retries: {url}") from last_error


class BinanceRestMarketData:
    def __init__(self, *, base_url: str = "https://fapi.binance.com", timeout_seconds: float = 6.0, retries: int = 2):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.retries = max(1, retries)
        self.headers = {
            "User-Agent": "langlang-trader/0.1",
            "Accept": "application/json",
            "Connection": "close",
        }

    def get_candles(self, symbol: str, bar: str = "1D", limit: int = 120) -> list[Candle]:
        query = parse.urlencode(
            {
                "symbol": _binance_symbol(symbol),
                "interval": _binance_interval(bar),
                "limit": str(limit),
            }
        )
        payload = self._get_json(f"{self.base_url}/fapi/v1/klines?{query}")
        rows = []
        for item in payload:
            rows.append(
                Candle(
                    symbol=symbol,
                    bar=bar,
                    ts=int(item[0]),
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5]),
                    vol_quote=_float_or_none(item[7]) if len(item) > 7 else None,
                    source="binance",
                )
            )
        return sorted(rows, key=lambda candle: candle.ts)

    def latest_price(self, symbol: str) -> float:
        return self.get_ticker(symbol).last

    def get_ticker(self, symbol: str) -> Ticker:
        query = parse.urlencode({"symbol": _binance_symbol(symbol)})
        payload = self._get_json(f"{self.base_url}/fapi/v1/ticker/24hr?{query}")
        if not payload or "lastPrice" not in payload:
            raise RuntimeError(f"Binance ticker request failed: {payload}")
        return Ticker(
            symbol=symbol,
            ts=int(payload.get("closeTime") or 0),
            last=float(payload["lastPrice"]),
            bid=_float_or_none(payload.get("bidPrice")),
            ask=_float_or_none(payload.get("askPrice")),
            volume_24h=_float_or_none(payload.get("quoteVolume")),
        )

    def get_order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        query = parse.urlencode({"symbol": _binance_symbol(symbol), "limit": str(depth)})
        payload = self._get_json(f"{self.base_url}/fapi/v1/depth?{query}")
        if not payload:
            raise RuntimeError(f"Binance order book request failed: {payload}")
        return OrderBook(
            symbol=symbol,
            ts=int(payload.get("lastUpdateId") or 0),
            bids=[OrderBookLevel(price=float(level[0]), qty=float(level[1])) for level in payload.get("bids", [])],
            asks=[OrderBookLevel(price=float(level[0]), qty=float(level[1])) for level in payload.get("asks", [])],
        )

    def _get_json(self, url: str):
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                with request.urlopen(request.Request(url, headers=self.headers), timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except URLError as exc:
                last_error = exc
                if attempt < self.retries - 1:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"Binance public request failed after retries: {url}") from last_error


class FallbackMarketData:
    def __init__(self, primary: MarketData, fallback: MarketData):
        self.primary = primary
        self.fallback = fallback

    def get_candles(self, symbol: str, bar: str = "1D", limit: int = 120) -> list[Candle]:
        try:
            return self.primary.get_candles(symbol, bar=bar, limit=limit)
        except Exception:
            return self.fallback.get_candles(symbol, bar=bar, limit=limit)

    def get_ticker(self, symbol: str) -> Ticker:
        try:
            return self.primary.get_ticker(symbol)
        except Exception:
            return self.fallback.get_ticker(symbol)

    def get_order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        try:
            return self.primary.get_order_book(symbol, depth=depth)
        except Exception:
            return self.fallback.get_order_book(symbol, depth=depth)

    def latest_price(self, symbol: str) -> float:
        try:
            return self.primary.latest_price(symbol)
        except Exception:
            return self.fallback.latest_price(symbol)


def _float_or_none(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def _binance_symbol(symbol: str) -> str:
    if symbol.endswith("-USDT-SWAP"):
        return symbol.replace("-USDT-SWAP", "USDT").replace("-", "")
    return symbol.replace("-", "")


def _binance_interval(bar: str) -> str:
    mapping = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "1H": "1h",
        "1h": "1h",
        "1D": "1d",
        "1d": "1d",
    }
    if bar not in mapping:
        raise ValueError(f"unsupported Binance kline interval: {bar}")
    return mapping[bar]
