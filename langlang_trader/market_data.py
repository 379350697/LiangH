from __future__ import annotations

from dataclasses import replace
import json
from http.client import RemoteDisconnected
import time
from typing import Any, Protocol
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

    def get_market_metrics(self, symbol: str) -> dict[str, Any]:
        ...


class SymbolMappedMarketData:
    def __init__(self, upstream: MarketData, symbol_map: dict[str, str]):
        self.upstream = upstream
        self.symbol_map = dict(symbol_map)

    def get_candles(self, symbol: str, bar: str = "1D", limit: int = 120) -> list[Candle]:
        exchange_symbol = self._exchange_symbol(symbol)
        rows = self.upstream.get_candles(exchange_symbol, bar=bar, limit=limit)
        return [replace(row, symbol=symbol) for row in rows]

    def latest_price(self, symbol: str) -> float:
        return self.upstream.latest_price(self._exchange_symbol(symbol))

    def get_ticker(self, symbol: str) -> Ticker:
        ticker = self.upstream.get_ticker(self._exchange_symbol(symbol))
        return replace(ticker, symbol=symbol)

    def get_order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        book = self.upstream.get_order_book(self._exchange_symbol(symbol), depth=depth)
        return replace(book, symbol=symbol)

    def get_market_metrics(self, symbol: str) -> dict[str, Any]:
        return dict(self.upstream.get_market_metrics(self._exchange_symbol(symbol)))

    def _exchange_symbol(self, symbol: str) -> str:
        return self.symbol_map.get(symbol, symbol)


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

    def get_market_metrics(self, symbol: str) -> dict[str, Any]:
        ticker = self.get_ticker(symbol)
        book = self.get_order_book(symbol)
        return {
            **market_microstructure_metrics(ticker=ticker, order_book=book),
            "funding_rate_status": "provider_limited",
            "funding_rate_last": "",
            "open_interest_status": "provider_limited",
            "open_interest_usd": "",
            "market_cap_status": "provider_limited",
            "market_cap_usd": "",
        }


_TRANSIENT_PUBLIC_ERRORS = (URLError, TimeoutError, OSError, RemoteDisconnected)


class OkxRestMarketData:
    def __init__(self, *, base_url: str = "https://www.okx.com", timeout_seconds: float = 6.0, retries: int = 3):
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

    def get_market_metrics(self, symbol: str) -> dict[str, Any]:
        ticker = self.get_ticker(symbol)
        order_book = self.get_order_book(symbol)
        metrics = {
            **market_microstructure_metrics(ticker=ticker, order_book=order_book),
            "market_cap_status": "provider_limited",
            "market_cap_usd": "",
        }
        funding_query = parse.urlencode({"instId": symbol})
        try:
            funding_payload = self._get_json(f"{self.base_url}/api/v5/public/funding-rate?{funding_query}")
            funding_rows = funding_payload.get("data") if funding_payload.get("code") == "0" else []
            funding_rate = _float_or_none(funding_rows[0].get("fundingRate")) if funding_rows else None
            metrics["funding_rate_last"] = funding_rate if funding_rate is not None else ""
            metrics["funding_rate_status"] = "available" if funding_rate is not None else "exchange_unavailable"
        except Exception as exc:
            metrics["funding_rate_last"] = ""
            metrics["funding_rate_status"] = "exchange_error"
            metrics["funding_rate_error"] = repr(exc)

        oi_query = parse.urlencode({"instType": "SWAP", "instId": symbol})
        try:
            oi_payload = self._get_json(f"{self.base_url}/api/v5/public/open-interest?{oi_query}")
            oi_rows = oi_payload.get("data") if oi_payload.get("code") == "0" else []
            row = oi_rows[0] if oi_rows else {}
            oi_usd = _float_or_none(row.get("oiUsd"))
            if oi_usd is None:
                oi_ccy = _float_or_none(row.get("oiCcy"))
                oi_usd = oi_ccy * ticker.last if oi_ccy is not None and ticker.last > 0 else None
            metrics["open_interest_usd"] = oi_usd if oi_usd is not None else ""
            metrics["open_interest_status"] = "available" if oi_usd is not None else "exchange_unavailable"
        except Exception as exc:
            metrics["open_interest_usd"] = ""
            metrics["open_interest_status"] = "exchange_error"
            metrics["open_interest_error"] = repr(exc)
        return metrics

    def _get_json(self, url: str) -> dict:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                with request.urlopen(request.Request(url, headers=self.headers), timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except _TRANSIENT_PUBLIC_ERRORS as exc:
                last_error = exc
                if attempt < self.retries - 1:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"OKX public request failed after retries: {url}; last_error={last_error!r}") from last_error


class BinanceRestMarketData:
    def __init__(self, *, base_url: str = "https://fapi.binance.com", timeout_seconds: float = 6.0, retries: int = 3):
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

    def get_market_metrics(self, symbol: str) -> dict[str, Any]:
        ticker = self.get_ticker(symbol)
        order_book = self.get_order_book(symbol)
        metrics = {
            **market_microstructure_metrics(ticker=ticker, order_book=order_book),
            "market_cap_status": "provider_limited",
            "market_cap_usd": "",
        }
        binance_symbol = _binance_symbol(symbol)
        try:
            funding_query = parse.urlencode({"symbol": binance_symbol})
            funding_payload = self._get_json(f"{self.base_url}/fapi/v1/premiumIndex?{funding_query}")
            funding_rate = _float_or_none(funding_payload.get("lastFundingRate"))
            metrics["funding_rate_last"] = funding_rate if funding_rate is not None else ""
            metrics["funding_rate_status"] = "available" if funding_rate is not None else "exchange_unavailable"
        except Exception as exc:
            metrics["funding_rate_last"] = ""
            metrics["funding_rate_status"] = "exchange_error"
            metrics["funding_rate_error"] = repr(exc)

        try:
            oi_query = parse.urlencode({"symbol": binance_symbol})
            oi_payload = self._get_json(f"{self.base_url}/fapi/v1/openInterest?{oi_query}")
            open_interest = _float_or_none(oi_payload.get("openInterest"))
            oi_usd = open_interest * ticker.last if open_interest is not None and ticker.last > 0 else None
            metrics["open_interest_usd"] = oi_usd if oi_usd is not None else ""
            metrics["open_interest_status"] = "available" if oi_usd is not None else "exchange_unavailable"
        except Exception as exc:
            metrics["open_interest_usd"] = ""
            metrics["open_interest_status"] = "exchange_error"
            metrics["open_interest_error"] = repr(exc)
        return metrics

    def _get_json(self, url: str):
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                with request.urlopen(request.Request(url, headers=self.headers), timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except _TRANSIENT_PUBLIC_ERRORS as exc:
                last_error = exc
                if attempt < self.retries - 1:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"Binance public request failed after retries: {url}; last_error={last_error!r}") from last_error


class FallbackMarketData:
    def __init__(self, primary: MarketData, fallback: MarketData):
        self.primary = primary
        self.fallback = fallback

    def get_candles(self, symbol: str, bar: str = "1D", limit: int = 120) -> list[Candle]:
        try:
            rows = self.primary.get_candles(symbol, bar=bar, limit=limit)
            if not rows:
                raise RuntimeError(f"empty market data response for {symbol} {bar}")
            return rows
        except Exception as primary_exc:
            try:
                rows = self.fallback.get_candles(symbol, bar=bar, limit=limit)
                if not rows:
                    raise RuntimeError(f"empty market data response for {symbol} {bar}")
                return rows
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"primary market data failed: {primary_exc!r}; fallback market data failed: {fallback_exc!r}"
                ) from fallback_exc

    def get_ticker(self, symbol: str) -> Ticker:
        try:
            return self.primary.get_ticker(symbol)
        except Exception as primary_exc:
            try:
                return self.fallback.get_ticker(symbol)
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"primary market data failed: {primary_exc!r}; fallback market data failed: {fallback_exc!r}"
                ) from fallback_exc

    def get_order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        try:
            return self.primary.get_order_book(symbol, depth=depth)
        except Exception as primary_exc:
            try:
                return self.fallback.get_order_book(symbol, depth=depth)
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"primary market data failed: {primary_exc!r}; fallback market data failed: {fallback_exc!r}"
                ) from fallback_exc

    def latest_price(self, symbol: str) -> float:
        try:
            return self.primary.latest_price(symbol)
        except Exception as primary_exc:
            try:
                return self.fallback.latest_price(symbol)
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"primary market data failed: {primary_exc!r}; fallback market data failed: {fallback_exc!r}"
                ) from fallback_exc

    def get_market_metrics(self, symbol: str) -> dict[str, Any]:
        try:
            return self.primary.get_market_metrics(symbol)
        except Exception as primary_exc:
            try:
                return self.fallback.get_market_metrics(symbol)
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"primary market metrics failed: {primary_exc!r}; fallback market metrics failed: {fallback_exc!r}"
                ) from fallback_exc


def market_microstructure_metrics(*, ticker: Ticker, order_book: OrderBook) -> dict[str, Any]:
    best_bid = ticker.bid
    best_ask = ticker.ask
    if best_bid is None and order_book.bids:
        best_bid = order_book.bids[0].price
    if best_ask is None and order_book.asks:
        best_ask = order_book.asks[0].price
    metrics: dict[str, Any] = {
        "ticker_volume_24h": ticker.volume_24h if ticker.volume_24h is not None else "",
        "book_depth_usdt_1pct": "",
        "book_depth_status": "unavailable",
        "spread_bps": "",
        "spread_status": "unavailable",
    }
    if best_bid is None or best_ask is None or best_bid <= 0 or best_ask <= 0:
        return metrics
    mid = (best_bid + best_ask) / 2.0
    if mid <= 0:
        return metrics
    metrics["spread_bps"] = ((best_ask - best_bid) / mid) * 10_000.0
    metrics["spread_status"] = "available"
    bid_floor = mid * 0.99
    ask_ceiling = mid * 1.01
    depth = 0.0
    for level in order_book.bids:
        if level.price >= bid_floor:
            depth += level.price * level.qty
    for level in order_book.asks:
        if level.price <= ask_ceiling:
            depth += level.price * level.qty
    metrics["book_depth_usdt_1pct"] = depth
    metrics["book_depth_status"] = "available"
    return metrics


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
        "4H": "4h",
        "4h": "4h",
        "1D": "1d",
        "1d": "1d",
    }
    if bar not in mapping:
        raise ValueError(f"unsupported Binance kline interval: {bar}")
    return mapping[bar]
