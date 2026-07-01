from __future__ import annotations

import asyncio
import http.client
import json
import socket
import time
import urllib.parse
from collections import deque
from collections.abc import AsyncIterator
from typing import Any

from .models import BookTick, TopBookTick, TradeTick


BOOK_STATUS_COLD = "cold"
BOOK_STATUS_BOOTSTRAPPING = "bootstrapping"
BOOK_STATUS_HOT = "hot"
BOOK_STATUS_REBUILDING = "rebuilding"
DEPTH_BUFFER_CAPACITY = 4096


class BinanceDepthSnapshotClient:
    def __init__(
        self,
        base_url: str = "https://fapi.binance.com",
        depth_limit: int = 1000,
        timeout_ms: int = 3000,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.depth_limit = depth_limit
        self.timeout_ms = timeout_ms

    async def fetch_snapshot(self, symbol: str) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                return await asyncio.to_thread(self._fetch_snapshot_sync, symbol.upper())
            except Exception as exc:
                last_exc = exc
                if attempt == 0:
                    await asyncio.sleep(0.1)
        assert last_exc is not None
        raise last_exc

    def _fetch_snapshot_sync(self, symbol: str) -> dict[str, Any]:
        parsed = urllib.parse.urlparse(self.base_url)
        host = parsed.hostname or parsed.netloc or parsed.path
        query = urllib.parse.urlencode({"symbol": symbol, "limit": self.depth_limit})
        connection = _IPv4HTTPSConnection(
            host,
            port=parsed.port,
            timeout=self.timeout_ms / 1000.0,
        )
        try:
            connection.request(
                "GET",
                f"/fapi/v1/depth?{query}",
                headers={"User-Agent": "liangh-market-maker/1.0"},
            )
            response = connection.getresponse()
            body = response.read()
            if response.status >= 400:
                raise RuntimeError(f"Binance depth snapshot failed status={response.status} body={body[:200]!r}")
            return json.loads(body.decode("utf-8"))
        finally:
            connection.close()


class LocalOrderBook:
    def __init__(self, symbol: str, venue: str = "binance_usdm") -> None:
        self.symbol = symbol.upper()
        self.venue = venue
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.last_update_id: int | None = None
        self.last_event_u: int | None = None
        self.status = BOOK_STATUS_COLD
        self.stale = True
        self.stale_since_ns: int | None = None
        self.sequence_gap_count = 0
        self.resync_count = 0
        self.fault_reason = ""
        self._buffer: deque[tuple[dict[str, Any], int]] = deque(maxlen=DEPTH_BUFFER_CAPACITY)

    @property
    def buffered_depth_event_count(self) -> int:
        return len(self._buffer)

    def apply_snapshot(
        self,
        last_update_id: int,
        bids: list[list[str]],
        asks: list[list[str]],
        event_time_ms: int,
        receive_time_ns: int,
    ) -> BookTick:
        snapshot_id = int(last_update_id)
        buffered_events = list(self._buffer)
        self._buffer.clear()
        self.bids = _levels_to_map(bids)
        self.asks = _levels_to_map(asks)
        self.last_update_id = snapshot_id
        self.last_event_u = None
        self.status = BOOK_STATUS_HOT
        self.stale = False
        self.stale_since_ns = None
        self.fault_reason = ""

        replay_events = [
            (event, event_receive_time_ns)
            for event, event_receive_time_ns in buffered_events
            if int(event["u"]) > snapshot_id
        ]
        for index, (event, event_receive_time_ns) in enumerate(replay_events):
            if not self._event_links_to_current(event, first_after_snapshot=index == 0):
                self._mark_rebuilding(
                    receive_time_ns=receive_time_ns,
                    reason=(
                        f"snapshot_boundary_miss lastUpdateId={snapshot_id} U={event.get('U')} "
                        f"u={event.get('u')} pu={event.get('pu')}"
                        if index == 0
                        else f"previous_link_mismatch expected={self.last_update_id} incoming_pu={event.get('pu')}"
                    ),
                    count_gap=index != 0,
                )
                self._buffer.append((event, event_receive_time_ns))
                return self._tick(
                    event_time_ms=event_time_ms,
                    receive_time_ns=receive_time_ns,
                    update_id=self.last_update_id,
                    sequence_gap=index != 0,
                )
            self._apply_depth_update(event)

        return self._tick(
            event_time_ms=event_time_ms,
            receive_time_ns=receive_time_ns,
            update_id=self.last_update_id,
            sequence_gap=False,
        )

    def apply_depth_event(self, event: dict[str, Any], receive_time_ns: int) -> BookTick:
        update_id = int(event["u"])
        event_time_ms = int(event.get("E", 0))
        if self.status != BOOK_STATUS_HOT or self.last_update_id is None:
            if self.status == BOOK_STATUS_COLD:
                self.status = BOOK_STATUS_BOOTSTRAPPING
            self.stale = True
            if self.stale_since_ns is None:
                self.stale_since_ns = receive_time_ns
            self._buffer.append((event, receive_time_ns))
            return self._tick(
                event_time_ms=event_time_ms,
                receive_time_ns=receive_time_ns,
                update_id=update_id,
                sequence_gap=False,
            )

        if update_id <= self.last_update_id:
            return self._tick(
                event_time_ms=event_time_ms,
                receive_time_ns=receive_time_ns,
                update_id=self.last_update_id,
                sequence_gap=False,
            )

        if not self._event_links_to_current(event, first_after_snapshot=False):
            self._mark_rebuilding(
                receive_time_ns=receive_time_ns,
                reason=f"previous_link_mismatch expected={self.last_update_id} incoming_pu={event.get('pu')}",
                count_gap=True,
            )
            self.resync_count += 1
            self._buffer.clear()
            self._buffer.append((event, receive_time_ns))
            return self._tick(
                event_time_ms=event_time_ms,
                receive_time_ns=receive_time_ns,
                update_id=update_id,
                sequence_gap=True,
            )

        self._apply_depth_update(event)
        self.status = BOOK_STATUS_HOT
        self.stale = False
        self.stale_since_ns = None
        self.fault_reason = ""
        return self._tick(
            event_time_ms=event_time_ms,
            receive_time_ns=receive_time_ns,
            update_id=update_id,
            sequence_gap=False,
        )

    def apply_book_ticker(self, event: dict[str, Any], receive_time_ns: int) -> TopBookTick:
        return TopBookTick(
            symbol=self.symbol,
            event_time_ms=int(event.get("E", 0)),
            receive_time_ns=receive_time_ns,
            best_bid=float(event["b"]),
            best_bid_qty=float(event["B"]),
            best_ask=float(event["a"]),
            best_ask_qty=float(event["A"]),
            update_id=int(event["u"]) if event.get("u") is not None else None,
            sequence_gap=False,
            stale=False,
            venue=self.venue,
            resync_count=self.resync_count,
            sequence_gap_count=self.sequence_gap_count,
            stale_since_ns=self.stale_since_ns,
        )

    def _event_links_to_current(self, event: dict[str, Any], first_after_snapshot: bool) -> bool:
        if self.last_update_id is None:
            return False
        first_update_id = int(event.get("U", event["u"]))
        final_update_id = int(event["u"])
        previous_update_id = _optional_int(event.get("pu"))
        if first_after_snapshot:
            return (
                previous_update_id == self.last_update_id
                or first_update_id <= self.last_update_id <= final_update_id
                or first_update_id <= self.last_update_id + 1 <= final_update_id
            )
        return previous_update_id == self.last_update_id

    def _apply_depth_update(self, event: dict[str, Any]) -> None:
        self._apply_levels(self.bids, event.get("b", []))
        self._apply_levels(self.asks, event.get("a", []))
        self.last_update_id = int(event["u"])
        self.last_event_u = int(event["u"])

    def _mark_rebuilding(self, receive_time_ns: int, reason: str, count_gap: bool) -> None:
        self.status = BOOK_STATUS_REBUILDING
        self.stale = True
        if self.stale_since_ns is None:
            self.stale_since_ns = receive_time_ns
        self.fault_reason = reason
        if count_gap:
            self.sequence_gap_count += 1

    @staticmethod
    def _apply_levels(side: dict[float, float], levels: list[list[str]]) -> None:
        for price_text, qty_text in levels:
            price = float(price_text)
            qty = float(qty_text)
            if qty <= 0:
                side.pop(price, None)
            else:
                side[price] = qty

    def _tick(self, event_time_ms: int, receive_time_ns: int, update_id: int | None, sequence_gap: bool) -> BookTick:
        best_bid = max(self.bids) if self.bids else None
        best_ask = min(self.asks) if self.asks else None
        return BookTick(
            symbol=self.symbol,
            event_time_ms=event_time_ms,
            receive_time_ns=receive_time_ns,
            best_bid=best_bid,
            best_bid_qty=self.bids.get(best_bid) if best_bid is not None else None,
            best_ask=best_ask,
            best_ask_qty=self.asks.get(best_ask) if best_ask is not None else None,
            update_id=update_id,
            sequence_gap=sequence_gap,
            stale=self.stale,
            venue=self.venue,
            source="l2_depth",
            book_status=self.status,
            resync_count=self.resync_count,
            sequence_gap_count=self.sequence_gap_count,
            stale_since_ns=self.stale_since_ns,
        )


class BinanceUsdmWebSocketMarketData:
    def __init__(
        self,
        symbol: str,
        snapshot_client: BinanceDepthSnapshotClient | None = None,
        reconnect_initial_backoff_s: float = 0.25,
        reconnect_max_backoff_s: float = 5.0,
    ) -> None:
        self.symbol = symbol.upper()
        self.book = LocalOrderBook(symbol=self.symbol)
        self.snapshot_client = snapshot_client or BinanceDepthSnapshotClient()
        self.needs_resync = False
        self.snapshot_error_count = 0
        self.last_snapshot_error = ""
        self.connection_error_count = 0
        self.last_connection_error = ""
        self.reconnect_count = 0
        self.reconnect_initial_backoff_s = reconnect_initial_backoff_s
        self.reconnect_max_backoff_s = reconnect_max_backoff_s
        self._snapshot_task: asyncio.Task[dict[str, Any]] | None = None
        self._last_snapshot_attempt_ns = 0

    @property
    def url(self) -> str:
        lower_symbol = self.symbol.lower()
        streams = "/".join(
            [
                f"{lower_symbol}@depth@100ms",
                f"{lower_symbol}@trade",
                f"{lower_symbol}@bookTicker",
            ]
        )
        return f"wss://fstream.binance.com/stream?streams={streams}"

    async def stream(self) -> AsyncIterator[BookTick | TopBookTick | TradeTick]:
        backoff_s = max(0.0, self.reconnect_initial_backoff_s)
        while True:
            try:
                async for event in self._stream_connection():
                    backoff_s = max(0.0, self.reconnect_initial_backoff_s)
                    yield event
            except Exception as exc:
                if not _is_retriable_stream_error(exc):
                    raise
                self.connection_error_count += 1
                self.last_connection_error = repr(exc)
            else:
                self.last_connection_error = "websocket stream ended"
            self.reconnect_count += 1
            self._reset_connection_state()
            if backoff_s > 0:
                await asyncio.sleep(backoff_s)
            backoff_s = min(
                max(0.0, self.reconnect_max_backoff_s),
                backoff_s * 2 if backoff_s > 0 else max(0.0, self.reconnect_initial_backoff_s),
            )

    async def _stream_connection(self) -> AsyncIterator[BookTick | TopBookTick | TradeTick]:
        try:
            import websockets
        except ModuleNotFoundError as exc:
            raise RuntimeError("Install websockets>=15.0 to use Binance market-data smoke") from exc

        async with websockets.connect(self.url, ping_interval=20, family=socket.AF_INET) as ws:
            async for message in ws:
                event = self.handle_message(message, receive_time_ns=time.monotonic_ns())
                if event is not None:
                    yield event
                snapshot_tick = self._complete_snapshot_if_ready()
                if snapshot_tick is not None:
                    yield snapshot_tick
                self._ensure_snapshot_task()

    def _reset_connection_state(self) -> None:
        if self._snapshot_task is not None and not self._snapshot_task.done():
            self._snapshot_task.cancel()
        self._snapshot_task = None
        self._last_snapshot_attempt_ns = 0
        self.needs_resync = False
        self.book = LocalOrderBook(symbol=self.symbol)

    def handle_message(
        self,
        message: str | bytes | dict[str, Any],
        receive_time_ns: int,
    ) -> BookTick | TopBookTick | TradeTick | None:
        payload = json.loads(message) if isinstance(message, (str, bytes, bytearray)) else message
        data = payload.get("data", payload)
        event_type = data.get("e")
        if event_type == "depthUpdate":
            tick = self.book.apply_depth_event(data, receive_time_ns=receive_time_ns)
            self.needs_resync = tick.stale or tick.sequence_gap
            return tick
        if event_type == "trade":
            return TradeTick(
                symbol=self.symbol,
                event_time_ms=int(data.get("E", 0)),
                receive_time_ns=receive_time_ns,
                price=float(data["p"]),
                qty=float(data["q"]),
                is_buyer_maker=bool(data["m"]),
                trade_id=data.get("t", ""),
            )
        if "bookTicker" in str(payload.get("stream", "")) or {"b", "B", "a", "A"}.issubset(data.keys()):
            return self.book.apply_book_ticker(data, receive_time_ns=receive_time_ns)
        return None

    def _ensure_snapshot_task(self) -> None:
        if self.book.status not in {BOOK_STATUS_BOOTSTRAPPING, BOOK_STATUS_REBUILDING}:
            return
        if self._snapshot_task is not None:
            return
        now_ns = time.monotonic_ns()
        if now_ns - self._last_snapshot_attempt_ns < 250_000_000:
            return
        self._last_snapshot_attempt_ns = now_ns
        self._snapshot_task = asyncio.create_task(self.snapshot_client.fetch_snapshot(self.symbol))
        self._snapshot_task.add_done_callback(_consume_task_exception)

    def _complete_snapshot_if_ready(self) -> BookTick | None:
        if self._snapshot_task is None or not self._snapshot_task.done():
            return None
        task = self._snapshot_task
        self._snapshot_task = None
        try:
            snapshot = task.result()
        except Exception as exc:
            self.snapshot_error_count += 1
            self.last_snapshot_error = repr(exc)
            return None
        tick = self.book.apply_snapshot(
            last_update_id=int(snapshot["lastUpdateId"]),
            bids=snapshot.get("bids", []),
            asks=snapshot.get("asks", []),
            event_time_ms=int(time.time() * 1000),
            receive_time_ns=time.monotonic_ns(),
        )
        self.needs_resync = tick.stale or tick.sequence_gap
        return tick


def _levels_to_map(levels: list[list[str]]) -> dict[float, float]:
    result: dict[float, float] = {}
    for price_text, qty_text in levels:
        qty = float(qty_text)
        if qty > 0:
            result[float(price_text)] = qty
    return result


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _is_retriable_stream_error(exc: Exception) -> bool:
    if isinstance(exc, (OSError, EOFError, TimeoutError, asyncio.TimeoutError)):
        return True
    module = exc.__class__.__module__
    name = exc.__class__.__name__
    if module.startswith("websockets.") and name.startswith("ConnectionClosed"):
        return True
    return False


class _IPv4HTTPSConnection(http.client.HTTPSConnection):
    def connect(self) -> None:
        raw_sock: socket.socket | None = None
        last_exc: OSError | None = None
        for family, socktype, proto, _canonname, address in socket.getaddrinfo(
            self.host,
            self.port,
            socket.AF_INET,
            socket.SOCK_STREAM,
        ):
            try:
                raw_sock = socket.socket(family, socktype, proto)
                raw_sock.settimeout(self.timeout)
                raw_sock.connect(address)
                break
            except OSError as exc:
                last_exc = exc
                if raw_sock is not None:
                    raw_sock.close()
                raw_sock = None
        if raw_sock is None:
            if last_exc is not None:
                raise last_exc
            raise OSError(f"could not resolve IPv4 address for {self.host}")
        self.sock = self._context.wrap_socket(raw_sock, server_hostname=self.host)


def _consume_task_exception(task: asyncio.Task[dict[str, Any]]) -> None:
    try:
        task.exception()
    except asyncio.CancelledError:
        return
