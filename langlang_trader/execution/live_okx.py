from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
import time
from typing import Any
from urllib.error import URLError
from urllib import request

from langlang_trader.config import ExecutionConfig
from langlang_trader.models import AccountSnapshot, OrderIntent, OrderResult, Position


class OkxRestTransport:
    def __init__(self, *, api_key: str, api_secret: str, passphrase: str, base_url: str = "https://www.okx.com"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.base_url = base_url.rstrip("/")

    @classmethod
    def from_env(cls) -> "OkxRestTransport":
        api_key = os.getenv("OKX_API_KEY", "")
        api_secret = os.getenv("OKX_API_SECRET", "")
        passphrase = os.getenv("OKX_API_PASSPHRASE", "")
        if not api_key or not api_secret or not passphrase:
            raise PermissionError("OKX_API_KEY, OKX_API_SECRET and OKX_API_PASSPHRASE are required for live mode")
        return cls(
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase,
            base_url=os.getenv("OKX_BASE_URL", "https://www.okx.com"),
        )

    def request(self, method: str, path: str, body: dict[str, Any] | None = None, auth: bool = False) -> dict[str, Any]:
        body_text = "" if body is None else json.dumps(body, separators=(",", ":"))
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "langlang-trader/0.1",
            "Accept": "application/json",
            "Connection": "close",
        }
        if auth:
            timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
            sign_text = f"{timestamp}{method.upper()}{path}{body_text}"
            digest = hmac.new(self.api_secret.encode(), sign_text.encode(), hashlib.sha256).digest()
            headers.update(
                {
                    "OK-ACCESS-KEY": self.api_key,
                    "OK-ACCESS-SIGN": base64.b64encode(digest).decode(),
                    "OK-ACCESS-TIMESTAMP": timestamp,
                    "OK-ACCESS-PASSPHRASE": self.passphrase,
                }
            )
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                req = request.Request(
                    f"{self.base_url}{path}",
                    data=None if body is None else body_text.encode(),
                    headers=headers,
                    method=method.upper(),
                )
                with request.urlopen(req, timeout=15) as response:
                    return json.loads(response.read().decode("utf-8"))
            except URLError as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"OKX request failed after retries: {path}") from last_error


class OkxLiveExecutor:
    def __init__(self, *, config: ExecutionConfig, transport: OkxRestTransport | Any | None = None):
        self.config = config
        self.transport = transport

    def get_account(self) -> AccountSnapshot:
        self._require_live_readiness()
        payload = self._transport().request("GET", "/api/v5/account/balance", auth=True)
        data = (payload.get("data") or [{}])[0]
        total_equity = float(data.get("totalEq") or 0.0)
        return AccountSnapshot(
            equity_usdt=total_equity,
            cash_usdt=total_equity,
            margin_used_usdt=0.0,
            realized_pnl_usdt=0.0,
        )

    def get_positions(self) -> list[Position]:
        self._require_live_readiness()
        self._transport().request("GET", "/api/v5/account/positions", auth=True)
        return []

    def place_order(self, intent: OrderIntent) -> OrderResult:
        self._require_live_readiness()
        body = {
            "instId": intent.symbol,
            "tdMode": "isolated",
            "side": intent.side.okx_order_side,
            "ordType": intent.order_type,
            "sz": self._fmt(intent.qty),
            "reduceOnly": "true" if intent.reduce_only else "false",
        }
        payload = self._transport().request("POST", "/api/v5/trade/order", body=body, auth=True)
        row = (payload.get("data") or [{}])[0]
        ok = payload.get("code") == "0" and row.get("sCode") in {None, "", "0"}
        status = "accepted" if ok else "rejected"
        return OrderResult(
            exchange_order_id=str(row.get("ordId") or ""),
            status=status,
            filled_qty=0.0,
            avg_price=None,
            fee=0.0,
            raw_payload=payload,
        )

    def cancel_order(self, order_id: str) -> OrderResult:
        self._require_live_readiness()
        payload = self._transport().request("POST", "/api/v5/trade/cancel-order", body={"ordId": order_id}, auth=True)
        return OrderResult(
            exchange_order_id=order_id,
            status="cancel_requested",
            filled_qty=0.0,
            avg_price=None,
            fee=0.0,
            raw_payload=payload,
        )

    def sync_fills(self) -> list[OrderResult]:
        self._require_live_readiness()
        self._transport().request("GET", "/api/v5/trade/fills", auth=True)
        return []

    def close_position(self, symbol: str, reason: str) -> OrderResult:
        self._require_live_readiness()
        payload = self._transport().request(
            "POST",
            "/api/v5/trade/close-position",
            body={"instId": symbol, "mgnMode": "isolated"},
            auth=True,
        )
        return OrderResult(
            exchange_order_id=f"close-position:{symbol}",
            status="close_requested",
            filled_qty=0.0,
            avg_price=None,
            fee=0.0,
            raw_payload={"reason": reason, "okx": payload},
        )

    def _transport(self) -> OkxRestTransport | Any:
        if self.transport is None:
            self.transport = OkxRestTransport.from_env()
        return self.transport

    def _require_live_readiness(self) -> None:
        if self.config.mode != "live" or self.config.exchange != "okx" or self.config.executor != "live_okx":
            raise PermissionError("live OKX execution requires mode=live, exchange=okx and executor=live_okx")
        if not self.config.allow_live_orders:
            raise PermissionError("live OKX execution requires allow_live_orders=true")
        if self.transport is None:
            self.transport = OkxRestTransport.from_env()

    @staticmethod
    def _fmt(value: float) -> str:
        return f"{value:.12g}"
