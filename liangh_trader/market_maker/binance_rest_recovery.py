from __future__ import annotations

import hashlib
import hmac
from urllib.parse import urlencode


RECOVERY_CONTEXTS = {"startup", "reconnect", "emergency", "listen_key_keepalive"}


class RecoveryContextError(RuntimeError):
    pass


class BinanceRestRecoveryClient:
    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://fapi.binance.com") -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")

    def build_start_listen_key_request(self, context: str) -> dict[str, object]:
        self._assert_recovery_context(context)
        return {
            "method": "POST",
            "path": "/fapi/v1/listenKey",
            "headers": {"X-MBX-APIKEY": self.api_key},
            "params": {},
            "context": context,
        }

    def build_keepalive_listen_key_request(self, listen_key: str, context: str = "listen_key_keepalive") -> dict[str, object]:
        self._assert_recovery_context(context)
        return {
            "method": "PUT",
            "path": "/fapi/v1/listenKey",
            "headers": {"X-MBX-APIKEY": self.api_key},
            "params": {"listenKey": listen_key},
            "context": context,
        }

    def build_open_orders_request(self, symbol: str, context: str, timestamp_ms: int) -> dict[str, object]:
        self._assert_recovery_context(context)
        params = {"symbol": symbol.upper(), "timestamp": int(timestamp_ms)}
        params["signature"] = self._sign(params)
        return {
            "method": "GET",
            "path": "/fapi/v1/openOrders",
            "headers": {"X-MBX-APIKEY": self.api_key},
            "params": params,
            "context": context,
        }

    def build_cancel_all_request(self, symbol: str, context: str, timestamp_ms: int) -> dict[str, object]:
        self._assert_recovery_context(context)
        params = {"symbol": symbol.upper(), "timestamp": int(timestamp_ms)}
        params["signature"] = self._sign(params)
        return {
            "method": "DELETE",
            "path": "/fapi/v1/allOpenOrders",
            "headers": {"X-MBX-APIKEY": self.api_key},
            "params": params,
            "context": context,
        }

    def _assert_recovery_context(self, context: str) -> None:
        if context not in RECOVERY_CONTEXTS:
            raise RecoveryContextError(f"REST recovery is not allowed in context={context}")

    def _sign(self, params: dict[str, object]) -> str:
        query = urlencode(sorted((key, str(value)) for key, value in params.items()))
        return hmac.new(self.api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
