from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
import time
from pathlib import Path
from typing import Any, Protocol
from urllib.error import URLError
from urllib import parse, request

from langlang_trader.config import UniverseConfig
from langlang_trader.models import utc_now_iso


@dataclass(frozen=True)
class UniverseSymbol:
    symbol: str
    base_ccy: str
    quote_ccy: str
    inst_type: str
    state: str
    is_reference: bool
    tradable: bool
    filter_reason: str
    raw_payload: dict[str, Any]
    source_exchange: str = "okx"
    exchange_symbol: str = ""
    execution_symbol: str = ""
    observed_only: bool = False
    liquidity_usdt_24h: float | None = None
    liquidity_rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UniverseSnapshot:
    mode: str
    generated_at: str
    symbols: list[str]
    reference_symbols: list[str]
    rows: list[UniverseSymbol]
    raw_payload: dict[str, Any]
    observed_symbols: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rows"] = [row.to_dict() for row in self.rows]
        return payload


class UniverseProvider(Protocol):
    def list_symbols(self) -> UniverseSnapshot:
        ...


class StaticUniverseProvider:
    def __init__(
        self,
        *,
        symbols: list[str],
        reference_symbols: list[str] | None = None,
        mode: str = "static",
    ):
        self.symbols = list(symbols)
        self.reference_symbols = list(reference_symbols or [])
        self.mode = mode

    def list_symbols(self) -> UniverseSnapshot:
        rows: list[UniverseSymbol] = []
        for symbol in self.reference_symbols:
            rows.append(
                UniverseSymbol(
                    symbol=symbol,
                    base_ccy=_base_from_inst_id(symbol),
                    quote_ccy="USDT",
                    inst_type="SWAP",
                    state="live",
                    is_reference=True,
                    tradable=False,
                    filter_reason="reference_market_anchor",
                    raw_payload={"instId": symbol, "source": "static"},
                    source_exchange="static",
                    exchange_symbol=symbol,
                )
            )
        for symbol in self.symbols:
            rows.append(
                UniverseSymbol(
                    symbol=symbol,
                    base_ccy=_base_from_inst_id(symbol),
                    quote_ccy="USDT",
                    inst_type="SWAP",
                    state="live",
                    is_reference=False,
                    tradable=True,
                    filter_reason="",
                    raw_payload={"instId": symbol, "source": "static"},
                    source_exchange="static",
                    exchange_symbol=symbol,
                    execution_symbol=symbol,
                )
            )
        return UniverseSnapshot(
            mode=self.mode,
            generated_at=utc_now_iso(),
            symbols=list(self.symbols),
            reference_symbols=list(self.reference_symbols),
            rows=rows,
            raw_payload={"source": "static"},
            observed_symbols=list(dict.fromkeys([*self.reference_symbols, *self.symbols])),
        )


class OkxUniverseProvider:
    def __init__(
        self,
        *,
        config: UniverseConfig | None = None,
        base_url: str = "https://www.okx.com",
        timeout_seconds: float = 8.0,
        retries: int = 2,
    ):
        self.config = config or UniverseConfig(mode="okx_all_usdt_swap")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.retries = max(1, retries)
        self.headers = {
            "User-Agent": "langlang-trader/0.1",
            "Accept": "application/json",
            "Connection": "close",
        }

    def list_symbols(self) -> UniverseSnapshot:
        query = parse.urlencode({"instType": "SWAP"})
        payload = self._get_json(f"{self.base_url}/api/v5/public/instruments?{query}")
        if payload.get("code") != "0":
            raise RuntimeError(f"OKX instruments request failed: {payload}")
        tickers_payload = self._get_json(f"{self.base_url}/api/v5/market/tickers?{query}")
        return self.snapshot_from_payload(payload, config=self.config, tickers_payload=tickers_payload)

    @classmethod
    def snapshot_from_payload(
        cls,
        payload: dict[str, Any],
        *,
        config: UniverseConfig | None = None,
        tickers_payload: dict[str, Any] | None = None,
    ) -> UniverseSnapshot:
        cfg = config or UniverseConfig(mode="okx_all_usdt_swap")
        liquidity_by_symbol = _okx_liquidity_from_tickers(tickers_payload)
        liquidity_ranks = _liquidity_ranks(liquidity_by_symbol)
        reference_set = set(cfg.reference_symbols)
        reference_symbols: list[str] = []
        symbols: list[str] = []
        rows: list[UniverseSymbol] = []
        for item in payload.get("data", []):
            symbol = str(item.get("instId") or "")
            inst_type = str(item.get("instType") or "")
            state = str(item.get("state") or "")
            settle_ccy = str(item.get("settleCcy") or "")
            quote_ccy = str(item.get("quoteCcy") or "")
            base_ccy = str(item.get("baseCcy") or _base_from_inst_id(symbol))
            is_reference = symbol in reference_set
            reasons: list[str] = []
            if inst_type != "SWAP":
                reasons.append("not_swap")
            if state != "live":
                reasons.append("not_live")
            if settle_ccy != "USDT" and quote_ccy != "USDT":
                reasons.append("not_usdt_settled")
            if not symbol.endswith("-USDT-SWAP"):
                reasons.append("not_usdt_swap_symbol")
            tradable = not reasons and not (is_reference and cfg.exclude_reference_symbols)
            filter_reason = "|".join(reasons)
            if is_reference and not reasons:
                filter_reason = "reference_market_anchor" if cfg.exclude_reference_symbols else ""
            liquidity = liquidity_by_symbol.get(symbol)
            liquidity_rank = liquidity_ranks.get(symbol)
            if (
                cfg.liquidity_top_n > 0
                and liquidity_by_symbol
                and not is_reference
                and not reasons
                and (liquidity_rank is None or liquidity_rank > cfg.liquidity_top_n)
            ):
                reasons.append(f"liquidity_rank_gt_{cfg.liquidity_top_n}" if liquidity_rank else "liquidity_rank_missing")
                filter_reason = "|".join(reasons)
                tradable = False
            if is_reference and not reasons:
                reference_symbols.append(symbol)
            if tradable:
                symbols.append(symbol)
            rows.append(
                UniverseSymbol(
                    symbol=symbol,
                    base_ccy=base_ccy,
                    quote_ccy=quote_ccy or settle_ccy,
                    inst_type=inst_type,
                    state=state,
                    is_reference=is_reference,
                    tradable=tradable,
                    filter_reason=filter_reason,
                    raw_payload=dict(item),
                    source_exchange="okx",
                    exchange_symbol=symbol,
                    execution_symbol=symbol if tradable else "",
                    observed_only=False,
                    liquidity_usdt_24h=liquidity,
                    liquidity_rank=liquidity_rank,
                )
            )
        return UniverseSnapshot(
            mode=cfg.mode if cfg.mode != "static" else "okx_all_usdt_swap",
            generated_at=utc_now_iso(),
            symbols=symbols,
            reference_symbols=[symbol for symbol in cfg.reference_symbols if symbol in set(reference_symbols)],
            rows=rows,
            raw_payload=payload,
            observed_symbols=list(dict.fromkeys([*reference_symbols, *symbols])),
        )

    def _get_json(self, url: str) -> dict[str, Any]:
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


class BinanceUniverseProvider:
    def __init__(
        self,
        *,
        config: UniverseConfig | None = None,
        base_url: str = "https://fapi.binance.com",
        timeout_seconds: float = 8.0,
        retries: int = 2,
    ):
        self.config = config or UniverseConfig(mode="binance_usdt_perp_observe", provider="binance")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.retries = max(1, retries)
        self.headers = {
            "User-Agent": "langlang-trader/0.1",
            "Accept": "application/json",
            "Connection": "close",
        }

    def list_symbols(self) -> UniverseSnapshot:
        payload = self._get_json(f"{self.base_url}/fapi/v1/exchangeInfo")
        tickers_payload = self._get_json(f"{self.base_url}/fapi/v1/ticker/24hr")
        return self.snapshot_from_payload(payload, config=self.config, tickers_payload=tickers_payload)

    @classmethod
    def snapshot_from_payload(
        cls,
        payload: dict[str, Any],
        *,
        config: UniverseConfig | None = None,
        tickers_payload: list[dict[str, Any]] | None = None,
    ) -> UniverseSnapshot:
        cfg = config or UniverseConfig(mode="binance_usdt_perp_observe", provider="binance")
        liquidity_by_symbol = _binance_liquidity_from_tickers(tickers_payload)
        liquidity_ranks = _liquidity_ranks(liquidity_by_symbol)
        reference_set = set(cfg.reference_symbols)
        reference_symbols: list[str] = []
        symbols: list[str] = []
        observed_symbols: list[str] = []
        rows: list[UniverseSymbol] = []
        for item in payload.get("symbols", []):
            exchange_symbol = str(item.get("symbol") or item.get("pair") or "")
            base_ccy = str(item.get("baseAsset") or _base_from_binance_symbol(exchange_symbol))
            quote_ccy = str(item.get("quoteAsset") or "")
            canonical_symbol = _canonical_swap_symbol(base_ccy, quote_ccy)
            status = str(item.get("status") or "")
            contract_type = str(item.get("contractType") or "")
            is_reference = canonical_symbol in reference_set
            reasons: list[str] = []
            if status != "TRADING":
                reasons.append("not_trading")
            if contract_type != "PERPETUAL":
                reasons.append("not_perpetual")
            if quote_ccy != "USDT":
                reasons.append("not_usdt_quote")
            if not canonical_symbol:
                reasons.append("invalid_symbol")
            tradable = not reasons and not (is_reference and cfg.exclude_reference_symbols)
            filter_reason = "|".join(reasons)
            if is_reference and not reasons:
                filter_reason = "reference_market_anchor" if cfg.exclude_reference_symbols else ""
            liquidity = liquidity_by_symbol.get(canonical_symbol)
            liquidity_rank = liquidity_ranks.get(canonical_symbol)
            if (
                cfg.liquidity_top_n > 0
                and liquidity_by_symbol
                and not is_reference
                and not reasons
                and (liquidity_rank is None or liquidity_rank > cfg.liquidity_top_n)
            ):
                reasons.append(f"liquidity_rank_gt_{cfg.liquidity_top_n}" if liquidity_rank else "liquidity_rank_missing")
                filter_reason = "|".join(reasons)
                tradable = False
            if is_reference and not reasons:
                reference_symbols.append(canonical_symbol)
            if not reasons:
                observed_symbols.append(canonical_symbol)
            if tradable:
                symbols.append(canonical_symbol)
            rows.append(
                UniverseSymbol(
                    symbol=canonical_symbol,
                    base_ccy=base_ccy,
                    quote_ccy=quote_ccy,
                    inst_type="PERPETUAL",
                    state=status,
                    is_reference=is_reference,
                    tradable=tradable,
                    filter_reason=filter_reason,
                    raw_payload=dict(item),
                    source_exchange="binance",
                    exchange_symbol=exchange_symbol,
                    execution_symbol="",
                    observed_only=False,
                    liquidity_usdt_24h=liquidity,
                    liquidity_rank=liquidity_rank,
                )
            )
        return UniverseSnapshot(
            mode=cfg.mode if cfg.mode != "static" else "binance_usdt_perp_observe",
            generated_at=utc_now_iso(),
            symbols=list(dict.fromkeys(symbols)),
            reference_symbols=[symbol for symbol in cfg.reference_symbols if symbol in set(reference_symbols)],
            rows=rows,
            raw_payload=payload,
            observed_symbols=list(dict.fromkeys(observed_symbols)),
        )

    def _get_json(self, url: str) -> dict[str, Any]:
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


class OkxBinanceUniverseProvider:
    def __init__(
        self,
        *,
        config: UniverseConfig | None = None,
        okx_base_url: str = "https://www.okx.com",
        binance_base_url: str = "https://fapi.binance.com",
    ):
        self.config = config or UniverseConfig(mode="okx_binance_usdt_swap_observe", provider="okx_binance")
        self.okx_provider = OkxUniverseProvider(config=UniverseConfig(mode="okx_all_usdt_swap"), base_url=okx_base_url)
        self.binance_provider = BinanceUniverseProvider(
            config=UniverseConfig(mode="binance_usdt_perp_observe", provider="binance"),
            base_url=binance_base_url,
        )

    def list_symbols(self) -> UniverseSnapshot:
        okx_snapshot = self.okx_provider.list_symbols()
        binance_snapshot = self.binance_provider.list_symbols()
        return self.snapshot_from_snapshots(okx_snapshot, binance_snapshot, config=self.config)

    @classmethod
    def snapshot_from_payloads(
        cls,
        okx_payload: dict[str, Any],
        binance_payload: dict[str, Any],
        *,
        config: UniverseConfig | None = None,
        okx_tickers_payload: dict[str, Any] | None = None,
        binance_tickers_payload: list[dict[str, Any]] | None = None,
        liquidity_top_n: int | None = None,
    ) -> UniverseSnapshot:
        cfg = config or UniverseConfig(mode="okx_binance_usdt_swap_observe", provider="okx_binance")
        if liquidity_top_n is not None:
            cfg = replace(cfg, liquidity_top_n=liquidity_top_n)
        okx_snapshot = OkxUniverseProvider.snapshot_from_payload(
            okx_payload,
            config=UniverseConfig(
                mode="okx_all_usdt_swap",
                provider="okx",
                reference_symbols=cfg.reference_symbols,
                exclude_reference_symbols=cfg.exclude_reference_symbols,
                liquidity_top_n=0,
            ),
            tickers_payload=okx_tickers_payload,
        )
        binance_snapshot = BinanceUniverseProvider.snapshot_from_payload(
            binance_payload,
            config=UniverseConfig(
                mode="binance_usdt_perp_observe",
                provider="binance",
                reference_symbols=cfg.reference_symbols,
                exclude_reference_symbols=cfg.exclude_reference_symbols,
                liquidity_top_n=0,
            ),
            tickers_payload=binance_tickers_payload,
        )
        return cls.snapshot_from_snapshots(okx_snapshot, binance_snapshot, config=cfg)

    @classmethod
    def snapshot_from_snapshots(
        cls,
        okx_snapshot: UniverseSnapshot,
        binance_snapshot: UniverseSnapshot,
        *,
        config: UniverseConfig | None = None,
    ) -> UniverseSnapshot:
        cfg = config or UniverseConfig(mode="okx_binance_usdt_swap_observe", provider="okx_binance")
        liquidity_by_symbol = _combined_liquidity(okx_snapshot, binance_snapshot)
        liquidity_ranks = _liquidity_ranks(liquidity_by_symbol)
        okx_executable = set(okx_snapshot.symbols)
        okx_references = set(okx_snapshot.reference_symbols)
        combined_rows = [
            _apply_liquidity_filter(
                row,
                liquidity_by_symbol=liquidity_by_symbol,
                liquidity_ranks=liquidity_ranks,
                top_n=cfg.liquidity_top_n,
            )
            for row in okx_snapshot.rows
        ]
        valid_binance_observed: list[str] = []
        binance_only_observed: list[str] = []
        overlap_observed: list[str] = []
        for row in binance_snapshot.rows:
            valid = row.filter_reason in {"", "reference_market_anchor"}
            observed_only = False
            tradable = False
            execution_symbol = ""
            filter_reason = row.filter_reason
            if valid and row.symbol not in okx_executable and row.symbol not in okx_references:
                observed_only = True
                filter_reason = "binance_observed_only_not_okx_executable"
                binance_only_observed.append(row.symbol)
            elif valid and row.symbol in okx_executable:
                filter_reason = "okx_executable_overlap"
                execution_symbol = row.symbol
                overlap_observed.append(row.symbol)
            elif valid and row.symbol in okx_references:
                filter_reason = "reference_market_anchor"
                execution_symbol = row.symbol
            if valid:
                valid_binance_observed.append(row.symbol)
            combined_rows.append(_apply_liquidity_filter(
                UniverseSymbol(
                    symbol=row.symbol,
                    base_ccy=row.base_ccy,
                    quote_ccy=row.quote_ccy,
                    inst_type=row.inst_type,
                    state=row.state,
                    is_reference=row.is_reference,
                    tradable=tradable,
                    filter_reason=filter_reason,
                    raw_payload=row.raw_payload,
                    source_exchange=row.source_exchange,
                    exchange_symbol=row.exchange_symbol,
                    execution_symbol=execution_symbol,
                    observed_only=observed_only,
                    liquidity_usdt_24h=liquidity_by_symbol.get(row.symbol),
                    liquidity_rank=liquidity_ranks.get(row.symbol),
                ),
                liquidity_by_symbol=liquidity_by_symbol,
                liquidity_ranks=liquidity_ranks,
                top_n=cfg.liquidity_top_n,
            )
            )
        observed_symbols = sorted({row.symbol for row in combined_rows if _is_observable_after_filters(row)})
        executable_symbols = sorted({row.symbol for row in combined_rows if row.source_exchange == "okx" and row.tradable})
        liquidity_excluded = sorted(
            {
                row.symbol
                for row in combined_rows
                if "liquidity_rank_gt_" in row.filter_reason or "liquidity_rank_missing" in row.filter_reason
            }
        )
        summary = {
            "okx_executable_count": len(executable_symbols),
            "okx_reference_count": len(okx_snapshot.reference_symbols),
            "binance_observed_count": len({symbol for symbol in valid_binance_observed if symbol in set(observed_symbols)}),
            "binance_only_observed_count": len({symbol for symbol in binance_only_observed if symbol in set(observed_symbols)}),
            "okx_binance_overlap_count": len({symbol for symbol in overlap_observed if symbol in set(observed_symbols)}),
            "combined_observed_count": len(observed_symbols),
            "liquidity_filter_top_n": cfg.liquidity_top_n,
            "liquidity_excluded_count": len(liquidity_excluded),
        }
        return UniverseSnapshot(
            mode=cfg.mode if cfg.mode != "static" else "okx_binance_usdt_swap_observe",
            generated_at=utc_now_iso(),
            symbols=executable_symbols,
            reference_symbols=list(dict.fromkeys([*okx_snapshot.reference_symbols, *binance_snapshot.reference_symbols])),
            rows=combined_rows,
            raw_payload={
                "summary": summary,
                "okx": okx_snapshot.raw_payload,
                "binance": binance_snapshot.raw_payload,
            },
            observed_symbols=observed_symbols,
        )


def write_universe_snapshot(path: str | Path, snapshot: UniverseSnapshot) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def read_universe_snapshot(path: str | Path) -> UniverseSnapshot:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [UniverseSymbol(**row) for row in payload.get("rows", [])]
    return UniverseSnapshot(
        mode=payload["mode"],
        generated_at=payload["generated_at"],
        symbols=list(payload.get("symbols", [])),
        reference_symbols=list(payload.get("reference_symbols", [])),
        rows=rows,
        raw_payload=dict(payload.get("raw_payload", {})),
        observed_symbols=list(payload.get("observed_symbols", [])),
    )


def _okx_liquidity_from_tickers(payload: dict[str, Any] | None) -> dict[str, float]:
    if not payload or payload.get("code") not in {None, "0"}:
        return {}
    values: dict[str, float] = {}
    for row in payload.get("data", []):
        symbol = str(row.get("instId") or "")
        if not symbol:
            continue
        value = _float_or_zero(row.get("volCcy24h") or row.get("vol24h") or row.get("volUsd24h"))
        if value > 0:
            values[symbol] = value
    return values


def _binance_liquidity_from_tickers(payload: list[dict[str, Any]] | None) -> dict[str, float]:
    if not payload:
        return {}
    values: dict[str, float] = {}
    for row in payload:
        exchange_symbol = str(row.get("symbol") or "")
        if not exchange_symbol.endswith("USDT"):
            continue
        canonical = _canonical_swap_symbol(_base_from_binance_symbol(exchange_symbol), "USDT")
        value = _float_or_zero(row.get("quoteVolume") or row.get("volume"))
        if canonical and value > 0:
            values[canonical] = value
    return values


def _combined_liquidity(okx_snapshot: UniverseSnapshot, binance_snapshot: UniverseSnapshot) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in [*okx_snapshot.rows, *binance_snapshot.rows]:
        if row.is_reference or row.liquidity_usdt_24h is None:
            continue
        values[row.symbol] = max(values.get(row.symbol, 0.0), float(row.liquidity_usdt_24h))
    return values


def _liquidity_ranks(values: dict[str, float]) -> dict[str, int]:
    return {
        symbol: idx
        for idx, (symbol, _) in enumerate(sorted(values.items(), key=lambda item: (-item[1], item[0])), start=1)
    }


def _apply_liquidity_filter(
    row: UniverseSymbol,
    *,
    liquidity_by_symbol: dict[str, float],
    liquidity_ranks: dict[str, int],
    top_n: int,
) -> UniverseSymbol:
    liquidity = liquidity_by_symbol.get(row.symbol)
    rank = liquidity_ranks.get(row.symbol)
    if top_n <= 0 or not liquidity_by_symbol or row.is_reference:
        return replace(row, liquidity_usdt_24h=liquidity, liquidity_rank=rank)
    if not _is_liquidity_filter_candidate(row):
        return replace(row, liquidity_usdt_24h=liquidity, liquidity_rank=rank)
    if rank is not None and rank <= top_n:
        return replace(row, liquidity_usdt_24h=liquidity, liquidity_rank=rank)
    reason = f"liquidity_rank_gt_{top_n}" if rank is not None else "liquidity_rank_missing"
    filter_reason = _append_filter_reason(row.filter_reason, reason)
    return replace(
        row,
        tradable=False,
        execution_symbol="",
        observed_only=False,
        filter_reason=filter_reason,
        liquidity_usdt_24h=liquidity,
        liquidity_rank=rank,
    )


def _is_liquidity_filter_candidate(row: UniverseSymbol) -> bool:
    return row.tradable or row.filter_reason in {
        "",
        "okx_executable_overlap",
        "binance_observed_only_not_okx_executable",
    }


def _is_observable_after_filters(row: UniverseSymbol) -> bool:
    if row.is_reference:
        return True
    if "liquidity_rank_gt_" in row.filter_reason or "liquidity_rank_missing" in row.filter_reason:
        return False
    return row.tradable or row.filter_reason in {"okx_executable_overlap", "binance_observed_only_not_okx_executable"}


def _append_filter_reason(existing: str, reason: str) -> str:
    parts = [part for part in existing.split("|") if part]
    if reason not in parts:
        parts.append(reason)
    return "|".join(parts)


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _base_from_inst_id(symbol: str) -> str:
    return symbol.split("-", 1)[0] if symbol else ""


def _base_from_binance_symbol(symbol: str) -> str:
    return symbol[:-4] if symbol.endswith("USDT") else symbol


def _canonical_swap_symbol(base_ccy: str, quote_ccy: str) -> str:
    if not base_ccy or not quote_ccy:
        return ""
    return f"{base_ccy}-{quote_ccy}-SWAP"
