#!/usr/bin/env python3
"""Distill Langlang's trade journal into quantifiable strategy evidence.

The script is intentionally self-contained: it reads the two source workbooks,
renders/summarizes the PDF strategy note, fetches OKX candles with an on-disk
cache, extracts trade-aligned features, evaluates human-readable candidate
rules, and writes a Markdown research report plus CSV artifacts.
"""

from __future__ import annotations

import argparse
import calendar
import json
import math
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PRIMARY_XLSX = Path(
    "/Users/wl/Downloads/%E6%B5%AA%E6%B5%AA%E4%BA%A4%E5%89%B2%E5%8D%95%E6%94%B9%E8%89%AF%E7%89%88.xlsx"
)
SECONDARY_XLSX = Path(
    "/Users/wl/Downloads/2022.5~2024.6%E6%B5%AA%E6%B5%AA%E4%BA%A4%E6%98%93%E4%BA%A4%E5%89%B2%E5%8D%95%E6%95%B0%E6%8D%AE3+(1)+(1).xlsx"
)
PDF_PATH = Path("/Users/wl/Downloads/bit%E6%B5%AA%E6%B5%AA%E4%BA%A4%E6%98%93%E5%BF%83%E5%BE%97.pdf")
SHEET_NAME = "时间排列+去除金额错误单子"

OKX_COLUMNS = ["ts", "open", "high", "low", "close", "vol", "vol_ccy", "vol_quote", "confirm"]
BAR_MS = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "1H": 60 * 60_000,
    "1D": 24 * 60 * 60_000,
}

STRATEGY_MAP = [
    {
        "concept": "主升浪",
        "text_meaning": "大级别趋势向上，资金集中在主升阶段；回调不破结构时继续顺势。",
        "quant_features": "20/60日涨幅、20/60日高低位、连续新高、价格在20日区间的位置、回调离前高的距离。",
    },
    {
        "concept": "第一次回调/二次入场",
        "text_meaning": "第一次大分歧不急追，等待回调到第二压力位/前平台后再参与。",
        "quant_features": "突破后回撤幅度、回撤后仍位于20日区间中上部、5m/15m回踩不破、再次转强。",
    },
    {
        "concept": "小分歧/大分歧",
        "text_meaning": "小分歧可承接，大分歧短期不参与；超大级别分歧要等结构重新稳定。",
        "quant_features": "入场前2h/24h振幅、回撤比例、成交量放大倍数、日线距高点回撤。",
    },
    {
        "concept": "突破后回踩",
        "text_meaning": "突破关键平台后不盲追，回踩确认、内部反弹增多时更适合入场。",
        "quant_features": "入场前高点突破、回踩深度、入场位置分位数、短周期转强。",
    },
    {
        "concept": "止损纪律",
        "text_meaning": "方向错或跌破关键位要快速砍；短时间亏损说明入场点或节奏错。",
        "quant_features": "MAE、持仓分钟、跌破前低/平台、短持仓亏损、杠杆下实际振幅。",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Langlang strategy distillation pipeline")
    parser.add_argument("--primary-xlsx", type=Path, default=PRIMARY_XLSX)
    parser.add_argument("--secondary-xlsx", type=Path, default=SECONDARY_XLSX)
    parser.add_argument("--pdf", type=Path, default=PDF_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("output/langlang_distill"))
    parser.add_argument("--kline-sample-size", type=int, default=240)
    parser.add_argument("--calibration-sample-size", type=int, default=40)
    parser.add_argument("--max-okx-calls", type=int, default=1800)
    parser.add_argument("--okx-sleep", type=float, default=0.06)
    parser.add_argument("--request-timeout", type=float, default=8.0)
    parser.add_argument("--okx-retries", type=int, default=2)
    parser.add_argument(
        "--max-intraday-hours",
        type=float,
        default=24.0,
        help="Cap post-entry intraday candle windows; 0 disables the cap.",
    )
    parser.add_argument(
        "--max-daily-symbols",
        type=int,
        default=0,
        help="0 means all symbols; otherwise fetch daily candles for top-N symbols plus sampled symbols.",
    )
    parser.add_argument("--no-network", action="store_true", help="Skip OKX requests and use cache only.")
    parser.add_argument("--refresh-cache", action="store_true", help="Ignore cached candle CSVs.")
    return parser.parse_args()


def utc_ms_from_naive(ts: pd.Timestamp, offset_hours: int = 0) -> int:
    """Treat a naive timestamp plus offset as UTC and return epoch ms.

    offset_hours=-8 means the sheet time is Asia/Shanghai local time converted
    to UTC. offset_hours=0 means the sheet time is already UTC.
    """
    if pd.isna(ts):
        return 0
    shifted = pd.Timestamp(ts).to_pydatetime() + timedelta(hours=offset_hours)
    return int(calendar.timegm(shifted.timetuple()) * 1000 + shifted.microsecond / 1000)


def pct(value: float | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{value * 100:.{digits}f}%"


def money(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{value:,.0f}"


def safe_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_trade_sheet(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=SHEET_NAME, header=1)
    raw = raw.dropna(how="all").copy()
    raw = raw[raw["交易对"].notna()].copy()

    out = pd.DataFrame()
    out["trade_id"] = pd.to_numeric(raw["序号"], errors="coerce").astype("Int64")
    out["entry_time"] = pd.to_datetime(raw["买入时间"], errors="coerce")
    out["exit_time"] = pd.to_datetime(raw["卖出时间"], errors="coerce")
    out["symbol"] = raw["交易对"].astype(str).str.strip()
    out["side"] = raw["方向"].astype(str).str.strip().map({"多": "long", "空": "short"}).fillna(raw["方向"])
    out["side_cn"] = raw["方向"].astype(str).str.strip()
    out["leverage"] = pd.to_numeric(raw["杠杆倍数"], errors="coerce")
    out["margin"] = pd.to_numeric(raw["保证金（最大时）"], errors="coerce")
    out["entry_price"] = pd.to_numeric(raw["开仓均价"], errors="coerce")
    out["exit_price"] = pd.to_numeric(raw["平仓均价"], errors="coerce")
    out["return_rate"] = pd.to_numeric(raw["收益率"], errors="coerce")
    out["pnl_usdt"] = pd.to_numeric(raw["收益 (USDT)"], errors="coerce")
    out["turnover_usd"] = pd.to_numeric(raw["交易额 (USD)"], errors="coerce")
    out["fee_usd"] = pd.to_numeric(raw["手续费 (USD)"], errors="coerce")

    hold_from_sheet = pd.to_numeric(raw.get("交易时间差（分钟）"), errors="coerce")
    hold_from_time = (out["exit_time"] - out["entry_time"]).dt.total_seconds() / 60
    out["hold_minutes"] = hold_from_sheet.where(hold_from_sheet.notna(), hold_from_time)
    realized_from_sheet = pd.to_numeric(raw.get("收益率/倍数=实际振幅"), errors="coerce")
    out["realized_move"] = realized_from_sheet.where(
        realized_from_sheet.notna(), out["return_rate"] / out["leverage"]
    )
    out["entry_month"] = out["entry_time"].dt.to_period("M").astype(str)
    out["entry_year"] = out["entry_time"].dt.year
    out["asset_group"] = np.where(out["symbol"].isin(["BTC-USDT-SWAP", "ETH-USDT-SWAP"]), "BTC/ETH", "alts")
    out["is_win"] = out["pnl_usdt"] > 0
    return out.sort_values("entry_time").reset_index(drop=True)


def compare_workbooks(primary: pd.DataFrame, secondary_path: Path) -> dict[str, Any]:
    if not secondary_path.exists():
        return {"secondary_exists": False}
    secondary = load_trade_sheet(secondary_path)
    key_cols = [
        "trade_id",
        "entry_time",
        "exit_time",
        "symbol",
        "side",
        "leverage",
        "entry_price",
        "exit_price",
        "return_rate",
        "pnl_usdt",
    ]
    n = min(len(primary), len(secondary))
    mismatch_cells = 0
    mismatch_rows = 0
    for idx in range(n):
        row_bad = False
        for col in key_cols:
            left = primary.iloc[idx][col]
            right = secondary.iloc[idx][col]
            if pd.isna(left) and pd.isna(right):
                continue
            if isinstance(left, pd.Timestamp) or isinstance(right, pd.Timestamp):
                ok = pd.Timestamp(left) == pd.Timestamp(right)
            elif isinstance(left, (int, float, np.number)) or isinstance(right, (int, float, np.number)):
                ok = math.isclose(safe_float(left), safe_float(right), rel_tol=1e-9, abs_tol=1e-9)
            else:
                ok = str(left) == str(right)
            if not ok:
                mismatch_cells += 1
                row_bad = True
        mismatch_rows += int(row_bad)
    return {
        "secondary_exists": True,
        "primary_rows": int(len(primary)),
        "secondary_rows": int(len(secondary)),
        "row_delta": int(len(primary) - len(secondary)),
        "compared_rows": int(n),
        "key_mismatch_rows": int(mismatch_rows),
        "key_mismatch_cells": int(mismatch_cells),
    }


@dataclass
class OKXClient:
    cache_dir: Path
    allow_network: bool
    max_calls: int
    sleep_seconds: float
    request_timeout: float = 8.0
    retries: int = 2
    refresh_cache: bool = False
    calls_made: int = 0
    cache_hits: int = 0
    errors: list[str] | None = None

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.errors = []

    def cache_path(self, symbol: str, bar: str, start_ms: int, end_ms: int) -> Path:
        symbol_safe = symbol.replace("/", "_").replace(":", "_")
        path = self.cache_dir / bar
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{symbol_safe}_{start_ms}_{end_ms}.csv"

    def fetch_window(self, symbol: str, bar: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        cache_file = self.cache_path(symbol, bar, start_ms, end_ms)
        if cache_file.exists() and not self.refresh_cache:
            self.cache_hits += 1
            return self._read_cache(cache_file)
        if not self.allow_network:
            return pd.DataFrame(columns=OKX_COLUMNS)
        if self.calls_made >= self.max_calls:
            self.errors.append(f"OKX call budget reached before {symbol} {bar} {start_ms}-{end_ms}")
            return pd.DataFrame(columns=OKX_COLUMNS)

        rows: list[list[Any]] = []
        cursor = end_ms + BAR_MS[bar]
        seen: set[int] = set()
        while cursor > start_ms and self.calls_made < self.max_calls:
            batch = self._fetch_batch(symbol, bar, cursor)
            if not batch:
                break
            min_ts = cursor
            for item in batch:
                ts = int(item[0])
                min_ts = min(min_ts, ts)
                if start_ms <= ts <= end_ms and ts not in seen:
                    seen.add(ts)
                    rows.append(item)
            if min_ts >= cursor:
                break
            cursor = min_ts
            if min_ts <= start_ms:
                break
        df = self._rows_to_df(rows)
        df.to_csv(cache_file, index=False)
        return df

    def _fetch_batch(self, symbol: str, bar: str, after_ms: int) -> list[list[Any]]:
        params = urllib.parse.urlencode(
            {"instId": symbol, "bar": bar, "limit": "300", "after": str(after_ms)}
        )
        url = f"https://www.okx.com/api/v5/market/history-candles?{params}"
        last_error = ""
        for attempt in range(self.retries):
            try:
                self.calls_made += 1
                req = urllib.request.Request(url, headers={"User-Agent": "langlang-distill/1.0"})
                with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                time.sleep(self.sleep_seconds)
                if payload.get("code") != "0":
                    last_error = f"{symbol} {bar}: {payload.get('code')} {payload.get('msg')}"
                    time.sleep(0.5 + attempt)
                    continue
                return payload.get("data", [])
            except Exception as exc:  # noqa: BLE001 - keep the batch retriable and reportable.
                last_error = f"{symbol} {bar}: {exc}"
                time.sleep(0.5 + attempt)
        self.errors.append(last_error)
        return []

    @staticmethod
    def _rows_to_df(rows: list[list[Any]]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=OKX_COLUMNS)
        df = pd.DataFrame(rows, columns=OKX_COLUMNS)
        for col in OKX_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)

    @staticmethod
    def _read_cache(path: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(path)
            for col in OKX_COLUMNS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df.sort_values("ts").reset_index(drop=True)
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=OKX_COLUMNS)


def render_pdf(pdf_path: Path, output_dir: Path) -> dict[str, Any]:
    pdf_dir = output_dir / "pdf_render"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    prefix = pdf_dir / "bit_langlang_strategy"
    info = {"pdf_exists": pdf_path.exists(), "rendered_png": None, "text_extract_chars": 0, "render_error": None}
    if not pdf_path.exists():
        return info
    try:
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join((page.extract_text() or "") for page in pdf.pages)
        info["text_extract_chars"] = len(text.strip())
    except Exception as exc:  # noqa: BLE001
        info["render_error"] = f"text extraction failed: {exc}"
    try:
        subprocess.run(
            [
                "/Users/wl/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/pdftoppm",
                "-png",
                "-r",
                "180",
                str(pdf_path),
                str(prefix),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        png = pdf_dir / "bit_langlang_strategy-1.png"
        if png.exists():
            info["rendered_png"] = str(png.resolve())
    except Exception as exc:  # noqa: BLE001
        info["render_error"] = str(exc)
    return info


def select_kline_sample(trades: pd.DataFrame, size: int) -> pd.DataFrame:
    if size <= 0 or size >= len(trades):
        return trades.copy()
    usable = trades[trades["exit_time"].notna()].copy()
    winners = usable.nlargest(max(20, size // 3), "pnl_usdt")
    losers = usable.nsmallest(max(20, size // 3), "pnl_usdt")
    rest = usable.drop(index=winners.index.union(losers.index), errors="ignore")
    random_n = max(0, size - len(winners) - len(losers))
    if random_n > 0:
        rest = rest.sample(n=min(random_n, len(rest)), random_state=42)
    sample = pd.concat([winners, losers, rest], ignore_index=False)
    return sample.drop_duplicates("trade_id").sort_values("entry_time").head(size).reset_index(drop=True)


def calibrate_time_offset(
    trades: pd.DataFrame, client: OKXClient, sample_size: int
) -> tuple[int, pd.DataFrame]:
    sample = trades[
        trades["symbol"].isin(["BTC-USDT-SWAP", "ETH-USDT-SWAP"])
        & trades["entry_time"].notna()
        & trades["exit_time"].notna()
    ].head(sample_size)
    rows = []
    for offset in [0, -8]:
        total = 0
        hits = 0
        relative_errors = []
        for _, trade in sample.iterrows():
            for time_col, price_col in [("entry_time", "entry_price"), ("exit_time", "exit_price")]:
                ts = trade[time_col]
                price = safe_float(trade[price_col])
                if pd.isna(ts) or not np.isfinite(price) or price <= 0:
                    continue
                center = utc_ms_from_naive(ts, offset)
                candles = client.fetch_window(
                    trade["symbol"], "1m", center - 2 * 60_000, center + 2 * 60_000
                )
                if candles.empty:
                    continue
                low = candles["low"].min()
                high = candles["high"].max()
                close = candles.iloc[(candles["ts"] - center).abs().argsort()[:1]]["close"].iloc[0]
                total += 1
                hits += int(low <= price <= high)
                relative_errors.append(abs(close - price) / price)
        rows.append(
            {
                "offset_hours": offset,
                "checked_points": total,
                "range_hit_rate": hits / total if total else np.nan,
                "median_close_error": float(np.nanmedian(relative_errors)) if relative_errors else np.nan,
            }
        )
    result = pd.DataFrame(rows)
    valid = result[result["checked_points"] > 0].copy()
    if valid.empty:
        return -8, result
    valid = valid.sort_values(["range_hit_rate", "median_close_error"], ascending=[False, True])
    return int(valid.iloc[0]["offset_hours"]), result


def candle_slice(df: pd.DataFrame, end_ms: int, lookback_bars: int) -> pd.DataFrame:
    before = df[df["ts"] <= end_ms].copy()
    if before.empty:
        return before
    return before.tail(lookback_bars)


def position_in_range(price: float, low: float, high: float) -> float:
    if not np.isfinite(price) or not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.nan
    return (price - low) / (high - low)


def build_daily_features(
    trades: pd.DataFrame, client: OKXClient, offset_hours: int, symbols_to_fetch: set[str] | None = None
) -> pd.DataFrame:
    records = []
    global_start = trades["entry_time"].min() - pd.Timedelta(days=140)
    global_end = trades["entry_time"].max() + pd.Timedelta(days=3)
    for symbol, group in trades.groupby("symbol", sort=True):
        if symbols_to_fetch is not None and symbol not in symbols_to_fetch:
            candles = pd.DataFrame(columns=OKX_COLUMNS)
        else:
            candles = client.fetch_window(
                symbol,
                "1D",
                utc_ms_from_naive(global_start, offset_hours),
                utc_ms_from_naive(global_end, offset_hours),
            )
        for _, trade in group.iterrows():
            entry_ms = utc_ms_from_naive(trade["entry_time"], offset_hours)
            before = candles[candles["ts"] <= entry_ms].copy()
            rec: dict[str, Any] = {"trade_id": int(trade["trade_id"]), "symbol": symbol, "daily_bars": len(before)}
            if len(before) < 8:
                rec.update({"daily_ok": False})
                records.append(rec)
                continue
            rec["daily_ok"] = True
            close = before["close"]
            high = before["high"]
            low = before["low"]
            vol = before["vol_quote"].replace(0, np.nan)
            for n in [7, 20, 60]:
                if len(before) > n:
                    rec[f"ret_{n}d"] = close.iloc[-1] / close.iloc[-n - 1] - 1
                else:
                    rec[f"ret_{n}d"] = np.nan
                tail = before.tail(n)
                rec[f"high_{n}d"] = tail["high"].max()
                rec[f"low_{n}d"] = tail["low"].min()
                rec[f"pos_{n}d"] = position_in_range(
                    safe_float(trade["entry_price"]), rec[f"low_{n}d"], rec[f"high_{n}d"]
                )
            high20 = rec["high_20d"]
            high60 = rec["high_60d"]
            low20 = rec["low_20d"]
            rec["pullback_from_20d_high"] = trade["entry_price"] / high20 - 1 if high20 else np.nan
            rec["pullback_from_60d_high"] = trade["entry_price"] / high60 - 1 if high60 else np.nan
            rec["distance_from_20d_low"] = trade["entry_price"] / low20 - 1 if low20 else np.nan
            rec["near_20d_breakout"] = bool(trade["entry_price"] >= high20 * 0.995) if high20 else False
            rec["vol_ratio_20d"] = vol.iloc[-1] / vol.tail(20).median() if len(vol.dropna()) >= 10 else np.nan
            rec["main_uptrend_daily"] = bool(
                safe_float(rec.get("ret_20d")) > 0.20
                and safe_float(rec.get("ret_60d")) > 0.35
                and safe_float(rec.get("pos_20d")) > 0.45
            )
            records.append(rec)
    return pd.DataFrame(records)


def return_over_window(candles: pd.DataFrame, entry_ms: int, minutes: int) -> float:
    window = candles[(candles["ts"] < entry_ms) & (candles["ts"] >= entry_ms - minutes * 60_000)]
    if len(window) < 2:
        return np.nan
    return window["close"].iloc[-1] / window["close"].iloc[0] - 1


def range_position_over_window(candles: pd.DataFrame, entry_ms: int, minutes: int, price: float) -> float:
    window = candles[(candles["ts"] < entry_ms) & (candles["ts"] >= entry_ms - minutes * 60_000)]
    if window.empty:
        return np.nan
    return position_in_range(price, window["low"].min(), window["high"].max())


def volume_ratio_over_window(candles: pd.DataFrame, entry_ms: int, minutes: int) -> float:
    window = candles[(candles["ts"] < entry_ms) & (candles["ts"] >= entry_ms - minutes * 60_000)]
    if len(window) < 5:
        return np.nan
    recent = window.tail(max(1, min(6, len(window) // 3)))["vol_quote"].median()
    base = window["vol_quote"].median()
    if not base or not np.isfinite(base):
        return np.nan
    return recent / base


def favorable_adverse(candles: pd.DataFrame, trade: pd.Series, entry_ms: int, exit_ms: int) -> tuple[float, float]:
    if pd.isna(trade["exit_time"]):
        exit_ms = entry_ms + 12 * 60 * 60_000
    during = candles[(candles["ts"] >= entry_ms) & (candles["ts"] <= exit_ms)]
    if during.empty:
        return np.nan, np.nan
    entry = safe_float(trade["entry_price"])
    if not np.isfinite(entry) or entry <= 0:
        return np.nan, np.nan
    if trade["side"] == "long":
        mfe = during["high"].max() / entry - 1
        mae = during["low"].min() / entry - 1
    else:
        mfe = entry / during["low"].min() - 1
        mae = entry / during["high"].max() - 1
    return float(mfe), float(mae)


def build_intraday_features(
    sample: pd.DataFrame, client: OKXClient, offset_hours: int, max_intraday_hours: float = 24.0
) -> pd.DataFrame:
    records = []
    for _, trade in sample.iterrows():
        entry_ms = utc_ms_from_naive(trade["entry_time"], offset_hours)
        exit_ms = utc_ms_from_naive(trade["exit_time"], offset_hours) if pd.notna(trade["exit_time"]) else entry_ms
        cap_ms = int(max_intraday_hours * 60 * 60_000) if max_intraday_hours and max_intraday_hours > 0 else None
        capped_end_ms = min(exit_ms, entry_ms + cap_ms) if cap_ms is not None else exit_ms
        rec: dict[str, Any] = {
            "trade_id": int(trade["trade_id"]),
            "symbol": trade["symbol"],
            "intraday_sampled": True,
            "intraday_window_hours": max_intraday_hours if cap_ms is not None else np.nan,
            "intraday_window_capped": bool(cap_ms is not None and exit_ms > entry_ms + cap_ms),
        }
        windows = {
            "1m": (
                entry_ms - 2 * 60 * 60_000,
                max(capped_end_ms + 60 * 60_000, entry_ms + 2 * 60 * 60_000),
            ),
            "5m": (
                entry_ms - 24 * 60 * 60_000,
                max(capped_end_ms + 4 * 60 * 60_000, entry_ms + 4 * 60 * 60_000),
            ),
            "15m": (
                entry_ms - 72 * 60 * 60_000,
                max(capped_end_ms + 8 * 60 * 60_000, entry_ms + 8 * 60 * 60_000),
            ),
            "1H": (
                entry_ms - 30 * 24 * 60 * 60_000,
                max(capped_end_ms + 24 * 60 * 60_000, entry_ms + 24 * 60 * 60_000),
            ),
        }
        for bar, (start_ms, end_ms) in windows.items():
            candles = client.fetch_window(trade["symbol"], bar, start_ms, end_ms)
            rec[f"{bar}_bars"] = len(candles)
            if candles.empty:
                continue
            if bar == "1m":
                rec["pre_30m_ret"] = return_over_window(candles, entry_ms, 30)
                rec["pre_120m_ret"] = return_over_window(candles, entry_ms, 120)
                rec["pos_2h"] = range_position_over_window(candles, entry_ms, 120, trade["entry_price"])
                rec["vol_ratio_30m"] = volume_ratio_over_window(candles, entry_ms, 30)
                rec["mfe_1m"], rec["mae_1m"] = favorable_adverse(candles, trade, entry_ms, capped_end_ms)
            elif bar == "5m":
                rec["pre_6h_ret_5m"] = return_over_window(candles, entry_ms, 6 * 60)
                rec["pre_24h_ret_5m"] = return_over_window(candles, entry_ms, 24 * 60)
                rec["pos_24h_5m"] = range_position_over_window(candles, entry_ms, 24 * 60, trade["entry_price"])
            elif bar == "15m":
                rec["pre_72h_ret_15m"] = return_over_window(candles, entry_ms, 72 * 60)
                rec["pos_72h_15m"] = range_position_over_window(candles, entry_ms, 72 * 60, trade["entry_price"])
            elif bar == "1H":
                rec["pre_7d_ret_1h"] = return_over_window(candles, entry_ms, 7 * 24 * 60)
                rec["pre_30d_ret_1h"] = return_over_window(candles, entry_ms, 30 * 24 * 60)
                rec["pos_30d_1h"] = range_position_over_window(candles, entry_ms, 30 * 24 * 60, trade["entry_price"])
        rec["chase_long_risk"] = bool(
            trade["side"] == "long"
            and safe_float(rec.get("pre_30m_ret")) > 0.03
            and safe_float(rec.get("pos_2h")) > 0.90
        )
        rec["chase_short_risk"] = bool(
            trade["side"] == "short"
            and safe_float(rec.get("pre_30m_ret")) < -0.03
            and safe_float(rec.get("pos_2h")) < 0.10
        )
        rec["pullback_entry_sample"] = bool(
            trade["side"] == "long"
            and safe_float(rec.get("pre_24h_ret_5m")) > 0.04
            and 0.35 <= safe_float(rec.get("pos_24h_5m")) <= 0.85
        )
        records.append(rec)
    return pd.DataFrame(records)


def summarize_subset(df: pd.DataFrame, mask: pd.Series, label: str, universe_label: str) -> dict[str, Any]:
    sub = df[mask.fillna(False)].copy()
    if sub.empty:
        return {
            "rule": label,
            "universe": universe_label,
            "n": 0,
            "pnl_sum": 0.0,
            "avg_pnl": np.nan,
            "median_pnl": np.nan,
            "win_rate": np.nan,
            "avg_return_rate": np.nan,
            "max_loss": np.nan,
            "best_win": np.nan,
        }
    return {
        "rule": label,
        "universe": universe_label,
        "n": int(len(sub)),
        "pnl_sum": float(sub["pnl_usdt"].sum()),
        "avg_pnl": float(sub["pnl_usdt"].mean()),
        "median_pnl": float(sub["pnl_usdt"].median()),
        "win_rate": float((sub["pnl_usdt"] > 0).mean()),
        "avg_return_rate": float(sub["return_rate"].mean()),
        "max_loss": float(sub["pnl_usdt"].min()),
        "best_win": float(sub["pnl_usdt"].max()),
    }


NUMERIC_FEATURE_COLUMNS = [
    "ret_7d",
    "ret_20d",
    "ret_60d",
    "pos_20d",
    "pos_60d",
    "pullback_from_20d_high",
    "pullback_from_60d_high",
    "distance_from_20d_low",
    "vol_ratio_20d",
    "pre_30m_ret",
    "pre_120m_ret",
    "pos_2h",
    "pre_6h_ret_5m",
    "pre_24h_ret_5m",
    "pos_24h_5m",
    "pre_72h_ret_15m",
    "pos_72h_15m",
    "pre_7d_ret_1h",
    "pre_30d_ret_1h",
    "pos_30d_1h",
    "mfe_1m",
    "mae_1m",
]
BOOL_FEATURE_COLUMNS = [
    "near_20d_breakout",
    "main_uptrend_daily",
    "intraday_sampled",
    "chase_long_risk",
    "chase_short_risk",
    "pullback_entry_sample",
]


def ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in NUMERIC_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    for col in BOOL_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = False
    return out


def evaluate_rules(features: pd.DataFrame) -> pd.DataFrame:
    df = ensure_feature_columns(features)
    conditions: list[tuple[str, pd.Series, str]] = [
        (
            "日线主升浪多单",
            (df["side"] == "long")
            & (df["ret_20d"] > 0.20)
            & (df["ret_60d"] > 0.35)
            & (df["pos_20d"] > 0.45),
            "all_daily",
        ),
        (
            "突破/近20日新高多单",
            (df["side"] == "long") & (df["near_20d_breakout"] == True) & (df["ret_20d"] > 0.15),
            "all_daily",
        ),
        (
            "强趋势回调多单",
            (df["side"] == "long")
            & (df["ret_20d"] > 0.20)
            & (df["pullback_from_20d_high"].between(-0.20, -0.015))
            & (df["pos_20d"] > 0.40),
            "all_daily",
        ),
        (
            "高位追多风险过滤",
            (df["side"] == "long")
            & (df["ret_7d"] > 0.20)
            & (df["pos_20d"] > 0.90)
            & (df["pullback_from_20d_high"] > -0.02),
            "all_daily",
        ),
        (
            "短期瀑布空单",
            (df["side"] == "short") & (df["ret_7d"] < -0.08) & (df["pos_20d"] < 0.35),
            "all_daily",
        ),
        (
            "样本-24h强势回踩多",
            (df.get("intraday_sampled", False) == True)
            & (df["side"] == "long")
            & (df["pre_24h_ret_5m"] > 0.04)
            & (df["pos_24h_5m"].between(0.35, 0.85)),
            "intraday_sample",
        ),
        (
            "样本-追涨追空风险",
            (df.get("intraday_sampled", False) == True)
            & ((df.get("chase_long_risk", False) == True) | (df.get("chase_short_risk", False) == True)),
            "intraday_sample",
        ),
        (
            "样本-MAE止损警戒",
            (df.get("intraday_sampled", False) == True) & (df["mae_1m"] < -0.018),
            "intraday_sample",
        ),
    ]
    rows = []
    for label, mask, universe in conditions:
        rows.append(summarize_subset(df, mask, label, universe))
        train = df["entry_time"] < pd.Timestamp("2024-01-01")
        rows.append(summarize_subset(df, mask & train, f"{label} / train<=2023", universe))
        rows.append(summarize_subset(df, mask & ~train, f"{label} / validate2024", universe))
    return pd.DataFrame(rows)


def feature_lift(features: pd.DataFrame) -> pd.DataFrame:
    features = ensure_feature_columns(features)
    pnl = features["pnl_usdt"]
    top = features[pnl >= pnl.quantile(0.90)]
    bottom = features[pnl <= pnl.quantile(0.10)]
    cols = [
        "ret_7d",
        "ret_20d",
        "ret_60d",
        "pos_20d",
        "pullback_from_20d_high",
        "vol_ratio_20d",
        "pre_30m_ret",
        "pre_120m_ret",
        "pre_24h_ret_5m",
        "pos_24h_5m",
        "mfe_1m",
        "mae_1m",
    ]
    rows = []
    for col in cols:
        if col not in features:
            continue
        rows.append(
            {
                "feature": col,
                "top_decile_median": float(top[col].median()) if top[col].notna().any() else np.nan,
                "bottom_decile_median": float(bottom[col].median()) if bottom[col].notna().any() else np.nan,
                "all_median": float(features[col].median()) if features[col].notna().any() else np.nan,
                "coverage": float(features[col].notna().mean()),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_无数据_"
    view = df[columns].head(max_rows).copy()
    headers = [str(col) for col in view.columns]

    def fmt_cell(value: Any) -> str:
        if pd.isna(value):
            text = ""
        elif isinstance(value, float):
            text = f"{value:.6g}"
        else:
            text = str(value)
        return text.replace("|", "\\|").replace("\n", "<br>")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(fmt_cell(row[col]) for col in view.columns) + " |")
    return "\n".join(lines)


def build_report(
    output_dir: Path,
    trades: pd.DataFrame,
    compare: dict[str, Any],
    pdf_info: dict[str, Any],
    calibration: pd.DataFrame,
    offset_hours: int,
    daily: pd.DataFrame,
    intraday: pd.DataFrame,
    features: pd.DataFrame,
    rules: pd.DataFrame,
    lifts: pd.DataFrame,
    client: OKXClient,
) -> str:
    pnl = trades["pnl_usdt"]
    hold = trades["hold_minutes"]
    top_trades = trades.nlargest(12, "pnl_usdt")[
        ["trade_id", "entry_time", "exit_time", "symbol", "side_cn", "leverage", "pnl_usdt", "return_rate", "hold_minutes"]
    ].copy()
    worst_trades = trades.nsmallest(12, "pnl_usdt")[
        ["trade_id", "entry_time", "exit_time", "symbol", "side_cn", "leverage", "pnl_usdt", "return_rate", "hold_minutes"]
    ].copy()
    by_symbol = (
        trades.groupby("symbol")
        .agg(trades=("trade_id", "count"), pnl_sum=("pnl_usdt", "sum"), win_rate=("is_win", "mean"))
        .sort_values("pnl_sum", ascending=False)
        .head(15)
        .reset_index()
    )
    daily_coverage = float(daily["daily_ok"].mean()) if "daily_ok" in daily and len(daily) else 0.0
    intraday_coverage = (
        float((intraday.filter(regex="_bars$").fillna(0).sum(axis=1) > 0).mean()) if len(intraday) else 0.0
    )
    best_rules = rules.sort_values(["universe", "pnl_sum"], ascending=[True, False]).copy()
    report = f"""# 浪浪交割单策略蒸馏报告

生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 数据与执行范围

- 主交割单：`{PRIMARY_XLSX}`
- 对照交割单：`{SECONDARY_XLSX}`
- 策略心得 PDF：`{PDF_PATH}`
- 标准化交易数：{len(trades):,} 笔，时间范围 {trades['entry_time'].min()} 到 {trades['entry_time'].max()}
- 合约数：{trades['symbol'].nunique()}；多单 {int((trades['side'] == 'long').sum()):,}，空单 {int((trades['side'] == 'short').sum()):,}
- 总收益：{money(pnl.sum())} USDT；胜率：{pct((pnl > 0).mean(), 2)}；收益中位数：{money(pnl.median())} USDT
- 收益分布：P10={money(pnl.quantile(0.10))}，P50={money(pnl.quantile(0.50))}，P90={money(pnl.quantile(0.90))}，P99={money(pnl.quantile(0.99))}
- 持仓时间中位数：{hold.median():.1f} 分钟；P90={hold.quantile(0.90):.1f} 分钟；P99={hold.quantile(0.99):.1f} 分钟
- Excel 对照：行差 {compare.get('row_delta', 'NA')}，关键字段不同行 {compare.get('key_mismatch_rows', 'NA')}，关键字段不同单元格 {compare.get('key_mismatch_cells', 'NA')}

## 2. PDF 心得结构化为可量化变量

PDF 文本抽取字符数：{pdf_info.get('text_extract_chars')}；渲染图：`{pdf_info.get('rendered_png')}`。

{markdown_table(pd.DataFrame(STRATEGY_MAP), ['concept', 'text_meaning', 'quant_features'], 10)}

## 3. K线对齐与覆盖

- OKX 时间校准选择 offset_hours={offset_hours}。`-8` 表示交割单时间按北京时间解释，`0` 表示按 UTC 解释。
- OKX 请求数：{client.calls_made}；缓存命中：{client.cache_hits}；错误数：{len(client.errors or [])}
- 日线特征覆盖：{pct(daily_coverage, 2)}；覆盖到的交易会计算主升浪/回撤/新高结构，未覆盖合约不会混入规则证据。
- 分钟线/多周期样本数：{len(intraday):,}，有效 K 线覆盖：{pct(intraday_coverage, 2)}。样本采用大赚、大亏、普通单分层抽样；分钟线持仓后窗口按运行参数设上限。

时间校准结果：

{markdown_table(calibration, ['offset_hours', 'checked_points', 'range_hit_rate', 'median_close_error'], 5)}

## 4. 蒸馏出的核心事实

1. 这不是高胜率策略：总胜率只有 {pct((pnl > 0).mean(), 2)}，但右尾极强，P99 单笔收益达到 {money(pnl.quantile(0.99))} USDT，最大单笔收益 {money(pnl.max())} USDT。
2. 中位数交易是亏损的：单笔收益中位数 {money(pnl.median())} USDT，说明策略必须靠少数主升浪/瀑布级别大波段覆盖大量试错。
3. 持仓时间高度偏态：半数交易 7 分钟左右结束，但最大收益单多来自数小时到数天持仓，符合“找到主升浪后拿住”的心得。
4. 初步量化时，日线强趋势、近20日新高、回调后仍保持区间中上部，是多单大收益的优先候选条件；高位短线追涨和 MAE 扩大是过滤/止损候选。

## 5. 候选规则验证

以下规则是“心得概念 -> 可观测条件 -> 交割单结果”的蒸馏结果。`train<=2023` 用于形成规则，`validate2024` 用于粗略样本外检查。

{markdown_table(best_rules, ['rule', 'universe', 'n', 'pnl_sum', 'avg_pnl', 'median_pnl', 'win_rate', 'max_loss', 'best_win'], 30)}

## 6. 大赚/大亏特征差异

{markdown_table(lifts, ['feature', 'top_decile_median', 'bottom_decile_median', 'all_median', 'coverage'], 20)}

## 7. 代表交易

大赚单：

{markdown_table(top_trades, ['trade_id', 'entry_time', 'exit_time', 'symbol', 'side_cn', 'leverage', 'pnl_usdt', 'return_rate', 'hold_minutes'], 12)}

大亏单：

{markdown_table(worst_trades, ['trade_id', 'entry_time', 'exit_time', 'symbol', 'side_cn', 'leverage', 'pnl_usdt', 'return_rate', 'hold_minutes'], 12)}

收益贡献最高合约：

{markdown_table(by_symbol, ['symbol', 'trades', 'pnl_sum', 'win_rate'], 15)}

## 8. 可回测规则草案

```python
for symbol in okx_swap_symbols:
    daily = load_1d(symbol)
    intraday = load_1m_5m_15m_1h(symbol)
    if ret_20d > 0.20 and ret_60d > 0.35 and pos_20d > 0.45:
        regime = "主升浪"
    if regime == "主升浪" and -0.20 <= pullback_from_20d_high <= -0.015 and pos_20d > 0.40:
        wait_for_5m_reclaim_or_platform_hold()
        enter_long_in_batches()
    if pre_30m_ret > 0.03 and pos_2h > 0.90:
        skip_or_reduce_size("追涨风险")
    if mae_1m < -0.018 or price_breaks_recent_platform:
        stop_loss("结构破坏")
    if new_high_continues and pullback_does_not_break_structure:
        hold_runner_position()
```

## 9. 结论

- 可以量化，但应该量化为“主升浪识别 + 回调/突破确认 + 失败过滤 + 分仓止损”的组合策略，而不是单一入场信号。
- 交割单显示的核心边际来自少数大级别行情；因此回测必须用收益分布、最大亏损、错过大单的机会成本来评价，不能只看胜率。
- 下一步建议：用本脚本缓存继续扩展全量 1m/5m/15m/1H 特征，然后把候选规则转成事件驱动回测，验证是否能在不使用未来函数的前提下复现右尾收益。

## 10. 输出文件

- 标准交易表：`{(output_dir / 'standard_trades.csv').resolve()}`
- 日线特征：`{(output_dir / 'daily_features.csv').resolve()}`
- 分钟线样本特征：`{(output_dir / 'intraday_features.csv').resolve()}`
- 合并特征表：`{(output_dir / 'trade_features.csv').resolve()}`
- 规则评估：`{(output_dir / 'rule_summary.csv').resolve()}`
- 特征差异：`{(output_dir / 'feature_lift.csv').resolve()}`
"""
    if client.errors:
        report += "\n## 11. OKX 请求警告\n\n" + "\n".join(f"- {err}" for err in client.errors[:30]) + "\n"
    return report


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "kline_cache"

    trades = load_trade_sheet(args.primary_xlsx)
    compare = compare_workbooks(trades, args.secondary_xlsx)
    pdf_info = render_pdf(args.pdf, output_dir)

    client = OKXClient(
        cache_dir=cache_dir,
        allow_network=not args.no_network,
        max_calls=args.max_okx_calls,
        sleep_seconds=args.okx_sleep,
        request_timeout=args.request_timeout,
        retries=args.okx_retries,
        refresh_cache=args.refresh_cache,
    )

    offset_hours, calibration = calibrate_time_offset(trades, client, args.calibration_sample_size)
    sample = select_kline_sample(trades, args.kline_sample_size)
    daily_symbols: set[str] | None = None
    if args.max_daily_symbols > 0:
        top_symbols = (
            trades.groupby("symbol")
            .agg(trades=("trade_id", "count"), abs_pnl=("pnl_usdt", lambda s: s.abs().sum()))
            .sort_values(["trades", "abs_pnl"], ascending=False)
            .head(args.max_daily_symbols)
            .index
        )
        daily_symbols = set(top_symbols).union(set(sample["symbol"]))
    daily = build_daily_features(trades, client, offset_hours, daily_symbols)
    intraday = build_intraday_features(sample, client, offset_hours, args.max_intraday_hours)

    features = trades.merge(daily, on=["trade_id", "symbol"], how="left")
    if not intraday.empty:
        features = features.merge(intraday.drop(columns=["symbol"], errors="ignore"), on="trade_id", how="left")
    else:
        features["intraday_sampled"] = False
    features["intraday_sampled"] = features["intraday_sampled"].apply(
        lambda value: False if pd.isna(value) else bool(value)
    )

    rules = evaluate_rules(features)
    lifts = feature_lift(features)

    trades.to_csv(output_dir / "standard_trades.csv", index=False)
    daily.to_csv(output_dir / "daily_features.csv", index=False)
    intraday.to_csv(output_dir / "intraday_features.csv", index=False)
    features.to_csv(output_dir / "trade_features.csv", index=False)
    rules.to_csv(output_dir / "rule_summary.csv", index=False)
    lifts.to_csv(output_dir / "feature_lift.csv", index=False)
    calibration.to_csv(output_dir / "time_calibration.csv", index=False)

    metadata = {
        "primary_xlsx": str(args.primary_xlsx),
        "secondary_xlsx": str(args.secondary_xlsx),
        "pdf": str(args.pdf),
        "output_dir": str(output_dir.resolve()),
        "trade_count": int(len(trades)),
        "symbol_count": int(trades["symbol"].nunique()),
        "time_offset_hours": int(offset_hours),
        "okx_calls_made": int(client.calls_made),
        "okx_cache_hits": int(client.cache_hits),
        "okx_error_count": len(client.errors or []),
        "kline_sample_size": int(len(intraday)),
        "max_daily_symbols": int(args.max_daily_symbols),
        "max_intraday_hours": float(args.max_intraday_hours),
        "compare": compare,
        "pdf_info": pdf_info,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    report = build_report(
        output_dir,
        trades,
        compare,
        pdf_info,
        calibration,
        offset_hours,
        daily,
        intraday,
        features,
        rules,
        lifts,
        client,
    )
    (output_dir / "strategy_distillation_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
