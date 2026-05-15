from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import DATA_DIR

log = logging.getLogger(__name__)

UNIVERSE_FILE = DATA_DIR / "universe_kospi200.parquet"
OHLCV_DIR = DATA_DIR / "ohlcv"


def _ensure_dirs() -> None:
    OHLCV_DIR.mkdir(parents=True, exist_ok=True)


def _yahoo_symbol(code: str, market: str = "KS") -> str:
    return f"{code}.{market}"


def fetch_kospi_universe(top_n: int = 200, force: bool = False) -> pd.DataFrame:
    """Return current KOSPI top N tickers by market cap, used as KOSPI200 proxy.

    KOSPI200 membership changes over time; this snapshot introduces some
    survivorship bias but is acceptable for an MVP.
    """
    _ensure_dirs()
    if UNIVERSE_FILE.exists() and not force:
        mtime = datetime.fromtimestamp(UNIVERSE_FILE.stat().st_mtime)
        if (datetime.now() - mtime) < timedelta(days=7):
            return pd.read_parquet(UNIVERSE_FILE)

    import FinanceDataReader as fdr

    kospi = fdr.StockListing("KOSPI")
    kospi = kospi[kospi["Market"] == "KOSPI"].copy()
    kospi = kospi.dropna(subset=["Marcap"])
    kospi = kospi.sort_values("Marcap", ascending=False).head(top_n)
    out = kospi[["Code", "Name", "Marcap"]].reset_index(drop=True)
    out.to_parquet(UNIVERSE_FILE)
    log.info("universe cached: %d tickers", len(out))
    return out


def _ohlcv_path(code: str) -> Path:
    return OHLCV_DIR / f"{code}.parquet"


def fetch_ohlcv(
    codes: Iterable[str],
    start: date,
    end: date,
    force: bool = False,
    sleep_sec: float = 0.0,
) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV for each ticker via yfinance and cache to parquet.

    Returns {code: DataFrame indexed by date with columns [Open, High, Low, Close, Volume]}.
    """
    import yfinance as yf

    _ensure_dirs()
    codes = list(codes)
    result: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for code in codes:
        path = _ohlcv_path(code)
        if path.exists() and not force:
            try:
                df = pd.read_parquet(path)
                if not df.empty and df.index.min().date() <= start and df.index.max().date() >= end:
                    result[code] = df.loc[
                        (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
                    ]
                    continue
            except Exception as e:  # noqa: BLE001
                log.warning("cache read failed for %s: %s", code, e)
        missing.append(code)

    if missing:
        symbols = [_yahoo_symbol(c) for c in missing]
        log.info("downloading %d tickers via yfinance...", len(symbols))
        df = yf.download(
            tickers=symbols,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        for code, sym in zip(missing, symbols):
            try:
                sub = df[sym].dropna(how="all") if sym in df.columns.levels[0] else None
            except Exception:
                sub = None
            if sub is None or sub.empty:
                log.warning("no data for %s (%s)", code, sym)
                continue
            sub.index = pd.to_datetime(sub.index).tz_localize(None)
            sub = sub[["Open", "High", "Low", "Close", "Volume"]].copy()
            sub.to_parquet(_ohlcv_path(code))
            result[code] = sub.loc[
                (sub.index >= pd.Timestamp(start)) & (sub.index <= pd.Timestamp(end))
            ]
            if sleep_sec:
                time.sleep(sleep_sec)

    return result


def load_panel(
    codes: Iterable[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    """Return a multi-index dataframe (date, code) -> OHLCV columns."""
    data = fetch_ohlcv(codes, start, end)
    frames = []
    for code, df in data.items():
        if df.empty:
            continue
        tmp = df.copy()
        tmp["code"] = code
        tmp.index.name = "date"
        frames.append(tmp.reset_index())
    if not frames:
        return pd.DataFrame(columns=["date", "code", "Open", "High", "Low", "Close", "Volume"])
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["date", "code"]).reset_index(drop=True)
    return panel


@dataclass
class MarketData:
    panel: pd.DataFrame
    names: dict[str, str]

    def trading_dates(self) -> list[pd.Timestamp]:
        return sorted(self.panel["date"].unique().tolist())

    def slice(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        m = (self.panel["date"] >= start) & (self.panel["date"] <= end)
        return self.panel.loc[m]

    def at(self, dt: pd.Timestamp) -> pd.DataFrame:
        return self.panel.loc[self.panel["date"] == dt]


def load_market_data(start: date, end: date, top_n: int = 200) -> MarketData:
    uni = fetch_kospi_universe(top_n=top_n)
    panel = load_panel(uni["Code"].tolist(), start, end)
    names = dict(zip(uni["Code"], uni["Name"]))
    return MarketData(panel=panel, names=names)
