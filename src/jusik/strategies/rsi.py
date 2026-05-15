from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Pick, Strategy


def _rsi(series: pd.Series, period: int) -> float:
    if len(series) < period + 1:
        return float("nan")
    diff = series.diff().dropna().tail(period)
    gain = diff.clip(lower=0).mean()
    loss = -diff.clip(upper=0).mean()
    if loss == 0:
        return 100.0
    rs = gain / loss
    return 100 - (100 / (1 + rs))


class RsiReversalStrategy(Strategy):
    """Oversold reversal: buy names whose RSI is below `threshold`.

    Lower RSI ranks first.  Period defaults to 14.
    """

    name = "rsi"

    def __init__(
        self,
        top_n: int = 5,
        period: int = 14,
        threshold: float = 30.0,
        min_price: float = 1_000.0,
        min_avg_volume: float = 100_000.0,
    ) -> None:
        self.top_n = top_n
        self.period = period
        self.threshold = threshold
        self.min_price = min_price
        self.min_avg_volume = min_avg_volume
        self.params = dict(
            top_n=top_n, period=period, threshold=threshold,
            min_price=min_price, min_avg_volume=min_avg_volume,
        )

    def select(self, asof: pd.Timestamp, history: pd.DataFrame) -> list[Pick]:
        hist = history[history["date"] <= asof]
        dates = sorted(hist["date"].unique())
        if len(dates) < self.period + 2:
            return []
        last_date = dates[-1]
        window_dates = dates[-(self.period + 1):]
        window = hist[hist["date"].isin(window_dates)]

        rows = []
        for code, g in window.groupby("code"):
            g = g.sort_values("date")
            rsi = _rsi(g["Close"], self.period)
            if np.isnan(rsi):
                continue
            last = g[g["date"] == last_date]
            if last.empty:
                continue
            rows.append({
                "code": code,
                "rsi": rsi,
                "last_close": float(last["Close"].iloc[0]),
                "avg_vol": float(g["Volume"].mean()),
            })

        df = pd.DataFrame(rows).set_index("code")
        if df.empty:
            return []
        df = df[(df["last_close"] >= self.min_price) & (df["avg_vol"] >= self.min_avg_volume)]
        df = df[df["rsi"] <= self.threshold]
        df = df.sort_values("rsi", ascending=True).head(self.top_n)
        if df.empty:
            return []
        w = 1.0 / len(df)
        return [Pick(code=c, weight=w, reason=f"rsi={r.rsi:.1f}") for c, r in df.iterrows()]
