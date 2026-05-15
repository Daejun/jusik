from __future__ import annotations

import pandas as pd

from .base import Pick, Strategy


class SmaCrossStrategy(Strategy):
    """Pick names whose short SMA is most above its long SMA on close basis.

    Score = SMA_short / SMA_long - 1.  Positive ranking emphasises trend
    continuation.  Optionally require the short SMA crossed *up* through long
    SMA today (fresh signal).
    """

    name = "sma_cross"

    def __init__(
        self,
        top_n: int = 5,
        short: int = 5,
        long: int = 20,
        fresh_only: bool = False,
        min_price: float = 1_000.0,
        min_avg_volume: float = 100_000.0,
    ) -> None:
        if short >= long:
            raise ValueError("short must be < long")
        self.top_n = top_n
        self.short = short
        self.long = long
        self.fresh_only = fresh_only
        self.min_price = min_price
        self.min_avg_volume = min_avg_volume
        self.params = dict(
            top_n=top_n, short=short, long=long,
            fresh_only=int(fresh_only),
            min_price=min_price, min_avg_volume=min_avg_volume,
        )

    def select(self, asof: pd.Timestamp, history: pd.DataFrame) -> list[Pick]:
        hist = history[history["date"] <= asof]
        dates = sorted(hist["date"].unique())
        if len(dates) < self.long + 1:
            return []
        window = hist[hist["date"].isin(dates[-(self.long + 1):])]
        last_date = dates[-1]
        prev_date = dates[-2]

        def _sma(g: pd.DataFrame, n: int, end: pd.Timestamp) -> float:
            tail = g[g["date"] <= end].tail(n)
            return tail["Close"].mean() if len(tail) == n else float("nan")

        rows = []
        for code, g in window.groupby("code"):
            short_now = _sma(g, self.short, last_date)
            long_now = _sma(g, self.long, last_date)
            if pd.isna(short_now) or pd.isna(long_now):
                continue
            score = short_now / long_now - 1
            last = g[g["date"] == last_date]
            if last.empty:
                continue
            last_close = float(last["Close"].iloc[0])
            avg_vol = float(g["Volume"].mean())
            cross_up = False
            if self.fresh_only:
                short_prev = _sma(g, self.short, prev_date)
                long_prev = _sma(g, self.long, prev_date)
                cross_up = (short_prev <= long_prev) and (short_now > long_now)
            rows.append({
                "code": code,
                "score": score,
                "last_close": last_close,
                "avg_vol": avg_vol,
                "cross_up": cross_up,
            })

        df = pd.DataFrame(rows).set_index("code")
        if df.empty:
            return []
        df = df[(df["last_close"] >= self.min_price) & (df["avg_vol"] >= self.min_avg_volume)]
        if self.fresh_only:
            df = df[df["cross_up"]]
        df = df[df["score"] > 0]
        df = df.sort_values("score", ascending=False).head(self.top_n)
        if df.empty:
            return []
        w = 1.0 / len(df)
        return [Pick(code=c, weight=w, reason=f"sma_score={r.score:+.3%}") for c, r in df.iterrows()]
