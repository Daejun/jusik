from __future__ import annotations

import pandas as pd

from .base import Pick, Strategy


class SmaCrossStrategy(Strategy):
    """Pick names whose short SMA is most above its long SMA on close basis.

    Score = SMA_short / SMA_long - 1.  Positive ranking emphasises trend
    continuation.  Optionally require a fresh crossover this bar.
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
        window_dates = dates[-(self.long + 1):]
        window = hist[hist["date"].isin(window_dates)]

        close = window.pivot(index="date", columns="code", values="Close").sort_index()
        vol = window.pivot(index="date", columns="code", values="Volume").sort_index()

        short_now = close.tail(self.short).mean()
        long_now = close.tail(self.long).mean()
        score = short_now / long_now - 1

        df = pd.DataFrame({
            "score": score,
            "last_close": close.iloc[-1],
            "avg_vol": vol.mean(),
        }).dropna()

        if self.fresh_only and len(close) >= self.long + 1:
            short_prev = close.iloc[-(self.short + 1):-1].mean()
            long_prev = close.iloc[-(self.long + 1):-1].mean()
            cross_up = (short_prev <= long_prev) & (short_now > long_now)
            df["cross_up"] = cross_up.reindex(df.index, fill_value=False)
            df = df[df["cross_up"]]

        df = df[(df["last_close"] >= self.min_price) & (df["avg_vol"] >= self.min_avg_volume)]
        df = df[df["score"] > 0]
        df = df.sort_values("score", ascending=False).head(self.top_n)
        if df.empty:
            return []
        w = 1.0 / len(df)
        return [Pick(code=c, weight=w, reason=f"sma_score={r.score:+.3%}") for c, r in df.iterrows()]
