from __future__ import annotations

import pandas as pd

from .base import Pick, Strategy


class GapDownStrategy(Strategy):
    """Mean-reversion bet: yesterday's biggest losers often bounce intraday.

    Picks the bottom `top_n` by previous-day return (most negative). Filters out
    extreme tail moves that are usually news-driven and unlikely to revert.
    """

    name = "gap_down"

    def __init__(
        self,
        top_n: int = 5,
        lookback: int = 1,
        min_price: float = 1_000.0,
        min_avg_volume: float = 100_000.0,
        min_prev_return: float = -0.20,
    ) -> None:
        self.top_n = top_n
        self.lookback = lookback
        self.min_price = min_price
        self.min_avg_volume = min_avg_volume
        self.min_prev_return = min_prev_return
        self.params = dict(
            top_n=top_n,
            lookback=lookback,
            min_price=min_price,
            min_avg_volume=min_avg_volume,
            min_prev_return=min_prev_return,
        )

    def select(self, asof: pd.Timestamp, history: pd.DataFrame) -> list[Pick]:
        hist = history[history["date"] <= asof]
        dates = sorted(hist["date"].unique())
        if len(dates) < self.lookback + 1:
            return []
        recent = hist[hist["date"].isin(dates[-(self.lookback + 1):])]
        first_close = recent.groupby("code").first()["Close"]
        last_row = recent[recent["date"] == dates[-1]].set_index("code")
        last_close = last_row["Close"]
        avg_vol = recent.groupby("code")["Volume"].mean()

        df = pd.DataFrame({
            "prev_return": last_close / first_close - 1,
            "last_close": last_close,
            "avg_vol": avg_vol,
        }).dropna()
        df = df[(df["last_close"] >= self.min_price) & (df["avg_vol"] >= self.min_avg_volume)]
        df = df[df["prev_return"] >= self.min_prev_return]
        df = df.sort_values("prev_return", ascending=True).head(self.top_n)
        if df.empty:
            return []
        w = 1.0 / len(df)
        return [
            Pick(code=c, weight=w, reason=f"prev_return={r.prev_return:.3%}")
            for c, r in df.iterrows()
        ]
