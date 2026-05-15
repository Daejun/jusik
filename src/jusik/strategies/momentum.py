from __future__ import annotations

import pandas as pd

from .base import Pick, Strategy


class MomentumStrategy(Strategy):
    """Buy the previous-day top gainers and hope momentum carries into next day's session.

    For each candidate, compute previous-day return = (Close / prior Close - 1) over
    the last `lookback` days, then pick the top `top_n` by that score. Optionally
    filter by min price and min average volume.
    """

    name = "momentum"

    def __init__(
        self,
        top_n: int = 5,
        lookback: int = 1,
        min_price: float = 1_000.0,
        min_avg_volume: float = 100_000.0,
        max_prev_return: float = 0.29,
    ) -> None:
        self.top_n = top_n
        self.lookback = lookback
        self.min_price = min_price
        self.min_avg_volume = min_avg_volume
        self.max_prev_return = max_prev_return
        self.params = dict(
            top_n=top_n,
            lookback=lookback,
            min_price=min_price,
            min_avg_volume=min_avg_volume,
            max_prev_return=max_prev_return,
        )

    def select(self, asof: pd.Timestamp, history: pd.DataFrame) -> list[Pick]:
        hist = history[history["date"] <= asof]
        if hist.empty:
            return []

        latest_dates = sorted(hist["date"].unique())
        if len(latest_dates) < self.lookback + 1:
            return []

        recent = hist[hist["date"].isin(latest_dates[-(self.lookback + 1):])]
        first_close = recent.groupby("code").first()["Close"]
        last_row = recent[recent["date"] == latest_dates[-1]].set_index("code")
        last_close = last_row["Close"]
        avg_vol = recent.groupby("code")["Volume"].mean()

        df = pd.DataFrame({
            "prev_return": last_close / first_close - 1,
            "last_close": last_close,
            "avg_vol": avg_vol,
        }).dropna()

        df = df[(df["last_close"] >= self.min_price) & (df["avg_vol"] >= self.min_avg_volume)]
        df = df[df["prev_return"] <= self.max_prev_return]
        df = df.sort_values("prev_return", ascending=False).head(self.top_n)

        if df.empty:
            return []

        weight = 1.0 / len(df)
        return [
            Pick(code=code, weight=weight, reason=f"prev_return={row.prev_return:.3%}")
            for code, row in df.iterrows()
        ]
