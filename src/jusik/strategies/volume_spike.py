from __future__ import annotations

import pandas as pd

from .base import Pick, Strategy


class VolumeSpikeStrategy(Strategy):
    """Buy names where yesterday's volume was a multiple of recent average.

    Score = (yesterday volume) / (avg volume over `window` days). Optionally
    require the previous day to be green so we ride continued interest.
    """

    name = "volume_spike"

    def __init__(
        self,
        top_n: int = 5,
        window: int = 20,
        require_green: bool = True,
        min_price: float = 1_000.0,
        min_avg_volume: float = 100_000.0,
    ) -> None:
        self.top_n = top_n
        self.window = window
        self.require_green = require_green
        self.min_price = min_price
        self.min_avg_volume = min_avg_volume
        self.params = dict(
            top_n=top_n,
            window=window,
            require_green=int(require_green),
            min_price=min_price,
            min_avg_volume=min_avg_volume,
        )

    def select(self, asof: pd.Timestamp, history: pd.DataFrame) -> list[Pick]:
        hist = history[history["date"] <= asof]
        dates = sorted(hist["date"].unique())
        if len(dates) < self.window + 1:
            return []
        window_dates = dates[-self.window:]
        last_date = dates[-1]

        windowed = hist[hist["date"].isin(window_dates)]
        avg_vol = windowed.groupby("code")["Volume"].mean()

        last = hist[hist["date"] == last_date].set_index("code")
        prev_last = hist[hist["date"] == dates[-2]].set_index("code")

        df = pd.DataFrame({
            "last_close": last["Close"],
            "last_open": last["Open"],
            "last_volume": last["Volume"],
            "prev_close": prev_last["Close"],
            "avg_vol": avg_vol,
        }).dropna()
        df["vol_ratio"] = df["last_volume"] / df["avg_vol"]
        df["day_return"] = df["last_close"] / df["prev_close"] - 1

        df = df[(df["last_close"] >= self.min_price) & (df["avg_vol"] >= self.min_avg_volume)]
        if self.require_green:
            df = df[df["day_return"] > 0]
        df = df.sort_values("vol_ratio", ascending=False).head(self.top_n)
        if df.empty:
            return []
        w = 1.0 / len(df)
        return [
            Pick(code=c, weight=w, reason=f"vol_ratio={r.vol_ratio:.2f}")
            for c, r in df.iterrows()
        ]
