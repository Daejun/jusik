from __future__ import annotations

import pandas as pd

from .base import Pick, Strategy


class LowVolStrategy(Strategy):
    """Pick names with lowest realised volatility over `window` days.

    Acts as a defensive baseline; tends to produce small, steady returns
    rather than fireworks.
    """

    name = "low_vol"

    def __init__(
        self,
        top_n: int = 5,
        window: int = 20,
        min_price: float = 1_000.0,
        min_avg_volume: float = 100_000.0,
    ) -> None:
        self.top_n = top_n
        self.window = window
        self.min_price = min_price
        self.min_avg_volume = min_avg_volume
        self.params = dict(
            top_n=top_n, window=window,
            min_price=min_price, min_avg_volume=min_avg_volume,
        )

    def select(self, asof: pd.Timestamp, history: pd.DataFrame) -> list[Pick]:
        hist = history[history["date"] <= asof]
        dates = sorted(hist["date"].unique())
        if len(dates) < self.window + 1:
            return []
        window_dates = dates[-self.window:]
        window = hist[hist["date"].isin(window_dates)]
        last_date = dates[-1]

        rows = []
        for code, g in window.groupby("code"):
            g = g.sort_values("date")
            ret = g["Close"].pct_change().dropna()
            if len(ret) < self.window - 1:
                continue
            last = g[g["date"] == last_date]
            if last.empty:
                continue
            rows.append({
                "code": code,
                "vol": float(ret.std()),
                "last_close": float(last["Close"].iloc[0]),
                "avg_vol": float(g["Volume"].mean()),
            })
        df = pd.DataFrame(rows).set_index("code")
        if df.empty:
            return []
        df = df[(df["last_close"] >= self.min_price) & (df["avg_vol"] >= self.min_avg_volume)]
        df = df[df["vol"] > 0]
        df = df.sort_values("vol", ascending=True).head(self.top_n)
        if df.empty:
            return []
        w = 1.0 / len(df)
        return [Pick(code=c, weight=w, reason=f"sigma={r.vol:.3%}") for c, r in df.iterrows()]
