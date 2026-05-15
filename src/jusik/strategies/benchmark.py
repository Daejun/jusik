from __future__ import annotations

import pandas as pd

from .base import Pick, Strategy


class BenchmarkStrategy(Strategy):
    """Trivial baseline: every day pick a fixed `code` (default KODEX 200 ETF)
    and trade its full open-to-close move.  Useful sanity check for the
    average intraday drift of the market.
    """

    name = "benchmark"

    def __init__(self, code: str = "069500", top_n: int = 1) -> None:
        self.code = code
        self.params = dict(code=code, top_n=top_n)

    def select(self, asof: pd.Timestamp, history: pd.DataFrame) -> list[Pick]:
        hist = history[(history["date"] <= asof) & (history["code"] == self.code)]
        if hist.empty:
            return []
        return [Pick(code=self.code, weight=1.0, reason="benchmark")]
