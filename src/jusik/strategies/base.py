from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class Pick:
    code: str
    weight: float
    reason: str = ""


class Strategy(ABC):
    name: str = "base"

    @abstractmethod
    def select(self, asof: pd.Timestamp, history: pd.DataFrame) -> list[Pick]:
        """Return picks for trading on the day AFTER `asof`.

        history: panel rows with date <= asof, columns [date, code, Open, High, Low, Close, Volume].
        Returned weights are fractions of the daily budget; sum should be <= 1.
        """

    def describe(self) -> dict:
        return {"name": self.name, "params": getattr(self, "params", {})}
