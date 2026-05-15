from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import math
import pandas as pd

from .config import TradeCost
from .strategies.base import Pick


@dataclass
class TradeRow:
    date: pd.Timestamp
    code: str
    shares: int
    open_price: float
    close_price: float
    buy_cost: float
    sell_proceeds: float
    pnl: float
    return_pct: float


@dataclass
class DayResult:
    date: pd.Timestamp
    budget: float
    invested: float
    pnl: float
    return_pct: float
    trades: list[TradeRow] = field(default_factory=list)
    note: str = ""


def simulate_day(
    asof: pd.Timestamp,
    picks: Sequence[Pick],
    ohlcv_day: pd.DataFrame,
    budget: float,
    cost: TradeCost,
) -> DayResult:
    """Buy at Open, sell at Close, using integer share counts.

    ohlcv_day: rows for `asof` with columns date, code, Open, High, Low, Close, Volume.
    """
    if not picks or ohlcv_day.empty:
        return DayResult(date=asof, budget=budget, invested=0.0, pnl=0.0, return_pct=0.0,
                         note="no picks or no data")

    day = ohlcv_day.set_index("code")
    buy_rate = cost.buy_cost_rate()
    sell_rate = cost.sell_cost_rate()

    trades: list[TradeRow] = []
    invested = 0.0
    pnl = 0.0

    for pick in picks:
        if pick.code not in day.index:
            continue
        row = day.loc[pick.code]
        open_price = float(row["Open"])
        close_price = float(row["Close"])
        if open_price <= 0 or math.isnan(open_price) or math.isnan(close_price):
            continue
        per_pick_budget = budget * pick.weight
        effective_buy_price = open_price * (1 + buy_rate)
        shares = int(per_pick_budget // effective_buy_price)
        if shares <= 0:
            continue
        buy_cost = shares * open_price * (1 + buy_rate)
        sell_proceeds = shares * close_price * (1 - sell_rate)
        trade_pnl = sell_proceeds - buy_cost
        trades.append(TradeRow(
            date=asof,
            code=pick.code,
            shares=shares,
            open_price=open_price,
            close_price=close_price,
            buy_cost=buy_cost,
            sell_proceeds=sell_proceeds,
            pnl=trade_pnl,
            return_pct=trade_pnl / buy_cost if buy_cost else 0.0,
        ))
        invested += buy_cost
        pnl += trade_pnl

    return DayResult(
        date=asof,
        budget=budget,
        invested=invested,
        pnl=pnl,
        return_pct=(pnl / budget) if budget else 0.0,
        trades=trades,
    )
