from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import pandas as pd

from .config import SimConfig, TradeCost
from .data import MarketData, load_market_data
from .portfolio import DayResult, TradeRow
from .strategies.base import Strategy

log = logging.getLogger(__name__)

BudgetMode = Literal["fixed", "compound"]


@dataclass(frozen=True)
class TradeMode:
    """Describes how each trade is priced and how long it is held.

    Three preset modes are exposed; see TRADE_MODES below.
    """

    name: str
    entry_col: str = "Open"
    exit_col: str = "Close"
    holding_days: int = 1
    step: int = 1
    overnight: bool = False


TRADE_MODES: dict[str, TradeMode] = {
    "intraday": TradeMode("intraday"),
    "overnight": TradeMode("overnight", entry_col="Close", exit_col="Open", overnight=True),
    "multiday5": TradeMode("multiday5", holding_days=5, step=5),
}


@dataclass
class RunResult:
    strategy_name: str
    strategy_params: dict
    start: pd.Timestamp
    end: pd.Timestamp
    initial_budget: float
    budget_mode: BudgetMode
    trade_mode: str
    daily: pd.DataFrame
    trades: pd.DataFrame
    summary: dict = field(default_factory=dict)


def _simulate_trade(
    decision_date: pd.Timestamp,
    buy_date: pd.Timestamp,
    sell_date: pd.Timestamp,
    picks,
    panel: pd.DataFrame,
    mode: TradeMode,
    budget: float,
    cost: TradeCost,
) -> DayResult:
    if not picks:
        return DayResult(date=buy_date, budget=budget, invested=0.0, pnl=0.0,
                         return_pct=0.0, note="no picks")
    buy_rows = panel[panel["date"] == buy_date].set_index("code")
    sell_rows = panel[panel["date"] == sell_date].set_index("code")
    if buy_rows.empty or sell_rows.empty:
        return DayResult(date=buy_date, budget=budget, invested=0.0, pnl=0.0,
                         return_pct=0.0, note="missing prices")

    buy_rate = cost.buy_cost_rate()
    sell_rate = cost.sell_cost_rate()
    trades: list[TradeRow] = []
    invested = 0.0
    pnl = 0.0

    for pick in picks:
        if pick.code not in buy_rows.index or pick.code not in sell_rows.index:
            continue
        bp = float(buy_rows.loc[pick.code, mode.entry_col])
        sp = float(sell_rows.loc[pick.code, mode.exit_col])
        if bp <= 0 or math.isnan(bp) or math.isnan(sp):
            continue
        per_pick_budget = budget * pick.weight
        effective_buy_price = bp * (1 + buy_rate)
        shares = int(per_pick_budget // effective_buy_price)
        if shares <= 0:
            continue
        buy_cost = shares * bp * (1 + buy_rate)
        sell_proceeds = shares * sp * (1 - sell_rate)
        trade_pnl = sell_proceeds - buy_cost
        trades.append(TradeRow(
            date=sell_date,
            code=pick.code,
            shares=shares,
            open_price=bp,
            close_price=sp,
            buy_cost=buy_cost,
            sell_proceeds=sell_proceeds,
            pnl=trade_pnl,
            return_pct=trade_pnl / buy_cost if buy_cost else 0.0,
        ))
        invested += buy_cost
        pnl += trade_pnl

    return DayResult(date=sell_date, budget=budget, invested=invested, pnl=pnl,
                     return_pct=(pnl / budget) if budget else 0.0, trades=trades)


def _trades_to_df(day_results: list[DayResult]) -> pd.DataFrame:
    rows = []
    for dr in day_results:
        for t in dr.trades:
            rows.append({
                "date": t.date,
                "code": t.code,
                "shares": t.shares,
                "open_price": t.open_price,
                "close_price": t.close_price,
                "buy_cost": t.buy_cost,
                "sell_proceeds": t.sell_proceeds,
                "pnl": t.pnl,
                "return_pct": t.return_pct,
            })
    return pd.DataFrame(rows)


def _daily_to_df(day_results: list[DayResult]) -> pd.DataFrame:
    rows = []
    equity = 0.0
    for dr in day_results:
        equity += dr.pnl
        rows.append({
            "date": dr.date,
            "budget": dr.budget,
            "invested": dr.invested,
            "pnl": dr.pnl,
            "return_pct": dr.return_pct,
            "cumulative_pnl": equity,
            "n_trades": len(dr.trades),
            "note": dr.note,
        })
    return pd.DataFrame(rows)


def _summarize(daily: pd.DataFrame, initial_budget: float, budget_mode: BudgetMode) -> dict:
    if daily.empty:
        return {}
    total_pnl = float(daily["cumulative_pnl"].iloc[-1])
    n_periods = int((daily["n_trades"] > 0).sum())
    win_periods = int((daily["pnl"] > 0).sum())
    if budget_mode == "fixed":
        total_return = total_pnl / initial_budget
    else:
        total_return = (daily["budget"].iloc[-1] + daily["pnl"].iloc[-1]) / initial_budget - 1
    daily_returns = daily.loc[daily["n_trades"] > 0, "return_pct"]
    sharpe = 0.0
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = float(daily_returns.mean() / daily_returns.std() * (252 ** 0.5))
    return {
        "total_pnl": total_pnl,
        "total_return": float(total_return),
        "trading_days": n_periods,
        "win_days": win_periods,
        "win_rate": (win_periods / n_periods) if n_periods else 0.0,
        "avg_daily_return": float(daily_returns.mean()) if not daily_returns.empty else 0.0,
        "stdev_daily_return": float(daily_returns.std()) if len(daily_returns) > 1 else 0.0,
        "sharpe_approx": sharpe,
        "max_daily_gain": float(daily_returns.max()) if not daily_returns.empty else 0.0,
        "max_daily_loss": float(daily_returns.min()) if not daily_returns.empty else 0.0,
    }


def run_backtest(
    strategy: Strategy,
    start: date,
    end: date,
    config: SimConfig | None = None,
    budget_mode: BudgetMode = "fixed",
    market: MarketData | None = None,
    trade_mode: str | TradeMode = "intraday",
) -> RunResult:
    cfg = config or SimConfig()
    mode = trade_mode if isinstance(trade_mode, TradeMode) else TRADE_MODES[trade_mode]
    if market is None:
        market = load_market_data(start, end, top_n=cfg.universe_size)

    panel = market.panel
    if panel.empty:
        raise RuntimeError("no market data loaded")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    panel = panel[(panel["date"] >= start_ts) & (panel["date"] <= end_ts)]
    if panel.empty:
        raise RuntimeError(f"market panel has no rows in [{start}, {end}]")

    dates = sorted(panel["date"].unique())
    day_results: list[DayResult] = []
    budget = cfg.budget

    def _iter_decisions():
        if mode.overnight:
            # signal on date i close, buy that close, sell next open
            for i in range(len(dates) - 1):
                yield dates[i], dates[i], dates[i + 1]
        else:
            # signal on date i-1 close, buy on i open, sell on i+holding-1 close
            i = 1
            while i + mode.holding_days - 1 < len(dates):
                decision = dates[i - 1]
                buy = dates[i]
                sell = dates[i + mode.holding_days - 1]
                yield decision, buy, sell
                i += mode.step

    for decision, buy_d, sell_d in _iter_decisions():
        history = panel[panel["date"] <= decision]
        if history.empty:
            continue
        picks = strategy.select(decision, history)
        if not picks:
            day_results.append(DayResult(date=sell_d, budget=budget, invested=0.0,
                                         pnl=0.0, return_pct=0.0, note="no picks"))
            continue
        dr = _simulate_trade(decision, buy_d, sell_d, picks, panel, mode,
                             budget=budget, cost=cfg.cost)
        day_results.append(dr)
        if budget_mode == "compound":
            budget = budget + dr.pnl
            if budget <= 0:
                log.warning("budget depleted at %s", sell_d)
                break

    daily = _daily_to_df(day_results)
    trades = _trades_to_df(day_results)
    summary = _summarize(daily, cfg.budget, budget_mode)
    return RunResult(
        strategy_name=strategy.name,
        strategy_params=getattr(strategy, "params", {}),
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
        initial_budget=cfg.budget,
        budget_mode=budget_mode,
        trade_mode=mode.name,
        daily=daily,
        trades=trades,
        summary=summary,
    )
