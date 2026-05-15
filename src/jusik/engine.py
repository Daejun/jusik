from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import pandas as pd

from .config import SimConfig, TradeCost
from .data import MarketData, load_market_data
from .portfolio import DayResult, simulate_day
from .strategies.base import Strategy

log = logging.getLogger(__name__)

BudgetMode = Literal["fixed", "compound"]


@dataclass
class RunResult:
    strategy_name: str
    strategy_params: dict
    start: pd.Timestamp
    end: pd.Timestamp
    initial_budget: float
    budget_mode: BudgetMode
    daily: pd.DataFrame
    trades: pd.DataFrame
    summary: dict = field(default_factory=dict)


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
    n_days = int((daily["n_trades"] > 0).sum())
    win_days = int((daily["pnl"] > 0).sum())
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
        "trading_days": n_days,
        "win_days": win_days,
        "win_rate": (win_days / n_days) if n_days else 0.0,
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
) -> RunResult:
    cfg = config or SimConfig()
    if market is None:
        market = load_market_data(start, end, top_n=cfg.universe_size)

    panel = market.panel
    if panel.empty:
        raise RuntimeError("no market data loaded")

    dates = sorted(panel["date"].unique())
    day_results: list[DayResult] = []
    budget = cfg.budget

    for i, dt in enumerate(dates):
        # use history strictly before dt to pick today's trades
        history = panel[panel["date"] < dt]
        if history.empty:
            continue
        prev_asof = history["date"].max()
        picks = strategy.select(prev_asof, history)
        ohlcv_day = panel[panel["date"] == dt]
        if not picks:
            day_results.append(DayResult(date=dt, budget=budget, invested=0.0, pnl=0.0,
                                         return_pct=0.0, note="no picks"))
            continue
        dr = simulate_day(dt, picks, ohlcv_day, budget=budget, cost=cfg.cost)
        day_results.append(dr)
        if budget_mode == "compound":
            budget = budget + dr.pnl
            if budget <= 0:
                log.warning("budget depleted at %s", dt)
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
        daily=daily,
        trades=trades,
        summary=summary,
    )
