from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import RESULTS_DIR
from .engine import RunResult

log = logging.getLogger(__name__)


def _run_id(result: RunResult) -> str:
    params_str = json.dumps(result.strategy_params, sort_keys=True)
    h = hashlib.sha1(params_str.encode()).hexdigest()[:6]
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{ts}_{result.strategy_name}_{h}"


def write_report(result: RunResult, root: Path | None = None) -> Path:
    root = root or RESULTS_DIR
    run_dir = root / _run_id(result)
    run_dir.mkdir(parents=True, exist_ok=True)

    result.daily.to_csv(run_dir / "daily_returns.csv", index=False)
    result.trades.to_csv(run_dir / "trades.csv", index=False)

    meta = {
        "strategy": result.strategy_name,
        "params": result.strategy_params,
        "start": result.start.isoformat(),
        "end": result.end.isoformat(),
        "initial_budget": result.initial_budget,
        "budget_mode": result.budget_mode,
        "summary": result.summary,
    }
    (run_dir / "summary.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    _plot_equity_curve(result, run_dir / "equity_curve.png")
    _plot_daily_returns(result, run_dir / "daily_returns.png")

    log.info("report written to %s", run_dir)
    return run_dir


def _plot_equity_curve(result: RunResult, path: Path) -> None:
    df = result.daily
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    if result.budget_mode == "compound":
        equity = df["budget"] + df["pnl"]
    else:
        equity = result.initial_budget + df["cumulative_pnl"]
    ax.plot(df["date"], equity, label="equity")
    ax.axhline(result.initial_budget, color="gray", linestyle="--", linewidth=1, label="initial")
    ax.set_title(f"{result.strategy_name}  {result.start.date()} ~ {result.end.date()}")
    ax.set_xlabel("date")
    ax.set_ylabel("equity (KRW)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_daily_returns(result: RunResult, path: Path) -> None:
    df = result.daily[result.daily["n_trades"] > 0]
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#2a9d8f" if r >= 0 else "#e76f51" for r in df["return_pct"]]
    ax.bar(df["date"], df["return_pct"] * 100, color=colors, width=0.9)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title("daily return (%)")
    ax.set_ylabel("%")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def print_summary(result: RunResult) -> None:
    s = result.summary
    if not s:
        print("(no results)")
        return
    print(f"strategy   : {result.strategy_name}  params={result.strategy_params}")
    print(f"period     : {result.start.date()} ~ {result.end.date()}  ({s.get('trading_days',0)} trading days)")
    print(f"budget mode: {result.budget_mode}  initial={result.initial_budget:,.0f}")
    print(f"total pnl  : {s['total_pnl']:>14,.0f}  ({s['total_return']:+.2%})")
    print(f"win rate   : {s['win_rate']:.1%}  ({s['win_days']}/{s['trading_days']})")
    print(f"avg daily  : {s['avg_daily_return']:+.3%}   sigma={s['stdev_daily_return']:.3%}")
    print(f"sharpe~    : {s['sharpe_approx']:.2f}")
    print(f"best/worst : {s['max_daily_gain']:+.2%}  /  {s['max_daily_loss']:+.2%}")
