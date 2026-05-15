from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import RESULTS_DIR, SimConfig
from .data import MarketData, load_market_data
from .engine import RunResult, run_backtest
from .strategies import build

log = logging.getLogger(__name__)


def compare_strategies(
    strategy_specs: list[dict],
    start,
    end,
    config: SimConfig | None = None,
    budget_mode: str = "fixed",
    out_dir: Path | None = None,
) -> Path:
    """Run multiple strategies on the same market data and write a comparison report.

    strategy_specs: list of {"name": str, "params": dict, "label": optional str}.
    """
    cfg = config or SimConfig()
    market: MarketData = load_market_data(start, end, top_n=cfg.universe_size)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = out_dir or (RESULTS_DIR / f"{ts}_compare")
    out.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, RunResult]] = []
    rows = []
    for spec in strategy_specs:
        name = spec["name"]
        params = spec.get("params", {})
        label = spec.get("label") or f"{name}({','.join(f'{k}={v}' for k,v in params.items())})"
        strategy = build(name, **params)
        result = run_backtest(
            strategy=strategy,
            start=start,
            end=end,
            config=cfg,
            budget_mode=budget_mode,
            market=market,
        )
        results.append((label, result))
        s = result.summary
        rows.append({
            "label": label,
            "strategy": name,
            "params": json.dumps(params, sort_keys=True),
            **{k: s.get(k) for k in (
                "total_pnl", "total_return", "trading_days", "win_rate",
                "avg_daily_return", "stdev_daily_return", "sharpe_approx",
                "max_daily_gain", "max_daily_loss",
            )},
        })

    leaderboard = pd.DataFrame(rows).sort_values("total_return", ascending=False)
    leaderboard.to_csv(out / "leaderboard.csv", index=False)

    _plot_compare_equity(results, cfg.budget, budget_mode, out / "equity_compare.png")
    _plot_compare_dist(results, out / "daily_return_dist.png")

    (out / "meta.json").write_text(json.dumps({
        "start": str(start),
        "end": str(end),
        "budget_mode": budget_mode,
        "initial_budget": cfg.budget,
        "specs": strategy_specs,
    }, indent=2, ensure_ascii=False))

    log.info("comparison report at %s", out)
    return out


def _plot_compare_equity(results, initial_budget, budget_mode, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for label, r in results:
        if r.daily.empty:
            continue
        equity = (r.daily["budget"] + r.daily["pnl"]) if budget_mode == "compound" \
            else initial_budget + r.daily["cumulative_pnl"]
        ax.plot(r.daily["date"], equity, label=label, linewidth=1.5)
    ax.axhline(initial_budget, color="gray", linestyle="--", linewidth=1)
    ax.set_title("equity curve comparison")
    ax.set_ylabel("equity (KRW)")
    ax.legend(loc="best", fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_compare_dist(results, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = []
    series = []
    for label, r in results:
        rets = r.daily.loc[r.daily["n_trades"] > 0, "return_pct"]
        if rets.empty:
            continue
        series.append(rets.values * 100)
        labels.append(label)
    if not series:
        plt.close(fig)
        return
    ax.boxplot(series, labels=labels, showmeans=True)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title("daily return distribution (%)")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
