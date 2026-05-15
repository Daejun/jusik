from __future__ import annotations

import itertools
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
from .engine import TRADE_MODES, run_backtest
from .strategies import build

log = logging.getLogger(__name__)


# default grids per strategy: lists of param dicts to materialise
DEFAULT_GRIDS: dict[str, dict[str, list]] = {
    "momentum": {"top_n": [3, 5], "lookback": [1, 3]},
    "gap_down": {"top_n": [3, 5], "lookback": [1, 3]},
    "volume_spike": {"top_n": [3, 5], "window": [20], "require_green": [True, False]},
    "sma_cross": {"top_n": [3, 5], "short": [5], "long": [20, 60], "fresh_only": [False]},
    "rsi": {"top_n": [3, 5], "period": [14], "threshold": [25.0, 30.0]},
    "low_vol": {"top_n": [5], "window": [20, 60]},
    "benchmark": {"code": ["069500"]},
}


def _expand(grid: dict[str, list]) -> list[dict]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def run_grid(
    start,
    end,
    config: SimConfig | None = None,
    budget_mode: str = "fixed",
    grids: dict[str, dict[str, list]] | None = None,
    out_dir: Path | None = None,
    extra_codes: list[str] | None = None,
    trade_modes: list[str] | None = None,
) -> Path:
    cfg = config or SimConfig()
    grids = grids or DEFAULT_GRIDS
    trade_modes = trade_modes or ["intraday"]

    market: MarketData = load_market_data(start, end, top_n=cfg.universe_size)

    if extra_codes:
        from .data import fetch_ohlcv
        extra = fetch_ohlcv(extra_codes, start, end)
        frames = [market.panel]
        for code, df in extra.items():
            tmp = df.copy()
            tmp["code"] = code
            tmp.index.name = "date"
            frames.append(tmp.reset_index())
        market.panel = pd.concat(frames, ignore_index=True).sort_values(["date", "code"]).reset_index(drop=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = out_dir or (RESULTS_DIR / f"{ts}_grid")
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    daily_frames: list[tuple[str, pd.DataFrame]] = []

    for tm in trade_modes:
        if tm not in TRADE_MODES:
            log.warning("unknown trade mode: %s — skip", tm)
            continue
        for strategy_name, grid in grids.items():
            for params in _expand(grid):
                base = f"{strategy_name}({','.join(f'{k}={v}' for k,v in params.items())})"
                label = f"[{tm}] {base}"
                try:
                    strategy = build(strategy_name, **params)
                    result = run_backtest(
                        strategy=strategy, start=start, end=end,
                        config=cfg, budget_mode=budget_mode, market=market,
                        trade_mode=tm,
                    )
                except Exception as e:  # noqa: BLE001
                    log.warning("%s failed: %s", label, e)
                    continue
                s = result.summary
                rows.append({
                    "label": label,
                    "trade_mode": tm,
                    "strategy": strategy_name,
                    "params": json.dumps(params, sort_keys=True),
                    **{k: s.get(k) for k in (
                        "total_pnl", "total_return", "trading_days", "win_rate",
                        "avg_daily_return", "stdev_daily_return", "sharpe_approx",
                        "max_daily_gain", "max_daily_loss",
                    )},
                })
                daily_frames.append((label, result.daily))
                log.info(
                    "%-70s ret=%+.2f%% sharpe=%+.2f wr=%.1f%%",
                    label,
                    (s.get("total_return", 0) or 0) * 100,
                    s.get("sharpe_approx", 0) or 0,
                    (s.get("win_rate", 0) or 0) * 100,
                )

    if not rows:
        log.warning("grid produced no results")
        return out

    lb = pd.DataFrame(rows).sort_values("sharpe_approx", ascending=False)
    lb.to_csv(out / "leaderboard.csv", index=False)

    top = lb.head(8)["label"].tolist()
    _plot_topk(top, daily_frames, cfg.budget, budget_mode, out / "top_equity.png")
    _plot_topk_sharpe(lb.head(15), out / "top_sharpe.png")

    (out / "meta.json").write_text(json.dumps({
        "start": str(start), "end": str(end),
        "budget_mode": budget_mode, "initial_budget": cfg.budget,
        "universe_size": cfg.universe_size,
        "strategies": list(grids.keys()),
        "total_runs": len(rows),
    }, indent=2, ensure_ascii=False))
    log.info("grid report at %s (%d runs)", out, len(rows))
    return out


def _plot_topk(top_labels, daily_frames, initial_budget, budget_mode, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, df in daily_frames:
        if label not in top_labels or df.empty:
            continue
        equity = (df["budget"] + df["pnl"]) if budget_mode == "compound" \
            else initial_budget + df["cumulative_pnl"]
        ax.plot(df["date"], equity, label=label, linewidth=1.2)
    ax.axhline(initial_budget, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"top-{len(top_labels)} by Sharpe — equity")
    ax.set_ylabel("equity (KRW)")
    ax.legend(fontsize=7, loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_topk_sharpe(top: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(top))))
    top = top.iloc[::-1]
    bars = ax.barh(top["label"], top["sharpe_approx"],
                   color=["#2a9d8f" if v >= 0 else "#e76f51" for v in top["sharpe_approx"]])
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Sharpe (annualised approx)")
    ax.set_title("leaderboard — top by Sharpe")
    for bar, val, ret in zip(bars, top["sharpe_approx"], top["total_return"]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f"  {val:+.2f} | ret={ret:+.1%}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
