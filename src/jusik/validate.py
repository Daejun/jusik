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
from .engine import TRADE_MODES, run_backtest
from .grid import DEFAULT_GRIDS, _expand
from .strategies import build

log = logging.getLogger(__name__)


def walk_forward(
    start,
    end,
    *,
    is_pct: float = 0.7,
    top_k: int = 5,
    rank_by: str = "sharpe_approx",
    config: SimConfig | None = None,
    grids: dict | None = None,
    trade_modes: list[str] | None = None,
    out_dir: Path | None = None,
) -> Path:
    """In-sample grid search, then validate top-K on the held-out tail.

    is_pct: fraction of the date range to use for in-sample tuning.
    """
    cfg = config or SimConfig()
    grids = grids or DEFAULT_GRIDS
    trade_modes = trade_modes or ["overnight", "multiday5"]

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = out_dir or (RESULTS_DIR / f"{ts}_walkforward")
    out.mkdir(parents=True, exist_ok=True)

    market: MarketData = load_market_data(start, end, top_n=cfg.universe_size)
    # extend with benchmark codes if needed
    if "benchmark" in grids:
        from .data import fetch_ohlcv
        extra = fetch_ohlcv(["069500"], start, end)
        frames = [market.panel]
        for c, df in extra.items():
            tmp = df.copy(); tmp["code"] = c; tmp.index.name = "date"
            frames.append(tmp.reset_index())
        market.panel = pd.concat(frames, ignore_index=True).sort_values(
            ["date", "code"]).reset_index(drop=True)

    dates = sorted(market.panel["date"].unique())
    if len(dates) < 100:
        raise RuntimeError(f"not enough dates ({len(dates)}) — extend the window")
    split_idx = int(len(dates) * is_pct)
    is_start = dates[0].date()
    is_end = dates[split_idx - 1].date()
    oos_start = dates[split_idx].date()
    oos_end = dates[-1].date()
    log.info("IS: %s ~ %s (%d days)  |  OOS: %s ~ %s (%d days)",
             is_start, is_end, split_idx, oos_start, oos_end, len(dates) - split_idx)

    is_rows = []
    for tm in trade_modes:
        if tm not in TRADE_MODES:
            continue
        for strategy_name, grid in grids.items():
            for params in _expand(grid):
                label = f"[{tm}] {strategy_name}({','.join(f'{k}={v}' for k,v in params.items())})"
                try:
                    s = build(strategy_name, **params)
                    r = run_backtest(strategy=s, start=is_start, end=is_end,
                                     config=cfg, budget_mode="fixed",
                                     market=market, trade_mode=tm)
                except Exception as e:  # noqa: BLE001
                    log.warning("%s failed: %s", label, e)
                    continue
                row = {
                    "label": label, "trade_mode": tm, "strategy": strategy_name,
                    "params": json.dumps(params, sort_keys=True),
                    **{k: r.summary.get(k) for k in (
                        "total_return", "sharpe_approx", "win_rate",
                        "avg_daily_return", "stdev_daily_return", "max_daily_loss",
                    )},
                }
                is_rows.append(row)

    is_df = pd.DataFrame(is_rows).sort_values(rank_by, ascending=False)
    is_df.to_csv(out / "is_leaderboard.csv", index=False)

    top = is_df.head(top_k).copy()
    log.info("top-%d IS picks:\n%s", top_k, top[["label", "total_return", "sharpe_approx"]])

    oos_rows = []
    is_eq_frames: list[tuple[str, pd.DataFrame]] = []
    oos_eq_frames: list[tuple[str, pd.DataFrame]] = []
    for _, row in top.iterrows():
        tm = row["trade_mode"]
        strategy_name = row["strategy"]
        params = json.loads(row["params"])
        label = row["label"]
        s = build(strategy_name, **params)
        # IS run again to get equity (we discarded daily earlier)
        is_r = run_backtest(strategy=s, start=is_start, end=is_end,
                            config=cfg, budget_mode="fixed",
                            market=market, trade_mode=tm)
        is_eq_frames.append((label, is_r.daily))
        oos_r = run_backtest(strategy=s, start=oos_start, end=oos_end,
                             config=cfg, budget_mode="fixed",
                             market=market, trade_mode=tm)
        oos_eq_frames.append((label, oos_r.daily))
        oos_rows.append({
            "label": label,
            "trade_mode": tm,
            "strategy": strategy_name,
            "params": row["params"],
            "is_return": row["total_return"],
            "is_sharpe": row["sharpe_approx"],
            "is_winrate": row["win_rate"],
            "oos_return": oos_r.summary.get("total_return", 0),
            "oos_sharpe": oos_r.summary.get("sharpe_approx", 0),
            "oos_winrate": oos_r.summary.get("win_rate", 0),
            "oos_max_loss": oos_r.summary.get("max_daily_loss", 0),
        })

    oos_df = pd.DataFrame(oos_rows)
    oos_df.to_csv(out / "comparison.csv", index=False)

    _plot_is_oos(is_eq_frames, oos_eq_frames, cfg.budget, out / "is_oos_equity.png",
                 is_end, oos_start)
    _plot_compare_bars(oos_df, out / "is_vs_oos_bars.png")

    (out / "meta.json").write_text(json.dumps({
        "start": str(start), "end": str(end),
        "is_split_pct": is_pct, "rank_by": rank_by, "top_k": top_k,
        "is_start": str(is_start), "is_end": str(is_end),
        "oos_start": str(oos_start), "oos_end": str(oos_end),
        "trade_modes": trade_modes,
    }, indent=2, ensure_ascii=False))

    log.info("walkforward report at %s", out)
    return out


def _plot_is_oos(is_frames, oos_frames, initial_budget, path, is_end, oos_start):
    fig, ax = plt.subplots(figsize=(11, 6))
    palette = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51",
               "#6a4c93", "#264653", "#6c757d"]
    for i, ((label, is_df), (_, oos_df)) in enumerate(zip(is_frames, oos_frames)):
        color = palette[i % len(palette)]
        # IS equity (solid)
        if not is_df.empty:
            eq = initial_budget + is_df["cumulative_pnl"]
            ax.plot(is_df["date"], eq, color=color, linewidth=1.4, label=label)
        # OOS equity (dashed, shifted so the IS endpoint isn't reused)
        if not oos_df.empty:
            eq = initial_budget + oos_df["cumulative_pnl"]
            ax.plot(oos_df["date"], eq, color=color, linewidth=1.4, linestyle="--")
    ax.axvline(pd.Timestamp(oos_start), color="black", linestyle=":", linewidth=1)
    ax.axhline(initial_budget, color="gray", linestyle="--", linewidth=0.7)
    ax.text(pd.Timestamp(oos_start), ax.get_ylim()[1] * 0.95,
            " OOS →", fontsize=9, va="top")
    ax.set_title("In-Sample (solid) vs Out-of-Sample (dashed)")
    ax.set_ylabel("equity (KRW, fixed budget per period)")
    ax.legend(fontsize=7, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_compare_bars(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(df))))
    y = range(len(df))
    width = 0.4
    ax.barh([i + width / 2 for i in y], df["is_return"], height=width,
            color="#2a9d8f", label="In-Sample")
    ax.barh([i - width / 2 for i in y], df["oos_return"], height=width,
            color="#e76f51", label="Out-of-Sample")
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["label"], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("total return")
    ax.set_title("IS vs OOS — total return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
