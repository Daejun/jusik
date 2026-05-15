from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import RESULTS_DIR, SimConfig, TradeCost
from .data import MarketData, load_market_data
from .engine import TRADE_MODES, run_backtest
from .strategies import build

log = logging.getLogger(__name__)


FORWARD_ROOT = RESULTS_DIR / "forward"


@dataclass
class ForwardConfig:
    name: str
    start_date: str
    strategy_name: str
    strategy_params: dict
    trade_mode: str = "multiday5"
    budget_mode: str = "compound"
    initial_budget: float = 10_000_000.0
    universe_size: int = 200

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "ForwardConfig":
        return cls(**d)


def session_dir(name: str) -> Path:
    return FORWARD_ROOT / name


def _load_config(name: str) -> ForwardConfig:
    p = session_dir(name) / "config.json"
    if not p.exists():
        raise FileNotFoundError(f"forward session '{name}' not found (expected {p})")
    return ForwardConfig.from_dict(json.loads(p.read_text()))


def list_sessions() -> list[str]:
    if not FORWARD_ROOT.exists():
        return []
    return sorted(p.name for p in FORWARD_ROOT.iterdir() if (p / "config.json").exists())


def init_session(
    name: str,
    strategy_name: str,
    strategy_params: dict,
    *,
    start_date: date | None = None,
    trade_mode: str = "multiday5",
    budget_mode: str = "compound",
    initial_budget: float = 10_000_000.0,
    universe_size: int = 200,
) -> ForwardConfig:
    if trade_mode not in TRADE_MODES:
        raise ValueError(f"unknown trade_mode: {trade_mode}. options: {list(TRADE_MODES)}")
    out = session_dir(name)
    if out.exists() and (out / "config.json").exists():
        raise FileExistsError(f"session '{name}' already exists at {out}")
    out.mkdir(parents=True, exist_ok=True)
    cfg = ForwardConfig(
        name=name,
        start_date=(start_date or (date.today() - timedelta(days=730))).isoformat(),
        strategy_name=strategy_name,
        strategy_params=strategy_params,
        trade_mode=trade_mode,
        budget_mode=budget_mode,
        initial_budget=initial_budget,
        universe_size=universe_size,
    )
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
    log.info("forward session initialised at %s", out)
    return cfg


def run_session(name: str, as_of: date | None = None) -> dict:
    cfg = _load_config(name)
    out = session_dir(name)

    end = as_of or (date.today() - timedelta(days=1))
    start = date.fromisoformat(cfg.start_date)
    if end <= start:
        raise ValueError(f"end ({end}) must be after start ({start})")

    sim = SimConfig(budget=cfg.initial_budget, universe_size=cfg.universe_size,
                    cost=TradeCost())
    market: MarketData = load_market_data(start, end, top_n=cfg.universe_size)
    if cfg.strategy_name == "benchmark":
        from .data import fetch_ohlcv
        code = cfg.strategy_params.get("code", "069500")
        extra = fetch_ohlcv([code], start, end)
        frames = [market.panel]
        for c, df in extra.items():
            tmp = df.copy(); tmp["code"] = c; tmp.index.name = "date"
            frames.append(tmp.reset_index())
        market.panel = pd.concat(frames, ignore_index=True).sort_values(
            ["date", "code"]).reset_index(drop=True)

    strategy = build(cfg.strategy_name, **cfg.strategy_params)
    result = run_backtest(
        strategy=strategy, start=start, end=end,
        config=sim, budget_mode=cfg.budget_mode,
        market=market, trade_mode=cfg.trade_mode,
    )

    result.daily.to_csv(out / "daily.csv", index=False)
    result.trades.to_csv(out / "trades.csv", index=False)
    summary = {
        "name": name,
        "as_of": end.isoformat(),
        "strategy": cfg.strategy_name,
        "params": cfg.strategy_params,
        "trade_mode": cfg.trade_mode,
        "budget_mode": cfg.budget_mode,
        "initial_budget": cfg.initial_budget,
        "summary": result.summary,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    _plot_equity(result, cfg, out / "equity.png")

    # tomorrow's picks: use latest data to suggest a pick
    next_picks = _suggest_next(strategy, market.panel, cfg, end)
    (out / "next_picks.json").write_text(json.dumps(next_picks, indent=2, ensure_ascii=False, default=str))

    summary["next_picks"] = next_picks
    return summary


def _suggest_next(strategy, panel: pd.DataFrame, cfg: ForwardConfig, end: date) -> dict:
    if panel.empty:
        return {"as_of": str(end), "picks": [], "note": "no data"}
    asof = panel["date"].max()
    picks = strategy.select(asof, panel)
    mode = TRADE_MODES[cfg.trade_mode]
    next_action = _next_action(mode, panel, cfg.trade_mode)
    return {
        "as_of": str(asof.date()),
        "trade_mode": cfg.trade_mode,
        "next_action": next_action,
        "picks": [
            {"code": p.code, "weight": p.weight, "reason": p.reason}
            for p in picks
        ],
    }


def _next_action(mode, panel: pd.DataFrame, mode_name: str) -> str:
    if mode_name == "intraday":
        return "buy at next session open, sell at same-day close"
    if mode_name == "overnight":
        return "buy at today's close (already executed), sell at next session open"
    if mode_name == "multiday5":
        return "buy at next session open, sell at close of 5th trading day"
    return ""


def _plot_equity(result, cfg: ForwardConfig, path: Path) -> None:
    df = result.daily
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    if cfg.budget_mode == "compound":
        equity = df["budget"] + df["pnl"]
    else:
        equity = cfg.initial_budget + df["cumulative_pnl"]
    ax.plot(df["date"], equity, color="#264653", linewidth=1.5)
    ax.axhline(cfg.initial_budget, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"{cfg.name} — {cfg.strategy_name}[{cfg.trade_mode}] {cfg.budget_mode}")
    ax.set_ylabel("equity (KRW)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def status(name: str) -> dict:
    out = session_dir(name)
    if not out.exists():
        raise FileNotFoundError(name)
    cfg = _load_config(name)
    info: dict = {"name": name, "config": cfg.to_dict()}
    s = out / "summary.json"
    if s.exists():
        info["latest"] = json.loads(s.read_text())
    n = out / "next_picks.json"
    if n.exists():
        info["next_picks"] = json.loads(n.read_text())
    return info
