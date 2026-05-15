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


def _format_won(amount: float) -> str:
    return f"{amount:>14,.0f}원"


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

    # last few closed trades for the daily report
    if not result.trades.empty:
        recent = result.trades.sort_values("date", ascending=False).head(8)
        summary["recent_trades"] = recent.to_dict(orient="records")
    else:
        summary["recent_trades"] = []

    # current equity for budget allocation
    if not result.daily.empty:
        last = result.daily.iloc[-1]
        equity_now = (last["budget"] + last["pnl"]) if cfg.budget_mode == "compound" \
            else cfg.initial_budget + last["cumulative_pnl"]
    else:
        equity_now = cfg.initial_budget
    summary["current_equity"] = float(equity_now)
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
        return "다음 거래일 시가 매수 → 같은 날 종가 매도"
    if mode_name == "overnight":
        return "다음 거래일 종가 매수 → 그 다음 거래일 시가 매도"
    if mode_name == "multiday5":
        return "다음 거래일 시가 매수 → 5거래일 후 종가 매도"
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


def format_summary(summary: dict, name_lookup: dict[str, str] | None = None) -> str:
    """Render run_session output as a friendly multi-line block."""
    name_lookup = name_lookup or {}
    name = summary["name"]
    asof = summary["as_of"]
    s = summary.get("summary", {}) or {}
    params = summary.get("params", {})
    initial = summary.get("initial_budget", 0)
    equity = summary.get("current_equity", initial)
    pnl_abs = equity - initial

    bar = "━" * 60
    lines = [
        "",
        bar,
        f"  📌 {name}   ({asof} 기준)",
        f"     {summary.get('strategy', '')}({', '.join(f'{k}={v}' for k,v in params.items())})",
        f"     {summary.get('trade_mode', '')} / {summary.get('budget_mode', '')}",
        bar,
        "",
        "📊 누적 성과",
        f"    초기예산        {initial:>14,.0f} 원",
        f"    현재자본        {equity:>14,.0f} 원  ({(equity/initial - 1) if initial else 0:+.2%})",
        f"    누적 손익       {pnl_abs:>+14,.0f} 원",
    ]
    if s:
        lines += [
            f"    승률            {s.get('win_rate',0):.1%}  "
            f"({s.get('win_days',0)}승 {s.get('trading_days',0)-s.get('win_days',0)}패)",
            f"    Sharpe~         {s.get('sharpe_approx',0):.2f}",
            f"    최고/최저 일수익  {s.get('max_daily_gain',0):+.2%}  /  {s.get('max_daily_loss',0):+.2%}",
        ]

    recent = summary.get("recent_trades") or []
    if recent:
        last_date = recent[0]["date"]
        lines += ["", f"📉 최근 청산 (마지막: {str(last_date)[:10]})"]
        day_pnl = 0.0
        latest = [r for r in recent if str(r["date"])[:10] == str(last_date)[:10]]
        for r in latest[:6]:
            name_kr = name_lookup.get(r["code"], "")
            arrow = "↑" if r["pnl"] >= 0 else "↓"
            lines.append(
                f"    {r['code']:>8} {name_kr:<8} "
                f"{int(r['shares']):>4}주  "
                f"{r['open_price']:>9,.0f}→{r['close_price']:>9,.0f}  "
                f"{arrow}{r['pnl']:>+11,.0f} 원 ({r['return_pct']:+.2%})"
            )
            day_pnl += r["pnl"]
        if latest:
            lines.append(f"    {'합계':>13}                                {day_pnl:>+11,.0f} 원")

    nxt = summary.get("next_picks") or {}
    picks = nxt.get("picks", [])
    lines += ["", "🔔 다음 거래일 진입 추천", f"    {nxt.get('next_action', '')}"]
    if not picks:
        lines.append("    (없음 — 전략이 후보를 찾지 못함)")
    else:
        for p in picks:
            name_kr = name_lookup.get(p["code"], "")
            alloc = equity * p["weight"]
            lines.append(
                f"    매수 {p['code']:>8} {name_kr:<8}  w={p['weight']:.0%}  "
                f"배분={alloc:>13,.0f} 원   ({p['reason']})"
            )

    lines += ["", f"📈 results/forward/{name}/equity.png", ""]
    return "\n".join(lines)


def run_all_sessions(as_of: date | None = None) -> list[dict]:
    names = list_sessions()
    if not names:
        return []
    results: list[dict] = []
    for n in names:
        try:
            results.append(run_session(n, as_of=as_of))
        except Exception as e:  # noqa: BLE001
            log.error("session %s failed: %s", n, e)
            results.append({"name": n, "error": str(e)})
    return results


def get_universe_names() -> dict[str, str]:
    """Best-effort: load cached universe with Korean names."""
    from .data import UNIVERSE_FILE
    if not UNIVERSE_FILE.exists():
        return {}
    try:
        df = pd.read_parquet(UNIVERSE_FILE)
        return dict(zip(df["Code"].astype(str), df["Name"].astype(str)))
    except Exception:  # noqa: BLE001
        return {}


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
