from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta

from .compare import compare_strategies
from .config import SimConfig, TradeCost
from .engine import run_backtest
from .forward import (
    init_session as forward_init,
    list_sessions as forward_list,
    run_session as forward_run,
    status as forward_status,
)
from .reporter import print_summary, write_report
from .strategies import REGISTRY, build


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _cmd_backtest(args: argparse.Namespace) -> int:
    cfg = SimConfig(
        budget=args.budget,
        top_n=args.top_n,
        universe_size=args.universe,
        cost=TradeCost(),
    )
    params: dict = {"top_n": args.top_n}
    if args.params:
        params.update(_parse_params(args.params))
    strategy = build(args.strategy, **params)
    result = run_backtest(
        strategy=strategy,
        start=args.start,
        end=args.end,
        config=cfg,
        budget_mode=args.budget_mode,
    )
    print_summary(result)
    out = write_report(result)
    print(f"\nartifacts: {out}")
    return 0


def _cmd_grid(args: argparse.Namespace) -> int:
    from .grid import DEFAULT_GRIDS, run_grid
    cfg = SimConfig(budget=args.budget, universe_size=args.universe)
    grids = DEFAULT_GRIDS
    if args.strategies:
        grids = {k: v for k, v in DEFAULT_GRIDS.items() if k in args.strategies}
    out = run_grid(
        start=args.start, end=args.end, config=cfg,
        budget_mode=args.budget_mode, grids=grids,
        extra_codes=["069500"] if "benchmark" in grids else None,
        trade_modes=args.trade_modes,
    )
    import pandas as pd
    lb = pd.read_csv(out / "leaderboard.csv")
    print("\n=== TOP 15 by Sharpe ===")
    cols = ["label", "total_return", "win_rate", "sharpe_approx",
            "avg_daily_return", "max_daily_loss"]
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}",
                           "display.max_colwidth", 60, "display.width", 200):
        print(lb.head(15)[cols].to_string(index=False))
    print(f"\nartifacts: {out}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    cfg = SimConfig(budget=args.budget, top_n=args.top_n, universe_size=args.universe)
    specs = [{"name": s, "params": {"top_n": args.top_n}} for s in args.strategies]
    out = compare_strategies(
        strategy_specs=specs,
        start=args.start,
        end=args.end,
        config=cfg,
        budget_mode=args.budget_mode,
    )
    import pandas as pd
    lb = pd.read_csv(out / "leaderboard.csv")
    print("\n=== leaderboard ===")
    cols = ["label", "total_return", "win_rate", "sharpe_approx", "max_daily_loss"]
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(lb[cols].to_string(index=False))
    print(f"\nartifacts: {out}")
    return 0


def _parse_params(items: list[str]) -> dict:
    out: dict = {}
    for kv in items:
        k, _, v = kv.partition("=")
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
        else:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


def _cmd_forward_init(args: argparse.Namespace) -> int:
    params = _parse_params(args.params)
    cfg = forward_init(
        name=args.name,
        strategy_name=args.strategy,
        strategy_params=params,
        start_date=args.start,
        trade_mode=args.trade_mode,
        budget_mode=args.budget_mode,
        initial_budget=args.budget,
        universe_size=args.universe,
    )
    print(f"created forward session '{cfg.name}' (start={cfg.start_date}, "
          f"strategy={cfg.strategy_name} {params}, mode={cfg.trade_mode}/{cfg.budget_mode})")
    print("next: jusik forward run", args.name)
    return 0


def _cmd_forward_run(args: argparse.Namespace) -> int:
    summary = forward_run(args.name, as_of=args.as_of)
    s = summary["summary"]
    print(f"\n=== {args.name} as of {summary['as_of']} ===")
    print(f"strategy : {summary['strategy']}  {summary['params']}")
    print(f"mode     : {summary['trade_mode']} / {summary['budget_mode']}")
    print(f"initial  : {summary['initial_budget']:,.0f}")
    if s:
        print(f"total pnl: {s['total_pnl']:>14,.0f}  ({s['total_return']:+.2%})")
        print(f"win rate : {s['win_rate']:.1%}  ({s['win_days']}/{s['trading_days']})")
        print(f"sharpe~  : {s['sharpe_approx']:.2f}")
    nxt = summary.get("next_picks", {})
    picks = nxt.get("picks", [])
    print(f"\n=== next picks (as_of {nxt.get('as_of')}) ===")
    print(f"action: {nxt.get('next_action')}")
    if not picks:
        print("(none — strategy did not produce candidates)")
    else:
        for p in picks:
            print(f"  - {p['code']:>8}  w={p['weight']:.3f}  {p['reason']}")
    return 0


def _cmd_forward_status(args: argparse.Namespace) -> int:
    info = forward_status(args.name)
    import json as _json
    print(_json.dumps(info, indent=2, ensure_ascii=False, default=str))
    return 0


def _cmd_forward_list(_: argparse.Namespace) -> int:
    names = forward_list()
    if not names:
        print("(no forward sessions yet)")
        return 0
    for n in names:
        print(f"  - {n}")
    return 0


def _cmd_list(_: argparse.Namespace) -> int:
    print("strategies:")
    for name, cls in REGISTRY.items():
        print(f"  - {name}: {cls.__doc__.splitlines()[0] if cls.__doc__ else ''}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="jusik", description="KOSPI200 open-buy/close-sell simulator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    bt = sub.add_parser("backtest", help="run historical backtest")
    bt.add_argument("--strategy", default="momentum", choices=list(REGISTRY))
    today = date.today()
    bt.add_argument("--start", type=_parse_date, default=today - timedelta(days=365))
    bt.add_argument("--end", type=_parse_date, default=today - timedelta(days=1))
    bt.add_argument("--budget", type=float, default=10_000_000)
    bt.add_argument("--top-n", type=int, default=5)
    bt.add_argument("--universe", type=int, default=200)
    bt.add_argument("--budget-mode", choices=["fixed", "compound"], default="fixed")
    bt.add_argument("--params", nargs="*", help="strategy params: key=value ...")
    bt.set_defaults(func=_cmd_backtest)

    ls = sub.add_parser("list", help="list strategies")
    ls.set_defaults(func=_cmd_list)

    fw = sub.add_parser("forward", help="forward (live-style) session management")
    fwsub = fw.add_subparsers(dest="action", required=True)

    fi = fwsub.add_parser("init", help="create a new forward session")
    fi.add_argument("name")
    fi.add_argument("--strategy", required=True, choices=list(REGISTRY))
    fi.add_argument("--params", nargs="*", default=[],
                    help="strategy params: key=value ...")
    fi.add_argument("--trade-mode", default="multiday5",
                    choices=["intraday", "overnight", "multiday5"])
    fi.add_argument("--budget", type=float, default=10_000_000)
    fi.add_argument("--budget-mode", default="compound", choices=["fixed", "compound"])
    fi.add_argument("--start", type=_parse_date, default=None)
    fi.add_argument("--universe", type=int, default=200)
    fi.set_defaults(func=_cmd_forward_init)

    fr = fwsub.add_parser("run", help="run / refresh a forward session")
    fr.add_argument("name")
    fr.add_argument("--as-of", type=_parse_date, default=None,
                    help="treat this date as 'yesterday' (debug only)")
    fr.set_defaults(func=_cmd_forward_run)

    fs = fwsub.add_parser("status", help="show forward session status")
    fs.add_argument("name")
    fs.set_defaults(func=_cmd_forward_status)

    fl = fwsub.add_parser("list", help="list forward sessions")
    fl.set_defaults(func=_cmd_forward_list)

    gr = sub.add_parser("grid", help="parameter grid search across strategies")
    gr.add_argument("--start", type=_parse_date, default=today - timedelta(days=730))
    gr.add_argument("--end", type=_parse_date, default=today - timedelta(days=1))
    gr.add_argument("--budget", type=float, default=10_000_000)
    gr.add_argument("--universe", type=int, default=200)
    gr.add_argument("--budget-mode", choices=["fixed", "compound"], default="fixed")
    gr.add_argument("--strategies", nargs="+", default=None,
                    help="subset of strategies (default: all in grid)")
    gr.add_argument("--trade-modes", nargs="+", default=["intraday"],
                    choices=["intraday", "overnight", "multiday5"],
                    help="trade modes to compare (default: intraday)")
    gr.set_defaults(func=_cmd_grid)

    cmp = sub.add_parser("compare", help="run all strategies and compare")
    cmp.add_argument("--strategies", nargs="+", default=list(REGISTRY))
    cmp.add_argument("--start", type=_parse_date, default=today - timedelta(days=365))
    cmp.add_argument("--end", type=_parse_date, default=today - timedelta(days=1))
    cmp.add_argument("--budget", type=float, default=10_000_000)
    cmp.add_argument("--top-n", type=int, default=5)
    cmp.add_argument("--universe", type=int, default=200)
    cmp.add_argument("--budget-mode", choices=["fixed", "compound"], default="fixed")
    cmp.set_defaults(func=_cmd_compare)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
