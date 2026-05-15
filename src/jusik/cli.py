from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta

from .compare import compare_strategies
from .config import SimConfig, TradeCost
from .engine import run_backtest
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
        for kv in args.params:
            k, _, v = kv.partition("=")
            try:
                params[k] = float(v) if "." in v else int(v)
            except ValueError:
                params[k] = v
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

    gr = sub.add_parser("grid", help="parameter grid search across strategies")
    gr.add_argument("--start", type=_parse_date, default=today - timedelta(days=730))
    gr.add_argument("--end", type=_parse_date, default=today - timedelta(days=1))
    gr.add_argument("--budget", type=float, default=10_000_000)
    gr.add_argument("--universe", type=int, default=200)
    gr.add_argument("--budget-mode", choices=["fixed", "compound"], default="fixed")
    gr.add_argument("--strategies", nargs="+", default=None,
                    help="subset of strategies (default: all in grid)")
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
