from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta

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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
