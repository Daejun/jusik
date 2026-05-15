from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "cache"
RESULTS_DIR = REPO_ROOT / "results"


@dataclass
class TradeCost:
    commission_bps: float = 1.5
    tax_bps_sell: float = 18.0
    slippage_bps: float = 5.0

    def buy_cost_rate(self) -> float:
        return (self.commission_bps + self.slippage_bps) / 10_000

    def sell_cost_rate(self) -> float:
        return (self.commission_bps + self.tax_bps_sell + self.slippage_bps) / 10_000


@dataclass
class SimConfig:
    budget: float = 10_000_000.0
    top_n: int = 5
    universe_size: int = 200
    cost: TradeCost = field(default_factory=TradeCost)
    min_price: float = 1_000.0
    min_avg_volume: float = 100_000.0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d
