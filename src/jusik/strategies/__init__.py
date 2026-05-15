from .base import Pick, Strategy
from .benchmark import BenchmarkStrategy
from .gap_down import GapDownStrategy
from .low_vol import LowVolStrategy
from .momentum import MomentumStrategy
from .rsi import RsiReversalStrategy
from .sma_cross import SmaCrossStrategy
from .volume_spike import VolumeSpikeStrategy

REGISTRY: dict[str, type[Strategy]] = {
    "momentum": MomentumStrategy,
    "gap_down": GapDownStrategy,
    "volume_spike": VolumeSpikeStrategy,
    "sma_cross": SmaCrossStrategy,
    "rsi": RsiReversalStrategy,
    "low_vol": LowVolStrategy,
    "benchmark": BenchmarkStrategy,
}


def build(name: str, **params) -> Strategy:
    if name not in REGISTRY:
        raise KeyError(f"unknown strategy: {name}. available: {list(REGISTRY)}")
    return REGISTRY[name](**params)
