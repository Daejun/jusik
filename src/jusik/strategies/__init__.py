from .base import Pick, Strategy
from .gap_down import GapDownStrategy
from .momentum import MomentumStrategy
from .volume_spike import VolumeSpikeStrategy

REGISTRY: dict[str, type[Strategy]] = {
    "momentum": MomentumStrategy,
    "gap_down": GapDownStrategy,
    "volume_spike": VolumeSpikeStrategy,
}


def build(name: str, **params) -> Strategy:
    if name not in REGISTRY:
        raise KeyError(f"unknown strategy: {name}. available: {list(REGISTRY)}")
    return REGISTRY[name](**params)
