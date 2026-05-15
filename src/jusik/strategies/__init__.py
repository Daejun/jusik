from .base import Pick, Strategy
from .momentum import MomentumStrategy

REGISTRY: dict[str, type[Strategy]] = {
    "momentum": MomentumStrategy,
}


def build(name: str, **params) -> Strategy:
    if name not in REGISTRY:
        raise KeyError(f"unknown strategy: {name}. available: {list(REGISTRY)}")
    return REGISTRY[name](**params)
