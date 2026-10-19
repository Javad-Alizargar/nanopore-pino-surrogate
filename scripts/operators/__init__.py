"""Exact-boundary parameter-conditioned neural operators for 2D axisymmetric PNP.

Package contents:
    boundary_ansatz.py   exact reservoir Dirichlet + positivity layer
    fno2d.py             parameter-conditioned Exact-Boundary 2D FNO
    deeponet.py          parameter-conditioned Exact-Boundary PI-DeepONet
    losses_and_metrics.py PNP residuals + exact SG current evaluator
    features.py          parameter feature map + normalization
    constants.py         physical constants + grid helpers
"""

from .boundary_ansatz import BoundaryAnsatz
from .deeponet import DeepONet
from .fno2d import FNO2D
from .features import ParamNormalizer, feature_stats, raw_features

__all__ = [
    "BoundaryAnsatz",
    "FNO2D",
    "DeepONet",
    "ParamNormalizer",
    "feature_stats",
    "raw_features",
    "build_model",
]


def build_model(kind: str, mode: str = "reparam", stats=None, **kwargs):
    """Factory for the two operator families.

    kind: "fno" or "deeponet"; mode: "reparam" or "direct_c".
    stats: dict from features.feature_stats (mean/std), optional.
    """
    normalizer = ParamNormalizer.from_stats(stats) if stats is not None else None
    kind = kind.lower()
    if kind in ("fno", "fno2d"):
        return FNO2D(mode=mode, normalizer=normalizer, **kwargs)
    if kind in ("deeponet", "pi-deeponet"):
        return DeepONet(mode=mode, normalizer=normalizer, **kwargs)
    raise ValueError(f"unknown model kind {kind!r}")
