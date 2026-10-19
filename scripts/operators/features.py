"""Parameter feature construction and normalization.

Maps the raw 4D operating vector mu = (V_app [mV], C0 [mol/m^3], q_w, R [nm]) to
the 7 physical features consumed by the operators, and provides a normalizer.
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn

from .constants import (
    A_LAMBDA_M2,
    FEATURE_NAMES,
    L_NM,
    V_T,
)


def raw_features(mu: torch.Tensor) -> torch.Tensor:
    """mu: (..., 4) -> features: (..., 7).

    Order: (psi_app, log10 C0, q_w, R_nm, alpha_r, beta, lambda_D/R).
    """
    v_mV = mu[..., 0]
    c0 = mu[..., 1]
    q_w = mu[..., 2]
    r_nm = mu[..., 3]

    psi_app = (v_mV * 1e-3) / V_T
    log_c0 = torch.log10(c0)
    alpha_r = (L_NM / r_nm) ** 2

    lam_d_nm = torch.sqrt(torch.as_tensor(A_LAMBDA_M2, dtype=mu.dtype) / c0) * 1e9
    beta = (L_NM / lam_d_nm) ** 2 / 2.0
    lam_d_over_r = lam_d_nm / r_nm

    return torch.stack(
        [psi_app, log_c0, q_w, r_nm, alpha_r, beta, lam_d_over_r], dim=-1
    )


def feature_stats(mu: np.ndarray) -> dict:
    """Compute per-feature mean/std from a raw (N,4) parameter array."""
    feats = raw_features(torch.as_tensor(np.asarray(mu), dtype=torch.float64)).numpy()
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    std[std == 0.0] = 1.0
    return {
        "names": FEATURE_NAMES,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


class ParamNormalizer(nn.Module):
    """Normalize mu -> features using fixed mean/std buffers (ONNX-friendly)."""

    def __init__(self, mean, std):
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32)
        self.register_buffer("feat_mean", mean)
        self.register_buffer("feat_std", std)

    def forward(self, mu: torch.Tensor):
        raw = raw_features(mu)
        norm = (raw - self.feat_mean.to(raw.dtype)) / self.feat_std.to(raw.dtype)
        return raw, norm

    @classmethod
    def from_stats(cls, stats):
        return cls(stats["mean"], stats["std"])

    @classmethod
    def from_file(cls, path):
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_stats(json.load(fh))


def save_feature_stats(stats: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
