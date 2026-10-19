import numpy as np


def relative_l2(pred: np.ndarray, truth: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(truth.ravel())), 1e-12)
    return float(np.linalg.norm((pred - truth).ravel()) / denom)


def max_abs(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.max(np.abs(pred - truth)))
