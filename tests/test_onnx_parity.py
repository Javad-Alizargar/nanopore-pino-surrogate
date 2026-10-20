#!/usr/bin/env python3
"""Minimal ONNX load/forward smoke test."""
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    ROOT / "results/models/deeponet_reparam_seed42.onnx",
    ROOT / "results/models/fno_reparam_seed42.onnx",
]


def main() -> None:
    for path in MODELS:
        assert path.exists(), f"missing {path}"
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        shape = sess.get_inputs()[0].shape
        dims = [d if isinstance(d, int) else 1 for d in shape]
        x = np.zeros(dims, dtype=np.float32)
        out = sess.run(None, {sess.get_inputs()[0].name: x})
        print(path.name, "ok", [np.asarray(o).shape for o in out])


if __name__ == "__main__":
    main()
