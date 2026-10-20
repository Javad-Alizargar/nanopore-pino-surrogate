#!/usr/bin/env python3
"""Self-contained CPC test run: load one 4D state, execute the exported
DeepONet ONNX model, verify physical invariants and the stored reference
summary, and print ``CPC TEST RUN PASSED``.

    python scripts/run_cpc_test_case.py
    python scripts/run_cpc_test_case.py --write-expected
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

BASE = Path(__file__).resolve().parents[1]
TEST = BASE / "test_run"


def summarise(arr: np.ndarray) -> dict:
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "center": float(arr[0, arr.shape[1] // 2, arr.shape[2] // 2]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-expected", action="store_true")
    args = ap.parse_args()

    cfg = json.loads((TEST / "sample_input.json").read_text())
    mu = np.asarray(cfg["mu"], dtype=np.float32).reshape(1, 4)
    model = BASE / cfg["model"]
    tol = float(cfg.get("tolerance", 1e-5))

    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    arrays = dict(zip(names, sess.run(None, {"mu": mu})))
    required = {"psi", "c_plus", "c_minus", "rho", "s"}
    if not required.issubset(arrays):
        raise SystemExit(f"missing ONNX outputs: {sorted(required - set(arrays))}")

    stats = {k: summarise(arrays[k]) for k in sorted(required)}
    if args.write_expected:
        (TEST / "expected_output.json").write_text(
            json.dumps({"model": cfg["model"], "tolerance": tol,
                        "summary": stats}, indent=2, sort_keys=True) + "\n")
        print("wrote test_run/expected_output.json")
        return

    expected = json.loads((TEST / "expected_output.json").read_text())
    errors = []
    for field, ref in expected["summary"].items():
        for stat, value in ref.items():
            delta = abs(stats[field][stat] - value)
            if delta > tol:
                errors.append(f"{field}.{stat}: |{stats[field][stat]:.8g}-{value:.8g}|={delta:.3g}")

    cp, cm, rho, ssum = (arrays["c_plus"], arrays["c_minus"],
                         arrays["rho"], arrays["s"])
    if min(cp.min(), cm.min()) <= 0:
        errors.append("non-positive concentration")
    if np.max(np.abs(rho - (cp - cm))) > tol:
        errors.append("charge identity failed")
    if np.max(np.abs(ssum - (cp + cm))) > tol:
        errors.append("salt-sum identity failed")
    if max(abs(cp[0, :, 0] - 1).max(), abs(cm[0, :, 0] - 1).max(),
           abs(cp[0, :, -1] - 1).max(), abs(cm[0, :, -1] - 1).max()) > 2e-4:
        errors.append("reservoir concentration mismatch")

    if errors:
        for e in errors:
            print("FAIL:", e)
        raise SystemExit(1)
    print(f"CPC TEST RUN PASSED — {Path(cfg['model']).name}, mu={mu[0].tolist()}, "
          f"5 fields, max |delta| <= {tol:g}")
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
