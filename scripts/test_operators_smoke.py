"""TASK 3.3 — operator unit tests, tiny smoke training, and ONNX export check.

Runs (deterministic, CPU/M1):
  1. exact reservoir boundary + strict positivity across random inputs, for both
     FNO-2D and DeepONet in both ``reparam`` and ``direct_c`` modes (f32 & f64);
  2. a 15-epoch smoke training on 16 states (loss decrease, finite gradients,
     checkpoint save/load round-trip, single-state inference latency);
  3. ONNX export of both models and output-equivalence vs onnxruntime (<1e-5),
     using the real-valued DFT-matrix FNO (no Complex64 ops).

Writes results/stage3_operator_smoke.json.
"""

import json
import os
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from operators import build_model, feature_stats  # noqa: E402
from operators.features import raw_features  # noqa: E402
from operators.losses_and_metrics import (  # noqa: E402
    GridTensors,
    sg_current_pA,
    pnp_physics_loss,
)

DATA = os.path.join(PROJECT_ROOT, "data", "processed", "pnp_parametric_240.npz")
OUT = os.path.join(PROJECT_ROOT, "results", "stage3_operator_smoke.json")
SEED = 20261018

MODES = ("reparam", "direct_c")
KINDS = ("fno", "deeponet")


class OnnxWrapper(nn.Module):
    """Return a tuple of tensors (ONNX prefers tuple outputs)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, mu):
        out = self.model(mu)
        return out["psi"], out["c_plus"], out["c_minus"], out["rho"], out["s"]


def _random_mu(n, dtype=torch.float64, rng=None):
    rng = rng or np.random.default_rng(SEED)
    v = rng.uniform(-80.0, 80.0, n)
    c0 = 10.0 ** rng.uniform(np.log10(5.0), np.log10(300.0), n)
    q = rng.uniform(-0.12, 0.0, n)
    r = rng.uniform(2.0, 10.0, n)
    return torch.tensor(np.stack([v, c0, q, r], axis=1), dtype=dtype)


def test_boundary_and_positivity(stats):
    results = {}
    mu32 = _random_mu(8, torch.float32)
    mu64 = _random_mu(8, torch.float64)
    for kind in KINDS:
        for mode in MODES:
            key = f"{kind}_{mode}"
            record = {}
            for tag, mu in (("float32", mu32), ("float64", mu64)):
                model = build_model(kind, mode, stats).to(mu.dtype).eval()
                with torch.no_grad():
                    out = model(mu)
                psi_app = raw_features(mu)[:, 0]
                e_psi0 = float((out["psi"][:, :, 0] - psi_app[:, None]).abs().max())
                e_psi1 = float(out["psi"][:, :, -1].abs().max())
                e_c0 = float((out["c_plus"][:, :, 0] - 1.0).abs().max())
                e_cm0 = float((out["c_minus"][:, :, 0] - 1.0).abs().max())
                e_rho0 = float(out["rho"][:, :, 0].abs().max())
                min_c = float(min(out["c_plus"].min(), out["c_minus"].min()))
                record[tag] = {
                    "max_abs_psi_z0_minus_psi_app": e_psi0,
                    "max_abs_psi_z1": e_psi1,
                    "max_abs_c_plus_z0_minus_1": e_c0,
                    "max_abs_c_minus_z0_minus_1": e_cm0,
                    "max_abs_rho_z0": e_rho0,
                    "min_concentration": min_c,
                }
                assert e_psi0 < 1e-15, (key, tag, e_psi0)
                assert e_psi1 < 1e-15, (key, tag, e_psi1)
                assert e_c0 < 1e-15 and e_cm0 < 1e-15, (key, tag, e_c0, e_cm0)
                assert e_rho0 < 1e-15, (key, tag, e_rho0)
                assert min_c > 0.0, (key, tag, min_c)
            results[key] = record
    return results


def _load_train(data, n=16):
    mu = data["mu"][data["splits"] == "train"][:n]
    keys = ["psi", "c_plus", "c_minus", "rho"]
    target = {
        k: torch.tensor(data[k][data["splits"] == "train"][:n], dtype=torch.float32)
        for k in keys
    }
    scales = {k: float(target[k].std().clamp_min(1e-3)) for k in keys}
    return mu, target, scales


def test_smoke_training(data, stats):
    torch.manual_seed(SEED)
    mu_np, target, scales = _load_train(data, 16)
    mu = torch.tensor(mu_np, dtype=torch.float32)
    grid = GridTensors(dtype=torch.float32)

    results = {}
    for kind in KINDS:
        for mode in MODES:
            key = f"{kind}_{mode}"
            model = build_model(kind, mode, stats)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            history = []
            grads_finite = True
            for epoch in range(15):
                opt.zero_grad()
                out = model(mu)
                loss_data = sum(
                    (((out[k] - target[k]) / scales[k]) ** 2).mean()
                    for k in ("psi", "c_plus", "c_minus")
                ) / 3.0
                loss_rho = (((out["rho"] - target["rho"]) / scales["rho"]) ** 2).mean()
                phys = pnp_physics_loss(out, mu, grid)
                loss = loss_data + 0.5 * loss_rho + 1e-4 * sum(v for v in phys.values())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        grads_finite = False
                opt.step()
                history.append(float(loss.detach()))

            # checkpoint round-trip
            with tempfile.TemporaryDirectory() as tmp:
                ckpt = os.path.join(tmp, f"{key}.pt")
                torch.save(model.state_dict(), ckpt)
                model2 = build_model(kind, mode, stats)
                model2.load_state_dict(torch.load(ckpt, weights_only=True))
                model.eval()
                model2.eval()
                with torch.no_grad():
                    o1 = model(mu[:1])["psi"]
                    o2 = model2(mu[:1])["psi"]
                ckpt_max_diff = float((o1 - o2).abs().max())

            results[key] = {
                "loss_first": history[0],
                "loss_last": history[-1],
                "loss_decreased": bool(history[-1] < history[0]),
                "gradients_finite": bool(grads_finite),
                "checkpoint_reload_max_diff": ckpt_max_diff,
                "n_params": int(sum(p.numel() for p in model.parameters())),
            }
            assert history[-1] < history[0], (key, history[0], history[-1])
            assert grads_finite, key
            assert ckpt_max_diff < 1e-6, (key, ckpt_max_diff)

    # inference latency (single state)
    latency = {}
    for kind in KINDS:
        for mode in MODES:
            for tag, dt in (("float32", torch.float32), ("float64", torch.float64)):
                model = build_model(kind, mode, stats).to(dt).eval()
                mu1 = _random_mu(1, dt)
                with torch.no_grad():
                    for _ in range(5):
                        model(mu1)
                    t0 = time.perf_counter()
                    for _ in range(30):
                        model(mu1)
                    ms = (time.perf_counter() - t0) / 30 * 1e3
                latency[f"{kind}_{mode}_{tag}"] = ms
    return results, latency


def test_onnx_export(stats, tol=1e-5):
    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover
        return {"skipped": f"onnxruntime unavailable: {exc}"}

    results = {}
    mu = _random_mu(4, torch.float32)
    with tempfile.TemporaryDirectory() as tmp:
        for kind in KINDS:
            for mode in MODES:
                key = f"{kind}_{mode}"
                model = build_model(kind, mode, stats).eval()
                wrapper = OnnxWrapper(model).eval()
                path = os.path.join(tmp, f"{key}.onnx")
                export_kwargs = dict(
                    input_names=["mu"],
                    output_names=["psi", "c_plus", "c_minus", "rho", "s"],
                    dynamic_axes={"mu": {0: "batch"}},
                    opset_version=17,
                )
                try:
                    torch.onnx.export(wrapper, (mu,), path, dynamo=False, **export_kwargs)
                    exporter = "legacy"
                except TypeError:
                    torch.onnx.export(wrapper, (mu,), path, **export_kwargs)
                    exporter = "default"

                sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                ort_out = sess.run(None, {"mu": mu.numpy()})
                with torch.no_grad():
                    ref = wrapper(mu)
                names = ["psi", "c_plus", "c_minus", "rho", "s"]
                diffs = {
                    names[i]: float(np.max(np.abs(ort_out[i] - ref[i].numpy())))
                    for i in range(len(names))
                }
                max_diff = max(diffs.values())
                results[key] = {
                    "exported": True,
                    "exporter": exporter,
                    "size_bytes": os.path.getsize(path),
                    "max_abs_diff": max_diff,
                    "per_output_max_abs_diff": diffs,
                    "matches_within_tol": bool(max_diff < tol),
                }
                assert max_diff < tol, (key, max_diff)
    return results


def main():
    t0 = time.perf_counter()
    data = np.load(DATA)
    stats = feature_stats(data["mu"][data["splits"] == "train"])

    grid = GridTensors(dtype=torch.float64)
    bench_mu = torch.tensor([[50.0, 100.0, -0.01, 5.0]], dtype=torch.float64)
    psi_ref = torch.tensor(data["psi"][144:145], dtype=torch.float64)
    cp_ref = torch.tensor(data["c_plus"][144:145], dtype=torch.float64)
    cm_ref = torch.tensor(data["c_minus"][144:145], dtype=torch.float64)
    iz = sg_current_pA(cp_ref, cm_ref, psi_ref, bench_mu, grid).numpy()[0]
    sg_current_check = {
        "mean_pA": float(iz.mean()),
        "stored_mean_pA": float(data["mean_current_pA"][144]),
        "max_abs_diff_vs_stored_fp32": float(np.max(np.abs(iz - data["i_z"][144]))),
    }

    boundary = test_boundary_and_positivity(stats)
    training, latency = test_smoke_training(data, stats)
    onnx = test_onnx_export(stats)

    result = {
        "stage": 3,
        "task": "3.3_operator_smoke_verification",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "torch_version": torch.__version__,
        "sg_current_evaluator_check": sg_current_check,
        "boundary_and_positivity": boundary,
        "smoke_training": training,
        "inference_latency_ms": latency,
        "onnx_export": onnx,
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "all_passed": True,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps({
        "sg_current_mean_pA": sg_current_check["mean_pA"],
        "training": {k: {"loss_first": round(v["loss_first"], 4),
                         "loss_last": round(v["loss_last"], 6),
                         "decreased": v["loss_decreased"]} for k, v in training.items()},
        "latency_ms": {k: round(v, 3) for k, v in latency.items()},
        "onnx_max_diff": {k: v.get("max_abs_diff") for k, v in onnx.items()},
        "wall_seconds": result["wall_seconds"],
    }, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
