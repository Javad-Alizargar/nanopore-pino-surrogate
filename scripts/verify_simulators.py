"""TASK 5.3c — Stage 5 interactive-simulator verification.

Produces ``results/stage5_simulator_verification.json`` covering:
  * web_simulator asset inventory (existence, bytes, SHA-256),
  * ONNX operating-range self-consistency (both operators reload in onnxruntime),
  * ONNX-vs-PyTorch numerical parity over the 48 interp_test states,
  * in-browser SG current formula vs the stored SG/FV dataset current,
  * Gradio app headless smoke test + live SG/FV solve timing/accuracy.

No paid calls; all local.
"""

import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from operators import build_model  # noqa: E402
import onnxruntime as ort  # noqa: E402

WEB = os.path.join(ROOT, "web_simulator")
MODELS = os.path.join(ROOT, "results", "models")
NPZ = os.path.join(ROOT, "data", "processed", "pnp_parametric_240.npz")
OUT = os.path.join(ROOT, "results", "stage5_simulator_verification.json")

NR, NZ = 21, 61


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory(paths):
    out = {}
    for p in paths:
        full = os.path.join(ROOT, p)
        out[p] = {"exists": os.path.isfile(full),
                  "bytes": os.path.getsize(full) if os.path.isfile(full) else 0,
                  "sha256": sha256(full) if os.path.isfile(full) else None}
    return out


# The shipped fno_reparam_seed42.onnx is the *300-epoch* re-export (TASK 5.1),
# so its ONNX-vs-PyTorch parity must be checked against the 300-epoch checkpoint.
CKPT_FOR_ONNX = {
    "deeponet": "deeponet_reparam_data_plus_physics_seed42.pt",
    "fno": "fno_reparam_seed42_300ep.pt",
}


def onnx_parity(kind, n_states):
    ckpt_path = os.path.join(MODELS, CKPT_FOR_ONNX[kind])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["kind"], ckpt["mode"], ckpt["stats"], **ckpt["size"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    onnx_path = os.path.join(WEB, "models", f"{kind}_reparam_seed42.onnx")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    d = np.load(NPZ, allow_pickle=True)
    interp = np.where(d["splits"] == "interp_test")[0]
    ood = np.where(d["splits"] == "ood_test")[0]
    idx = np.concatenate([interp, ood])[:n_states]
    mu = d["mu"][idx].astype(np.float32)

    with torch.no_grad():
        t = model(torch.as_tensor(mu))
    names = [o.name for o in sess.get_outputs()]
    o = sess.run(None, {sess.get_inputs()[0].name: mu})
    diffs = {}
    for n, arr in zip(names, o):
        tv = t[n].numpy()
        diffs[n] = float(np.max(np.abs(tv - arr)))
    return {"checkpoint": CKPT_FOR_ONNX[kind], "onnx_epochs": ckpt.get("epochs", "110"),
            "states_checked": int(len(idx)), "max_abs_diff": diffs,
            "max_abs_diff_overall": max(diffs.values()),
            "parity_ok": max(diffs.values()) < 1e-5}


def js_current_parity():
    d = np.load(NPZ, allow_pickle=True)
    r, z = d["r"], d["z"]
    dz = float(z[1] - z[0])
    edges = np.empty(r.size + 1)
    edges[0], edges[-1] = 0.0, 1.0
    edges[1:-1] = 0.5 * (r[:-1] + r[1:])
    w = 0.5 * (edges[1:] ** 2 - edges[:-1] ** 2)
    F, Dc, Da, L = 96485.33212, 1.96e-9, 2.03e-9, 30.0

    def B(x):
        xc = np.clip(x, -80, 80)
        small = np.abs(xc) < 1e-8
        ser = 1 - xc / 2 + xc ** 2 / 12
        xs = np.where(small, 1.0, xc)
        return np.where(small, ser, xs / np.expm1(xs))

    def cur(cp, cm, psi, mu):
        cpb, cmb, psib = (a.reshape(NR, NZ) for a in (cp, cm, psi))
        dpsi = psib[:, 1:] - psib[:, :-1]
        jp = -(cpb[:, 1:] * B(-dpsi) - cpb[:, :-1] * B(dpsi)) / dz
        jm = -(cmb[:, 1:] * B(dpsi) - cmb[:, :-1] * B(-dpsi)) / dz
        dimless = 2 * np.pi * np.sum((Dc * jp - Da * jm) * w[:, None], axis=0)
        return dimless * (F * mu[1] * (mu[3] * 1e-9) ** 2 / (L * 1e-9)) * 1e12

    # benchmark preset states (as shipped in model_meta.json)
    meta = json.load(open(os.path.join(WEB, "data", "model_meta.json")))
    rows = []
    for st in meta["benchmark_states"]:
        i = st["id"]
        I = cur(d["c_plus"][i], d["c_minus"][i], d["psi"][i], d["mu"][i])
        ref = float(d["mean_current_pA"][i])
        rel = abs(I.mean() - ref) / (abs(ref) if abs(ref) > 1e-9 else 1.0)
        rows.append({"id": i, "js_mean_pA": float(I.mean()),
                     "sgfv_mean_pA": ref, "rel_diff": float(rel)})
    return {"max_rel_diff": max(r["rel_diff"] for r in rows), "per_state": rows}


def gradio_smoke():
    import app_gradio

    t0 = time.perf_counter()
    op = app_gradio.predict("deeponet", 50.0, 100.0, -0.01, 5.0)
    op_ms = op["ms"]
    ref = app_gradio.run_reference(50.0, 100.0, -0.01, 5.0)
    Iref = app_gradio.sg_current_pA(ref["c_plus"], ref["c_minus"], ref["psi"], ref["mu"])
    fn = app_gradio.predict("fno", 50.0, 100.0, -0.01, 5.0)
    total = time.perf_counter() - t0
    return {
        "deeponet_mean_I_pA": float(op["i_z"].mean()),
        "deeponet_infer_ms": float(op_ms),
        "fno_mean_I_pA": float(fn["i_z"].mean()),
        "fno_infer_ms": float(fn["ms"]),
        "live_reference_wall_s": float(ref["wall_s"]),
        "live_reference_method": ref["method"],
        "live_reference_iterations": ref["iterations"],
        "live_reference_residual": float(ref["residual"]),
        "deeponet_vs_live_psi_rel_l2": app_gradio.rel_l2(op["psi"], ref["psi"]),
        "deeponet_vs_live_rho_rel_l2": app_gradio.rel_l2(op["rho"], ref["rho"]),
        "deeponet_vs_live_I_rel_l2": app_gradio.rel_l2(op["i_z"], Iref),
        "sgfv_mean_I_pA": float(Iref.mean()),
        "headless_total_s": float(total),
        "source": "scripts/app_gradio.py (functions imported; same code path as UI)",
    }


def main():
    web_files = [
        "web_simulator/index.html", "web_simulator/style.css", "web_simulator/app.js",
        "web_simulator/data/model_meta.json",
        "web_simulator/models/deeponet_reparam_seed42.onnx",
        "web_simulator/models/fno_reparam_seed42.onnx",
    ]
    report = {
        "stage": 5,
        "task": "5.3",
        "generated_local_only": True,
        "web_assets": inventory(web_files),
        "onnx_reload_ok": {},
        "onnx_parity_vs_pytorch": {},
        "js_current_formula": js_current_parity(),
        "gradio_app": gradio_smoke(),
    }
    for kind in ("deeponet", "fno"):
        sess = ort.InferenceSession(os.path.join(WEB, "models", f"{kind}_reparam_seed42.onnx"),
                                    providers=["CPUExecutionProvider"])
        report["onnx_reload_ok"][kind] = [o.name for o in sess.get_outputs()]
        report["onnx_parity_vs_pytorch"][kind] = onnx_parity(kind, 48)

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("wrote", OUT)
    print(json.dumps({
        "js_max_rel_diff": report["js_current_formula"]["max_rel_diff"],
        "deeponet_parity": report["onnx_parity_vs_pytorch"]["deeponet"]["max_abs_diff_overall"],
        "fno_parity": report["onnx_parity_vs_pytorch"]["fno"]["max_abs_diff_overall"],
        "gradio_I_rel_l2": report["gradio_app"]["deeponet_vs_live_I_rel_l2"],
        "gradio_ref_s": report["gradio_app"]["live_reference_wall_s"],
    }, indent=2))


if __name__ == "__main__":
    main()
