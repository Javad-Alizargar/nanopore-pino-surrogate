"""TASK 5.3b — local interactive Gradio simulator.

Loads the exported ONNX operators (client-side-style inference, no server model
calls) and offers an optional **live SG/FV solver comparison**: the verified
float64 finite-volume Scharfetter-Gummel reference is solved at the currently
selected parameters and compared field-by-field against the surrogate.

Run:
    .venv/bin/python scripts/app_gradio.py            # launch web UI
    .venv/bin/python scripts/app_gradio.py --smoke    # headless self-test
"""

import argparse
import json
import os
import sys
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import onnxruntime as ort  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from generate_parametric_dataset import solve_state  # noqa: E402
from operators.constants import (  # noqa: E402
    A_LAMBDA_M2,
    DIFF_ANION,
    DIFF_CATION,
    EPSILON_0,
    FARADAY,
    L_NM,
    NR,
    NZ,
    REL_PERMITTIVITY,
    V_T,
    annular_weights,
    make_grid,
)

WEB = os.path.join(PROJECT_ROOT, "web_simulator")
META_PATH = os.path.join(WEB, "data", "model_meta.json")
MODONS = {
    "deeponet": os.path.join(WEB, "models", "deeponet_reparam_seed42.onnx"),
    "fno": os.path.join(WEB, "models", "fno_reparam_seed42.onnx"),
}
MODEL_LABEL = {
    "deeponet": "PI-DeepONet (reparam) — primary",
    "fno": "Exact-Boundary FNO-2D (reparam) — benchmark",
}

with open(META_PATH, "r", encoding="utf-8") as fh:
    META = json.load(fh)

_R, _Z, *_ = make_grid(NR, NZ)
_WEIGHTS = annular_weights(_R)
_DZ = float(_Z[1] - _Z[0])

_SESSIONS = {}


def _session(key):
    if key not in _SESSIONS:
        _SESSIONS[key] = ort.InferenceSession(MODONS[key], providers=["CPUExecutionProvider"])
    return _SESSIONS[key]


# --------------------------------------------------------------------------- #
# Physics helpers (mirror the in-browser evaluator)
# --------------------------------------------------------------------------- #
def _bernoulli(x):
    xc = np.clip(x, -80.0, 80.0)
    small = np.abs(xc) < 1e-8
    series = 1.0 - xc / 2.0 + xc ** 2 / 12.0
    xs = np.where(small, 1.0, xc)
    exact = xs / np.expm1(xs)
    return np.where(small, series, exact)


def sg_current_pA(cp, cm, psi, mu):
    cpb = np.asarray(cp, dtype=float).reshape(NR, NZ)
    cmb = np.asarray(cm, dtype=float).reshape(NR, NZ)
    psib = np.asarray(psi, dtype=float).reshape(NR, NZ)
    dpsi = psib[:, 1:] - psib[:, :-1]
    jp = -(cpb[:, 1:] * _bernoulli(-dpsi) - cpb[:, :-1] * _bernoulli(dpsi)) / _DZ
    jm = -(cmb[:, 1:] * _bernoulli(dpsi) - cmb[:, :-1] * _bernoulli(-dpsi)) / _DZ
    dimless = 2.0 * np.pi * np.sum((DIFF_CATION * jp - DIFF_ANION * jm) * _WEIGHTS[:, None], axis=0)
    c0, r_m = float(mu[1]), float(mu[3]) * 1e-9
    scale = FARADAY * c0 * r_m ** 2 / (L_NM * 1e-9)
    return dimless * scale * 1e12


def lambda_d_nm(c0):
    return float(np.sqrt(A_LAMBDA_M2 / c0) * 1e9)


def sigma_mC(mu):
    return float(mu[2]) * REL_PERMITTIVITY * EPSILON_0 * V_T / (float(mu[3]) * 1e-9) * 1000.0


def ood_flags(mu):
    R = META["ranges_in_distribution"]
    f = []
    if not (R["V_app_mV"][0] <= mu[0] <= R["V_app_mV"][1]):
        f.append("V_app")
    if not (R["C0"][0] <= mu[1] <= R["C0"][1]):
        f.append("C0")
    if not (R["q_w"][0] <= mu[2] <= R["q_w"][1]):
        f.append("q_w")
    if not (R["R_nm"][0] <= mu[3] <= R["R_nm"][1]):
        f.append("R")
    return f


# --------------------------------------------------------------------------- #
# Inference / reference
# --------------------------------------------------------------------------- #
def predict(model_key, V_mV, C0, q_w, R_nm):
    mu = np.array([[V_mV, C0, q_w, R_nm]], dtype=np.float32)
    sess = _session(model_key)
    t0 = time.perf_counter()
    out = sess.run(None, {sess.get_inputs()[0].name: mu})
    ms = (time.perf_counter() - t0) * 1e3
    names = [o.name for o in sess.get_outputs()]
    res = {n: a.reshape(NR, NZ).astype(np.float64) for n, a in zip(names, out)}
    res["mu"] = mu[0].astype(float)
    res["ms"] = ms
    res["i_z"] = sg_current_pA(res["c_plus"], res["c_minus"], res["psi"], res["mu"])
    return res


def run_reference(V_mV, C0, q_w, R_nm):
    t0 = time.perf_counter()
    sol, cfg, grid, method, n_cont = solve_state(V_mV, C0, q_w, R_nm)
    wall = time.perf_counter() - t0
    return {
        "psi": np.asarray(sol.psi, dtype=float),
        "c_plus": np.asarray(sol.c_plus, dtype=float),
        "c_minus": np.asarray(sol.c_minus, dtype=float),
        "rho": np.asarray(sol.c_plus, dtype=float) - np.asarray(sol.c_minus, dtype=float),
        "mu": np.array([V_mV, C0, q_w, R_nm], dtype=float),
        "method": method,
        "n_cont": int(n_cont),
        "residual": float(sol.residual),
        "iterations": int(sol.iterations),
        "wall_s": wall,
    }


def rel_l2(a, b):
    b = np.asarray(b, dtype=float)
    d = np.linalg.norm(b)
    return float(np.linalg.norm(np.asarray(a, dtype=float) - b) / (d if d > 0 else 1.0))


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _imshow_row(titles, fields, cmap="viridis", diverging=False):
    fig, axes = plt.subplots(1, len(fields), figsize=(3.0 * len(fields), 4.6), dpi=130)
    if len(fields) == 1:
        axes = [axes]
    for ax, title, field in zip(axes, titles, fields):
        arr = np.asarray(field, dtype=float).reshape(NR, NZ)
        if diverging:
            v = np.max(np.abs(arr)) or 1.0
            im = ax.imshow(arr, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
        else:
            im = ax.imshow(arr, origin="lower", aspect="auto", cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("z index", fontsize=8)
        ax.set_ylabel("r index", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _current_fig(pred, ref=None):
    fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=130)
    z = _Z[:-1] + _DZ / 2.0
    ax.plot(z, pred, "-", color="#3fb6d0", lw=2, label="operator")
    if ref is not None:
        ax.plot(z, ref, "--", color="#e69f00", lw=2, label="SG/FV reference")
    ax.set_xlabel("z/L")
    ax.set_ylabel("I(z) [pA]")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _fig_to_array(fig):
    """Render a Matplotlib figure to an RGB uint8 numpy array (Gradio-friendly)."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return np.ascontiguousarray(buf[:, :, :3])


# --------------------------------------------------------------------------- #
# Gradio callbacks
# --------------------------------------------------------------------------- #
def ui_predict(model_key, V_mV, C0, q_w, R_nm):
    res = predict(model_key, V_mV, C0, q_w, R_nm)
    I = res["i_z"]
    lam = lambda_d_nm(C0)
    flags = ood_flags([V_mV, C0, q_w, R_nm])
    readout = (
        f"**Model:** {MODEL_LABEL[model_key]}\n\n"
        f"**Mean axial current:** {I.mean():.3f} pA\n\n"
        f"**Inference latency:** {res['ms']:.2f} ms (single state)\n\n"
        f"**lambda_D = {lam:.3f} nm**, lambda_D/R = {lam / R_nm:.3f}, |sigma| = {abs(sigma_mC([V_mV, C0, q_w, R_nm])):.1f} mC/m^2\n\n"
        f"**Regime:** {'in-distribution' if not flags else 'OOD: ' + ', '.join(flags)}"
    )
    row = _fig_to_array(_imshow_row(
        ["psi", "c_plus", "c_minus", "rho"],
        [res["psi"], res["c_plus"], res["c_minus"], res["rho"]],
    ))
    cur = _fig_to_array(_current_fig(I))
    return readout, row, cur


def ui_compare(model_key, V_mV, C0, q_w, R_nm):
    res = predict(model_key, V_mV, C0, q_w, R_nm)
    ref = run_reference(V_mV, C0, q_w, R_nm)
    metrics = {
        "psi rel-L2": rel_l2(res["psi"], ref["psi"]),
        "c_plus rel-L2": rel_l2(res["c_plus"], ref["c_plus"]),
        "c_minus rel-L2": rel_l2(res["c_minus"], ref["c_minus"]),
        "rho rel-L2": rel_l2(res["rho"], ref["rho"]),
        "I rel-L2": rel_l2(res["i_z"], sg_current_pA(ref["c_plus"], ref["c_minus"], ref["psi"], ref["mu"])),
        "I mean op [pA]": float(res["i_z"].mean()),
        "I mean SG/FV [pA]": float(sg_current_pA(ref["c_plus"], ref["c_minus"], ref["psi"], ref["mu"]).mean()),
    }
    txt = (
        f"Live SG/FV solve: **{ref['wall_s']:.3f} s** ({ref['method']}, "
        f"{ref['iterations']} iters, residual {ref['residual']:.2e})\n\n"
        + "\n\n".join(f"- **{k}:** {v:.3e}" if v < 1 else f"- **{k}:** {v:.3f}" for k, v in metrics.items())
    )
    epsi = res["psi"] - ref["psi"]
    erho = res["rho"] - ref["rho"]
    row = _fig_to_array(_imshow_row(
        ["psi (op)", "psi error", "rho (op)", "rho error"],
        [res["psi"], epsi, res["rho"], erho],
        diverging=False,
    ))
    # re-render error panels with diverging scale
    err_fig = _fig_to_array(_imshow_row(["psi error", "rho error"], [epsi, erho], diverging=True))
    Iref = sg_current_pA(ref["c_plus"], ref["c_minus"], ref["psi"], ref["mu"])
    cur = _fig_to_array(_current_fig(res["i_z"], Iref))
    return txt, row, err_fig, cur


def build_ui():
    import gradio as gr

    with gr.Blocks(title="Nanopore PNP Neural-Operator Simulator") as demo:
        gr.Markdown(
            "# Nanopore PNP Neural-Operator Simulator\n"
            "Parameter-conditioned surrogate over mu = (V_app, C0, q_w, R). "
            "Primary architecture: Multi-Scale Radial PI-DeepONet (charge-balanced "
            "reparameterization); Exact-Boundary FNO-2D shown as the spectral benchmark."
        )
        with gr.Row():
            with gr.Column(scale=1):
                model = gr.Dropdown(
                    choices=[("PI-DeepONet (reparam) \u2014 primary", "deeponet"),
                             ("Exact-Boundary FNO-2D (reparam) \u2014 benchmark", "fno")],
                    value="deeponet", label="Surrogate model",
                )
                V = gr.Slider(-75, 75, value=50, step=0.5, label="V_app [mV]")
                C0 = gr.Slider(10, 200, value=100, step=1, label="C0 [mol/m^3]")
                qw = gr.Slider(-0.10, 0.0, value=-0.01, step=0.001, label="q_w (dimensionless wall charge)")
                R = gr.Slider(2.0, 8.0, value=5.0, step=0.1, label="R [nm]")
                predict_btn = gr.Button("Predict (embedded ONNX)", variant="primary")
                compare_btn = gr.Button("Run Live SG/FV Solver Comparison (~0.2-2 s)")
            with gr.Column(scale=2):
                readout = gr.Markdown()
                row_img = gr.Image(label="fields", type="numpy")
                cur_img = gr.Image(label="axial current I(z)", type="numpy")
                with gr.Row():
                    compare_md = gr.Markdown()
                    err_img = gr.Image(label="error maps", type="numpy")

        predict_btn.click(ui_predict, [model, V, C0, qw, R], [readout, row_img, cur_img])
        compare_btn.click(ui_compare, [model, V, C0, qw, R], [compare_md, row_img, err_img, cur_img])
        return demo


# --------------------------------------------------------------------------- #
def smoke():
    """Headless self-test used by TASK 5.3c verification."""
    report = {}
    for key in ("deeponet", "fno"):
        res = predict(key, 50.0, 100.0, -0.01, 5.0)
        report[key] = {
            "mean_I_pA": float(res["i_z"].mean()),
            "ms": float(res["ms"]),
            "psi_range": [float(res["psi"].min()), float(res["psi"].max())],
        }
    ref = run_reference(50.0, 100.0, -0.01, 5.0)
    op = predict("deeponet", 50.0, 100.0, -0.01, 5.0)
    report["live_reference"] = {
        "wall_s": float(ref["wall_s"]),
        "method": ref["method"],
        "iterations": ref["iterations"],
        "residual": float(ref["residual"]),
        "psi_rel_l2_vs_deeponet": rel_l2(op["psi"], ref["psi"]),
        "rho_rel_l2_vs_deeponet": rel_l2(op["rho"], ref["rho"]),
        "I_rel_l2_vs_deeponet": rel_l2(op["i_z"], sg_current_pA(ref["c_plus"], ref["c_minus"], ref["psi"], ref["mu"])),
        "I_mean_deeponet_pA": float(op["i_z"].mean()),
        "I_mean_sgfv_pA": float(sg_current_pA(ref["c_plus"], ref["c_minus"], ref["psi"], ref["mu"]).mean()),
    }
    print(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="headless self-test")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    if args.smoke:
        smoke()
        return
    demo = build_ui()
    demo.queue()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share,
                show_error=True)


if __name__ == "__main__":
    main()
