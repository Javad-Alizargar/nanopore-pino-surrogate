"""TASK 5B.1b — zero-dependency interactive web server.

Serves the static ``web_simulator/`` app AND exposes a local JSON API:

    GET  /                 -> web_simulator/index.html (and all static assets)
    GET  /api/health       -> {"ok": true, ...}
    POST /api/solve_sg     -> run the verified float64 SG/FV solver live
                              body:  {"V_app":50,"C0":100,"qw":-0.01,"R":5.0}
                              reply: mean_current_pA, i_z, fields, residual,
                                     iterations, runtime_s, method

Uses only the Python standard library (``http.server``) plus numpy + the
in-repo solver.  Runs on 127.0.0.1:8765 by default.

    .venv/bin/python scripts/serve_interactive.py [--port 8765]
"""

import argparse
import json
import os
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

WEB = os.path.join(ROOT, "web_simulator")

# reuse the verified solver path (direct solve + continuation fallback)
from generate_parametric_dataset import solve_state  # noqa: E402

NR, NZ = 21, 61
FARADAY, DIFF_CATION, DIFF_ANION, L_NM = 96485.33212, 1.96e-9, 2.03e-9, 30.0


def _annular_weights(r):
    edges = np.empty(r.size + 1)
    edges[0], edges[-1] = 0.0, 1.0
    edges[1:-1] = 0.5 * (r[:-1] + r[1:])
    return 0.5 * (edges[1:] ** 2 - edges[:-1] ** 2)


def _bernoulli(x):
    xc = np.clip(x, -80.0, 80.0)
    small = np.abs(xc) < 1e-8
    series = 1.0 - xc / 2.0 + xc ** 2 / 12.0
    xs = np.where(small, 1.0, xc)
    return np.where(small, series, xs / np.expm1(xs))


def sg_current_pA(cp, cm, psi, mu):
    r = np.linspace(0.0, 1.0, NR)
    w = _annular_weights(r)
    dz = 1.0 / (NZ - 1)
    cpb, cmb, psib = (np.asarray(a, float).reshape(NR, NZ) for a in (cp, cm, psi))
    dpsi = psib[:, 1:] - psib[:, :-1]
    jp = -(cpb[:, 1:] * _bernoulli(-dpsi) - cpb[:, :-1] * _bernoulli(dpsi)) / dz
    jm = -(cmb[:, 1:] * _bernoulli(dpsi) - cmb[:, :-1] * _bernoulli(-dpsi)) / dz
    dimless = 2.0 * np.pi * np.sum((DIFF_CATION * jp - DIFF_ANION * jm) * w[:, None], axis=0)
    c0, r_m = float(mu[1]), float(mu[3]) * 1e-9
    return dimless * (FARADAY * c0 * r_m ** 2 / (L_NM * 1e-9)) * 1e12


def solve_sg(V_mV, C0, q_w, R_nm):
    t0 = time.perf_counter()
    sol, cfg, grid, method, n_cont = solve_state(float(V_mV), float(C0), float(q_w), float(R_nm))
    wall = time.perf_counter() - t0
    mu = [float(V_mV), float(C0), float(q_w), float(R_nm)]
    cp = np.asarray(sol.c_plus, float)
    cm = np.asarray(sol.c_minus, float)
    psi = np.asarray(sol.psi, float)
    i_z = sg_current_pA(cp, cm, psi, mu)
    return {
        "mu": mu,
        "mean_current_pA": float(i_z.mean()),
        "i_z": [float(v) for v in i_z],
        "psi": [float(f"{v:.6g}") for v in psi.ravel()],
        "c_plus": [float(f"{v:.6g}") for v in cp.ravel()],
        "c_minus": [float(f"{v:.6g}") for v in cm.ravel()],
        "rho": [float(f"{v:.6g}") for v in (cp - cm).ravel()],
        "residual": float(sol.residual),
        "iterations": int(sol.iterations),
        "runtime_s": float(wall),
        "method": method,
        "n_continuation_steps": int(n_cont),
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=WEB, **kw)

    def log_message(self, fmt, *args):
        sys.stderr.write("[serve_interactive] " + (fmt % args) + "\n")

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path == "/api/health":
            return self._json(200, {"ok": True, "service": "serve_interactive",
                                    "sgfv_reference_pA": 196.13722469169718})
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/solve_sg":
            return self._json(404, {"error": "unknown endpoint"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            V = float(payload.get("V_app", 50.0))
            C0 = float(payload.get("C0", 100.0))
            qw = float(payload.get("qw", -0.01))
            R = float(payload.get("R", 5.0))
        except Exception as e:  # noqa: BLE001
            return self._json(400, {"error": f"bad request: {e}"})
        try:
            result = solve_sg(V, C0, qw, R)
        except Exception as e:  # noqa: BLE001
            return self._json(500, {"error": f"solver failed: {e}"})
        return self._json(200, result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve_interactive] serving {WEB} on http://{args.host}:{args.port}", flush=True)
    print(f"[serve_interactive] live SG/FV endpoint: POST http://{args.host}:{args.port}/api/solve_sg", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
