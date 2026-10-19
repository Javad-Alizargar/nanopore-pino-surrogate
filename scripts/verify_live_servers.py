"""TASK 5B.1b — live end-to-end verification of both interactive apps.

Starts ``scripts/serve_interactive.py`` (127.0.0.1:8765) and
``scripts/app_gradio.py`` (127.0.0.1:7860) as detached background servers, waits
until each is listening, probes every endpoint over real HTTP (GET /, /config,
/api/health, POST /api/solve_sg, and the Gradio prediction API), writes
``results/stage5b_live_server_verification.json`` and **leaves both servers
running** so the user can open them immediately.

    .venv/bin/python scripts/verify_live_servers.py [--stop]
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, ".venv", "bin", "python")
OUT = os.path.join(ROOT, "results", "stage5b_live_server_verification.json")
LOGDIR = "/tmp"

INTERACTIVE = ("127.0.0.1", 8765)
GRADIO = ("127.0.0.1", 7860)


def port_up(host, port, timeout=1.0):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def wait_port(host, port, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if port_up(host, port):
            return True
        time.sleep(0.5)
    return False


def http_get(url, timeout=15):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def http_post_json(url, obj, timeout=60):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read())


def start(script, port, log):
    lf = open(os.path.join(LOGDIR, log), "w")
    p = subprocess.Popen([PY, os.path.join(HERE, script), "--port", str(port)],
                         cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                         start_new_session=True)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop", action="store_true", help="stop servers instead of starting")
    args = ap.parse_args()
    if args.stop:
        subprocess.run(["pkill", "-f", "scripts/serve_interactive.py"], check=False)
        subprocess.run(["pkill", "-f", "scripts/app_gradio.py"], check=False)
        print("stopped.")
        return

    report = {"stage": "5B.1b", "generated_local_only": True, "servers": {}}

    # ---- interactive ------------------------------------------------------
    procs = {}
    if port_up(*INTERACTIVE):
        report["servers"]["interactive"] = {"note": "already running"}
    else:
        procs["interactive"] = start("serve_interactive.py", INTERACTIVE[1], "serve_int.log")
        if not wait_port(*INTERACTIVE, timeout=30):
            raise RuntimeError("serve_interactive failed to listen")
    t0 = time.time()
    status, body = http_get(f"http://{INTERACTIVE[0]}:{INTERACTIVE[1]}/")
    assert status == 200 and b"Nanopore" in body, "index.html not served"
    _, health = http_get(f"http://{INTERACTIVE[0]}:{INTERACTIVE[1]}/api/health")
    t_solve = time.time()
    s_status, solved = http_post_json(f"http://{INTERACTIVE[0]}:{INTERACTIVE[1]}/api/solve_sg",
                                      {"V_app": 50, "C0": 100, "qw": -0.01, "R": 5.0})
    solve_wall = time.time() - t_solve
    report["servers"]["interactive"] = {
        "url": f"http://{INTERACTIVE[0]}:{INTERACTIVE[1]}",
        "index_http": status,
        "index_contains_Nanopore": b"Nanopore" in body,
        "health": json.loads(health),
        "solve_sg_http": s_status,
        "solve_sg_mean_current_pA": solved["mean_current_pA"],
        "solve_sg_method": solved["method"],
        "solve_sg_iterations": solved["iterations"],
        "solve_sg_residual": solved["residual"],
        "solve_sg_runtime_s": solved["runtime_s"],
        "solve_sg_roundtrip_s": round(solve_wall, 3),
        "solve_sg_fields_ok": len(solved["psi"]) == 21 * 61 and len(solved["i_z"]) == 60,
        "note": "served web_simulator/ + live SG/FV JSON API",
    }

    # ---- gradio -----------------------------------------------------------
    if port_up(*GRADIO):
        report["servers"]["gradio"] = {"note": "already running"}
    else:
        procs["gradio"] = start("app_gradio.py", GRADIO[1], "gradio.log")
        if not wait_port(*GRADIO, timeout=60):
            raise RuntimeError("app_gradio failed to listen")
    g_status, g_body = http_get(f"http://{GRADIO[0]}:{GRADIO[1]}/")
    c_status, c_body = http_get(f"http://{GRADIO[0]}:{GRADIO[1]}/config")
    gradio_entry = {
        "url": f"http://{GRADIO[0]}:{GRADIO[1]}",
        "index_http": g_status,
        "config_http": c_status,
    }
    try:
        from gradio_client import Client
        cl = Client(f"http://{GRADIO[0]}:{GRADIO[1]}/", verbose=False)
        md, fields, cur = cl.predict("deeponet", 50.0, 100.0, -0.01, 5.0, api_name="/ui_predict")
        md2, fields2, err2, cur2 = cl.predict("deeponet", 50.0, 100.0, -0.01, 5.0, api_name="/ui_compare")
        gradio_entry.update({
            "predict_api": "ok",
            "predict_readout": md.splitlines()[2] if len(md.splitlines()) > 2 else md[:60],
            "predict_images_exist": all(os.path.exists(p) for p in (fields, cur)),
            "compare_api": "ok",
            "compare_live_solve_head": md2.splitlines()[0][:80],
            "compare_images_exist": all(os.path.exists(p) for p in (fields2, err2, cur2)),
        })
    except Exception as e:  # noqa: BLE001
        gradio_entry["api_error"] = repr(e)
    report["servers"]["gradio"] = gradio_entry

    # ---- leave running ----------------------------------------------------
    for name, p in procs.items():
        report["servers"][name] = dict(report["servers"].get(name, {}))
        report["servers"][name]["pid"] = p.pid
        report["servers"][name]["left_running"] = True

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    print("\nwrote", OUT)
    print("SERVERS LEFT RUNNING:",
          {"serve_interactive": f"http://{INTERACTIVE[0]}:{INTERACTIVE[1]}",
           "gradio": f"http://{GRADIO[0]}:{GRADIO[1]}"})
    print("Stop with: .venv/bin/python scripts/verify_live_servers.py --stop")


if __name__ == "__main__":
    main()
