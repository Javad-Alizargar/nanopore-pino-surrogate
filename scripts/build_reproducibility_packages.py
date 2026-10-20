#!/usr/bin/env python3
"""Build the two release packages for the nanopore PINO surrogate.

Public package  (reproducibility/public_repo/): legally shareable allowlist,
no credentials, no governing spec, no private logs.
Private package (reproducibility/github/): full project state, secret file
types excluded, produced for the authorized private GitHub repository.

Both packages get a SHA-256 manifest. Exact values stay in results/.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "reproducibility" / "public_repo"
PRIV = ROOT / "reproducibility" / "github"

PUBLIC_FILES = [
    "scripts/app_gradio.py",
    "scripts/serve_interactive.py",
    "scripts/build_web_bundle.py",
    "scripts/build_web_simulator.py",
    "scripts/generate_parametric_dataset.py",
    "scripts/make_data_manifest.py",
    "scripts/train_and_evaluate_suite.py",
    "scripts/fno_extended_check.py",
    "scripts/run_inverse_recovery_demo.py",
    "scripts/run_stage10_upgrades.py",
    "scripts/run_stage11_reviewer_upgrades.py",
    "scripts/enforce_sequential_numbering_and_inline_figures.py",
    "scripts/verify_simulators.py",
    "scripts/verify_live_servers.py",
    "scripts/test_operators_smoke.py",
    "scripts/test_web_simulator_dom.js",
    "scripts/benchmark_sg_m1.py",
    "results/statistics.json",
    "results/dataset_manifest.json",
    "results/dataset_manifest_384.json",
    "results/stage10_upgrades.json",
    "results/stage11_upgrades.json",
    "results/stage11_numbering_report.json",
    "results/stage2_sg_baseline_verification.json",
    "results/stage3_operator_smoke.json",
    "results/stage5_simulator_verification.json",
    "results/stage5b_live_server_verification.json",
    "results/inverse_recovery_demo.json",
    "data/processed/pnp_parametric_240.npz",
    "data/processed/pnp_parametric_384.npz",
    "results/models/deeponet_reparam_seed42.onnx",
    "results/models/fno_reparam_seed42.onnx",
    "results/models/deeponet_reparam_data_plus_physics_seed42.pt",
    "results/models/fno_reparam_data_plus_physics_seed42.pt",
]
PUBLIC_TREES = [
    "scripts/operators",
    "scripts/figures",
    "scripts/sg_baseline",
    "web_simulator",
]
PUBLIC_GLOBS = [
    "results/tables/*.csv",
    "results/figure_data/*.json",
    "results/figure_data/*.npz",
    "manuscript/figures/*.png",
]

PRIVATE_EXCLUDE_DIRS = {".git", ".venv", "venv", "__pycache__", "apis", "APIS",
                        "logs", "workers", ".run_state", "reproducibility",
                        "node_modules", ".mypy_cache", ".pytest_cache"}
PRIVATE_EXCLUDE_SUFFIX = {".pyc", ".pyo", ".sqlite", ".sqlite-wal", ".sqlite-journal",
                          ".lock", ".DS_Store", ".key", ".pem"}
PRIVATE_EXCLUDE_NAMES = {"STOP_REQUESTED"}

REQUIREMENTS = """numpy==2.5.4
scipy==1.18.1
torch==2.14.1
matplotlib==3.11.2
onnx==1.23.2
onnxruntime==1.31.0
gradio==6.30.0
python-docx==1.2.0
Pillow==12.3.0
PyYAML==6.0.3
requests==2.34.2
pandas==3.0.6
"""

GITIGNORE = """.venv/
__pycache__/
*.pyc
*.npy
*.npz.bak
.DS_Store
*.key
*.pem
.env
apis/
logs/
"""

README = """# nanopore-pino-surrogate

Parameter-conditioned physics-informed neural operator (PINO) for steady
axisymmetric Poisson--Nernst--Planck (PNP) transport in a charged cylindrical
nanopore. The repository contains the conservative Scharfetter--Gummel
finite-volume reference solver, the 240-state benchmark and a 384-state scaling
expansion, a multi-scale radial physics-informed DeepONet, a real-DFT Fourier
neural operator, the inverse recovery demo, a trunk ablation, a data-scaling
study, a Fisher-information and multi-protocol identifiability analysis, a
four-way constraint-scheme benchmark, an exact SG/FV Fisher validation, a
physics-weight sweep, and a static browser simulator.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/test_operators_smoke.py          # ONNX/PyTorch parity smoke test
python scripts/run_inverse_recovery_demo.py     # five-state inverse demo
python scripts/run_stage10_upgrades.py          # trunk ablation, scaling law, identifiability
python scripts/run_stage11_reviewer_upgrades.py # 4-way constraints, physics sweep, exact Fisher
python scripts/serve_interactive.py             # live SG/FV + operator service
python scripts/app_gradio.py                    # Gradio comparison app
```

Open `web_simulator/index.html`, or use the hosted portal at
<https://pino.statwhy.com/> (ONNX operator predictions, verified benchmark
cases, and a live SG/FV reference solver).

## Layout

| Path | Contents |
|---|---|
| `scripts/operators/` | DeepONet/FNO, boundary ansatz, SG losses, constants |
| `scripts/sg_baseline/` | conservative SG/FV reference solver |
| `scripts/train_and_evaluate_suite.py` | three-seed training/evaluation |
| `scripts/run_inverse_recovery_demo.py` | radius/charge inverse demo |
| `scripts/run_stage10_upgrades.py` | trunk ablation, 384-state scaling law, Fisher/multi-protocol identifiability |
| `scripts/run_stage11_reviewer_upgrades.py` | four-way constraint benchmark, physics-weight sweep, exact SG/FV Fisher validation |
| `scripts/enforce_sequential_numbering_and_inline_figures.py` | first-appearance reference/table renumbering and inline-figure audit |
| `scripts/figures/` | figure generation and layout verification |
| `results/` | statistics, tables 1--6 and S3, S14--S18, verification JSONs, export models |
| `data/processed/` | 240-state dataset and 384-state scaling expansion |
| `web_simulator/` | static browser simulator (ONNX + JS SG engine) |
| `tests/` | smoke and parity tests |

## Reproducibility notes

- The dataset was generated on an Apple M1 with Python 3.14, seed 20261018.
- `results/statistics.json` and `results/tables/*.csv` hold full-precision
  values; the manuscript reports 3--4 significant figures.
- Fixed physical settings: $T=298.15$ K, $\\varepsilon_r=78.5$,
  $D_+=1.96\\times10^{-9}$, $D_-=2.03\\times10^{-9}$ m$^2$ s$^{-1}$, $L=30$ nm.
- License: MIT. Archived copy to be deposited on Zenodo at acceptance.
"""

ARCH_MAP = """# Architecture map

| Stage | Script | Inputs | Outputs | Figures/Tables |
|---|---|---|---|---|
| Dataset | `scripts/generate_parametric_dataset.py` | `scripts/operators/constants.py`, `scripts/sg_baseline/` | `data/processed/pnp_parametric_240.npz`, `results/dataset_manifest.json` | Table 1 |
| Training/eval | `scripts/train_and_evaluate_suite.py` | dataset, `scripts/operators/` | `results/models/*.pt`, `results/statistics.json`, `results/tables/table3_*.csv` | Tables 2--5, Figures 2--4 |
| FNO budget | `scripts/fno_extended_check.py` | dataset | `results/tables/table5_*.csv` | Table 5 |
| Inverse | `scripts/run_inverse_recovery_demo.py` | dataset, ONNX models | `results/inverse_recovery_demo.json`, `results/tables/table6_*.csv` | Table 6, Figure 6B |
| Timing | `scripts/benchmark_sg_m1.py` | models, solver | `results/tables/table4_*.csv` | Figure 6A |
| Stage-10 upgrades | `scripts/run_stage10_upgrades.py` | 240/384-state datasets, `scripts/operators/` | `results/stage10_upgrades.json`, `results/tables/table_s14_*.csv`, `table_s15_*.csv`, `data/processed/pnp_parametric_384.npz` | Figures 3E,F, 4D; Tables S3, S15 |
| Stage-11 upgrades | `scripts/run_stage11_reviewer_upgrades.py` | 240-state dataset, ONNX operator, SG/FV solver | `results/stage11_upgrades.json`, `results/tables/table_sgfv_fisher_validation.csv`, `table_constraint_schemes_4way.csv`, `table_physics_weight_sensitivity.csv` | Figures 3A--D, 6C,D; Tables S16--S18 |
| Figures | `scripts/figures/*.py` | `results/*` | `figures/*.png`, `pdf` | Figures 1--6 |
| Deployment | `scripts/build_web_simulator.py`, `scripts/build_web_bundle.py` | ONNX models | `web_simulator/` | -- |
| Live apps | `scripts/serve_interactive.py`, `scripts/app_gradio.py` | solver, models | HTTP/Gradio services | -- |

Constraint handling lives in `scripts/operators/boundary_ansatz.py`
(exact reservoir Dirichlet + charge-balanced reconstruction). The SG face
flux and physics residuals live in `scripts/operators/losses_and_metrics.py`.
"""

TESTS_README = """# Tests

- `test_onnx_parity.py` checks that both ONNX models load and that an
  onnxruntime forward pass runs on a random normalized input.
- `run_tests.sh` runs the package smoke tests. Some tests need the dataset
  under `data/processed/` and the models under `results/models/`.
"""

TEST_ONNX = '''#!/usr/bin/env python3
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
'''

RUN_TESTS = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python test_onnx_parity.py
python ../scripts/test_operators_smoke.py
"""

INDEX = """<!doctype html>
<html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="0; url=web_simulator/index.html">
<title>nanopore-pino-surrogate</title></head>
<body><p>Redirecting to the <a href="web_simulator/index.html">browser simulator</a>.</p></body>
</html>
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    for path in src.rglob("*"):
        if path.is_dir():
            continue
        if "__pycache__" in path.parts or path.suffix in (".pyc", ".DS_Store"):
            continue
        copy_file(path, dst / path.relative_to(src))


def copy_private(src: Path, dst: Path) -> None:
    for path in src.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(ROOT)
        if any(part in PRIVATE_EXCLUDE_DIRS for part in rel.parts):
            continue
        if path.suffix in PRIVATE_EXCLUDE_SUFFIX or path.name in PRIVATE_EXCLUDE_NAMES:
            continue
        copy_file(path, dst / rel)


def build_public() -> dict:
    if PUB.exists():
        shutil.rmtree(PUB)
    PUB.mkdir(parents=True)
    copied = []
    for rel in PUBLIC_FILES:
        src = ROOT / rel
        if src.exists():
            copy_file(src, PUB / rel)
            copied.append(rel)
    for rel in PUBLIC_TREES:
        src = ROOT / rel
        if src.exists():
            copy_tree(src, PUB / rel)
            copied.append(rel + "/**")
    for pattern in PUBLIC_GLOBS:
        for src in ROOT.glob(pattern):
            rel = src.relative_to(ROOT)
            copy_file(src, PUB / rel)
            copied.append(str(rel))
    (PUB / "requirements.txt").write_text(REQUIREMENTS)
    (PUB / ".gitignore").write_text(GITIGNORE)
    (PUB / "README.md").write_text(README)
    (PUB / "ARCHITECTURE_MAP.md").write_text(ARCH_MAP)
    (PUB / "LICENSE").write_text(
        "MIT License\n\nCopyright (c) 2026 Javad Alizargar\n\n"
        "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
        "of this software and associated documentation files (the \"Software\"), to deal\n"
        "in the Software without restriction, including without limitation the rights\n"
        "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n"
        "copies of the Software, and to permit persons to whom the Software is\n"
        "furnished to do so, subject to the following conditions:\n\n"
        "The above copyright notice and this permission notice shall be included in all\n"
        "copies or substantial portions of the Software.\n\n"
        "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n"
        "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n"
        "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n"
        "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n"
        "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n"
        "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n"
        "SOFTWARE.\n")
    (PUB / "index.html").write_text(INDEX)
    (PUB / ".nojekyll").write_text("")
    if (PUB / "web_simulator").exists():
        (PUB / "web_simulator" / ".nojekyll").write_text("")
    (PUB / "tests").mkdir(exist_ok=True)
    (PUB / "tests" / "README.md").write_text(TESTS_README)
    (PUB / "tests" / "test_onnx_parity.py").write_text(TEST_ONNX)
    (PUB / "tests" / "run_tests.sh").write_text(RUN_TESTS)
    (PUB / "tests" / "run_tests.sh").chmod(0o755)
    fig_src = ROOT / "manuscript" / "figures"
    if fig_src.exists():
        for png in sorted(fig_src.glob("*.png")):
            copy_file(png, PUB / "figures" / png.name)
    return manifest(PUB, copied)


def build_private() -> dict:
    if PRIV.exists():
        shutil.rmtree(PRIV)
    PRIV.mkdir(parents=True)
    copy_private(ROOT, PRIV)
    (PRIV / ".gitignore").write_text(GITIGNORE + "download/\ndata/processed/\n")
    (PRIV / "README_PRIVATE.md").write_text(
        "# Private research repository\n\n"
        "Full project state for `nanopore_pino_surrogate`. Public release set:\n"
        "`reproducibility/public_repo/`. Credentials (`apis/`) and the virtual\n"
        "environment are excluded. Rebuild with\n"
        "`python scripts/build_reproducibility_packages.py`.\n")
    return manifest(PRIV, None)


def manifest(root: Path, copied: list[str] | None) -> dict:
    entries = []
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_dir() or path.name == "PACKAGE_MANIFEST.json":
            continue
        size = path.stat().st_size
        total += size
        entries.append({
            "path": str(path.relative_to(root)),
            "bytes": size,
            "sha256": sha256(path),
        })
    data = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "root": str(root.relative_to(ROOT)),
        "file_count": len(entries),
        "total_bytes": total,
        "files": entries,
    }
    (root / "PACKAGE_MANIFEST.json").write_text(json.dumps(data, indent=2))
    return data


def main() -> None:
    pub = build_public()
    priv = build_private()
    summary = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "public_repo": {k: pub[k] for k in ("file_count", "total_bytes")},
        "private_repo": {k: priv[k] for k in ("file_count", "total_bytes")},
    }
    (ROOT / "reproducibility" / "PACKAGE_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
