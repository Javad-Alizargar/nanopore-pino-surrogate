#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python test_onnx_parity.py
python ../scripts/test_operators_smoke.py
