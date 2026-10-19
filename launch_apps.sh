#!/bin/bash
# Launch BOTH interactive apps (same behaviour as launch_apps.command).
#   - static ONNX browser simulator + live Python SG/FV API : http://127.0.0.1:8765
#   - Gradio app                                            : http://127.0.0.1:7860
set -e
cd "$(dirname "$0")"

PY=".venv/bin/python"
if [ ! -x "$PY" ]; then PY="python3"; fi
NO_OPEN="${NO_OPEN:-0}"

echo "Starting live SG/FV + static simulator on http://127.0.0.1:8765 ..."
"$PY" scripts/serve_interactive.py &
SERVE_PID=$!

echo "Starting Gradio app on http://127.0.0.1:7860 ..."
"$PY" scripts/app_gradio.py --port 7860 &
GRADIO_PID=$!

cleanup() { kill "$SERVE_PID" "$GRADIO_PID" 2>/dev/null || true; }
trap cleanup INT TERM

sleep 4
if [ "$NO_OPEN" != "1" ]; then
  open "http://127.0.0.1:8765" 2>/dev/null || true
  open "http://127.0.0.1:7860" 2>/dev/null || true
fi
echo "Both apps running. Press Ctrl+C to stop."
wait
