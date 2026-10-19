#!/bin/bash
# Double-click in macOS Finder to launch BOTH interactive apps.
#   - static ONNX browser simulator + live Python SG/FV API : http://127.0.0.1:8765
#   - Gradio app                                            : http://127.0.0.1:7860
# Close this Terminal window to stop both servers.
cd "$(dirname "$0")" || exit 1
PY=".venv/bin/python"
if [ ! -x "$PY" ]; then PY="python3"; fi

echo "Starting live SG/FV + static simulator on http://127.0.0.1:8765 ..."
"$PY" scripts/serve_interactive.py &
SERVE_PID=$!

echo "Starting Gradio app on http://127.0.0.1:7860 ..."
"$PY" scripts/app_gradio.py --port 7860 &
GRADIO_PID=$!

sleep 4
open "http://127.0.0.1:8765"
open "http://127.0.0.1:7860"

echo "Both apps running. Press Ctrl+C to stop."
trap 'kill $SERVE_PID $GRADIO_PID 2>/dev/null' INT TERM
wait
