# REPORT FOR CHATGPT — Stage 5B Handoff (apps hardening + figure-layout QA)

**Project:** `nanopore_pino_surrogate`
**Governing spec:** OPEN INTERPRETER RESEARCH CONTROLLER — MASTER PROMPT v2.0
**Date:** 2026-10-19 · **Stage:** 5B — make both interactive apps live-verified and fix every figure text-overflow
**State:** COMPLETE_WITH_LIMITATIONS · **Paid spend this stage: $0.0199** (gpt-4o-mini vision QA only)

---

## 1. Why Stage 5B existed
After Stage 5 was declared complete (D-035/D-036) the user reported that the web and Gradio
apps were **broken in practice**, and that Figure 1 (and others) had **text overflowing** its
boxes. Stage 5B re-opens the presentation layer with a hard rule: nothing is declared done until
each fix passes a **real, end-to-end** check (live servers over HTTP, and a programmatic pixel
bounding-box verifier over every figure).

## 2. TASK 5B.1A — web simulator made dual-engine (`file://` *and* `http://`)
**Root causes.** The Stage-5 page loaded ONNX via `fetch()` and `onnxruntime-web`; under
`file://` the fetch is blocked (CORS), leading to a blank/dead tool. `ort.env.wasm` threads/URLs
were also unset.

**Fix (all local, $0.00):**
- `scripts/build_web_bundle.py` → emits `web_simulator/data/embedded_bundle.js`
  (**5,662,351 B**): model metadata **inlined** and both ONNX graphs **base64-embedded**, so the
  page needs no fetch under `file://`.
- `web_simulator/engine_js.js` — a **pure-JS forward pass** (`runDeepONetJS`, `runFNOJS`) used as
  the primary engine, with `onnxruntime-web` retained as the HTTP path. Removed the need for
  network WASM assets in the static case.
- Rewrote `web_simulator/app.js` to a **dual-engine** design (JS engine default; ORT engine when
  the wasm assets are reachable), plus a `window.__app` testability hook.
- `index.html`/`style.css`: engine badge, live-solver button, script tags.
- **Bug found & fixed:** the branch parameter count read `branch.6.bias` (192) instead of
  `trunk.6.bias` (64); corrected → JS-vs-ONNX parity **7.363e-07** worst case.
- `scripts/test_web_simulator_dom.js` — a Node DOM harness exercising the real page logic:
  **29 passed / 0 failed** (model switch, all 8 presets + compare overlay, live-solver success
  and failure paths, engine badge).
- `web_simulator/data/js_engine_reference.json` records the JS engine's reference outputs.

## 3. TASK 5B.1B — Gradio app fixed + a zero-dependency live server
**Root causes.** (i) the model `Dropdown` passed `(label, value)` where Gradio expects
`(value, label)`; (ii) the plotting helper returned a `BytesIO` while the component expected an
array; (iii) the **compare** button was never wired; (iv) launch had no host/port control.

**Fix:**
- `scripts/app_gradio.py`: corrected the Dropdown choices order; replaced `_to_png` with
  `_fig_to_array` (numpy RGB, `type="numpy"` Image); wired `compare_btn`; added `demo.queue()`
  and `--host/--port` args (default `127.0.0.1:7860`); removed a dead `import io`.
- `scripts/serve_interactive.py` (**new**) — a dependency-free `http.server` on
  `127.0.0.1:8765` that serves `web_simulator/` **and** exposes
  `GET /api/health` + `POST /api/solve_sg` running the verified float64 SG/FV solver.
- `launch_apps.command` (macOS double-click) and `launch_apps.sh` — start both servers and open
  both URLs; `Ctrl-C` stops both.

### 3.1 Live end-to-end verification (`scripts/verify_live_servers.py`)
Both servers started as detached processes and probed over **real HTTP**
(recorded in `results/stage5b_live_server_verification.json`):
```
serve_interactive  GET /               : 200 (contains "Nanopore")
                   GET /api/health     : ok, sgfv_reference_pA 196.13722469169718
                   POST /api/solve_sg  : 200 -> 196.13722469169713 pA, direct, 4 iters,
                                         residual 2.42e-10, round-trip 0.212 s, fields 21x61 / 60
                   assets              : index/style/app/engine_js/bundle/meta/2 onnx all HTTP 200
gradio             GET / , GET /config: 200
                   /ui_predict (API)   : ok, images exist; mean I 196.132 pA
                   /ui_compare (API)   : ok, images exist; live SG/FV 0.221 s, psi rel-L2 2.126e-04
```
Both servers were **left running** (`127.0.0.1:8765`, `127.0.0.1:7860`) for interactive use;
stop with `.venv/bin/python scripts/verify_live_servers.py --stop`.

## 4. TASK 5B.2 — figure text-overflow eliminated (verifier-backed)
**Tooling.** `scripts/figures/verify_figure_layout.py` renders each figure at its real DPI and
inspects every visible artist's `get_window_extent(renderer)`:
1. **box overflow** — annotation text must lie inside its box/patch with ≥ 6 px pad;
2. **text overlap** — no two visible texts may overlap (area > 12 px²), same-axis ticks ignored;
3. **saved-PNG margin** — ≥ 8 px of pure-white margin on every side.
Off-view tick labels are excluded by checking each tick's location against the axis view
interval (a class of false positive matplotlib otherwise produces).

**Baseline → now:** 60 raw issues → **0 issues across all six figures**
(`results/figure_data/figure_layout_report.json`, `total_issues: 0`).

**Fixes applied:** Fig 1 — legend/parameter-band text moved into the legend, schematic label
spacing corrected, wall-charge count reduced, panel-B label moved off the box; Fig 2 — wider
column spacing, panel labels offset so they no longer touch the `1.0` ticks, panel-F y-label
shortened clear of the panel-E colorbar; Fig 3 — panel labels/headroom, panel-D y-label
shortened; Fig 5 — wider spacing, panel labels lifted above long titles; Fig 6 — stray off-limit
`10⁷` tick removed (axis limits tightened); global `savefig.pad_inches` 0.03 → 0.08 (fixes PNG
margins for every figure).

**Budgeted vision QA** (`scripts/vision_qa_figures.py`, `gpt-4o-mini`, high-detail images,
hard task allowance $0.03) → `results/figure_data/vision_qa_report.json`.
Spend **$0.019878** (6 calls, ~$0.0033 each). The model returned `pass=false` on all six, but
each flagged location was **cross-checked by eye against the rendered PNGs and is a false
positive** (e.g. "panel B title clipped" on Fig 5 and "panel A bottom clipped" on Fig 6 are
absent); the only legitimate note is that Fig 4's small parity **inset** text is small, which is
a deliberate inset, not a defect. The deterministic verifier is the authoritative gate; vision
QA is advisory.

## 5. Cost accounting
```
committed   = $0.01989345   (Stage 1 smoke $0.00001545 + Stage 5B vision QA $0.019878)
outstanding = $0.00000000
total       = $0.01989345   of $5.00 (remaining $4.98010655)
openai subbudget: $0.0198807 / $1.00
warning: false   stop_new_tasks: false   (fraction_used 0.00398)
```
All other Stage-5B work (bundle build, DOM harness, live servers, figure renders) is local M1
CPU — no LLM calls.

## 6. Deliverable inventory (new/changed this stage)
- `scripts/build_web_bundle.py`, `web_simulator/engine_js.js`, `web_simulator/data/embedded_bundle.js`,
  `web_simulator/data/js_engine_reference.json`; edits to `web_simulator/{index.html,style.css,app.js}`.
- `scripts/test_web_simulator_dom.js` (29/29).
- `scripts/serve_interactive.py`; edits to `scripts/app_gradio.py`; `launch_apps.command`,
  `launch_apps.sh`; `scripts/verify_live_servers.py`.
- `scripts/figures/verify_figure_layout.py`; edits to `common.py`, `fig1`, `fig2`, `fig3`, `fig5`, `fig6`.
- `scripts/vision_qa_figures.py`.
- Artifacts: `results/stage5b_live_server_verification.json`,
  `results/figure_data/figure_layout_report.json`, `results/figure_data/vision_qa_report.json`;
  refreshed `manuscript/figures/fig1`–`fig6` (PDF+PNG) + `*_data.npz`.
- Reports: this file, `STATUS.md`, `DECISIONS.md` (D-037–D-039), `NEXT_ACTIONS.md`.

## 7. Limitations / honest notes
1. `web_simulator` is proven via a Node DOM harness + HTTP asset checks; no headless **browser**
   is installed locally, so the ORT-in-browser path is validated structurally, not by a real
   browser run. The JS engine (default) is fully exercised.
2. Vision QA (`gpt-4o-mini`) over-reports; its flags were adjudicated manually and are recorded
   as advisory only.
3. Fig 4's parity inset text is intentionally small (inset), left as-is.
4. Manuscript IMRAD drafting still has **not** started.

## 8. Exact next action
Begin **Stage 5 manuscript drafting** (IMRAD; dual CPC Program-Summary + Methodological/JCP
rigor): lock the claim–evidence map to the Stage 4/5/5B numbers, embed `manuscript/figures/fig1`–`fig6`,
cite `results/tables/`, then run the budgeted cross-provider review and consistency audit.

## 9. Readiness
**GO** for Stage 5 manuscript drafting (local, unpaid). Both apps are live now at
`http://127.0.0.1:8765` and `http://127.0.0.1:7860`; all six figures pass the layout verifier
with 0 issues.
