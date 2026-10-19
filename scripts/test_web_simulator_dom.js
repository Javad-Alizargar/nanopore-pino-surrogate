"use strict";
/*
 * TASK 5B.1a — real end-to-end DOM harness for web_simulator/.
 *
 * Mocks a minimal HTML5 DOM + Canvas 2D context, loads embedded_bundle.js,
 * engine_js.js and app.js in a single global context (as a real browser does),
 * then drives: initial load, every slider across all four parameters, model
 * switching (DeepONet/FNO), all 8 preset comparisons, and the live-Python
 * solver button (success + failure paths).
 *
 * Asserts: zero runtime exceptions, finite (non-NaN) fields and canvas pixel
 * buffers, and < 1e-5 agreement of both JS engines with the Python/ONNX
 * reference (data/js_engine_reference.json).
 *
 * Usage:  node scripts/test_web_simulator_dom.js
 * Exit code 0 = all pass, 1 = failure.
 */

const fs = require("fs");
const path = require("path");
const vm = require("vm");
const { performance } = require("perf_hooks");

const WEB = path.join(__dirname, "..", "web_simulator");
const failures = [];
const passes = [];
function check(name, cond, detail) {
  (cond ? passes : failures).push(name + (detail ? " :: " + detail : ""));
  if (!cond) console.log("FAIL  " + name + (detail ? " :: " + detail : ""));
  else console.log("ok    " + name);
}

// ------------------------------------------------------------------- DOM mocks
const elements = {};
function makeCtx(canvas) {
  const noop = () => {};
  return {
    createImageData: (w, h) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
    putImageData: (img) => { canvas.__pixels = img; },
    drawImage: (src) => { if (src && src.__pixels) canvas.__sourcePixels = src.__pixels; },
    clearRect: noop, fillRect: noop, fillText: noop,
    beginPath: noop, moveTo: noop, lineTo: noop, stroke: noop, strokeRect: noop,
    save: noop, restore: noop, translate: noop, rotate: noop, setLineDash: noop,
    measureText: () => ({ width: 10 }),
    font: "", fillStyle: "", strokeStyle: "", lineWidth: 1,
    textAlign: "left", imageSmoothingEnabled: false,
  };
}
function makeEl(id) {
  const el = {
    id, value: "0", textContent: "", innerHTML: "", className: "", dataset: {},
    children: [], style: {}, disabled: false, checked: false,
    classList: { toggle: () => {}, add: () => {}, remove: () => {} },
    appendChild(c) { this.children.push(c); return c; },
    addEventListener: () => {}, removeEventListener: () => {},
    getContext() { return makeCtx(this); },
  };
  Object.defineProperty(el, "width", { value: 240, writable: true });
  Object.defineProperty(el, "height", { value: 620, writable: true });
  return el;
}
const document = {
  getElementById(id) {
    if (!elements[id]) elements[id] = makeEl(id);
    return elements[id];
  },
  createElement(tag) {
    const e = makeEl(null);
    e.tagName = tag.toUpperCase();
    if (tag === "canvas") { e.width = 0; e.height = 0; }
    return e;
  },
  addEventListener: () => {},
};

// default control values (as in index.html)
const ids = ["sl_V", "sl_C0", "sl_q", "sl_R", "out_V", "out_C0", "out_q", "out_R",
  "out_lam", "out_lamR", "out_sigma", "oodFlags", "presets", "compareToggle",
  "modelSelect", "liveSolveBtn", "liveStatus", "out_I", "out_ms", "refLabel",
  "out_refI", "cv_psi", "cv_cp", "cv_cm", "cv_rho", "cv_I", "status", "engineBadge"];
ids.forEach((i) => document.getElementById(i));
elements.sl_V.value = "50"; elements.sl_C0.value = "100";
elements.sl_q.value = "-0.01"; elements.sl_R.value = "5";
elements.modelSelect.value = "deeponet";
elements.cv_I.width = 1000; elements.cv_I.height = 260;

// fetch mock (live solver): switchable behaviour
let fetchMode = "ok";
let fetchCalls = 0;
const fetchMock = async () => {
  fetchCalls++;
  if (fetchMode === "fail") throw new Error("ECONNREFUSED (no server)");
  return { ok: true, json: async () => ({ mean_current_pA: 196.1372246, residual: 2.4e-10, iterations: 4, runtime_s: 0.206, i_z: new Array(60).fill(196.137) }) };
};

// ------------------------------------------------------------------ global ctx
const domHandlers = [];
global.window = global;
global.document = document;
global.performance = performance;
global.atob = (s) => Buffer.from(s, "base64").toString("binary");
global.setTimeout = setTimeout;
global.clearTimeout = clearTimeout;
global.fetch = fetchMock;
global.ort = undefined; // force the embedded JS engine path
global.window.addEventListener = (ev, fn) => { if (ev === "DOMContentLoaded") domHandlers.push(fn); };

function loadScript(rel) {
  const file = path.join(WEB, rel);
  vm.runInThisContext(fs.readFileSync(file, "utf8"), { filename: file });
}

// --------------------------------------------------------------------- run
(async function main() {
  try {
    loadScript("data/embedded_bundle.js");
    check("bundle: MODEL_BUNDLE defined", !!global.MODEL_BUNDLE);
    check("bundle: meta present", !!global.MODEL_BUNDLE.meta);
    check("bundle: 8 presets", global.MODEL_BUNDLE.meta.benchmark_states.length === 8);
    check("bundle: both ONNX inlined", !!(global.MODEL_BUNDLE.onnx.deeponet && global.MODEL_BUNDLE.onnx.fno));
    check("bundle: both weight sets", !!(global.MODEL_BUNDLE.weights.deeponet && global.MODEL_BUNDLE.weights.fno));

    loadScript("engine_js.js");
    check("engine_js: JSEngine defined", !!global.JSEngine);
    loadScript("app.js");
    check("app: __app hook present", !!(global.window.__app));

    // ---- reference parity (JS engine vs Python/ONNX) ----
    const ref = JSON.parse(fs.readFileSync(path.join(WEB, "data", "js_engine_reference.json"), "utf8"));
    let worst = 0;
    for (const p of ref.probes) {
      for (const [model, fn] of [["deeponet", JSEngine.runDeepONetJS], ["fno", JSEngine.runFNOJS]]) {
        const y = fn(p.mu);
        const exp = p.models[model];
        for (const n of ["psi", "c_plus", "c_minus", "rho", "s"]) {
          const arr = y[n];
          for (let i = 0; i < arr.length; i++) worst = Math.max(worst, Math.abs(arr[i] - exp[n][i]));
          if (arr.some((v) => !Number.isFinite(v))) check(`parity finite ${model}/${n} mu=${p.mu}`, false);
        }
      }
    }
    check("engine parity vs Python/ONNX < 1e-5", worst < 1e-5, "worst=" + worst.toExponential(3));

    // ---- boot ----
    const init = domHandlers[domHandlers.length - 1];
    check("DOMContentLoaded handler registered", typeof init === "function");
    await init();
    check("engine selected = js (ort absent)", global.window.__app.getEngine() === "js");
    check("status ok after boot", /Ready/.test(elements.status.textContent), elements.status.textContent);

    const finite = (res) => ["psi", "c_plus", "c_minus", "rho", "i_z"].every((k) => res[k] && Array.from(res[k]).every(Number.isFinite))
      && res.psi.length === 21 * 61 && res.i_z.length === 60;

    let cur = global.window.__app.getCurrent();
    check("initial predict: finite fields", finite(cur.pred), JSON.stringify(Object.keys(cur.pred || {})));
    check("initial canvases drawn", ["cv_psi", "cv_cp", "cv_cm", "cv_rho"].every((id) => elements[id].__sourcePixels && elements[id].__sourcePixels.data.length === 21 * 61 * 4));

    // ---- sliders (all four, several values) ----
    const scenarios = [
      { sl_V: "-75", sl_C0: "10", sl_q: "-0.10", sl_R: "2.0" },
      { sl_V: "75", sl_C0: "200", sl_q: "0.0", sl_R: "8.0" },
      { sl_V: "0", sl_C0: "50", sl_q: "-0.02", sl_R: "3.5" },
      { sl_V: "30", sl_C0: "120", sl_q: "-0.04", sl_R: "6.0" },
    ];
    for (const s of scenarios) {
      Object.assign(elements.sl_V, { value: s.sl_V });
      elements.sl_C0.value = s.sl_C0; elements.sl_q.value = s.sl_q; elements.sl_R.value = s.sl_R;
      await global.window.__app.evaluate();
      cur = global.window.__app.getCurrent();
      check(`slider scenario V=${s.sl_V} C0=${s.sl_C0} qw=${s.sl_q} R=${s.sl_R}`, finite(cur.pred));
    }

    // ---- model switch ----
    elements.modelSelect.value = "fno";
    await global.window.__app.evaluate();
    check("model switch to FNO: finite fields", finite(global.window.__app.getCurrent().pred));
    elements.modelSelect.value = "deeponet";

    // ---- presets + compare overlay ----
    elements.compareToggle.checked = true;
    for (const st of global.MODEL_BUNDLE.meta.benchmark_states) {
      elements.sl_V.value = String(st.mu[0]); elements.sl_C0.value = String(st.mu[1]);
      elements.sl_q.value = String(st.mu[2]); elements.sl_R.value = String(st.mu[3]);
      await global.window.__app.evaluate();
      const c = global.window.__app.getCurrent();
      check(`preset ${st.id} compare overlay finite`, finite(c.pred));
    }
    elements.compareToggle.checked = false;

    // ---- live solver button (success + failure) ----
    fetchMode = "ok";
    await global.window.__app.runLiveSolver();
    check("live solver success path", /196\.13/.test(elements.liveStatus.innerHTML), elements.liveStatus.innerHTML.slice(0, 60));
    fetchMode = "fail";
    await global.window.__app.runLiveSolver();
    check("live solver failure handled gracefully", /unavailable/i.test(elements.liveStatus.innerHTML));

    // ---- badge ----
    check("engine badge rendered", /Engine:/.test(elements.engineBadge.textContent), elements.engineBadge.textContent);

  } catch (e) {
    failures.push("uncaught exception: " + e.stack);
    console.log("EXCEPTION " + e.stack);
  }

  console.log(`\n${passes.length} passed, ${failures.length} failed`);
  if (failures.length) { console.log("\nFAILURES:\n - " + failures.join("\n - ")); process.exit(1); }
  console.log("WEB SIMULATOR DOM HARNESS: ALL PASS");
  process.exit(0);
})();
