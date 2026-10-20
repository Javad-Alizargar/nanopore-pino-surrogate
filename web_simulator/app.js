"use strict";
/*
 * Nanopore PNP neural-operator simulator — dual-engine, zero-failure front-end.
 *
 * Engine 1 (preferred): onnxruntime-web WASM running the embedded (base64) ONNX
 *   operators. Requires the CDN .wasm; used automatically when available.
 * Engine 2 (fallback): pure-JS Float32 forward passes (engine_js.js) reading the
 *   embedded PyTorch weights. Needs no network/WASM → works under file://.
 *
 * All metadata (grid, constants, ranges, presets) and both ONNX binaries come
 * from data/embedded_bundle.js, so nothing is fetched over the network.
 */

const NR = 21, NZ = 61, N = NR * NZ;
let META = null;
const SESSIONS = {};
let ENGINE = "js"; // "onnx" | "js"
const CURRENT = { mu: [50, 100, -0.01, 5], pred: null, preset: null };
// Same-origin when served over http(s) (deployed portal), localhost otherwise.
const LOCAL_API = "http://127.0.0.1:8765/api/solve_sg";
const LIVE_API = (typeof location !== "undefined" && /^https?:$/.test(location.protocol))
  ? "/api/solve_sg" : LOCAL_API;

// ------------------------------------------------------------------ colormap
const VIRIDIS_STOPS = [
  [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
  [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37],
];
function buildLUT(stops, n = 256) {
  const lut = new Uint8ClampedArray(n * 3);
  for (let k = 0; k < n; k++) {
    const t = k / (n - 1) * (stops.length - 1);
    const i = Math.min(Math.floor(t), stops.length - 2);
    const f = t - i;
    for (let c = 0; c < 3; c++) lut[k * 3 + c] = stops[i][c] + (stops[i + 1][c] - stops[i][c]) * f;
  }
  return lut;
}
const LUT_MAIN = buildLUT(VIRIDIS_STOPS);
const DIVERGING = [
  [49, 54, 149], [69, 117, 180], [171, 217, 233], [255, 255, 255],
  [253, 174, 97], [215, 48, 39], [165, 0, 38],
];
const LUT_ERR = buildLUT(DIVERGING);

// ------------------------------------------------------------- physics helpers
function bernoulli(x) {
  const xc = Math.max(-80, Math.min(80, x));
  if (Math.abs(xc) < 1e-8) return 1 - xc / 2 + xc * xc / 12;
  return xc / (Math.exp(xc) - 1);
}
function sgCurrentProfile(cp, cm, psi, mu) {
  const { dz, weights } = META.grid;
  const { FARADAY: F, D_cation: Dc, D_anion: Da, L_nm } = META.constants;
  const c0 = mu[1], rM = mu[3] * 1e-9;
  const scale = F * c0 * rM * rM / (L_nm * 1e-9);
  const out = new Float64Array(NZ - 1);
  for (let k = 0; k < NZ - 1; k++) {
    let dimless = 0;
    for (let i = 0; i < NR; i++) {
      const a = i * NZ + k, b = i * NZ + k + 1;
      const dpsi = psi[b] - psi[a];
      const jp = -(cp[b] * bernoulli(-dpsi) - cp[a] * bernoulli(dpsi)) / dz;
      const jm = -(cm[b] * bernoulli(dpsi) - cm[a] * bernoulli(-dpsi)) / dz;
      dimless += (Dc * jp - Da * jm) * weights[i];
    }
    dimless *= 2 * Math.PI;
    out[k] = dimless * scale * 1e12;
  }
  return out;
}
const meanOf = (arr) => arr.reduce((s, v) => s + v, 0) / arr.length;
const minMax = (arr) => { let lo = Infinity, hi = -Infinity; for (const v of arr) { if (v < lo) lo = v; if (v > hi) hi = v; } return [lo, hi]; };

// -------------------------------------------------------------------- rendering
function drawHeatmap(id, field, vmin, vmax, lut) {
  const canvas = document.getElementById(id);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const vlo = vmin, span = (vmax - vmin) || 1;
  const off = document.createElement("canvas");
  off.width = NR; off.height = NZ;
  const octx = off.getContext("2d");
  const img = octx.createImageData(NR, NZ);
  for (let j = 0; j < NZ; j++) {
    for (let i = 0; i < NR; i++) {
      let t = (field[i * NZ + j] - vlo) / span;
      t = Math.max(0, Math.min(1, t));
      const c = Math.round(t * 255) * 3;
      const p = (j * NR + i) * 4;
      img.data[p] = lut[c]; img.data[p + 1] = lut[c + 1]; img.data[p + 2] = lut[c + 2]; img.data[p + 3] = 255;
    }
  }
  octx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "rgba(255,255,255,0.25)";
  ctx.strokeRect(0.5, 0.5, canvas.width - 1, canvas.height - 1);
}

function drawCurrent(pred, ref) {
  const canvas = document.getElementById("cv_I");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height, pad = 46;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#06090d"; ctx.fillRect(0, 0, W, H);
  const all = ref ? Array.from(pred).concat(Array.from(ref)) : Array.from(pred);
  let [lo, hi] = minMax(all);
  const padY = (hi - lo) * 0.12 || 1; lo -= padY; hi += padY;
  const X = (k) => pad + k / (NZ - 2) * (W - pad - 14);
  const Y = (v) => H - pad - (v - lo) / (hi - lo) * (H - 2 * pad);
  ctx.strokeStyle = "#30363d"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, 10); ctx.lineTo(pad, H - pad); ctx.lineTo(W - 14, H - pad); ctx.stroke();
  ctx.fillStyle = "#8b949e"; ctx.font = "12px sans-serif";
  ctx.save(); ctx.translate(14, H / 2); ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center"; ctx.fillText("I(z) [pA]", 0, 0); ctx.restore();
  ctx.textAlign = "center"; ctx.fillText("z/L", W / 2, H - 12);
  ctx.textAlign = "right";
  for (let g = 0; g <= 4; g++) {
    const v = lo + (hi - lo) * g / 4;
    ctx.fillText(v.toFixed(3), pad - 6, Y(v) + 4);
    ctx.strokeStyle = "#1c2430"; ctx.beginPath(); ctx.moveTo(pad, Y(v)); ctx.lineTo(W - 14, Y(v)); ctx.stroke();
  }
  const plot = (arr, color, dashed) => {
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash(dashed ? [6, 4] : []);
    ctx.beginPath();
    for (let k = 0; k < arr.length; k++) { const x = X(k), y = Y(arr[k]); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
    ctx.stroke(); ctx.setLineDash([]);
  };
  if (ref) plot(ref, "#e69f00", true);
  plot(pred, "#3fb6d0", false);
  ctx.textAlign = "left"; ctx.fillStyle = "#3fb6d0"; ctx.fillText("operator", pad + 8, 22);
  if (ref) { ctx.fillStyle = "#e69f00"; ctx.fillText("SG/FV reference", pad + 78, 22); }
}

// --------------------------------------------------------------- derived labels
function lambdaD_nm(c0) { return Math.sqrt(META.constants.A_LAMBDA_M2 / c0) * 1e9; }
function sigma_mC(mu) {
  const { EPSILON_0: e0, REL_PERMITTIVITY: er, V_T } = META.constants;
  return mu[2] * er * e0 * V_T / (mu[3] * 1e-9) * 1000;
}
function oodFlags(mu) {
  const R = META.ranges_in_distribution;
  const f = [];
  if (mu[0] < R.V_app_mV[0] || mu[0] > R.V_app_mV[1]) f.push("V_app OOD");
  if (mu[1] < R.C0[0] || mu[1] > R.C0[1]) f.push("C0 OOD");
  if (mu[2] < R.q_w[0] || mu[2] > R.q_w[1]) f.push("q_w OOD");
  if (mu[3] < R.R_nm[0] || mu[3] > R.R_nm[1]) f.push("R OOD");
  return f;
}

// ------------------------------------------------------------------- inference
function b64ToBytes(b64) {
  const bin = atob(b64), len = bin.length, out = new Uint8Array(len);
  for (let i = 0; i < len; i++) out[i] = bin.charCodeAt(i);
  return out;
}

async function initEngine() {
  // Try ONNX Runtime Web first; fall back to the embedded JS engine.
  try {
    if (typeof ort === "undefined") throw new Error("onnxruntime-web unavailable (offline)");
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    ort.env.wasm.numThreads = 1;
    for (const k of ["deeponet", "fno"]) {
      const bytes = b64ToBytes(MODEL_BUNDLE.onnx[k]);
      SESSIONS[k] = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
    }
    ENGINE = "onnx";
    return ENGINE;
  } catch (e) {
    ENGINE = "js";
    return ENGINE;
  }
}

async function runModel(modelKey, mu) {
  if (ENGINE === "onnx") {
    const sess = SESSIONS[modelKey];
    const inName = sess.inputNames[0];
    const tensor = new ort.Tensor("float32", Float32Array.from(mu), [1, 4]);
    const t0 = performance.now();
    const y = await sess.run({ [inName]: tensor });
    const ms = performance.now() - t0;
    const f = (n) => Array.from(y[n].data);
    return { psi: f("psi"), c_plus: f("c_plus"), c_minus: f("c_minus"), rho: f("rho"), ms };
  }
  const t0 = performance.now();
  const y = modelKey === "deeponet" ? JSEngine.runDeepONetJS(mu) : JSEngine.runFNOJS(mu);
  const ms = performance.now() - t0;
  return { psi: Array.from(y.psi), c_plus: Array.from(y.c_plus), c_minus: Array.from(y.c_minus), rho: Array.from(y.rho), ms };
}

// ------------------------------------------------------------------ UI wiring
function readSliders() {
  return [
    parseFloat(document.getElementById("sl_V").value),
    parseFloat(document.getElementById("sl_C0").value),
    parseFloat(document.getElementById("sl_q").value),
    parseFloat(document.getElementById("sl_R").value),
  ];
}
function writeSliders(mu) {
  document.getElementById("sl_V").value = mu[0];
  document.getElementById("sl_C0").value = mu[1];
  document.getElementById("sl_q").value = mu[2];
  document.getElementById("sl_R").value = mu[3];
}
function updateDerived(mu) {
  document.getElementById("out_V").textContent = mu[0].toFixed(1) + " mV";
  document.getElementById("out_C0").textContent = mu[1].toFixed(0) + " mol/m\u00B3";
  document.getElementById("out_q").textContent = mu[2].toFixed(4);
  document.getElementById("out_R").textContent = mu[3].toFixed(1) + " nm";
  const lam = lambdaD_nm(mu[1]);
  document.getElementById("out_lam").textContent = lam.toFixed(3) + " nm";
  document.getElementById("out_lamR").textContent = (lam / mu[3]).toFixed(3);
  document.getElementById("out_sigma").textContent = sigma_mC(mu).toFixed(1) + " mC/m\u00B2";
  const flags = oodFlags(mu);
  const box = document.getElementById("oodFlags");
  box.innerHTML = flags.length
    ? '<span class="badge id">in-distribution</span>' + flags.map((f) => `<span class="badge ood">${f}</span>`).join("")
    : '<span class="badge id">in-distribution \u2713</span>';
}
function setBadge(ms) {
  const el = document.getElementById("engineBadge");
  const name = ENGINE === "onnx" ? "ONNX Runtime Web" : "Embedded JS";
  el.textContent = `Ready \u00B7 Engine: ${name} \u00B7 Inference: ${ms.toFixed(2)} ms`;
  el.className = "engine-badge" + (ENGINE === "js" ? " warn" : "");
}
function updateReadout(I, ms) {
  document.getElementById("out_I").textContent = I.toFixed(3) + " pA";
  document.getElementById("out_ms").textContent = ms.toFixed(2) + " ms";
  setBadge(ms);
}

let timer = null;
function schedule() { clearTimeout(timer); timer = setTimeout(evaluate, 60); }

async function evaluate() {
  const modelKey = document.getElementById("modelSelect").value;
  const mu = readSliders();
  CURRENT.mu = mu;
  updateDerived(mu);
  let res;
  try {
    res = await runModel(modelKey, mu);
  } catch (e) {
    setStatus("Inference error: " + e.message, "err"); return;
  }
  CURRENT.pred = res;
  const I = sgCurrentProfile(res.c_plus, res.c_minus, res.psi, mu);
  res.i_z = I;
  CURRENT.I = I;
  updateReadout(meanOf(I), res.ms);

  const compare = document.getElementById("compareToggle").checked && CURRENT.preset;
  const ref = CURRENT.preset;
  if (compare) {
    const err = (f, r) => f.map((v, i) => v - r[i]);
    const sym = (a) => { const m = Math.max.apply(null, a.map(Math.abs)) || 1; return [-m, m]; };
    let e = err(res.psi, ref.psi), ab = sym(e); drawHeatmap("cv_psi", e, ab[0], ab[1], LUT_ERR);
    e = err(res.c_plus, ref.c_plus); ab = sym(e); drawHeatmap("cv_cp", e, ab[0], ab[1], LUT_ERR);
    e = err(res.c_minus, ref.c_minus); ab = sym(e); drawHeatmap("cv_cm", e, ab[0], ab[1], LUT_ERR);
    e = err(res.rho, ref.rho); ab = sym(e); drawHeatmap("cv_rho", e, ab[0], ab[1], LUT_ERR);
    document.getElementById("refLabel").textContent = "SG/FV (preset)";
    document.getElementById("out_refI").textContent = ref.mean_current_pA.toFixed(3) + " pA";
    drawCurrent(I, ref.i_z);
  } else {
    const axes = { psi: minMax(res.psi), c_plus: minMax(res.c_plus), c_minus: minMax(res.c_minus), rho: minMax(res.rho) };
    drawHeatmap("cv_psi", res.psi, axes.psi[0], axes.psi[1], LUT_MAIN);
    drawHeatmap("cv_cp", res.c_plus, axes.c_plus[0], axes.c_plus[1], LUT_MAIN);
    drawHeatmap("cv_cm", res.c_minus, axes.c_minus[0], axes.c_minus[1], LUT_MAIN);
    drawHeatmap("cv_rho", res.rho, axes.rho[0], axes.rho[1], LUT_MAIN);
    if (ref) {
      document.getElementById("refLabel").textContent = "SG/FV (preset)";
      document.getElementById("out_refI").textContent = ref.mean_current_pA.toFixed(3) + " pA";
      drawCurrent(I, ref.i_z);
    } else {
      document.getElementById("refLabel").textContent = "SG/FV reference";
      document.getElementById("out_refI").textContent = "\u2014";
      drawCurrent(I, null);
    }
  }
}

function setStatus(text, cls) {
  const el = document.getElementById("status");
  el.textContent = text;
  el.className = "status" + (cls ? " " + cls : "");
}

// --------------------------------------------------------------- tab switching
function setTab(name) {
  const panels = document.querySelectorAll(".tabpanel");
  Array.prototype.forEach.call(panels, (p) => p.classList.toggle("active", p.id === name));
  const tabs = document.querySelectorAll(".tab");
  Array.prototype.forEach.call(tabs, (t) => {
    const on = t.dataset.tab === name;
    t.classList.toggle("active", on);
    t.setAttribute("aria-selected", on ? "true" : "false");
  });
  return name;
}

function initTabs() {
  const tabs = document.querySelectorAll(".tab");
  Array.prototype.forEach.call(tabs, (t) => {
    t.onclick = () => setTab(t.dataset.tab);
  });
  const links = document.querySelectorAll("a.nav-link[href^='#tab-']");
  Array.prototype.forEach.call(links, (a) => {
    a.onclick = (ev) => { if (ev && ev.preventDefault) ev.preventDefault(); setTab(a.getAttribute("href").slice(1)); };
  });
}

function buildPresets() {
  const box = document.getElementById("presets");
  box.innerHTML = "";
  META.benchmark_states.forEach((st) => {
    const b = document.createElement("button");
    b.textContent = `${st.label} \u2014 I=${st.mean_current_pA.toFixed(2)} pA`;
    b.dataset.id = st.id;
    b.onclick = () => {
      CURRENT.preset = st;
      writeSliders(st.mu);
      Array.prototype.forEach.call(box.children, (c) => c.classList.toggle("active", c === b));
      evaluate();
    };
    box.appendChild(b);
  });
}

// --------------------------------------------------------- live Python solver
async function runLiveSolver() {
  const btn = document.getElementById("liveSolveBtn");
  const out = document.getElementById("liveStatus");
  const mu = readSliders();
  btn.disabled = true; out.textContent = "Solving\u2026";
  try {
    const body = JSON.stringify({ V_app: mu[0], C0: mu[1], qw: mu[2], R: mu[3] });
    const t0 = performance.now();
    const resp = await fetch(LIVE_API, { method: "POST", headers: { "Content-Type": "application/json" }, body });
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const j = await resp.json();
    const ms = performance.now() - t0;
    out.innerHTML = `SG/FV mean I = <b>${j.mean_current_pA.toFixed(4)} pA</b> \u00B7 ` +
      `residual ${j.residual.toExponential(2)} \u00B7 ${j.iterations} iters \u00B7 ${j.runtime_s.toFixed(3)} s ` +
      `(round-trip ${ms.toFixed(0)} ms)`;
    // overlay the live SG/FV current
    if (CURRENT.I) drawCurrent(CURRENT.I, j.i_z);
  } catch (e) {
    out.innerHTML = `Live solver unavailable (${e.message}). Start <code>scripts/serve_interactive.py</code> to enable.`;
  } finally {
    btn.disabled = false;
  }
}

async function init() {
  try {
    if (!window.MODEL_BUNDLE) throw new Error("embedded_bundle.js not loaded");
    META = MODEL_BUNDLE.meta;
    setStatus("Initialising inference engine\u2026");
    const eng = await initEngine();
    setStatus(eng === "onnx"
      ? "Ready \u2014 ONNX Runtime Web inference active."
      : "Ready \u2014 running on the embedded JS engine (offline mode).",
      "ok");
    buildPresets();
    initTabs();
    document.getElementById("modelSelect").onchange = schedule;
    ["sl_V", "sl_C0", "sl_q", "sl_R"].forEach((id) => { document.getElementById(id).oninput = schedule; });
    document.getElementById("compareToggle").onchange = evaluate;
    document.getElementById("liveSolveBtn").onclick = runLiveSolver;
    await evaluate();
  } catch (e) {
    setStatus("Initialization failed: " + e.message, "err");
    const el = document.getElementById("engineBadge");
    if (el) { el.textContent = "Engine: failed"; el.className = "engine-badge err"; }
  }
}
if (typeof window !== "undefined") {
  // testability hook (used by scripts/test_web_simulator_dom.js)
  window.__app = {
    init, evaluate, runModel, sgCurrentProfile, buildPresets, runLiveSolver, setBadge,
    setTab, initTabs,
    getCurrent: () => CURRENT, getEngine: () => ENGINE, getMeta: () => META,
  };
  window.addEventListener("DOMContentLoaded", init);
}
