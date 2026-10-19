"use strict";
/*
 * Pure-JavaScript Float32 forward passes for the two exported operators.
 * Weights are read from window.MODEL_BUNDLE.weights (base64 float32 blobs
 * dumped from the PyTorch state_dicts).  This is the offline fallback engine:
 * it needs no network, no WASM and no onnxruntime — it runs under file://.
 *
 * Public API (attached to window.JSEngine):
 *   runDeepONetJS(mu) -> {psi,c_plus,c_minus,rho,s}   (Float32Array, length NR*NZ)
 *   runFNOJS(mu)      -> same
 */
(function (global) {
  const NR = 21, NZ = 61, N = NR * NZ;
  const V_T = 8.314462618 * 298.15 / 96485.33212;
  const A_LAMBDA_M2 = 8.8541878128e-12 * 78.5 * 8.314462618 * 298.15 / (2.0 * 96485.33212 * 96485.33212);
  const RADIAL_SCALES = [1.0, 3.0, 10.0, 30.0];

  let BN = null; // decoded tensors per model

  function b64ToBytes(b64) {
    const bin = atob(b64);
    const len = bin.length;
    const out = new Uint8Array(len);
    for (let i = 0; i < len; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  function buildTensorMap(w) {
    const bytes = b64ToBytes(w.blob_b64);
    const f32 = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
    const map = {};
    for (const e of w.layout) map[e.name] = { data: f32.subarray(e.offset, e.offset + e.length), shape: e.shape };
    return map;
  }

  function ensureInit() {
    if (BN) return BN;
    const W = global.MODEL_BUNDLE.weights;
    BN = {
      deeponet: { t: buildTensorMap(W.deeponet), mean: new Float32Array(W.deeponet.feat_mean), std: new Float32Array(W.deeponet.feat_std) },
      fno: { t: buildTensorMap(W.fno), mean: new Float32Array(W.fno.feat_mean), std: new Float32Array(W.fno.feat_std) },
    };
    return BN;
  }

  // ---- numeric helpers ----
  function erf(x) {
    const s = x < 0 ? -1 : 1;
    const ax = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * ax);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-ax * ax);
    return s * y;
  }
  const gelu = (x) => 0.5 * x * (1 + erf(x * 0.7071067811865476));
  const clamp = (x, lo, hi) => (x < lo ? lo : x > hi ? hi : x);

  // y = W(out,in) x + b(out)
  function linear(x, W, b, out) {
    const inN = W.shape[1], outN = W.shape[0], w = W.data;
    for (let o = 0; o < outN; o++) {
      let acc = b.data[o];
      const base = o * inN;
      for (let i = 0; i < inN; i++) acc += w[base + i] * x[i];
      out[o] = acc;
    }
    return out;
  }
  function linearGelu(x, W, b, out) { linear(x, W, b, out); for (let i = 0; i < out.length; i++) out[i] = gelu(out[i]); return out; }

  function features(mu, mean, std) {
    const V = mu[0], C0 = mu[1], qw = mu[2], Rn = mu[3];
    const psi_app = V * 1e-3 / V_T;
    const logC0 = Math.log10(C0);
    const alpha = Math.pow(30.0 / Rn, 2);
    const lam_nm = Math.sqrt(A_LAMBDA_M2 / C0) * 1e9;
    const beta = Math.pow(30.0 / lam_nm, 2) / 2.0;
    const lamr = lam_nm / Rn;
    const raw = [psi_app, logC0, qw, Rn, alpha, beta, lamr];
    const norm = new Float32Array(7);
    for (let i = 0; i < 7; i++) norm[i] = (raw[i] - mean[i]) / std[i];
    return { raw, psi_app, norm };
  }

  // exact-boundary positive ansatz (mode = reparam); raw3 is (3,N) channel-major
  function ansatz(raw3, psi_app) {
    const psi = new Float32Array(N), cp = new Float32Array(N), cm = new Float32Array(N),
          rho = new Float32Array(N), s = new Float32Array(N);
    const gpsi = 0, gs = N, grho = 2 * N;
    for (let j = 0; j < NZ; j++) {
      const z = j / (NZ - 1);
      const env = z * (1 - z);
      for (let i = 0; i < NR; i++) {
        const k = i * NZ + j;
        psi[k] = psi_app * (1 - z) + env * raw3[gpsi + k];
        const rv = env * raw3[grho + k];
        const sv = 2.0 * Math.exp(clamp(env * raw3[gs + k], -15, 15));
        const root = Math.sqrt(sv * sv + rv * rv);
        cp[k] = 0.5 * (root + rv);
        cm[k] = 0.5 * (root - rv);
        rho[k] = rv;
        s[k] = sv;
      }
    }
    return { psi, c_plus: cp, c_minus: cm, rho, s };
  }

  // ---------------------------------------------------------------- DeepONet
  function runDeepONetJS(mu) {
    const { t, mean, std } = ensureInit().deeponet;
    const f = features(mu, mean, std);
    const P = t["trunk.6.bias"].shape[0];        // p (64)
    const N_OUT = t["branch.6.bias"].shape[0] / P; // n_outputs (3)

    // branch: 7 ->64->64->64->(3*P)
    const b0 = new Float32Array(64), b1 = new Float32Array(64), b2 = new Float32Array(64);
    linearGelu(f.norm, t["branch.0.weight"], t["branch.0.bias"], b0);
    linearGelu(b0, t["branch.2.weight"], t["branch.2.bias"], b1);
    linearGelu(b1, t["branch.4.weight"], t["branch.4.bias"], b2);
    const branch = new Float32Array(N_OUT * P);
    linear(b2, t["branch.6.weight"], t["branch.6.bias"], branch); // (n_out*P)

    // trunk over grid
    const trunk = new Float32Array(N * P);
    const feats = new Float32Array(2 + RADIAL_SCALES.length);
    const h0 = new Float32Array(64), h1 = new Float32Array(64), h2 = new Float32Array(64);
    for (let i = 0; i < NR; i++) {
      const r = i / (NR - 1);
      for (let j = 0; j < NZ; j++) {
        const z = j / (NZ - 1);
        feats[0] = r; feats[1] = z;
        for (let sI = 0; sI < RADIAL_SCALES.length; sI++) feats[2 + sI] = Math.exp(-RADIAL_SCALES[sI] * (1 - r));
        linearGelu(feats, t["trunk.0.weight"], t["trunk.0.bias"], h0);
        linearGelu(h0, t["trunk.2.weight"], t["trunk.2.bias"], h1);
        linearGelu(h1, t["trunk.4.weight"], t["trunk.4.bias"], h2);
        const base = (i * NZ + j) * P;
        linear(h2, t["trunk.6.weight"], t["trunk.6.bias"], trunk.subarray(base, base + P));
      }
    }

    // out[o, idx] = sum_p branch[o*P+p]*trunk[idx*P+p] + bias[o]
    const raw3 = new Float32Array(N_OUT * N);
    const bias = t["bias"].data;
    for (let o = 0; o < N_OUT; o++) {
      const bo = o * P, do_ = o * N;
      for (let idx = 0; idx < N; idx++) {
        const tb = idx * P;
        let acc = bias[o];
        for (let p = 0; p < P; p++) acc += branch[bo + p] * trunk[tb + p];
        raw3[do_ + idx] = acc;
      }
    }
    return ansatz(raw3, f.psi_app);
  }

  // --------------------------------------------------------------------- FNO
  // small dense helpers (row-major): C(aRows,bCols) = A(aRows,aCols) B(aCols,bCols)
  function mm(A, aRows, aCols, B, bCols, C) {
    for (let i = 0; i < aRows; i++) {
      const ai = i * aCols, ci = i * bCols;
      for (let k = 0; k < bCols; k++) C[ci + k] = 0;
      for (let t = 0; t < aCols; t++) {
        const a = A[ai + t];
        if (a === 0) continue;
        const bt = t * bCols;
        for (let k = 0; k < bCols; k++) C[ci + k] += a * B[bt + k];
      }
    }
  }

  function buildFNOPlan(t) {
    // fixed spectral basis matrices from the checkpoint shapes
    const Kr = t["spectral.0.weight_real"].shape[2];
    const Kz = t["spectral.0.weight_real"].shape[3];
    const C = t["spectral.0.weight_real"].shape[1]; // out channels
    const Cin = t["spectral.0.weight_real"].shape[0];
    const Cr = new Float32Array(Kr * NR), Sr = new Float32Array(Kr * NR);
    for (let kr = 0; kr < Kr; kr++)
      for (let r = 0; r < NR; r++) {
        const ang = 2 * Math.PI * kr * r / NR;
        Cr[kr * NR + r] = Math.cos(ang); Sr[kr * NR + r] = Math.sin(ang);
      }
    const Cz = new Float32Array(Kz * NZ), Sz = new Float32Array(Kz * NZ);
    for (let kz = 0; kz < Kz; kz++)
      for (let z = 0; z < NZ; z++) {
        const ang = 2 * Math.PI * kz * z / NZ;
        Cz[kz * NZ + z] = Math.cos(ang); Sz[kz * NZ + z] = Math.sin(ang);
      }
    return { Kr, Kz, C, Cin, Cr, Sr, Cz, Sz };
  }

  function spectralForward(layer, t, plan, x, out) {
    const { Kr, Kz, C } = plan;
    const { Cr, Sr, Cz, Sz } = plan;
    const M = Kr * Kz;
    const Wr = t[`spectral.${layer}.weight_real`].data;
    const Wi = t[`spectral.${layer}.weight_imag`].data;
    // Re/Im per channel: store as [c*M + m] and [C*M + c*M + m]
    const Re = new Float32Array(C * M), Im = new Float32Array(C * M);
    const Ar = new Float32Array(Kr * NZ), Sx = new Float32Array(Kr * NZ);
    const tmpA = new Float32Array(Kr * Kz), tmpB = new Float32Array(Kr * Kz);
    const xc = new Float32Array(NR * NZ);
    for (let c = 0; c < C; c++) {
      const off = c * N;
      for (let k = 0; k < N; k++) xc[k] = x[off + k];
      mm(Cr, Kr, NR, xc, NZ, Ar);
      mm(Sr, Kr, NR, xc, NZ, Sx);
      // Re = Ar Cz^T - Sx Sz^T ; Im = -(Sx Cz^T + Ar Sz^T)
      for (let a = 0; a < Kr; a++) {
        const ar = a * NZ, sr = a * NZ;
        for (let b = 0; b < Kz; b++) {
          let re = 0, im = 0;
          for (let z = 0; z < NZ; z++) {
            const cb = Cz[b * NZ + z], sb = Sz[b * NZ + z];
            re += Ar[ar + z] * cb - Sx[sr + z] * sb;
            im += -(Sx[sr + z] * cb + Ar[ar + z] * sb);
          }
          const m = a * Kz + b;
          Re[c * M + m] = re; Im[c * M + m] = im;
        }
      }
    }
    // Yre[m,oc], Yim[m,oc]
    const Yre = new Float32Array(M * C), Yim = new Float32Array(M * C);
    for (let m = 0; m < M; m++) {
      for (let oc = 0; oc < C; oc++) {
        let yr = 0, yi = 0;
        for (let ic = 0; ic < C; ic++) {
          const widx = ((ic * C + oc) * Kr + Math.floor(m / Kz)) * Kz + (m % Kz);
          const wr = Wr[widx], wi = Wi[widx];
          const re = Re[ic * M + m], im = Im[ic * M + m];
          yr += re * wr - im * wi;
          yi += re * wi + im * wr;
        }
        Yre[m * C + oc] = yr; Yim[m * C + oc] = yi;
      }
    }
    // reconstruct per out channel
    const Y = new Float32Array(Kr * Kz), Yi = new Float32Array(Kr * Kz);
    const CrT = new Float32Array(NR * Kr), SrT = new Float32Array(NR * Kr);
    for (let r = 0; r < NR; r++) for (let a = 0; a < Kr; a++) { CrT[r * Kr + a] = Cr[a * NR + r]; SrT[r * Kr + a] = Sr[a * NR + r]; }
    const rr = new Float32Array(NR * NZ), ss = new Float32Array(NR * NZ), sc = new Float32Array(NR * NZ), cs = new Float32Array(NR * NZ);
    const t1 = new Float32Array(NR * Kz), t2 = new Float32Array(NR * Kz), t3 = new Float32Array(NR * Kz);
    for (let oc = 0; oc < C; oc++) {
      for (let m = 0; m < M; m++) { Y[m] = Yre[m * C + oc]; Yi[m] = Yim[m * C + oc]; }
      mm(CrT, NR, Kr, Y, Kz, t1);   mm(t1, NR, Kz, Cz, NZ, rr);
      mm(SrT, NR, Kr, Y, Kz, t2);   mm(t2, NR, Kz, Sz, NZ, ss);
      mm(SrT, NR, Kr, Yi, Kz, t3);  mm(t3, NR, Kz, Cz, NZ, sc);
      mm(CrT, NR, Kr, Yi, Kz, t1);  mm(t1, NR, Kz, Sz, NZ, cs);
      const off = oc * N, scale = 1.0 / (NR * NZ);
      for (let k = 0; k < N; k++) out[off + k] = (rr[k] - ss[k] - sc[k] - cs[k]) * scale;
    }
    return out;
  }

  function runFNOJS(mu) {
    const { t, mean, std } = ensureInit().fno;
    const f = features(mu, mean, std);
    const plan = buildFNOPlan(t);
    const C = plan.C; // width

    // lift input (2 coord + 7 features) -> width
    const liftW = t["lift.weight"], liftB = t["lift.bias"];
    const xin = new Float32Array(2 + 7);
    for (let i = 0; i < 7; i++) xin[2 + i] = f.norm[i];
    let h = new Float32Array(C * N);
    const hc = new Float32Array(C);
    const xc = new Float32Array(2 + 7);
    for (let i = 0; i < NR; i++) {
      xin[0] = 2 * (i / (NR - 1)) - 1;
      for (let j = 0; j < NZ; j++) {
        xin[1] = 2 * (j / (NZ - 1)) - 1;
        linear(xin, liftW, liftB, hc);
        for (let c = 0; c < C; c++) h[c * N + i * NZ + j] = hc[c];
      }
    }

    const P = new Float32Array(C * N), S = new Float32Array(C * N);
    const pw = new Float32Array(C), sc2 = new Float32Array(C), sh2 = new Float32Array(C);
    const film0 = new Float32Array(C), film1 = new Float32Array(2 * C);
    for (let L = 0; L < 4; L++) {
      spectralForward(L, t, plan, h, S);
      // pointwise conv + activation
      for (let i = 0; i < NR; i++) for (let j = 0; j < NZ; j++) {
        const idx = i * NZ + j;
        for (let c = 0; c < C; c++) pw[c] = h[c * N + idx];
        linear(pw, t[`pointwise.${L}.weight`], t[`pointwise.${L}.bias`], sc2);
        for (let c = 0; c < C; c++) h[c * N + idx] = gelu(S[c * N + idx] + sc2[c]);
      }
      // FiLM on normalized params
      linearGelu(f.norm, t[`film.${L}.net.0.weight`], t[`film.${L}.net.0.bias`], film0);
      linear(film0, t[`film.${L}.net.2.weight`], t[`film.${L}.net.2.bias`], film1);
      for (let c = 0; c < C; c++) { sc2[c] = film1[c]; sh2[c] = film1[C + c]; }
      for (let c = 0; c < C; c++) {
        const a = 1 + sc2[c], b = sh2[c], off = c * N;
        for (let k = 0; k < N; k++) h[off + k] = h[off + k] * a + b;
      }
    }

    // projection width -> 3
    const raw3 = new Float32Array(3 * N);
    const projW = t["proj.weight"], projB = t["proj.bias"];
    const pc = new Float32Array(C), po = new Float32Array(3);
    for (let i = 0; i < NR; i++) for (let j = 0; j < NZ; j++) {
      const idx = i * NZ + j;
      for (let c = 0; c < C; c++) pc[c] = h[c * N + idx];
      linear(pc, projW, projB, po);
      for (let o = 0; o < 3; o++) raw3[o * N + idx] = po[o];
    }
    return ansatz(raw3, f.psi_app);
  }

  global.JSEngine = { runDeepONetJS, runFNOJS, ensureInit };
})(typeof window !== "undefined" ? window : globalThis);
