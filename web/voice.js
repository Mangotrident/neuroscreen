/* Line-by-line JavaScript port of neuroscreen/voice_features.py.
   Input: Float64Array/Float32Array at 16 kHz. Output: same feature dictionary as Python.
   tests/parity.js checks the two implementations agree on real recordings. */
(function (root) {
  "use strict";
  const SR = 16000, FRAME = 640, HOP = 160, FMIN = 60.0, FMAX = 500.0;
  const VOICING_THRESHOLD = 0.45, MAX_PERIOD_FACTOR = 1.3, MAX_AMP_FACTOR = 1.6;
  const FEATURE_NAMES = [
    "jitter_local", "jitter_rap", "jitter_ppq5", "period_cv",
    "shimmer_local", "shimmer_db", "shimmer_apq3", "shimmer_apq5", "shimmer_apq11",
    "hnr_db", "cpp_db",
    "f0_sd_semitones", "f0_range_semitones", "f0_tremor_ratio",
    "amp_sd_db", "amp_tremor_ratio", "voiced_fraction",
  ];

  // ---------- small numeric helpers matching numpy behaviour ----------
  function roundHalfEven(v) {           // Python's round()
    const f = Math.floor(v), d = v - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return (f % 2 === 0) ? f : f + 1;
  }
  const mean = (a) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / a.length; };
  const std = (a) => { const m = mean(a); let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - m) ** 2; return Math.sqrt(s / a.length); };
  function median(a) { const b = Array.from(a).sort((x, y) => x - y); const n = b.length; return n % 2 ? b[(n - 1) / 2] : 0.5 * (b[n / 2 - 1] + b[n / 2]); }
  function percentile(a, q) {           // numpy default ("linear")
    const b = Array.from(a).sort((x, y) => x - y); const pos = (b.length - 1) * q / 100;
    const lo = Math.floor(pos), hi = Math.ceil(pos); return b[lo] + (b[hi] - b[lo]) * (pos - lo);
  }
  function argmax(a, lo, hi) { let k = lo, m = -Infinity; for (let i = lo; i < hi; i++) if (a[i] > m) { m = a[i]; k = i; } return k - lo; }
  function polyfit1(x, y) {             // least-squares line, returns [slope, intercept]
    const n = x.length; let sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (let i = 0; i < n; i++) { sx += x[i]; sy += y[i]; sxx += x[i] * x[i]; sxy += x[i] * y[i]; }
    const slope = (n * sxy - sx * sy) / (n * sxx - sx * sx); return [slope, (sy - slope * sx) / n];
  }
  const hann = (n) => { const w = new Float64Array(n); for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / n); return w; };
  const hamming = (n) => { const w = new Float64Array(n); for (let i = 0; i < n; i++) w[i] = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (n - 1)); return w; };

  // in-place radix-2 complex FFT (n must be a power of two)
  function fft(re, im, inverse) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit;
      if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = 2 * Math.PI / len * (inverse ? 1 : -1), wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let k = 0; k < len / 2; k++) {
          const a = i + k, b = a + len / 2;
          const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
          re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
          const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
        }
      }
    }
    if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
  }
  // autocorrelation (lags 0..maxLag-1) of a real segment via zero-padded FFT
  function autocorr(seg, nfft, maxLag) {
    const re = new Float64Array(nfft), im = new Float64Array(nfft);
    re.set(seg); fft(re, im, false);
    for (let i = 0; i < nfft; i++) { re[i] = re[i] * re[i] + im[i] * im[i]; im[i] = 0; }
    fft(re, im, true); return re.slice(0, maxLag);
  }

  function frameRmsDb(x) {
    const n = 1 + Math.max(0, Math.floor((x.length - FRAME) / HOP)); const out = new Float64Array(n);
    for (let i = 0; i < n; i++) { let s = 0; const a = i * HOP, b = Math.min(a + FRAME, x.length); for (let j = a; j < b; j++) s += x[j] * x[j]; out[i] = 10 * Math.log10(s / (b - a) + 1e-12); }
    return out;
  }

  function selectRegion(x) {
    const db = frameRmsDb(x); let mx = -Infinity; for (const v of db) mx = Math.max(mx, v);
    let bestLen = 0, bestStart = 0, runStart = null;
    for (let i = 0; i <= db.length; i++) {
      const v = i < db.length && db[i] > mx - 25.0;
      if (v && runStart === null) runStart = i;
      else if (!v && runStart !== null) { if (i - runStart > bestLen) { bestLen = i - runStart; bestStart = runStart; } runStart = null; }
    }
    let a = bestStart * HOP, b = (bestStart + bestLen - 1) * HOP + FRAME;
    const trim = Math.trunc(0.15 * (b - a)); a += trim; b -= trim;
    const maxLen = 4 * SR;
    if (b - a > maxLen) { const c = Math.floor((a + b) / 2); a = c - maxLen / 2; b = c + maxLen / 2; }
    return x.slice(a, b);
  }

  function autocorrPitch(x) {
    const win = hann(FRAME), nfft = 2048;
    const rw0 = autocorr(win, nfft, FRAME); const rw = rw0.map((v) => v / rw0[0]);
    const lagMin = Math.floor(SR / FMAX), lagMax = Math.ceil(SR / FMIN);
    const n = 1 + Math.max(0, Math.floor((x.length - FRAME) / HOP));
    const f0 = new Float64Array(n), rbest = new Float64Array(n);
    const seg = new Float64Array(FRAME);
    for (let i = 0; i < n; i++) {
      let m = 0; for (let j = 0; j < FRAME; j++) m += x[i * HOP + j]; m /= FRAME;
      for (let j = 0; j < FRAME; j++) seg[j] = (x[i * HOP + j] - m) * win[j];
      let r = autocorr(seg, nfft, FRAME);
      if (r[0] <= 1e-12) continue;
      const r0 = r[0]; for (let j = 0; j < FRAME; j++) r[j] = (r[j] / r0) / Math.max(rw[j], 1e-3);
      const segR = r.slice(lagMin, lagMax + 1);
      const cands = []; for (let k = 1; k < segR.length - 1; k++) if (segR[k] > segR[k - 1] && segR[k] >= segR[k + 1]) cands.push(k);
      if (!cands.length) continue;
      let best = cands[0]; for (const k of cands) if (segR[k] > segR[best]) best = k;
      for (const k of cands) if (segR[k] >= 0.85 * segR[best]) { best = k; break; }
      const rv = segR[best]; if (rv < VOICING_THRESHOLD) continue;
      const y0 = segR[best - 1], y1 = segR[best], y2 = segR[best + 1], denom = y0 - 2 * y1 + y2;
      const delta = Math.abs(denom) > 1e-12 ? 0.5 * (y0 - y2) / denom : 0.0;
      f0[i] = SR / (lagMin + best + delta); rbest[i] = Math.min(Math.max(rv, 1e-4), 0.9999);
    }
    const sm = Float64Array.from(f0);
    for (let i = 0; i < n; i++) if (f0[i] > 0) {
      const w = []; for (let j = Math.max(0, i - 2); j < Math.min(n, i + 3); j++) if (f0[j] > 0) w.push(f0[j]);
      sm[i] = median(w);
    }
    return [sm, rbest];
  }

  function cycleMarks(x, f0) {
    const voiced = Array.from(f0, (v) => v > 0); const nv = voiced.filter(Boolean).length;
    if (nv < 3) return [[], []];
    let mx = -Infinity, mn = Infinity; for (const v of x) { if (v > mx) mx = v; if (v < mn) mn = v; }
    const sig = mx >= -mn ? x : x.map((v) => -v);
    const vi = []; voiced.forEach((v, i) => { if (v) vi.push(i); });
    const localPeriod = (t) => {
      let i = roundHalfEven((t * SR - FRAME / 2) / HOP); i = Math.min(Math.max(i, 0), f0.length - 1);
      if (f0[i] > 0) return 1.0 / f0[i];
      let j = vi[0], bd = Infinity; for (const k of vi) { const d = Math.abs(k - i); if (d < bd) { bd = d; j = k; } }
      return 1.0 / f0[j];
    };
    const firstV = voiced.indexOf(true);
    const tStart = (firstV * HOP + FRAME / 2) / SR;
    const t0 = localPeriod(tStart);
    const a = Math.trunc(tStart * SR), b = Math.min(sig.length, a + Math.trunc(t0 * SR) + 1);
    if (b - a < 3) return [[], []];
    let pos = a + argmax(sig, a, b); const marks = [pos];
    while (true) {
      const k = roundHalfEven(pos);
      const T = roundHalfEven(localPeriod(k / SR) * SR);
      const lo = Math.trunc(0.8 * T), hi = Math.trunc(1.25 * T) + 1;
      if (k + hi + T + 1 >= sig.length || T < 4) break;
      let rm = 0; for (let j = 0; j < T; j++) rm += sig[k + j]; rm /= T;
      let nref = 0; for (let j = 0; j < T; j++) nref += (sig[k + j] - rm) ** 2; nref = Math.sqrt(nref) + 1e-12;
      const cc = new Float64Array(hi - lo + 1);
      for (let L = lo; L <= hi; L++) {
        let sm = 0; for (let j = 0; j < T; j++) sm += sig[k + L + j]; sm /= T;
        let dot = 0, ss = 0;
        for (let j = 0; j < T; j++) { const s = sig[k + L + j] - sm; dot += (sig[k + j] - rm) * s; ss += s * s; }
        cc[L - lo] = dot / (nref * (Math.sqrt(ss) + 1e-12));
      }
      const j = argmax(cc, 0, cc.length); let delta = 0.0;
      if (j > 0 && j < cc.length - 1) { const y0 = cc[j - 1], y1 = cc[j], y2 = cc[j + 1], den = y0 - 2 * y1 + y2; delta = Math.abs(den) > 1e-12 ? 0.5 * (y0 - y2) / den : 0.0; }
      pos = pos + (lo + j + delta); marks.push(pos);
    }
    const times = marks.map((m) => m / SR), amps = [];
    for (let j = 0; j < marks.length - 1; j++) {
      const s = roundHalfEven(marks[j]), e = roundHalfEven(marks[j + 1]);
      let hiV = -Infinity, loV = Infinity; for (let q = s; q < e; q++) { if (x[q] > hiV) hiV = x[q]; if (x[q] < loV) loV = x[q]; }
      amps.push(e > s ? hiV - loV : 0.0);
    }
    return [times, amps];
  }

  function jitterShimmer(times, amps) {
    const T = []; for (let i = 1; i < times.length; i++) T.push(times[i] - times[i - 1]);
    const ok = T.map((t) => t > 1.0 / FMAX * 0.9 && t < 1.0 / FMIN * 1.1);
    const out = {};
    const okAll = (lo, hi) => { for (let q = lo; q < hi; q++) if (!ok[q]) return false; return true; };
    const ratio = (w) => Math.max(...w) / Math.min(...w);
    const diffs = [], rap = [], ppq = [];
    for (let i = 1; i < T.length; i++) if (ok[i] && ok[i - 1] && Math.max(T[i], T[i - 1]) / Math.min(T[i], T[i - 1]) <= MAX_PERIOD_FACTOR) diffs.push(Math.abs(T[i] - T[i - 1]));
    for (let i = 1; i < T.length - 1; i++) { const w = T.slice(i - 1, i + 2); if (okAll(i - 1, i + 2) && ratio(w) <= MAX_PERIOD_FACTOR) rap.push(Math.abs(T[i] - mean(w))); }
    for (let i = 2; i < T.length - 2; i++) { const w = T.slice(i - 2, i + 3); if (okAll(i - 2, i + 3) && ratio(w) <= MAX_PERIOD_FACTOR) ppq.push(Math.abs(T[i] - mean(w))); }
    const Tok = T.filter((_, i) => ok[i]); const meanT = Tok.length ? mean(Tok) : NaN;
    out.jitter_local = diffs.length ? mean(diffs) / meanT : NaN;
    out.jitter_rap = rap.length ? mean(rap) / meanT : NaN;
    out.jitter_ppq5 = ppq.length ? mean(ppq) / meanT : NaN;
    out.period_cv = Tok.length > 2 ? std(Tok) / meanT : NaN;
    const A = amps, okA = A.map((v) => v > 1e-6);
    const sd = [], sdb = [];
    for (let i = 1; i < A.length; i++) if (okA[i] && okA[i - 1] && Math.max(A[i], A[i - 1]) / Math.min(A[i], A[i - 1]) <= MAX_AMP_FACTOR) { sd.push(Math.abs(A[i] - A[i - 1])); sdb.push(Math.abs(20 * Math.log10(A[i] / A[i - 1]))); }
    const Aok = A.filter((_, i) => okA[i]); const meanA = Aok.length ? mean(Aok) : NaN;
    out.shimmer_local = sd.length ? mean(sd) / meanA : NaN;
    out.shimmer_db = sdb.length ? mean(sdb) : NaN;
    for (const [name, half] of [["shimmer_apq3", 1], ["shimmer_apq5", 2], ["shimmer_apq11", 5]]) {
      const vals = [];
      for (let i = half; i < A.length - half; i++) {
        const w = A.slice(i - half, i + half + 1); let allOk = true; for (let q = i - half; q <= i + half; q++) if (!okA[q]) allOk = false;
        if (allOk && ratio(w) <= MAX_AMP_FACTOR) vals.push(Math.abs(A[i] - mean(w)));
      }
      out[name] = vals.length ? mean(vals) / meanA : NaN;
    }
    return out;
  }

  function cpp(x, f0) {
    const win = hamming(FRAME), nfft = 1024;
    const q = Array.from({ length: nfft }, (_, i) => i / SR);
    const qmin = 1.0 / FMAX, qmax = 1.0 / FMIN;
    const band = [], regX = [], regI = [];
    for (let i = 0; i < nfft; i++) { if (q[i] >= qmin && q[i] <= qmax) band.push(i); if (q[i] >= 0.001 && q[i] <= qmax) { regX.push(q[i]); regI.push(i); } }
    const vals = [];
    for (let i = 0; i < f0.length; i++) {
      if (!(f0[i] > 0)) continue;
      if (i * HOP + FRAME > x.length) continue;
      const re = new Float64Array(nfft), im = new Float64Array(nfft);
      for (let j = 0; j < FRAME; j++) re[j] = x[i * HOP + j] * win[j];
      fft(re, im, false);
      for (let j = 0; j < nfft; j++) { re[j] = 10 * Math.log10(re[j] * re[j] + im[j] * im[j] + 1e-12); im[j] = 0; }
      fft(re, im, true);
      const cdb = re.map((c) => 10 * Math.log10(c * c + 1e-12));
      let k = band[0]; for (const b of band) if (cdb[b] > cdb[k]) k = b;
      const [slope, icpt] = polyfit1(regX, regI.map((j) => cdb[j]));
      vals.push(cdb[k] - (slope * q[k] + icpt));
    }
    return vals.length ? mean(vals) : NaN;
  }

  function tremorRatio(contour) {
    const n = contour.length; if (n < 64) return NaN;
    const m = mean(contour); let c = contour.map((v) => v - m);
    if (c.every((v) => Math.abs(v) < 1e-12)) return NaN;
    const t = c.map((_, i) => i); const [s, b] = polyfit1(t, c); c = c.map((v, i) => v - (s * i + b));
    const w = hann(n); const dt = HOP / SR; let total = 0, trem = 0;
    for (let k = 0; k <= Math.floor(n / 2); k++) {
      const f = k / (n * dt); if (f < 1 || f > 20) continue;
      let re = 0, im = 0; for (let j = 0; j < n; j++) { const v = c[j] * w[j], a = -2 * Math.PI * k * j / n; re += v * Math.cos(a); im += v * Math.sin(a); }
      const P = re * re + im * im; total += P; if (f >= 3 && f <= 10) trem += P;
    }
    return total > 0 ? trem / total : NaN;
  }

  function longestRun(values, mask, semitones) {
    let best = [0, 0], cur = null;
    for (let i = 0; i <= mask.length; i++) {
      const m = i < mask.length && mask[i];
      if (m && cur === null) cur = i;
      else if (!m && cur !== null) { if (i - cur > best[1] - best[0]) best = [cur, i]; cur = null; }
    }
    let v = Array.from(values.slice(best[0], best[1]));
    if (semitones && v.length) { const md = median(v); v = v.map((u) => 12 * Math.log2(u / md)); }
    return v;
  }

  // x: samples at 16 kHz. Also returns intermediate signals for plotting.
  function extract(x16) {
    let x = Float64Array.from(x16); const m = mean(x); x = x.map((v) => v - m);
    let pk = 0; for (const v of x) pk = Math.max(pk, Math.abs(v)); if (pk > 0) x = x.map((v) => v / pk);
    x = selectRegion(x);
    const [f0, r] = autocorrPitch(x);
    const voiced = Array.from(f0, (v) => v > 0);
    const feats = {}; for (const k of FEATURE_NAMES) feats[k] = NaN;
    const nv = voiced.filter(Boolean).length;
    feats.voiced_fraction = voiced.length ? nv / voiced.length : 0;
    const debug = { region: x, f0, times: [], amps: [] };
    if (nv < 20) return { feats, debug };
    const [times, amps] = cycleMarks(x, f0); debug.times = times; debug.amps = amps;
    if (times.length > 12) Object.assign(feats, jitterShimmer(times, amps));
    const rv = []; voiced.forEach((v, i) => { if (v) rv.push(r[i]); });
    feats.hnr_db = mean(rv.map((v) => 10 * Math.log10(v / (1 - v))));
    feats.cpp_db = cpp(x, f0);
    const fv = []; voiced.forEach((v, i) => { if (v) fv.push(f0[i]); });
    const md = median(fv); const st = fv.map((v) => 12 * Math.log2(v / md));
    feats.f0_sd_semitones = std(st);
    feats.f0_range_semitones = percentile(st, 95) - percentile(st, 5);
    feats.f0_tremor_ratio = tremorRatio(longestRun(f0, voiced, true));
    const db = frameRmsDb(x).slice(0, f0.length);
    const dbv = []; voiced.forEach((v, i) => { if (v) dbv.push(db[i]); });
    feats.amp_sd_db = std(dbv);
    feats.amp_tremor_ratio = tremorRatio(longestRun(db, voiced, false));
    return { feats, debug };
  }

  // logistic model exported by scripts/export_model.py
  function predict(model, feats) {
    const z = model.features.map((f, i) => {
      let v = feats[f];
      if (model.log_features.includes(f) && !Number.isNaN(v)) v = Math.log(Math.max(v, 1e-6));
      if (v === undefined || Number.isNaN(v)) v = model.impute_median[i];
      return (v - model.scaler_mean[i]) / model.scaler_scale[i];
    });
    let s = model.intercept; z.forEach((v, i) => { s += v * model.coef[i]; });
    return { prob: 1 / (1 + Math.exp(-s)), contributions: Object.fromEntries(model.features.map((f, i) => [f, z[i] * model.coef[i]])) };
  }

  // Synthetic sustained /a/: a glottal pulse train with chosen cycle-to-cycle period and amplitude
  // variation, shaped by three formant resonators, plus breath noise. Used to demo and to check that
  // the extractor recovers what was injected.
  function simulate({ f0 = 125, jitter = 0.004, shimmer = 0.03, noiseDb = -35, seconds = 3, seed = 11 } = {}) {
    let s = seed >>> 0;
    const rnd = () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
    const n = Math.round(seconds * SR), src = new Float64Array(n);
    // E|g_i - g_{i-1}| = 2/sqrt(pi) for unit gaussians, so scale to hit the requested local jitter/shimmer
    const kj = jitter / 1.1284, ks = shimmer / 1.1284;
    let t = 0.01 * SR;
    while (t < n - 2) {
      const T = (SR / f0) * (1 + kj * gauss());
      const A = 1 + ks * gauss();
      // Rosenberg-like pulse: smooth opening, sharp closure
      const L = Math.round(0.6 * T);
      for (let k = 0; k < L && t + k < n; k++) {
        const ph = k / L, idx = Math.floor(t + k);
        src[idx] += A * (ph < 0.66 ? 0.5 * (1 - Math.cos(Math.PI * ph / 0.66)) : Math.cos(Math.PI * (ph - 0.66) / (2 * 0.34)));
      }
      t += T;
    }
    // differentiate (lip radiation) then cascade formant resonators
    let y = new Float64Array(n); for (let i = 1; i < n; i++) y[i] = src[i] - src[i - 1];
    for (const [F, B] of [[730, 90], [1090, 110], [2440, 160]]) {
      const r = Math.exp(-Math.PI * B / SR), c = 2 * r * Math.cos(2 * Math.PI * F / SR), d = -r * r, g = 1 - c - d;
      const o = new Float64Array(n);
      for (let i = 0; i < n; i++) o[i] = g * y[i] + c * (i > 0 ? o[i - 1] : 0) + d * (i > 1 ? o[i - 2] : 0);
      y = o;
    }
    let rms = 0; for (const v of y) rms += v * v; rms = Math.sqrt(rms / n);
    const nAmp = rms * Math.pow(10, noiseDb / 20);
    for (let i = 0; i < n; i++) y[i] += nAmp * gauss();
    // fade in/out like a real phonation
    const f = Math.round(0.15 * SR); for (let i = 0; i < f; i++) { y[i] *= i / f; y[n - 1 - i] *= i / f; }
    return y;
  }

  const api = { SR, FEATURE_NAMES, extract, predict, simulate };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.NeuroVoice = api;
})(typeof self !== "undefined" ? self : this);
