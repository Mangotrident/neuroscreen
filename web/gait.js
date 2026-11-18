/* Walking check: gait measures from MediaPipe Pose landmarks (33 points per frame) while a person
   walks across the camera's view (side-on, 4-6 metres).
   Input: frames = [{t: seconds, lm: [[x, y, z, visibility], ...33] in pixels}] (no person -> lm: null),
          heightCm = the person's height (stored once in their profile).
   Scale comes from leg length: hip-to-ankle is about 0.49 of standing height in adults (hip joint at
   ~0.53 H, ankle at ~0.04 H), so pixels convert to metres without any calibration object.
   Measures: walking speed (the "sixth vital sign"), cadence, step length, stride-time variability,
   arm-swing amplitude per side and its asymmetry, and forward trunk lean. */
(function (root) {
  "use strict";
  const L = { sh: [11, 12], wr: [15, 16], hip: [23, 24], ank: [27, 28] };
  const LEG_FRACTION = 0.49;
  const d2 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const median = (a) => { const b = [...a].sort((x, y) => x - y); const n = b.length; return n ? (n % 2 ? b[(n - 1) / 2] : (b[n / 2 - 1] + b[n / 2]) / 2) : NaN; };
  const pct = (a, q) => { const b = [...a].sort((x, y) => x - y); if (!b.length) return NaN; const p = (b.length - 1) * q / 100, lo = Math.floor(p), hi = Math.ceil(p); return b[lo] + (b[hi] - b[lo]) * (p - lo); };
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  const sd = (a) => { const m = mean(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length); };
  const smooth = (a, w) => a.map((_, i) => { let s = 0, c = 0; for (let j = Math.max(0, i - w); j <= Math.min(a.length - 1, i + w); j++) { s += a[j]; c++; } return s / c; });
  const mid = (lm, [a, b]) => [(lm[a][0] + lm[b][0]) / 2, (lm[a][1] + lm[b][1]) / 2];

  function linfit(x, y) {
    const n = x.length, mx = mean(x), my = mean(y); let sxy = 0, sxx = 0;
    for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; }
    const b = sxy / sxx; return [b, my - b * mx];
  }

  // local extrema with a minimum prominence, alternating max/min
  function extrema(sig, minProm) {
    const out = [];
    for (let i = 1; i < sig.length - 1; i++) {
      const isMax = sig[i] >= sig[i - 1] && sig[i] > sig[i + 1], isMin = sig[i] <= sig[i - 1] && sig[i] < sig[i + 1];
      if (!isMax && !isMin) continue;
      const type = isMax ? 1 : -1;
      if (out.length && out[out.length - 1].type === type) {
        const last = out[out.length - 1];
        if ((type === 1 && sig[i] > sig[last.i]) || (type === -1 && sig[i] < sig[last.i])) out[out.length - 1] = { i, type };
        continue;
      }
      if (out.length && Math.abs(sig[i] - sig[out[out.length - 1].i]) < minProm) continue;
      out.push({ i, type });
    }
    return out;
  }

  function analyse(frames, heightCm) {
    const vis = (lm, ids) => Math.min(...ids.map((k) => (lm[k][3] ?? 1)));
    const ok = frames.filter((f) => f.lm && vis(f.lm, [23, 24, 27, 28]) > 0.5);
    const quality = { frames: frames.length, person_found_pct: frames.length ? (100 * ok.length) / frames.length : 0 };
    if (!heightCm || heightCm < 100 || heightCm > 230) return { error: "Enter the person's height (in cm) so distances can be measured.", quality };
    if (ok.length < 20) return { error: "Couldn't follow a whole person walking. Walk across the camera's view, side-on, with your whole body in frame.", quality };
    const t = ok.map((f) => f.t);
    const legPx = pct(ok.map((f) => (d2(f.lm[23], f.lm[27]) + d2(f.lm[24], f.lm[28])) / 2), 90);
    const mPerPx = (LEG_FRACTION * heightCm / 100) / legPx;
    const pelvisX = ok.map((f) => mid(f.lm, L.hip)[0] * mPerPx);
    const dir = Math.sign(pelvisX[pelvisX.length - 1] - pelvisX[0]) || 1;
    // steady middle of the walk: drop the first and last 15% of frames (starting and stopping)
    const a = Math.floor(0.15 * ok.length), b = Math.ceil(0.85 * ok.length);
    const tm = t.slice(a, b), xm = pelvisX.slice(a, b);
    const [slope] = linfit(tm, xm);
    const speed = Math.abs(slope);
    const travelled = Math.abs(pelvisX[pelvisX.length - 1] - pelvisX[0]);
    quality.duration_s = t[t.length - 1] - t[0]; quality.distance_m = travelled;
    if (travelled < 1.5 || speed < 0.2) return { error: `Only ${travelled.toFixed(1)} m of sideways walking was seen. Walk across the camera's view, side-on, for at least 3 metres.`, quality };

    // steps: the signed forward distance between the ankles peaks at each heel strike
    const sep = smooth(ok.map((f) => dir * (f.lm[27][0] - f.lm[28][0]) * mPerPx), 1);
    const amp = pct(sep, 95) - pct(sep, 5);
    const ex = extrema(sep, 0.3 * amp);
    // refine each step time between frames with a parabola through the peak and its neighbours
    const stepT = ex.map((e) => {
      const i = e.i; if (i < 1 || i >= sep.length - 1) return t[i];
      const y0 = sep[i - 1], y1 = sep[i], y2 = sep[i + 1], den = y0 - 2 * y1 + y2;
      const dlt = Math.abs(den) > 1e-12 ? Math.max(-0.5, Math.min(0.5, 0.5 * (y0 - y2) / den)) : 0;
      return t[i] + dlt * (t[i + 1] - t[i - 1]) / 2;
    });
    const metrics = { speed_m_s: speed, cadence_steps_min: NaN, step_length_m: NaN, stride_time_cv_pct: NaN, arm_swing_left: NaN, arm_swing_right: NaN, arm_swing_asym_pct: NaN, trunk_lean_deg: NaN };
    if (stepT.length >= 4) {
      const stepMean = (stepT[stepT.length - 1] - stepT[0]) / (stepT.length - 1);
      metrics.cadence_steps_min = 60 / stepMean;
      metrics.step_length_m = speed * stepMean;
      const strides = []; for (let i = 2; i < stepT.length; i++) strides.push(stepT[i] - stepT[i - 2]);
      if (strides.length >= 3) metrics.stride_time_cv_pct = (100 * sd(strides)) / mean(strides);
    }
    quality.steps_found = stepT.length;

    // arm swing: wrist movement forward/back relative to its shoulder, in metres (peak to peak)
    const swing = (side) => {
      const v = ok.filter((f) => vis(f.lm, [L.wr[side], L.sh[side]]) > 0.3).map((f) => dir * (f.lm[L.wr[side]][0] - f.lm[L.sh[side]][0]) * mPerPx);
      return v.length > 10 ? pct(v, 95) - pct(v, 5) : NaN;
    };
    metrics.arm_swing_left = swing(0); metrics.arm_swing_right = swing(1);
    if (Number.isFinite(metrics.arm_swing_left) && Number.isFinite(metrics.arm_swing_right)) {
      const hi = Math.max(metrics.arm_swing_left, metrics.arm_swing_right);
      metrics.arm_swing_asym_pct = hi > 0 ? (100 * Math.abs(metrics.arm_swing_left - metrics.arm_swing_right)) / hi : NaN;
    }
    // trunk lean: angle of hips->shoulders from vertical, positive = leaning in the walking direction
    metrics.trunk_lean_deg = median(ok.map((f) => {
      const h = mid(f.lm, L.hip), s = mid(f.lm, L.sh);
      return (Math.atan2(dir * (s[0] - h[0]), h[1] - s[1]) * 180) / Math.PI;
    }));
    quality.far_side_visibility = median(ok.map((f) => Math.min(f.lm[15][3] ?? 1, f.lm[16][3] ?? 1)));
    return { metrics, quality, series: { t, pelvisX, sep } };
  }

  // synthetic walker for tests and the demo (side view, pixels; image y points down)
  function simulate({ heightCm = 170, speed = 1.2, cadence = 110, armL = 0.30, armR = 0.30, lean = 4, strideJitter = 0.02, seconds = 6, fps = 30, seed = 3 } = {}) {
    let s = seed >>> 0; const rnd = () => { s = (s + 0x6d2b79f5) >>> 0; let x = s; x = Math.imul(x ^ (x >>> 15), x | 1); x ^= x + Math.imul(x ^ (x >>> 7), x | 61); return ((x ^ (x >>> 14)) >>> 0) / 4294967296; };
    const pxPerM = 300, H = heightCm / 100, stepT = 60 / cadence, stepLen = speed * stepT;
    // stride durations with the requested variability (uniform noise scaled so its SD = strideJitter)
    const strides = [], bounds = [0];
    while (bounds[bounds.length - 1] < seconds + 2) { const d = 2 * stepT * (1 + strideJitter * Math.sqrt(3) * (rnd() * 2 - 1)); strides.push(d); bounds.push(bounds[bounds.length - 1] + d); }
    const phaseAt = (t) => { let c = 0; while (bounds[c + 1] <= t) c++; return 2 * Math.PI * (c + (t - bounds[c]) / strides[c]); };
    const frames = [];
    for (let k = 0; k < seconds * fps; k++) {
      const t = k / fps, phase = phaseAt(t);
      const px = 100 + speed * t * pxPerM, hipY = 600 - 0.53 * H * pxPerM, ankY = 600 - 0.04 * H * pxPerM;
      const lm = Array.from({ length: 33 }, () => [px, hipY, 0, 0.9]);
      const legDx = (stepLen / 2) * Math.sin(phase) * pxPerM, armDx = (side) => (side ? armR : armL) / 2 * Math.sin(phase + Math.PI) * pxPerM;
      const shY = 600 - 0.82 * H * pxPerM, leanDx = Math.tan((lean * Math.PI) / 180) * (hipY - shY);
      lm[23] = [px - 4, hipY, 0, 0.95]; lm[24] = [px + 4, hipY, 0, 0.95];
      lm[11] = [px + leanDx - 4, shY, 0, 0.95]; lm[12] = [px + leanDx + 4, shY, 0, 0.95];
      // keep hip-to-ankle length fixed: the ankle drops slightly as the leg swings out
      const legLen = (0.53 - 0.04) * H * pxPerM;
      const ankle = (dx) => [px + dx, hipY + Math.sqrt(Math.max(1, legLen ** 2 - dx ** 2)), 0, 0.9];
      lm[27] = ankle(legDx); lm[28] = ankle(-legDx);
      lm[15] = [lm[11][0] - armDx(0), shY + 0.35 * H * pxPerM, 0, 0.8];
      lm[16] = [lm[12][0] + armDx(1), shY + 0.35 * H * pxPerM, 0, 0.8];
      for (const p of lm) { p[0] += 0.8 * (rnd() - 0.5); p[1] += 0.8 * (rnd() - 0.5); }
      frames.push({ t, lm });
    }
    return frames;
  }

  const api = { analyse, simulate, LEG_FRACTION };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.NeuroGait = api;
})(typeof self !== "undefined" ? self : this);
