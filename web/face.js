/* Smile check: facial expressivity from MediaPipe Face Mesh landmarks (468 points per frame).
   Input: frames = [{t: seconds, lm: [[x, y, z], ...468] in pixels}] (missing face -> lm: null).
   Every distance is divided by the inter-ocular distance (outer eye corners), and measured in a
   face-aligned frame (x along the eye line), so camera distance and head tilt cancel out.
   Measures follow the hypomimia literature: smile amplitude, onset speed, left-right asymmetry,
   cheek raise (eye narrowing during a genuine smile), blink rate and resting facial motion. */
(function (root) {
  "use strict";
  const I = {
    eyeOutR: 33, eyeOutL: 263, mouthR: 61, mouthL: 291, lipUp: 13, lipLow: 14, nose: 1,
    eyeR: [33, 160, 158, 133, 153, 144], eyeL: [362, 385, 387, 263, 373, 380],
  };
  const d2 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const median = (a) => { const b = [...a].sort((x, y) => x - y); const n = b.length; return n ? (n % 2 ? b[(n - 1) / 2] : (b[n / 2 - 1] + b[n / 2]) / 2) : NaN; };
  const pct = (a, q) => { const b = [...a].sort((x, y) => x - y); if (!b.length) return NaN; const p = (b.length - 1) * q / 100, lo = Math.floor(p), hi = Math.ceil(p); return b[lo] + (b[hi] - b[lo]) * (p - lo); };

  // eye aspect ratio (Soukupova & Cech 2016): vertical lid gaps over eye width
  const ear = (lm, e) => (d2(lm[e[1]], lm[e[5]]) + d2(lm[e[2]], lm[e[4]])) / (2 * d2(lm[e[0]], lm[e[3]]));

  // express a point in the face frame: origin between the outer eye corners, x along the eye line, units of IOD
  function faceFrame(lm) {
    const a = lm[I.eyeOutR], b = lm[I.eyeOutL];
    const ox = (a[0] + b[0]) / 2, oy = (a[1] + b[1]) / 2, iod = d2(a, b);
    const ux = (b[0] - a[0]) / iod, uy = (b[1] - a[1]) / iod;            // eye-line direction
    const vx = -uy, vy = ux;                                                // perpendicular (down the face)
    return { iod, map: (p) => [((p[0] - ox) * ux + (p[1] - oy) * uy) / iod, ((p[0] - ox) * vx + (p[1] - oy) * vy) / iod] };
  }

  function smooth(a, w) {
    return a.map((_, i) => { let s = 0, c = 0; for (let j = Math.max(0, i - w); j <= Math.min(a.length - 1, i + w); j++) { s += a[j]; c++; } return s / c; });
  }

  function analyse(frames) {
    const ok = frames.filter((f) => f.lm);
    const quality = { frames: frames.length, face_found_pct: frames.length ? (100 * ok.length) / frames.length : 0 };
    if (ok.length < 15) return { error: "No face found for long enough. Face the camera in good light, about an arm's length away.", quality };
    const t = ok.map((f) => f.t), dur = t[t.length - 1] - t[0];
    const F = ok.map((f) => faceFrame(f.lm));
    const P = ok.map((f, i) => ({ mr: F[i].map(f.lm[I.mouthR]), ml: F[i].map(f.lm[I.mouthL]), up: F[i].map(f.lm[I.lipUp]), lo: F[i].map(f.lm[I.lipLow]) }));
    // head turn: the nose sits midway between the eyes when facing the camera
    const yaw = ok.map((f, i) => F[i].map(f.lm[I.nose])[0]);
    quality.head_turn = median(yaw.map(Math.abs));

    const width = smooth(P.map((p) => Math.hypot(p.ml[0] - p.mr[0], p.ml[1] - p.mr[1])), 1);
    // corner lift: how far the corners sit above the lip centre line (positive = raised)
    const lift = smooth(P.map((p) => ((p.up[1] + p.lo[1]) / 2) - (p.mr[1] + p.ml[1]) / 2), 1);
    const w0 = pct(width, 10), l0 = pct(lift, 10);
    const S = width.map((w, i) => (w - w0) + (lift[i] - l0));          // smile index, IOD units
    const peak = pct(S, 98);
    const earv = ok.map((f) => (ear(f.lm, I.eyeR) + ear(f.lm, I.eyeL)) / 2);

    // smile episodes: above half of the peak for at least 0.3 s
    const thr = 0.5 * peak, events = [];
    let start = null;
    for (let i = 0; i <= S.length; i++) {
      const on = i < S.length && S[i] > thr;
      if (on && start === null) start = i;
      if (!on && start !== null) { if (t[i - 1] - t[start] >= 0.3) events.push([start, i - 1]); start = null; }
    }
    const metrics = { smile_amplitude: NaN, width_gain_pct: NaN, onset_s: NaN, asymmetry_pct: NaN, cheek_raise_pct: NaN, repeat_ratio: NaN, blink_rate_per_min: NaN, rest_motion: NaN };
    const neutralIdx = S.map((v, i) => i).filter((i) => S[i] < 0.2 * peak);
    if (peak > 0.04 && events.length) {
      metrics.smile_amplitude = peak;
      const ev = events[0];
      let pi = ev[0]; for (let i = ev[0]; i <= ev[1]; i++) if (S[i] > S[pi]) pi = i;
      metrics.width_gain_pct = 100 * (width[pi] / w0 - 1);
      // onset: last time before the peak at 10% of it, to first time at 90%
      let a = pi; while (a > 0 && S[a] > 0.1 * S[pi]) a--;
      let b = a; while (b < pi && S[b] < 0.9 * S[pi]) b++;
      metrics.onset_s = t[b] - t[a];
      // asymmetry: each corner's movement from its neutral position
      const nr = [median(neutralIdx.map((i) => P[i].mr[0])), median(neutralIdx.map((i) => P[i].mr[1]))];
      const nl = [median(neutralIdx.map((i) => P[i].ml[0])), median(neutralIdx.map((i) => P[i].ml[1]))];
      const dR = Math.hypot(P[pi].mr[0] - nr[0], P[pi].mr[1] - nr[1]), dL = Math.hypot(P[pi].ml[0] - nl[0], P[pi].ml[1] - nl[1]);
      metrics.asymmetry_pct = (100 * Math.abs(dR - dL)) / (dR + dL);
      const earN = median(neutralIdx.map((i) => earv[i]));
      const earS = median(earv.slice(Math.max(ev[0], pi - 2), Math.min(ev[1], pi + 2) + 1));
      metrics.cheek_raise_pct = (100 * (earN - earS)) / earN;
      if (events.length > 1) {
        const pk = (e) => Math.max(...S.slice(e[0], e[1] + 1));
        metrics.repeat_ratio = pk(events[events.length - 1]) / pk(events[0]);
      }
    }
    // blinks: eye aspect ratio falls below 60% of its median, briefly (< 0.5 s)
    const earMed = median(earv); let blinks = 0, inB = false, bStart = 0;
    for (let i = 0; i < earv.length; i++) {
      const closed = earv[i] < 0.6 * earMed;
      if (closed && !inB) { inB = true; bStart = t[i]; }
      if (!closed && inB) { inB = false; if (t[i] - bStart < 0.5) blinks++; }
    }
    if (dur >= 5) metrics.blink_rate_per_min = (60 * blinks) / dur;
    // resting facial motion: median movement of all landmarks between neutral frames, IOD per second
    const mv = [];
    for (let k = 1; k < neutralIdx.length; k++) {
      const i = neutralIdx[k], j = neutralIdx[k - 1]; if (i !== j + 1) continue;
      const A = ok[i].lm, B = ok[j].lm, fa = F[i], fb = F[j]; let s = 0, n = 0;
      for (let q = 0; q < A.length; q += 4) { const pa = fa.map(A[q]), pb = fb.map(B[q]); s += Math.hypot(pa[0] - pb[0], pa[1] - pb[1]); n++; }
      mv.push(s / n / (t[i] - t[j]));
    }
    if (mv.length > 5) metrics.rest_motion = median(mv);
    quality.duration_s = dur; quality.smiles_found = events.length;
    if (quality.head_turn > 0.25) quality.warning = "The head was turned away from the camera, which makes the asymmetry figure unreliable.";
    if (!events.length) quality.warning = "No clear smile was found. Hold a relaxed face, then give one big smile.";
    return { metrics, quality, series: { t, smile: S, ear: earv } };
  }

  // synthetic landmark generator for tests and the demo: a face template with animated mouth corners and eyelids
  function simulate({ seconds = 10, fps = 30, smiles = [[2.5, 4.5], [6.5, 8.5]], amplitude = 0.12, onset = 0.35, asym = 0, blinksPerMin = 15, seed = 5 } = {}) {
    let s = seed >>> 0; const rnd = () => { s = (s + 0x6d2b79f5) >>> 0; let x = s; x = Math.imul(x ^ (x >>> 15), x | 1); x ^= x + Math.imul(x ^ (x >>> 7), x | 61); return ((x ^ (x >>> 14)) >>> 0) / 4294967296; };
    const base = []; for (let i = 0; i < 468; i++) base.push([300 + 120 * (rnd() - 0.5), 300 + 150 * (rnd() - 0.5), 0]);
    const set = (i, x, y) => { base[i] = [x, y, 0]; };
    set(33, 240, 260); set(263, 360, 260); set(133, 280, 262); set(362, 320, 262);
    set(160, 252, 254); set(158, 268, 254); set(153, 268, 266); set(144, 252, 266);
    set(385, 332, 254); set(387, 348, 254); set(373, 348, 266); set(380, 332, 266);
    set(1, 300, 310); set(61, 270, 360); set(291, 330, 360); set(13, 300, 356); set(14, 300, 366);
    const blinkTimes = []; for (let x = 1.2; x < seconds; x += 60 / blinksPerMin) blinkTimes.push(x + 0.3 * (rnd() - 0.5));
    const frames = [];
    for (let k = 0; k < seconds * fps; k++) {
      const tt = k / fps; let sm = 0;
      for (const [a, b] of smiles) {
        const rise = (x) => x <= 0 ? 0 : x >= 1 ? 1 : 0.5 - 0.5 * Math.cos(Math.PI * x);
        sm = Math.max(sm, Math.min(rise((tt - a) / (onset / 0.8)), 1 - rise((tt - b) / 0.4)));
      }
      const lm = base.map((p) => [p[0] + 0.15 * (rnd() - 0.5), p[1] + 0.15 * (rnd() - 0.5), 0]);
      const iod = 120, A = amplitude * iod * sm;
      lm[61] = [270 - A * 0.55 * (1 - asym), 360 - A * 0.45 * (1 - asym), 0];
      lm[291] = [330 + A * 0.55 * (1 + asym), 360 - A * 0.45 * (1 + asym), 0];
      const squint = 1 - 0.25 * sm; let lid = 1;
      for (const bt of blinkTimes) if (Math.abs(tt - bt) < 0.08) lid = 0.15;
      for (const [u, l] of [[160, 144], [158, 153], [385, 380], [387, 373]]) {
        const cy = 260, h = 6 * squint * lid; lm[u] = [base[u][0], cy - h, 0]; lm[l] = [base[l][0], cy + h, 0];
      }
      frames.push({ t: tt, lm });
    }
    return frames;
  }

  const api = { analyse, simulate };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.NeuroFace = api;
})(typeof self !== "undefined" ? self : this);
