/* Spiral-drawing kinematics. Input: strokes = [[{x, y, t(ms), p}, ...], ...] in any pixel scale.
   Every measure is scale-free (relative to the spiral's own radius) so a phone, a tablet and a mouse
   are at least comparable. These are measurements, not a validated classifier. */
(function (root) {
  "use strict";
  const FS = 100; // resample to 100 Hz

  function resample(strokes) {
    const out = [];
    for (const s of strokes) {
      if (s.length < 2) continue;
      const t0 = s[0].t, t1 = s[s.length - 1].t;
      let j = 0;
      for (let t = t0; t <= t1; t += 1000 / FS) {
        while (j < s.length - 2 && s[j + 1].t < t) j++;
        const a = s[j], b = s[j + 1], dt = b.t - a.t || 1, u = Math.min(1, Math.max(0, (t - a.t) / dt));
        out.push({ x: a.x + (b.x - a.x) * u, y: a.y + (b.y - a.y) * u, p: (a.p ?? 0.5) + ((b.p ?? 0.5) - (a.p ?? 0.5)) * u, t });
      }
    }
    return out;
  }

  function unwrap(angles) {
    const out = [angles[0]];
    for (let i = 1; i < angles.length; i++) {
      let d = angles[i] - angles[i - 1];
      while (d > Math.PI) d -= 2 * Math.PI;
      while (d < -Math.PI) d += 2 * Math.PI;
      out.push(out[i - 1] + d);
    }
    return out;
  }

  // fit r = a + b*theta around a candidate centre; returns rms residual and the fit
  function fitAt(P, cx, cy) {
    const th = unwrap(P.map((q) => Math.atan2(q.y - cy, q.x - cx)));
    const r = P.map((q) => Math.hypot(q.x - cx, q.y - cy));
    const n = P.length; let sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (let i = 0; i < n; i++) { sx += th[i]; sy += r[i]; sxx += th[i] * th[i]; sxy += th[i] * r[i]; }
    const b = (n * sxy - sx * sy) / (n * sxx - sx * sx || 1), a = (sy - b * sx) / n;
    let ss = 0; const res = new Float64Array(n);
    for (let i = 0; i < n; i++) { res[i] = r[i] - (a + b * th[i]); ss += res[i] * res[i]; }
    return { rms: Math.sqrt(ss / n), a, b, th, r, res, cx, cy };
  }

  function findCentre(P) {
    let minx = Infinity, maxx = -Infinity, miny = Infinity, maxy = -Infinity;
    for (const q of P) { minx = Math.min(minx, q.x); maxx = Math.max(maxx, q.x); miny = Math.min(miny, q.y); maxy = Math.max(maxy, q.y); }
    const size = Math.max(maxx - minx, maxy - miny);
    let best = fitAt(P, (minx + maxx) / 2, (miny + maxy) / 2);
    for (const [span, steps] of [[0.35 * size, 14], [0.05 * size, 10], [0.008 * size, 10]]) {
      const c0x = best.cx, c0y = best.cy;
      for (let i = -steps / 2; i <= steps / 2; i++) for (let j = -steps / 2; j <= steps / 2; j++) {
        const f = fitAt(P, c0x + (i * span) / steps, c0y + (j * span) / steps);
        if (f.rms < best.rms) best = f;
      }
    }
    return best;
  }

  function spectrum(sig, fs) {
    // remove slow drift with a 0.5 s moving average, then Hann window + DFT
    const n = sig.length, w = Math.round(fs / 2), hp = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0, c = 0; for (let j = Math.max(0, i - w); j < Math.min(n, i + w + 1); j++) { s += sig[j]; c++; }
      hp[i] = sig[i] - s / c;
    }
    const freqs = [], power = [];
    for (let k = 1; k <= Math.floor(n / 2); k++) {
      const f = (k * fs) / n; if (f > 20) break;
      let re = 0, im = 0;
      for (let j = 0; j < n; j++) { const v = hp[j] * (0.5 - 0.5 * Math.cos((2 * Math.PI * j) / n)), a = (-2 * Math.PI * k * j) / n; re += v * Math.cos(a); im += v * Math.sin(a); }
      freqs.push(f); power.push(re * re + im * im);
    }
    return { freqs, power };
  }

  function analyse(strokes) {
    const P = resample(strokes);
    if (P.length < FS * 1.5) return { error: "Draw a bit longer: at least 1.5 seconds of drawing is needed." };
    const fit = findCentre(P);
    let R = 0; for (const v of fit.r) R = Math.max(R, v);
    const turns = Math.abs(fit.th[fit.th.length - 1] - fit.th[0]) / (2 * Math.PI);
    if (turns < 1.5) return { error: "That doesn't look like a spiral yet: draw at least two turns from the centre outward." };
    const resNorm = Array.from(fit.res, (v) => v / R);
    const sp = spectrum(resNorm, FS);
    let total = 0, band = 0, peakF = 0, peakP = -1;
    sp.freqs.forEach((f, i) => {
      if (f >= 1) total += sp.power[i];
      if (f >= 4 && f <= 12) { band += sp.power[i]; if (sp.power[i] > peakP) { peakP = sp.power[i]; peakF = f; } }
    });
    // speed in radii per second, and dimensionless jerk on a lightly smoothed path
    const sm = (k) => { const o = []; for (let i = 0; i < P.length; i++) { let s = 0, c = 0; for (let j = Math.max(0, i - 2); j <= Math.min(P.length - 1, i + 2); j++) { s += P[j][k]; c++; } o.push(s / c); } return o; };
    const xs = sm("x"), ys = sm("y"), dt = 1 / FS, speed = [];
    for (let i = 1; i < P.length; i++) speed.push(Math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]) / dt / R);
    const meanSp = speed.reduce((a, b) => a + b, 0) / speed.length;
    const sdSp = Math.sqrt(speed.reduce((a, b) => a + (b - meanSp) ** 2, 0) / speed.length);
    let jerk2 = 0, L = 0;
    for (let i = 3; i < P.length; i++) {
      const jx = (xs[i] - 3 * xs[i - 1] + 3 * xs[i - 2] - xs[i - 3]) / dt ** 3, jy = (ys[i] - 3 * ys[i - 1] + 3 * ys[i - 2] - ys[i - 3]) / dt ** 3;
      jerk2 += (jx * jx + jy * jy) * dt;
    }
    for (let i = 1; i < P.length; i++) L += Math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]);
    const duration = P.length / FS;
    const pressures = P.map((q) => q.p);
    const pm = pressures.reduce((a, b) => a + b, 0) / pressures.length;
    const pcv = pm > 0 ? Math.sqrt(pressures.reduce((a, b) => a + (b - pm) ** 2, 0) / pressures.length) / pm : NaN;
    return {
      metrics: {
        tremor_band_share: total > 0 ? band / total : NaN,
        tremor_peak_hz: peakF,
        radial_irregularity_pct: (100 * fit.rms) / R,
        speed_cv: sdSp / meanSp,
        log_dimensionless_jerk: Math.log10(Math.sqrt(0.5 * jerk2 * duration ** 5 / (L * L))),
        duration_s: duration,
        turns,
        pen_lifts: Math.max(0, strokes.filter((s) => s.length > 1).length - 1),
        pressure_cv: pcv,
      },
      fit, R, spectrum: sp, points: P,
    };
  }

  // a synthetic spiral for the demo, optionally with a tremor of given frequency and amplitude (fraction of radius)
  function simulate({ tremorHz = 0, tremorAmp = 0, seconds = 9, turns = 3.5, seed = 7 } = {}) {
    let s = seed >>> 0; const rnd = () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    const pts = [], R = 150, fs = 120, n = Math.round(seconds * fs);
    let drift = 0;
    for (let i = 0; i < n; i++) {
      const u = i / (n - 1), th = u * turns * 2 * Math.PI, t = (i / fs) * 1000;
      drift = 0.97 * drift + 0.03 * (rnd() - 0.5) * 0.08;
      const trem = tremorAmp * Math.sin(2 * Math.PI * tremorHz * (i / fs) + 0.3) * (0.7 + 0.3 * rnd());
      const r = R * u * (1 + drift) + R * trem + (rnd() - 0.5) * 0.6;
      pts.push({ x: 200 + r * Math.cos(th), y: 200 + r * Math.sin(th), t, p: 0.5 });
    }
    return [pts];
  }

  const api = { analyse, simulate };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.NeuroSpiral = api;
})(typeof self !== "undefined" ? self : this);
