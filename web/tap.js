/* Tap test: alternate between two on-screen targets with one index finger, as fast and evenly as
   possible, for a fixed time per hand (the two-target design used in mPower and the clinic's
   finger-tapping item of MDS-UPDRS 3.4).
   Input per hand: { duration_s, targets: [{x, y, r}, {x, y, r}], taps: [{t (s), x, y}] }.
   Measures: speed, rhythm variability, decrement (slowing over the trial), accuracy, alternation
   errors, and left-right asymmetry. */
(function (root) {
  "use strict";
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  const sd = (a) => { const m = mean(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length); };

  function analyseHand({ duration_s, targets, taps }) {
    const T = taps.filter((p) => p.t >= 0 && p.t <= duration_s).sort((a, b) => a.t - b.t);
    if (T.length < 6) return { error: "Too few taps. Alternate between the two circles as fast as you comfortably can." };
    const nearest = T.map((p) => {
      const d = targets.map((g) => Math.hypot(p.x - g.x, p.y - g.y) / g.r);
      const k = d[0] <= d[1] ? 0 : 1; return { k, dist: d[k] };
    });
    const hits = nearest.filter((n) => n.dist <= 1).length;
    let sameTarget = 0; for (let i = 1; i < nearest.length; i++) if (nearest[i].k === nearest[i - 1].k) sameTarget++;
    const iti = T.slice(1).map((p, i) => p.t - T[i].t);
    // rate in the first and last thirds of the trial
    // (timed from the first tap, so a slow start doesn't count as speeding up)
    const t0 = T[0].t, third = (duration_s - t0) / 3;
    const rate = (a, b) => T.filter((p) => p.t >= a && p.t < b).length / (b - a);
    const r1 = rate(t0, t0 + third), r3 = rate(t0 + 2 * third, duration_s + 1e-9);
    return {
      taps: T.length,
      rate_hz: T.length / duration_s,
      iti_ms: 1000 * mean(iti),
      iti_cv_pct: (100 * sd(iti)) / mean(iti),
      decrement_pct: r1 > 0 ? (100 * (r1 - r3)) / r1 : 0,
      miss_pct: (100 * (T.length - hits)) / T.length,
      error_radius: mean(nearest.map((n) => n.dist)),
      alternation_errors_pct: (100 * sameTarget) / (T.length - 1),
    };
  }

  function analyse(right, left) {
    const R = right ? analyseHand(right) : null, Lh = left ? analyseHand(left) : null;
    const out = { right: R, left: Lh, asymmetry_pct: NaN };
    if (R && Lh && !R.error && !Lh.error) out.asymmetry_pct = (100 * Math.abs(R.rate_hz - Lh.rate_hz)) / Math.max(R.rate_hz, Lh.rate_hz);
    return out;
  }

  // synthetic tapper: base rate, timing jitter, slowing across the trial, and aim scatter
  function simulate({ duration_s = 15, rateHz = 5, jitter = 0.08, decrement = 0, aim = 0.35, seed = 9 } = {}) {
    let s = seed >>> 0; const rnd = () => { s = (s + 0x6d2b79f5) >>> 0; let x = s; x = Math.imul(x ^ (x >>> 15), x | 1); x ^= x + Math.imul(x ^ (x >>> 7), x | 61); return ((x ^ (x >>> 14)) >>> 0) / 4294967296; };
    const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
    const targets = [{ x: 100, y: 200, r: 45 }, { x: 300, y: 200, r: 45 }];
    const taps = []; let t = 0.3, k = 0;
    while (t < duration_s) {
      const g = targets[k % 2];
      taps.push({ t, x: g.x + aim * g.r * gauss() * 0.7, y: g.y + aim * g.r * gauss() * 0.7 });
      const r = rateHz * (1 - decrement * (t / duration_s));
      t += Math.max(0.05, (1 / r) * (1 + jitter * gauss())); k++;
    }
    return { duration_s, targets, taps };
  }

  const api = { analyse, analyseHand, simulate };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.NeuroTap = api;
})(typeof self !== "undefined" ? self : this);
