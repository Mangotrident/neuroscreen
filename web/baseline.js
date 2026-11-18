/* Per-person baseline and change engine.
   Each person is compared with their own first weeks, not with population norms, because the robot,
   room and microphone stay the same and people differ a lot from each other.
   - baseline: median and MAD of a metric over the person's first sessions (robust to bad days)
   - sustained change: the median of the last 5 sessions is far outside baseline (judged against its
     standard error) and at least 4 of those 5 sessions point the same way
   - sudden change (one-sided metrics only): today far outside baseline while yesterday was normal
   - patterns: changes are grouped by domain (voice, face, walking, hands) and combined into
     plain-language prompts. Prompts point to an area to discuss with a doctor; they never name a disease. */
(function (root) {
  "use strict";
  const REGISTRY = {
    voice_score:          { label: "Voice model score",          domain: "voice", worse: "up" },
    jitter_local:         { label: "Voice jitter",               domain: "voice", worse: "up" },
    shimmer_local:        { label: "Voice shimmer",              domain: "voice", worse: "up" },
    hnr_db:               { label: "Voice clarity (HNR)",        domain: "voice", worse: "down" },
    smile_amplitude:      { label: "Smile size",                 domain: "face",  worse: "down" },
    onset_s:              { label: "Smile speed (onset time)",   domain: "face",  worse: "up" },
    asymmetry_pct:        { label: "Smile asymmetry",            domain: "face",  worse: "up", acute: true, acuteMin: 15 },
    blink_rate_per_min:   { label: "Blink rate",                 domain: "face",  worse: "down" },
    speed_m_s:            { label: "Walking speed",              domain: "gait",  worse: "down" },
    step_length_m:        { label: "Step length",                domain: "gait",  worse: "down" },
    stride_time_cv_pct:   { label: "Stride variability",         domain: "gait",  worse: "up" },
    arm_swing_asym_pct:   { label: "Arm-swing asymmetry",        domain: "gait",  worse: "up" },
    trunk_lean_deg:       { label: "Forward lean",               domain: "gait",  worse: "up" },
    tap_rate_hz:          { label: "Tapping speed",              domain: "hand",  worse: "down" },
    tap_decrement_pct:    { label: "Tapping slow-down",          domain: "hand",  worse: "up" },
    tap_iti_cv_pct:       { label: "Tapping rhythm variability", domain: "hand",  worse: "up" },
    tap_asymmetry_pct:    { label: "Left-right tapping gap",     domain: "hand",  worse: "up", acute: true, acuteMin: 20 },
    tremor_band_share:    { label: "Spiral tremor",              domain: "hand",  worse: "up" },
  };
  const DOMAIN_NAMES = { voice: "voice", face: "facial expression", gait: "walking", hand: "hand movement" };
  let Z_SUSTAINED = 4, Z_ACUTE = 5, MIN_BASELINE = 6, K_RECENT = 5;

  const median = (a) => { const b = [...a].sort((x, y) => x - y); const n = b.length; return n % 2 ? b[(n - 1) / 2] : (b[n / 2 - 1] + b[n / 2]) / 2; };

  // Theil-Sen slope: median of pairwise slopes, robust to outliers
  function theilSen(x, y) {
    const s = []; for (let i = 0; i < x.length; i++) for (let j = i + 1; j < x.length; j++) if (x[j] !== x[i]) s.push((y[j] - y[i]) / (x[j] - x[i]));
    return s.length ? median(s) : 0;
  }

  const dayNum = (d) => Math.round(new Date(d + "T00:00:00Z").getTime() / 86400000);

  function evaluate(sessions) {
    const S = [...sessions].sort((a, b) => (a.date < b.date ? -1 : 1));
    const results = {};
    for (const [key, meta] of Object.entries(REGISTRY)) {
      const pts = S.map((s) => ({ d: dayNum(s.date), date: s.date, v: s.metrics[key] })).filter((p) => Number.isFinite(p.v));
      if (pts.length < MIN_BASELINE + K_RECENT) continue;
      const nb = Math.max(MIN_BASELINE, Math.min(14, pts.length - K_RECENT));
      const base = pts.slice(0, nb).map((p) => p.v);
      const med = median(base);
      const mad = 1.4826 * median(base.map((v) => Math.abs(v - med)));
      // the larger of the robust and classical spread: a short baseline can make the MAD alone too small
      const bm = base.reduce((a, b) => a + b, 0) / base.length;
      const bsd = Math.sqrt(base.reduce((a, b) => a + (b - bm) ** 2, 0) / (base.length - 1));
      const scale = Math.max(mad, bsd, 0.05 * Math.abs(med), 1e-9);
      const sign = meta.worse === "up" ? 1 : -1;
      const z = pts.map((p) => (sign * (p.v - med)) / scale);
      // sustained change: the median of the last 5 sessions sits far beyond baseline (using its
      // standard error, inflated for a short baseline) and most of those sessions agree
      const seMed = (1.2533 / Math.sqrt(K_RECENT)) * (1 + 1 / Math.sqrt(nb));
      const windowFlag = (end) => {
        const w = z.slice(end - K_RECENT, end);
        const zr = w.length === K_RECENT ? median(w) / seMed : 0;
        return { zr, hit: zr >= Z_SUSTAINED && w.filter((v) => v >= 1.5).length >= 4 };
      };
      const now = windowFlag(z.length), prev = windowFlag(z.length - 1);
      const zRecent = now.zr;
      // must hold on two consecutive days, so one noisy session can't trigger it
      const sustained = now.hit && prev.hit;
      const jump = pts[pts.length - 1].v - med;
      const acute = !!meta.acute && z[z.length - 1] >= Z_ACUTE && z[z.length - 2] < 2 && jump >= meta.acuteMin;
      const lastDays = pts.filter((p) => p.d >= pts[pts.length - 1].d - 28);
      const trendPerWeek = 7 * theilSen(lastDays.map((p) => p.d), lastDays.map((p, i) => z[pts.length - lastDays.length + i]));
      results[key] = { ...meta, key, baseline: med, scale, latest: pts[pts.length - 1].v, z_latest: z[z.length - 1], z_recent: zRecent, sustained, acute, trend_z_per_week: trendPerWeek, series: pts.map((p, i) => ({ date: p.date, v: p.v, z: z[i] })) };
    }
    return { metrics: results, patterns: patterns(results) };
  }

  function patterns(R) {
    const out = [];
    const flagged = Object.values(R).filter((m) => m.sustained);
    const byDomain = {}; for (const m of flagged) (byDomain[m.domain] = byDomain[m.domain] || []).push(m);
    const names = (ms) => ms.map((m) => m.label.toLowerCase()).join(", ");
    const Names = (ms) => { const t = names(ms); return t.charAt(0).toUpperCase() + t.slice(1); };

    const acute = Object.values(R).filter((m) => m.acute);
    if (acute.length) out.push({
      level: "urgent", title: "Sudden one-sided change today",
      text: `${Names(acute)} changed sharply since yesterday. Check in now: if you see a drooping face, arm weakness or slurred speech, call emergency services immediately.`,
      metrics: acute.map((m) => m.key),
    });
    const motorDomains = ["face", "gait", "hand", "voice"].filter((d) => byDomain[d]);
    if (motorDomains.length >= 2) out.push({
      level: "discuss", title: `Changes in ${motorDomains.length} movement tests`,
      text: `Over recent weeks, ${motorDomains.map((d) => DOMAIN_NAMES[d]).join(", ")} have all drifted from this person's usual (${names(flagged)}). One test changing is often noise; several together are worth raising with a doctor or neurologist. Bring this summary.`,
      metrics: flagged.map((m) => m.key),
    });
    else {
      if (byDomain.gait) out.push({
        level: "watch", title: "Walking has changed",
        text: `${Names(byDomain.gait)} moved away from usual. Ask about pain, dizziness, new medicines or a recent fall; a falls-risk check with the doctor is worth booking if it persists.`,
        metrics: byDomain.gait.map((m) => m.key),
      });
      if (byDomain.voice) out.push({
        level: "watch", title: "Voice has changed",
        text: `${Names(byDomain.voice)} differ from usual. Ask about a cold, sore throat or breathlessness. If it lasts beyond two weeks, mention it to the doctor.`,
        metrics: byDomain.voice.map((m) => m.key),
      });
      if (byDomain.face) out.push({ level: "watch", title: "Facial expression has changed", text: `${Names(byDomain.face)} differ from usual. Low mood, tiredness and some medicines can do this; ask how they've been feeling.`, metrics: byDomain.face.map((m) => m.key) });
      if (byDomain.hand) out.push({ level: "watch", title: "Hand tests have changed", text: `${Names(byDomain.hand)} differ from usual. Ask about new medicines, caffeine, poor sleep or hand pain.`, metrics: byDomain.hand.map((m) => m.key) });
    }
    // voice getting rougher at the same time as walking slows: breathing or fluid problems are one possibility
    if (byDomain.voice && R.speed_m_s && R.speed_m_s.trend_z_per_week >= 0.7) out.push({
      level: "discuss", title: "Voice and stamina dropped together",
      text: "Voice quality and walking speed have both been declining. Check for breathlessness, swollen ankles, fever or cough, and call the doctor if any are present.",
      metrics: ["speed_m_s", ...byDomain.voice.map((m) => m.key)],
    });
    const recentSpeed = R.speed_m_s ? median(R.speed_m_s.series.slice(-K_RECENT).map((p) => p.v)) : NaN;
    if (R.speed_m_s && R.speed_m_s.series.length >= K_RECENT && recentSpeed < 1.0 && R.speed_m_s.baseline >= 1.0) out.push({
      level: "watch", title: "Walking speed below 1.0 m/s",
      text: `Typical walking speed over the last ${K_RECENT} checks is ${recentSpeed.toFixed(2)} m/s. Clinicians treat walking slower than 1.0 m/s in older adults as a yellow flag for falls and frailty.`,
      metrics: ["speed_m_s"],
    });
    return out;
  }

  // ---- synthetic histories for tests and the demo ----
  const TYPICAL = { // [mean, between-day SD] for a healthy older adult measured by the same device
    voice_score: [0.45, 0.06], jitter_local: [0.006, 0.0012], shimmer_local: [0.035, 0.006], hnr_db: [21, 1.5],
    smile_amplitude: [0.18, 0.02], onset_s: [0.45, 0.06], asymmetry_pct: [6, 2.5], blink_rate_per_min: [16, 3],
    speed_m_s: [1.15, 0.06], step_length_m: [0.62, 0.03], stride_time_cv_pct: [2.2, 0.5], arm_swing_asym_pct: [12, 5], trunk_lean_deg: [5, 1.2],
    tap_rate_hz: [4.6, 0.3], tap_decrement_pct: [3, 4], tap_iti_cv_pct: [9, 1.5], tap_asymmetry_pct: [8, 4], tremor_band_share: [0.25, 0.07],
  };
  const DRIFT = { // change at full effect, in units of the metric
    motor: { smile_amplitude: -0.07, onset_s: 0.25, arm_swing_asym_pct: 30, tap_decrement_pct: 18, tap_rate_hz: -1.2, shimmer_local: 0.02, jitter_local: 0.004, voice_score: 0.25 },
    mobility: { speed_m_s: -0.3, step_length_m: -0.12, stride_time_cv_pct: 2 },
    voice_cold: { hnr_db: -5, shimmer_local: 0.025, jitter_local: 0.004 },
  };
  function simulateHistory({ days = 56, scenario = "stable", changeStart = 28, rampDays = 21, seed = 1, skipProb = 0.15, acuteDay = null } = {}) {
    let s = seed >>> 0; const rnd = () => { s = (s + 0x6d2b79f5) >>> 0; let x = s; x = Math.imul(x ^ (x >>> 15), x | 1); x ^= x + Math.imul(x ^ (x >>> 7), x | 61); return ((x ^ (x >>> 14)) >>> 0) / 4294967296; };
    const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
    const out = [], start = Date.UTC(2026, 8, 1);
    for (let d = 0; d < days; d++) {
      if (rnd() < skipProb) continue;
      const f = Math.min(1, Math.max(0, (d - changeStart) / rampDays));
      const metrics = {};
      for (const [k, [m, sdv]] of Object.entries(TYPICAL)) {
        const drift = (DRIFT[scenario] && DRIFT[scenario][k]) || 0;
        metrics[k] = m + drift * f + sdv * gauss();
      }
      if (acuteDay !== null && d === acuteDay) { metrics.asymmetry_pct += 40; metrics.tap_asymmetry_pct += 45; }
      out.push({ date: new Date(start + d * 86400000).toISOString().slice(0, 10), metrics });
    }
    return out;
  }

  const setThreshold = (z) => { Z_SUSTAINED = z; };
  const api = { REGISTRY, evaluate, simulateHistory, setThreshold };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.NeuroBaseline = api;
})(typeof self !== "undefined" ? self : this);
