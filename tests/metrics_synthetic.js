// Injects known values into synthetic landmark / tap streams and checks each analyser recovers them.
const fs = require("fs"), path = require("path");
const F = require("../web/face.js"), G = require("../web/gait.js"), T = require("../web/tap.js"), S = require("../web/spiral.js");
const out = { face: {}, gait: {}, tap: {}, spiral: {} }; let fails = 0;
const check = (group, name, got, want, tol) => { const ok = Math.abs(got - want) <= tol; if (!ok) fails++; out[group][name] = { injected: want, measured: +got.toFixed(4), ok }; };
// face
const f0 = F.analyse(F.simulate({})).metrics;
check("face", "asymmetry_pct (inject 40%)", F.analyse(F.simulate({ asym: 0.4 })).metrics.asymmetry_pct, 40, 3);
check("face", "cheek_raise_pct (inject 25%)", f0.cheek_raise_pct, 25, 2);
check("face", "blink_rate_per_min (inject 6)", F.analyse(F.simulate({ blinksPerMin: 6 })).metrics.blink_rate_per_min, 6, 1.5);
check("face", "smile size ratio weak/normal (inject 0.42)", F.analyse(F.simulate({ amplitude: 0.05 })).metrics.smile_amplitude / f0.smile_amplitude, 0.05 / 0.12, 0.05);
check("face", "onset slower when injected slower (1 = yes)", F.analyse(F.simulate({ onset: 1.0 })).metrics.onset_s > f0.onset_s * 2 ? 1 : 0, 1, 0);
// gait
const g = (o) => G.analyse(G.simulate(o), o.heightCm || 170).metrics;
check("gait", "speed_m_s (inject 1.2)", g({}).speed_m_s, 1.2, 0.02);
check("gait", "speed_m_s (inject 0.8)", g({ speed: 0.8, cadence: 95 }).speed_m_s, 0.8, 0.02);
check("gait", "speed_m_s, 155 cm person (inject 1.2)", g({ heightCm: 155 }).speed_m_s, 1.2, 0.02);
check("gait", "cadence (inject 110)", g({}).cadence_steps_min, 110, 2);
check("gait", "step_length_m (inject 0.655)", g({}).step_length_m, 1.2 * 60 / 110, 0.02);
check("gait", "arm_swing_asym_pct (inject 60%)", g({ armR: 0.12 }).arm_swing_asym_pct, 60, 3);
check("gait", "trunk_lean_deg (inject 15)", g({ lean: 15 }).trunk_lean_deg, 15, 0.5);
check("gait", "stride variability rises with injected variability (1 = yes)", g({ strideJitter: 0.06, seconds: 10 }).stride_time_cv_pct > 2 * g({ strideJitter: 0.01, seconds: 10 }).stride_time_cv_pct ? 1 : 0, 1, 0);
// tap
const t = (o) => T.analyseHand(T.simulate(o));
check("tap", "rate_hz (inject 5)", t({}).rate_hz, 5, 0.3);
check("tap", "rate_hz (inject 3)", t({ rateHz: 3 }).rate_hz, 3, 0.2);
check("tap", "rhythm variability higher when injected (1 = yes)", t({ jitter: 0.3 }).iti_cv_pct > 2.5 * t({}).iti_cv_pct ? 1 : 0, 1, 0);
check("tap", "decrement_pct (inject 40% linear slowing, ~28% expected)", t({ decrement: 0.4 }).decrement_pct, 28, 6);
check("tap", "decrement_pct (no slowing)", t({}).decrement_pct, 0, 6);
// spiral
check("spiral", "tremor_peak_hz (inject 5)", S.analyse(S.simulate({ tremorHz: 5, tremorAmp: 0.02 })).metrics.tremor_peak_hz, 5, 0.3);
check("spiral", "tremor_peak_hz (inject 9)", S.analyse(S.simulate({ tremorHz: 9, tremorAmp: 0.01 })).metrics.tremor_peak_hz, 9, 0.3);
fs.writeFileSync(path.join(__dirname, "..", "results", "synthetic_recovery.json"), JSON.stringify(out, null, 1));
for (const [grp, r] of Object.entries(out)) for (const [k, v] of Object.entries(r)) console.log((v.ok ? "ok  " : "FAIL") + " " + grp.padEnd(7) + k.padEnd(62) + v.measured);
console.log(fails ? `${fails} FAILED` : "ALL RECOVERED"); process.exit(fails ? 1 : 0);
