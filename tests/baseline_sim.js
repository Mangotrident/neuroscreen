// How often does the engine raise a prompt for a stable person, and how fast does it catch a real drift?
// Every simulated day is evaluated with only the sessions up to that day, as the robot would.
const B = require("../web/baseline.js");
function firstAlert(hist, levels) {
  for (let i = 10; i <= hist.length; i++) {
    const p = B.evaluate(hist.slice(0, i)).patterns.filter((x) => levels.includes(x.level));
    if (p.length) return { day: hist[i - 1].date, idx: i, titles: p.map((x) => x.title) };
  }
  return null;
}
const N = 200, out = {};
let falseAny = 0, falseDiscuss = 0;
for (let k = 0; k < N; k++) {
  const h = B.simulateHistory({ scenario: "stable", seed: 1000 + k, days: 56 });
  const levels = new Set();
  for (let i = 10; i <= h.length; i++) for (const p of B.evaluate(h.slice(0, i)).patterns) levels.add(p.level);
  if (levels.size) falseAny++;
  if (levels.has("discuss") || levels.has("urgent")) falseDiscuss++;
}
out.note = "Simulated people with assumed day-to-day variability; real variability must be measured in a pilot.";
out.stable_people_with_any_prompt_in_8_weeks_pct = (100 * falseAny) / N;
out.stable_people_with_discuss_or_urgent_prompt_pct = (100 * falseDiscuss) / N;
for (const scen of ["motor", "mobility"]) {
  let caught = 0; const delays = [];
  for (let k = 0; k < N; k++) {
    const h = B.simulateHistory({ scenario: scen, seed: 5000 + k, days: 70, changeStart: 28, rampDays: 21 });
    const want = scen === "motor" ? ["discuss"] : ["watch", "discuss"];
    const a = firstAlert(h, want);
    if (a) { const d = (Date.parse(a.day) - Date.UTC(2026, 8, 1)) / 86400000 - 28; if (d >= 0) { caught++; delays.push(d); } }
  }
  delays.sort((a, b) => a - b);
  out[`${scen}_drift_caught_pct`] = (100 * caught) / N;
  out[`${scen}_median_days_after_change_starts`] = delays[Math.floor(delays.length / 2)];
}
let acuteCaught = 0;
for (let k = 0; k < N; k++) {
  const h = B.simulateHistory({ scenario: "stable", seed: 9000 + k, days: 40, acuteDay: 39, skipProb: 0 });
  if (B.evaluate(h).patterns.some((p) => p.level === "urgent")) acuteCaught++;
}
out.sudden_one_sided_change_caught_same_day_pct = (100 * acuteCaught) / N;
console.log(JSON.stringify(out, null, 1));
require("fs").writeFileSync(require("path").join(__dirname, "..", "results", "baseline_sim.json"), JSON.stringify(out, null, 1));
