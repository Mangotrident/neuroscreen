/* UI for the tap test, smile check, walking check and weekly view. Runs after the voice/spiral UI. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) || !Number.isFinite(v)) ? "–" : Number(v).toFixed(d);
  const FACE = window.NeuroFace, GAIT = window.NeuroGait, TAP = window.NeuroTap, BASE = window.NeuroBaseline, VISION = window.NeuroVision;
  const R = RESULTS;

  function setupCanvas(cv) {
    const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight || Number(cv.getAttribute("height"));
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
    const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); return { ctx, w, h };
  }
  function tiles(id, list) {
    $(id).innerHTML = list.map(([v, k, warn]) => `<div class="tile${warn ? " warn" : ""}"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");
  }
  function status(id, msg, err) { $(id).textContent = msg || ""; $(id).classList.toggle("err", !!err); }

  // generic line plot: series = [{x: [], y: [], color}], marks = [{x, color}]
  function plot(cv, series, { marks = [], label = "", yMin, yMax, xLabel = "s" } = {}) {
    const { ctx, w, h } = setupCanvas(cv); ctx.clearRect(0, 0, w, h);
    const xs = series.flatMap((s) => s.x), ys = series.flatMap((s) => s.y).filter(Number.isFinite);
    if (!xs.length) return;
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = yMin ?? Math.min(...ys), y1 = yMax ?? Math.max(...ys);
    const padL = 8, padR = 8, padT = 16, padB = 18;
    const X = (x) => padL + ((x - x0) / (x1 - x0 || 1)) * (w - padL - padR), Y = (y) => h - padB - ((y - y0) / (y1 - y0 || 1)) * (h - padT - padB);
    ctx.strokeStyle = css("--grid"); ctx.beginPath(); ctx.moveTo(padL, h - padB); ctx.lineTo(w - padR, h - padB); ctx.stroke();
    for (const m of marks) { ctx.strokeStyle = m.color; ctx.globalAlpha = 0.7; ctx.beginPath(); ctx.moveTo(X(m.x), padT - 4); ctx.lineTo(X(m.x), padT + 4); ctx.stroke(); ctx.globalAlpha = 1; }
    for (const s of series) {
      ctx.strokeStyle = s.color; ctx.lineWidth = 1.6; ctx.beginPath(); let started = false;
      s.x.forEach((x, i) => { const y = s.y[i]; if (!Number.isFinite(y)) return; started ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)); started = true; });
      ctx.stroke();
    }
    ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono");
    ctx.fillText(label, padL, 11); ctx.fillText(`${fmt(x1 - x0, 0)} ${xLabel}`, w - padR - 40, h - 4);
  }

  // ---------------- TAP ----------------
  const tc = $("tapCanvas"); let tapState = null; const tapData = { right: null, left: null };
  const TAP_SECONDS = 15;
  function tapTargets() { const w = tc.clientWidth, h = tc.clientHeight; const r = Math.min(w * 0.13, h * 0.32); return [{ x: w * 0.27, y: h / 2, r }, { x: w * 0.73, y: h / 2, r }]; }
  function drawTap(flash) {
    const { ctx, w, h } = setupCanvas(tc); ctx.clearRect(0, 0, w, h);
    tapTargets().forEach((g, i) => {
      ctx.beginPath(); ctx.arc(g.x, g.y, g.r, 0, 2 * Math.PI);
      ctx.fillStyle = flash === i ? css("--accent") : css("--card"); ctx.fill();
      ctx.lineWidth = 2; ctx.strokeStyle = css("--accent"); ctx.stroke();
    });
    ctx.fillStyle = css("--muted"); ctx.font = "12px " + css("--font-body"); ctx.textAlign = "center";
    ctx.fillText(tapState ? `${tapState.hand} hand · ${Math.max(0, TAP_SECONDS - (performance.now() - tapState.t0) / 1000).toFixed(0)} s` : "Press Start, then tap left, right, left, right…", w / 2, h - 10);
    ctx.textAlign = "start";
  }
  function startTap(hand) {
    tapState = { hand, t0: performance.now(), taps: [], targets: tapTargets() };
    $("tapSource").textContent = "your taps"; status("tapStatus", `Tap now with your ${hand} hand`);
    const tick = () => { if (!tapState) return; drawTap(); if ((performance.now() - tapState.t0) / 1000 >= TAP_SECONDS) finishTap(); else requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  }
  function finishTap() {
    const s = tapState; tapState = null;
    tapData[s.hand] = { duration_s: TAP_SECONDS, targets: s.targets, taps: s.taps };
    status("tapStatus", tapData.right && tapData.left ? "" : `Done. Now do the ${s.hand === "right" ? "left" : "right"} hand.`);
    renderTap();
  }
  tc.addEventListener("pointerdown", (e) => {
    if (!tapState) return;
    const r = tc.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    tapState.taps.push({ t: (performance.now() - tapState.t0) / 1000, x, y });
    const g = tapState.targets, k = Math.hypot(x - g[0].x, y - g[0].y) < Math.hypot(x - g[1].x, y - g[1].y) ? 0 : 1; drawTap(k);
    e.preventDefault();
  });
  function renderTap() {
    const res = TAP.analyse(tapData.right, tapData.left);
    const row = (hand, m) => m && !m.error ? [
      [`${fmt(m.rate_hz, 1)}/s`, `${hand}: taps per second`],
      [`${fmt(m.iti_cv_pct, 0)}%`, `${hand}: rhythm variability`, m.iti_cv_pct > 20],
      [`${fmt(m.decrement_pct, 0)}%`, `${hand}: slowing over 15 s`, m.decrement_pct > 15],
    ] : m && m.error ? [["–", `${hand}: ${m.error}`]] : [];
    tiles("tapTiles", [...row("Right", res.right), ...row("Left", res.left), ...(Number.isFinite(res.asymmetry_pct) ? [[`${fmt(res.asymmetry_pct, 0)}%`, "gap between hands", res.asymmetry_pct > 25]] : [])]);
    const series = [];
    for (const [hand, color] of [["right", css("--accent")], ["left", css("--pd")]]) {
      const d = tapData[hand]; if (!d || d.taps.length < 3) continue;
      const T = d.taps.map((p) => p.t); series.push({ x: T.slice(1), y: T.slice(1).map((t, i) => 1000 * (t - T[i])), color });
    }
    if (series.length) plot($("tapPlot"), series, { label: "ms between taps (blue right, red left)" });
    drawTap();
  }
  $("tapRight").addEventListener("click", () => startTap("right"));
  $("tapLeft").addEventListener("click", () => startTap("left"));
  $("tapSyn").addEventListener("click", () => {
    tapData.right = TAP.simulate({ rateHz: 4.8, seed: 21 }); tapData.left = TAP.simulate({ rateHz: 4.3, decrement: 0.15, seed: 22 });
    $("tapSource").textContent = "synthetic example"; status("tapStatus", ""); renderTap();
  });

  // ---------------- FACE ----------------
  function renderFace(r, label) {
    $("faceSource").textContent = label;
    if (r.error) { status("faceStatus", r.error, true); return; }
    status("faceStatus", r.quality.warning || "", !!r.quality.warning);
    const m = r.metrics;
    tiles("faceTiles", [
      [`+${fmt(m.width_gain_pct, 0)}%`, "mouth widening in the smile"],
      [`${fmt(m.onset_s, 2)} s`, "time to reach full smile"],
      [`${fmt(m.asymmetry_pct, 0)}%`, "left-right difference", m.asymmetry_pct > 25],
      [`${fmt(m.cheek_raise_pct, 0)}%`, "eye narrowing (a genuine smile)"],
      [`${fmt(m.blink_rate_per_min, 0)}/min`, "blink rate"],
      [Number.isFinite(m.repeat_ratio) ? `${fmt(m.repeat_ratio, 2)}×` : "–", "second smile vs first"],
    ]);
    const s = r.series, earMed = [...s.ear].sort((a, b) => a - b)[Math.floor(s.ear.length / 2)];
    const marks = []; let closed = false;
    s.ear.forEach((e, i) => { const c = e < 0.6 * earMed; if (c && !closed) marks.push({ x: s.t[i], color: css("--muted") }); closed = c; });
    plot($("facePlot"), [{ x: s.t, y: s.smile, color: css("--accent") }], { marks, label: "smile size over time · ticks = blinks", yMin: 0 });
  }
  async function runVision(kind, getFrames, label, statusId, render) {
    try {
      status(statusId, "Loading the on-device model…");
      const { frames } = await getFrames((p) => status(statusId, `Analysing… ${Math.round(100 * p)}%`));
      status(statusId, "");
      render(frames, label);
    } catch (e) {
      status(statusId, String(e && e.message || e).includes("Permission") || (e && e.name === "NotAllowedError") ?
        "The camera is blocked here. Upload a short video instead, or open demo.html from the repo on a device with a camera." : `Couldn't analyse that: ${e.message || e}`, true);
    }
  }
  $("faceFile").addEventListener("change", (e) => {
    const f = e.target.files[0]; if (!f) return; e.target.value = "";
    runVision("face", (cb) => VISION.fromFile("face", f, { fps: 15, maxSeconds: 15, onProgress: cb }), f.name, "faceStatus", (fr, l) => renderFace(FACE.analyse(fr), l));
  });
  $("faceCam").addEventListener("click", () => {
    const v = $("faceVideo"); v.hidden = false;
    runVision("face", (cb) => VISION.fromCamera("face", v, { seconds: 10, onProgress: cb }), "your camera", "faceStatus", (fr, l) => { v.hidden = true; renderFace(FACE.analyse(fr), l); });
  });
  $("faceSynA").addEventListener("click", () => renderFace(FACE.analyse(FACE.simulate({})), "synthetic: typical"));
  $("faceSynB").addEventListener("click", () => renderFace(FACE.analyse(FACE.simulate({ amplitude: 0.06, onset: 0.9, blinksPerMin: 7, seed: 8 })), "synthetic: reduced expression"));

  // ---------------- WALK ----------------
  function renderGait(r, label) {
    $("gaitSource").textContent = label;
    if (r.error) { status("gaitStatus", r.error, true); return; }
    status("gaitStatus", "");
    const m = r.metrics;
    tiles("gaitTiles", [
      [`${fmt(m.speed_m_s, 2)} m/s`, "walking speed", m.speed_m_s < 1.0],
      [`${fmt(m.cadence_steps_min, 0)}`, "steps per minute"],
      [`${fmt(m.step_length_m, 2)} m`, "step length"],
      [`${fmt(m.stride_time_cv_pct, 1)}%`, "stride-time variability", m.stride_time_cv_pct > 4],
      [`${fmt(m.arm_swing_left, 2)} / ${fmt(m.arm_swing_right, 2)} m`, "arm swing, left / right"],
      [`${fmt(m.arm_swing_asym_pct, 0)}%`, "arm-swing asymmetry", m.arm_swing_asym_pct > 40],
      [`${fmt(m.trunk_lean_deg, 0)}°`, "forward lean"],
      [`${fmt(r.quality.distance_m, 1)} m`, `walked · ${r.quality.steps_found} steps found`],
    ]);
    const s = r.series;
    plot($("gaitPlot"), [{ x: s.t, y: s.sep, color: css("--accent") }], { label: "ankle separation (m) · each peak is a step" });
  }
  const height = () => Number($("gaitHeight").value);
  $("gaitFile").addEventListener("change", (e) => {
    const f = e.target.files[0]; if (!f) return; e.target.value = "";
    runVision("pose", (cb) => VISION.fromFile("pose", f, { fps: 20, maxSeconds: 15, onProgress: cb }), f.name, "gaitStatus", (fr, l) => renderGait(GAIT.analyse(fr, height()), l));
  });
  $("gaitCam").addEventListener("click", () => {
    const v = $("gaitVideo"); v.hidden = false;
    runVision("pose", (cb) => VISION.fromCamera("pose", v, { seconds: 8, onProgress: cb }), "your camera", "gaitStatus", (fr, l) => { v.hidden = true; renderGait(GAIT.analyse(fr, height()), l); });
  });
  $("gaitSynA").addEventListener("click", () => renderGait(GAIT.analyse(GAIT.simulate({ heightCm: height() }), height()), "synthetic: typical"));
  $("gaitSynB").addEventListener("click", () => renderGait(GAIT.analyse(GAIT.simulate({ heightCm: height(), speed: 0.85, cadence: 98, armR: 0.1, lean: 11, strideJitter: 0.06, seconds: 8, seed: 7 }), height()), "synthetic: slower, uneven arms"));

  // ---------------- WEEKLY ----------------
  const SCEN = {
    stable: { args: { scenario: "stable", seed: 41, days: 63 }, name: "Stable" },
    motor: { args: { scenario: "motor", seed: 42, days: 63, changeStart: 28, rampDays: 21 }, name: "Slow drift across several tests" },
    mobility: { args: { scenario: "mobility", seed: 43, days: 63, changeStart: 28, rampDays: 21 }, name: "Walking decline only" },
    acute: { args: { scenario: "stable", seed: 44, days: 50, acuteDay: 49, skipProb: 0 }, name: "Sudden one-sided change" },
  };
  const SHOW = ["speed_m_s", "arm_swing_asym_pct", "smile_amplitude", "tap_decrement_pct", "shimmer_local", "asymmetry_pct"];
  function mini(m) {
    const W = 300, H = 92, pad = 6, pts = m.series, n = pts.length;
    const lo = m.baseline - 4 * m.scale, hi = m.baseline + 4 * m.scale;
    const vmin = Math.min(lo, ...pts.map((p) => p.v)), vmax = Math.max(hi, ...pts.map((p) => p.v));
    const X = (i) => pad + (i / (n - 1)) * (W - 2 * pad), Y = (v) => H - 16 - ((v - vmin) / (vmax - vmin || 1)) * (H - 30);
    const nb = Math.max(6, Math.min(14, n - 5));
    let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="${m.label}: daily values against baseline">`;
    s += `<rect x="${pad}" y="${Y(m.baseline + 2 * m.scale)}" width="${W - 2 * pad}" height="${Math.max(1, Y(m.baseline - 2 * m.scale) - Y(m.baseline + 2 * m.scale))}" fill="var(--control)" opacity="0.16"></rect>`;
    s += `<line x1="${X(nb - 1)}" x2="${X(nb - 1)}" y1="10" y2="${H - 14}" stroke="var(--muted)" stroke-dasharray="3 3"></line>`;
    s += `<polyline fill="none" stroke="var(--ink)" stroke-width="1.2" opacity="0.55" points="${pts.map((p, i) => `${X(i)},${Y(p.v)}`).join(" ")}"></polyline>`;
    pts.forEach((p, i) => { const bad = i >= nb && p.z >= 1.5; s += `<circle cx="${X(i)}" cy="${Y(p.v)}" r="${bad ? 2.6 : 1.8}" fill="${bad ? "var(--pd)" : "var(--ink)"}" opacity="${bad ? 1 : 0.6}"></circle>`; });
    s += `<text x="${pad}" y="${H - 2}" font-size="10.5" fill="var(--muted)" font-family="var(--font-mono)">${pts[0].date.slice(5)}</text><text x="${W - pad}" y="${H - 2}" font-size="10.5" fill="var(--muted)" font-family="var(--font-mono)" text-anchor="end">${pts[n - 1].date.slice(5)}</text>`;
    return `<div class="mini"><div class="mini-head"><span>${m.label}</span>${m.sustained ? '<span class="pill bad">changed</span>' : m.acute ? '<span class="pill bad">sudden</span>' : '<span class="pill ok">usual</span>'}</div>${s}</svg></div>`;
  }
  function showWeekly(key) {
    const sc = SCEN[key], hist = BASE.simulateHistory(sc.args), ev = BASE.evaluate(hist);
    for (const k of Object.keys(SCEN)) $("sc" + k[0].toUpperCase() + k.slice(1)).classList.toggle("primary", k === key);
    $("wkTitle").textContent = `What Piko would tell the family · ${sc.name}`;
    const lvl = { urgent: "bad", discuss: "mid", watch: "mid" };
    $("wkPrompts").innerHTML = ev.patterns.length ? ev.patterns.map((p) => `<div class="prompt"><span class="pill ${lvl[p.level]}">${p.level}</span><div><b>${p.title}</b><p class="small">${p.text}</p></div></div>`).join("") :
      `<div class="prompt"><span class="pill ok">all usual</span><div><b>Nothing to report this week</b><p class="small">All ${Object.keys(ev.metrics).length} measures are within this person's usual range. Piko says nothing, so families aren't trained to ignore it.</p></div></div>`;
    const flaggedFirst = [...SHOW].sort((a, b) => ((ev.metrics[b] && (ev.metrics[b].sustained || ev.metrics[b].acute)) ? 1 : 0) - ((ev.metrics[a] && (ev.metrics[a].sustained || ev.metrics[a].acute)) ? 1 : 0));
    $("wkCharts").innerHTML = flaggedFirst.filter((k) => ev.metrics[k]).map((k) => mini(ev.metrics[k])).join("");
    const b = R.baseline_sim;
    $("wkStats").textContent = `In 200 simulated stable people over 8 weeks, ${fmt(b.stable_people_with_any_prompt_in_8_weeks_pct, 0)}% got any prompt and ${fmt(b.stable_people_with_discuss_or_urgent_prompt_pct, 0)}% got a "discuss with a doctor" prompt. A slow drift across several tests was caught in ${fmt(b.motor_drift_caught_pct, 0)}% of simulated people, a median ${b.motor_median_days_after_change_starts} days after it began. These rates depend on how much real people vary from day to day, which a pilot has to measure.`;
  }
  for (const k of Object.keys(SCEN)) $("sc" + k[0].toUpperCase() + k.slice(1)).addEventListener("click", () => showWeekly(k));

  let started = false;
  window.initExtra = function () {
    if (started) return; started = true;
    tapData.right = TAP.simulate({ rateHz: 4.8, seed: 21 }); tapData.left = TAP.simulate({ rateHz: 4.3, decrement: 0.15, seed: 22 }); renderTap();
    renderFace(FACE.analyse(FACE.simulate({})), "synthetic: typical");
    renderGait(GAIT.analyse(GAIT.simulate({ heightCm: height() }), height()), "synthetic: typical");
    showWeekly("motor");
  };
  // run once fonts are ready (canvas text uses them); the voice/spiral script may finish first
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(window.initExtra); else window.initExtra();
})();
