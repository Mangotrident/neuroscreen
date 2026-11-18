(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? "–" : Number(v).toFixed(d);

  // ---------------- findings strip ----------------
  const R = RESULTS;
  const findings = [
    { num: fmt(R.portable.subj_auc, 2), tag: "browser model, unseen people", text: `AUC on ${R.sakar_n} people held out by person (Sakar 2018, 10 voice measures the browser computes).` },
    { num: fmt(R.external.oxford.auc, 2), tag: "different country", text: `AUC on the Oxford cohort (${R.external.oxford.n_subjects} people, US recordings) it never trained on. It ranks people well, but its threshold needs recalibrating.` },
    { num: `${fmt(R.leak.leaky_subj, 2)}<span class="arrow"> → </span>${fmt(R.leak.grouped_subj, 2)}`, tag: "leakage", text: "The full 752-feature model's AUC with a person's recordings allowed on both sides of the split, then with each person kept on one side." },
    { num: fmt(R.shortcut.silence_only_auc, 2), tag: "shortcut", warn: true, text: "AUC from the background silence of each Italian file, with no voice at all. The near-perfect voice score there is a recording artefact." },
  ];
  $("findings").innerHTML = findings.map((f) => `<div class="finding ${f.warn ? "warn" : ""}"><div class="tag">${f.tag}</div><div class="num">${f.num}</div><p>${f.text}</p></div>`).join("");

  // ---------------- canvas helpers ----------------
  function setupCanvas(cv) {
    const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight || Number(cv.getAttribute("height"));
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
    const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); return { ctx, w, h };
  }

  // ---------------- voice ----------------
  const BM = [
    { f: "jitter_local", name: "Jitter (local)", unit: "%", k: 100, d: 2, log: true },
    { f: "jitter_rap", name: "Jitter (RAP)", unit: "%", k: 100, d: 2, log: true },
    { f: "jitter_ppq5", name: "Jitter (PPQ5)", unit: "%", k: 100, d: 2, log: true },
    { f: "period_cv", name: "Period variability", unit: "%", k: 100, d: 2, log: true },
    { f: "shimmer_local", name: "Shimmer (local)", unit: "%", k: 100, d: 1, log: true },
    { f: "shimmer_db", name: "Shimmer", unit: "dB", k: 1, d: 2, log: false },
    { f: "shimmer_apq3", name: "Shimmer (APQ3)", unit: "%", k: 100, d: 1, log: true },
    { f: "shimmer_apq5", name: "Shimmer (APQ5)", unit: "%", k: 100, d: 1, log: true },
    { f: "shimmer_apq11", name: "Shimmer (APQ11)", unit: "%", k: 100, d: 1, log: true },
    { f: "hnr_db", name: "Harmonics-to-noise", unit: "dB", k: 1, d: 1, log: false },
  ];

  function strip(bm, value) {
    const ref = MODEL.reference[bm.f], c = ref.control, p = ref.parkinsons;
    const vals = [c["10"], c["90"], p["10"], p["90"]]; if (Number.isFinite(value)) vals.push(value);
    let lo = Math.min(...vals), hi = Math.max(...vals);
    const T = (v) => bm.log ? Math.log(Math.max(v, 1e-6)) : v;
    let a = T(lo), b = T(hi); const pad = (b - a) * 0.08 || 1; a -= pad; b += pad;
    const X = (v) => (100 * (T(v) - a) / (b - a)).toFixed(2);
    const band = (q, color, y) => `<rect x="${X(q["25"])}%" y="${y}" width="${Math.max(0.5, X(q["75"]) - X(q["25"]))}%" height="7" rx="3" fill="${color}" opacity="0.75"></rect>
      <rect x="${X(q["50"])}%" y="${y - 1}" width="2" height="9" fill="${color}"></rect>`;
    const you = Number.isFinite(value) ? `<rect x="${X(value)}%" y="1" width="3" height="24" rx="1.5" fill="var(--ink)" transform="translate(-1.5,0)"></rect>` : "";
    return `<svg width="100%" height="26" role="img" aria-label="${bm.name}: your value against the control and Parkinson's ranges">
      <line x1="0" x2="100%" y1="13" y2="13" stroke="var(--grid)"></line>${band(c, "var(--control)", 4)}${band(p, "var(--pd)", 15)}${you}</svg>`;
  }

  function renderVoice(feats, debug, sourceLabel) {
    const out = VOICE.predict(MODEL, feats);
    $("voiceSource").textContent = sourceLabel;
    $("scoreNum").textContent = fmt(out.prob, 2);
    $("scoreMark").style.left = (100 * out.prob).toFixed(1) + "%";
    const near = Math.abs(out.prob - 0.5) < 0.1;
    $("scoreText").textContent = near ? "Between the two groups: this voice doesn't lean either way." :
      out.prob > 0.5 ? "These measures sit closer to the Parkinson's group of the training clinic." : "These measures sit closer to the control group of the training clinic.";
    $("biomarkers").innerHTML = BM.map((bm) => {
      const v = feats[bm.f], shown = Number.isFinite(v) ? (v * bm.k).toFixed(bm.d) + " " + bm.unit : "–";
      return `<div class="bm"><div class="name">${bm.name}</div><div class="val">${shown}</div><div>${strip(bm, v)}</div></div>`;
    }).join("");
    const top = Object.entries(out.contributions).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 2)
      .map(([f, c]) => `${BM.find((b) => b.f === f).name.toLowerCase()} (${c > 0 ? "toward Parkinson's" : "toward control"})`);
    $("voiceNote").textContent = `Biggest influences on this score: ${top.join(" and ")}. Reference ranges are per person from the training clinic (${R.sakar_n} people). Cycles found: ${debug.times.length}.`;
    drawWave(debug);
  }

  function drawWave(debug) {
    const cv = $("waveCanvas"); const { ctx, w, h } = setupCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const x = debug.region; if (!x || !x.length) return;
    const SRv = VOICE.SR, span = Math.round(0.06 * SRv), start = Math.max(0, Math.floor(x.length / 2 - span / 2));
    let peak = 1e-9; for (let i = start; i < start + span && i < x.length; i++) peak = Math.max(peak, Math.abs(x[i]));
    ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
    ctx.strokeStyle = css("--accent"); ctx.globalAlpha = 0.85;
    for (const t of debug.times) {
      const s = t * SRv; if (s < start || s > start + span) continue;
      const px = ((s - start) / span) * w; ctx.beginPath(); ctx.moveTo(px, 6); ctx.lineTo(px, 16); ctx.stroke();
    }
    ctx.globalAlpha = 1; ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.3; ctx.beginPath();
    for (let i = 0; i < span && start + i < x.length; i++) {
      const px = (i / span) * w, py = h / 2 - (x[start + i] / peak) * (h / 2 - 22);
      i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    }
    ctx.stroke();
    ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono");
    ctx.fillText("60 ms · ticks = vocal-fold cycles", 8, h - 7);
  }

  function setVoiceBusy(b, msg, isErr) {
    for (const id of ["recBtn", "synSteady", "synRough"]) $(id).disabled = b;
    $("voiceStatus").textContent = msg || ""; $("voiceStatus").classList.toggle("err", !!isErr);
  }

  function analyseSamples(x16, label) {
    const { feats, debug } = VOICE.extract(x16);
    if (!Number.isFinite(feats.jitter_local)) {
      setVoiceBusy(false, "Couldn't find a steady voiced sound. Hold one vowel for 4 to 5 seconds, close to the microphone, in a quiet room.", true);
      return;
    }
    renderVoice(feats, debug, label);
    setVoiceBusy(false, "");
  }

  async function to16k(buffer) {
    const len = Math.ceil(buffer.duration * VOICE.SR);
    const off = new OfflineAudioContext(1, len, VOICE.SR);
    const src = off.createBufferSource(); src.buffer = buffer; src.connect(off.destination); src.start();
    const out = await off.startRendering(); return out.getChannelData(0);
  }

  $("voiceFile").addEventListener("change", async (e) => {
    const file = e.target.files[0]; if (!file) return;
    setVoiceBusy(true, "Analysing " + file.name + "…");
    try {
      const buf = await file.arrayBuffer();
      const decoded = await new OfflineAudioContext(1, 1, VOICE.SR).decodeAudioData(buf);
      let x = decoded.getChannelData(0);
      if (decoded.numberOfChannels > 1) { const y = decoded.getChannelData(1); x = x.map((v, i) => 0.5 * (v + y[i])); }
      if (decoded.sampleRate !== VOICE.SR) x = await to16k(decoded);
      setTimeout(() => analyseSamples(x, file.name), 20);
    } catch (err) {
      setVoiceBusy(false, "Couldn't read that file. Try a WAV, M4A or MP3 recording.", true);
    }
    e.target.value = "";
  });

  $("recBtn").addEventListener("click", async () => {
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 } });
    } catch (err) {
      setVoiceBusy(false, "The microphone is blocked here. Upload a recording instead (a phone voice memo works), or open demo.html from the repo in your own browser to record live.", true);
      return;
    }
    const ac = new (window.AudioContext || window.webkitAudioContext)();
    const src = ac.createMediaStreamSource(stream), proc = ac.createScriptProcessor(4096, 1, 1), chunks = [];
    proc.onaudioprocess = (ev) => chunks.push(new Float32Array(ev.inputBuffer.getChannelData(0)));
    src.connect(proc); proc.connect(ac.destination);
    let left = 5; setVoiceBusy(true, `Recording… hold "aaah" for ${left} s`);
    const timer = setInterval(() => { left -= 1; if (left > 0) $("voiceStatus").textContent = `Recording… hold "aaah" for ${left} s`; }, 1000);
    await new Promise((r) => setTimeout(r, 5200));
    clearInterval(timer); proc.disconnect(); src.disconnect(); stream.getTracks().forEach((t) => t.stop());
    const n = chunks.reduce((a, c) => a + c.length, 0), all = new Float32Array(n); let o = 0; for (const c of chunks) { all.set(c, o); o += c.length; }
    const buf = ac.createBuffer(1, n, ac.sampleRate); buf.copyToChannel(all, 0); ac.close();
    $("voiceStatus").textContent = "Analysing…";
    const x16 = await to16k(buf);
    setTimeout(() => analyseSamples(x16, "your recording"), 20);
  });

  $("synSteady").addEventListener("click", () => { setVoiceBusy(true, "Synthesising…"); setTimeout(() => analyseSamples(VOICE.simulate({ jitter: 0.003, shimmer: 0.025, noiseDb: -35, seed: 11 }), "synthetic steady voice"), 20); });
  $("synRough").addEventListener("click", () => { setVoiceBusy(true, "Synthesising…"); setTimeout(() => analyseSamples(VOICE.simulate({ jitter: 0.015, shimmer: 0.09, noiseDb: -18, seed: 12 }), "synthetic rough voice"), 20); });

  // ---------------- spiral ----------------
  const sc = $("spiralCanvas");
  let strokes = [], drawing = false, current = null, lastResult = null, lastPoints = null;

  function drawSpiral() {
    const { ctx, w, h } = setupCanvas(sc);
    ctx.clearRect(0, 0, w, h);
    const step = w / 16; ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1;
    for (let i = 1; i < 16; i++) { ctx.beginPath(); ctx.moveTo(i * step, 0); ctx.lineTo(i * step, h); ctx.stroke(); ctx.beginPath(); ctx.moveTo(0, i * step); ctx.lineTo(w, i * step); ctx.stroke(); }
    const scale = w / (sc._logical || w);
    if (lastResult && lastResult.fit && lastPoints === strokes) {
      const f = lastResult.fit; ctx.setLineDash([5, 5]); ctx.strokeStyle = css("--accent"); ctx.globalAlpha = 0.6; ctx.lineWidth = 1.4; ctx.beginPath();
      const t0 = Math.min(f.th[0], f.th[f.th.length - 1]), t1 = Math.max(f.th[0], f.th[f.th.length - 1]);
      for (let k = 0; k <= 400; k++) { const th = t0 + (t1 - t0) * k / 400, r = f.a + f.b * th; const px = (f.cx + r * Math.cos(th)) * scale, py = (f.cy + r * Math.sin(th)) * scale; k ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
      ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
      const P = lastResult.points, res = f.res, Rr = lastResult.R, cC = css("--control"), cP = css("--pd");
      ctx.lineWidth = 2.4; ctx.lineCap = "round";
      for (let i = 1; i < P.length; i++) {
        if (P[i].t - P[i - 1].t > 30) continue;
        const e = Math.min(1, Math.abs(res[i]) / (0.04 * Rr));
        ctx.strokeStyle = e > 0.5 ? cP : cC; ctx.globalAlpha = 0.45 + 0.55 * e;
        ctx.beginPath(); ctx.moveTo(P[i - 1].x * scale, P[i - 1].y * scale); ctx.lineTo(P[i].x * scale, P[i].y * scale); ctx.stroke();
      }
      ctx.globalAlpha = 1;
    } else {
      ctx.strokeStyle = css("--ink"); ctx.lineWidth = 2.2; ctx.lineCap = "round"; ctx.lineJoin = "round";
      for (const s of strokes) { ctx.beginPath(); s.forEach((p, i) => i ? ctx.lineTo(p.x * scale, p.y * scale) : ctx.moveTo(p.x * scale, p.y * scale)); ctx.stroke(); }
    }
  }

  function renderSpiral(result, label) {
    $("spiralSource").textContent = label;
    if (result.error) { $("spiralStatus").textContent = result.error; $("spiralStatus").classList.add("err"); return; }
    $("spiralStatus").textContent = ""; $("spiralStatus").classList.remove("err");
    const m = result.metrics;
    const tiles = [
      [fmt(100 * m.tremor_band_share, 0) + "%", "of the wobble sits in the 4–12 Hz tremor band"],
      [m.tremor_band_share > 0.5 ? fmt(m.tremor_peak_hz, 1) + " Hz" : "none", "dominant tremor frequency"],
      [fmt(m.radial_irregularity_pct, 2) + "%", "average drift from the ideal spiral, as a share of its radius"],
      [fmt(m.speed_cv, 2), "speed variability (lower is steadier)"],
      [fmt(m.duration_s, 1) + " s", `drawing time · ${fmt(m.turns, 1)} turns`],
      [String(m.pen_lifts), "pen lifts"],
    ];
    $("spiralTiles").innerHTML = tiles.map(([v, k]) => `<div class="tile"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");
    $("spiralNote").textContent = `These are measurements, not a diagnosis. On tablet recordings of ${R.spiral.n_people} people, a spiral-kinematics model reached AUC ${fmt(R.spiral.static_spiral_only.auc, 2)} (static spiral) on held-out people, but its features depend on the tablet, so this browser test reports the measures only.`;
    lastResult = result; lastPoints = strokes; drawSpiral(); drawSpectrum(result);
    $("spiralHint").textContent = "Dashed line: best-fit spiral. Red segments drift furthest from it.";
  }

  function drawSpectrum(result) {
    const cv = $("specCanvas"); const { ctx, w, h } = setupCanvas(cv); ctx.clearRect(0, 0, w, h);
    const { freqs, power } = result.spectrum, padL = 8, padB = 20, fMax = 20;
    const X = (f) => padL + (f / fMax) * (w - 2 * padL), pm = Math.max(...power.filter((_, i) => freqs[i] >= 1)) || 1;
    ctx.fillStyle = css("--accent"); ctx.globalAlpha = 0.09; ctx.fillRect(X(4), 4, X(12) - X(4), h - padB - 4); ctx.globalAlpha = 1;
    ctx.strokeStyle = css("--grid"); ctx.beginPath(); ctx.moveTo(padL, h - padB); ctx.lineTo(w - padL, h - padB); ctx.stroke();
    ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono");
    for (const f of [0, 4, 8, 12, 16, 20]) { ctx.fillText(String(f), X(f) - 4, h - 6); }
    ctx.fillText("Hz", w - padL - 16, h - padB - 6);
    ctx.fillText("tremor band", X(4) + 4, 14);
    ctx.beginPath(); ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.5;
    freqs.forEach((f, i) => { const y = h - padB - (power[i] / pm) * (h - padB - 22); i ? ctx.lineTo(X(f), y) : ctx.moveTo(X(f), y); });
    ctx.stroke();
  }

  function toLogical(e) { const r = sc.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }
  sc.addEventListener("pointerdown", (e) => {
    if (lastResult && lastPoints === strokes) { strokes = []; lastResult = null; }
    sc._logical = sc.clientWidth; drawing = true; sc.setPointerCapture(e.pointerId);
    const p = toLogical(e); current = [{ x: p.x, y: p.y, t: e.timeStamp, p: e.pressure || 0.5 }]; strokes.push(current);
    $("spiralSource").textContent = "your drawing"; $("spiralHint").textContent = "Lift to finish. Analysis runs when you stop for a moment.";
    clearTimeout(sc._t);
  });
  sc.addEventListener("pointermove", (e) => {
    if (!drawing) return;
    const evs = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
    for (const ev of evs) { const p = toLogical(ev); current.push({ x: p.x, y: p.y, t: ev.timeStamp, p: ev.pressure || 0.5 }); }
    drawSpiral();
  });
  const endStroke = () => {
    if (!drawing) return; drawing = false;
    clearTimeout(sc._t); sc._t = setTimeout(() => renderSpiral(SPIRAL.analyse(strokes), "your drawing"), 900);
  };
  sc.addEventListener("pointerup", endStroke); sc.addEventListener("pointercancel", endStroke);

  function showSimulated(opts, label) {
    const w = sc.clientWidth; sc._logical = w;
    const sim = SPIRAL.simulate(opts);
    strokes = sim.map((s) => s.map((p) => ({ ...p, x: (p.x / 400) * w, y: (p.y / 400) * w })));
    renderSpiral(SPIRAL.analyse(strokes), label);
  }
  $("clearSpiral").addEventListener("click", () => { strokes = []; lastResult = null; drawSpiral(); $("spiralTiles").innerHTML = ""; $("spiralNote").textContent = ""; $("spiralSource").textContent = "your drawing"; $("spiralHint").textContent = "Draw here: start at the centre, go outward."; const { ctx, w, h } = setupCanvas($("specCanvas")); ctx.clearRect(0, 0, w, h); sc.focus(); });
  $("synSpiralSteady").addEventListener("click", () => showSimulated({}, "synthetic steady hand"));
  $("synSpiralTremor").addEventListener("click", () => showSimulated({ tremorHz: 5, tremorAmp: 0.02 }, "synthetic 5 Hz tremor"));
  window.addEventListener("resize", () => { clearTimeout(window._rz); window._rz = setTimeout(() => { if (lastResult && lastPoints === strokes) showSimulatedRescale(); else drawSpiral(); }, 150); });
  function showSimulatedRescale() { drawSpiral(); }

  // ---------------- evidence charts ----------------
  function bars(rows, opts = {}) {
    const W = 520, rowH = 34, top = 6, left = opts.left || 190, right = 46, H = top + rows.length * rowH + 26;
    const X = (v) => left + v * (W - left - right);
    let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="${opts.aria || "bar chart"}" style="max-width:${W}px;font-family:var(--font-body)">`;
    for (const v of [0, 0.5, 1]) s += `<line x1="${X(v)}" x2="${X(v)}" y1="${top}" y2="${H - 22}" stroke="var(--grid)" ${v === 0.5 ? 'stroke-dasharray="4 3"' : ""}></line><text x="${X(v)}" y="${H - 6}" font-size="11" fill="var(--muted)" text-anchor="middle" font-family="var(--font-mono)">${v.toFixed(1)}</text>`;
    s += `<text x="${X(0.5) + 4}" y="${H - 22 - 4}" font-size="10" fill="var(--muted)">chance</text>`;
    rows.forEach((r, i) => {
      const y = top + i * rowH;
      s += `<text x="${left - 10}" y="${y + 20}" font-size="12.5" fill="var(--ink)" text-anchor="end">${r.label}</text>`;
      s += `<rect x="${X(0)}" y="${y + 8}" width="${X(r.value) - X(0)}" height="16" rx="3" fill="${r.color}"></rect>`;
      s += `<text x="${X(r.value) + 6}" y="${y + 21}" font-size="12.5" fill="var(--ink)" font-family="var(--font-mono)">${r.value.toFixed(3)}</text>`;
    });
    return s + "</svg>";
  }
  $("leakText").textContent = `Sakar 2018, ${R.sakar_n} people with 3 recordings each, all 752 published features, gradient-boosted trees. Leaky: random 5-fold over recordings. Held out: each person's recordings stay together.`;
  $("leakChart").innerHTML = bars([
    { label: "Leaky split, per recording", value: R.leak.leaky_rec, color: "var(--pd)" },
    { label: "Held out, per recording", value: R.leak.grouped_rec, color: "var(--accent)" },
    { label: "Leaky split, per person", value: R.leak.leaky_subj, color: "var(--pd)" },
    { label: "Held out, per person", value: R.leak.grouped_subj, color: "var(--accent)" },
  ], { aria: "AUC with and without person-level leakage" });

  const dir = R.shortcut.biomarker_directions;
  $("shortcutText").textContent = `28 Parkinson's vs 22 age-matched controls. The typical Parkinson's file has a background noise floor about ${fmt(R.shortcut.noise_floor_db_median.pd - R.shortcut.noise_floor_db_median.control, 0)} dB louder than a control file, and jitter, shimmer and HNR all point the wrong way (jitter is ${fmt(100 * dir.jitter_local.pd_median, 2)}% in Parkinson's vs ${fmt(100 * dir.jitter_local.control_median, 2)}% in controls). The model was hearing the recording setup.`;
  $("shortcutChart").innerHTML = bars([
    { label: "Voice model, in-dataset", value: R.shortcut.in_dataset_auc, color: "var(--pd)" },
    { label: "Background silence only", value: R.shortcut.silence_only_auc, color: "var(--pd)" },
    { label: "Clinic-trained model", value: R.shortcut.physiology_model_transfer_auc, color: "var(--accent)" },
  ], { aria: "AUC of three models on the Italian dataset" });

  const ex = R.external;
  const pill = (auc) => auc >= 0.85 ? `<span class="pill ok">strong ranking</span>` : auc >= 0.7 ? `<span class="pill mid">moderate</span>` : `<span class="pill bad">fails</span>`;
  const extRows = [
    ["Sakar 2018 (Turkey), held out by person", `${R.sakar_n} (${R.sakar_pd} PD)`, R.portable.subj_auc, R.portable.subj_sens, R.portable.subj_spec, "Same clinic as training. 5-fold × 10 repeats."],
    ["Oxford (US), never seen", `${ex.oxford.n_subjects} (${ex.oxford.n_pd} PD)`, ex.oxford.auc, ex.oxford.sens, ex.oxford.spec, "MDVP software instead of Praat; no period-variability measure."],
    ["Italian (Bari), never seen", `${ex.italian.n_subjects} (${ex.italian.n_pd} PD)`, ex.italian.auc, ex.italian.sens, ex.italian.spec, "Run through the browser extractor on raw audio. Recording setup differs by group."],
  ];
  $("extTable").innerHTML = `<thead><tr><th>Test set</th><th>People</th><th>AUC</th><th>Sensitivity</th><th>Specificity</th><th></th><th>Notes</th></tr></thead><tbody>` +
    extRows.map((r) => `<tr><td>${r[0]}</td><td class="n">${r[1]}</td><td class="n">${fmt(r[2], 3)}</td><td class="n">${fmt(r[3], 2)}</td><td class="n">${fmt(r[4], 2)}</td><td>${pill(r[2])}</td><td class="small muted">${r[5]}</td></tr>`).join("") + "</tbody>";
  $("extText").textContent = "Sensitivity and specificity use the 0.5 threshold set in the training clinic. On Oxford the model ranks people well (AUC above 0.9) but flags too few of them at that threshold, so each new site needs its own threshold. On the Italian set it fails because that dataset's groups were recorded differently.";

  const SR = R.synthetic;
  const standsRows = [
    ["Voice", "Own extractor; tracks Praat (Spearman 0.61 to 0.87); browser equals Python to 1e-11", `Logistic model trained on ${R.sakar_n} people. AUC ${fmt(R.portable.subj_auc, 2)} on held-out people, ${fmt(R.external.oxford.auc, 2)} on an external US cohort`, "Score + 10 measures"],
    ["Spiral", "Pen or finger path", `Recovers injected 5 Hz and 9 Hz tremor exactly. Tablet model AUC ${fmt(R.spiral.static_spiral_only.auc, 2)} (not reused: device-specific)`, "Measures"],
    ["Tap test", "Touch events", `Recovers injected speed (${fmt(SR.tap["rate_hz (inject 5)"].measured, 2)} vs 5 taps/s), rhythm and slowing`, "Measures"],
    ["Smile", "Google MediaPipe Face Mesh, 468 points", `Recovers injected asymmetry (${fmt(SR.face["asymmetry_pct (inject 40%)"].measured, 1)}% vs 40%), cheek raise and blink rate in synthetic faces`, "Measures"],
    ["Walking", "Google MediaPipe Pose, 33 points", `Recovers injected speed (${fmt(SR.gait["speed_m_s (inject 1.2)"].measured, 2)} vs 1.20 m/s), cadence, step length, arm asymmetry and lean`, "Measures"],
    ["Weekly engine", "Person's own baseline", `200 simulated stable people: ${fmt(R.baseline_sim.stable_people_with_any_prompt_in_8_weeks_pct, 0)}% got any prompt in 8 weeks. Multi-test drift caught in ${fmt(R.baseline_sim.motor_drift_caught_pct, 0)}%`, "Plain-language prompts"],
  ];
  $("standsTable").innerHTML = `<thead><tr><th>Check</th><th>Perception</th><th>Validated how</th><th>Output</th></tr></thead><tbody>` + standsRows.map((r) => `<tr><td>${r[0]}</td><td class="small">${r[1]}</td><td class="small">${r[2]}</td><td class="small muted">${r[3]}</td></tr>`).join("") + "</tbody>";

  // ---------------- initial state: examples already analysed ----------------
  function init() {
    analyseSamples(VOICE.simulate({ jitter: 0.003, shimmer: 0.025, noiseDb: -35, seed: 11 }), "synthetic steady voice");
    showSimulated({ tremorHz: 5, tremorAmp: 0.02 }, "synthetic 5 Hz tremor");
    if (window.initExtra) window.initExtra();
  }
  const VOICE = window.NeuroVoice, SPIRAL = window.NeuroSpiral;
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(init); else init();
})();
