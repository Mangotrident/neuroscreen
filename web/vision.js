/* Runs Google's MediaPipe Face Mesh and Pose models in the browser (WebAssembly, on-device) and turns
   a video file or live camera into landmark time series for face.js and gait.js.
   Model files load from window.NEUROSCREEN_MEDIAPIPE_BASE: "vendor/" when shipped with the page
   (offline, for the robot), or the jsDelivr copy of the same npm packages. */
(function (root) {
  "use strict";
  const VERSIONS = { face_mesh: "0.4.1633559619", pose: "0.5.1675469404" };
  const base = (pkg) => {
    const b = root.NEUROSCREEN_MEDIAPIPE_BASE || "vendor/";
    return b.startsWith("http") ? `${b}@mediapipe/${pkg}@${VERSIONS[pkg]}/` : `${b}${pkg}/`;
  };
  // some hosts only serve common file types: the build can append ".wasm" to binary model files (they are fetched as raw bytes, so the type does not matter)
  const RENAMED = /\.(binarypb|data|tflite)$/;
  const fileName = (f) => (root.NEUROSCREEN_MEDIAPIPE_RENAMED && RENAMED.test(f) ? f + ".wasm" : f);
  const loaded = {};
  function loadScript(src) {
    return new Promise((res, rej) => { const s = document.createElement("script"); s.src = src; s.crossOrigin = "anonymous"; s.onload = res; s.onerror = () => rej(new Error("Couldn't load " + src)); document.head.appendChild(s); });
  }

  async function model(kind) {
    if (loaded[kind]) return loaded[kind];
    if (kind === "face") {
      await loadScript(base("face_mesh") + "face_mesh.js");
      const m = new root.FaceMesh({ locateFile: (f) => base("face_mesh") + fileName(f) });
      m.setOptions({ maxNumFaces: 1, refineLandmarks: false, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
      await m.initialize();
      loaded[kind] = m;
    } else {
      await loadScript(base("pose") + "pose.js");
      const m = new root.Pose({ locateFile: (f) => base("pose") + fileName(f) });
      m.setOptions({ modelComplexity: 0, smoothLandmarks: true, enableSegmentation: false, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
      await m.initialize();
      loaded[kind] = m;
    }
    return loaded[kind];
  }

  // one frame -> landmarks in pixels (or null)
  async function detect(kind, m, source, W, H) {
    let out = null;
    m.onResults((r) => {
      if (kind === "face") { const f = r.multiFaceLandmarks && r.multiFaceLandmarks[0]; out = f ? f.map((p) => [p.x * W, p.y * H, p.z * W]) : null; }
      else { const p = r.poseLandmarks; out = p ? p.map((q) => [q.x * W, q.y * H, q.z * W, q.visibility ?? 1]) : null; }
    });
    await m.send({ image: source });
    return out;
  }

  // step through a video file at a fixed frame rate (deterministic, independent of playback speed)
  async function fromFile(kind, file, { fps = 15, maxSeconds = 20, onProgress } = {}) {
    const m = await model(kind);
    const v = document.createElement("video");
    v.muted = true; v.playsInline = true; v.preload = "auto"; v.src = URL.createObjectURL(file);
    await new Promise((res, rej) => { v.onloadedmetadata = res; v.onerror = () => rej(new Error("This browser can't play that video. Try an MP4 (H.264) file.")); });
    const W = v.videoWidth, H = v.videoHeight, dur = Math.min(v.duration, maxSeconds);
    const frames = [];
    for (let t = 0; t < dur; t += 1 / fps) {
      v.currentTime = t;
      await new Promise((res) => { v.onseeked = res; });
      frames.push({ t, lm: await detect(kind, m, v, W, H) });
      if (onProgress) onProgress(t / dur);
    }
    URL.revokeObjectURL(v.src);
    return { frames, width: W, height: H };
  }

  // live camera for a fixed time (used on the robot or a local copy; hosted viewers block the camera)
  async function fromCamera(kind, videoEl, { seconds = 10, onProgress } = {}) {
    const m = await model(kind);
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" }, audio: false });
    videoEl.srcObject = stream; await videoEl.play();
    const W = videoEl.videoWidth, H = videoEl.videoHeight, frames = [], t0 = performance.now();
    while ((performance.now() - t0) / 1000 < seconds) {
      const t = (performance.now() - t0) / 1000;
      frames.push({ t, lm: await detect(kind, m, videoEl, W, H) });
      if (onProgress) onProgress(t / seconds);
      await new Promise((r) => requestAnimationFrame(r));
    }
    stream.getTracks().forEach((tr) => tr.stop());
    return { frames, width: W, height: H };
  }

  root.NeuroVision = { model, fromFile, fromCamera };
})(typeof self !== "undefined" ? self : this);
