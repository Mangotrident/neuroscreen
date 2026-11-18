// Runs the browser extractor on the fixtures and compares with the Python numbers.
const fs = require("fs"), path = require("path");
const V = require("../web/voice.js");
const dir = path.join(__dirname, "fixtures");
const meta = JSON.parse(fs.readFileSync(path.join(dir, "expected.json")));
let worst = {}, fails = 0;
for (const m of meta) {
  const buf = fs.readFileSync(path.join(dir, m.file));
  const x = new Float64Array(buf.buffer, buf.byteOffset, buf.length / 8);
  const { feats } = V.extract(x);
  for (const k of V.FEATURE_NAMES) {
    const a = m.py[k], b = feats[k];
    if (a === null && Number.isNaN(b)) continue;
    const rel = Math.abs(a - b) / Math.max(1e-9, Math.abs(a));
    worst[k] = Math.max(worst[k] || 0, rel);
    if (!(rel < 1e-6)) fails++;
  }
}
for (const k of V.FEATURE_NAMES) console.log(k.padEnd(20), "max relative diff", (worst[k] || 0).toExponential(2));
console.log(fails === 0 ? "PARITY OK on " + meta.length + " recordings" : fails + " mismatches");
process.exit(fails === 0 ? 0 : 1);
