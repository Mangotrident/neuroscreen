"""Dump 16 kHz audio + Python features for a sample of recordings so tests/parity.js can compare."""
import sys, os, json, numpy as np, pandas as pd, soundfile as sf, librosa
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuroscreen.voice_features import extract, FEATURE_NAMES
df = pd.read_csv("data/italian_vowel_features.csv").sample(24, random_state=1)
meta = []
for i, p in enumerate(df.path):
    x, sr = sf.read(p, dtype="float64"); x = x if x.ndim == 1 else x.mean(1)
    x16 = librosa.resample(x, orig_sr=sr, target_sr=16000) if sr != 16000 else x
    x16.astype("<f8").tofile(f"tests/fixtures/{i}.f64")
    f = extract(x16, 16000)
    meta.append({"file": f"{i}.f64", "py": {k: (None if np.isnan(f[k]) else float(f[k])) for k in FEATURE_NAMES}})
json.dump(meta, open("tests/fixtures/expected.json", "w"))
print("wrote", len(meta))
