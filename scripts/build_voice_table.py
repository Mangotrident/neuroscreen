"""Extract features for every sustained-vowel recording of the Italian PD dataset.
Main comparison: Parkinson's (28) vs age-matched elderly healthy controls (22). Young controls have no vowels."""
import os as _os
RAW = _os.environ.get("NEUROSCREEN_DATA", "data/raw")
import glob, os, re, sys
import numpy as np, pandas as pd, soundfile as sf, parselmouth
from parselmouth.praat import call
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from neuroscreen.voice_features import extract, SR
import librosa

ROOT = f"{RAW}/italian/data"
GROUPS = {"28 People with Parkinson's disease": 1, "22 Elderly Healthy Control": 0}

def praat_feats(x):
    snd = parselmouth.Sound(x, sampling_frequency=SR)
    pp = call(snd, "To PointProcess (periodic, cc)", 60, 500)
    harm = call(snd, "To Harmonicity (cc)", 0.01, 60, 0.1, 1.0)
    j = lambda m: call(pp, f"Get jitter ({m})", 0, 0, 0.0001, 0.02, 1.3)
    sh = lambda m: call([snd, pp], f"Get shimmer ({m})", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    return {
        "praat_jitter_local": j("local"), "praat_jitter_rap": j("rap"), "praat_jitter_ppq5": j("ppq5"),
        "praat_period_cv": call(pp, "Get stdev period", 0, 0, 0.0001, 0.02, 1.3) / call(pp, "Get mean period", 0, 0, 0.0001, 0.02, 1.3),
        "praat_shimmer_local": sh("local"), "praat_shimmer_db": sh("local_dB"), "praat_shimmer_apq3": sh("apq3"),
        "praat_shimmer_apq5": sh("apq5"), "praat_shimmer_apq11": sh("apq11"),
        "praat_hnr_db": call(harm, "Get mean", 0, 0),
    }

def one(args):
    path, label, subject = args
    x, sr = sf.read(path, dtype="float64")
    if x.ndim > 1: x = x.mean(1)
    f = extract(x, sr)
    x16 = librosa.resample(x, orig_sr=sr, target_sr=SR) if sr != SR else x
    from neuroscreen.voice_features import select_region
    x16 = x16 - x16.mean(); x16 = x16 / (np.abs(x16).max() + 1e-12)
    try: f.update(praat_feats(select_region(x16)))
    except Exception as e: print('praat fail', path, e)
    base = os.path.basename(path)
    m = re.match(r"(V[AEIOU])(\d)\D+?(\d{2})([MF])", base)
    f.update(dict(path=path, subject=subject, label=label, vowel=m.group(1) if m else None,
                  take=int(m.group(2)) if m else None, sex=m.group(4) if m else None,
                  orig_sr=sr))
    return f

jobs = []
for g, lab in GROUPS.items():
    for wav in glob.glob(f"{ROOT}/{g}/**/V[AEIOU][12]*.wav", recursive=True):
        subject = os.path.relpath(os.path.dirname(wav), ROOT)
        jobs.append((wav, lab, subject))
print(len(jobs), "recordings")
with Pool(2) as p:
    rows = p.map(one, jobs, chunksize=8)
df = pd.DataFrame(rows)
df.to_csv("data/italian_vowel_features.csv", index=False)
print(df.groupby("label").subject.nunique(), df.shape)
print(df.isna().mean().sort_values().tail(6))
