"""Is the Italian dataset's near-perfect score real? Three checks:
1. a model that only sees the background silence of each file (no voice at all),
2. the direction of each voice biomarker (Parkinson's should raise jitter/shimmer and lower HNR),
3. the Sakar-trained physiology model applied to it (from evaluate_sakar.py)."""
import json, os, sys, re
import numpy as np, pandas as pd, soundfile as sf, librosa
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate_voice import grouped_cv

df = pd.read_csv("data/italian_vowel_features.csv")
rows = []
for p in df.path:
    x, sr = sf.read(p); x = x if x.ndim == 1 else x.mean(1)
    x = librosa.resample(x, orig_sr=sr, target_sr=16000) if sr != 16000 else x
    fr = librosa.util.frame(x, frame_length=400, hop_length=160)
    e = 10 * np.log10((fr ** 2).mean(0) + 1e-12)
    quiet = fr[:, e <= np.percentile(e, 10)]
    S = np.abs(np.fft.rfft(quiet * np.hanning(400)[:, None], axis=0)).mean(1)
    f = np.fft.rfftfreq(400, 1 / 16000)
    rows.append(dict(noise_floor_db=np.percentile(e, 5), silence_centroid_hz=(S * f).sum() / S.sum() if S.sum() > 0 else np.nan))
d = pd.concat([df, pd.DataFrame(rows)], axis=1)
mk = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))
res, _ = grouped_cv(d, mk, feats=["noise_floor_db", "silence_centroid_hz"])

subj = d.groupby("subject")[["label", "noise_floor_db", "silence_centroid_hz", "jitter_local", "shimmer_local", "hnr_db"]].mean()
expected = {"jitter_local": "higher in PD", "shimmer_local": "higher in PD", "hnr_db": "lower in PD"}
directions = {}
for f, exp in expected.items():
    pd_med = float(subj[subj.label == 1][f].median()); hc_med = float(subj[subj.label == 0][f].median())
    observed = "higher in PD" if pd_med > hc_med else "lower in PD"
    directions[f] = {"pd_median": pd_med, "control_median": hc_med, "expected": exp, "observed": observed}

voice = json.load(open("results/voice_eval.json"))
sak = json.load(open("results/sakar_eval.json"))
out = {
    "in_dataset_auc": voice["models_all_vowels"][voice["selected_model"]]["auc"]["mean"],
    "silence_only_auc": res["auc"]["mean"], "silence_only_bacc": res["balanced_accuracy"]["mean"],
    "physiology_model_transfer_auc": sak["external"]["italian_ours_vowel_a"]["auc"],
    "noise_floor_db_median": {"pd": float(subj[subj.label == 1].noise_floor_db.median()), "control": float(subj[subj.label == 0].noise_floor_db.median())},
    "biomarker_directions": directions,
}
json.dump(out, open("results/shortcut.json", "w"), indent=1)
print(json.dumps(out, indent=1))
