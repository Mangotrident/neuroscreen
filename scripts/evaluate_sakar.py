"""Train the deployable voice model on Sakar et al. 2018 (252 people, one clinic, one microphone),
evaluate it subject-independently, then test it on two datasets it has never seen:
the Italian PD dataset (our own feature extractor, run on raw audio) and the Oxford dataset (MDVP features)."""
import os as _os
RAW = _os.environ.get("NEUROSCREEN_DATA", "data/raw")
import json, os, sys, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
warnings.filterwarnings("ignore")

PORTABLE = ["jitter_local", "jitter_rap", "jitter_ppq5", "period_cv", "shimmer_local", "shimmer_db",
            "shimmer_apq3", "shimmer_apq5", "shimmer_apq11", "hnr_db"]
LOGGED = [f for f in PORTABLE if f not in ("shimmer_db", "hnr_db")]

SAKAR_MAP = {"locPctJitter": "jitter_local", "rapJitter": "jitter_rap", "ppq5Jitter": "jitter_ppq5",
             "locShimmer": "shimmer_local", "locDbShimmer": "shimmer_db", "apq3Shimmer": "shimmer_apq3",
             "apq5Shimmer": "shimmer_apq5", "apq11Shimmer": "shimmer_apq11", "meanHarmToNoiseHarmonicity": "hnr_db"}
OXFORD_MAP = {"MDVP:RAP": "jitter_rap", "MDVP:PPQ": "jitter_ppq5", "MDVP:Shimmer": "shimmer_local",
              "MDVP:Shimmer(dB)": "shimmer_db", "Shimmer:APQ3": "shimmer_apq3", "Shimmer:APQ5": "shimmer_apq5",
              "MDVP:APQ": "shimmer_apq11", "HNR": "hnr_db"}


def load_sakar():
    s = pd.read_csv(f"{RAW}/sakar3/pd_speech_features.csv")
    d = s.rename(columns=SAKAR_MAP)
    d["period_cv"] = s.stdDevPeriodPulses / s.meanPeriodPulses
    d["subject"] = s["id"]; d["label"] = s["class"]; d["sex"] = s["gender"]
    return s, d


def load_oxford():
    o = pd.read_csv(f"{RAW}/sakar2/Data - Parkinsons")
    d = o.rename(columns=OXFORD_MAP)
    d["jitter_local"] = o["MDVP:Jitter(%)"] / 100.0
    d["period_cv"] = np.nan
    d["subject"] = o["name"].str.rsplit("_", n=1).str[0]; d["label"] = o["status"]
    return o, d


def make_pre(feats):
    idx = [feats.index(f) for f in feats if f in LOGGED]
    def tf(X):
        X = np.array(X, dtype=float, copy=True); X[:, idx] = np.log(np.clip(X[:, idx], 1e-6, None)); return X
    return [FunctionTransformer(tf), SimpleImputer(strategy="median"), StandardScaler()]


def model_zoo(feats):
    return {
        "logistic": lambda: make_pipeline(*make_pre(feats), LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced")),
        "svm_rbf": lambda: make_pipeline(*make_pre(feats), SVC(C=1.0, probability=True, class_weight="balanced")),
        "lightgbm": lambda: make_pipeline(*make_pre(feats), LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7,
                                                                            min_child_samples=10, class_weight="balanced", verbose=-1, random_state=0)),
    }


def subject_metrics(y_subj, p_subj, thr=0.5):
    pred = (p_subj >= thr).astype(int)
    return dict(auc=roc_auc_score(y_subj, p_subj), bacc=balanced_accuracy_score(y_subj, pred),
                sens=float(((pred == 1) & (y_subj == 1)).sum() / (y_subj == 1).sum()),
                spec=float(((pred == 0) & (y_subj == 0)).sum() / (y_subj == 0).sum()))


def grouped(d, feats, mk, repeats=10, leaky=False):
    rows = []
    for r in range(repeats):
        cv = StratifiedKFold(5, shuffle=True, random_state=r) if leaky else StratifiedGroupKFold(5, shuffle=True, random_state=r)
        p = np.zeros(len(d))
        split = cv.split(d, d.label) if leaky else cv.split(d, d.label, d.subject)
        for tr, te in split:
            m = mk(); m.fit(d.iloc[tr][feats].values, d.iloc[tr].label.values)
            p[te] = m.predict_proba(d.iloc[te][feats].values)[:, 1]
        rec = dict(rec_auc=roc_auc_score(d.label, p), rec_bacc=balanced_accuracy_score(d.label, (p >= .5).astype(int)))
        ps = pd.Series(p).groupby(d.subject.values).mean(); ys = d.groupby("subject").label.first().loc[ps.index]
        rec.update({f"subj_{k}": v for k, v in subject_metrics(ys.values, ps.values).items()})
        rows.append(rec)
    df = pd.DataFrame(rows)
    return {k: {"mean": float(df[k].mean()), "sd": float(df[k].std())} for k in df.columns}


def external(model, d, feats, name):
    p = model.predict_proba(d[feats].values)[:, 1]
    ps = pd.Series(p).groupby(d.subject.values).mean(); ys = d.groupby("subject").label.first().loc[ps.index]
    m = subject_metrics(ys.values, ps.values)
    m["n_subjects"] = int(len(ys)); m["n_pd"] = int(ys.sum())
    print(f"  external {name:38s} AUC {m['auc']:.3f} bAcc {m['bacc']:.3f} sens {m['sens']:.2f} spec {m['spec']:.2f} (n={len(ys)})")
    return m


def main():
    raw, d = load_sakar()
    out = {"train_dataset": "Sakar et al. 2018, UCI 'Parkinson's Disease Classification' (CC BY 4.0): 252 people "
                            "(188 PD, 64 controls), 3 sustained /a/ each, one clinic and microphone"}
    all_feats = [c for c in raw.columns if c not in ("id", "class", "gender")]
    d_all = raw.copy(); d_all["subject"] = raw["id"]; d_all["label"] = raw["class"]
    print("Sakar benchmark, all 752 features:")
    mk_all = lambda: make_pipeline(SimpleImputer(), StandardScaler(), LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15,
                                                                                      class_weight="balanced", verbose=-1, random_state=0))
    out["sakar_all_features_grouped"] = grouped(d_all, all_feats, mk_all, repeats=5)
    out["sakar_all_features_leaky"] = grouped(d_all, all_feats, mk_all, repeats=5, leaky=True)
    print("  grouped", {k: round(v["mean"], 3) for k, v in out["sakar_all_features_grouped"].items()})
    print("  leaky  ", {k: round(v["mean"], 3) for k, v in out["sakar_all_features_leaky"].items()})

    print("Sakar, the 10 portable features the browser can compute:")
    out["sakar_portable_grouped"] = {}
    for name, mk in model_zoo(PORTABLE).items():
        res = grouped(d, PORTABLE, mk)
        out["sakar_portable_grouped"][name] = res
        print(f"  {name:10s}", {k: round(v["mean"], 3) for k, v in res.items()})
    subj = d.groupby("subject").agg(label=("label", "first"), sex=("sex", "first"))
    out["sakar_sex_only_auc"] = float(roc_auc_score(subj.label, subj.sex))

    # external validation: fit on ALL of Sakar, test elsewhere
    chosen = "logistic"   # chosen for transfer: linear, monotone in each feature, trivially portable to JS
    out["chosen_model"] = chosen
    ext = {}
    final = model_zoo(PORTABLE)[chosen](); final.fit(d[PORTABLE].values, d.label.values)
    it = pd.read_csv("data/italian_vowel_features.csv")
    it_a = it[it.vowel == "VA"]
    print("External validation (model never saw these people, countries or microphones):")
    ext["italian_ours_vowel_a"] = external(final, it_a, PORTABLE, "Italian, our extractor, /a/")
    praat_cols = {f"praat_{f}": f for f in PORTABLE}
    it_p = it_a.drop(columns=PORTABLE).rename(columns=praat_cols)
    ext["italian_praat_vowel_a"] = external(final, it_p, PORTABLE, "Italian, Praat reference, /a/")
    ext["italian_ours_all_vowels"] = external(final, it, PORTABLE, "Italian, our extractor, all vowels")
    # Oxford has no period-variability measure: refit without it
    feats9 = [f for f in PORTABLE if f != "period_cv"]
    final9 = model_zoo(feats9)[chosen](); final9.fit(d[feats9].values, d.label.values)
    _, ox = load_oxford()
    ext["oxford_mdvp"] = external(final9, ox, feats9, "Oxford (MDVP features, US)")
    out["external"] = ext

    coef = final[-1].coef_[0]
    out["coefficients"] = dict(zip(PORTABLE, coef.round(4).tolist()))
    print("coefficients (+ = pushes toward PD):", out["coefficients"])
    json.dump(out, open("results/sakar_eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
