"""Subject-independent evaluation of the voice module.

Every split keeps all recordings of a person on one side (StratifiedGroupKFold), repeated 20 times.
We also report what a leaky random split would have claimed, a sex-only baseline, a run without the
10 Parkinson's patients recorded at 44.1 kHz, and a label-permutation test.
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from neuroscreen.voice_features import FEATURE_NAMES

warnings.filterwarnings("ignore")
LOG_FEATS = [f for f in FEATURE_NAMES if f.startswith(("jitter", "shimmer_local", "shimmer_apq"))]
LOG_IDX = [FEATURE_NAMES.index(f) for f in LOG_FEATS]


def log_some(X):
    X = np.array(X, dtype=float, copy=True)
    X[:, LOG_IDX] = np.log(np.clip(X[:, LOG_IDX], 1e-6, None))
    return X


def models():
    pre = lambda: [FunctionTransformer(log_some), SimpleImputer(strategy="median"), StandardScaler()]
    return {
        "logistic": lambda: make_pipeline(*pre(), LogisticRegression(C=0.3, max_iter=2000)),
        "svm_rbf": lambda: make_pipeline(*pre(), SVC(C=1.0, gamma="scale", probability=True)),
        "random_forest": lambda: make_pipeline(*pre(), RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=0)),
        "lightgbm": lambda: make_pipeline(*pre(), LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=10, verbose=-1, random_state=0)),
    }


def grouped_cv(df, make_model, feats=FEATURE_NAMES, repeats=20, folds=5, seed=0, test_mask=None):
    """Returns subject-level AUC / balanced accuracy per repeat plus pooled subject predictions."""
    subj = df.groupby("subject").label.first()
    aucs, baccs, sens, spec = [], [], [], []
    pooled = []
    for r in range(repeats):
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + r)
        probs = pd.Series(index=df.index, dtype=float)
        for tr, te in cv.split(df, df.label, df.subject):
            m = make_model()
            m.fit(df.iloc[tr][feats].values, df.iloc[tr].label.values)
            probs.iloc[te] = m.predict_proba(df.iloc[te][feats].values)[:, 1]
        use = df.index if test_mask is None else df.index[test_mask]
        sp = probs.loc[use].groupby(df.loc[use, "subject"]).mean()
        y = subj.loc[sp.index]
        aucs.append(roc_auc_score(y, sp))
        pred = (sp >= 0.5).astype(int)
        baccs.append(balanced_accuracy_score(y, pred))
        sens.append(((pred == 1) & (y == 1)).sum() / (y == 1).sum())
        spec.append(((pred == 0) & (y == 0)).sum() / (y == 0).sum())
        pooled.append(pd.DataFrame({"subject": sp.index, "prob": sp.values, "label": y.values, "repeat": r}))
    s = lambda v: {"mean": float(np.mean(v)), "sd": float(np.std(v))}
    return {"auc": s(aucs), "balanced_accuracy": s(baccs), "sensitivity": s(sens), "specificity": s(spec),
            "n_subjects": int(len(subj))}, pd.concat(pooled)


def leaky_cv(df, make_model, feats=FEATURE_NAMES, repeats=20, folds=5):
    """What most GitHub projects do: random splits over recordings, same person on both sides."""
    aucs, accs = [], []
    for r in range(repeats):
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=r)
        probs = np.zeros(len(df))
        for tr, te in cv.split(df, df.label):
            m = make_model()
            m.fit(df.iloc[tr][feats].values, df.iloc[tr].label.values)
            probs[te] = m.predict_proba(df.iloc[te][feats].values)[:, 1]
        aucs.append(roc_auc_score(df.label, probs))
        accs.append(balanced_accuracy_score(df.label, (probs >= 0.5).astype(int)))
    return {"auc": float(np.mean(aucs)), "balanced_accuracy": float(np.mean(accs))}


def main():
    df = pd.read_csv("data/italian_vowel_features.csv").reset_index(drop=True)
    out = {"dataset": "Italian Parkinson's Voice and Speech (Dimauro & Girardi, IEEE DataPort 2019, CC BY 4.0)",
           "groups": "28 Parkinson's vs 22 age-matched elderly controls; 495 sustained-vowel recordings",
           "protocol": "StratifiedGroupKFold, 5 folds x 20 repeats, all of a person's recordings on one side; "
                       "subject score = mean probability over that person's recordings"}
    M = models()
    out["models_all_vowels"] = {}
    for name, mk in M.items():
        res, _ = grouped_cv(df, mk)
        out["models_all_vowels"][name] = res
        print(f"{name:14s} AUC {res['auc']['mean']:.3f}±{res['auc']['sd']:.3f}  bAcc {res['balanced_accuracy']['mean']:.3f}")
    best = max(out["models_all_vowels"], key=lambda k: out["models_all_vowels"][k]["auc"]["mean"])
    out["selected_model"] = best
    print("selected:", best)

    # the demo records an /a/: evaluate using only /a/ recordings at test time (model trained on all vowels)
    res_a, pooled_a = grouped_cv(df, M[best], test_mask=(df.vowel == "VA").values)
    out["demo_setting_vowel_a_only"] = res_a
    print("test on /a/ only", res_a["auc"], res_a["balanced_accuracy"])

    res, pooled = grouped_cv(df, M[best])
    # ROC from pooled subject predictions of repeat 0
    p0 = pooled[pooled.repeat == 0]
    fpr, tpr, thr = roc_curve(p0.label, p0.prob)
    out["roc_repeat0"] = {"fpr": fpr.round(4).tolist(), "tpr": tpr.round(4).tolist()}
    out["subject_scores_repeat0"] = p0[["prob", "label"]].round(4).values.tolist()

    out["leaky_random_split"] = leaky_cv(df, M[best])
    print("leaky:", out["leaky_random_split"])

    # confound checks
    subj = df.groupby("subject").agg(label=("label", "first"), sex=("sex", "first"), sr=("orig_sr", "max"))
    out["sex_only_baseline_auc"] = float(roc_auc_score(subj.label, (subj.sex == "M").astype(int)))
    keep = df.subject.isin(subj.index[subj.sr == 16000])
    res16, _ = grouped_cv(df[keep].reset_index(drop=True), M[best])
    out["without_44khz_patients"] = res16
    print("16k only:", res16["auc"], res16["balanced_accuracy"])

    # permutation test: shuffle labels across subjects
    rng = np.random.default_rng(0)
    null = []
    for i in range(100):
        perm = dict(zip(subj.index, rng.permutation(subj.label.values)))
        d2 = df.copy(); d2["label"] = d2.subject.map(perm)
        r, _ = grouped_cv(d2, M[best], repeats=1, seed=1000 + i)
        null.append(r["auc"]["mean"])
    out["permutation_test"] = {"n": 100, "null_auc_mean": float(np.mean(null)), "null_auc_95pct": float(np.percentile(null, 95)),
                               "p_value": float((1 + sum(n >= res["auc"]["mean"] for n in null)) / 101)}
    print("perm:", out["permutation_test"])

    # which features carry signal (fit on everything, logistic coefficients for interpretability)
    lr = M["logistic"](); lr.fit(df[FEATURE_NAMES].values, df.label.values)
    coefs = lr[-1].coef_[0]
    out["logistic_coefficients"] = dict(sorted(zip(FEATURE_NAMES, coefs.round(3).tolist()), key=lambda t: -abs(t[1])))
    json.dump(out, open("results/voice_eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
