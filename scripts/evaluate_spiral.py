"""Spiral-drawing baseline on the UCI 'Parkinson Disease Spiral Drawings Using Digitized Graphics Tablet'
features (Isenkul et al. 2014; 62 Parkinson's, 15 controls; one row per person, so no person can leak)."""
import os as _os
RAW = _os.environ.get("NEUROSCREEN_DATA", "data/raw")
import json, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")

d = pd.read_csv(f"{RAW}/spiral_repo/data.csv", index_col=0)
X, y = np.log1p(d.drop(columns="target").abs()), d.target.values
out = {"dataset": "UCI digitized-tablet spiral drawings (Isenkul et al. 2014), features from github.com/shiwalisingh0/Parkinson-disease-detection-",
       "n_people": int(len(d)), "n_pd": int(y.sum()), "n_control": int((1 - y).sum())}
for name, cols in (("all_tests", list(X.columns)), ("static_spiral_only", [c for c in X.columns if c.endswith("_st")])):
    aucs, baccs = [], []
    for r in range(20):
        mk = make_pipeline(SimpleImputer(), RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced", random_state=0))
        p = cross_val_predict(mk, X[cols], y, cv=StratifiedKFold(5, shuffle=True, random_state=r), method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y, p)); baccs.append(balanced_accuracy_score(y, (p >= 0.5).astype(int)))
    out[name] = {"auc": float(np.mean(aucs)), "auc_sd": float(np.std(aucs)), "balanced_accuracy": float(np.mean(baccs))}
    print(name, out[name])
json.dump(out, open("results/spiral_eval.json", "w"), indent=1)
