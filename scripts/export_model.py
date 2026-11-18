"""Fit the deployable voice model on all of Sakar 2018 and export everything the browser needs:
the preprocessing constants, the logistic weights, and per-group reference ranges for each biomarker."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from evaluate_sakar import load_sakar, model_zoo, PORTABLE, LOGGED

_, d = load_sakar()
m = model_zoo(PORTABLE)["logistic"]()
m.fit(d[PORTABLE].values, d.label.values)
imp, sc, lr = m[1], m[2], m[3]

ref = {}
for f in PORTABLE:
    ref[f] = {}
    for lab, name in ((0, "control"), (1, "parkinsons")):
        # one value per person (mean of their 3 recordings) so nobody counts three times
        v = d[d.label == lab].groupby("subject")[f].mean().dropna()
        ref[f][name] = {str(q): float(np.percentile(v, q)) for q in (10, 25, 50, 75, 90)}

out = {
    "features": PORTABLE,
    "log_features": [f for f in PORTABLE if f in LOGGED],
    "impute_median": imp.statistics_.tolist(),
    "scaler_mean": sc.mean_.tolist(),
    "scaler_scale": sc.scale_.tolist(),
    "coef": lr.coef_[0].tolist(),
    "intercept": float(lr.intercept_[0]),
    "reference": ref,
    "trained_on": "Sakar et al. 2018 (UCI, CC BY 4.0): 252 people, 756 sustained /a/ recordings",
}
os.makedirs("web", exist_ok=True)
json.dump(out, open("web/voice_model.json", "w"), indent=1)

# sanity: the hand-written forward pass must equal sklearn's
def forward(x):
    x = np.array(x, float)
    for i, f in enumerate(PORTABLE):
        if f in LOGGED:
            x[i] = np.log(max(x[i], 1e-6)) if not np.isnan(x[i]) else np.nan
    x = np.where(np.isnan(x), out["impute_median"], x)
    z = (x - np.array(out["scaler_mean"])) / np.array(out["scaler_scale"])
    return 1 / (1 + np.exp(-(z @ np.array(out["coef"]) + out["intercept"])))

X = d[PORTABLE].values[:50]
diff = np.max(np.abs(np.array([forward(r) for r in X]) - m.predict_proba(X)[:, 1]))
print("max |manual - sklearn| =", diff)
assert diff < 1e-9
