"""Assemble dist/index.html (and dist/demo.html for local use with a live microphone) from the template,
the browser code, the exported model and the evaluation results. All numbers on the page come from results/."""
import os as _os
RAW = _os.environ.get("NEUROSCREEN_DATA", "data/raw")
import json, os
os.makedirs("dist", exist_ok=True)
sak = json.load(open("results/sakar_eval.json"))
short = json.load(open("results/shortcut.json"))
spiral = json.load(open("results/spiral_eval.json"))
import pandas as pd
s = pd.read_csv(f"{RAW}/sakar3/pd_speech_features.csv").groupby("id")["class"].first()
port = sak["sakar_portable_grouped"][sak["chosen_model"]]
R = {
    "sakar_n": int(len(s)), "sakar_pd": int(s.sum()),
    "portable": {k: port[k]["mean"] for k in ("subj_auc", "subj_bacc", "subj_sens", "subj_spec")},
    "leak": {"grouped_subj": sak["sakar_all_features_grouped"]["subj_auc"]["mean"], "leaky_subj": sak["sakar_all_features_leaky"]["subj_auc"]["mean"],
             "grouped_rec": sak["sakar_all_features_grouped"]["rec_auc"]["mean"], "leaky_rec": sak["sakar_all_features_leaky"]["rec_auc"]["mean"]},
    "external": {"oxford": sak["external"]["oxford_mdvp"], "italian": sak["external"]["italian_ours_vowel_a"]},
    "shortcut": short,
    "spiral": spiral,
}
html = open("web/template.html").read()
html = html.replace("/*__VOICE_JS__*/", open("web/voice.js").read())
html = html.replace("/*__SPIRAL_JS__*/", open("web/spiral.js").read())
html = html.replace("__MODEL_JSON__", json.dumps(json.load(open("web/voice_model.json"))))
html = html.replace("__RESULTS_JSON__", json.dumps(R))
open("dist/index.html", "w").write(html)
open("dist/demo.html", "w").write(html)
print("dist/index.html", len(html) // 1024, "KB")
