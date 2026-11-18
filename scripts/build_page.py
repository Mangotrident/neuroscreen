"""Assemble the demo page from the template, the browser code, the exported model and every results file.
dist/artifact/  : page + vendored MediaPipe models (offline; what the robot or a hosted viewer loads)
dist/demo.html  : same page loading MediaPipe from jsDelivr (open locally to use mic + camera)"""
import json, os, shutil
RAW = os.environ.get("NEUROSCREEN_DATA", "data/raw")
os.makedirs("dist/artifact", exist_ok=True)
sak = json.load(open("results/sakar_eval.json"))
# people and people with PD in the Sakar CSV; checked against the CSV when the data is downloaded,
# so the page can be built from a fresh clone without the 1.5 GB download
SAKAR_N, SAKAR_PD = 252, 188
csv = f"{RAW}/sakar3/pd_speech_features.csv"
if os.path.exists(csv):
    import pandas as pd
    s = pd.read_csv(csv).groupby("id")["class"].first()
    assert (len(s), int(s.sum())) == (SAKAR_N, SAKAR_PD), "Sakar counts changed: update SAKAR_N and SAKAR_PD"
port = sak["sakar_portable_grouped"][sak["chosen_model"]]
R = {
    "sakar_n": SAKAR_N, "sakar_pd": SAKAR_PD,
    "portable": {k: port[k]["mean"] for k in ("subj_auc", "subj_bacc", "subj_sens", "subj_spec")},
    "leak": {"grouped_subj": sak["sakar_all_features_grouped"]["subj_auc"]["mean"], "leaky_subj": sak["sakar_all_features_leaky"]["subj_auc"]["mean"],
             "grouped_rec": sak["sakar_all_features_grouped"]["rec_auc"]["mean"], "leaky_rec": sak["sakar_all_features_leaky"]["rec_auc"]["mean"]},
    "external": {"oxford": sak["external"]["oxford_mdvp"], "italian": sak["external"]["italian_ours_vowel_a"]},
    "shortcut": json.load(open("results/shortcut.json")),
    "spiral": json.load(open("results/spiral_eval.json")),
    "synthetic": json.load(open("results/synthetic_recovery.json")),
    "baseline_sim": json.load(open("results/baseline_sim.json")),
}
html = open("web/template.html").read()
for key, f in (("VOICE", "voice.js"), ("SPIRAL", "spiral.js"), ("FACE", "face.js"), ("GAIT", "gait.js"), ("TAP", "tap.js"),
               ("BASELINE", "baseline.js"), ("VISION", "vision.js"), ("UI", "ui_base.js"), ("UI_EXTRA", "ui_extra.js")):
    html = html.replace(f"/*__{key}_JS__*/", open(f"web/{f}").read())
html = html.replace("__MODEL_JSON__", json.dumps(json.load(open("web/voice_model.json"))))
html = html.replace("__RESULTS_JSON__", json.dumps(R))
open("dist/artifact/index.html", "w").write(html.replace("__MP_BASE__", "vendor/").replace("__MP_RENAMED__", "true"))
if os.path.isdir("web/vendor"):
    # binary model files get a ".wasm" suffix so any static host serves them; vision.js maps the names back
    shutil.rmtree("dist/artifact/vendor", ignore_errors=True)
    for pkg in os.listdir("web/vendor"):
        os.makedirs(f"dist/artifact/vendor/{pkg}", exist_ok=True)
        for f in os.listdir(f"web/vendor/{pkg}"):
            if os.path.getsize(f"web/vendor/{pkg}/{f}") == 0:
                continue
            dst = f + ".wasm" if f.endswith((".binarypb", ".data", ".tflite")) else f
            shutil.copy(f"web/vendor/{pkg}/{f}", f"dist/artifact/vendor/{pkg}/{dst}")
cdn = html.replace("__MP_BASE__", "https://cdn.jsdelivr.net/npm/").replace("__MP_RENAMED__", "false")
open("dist/demo.html", "w").write(cdn)
open("dist/index.html", "w").write(cdn)
print("built", len(html) // 1024, "KB")
