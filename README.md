# NeuroScreen

**Five everyday health checks a home robot can run (voice, smile, walking, tapping and spiral drawing), each compared with the person's own normal, with every accuracy claim tested on people the models have never seen.**

NeuroScreen runs entirely on-device, in a browser or a robot's web view. It is a research demo, not a medical device.

## The five checks

| Check | Input | Perception | What it measures | Validated how |
|---|---|---|---|---|
| Voice | 5 s "aaah" (mic) | Own extractor, JS port of the Python one | Jitter, shimmer, period variability, HNR, CPP, pitch and loudness tremor, plus a model score | Model trained on 252 people; AUC 0.76 on held-out people, 0.91 on an external US cohort (below) |
| Spiral | Finger or stylus (screen) | Pen path | Tremor-band share and frequency, drift from the ideal spiral, speed variability | Recovers injected 5 Hz and 9 Hz tremor exactly |
| Tap test | Two-target tapping, 15 s per hand (screen) | Touch events | Taps per second, rhythm variability, slowing over the trial, misses, left-right gap | Recovers injected rate, rhythm and slowing |
| Smile | 10 s video (camera) | Google MediaPipe Face Mesh, 468 points | Smile size, time to full smile, left-right asymmetry, cheek raise, blink rate, resting facial motion | Recovers injected asymmetry (40.4% vs 40%), cheek raise and blink rate in synthetic faces |
| Walking | Walk 3–5 m across the camera's view | Google MediaPipe Pose, 33 points, scaled by height | Walking speed (m/s), cadence, step length, stride variability, arm swing per side and asymmetry, forward lean | Recovers injected speed (1.19 vs 1.20 m/s), cadence, step length, arm asymmetry and lean |

Smile, walking and tap measures have **no disease model**. I searched for openly licensed, reproducible labelled data and found none: the one hypomimia repo ships only PCA-compressed features, and mPower tapping data sits behind a data-use agreement. These checks report measurements and changes from the person's own baseline. All 20 synthetic-recovery checks pass (`node tests/metrics_synthetic.js`). End to end, the MediaPipe models were run in headless Chrome on real video, through the page's own upload path.

### Weekly change engine (`web/baseline.js`)

Each person's first two weeks (6 to 14 sessions) set their baseline (median and spread). A **sustained change** needs the median of the last 5 sessions to sit far outside that baseline, judged against its standard error, with 4 of those 5 sessions agreeing, on two consecutive days. A **sudden one-sided change** (smile asymmetry, left-right tapping gap) triggers an immediate safety prompt. Changes are grouped by domain (voice, face, walking, hands) into plain-language prompts that suggest what to ask or whom to see. They never name a disease.

| Simulation (200 people each, `node tests/baseline_sim.js`) | Result |
|---|---|
| Stable people who got any prompt in 8 weeks | 2% |
| Stable people who got a "discuss with a doctor" prompt | 0% |
| Slow drift across several tests caught | 100%, median 16 days after it began |
| Walking-only decline caught | 98%, median 13 days |
| Sudden one-sided change caught the same day | 100% |

The simulated day-to-day variability is an assumption. A pilot has to measure the real variability before these rates mean anything.

## Voice results

All numbers are subject-independent: every recording from one person sits on the same side of every split.

| Experiment | Data | AUC |
|---|---|---|
| Full 752-feature voice model, held out by person | Sakar 2018, 252 people (188 PD) | **0.900** |
| Same model, leaky random split over recordings | Sakar 2018 | 0.986 |
| Browser model (10 features the browser computes), held out by person | Sakar 2018 | **0.763** |
| Browser model, external cohort never seen in training | Oxford / Little 2007, 32 people (24 PD) | **0.906** |
| Browser model, external cohort never seen in training | Italian PD dataset, 50 people (28 PD) | 0.451 |
| Voice model trained and tested within the Italian dataset | Italian, 50 people | 0.986 |
| A model that only hears each Italian file's background silence | Italian, 50 people | 0.837 |
| Spiral kinematics, static spiral (random forest) | UCI tablet spirals, 77 people (62 PD) | 0.806 |

### Three findings

1. **Leakage inflates scores.** Letting a person's recordings appear in both the training and test sets lifts the Sakar model from 0.900 to 0.986 AUC. Many public notebooks on these datasets report the leaky number.
2. **The Italian dataset's near-perfect score is a recording artefact.** Within that dataset a voice model reaches 0.986 AUC (permutation test p = 0.01), yet:
   - a classifier fed only the background silence of each file reaches 0.837;
   - Parkinson's files have a median noise floor about 26 dB louder than control files;
   - jitter, shimmer and HNR all point the opposite way to the physiology (for example, jitter is 0.41% in Parkinson's vs 0.71% in controls).

   The models were learning the recording setup. A model trained on a clean clinic dataset scores 0.45 there, below chance.
3. **The interpretable biomarkers transfer when the recording setup is consistent.** Ten perturbation measures (jitter, shimmer, period variability, HNR) give 0.763 AUC in-clinic and 0.906 on the US Oxford cohort they never saw. At the training clinic's 0.5 threshold, though, sensitivity on Oxford is only 0.17. The ranking transfers; the threshold does not, so each site needs its own calibration.

Confound checks: sex alone gives 0.61 AUC on both datasets. Removing the 10 Italian patients recorded at 44.1 kHz (all controls were 16 kHz) does not lower the Italian in-dataset score (0.996). That points to the recording environment rather than the sample rate.

## What runs in the browser

- **Voice** (`web/voice.js`): resample to 16 kHz → keep the steady middle of the vowel → autocorrelation pitch → vocal-fold cycles by waveform matching (Praat's "cc" idea) → jitter (local, RAP, PPQ5), period variability, shimmer (local, dB, APQ3/5/11), HNR, CPP and tremor-band ratios → balanced logistic regression trained on Sakar 2018.
  - It is a line-by-line port of `neuroscreen/voice_features.py`. `node tests/parity.js` checks the two agree to within 1e-11 on 24 real recordings.
  - Against Praat on 495 real recordings, our measures track Praat's with Spearman ρ 0.61–0.87 and median ratios of 0.91–1.10.
- **Spiral** (`web/spiral.js`): resample the pen path to 100 Hz → search for the centre of the best-fitting Archimedean spiral → radial wobble relative to the spiral's radius → share of wobble energy in the 4–12 Hz tremor band, dominant tremor frequency, speed variability, smoothness. On synthetic spirals it recovers injected 5 Hz and 9 Hz tremor exactly.
  - These are measurements only. The tablet model's features (speed in tablet units, sample counts) are device-specific, so they are not reused in the browser.

## Run it

```bash
pip install -r requirements.txt
bash scripts/get_data.sh                 # ~1.5 GB, all datasets CC BY 4.0
bash scripts/get_vendor.sh               # MediaPipe face + pose models for offline use (optional)
python scripts/build_voice_table.py      # features for 495 Italian vowel recordings (+ Praat reference)
python scripts/evaluate_voice.py         # Italian within-dataset evaluation, permutation test
python scripts/evaluate_sakar.py         # Sakar benchmark, leakage comparison, external validation
python scripts/shortcut_check.py         # silence-only model, biomarker directions
python scripts/evaluate_spiral.py        # spiral baseline
python scripts/export_model.py           # browser model -> web/voice_model.json
python tests/make_fixtures.py && node tests/parity.js     # browser voice code == Python
node tests/metrics_synthetic.js          # face, walking, tap and spiral measures recover injected values
node tests/baseline_sim.js               # false-alarm and detection rates of the weekly engine
python scripts/build_page.py             # -> dist/demo.html and dist/artifact/
```

**To try it live** with microphone and camera, serve `dist/` locally (`python -m http.server` inside `dist/`) and open `http://localhost:8000/demo.html`. It loads the MediaPipe models from jsDelivr. `dist/artifact/` is the same page with the models bundled, for offline use on the robot (run `get_vendor.sh` first). Hosted viewers block the camera and microphone, so there the page accepts uploaded audio and video instead.

**On Piko:** each check is a pure function from a landmark, touch or audio stream to a dictionary of measures (`web/face.js`, `gait.js`, `tap.js`, `spiral.js`, `voice.js`). The robot calls them on its own camera, microphone and screen events, stores one session per day, and passes the history to `NeuroBaseline.evaluate()` for the family summary.

## Limits

- The deployable model comes from one Turkish clinic. Microphones, rooms and compression shift every measure.
- The 0.5 threshold does not transfer between sites. Recalibrate before any screening use.
- Synthetic voices in the demo show the pipeline working. On them the extractor reads roughly half the injected jitter (Praat reads about 80%), while on real recordings it tracks Praat closely.
- No Alzheimer's or stroke/dysarthria module. DementiaBank (Pitt, ADReSS) and TORGO need access agreements. DementiaBank is free for researchers with a faculty sponsor.
- Face, walking and tap measures have not been validated against disease labels, and nobody has measured their day-to-day reliability on robot hardware yet. That pilot is the next step.
- Walking assumes a side-on path roughly parallel to the camera. Walking towards the camera is rejected by the quality check rather than measured badly.

## Next steps

1. Run a 2-week pilot on robot hardware: measure each check's day-to-day variability per person, then set the weekly engine's thresholds from real data.
2. Get DementiaBank access and add the linguistic module (Whisper timestamps → pause and lexical features), evaluated leave-one-subject-out.
3. Find raw digitized-tablet spiral data (x, y, t, pressure) so the browser spiral features can be validated directly.
4. Calibrate per site, using a few labelled people at each new site to set the threshold.

## Data and licences

- Sakar, C.O. et al. (2018). *A comparative analysis of speech signal processing algorithms for Parkinson's disease classification…* Applied Soft Computing. UCI ML Repository, CC BY 4.0.
- Little, M.A. et al. (2007). *Exploiting nonlinear recurrence and fractal scaling properties for voice disorder detection.* UCI ML Repository, CC BY 4.0.
- Dimauro, G. & Girardi, F. (2019). *Italian Parkinson's Voice and Speech.* IEEE DataPort, doi:10.21227/aw6b-tg17, CC BY 4.0.
- Isenkul, M.E. et al. (2014). *Improved spiral test using digitized graphics tablet for monitoring Parkinson's disease.* UCI ML Repository, CC BY 4.0.
- Google MediaPipe Face Mesh 0.4 and Pose 0.5 web packages (Apache-2.0), used unmodified for face and body landmarks.
