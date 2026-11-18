# NeuroScreen

**Can a sustained "aaah" and a hand-drawn spiral reveal Parkinson's disease, and how much of the accuracy reported for this survives honest testing?**

NeuroScreen is a voice and spiral-drawing screening demo that runs entirely in the browser, plus an evaluation that keeps every person out of their own test set. It is a research demo, not a medical device.

## Results

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
python scripts/build_voice_table.py      # features for 495 Italian vowel recordings (+ Praat reference)
python scripts/evaluate_voice.py         # Italian within-dataset evaluation, permutation test
python scripts/evaluate_sakar.py         # Sakar benchmark, leakage comparison, external validation
python scripts/shortcut_check.py         # silence-only model, biomarker directions
python scripts/evaluate_spiral.py        # spiral baseline
python scripts/export_model.py           # browser model -> web/voice_model.json
python tests/make_fixtures.py && node tests/parity.js
python scripts/build_page.py             # -> dist/index.html and dist/demo.html
```

To record from your microphone, serve the folder locally (`python -m http.server` in `dist/`) and open `http://localhost:8000/demo.html`. Hosted artifact viewers block microphone access, so there the page offers file upload instead. A phone voice memo works, but uncompressed WAV is best.

## Limits

- The deployable model comes from one Turkish clinic. Microphones, rooms and compression shift every measure.
- The 0.5 threshold does not transfer between sites. Recalibrate before any screening use.
- Synthetic voices in the demo show the pipeline working. On them the extractor reads roughly half the injected jitter (Praat reads about 80%), while on real recordings it tracks Praat closely.
- No Alzheimer's or stroke/dysarthria module. DementiaBank (Pitt, ADReSS) and TORGO need access agreements. DementiaBank is free for researchers with a faculty sponsor.

## Next steps

1. Get DementiaBank access and add the linguistic module (Whisper timestamps → pause and lexical features), evaluated leave-one-subject-out.
2. Find raw digitized-tablet spiral data (x, y, t, pressure) so the browser spiral features can be validated directly.
3. Calibrate per site, using a few labelled people at each new site to set the threshold.

## Data and licences

- Sakar, C.O. et al. (2018). *A comparative analysis of speech signal processing algorithms for Parkinson's disease classification…* Applied Soft Computing. UCI ML Repository, CC BY 4.0.
- Little, M.A. et al. (2007). *Exploiting nonlinear recurrence and fractal scaling properties for voice disorder detection.* UCI ML Repository, CC BY 4.0.
- Dimauro, G. & Girardi, F. (2019). *Italian Parkinson's Voice and Speech.* IEEE DataPort, doi:10.21227/aw6b-tg17, CC BY 4.0.
- Isenkul, M.E. et al. (2014). *Improved spiral test using digitized graphics tablet for monitoring Parkinson's disease.* UCI ML Repository, CC BY 4.0.
