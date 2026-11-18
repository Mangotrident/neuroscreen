"""Sustained-vowel voice features, written so the exact same algorithm can run in the browser.

Everything here uses plain numpy (no scipy / Praat calls) and simple, deterministic steps:
frame autocorrelation pitch -> cycle peak picking -> jitter / shimmer / HNR / CPP / tremor.
web/voice.js is a line-by-line port; tests/test_parity.py checks the two agree.
"""
import numpy as np

SR = 16000            # all audio is resampled to 16 kHz first (removes the 44.1 kHz vs 16 kHz recording confound)
FRAME = 640           # 40 ms analysis frame
HOP = 160             # 10 ms hop
FMIN, FMAX = 60.0, 500.0
VOICING_THRESHOLD = 0.45
MAX_PERIOD_FACTOR = 1.3     # Praat default: neighbouring periods may differ by at most 30%
MAX_AMP_FACTOR = 1.6        # Praat default for shimmer

FEATURE_NAMES = [
    "jitter_local", "jitter_rap", "jitter_ppq5", "period_cv",
    "shimmer_local", "shimmer_db", "shimmer_apq3", "shimmer_apq5", "shimmer_apq11",
    "hnr_db", "cpp_db",
    "f0_sd_semitones", "f0_range_semitones", "f0_tremor_ratio",
    "amp_sd_db", "amp_tremor_ratio", "voiced_fraction",
]


def _hann(n):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def _frame_rms_db(x):
    n = 1 + max(0, (len(x) - FRAME) // HOP)
    out = np.empty(n)
    for i in range(n):
        seg = x[i * HOP:i * HOP + FRAME]
        out[i] = 10 * np.log10(np.mean(seg * seg) + 1e-12)
    return out


def select_region(x):
    """Pick the steady middle of the phonation: frames within 25 dB of the loudest,
    take the longest such run, drop 15% at each end, keep at most 4 s from the centre."""
    db = _frame_rms_db(x)
    loud = db > (db.max() - 25.0)
    best_len, best_start, run_start = 0, 0, None
    for i, v in enumerate(np.append(loud, False)):
        if v and run_start is None:
            run_start = i
        elif not v and run_start is not None:
            if i - run_start > best_len:
                best_len, best_start = i - run_start, run_start
            run_start = None
    a = best_start * HOP
    b = (best_start + best_len - 1) * HOP + FRAME
    trim = int(0.15 * (b - a))
    a, b = a + trim, b - trim
    max_len = 4 * SR
    if b - a > max_len:
        c = (a + b) // 2
        a, b = c - max_len // 2, c + max_len // 2
    return x[a:b]


def _autocorr_pitch(x):
    """Per-frame F0 (Hz, 0 = unvoiced) and the normalised autocorrelation at the chosen lag."""
    win = _hann(FRAME)
    # autocorrelation of the window, used to undo the window's taper (as Praat does)
    nfft = 2048
    W = np.fft.rfft(win, nfft)
    rw = np.fft.irfft(W * np.conj(W), nfft)[:FRAME]
    rw = rw / rw[0]
    lag_min = int(np.floor(SR / FMAX))
    lag_max = int(np.ceil(SR / FMIN))
    n = 1 + max(0, (len(x) - FRAME) // HOP)
    f0 = np.zeros(n)
    rbest = np.zeros(n)
    for i in range(n):
        seg = x[i * HOP:i * HOP + FRAME]
        seg = (seg - seg.mean()) * win
        S = np.fft.rfft(seg, nfft)
        r = np.fft.irfft(S * np.conj(S), nfft)[:FRAME]
        if r[0] <= 1e-12:
            continue
        r = r / r[0]
        r = r / np.maximum(rw, 1e-3)
        seg_r = r[lag_min:lag_max + 1]
        # local maxima candidates
        cands = [k for k in range(1, len(seg_r) - 1) if seg_r[k] > seg_r[k - 1] and seg_r[k] >= seg_r[k + 1]]
        if not cands:
            continue
        best = max(cands, key=lambda k: seg_r[k])
        # prefer the shortest lag (highest F0) whose peak is nearly as strong: avoids octave-down errors
        for k in cands:
            if seg_r[k] >= 0.85 * seg_r[best]:
                best = k
                break
        rv = seg_r[best]
        if rv < VOICING_THRESHOLD:
            continue
        # parabolic interpolation of the peak
        y0, y1, y2 = seg_r[best - 1], seg_r[best], seg_r[best + 1]
        denom = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
        lag = lag_min + best + delta
        f0[i] = SR / lag
        rbest[i] = min(max(rv, 1e-4), 0.9999)
    # 5-point median smoothing over voiced frames
    sm = f0.copy()
    for i in range(n):
        if f0[i] > 0:
            w = [v for v in f0[max(0, i - 2):i + 3] if v > 0]
            sm[i] = float(np.median(w))
    return sm, rbest


def _cycle_marks(x, f0):
    """Peak-pick one waveform peak per glottal cycle, guided by the frame F0 contour.
    Returns peak times (s) and peak-to-peak amplitudes per cycle."""
    voiced = f0 > 0
    if voiced.sum() < 3:
        return np.array([]), np.array([])
    # polarity: use the sign with the larger extreme
    sig = x if np.max(x) >= -np.min(x) else -x
    frame_t = (np.arange(len(f0)) * HOP + FRAME / 2) / SR

    def local_period(t):
        i = int(round((t * SR - FRAME / 2) / HOP))
        i = min(max(i, 0), len(f0) - 1)
        if f0[i] > 0:
            return 1.0 / f0[i]
        vi = np.where(voiced)[0]
        j = vi[np.argmin(np.abs(vi - i))]
        return 1.0 / f0[j]

    t_start = frame_t[np.argmax(voiced)]
    t0 = local_period(t_start)
    a = int(t_start * SR)
    b = min(len(sig), a + int(t0 * SR) + 1)
    if b - a < 3:
        return np.array([]), np.array([])
    # first mark = largest peak in the first period
    pos = float(a + int(np.argmax(sig[a:b])))
    marks = [pos]
    # waveform matching (like Praat's "cc" method): the next cycle starts where the
    # current cycle's waveform best correlates with what follows
    while True:
        k = int(round(pos))
        T = int(round(local_period(k / SR) * SR))
        lo, hi = int(0.8 * T), int(1.25 * T) + 1
        if k + hi + T + 1 >= len(sig) or T < 4:
            break
        ref = sig[k:k + T]
        ref = ref - ref.mean()
        nref = np.sqrt(np.dot(ref, ref)) + 1e-12
        cc = np.empty(hi - lo + 1)
        for j, L in enumerate(range(lo, hi + 1)):
            seg = sig[k + L:k + L + T]
            seg = seg - seg.mean()
            cc[j] = np.dot(ref, seg) / (nref * (np.sqrt(np.dot(seg, seg)) + 1e-12))
        j = int(np.argmax(cc))
        delta = 0.0
        if 0 < j < len(cc) - 1:
            y0, y1, y2 = cc[j - 1], cc[j], cc[j + 1]
            denom = y0 - 2 * y1 + y2
            delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
        pos = pos + (lo + j + delta)   # the matched lag is this cycle's period
        marks.append(pos)
    times = np.array(marks) / SR
    amps = []
    for j in range(len(marks) - 1):
        seg = x[int(round(marks[j])):int(round(marks[j + 1]))]
        amps.append(float(seg.max() - seg.min()) if len(seg) else 0.0)
    return times, np.array(amps)




def _jitter_shimmer(times, amps):
    T = np.diff(times)
    ok = (T > 1.0 / FMAX * 0.9) & (T < 1.0 / FMIN * 1.1)
    out = {}
    # jitter on runs of valid periods with neighbour ratio <= MAX_PERIOD_FACTOR
    diffs, rap, ppq = [], [], []
    for i in range(1, len(T)):
        if ok[i] and ok[i - 1] and max(T[i], T[i - 1]) / min(T[i], T[i - 1]) <= MAX_PERIOD_FACTOR:
            diffs.append(abs(T[i] - T[i - 1]))
    for i in range(1, len(T) - 1):
        w = T[i - 1:i + 2]
        if ok[i - 1:i + 2].all() and w.max() / w.min() <= MAX_PERIOD_FACTOR:
            rap.append(abs(T[i] - w.mean()))
    for i in range(2, len(T) - 2):
        w = T[i - 2:i + 3]
        if ok[i - 2:i + 3].all() and w.max() / w.min() <= MAX_PERIOD_FACTOR:
            ppq.append(abs(T[i] - w.mean()))
    meanT = T[ok].mean() if ok.any() else np.nan
    out["jitter_local"] = np.mean(diffs) / meanT if diffs else np.nan
    out["jitter_rap"] = np.mean(rap) / meanT if rap else np.nan
    out["jitter_ppq5"] = np.mean(ppq) / meanT if ppq else np.nan
    out["period_cv"] = float(np.std(T[ok]) / meanT) if ok.sum() > 2 else np.nan

    A = amps
    okA = A > 1e-6
    sd, sdb = [], []
    for i in range(1, len(A)):
        if okA[i] and okA[i - 1] and max(A[i], A[i - 1]) / min(A[i], A[i - 1]) <= MAX_AMP_FACTOR:
            sd.append(abs(A[i] - A[i - 1]))
            sdb.append(abs(20 * np.log10(A[i] / A[i - 1])))
    meanA = A[okA].mean() if okA.any() else np.nan
    out["shimmer_local"] = np.mean(sd) / meanA if sd else np.nan
    out["shimmer_db"] = np.mean(sdb) if sdb else np.nan
    for name, half in (("shimmer_apq3", 1), ("shimmer_apq5", 2), ("shimmer_apq11", 5)):
        vals = []
        for i in range(half, len(A) - half):
            w = A[i - half:i + half + 1]
            if okA[i - half:i + half + 1].all() and w.max() / w.min() <= MAX_AMP_FACTOR:
                vals.append(abs(A[i] - w.mean()))
        out[name] = np.mean(vals) / meanA if vals else np.nan
    return out


def _cpp(x, f0):
    """Cepstral peak prominence (dB), averaged over voiced frames."""
    win = np.hamming(FRAME)
    nfft = 1024
    q = np.arange(nfft) / SR
    qmin, qmax = 1.0 / FMAX, 1.0 / FMIN
    band = (q >= qmin) & (q <= qmax)
    reg = (q >= 0.001) & (q <= qmax)
    vals = []
    for i in np.where(f0 > 0)[0]:
        seg = x[i * HOP:i * HOP + FRAME]
        if len(seg) < FRAME:
            continue
        spec = np.abs(np.fft.fft(seg * win, nfft)) ** 2
        logspec = 10 * np.log10(spec + 1e-12)
        ceps = np.real(np.fft.ifft(logspec))
        ceps_db = 10 * np.log10(ceps ** 2 + 1e-12)
        k = np.where(band)[0][np.argmax(ceps_db[band])]
        slope, intercept = np.polyfit(q[reg], ceps_db[reg], 1)
        vals.append(ceps_db[k] - (slope * q[k] + intercept))
    return float(np.mean(vals)) if vals else np.nan


def _tremor_ratio(contour):
    """Share of a 100 Hz-sampled contour's variance that sits in the 3-10 Hz tremor band."""
    c = contour - np.mean(contour)
    if len(c) < 64 or np.allclose(c, 0):
        return np.nan
    # remove slow drift with a linear detrend
    t = np.arange(len(c))
    c = c - np.polyval(np.polyfit(t, c, 1), t)
    P = np.abs(np.fft.rfft(c * _hann(len(c)))) ** 2
    f = np.fft.rfftfreq(len(c), d=HOP / SR)
    total = P[(f >= 1) & (f <= 20)].sum()
    trem = P[(f >= 3) & (f <= 10)].sum()
    return float(trem / total) if total > 0 else np.nan


def extract(x, sr):
    """x: mono float waveform, sr: its sample rate. Returns dict of FEATURE_NAMES."""
    x = np.asarray(x, dtype=np.float64)
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    x = x - np.mean(x)
    peak = np.max(np.abs(x))
    if peak > 0:
        x = x / peak
    x = select_region(x)
    f0, r = _autocorr_pitch(x)
    voiced = f0 > 0
    feats = {k: np.nan for k in FEATURE_NAMES}
    feats["voiced_fraction"] = float(voiced.mean()) if len(voiced) else 0.0
    if voiced.sum() < 20:
        return feats
    times, amps = _cycle_marks(x, f0)
    if len(times) > 12:
        feats.update(_jitter_shimmer(times, amps))
    rv = r[voiced]
    feats["hnr_db"] = float(np.mean(10 * np.log10(rv / (1 - rv))))
    feats["cpp_db"] = _cpp(x, f0)
    st = 12 * np.log2(f0[voiced] / np.median(f0[voiced]))
    feats["f0_sd_semitones"] = float(np.std(st))
    feats["f0_range_semitones"] = float(np.percentile(st, 95) - np.percentile(st, 5))
    # tremor on the longest voiced run of the semitone contour
    feats["f0_tremor_ratio"] = _tremor_ratio(_longest_run(f0, voiced, semitones=True))
    db = _frame_rms_db(x)[:len(f0)]
    feats["amp_sd_db"] = float(np.std(db[voiced]))
    feats["amp_tremor_ratio"] = _tremor_ratio(_longest_run(db, voiced, semitones=False))
    return feats


def _longest_run(values, mask, semitones):
    best, cur = (0, 0), None
    for i, m in enumerate(np.append(mask, False)):
        if m and cur is None:
            cur = i
        elif not m and cur is not None:
            if i - cur > best[1] - best[0]:
                best = (cur, i)
            cur = None
    v = values[best[0]:best[1]]
    if semitones and len(v):
        v = 12 * np.log2(v / np.median(v))
    return v
