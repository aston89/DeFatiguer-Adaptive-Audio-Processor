import numpy as np
import subprocess
import tempfile
import os
import librosa
import soundfile as sf
from scipy.ndimage import gaussian_filter1d

# =========================================================
# CONFIG
# =========================================================

EPS = 1e-9

# HIGH RESOLUTION
N_FFT = 8192
HOP_LENGTH = 1024

# PSYCHOACOUSTIC BAND COUNT
NUM_BANDS = 192

# GAIN LIMITS
MIN_GAIN = 0.75
MAX_GAIN = 1.25

# HOW AGGRESSIVE THE REDISTRIBUTION IS
GAIN_STRENGTH = 0.22

# SMOOTHING
FREQ_SMOOTH_SIGMA = 1.5
TIME_SMOOTH_SIGMA = 2.0


# =========================================================
# LOAD / UTIL
# =========================================================
def ensure_stereo(y):
    if y.ndim == 1:
        return np.vstack([y, y])
    if y.shape[0] == 1:
        return np.vstack([y[0], y[0]])
    return y[:2, :]


def load_audio_any(path, sr=44100):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    subprocess.run([
        "ffmpeg", "-y",
        "-i", path,
        "-ar", str(sr),
        "-ac", "2",
        tmp_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    y, sr = librosa.load(tmp_path, sr=sr, mono=False)
    os.remove(tmp_path)

    y = ensure_stereo(y)
    return y, sr


# =========================================================
# PSYCHOACOUSTIC BANDS
# =========================================================
def make_bands(mode="stable"):

    if mode == "stable":
        num_bands = NUM_BANDS
    else:
        num_bands = NUM_BANDS * 2

    mel_edges = np.linspace(
        librosa.hz_to_mel(20, htk=True),
        librosa.hz_to_mel(20000, htk=True),
        num_bands + 1
    )

    hz_edges = librosa.mel_to_hz(mel_edges, htk=True)

    return [
        (hz_edges[i], hz_edges[i + 1])
        for i in range(num_bands)
    ]


def band_labels(bands):
    return [f"{int(round(low))}-{int(round(high))}" for low, high in bands]

# =========================================================
# SMOOTH FIELD
# =========================================================
def smooth_gain_field(field):

    # smooth across frequency
    field = gaussian_filter1d(
        field,
        sigma=FREQ_SMOOTH_SIGMA,
        axis=0
    )

    # smooth across time
    field = gaussian_filter1d(
        field,
        sigma=TIME_SMOOTH_SIGMA,
        axis=1
    )

    return field

# =========================================================
# SPECTRAL MATRIX
# =========================================================
def spectral_matrix(signal, sr, bands):
    S = np.abs(librosa.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    M = []
    for low, high in bands:
        idx = np.where((freqs >= low) & (freqs < high))[0]
        if len(idx) == 0:
            M.append(np.zeros(S.shape[1]))
        else:
            M.append(np.mean(S[idx, :], axis=0))

    return np.array(M)


# =========================================================
# ARTIFACT FIELD
# =========================================================
def artifact_stability_field(signal, sr, bands):
    M = spectral_matrix(signal, sr, bands)

    if M.shape[1] < 2:
        return np.ones_like(M)

    M = M / (np.max(M, axis=1, keepdims=True) + EPS)

    temporal_var = np.abs(np.diff(M, axis=1))
    stability = 1.0 / (1.0 + temporal_var)
    stability = np.pad(stability, ((0, 0), (1, 0)), mode="edge")

    cross = []
    for t in range(M.shape[1]):
        vec = M[:, t]
        vec = vec / (np.sum(vec) + EPS)
        cross.append(vec)

    cross = np.array(cross).T
    cross_inst = np.std(cross, axis=0)
    cross_inst = 1.0 / (1.0 + cross_inst)
    cross_field = np.tile(cross_inst, (M.shape[0], 1))

    return 0.7 * stability + 0.3 * cross_field


# =========================================================
# TEMPORAL CONTRAST
# =========================================================
def temporal_contrast_field(ASF):
    if ASF.shape[1] < 2:
        return np.zeros_like(ASF)

    delta = np.abs(np.diff(ASF, axis=1))
    delta = np.pad(delta, ((0, 0), (1, 0)), mode="edge")

    contrast = np.abs(delta - np.roll(delta, 1, axis=1))
    contrast = contrast / (np.max(contrast) + EPS)

    return contrast


# =========================================================
# STEREO / PHASE / DYNAMICS
# =========================================================
def stereo_incoherence(L, R):
    coef = np.corrcoef(L, R)[0, 1]
    if not np.isfinite(coef):
        diff = np.mean(np.abs(L - R))
        scale = np.mean(np.abs(L + R)) + EPS
        return float(np.clip(diff / scale, 0, 1))
    return float(1.0 - np.clip(coef, -1.0, 1.0))


def phase_instability(L, R):
    S_L = librosa.stft(L, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_R = librosa.stft(R, n_fft=N_FFT, hop_length=HOP_LENGTH)
    diff = np.unwrap(np.angle(S_L) - np.angle(S_R))
    return float(np.clip(np.log1p(np.std(diff)) / 5.0, 0, 1))


def dynamic_instability(signal):
    rms = librosa.feature.rms(y=signal, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    if len(rms) < 2:
        return 0.0
    return float(np.clip(np.log1p(np.mean(np.abs(np.diff(rms)))), 0, 1))


# =========================================================
# CORE ENGINE
# =========================================================
def compute_fatigue(y, sr, mode="stable"):
    bands = make_bands(mode)

    L, R = y[0], y[1]
    mono = (L + R) / 2

    stereo = stereo_incoherence(L, R)
    phase = phase_instability(L, R)
    dyn = dynamic_instability(mono)

    ASF = artifact_stability_field(mono, sr, bands)
    stability = float(np.mean(ASF))
    contrast = float(np.mean(temporal_contrast_field(ASF)))

    score = (
        0.30 * stereo +
        0.30 * phase +
        0.20 * dyn +
        0.10 * (1.0 - stability) +
        0.10 * contrast
    )

    return float(np.clip(score, 0, 1)), {
        "stereo": float(stereo),
        "phase": float(phase),
        "dynamic": float(dyn),
        "artifact_stability": float(stability),
        "temporal_contrast": float(contrast)
    }


# =========================================================
# PERCEPTUAL CLEANUP
# =========================================================
def gentle_cleanup(y, sr, mode="stable", return_debug=False):

    bands = make_bands(mode)

    L, R = y[0], y[1]

    mid = (L + R) / 2
    side = (L - R) / 2

    S = librosa.stft(
        mid,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    freqs = librosa.fft_frequencies(
        sr=sr,
        n_fft=N_FFT
    )

    ASF = artifact_stability_field(mid, sr, bands)
    contrast = temporal_contrast_field(ASF)

    # =====================================================
    # PERCEPTUAL TARGET
    # =====================================================

    target = np.median(ASF)

    # instability field
    instability = (1.0 - ASF)

    # combined penalty
    penalty = (
        0.75 * instability +
        0.25 * contrast
    )

    # normalize
    penalty = penalty / (np.max(penalty) + EPS)

    # =====================================================
    # BUILD GAIN FIELD
    # =====================================================

    band_gain_map = np.ones_like(penalty)

    for i in range(len(bands)):

        deviation = target - penalty[i, :]

        band_gain = (
            1.0 +
            GAIN_STRENGTH * deviation
        )

        band_gain = np.clip(
            band_gain,
            MIN_GAIN,
            MAX_GAIN
        )

        band_gain_map[i, :] = band_gain

    # =====================================================
    # SMOOTHING
    # =====================================================

    band_gain_map = smooth_gain_field(
        band_gain_map
    )

    # =====================================================
    # APPLY TO FFT
    # =====================================================

    gain = np.ones_like(S)

    band_avg_gain = []

    for i, (low, high) in enumerate(bands):

        idx = np.where(
            (freqs >= low) &
            (freqs < high)
        )[0]

        if len(idx) == 0:
            band_avg_gain.append(1.0)
            continue

        gain[idx, :] *= band_gain_map[i, :]

        band_avg_gain.append(
            float(np.mean(band_gain_map[i, :]))
        )

    # =====================================================
    # RECONSTRUCT
    # =====================================================

    mid_clean = librosa.istft(
        S * gain,
        hop_length=HOP_LENGTH,
        length=len(mid)
    )

    # preserve original stereo image
    L_out = mid_clean + side
    R_out = mid_clean - side

    y_out = np.vstack([
        L_out,
        R_out
    ])

    # =====================================================
    # DEBUG
    # =====================================================

    if not return_debug:
        return y_out

    debug = {
        "bands": bands,
        "band_labels": band_labels(bands),
        "ASF": ASF,
        "contrast": contrast,
        "gain": gain,
        "band_gain_map": band_gain_map,
        "band_avg_gain": np.array(
            band_avg_gain,
            dtype=float
        ),
        "mid_clean": mid_clean
    }

    return y_out, debug

# =========================================================
# TEXT REPORT
# =========================================================
def generate_text_report(
    txt_path,
    mode,
    original_score,
    cleaned_score,
    original_metrics,
    cleaned_metrics,
    debug
):

    lines = []

    lines.append("====================================")
    lines.append(" PERCEPTUAL FATIGUE REPORT")
    lines.append("====================================")
    lines.append("")

    lines.append(f"Mode: {mode}")
    lines.append("")

    # =====================================================
    # GLOBAL SCORES
    # =====================================================

    lines.append("GLOBAL SCORES")
    lines.append("------------------------------------")
    lines.append(f"Original fatigue : {original_score:.6f}")
    lines.append(f"Cleaned fatigue  : {cleaned_score:.6f}")
    lines.append(
        f"Delta             : {(cleaned_score - original_score):+.6f}"
    )
    lines.append("")

    # =====================================================
    # METRICS
    # =====================================================

    lines.append("METRICS")
    lines.append("------------------------------------")

    for key in original_metrics.keys():

        before = original_metrics[key]
        after = cleaned_metrics[key]

        delta = after - before

        lines.append(
            f"{key:24s} "
            f"{before:.6f} -> {after:.6f} "
            f"({delta:+.6f})"
        )

    lines.append("")

    # =====================================================
    # BAND ANALYSIS
    # =====================================================

    lines.append("BAND ANALYSIS")
    lines.append("------------------------------------")

    labels = debug["band_labels"]
    avg_gain = debug["band_avg_gain"]

    for label, gain in zip(labels, avg_gain):

        if gain > 1.01:
            action = "BOOST"
        elif gain < 0.99:
            action = "CUT"
        else:
            action = "NEUTRAL"

        lines.append(
            f"{label:16s} "
            f"gain={gain:.4f} "
            f"{action}"
        )

    lines.append("")

    # =====================================================
    # STRONGEST MODIFICATIONS
    # =====================================================

    lines.append("MOST MODIFIED BANDS")
    lines.append("------------------------------------")

    order = np.argsort(np.abs(avg_gain - 1.0))[::-1]

    for idx in order[:20]:

        label = labels[idx]
        gain = avg_gain[idx]

        deviation = gain - 1.0

        lines.append(
            f"{label:16s} "
            f"gain={gain:.4f} "
            f"deviation={deviation:+.4f}"
        )

    lines.append("")

    # =====================================================
    # INTERPRETATION
    # =====================================================

    lines.append("INTERPRETATION")
    lines.append("------------------------------------")

    if cleaned_score < original_score:
        lines.append(
            "The perceptual fatigue score improved."
        )
    else:
        lines.append(
            "The perceptual fatigue score increased."
        )

    avg_dev = np.mean(np.abs(avg_gain - 1.0))

    if avg_dev < 0.01:
        lines.append(
            "Very conservative redistribution."
        )

    elif avg_dev < 0.03:
        lines.append(
            "Moderate perceptual redistribution."
        )

    else:
        lines.append(
            "Aggressive spectral reinterpretation detected."
        )

    lines.append("")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# =========================================================
# PATH UTIL
# =========================================================
def derive_txt_path(base_path):
    root, _ = os.path.splitext(base_path)
    return root + "_report.txt"


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Perceptual fatigue redistribution tool"
    )

    parser.add_argument(
        "input",
        help="Input audio file"
    )

    parser.add_argument(
        "output",
        help="Output audio file"
    )

    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply perceptual redistribution"
    )

    args = parser.parse_args()

    # =====================================================
    # VALIDATION
    # =====================================================

    if not args.fix:
        parser.error(
            "Specify --fix to process the file"
        )

    # =====================================================
    # LOAD
    # =====================================================

    mode = "stable"

    y, sr = load_audio_any(args.input)

    # =====================================================
    # ANALYSIS BEFORE
    # =====================================================

    original_score, original_metrics = compute_fatigue(
        y,
        sr,
        mode=mode
    )

    print("\n==============================")
    print(" ORIGINAL")
    print("==============================\n")

    print(f"Fatigue score: {original_score:.4f}\n")

    for k, v in original_metrics.items():
        print(f"{k:28s}: {v:.4f}")

    # =====================================================
    # PROCESS
    # =====================================================

    y_clean, debug = gentle_cleanup(
        y,
        sr,
        mode=mode,
        return_debug=True
    )

    # =====================================================
    # ANALYSIS AFTER
    # =====================================================

    cleaned_score, cleaned_metrics = compute_fatigue(
        y_clean,
        sr,
        mode=mode
    )

    print("\n==============================")
    print(" CLEANED")
    print("==============================\n")

    print(f"Fatigue score: {cleaned_score:.4f}\n")

    for k, v in cleaned_metrics.items():
        print(f"{k:28s}: {v:.4f}")

    # =====================================================
    # SAVE AUDIO
    # =====================================================

    sf.write(
        args.output,
        np.clip(y_clean.T, -1.0, 1.0),
        sr
    )

    print(f"\nSaved audio -> {args.output}")

    # =====================================================
    # SAVE REPORT
    # =====================================================

    txt_path = derive_txt_path(args.output)

    generate_text_report(
        txt_path=txt_path,
        mode=mode,
        original_score=original_score,
        cleaned_score=cleaned_score,
        original_metrics=original_metrics,
        cleaned_metrics=cleaned_metrics,
        debug=debug
    )

    print(f"Saved report -> {txt_path}")
