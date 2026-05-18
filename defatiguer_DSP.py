import numpy as np
import subprocess
import tempfile
import os
import librosa
import soundfile as sf

EPS = 1e-9


# =========================================================
# LOAD AUDIO
# =========================================================
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
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    y, sr = librosa.load(tmp_path, sr=sr, mono=False)
    os.remove(tmp_path)

    if y.ndim == 1:
        y = np.vstack([y, y])

    return y, sr


# =========================================================
# BAND GENERATOR (STABLE vs EXPERIMENTAL)
# =========================================================
def make_bands(mode="stable"):
    if mode == "stable":
        return [
            (20, 60), (60, 120), (120, 200),
            (200, 300), (300, 400), (400, 600),
            (600, 800), (800, 1000),
            (1000, 1200), (1200, 1500),
            (1500, 2000), (2000, 2500),
            (2500, 3000), (3000, 3500),
            (3500, 4000), (4000, 5000),
            (5000, 6000), (6000, 7000),
            (7000, 8000), (8000, 10000),
            (10000, 12000), (12000, 14000),
            (14000, 16000), (16000, 20000)
        ]

    # experimental: log-like densification (48-ish, controlled)
    edges = np.geomspace(20, 20000, 49)
    return [(edges[i], edges[i+1]) for i in range(len(edges)-1)]


# =========================================================
# SPECTRAL MATRIX
# =========================================================
def spectral_matrix(signal, sr, bands):
    S = np.abs(librosa.stft(signal))
    freqs = librosa.fft_frequencies(sr=sr)

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

    M = M / (np.max(M, axis=1, keepdims=True) + EPS)

    temporal_var = np.abs(np.diff(M, axis=1))
    stability = 1.0 / (1.0 + temporal_var)

    stability = np.pad(stability, ((0,0),(1,0)), mode='edge')

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
    delta = np.abs(np.diff(ASF, axis=1))
    delta = np.pad(delta, ((0,0),(1,0)), mode='edge')

    contrast = np.abs(delta - np.roll(delta, 1, axis=1))
    contrast = contrast / (np.max(contrast) + EPS)

    return contrast


# =========================================================
# STEREO / PHASE / DYNAMICS
# =========================================================
def stereo_incoherence(L, R):
    return 1 - np.clip(np.corrcoef(L, R)[0, 1], -1, 1)


def phase_instability(L, R):
    S_L = librosa.stft(L)
    S_R = librosa.stft(R)
    diff = np.unwrap(np.angle(S_L) - np.angle(S_R))
    return np.clip(np.log1p(np.std(diff)) / 5.0, 0, 1)


def dynamic_instability(signal):
    rms = librosa.feature.rms(y=signal)[0]
    return np.clip(np.log1p(np.mean(np.abs(np.diff(rms)))), 0, 1)


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

    stability = np.mean(ASF)
    contrast = np.mean(temporal_contrast_field(ASF))

    score = (
        0.30 * stereo +
        0.30 * phase +
        0.20 * dyn +
        0.10 * (1.0 - stability) +
        0.10 * contrast
    )

    return np.clip(score, 0, 1), {
        "stereo": stereo,
        "phase": phase,
        "dynamic": dyn,
        "artifact_stability": stability,
        "temporal_contrast": contrast
    }


# =========================================================
# CLEANUP
# =========================================================
def gentle_cleanup(y, sr, mode="stable"):
    bands = make_bands(mode)

    L, R = y[0], y[1]
    mid = (L + R) / 2
    side = (L - R) / 2

    S = librosa.stft(mid)
    freqs = librosa.fft_frequencies(sr=sr)

    ASF = artifact_stability_field(mid, sr, bands)
    contrast = temporal_contrast_field(ASF)

    gain = np.ones_like(S)

    for i, (low, high) in enumerate(bands):
        idx = np.where((freqs >= low) & (freqs < high))[0]
        if len(idx) == 0:
            continue

        penalty = ASF[i, :] * (1.0 + contrast[i, :])

        gain[idx, :] *= np.clip(0.92 + 0.08 * (1.0 - penalty), 0.88, 1.0)

    mid_clean = librosa.istft(S * gain)

    L_out = mid_clean + side
    R_out = mid_clean - side

    return np.vstack([L_out, R_out])


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python defatiguer.py input output [--fix|--fix-experimental]")
        sys.exit(1)

    inp, out = sys.argv[1], sys.argv[2]

    mode = "stable"
    if "--fix-experimental" in sys.argv:
        mode = "experimental"

    y, sr = load_audio_any(inp)

    score, metrics = compute_fatigue(y, sr, mode=mode)

    print("\n==============================")
    print(" FATIGUE ANALYSIS")
    print("==============================\n")
    print(f"Mode: {mode}")
    print(f"Score (0–1): {score:.3f}\n")

    for k, v in metrics.items():
        print(f"{k:28s}: {v:.4f}")

    if "--fix" in sys.argv or "--fix-experimental" in sys.argv:
        y2 = gentle_cleanup(y, sr, mode=mode)
        sf.write(out, y2.T, sr)
        print(f"\nSaved -> {out}")
