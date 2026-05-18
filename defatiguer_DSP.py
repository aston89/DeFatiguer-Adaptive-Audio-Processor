import numpy as np
import subprocess
import tempfile
import os
import librosa
import soundfile as sf

EPS = 1e-9
N_FFT = 2048
HOP_LENGTH = 512


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

    # experimental: densificazione logaritmica controllata (48 bande)
    edges = np.geomspace(20, 20000, 49)
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def band_labels(bands):
    labels = []
    for low, high in bands:
        labels.append(f"{int(round(low))}-{int(round(high))}")
    return labels


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
# CLEANUP
# =========================================================
def gentle_cleanup(y, sr, mode="stable", return_debug=False):
    bands = make_bands(mode)

    L, R = y[0], y[1]
    mid = (L + R) / 2
    side = (L - R) / 2

    S = librosa.stft(mid, n_fft=N_FFT, hop_length=HOP_LENGTH)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    ASF = artifact_stability_field(mid, sr, bands)
    contrast = temporal_contrast_field(ASF)

    gain = np.ones_like(S)
    band_gain_map = np.ones((len(bands), S.shape[1]))
    band_avg_gain = []

    for i, (low, high) in enumerate(bands):
        idx = np.where((freqs >= low) & (freqs < high))[0]
        if len(idx) == 0:
            band_avg_gain.append(1.0)
            continue

        penalty = ASF[i, :] * (1.0 + contrast[i, :])
        band_gain = np.clip(0.92 + 0.08 * (1.0 - penalty), 0.88, 1.0)

        gain[idx, :] *= band_gain
        band_gain_map[i, :] = band_gain
        band_avg_gain.append(float(np.mean(band_gain)))

    mid_clean = librosa.istft(
        S * gain,
        hop_length=HOP_LENGTH,
        length=len(mid)
    )

    L_out = mid_clean + side
    R_out = mid_clean - side
    y_out = np.vstack([L_out, R_out])

    if not return_debug:
        return y_out

    debug = {
        "bands": bands,
        "band_labels": band_labels(bands),
        "ASF": ASF,
        "contrast": contrast,
        "gain": gain,
        "band_gain_map": band_gain_map,
        "band_avg_gain": np.array(band_avg_gain, dtype=float),
        "mid_clean": mid_clean
    }
    return y_out, debug


# =========================================================
# PLOTLY REPORT
# =========================================================
def generate_plotly_report_html(html_path, y, y_clean, sr, mode, original_metrics, cleaned_metrics, debug):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio
    except Exception as e:
        raise RuntimeError(
            "Plotly non è disponibile in questo ambiente. "
            "Installa plotly per usare --graph."
        ) from e

    metric_names = [
        "stereo",
        "phase",
        "dynamic",
        "artifact_stability",
        "temporal_contrast",
        "fatigue_score"
    ]

    original_score, _ = compute_fatigue(y, sr, mode=mode)
    cleaned_score, _ = compute_fatigue(y_clean, sr, mode=mode)

    original_values = [
        original_metrics["stereo"],
        original_metrics["phase"],
        original_metrics["dynamic"],
        original_metrics["artifact_stability"],
        original_metrics["temporal_contrast"],
        original_score
    ]

    cleaned_values = [
        cleaned_metrics["stereo"],
        cleaned_metrics["phase"],
        cleaned_metrics["dynamic"],
        cleaned_metrics["artifact_stability"],
        cleaned_metrics["temporal_contrast"],
        cleaned_score
    ]

    labels = debug["band_labels"]
    band_gain_map = debug["band_gain_map"]
    avg_gain = debug["band_avg_gain"]
    times = np.arange(band_gain_map.shape[1]) * HOP_LENGTH / sr

    fig = make_subplots(
        rows=3,
        cols=1,
        vertical_spacing=0.10,
        row_heights=[0.28, 0.42, 0.30],
        subplot_titles=(
            "Metriche prima / dopo",
            "Mappa di modifica per banda e tempo",
            "Modifica media per banda"
        )
    )

    fig.add_trace(
        go.Bar(x=metric_names, y=original_values, name="Originale"),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=metric_names, y=cleaned_values, name="Corretto"),
        row=1, col=1
    )

    fig.add_trace(
        go.Heatmap(
            z=band_gain_map,
            x=times,
            y=labels,
            colorscale="Viridis",
            zmin=0.88,
            zmax=1.0,
            colorbar=dict(title="Gain")
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(x=labels, y=avg_gain, name="Gain medio"),
        row=3, col=1
    )

    fig.update_layout(
        title=f"Defatiguer report — mode: {mode}",
        barmode="group",
        height=1200,
        width=1500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_yaxes(title_text="Valore", row=1, col=1)
    fig.update_xaxes(title_text="Metrica", row=1, col=1)

    fig.update_yaxes(title_text="Banda", autorange="reversed", row=2, col=1)
    fig.update_xaxes(title_text="Tempo (s)", row=2, col=1)

    fig.update_yaxes(title_text="Gain medio", row=3, col=1)
    fig.update_xaxes(title_text="Banda", tickangle=45, row=3, col=1)

    pio.write_html(fig, file=html_path, include_plotlyjs="cdn", full_html=True)


# =========================================================
# PATH UTIL
# =========================================================
def derive_html_path(base_path):
    root, _ = os.path.splitext(base_path)
    return root + "_graphs.html"


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Defatiguer audio: fix, experimental fix, and/or graph report."
    )
    parser.add_argument("input", help="Path del file audio in input")
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Path of audio file output (needed for --fix / --fix-experimental)"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply correction with 24 bands and save output file"
    )
    parser.add_argument(
        "--fix-experimental",
        action="store_true",
        help="Apply correction with 48 bands and save output file"
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Generate HTML graphs with Plotly"
    )

    args = parser.parse_args()

    if not (args.fix or args.fix_experimental or args.graph):
        parser.error("you have to specify at least one of these --fix, --fix-experimental, --graph")

    if args.fix and args.fix_experimental:
        parser.error("Use only one between --fix and --fix-experimental")

    if (args.fix or args.fix_experimental) and not args.output:
        parser.error("with --fix or --fix-experimental you have to specify output path")

    mode = "experimental" if args.fix_experimental else "stable"

    y, sr = load_audio_any(args.input)

    score, metrics = compute_fatigue(y, sr, mode=mode)

    print("\n==============================")
    print(" FATIGUE ANALYSIS")
    print("==============================\n")
    print(f"Mode: {mode}")
    print(f"Score (0–1): {score:.3f}\n")

    for k, v in metrics.items():
        print(f"{k:28s}: {v:.4f}")

    y_clean = None
    clean_metrics = None
    debug = None

    need_cleanup = args.fix or args.fix_experimental or args.graph
    if need_cleanup:
        y_clean, debug = gentle_cleanup(y, sr, mode=mode, return_debug=True)
        clean_score, clean_metrics = compute_fatigue(y_clean, sr, mode=mode)

        print("\n==============================")
        print(" AFTER CLEANUP")
        print("==============================\n")
        print(f"Score (0–1): {clean_score:.3f}\n")

        for k, v in clean_metrics.items():
            print(f"{k:28s}: {v:.4f}")

    if args.fix or args.fix_experimental:
        sf.write(args.output, np.clip(y_clean.T, -1.0, 1.0), sr)
        print(f"\nSaved -> {args.output}")

    if args.graph:
        html_base = args.output if args.output else args.input
        html_path = derive_html_path(html_base)
        generate_plotly_report_html(
            html_path=html_path,
            y=y,
            y_clean=y_clean,
            sr=sr,
            mode=mode,
            original_metrics=metrics,
            cleaned_metrics=clean_metrics,
            debug=debug
        )
        print(f"Graph HTML -> {html_path}")




