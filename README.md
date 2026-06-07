# DeFatiguer: Adaptive audio processor
A lightweight perceptual audio stabilization tool designed to reduce listening fatigue in problematic audio material without altering its overall aesthetic identity.
It operates as a non-destructive analysis-and-attenuation system, targeting micro-instabilities in spectral, temporal and stereo domains rather than applying global mastering-style transformations.

Defatiguer presents a particularly interesting utility in **broadcast scenarios**, especially in traditional radio environments where prolonged listening is common.
By addressing spectral instabilities and phase incoherence, it allows for more comfortable extended listening sessions, reducing auditory fatigue that can arise from continuous exposure to potentially problematic material.
This can be especially valuable for formats that rely on long-term engagement such as **heavy electronic compressed music**, talk radio or all-day music programming.

Additionally, Defatiguer shows promise as a **valuable tool in pre-loudness maximization workflows**. In these situations, its ability to mitigate various types of instabilities can effectively "clean up" the signal before it hits the final limiter.
This can allow the limiter to work more efficiently and assertively without introducing undesirable artifacts, ultimately permitting a hotter final output while preserving a greater degree of sonic integrity.
In practical terms : it essentially allows the limiter to **"pump harder"** (to borrow some colorful engineer slang) in a way that enhances overall loudness and impact without sacrificing quality. This positions Defatiguer as a strategic insert ahead of loudness maximization stages in mastering chains, particularly for genres or formats that aim for competitive loudness levels.

The core idea is simple: instead of “improving the sound”, DeFatiguer reduces the perceptual effort required to follow it.

---

# What it is

DeFatiguer is not a mastering tool and is not a creative effect, it sits in a different category:

> **perceptual stabilization layer for pre-existing audio material**

It analyzes an input signal and applies extremely conservative, locally-adaptive corrections that aim to reduce:
- Spectral masking instability
- Temporal fluctuation of energy distribution
- Stereo phase incoherence
- Micro-dynamic irregularities
- Rapid perceptual attention shifts

The system is intentionally designed to remain close to the original signal, often producing changes that are subtle or not immediately obvious in short A/B comparisons.

---

# Who it is for

DeFatiguer is intended for:
- Producers working with low-quality or inconsistent source material
- Audio restoration workflows (non-critical archival or informal content)
- Sample preparation (loops, stems, reused material)
- Listening optimization for long sessions
- Rough mix stabilization before proper mastering

It is especially useful when:
- The source material is already “acceptable” but tiring, harsh, or inconsistent over time
- Conventional EQ/compression would be too destructive
- The goal is perceptual comfort rather than sonic redesign

---

# Core Principle

The system is built around a single assumption:
> **Listening fatigue is not caused by global defects but by local instability in perceptual attention.**

Instead of correcting frequency balance or dynamics directly, DeFatiguer models how attention shifts over time and frequency, then reduces unnecessary volatility in that structure.

---

# Architecture Overview

The processing pipeline is composed of several main stages:
1. Psychoacoustic band definition
2. Spectral matrix construction
3. Artifact Stability Field (ASF) estimation
4. Temporal Contrast Field calculation
5. Stereo, phase, and dynamic instability measurement
6. Perceptual fatigue scoring
7. Gentle Cleanup (stereo-preserving correction)

Each stage contributes to a hierarchical model of “perceptual stability”.

---

# How it Works

1. Psychoacoustic Band Definition
The system operates on 192 psychoacoustic bands derived from Mel frequencies.

2. Spectral Matrix Construction
The signal is transformed into a time-frequency representation using the Short-Time Fourier Transform (STFT). Instead of operating on raw FFT bins, the spectrum is grouped into the defined perceptual bands.

3. Artifact Stability Field (ASF)
The ASF estimates how stable each spectral region is over time. It is composed of two components:

3.1 Temporal stability
For each band:
- Frame-to-frame variation is computed.
- Variation is converted into stability using a non-linear inverse transform.

3.2 Cross-band coherence
For each time frame:
- Normalized spectral distribution across bands is computed.
- Variance across bands over time is measured.

4. Temporal Contrast Field
The Temporal Contrast Field measures the instability of instability. It computes the temporal derivative of the ASF and normalizes it into a bounded field.

5. Stereo, Phase, and Dynamic Instability
The following metrics are calculated:
- Stereo incoherence: Measures the degree of phase or amplitude imbalance between channels.
- Phase instability: Represents broadband phase instability across the stereo field.
- Dynamic instability: Represents micro-dynamic irregularity in the RMS envelope.

6. Perceptual Fatigue Scoring
A final fatigue score, bounded between 0 and 1, is computed based on the combination of stereo incoherence, phase instability, dynamic instability, spectral structural instability, and temporal contrast.

7. Gentle Cleanup (stereo-preserving correction)
The correction stage applies stability-aware attenuation:
- Mid/Side decomposition is performed.
- Gain shaping is applied based on the stability metrics.
- The signal is reconstructed, preserving the original stereo image.

---

# Usage

Basic analysis
python DeFatiguer_DSP.py input.wav

Perceptual redistribution
python DeFatiguer_DSP.py input.wav output.wav --fix

Command-line options:
- --fix: Apply the perceptual redistribution to reduce fatigue.
- input.wav: The input audio file to process. 
- output.wav: The output audio file path.

Output Metrics (analysis mode):
- stereo: Stereo coherence between left and right channels.
- phase: Broadband phase instability across the stereo field.
- dynamic: Micro-dynamic irregularity in the RMS envelope.
- artifact_stability: Stability of spectral structure across time and frequency bands.
- temporal_contrast: Rapid changes in stability patterns over time.

Output Files
- Audio: The processed audio file (output.wav).
- Text Report: A detailed text report is generated alongside the audio, offering insights into the original and cleaned audio metrics.

---

## Related tool

Alongside *DeFatiguer*, there is a complementary tool called **[SpectralGravity Processor](https://github.com/aston89/SpectralGravity-Processor)**, which approaches audio processing from a different conceptual angle.
While DeFatiguer is fundamentally a *perceptual instability analyzer + corrective field system* (as a combination of temporal, spectral, stereo and phase instabilities), SpectralGravity works more like a *macro-dynamic balancing engine*.
Instead of trying to detect and suppress “fatiguing micro-artifacts”, SpectralGravity operates on broader structural elements of the signal: it splits audio into coarse spectral regions (low / mid / high) and applies **intensity-dependent gain shaping driven by envelope stability, stereo correlation, and energy distribution**.
The goal of SpectralGravity is not to “clean artifacts” but to maintain a globally coherent energy flow across time and spectrum.

In simpler terms:
* **DeFatiguer** - focuses on *micro-instabilities* (local perceptual roughness, masking irregularities, phase tension, temporal contrast spikes) and applies subtle corrective fields that behave almost like perceptual smoothing.
* **SpectralGravity** - focuses on *macro balance and movement* (how energy is distributed across bands and channels over time), applying controlled gain dynamics and stereo-aware shaping to preserve overall structural stability.

The two tools therefore complement each other but do not overlap in intent:
* **DeFatiguer** - tries to reduce “what feels tiring when you listen closely”.
* **SpectralGravity** - tries to stabilize “how the mix behaves as a system over time”.

A deeper technical overview of SpectralGravity is available in its [dedicated repository](https://github.com/aston89/SpectralGravity-Processor).
