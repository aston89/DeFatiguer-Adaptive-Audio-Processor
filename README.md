# DeFatiguer

Defatiguer is a lightweight perceptual audio stabilisation tool designed to reduce listening fatigue in problematic audio material without altering its overall aesthetic identity.

It operates as a non-destructive analysis-and-attenuation system that targets micro-instabilities in spectral, temporal, and stereo domains, rather than applying global mastering-style transformations.

The core idea is simple: instead of “improving the sound”, Defatiguer reduces the perceptual effort required to follow it.

---

# What it is

Defatiguer is not a mastering tool and it is not a creative effect.

It sits in a different category:

> a perceptual stabilisation layer for pre-existing audio material

It analyses an input signal and applies extremely conservative, locally-adaptive corrections that aim to reduce:

- spectral masking instability
- temporal fluctuation of energy distribution
- stereo phase incoherence
- micro-dynamic irregularities
- rapid perceptual attention shifts

The system is intentionally designed to remain close to the original signal, often producing changes that are subtle or not immediately obvious in short A/B comparisons.

---

# Who it is for

Defatiguer is intended for:

- producers working with low-quality or inconsistent source material
- audio restoration workflows (non-critical archival or informal content)
- sample preparation (loops, stems, reused material)
- listening optimisation for long sessions
- rough mix stabilisation before proper mastering

It is especially useful when:

- the source material is already “acceptable” but tiring, harsh or inconsistent over time
- conventional EQ/compression would be too destructive
- the goal is perceptual comfort rather than sonic redesign

It is not intended for:

- final commercial mastering
- creative sound design
- stylistic enhancement or “coloration”
- replacing mixing decisions

---

# Core principle

The system is built around a single assumption:

> listening fatigue is not caused by global defects but by local instability in perceptual attention.

Instead of correcting frequency balance or dynamics directly, DeFatiguer models how attention shifts over time and frequency, then reduces unnecessary volatility in that structure.

---

# Architecture overview

The processing pipeline is composed of four main stages:

1. spectral decomposition into perceptual bands  
2. temporal stability field estimation  
3. contrast-driven instability detection  
4. stereo-preserving micro-attenuation synthesis  

Each stage contributes to a hierarchical model of “perceptual stability”.

---

# How it works

## 1. Audio loading and normalization

Any input format is first converted to a consistent internal representation:

- via ffmpeg decoding
- resampled to a fixed sampling rate (default 44.1 kHz)
- converted to stereo if necessary

This ensures all downstream analysis operates in a uniform signal space.

No normalization of loudness or dynamics is applied at this stage, preserving the original energy structure.

## 2. Spectral matrix construction

The signal is transformed using Short-Time Fourier Transform (STFT), producing a time-frequency representation.

Instead of operating on raw FFT bins, the spectrum is grouped into perceptual bands.

Two modes exist:

### Stable mode (24 bands)
Fixed logarithmic segmentation of the audible spectrum:

- low frequencies are densely represented
- mid-range is moderately resolved
- high frequencies are coarsely grouped

This mode prioritises stability and predictable behaviour.

### Experimental mode (48 bands)
A geometrically spaced log distribution of bands across 20 Hz – 20 kHz.

This increases spectral resolution while maintaining structural consistency.

Each band is reduced to a time-series energy envelope:

> M[band, time] = mean spectral energy in that band over time

This forms the base perceptual representation.

## 3. Artifact Stability Field (ASF)

The ASF is the core perceptual model of the system.

It estimates how stable each spectral region is over time.

It is composed of two components:

### 3.1 Temporal stability

For each band:

- compute frame-to-frame variation
- convert variation into stability using a non-linear inverse transform:

> stability = 1 / (1 + temporal_variation)

This captures how “volatile” a band is over time.

### 3.2 Cross-band coherence

For each time frame:

- compute normalized spectral distribution across bands
- measure variance across bands over time

This captures whether the spectral energy distribution is stable or constantly rebalancing.

### 3.3 Fusion

The final ASF is:

> ASF = 0.7 × temporal stability + 0.3 × cross-band coherence

This produces a band × time field representing perceptual stability.

## 4. Temporal Contrast Field

The Temporal Contrast Field measures instability of instability.

Instead of looking at raw variation, it measures:

- how stability itself changes over time
- how rapidly perceptual conditions fluctuate

It is computed as:

- temporal derivative of ASF
- difference between consecutive derivative states
- normalization into a bounded field

This identifies “jitter-like” perceptual behaviour:
situations where stability is not just low but inconsistent.

## 5. Global perceptual scoring

The fatigue score is not a physical metric but a perceptual proxy.

It combines:

- stereo incoherence (phase correlation between channels)
- phase instability (STFT phase divergence)
- dynamic instability (RMS envelope variation)
- lack of artifact stability
- temporal contrast

The final score is:

> weighted instability + bounded non-linear compression

This ensures:

- values remain in [0,1]
- extreme cases do not dominate
- mid-range differences remain meaningful

## 6. Gentle Cleanup (stereo-preserving correction)

The correction stage does not perform EQ or compression.

Instead, it applies:

### 6.1 Mid/Side decomposition

- Mid = (L + R) / 2
- Side = (L - R) / 2

All processing is applied only to the Mid channel.

This preserves stereo width and spatial integrity.

### 6.2 Stability-aware attenuation

For each spectral band:

- retrieve ASF and temporal contrast values
- compute a penalty term:

> penalty = ASF × (1 + temporal_contrast)

This ensures that:
- unstable regions are more likely to be attenuated
- rapidly fluctuating instability is penalized more heavily

### 6.3 Gain shaping

Gain is applied in a very narrow range:

- typically between 0.88 and 1.0
- fully bounded to avoid artifacts

This ensures corrections remain subtle and non-destructive.

### 6.4 Reconstruction

- inverse STFT applied to processed Mid
- Side channel reintroduced unchanged
- stereo image restored

Result:

> perceptual stabilisation without spatial collapse

## 7. Output philosophy

Defatiguer is intentionally conservative.

It prioritises:

- preservation of original mix intent
- avoidance of audible processing artifacts
- reduction of perceptual fatigue rather than tonal reshaping

In most cases:

- short A/B comparisons may sound identical
- long listening reveals reduced cognitive load

---

Perfetto, questa è la parte “product-grade” del progetto. Ti lascio una sezione README aggiornata, pensata per essere chiara, leggibile e coerente con la nuova CLI e la doppia modalità.

---

# Usage

Defatiguer can operate in three modes:

* analysis only (no audio modification)
* corrective processing (`--fix`)
* experimental corrective processing (`--fix-experimental`)
* diagnostic visualization (`--graph`)

### Basic analysis

```bash
python defatiguer.py input.wav
```

This runs a full fatigue analysis and prints a set of perceptual metrics without modifying the audio.

### Standard correction (stable mode)

```bash
python defatiguer.py input.wav output.wav --fix
```

Applies a conservative 24-band correction model.

This mode is designed to:

* preserve the original stereo image
* apply minimal spectral attenuation only where instability is detected
* avoid altering the overall tonal character of the mix

### Experimental correction (high-resolution mode)

```bash
python defatiguer.py input.wav output.wav --fix-experimental
```

Uses a 48-band logarithmic spectral decomposition.

Compared to stable mode, this version:

* increases spectral resolution significantly
* produces more localized corrective actions in time-frequency space
* reacts more precisely to narrowband instability events
* is more sensitive to transient spectral irregularities

This mode can produce more noticeable micro-adjustments, but still operates within a conservative gain envelope to avoid destructive processing.

### Visualization mode

```bash
python defatiguer.py input.wav output.wav --graph
```

Generates an interactive HTML report including:

* before/after fatigue metrics
* time-frequency gain maps
* per-band correction intensity over time

The output file will be saved as:

```
*_graphs.html
```

## Output metrics

The analysis engine produces a set of normalized perceptual indicators:

### stereo

Estimates stereo coherence between left and right channels.
Higher values indicate stronger phase or amplitude imbalance between channels.

### phase

Measures broadband phase instability across the stereo field.
High values often correlate with perceived spatial “smearing” or loss of focus.

### dynamic

Represents micro-dynamic irregularity in the RMS envelope of the signal.
High values indicate unstable or overly uneven energy distribution over time.

### artifact_stability

A synthetic field derived from spectral structure stability across time and frequency bands.

High values indicate consistent spectral behavior (lower perceived fatigue).

### temporal_contrast (if enabled in experimental flows)

Measures rapid changes in stability patterns over time, highlighting “attention jitter” in the spectral domain.

## Fatigue score

The final score is a normalized combination of:

* stereo incoherence
* phase instability
* dynamic instability
* spectral structural instability
* temporal contrast (when available)

The score is bounded between 0 and 1:

* **0.0 – 0.3** → low perceived fatigue
* **0.3 – 0.6** → moderate instability, potentially noticeable in long listening sessions
* **0.6 – 1.0** → high fatigue risk (masking issues, spatial blur, or unstable spectral behavior)

This score is not a loudness or mastering metric.
It is a *perceptual stability heuristic*.

## Design philosophy

Defatiguer is not a mastering tool and does not attempt to “improve sound quality” in a traditional sense.

Instead, it models audio as a **time-varying perceptual system**, where discomfort arises from instability patterns rather than absolute spectral content.

The corrective engine therefore:

* avoids global EQ decisions
* avoids compression-based loudness normalization
* avoids stylistic assumptions about genre or target sound
* applies only localized, stability-driven attenuation

The goal is not to “change the mix”, but to reduce perceptual load caused by unstable spectral and spatial micro-events.

## Stable vs Experimental mode

Stable mode (24 bands):

* smoother corrections
* safer for mastered or commercial material
* minimal spectral disturbance

Experimental mode (48 bands):

* higher resolution spectral model
* better sensitivity to narrow artifacts
* slightly more aggressive micro-corrections
* recommended for analysis, restoration, or damaged audio sources

---

# Limitations

Defatiguer does not:

- replace proper mixing decisions
- improve badly recorded audio in a conventional sense
- reconstruct missing information
- perform creative mastering
- guarantee perceptible improvement in short listening sessions

Its effects are primarily:

> temporal, relational and cognitive rather than spectral

---

# Design philosophy

The system is built on a constraint:

> fewer decisions, higher relevance per decision

It deliberately avoids:

- global style classification
- genre inference
- heavy statistical modelling
- neural networks or learned representations

Instead it relies on:

- structured spectral decomposition
- temporal stability fields
- bounded corrective logic

---

# Related tool: [SpectralGravity Processor](https://github.com/aston89/SpectralGravity-Processor)

Alongside *defatiguer*, there is a complementary tool called **[SpectralGravity Processor](https://github.com/aston89/SpectralGravity-Processor)**, which approaches audio processing from a different conceptual angle.
While defatiguer is fundamentally a *perceptual instability analyzer + corrective field system* (it models “listening fatigue” as a combination of temporal, spectral, stereo and phase instabilities and then applies localized corrective attenuation), SpectralGravity works more like a *macro-dynamic balancing engine*.
Instead of trying to detect and suppress “fatiguing micro-artifacts”, SpectralGravity operates on broader structural elements of the signal: it splits audio into coarse spectral regions (low / mid / high bands) and applies **intensity-dependent gain shaping driven by envelope stability, stereo correlation, and energy distribution**. The goal is not to “clean artifacts”, but to maintain a globally coherent energy flow across time and spectrum.

In simpler terms:
* **defatiguer** - focuses on *micro-instabilities* (local perceptual roughness, masking irregularities, phase tension, temporal contrast spikes) and applies subtle corrective fields that behave almost like perceptual smoothing.
* **SpectralGravity** - focuses on *macro balance and movement* (how energy is distributed across bands and channels over time), applying controlled gain dynamics and stereo-aware shaping to preserve overall structural stability.

The two tools therefore complement each other but do not overlap in intent:

defatiguer tries to reduce “what feels tiring when you listen closely”,
SpectralGravity tries to stabilize “how the mix behaves as a system over time”.

Because of this difference, SpectralGravity is intentionally not designed to perform fine corrective masking or artifact suppression. Instead, it prioritizes predictable macro-dynamics and conservative band-level modulation, often producing results that feel more “leveled” rather than “cleaned”.

A deeper technical overview of SpectralGravity is available in its [dedicated repository](https://github.com/aston89/SpectralGravity-Processor).
