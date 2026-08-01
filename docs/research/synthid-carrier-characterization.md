# SynthID Carrier Signal Characterization

## Summary

Analysis of 30 pure-black 2400x1792 Gemini-generated images reveals the SynthID
carrier's structure in detail. The carrier is a frequency-domain watermark with
a deterministic magnitude envelope and per-image random phase encoding.

## Key Findings

### 1. Common Structure (Deterministic Component)

**Spatial domain:** The average of 30 carrier-only images shows a very stable
spatial pattern. Cross-sample spatial correlation is **0.969** (Green channel) —
the carrier's spatial structure is nearly identical across images. The average
pattern has RMS amplitude ~3.5/255 (1.4% of full scale) and peaks at ~128/255.

**Spectral domain:** The magnitude spectrum has a stable "envelope" shape:
- Follows approximately 1/f^1.3 power law (R² = 0.98)
- Steeper than 1/f pink noise (slope -1.0), closer to brown noise territory
- High energy at low frequencies, rolling off toward high frequencies
- Log-log slope: **-2.6** for the power spectrum (= -1.3 for magnitude)

**Channel weighting:** All three channels carry the same spectral *shape*
(cross-channel shape correlation > 0.96), but with different strengths:
- Green: 36.3% of total carrier energy (strongest)
- Blue: 32.6% (0.90× green)
- Red: 31.1% (0.86× green)

> **Caveat (not reproducible on the shipped fixtures):** the 36.3/32.6/31.1%
> split above was not reproducible on `test-images/gemini-3.1-pro/2400x1792/`.
> The 30 RAW pure-black frames in `pure-black/` (this doc's methodology) measure
> closer to G=33.6% / B=33.0% / R=33.4% (1.6% spread, G > R > B ordering); an
> independent probe measured a 4.8% spread with the same G > R > B ordering.
> The 9-frame multi-color `pure-*-gemini.png` set measures B > R > G (biased
> by clipping harmonics in the saturated ON channels). The two fixture types
> agree only on **B > R** (the load-bearing claim encoded by `kChannelWeights`
> = {0.85, 1.0, 0.70} for {B, G, R}); the G position is not robustly
> resolvable across capture types. The WS1b test in
> `tests/unit/codebook_subtractor_test.cpp` therefore asserts only B > R.

### 2. Per-Image Variation (Stochastic Component)

The per-image variation (deviation from the common template) is **random and
independent** across images:
- Cross-residual correlation: mean = -0.007 (essentially zero)
- This is the encoded payload — unique data embedded per image
- The variation has no correlated structure (pure noise-like)

### 3. Phase Structure — Critical Finding

Phase is **deterministic at low frequencies** and **random at high frequencies**:
- Deterministic (phase_std < 0.3) up to radius ~100 (~12% of max radius)
- Semi-deterministic (phase_std < 1.0) up to radius ~350-430
- Fully random (phase_std > 1.4) from radius ~640-800

This gradient means:
- **Low frequencies (r < 100):** Phase is nearly identical across all 30 samples.
  These bins carry the carrier's structural template, not data.
- **Mid frequencies (r 100-400):** Phase becomes increasingly variable — this
  is where the per-image data encoding lives.
- **High frequencies (r > 600):** Phase is essentially random — either carrying
  the bulk of the payload, or too variable to use.

### 4. Frequency Band Behavior

| Band | Radius | Avg Magnitude (Green) | Phase Behavior | Data Capacity |
|------|--------|----------------------|----------------|---------------|
| DC | 0-3 | 2678 | Fully deterministic | None |
| Very low | 3-10 | 1493 | Deterministic | Minimal |
| Low | 10-30 | 783 | Mostly deterministic | Low |
| Low-mid | 30-80 | 145 | Semi-deterministic | Moderate |
| Mid | 80-200 | 40 | Semi-random | High |
| Mid-high | 200-400 | 13 | Mostly random | Very high |
| High | 400-600 | 6 | Random | Maximum |
| Very high | 600-800 | 4 | Random | Maximum |
| Ultra high | 800-895 | 3 | Random | Maximum |

### 5. No Tiling or Repetition

The autocorrelation peaks at exactly half-image dimensions (1200, 896) are a
Fourier-domain conjugate symmetry artifact, NOT spatial tiling. The carrier:
- Has NO quadrant symmetry (correlations 0.03-0.48)
- Has NO mirror symmetry (flip correlations ≈ 0)
- Is NOT tiled or repeated
- Spans the full image as a single contiguous pattern

### 6. Visible Watermark (Separate from SynthID)

The Gemini visible watermark:
- Is **identical** across all 30 images (r = 0.9999)
- Located at bottom-right: rows 1632-1727, cols 2240-2335 (96×96 pixels)
- Amplitude: ~0.5% of pixel range
- Completely separate from SynthID in both frequency and spatial domain

### 7. Carrier Operating Parameters

| Parameter | Value |
|-----------|-------|
| RMS amplitude | 3.5/255 (1.4% of 8-bit range) |
| Peak amplitude | 128/255 (50% of range, rare) |
| Frequency range | Full spectrum, dominant at low-mid frequencies |
| Phase encoding | Starts at r~30, peaks at r~100-400 |
| Carrier energy (no DC) | 90-97% of total on black, 0% on white channels |
| Noise color | ~1/f^1.3 (between pink and brown) |
| Per-image variation | Random, independent, ~3.6 std in magnitude |

### 8. Per-Bin Magnitude Distribution

At specific frequency bins, the magnitude across samples:
- Low freq (r=10): CV=0.03, highly stable, Mean/Std=30.8 (NOT Rayleigh)
- Mid freq (r=100): CV=0.18, moderately stable
- High freq (r=200): CV=0.52, Mean/Std=1.94 ≈ Rayleigh (complex Gaussian)

The high-frequency carrier bins appear to follow Rayleigh distribution, suggesting
the carrier at those frequencies is essentially complex Gaussian noise. Low-frequency
bins are much more deterministic.

### 9. Carrier on Different Backgrounds

On pure-black images, carrier energy is 90-97% of total spectral energy.
On saturated channels (white, red, green, blue), carrier energy drops to ~0%
because the DC component dominates. The carrier is present on ALL backgrounds,
but only visible when the background has no spectral content.

Key insight: The carrier is **additive**. On a black pixel (value 0), the carrier
adds ~1.4% RMS noise. On a white pixel (value 255), the same carrier is present
but represents ~0% of the spectral energy relative to the DC component.

## 10. Spectral Visual Analysis

Visual inspection of the FFT and phase coherence images reveals structure not
captured by radial averaging alone.

### FFT Magnitude Spectrum

The magnitude spectrum is radially symmetric with concentric ripple rings and
no directional streaks. Energy distribution is isotropic — no preferred angular
orientation in magnitude. The 1/f^1.3 rolloff is visible as a bright center
fading smoothly outward.

### Phase Coherence — Angular Structure (New Finding)

Phase coherence maps (30-image circular mean) show **fourfold symmetry with
channel-specific angular stripe patterns**:

- **Blue channel**: Vertical and horizontal coherence stripes (0°/90°/180°/270°)
- **Green channel**: Diagonal coherence stripes (45°/135°/225°/315°)
- **Red channel**: Mixed/combined pattern (both axes present, weaker)

This angular modulation is present only in the phase coherence (consistency
across samples), NOT in the magnitude spectrum. It indicates that the carrier's
deterministic phase structure has directional encoding — different channels
carry phase information along different angular orientations.

**Interpretation:** The fourfold symmetry likely arises from the carrier's
frequency-domain construction. The per-channel angular offset (0° vs 45°)
suggests a directional filter bank or angularly-modulated ring structure.
This could be:
1. An artifact of the embedding process (e.g., real/imaginary splitting)
2. Intentional angular diversity to improve robustness against rotation
3. A consequence of the DCT/FFT ring structure used in SynthID

The angular structure is strongest at mid-frequencies (r ~100-300) where
phase transitions from deterministic to random. This is the same band where
per-image data encoding occurs.

### Content Image FFT Structure

On content images (pop-art, photo), the FFT is completely dominated by image
content. The bilateral filter noise residual retains "sketch-like outlines" of
edges and textures. The carrier's spectral signature is invisible beneath
content energy. This visually confirms the SNR gap documented in Section 11.

### Generated Visualizations

All images saved in `docs/research/images/`:
- `carrier_spatial_average.png` — Per-channel spatial pattern (enhanced range)
- `carrier_fft_magnitude.png` — FFT magnitude spectra (log scale, full + zoomed)
- `carrier_phase_coherence.png` — Phase coherence maps revealing angular stripes
- `carrier_radial_profiles.png` — Radial magnitude, coherence, CV, and power law
- `carrier_backgrounds_comparison.png` — Carrier on black/white/red/green/blue
- `carrier_content_comparison.png` — Black reference vs pop-art vs photo

## 11. Phase Coherence Detection Tests

### Test Method

Extracted a low-frequency phase template (r <= 100) from 30 pure-black images
using circular mean (unit vector averaging). Then tested phase coherence on
various images by comparing their noise residual's FFT phase against the template.

### Results: Per-Channel Coherence vs Channel Brightness

| Image      | Ch | Mean  | Std    | PhaseCoh | SNR_est |
|------------|-----|-------|--------|----------|---------|
| black      | B/G/R | 0.014 | 0.014 | ~0.98    | 1.000   |
| white      | B/G/R | 0.986 | 0.014 | ~0.01    | 0.014   |
| red        | B   | 0.004 | 0.006 | ~0.96    | 2.333   |
| red        | R   | 0.986 | 0.015 | ~0.03    | 0.014   |
| pop-art    | B/G/R | ~0.5  | ~0.25 | 0.04-0.05 | ~0.056 |
| photo      | B/G/R | ~0.4  | ~0.20 | 0.02-0.08 | ~0.070 |

**Key insight:** Phase coherence tracks channel darkness, not carrier presence.
On bright channels, the carrier is present but completely swamped by the image
content in the noise residual.

### Results: Noise Residual Phase Coherence

Even using bilateral filter denoising to extract the noise residual:

| Image       | Best Channel | Coherence | Avg Coherence |
|-------------|-------------|-----------|---------------|
| black       | any         | 0.98      | 0.98          |
| white       | B           | 0.04      | 0.02          |
| pop-art     | R           | 0.051     | 0.045         |
| photo       | R           | 0.083     | 0.024         |
| Random noise| —           | 0.083     | —             |

**Critical finding:** Content images (pop-art, photo) show phase coherence of
0.04-0.08, which is **indistinguishable from random noise** (0.08). The carrier
IS present but at 1.4% RMS, it is 10-30x weaker than the content noise in the
bilateral filter residual.

### Content-Normalized Weighting

Weighting phase comparison by carrier_magnitude / noise_magnitude did NOT help.
The spectral shape of the noise residual on content images is dominated by image
content (edges, textures), not carrier structure.

## Implications for Detection and Removal

### Detection

1. **Reliable on uniform/dark images** — Phase coherence > 0.90 on channels where
   the mean pixel value is below ~0.1. The carrier template matches perfectly.

2. **Unreliable on content images** — Phase coherence drops to random baseline
   (~0.04-0.08) on content images. The bilateral filter residual is dominated by
   image content, not carrier signal.

3. **The fundamental SNR gap** — Carrier RMS is ~1.4% (3.5/255). Content noise in
   the bilateral filter residual is 10-30x stronger. No amount of spectral
   weighting bridges this gap.

4. **Channel-based approach works partially** — On images with some dark channels
   (e.g., red image has dark B/G channels), those specific channels show high
   coherence. But real-world images rarely have truly dark channels.

### Removal

1. **Magnitude subtraction works for uniform images** — The carrier magnitude
   template is well-characterized and stable. Subtract it from the image's FFT
   magnitude.

2. **Phase disruption is needed for content images** — Since detection fails on
   content images, the only viable approach is blind disruption: perturb phase
   in the r=30-400 band where SynthID data is encoded, without trying to detect
   the carrier first.

3. **Targeted phase noise** — Apply phase perturbation weighted by the known
   carrier strength at each frequency. Low frequencies (r < 30) need no
   disruption; mid frequencies (r 100-400) need the most.

4. **Angular-aware disruption** — The fourfold symmetry in phase coherence
   (Section 10) suggests that phase disruption should cover all angular
   orientations, or match the per-channel angular pattern (B: H/V, G: diagonal,
   R: mixed) for maximum effectiveness.

5. **Noise residual subtraction** — An alternative: bilateral filter to extract
   noise residual, subtract it entirely (removes carrier + image noise), then
   add back synthetic noise. This is destructive but effective.

### Procedural Reproduction

The carrier's deterministic component CAN be reproduced:
1. Generate a 1/f^1.3 magnitude envelope (well-defined)
2. Use the average phase template for r < 30 (deterministic)
3. Add random phase for r > 30 (data encoding region)
4. Scale by channel weights (B:0.90, G:1.0, R:0.86 of base)
5. Scale by carrier amplitude (RMS ~3.5/255)

The random phase component means we can reproduce the carrier *structure* but
not the specific per-image data. This is sufficient for uniform-image detection
and template-based removal.

## Open Questions

1. **Can machine learning detect the carrier on content images?** A CNN trained
   on many Gemini vs non-Gemini images might learn subtle statistical patterns
   that our spectral analysis misses.

2. **Is there a secret key we're missing?** SynthID uses a keyed hash for the
   ring hash mechanism. Without the key, detection relies on observing the
   carrier directly.

3. **Does the carrier change across Gemini model versions?** Our analysis is
   based on Gemini 3.1 Pro images. Different models may use different carrier
   parameters.
