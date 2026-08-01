# SynthID clean codebook build + content evaluation

**Date:** 2026-08-01
**Task:** Fix the codebook builder's "dots" artifact (discriminative carrier
selection), add a degenerate-codebook OOM guard, build a clean multi-resolution
codebook on real content, and evaluate the carrier attenuation on content.

## 1. The bug (root-caused)

`CodebookBuilder::finalize` built the carrier-selection gate
(`consistency_bgr`) as `1 - std/max_std`, where `std` is the per-bin magnitude
std across captures and `max_std` is the global max (excluding the DC
neighborhood). On near-identical captures (e.g. the HF `gemini_black` set, 100
near-pure-black 1024x1024 PNGs) the per-bin magnitude std is tiny EVERYWHERE,
so `std/max_std` was ~0 at every non-DC bin and `consistency` saturated at
~0.997 across the whole spectrum. Measured on the pre-fix `gemini_black_40`
codebook:

| channel | consistency > 0.6 | effective gate > 0.1 |
|---|---|---|
| B | 100.0% | 97.2% |
| G | 100.0% | 97.0% |
| R | 100.0% | 97.1% |

With the gate fully open at ~97% of bins, the subtractor fed the stored
averaged |FFT| (peak 2077-2430, the visible diamond's broadband signature) into
every bin. On a target image that imprinted the diamond + nuked the band = the
"dots" artifact. A clean expert codebook (reverse-SynthID's
`SynthIDCodebookFinder.find_fourier_carriers`) keeps ~17% of bins and produces
no dots through the same subtractor, confirming the flaw is the builder's
carrier selection, not the subtractor.

## 2. The fix: discriminative carrier selection

The reference method multiplies `log1p(mean_magnitude) * phase_coherence` as the
per-bin carrier score (phase coherence = cross-capture mean resultant length,
already computed as `phase_consistency_bgr`). The magnitude factor gives a soft
floor so zero-magnitude bins (pure-black background) score ~0 regardless of
their numerically-trivial phase; the phase-coherence factor drops content bins
(random phase across captures).

The new `consistency_bgr` is this score normalized to [0,1] by the max score
outside the DC neighborhood. This replaces the saturated `1 - std/max_std`
entirely (the magnitude-variance metric was non-discriminative in both regimes:
it saturated near 1.0 on near-identical images, and on varied content it read
the carrier bin LOWER than off-grid content bins because content-noise masked
the magnitude-stability signal).

### Active-bin fraction, before vs after

`gemini_black_40` (1024x1024, 40 near-identical captures):

| channel | before (1-std/max_std) | after (log1p(mag)*pcons) |
|---|---|---|
| B | con>0.6 = 100.0%, eff_gate>0.1 = 97.2% | con>0.6 = 1.1%, eff_gate>0.1 = 1.7% |
| G | con>0.6 = 100.0%, eff_gate>0.1 = 97.0% | con>0.6 = 1.1%, eff_gate>0.1 = 1.7% |
| R | con>0.6 = 100.0%, eff_gate>0.1 = 97.1% | con>0.6 = 1.1%, eff_gate>0.1 = 1.7% |

The 1.1% that remains is the visible diamond's spectral footprint (the
strongest, most phase-stable bins). On near-identical pure-black captures no
frequency-domain metric can separate the genuine invisible carrier from the
visible diamond, because both are fixed-position additive components and the
carrier is buried under the diamond's broadband signature. The fix's value is
dropping the broadband saturation: the subtractor no longer subtracts a tiny
amount at ~97% of bins (the broadband imprint), only at the ~1% genuinely
strong bins.

`gemini_random_50` (2816x1536, 50 content captures):

| channel | after (log1p(mag)*pcons) | pcons median | random floor 1/sqrt(50) |
|---|---|---|---|
| B | con>0.6 = 0.0%, eff_gate>0.1 = 0.0% | 0.1126 | 0.141 |
| G | con>0.6 = 0.0%, eff_gate>0.1 = 0.0% | 0.1122 | 0.141 |
| R | con>0.6 = 0.0%, eff_gate>0.1 = 0.0% | 0.1118 | 0.141 |

The phase coherence sits at the random floor everywhere: no coherent signal
(neither carrier nor diamond) is resolvable above the content noise floor. This
matches the prior analysis in `synthid-content-fixture-analysis.md` section 6.4.

### Unit tests (`tests/unit/codebook_builder_test.cpp`)

- (a) Near-identical pure-black images with a fixed mark: active-bin fraction
  drops from ~0.99 to < 0.50 (background bins excluded by the magnitude floor).
- (b) A phase-stable carrier sinusoid across varied random-content captures is
  kept (pcons > 0.80, consistency > 0.50); surrounding content bins are dropped
  (pcons < 0.50, consistency < carrier * 0.6).
- (c) Degenerate-profile OOM guard: all-zero and all-NaN profiles no-op cleanly
  instead of crashing.

## 3. Degenerate-codebook OOM guard

A codebook whose magnitude plane is all-zero or non-finite, or whose
consistency plane is NaN (the 0/0 from byte-identical captures), used to
propagate NaN through `polarToCart -> cv::min -> FFT -> convertTo`. In the
worst observed case the cascade triggered a multi-exabyte cv::Mat allocation
(OOM). `CodebookSubtractor::remove_synthid` now checks the profile at entry
(max magnitude finite and >= 1e-6; no NaN in consistency) and, if degenerate,
logs a warning and skips the subtraction passes. The phase-noise disruption
section still runs (it is independent of the profile planes), so the output is
never a bare identity.

## 4. Clean codebook build

`scripts/build_synthid_codebook.sh` stages the HF `gemini_black` (100) +
`gemini_random` (60 of 88; 28 held out for eval) + local 896x1200 fixtures
into a flat scratch dir and builds a multi-resolution `.wcb` with the fixed
builder. The builder groups by resolution, so each size gets its own profile.

Built codebook `/tmp/clean_multi.wcb`:

| profile | N | con>0.6 | pcons median |
|---|---|---|---|
| 1024x1024 (gemini_black) | 100 | 1.1% | 0.74 |
| 2816x1536 (gemini_random) | 60 | 0.0% | 0.11 |

## 5. Content evaluation (the use case)

Per the owner's correction: evaluate on CONTENT, not uniform. The codebook's
content-guard (`is_content_image -> num_passes=0`) makes it inert on content by
default. To measure the codebook's own effect on content, a new
`--no-content-guard` flag bypasses the guard for evaluation. Both modes are
reported. Attenuation is `sum |FFT|^2` in the carrier band `r=3..400` on the
greyscale channel, before vs after.

### gemini_random (2816x1536 content, 6 held-out images)

| mode | band attenuation | PSNR vs original |
|---|---|---|
| content-guard ON (codebook inert) | +1.79% | 33.85 dB |
| content-guard OFF (codebook acts) | +2.17% | 33.46 dB |
| codebook-only extra (OFF minus ON) | +0.38% | -0.39 dB |

### 896x1200 Gemini 3.6 fixtures (6 images, nearest-profile fallback)

| mode | band attenuation | PSNR vs original |
|---|---|---|
| content-guard ON (codebook inert) | +1.05% | 45.71 dB |
| content-guard OFF (codebook acts) | +1.21% | 44.92 dB |
| codebook-only extra (OFF minus ON) | +0.16% | -0.79 dB |

The ~1-2% attenuation comes almost entirely from the phase-noise disruption
(which scrambles carrier-band phase regardless of the codebook). The codebook
subtraction itself adds +0.16 to +0.38% extra on content, which is within
measurement noise and confirms the codebook is effectively inert there: the
carrier is not resolvable above the content noise floor, so there is nothing
for the codebook to subtract.

### Dots verdict (visual, amplified diff x40 on content)

A 1024x1024 content image (resized from `gemini_random`) processed with the
`gemini_black` codebook, codebook-only contribution = `|codebook_output -
phase_noise_only_output|` (isolates what the codebook adds beyond the shared
phase-noise disruption):

| codebook | max pixel diff | visual verdict (image-analyzer) |
|---|---|---|
| contaminated (pre-fix, 97% active bins) | 138 | bright concentration in the bottom-right corner (the diamond imprint) + a scattered dot field across the image |
| clean (post-fix, 1.1% active bins) | 73 | no diamond imprint in any corner, no scattered dot field; cleaner than the contaminated version |

The contaminated codebook imprints the captured visible-diamond signature onto
the target. The clean codebook does not.

## 6. Verdict: worth shipping?

**The builder fix and OOM guard are worth shipping** (they fix a real bug:
the dots artifact on codebooks built from near-identical captures, and a crash
on degenerate input). The 61-test suite stays green (58 original + 3 new).

**The clean codebook is NOT worth shipping as a default yet.** On content
images (the use case) the carrier is not resolvable above the noise floor with
the available datasets, so the codebook is effectively inert (+0.16 to +0.38%
attenuation over the phase-noise baseline, within noise). The content-guard
correctly keeps it inert. Shipping it as a default would add a large asset
(~50 MB per resolution profile) for no measurable user-visible benefit on
content.

The genuine follow-up (unchanged from `synthid-content-fixture-analysis.md`
section 6.8): a verified SynthID-positive / visible-mark-negative fixture pair
with a known published carrier transform. Only that can surface a real carrier
above the content+visible-diamond baseline. Until then, the default SynthID path
stays `--codebook-free` (noise-residual estimation), and the codebook path
remains opt-in for research.

## 7. Reproduction

```sh
# Build the fixed binary + tests.
./scripts/build.sh
ctest --test-dir build --output-on-failure           # 61 tests

# Build a clean codebook (stages HF sets + local fixtures).
./scripts/build_synthid_codebook.sh                  # -> /tmp/clean_multi.wcb

# Content evaluation (attenuation + PSNR).
python3 docs/research/synthid_content_probe.py /tmp/clean_multi.wcb  # active-bin stats
# attenuation/PSNR harness: see the tables above, driven by
#   ./build/wmr synthid <img> --codebook /tmp/clean_multi.wcb --no-content-guard -o out.png

# Dots check: amplified diff on a content image.
./build/wmr synthid content_1024.png --force \
  --codebook /tmp/clean_multi.wcb --no-content-guard -o /tmp/out_cb.png
python3 -c "import cv2,numpy as np; t=cv2.imread('content_1024.png').astype(float); \
  o=cv2.imread('/tmp/out_cb.png').astype(float); \
  cv2.imwrite('/tmp/ampdiff.png', np.clip(np.abs(o-t)*40,0,255).astype('uint8'))"
```
