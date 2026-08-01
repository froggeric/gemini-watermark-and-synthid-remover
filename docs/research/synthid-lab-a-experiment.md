# WS3: LAB `a`-channel experiment (`--lab-a`)

**Date:** 2026-08-01
**Task:** Phase 1 WS3. Test whether a LAB `a` (green-red opponent) channel path
beats the BGR path for SynthID suppression/detection on our fixtures.
**Prior art:** `vitotitto/synthid-fingerprint-analysis` reported SynthID most
detectable in LAB `a` (AUC 0.977 / 0.926). Our subtractor/detector work in BGR
(weights `{B:0.85, G:1.0, R:0.70}`).
**Verdict: NO SHIP.** The `a`-channel path is dramatically worse on every
fixture and every split. `--lab-a` ships as an inert, documented experiment flag
so a future dev can re-test on chrominance-bearing content.

---

## 1. What shipped (code)

- `--lab-a` flag on `remove` / `synthid` (`src/cli/cli_app.{hpp,cpp}`), threaded
  through `RemovalConfig::lab_a` (`src/synthid/codebook_subtractor.hpp`).
  Default off = byte-identical to today's BGR path (the lab-a branch returns
  early before any BGR code).
- `NoiseResidualSubtractor::remove_synthid_lab_a`
  (`src/synthid/noise_residual_subtractor.{hpp,cpp}`): converts BGR -> Lab,
  runs the full codebook-free algorithm on channel index 1 (`a`, single plane,
  weight 1.0), keeps L and b byte-identical, merges back, Lab -> BGR. Self-
  contained so the default BGR path is untouched.
- `SynthidDetector::ColorSpace { BGR, LabA }` sibling
  (`src/detection/synthid_detector.{hpp,cpp}`): the LabA path scores on `a`
  against the green-channel codebook profile (the closest single-channel BGR
  proxy for the green-red opponent). Measurement proxy, not a calibrated
  `a`-space codebook.
- Unit tests: `tests/unit/lab_a_experiment_test.cpp` (engaged path differs from
  BGR, valid image, default-off byte-identical, detector LabA finite + differs).

The path is genuinely engaged: on a content fixture, lab-a output differs from
both the input (max abs diff 3/255) and the BGR output (max abs diff 12/255).
This is NOT a vacuous comparison.

## 2. Methodology

Metric (LOWER = better suppression): **carrier-band residual energy**
`sum |FFT|^2` over radius `r` in `[3, 400]`, on the OUTPUT image, planes
normalized to `[0,1]` (same convention as the C++ path). Reported per BGR
channel and as the BGR total, plus the `a`-channel energy of the output
(convert output -> Lab, channel index 1).

Fixtures:
- **Content set:** `reference-images/896x1200-gemini36/` (10 Gemini 3.6 images).
- **Pure-black set:** `test-images/gemini-3.1-pro/2400x1792/pure-black/`
  (30 uniform captures; first 10 sampled).

Both paths run at default strength (`--synthid-strength 0.50` = Moderate), the
only difference is `--lab-a`. Probe: `docs/research/synthid_lab_a_probe.py`
(drives the real `./build/wmr` binary, no reimplementation).

Cross-validation: the 10 content images are split into disjoint halves `5a` /
`5b`. A real effect reproduces on both, a fluke does not.

## 3. Raw numbers

### 3.1 Content set aggregate (mean across fixtures)

| split | orig_total | bgr_total | laba_total | lab/bgr % | bgr_a | laba_a | bgr_suppr | laba_suppr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| content-all (10) | 4,192,214 | 2,758,088 | 10,329,126 | **374.5%** | 1,532,551 | 15,508 | +34.2% | -146.4% |
| split-5a | 4,203,886 | 2,768,787 | 10,338,985 | **373.4%** | 1,545,263 | 15,021 | +34.1% | -145.9% |
| split-5b | 4,180,541 | 2,747,437 | 10,319,267 | **375.6%** | 1,519,844 | 15,995 | +34.3% | -146.8% |

`bgr_suppr` / `laba_suppr` = % change vs `orig_total` (positive = suppressed,
negative = added energy).

The lab-a path leaves a residual **3.74x the BGR residual** and **146% higher
than the original**. The BGR path suppresses 34%. The split reproduces within
1% on both halves.

The `a`-channel column tells the mechanism: lab-a does crush the `a` plane
(laba_a = 15,508 vs bgr_a = 1,532,551, a 99x reduction), but the Lab -> BGR
color matrix spreads that `a`-plane disruption into B and R, and the total BGR
carrier-band energy goes UP.

### 3.2 Per-fixture content set (selected, full table in probe output)

| fixture (abbrev) | orig_total | bgr_total | laba_total | lab/bgr % |
|---|---:|---:|---:|---:|
| 5ibg2e | 4,161,961 | 2,746,247 | 10,275,085 | 374.2% |
| gawws5g | 4,294,764 | 2,838,145 | 10,394,505 | 366.2% |
| ptrgzj | 4,229,744 | 2,770,655 | 10,496,132 | 378.8% |
| r3hwh1 | 4,062,814 | 2,663,402 | 9,972,112 | 374.4% |

Every fixture without exception: lab-a is worse, range 366-379% of BGR.

### 3.3 Pure-black sample (first 10 of 30)

| fixture (abbrev) | orig_total | bgr_total | laba_total | lab/bgr % |
|---|---:|---:|---:|---:|
| 1u7nhn | 50,250,572 | 27,612,496 | 110,026,244 | 398.5% |
| 26htg4 | 18,963,332 | 27,408,273 | 72,565,088 | 264.8% |
| 5rm5gj | 13,077,284 | 27,244,537 | 59,505,751 | 218.4% |
| 5xoj2a | 26,383,075 | 27,070,988 | 89,172,140 | 329.4% |

Range 218-398% of BGR. Pure-black images are uniform (std < 0.05), so both
paths replace with mean+noise. The BGR path replaces all 3 channels; lab-a
keeps L (luminance) byte-identical. The SynthID carrier on these captures lives
in luminance, so lab-a retains it and the Lab -> BGR spread inflates the total.

## 4. Root cause

All 10 content fixtures are **luminance-only**: the `a` and `b` channels are
near-flat (a-std ~0.0004 in [0,1], b-std ~0.0003), while L carries all the
structure (L-std ~0.50). Measured directly:

```
896x1200-gemini36/*.png: BGRstd=0.499 Lstd=0.499 astd=0.0004 bstd=0.0003
```

The SynthID carrier on these images is in luminance, not chrominance. The `a`
channel carries no carrier signal to suppress, so:

1. Disrupting `a` does not touch the real carrier (it lives in L, which is kept
   byte-identical).
2. The `a`-plane disruption, after the Lab -> BGR matrix, injects broadband
   energy into B and R, so the BGR-total residual goes UP, not down.

The vitotitto "most detectable in `a`" finding does not reproduce here. Likely
reasons: (a) their fixtures were color photographs with real chrominance, ours
are luminance-only; (b) SynthID version/rendering differences. Our fixtures do
not exercise the regime where an `a`-channel path could help.

## 5. Detection-separation (metric b)

No same-dimension SynthID-NEGATIVE fixtures exist for a separation/AUC test:
- Content codebook is 1200x896 (HxW).
- The only candidate clean photo (`test-images/poster-artnight.png`) is
  2400x1792 (HxW), a different size, so a codebook built on the content set has
  no matching profile for it. Resizing it would change the spectral content and
  invalidate the comparison.
- The pure-black set (1792x2400 HxW) is transposed relative to the poster.

Per the task guidance ("if no negatives exist, say so and rely on the residual
metric"), the residual metric (section 3) is the load-bearing result. The
`SynthidDetector` LabA sibling path is unit-tested
(`tests/unit/lab_a_experiment_test.cpp`) and produces a finite, in-range
confidence that differs from BGR on a synthetic input; it is available for a
future separation test if chrominance-bearing positives and negatives at a
common size are added.

## 6. Verdict

**NO SHIP.** The `a`-channel path is 3.74x worse than BGR on the content set
(cross-validated 373.4% / 375.6% on the two splits) and 2.2-4.0x worse on
pure-black. It leaves the luminance carrier untouched and inflates the BGR
total via the Lab -> BGR color matrix. The vitotitto hypothesis does not hold
on our (luminance-only) fixtures.

`--lab-a` remains as an inert, documented experiment flag (wired on
`remove`/`synthid`, default off = byte-identical). A future dev can re-test on
chrominance-bearing color content (real photographs, not luminance-only
patterns) where the `a`-channel signal might actually carry the carrier.
