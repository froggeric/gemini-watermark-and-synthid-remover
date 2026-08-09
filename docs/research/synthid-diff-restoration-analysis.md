# SynthID regen detail-restoration (diff-gating) analysis

Date: 2026-08-07/08. Unreleased research + feature design. Companion to
`synthid-spectral-removal-record.md` and `synthid-light-reconstruction-attacks.md`.

## TL;DR

`--synthid-attack regen` clears SynthID but costs fidelity (~7-29 dB PSNR at the
validated strength 0.10 knee, content-dependent). Two findings let us recover
some of that fidelity without reintroducing a detectable watermark:

1. **A dilution gate works on normal-content images.** Restoring the top-5%-magnitude
   pixels of the diff `D = O - R` buys +7 dB (PSNR 29.4 -> 36.7 dB on iu84; faces and
   foliage visibly sharpen) and **cleared Google's "Verify with SynthID"**; 10%+ was
   detected. The gate works by *dilution* (few enough watermark-bearing pixels that the
   total restored watermark energy stays under the detector's aggregate threshold), not
   by separating the watermark from the detail.
2. **A characterized-carrier Wiener attenuation helps partially and lifts the ceiling
   one notch on normal content.** Attenuating the diff's low-mid (carrier-band) green
   energy moves dark-content detection from categorical to partial (dose-dependent),
   and on normal content shifts the keep cliff from 5-6 to 6-7 (+0.6 dB free, visually
   transparent). It plateaus at "partial" on dark content even at maximum attenuation
   (the content-adaptive gain mask is unreachable by global spectral means).

A 10-image clearance study then identified the predictor: **mean luminance cleanly
separates clear from fail** (the 4 dimmest images fail at keep5+B3; the 6 brightest
clear; threshold ~128, 10/10, monotone, physically grounded in carrier-SNR). The
shipped design is **automatic**: measure luminance, restore bright images (>=128), full
regen dim images (<128), with two override flags. It is best-effort (no in-process
verifier); `--no-regen-restore-detail` gives guaranteed full removal.

This doc supersedes the earlier "diff-filtering ruled out" framing in places: linear
spectral filtering cannot *fully* clear a watermarked image (reverse-SynthID V3 stripped
91% of carrier-bin energy and was still detected), but in the dilution+attenuation
regime, on near-threshold cases, attenuation measurably helps and is the basis of the
feature.

## Setup

- `O` = watermarked original; `R` = `wmr synthid O --synthid-attack regen` (strength 0.10,
  50 steps, the validated knee); `D = O - R` (signed BGR float).
- Gate metric: combined per-pixel L2 norm `sqrt(dB^2 + dG^2 + dR^2)` (combined, not
  per-channel or luminance: the reconstruction error is luminance-correlated so a
  combined norm captures "did this pixel change" without chroma cancellation; per-channel
  fragments one decision into three without improving separation).
- Verification: Google's manual "Verify with SynthID" (~10/day/account, binary or
  categorical-vs-partial). Within-run controls (a less-attenuated variant detected while
  a more-attenuated one clears) make single verdicts trustworthy.

## The dilution gate (the win on normal content)

PSNR vs `O` as the keep-fraction grows (iu84, the double-watermark hardest case):

| keep% | PSNR(R',O) | verdict |
|---|---|---|
| 0 (regen alone) | 29.45 dB | (cleared by regen) |
| 5 | 36.70 dB | **CLEAR** |
| 6 | 37.38 dB | detected |
| 8 | 38.47 dB | detected |
| 10 | 39.38 dB | detected |

Supporting measurements: diff energy is highly concentrated (top-5% of pixels hold 81%);
high-intensity diff co-locates with texture ~4x over random (where a content-conditional
watermark concentrates); hf-band energy is restored to the original's level at keep5 yet
keep5 still clears, so generic high-frequency energy does not predict detection (the
detector keys on a structured pattern, not hf magnitude).

**Why it works (dilution, not separation):** the detector is a normalized linear
correlator against the watermark key (reverse-engineered by Allen Kuo; consistent with
the SynthID paper's conformal-p-value framing). Detection score is proportional to total
restored watermark energy = fraction-restored x per-pixel amplitude. keep5 stays under
the threshold; keep6 crosses it. The gate does not estimate or remove the watermark; it
keeps few enough watermark-bearing pixels that the aggregate stays under threshold.

**Regime-A wash check (amplitude scaling is strictly worse):** at matched ~5% watermark
budget, spreading restoration over more pixels at lower amplitude collapses the recovered
detail (restored energy 81% -> 22% -> 6% -> 4%; energy falls with scale^2) and PSNR drops
monotonically (36.7 -> 34.3 -> 31.8 -> 31.3). The concentrated full-strength magnitude
gate is Pareto-optimal; diff-scaling is dominated. (Keep this in mind for the attenuation
design: scaling `a` down at fixed keep is the same dominated trade.)

## Can we filter W out of the diff? (attenuation investigation)

Goal: turn `D = reconstruction_error + W` into `reconstruction_error + (1-gamma)W` so more
pixels can be restored at lower per-pixel watermark amplitude. Two facts framed this and
prevented over-claiming: V3 (aloshdenny/reverse-SynthID) removed 91.4% of carrier-bin
energy at 43.5 dB / 0.997 SSIM on a full image and was *still detected*; and a random
perturbation is orthogonal to the detector key and does not shift the cosine (Kuo). So
linear filtering cannot *fully* clear. The bar here is lower: attenuate W enough to tip a
near-threshold case.

### Carrier signature (prior), from synthid-carrier-characterization.md

Magnitude envelope 1/f^1.3 (power slope -2.6), isotropic; channel weighting G=1.0 > B=0.90
> R=0.86 (same spectral shape across channels, corr > 0.96); phase deterministic at r<100,
random at high freq, with a per-channel fourfold angular coherence at r~100-400; RMS
amplitude ~3.5/255 on black frames (below the content noise floor on real images).

### Attenuator B (characterized-carrier Wiener) — the one that works

Per channel, a Wiener-style shrink in the FFT domain, using the signature as the carrier
magnitude prior:

```
carrier_power  = (K_c * r^-1.3)^2           # K_c scaled by ch weight; K calibrated so the
                                           # prior at r=30 ~= 50% of measured |F_D| near r=30
residual_power = max(|F_D|^2 - carrier_power, eps)
alpha          = carrier_power / (carrier_power + residual_power)
reliability    = dc_ramp(r, 25) * highfreq_rolloff(r, 600)
shrink         = max(1 - gamma * alpha * reliability, 0.0)   # capped at 0 (never invert)
F_att          = F_D * shrink
```

Attenuator A (the literal removed "codebook-free" bilateral core: `noise = D - bilateral(D)`,
subtract scaled noise-residual FFT magnitude) is dominated by B on both axes (less
on-target reduction, 5-10x the kept-region fidelity cost), so B is the basis of the feature.

### Results on the hardest image (e3hcto, dark, ~49% pure-black)

| variant | low-mid G reduction | fidelity cost | verdict |
|---|---|---|---|
| keep5 baseline | 0% | 0 | categorical fail |
| B1 (gamma=1) | 18.3% | 0.02 dB | categorical fail |
| B2 (gamma=2) | 29.5% | 0.08 dB | **partial** |
| B3 (gamma=4, capped) | 45.5% | 0.41 dB | partial |
| B4 (gamma=20, ~notch) | 84.2% | 0.85 dB | partial |

Real signal, dose-dependent (categorical -> partial between B1 and B2), then a **plateau**:
pushing low-mid attenuation from 30% to 84% does not move past partial. The residual it
cannot reach is the content-adaptive gain mask (spatially concentrated), which a global
low-mid filter cannot target. This is the diff-regime echo of V3. Lowering the keep (3, 2)
with and without B3 also fails on e3hcto: the darkest content is beyond dilution+attenuation
at any useful keep and needs full regen.

### Ceiling raise on normal content (iu84)

| variant | verdict |
|---|---|
| keep5 baseline / B2 / B3 | clear |
| keep6 baseline | detected (the cliff) |
| **keep6 B3** | **clear** |
| keep8 B3 | detected |

B3 shifts iu84's cliff from 5-6 to 6-7: +1 keep-notch = +0.6 dB on faces/foliage, visually
transparent (the low-mid band carries no restored detail). Not the keep8 leap, but free.

### Two corrections to earlier theoretical calls

1. A power-spectrum measurement of the diff showed the carrier signature absent (low-mid
   recon-error-dominated, no clean G>B>R, slope ~-1 not -2.6), suggesting the attenuator
   would be a no-op. The verifier disagreed: the detectable carrier component *is* in the
   low-mid band, exactly as the characterization predicted; it is a small *correlated*
   fraction buried under recon-error noise, and the verifier's correlation is far more
   sensitive than a total-power spectral read. Lesson: test, do not gate on a coarse
   spectral measurement (see memory `empirical-first-synthid-experiments`).
2. The original "diff-filtering ruled out" framing was too strong for the dilution+
   attenuation regime. Linear filtering cannot fully clear a watermarked image (V3), but
   attenuation measurably helps near-threshold cases and is shippable behind a luminance
   gate.

## The 10-image clearance study (the luminance gate)

Ran keep5+B3 across 10 varied images and verified each. 6 clear, 4 fail. Tested 15 factors;
**mean luminance is the only clean separator** (10/10, monotone, gap [107, 130]):

| stem | verdict | luma | B/G/R | dark<10% | texture |
|---|---|---|---|---|---|
| e3hcto | FAIL | 25 | 26/24/26 | 77% | 9.9 |
| aipro1 | FAIL | 87 | 85/88/88 | 0% | 17.7 |
| baduanjin | FAIL | 102 | 99/105/98 | 1% | 9.1 |
| mockup | FAIL | 107 | 93/106/116 | 1% | 7.0 |
| iau18v | CLEAR | 130 | 108/128/143 | 3% | 4.9 |
| dance-summer | CLEAR | 141 | 94/132/174 | 10% | 14.0 |
| iu84 | CLEAR | 171 | 176/173/165 | 10% | 9.5 |
| x1hsoq | CLEAR | 215 | 208/216/217 | 5% | 19.9 |
| 5nvcl6 | CLEAR | 217 | 192/227/206 | 0% | 20.7 |
| volcanic | CLEAR | 234 | 234/234/234 | 8% | 5.9 |

The 4 dimmest fail; the 6 brightest clear; threshold anywhere in the gap (~128) classifies
10/10. What does *not* separate (all overlap): **blue channel mean** (color-confounded:
baduanjin B=99 fails, dance-summer B=94 clears); **dark-pixel fraction** (baduanjin is 1%
near-black yet fails; 5nvcl6 is 0% yet clears; it is overall brightness, not dark patches);
**complexity** (texture/gradient); **uniform-area fraction**; **diff stats**.

**Mechanism:** the carrier is an additive perturbation of roughly fixed amplitude, so its
signal-to-content ratio scales ~1/luminance. On dim images it is a larger fraction of every
pixel and stays detectable after the carrier-bearing diff pixels are restored; on bright
images it drops under threshold. Green and red means also separate cleanly; luminance
(G-weighted) captures it.

## The design (what ships)

**Automatic, luminance-gated, with two override flags.** Default (no flag): measure mean
luminance; >= 128 -> keep5 + Wiener-B3 restoration; < 128 -> full regen (no restoration).
`--regen-restore-detail` forces restoration (user accepts the risk). `--no-regen-restore-detail`
forces full regen (guaranteed removal). "Both toggles" works because the default is a third
state (auto).

Why automatic over off-by-default+opt-in: the study validated the predictor (10/10,
physically grounded), which was the condition for adaptive being justified. Best UX: bright
images (the majority) get the fidelity boost automatically; dim images auto-fallback to safe
full regen. The error modes are asymmetric in our favor (a wrong "dim" call -> full regen,
safe; the dangerous direction, a "bright" call that silently fails, is 0/6 on this set and
mitigated by the mechanism + the `--no-` escape).

Parameters: keep = 5% (top-5% by combined L2 magnitude of D); Wiener gamma = 4 (capped),
dc_radius = 25, high-freq cutoff = 600, channel weights G/B/R = 1.0/0.90/0.86, prior
calibrated to ~50% of measured |F_D| at r=30. Luminance threshold = 128 (the 8-bit midpoint,
lands inside the empirical gap; consider nudging to ~135 for extra margin since the
below-threshold path is safe full regen).

**Implementation note (open dependency):** the Wiener attenuation needs a 2D FFT per channel.
FFTW3 was removed in 1.16.0. Options: vendor a lightweight MIT FFT (kissfft) for
cross-platform, or use Accelerate/vDSP on macOS. Resolve before implementing.

## Honest caveats

- **n = 10.** The luminance cutoff (~128) and the keep/attenuation params are from a small,
  varied set. iau18v (130) and mockup (107) sit on the line, so the exact threshold is
  tentative. More images would tighten it. The *signal* (dim -> fail) is strong, monotone,
  and matches the carrier physics.
- **No in-process verifier.** The automatic mode is best-effort, not a guarantee. A >= 128
  image could still silently fail restoration. Mitigations: conservative threshold, honest
  docs, the `--no-regen-restore-detail` escape, and ongoing data collection.
- **It is dilution, not removal.** The restored pixels still carry the original watermark;
  the gate keeps the total under the detector's threshold. Full regen destroys the
  watermark; this hides what remains in a small, bright area. Document plainly.
- **Dark content gets no restoration.** e3hcto (and the dim regime generally) cannot be
  restored at any useful keep + attenuation; the auto-gate routes them to full regen. This
  is the measured lower boundary of the feature.
- **Forensic footprint.** Regen outputs (with or without restoration) carry a detectable
  "was-attacked" signature (~99.64% TPR, arXiv 2605.09203), a separate axis no diff
  processing lifts.

## Sources

- SynthID-Image paper: https://arxiv.org/abs/2510.09263
- aloshdenny/reverse-SynthID (V3 91%-strip-still-detected): https://github.com/aloshdenny/reverse-SynthID
- Allen Kuo (linear-correlator detector, orthogonality): https://allenkuo.medium.com/synthid-image-watermark-research-report-9b864b19f9cf
- Yang, "Can Simple Averaging Defeat Modern Watermarks?" NeurIPS 2024
- Goonatilake & Ateniese, forensic cost of regen removal: https://arxiv.org/abs/2605.09203
- In-repo: `synthid-carrier-characterization.md` (carrier signature), `synthid-spectral-removal-record.md`
  (why the spectral path was inert on content), `synthid-light-reconstruction-attacks.md` (regen knee).
