# SynthID detection: validation against Google-verified ground truth

Date: 2026-08-05. Task 1 of the post-v1.15.0 follow-ups (plan at
`~/.claude/plans/post-v1.15-followups.md`). This is the first time the spectral
SynthID detector is scored against an EXTERNAL ground truth (Google's official
in-app "Verify with SynthID" verdict). Earlier SynthID research
(`synthid-investigation-summary.md`, `synthid-codebook-not-viable.md`,
`codebook-free-removal-research.md`) concluded detection is not viable from
internal probes and theory; this doc confirms that conclusion with real labels.

Read `synthid-investigation-summary.md` first for the broader SynthID context.
This doc is scoped to ONE question: does our detector agree with Google?

## The question and why it matters

`wmr detect` exposes a spectral SynthID detector (`src/detection/synthid_detector.cpp`)
that scores an image against a carrier codebook and reports a confidence plus four
sub-scores (noise correlation, carrier phase, structure ratio, multi-scale
consistency). The detector was deliberately deprioritized in commit `9bc9ad1`
("correct SynthID detector, drop SynthID from active scope"): thresholds were
raised above the random baseline so all images report "not detected", because
prior research found the FFT path has no discriminative power against SynthID's
neural encoder-decoder (arXiv 2510.09263).

That "no discriminative power" conclusion was drawn from internal probes (carrier
energy ratios, invariance tables, false-positive checks on synthetic frames). It
had never been checked against Google's actual verdict. The owner has now run
Google's official in-app detector on a small labeled set. This doc scores our
detector against those labels.

## Ground-truth set (Google verdicts)

All images are 1728x2432. The "verified" set is Google-positive (SynthID
present); the "free" set is Google-negative (SynthID removed by a regen pass the
owner confirmed clean). Naming: `*-detected.png` = a regen output where SynthID
survived (regen too weak); `*-free.png` = a regen output Google confirmed clean;
`*-cpu.png` = the CPU-backend regen output.

| Image | Google verdict | Notes |
|-------|----------------|-------|
| `reference-images/synthid-verified/poster-baduanjin.png` | POSITIVE | original Gemini generation |
| `reference-images/synthid-verified/test-regen003-detected.png` | POSITIVE | regen output, SynthID survived |
| `reference-images/synthid-verified/test-regen005-detected.png` | POSITIVE | regen output, SynthID survived |
| `reference-images/synthid-verified/test-regen009-detected.png` | POSITIVE | regen output, SynthID survived |
| `reference-images/synthid-verified/test-regen005-cpu.png` | POSITIVE | CPU-backend regen, SynthID survived |
| `reference-images/synthid-free/test-regen010-free.png` | NEGATIVE | regen output, Google-confirmed clean |
| `reference-images/synthid-free/test-regen011-free.png` | NEGATIVE | regen output, Google-confirmed clean |
| `reference-images/synthid-free/test-regen015-free.png` | NEGATIVE | regen output, Google-confirmed clean |

## Method

Two codebooks were built and each was used to score all 8 images. The detector
resizes the codebook profile to the image size (INTER_LINEAR), so a 2400x1792
codebook applies to the 1728x2432 set.

1. **Contaminated 9-sample codebook** (`/tmp/wmr-synthid-solidcolors-2400.wcb`):
   built from `test-images/gemini-3.1-pro/2400x1792/` (the 9 pure-color frames).
   These source frames still carry the VISIBLE Gemini diamond (`[VISIBLE V1]
   detected 80.0%`), so the diamond's spectrum leaks into the carrier profile.
   This is the naive build.
2. **Clean 30-sample codebook** (`/tmp/wmr-synthid-black-clean-30.wcb`): built
   from `test-images/gemini-3.1-pro/2400x1792/pure-black-cleaned/` (30 distinct
   pure-black Gemini generations with the visible diamond already removed;
   `[VISIBLE] not detected`, mean pixel ~1/255). On pure black the carrier is
   the dominant spectral signal, so this is the best-case carrier reference with
   no visible-mark contamination. This build addresses the methodological
   requirement that codebook sources have the visible watermark stripped first.

Sub-scores were captured by a temporary instrumentation of `process_detect`
(`src/cli/cli_app.cpp`) that prints the raw sub-scores for every image; the
instrumentation was reverted after the run (the working tree is unchanged).

## Results: contaminated 9-sample codebook

| Image | Google | confidence % | noise | phase | struct | ms_cons |
|-------|--------|-------------:|------:|------:|-------:|--------:|
| poster-baduanjin | POS | 59.8950 | 0.5018 | 0.4996 | 0.9925 | 0.4994 |
| test-regen003-detected | POS | 59.8907 | 0.5019 | 0.4995 | 0.9924 | 0.4992 |
| test-regen005-cpu | POS | 59.9183 | 0.5018 | 0.5001 | 0.9926 | 0.4996 |
| test-regen005-detected | POS | 59.8907 | 0.5019 | 0.4995 | 0.9924 | 0.4992 |
| test-regen009-detected | POS | 59.8907 | 0.5019 | 0.4995 | 0.9924 | 0.4992 |
| test-regen010-free | NEG | 59.8957 | 0.5020 | 0.4996 | 0.9924 | 0.4993 |
| test-regen011-free | NEG | 59.8957 | 0.5020 | 0.4996 | 0.9924 | 0.4993 |
| test-regen015-free | NEG | 59.8960 | 0.5020 | 0.4996 | 0.9924 | 0.4992 |

## Results: clean 30-sample codebook

| Image | Google | confidence % | noise | phase | struct | ms_cons |
|-------|--------|-------------:|------:|------:|-------:|--------:|
| poster-baduanjin | POS | 61.4170 | 0.5599 | 0.5004 | 0.9654 | 0.4993 |
| test-regen003-detected | POS | 61.3587 | 0.5583 | 0.5004 | 0.9652 | 0.4995 |
| test-regen005-cpu | POS | 61.3295 | 0.5576 | 0.5003 | 0.9652 | 0.4995 |
| test-regen005-detected | POS | 61.3587 | 0.5583 | 0.5004 | 0.9652 | 0.4995 |
| test-regen009-detected | POS | 61.3587 | 0.5583 | 0.5004 | 0.9652 | 0.4995 |
| test-regen010-free | NEG | 61.3665 | 0.5584 | 0.5005 | 0.9653 | 0.4996 |
| test-regen011-free | NEG | 61.3665 | 0.5584 | 0.5005 | 0.9653 | 0.4996 |
| test-regen015-free | NEG | 61.3751 | 0.5587 | 0.5004 | 0.9653 | 0.4997 |

## Confusion matrix and discrimination

The detector's content-image threshold is 0.70 (struct_ratio above 0.80 flags a
content image; uniform images use 0.55). Every test image is content
(struct_ratio ~0.965), so all 8 are gated at 0.70 and all score ~0.613, so all
report "not detected".

At the operating threshold (0.70), all 8 predicted NEGATIVE:

| | Google POS | Google NEG |
|---|---|---|
| wmr POS | 0 | 0 |
| wmr NEG | 5 | 3 |

Accuracy 3/8 (37.5%, the negative base rate). Recall 0/5 (0%): the detector
misses every real SynthID-positive image.

Threshold tuning cannot rescue this. The score distributions fully overlap:
- Clean codebook POS range: 61.3295 to 61.4170% (mean 61.365%)
- Clean codebook NEG range: 61.3665 to 61.3751% (mean 61.369%)
- The NEG mean (61.369) is HIGHER than the POS mean (61.365); the NEG range sits
  entirely inside the POS range. Total spread across all 8 images is 0.087
  percentage points.
- ROC AUC (clean codebook) = 0.20. Below 0.5 (random); the score is slightly
  anti-correlated with the label, which at N=8 is noise. A real detector needs
  AUC well above 0.9. There is no threshold that separates the two classes.

The contaminated codebook is no better: every image scores 59.89 to 59.92%, an
even tighter cluster.

## Why the score is a constant (not a measurement of SynthID)

The sub-score breakdown shows the confidence is a property of "a natural
1728x2432 image matched against this codebook", not of the carrier:
- `struct_ratio` is ~0.965 for every image (spectral-shape similarity to the
  carrier profile). This is content-driven and near-identical across natural
  images. It is the dominant term.
- `noise_corr` and `carrier_phase` sit at the random baseline (~0.50, or ~0.558
  on the pure-black codebook from a mild spurious shape match). They do not move
  with the Google label.
- `ms_consistency` is ~0.50 for every image (the `9bc9ad1` fix removed the old
  x2.0 multiplier that had pinned it near 1.0 for all images, including random
  noise).

The carrier amplitude is ~0.025/255 (sub-LSB), below the content noise floor, so
it contributes negligibly to the FFT magnitude the scores are built from. This
matches the earlier "carrier energy is negligible on content images" finding in
`codebook-free-removal-research.md` (carrier is 0.06 to 0.09% of total spectral
energy on content). The clean pure-black codebook raises `noise_corr` from 0.50
to 0.558 (the pure-black carrier shape correlates weakly with any natural
image's noise spectrum) but raises it IDENTICALLY for positives and negatives,
so it adds a constant offset, not discrimination.

## Verdict

Our spectral SynthID detector has no usable discriminative power for SynthID on
content images. Against the first external ground truth available (8 images,
Google's official verdict), the confidence is a near-constant ~61% regardless of
whether SynthID is present, the score distributions fully overlap, and ROC AUC
is 0.20 (noise at this N). No threshold or codebook separates the classes.

This empirically confirms the prior theoretical conclusion
(`synthid-investigation-summary.md`, `synthid-codebook-not-viable.md`) against
real Google labels. It is consistent with SynthID-Image being a
content-conditional neural watermark whose carrier sits below the content noise
floor. It is also consistent with reverse-SynthID's finding that purely spectral
rounds (their 01 to 05) failed to clear Google's detector; only their
non-spectral Round 06 (diffusion regeneration) worked, which is the basis of our
`--synthid-attack regen` path.

"Finetune the threshold" is not a possible outcome: there is no signal to
threshold on. The score would have to vary with the label for a threshold to
separate the classes, and it does not.

## Implications for `wmr detect`

- `wmr detect` does not report SynthID unless `--codebook <path>` is passed, and
  no codebook ships with the project. So end users never see the (non-functional)
  SynthID line. This is the correct, honest state by omission. There is no
  regression: `process_detect`'s SynthID block has been gated on `--codebook`
  since the detect subcommand was introduced (`165cb129`), and
  `SynthidDetector::detect()` has no codebook-free overload.
- The detector should not be exposed as a working SynthID feature. Google
  provides no public SynthID-Image verifier, and this validation shows our
  spectral detector cannot substitute for one. The honest product position: wmr
  does not detect SynthID; it can suppress it (heuristic, `--codebook-free`) or
  remove it (lossy, validated, `--synthid-attack regen`).
- Detection on UNIFORM images (pure black / solid color) does work, because there
  the carrier is the dominant spectral signal. But uniform images are an
  artificial case (a real photo is never a flat field), so this is not a useful
  product feature.

## Caveats

- N = 8 (5 positive, 3 negative). Small, but the result is not borderline: the
  distributions fully overlap and the score is a content-property constant, so a
  larger set would tighten the means without creating separation.
- The codebooks were built from 2400x1792 sources and applied to 1728x2432
  images via the detector's resize path. The clean 30-sample pure-black codebook
  is the best-case carrier reference (carrier-dominant, no visible-mark
  leakage), so if detection were possible it would show here.
- The negative images are regen outputs (diffusion-processed), not clean
  "never-watermarked" frames. Even so they score identically to the positives,
  which strengthens (not weakens) the no-discrimination conclusion: substantial
  image processing did not move the content-driven score.
- No public verifier exists. Google's in-app verdict is the only external signal
  and is itself qualitative and account-gated. These 8 labels are the owner's
  manual checks.
