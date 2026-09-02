# Antidetect M0: detector baselines, Stage-A per-ingredient A/B, and dose calibration

Measured 2026-09-01 on the dev Mac (arm64, 16 GB), single machine, by the
harness in `experiments/antidetect-eval/` (see its README for reproduction;
all numbers below come from `experiments/antidetect-eval/results/`, per-image
rows in `per_image.csv`, 3042 scores over 39 fixtures x 26 conditions x 3
detectors). This document is the artifact of record the C++ Stage-A constants
(`src/core/antidetect_physics.cpp`, `dose_for_strength`) are calibrated
against.

## Executive summary

**Update 2026-09-01 (post-calibration, C++ commit c5a551a):** the per-ingredient
A/B below led to a RECALIBRATED production ladder -- bilateral + randomized
{edgeaware, bilinear} mosaic + JPEG q88-96, sensor noise only at strength
>= 0.75, CA and vignette OFF at every strength (default output quality went
from ~13/255 to ~2/255 mean perturbation). The original literature-anchor
ladder is retained in the harness as `ladder="legacy"`; every `stack-*` row
in `results/per_image.csv` was measured under that PRE-calibration ladder and
is historical. Details in "Post-calibration decision" below. The summary
points 1-6 stand as measured facts.

1. **Of the three local detectors, only Corvi (GRIP Grag2021_latent) flags
   the flagship fixture population (Gemini 3.6 896x1200 portraits): 10/10
   flagged at baseline, with small margins (+0.02..+0.36 logit).** commfor
   (CommunityForensics ViT) misses all of them; it instead flags the large
   2400-2816 px outputs and AI paintings (5/25). NPR is inverted/broken on
   this population (AUROC 0.64, and it flags 2/14 real camera photos).
2. **The full Stage-A stack at any strength in 0.25..1.00 clears every
   Corvi-positive image with margin**: worst surviving AI score after
   stack-0.50 is -1.21 (from +2.15 at baseline); after stack-1.00, -0.98.
   Real photos stay far below threshold throughout (worst real +stack-1.00:
   -2.61; zero new false positives at any dose for commfor/corvi).
3. **The single most cost-effective ingredient against Corvi is the JPEG
   cycle**: one q96 cycle alone moves the portraits from +0.18 mean to -2.76
   (a ~3-logit drop, the largest per-unit-visibility effect measured). The
   mosaic+demosaic (-0.84..-1.65) and CA (-0.99) are next. Noise and
   bilateral flip the marginal positives too, but with ~10x smaller deltas.
4. **More strength is not better**: commfor's flagged count rises from 2-4
   (strengths 0.25-0.75) to 6 at strength 1.00 -- heavy bilateral/noise/mosaic
   doses manufacture the very artifacts these detectors read. Keep the
   default at 0.5; treat > 0.75 as an explicit user override.
5. **Corvi suppression survives downstream JPEG; commfor clearance does not
   fully**: re-encoding the stack-0.50 output at q90/q75/q50 keeps Corvi at
   25/25 below threshold (worst -0.65 even at q50), but q90 re-encoding
   creates 3 NEW commfor positives on portraits (+1.1 each, from -3.6..-5.0
   at baseline) while q75/q50 bring back paint-lioness (+4.4/+3.8). Zero
   real false positives at every quality for commfor/corvi. Practical read:
   the physics stack is robust for the detector that actually flags Gemini,
   and partly reversible for the ViT by platform re-compression.
6. **A porting hazard measured, not hypothetical**: commfor's logit moves by
   2-4 depending on the resize kernel used for its 440-px preprocessing
   (non-antialiased cv2 cubic vs canonical PIL bicubic). Any C++ port of a
   ViT detector must pin an antialiased downscale (cv2 INTER_AREA tracks PIL
   bicubic within ~1.0 logit here) and lock it with a parity test.

## Scope

- Stage A "camera-statistics restoration" (the physics pipeline), measured
  against three local AI-image detectors on a small local fixture set.
- Not measured here: Stage B (adversarial Square Attack), closed detectors
  (SynthID, platform-side classifiers), online services, and video.

## Fixtures

| set | n | what |
|---|---|---|
| AI | 25 | Gemini-generated originals from the repo's reference sets: 10x Gemini 3.6 896x1200 portraits, 2x regen-lab originals, 3x VAE-testing originals, 2x 2400x1792, 2x 896x1200 test plates, 2x 2816x1536 paid-tier originals, 3x AI paintings (2400x1792), 1x AI poster. No regen outputs (different population), no flat-color plates. |
| real | 14 | Wikimedia Commons "quality image" camera photographs. 3 are untouched camera originals (3474-4691 px); 11 are width-2000 JPEG derivatives (rescaled by the Commons CDN; attribution in `fixtures/real/manifest.json`). |

Known skew: the real set is landscape-heavy and camera-JPEG-sourced; the AI
set is one generator family (Gemini). Counts are small; treat numbers as
directional, not population statistics.

## Detectors

| name | version / weights | preprocessing | score convention |
|---|---|---|---|
| commfor | `buildborderless/CommunityForensics-DeepfakeDet-ViT` `onnx/model.onnx` (fp32, 87,442,080 B, sha256 a42c7d74...) -- the FIXED 2026-07-22 weight regeneration | shortest-edge 440 (PIL bicubic, antialiased) -> center crop 384 -> CLIP norm | single logit; logit > 0 (sigmoid > 0.5) = fake |
| corvi | GRIP `Grag2021_latent` (`buloutian/corvi-2022-mirror` checkpoint, 282,549,121 B, sha256 0168451e...), res50stride1 | full-res, ImageNet norm, spatial mean of per-pixel logits; inputs > 1.2 MPix bicubic-capped (stride-1 activation budget; the cap applies identically to baseline and conditions) | logit > 0 = fake (Corvi et al. 2022) |
| npr | `chuangchuangtan/NPR-DeepfakeDetection` `NPR.pth` (17,393,733 B) | full-res, ImageNet norm; model computes x - nearest_up(nearest_down(x, 0.5)), conv1 input x 2/3 | logit > 0 = fake (their validate.py) |

The deprecated `CommunityForensics-DeepfakeDet-ViT-ONNX` repo must not be
used (self-declared broken weights; see the fixed repo's CHANGELOG for the
2026-07-22 regeneration).

### Resize-kernel sensitivity of commfor (a porting hazard)

The ViT's logit depends strongly on HOW the 440-px shortest-edge resize is
done (measured, 4 fixtures):

| image | PIL bicubic (canonical) | cv2 INTER_AREA | cv2 INTER_LINEAR | cv2 INTER_CUBIC |
|---|---|---|---|---|
| gem36-01 (AI) | -4.15 | -4.02 | -2.00 | -1.87 |
| gem36-04 (AI) | -3.17 | -2.18 | -1.05 | -0.98 |
| commons-00 (real) | -1.72 | -1.73 | -5.21 | -5.99 |
| commons-03 (real) | -9.45 | -8.83 | -9.25 | -9.54 |

Non-antialiased downsampling (plain INTER_LINEAR/INTER_CUBIC) shifts logits
by 2-4 and moves AI images TOWARD the fake side -- it manufactures
high-frequency energy the ViT reads as generative. Any C++ port of this
detector MUST use an antialiased downscale (INTER_AREA tracks PIL bicubic
within ~1.0 logit on these fixtures) and pin it in a parity test against the
ONNX reference numbers above.

## Method

- A/B engine: `experiments/antidetect-eval/physics_reference.py`, a 1:1
  NumPy/OpenCV mirror of `src/core/antidetect_physics.cpp` (same op order,
  formulas, u8 quantization points, 3-kernel bank, dose ladder; RNG is numpy
  PCG64 instead of std::mt19937_64, so distribution-identical, not
  byte-identical).
- 26 conditions: baseline; bilateral/noise/CA/vignette each alone at
  strengths 0.25/0.50/1.00; JPEG cycle alone at q96/92/88; mosaic+demosaic
  alone per kernel (malvar5x5/bilinear/edgeaware); the full stack at
  strengths 0.25/0.50/0.75/1.00; JPEG survival (stack-0.5 re-encoded at
  q90/75/50).
- Seeds: crc32(slug + condition); every condition image is reproducible.
- Every condition runs on BOTH sets; deltas are per-image vs that fixture's
  own baseline. Full tables: `results/baseline.csv`, `results/ab_summary.csv`,
  `results/calibration.md`.

## Results

### Baseline (no ops)

| detector | AI mean (n=25) | real mean (n=14) | AUROC | AI flagged@0 | real flagged@0 (FP) |
|---|---|---|---|---|---|
| commfor | -2.53 | -7.75 | 0.906 | 5/25 | 0/14 |
| corvi | -2.48 | -9.81 | 0.923 | 12/25 | 0/14 |
| npr | -31.42 | -39.45 | 0.643 | 1/25 | 2/14 |

Which AI images are flagged: corvi -> all 10 gem36 portraits + the 2
896x1200 test plates (margins +0.02..+0.36, one outlier +2.15); commfor ->
the 2 wide 2400x1792, the 2 paid 2816x1536, 1 regen-lab original, 1 AI
painting (margins +1.0..+4.7); npr -> 1 AI painting only.

Real subset split (native camera originals vs Commons-rescaled
derivatives): commfor -3.91 vs -8.79, corvi -8.41 vs -10.19, and
**npr +7.21 (native) vs -52.18 (rescaled)** -- NPR reads ordinary resized
real photos as strongly fake. NPR is an up-sampling-artifact detector, not
an AI-image detector, on this population; its 2 real FPs at baseline are
rescaled photos.

### Per-ingredient A/B (medium dose; deltas are per-image score movement)

| condition | corvi AI delta | commfor AI delta | npr AI delta | corvi real delta | notes |
|---|---|---|---|---|---|
| bilateral-0.50 | +0.16 | +1.04 | +75.2 | +0.20 | wrong direction for commfor; npr explodes |
| noise-0.50 | -0.32 | +0.99 | +34.2 | +1.56 | corvi's cheapest flip (q96-scale margins) |
| ca-0.50 | -0.99 | +0.93 | +47.4 | -0.91 | good corvi mover |
| vignette-0.50 | -0.25 | -0.33 | +10.3 | +0.27 | only op that lowers commfor |
| jpeg-92 | -0.66 | -0.12 | +3.9 | -0.07 | on portraits, jpeg-96 alone = **-2.94 delta** |
| mosaic-malvar5x5 | -1.65 | +1.77 | +75.8 | -0.19 | strongest per-op corvi delta |
| mosaic-bilinear | -1.09 | +1.46 | +59.9 | -0.73 | |
| mosaic-edgeaware | -1.32 | +1.18 | +59.6 | -0.40 | kernel choice is second-order |

Reading the "flips" (baseline > 0 -> after <= 0): corvi's 12 positives flip
under essentially every op (their margins are tiny), so for corvi the DELTA
and the surviving margin are the meaningful metrics, not flip counts. For
commfor the flips are meaningful (its positives have 1-4 logit margins):
4 flip under every stack strength <= 0.75.

Full-stack ladder (AI set; "flipped" = of the detector's own baseline
positives; real new FP = real images crossing 0 upward):

| condition | corvi below 0 / flipped | commfor below 0 / flipped | npr below 0 / flipped | real new FP (corvi/commfor/npr) |
|---|---|---|---|---|
| stack-0.25 | 25/25 / 12 | 23/25 / 4 | 25/25 / 1 | 0 / 0 / 1 |
| stack-0.50 | 25/25 / 12 | 23/25 / 4 | 24/25 / 1 | 0 / 0 / 1 |
| stack-0.75 | 25/25 / 12 | 22/25 / 4 | 22/25 / 1 | 0 / 0 / 2 |
| stack-1.00 | 25/25 / 12 | 19/25 / 4 (6 flagged: backfire) | 23/25 / 1 | 0 / 0 / 2 |

Worst-case surviving AI logits (the margin that matters):

| condition | corvi worst AI | commfor worst AI | npr worst AI |
|---|---|---|---|
| baseline | +2.15 | +4.70 | +9.67 |
| stack-0.25 | -1.36 | +1.65 | -16.4 |
| stack-0.50 | -1.21 | +0.94 | +3.07 |
| stack-1.00 | -0.98 | +1.49 | +21.2 |

JPEG survival (stack-0.50 output re-encoded; "flagged" = AI images above 0,
composition vs each image's baseline):

| re-encode | corvi worst AI / flagged | commfor worst AI / flagged | real new FP (all det) |
|---|---|---|---|
| q90 | -1.10 / 0 | +1.13 / 4 (3 NEW: gem36-03/09/10, +1.1 each) | 0 (npr: 1) |
| q75 | -0.78 / 0 | +4.43 / 2 (paint-lioness +4.43, paid-36 +1.37) | 0 |
| q50 | -0.65 / 0 | +3.75 / 1 (paint-lioness) | 0 |

The q90 commfor positives are newly-created by the re-encode (their
baselines are -3.6..-5.0): aggressive re-compression of a stack output can
re-manufacture the high-frequency cues the ViT reads. For the detector that
actually flags Gemini (corvi), suppression is robust across the whole
q90..q50 range.

### NPR cautionary finding

Every Stage-A op pushes NPR scores UP (toward its fake class), on real
photos as much as AI (bilateral-0.50 on real: +114.7, 12/14 real new FPs;
mosaic-bilinear on real: 11/14 new FPs). Mosaic+demosaic IS the
interpolation artifact NPR detects. This is not fixable by dose tuning: any
camera-simulation CFA pass is an up/down-resample signature. NPR's
practical threat value is low precisely because plain resized real photos
trigger it too (-52 mean on the rescaled real subset), but Stage B's
surrogate ensemble should NOT include NPR-style residual detectors as
weighted voters, or it will optimize against a detector that flags the
world.

## Post-calibration decision (what shipped, C++ commit c5a551a)

The per-ingredient A/B was re-read as cost/benefit (cleared = baseline
positives now at or below the detector's threshold; perturbation = mean
|dx| over the AI set; commfor p = mean sigmoid over the AI set, baseline
0.216):

| op @ dose | perturbation /255 | corvi cleared (of 12) | commfor cleared (of 5) | commfor mean p |
|---|---|---|---|---|
| jpeg-96 | 0.72 | **12/12** | 1/5 | 0.18 |
| jpeg-92 | 1.03 | **12/12** | 2/5 | 0.19 |
| jpeg-88 | 1.27 | **12/12** | **3/5** | 0.11 |
| mosaic-edgeaware | 1.84 | **12/12** | 1/5 | 0.28 |
| mosaic-bilinear | 1.86 | **12/12** | 1/5 | 0.32 |
| mosaic-malvar5x5 | 5.15 | 1/12 | 1/5 | 0.32 |
| bilateral-0.25 | 0.89 | **12/12** | 0/5 | 0.30 |
| bilateral-0.50 | 1.11 | 11/12 | 0/5 | 0.29 |
| bilateral-1.00 | 1.32 | 11/12 | 0/5 | 0.27 |
| noise-0.25 | 2.45 | **12/12** | 1/5 | **0.46 (worse than baseline 0.22)** |
| noise-0.50 | 3.38 | **12/12** | 3/5 | 0.35 |
| noise-1.00 | 4.64 | **12/12** | 3/5 | 0.39 |
| ca-0.25 / 0.50 / 1.00 | 2.13 / 2.12 / 2.79 | 9, 7, 11 of 12 (never all) | 1/5 | 0.24 / 0.30 / 0.19 |
| vignette-0.25 / 0.50 / 1.00 | 6.61 / 9.16 / 14.25 | 11/12 (never all) | **0/5 at every dose** | 0.21 / 0.21 / 0.20 |

Reading: the JPEG cycle does almost all of the detector work at ~1/255 of
visible perturbation; the cheap mosaic kernels (edgeaware / bilinear) add
full corvi coverage at ~1.9/255; everything else is measured dead weight --
vignette never fully clears corvi and never clears commfor while costing up
to 14.3/255 (the main visible-degradation source; manual verification of
the first build measured ~13/255 mean perturbation / PSNR ~23 dB at the old
default), CA clears nothing fully at 2-3/255, low-dose noise makes commfor
MORE suspicious (mean p 0.216 -> 0.462 at the 0.25 dose), and the malvar
kernel costs ~3x edgeaware in compute (32.3 vs 9.0 ms per 1.1 MPix in the
Python mirror; 2.8x in the C++ measurement) while being the one kernel
whose output corvi still flags (11/12 remain, i.e. it clears only 1/12).

Dose ladder transcribed into `dose_for_strength` (the CALIBRATED ladder;
`physics_reference.py` mirrors it, `ladder="legacy"` reproduces the
pre-calibration values below it):

| strength | bilateral_d | noise_a / noise_b | ca_px | vignette_k | jpeg_q | kernel draw |
|---|---|---|---|---|---|---|
| <= 0.05 (off) | 0 | 0 / 0 | 0 | 0 | - | - |
| 0.50 (default) | 8 | 0 / 0 | 0 | 0 | U[88,96] | U{edgeaware, bilinear} |
| 1.00 (max) | 11 | 2.1e-3 / 2.0e-5 | 0 | 0 | U[88,96] | U{edgeaware, bilinear} |

- noise ramps in only above strength 0.75: `t = clamp((s-0.75)/0.25)`,
  `a = 2.1e-3 t`, `b = 2.0e-5 t` (full dose measurably helps commfor, 3/5
  cleared at 4.6/255; the sub-0.5 doses hurt it).
- CA and vignette stay at 0 in the ladder; the ops remain available for
  experiments via direct op calls in the harness.
- Production kernel draw excludes malvar (cost + the only kernel corvi
  still flags). malvar stays in the bank for explicit-kernel experiments.
- CAVEAT on the historical data: every `stack-*`, `vignette-*`, `ca-*`,
  `noise-*` (dose values) row in `results/per_image.csv` was produced under
  the PRE-calibration ladder (noise_a = 2.1e-3 s, ca_px = 0.5+0.7 s,
  vignette 0.10+0.25 s, kernel draw over all three kernels). Those rows are
  the calibration INPUT and are historical; re-running `run_eval.py` today
  applies the calibrated ladder (use `--legacy-stack` to reproduce them).

The earlier recommendation recorded here at first measurement ("confirm the
literature-anchor ladder, default 0.5, keep CA/vignette") is superseded by
the above; it remains true that strength > 0.75 backfires for commfor under
the legacy heavy ladder (6 flagged at 1.00 vs 2 at 0.50) and that the JPEG
cycle is the load-bearing op.

### Real-set false-positive movement (the production-relevant FP side)

How the pass moves REAL camera photos (n=14) across each threshold; "new
FP" = images that were below 0 at baseline and above 0 after the op:

- **corvi**: 0/14 flagged at baseline, 0 new FPs under every single-op and
  stack condition, including stack-1.00 and survival q50. The worst real
  score anywhere in the grid is -2.61 (stack-1.00), still far below 0.
- **commfor**: 0/14 at baseline, 0 new FPs under every condition (worst
  real after stack-1.00: -6.94). Real photos move AWAY from the threshold
  under the stack (baseline max -1.72 -> -6.16 at stack-0.50).
- **npr**: 2/14 flagged at baseline (both Commons-rescaled derivatives).
  Stage-A ops ADD real FPs (bilateral-0.50: +12, mosaic-bilinear: +11,
  noise-0.50: +10, stack-0.50: +1, stack-1.00: +2, survival-q90: +1) --
  further evidence NPR-style residual detectors must not be Stage-B voters.

Bottom line for the shipped defaults: the calibrated pass is
false-positive-free on the real set for both production surrogates
(commfor, corvi) at every strength.

Per-detector "flip"/report thresholds (for the stage-C report):

Per-detector "flip"/report thresholds (for the stage-C report):

| detector | threshold | baseline AI-positive band | post-stack-0.5 worst AI | verdict |
|---|---|---|---|---|
| corvi | 0 (logit) | +0.02..+2.15 | -1.21 | cleared, robust |
| commfor | 0 (logit) | +1.0..+4.7 | +0.94 | 1 residual (AI painting) |
| npr | 0 (logit) | (inverted; do not use as a vote) | +3.07 | excluded |

## Corvi ONNX export

`experiments/antidetect-eval/export/corvi-grag2021-latent.onnx`
(export_corvi_onnx.py; opset 18, single self-contained file, ~94 MB):

- Graph: input `image` (N,3,H,W) float32 RGB in [0,1], dynamic H/W ->
  ImageNet normalization baked in -> res50stride1 -> spatial mean ->
  output `logit` (N,1); logit > 0 = AI. sigmoid(logit) = p_fake.
- torch-vs-ONNX parity (CPU, 3 fixtures; inputs > 1.2 MPix capped as in
  scoring): AI 896x1200 |d| = 3.3e-05; AI 2400x1792 (capped) |d| = 7.4e-06;
  REAL 3474x2316 (capped) |d| = 1.9e-06. Assert < 1e-3 in the script
  (`results/parity.log`).
- sha256 `7f8a33d4d4bf89ee30251a13058b9d0c0c550d4f15f755cec77ad3fdfae0d242`
  (94,262,471 bytes).
- Export notes: the torch 2.13 dynamo exporter emits opset 18 (17 fails its
  version converter for this graph) and writes weights as external data by
  default; the script inlines them (`onnx.save_model(...,
  save_as_external_data=False)`) so the artifact is one file. onnxruntime
  >= 1.16 runs opset 18 (wmr vendors 1.27.1).
- Intended upload path `antidetect/corvi-grag2021-latent.onnx` in
  `froggeric/wmr` -- see `experiments/antidetect-eval/export/UPLOAD-MANIFEST.md`
  for sha256s. NOT uploaded by this harness.

## Limitations

- Single machine, CPU inference for corvi/npr (MPS on the shared dev box
  stalls under GPU contention; logits verified identical between devices).
- 25 AI fixtures from ONE generator family (Gemini 3.x) and 14 real photos
  (3 native originals + 11 Commons-rescaled derivatives). No SDXL/MJ/Flux
  population; the commfor finding in particular is Gemini-specific.
- corvi inputs above 1.2 MPix are bicubic-capped for the stride-1
  activation budget; baseline and conditions share the cap per fixture, so
  deltas are internally consistent, but absolute corvi logits on large
  images are not "native-resolution" numbers. (Corvi is
  resolution-sensitive: the large AI images score -5..-11 partly because of
  the cap and partly content.)
- commfor sees a 384x384 center crop only (its fixed preprocessing); its
  verdicts on non-center-subject images reflect that crop.
- The physics mirror is distribution-identical to the C++ TU, not
  byte-identical (numpy PCG64 vs std::mt19937_64). The C++ unit tests
  remain the byte-level contract.
- Detector score conventions (logit > 0 = fake) verified empirically
  against the fixture baseline for commfor and corvi; NPR's convention
  follows its own validate.py but its behavior on this population is
  inverted/unstable -- treat NPR as manual-follow-up, not a shipped
  adversary.
