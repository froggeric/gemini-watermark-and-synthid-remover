# Anti-detection research archive (closed 2026-09-02)

A self-contained archive of a research program: **making AI-generated images
pass AI-image detectors** (a `wmr antidetect` pipeline: camera-statistics
physics + an adversarial Square Attack against local surrogate detectors).
The feature was fully built, calibrated, and measured — and then **removed
from the app before any release** because the measurements showed it does
not achieve the goal against the detectors that matter, and shipping it
would have been misleading. This archive keeps the methodology, the
findings, and the code so someone else can build on it.

**The verification oracle for all external numbers: [Hive Detect](https://hivedetect.ai)**
(Hive's official public demo; the best independently measured commercial
detector — peer-reviewed CCS 2024: 98.03% accuracy, 0% FPR; survives JPEG
q15). Free, no signup for single images.

## The verdict, in one table

Every row is an owner-uploaded image scored by Hive Detect (2026-09-02):

| # | input | treatment | Hive verdict | attribution |
|---|---|---|---|---|
| 1 | Gemini 2400x1792 | physics @ default (invisible, ~2/255, PSNR 42 dB) | **99.9% AI** | gemini3 64.6% (correct) |
| 2 | Gemini 2400x1792 | SDXL regen + physics + attack | **95% AI** | leonardo 85% (family swap) |
| 3 | ChatGPT image | physics @ default (no watermark present) | **99.9% AI** | gptimage2 87.3% (correct) |
| 4 | ChatGPT image | physics @ max strength (visible) | **99.9% AI** | gptimage2 94.3%; deepfake-likely 15.2% -> **0%** |
| F | Gemini 2400x1792 | real eps-8/255 craft (flips the local ViT ensemble at LPIPS 0.000) | **99.9% AI** | gemini3 **98.4%** (sharpened) |
| G | ChatGPT image | max-aggression craft (gate off; visibly degraded) | **99.8% AI** | gptimage2 97.5%; deepfake-likely **10.2%** (worse) |

Nothing moved a top commercial detector: not invisible physics, not visible
physics, not regeneration, not imperceptible adversarial craft, not visibly
aggressive craft, with or without a watermark. Full detail, including the
confound-elimination chain (dead-oracle pin, mean-objective freeze, two
rendering defects that had silently disabled the attack since day one):
`docs/antidetect-implementation-record.md`.

## What works (and is worth knowing)

- **Against open/forensic detectors, camera-statistics physics works.** The
  camera JPEG cycle alone clears all 12 corvi-flagged images at 0.7/255
  (~49 dB); the calibrated stack (bilateral + randomized-kernel CFA
  round-trip + JPEG, ~2/255, PSNR ~36-42 dB) clears corvi 12/12, commfor
  3-5/5, with **0/14 real-photo false positives**. Tables:
  `docs/antidetect-m0-calibration.md` (3,042 measurements: 39 fixtures x 26
  conditions x 3 detectors).
- **The fixed Square Attack flips local detectors imperceptibly.** commfor
  0.991 -> 0.435 at eps 8/255 in 1000 queries (~28 min with the ORT CoreML
  execution provider), PSNR 37 dB / LPIPS 0.000. The DCT low-band
  projection concentrates eps-8 into smooth color shifts.
- **Hive's face/deepfake classifier is fragile** (cleared by max-strength
  noise; raised by aggressive craft). Its generator attribution is not.

## Why transfer fails (the mechanism we settled on)

We must craft in the DCT low band because that band survives JPEG
re-encoding — and that is exactly the band commercial/committee detectors
are hardened against by training augmentations (blur, color jitter,
compression). **The JPEG-robust band and the transfer-susceptible band are
structurally opposed.** That tension, not query budget or eps, is the wall.
The literature's only measured Hive degradation (~100 -> 74-77%) came from
DI-style multi-model crafting at visible eps, and even that is degrade, not
defeat. The one untried path: craft against a **committee-like local
surrogate** (self-trained DINOv2/DINOv3 ensemble) — see
`docs/anti-detection-pipeline-research.md` for the design and the honest
expectations.

## Hard-won methodological lessons (all measured here)

1. A unit test can validate a rendering bug: the mock attack test passed
   BECAUSE a round-half-to-even artifact flipped its odd-valued base
   instantly. Make mock flips require real perturbation energy.
2. Verify an adversarial attack actually ACCEPTS moves. PSNR ~99 dB vs the
   base after a "full run" = zero accepts = something upstream is dead.
3. Verify an oracle discriminates on known-labeled inputs before trusting
   any attack result (we pinned a broken checkpoint for a day: near-zero
   uncorrelated logits; the fixed one is
   `buildborderless/CommunityForensics-DeepfakeDet-ViT` `onnx/model.onnx`,
   sha256 `a42c7d74...` — NOT the deprecated `-ONNX` sibling repo).
4. A saturated ensemble member freezes a mean-minimizing attack; use margin
   (sum of max(0, score - threshold)) so already-won detectors stop vetoing.
5. LPIPS layer-2 does not see smooth low-band damage (vignette, accumulated
   craft): metric-only "imperceptible" claims need human eyes.
6. Per-ingredient dose calibration beats physical realism: the
   literature-anchored ladder cost 13/255 for near nothing; the measured
   frontier does the work at ~2/255 (vignette and CA clear NOTHING at any
   dose; low-dose noise makes a ViT MORE suspicious).

## How to explore / revive

- This directory is self-contained: `src/` (the pipeline sources + the
  LPIPS weights header), `tests/`, `docs/` (the four research records),
  `eval/` (the standalone Python harness that produced the calibration
  tables; see `eval/README.md`).
- **To revive the feature as a building `wmr antidetect`**: start from the
  v1.16.11-era tree, copy `src/*.cpp|hpp` into `src/core/` (and
  `src/lpips_alex_l2.h` into `assets/`), the `tests/` files into
  `tests/unit/`, then apply `wiring-against-1.16.11.patch` (the exact
  cumulative diff of the feature's changes to the shared files: CMake
  gates + the shared ORT fetch, CLI subcommand + chained flags, vcpkg
  feature, build.sh, release CI). That reconstructs the complete,
  suite-green feature state that was measured for this record.
- Surrogate models (never uploaded anywhere; pins + SHAs in
  `eval/UPLOAD-MANIFEST.md`): commfor (MIT) from the repo above; corvi
  (Apache-2.0) exported from `buloutian/corvi-2022-mirror` via
  `eval/export_corvi_onnx.py` (parity ~3e-5). NPR is measured INVERTED on
  our fixtures (flags real photos) — do not ensemble it.
- The ORT CoreML EP comes with the stock osx-arm64 prebuilt; corvi's
  stride-1 ResNet-50 needs its input capped at 1.2 MPix (that cap is part
  of the calibrated threshold contract).
