# Anti-detection pipeline: implementation record

Date: 2026-09-01. Status: v1.17.0 (committed; release pending the owner's manual
verification).
Companion docs: the detector-side landscape (`ai-image-detection-landscape.md`),
the attack-side research that chose this architecture
(`anti-detection-pipeline-research.md`), and the M0 calibration tables once they
land (`antidetect-m0-calibration.md`).

This record describes what shipped, the design decisions behind it, and what was
measured on the way. It is written so a future contributor can change the
pipeline without re-deriving the constraints.

## What shipped

`wmr antidetect <in> -o <out>` and a chained `--antidetect` flag on
`remove` / `synthid` (batch included). Two stages:

- **Stage A, physics** (`src/core/antidetect_physics.{hpp,cpp}`): re-injects
  camera-pipeline statistics. Pure OpenCV, zero model downloads, deterministic
  under `--seed`.
- **Stage B, adversarial** (`square_attack.{hpp,cpp}`,
  `detector_suite_ort.{hpp,cpp}`, `antidetect_adversarial.cpp`): a bounded
  Square Attack against the local surrogate detector suite via ONNX Runtime,
  LPIPS-gated. Requires the surrogate models (downloaded on first use,
  SHA256-pinned); degrades to physics when absent.

Method selection: `--method full` (default; A then B) / `auto` (full, silently
degrading to physics) / `physics` (A only) / `adversarial` (B only; refuses
with exit 1 and an unchanged image when models are unavailable).

## Stage A: op order and why

The order mirrors a real camera pipeline, because the point is to produce
statistics that are *correlated the way sensor data is*:

1. **Bilateral pre-clean.** Removes the smooth diffusion-decode trace (the
   strongest counter-signal: removal-trace classifiers detect regen output at
   over 99%; bilateral filtering measurably degrades them) without killing
   real edges.
2. **sRGB -> linear LUT**, noise added in linear: Poisson-Gaussian with
   `sigma(x) = sqrt(a*x + b)` per channel, so shot noise scales with
   signal the way physics dictates.
3. **Noise BEFORE the mosaic** so it passes through demosaic interpolation
   the way real sensor noise does (correlated, band-limited) rather than
   sitting on top as i.i.d. pixels an oracle can spot.
4. **Bayer RGGB mosaic -> demosaic with a randomized kernel** (Malvar-He-Cutler
   5x5 / bilinear cross / bilinear diagonal / edge-aware, drawn per image).
   Randomization is mandatory: published counter-forensics detect a
   *fixed-kernel* re-demosaic at 99.9% (Chen/Zhao/Stamm-class); a randomized
   kernel bank moves the CFA statistics into a distribution rather than a
   point the detector can fingerprint.
5. **Lateral chromatic aberration** (sub-pixel opposite-sign R/B warp) and
   **vignetting**, both weak.
6. **One camera JPEG cycle** (4:2:0, q88-96), unless `--no-jpeg-cycle`. Also
   the carrier that Stage B's DCT-band projection is designed to survive.

Kernel constants (Malvar-He-Cutler) were verified against the
`colour-demosaicing` reference implementation; the initial suspicion that they
were wrong turned out to be two OpenCV bugs (below).

Doses scale with `--strength` via `dose_for_strength()` (identity below
s = 0.05 so "no-op" really is one; the CA/vignette floors once broke that).
**The ladder is calibrated** (2026-09-01, from the M0 per-ingredient A/B in
`antidetect-m0-calibration.md`): the first shipped constants were
physical-realism literature anchors, and manual verification found the
default-strength output visibly degraded (~13/255 mean perturbation, PSNR
~23 dB) while the A/B showed the expensive ingredients buy almost nothing.
Measured on 25 AI fixtures x 3 detectors: the JPEG cycle alone clears all
12 corvi-flagged images at 0.7/255 and 3/5 commfor-flagged at 1.3/255;
vignette clears nothing at any dose while costing up to 14/255 (the single
biggest visible-degradation source); CA changes nothing at 2-3/255; low-dose
noise makes commfor MORE suspicious (mean p_fake 0.22 -> 0.46; only the
full 1.0 dose helps, 3/5 cleared); malvar costs 2.8x edgeaware and is the
one kernel whose output corvi still flags. The calibrated default is
therefore bilateral + randomized {edgeaware, bilinear} CFA round-trip +
JPEG cycle (~2/255, PSNR ~36-40 dB, invisible), noise ramping in only at
s >= 0.75, CA/vignette 0 at every strength. The full stack retains a
one-image commfor-margin advantage (4/5 vs 2-3/5) at 6-8x the perturbation:
that trade is left to explicit high `--strength`, not the default.

## Stage B: Square Attack, DCT-band projection, LPIPS gate

Square Attack (Andriushchenko/Croce/Flammarion/Hein, arXiv 1912.00049;
reference code BSD-3; reimplemented from the paper, no code copied) is a
score-based random search: propose square-shaped perturbations at random
positions/sizes/colors, accept when the ensemble's mean p_fake improves. It
needs no gradients, which matters because ORT is inference-only (the
backward-graph export that would enable gradient attacks is a documented
follow-up, not v1).

Three quality constraints compose:

- **L∞ box**: eps <= 4/255 at full strength (`(1 + 3*strength)/255`).
- **DCT-band projection** (`dct_band_weight`, keep_frac 0.2): after each
  accepted step the perturbation is projected onto the lowest DCT corner, the
  band JPEG quantization preserves best (the FBA2D/DuFIA finding). The eps
  clamp must run AFTER the projection: the projection redistributes energy and
  can push pixels back outside the box.
- **LPIPS gate**: a candidate is accepted only if `LPIPS(base, candidate) <=
  lpips_budget` (default 0.05). The metric is a hand-ported AlexNet layer-2
  LPIPS (weights extracted from the official calibration into
  `assets/lpips_alex_l2.h`, golden-tested against torch to ~1e-5). Layer 2
  (conv1+pool1+conv2) is the AEROBLADE-style configuration: enough depth to be
  perceptually meaningful, small enough to run per query without a GPU.

Query budget: `200 + 800*strength`, capped at 1000 (at ~6-7 q/s CPU against
commfor on an M4 that is ~90 s for the default strength 0.5). Early stop when
every surrogate flips below its threshold.

## Surrogate suite

`kSurrogateManifest` (`detector_suite.hpp`, constexpr, ORT-free header) is the
single source: key, filename, SHA256 pin, preprocessing contract
(`BakedGraph` = the graph does everything, feed float RGB [0,255];
`ResizeCropNorm` = shortest-side resize + center crop + ImageNet norm), and
the flip threshold. Adding a surrogate = one entry + one pinned file; no logic
changes.

- **commfor** (Community-Forensics ViT-S/16 @ 384): pinned, the FIXED fp32
  ONNX (87 MB, weights regenerated 2026-07-22), MIT. The "what the public
  actually runs" detector. Prep gotcha: the shortest-side resize must be
  ANTIALIASED (cv::INTER_AREA on downscale; a plain bicubic/linear downscale
  manufactures high-frequency energy the ViT reads as generative, measured
  2-4 logit shifts toward fake). The first pin (the sibling `-ONNX` repo's
  fp16 export) predates the weight fix and is a dead oracle — see discovery 0.
- **corvi** (GRIP Grag2021_latent): Apache-2.0, ~282 MB. Pin lands with M0
  (an empty SHA in the manifest = cleanly disabled, logged in the fetch note,
  never an error).
- **npr** (NPR-DeepfakeDetection): upstream has no license file; documented,
  not pinned, not shipped (license is the owner's call; nothing is excluded
  silently).

Models resolve from `$WMR_ANTIDETECT_MODELS_DIR` > exe-relative dirs >
`~/.cache/wmr/antidetect`, download pinned from wmr's own model host with the
`.sha256.ok` sidecar fast path (the CoreML-fetch pattern), and are clearable
via `wmr cache --clear-antidetect-models` (user-cache copies only).

## Build gates and platform split

- `WMR_BUILD_ANTIDETECT` (default OFF in CMake, ON in `scripts/build.sh` and
  CI): physics + LPIPS + facade + CLI + fetch. The CLI is always registered;
  an OFF build prints a build-free error.
- `WMR_BUILD_ANTIDETECT_ADVERSARIAL` (default ON): Stage B + ORT. Auto-forced
  OFF when the parent is OFF or on a mac-x86_64 cross-compile
  (`CMAKE_OSX_ARCHITECTURES` lacking the host processor), because **ORT 1.27.1
  publishes no osx-x86_64 binary**. The mac-x86_64 release is physics-only
  by design; building ORT for x86_64-osx is the documented follow-up.
- ORT itself is the shared pinned fetch (`if(WMR_NEED_ORT)`): MI-GAN on
  linux/windows and antidetect on mac-arm64/linux/windows. The mac-arm64
  package now ships `lib/libonnxruntime.1.dylib` (bundled + signed by the
  existing scripts).

## Test-time discoveries (the expensive ones)

0. **The first commfor pin was a dead oracle (caught 2026-09-01, pre-commit).**
   The ready-made fp16 ONNX in
   `buildborderless/CommunityForensics-DeepfakeDet-ViT-ONNX` predates that
   repo's 2026-07-22 weight fix (its own CHANGELOG: the old export was
   converted from different weights with a wrong classifier head). Measured on
   four fixtures: the fixed fp32 scores +4.1/-3.4/-1.7/-4.4 logits (clear,
   correct discrimination) while the stale fp16 gives -0.01/-0.55/+0.36/+0.15
   (near-zero, uncorrelated). Every early end-to-end run therefore showed a
   flat 0.535 score — the Square Attack was optimizing against noise. Re-pinned
   to the fixed repo's fp32 `onnx/model.onnx` (87,442,080 B). Rule: when
   validating an adversarial stage, FIRST verify the oracle discriminates on
   known-labeled inputs (fake should score high, real low) — a dead oracle
   looks exactly like "the attack doesn't work".
0b. **The Malvar blue plane had a double-swapped kernel phase (caught the same
   day, by the colorful-content rule).** `assemble_plane`'s `chroma_is_red=
   false` branch already mirrors the G-phase masks for blue (blue at a gr site
   is vertical, at a gb site horizontal); the Malvar blue call ALSO swapped the
   (row, col) kernel arguments — two swaps cancel, so blue G-sites were
   estimated with wrong-orientation kernels. Invisible on neutral content
   (every kernel estimates the same value when R==G==B, so all the neutral unit
   tests passed) but a ~10 dB PSNR crater on colorful content (~25% of default
   runs drew the malvar kernel). Fixed by passing (horizontal, vertical) order
   for both planes; regression test added on smooth colorful gradients.
1. **SSIM luminance and contrast terms are MULTIPLICATIVE.** An additive
   formula scores ~1 for literally everything (including flat+sigma15 noise
   that should read ~0.4). Caught only because the report test asserted a
   meaningful SSIM on degraded content.
2. **Masked OpenCV ops leave non-mask entries of a pre-existing dst
   untouched.** The Bayer mosaic produced garbage (mean 223 on a neutral 128
   solid) because masked writes accumulated stale channel values. Zero the
   scratch between masked writes.
3. **cv::Range has no stride constructor.** Build per-site Bayer masks as
   2x2-tiled masks via `cv::repeat`.
4. **Flat-content LPIPS layer-2 is exactly scale-invariant** (per-channel
   spatial unit normalization cancels the scale of a constant map). Budget
   tests must use textured content; a flat base can make a tight budget test
   pass vacuously.
5. **The greedy Square Attack needs a scorer with a reachable target inside
   the eps box.** A continuous mock whose boundary lies outside the box never
   flips; a step function gives the greedy loop nothing to accept. The test
   mock is a steep ramp crossing the boundary inside eps.
6. **PSNR reads misleadingly low on dark content** (~23 dB on a near-black
   image while LPIPS is 0.001): the physics noise is added in linear domain
   where shadows carry little energy, and sRGB amplifies it. Judge quality by
   LPIPS on dark content; the help text's PSNR band applies to normal-content
   images.
7. **The early "flat 0.535" observation was the dead oracle, not attack
   weakness.** With the fixed surrogate, Stage A physics ALONE takes a
   strongly-flagged fake image (raw commfor logit +4.1, p_fake 0.98) to
   p_fake 0.007 on the 2400x1792 poster fixture — the adversarial stage then
   finds nothing left to push (0.007 -> 0.007). One fixture, one detector;
   the M0 tables (39 fixtures x 26 conditions x 3 detectors) are the real
   measurement. But it is the first direct evidence for the research bet that
   correlated camera statistics sits outside these detectors' training
   distribution.

## Verification performed

- Unit: physics op behaviors (mosaic/demosaic round-trip on neutral AND
  colorful content, noise monotonicity, dose-ladder identity below 0.05,
  determinism), LPIPS golden vs torch (~1e-5), SSIM semantics, report wording
  locks, Square Attack (flips a reachable mock within the eps box, respects
  budget + LPIPS gate, deterministic, DCT projection kills high-frequency
  energy), downloader (file:// + SHA + refusal), ORT smoke (SKIPs unless
  `WMR_TEST_ANTIDETECT_MODELS` is set).
- Full suite green with the feature ON.
- End-to-end: `wmr antidetect <image> --method full --seed 42` runs physics +
  600 adversarial queries (~90 s CPU), prints the honest report, and reruns
  byte-identically under the same seed.
- OFF build (`-DWMR_BUILD_ANTIDETECT=OFF`): no pipeline code in the binary,
  subcommand prints the build-free error.
- Degrade paths: cold cache + `--no-download` -> physics + exit 0; explicit
  `--method adversarial` cold -> exit 1, no output written.

## The eps-8 transfer program (2026-09-02): what the experiments found

A sequence of owner-requested experiments against Hive Detect, run at eps
8/255 (the literature's regime) with env-gated escapes, produced three
discoveries before producing a single valid artifact:

1. **The local oracle saturates on everything we throw at it.** The raw
   ChatGPT image scores commfor 0.001 / corvi 0.000 - the local ensemble
   is too weak to flag it, so the attack early-stops at 0 queries. A
   transfer artifact needs a locally-flagged input (the Gemini poster,
   commfor 0.983).
2. **A mean objective freezes on a saturated ensemble member.** Artifact
   C: commfor 0.990 -> 0.996 over 1000 queries while corvi sat at 0 -
   any candidate nudging corvi up vetoed acceptance. Fixed with a margin
   objective (sum of max(0, score - threshold)): already-won detectors
   stop mattering.
3. **The attack had never actually run.** Artifacts C/D/E accepted ZERO
   moves in 3000 combined queries (PSNR ~99 dB vs base). Root causes (see
   commit c6b88a3): a 255x eps-domain mismatch (increments ~0.03 of one
   level instead of 8) composed with a round-half-to-even artifact that
   bumped every odd-valued pixel +1 (the mock unit test passed BECAUSE of
   this - it validated the bug). Both fixed; the test now earns its flip
   with real energy.

The first run of the fixed attack (artifact F, poster, eps 8/255, 1000
queries, ~28 min on the CoreML EP): **commfor 0.991 -> 0.435** (flipped
across the decision boundary), corvi 0.112 -> 0.067, at PSNR 37.3 dB /
LPIPS 0.000 - the DCT band projection keeps eps-8 imperceptible while
moving the detector. Every earlier "adversarial" Hive upload (D: 99.9%,
gemini3 45.5% / wan 44.4%) tested an artifact containing no attack.

**The transfer verdict (F uploaded to Hive Detect): 99.9% AI, 98.4%
gemini3.** A perturbation that flips the local ViT ensemble at
imperceptible quality moved a top commercial detector not at all - the
attribution actually sharpened (98.4% vs D's 45.5/44.4 split; the smooth
low-band perturbation apparently removed the noise that had Hive
second-guessing its generator attribution). The cheap transfer path is
measured-dead end to end: working attack, right eps regime, 1000
queries, local flip achieved, zero commercial movement. Mechanistic
reading: we crafted in the DCT low band because it survives JPEG - but
that is exactly the band committee/commercial detectors are hardened
against by training augmentations (blur, color jitter, compression). The
JPEG-robust band and the transfer-susceptible band are structurally
opposed; that tension, not query budget or eps, is the wall. What
remains untried: crafting against a committee-LIKE local surrogate
(self-trained DINOv2/DINOv3 ensemble) - the literature's only measured
Hive degradation (~100 -> 74-77%) came from multi-model crafting at
visible eps, and even that is degrade, not defeat.

**Watermark-hypothesis control (G, same attack on the watermark-free
ChatGPT image, maximally aggressive: eps 8, perceptual gate disabled,
sum-of-scores objective): 99.8% AI, 97.5% gptimage2.** The visible
diamond was NOT the transfer blocker - watermark-free craft moves Hive
just as little. Two further measurements fell out: (a) the aggressive
noise RAISED Hive's face-classifier suspicion (deepfake-likely
0.1-2.2% -> 10.2%) - aggression is counterproductive on the face axis
too; (b) the owner judged G's quality "too degraded to be unusable"
while every automated metric read benign (PSNR 32.6 dB, SSIM 0.946,
LPIPS 0.000) - a second confirmed instance of the LPIPS-layer-2 low-band
blind spot (the vignette lesson): accumulated smooth low-band shifts are
invisible to the metric and visible to humans. The eps-4/255 CLI cap +
the 0.05 LPIPS gate are the shipped guards against exactly this; the
experiment envs bypass them deliberately and the outputs are not product
representative. Any future "imperceptible" claim from these metrics
alone needs human eyes behind it.

Supporting changes from the same program: the CoreML execution provider
for the ORT suite (corvi hot path 9.9s -> 1.4s per forward, ~3x
end-to-end query rate; providers differ ~0.01 logit, so pinned A/Bs must
pin the provider via WMR_AD_DISABLE_COREML), and honest progress
summaries for early-stopped runs.

## External validation (owner-run, 2026-09-02)

Three-way A/B against Hive Detect (hivedetect.ai, the official demo of
the best independently measured commercial detector), one image
(`2400x1792-test2-gemini.png`):

| variant | Hive verdict | attribution |
|---|---|---|
| `remove --antidetect` (no regen; original pixels + calibrated physics; local suite 0.008/0.000) | **99.9% AI** | 64.6% gemini3, 30.9% imagen4, 2.3% seedream |
| `synthid --antidetect` (SDXL regen + physics + adversarial; local suite 0.040/0.217) | **95% AI** | 85.2% leonardo, 4.6% krea, 1.5% qwen, 2.2% zimage |
| ChatGPT image, no watermark present (pure physics on untouched GPT-Image pixels, the cleanest control; local suite 0.006/0.000) | **99.9% AI** (+15.2% "deepfake-likely", Hive's separate face classifier on the photorealistic human subject) | 87.3% gptimage2, 7.7% meta |
| Same ChatGPT image at `--strength 1 --lpips-budget 0.2` (maximum: noise 6/255, PSNR 30.1, SSIM 0.698 — visibly degraded) | **99.9% AI** (**0% deepfake-likely**: the heavy noise DID clear Hive's face/manipulation classifier) | 94.3% gptimage2, 4.7% gptimage1_5 |

(The raw original was not separately uploaded; the gemini3/imagen4
attribution on the no-regen variant implies the control would flag at
least as strongly.)

Readings, in order of importance:

1. **The calibrated invisible dose does not touch Hive's signal.** Hive
   read through ~2/255 of bilateral + CFA + q93 JPEG and still named the
   correct generator family. Our physics was calibrated to clear OPEN
   detectors (corvi 12/12, commfor 3/5) at imperceptible cost; a
   58-generator commercial ensemble trained on Gemini/Imagen output is a
   different weight class, as the honesty lock always scoped.
2. **Regen lowered Hive's confidence (95 vs 99.9) and scrambled the
   attribution** (Leonardo-family instead of Gemini) but both variants
   are flagged. Regen changes WHICH generator is detected, not whether.
3. The adversarial stage early-stopped in both runs (local surrogates
   already below threshold): no eps<=4/255 attack crafted on a 2-model
   open ensemble transfers to a dissimilar commercial one (the
   literature's measured Hive drops, 100 -> 73.8-76.6, required eps
   8/255 with DI-style multi-model crafting, and even that degrades
   rather than defeats).

Empirical bottom line for v1 scope: `wmr antidetect` clears the open /
forensic detector families it was calibrated against (with 0/14 real
false positives) and is ineffective against top commercial detectors at
its default imperceptible dose. Passing or degrading Hive-class is the
documented future work: a self-trained DINOv2-committee surrogate
(architecturally similar to Hive/Pangram foundations), eps-8 transfer
crafting, an explicit visible-quality trade-off, and the honest
expectation of degrade-not-defeat (analog print/rescan remains the only
measured near-total killer of commercial detection, and it is not a
software path).

## Open items

- Transcribe M0's calibrated dose ladder + flip thresholds into
  `dose_for_strength` + `kSurrogateManifest`; pin corvi; upload both ONNX
  files to the model host.
- Revisit the default strength/eps once the ensemble (commfor + corvi) is
  live; the single-surrogate flat result above suggests defaults may need to
  start higher, but that is a calibration decision, not a code decision.
- mac-x86_64 adversarial (build ORT for x86_64-osx).
- Phase-2: backward-graph ONNX exports for gradient attacks (DI/DuFIA-class),
  self-trained DINOv2 committee surrogate, JPEG quant-table vendor variance.
