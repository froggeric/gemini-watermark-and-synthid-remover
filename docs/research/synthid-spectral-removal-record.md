# SynthID spectral detection + suppression: removal record

Date: 2026-08-05. Version: 1.16.0. Branch: `synthid-spectral-removal`.

This is the canonical decision record for the removal of wmr's spectral SynthID
detection and suppression path in v1.16.0. It records what was deleted, what was
kept, why, what was tried over the whole SynthID effort, the data blocker, and the
conditions under which detection could be revisited. Read this before reopening any
SynthID detection work.

## TL;DR

The spectral SynthID detector and suppressor did not work. They are removed. The
only SynthID operation that ships is `--synthid-attack regen` (lossy SDXL img2img,
the single attack the published literature reports as validated against
SynthID-Image). Detection is out of wmr entirely: no public SynthID-Image verifier
exists, so any detection claim is unverifiable, and our detector had no
discriminative power anyway.

## What was removed

Source (C++):
- `src/synthid/` (the whole directory): `codebook_builder.{hpp,cpp}`,
  `codebook_subtractor.{hpp,cpp}`, `noise_residual_subtractor.{hpp,cpp}`,
  `spectral_codebook.{hpp,cpp}`.
- `src/detection/synthid_detector.{hpp,cpp}`.
- `src/core/fft_context.{hpp,cpp}` (spectral-only; nothing else in the tree used FFT).

Tests:
- `tests/unit/spectral_codebook_test.cpp`, `tests/unit/codebook_subtractor_test.cpp`,
  `tests/unit/codebook_builder_test.cpp`, `tests/unit/fft_context_test.cpp`,
  `tests/unit/synthid_wording_test.cpp` (the spectral honesty-lock; obsolete once
  spectral is gone), `tests/unit/lab_a_experiment_test.cpp` (the `--lab-a` experiment
  test, depended entirely on the removed subtractors).

Dev scripts (spectral-only):
- `scripts/build_synthid_codebook.sh`, `scripts/verify_removal.py`,
  `scripts/visualize_spectral.py`, `docs/research/synthid_content_probe.py`,
  `docs/research/synthid_content_ring_probe.py`, `docs/research/synthid_lab_a_probe.py`.
- Seven uncited spectral dev-analysis scripts that referenced the removed `.wcb`
  format/code and could no longer run: `scripts/analyze_carrier{,_deep}.py`,
  `scripts/analyze_channel_detector.py`, `scripts/analyze_phase_detector.py`,
  `scripts/build_codebook_from_folder.py`, `scripts/build_differential_codebook.py`,
  `scripts/convert_npz_to_wcb.py`. (Their findings live on in the research docs; none
  was cited by a kept doc.) `docs/research/ws2b_leakage_probe.py` is KEPT because two
  kept docs (`synthid-48-96-leakage-check.md`, `synthid-content-fixture-analysis.md`)
  cite it as their reproducibility anchor.

Build:
- FFTW3 dependency dropped from `CMakeLists.txt`, `tests/CMakeLists.txt`,
  `vcpkg.json`, `scripts/build.sh`, `CMakePresets.json`.
- CLI flags removed: `--codebook`, `--codebook-free`, `--phase-adaptive`, `--lab-a`,
  `--no-content-guard`, `--synthid-strength`, `--carrier-grid`, `--synthid` (on
  `remove`). The `build-codebook` subcommand is gone. `wmr detect` is visible-only.

`--synthid-attack` is kept as a pluggable method selector so a future SynthID method
is a localized add (one new IsMember value + one new dispatch branch). Its valid set
today is `{"regen"}`, default `regen`.

## What was kept (byte-for-byte)

- The diffusion-regen path: `src/core/regenerator.{hpp,cpp}`,
  `src/core/model_downloader.{hpp,cpp}`, `src/core/coreml_sd_*`,
  `external/stable-diffusion.cpp`, the `WMR_BUILD_REGEN` / `WMR_BUILD_AI_COREML_SD`
  CMake blocks, the `DiffusionRegen` enum value + `regen_*` InpaintConfig fields.
- The visible-watermark pipeline (NccDetector, still_geometry, blend, inpaint, video,
  NotebookLM, MI-GAN, AI denoise).
- Every research doc under `docs/research/` (the permanent findings record).

## Why (the evidence, one paragraph per reason)

1. **The spectral detector had no discriminative power.** Scored against 8
   Google-official-verifier-labeled images, the detector's ROC AUC was **0.20**
   (worse than random; its score is a content-property constant at ~61%). This is the
   first and only time the detector was scored against external ground truth.
   See `synthid-detection-validation.md`.

2. **No reliable third-party SynthID-Image detection exists.** A deep study across
   GitHub, HuggingFace, and the literature (14 sources deep-read, 16 adversarially
   verified claims) found every public attempt falls into one of four boxes
   (broken, unverifiable, surrogate-only, unrelated). See
   `synthid-detection-feasibility.md`.

3. **The spectral suppressors are inert on content images.** The SynthID-Image
   carrier amplitude is ~0.025/255 (sub-LSB), below the content noise floor. A clean
   codebook (fixed builder, or reverse-SynthID's external one through our subtractor)
   attenuates the carrier band by only +0.16 to +0.38% over the phase-noise baseline
   (within noise; PSNR 33 to 45 dB = no damage, but no suppression either). See
   `synthid-clean-codebook-eval.md` and `synthid-codebook-not-viable.md`.

The honest framing already shipped ("suppress, heuristic; not a verifiable removal")
described a path that did not even suppress. Keeping it would mean shipping a
capability that provably does not work. Removing it is honest code.

## What was tried over the whole SynthID effort

- **Spectral codebook subtraction** (`CodebookSubtractor`): multi-pass magnitude +
  phase subtraction from a `.wcb` codebook. Result: inert on content; the "dots"
   artifact was a builder bug (captured the visible diamond's broadband FFT) that
  the discriminative carrier-selection fix resolved, but a clean codebook is also
  inert on content. See `synthid-codebook-not-viable.md`,
  `synthid-clean-codebook-eval.md`.
- **Codebook-free noise residual** (`NoiseResidualSubtractor`): bilateral-filter
  denoise, take the residual, FFT, estimate the carrier. Result: inert on content
  for the same carrier-below-noise-floor reason. See
  `codebook-free-removal-research.md`, `synthid-codebook-vs-codebook-free-gate.md`.
- **LAB `a`-channel attack** (`--lab-a`): the carrier is luminance-based, so
  operating on the chrominance `a` channel was tried. Result: 3.74x worse than BGR.
  See `synthid-lab-a-experiment.md`.
- **Carrier-bin seeding** (`--carrier-grid`): seed candidate SynthID carrier bins
  (the (48,96) grid from a published reverse-SynthID repo) at consistency 1.0. Result:
  INCONCLUSIVE_LEAKAGE_SUSPECTED; the off-grid control also collapsed, so the signal
  is non-discriminative (likely leakage from the visible 48/96px diamond footprint).
  See `synthid-48-96-leakage-check.md`.
- **Discriminative builder fix + OOM guard**: real correctness fixes to the builder
  (replaced the saturated `1 - std/max_std` gate with
  `normalize(log1p(mean_magnitude) * phase_coherence)`; guard against all-zero/NaN
  profiles crashing the subtractor). Result: fixed the dots artifact, but confirmed
  the clean codebook is inert on content. These fixes shipped in 1.15.0; the code
  itself is removed here because the codebook is not a viable feature.

## The data blocker

There is no way to obtain the labeled data needed to build or validate a SynthID
detector:

- **No public SynthID-Image verifier.** Google's only open SynthID code is
  text-only (`google-deepmind/synthid-text`). The Gemini in-app "Verify with
  SynthID" tool is account-gated, rate-limited (~10 images per 24h), emits
  natural-language verdicts, and cannot be batched.
- **No controlled same-generator watermarked-vs-unwatermarked pairs.** Gemini
  always watermarks its output, so we cannot get the unwatermarked counterpart of
  any Gemini image. A detector trained on real-content pairs cannot exist without
  the negative class.
- **The only public "surrogate" labels are confounded.** The `fyxme/synthid-detector`
  HuggingFace space labels its inputs, but those labels are themselves an
  unverifiable surrogate (its accuracy vs the real extractor is unknown), so it
  cannot serve as ground truth.

SynthID-Image is also content-conditional (arXiv 2510.09263): the encoder maps
`f(image, payload) -> image`, per-image, detected by a trained extractor via
conformal p-value. It is not a fixed carrier a frequency-domain subtractor can
isolate.

## Conditions to revisit

Detection stays out of wmr until at least one of these is true. Until then, regen is
the only SynthID operation wmr performs.

1. **Google publishes a public SynthID-Image verifier or feature extractor.** That
   would give a ground-truth label source and make detection measurable.
2. **A controlled same-generator watermarked-vs-unwatermarked dataset becomes
   available** (a generator that can be run with the watermark on and off, paired on
   identical prompts). That would give the negative class needed to train a
   detector.
3. **A peer-reviewed third-party detector appears with real held-out validation**
   (not surrogate labels, not the prover model). That would be worth re-implementing.

A `--synthid-attack regen` measurement improvement (lower strength, GPU/ANE placement
for speed, a public verifier to measure against) is separate from detection and can
proceed independently; the regen path is kept.

## Pointer map (detailed docs)

- `synthid-detection-validation.md` - the detector scored ROC AUC 0.20 on
  Google-verifier-labeled images.
- `synthid-detection-feasibility.md` - the 14-source deep study of third-party
  SynthID detection.
- `synthid-clean-codebook-eval.md` - a clean codebook is inert on content
  (+0.16 to +0.38% attenuation over baseline).
- `synthid-codebook-not-viable.md` - the "dots" artifact root cause + the
  content-inertness verdict.
- `codebook-free-removal-research.md` - the noise-residual path research.
- `synthid-codebook-vs-codebook-free-gate.md` - the two subtractors compared.
- `synthid-lab-a-experiment.md` - the LAB `a`-channel experiment (3.74x worse).
- `synthid-48-96-leakage-check.md` - the carrier-bin leakage investigation.
- `synthid-content-fixture-analysis.md` - the content-fixture probe results.
- `synthid-carrier-characterization.md` - carrier amplitude measurement
  (~0.025/255, sub-LSB).
- `synthid-external-codebook-test.md` - reverse-SynthID's expert codebook through our
  subtractor (also inert on content).
- `synthid-regen-validation.md` - the regen path validation (the kept path).
- `synthid-investigation-summary.md` - the capstone summary (updated with a 1.16.0
  note pointing here).
