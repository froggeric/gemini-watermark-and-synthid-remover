# SynthID investigation: summary + state (2026-08-02)

> **Update 2026-08-05 (v1.16.0): the spectral path was removed.** Following the
> validation below (detector ROC AUC 0.20 on Google-verifier-labeled images; a clean
> codebook inert on content), the spectral SynthID detection + suppression code was
> deleted in v1.16.0. The only SynthID operation wmr ships now is
> `--synthid-attack regen` (lossy SDXL img2img). The decision record, with the
> evidence and the conditions to revisit detection, is at
> `synthid-spectral-removal-record.md`. The rest of this doc is the pre-removal
> investigation record (kept for reference).

Capstone of the SynthID work on branch `synthid-correctness-pack` (merged to `main`, not released; version 1.15.0 in CMakeLists, no tag). This doc ties together the detailed write-ups in `docs/research/synthid-*.md` and records conclusions + next steps. Read this first when resuming SynthID work.

## What SynthID-Image actually is (ground truth)
Gemini image / Veo video carry **SynthID-Image**, a **content-conditional neural watermark** (arXiv 2510.09263): encoder `f(image, payload) -> image`, per-image, detected by a trained extractor via conformal p-value. Not a fixed carrier. The only official open code is `google-deepmind/synthid-text` (text-only). Google's "Verify with SynthID" IS an official but **manual verifier** (in-app, account-gated, ~10 image checks/24h, natural-language output; no API), so removal cannot be auto-confirmed in-process (the default knee WAS validated against it). The invisible carrier amplitude is ~0.025/255 (sub-LSB), below the content noise floor. SynthID is now multi-vendor (Google licensed it; OpenAI GPT-image carries it).

## The visible diamond (separate, exact)
The visible Gemini/Veo diamond is a constant-alpha overlay removed by **exact reverse-alpha-blend** (`original = (watermarked - alpha*255)/(1-alpha)`). This path is correct and best-in-class; untouched by this work. NOTE a gap surfaced: the HF `gemini_black` (Gemini 3.1-class) diamond peaks at alpha ~0.56, while our captured alpha `v2_diamond_48_still` (Gemini 3.6) peaks ~0.30, so `wmr visible` under-removes that variant (a visible-remover coverage gap, not yet fixed).

## Our spectral path (CodebookSubtractor / NoiseResidualSubtractor)
Frequency-domain (FFTW3) carrier suppression. Honest framing shipped (WS1a): "suppress (heuristic; not a verifiable removal)", not "remove" (locked by a source-scan test). Per-channel BGR weights {0.85, 1.0, 0.70} (B>R verified empirically; G-prominence within noise). Default path is `--codebook-free` (NoiseResidualSubtractor).

## What shipped on this branch (correctness wins)
1. **WS1a** — honesty reframe + auto-discovering wording-lock test.
2. **WS2a** — fixed `CodebookBuilder::finalize` circular-phase bug (arithmetic mean -> cos/sin atan2); added `phase_consistency_bgr` plane; `.wcb` -> `WMRCB02` (v1 backward-compat); subtractor soft-gated on phase-consistency.
3. **WS1b** — R/B channel-weight verification probe + synthetic WS2a gate regression probe.
4. **WS2b** — carrier-bin seeding helpers (`seed_carrier_bins` dual-plane, `merge_from` phase-preserving, `find_exact_profile`) + `--carrier-grid`; (48,96) leakage gate INCONCLUSIVE -> helpers ship inert.
5. **Phase 0.5** — fixed `consistency_bgr` DC-saturation (excluded DC neighborhood from max_std; the magnitude gate was a near-no-op before).
6. **WS3** — LAB-`a` channel experiment (`--lab-a`); 3.74x worse than BGR (carrier is luminance-based); ships inert + documented.
7. **Builder fix (the dots bug)** — replaced the saturated `1 - std/max_std` carrier gate with reverse-SynthID's discriminative `normalize(log1p(mean_magnitude) * phase_coherence)`. Active-bin fraction on near-identical captures dropped 100% -> 1.1% (excludes the visible diamond's broadband FFT). Fixes the "dots" artifact.
8. **OOM guard** — degenerate (all-zero/NaN) codebook profiles no longer crash the subtractor (multi-exabyte alloc); skipped with a warning.

## The "dots" artifact: root cause + fix (the main investigation)
Symptom: `wmr synthid --codebook <cb>` left dozens of tiny fixed-position luminous dots, scattered uniformly, strength-scaled. Three wrong theories first (diamond blob, phase-noise, content). Actual root cause (confirmed visually via image analysis + isolation): **our codebook BUILDER, fed near-identical pure-black captures, saturated the magnitude-variability consistency gate (~0.997 everywhere) and captured the VISIBLE diamond's broadband FFT as if it were carrier (active at 100% of bins). Subtracting that codebook imprinted the diamond's spectrum as a scattered dot field.** NOT a subtractor bug: a clean expert codebook (reverse-SynthID's) run through our unchanged subtractor produces no dots. Fixed by the discriminative carrier-selection metric (#7). Detailed in `synthid-codebook-not-viable.md`, `synthid-external-codebook-test.md`, `synthid-clean-codebook-eval.md`.

## Is a fixed spectral codebook worth shipping? (No, as a feature)
Settled by measuring on **content** images (the real use case; uniform is artificial because `--codebook-free` "wins" there only by regenerating a flat field):
- Even a **clean** codebook (fixed builder, or reverse-SynthID's external one through our subtractor) is **inert on content**: carrier-band attenuation +0.16 to +0.38% over the phase-noise baseline (within noise); PSNR 33-45 dB (no damage, but no suppression either). The carrier (~0.025/255) is not resolvable above the content noise floor.
- reverse-SynthID's expert codebook does **not** outperform ours through our subtractor (both inert on content). Their own pure-spectral rounds (01-05) failed to clear Google's detector; only their non-spectral Round-06 (SD-VAE + warp + JPEG = diffusion regeneration) worked.
- Convergence: our evidence + reverse-SynthID's evidence agree — a fixed spectral codebook cannot defeat a content-conditional carrier below the noise floor.

So: the **builder fix + OOM guard ship** (real correctness). The **codebook is NOT shipped as a default feature** (inert on content); the build infrastructure (`scripts/build_synthid_codebook.sh`) is committed and reproducible for anyone who wants to build one, and a `.wcb` could be hosted at a future release. Default SynthID path stays `--codebook-free`.

## Data sources used
- HF dataset `aoxo/reverse-synthid`: `gemini_black` (101 pure-black 1024x1024), `gemini_white` (253), `gemini_random` (88 content 2816x1536), + Veo/Sora video dirs, + dall-e-3 (OpenAI, also SynthID via licensing). Gemini image dirs are the SynthID-bearing still-image source.
- reverse-SynthID's prebuilt codebook `spectral_codebook_v4.npz` (per-model x 7 resolutions x {mag,phase,cons,cw,scales}; mag log2-encoded, phase int8 x pi/127, half-spectrum rfft2; cons = cross-color phase coherence).

## State at merge (2026-08-02)
- Branch `synthid-correctness-pack` merged to `main` (20 commits). Not tagged; version 1.15.0 in CMakeLists (the `[Unreleased]` CHANGELOG block holds the builder fix + OOM; `[1.15.0]` holds Phase 0).
- 58+3 = 61 tests green. Default SynthID path unchanged for users (`--codebook-free`).
- Untracked local fixtures (reference-images/, reference-videos/, test-images/poster-artnight.png) NOT committed.

## Next steps (priority order)
1. **Phase 2: native C++ diffusion-regen mode** (`--synthid-attack regen`) via `leejet/stable-diffusion.cpp` (Metal on mac, CUDA/Vulkan on linux/windows, CPU fallback), day-1 tiled 4K, behind `WMR_BUILD_REGEN`, model downloaded on first run to `~/.cache/wmr/` (SHA256-pinned). This is the ONLY validated SynthID-removal technique (= reverse-SynthID's Round 06). The spec is in `~/.claude/plans/what-is-the-git-lovely-narwhal.md` (WS4). Hard constraints: no Python at runtime, HW acceleration mandatory.
2. **Release 1.15.0** (tag + push tag -> fires release.yml) when ready: ships the honesty reframe + all the codebook correctness fixes (phase, consistency-DC, discriminative carrier selection, OOM guard). The builder fix + OOM are in `[Unreleased]`; promote to `[1.15.0]` (or `[1.16.0]` if more lands) at cut time.
3. **(Optional) wmr-visible alpha gap**: capture the correct alpha for the Gemini 3.1-class diamond (peaks ~0.56) so `wmr visible` can clean those HF images (only matters if we want to build codebooks from that dataset; the codebook itself is inert on content, so low priority).
4. **(Optional) codebook as a hosted asset**: if ever wanted, host a `.wcb` built via `scripts/build_synthid_codebook.sh` on the GitHub release + add runtime download/resolution (mirror the regen model-download pattern). Low priority given content-inertness.
