# wmr: Watermark Remover

[![GitHub release](https://img.shields.io/github/v/release/froggeric/gemini-watermark-and-synthid-remover?label=release)](https://github.com/froggeric/gemini-watermark-and-synthid-remover/releases)
[![CI](https://img.shields.io/github/actions/workflow/status/froggeric/gemini-watermark-and-synthid-remover/release.yml?branch=main&label=CI)](https://github.com/froggeric/gemini-watermark-and-synthid-remover/actions/workflows/release.yml)
[![macOS](https://img.shields.io/badge/macOS-signed%20%26%20notarized-brightgreen)](https://github.com/froggeric/gemini-watermark-and-synthid-remover/releases)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/std/the-standard)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`wmr` is a command-line tool that removes **visible** watermarks from images and videos made by Google **Gemini**, **Veo**, and **NotebookLM**, and can scrub the invisible **SynthID** watermark through lossy SDXL regeneration. Every release package is a single self-contained binary that runs on a clean system with nothing to install. macOS builds are Developer ID signed and Apple-notarized, so they pass Gatekeeper with no extra steps.

## Table of contents

- [What it does](#what-it-does)
- [Quick start](#quick-start)
- [Usage reference](#usage-reference)
  - [Common flags](#common-flags)
  - [Still-image geometry](#still-image-geometry)
  - [SynthID (invisible watermark)](#synthid-invisible-watermark)
  - [Video](#video)
  - [Residual cleanup (optional)](#residual-cleanup-optional)
  - [cache subcommand](#cache-subcommand)
- [How it works (for researchers)](#how-it-works-for-researchers)
  - [Visible-mark pipeline](#visible-mark-pipeline)
  - [SynthID reality](#synthid-reality)
  - [Detail-restoration](#detail-restoration)
  - [Progress UX](#progress-ux)
  - [CoreML cache management](#coreml-cache-management)
- [Building from source](#building-from-source)
- [Platform support](#platform-support)
- [Privacy](#privacy)
- [License](#license)
- [Credits](#credits)

## What it does

| Watermark | Found on | How wmr handles it | Status |
|-----------|----------|--------------------|--------|
| Gemini diamond logo | Gemini images | Exact reverse alpha-blend (mathematically inverts the overlay) | Exact removal |
| Veo video watermark | Veo videos | Per-frame reverse alpha-blend + edge cleanup | Exact removal |
| NotebookLM logo + wordmark | NotebookLM videos | Per-scene AI inpaint (MI-GAN on the Neural Engine or ONNX Runtime CPU) | Inpaint |
| SynthID (invisible) | Gemini images | Lossy SDXL img2img regeneration | Lossy; validated against Google's official manual verifier (see below) |

`detect` locates the **visible** watermark without modifying the file. wmr does **not** detect SynthID: Google's "Verify with SynthID" is a manual in-app tool with no API wmr can drive, and the spectral detector shipped before 1.16.0 had no discriminative power (it scored ROC AUC 0.20 on images labeled by that Google verifier, so it was removed). See [`docs/research/synthid-spectral-removal-record.md`](docs/research/synthid-spectral-removal-record.md) for the full evidence.

The SynthID removal is **lossy** and cannot be confirmed in-process (there is no verifier API). The default strength was validated against Google's manual verifier on a small, varied set including a double-watermarked image; see [SynthID reality](#synthid-reality) and the linked research docs for the honest scope.

## Quick start

### 1. Download

Grab a prebuilt binary from the [Releases page](https://github.com/froggeric/gemini-watermark-and-synthid-remover/releases). Every package is self-contained (it bundles the AI models and any runtime libraries it needs).

| Asset | Platform | Run |
|-------|----------|-----|
| `wmr-macos-arm64.zip` | macOS 14+ (Apple Silicon) | `unzip wmr-macos-arm64.zip && cd wmr-macos-arm64 && ./wmr` |
| `wmr-macos-x86_64.zip` | macOS 14+ (Intel) | `unzip wmr-macos-x86_64.zip && cd wmr-macos-x86_64 && ./wmr` |
| `wmr-linux-x86_64.tar.gz` | Linux | `tar xzf wmr-linux-x86_64.tar.gz && cd wmr-linux-x86_64 && ./wmr` |
| `wmr-windows-x86_64.zip` | Windows | extract, then run `wmr.exe` |

- macOS ships native CoreML MI-GAN for NotebookLM (Neural Engine, about 28 ms/frame) and, on Apple Silicon, a native CoreML SDXL pipeline for SynthID regen (much faster than CPU).
- Linux and Windows ship ONNX Runtime MI-GAN (about 225 ms/frame, CPU) and run SynthID regen on the CPU (correct, but slow).
- macOS builds are Developer ID signed and notarized, so Gatekeeper allows them on first launch (a one-time online check). If Gatekeeper still blocks a build (for example on an offline machine), run `xattr -dr com.apple.quarantine <extracted-dir>`.
- Third-party licenses: [`LICENSE-THIRD-PARTY.md`](LICENSE-THIRD-PARTY.md).

### 2. Remove your watermark

```bash
# Gemini image (auto-detects + removes the sparkle logo)
wmr remove image.png -o clean.png

# Gemini / Veo video (auto-detects the watermark position + size)
wmr video video.mp4 -o clean.mp4

# Veo legacy text watermark
wmr video veo.mp4 --legacy -o clean.mp4

# NotebookLM video (auto-detects the logo + wordmark)
wmr video notebooklm.mp4 --notebooklm -o clean.mp4

# SynthID invisible watermark (lossy SDXL regen)
wmr synthid image.png -o clean.png
```

Batch a folder: `wmr remove folder/ -o cleaned/ --recursive`.

> **Which command?** Image with a visible mark -> `remove`. Gemini/Veo video -> `video`. NotebookLM video -> `video --notebooklm`. Invisible (SynthID) -> `synthid`. Just locate -> `detect`.

> **Gemini 3.6 Flash images:** the visible diamond is auto-detected, so `wmr remove` just works. If a mark is too faint to confirm (it blends with a light background), force its position: `wmr remove img.png --geo-preset gemini36-portrait -o clean.png` (named presets per resolution) or `wmr remove img.png --rect x,y,w,h -o clean.png` (exact box). Use `wmr detect img.png -v` to read the detected geometry.

Supported inputs: PNG, JPEG, WebP images; MP4 and other FFmpeg-supported video.

## Usage reference

| Command | Does |
|---------|------|
| `remove` (default) | Auto-detect + remove visible watermarks. Optional `--synthid-attack regen` also scrubs SynthID. |
| `synthid` | Scrub only the SynthID invisible watermark (lossy SDXL regen runs by default). |
| `video` | Remove watermarks from video (Gemini, Veo, NotebookLM). |
| `detect` | Detect the visible watermark without modifying. |
| `cache` | Manage wmr's local caches (today: `--clear-coreml` on macOS). |

Run `wmr <subcommand> --help` for the full flag list.

### Common flags

| Flag | Applies to | Description |
|------|------------|-------------|
| `-o, --output` | most | Output path (required for single files; batch defaults to `cleaned/`) |
| `-f, --force` | remove, synthid, video | Skip detection, assume a watermark is present |
| `--no-progress` | all | Suppress progress output (errors + final summary only) |
| `--no-update-check` | all | Skip the update check (also `WMR_NO_UPDATE_CHECK=1`) |
| `-v, --verbose` / `-V, --version` | all | Debug verbosity / print version |
| `-r, --recursive` | remove | Process directories recursively |

### Still-image geometry

wmr auto-detects the visible mark's corner and size, so most images need no flags. Override the position when a mark is too faint to confirm.

| Flag | Applies to | Description |
|------|------------|-------------|
| `--rect x,y,w,h` | remove, detect | Force the watermark box; removal runs there even on a faint mark |
| `--geo-preset <name>` | remove, detect | Named geometry, e.g. `gemini36-portrait` (896x1200) |
| `--no-auto-geometry` | remove, detect | Skip the content-based position search; use the model position |
| `--force-small` / `--force-large` | remove | Force 48x48 / 96x96 Gemini logo size |
| `--legacy` | remove, detect | Pin the legacy Gemini (pre-3.5) V1 profile |
| `--no-legacy` | remove, detect | Pin the current (Gemini 3.5+) V2 profile; disable auto fallback |

The watermark's size depends on the output resolution: 48x48 for small images (short side up to 1024; Gemini 3.6, replacing the 36x36 mark Gemini 3.5 used) and 96x96 for larger ones, in the bottom-right corner. Gemini 3.5 and 3.6 are both supported through multi-template detection.

### SynthID (invisible watermark)

wmr does **not** detect SynthID (see [SynthID reality](#synthid-reality)). The only SynthID operation is `--synthid-attack regen`: a lossy SDXL img2img scrub, opt-in on `remove` and the default on `synthid`.

```bash
wmr synthid image.png -o clean.png                            # regen runs by default
wmr remove image.png --synthid-attack regen -o clean.png      # visible diamond first, then regen
```

| Flag | Description |
|------|-------------|
| `--synthid-attack regen` | SynthID attack method (default and only value: `regen`). On `remove`, regen runs only when this is passed. |
| `--regen-strength` | img2img strength, 0.02 to 0.15 (default 0.10, the validated minimum) |
| `--regen-steps` | sampler steps (default 50) |
| `--regen-restore-detail` | force detail-restoration on (you accept the risk on dim content) |
| `--no-regen-restore-detail` | force full regen (guaranteed removal) |
| `--regen-backend {auto\|cpu\|metal\|vulkan\|coreml}` | runtime backend (default `auto`) |
| `--regen-no-download` | refuse the first-run model fetch (errors if the model is absent) |
| `--regen-no-tile` | disable tiled img2img (whole-image; fails above about 1024px on most GPUs) |
| `--regen-model-path` / `--regen-vae-path` | point at a local model / VAE instead of the pinned download |

**What you should expect.** At the strength that reliably scrubs SynthID (0.10) the output is visibly smoothed and simplified. Measured fidelity is about 29 to 41 dB PSNR versus the input across a varied test set. Lowering the strength is **not recommended**: lighter strengths (0.04 to 0.08) cleared singly-watermarked images in testing but missed a double-watermarked image, so 0.10 is the default. Regen also leaves a forensic footprint (a regen output is itself a detectable diffusion output); this is an attack on the watermark, not an invisibility guarantee.

**Model download.** SynthID regen downloads about 6.5 GB on first use (the SDXL base checkpoint and the fp16-fix VAE, SHA256-pinned, cached in `~/.cache/wmr/`). The CoreML backend (macOS Apple Silicon, about 4.5 GB) auto-downloads from `huggingface.co/froggeric/wmr`. Pass `--regen-no-download` to refuse the fetch.

**Backend selection.** All backends use the same SDXL base model and the same deterministic-Euler img2img schedule, so the validated strength (0.10 @ 50 steps = 5 actual denoise steps) applies uniformly.

| Backend | Platform | Performance | Notes |
|---------|----------|-------------|-------|
| `auto` (default) | all | macOS Apple Silicon prefers CoreML when models are present; else CPU | Linux, Windows, and macOS Intel fall back to CPU. |
| `coreml` | macOS Apple Silicon | about 20s per 1024-tile on M4, plus a one-time ~50s load | Native CoreML SDXL pipeline; much faster than CPU. |
| `cpu` | all | about 231s per 896x1200 tile | sdcpp via stable-diffusion.cpp. The default on Linux, Windows, and macOS Intel release binaries. |
| `metal` | macOS Apple Silicon (arm64 binary only) | not recommended | sdcpp Metal backend is unstable upstream on Apple Silicon; prefer `auto` or `coreml`. |

Set `$WMR_COREML_SD_MODELS_DIR` to place the CoreML models elsewhere (defaults to `~/.cache/wmr/coreml-sdxl/`). Set `$WMR_COREML_SD_COMPUTE_UNITS` (`all` default, `cpu_gpu`, `cpu_ane`, `cpu`) to override the CoreML compute units.

On macOS Apple Silicon the CoreML UNet uses `ORIGINAL` attention and is GPU-bound. The Neural Engine is unused (SDXL's large fp16 attention matmuls are ANE-ineligible, so CoreML places the UNet on the GPU). On upgrade, wmr re-verifies each cached model against its pinned SHA and re-downloads only what changed, removing the old copy first, so the cache does not grow.

### Video

| Flag | Description |
|------|-------------|
| `--notebooklm` | Target the NotebookLM logo + wordmark |
| `--rect x,y,w,h` | Manual watermark rect (overrides auto-detect; Gemini, Veo, and NotebookLM) |
| `--notebooklm-method {auto\|ns\|migan}` | Inpaint method override (`auto` = platform default: MI-GAN everywhere on Apple Silicon, complexity-gated elsewhere) |
| `--complexity-threshold` | NS vs MI-GAN gate (default 15; consulted only on non-arm64 `auto`) |
| `--variant` | Force geometry: `720p-1`, `720p-2`, `1080p` (otherwise auto-detected) |
| `--no-auto-geometry` | Skip the content-based geometry search; fall back to the resolution guess |
| `--no-edge-cleanup` | Pure reverse-blend; skip the Gemini diamond edge cleanup (cleanup is on by default) |
| `--legacy` | Use the Veo legacy text profile |
| `--scenes` | Split multi-scene videos into separate files |
| `--scene-threshold` | Scene-cut sensitivity 0.0 to 1.0 (default 0.3) |
| `--crf` / `--preset` / `--codec` | Encode settings (default CRF 14, `slow`, `libx264`) |
| `--inpaint-strength` | Inpaint strength 0.0 to 1.0 (default 0.85) |

**Measuring a NotebookLM `--rect`:** if auto-detection misses, grab a full-resolution frame and measure the mark's `x,y,width,height` in any image editor:

```bash
ffmpeg -ss 30 -i input.mp4 -frames:v 1 frame.png   # then measure the mark in frame.png
```

Leave about a 1px border around the mark, and pick a frame where it is clearly visible.

### Residual cleanup (optional)

The default visible removal is a pure reverse-alpha-blend (the exact mathematical inversion, no blur). Optionally clean up residual artifacts from an imperfect reversal with `--denoise`. The opt-in cleanup is **residual-only**: it only touches pixels where the reverse-blend left a real residual, so it never blurs a clean removal.

```bash
wmr remove in.png -o out.png                   # default: exact reverse-blend, no cleanup
wmr remove in.png --denoise ns -o out.png      # residual-only Navier-Stokes
wmr remove in.png --denoise telea -o out.png   # residual-only Telea
wmr remove in.png --denoise ai -o out.png      # AI (FDnCNN, release builds only)
wmr remove in.png --strength 150 -o out.png    # cleanup strength (0-300%)
```

| Flag | Range | Default | Notes |
|------|-------|---------|-------|
| `--denoise` | `off\|soft\|ns\|telea` (release adds `ai`) | `off` | Residual cleanup method (`off` = exact reverse-blend) |
| `--sigma` | 1 to 150 | 50 | FDnCNN noise level (AI only) |
| `--strength` | 0 to 300 % | 120 | Cleanup strength |
| `--radius` | 1 to 25 | 10 | Gaussian / TELEA / NS radius |

### cache subcommand

`wmr cache` manages wmr's local caches. Today it exposes one operation (macOS only):

```bash
wmr cache --clear-coreml   # clear the CoreML execution cache; CoreML recompiles on the next regen
```

`--clear-coreml` removes wmr's app-scoped compiled-Metal cache (`~/Library/Caches/wmr/com.apple.e5rt.e5bundlecache/`). It never touches the shared `~/Library/Caches/CoreML` or the model `.mlpackage` files. It is a no-op on Linux and Windows. See [CoreML cache management](#coreml-cache-management).

## How it works (for researchers)

This section is for technical readers who want the mechanism and the evidence. The detailed findings live under [`docs/research/`](docs/research/); the links below are the entry points.

### Visible-mark pipeline

Visible Gemini and Veo watermarks are alpha-blended overlays: `watermarked = a * logo + (1 - a) * original`. With the logo and its alpha map known, removal inverts the blend exactly: `original = (watermarked - a * logo) / (1 - a)`. This reverse-blend is the default and is mathematically exact. The optional `--denoise` cleanup is residual-only (it predicts clean content with `cv::inpaint` and blends it in only where the reversal left a real residual, never blurring a clean removal).

The pipeline has three stages, orchestrated by `WatermarkEngine`:

1. **Detect.** A three-stage NCC detector (spatial template match, gradient match on Sobel magnitudes, variance analysis) fused as `spatial*0.50 + gradient*0.30 + variance*0.20`, threshold 0.35.
2. **Geometry auto-detection.** A polarity-invariant NCC template match against native per-size alpha captures, anchored on the predicted position then widened to the corner. Still images run the search once at the CLI layer (anchored first, widened only if the hit is not trusted); video aggregates about 12 sampled frames so the static mark wins over transient content. Templates are captured per size and never resized, because resizing smears the anti-aliased edges the exact reversal depends on.
3. **Remove.** The reverse-blend per pixel (still) or per frame (video). Video adds shot-level detection, an occlusion gate, and a residual-gated edge cleanup that repairs the faint halo at the diamond's edge on compressed footage (a safe no-op when the mark sits on a uniform background). Audio is passed through untouched.

The 48px Gemini mark (Gemini 3.6) and the 36px mark (Gemini 3.5) are both supported through multi-template detection. The 48px alpha is the average of 10 distinct clean captures, which suppresses per-pixel capture noise. Video removes with the same clean per-size alpha used for stills.

**NotebookLM** marks are semi-transparent and color-adaptive (not a reversible alpha overlay; the alpha is approximately zero, so there is no clean inverse). They are removed by AI inpainting. [MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN) (MIT, ICCV 2023) synthesizes the missing region; on Apple Silicon it runs on the Neural Engine (about 28 ms/frame), elsewhere on ONNX Runtime CPU (about 225 ms/frame), falling back to Navier-Stokes. Every scene is inpainted, with the method chosen per scene: MI-GAN everywhere on Apple Silicon; elsewhere a complexity gate picks MI-GAN for textured backgrounds and NS for uniform ones.

### SynthID reality

SynthID-Image (arXiv [2510.09263](https://arxiv.org/abs/2510.09263)) is a **content-conditional neural watermark** Google attaches to Gemini image output ([Google's SynthID page](https://deepmind.google/models/synthid/)). The encoder maps `f(image, payload) -> image` per image, detected by a trained extractor via a conformal p-value. It is not a fixed carrier a frequency-domain subtractor can isolate.

**wmr does not detect SynthID**, for two reasons that are documented in the research record:

1. **There is no public verifier API.** Google's "Verify with SynthID" is a manual in-app tool, account-gated and rate-limited (about 10 checks per day), emitting natural-language verdicts. There is no batchable ground-truth label source, and Google's only open SynthID code is text-only. Without labeled data there is no way to train or validate a detector. See [`synthid-detection-feasibility.md`](docs/research/synthid-detection-feasibility.md).
2. **The detector we shipped before 1.16.0 had no discriminative power.** Scored for the first time against 8 images labeled by Google's official verifier, it scored ROC AUC 0.20 (worse than random; its score is a content-property constant at about 61%). See [`synthid-detection-validation.md`](docs/research/synthid-detection-validation.md).

The spectral suppression path (codebook subtraction, noise-residual subtraction, an LAB `a`-channel attack, carrier-bin seeding) was tried over the whole SynthID effort and **removed in 1.16.0** because it did not work: the carrier sits below the content noise floor on real images (10 to 30x weaker than the content residual), so a clean codebook is inert on content (0.16 to 0.38% attenuation over the phase-noise baseline). The canonical decision record, including every method tried and the conditions under which detection could be revisited, is [`synthid-spectral-removal-record.md`](docs/research/synthid-spectral-removal-record.md). The carrier characterization is in [`synthid-carrier-characterization.md`](docs/research/synthid-carrier-characterization.md).

**The only validated removal is low-strength SDXL img2img regeneration.** `--synthid-attack regen` regenerates the whole image with the SDXL base model at a low starting noise. This is the single attack the published literature reports as validated against SynthID-Image, and the one wmr validated. The default strength 0.10 at 50 steps (5 actual denoise steps) cleared a varied 9-image set, including an image-to-image generation that carries SynthID twice, confirmed over two rounds of Google's manual verifier. This is a small sample (the verifier is rate-limited to about 10 checks/day), not a large-scale proof, but it spans posters, mockups, and AI art and includes the hardest known case. Lighter strengths (0.04 to 0.08) cleared singly-watermarked images but missed the double-watermark case, so 0.10 is the default. See [`synthid-regen-validation.md`](docs/research/synthid-regen-validation.md) and [`synthid-light-reconstruction-attacks.md`](docs/research/synthid-light-reconstruction-attacks.md).

Because Google exposes no verifier API, success cannot be confirmed in-process. The removal is verifiable, but only by a human using the manual in-app tool.

### Detail-restoration

Regen is lossy. An optional **detail-restoration** pass recovers much of that fidelity on the content where it is safe. After regen produces `R`, the top-5%-magnitude pixels of the diff `O - R` are restored onto `R`, after a characterized-carrier Wiener filter suppresses the SynthID component in the low-mid band. On normal-content images this sharpens faces and foliage (about +7 dB PSNR on the test image) and still clears Google's verifier.

Two findings from [`synthid-diff-restoration-analysis.md`](docs/research/synthid-diff-restoration-analysis.md) drive the design:

1. **A dilution gate works on normal content.** Restoring the top 5% of the diff buys fidelity and still clears the verifier; restoring 10% or more is detected. The gate works by *dilution* (few enough watermark-bearing pixels that the total restored watermark energy stays under the detector's aggregate threshold), not by separating the watermark from the detail.
2. **Mean luminance predicts whether restoration stays clear.** A 10-image clearance study found brightness cleanly separates clear from fail (the 4 dimmest images fail; the 6 brightest clear; threshold about 128, 10/10, monotone). The SynthID carrier is an additive perturbation whose signal-to-content ratio scales as roughly 1/luminance, so it stays detectable on dim images once the carrier-bearing pixels are restored.

The shipped behavior is **automatic and luminance-gated**: bright images (mean luminance >= 128) get the restoration; dim images (< 128) get full regen (no restoration). Override with `--regen-restore-detail` (force restoration on; you accept the risk) or `--no-regen-restore-detail` (force full regen, guaranteed removal).

It is **best-effort, not a guarantee**: there is no in-process verifier, and the luminance gate is empirical (a 10-image sample). Detail-restoration *dilutes* the watermark on the restored pixels (below the detector's aggregate threshold); it does not remove it. Use `--no-regen-restore-detail` when you need guaranteed removal.

### Progress UX

Long operations (regen, video, batch) print perception-grounded progress: a stage frame (`[1/4]` ... `[4/4]`) so you always know what is happening and what is next; per-tile and per-frame progress with count, rate, backend, and an honest ETA (a rolling average, hidden until enough samples so it never gives a misleading early guess); one-time costs (the first-run download, the first CoreML compile) labeled as one-time so they do not read as stalls; and elapsed time, tile count, and backend on completion. Output is TTY-aware: a refreshing progress bar on an interactive terminal; clean append-only milestone lines when piped, so CI logs stay linear. Progress writes to stderr, so `wmr ... > /dev/null` still shows it and stdout stays clean. `--no-progress` suppresses it. See [`cli-progress-ux-design.md`](docs/research/cli-progress-ux-design.md) for the design principles and the per-operation spec.

### CoreML cache management

On macOS, CoreML keeps an app-scoped compiled-Metal execution cache (`~/Library/Caches/wmr/com.apple.e5rt.e5bundlecache/`) with no eviction. Across model re-pins, wmr upgrades, and macOS upgrades it was observed growing to about 138 GB, and stale entries produced noisy Apple-framework warnings. Since 1.16.7 wmr auto-manages this cache: at CoreML init it clears the cache when stale (the wmr version, the model pin, or the macOS version changed since the last clear, tracked through a sidecar) or when it exceeds a size threshold, then lets CoreML recompile (a one-time ~30s). It only touches the app-scoped cache; the shared `~/Library/Caches/CoreML` and the model `.mlpackage` files are never touched. Linux and Windows are unaffected. `wmr cache --clear-coreml` clears it on demand. See [`coreml-execution-cache-management.md`](docs/research/coreml-execution-cache-management.md).

The CoreML SD pipeline is a from-scratch Objective-C++ port (an Euler scheduler over Accelerate, VAE encode and decode, baked empty-prompt embeddings; text encoders are not shipped). The GPU/ANE placement investigation, including why the Neural Engine is unused for SDXL and why 6-bit weight palettization was tested and rejected (it is slower, not faster, on the GPU), is in [`coreml-sd-placement.md`](docs/research/coreml-sd-placement.md).

## Building from source

Requires CMake 3.21+, a C++20 compiler, and Ninja.

**macOS (Homebrew), recommended:** `scripts/build.sh` verifies the required formulas and self-heals a stale cache:

```bash
brew install cmake ninja opencv ffmpeg fmt spdlog cli11 catch2
scripts/build.sh                 # release + tests
RUN_TESTS=0 scripts/build.sh     # build only
```

**vcpkg (all platforms):**

```bash
cmake -B build -S . -GNinja
cmake --build build
```

To match the release binaries (AI denoise, MI-GAN, SynthID regen, CoreML SD), set `WMR_AI_MIGAN=1 WMR_AI_DENOISE=1 scripts/build.sh` on macOS (also install `vulkan-volk`, `vulkan-loader`, `vulkan-headers`, `molten-vk`). Tests: `ctest --test-dir build --output-on-failure`.

See [`CLAUDE.md`](CLAUDE.md) for the full build matrix, platform quirks, and the design notes behind each watermark path, and [`CHANGELOG.md`](CHANGELOG.md) for version history.

## Platform support

| Platform | Visible (image + video) | NotebookLM | SynthID regen | Update check |
|----------|-------------------------|------------|---------------|--------------|
| macOS 14+ Apple Silicon | exact reverse-blend | CoreML MI-GAN (Neural Engine) | CoreML SDXL (fast) | yes |
| macOS 14+ Intel | exact reverse-blend | CoreML MI-GAN | CPU (sdcpp, slow) | yes |
| Linux x86_64 | exact reverse-blend | ONNX Runtime MI-GAN (CPU) | CPU (sdcpp, slow) | yes |
| Windows x86_64 | exact reverse-blend | ONNX Runtime MI-GAN (CPU) | CPU (sdcpp, slow) | yes |

All packages ship the visible-removal alpha maps and the MI-GAN model. The SynthID regen model (about 6.5 GB CPU, or 4.5 GB CoreML on macOS) is downloaded on first use, not bundled. macOS packages are Developer ID signed and notarized.

### Help wanted: faster SynthID regen on Linux and Windows

`--synthid-attack regen` runs everywhere, but off Apple Silicon it is CPU-only and slow (minutes per image). The Linux and Windows release binaries ship CPU-only because the sdcpp ([leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) + ggml) GPU backends have build gaps we could not CI-validate (Vulkan needs `glslc` and the loader on the Linux runner; sdcpp on MSVC was never green in CI). If you work with Vulkan or CUDA on ggml / stable-diffusion.cpp and can help wire and validate a GPU regen backend for Linux and Windows, that is the biggest speedup left. Build locally with `WMR_BUILD_REGEN=ON` (no `WMR_REGEN_CPU_ONLY`), point `--regen-backend vulkan` or `cuda` at a GPU, and report the timings and whether the output still clears Google's SynthID verifier. Open an issue to coordinate.

Other welcome contributions: new watermark profiles (still or video geometry and alpha captures), higher-fidelity or faster inpaint backends, and video SynthID.

## Privacy

For a visible-only user, wmr makes zero network calls. The one network feature is the **update check** (notify-only, on by default since 1.16.4): once per day it fetches the latest release tag and prints a short notice to stderr after the command finishes. It never downloads or replaces the binary.

- **Zero payload.** The check is a single HTTPS GET of the latest release tag with a versionless `User-Agent: wmr`. No version, OS, arch, or id is sent.
- **Opt out:** `--no-update-check`, `WMR_NO_UPDATE_CHECK=1`, `CI`, or `DO_NOT_TRACK=1`; it also auto-disables when stderr is not a terminal.
- **Throttle:** at most once per 24h; override with `WMR_UPDATE_CHECK_INTERVAL=<seconds>`. Cache at `~/.cache/wmr/update-check.json`.
- Build without it: `cmake -DWMR_UPDATE_CHECK=OFF ...`.

The only other network call is the SynthID regen model download, which runs only when you use `--synthid-attack regen` (and you can refuse it with `--regen-no-download`).

## License

MIT. Bundled third-party licenses (NCNN, volk, KAIR/FDnCNN, MI-GAN, ONNX Runtime, OpenCV) are in [`LICENSE-THIRD-PARTY.md`](LICENSE-THIRD-PARTY.md).

## Credits

Built on research and code from:

- [GeminiWatermarkTool](https://github.com/allenk/GeminiWatermarkTool) and [VeoWatermarkRemover](https://github.com/allenk/VeoWatermarkRemover): reverse alpha-blend, NCC detection, inpainting, FDnCNN conversion (Allen Kuo)
- [reverse-SynthID](https://github.com/aloshdenny/reverse-SynthID): SynthID spectral analysis
- [Picsart-AI-Research/MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN): MI-GAN inpainting (MIT, ICCV 2023)
- [KAIR/FDnCNN](https://github.com/csjcai/KAIR), [Tencent/ncnn](https://github.com/Tencent/ncnn), [zeux/volk](https://github.com/zeux/volk), [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime): AI inference stack
