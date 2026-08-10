# Project Roadmap: wmr — Unified Watermark Removal CLI

## Overview

`wmr` is a C++20 CLI tool that removes watermarks from Google-generated content. It merges functionality from two reference projects:

- **GeminiWatermarkTool** (allenk/GeminiWatermarkTool) — visible sparkle-logo watermark removal via reverse alpha blending, NCC detection, inpainting
- **reverse-SynthID** (layerd-filtering/reverse-SynthID) — SynthID invisible watermark removal via V3 spectral codebook subtraction, Bayesian detection

Phase 1 (visible watermark removal) is complete. This spec defines Phases 2–7.

## Current Status (updated 2026-08-10, v1.16.8)

This spec was the original plan. The project shipped past it and diverged on SynthID. The design text below is kept as a historical record; the real outcome per phase:

| Phase | Plan | Actual |
|-------|------|--------|
| 1. Visible removal | reverse alpha-blend | Shipped. |
| 2. Detection + inpaint | 3-stage NCC + cleanup | Shipped. The default cleanup is now OFF (pure reverse-blend, the exact inversion); cleanup is opt-in and residual-only. |
| 3. SynthID spectral removal | FFT codebook subtraction | Built, then **removed in 1.16.0**. It did not work (the detector scored ROC AUC 0.20 against Google-labeled images, and a clean codebook is inert on real content because the carrier sits below the noise floor). Superseded by diffusion regen. Decision record: `docs/research/synthid-spectral-removal-record.md`. |
| 4. SynthID detection + codebook builder | Bayesian 4-method detector | Resolved as **not achievable**. No reliable third-party SynthID detector exists, and Google's verifier is manual-only with no API, so detection cannot be automated. `wmr detect` is visible-only. Records: `docs/research/synthid-detection-feasibility.md`, `synthid-detection-validation.md`. |
| 5. Unified CLI + batch | subcommands + directory batch | Shipped (`remove`, `synthid`, `detect`, `video`, `cache`). |
| 6. AI denoise | FDnCNN cleanup | Shipped (FDnCNN via NCNN+Vulkan), plus MI-GAN inpainting (CoreML on mac, ORT elsewhere), beyond the plan. |
| 7. Video | frame-by-frame | Shipped, well beyond the plan: Gemini diamond, Veo text, NotebookLM, shot detection and splitting, audio passthrough, per-frame occlusion gate, residual-gated edge cleanup. |

### Beyond the original roadmap (v1.15.0 to v1.16.8)

Major work added after this plan was written:

- **SynthID removal via diffusion regen** (1.15.0 to 1.16.3). Low-strength SDXL img2img is the only SynthID-Image scrub the literature reports as validated. macOS Apple Silicon uses a native CoreML SDXL pipeline (GPU-bound ORIGINAL-attention UNet, fp16-fix VAE, models auto-download from `huggingface.co/froggeric/wmr`); linux, windows, and mac-Intel use the sdcpp CPU backend (cross-platform since 1.16.2). Default strength 0.10 at 50 steps, validated against Google's in-app verifier. Optional detail-restoration (1.16.6) recovers fidelity on bright images.
- **NotebookLM video watermark removal** (1.6 to 1.10.1). Per-scene MI-GAN inpaint (CoreML on Apple Silicon, ORT elsewhere), complexity-gated.
- **Update check** (1.16.4). Notify-only GitHub-release check, zero payload, opt-out.
- **CoreML cache management** (1.16.7). Auto-manages the e5rt execution cache so it cannot balloon.
- **Still-image auto-geometry** (1.12 to 1.16.8). Content-detected position and size for the Gemini 3.6 48px diamond, with a content-suppressed snap (1.16.8) for exact pixel placement.

Open item: the NotebookLM MI-GAN runtime under Rosetta on Intel Mac is unverified in CI (the link is verified and the `.mlpackage` ships; fallback is a one-line disable for x86_64).

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| AI denoise | Traditional first, AI later (Phase 6) | Avoids heavy NCNN+Vulkan dependency until core pipeline is proven |
| Detection mode | Detect then remove | More robust than force-process; user can override with `--force` |
| SynthID codebook | Bundled + user-build | Pre-built codebooks for common resolutions; power users can build custom ones |
| Video support | Images first, video later (Phase 7) | Core image pipeline must be solid before adding frame-level complexity |
| FFT library | FFTW3 | Industry standard, vcpkg-available, GPL-compatible for open-source |
| Test framework | Catch2 | Header-only option, C++20 support, vcpkg-available |

## Architecture

### Module Structure

```
src/
├── main.cpp
├── cli/
│   ├── cli_app.hpp/cpp              # CLI parsing, orchestration
│   └── batch_processor.hpp/cpp      # Directory batch processing (Phase 5)
├── core/
│   ├── types.hpp                    # Shared types, enums, constants
│   ├── blend_modes.hpp/cpp          # Alpha blend math (done)
│   ├── watermark_engine.hpp/cpp     # Visible watermark engine (done)
│   ├── inpaint.hpp/cpp              # Traditional inpainting (Phase 2)
│   └── fft_context.hpp/cpp          # FFT wrapper (Phase 3)
├── detection/
│   ├── ncc_detector.hpp/cpp         # NCC visible watermark detection (Phase 2)
│   └── synthid_detector.hpp/cpp     # Bayesian SynthID detection (Phase 4)
├── synthid/
│   ├── spectral_codebook.hpp/cpp    # Codebook data structures (Phase 3)
│   ├── codebook_subtractor.hpp/cpp  # V3 FFT subtraction (Phase 3)
│   └── codebook_builder.hpp/cpp     # Build codebook from refs (Phase 4)
└── assets/
    ├── embedded_assets.hpp          # Visible watermark PNGs (done)
    └── codebook_assets.hpp          # Bundled SynthID codebooks (Phase 3)
```

### Dependency Graph

```
cli_app → watermark_engine → blend_modes, ncc_detector, inpaint
        → codebook_subtractor → spectral_codebook, fft_context
        → synthid_detector → fft_context, spectral_codebook
        → codebook_builder → fft_context, spectral_codebook
```

Each module exposes a minimal public interface. The CLI layer orchestrates but contains no business logic.

---

## Phase 1: Visible Watermark Removal — DONE

Reverse alpha blending, embedded PNG assets, basic CLI with force-process. Committed as b885d61.

**Delivered:** `wmr input.png -o output.png` removes Gemini visible sparkle watermark from bottom-right corner.

---

## Phase 2: Visible Watermark Detection + Inpainting

### Goal

Make visible watermark removal robust: detect the watermark before removing it, and clean up residuals with traditional inpainting.

### Detection: Three-Stage NCC Pipeline

Ported from GeminiWatermarkTool's `detect_watermark()` and `detect_watermark_guided()`.

**Stage 1 — Spatial NCC:**
- Normalize alpha map template and image patch
- Compute normalized cross-correlation at watermark ROI (bottom-right corner)
- Multi-scale: downsample 4x for coarse search, refine at full resolution
- Threshold: 0.6 for coarse, 0.7 for fine

**Stage 2 — Gradient NCC:**
- Apply Sobel filter to both template and image patch
- NCC on gradient magnitude — robust to brightness/contrast differences
- Weight: 0.3 in final confidence

**Stage 3 — Variance-Guided NCC:**
- Compute local variance in watermark region
- Weight high-variance areas more (they contain more structure information)
- Weight: 0.2 in final confidence

**Guided Multi-Scale Search:**
- For large images (>2048px), start at 1/4 resolution
- Coarse NCC → best candidate region
- Refine at 1/2 resolution, then full resolution
- Reduces search space from O(n²) to O(n) per scale

**Result:** `DetectionResult { bool found; float confidence; cv::Point position; WatermarkSize size; }`

**Fallback:** If detection confidence < 0.7, use default position (Phase 1 behavior) with a warning. `--force` skips detection entirely.

### Inpainting: Traditional Methods

Three methods applied sequentially after watermark removal:

1. **Gaussian Soft Inpaint** — Blur the watermark boundary region (3-5px band) to seamlessly blend the removal area with surrounding pixels. This is the primary cleanup method.

2. **TELEA** (OpenCV `cv::inpaint` with `INPAINT_TELEA`) — Fast marching method. Applied to remaining thin artifacts after Gaussian blend. Good for hairline edges.

3. **Navier-Stokes** (OpenCV `cv::inpaint` with `INPAINT_NS`) — Fluid-dynamics-based. Applied to any remaining textured artifacts. Better for regions with complex texture.

**Application strategy:** Gaussian blend first (always). If residual artifact energy > threshold, apply TELEA. If still > threshold, apply Navier-Stokes.

### Files

| File | Purpose |
|------|---------|
| `src/detection/ncc_detector.hpp` | NccDetector class declaration |
| `src/detection/ncc_detector.cpp` | Three-stage NCC + guided search |
| `src/core/inpaint.hpp` | Inpainting function declarations |
| `src/core/inpaint.cpp` | Gaussian, TELEA, Navier-Stokes |
| `src/core/types.hpp` | Add DetectionResult struct |

### CLI Changes

```
wmr input.png                    # Detect → remove → inpaint (default)
wmr input.png --force            # Skip detection, remove at default position
wmr input.png --detect-only      # Report detection result without modifying
```

### Dependencies

None (all OpenCV built-ins).

### Verification

1. Run against `test-images/2400x1792-gemini.png` — detection should find watermark with confidence > 0.8
2. Compare output quality with Phase 1 output — inpainted result should be smoother at boundary
3. Run against non-watermarked image — detection should report `found=false`, image unchanged

---

## Phase 3: SynthID Removal Foundation — SUPERSEDED (built, removed in 1.16.0; see Current Status)

### Goal

Deliver working SynthID invisible watermark removal. Users can remove SynthID from images using bundled codebooks.

### FFT Infrastructure

Thin wrapper around FFTW3:

```cpp
class FftContext {
public:
    // Forward 2D FFT: CV_32FC1 → CV_32FC2 (complex), one channel at a time
    cv::Mat forward(const cv::Mat& channel);

    // Inverse 2D FFT: CV_32FC2 (complex) → CV_32FC1
    cv::Mat inverse(const cv::Mat& complex);

    // Get magnitude and phase from complex representation
    static cv::Mat magnitude(const cv::Mat& complex);
    static cv::Mat phase(const cv::Mat& complex);
};
```

- Plan caching for repeated same-size transforms
- Input: float32 CV_32FC1 (single channel)
- Output: CV_32FC2 (complex, single channel)
- Callers split/recombine BGR channels externally

### Spectral Codebook

Per-resolution profiles storing the SynthID watermark signature:

```cpp
struct SpectralProfile {
    int width, height;
    cv::Mat magnitude_bgr[3];   // CV_32FC1 per channel
    cv::Mat phase_bgr[3];       // CV_32FC1 per channel
    cv::Mat consistency_bgr[3]; // CV_32FC1 per channel (quality metric)
    int sample_count;
};

class SpectralCodebook {
public:
    void load(const std::string& path);
    void save(const std::string& path) const;

    // Get profile for resolution (exact or nearest + resize)
    const SpectralProfile& get_profile(int width, int height) const;

    bool has_profile(int width, int height) const;
    void add_profile(const SpectralProfile& profile);

private:
    std::map<std::pair<int,int>, SpectralProfile> profiles_;
    std::string path_;
};
```

**Binary serialization format:**
```
Header: "WMRCB01" (7 bytes magic) + uint32 profile_count
Per profile: uint32 w, uint32 h, uint32 sample_count
  Per channel (3x): uint32 rows, uint32 cols, float data[rows*cols]
```

No compression needed — profiles are small (H × (W/2+1) for r2c FFT output per channel).

### V3 Spectral Codebook Subtraction Pipeline

Ported from `reverse-SynthID/V3_codebook_subtraction.py`:

1. **Profile selection:** Look up exact `(H, W)` match in codebook. If not found, find nearest resolution and resize profile via spatial-domain interpolation.

2. **Per-channel FFT subtraction:**
   ```
   for each channel (B, G, R):
       F = FFT(image_channel)
       D = magnitude(F) * exp(j * phase(F))  // complex spectrum
       profile = codebook.get_profile(H, W).channel[c]

       D_clean = D - strength * profile.magnitude * exp(j * profile.phase)
       D_clean = clamp_magnitude(D_clean, max(0, min_mag), max_mag)  // safety caps

       clean_channel = IFFT(D_clean)
   ```

3. **Safety caps:** Clamp cleaned magnitude to `[min_observed * 0.5, max_observed * 1.5]` to prevent artifacts from over-subtraction.

4. **Strength parameter:** Default 1.0. `--strength 0.5` for gentle removal, `--strength 2.0` for aggressive.

### Bundled Codebooks

Pre-built codebooks for common Google image resolutions, generated from reference pure-black and pure-white images produced by Gemini/Imagen:

- Common resolutions: 512x512, 768x768, 1024x1024, 1024x1792, 1792x1024, 1792x1792
- Stored as compressed binary blobs in `assets/codebook_assets.hpp` (same constexpr pattern as embedded_assets.hpp)
- Loaded at runtime into SpectralCodebook

**Source:** Generated from reverse-SynthID reference images using the codebook builder (Phase 4).

### Files

| File | Purpose |
|------|---------|
| `src/core/fft_context.hpp` | FftContext class declaration |
| `src/core/fft_context.cpp` | FFTW3 wrapper implementation |
| `src/synthid/spectral_codebook.hpp` | SpectralProfile + SpectralCodebook |
| `src/synthid/spectral_codebook.cpp` | Codebook load/save/query |
| `src/synthid/codebook_subtractor.hpp` | V3 subtraction pipeline |
| `src/synthid/codebook_subtractor.cpp` | FFT subtraction with safety caps |
| `src/assets/codebook_assets.hpp` | Bundled codebook binary blobs |

### CLI Changes

Phase 3 adds a `--synthid` flag (subcommands come in Phase 5):

```
wmr input.png --synthid                       # Remove SynthID using bundled codebook
wmr input.png --synthid --codebook custom.cb  # Use custom codebook
wmr input.png --synthid --strength 0.5        # Gentle removal
wmr input.png --synthid --codebook-dir ./cb/  # Load codebooks from directory
```

Note: Phase 5 restructures the CLI into subcommands (`wmr synthid input.png`).

### Dependencies

- FFTW3 (via vcpkg: `fftw3`)

### Verification

1. Generate a SynthID-watermarked image using Google's tools
2. Run `wmr synthid test.png -o clean.png`
3. Verify SynthID detection score drops significantly after removal
4. Visual inspection — no artifacts introduced

---

## Phase 4: SynthID Detection + Codebook Building — NOT ACHIEVABLE (see Current Status)

### Goal

Add SynthID watermark detection and the ability for users to build custom codebooks from their own reference images.

### Detection: Multi-Method Bayesian Fusion

Ported from `reverse-SynthID/V3_codebook_subtraction.py` detection pipeline.

Four detection methods fused with weighted averaging:

**Method 1 — Noise Correlation (weight: 0.35):**
- Extract noise residual by subtracting denoised version (bilateral filter as cheap denoiser)
- Compute correlation between noise residual and known SynthID carrier pattern
- High correlation → SynthID present

**Method 2 — Carrier Phase Matching (weight: 0.35):**
- FFT of image → extract phase at known carrier frequency bins
- Compare against expected phase pattern for that resolution
- Phase coherence score → SynthID present

**Method 3 — Structure Ratio (weight: 0.15):**
- Ratio of energy at carrier frequencies vs surrounding frequencies
- SynthID adds structured energy at specific bins
- Elevated ratio → SynthID present

**Method 4 — Multi-Scale Consistency (weight: 0.15):**
- Run detection at multiple resolutions (original, 1/2, 1/4)
- SynthID is resolution-dependent — consistent detection across scales is strong signal
- Inconsistent → likely false positive

**Fusion:**
```
score = 0.35 * noise_corr + 0.35 * carrier_phase + 0.15 * structure_ratio + 0.15 * multi_scale
detected = score > threshold (default 0.5)
```

**Target accuracy:** ~90% (matching reverse-SynthID Python implementation).

### Codebook Builder

`wmr build-codebook` subcommand:

```
wmr build-codebook ./reference_images/ -o my_codebook.cb
```

**Input:** Directory of reference images (pure black or pure white images generated by Gemini/Imagen — these contain SynthID but no visible content).

**Process:**
1. For each reference image:
   - Compute per-channel FFT
   - Extract magnitude and phase
2. Group by resolution `(H, W)`
3. For each resolution group:
   - Average magnitude across samples
   - Average phase across samples
   - Compute consistency (standard deviation — low = good quality)
4. Build `SpectralCodebook` with all profiles
5. Validate: flag resolutions with < 3 samples or high inconsistency
6. Save to binary format

**Quality gates:**
- Minimum 3 reference images per resolution
- Consistency score < 0.1 (standard deviation of magnitude)
- Warn on low-quality profiles, don't reject (user may have limited references)

### Files

| File | Purpose |
|------|---------|
| `src/detection/synthid_detector.hpp` | SynthIDDetector class declaration |
| `src/detection/synthid_detector.cpp` | 4-method Bayesian detection |
| `src/synthid/codebook_builder.hpp` | Codebook builder declaration |
| `src/synthid/codebook_builder.cpp` | Build codebook from reference images |

### CLI Changes

```
wmr detect input.png                       # Report all detected watermarks
wmr build-codebook ./refs/ -o codebook.cb  # Build custom codebook
```

### Verification

1. Detect SynthID on known-watermarked images — should report `detected=true`
2. Detect SynthID on clean images — should report `detected=false`
3. Build codebook from 5+ reference images — should produce valid codebook
4. Use custom codebook for removal — should work same as bundled

---

## Phase 5: Unified CLI + Batch Processing

### Goal

Merge visible and SynthID pipelines into a single auto-detecting workflow with batch support and comprehensive tests.

### Unified Pipeline

```
wmr input.png
  → Load image
  → Visible detection (NCC)
     → If found: remove visible watermark + inpaint
  → SynthID detection (Bayesian)
     → If found: remove SynthID
  → Save output
```

Both checks always run. An image can have both watermarks. Removal is sequential: visible first (spatial domain), then SynthID (frequency domain).

### Subcommand Structure

```
wmr input.png                   # Auto-detect and remove all (default behavior)
wmr remove input.png            # Explicit alias for default
wmr detect input.png            # Report what's detected, don't modify
wmr visible input.png           # Force visible-only processing
wmr synthid input.png           # Force SynthID-only processing
wmr build-codebook <dir> -o <cb># Build custom SynthID codebook
wmr --help                      # Usage
wmr --version                   # Version
```

### Batch Processing

```
wmr remove ./images/            # Process all images in directory
wmr remove ./images/ -r         # Recursive
wmr remove ./images/ -o ./out/  # Output to separate directory
```

- Supported formats: .png, .jpg, .jpeg, .webp
- Preserves directory structure when using `-o`
- Reports per-file results and summary

### Test Suite

Using Catch2 v3:

**Unit tests:**
- `blend_modes_test.cpp` — alpha map calculation, forward/reverse blending
- `fft_context_test.cpp` — FFT round-trip, magnitude/phase extraction
- `spectral_codebook_test.cpp` — load/save, profile lookup, fallback
- `ncc_detector_test.cpp` — detection on known watermarked/clean images
- `synthid_detector_test.cpp` — detection accuracy on marked/clean images
- `codebook_builder_test.cpp` — build from references, quality validation
- `inpaint_test.cpp` — inpainting methods produce valid output

**Integration tests:**
- `visible_pipeline_test.cpp` — end-to-end visible removal with detection
- `synthid_pipeline_test.cpp` — end-to-end SynthID removal with detection
- `unified_pipeline_test.cpp` — auto-detect both watermarks, remove both
- `batch_test.cpp` — batch processing on directory of mixed images

### Files

| File | Purpose |
|------|---------|
| `src/cli/batch_processor.hpp` | Batch processing declaration |
| `src/cli/batch_processor.cpp` | Directory traversal, per-file orchestration |
| `tests/CMakeLists.txt` | Test executable and linking |
| `tests/unit/*.cpp` | Unit tests (7 files) |
| `tests/integration/*.cpp` | Integration tests (4 files) |

### CLI App Restructure

`cli_app.cpp` restructured to use subcommands via CLI11:

```cpp
auto* remove_cmd = app.add_subcommand("remove", "Auto-detect and remove watermarks");
auto* detect_cmd = app.add_subcommand("detect", "Detect watermarks without modifying");
auto* visible_cmd = app.add_subcommand("visible", "Remove visible watermark only");
auto* synthid_cmd = app.add_subcommand("synthid", "Remove SynthID only");
auto* build_cmd = app.add_subcommand("build-codebook", "Build SynthID codebook");
```

Default (no subcommand) falls through to `remove` behavior for backward compatibility.

### Dependencies

- Catch2 (via vcpkg: `catch2`)

### Verification

1. `wmr test-images/2400x1792-gemini.png` — auto-detects visible, removes it
2. `wmr test-images/synthid-test.png` — auto-detects SynthID, removes it
3. `wmr detect test-images/2400x1792-gemini.png` — reports visible found, SynthID not found
4. `ctest --test-dir build/` — all tests pass
5. `wmr remove ./test-images/ -o ./test-output/` — batch processes all images

---

## Phase 6: AI Denoise (Optional)

### Goal

Improve residual cleanup quality using FDnCNN neural network via NCNN+Vulkan.

### Design

- Optional CMake flag: `WMR_ENABLE_AI_DENOISE=ON`
- NCNN inference with Vulkan GPU acceleration
- FDnCNN model (pre-trained, bundled in assets/)
- Applied as final cleanup step after traditional inpainting

### Scope

- Model files: `assets/fdncnn_model.bin`, `assets/fdncnn_model.param`
- Build flag gates compilation of AI denoise module
- Falls back to traditional-only if not enabled

### Dependencies (optional)

- ncnn (via vcpkg: `ncnn`)
- Vulkan headers (system)

---

## Phase 7: Video Support (Future)

### Goal

Extend `wmr` to process video files (MP4, AVI, MOV) for Veo-generated content.

### Design

- Frame-by-frame extraction via OpenCV VideoCapture
- Each frame processed through both visible and SynthID pipelines
- Re-encode via OpenCV VideoWriter or ffmpeg subprocess
- CLI: `wmr video input.mp4 -o output.mp4`

### Scope

- Video I/O (OpenCV videoio module)
- Per-frame processing with optional temporal consistency
- Progress reporting (frame N/total)
- Audio passthrough (ffmpeg subprocess)

---

## Build System Evolution

### vcpkg.json Changes Per Phase

| Phase | Additions |
|-------|-----------|
| Phase 2 | None |
| Phase 3 | `fftw3` |
| Phase 4 | None |
| Phase 5 | `catch2` |
| Phase 6 | `ncnn` (optional) |
| Phase 7 | None (OpenCV videoio already linked) |

### CMakeLists.txt Changes Per Phase

| Phase | Changes |
|-------|---------|
| Phase 2 | Add `src/detection/`, `src/core/inpaint.cpp` to SOURCES |
| Phase 3 | Add `src/synthid/`, `src/core/fft_context.cpp`; find_package(FFTW3); link `fftw3` |
| Phase 4 | Add `src/detection/synthid_detector.cpp`, `src/synthid/codebook_builder.cpp` |
| Phase 5 | Add `src/cli/batch_processor.cpp`; test CMakeLists.txt; link Catch2 |
| Phase 6 | Conditional NCNN linking, optional sources |

---

## Reference Projects

### GeminiWatermarkTool (allenk/GeminiWatermarkTool)

**Key source files for porting:**
- `src/detect_watermark.cpp` — NCC detection (Phase 2)
- `src/detect_watermark_guided.cpp` — multi-scale search (Phase 2)
- `src/inpaint.cpp` — Gaussian/TELEA/NS inpainting (Phase 2)
- `src/embedded_assets.hpp` — constexpr PNG arrays (done)

### reverse-SynthID (layerd-filtering/reverse-SynthID)

**Key source files for porting:**
- `V3_codebook_subtraction.py` — V3 removal pipeline + detection (Phases 3, 4)
- `codebook_handler.py` — codebook I/O (Phase 3)
- `detection_pipeline.py` — 4-method Bayesian fusion (Phase 4)

### Python → C++ Equivalents

| Python | C++ |
|--------|-----|
| numpy arrays | cv::Mat (CV_32FC1/CV_32FC3) |
| numpy.fft / scipy.fft | FFTW3 via FftContext |
| numpy.savez / numpy.load | Custom binary format |
| sklearn.decomposition (PCA/ICA) | Not needed for V3 pipeline |
| pywt (wavelet denoise) | Not needed for V3 pipeline |
| PIL.Image | cv::imread/imwrite |
