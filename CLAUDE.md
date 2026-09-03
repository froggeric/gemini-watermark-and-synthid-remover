# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

**Recommended (macOS/Homebrew), resilient to `brew upgrade`:**
```bash
scripts/build.sh                 # Release build + tests; self-heals a stale cache
BUILD_TYPE=Debug scripts/build.sh
RUN_TESTS=0 scripts/build.sh     # build only
```
`scripts/build.sh` verifies the required Homebrew formulas, auto-wipes + reconfigures when a cached dependency path has vanished after an upgrade, and configures against the stable `/opt/homebrew/opt/…` symlinks. Binaries: `build/wmr`, `build/tests/wmr_tests`. Test suite needs Catch2 (`brew install catch2`).

Or, arm64 preset: `cmake --preset mac-homebrew-Release && cmake --build --preset mac-homebrew-Release`.

**System libs (macOS/Homebrew), manual, no vcpkg:**
```bash
cmake -B build -S . -GNinja \
  -DCMAKE_PREFIX_PATH="$(brew --prefix opencv);$(brew --prefix ffmpeg);$(brew --prefix catch2);$(brew --prefix fmt);$(brew --prefix spdlog);$(brew --prefix cli11)" \
  -DOpenCV_DIR=$(brew --prefix opencv)/lib/cmake/opencv4 \
  -DFFMPEG_ROOT=$(brew --prefix ffmpeg) \
  -DWMR_BUILD_TESTS=ON
cmake --build build
```

**vcpkg (all platforms):**
```bash
cmake -B build -S . -GNinja
cmake --build build
```

**Tests:**
```bash
ctest --test-dir build --output-on-failure
./build/tests/wmr_tests "[v2]"          # single tag (path: tests/wmr_tests)
```
Integration tests need project root as CWD (they look for `test-images/` relative to CWD). Tests use `SKIP` macro if test data is absent, so they don't fail without it.

## Architecture

Single-pass C++20 CLI tool. No libraries, everything compiles into one `wmr` executable.

### Pipeline: Detect → Remove → Inpaint

`WatermarkEngine` (src/core/) orchestrates the image pipeline:

1. **NccDetector** (detection/), 3-stage NCC: spatial template match (cv::matchTemplate), gradient match (Sobel magnitudes), variance analysis. Fusion: spatial×0.50 + gradient×0.30 + variance×0.20. Threshold: 0.35.
2. **Reverse alpha blend** (core/blend_modes), `original = (watermarked - alpha*255) / (1-alpha)`. Alpha maps decoded from embedded PNGs (assets/embedded_assets.hpp).
3. **Inpaint** (core/inpaint) — OFF by default since 1.14.0: the default is the pure
   reverse-alpha-blend (the exact inversion). The opt-in `--denoise ns|telea` cleanup
   is **residual-only** (predicts clean content with cv::inpaint and blends it in only
   where the reverse-blend left a real residual, never blurring a clean removal).

### AI Denoise (optional, OFF by default)

An FDnCNN denoiser (`src/core/ai_denoise.{hpp,cpp}`, NCNN + Vulkan, CPU fallback) is an optional residual-cleanup method, gated on `WMR_BUILD_AI_DENOISE`. Since 1.14.0 the default cleanup is **off** (pure reverse-alpha-blend, the exact inversion); AI is opt-in via `--denoise ai` (it denoises the reverse-blended region, so prefer it only when there is real residual). The lean OFF build is provably AI-free.

- **Build:** `scripts/build.sh` builds FULL by default on macOS (AI denoise + MI-GAN + SynthID regen + CoreML SD; inits the NCNN + stable-diffusion.cpp submodules, checks `vulkan-volk`/`molten-vk`/`openssl@3`). There is no separate "lean" build. Set `WMR_AI_DENOISE=0` / `WMR_AI_MIGAN=0` / `WMR_BUILD_REGEN=0` to opt a feature out. On linux/windows, regen + CoreML SD are off by default (sdcpp build gaps — see CI); AI denoise + MI-GAN are on. CI uses the vcpkg `ai-denoise` manifest feature (`volk`), no Vulkan SDK install. NCNN is a git submodule; volk comes from vcpkg.
- **CLI:** `--denoise` is always available (not AI-gated): default `off` (pure reverse-blend), or `soft|ns|telea` residual-only cleanup; `ai` only when built. `--strength`/`--radius` tune it; `--sigma` is AI-only. OFF build has `--inpaint-strength`.
- **Dispatch:** `WatermarkEngine::remove_watermark_detected` takes an `InpaintConfig` overload (the `float` overload forwards). AI dispatches in the engine (engine-level, not in `inpaint.cpp`), keeps ncnn headers out of the inpaint TU. All AI symbols are `#ifdef WMR_AI_DENOISE`-guarded so the OFF build compiles with zero AI knowledge.
- **Singleton lifetime:** `WatermarkEngine::denoiser()` returns an **intentionally-leaked** heap `NcnnDenoiser` (never destroyed). Destroying the embedded `ncnn::Net` during C++ static teardown races ncnn's global Vulkan-device teardown → EXC_BAD_ACCESS in `VulkanDevice::vkdevice()` at exit (only on the GPU path). Leaking the singleton is the standard fix for a process singleton owning a Vulkan context. Do NOT turn it back into a static-local.
- **Release build:** single full-package build per (OS, arch), no lean/full split, no separate AI tarball. A separate `tests` job builds AI+TESTS ON to cover the AI/routing unit tests. The `build` matrix has 4 legs, all `WMR_BUILD_AI_DENOISE=ON` + `WMR_BUILD_AI_MIGAN=ON`:
  - **mac arm64 (native):** the only non-system dynamic deps are the Vulkan loader (`libvulkan.1.dylib`, a hard dyld load command forced by `-DVulkan_LIBRARY`) + MoltenVK (`libMoltenVK.dylib`, dlopened at runtime); everything else is static. `scripts/bundle_macos_vulkan.sh` bundles both (load cmd → `@rpath`, `@loader_path/lib` rpath, co-located `MoltenVK_icd.json`, ad-hoc re-sign) + a `wmr` launcher that sets `VK_ICD_FILENAMES`, so the tarball runs on a clean macOS (no SDK/Homebrew/MoltenVK). CI installs `vulkan-loader` + `molten-vk` on this leg only.
  - **mac x86_64 (cross-compiled on the arm64 runner via `x64-osx` triplet + Rosetta; the only Intel runner `macos-13` was retired):** built `-DWMR_NCNN_VULKAN=OFF` (CPU-only AI). On APPLE, NCNN's simplevk does *static* Vulkan linkage and needs a build-time `libvulkan`, but there's no x86_64 Vulkan dylib on the arm64 cross-build runner, so Vulkan is compiled out (`ai_denoise.cpp` guards the GPU calls behind `#if NCNN_VULKAN`; NCNN propagates it via `platform.h`'s `#cmakedefine01`).
  - **linux / windows:** pass no `Vulkan_LIBRARY` → NCNN `simplevk` (runtime `dlopen`, no Vulkan at build time, graceful CPU fallback).
  - **Windows CI gotchas (1.7.1):** (1) the `cmake` step must run inside the MSVC dev env (`vswhere` + `vcvarsall.bat x64`, `shell: cmd` in `release.yml`) or CMake picks MinGW gcc from PATH → MSVC-vs-MinGW link failure; (2) `CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded` is set as a *variable* (not just the wmr target property) so the `ncnn` subproject inherits `/MT` to match the `x64-windows-static` deps, else LNK2005 (`msvcprt` vs `libcpmt`) + LNK2019 (`__imp_*` ucrt stubs).
  - **MI-GAN (1.10.0) is platform-split.**
    - **macOS (arm64 + x86_64) = native CoreML:** `WMR_BUILD_AI_MIGAN=ON` → the CMake `if(APPLE)` branch compiles `migan_coreml_inpainter.mm`, links the system CoreML framework (no ORT, no vendored lib), and ships the 14 MB `migan_512_places2_fp16.mlpackage` dir (Git LFS via `assets/.../weight.bin`; `lfs: enable_migan` on checkout). mac x86_64 is now a **tarball** (was a bare binary) so the `.mlpackage` sits next to it; Intel Macs gain MI-GAN (were OFF in 1.9.0).
    - **linux/windows = ORT** (unchanged from 1.9.0): the ONNX Runtime shared lib + 27 MB `migan_pipeline_v2.onnx`, shipped as **archives** (`wmr` + `libonnxruntime.so.1`/`onnxruntime.dll` + model; linux `patchelf --set-rpath '$ORIGIN'`). ORT = official v1.27.1 prebuilt fetched at CMake configure (NOT the vcpkg port, heavy source build would threaten the Windows 6 h cap), `IMPORTED` target `wmr_ort`, SHA256-pinned; windows post-build-copies `onnxruntime.dll` next to `wmr.exe` (exe-dir search).
  - Licenses ship in `LICENSE-THIRD-PARTY.md`.

### SynthID (regen only, since 1.16.0)

The spectral SynthID detection + suppression path (CodebookSubtractor, NoiseResidualSubtractor, CodebookBuilder, SynthidDetector, FftContext, FFTW3) was **removed in 1.16.0** because it did not work: the detector scored ROC AUC 0.20 on Google-verifier-labeled images, and a clean codebook is inert on content (the carrier is ~0.025/255, below the noise floor). wmr does **not** detect SynthID (Google's "Verify with SynthID" is a manual in-app tool with no API, so it cannot drive an automated detector). The decision record, evidence, and the conditions to revisit detection are in `docs/research/synthid-spectral-removal-record.md`.

The only SynthID operation is `--synthid-attack regen` (lossy SDXL img2img). `--synthid-attack` is a pluggable method selector (IsMember `{"regen"}`, default `regen`); a future method is one new IsMember value + one new dispatch branch. On `remove`, regen runs only when `--synthid-attack` is passed explicitly (lossy + ~6.5 GB download, so opt-in); on `synthid`, regen is the default.

### SynthID diffusion-regen + CoreML SDXL backend (Phase 2 + 3)

- `--synthid-attack regen` (the only method since 1.16.0) = low-strength SDXL img2img of the whole image (the only SynthID-Image scrub the literature reports as validated; LOSSY (validated against Google's manual "Verify with SynthID" verifier; default strength 0.10)). Behind `WMR_BUILD_REGEN` (sdcpp CPU backend, vendored `external/stable-diffusion.cpp`) + `WMR_BUILD_AI_COREML_SD` (native CoreML mac). Knobs: `--regen-backend {auto,cpu,metal,vulkan,coreml}`, `--regen-strength/-steps`, `--regen-no-download`.
- **CoreML SD pipeline** (`src/core/coreml_sd_{scheduler,pipeline.mm,model_fetch}` + `regenerator.cpp`): from-scratch ObjC++ port (Euler scheduler over Accelerate + VAE encode/decode + baked empty-prompt embeds; text encoders NOT shipped). mac `auto` resolves the backend FIRST and BOOTSTRAPS CoreML: downloads its models on first run, CPU fallback only if that fails or `--regen-no-download`; the sdcpp checkpoint is fetched ONLY when the sdcpp path actually runs (before 1.16.11 it was downloaded before the backend was chosen, so every fresh Mac burned 7.2 GB then ran the several-times-slower CPU path). Since 1.16.3 the UNet uses `ORIGINAL` attention and is GPU-bound; the **ANE is unused** (SDXL's large fp16 attention matmuls are ANE-ineligible, so CoreML places the UNet on the GPU under `MLComputeUnitsAll`); the prior `SPLIT_EINSUM` UNet was GPU-bound too (it did NOT collapse to CPU as once thought). ~20s/1024-tile on M4, ~1.2x faster than SPLIT_EINSUM. `$WMR_COREML_SD_COMPUTE_UNITS`={all,cpu_gpu,cpu_ane,cpu} overrides the unit (A/B + force-GPU escape). Models auto-download from `huggingface.co/froggeric/wmr` (SHA256-pinned in `coreml_sd_model_fetch.cpp`, tar.gz-extracted; `--regen-no-download` refuses).
- **Download progress + leftover cleanup (1.16.11):** an HF fetch is 302 -> CDN and curl fires the xferinfo cb during the redirect body too, reporting the ~1 KB redirect page's Content-Length as the total; `download_pinned_file` must suppress redirect-phase ticks (a `Location:` header on a 3xx marks them; the write cb alone is NOT a safe gate, an empty final body never fires it), adds `have` to the 206-resume total, and parses `HTTP/` status lines generically (the old `HTTP/1.`-only prefix missed HTTP/2 = all of HF, so `server_partial` never fired there). `ByteProgress::set_total` lets the LAST non-zero total win, percent clamps to 100, TTY lines cap to the terminal width via `terminal_width()`/`format_byte_line()` (bar shrinks, then the label truncates BEFORE bolding; a wrapped line cannot be rewritten by \r), and piped milestone lines bucket on the clamped percent. The displayed rate is a `WindowedRate` (4s re-anchoring window, blended 50/50 with the previous window, since-start average only until the first window closes, ETA gated on TWO closed windows): per-tick EWMA rates swung 5x-300x on bursty CDN delivery and made the ETA jump 15s<->15min. `make_byte_progress` (model_downloader) is shared by the sdcpp + CoreML fetches (the CoreML 5-6 GB fetch was silent before). Once CoreML is in use, `offer_leftover_cpu_models_cleanup` (regenerator.cpp) offers the one-time y/N delete of the cached sdcpp models (TTY-gated prompt; framed stderr notice otherwise); `wmr cache --clear-cpu-models` is the manual path. Regression tests: `regen downloader redirect total` (local python 302 server; posix-only) + `format_byte_line` cases in progress_test.
- **HF model uploads (`froggeric/wmr`):** use `hf upload froggeric/wmr <local> <path-in-repo>` (NOT `huggingface-cli`, which is deprecated and no-ops); it auto-uses the cached `~/.cache/huggingface/token` and auto-tracks `.safetensors`/`.bin` as LFS. Mirroring exact upstream bytes keeps the LFS content-oid (= the SHA256 pin) unchanged. The model-card `license_name` must be lowercase; prefer `license: creativeml-openrail-m`. To delete an HF file: the `hf` CLI has no delete subcommand; use `python3 -c "from huggingface_hub import HfApi; HfApi().delete_file('<path-in-repo>','froggeric/wmr')"`. wmr's runtime never touches `~/.cache/huggingface` (downloads are plain curl into `~/.cache/wmr/`); that dir only holds `hf upload` blobs + the token, so `rm -rf ~/.cache/huggingface/hub` is safe but keep `token`.
- **HF housekeeping:** never keep superseded model files on HF as "rollback" copies (confusing + bad practice). A re-pin reaches cached users via the cache fast-path, so rollback is a code revert + re-convert, not a preserved HF artifact.
- **CoreML model cache fast path (FIXED 1.16.3):** `ensure_coreml_model_files` (`coreml_sd_model_fetch.cpp`) SHA-verifies each cached archive against its current pin via an O(1) sidecar `<archive>.sha256.ok` (the warm path no longer trusts existence alone), so a re-pinned UNet/embeds DOES reach already-cached users. On a re-pin it removes the old archive and extracted `.mlpackage` **before** downloading the new one (peak disk ~= steady state, no accumulation). The embeds are byte-identical across UNet variants, so `kSha256Embeds` is unchanged.
  - CoreML keeps its own compile caches, separate from the wmr model cache: the app-scoped e5rt execution cache `~/Library/Caches/wmr/com.apple.e5rt.e5bundlecache` (observed ballooning to **~138 GB** across model versions/tile configs, with no eviction API — far beyond the ~5 GB once estimated) AND the shared `~/Library/Caches/CoreML` (across all CoreML apps). A stale/partial entry there surfaces as `The file "manifest.plist" couldn't be opened` + `Unable to load MPSGraphExecutable ... @ GetMPSGraphExecutable` (non-fatal; CoreML recovers and recompiles, the regen still completes).
  - **`~/.cache/wmr/` layout** (for disk-cleanup requests): `coreml-sdxl/` = the LIVE regen models (~11 GB, in active use — never delete); `coreml-sdxl-original-*` = stale UNet-variant A/B leftovers (deletable); `coreml-sdxl-venv/` + `ml-stable-diffusion/` = the dormant model-CONVERSION toolchain (~0.9 GB, keep unless you accept rebuilding it on the next re-pin); `update-check.json`; feature subdirs (e.g. `antidetect/`) die with their feature.
  - **Since 1.16.7 the app-scoped `com.apple.e5rt.e5bundlecache` is auto-managed** (cleared on staleness/bloat by `coreml_cache.cpp`; see the "CoreML cache management" section), and `wmr cache --clear-coreml` clears it on demand. Only the shared `~/Library/Caches/CoreML` is still manual: `rm -rf ~/Library/Caches/CoreML` (the next run recompiles, one-time).
  - These cache warnings are Apple framework `os_log`/`nslog` lines, not suppressible from wmr (unlike FFmpeg's `av_log_set_level`).
- **CoreML weight palettization = dead end on the GPU (tested + rejected 2026-08-06):** 6-bit k-means (`coremltools.optimize.coreml.palettize_weights`, `OpPalettizerConfig(nbits=6,mode="kmeans",granularity="per_tensor")`) shrinks the UNet 4.8→1.8 GB but is ~1.9x SLOWER (348s vs fp16 183s) + a ~5-6 min one-time compile. The speedup is ANE-only (JIT LUT-decompression); SDXL's ANE is unused, so on the GPU the dequant is a net cost. Don't re-try; see `docs/research/coreml-sd-placement.md`.
- **Testing a CoreML model before it's the canonical pin:** the `wmr` binary can't run an un-pinned model directly (`ensure_coreml_models` re-downloads the canonical one over it). Use a throwaway models dir: placeholder `<file>.tar.gz` + a `<file>.tar.gz.sha256.ok` sidecar holding the CURRENT pin (makes `fetch_one` skip verify) + the candidate `.mlpackage`, via `WMR_COREML_SD_MODELS_DIR`; or load the pipeline directly (the `[coreml-sd][pipeline]` smoke test / `CoreMLSDPipeline::initialize`), bypassing the bootstrap. Verify a model's IO with a throwaway `.mm` spike: `clang++ -std=c++20 -fobjc-arc spike.mm -framework CoreML -framework Foundation` (`-fobjc-arc`, hyphen not `=`), printing `model.modelDescription.inputDescriptionsByName`.
- **Conversion (dev-only, `apple/ml-stable-diffusion`):** tag **1.1.1** (NOT HEAD, which drags in `diffusionkit`); install `pytest` (coremltools' testing_utils imports it); pin `torch==2.7.0 coremltools==9.0 diffusers==0.39.0 transformers==4.51.3` (newer torch breaks the trace); apply the one-line `_cast` patch in `coremltools/.../torch/ops.py` (`int(len-1 array)` → `.item()` first); **MANDATORY `--custom-vae-version madebyollin/sdxl-vae-fp16-fix`** (the base VAE overflows/darkens in fp16). Full recipe + the C++ IO specs: `~/.claude/plans/coreml-sdxl-phase3.md`.
- **Regen is cross-platform since 1.16.2** (was macOS-Apple-Silicon-only in 1.15.0-1.16.1). sdcpp CPU regen now builds on linux/windows/mac-Intel via `WMR_REGEN_CPU_ONLY` (forces the sdcpp + ggml GPU backends off; ggml defaults `GGML_METAL=ON` on APPLE regardless of `SD_METAL`, so both `SD_*` and `GGML_*` must be forced off) + a macOS cross-arch guard setting `GGML_NATIVE=OFF` when target arch != host (fixes the `-mcpu=apple-m1` cross-compile error). macOS arm64 keeps Metal. The CPU path uses deterministic Euler (`EULER_SAMPLE_METHOD`, NOT `EULER_A`) + the SDXL discrete schedule, matching the validated CoreML `CoreMLSDEulerScheduler` so the 0.10 @ 50-step knee applies uniformly. See CI below.
- **Validated removal knee + the N factor.** Default `--regen-strength 0.10` @ `--regen-steps 50` (5 actual denoise steps) is the validated minimum: cleared 9/9 varied images incl. a double-watermarked image-to-image generation, 2 rounds of Google's detector. Removal is governed by `strength` (the schedule's starting noise), NOT by `steps` (N = granularity) — N=20 and N=50 at s=0.10 are removal-equivalent; N=50 is the tested config. Lighter strengths are NOT reliably safe (0.08/4-step clears singles but misses the double-watermark case; VAE-only round-trip ~40%). Don't lower the default without re-validating against Google's detector. See `docs/research/synthid-light-reconstruction-attacks.md`.
- **Alternative img2img backbones evaluated + rejected (2026-08):** diffusers sweep of Flux.1-schnell, Z-Image-Turbo, SDXL-Turbo, SDXL-Lightning vs SDXL-base. All 4 rejected on visual fidelity: flow models (Flux, Z-Image) destroy text/faces; distilled SDXLs (Turbo, Lightning) show grid/hallucination. SDXL-base is the only acceptable backbone (0.10 is the validated min). The mean-diff metric HID the damage (visual inspection was decisive). Reusable eval protocol + harness: `docs/research/regen-model-evaluation-protocol.md`, `experiments/regen-model-sweep/sweep.py`. Model-selection research may use Python/diffusers; a native port (CoreML/sdcpp) is done only for a proven winner.
- **Scheduler step-collapse gotcha:** actual denoise steps = `int(N*strength)`, truncated. At N=20 every strength <0.10 collapses to a single light step (`int(20*s)` is 0 or 1, both clamp to 1 step). To sweep sub-0.10 strength, raise N (e.g. N=50, where s=0.04/0.06/0.08 = 2/3/4 distinct steps). See `coreml_sd_scheduler.cpp`.
- **SynthID validation oracle:** Google's "Verify with SynthID" IS an official, public verifier (it is how the removal knee was validated), but it is manual-only (in-app), rate-limited (~10 checks/day/account), natural-language, and noisy (documented hallucinations; went 2/3 in third-party tests). There is no verifier API, so wmr cannot confirm a given output in-process and detection cannot be automated. Plan "did it clear?" sweeps around the ~10/day budget, and don't over-interpret a single verdict on near-identical images (it may be detector variance).

### Video Processing

`VideoProcessor` → `VideoReader` + `VideoWriter` (video/):

- Shot-level detection: samples 12 frames across first 90% of video, takes median position
- Per-frame: occlusion gate (skip if NCC < 0.35), position refinement (±4px tolerance vs shot anchor)
- Audio passthrough via fresh input context with timestamp rescaling
- Audio streams created before MP4 header write (valid moov atom)

### Still-image Watermark Geometry (auto-detect, Gemini 3.6+)

Gemini 3.6 Flash's small diamond is **48px** (NOT the 36px Gemini 3.5 still alpha) at
~0.30 alpha for BOTH stills and video. A pristine measurement across 3 clean 1280x720
videos (360 frames, mark on a uniform dark patch) gives the video alpha peak 0.304 / center
0.301, matching the still alpha (shape diff 0.5%). An earlier 720x1280 reading of "video
peak ~0.39 / stronger render" was inflated by busy-content background-estimation error —
disregard it; video ≈ still ≈ 0.30. The removal alpha `v2_diamond_48_still`
(`get_v2_diamond_alpha_48_still`) is the **average of 10 distinct Gemini 3.6 generations**,
each with the mark on a near-uniform-black patch (so `correct_alpha_for_background` is a
no-op); averaging suppresses per-pixel capture noise ~3x vs a single capture (the SynthID
carrier in the mark region is only ~0.025/255 and content-correlated, so the gain is
shot-noise, not SynthID). The VIDEO path removes with this same alpha (see
`select_video_alpha` below) as the single clean source for both. The faint residual
sometimes seen on video is H.264 compression ringing at the diamond edge (not under-removal);
the video path now repairs it with an opt-out edge cleanup (see Video Watermark Geometry).
It sits at a margin the position model
(`v2_small_config_from_dims`) no longer predicts (model 36px@84 vs real 48px@(96,96) at 896x1200). Still images now
content-detect the position AND size like the video path (shipped 1.12.0), adapted
for the single-image reality. Pure unit `src/detection/still_geometry.{hpp,cpp}`
(OpenCV-only, no FFmpeg, links in the test exe like `geometry_detector`).

- **Multi-template (36 AND 48):** `locate_still_watermark_hybrid` is fed both the
  36px (Gemini 3.5) and 48px (Gemini 3.6) diamond alphas; the winner
  (`StillGeometryHit::template_index`) selects the matched-size alpha for removal.
  A single-size assumption fails one of the two Gemini generations (1.13.0 shipped
  36px-only and under-covered 3.6's 48px mark).
- **Why anchored, not a blind corner scan:** video's blind scan works because it
  aggregates ~12 frames (the static mark wins over transient content). A single
  busy still has no such advantage: a faint mark (~0.48 NCC) is beaten by corner
  content (~0.51-0.54) in a blind scan. So the search is **anchored** on the
  model-predicted top-left first (±`kStillAnchorPad`=40 px), then **widened** to the
  bottom-right corner (`max(0,W-320) x max(0,H-320)`) only if the anchored hit is
  not trusted.
- **Trust gate** (reuses `decide_auto_geometry`): a hit that **snaps** to a
  calibrated preset (`snap_still_to_known`, center L1 <= 40 within the size +
  short-side-tier gate) is trusted at `kStillMinConfidence` (0.45); a raw off-table
  hit must clear `kStillHighConfidence` (0.60) or it falls back to the model.
- **Scope:** the search runs for **every V2 profile** (small AND large), not just
  V2-small. Gemini 3.6 stamps a 48px diamond at margin (96,96) even on large (>1024px)
  outputs (e.g. 2400x1792), where `get_watermark_size` wrongly picks Large (96px model)
  and plain `remove` used to find nothing; the search recovers that 48px mark via the
  bottom-right corner window. **V2-large stays byte-identical:** on a real Gemini 3.5
  large image (genuine 96px mark) the 48px template scores ~0.43, below the 0.45
  min-confidence and the 0.60 raw-trust bars, so the search finds nothing trusted and
  returns the model position (verified: identical md5 pre/post the un-gate). V1 keeps
  the model. `WatermarkEngine::resolve_still_geometry` is called **once** at the CLI
  layer (not inside `detect_watermark`, which runs per variant attempt) and returns
  `{pos, alpha}`: the matched alpha is threaded as `custom_alpha` through
  `detect_watermark` and the remover, so a 48px detection removes with the 48px alpha
  (and a `--rect`/`--geo-preset` override too, not the default 36px).
- **Calibrated preset table** `kStillPresets[]` (first entry `gemini36-portrait`,
  896x1200 -> 48px @ margin (96,96)); `kStillPresetNames[]` drives the `--geo-preset`
  IsMember validator (help lists names, unknown rejected). The model is the fallback
  for uncalibrated resolutions.
- **Still-path fixtures:** `test-images/896x1200-test4-gemini36.png` = clear mark on
  near-uniform black (the `v2_diamond_48_still` capture source); `...-test3-...` = faint
  mark on a busy poster (the hard case; auto-detect can fail, so force with
  `--geo-preset gemini36-portrait`).
- **Precedence:** `--rect` > `--geo-preset` > auto-detect > model. New flags on
  `remove`/`visible`/`detect`: `--rect x,y,w,h` (shared `parse_rect` helper, also
  used by video), `--geo-preset <name>`, `--no-auto-geometry`. An explicit
  `--rect`/`--geo-preset` **forces removal at that position** even when the
  detector's confidence is below the 0.35 gate (a faint mark the search localized
  but could not confirm); the override's logo_size picks the removal alpha (a 48px
  box or the 48px preset -> 48px alpha). `--force` still uses the model position
  (unchanged).
- **Polarity caveat:** `locate_still_watermark_hybrid` is polarity-invariant
  (`max(|mx|,|mn|)`), but `NccDetector`'s stage-1 spatial NCC is max-only, so a
  dark-on-bright mark the geometry search localizes can still be rejected by the
  downstream fusion. Bright marks (the common case) confirm; inverted marks fall
  back to `--rect`/`--geo-preset`.
- **Snap generalization + content-suppressed refinement:** `enable_snap` is
  `!explicit_override && (force_position.has_value() || (V2 && Small))` at the three
  CLI sites (`process_single_image`'s `try_remove`, `process_detect`'s `report`, the
  batch path). An explicit `--rect`/`--geo-preset` DISABLES the snap (1.16.8): the snap
  was overriding the user's forced position, so `--rect` was silently ignored. When the
  snap fires (spatial NCC >= 0.60) it re-localizes on a median-background-subtracted
  prominence (kernel 9), not the raw image: on busy content the raw NCC peak straddles
  two integer cells and minMaxLoc picks the wrong one, and the hard diamond edge (a
  ~0.30 plateau, ~1px boundary) turns a 1px error into a visible light/shadow emboss.
  Integer localization only (Gemini places marks at integer margins); a sub-pixel alpha
  shift was tried and reverted (it biased ~0.4px on busy content and reintroduced the
  emboss). `remove_watermark` (`--force`) is unchanged (no force_position, no snap).
- **Follow-up:** recalibrate `v2_small_config_from_dims` (or grow the preset table)
  once more 3.6 resolutions are measured, so the model fallback is accurate too.

### Video Watermark Geometry (auto-detect, default for `VideoVariant::Auto`)

`VideoVariant::Auto` no longer guesses by resolution only; it content-detects the
corner + logo size so `--variant` is rarely needed. Two layers:

- **Pure detector** `src/video/geometry_detector.{hpp,cpp}` (OpenCV only, no
  FFmpeg, so it is in the test link like `notebooklm_gates`): `detect_geometry_in_frames`
  (a clone of `NotebookLMDetector::match_mark`, but matching the real alpha assets
  at **native size** as separate templates, no scale ladder), `snap_geometry_to_known`
  (snap to the `get_video_watermark_geometry` table by center L1, tol 40 px),
  `rect_to_watermark_position`, `effective_alpha_size` (single source of truth for the
  `logo_size > 48 / > 68` alpha gate), and `decide_auto_geometry` (the regression gate).
- **FFmpeg-linked glue** in `VideoProcessor`: `auto_detect_geometry` (samples ~12
  frames over the first 90%, builds CV_8U templates from the engine alphas + a
  corner window, calls the pure detector) and `resolve_effective_geometry`, the
  single chokepoint called once in `process()` before the scenes/std fork (geometry
  is constant across an export). `detect_in_shot` now takes `geo` as a parameter.
  `select_video_alpha` is the single helper that turns a `geo` into an alpha Mat +
  top-left + bbox; it routes the small/large pick through `effective_alpha_size`
  (so the `>48/>68` gate truly has one source) and is called from `detect_in_shot`
  and once in `process()` (its anchor reused by both `--force` branches). **The 48px Gemini
  alpha (removal AND detection template) is the STILL capture** (`get_v2_diamond_alpha_48_still`,
  the averaged clean mask) — there is no separate "video" 48px alpha; the old `v2_diamond_48`
  video capture (~5% wrong shape, mis-calibrated) has been removed entirely. Only the 96px large
  template uses the video `get_v2_diamond_alpha_large`. The masks are also exported as standalone
  PNGs in `assets/watermark-masks/` for community reuse.

  Templates: diamond `{48, 96}` (`get_v2_diamond_alpha_small/_large`), Veo text
  `{68x30, 99x43}`. The 36 diamond is still-only, never video. Corner window
  `max(0,W-320) x max(0,H-320)` (Gemini) / `W-200 x H-120` (Veo). dtype is 8U/8U
  (alpha `CV_32FC1 -> CV_8U` once, frames `cvtColor` grayscale), the `match_mark`
  convention, NOT the `NccDetector` float-image path. Polarity-invariant: the
  location follows the polarity (`loc_mn` when `|min|` wins, `loc_mx` when `|max|`
  wins); the Jiwoks fork used `loc_mx` always and got the wrong corner.

  **Snap is position-based, not size-based:** 720p-1 and 720p-2 both use the 48px
  diamond at different margins (`P720_2.logo_size=44` is vestigial; it gates to
  the 48 alpha), so size cannot tell them apart. `snap_geometry_to_known` only
  considers variants whose `effective_alpha_size` equals the detected size, then
  snaps by center distance. Their centers are ~75 px L1 apart (tol 40; nearer wins).

- **Precedence** (`resolve_effective_geometry`): `--rect` > auto-detect > `--variant`
  > resolution guess. `--no-auto-geometry` opts out; `--force` skips the search
  (uses `--variant`/resolution). `--rect` was renamed from `notebooklm_rect` and
  now serves Gemini/Veo too (still consumed by NotebookLM).
- **Regression gate** (`decide_auto_geometry`, pure, unit-tested): a snapped
  on-table detection is trusted; a raw off-table detection must score >=
  `kAutoOverrideRawScore` (0.60, file-local in `video_processor.cpp`) to override
  the resolution guess; otherwise keep today's behavior. So a busy-corner false
  positive cannot regress a video that already works. `kAutoGeometryMinConfidence`
  is 0.45 (same as NotebookLM). Log line `Geometry: margin=.. logo_size=.. (source=..,
  score=..)` names the branch that ran.

### Video diamond edge cleanup (default-on; `--no-edge-cleanup` to disable)

After the per-frame reverse-blend, the Gemini/Veo diamond path repairs the faint border/halo
that survives at the diamond's edge (H.264 ringing + any under-removal). `wmr::inpaint_diamond_edges`
(`src/core/inpaint.{hpp,cpp}`, the validated "U4" recipe) operates on a **thin ring around the
footprint boundary only** — `band = dilate(footprint, r3) - erode(footprint, r3)` — leaving the
recovered interior and the untouched exterior byte-for-byte intact (unlike `inpaint_residual`,
which can touch the whole footprint). TELEA inpaint the band -> `ref0`; gate to edge pixels where
`max-ch |cur - ref0| > 14` (so a clean reversal is untouched); re-inpaint the gated mask -> blend.
Called from all 4 reverse-blend sites in `process()` via the file-local `apply_edge_cleanup`,
guarded by `config.edge_cleanup && profile != VeoLegacy`. **Safe no-op on clean footage:** when
the mark sits on a uniform background there is no edge residual, the gate fires on 0 pixels, and
nothing changes (verified on 3 Gemini 3.6 1280x720 clips: ghost 0.0015 with/without). It only
fires on busy/compressed footage where a halo is actually present. `EdgeCleanupConfig{ring_radius=3,
residual_thresh=14, inpaint_radius=3, strength=1.0}`.

- **1280x720 (Gemini 3.6 horizontal)** auto-detects cleanly: margin **(96,96)**, logo_size **48**,
  score **1.00** — the same margin as the `gemini36-portrait` still preset. No new variant needed
  (raw detection clears the 0.60 gate). The mark sits on a persistent near-uniform dark patch, so
  detection is trivial (12/12 sampled frames).

### Scene Detection and Splitting (opt-in via `--scenes`)

`SceneDetector` (video/scene_detector), combined BGR Bhattacharyya + MAD:

- Per-channel BGR histogram distance (max across channels) + mean absolute pixel difference
- Combined metric: `max(per_channel_bhatt, mad)`, catches chromatic and structural scene changes
- Default threshold 0.30, minimum scene length 15 frames
- Scans for scene boundaries, splits video into separate MP4 files at cuts
- `SceneInfo` contains only `start_frame`/`end_frame` (half-open interval)
- Single full-video watermark detection via `detect_in_shot()` (default params), applied uniformly across all split files
- `VideoWriter::copy_audio_range(start_sec, end_sec)`, seek-based audio copy with PTS offset subtraction
- Reader reads sequentially across scenes (no seeking within the loop)
- Each output file: I-frame at start, trimmed audio, correct container duration
- `-o` specifies output directory (defaults to `<input>_scenes/`); rejects file paths
- Output naming: `<stem>_<NNN>.mp4` with dynamic zero-padding

### NotebookLM Video Watermark (opt-in via `--notebooklm`)

`NotebookLMDetector` (video/notebooklm_detector.cpp) + `VideoProcessor::process_notebooklm`, removes the NotebookLM rainbow logo + "NotebookLM" wordmark from generated videos (cinematic / explainer / short-portrait exports).

- **Why a separate path**: the NotebookLM mark is semi-transparent, color-adaptive (light-on-dark / dark-on-light, scene-dependent), and H.264-compressed, NOT a reversible constant-alpha overlay. A temporal reverse-alpha recovery was investigated and **ruled out** (α≈0; the mark is adaptive with no mathematical inverse). Removal is spatial inpaint, chosen **per scene** by an adaptive dispatch.
- **Per-scene dispatch** (`process_notebooklm`, 1.6.0+; FSR routing 1.7.0; always-inpaint 1.7.1; **MI-GAN default 1.9.0**; **1.10.1: arm64 MI-GAN-everywhere default + `--notebooklm-method` toggle**; FSR + LaMa removed). `SceneDetector::detect_boundaries` splits the video; **every scene is inpainted** via `inpaint_mark_roi`, with the method chosen per scene by `resolve_inpaint_method` (notebooklm_gates, pure, arch-agnostic, unit-tested; signature `resolve_inpaint_method(complexity, threshold, has_migan, requested, platform_default)`).
  - **Platform default:** MI-GAN-everywhere on Apple Silicon (`#if defined(WMR_AI_MIGAN_COREML) && defined(__arm64__)`, the ANE makes MI-GAN fast, so the complexity pass is *skipped*). Elsewhere (x86_64 incl. a Rosetta-translated arm64 binary, linux/windows) it's the **complexity gate** (`background_complexity`, Sobel gradient-energy in a gapped band around the mark): intricate (score ≥ `notebooklm_complexity_threshold`) → **MI-GAN** (CoreML on mac, ORT on linux/windows; `WMR_AI_MIGAN`), uniform → **NS**.
  - **Override:** `--notebooklm-method {auto|ns|migan}` (`auto` = platform default; `ns`/`migan` force one). NS is always the MI-GAN-unavailable fallback (incl. a runtime model-load failure, caught by `inpaint_mark_roi`'s `is_ready()` check).
  - **Always-inpaint rationale (1.7.1):** a per-scene **presence gate** existed in 1.6.0–1.7.0 and was removed, because template matching couldn't reliably separate a faint-but-present mark from a genuinely-absent scene for this semi-transparent mark, so skipping risked leaving watermarks; inpainting an already-clean patch is imperceptible.
  - Single-file output; audio copied once.
- **Detection** (whole-video, for the bbox): template matching, multi-scale `|TM_CCOEFF_NORMED|` against each of ~12 sampled frames, keep the highest-scoring (polarity-invariant; stable across scene cuts). Template = embedded `notebooklm_mark_png` (98×14 grayscale, `assets/embedded_assets.hpp`). The detected bbox snaps to user-measured exact coordinates per known export mode (`kKnownModes`); unknown resolutions use the raw detection. Min confidence 0.45. **(1.9.0)** the explainer-mode rect was corrected from `(1105,660,131,16)` → `(1085,658,153,20)`, the old one started at the spiral logo's right edge, leaving ~18px of the logo unmasked (NS/FSR/LaMa hid it by blurring; MI-GAN's precise fill exposed it).
- **CLI**: `wmr video in.mp4 -o out.mp4 --notebooklm` (auto-detect); `--rect x,y,w,h` manual override; `--complexity-threshold` (NS↔MI-GAN gate, default 15; gated platforms only, arm64-auto skips it); `--notebooklm-method {auto|ns|migan}` (1.10.1+; `auto`=platform default, MI-GAN-everywhere on Apple Silicon, complexity-gated elsewhere; `ns`/`migan` force one). Config: `VideoWatermarkConfig::{notebooklm_rect, notebooklm_complexity_threshold, notebooklm_method}`.
- **Methods** (`inpaint_mark_roi`, static in video_processor.cpp): **MI-GAN** (`migan` branch) = `MiganInpainter::inpaint_hole` (leaked singleton, `#ifdef WMR_AI_MIGAN`). Two platform impls share one interface (`migan_inpainter.hpp`):
  - **macOS (1.10.0)**, `src/core/migan_coreml_inpainter.mm` (ObjC++, `#ifdef WMR_AI_MIGAN_COREML`): native CoreML fp16 mlprogram on the ANE, **~28 ms/frame** (~11× over ORT-CPU; A/B-verified to match the ORT baseline within Δ1.9/255). The bare `Generator(resolution=512)` is resolution-locked, so it crops a square around the mark → resizes to 512² → builds the (1,4,512,512) f32 input `cat([mask−0.5, img×mask])` (img in [−1,1], mask 0=hole) → predicts → denormalizes → soft-pastes (dilated+blurred mask). Two-step load (`+compileModelAtURL:` cached, then `+modelWithContentsOfURL:configuration:error:`, a CLASS factory, not `[[MLModel alloc] initWith…]` which doesn't exist), `MLComputeUnitsAll` (ANE preferred). **CoreML may deliver the output as Float16 OR Float32, handle both** (`outMA.dataType` check). Replaces ORT entirely on mac: CoreML is a system framework (no vendored lib), the `.mlpackage` is arch-neutral → **both arm64 AND x86_64** ship it (Intel Macs gain MI-GAN; was OFF). Resolved via `$WMR_COREML_MODEL` / `<exedir>/migan_512_places2_fp16.mlpackage` / `<exedir>/../share/wmr/...`. First load compiles the .mlpackage (one-time, cached by CoreML).
  - **Linux/Windows (1.9.0)**, `src/core/migan_inpainter.cpp` (ORT): feeds the **whole frame** (RGB uint8 NCHW, dynamic-res) + mask (0=hole, dilated 5×5) → uint8 result. ~225 ms/frame CPU. ORT = vendored official v1.27.1 prebuilt (NOT the vcpkg port, heavy source build threatens the Windows 6 h cap): `file(DOWNLOAD)`+SHA256, `IMPORTED wmr_ort`; linux `patchelf $ORIGIN` + libonnxruntime.so.1, windows onnxruntime.dll in exe dir. Model `assets/migan_pipeline_v2.onnx` (~27 MB, LFS) via `$WMR_MIGAN_MODEL` / `<exedir>/…`.
  - **(1.10.0) mac no longer fetches ORT**, the `if(WMR_BUILD_AI_MIGAN)` block splits `if(APPLE)` (CoreML: `enable_language(OBJCXX)`, `find_library CoreML+Foundation`, `.mm`, `copy_directory` the `.mlpackage`) / `else()` (ORT). `video_processor.cpp` dispatch + leaked singleton are byte-identical for both.
  - **NS** = `cv::inpaint` Navier-Stokes on a lean `radius+4` padded crop (uniform-scene + MI-GAN-unavailable fallback). MI-GAN is MIT (Picsart, ICCV 2023). **Dead weight:** vcpkg `opencv4[contrib]`/xphoto is still built though FSR is gone, drop in a follow-up.
- **History note, "CoreML is slower" was WRONG:** the 1.9.0 finding "GPU (CoreML) tested, SLOWER (602 ms)" was ORT's *CoreML execution provider* (only 375/559 nodes on CoreML → partition overhead). A **native** `coremltools` mlprogram (whole graph in one MIL program, no partitioning) is the opposite, 28 ms, 11× faster. Don't re-conclude CoreML is slow from the ORT-EP number.
- **Known limitation** (1.10.0): MI-GAN is sharp + reliable on cartoons/textures. macOS CoreML ~28 ms/frame on the ANE; Linux/Windows ORT ~225 ms/frame. macOS requires 14+ (the .mlpackage targets `minimum_deployment_target=macOS14`; the Conv/LeakyRelu/Resize graph needs only macOS12+ mlprogram, re-verified 28 ms / Δ1.9 hold at macOS14). CoreML on the GitHub `macos-14` runner falls back to CPU (paravirtualized ANE), CI smoke proves load+predict+teardown, not the 28 ms (arm64-ANE-native, verified locally). x86_64 CoreML *link* is verified (CoreML.tbd has an x86_64 slice); x86_64 *runtime under Rosetta* is the CI Step-0 open item (fallback: flip `enable_migan`→`false` for x86_64 only, the CMake split makes it a one-line matrix revert).

### Anti-detection research: CLOSED, archived, NOT in the app

A `wmr antidetect` pipeline (camera-statistics physics + Square Attack vs local surrogate detectors) was fully built, calibrated, and externally measured on 2026-09-02, then **removed from the app before any release**: it clears open/forensic detectors (corvi 12/12, commfor 3-5/5, 0/14 real-photo FPs) but nothing moves a top commercial detector (Hive Detect stayed 99.9% AI across invisible physics, max-strength physics, regen, imperceptible eps-8 craft that flips the local ensemble, and visibly aggressive craft, with and without a watermark). Shipping it would have been misleading. **Do not re-add it to the app.** The methodology, findings, and code are archived separately in `research/antidetect/` (entry point: its README, with the verdict table, the hivedetect.ai verification oracle, and `wiring-against-1.16.11.patch` to reconstruct the building feature on a 1.16.11-era tree). The one untried path (self-trained DINOv2/DINOv3 committee surrogate, degrade-not-defeat ceiling at visible quality) is documented there.

### Update check (notify-only, default ON since 1.16.4)

On a real subcommand, if a human is at a TTY and has not opted out, wmr fetches the latest GitHub release tag (once per 24h, cached at `~/.cache/wmr/update-check.json`) and prints a styled three-line notice to stderr when newer. **Notify-only**: it never downloads a binary, never runs fetched code, and can never delay or break the real command (it runs at a single chokepoint at the END of `run_cli`, after the subcommand's output; `rc` is returned untouched).
- **Build:** `WMR_UPDATE_CHECK` (default ON) links `curl` (now a base vcpkg dep; also pulled by `WMR_BUILD_REGEN`) and `#ifdef`-guards all of `src/core/update_check.{hpp,cpp}`. `WMR_UPDATE_CHECK=OFF` is update-check-free and curl-free-from-this-feature (`nm wmr | grep update_check` is empty). `curl` was added to the base vcpkg deps so a default-ON configure works even with regen OFF; locally on macOS `find_package(CURL)` resolves the system/SDK `libcurl.tbd`.
- **Privacy hard invariant: zero payload.** The fetch sends only the fixed HTTPS `/releases/latest` URL, a versionless `User-Agent: wmr`, the fixed `Accept` header, and the optional GitHub-issued `If-None-Match` ETag. No version/OS/arch/id anywhere (not in the UA, URL, or headers). A visible-only user otherwise makes zero network calls, so this is the first; it is documented in the README and the notice, and is opt-out.
- **Suppression (before any I/O):** `--no-update-check`; `WMR_NO_UPDATE_CHECK`, `CI`, `DO_NOT_TRACK=1` env; non-TTY stderr. Structural skip: `--version`/`--help`/no-args/parse-errors return from the `CLI::ParseError` catch (`cli_app.cpp` ~`:642`) or the no-args early return, both upstream of the post-dispatch chokepoint, so they never reach the check.
- **Gotchas:** (1) api.github.com returns `Content-Encoding: gzip` by default; `fetch_latest_release` MUST set `CURLOPT_ACCEPT_ENCODING ""` or the JSON arrives as raw gzip and the check silently never works (the `[update-check][smoke]` test, SKIP unless `WMR_NETWORK_TESTS=1`, guards this). (2) The orchestrator is split: a gate-free `run_update_check` core (CI-testable; driven directly by the AC-10 test) and the gated `maybe_check_for_update` entry `run_cli` calls. (3) `parse_version` rejects overflow via `errno==ERANGE || n > numeric_limits<unsigned>::max()` (`strtoul` is `unsigned long`, 64-bit on mac/linux, so the `>UINT_MAX` arm is what catches `99999999999`). (4) The cache path comes from `src/core/paths.cpp` `user_cache_dir()`, extracted from `regenerator.cpp` so it resolves when regen is OFF; `paths.cpp` is in the top-level `SOURCES` AND the test `LIB_SOURCES`, and must NOT be re-listed in the `WMR_UPDATE_CHECK` test mirror block (duplicate-source error). (5) `print_header`'s dash rule is the shared `wmr::kHeaderRule` (44 dashes); `format_notice` reuses it so the two framed blocks cannot drift.

### Detail-restoration (post-regen, 1.16.6+)

After SynthID regen produces R, `wmr::restore_detail(O, R, cfg)` (`src/core/regen_restore.{hpp,cpp}`, `#ifdef WMR_BUILD_REGEN`) optionally restores the top-5%-magnitude diff pixels (`O - R`) after a characterized-carrier Wiener filter (`kissfft`, gamma=4, capped) suppresses the watermark component in the low-mid band. **Luminance-gated by default** (`RestoreMode::Auto`): bright images (mean luma >= 128) restore; dim images skip (full regen, the safe path). `--regen-restore-detail` forces on; `--no-regen-restore-detail` forces off. Best-effort (no in-process verifier); it *dilutes* the watermark below the aggregate detection threshold, it does not remove it. The luminance gate is empirical (10-image clearance study; mean luminance is the clean predictor of failure; the carrier SNR scales ~1/luminance). See `docs/research/synthid-diff-restoration-analysis.md`. The multi-tile path must `.clone()` the pre-regen original before writing R back (a shallow copy silently overwrote O with R, making restoration a no-op on >1024px images; the unit test missed it because it used separate buffers). `external/kissfft/` (BSD-3): float build, the inverse has no 1/N scaling, so `regen_restore.cpp` applies it explicitly to match numpy's `ifft2`.

### Progress UX (1.16.6+)

`src/cli/progress.{hpp,cpp}` (always built, not feature-gated) provides `ProgressReporter` (discrete: tiles/frames/batch), `ByteProgress` (downloads, curl-xferinfo-driven), `Stage` (the `[k/N]` frame), `RateEstimator` (EWMA; ETA hidden until >=3 samples + >=5% done; single point estimate, no range). Progress writes to **stderr** (spdlog info stays stdout). TTY-detected (`isatty` + `NO_COLOR`/`CI`/`TERM` gates): refreshing `\r` bar on terminal; append-only milestone lines when piped. `--no-progress` suppresses. TTY intra-unit refresh thread (~800ms) moves the bar + ETA within a tile/frame. Used by regen (`regenerator.cpp`), video (`video_processor.cpp`, where `av_log_set_level(AV_LOG_QUIET)` silences FFmpeg/x264 noise), batch (`batch_processor.cpp`). Design: `docs/research/cli-progress-ux-design.md`. Manual TTY verification: `script -q /tmp/cap.raw bash -c 'stty cols 50; ...'` gives a pty (kill the run with `pkill -x wmr`; `pkill -f` matches the bash -c wrapper's own cmdline and kills the session).

### CoreML cache management (macOS, 1.16.7+)

`src/core/coreml_cache.{hpp,cpp}` (`#if WMR_BUILD_AI_COREML_SD && __APPLE__`) auto-manages the CoreML e5rt execution cache (`~/Library/Caches/wmr/com.apple.e5rt.e5bundlecache`, NOT `~/.cache/wmr/`): clears on staleness (wmr version / model pin / macOS version changed, via a sidecar `<e5bundlecache>.wmr_meta`) or bloat (> 6 GB, early-exit walk). Called from `regenerator.cpp` CoreML init. `wmr cache --clear-coreml` clears on demand. Only the app-scoped cache; never `~/Library/Caches/CoreML` (shared).

### CLI

CLI11 subcommands in src/cli/: `remove` (default), `synthid`, `detect`, `video`, `cache`, `metadata`. (`visible` was removed in 1.16.6; use `remove`.) Directory inputs to remove/synthid trigger batch mode (sequential, outputs to `cleaned/` subdirectory).

### Provenance metadata strip (since 1.16.9)

`src/metadata/` (`wmr::provenance`), ungated, ships in every binary. A distinct concern from the visible diamond (`WatermarkEngine` reverse alpha-blend) and from SynthID (`--synthid-attack regen`). The metadata layer **never decodes pixels** and stays OUT of `WatermarkEngine`: it reads raw container bytes, sniffs format by magic (never by extension), and rewrites the container. PNG IDAT bytes and JPEG entropy bytes are copied verbatim (only metadata chunks/markers are dropped), so removal is bit-identical on pixels.

- **Files:** `provenance_constants.hpp` (header-only constexpr; the single source of truth), `png_chunks.{hpp,cpp}`, `jpeg_markers.{hpp,cpp}`, `provenance.{hpp,cpp}` (public API + `sniff_format` dispatch). Pure TUs: the `.cpp`/`.hpp` include only `<span>`, `<vector>`, `<string>`, `<string_view>`, `<optional>`, `<cstdint>`, `<cstddef>`, `<array>`. NO OpenCV or FFmpeg headers (the pure-TU rule from `notebooklm_gates`). `provenance.cpp` dispatches on `sniff_format`: Png -> `scan_png`/`rewrite_png_strip_ai`; Jpeg -> `scan_jpeg`/`rewrite_jpeg_strip_ai`; anything else -> `{ok=true, supported=false}` (caller copies input through unchanged).
- **Fail-safe is supreme:** any malformed input is returned UNCHANGED, never truncated or clamped. Every offset arithmetic uses 64-bit `size_t` and validates `off + (len + 12) <= size` as a single check before reading; PNG `len` is 4 bytes (max 2^32-1), ISOBMFF `largesize` is validated against remaining. No allocation from an untrusted length without validation. `RewriteResult.ok == false` => caller copies input bytes verbatim.
- **Dual type+content C2PA detection:** the strip matches BOTH the storage type (PNG `caBX` chunk; JPEG APP11 carrying the JUMBF "c2pa" content-type UUID `6332706100110010800000aa00389b71` at payload offset [24..40]; ISOBMFF `uuid` box extended type `D8FEC3D6-1B0E-483C-9297-582887EC4881`) AND a content sniff (`kAiSubstrings`, case-insensitive) over APP1 XMP / APP13 IPTC / `tEXt`/`iTXt` values. So a manifest under a variant label is still caught by the content sniff. The constants were cross-verified against the C2PA 2.4 spec, the `contentauth/c2pa-rs` reference impl, and a real C2PA fixture hexdump.
- **Detection-removal parity invariant:** `provenance_constants.hpp` is the SINGLE shared header used by both the reporter (`scan_*`) and the remover (`rewrite_*_strip_ai`), so detection and removal cannot drift. The parity unit test asserts the remover drops exactly what the reporter flagged. Keep new keys/substrings in the shared header only, never duplicated in the `.cpp`.
- **JPEG strip drops ALL APP11 (0xEB)** unconditionally (rare; only JUMBF/C2PA/JPEG-XS use it), so the envelope segment layout matters only for the reporter's `has_c2pa` flag. From the first SOS (0xDA) / EOI (0xD9) the remainder is copied verbatim (covers progressive/multi-scan).
- **zTXt gap (Phase 2):** v1 matches `zTXt` by KEY only (no zlib inflate). A `zTXt parameters` is dropped by key; a `zTXt` with a benign key but an AI value is NOT scanned (kept). Phase 2 adds inflate (link zlib via `find_package(ZLIB REQUIRED)`; it is transitively present via libopencv_imgcodecs but NOT directly linked to `build/wmr` today). EXIF surgical scrub is also Phase 2: v1 drops an APP1-EXIF carrying an AI marker WHOLE (no IFD rewrite, the landmine), and keeps standard non-AI EXIF verbatim.
- **The remove-path strip is a defensive no-op today (DECISION A):** `wmr remove`/`synthid` strip provenance from the output by default via `post_write_provenance_strip` (`provenance.hpp`), hooked at the two `cv::imwrite` sites (`cli_app.cpp` ~`:394` single image, `batch_processor.cpp` ~`:193` batch), guarded by `!opts.keep_provenance`. But `cv::imwrite` ALREADY drops ALL container metadata on write and injects nothing (verified at the byte level on the real binary: a PNG output is exactly `IHDR + IDAT(s) + IEND`, a JPEG output is a default APP0/JFIF + DQT/SOF0/DHT/SOS). So the post-write scan finds nothing and the file is NOT rewritten (true no-op, temp-file+rename only fires when there is something to strip). Its value is the GUARANTEE independent of encoder. The ACTIVE, load-bearing path is the standalone `wmr metadata`, which losslessly strips USER input files that DO carry metadata. Do NOT claim the remove-path strip does active forensic work today.
- **`--keep-provenance`** (remove/synthid/video): opt-OUT of the default strip. Default is to strip (`keep_provenance = false`). `provenance_strip_help_text()` is an inline function (mirrors `synthid_attack_help_text()`) so the wording test asserts on it without linking `cli_app.cpp`.
- **v1 video stance:** `wmr video` is a full FFmpeg re-encode (`video_writer.cpp`: `avcodec_send_frame` + a fresh `AVFormatContext` + `avformat_write_header`; NO stream copy / remux). A C2PA `uuid`/`jumb` box in the input is NOT carried into the output by construction (FFmpeg writes a fresh moov/mdat). So the video output is already free of input C2PA. v1 does NOT run an explicit ISOBMFF strip and makes NO byte-level guarantee (Phase 4 offset-preserving retype). FFmpeg may write its own `udta` ("encoder: Lavf") tag; that is encoder metadata, not C2PA/AI provenance, and is out of scope. `--keep-provenance` on `video` is a documented no-op for CLI symmetry.
- **CMake pattern (ungated SOURCES + LIB_SOURCES, `paths.cpp` precedent):** the three `.cpp` go in the top-level `CMakeLists.txt` `SOURCES` AND the test `LIB_SOURCES` (ungated, so the lean/AI-off build compiles them unchanged), and are NOT repeated in any `target_sources(wmr_tests PRIVATE ...)` mirror block (that is for feature-gated sources and would be a no-op since `wmr_tests` already exists). The header-only `provenance_constants.hpp` needs no listing. v1 needs no `find_package` change (no zlib, no new deps). Spec: `docs/research/provenance-metadata-strip-plan.md`.

## Key Conventions

- Alpha maps are constexpr PNG byte arrays decoded at runtime via `cv::imdecode`
- **Validate AI image ops on COLORFUL natural content with a tight diff gate**, not gray/synthetic fixtures. A uniform color shift or channel swap is invisible on near-gray (R~=G~=B) images: it hid the Phase-2 Metal collapse AND the Phase-3 planar-layout + fp16-VAE bugs until a colorful poster was tested. The env-gated `WMR_COREML_SD_DIAG_IMAGE` natural-image diagnostic on the CoreML smoke test exists for this.
- **Debugging a removal residual:** the reverse-blend reproduces bit-exact in Python
  (`out = (syn - a*255)/(1-a)` with the mask PNG from `assets/watermark-masks/` placed at the
  detected integer region; max abs diff 0 vs `build/wmr`). Measure the residual against a
  content-fill proxy (the MI-GAN output, or `cv2.inpaint` TELEA of the mark footprint) and
  localize the true mark with a median-background-subtracted NCC. A residual shaped like the
  mark means wrong position or strength, not a wrong mask (the masks are faithful: they match
  their source captures; a stronger community mask over-removes).
- Capturing a watermark alpha from a real image: crop at the mark's TRUE top-left so the
  full mark fills the crop (a 2px clip reads as a reversal border); do NOT median-denoise
  (it erodes the faint outer edge); a pure-black background makes
  `correct_alpha_for_background` a no-op so you get the true alpha. Intensity is
  source-dependent (PlayerYK `bg_48` peaks 0.506, our 3.6 images ~0.31: same shape,
  different strength, so it would over-remove). Never resize an alpha between sizes
  (48/96); use a native capture per size, since resizing smears the anti-aliased edges
  the exact reverse-blend depends on (see also the video path's native-size templates).
  A capture from a real Gemini image bakes in its **SynthID carrier**, but the practical
  effect on the extracted alpha is small: across 10 distinct black-background generations
  the per-pixel std in the mark region was only ~0.025/255 and averaging moved the alpha
  ~0.0001. So averaging buys shot-noise reduction, not a SynthID fix, and only helps when
  the captures differ in the mark region (identical black content keeps the carrier
  correlated). SynthID-strip the capture first only if you need a truly pristine alpha.
- Measuring an alpha from a real capture on non-uniform content: the diamond is inscribed
  in its 48/96 box, so the box-corner pixels are clean background. Bilinear-interpolate the
  4 corners for a per-pixel bg estimate, then alpha = (patch - bg)/(255 - bg); median over
  many frames suppresses the varying content.
- Validating a measured alpha by removal: never score it on the same frames you derived it
  from (a median-derived alpha reads ~0 residual on its own frames by construction, since
  the median of deviations from the median is 0). Derive from one half of the frames, test
  on the other, and prefer dark/smooth-background frames, since bg-estimation noise can
  mask small alpha differences.
- Still-image watermark geometry is profile-aware (`WatermarkVariant::V1`/`V2`, default V2 with auto V2→V1 fallback; `--legacy` pins V1): V1 (legacy, pre-3.5) → 48×48 if either dim ≤ 1024 else 96×96, margins {32,32}/{64,64}; V2 (Gemini 3.5+) → large 96×96 @192px, small 36×36 with aspect-aware margin (`v2_small_config_from_dims`) + the content-suppressed snap (see "Snap generalization" under Still-image Geometry; an explicit `--rect`/`--geo-preset` disables it). `WatermarkSize` (Small/Large) is a size class, not a pixel count (V2 Small = 36px alpha). Still `WatermarkVariant` is distinct from video `VideoVariant`.
- Video encoding defaults: libx264, CRF 14, High profile, slow preset
- Test executable re-compiles library sources but does NOT link the main binary / `cli_app.cpp` (it is not in `tests/CMakeLists.txt`). So `run_cli`, CLI dispatch, and control-flow changes are NOT covered by `ctest` — verify with manual invocations (`wmr --version`, `wmr` no-args, `wmr <subcmd> <fixture>`, a bad flag). New library sources go in both `CMakeLists.txt` and `tests/CMakeLists.txt`. For position/detection changes, a library-level round-trip regression test forward-blends the alpha at a known position (`add_watermark_alpha_blend`) and asserts `detect_watermark` recovers it exactly; see "V2 small snap recovers the exact mark position on busy content" in `tests/integration/visible_pipeline_test.cpp`.
- Feature-gated sources go in a `target_sources(wmr_tests PRIVATE ...)` mirror block under `if(WMR_BUILD_<FEATURE>)` (mirror `WMR_BUILD_AI_DENOISE`), NOT the top `TEST_SOURCES`/`LIB_SOURCES` lists (`wmr_tests` is already created, so list mutation is a no-op). An ungated source shared by a feature and the tests (e.g. `src/core/paths.cpp`) goes in the top-level `SOURCES` AND the test `LIB_SOURCES`, and must NOT be re-listed in a feature mirror block (duplicate-source error when two features are ON).
- `wmr --version` is the `APP_VERSION` define (`=project(wmr VERSION …)`) baked at CMake **configure** time (cached as `CMAKE_PROJECT_VERSION`). Editing the version doesn't change `build/wmr` until a reconfigure, `cmake --build build` reconfigures automatically when `CMakeLists.txt` changed.
- Integrating an ONNX inpainter: **probe its IO empirically** (input/output dtype + range, mask polarity, e.g. MI-GAN is uint8 RGB + mask 0=hole; Carve/LaMa-ONNX *outputs* [0,255], not [0,1]); don't assume. Then **verify the actual output is a valid image** (per-channel mean ≈ the original scene, not ~0/~255 saturated), a white/black output = a scale/polarity bug. Don't trust VLM quality judgments on small/upscaled crops; verify objectively (pixel mean, tensor diff vs a known-good reference).
- Before commit/merge: stage the intended files by EXPLICIT path, then `git status`. Never `git add -A`/`git add .` — this repo carries persistent untracked reference/test data that must not be staged (`reference-images/`, `reference-videos/`, `test-images/poster-artnight.png`, `.playwright-cli/`, `images/`). The test exe and dev `build/` compile from the **working tree**, which masks un-staged changes (the MI-GAN swap once shipped a commit with only the *new* files staged, missing modifications/deletions to existing source — caught only by `git status` pre-merge).
- Verify a feature's OFF build is clean: `cmake --preset mac-homebrew-Debug -DWMR_<FEATURE>=OFF -B build-off && cmake --build build-off --target wmr`, then `nm build-off/wmr | grep <feature>` (symbols) and `otool -L build-off/wmr | grep <dep>` (e.g. curl). `mac-homebrew-Debug` exists alongside `-Release`; `build-*/` is gitignored. A dev-mac build has every feature ON, so a file included inside one feature's guard (e.g. the `WMR_BUILD_AI_COREML_SD && __APPLE__` block in cli_app.cpp) but used under another (`WMR_BUILD_REGEN`) compiles locally and only breaks the CoreML-SD-OFF CI legs (cost the first v1.16.11 tag run); the catching config is `-DWMR_BUILD_REGEN=ON -DWMR_BUILD_AI_COREML_SD=OFF`.
- Removing an unreleased feature (never pushed/tagged): if origin hasn't moved, `git branch <name> HEAD` (safety), reset main to the pre-feature commit, and re-land ONE archive commit — new files copied into a standalone folder plus `git diff <pre-feature> <name> -- <shared files>` saved as a wiring patch. Then delete the branch: the folder is self-contained and no stale ref lingers (surgical un-picking of interleaved fix commits is strictly worse).
- Closed-program archives live in `research/<topic>/` (self-contained: README with verdict + verification oracle, sources, tests, wiring patch; see `research/antidetect/`), separate from `docs/research/` which holds the ACTIVE product research records. Do not mix them.
- License/redistribution suitability is the project owner's call, not a hard pre-filter. When evaluating dependencies, rank on technical merit and report license facts separately; don't silently exclude GPL/CC-NC options before the owner decides.

## Platform Quirks

- CMakePresets.json is macOS-only (arm64, despite "x64" naming). Linux/Windows use manual cmake invocation.
- FFmpeg found via custom `cmake/FindFFMPEG.cmake` (pkg-config primary, `FFMPEG_ROOT` fallback). Creates imported targets `FFMPEG::avformat` etc.
- Linux links static libgcc/libstdc++; MSVC uses static CRT.
- **MSVC `/bigobj` for sdcpp (C1128):** `external/stable-diffusion.cpp/src/stable-diffusion.cpp` is a very large TU that exceeds MSVC's COFF section count (`fatal error C1128: number of sections exceeded object file format limit`); the `WMR_BUILD_REGEN` block adds `/bigobj` to `CMAKE_CXX_FLAGS`/`CMAKE_C_FLAGS` under MSVC (before `add_subdirectory`) so ggml + sdcpp inherit it. Re-check if a new large sdcpp TU trips C1128.
- **Local build is DYNAMIC; CI is STATIC.** The Homebrew `build/wmr` links OpenCV/FFmpeg/fmt/spdlog dynamically (~10 MB); CI's vcpkg build is fully static (lean release binaries are ~29 MB single self-contained files, `otool -L` shows only system frameworks). Don't judge CI portability from the local binary, inspect the downloaded release binary (`gh release download`). To hand-test on another Mac: `~/github/vcpkg` exists (bootstrapped 2026-08-27, warm arm64-osx cache); configure `build-static` with the CI mac-arm64 flags + that toolchain, then `scripts/bundle_macos_vulkan.sh build-static/wmr dist/<name>` + copy the MI-GAN `.mlpackage` next to it. Copy the whole folder; `build-static/wmr` alone still has a hard libvulkan load command.
- macOS runners and the local Mac are BSD, not GNU: `base64` decodes with `-D` and encodes with `-i` (no `--decode`/`-w0`); `xargs` has no `-r`/`--no-run-if-empty` (use `find -exec`); `sed -i` needs `''`; there is no `timeout` command; `ps` has no `--sort` (pipe through `sort -rk4`). zsh does NOT word-split unquoted vars: `R="-R slug"; gh ... $R` passes one garbage argument (inline the flags instead). Ad-hoc OpenCV spikes: headers live under `$(brew --prefix opencv)/include/opencv5` (OpenCV 5), not `include/opencv2` — compile with `-I$(brew --prefix opencv)/include/opencv5 -lopencv_core -lopencv_imgproc`.
- **Background Bash tasks can launch from a stale session cwd** (a foreground `cd` does not reliably propagate to them, and the foreground cwd persists across tool calls, silently relocating later commands): use ABSOLUTE paths for the binary, input, and output in every `run_in_background` command. Related: a pipe masks exit codes (`cmd | grep` reports grep's status) — when verifying exit codes, redirect to a file and check `$?` unwrapped (or use zsh `$pipestatus`).
- Extract one video frame by index: `ffmpeg -i in.mp4 -vf "select='eq(n,N)'" -vsync 0 -frames:v 1 out.png`, with N a plain integer (not zero-padded; zsh rejects the `10#$n` base-prefix trick). Every Kth frame: `select='not(mod(n,K))'`.
- GitHub macOS runners use a paravirtualized Metal GPU (`AppleParavirtDevice`) that throws during MoltenVK `vkCreateInstance` (`newArgumentEncoderWithLayout:`). The GPU path can't run in CI, the `ai-denoise` job verifies CPU only (`VK_ICD_FILENAMES=/nonexistent`); verify GPU out-of-band on real Apple Silicon (on the dev Mac: `wmr remove <img> --denoise ai` and confirm the log shows `Vulkan GPU`, not the CPU fallback).
- CMake post-build model copies (NCNN `ai_denoise_model`, MI-GAN `migan_pipeline_v2.onnx`) are guarded by `if(EXISTS assets/…)` evaluated at **configure time**. Adding/renaming a model under `assets/` after configuring won't copy it next to the binary → silent "model not found" → runtime fallback (NS). Reconfigure (or touch `CMakeLists.txt`) after changing a model asset. The CoreML `.mlpackage` is a **directory**, use `copy_directory` (not `copy_if_different`), `cp -R` in the bundle step, and Git LFS can't directory-glob so the weight blob needs a full-path rule (`assets/migan_512_places2_fp16.mlpackage/Data/com.apple.CoreML/weights/weight.bin filter=lfs`).
- MI-GAN model assets are Git LFS: `assets/migan_pipeline_v2.onnx` (linux/windows) and `assets/migan_512_places2_fp16.mlpackage/Data/com.apple.CoreML/weights/weight.bin` (mac). A plain `git clone` leaves LFS pointer files, so a `WMR_AI_MIGAN=1` build silently fails to find the model and falls back to NS. Run `git lfs install && git lfs pull` after cloning (CI uses `lfs: enable_migan` on checkout).
- **OBJCXX must be enabled once, up front, `if(APPLE)`** (right after `project(... LANGUAGES CXX)`). NOT in `project(LANGUAGES)` (linux/windows have no `cc1objplus` → configure fails), and NOT via a late `enable_language(OBJCXX)` after `add_subdirectory(external/stable-diffusion.cpp)` (the submodule's `project()` resets it → `CMAKE_OBJCXX_COMPILE_OBJECT not set`). The MI-GAN block's `enable_language(OBJCXX)` only works because it runs before sdcpp.
- **Dev Mac is 16 GB unified memory.** Big diffusers models (Flux/Chroma/SD3.5/Qwen, ~12-20 GB) jetsam-kill or need GGUF Q4/Q5 + `enable_model_cpu_offload` (`enable_sequential_cpu_offload` raises `KeyError(None)` on quantized tensors); only ~7 GB models (SDXL-family) run clean. Expect slow inference + OOM-kills for larger models; test those on a bigger box. Before killing a heavy agent-run Python on RAM grounds, sample its RSS twice ~20 s apart: a stable high RSS is fine, only a growth rate that crosses the remaining ETA predicts an OOM (and design eval harnesses to resume, so a kill is cheap).
- **Gated HF repos:** the cached token (`~/.cache/huggingface/token`, user froggeric) is valid, but gated repos (e.g. FLUX.1-schnell, SD3.5) still 403 until you accept their license on the HF page in that account. `qwen/Qwen-Image` and `Tongyi-MAI/Z-Image-Turbo` are non-gated.

## CI & Releases

- `.github/workflows/release.yml`: the `build` matrix (4 legs: mac arm64 native + bundled Vulkan/MoltenVK + CoreML `.mlpackage`; mac x86_64 cross-compiled + CoreML `.mlpackage` (zip); linux; windows, all `WMR_BUILD_AI_MIGAN=ON` + `WMR_BUILD_AI_DENOISE=ON`; on mac `WMR_BUILD_AI_MIGAN` routes to CoreML, elsewhere ORT) + a `tests` job (AI+TESTS ON, ubuntu/ORT); `release` (`needs: [build, tests]`, `if: v*` tag) attaches the 4 packages (mac arm64 zip, mac x86_64 zip, linux tarball, windows zip) + `LICENSE-THIRD-PARTY.md`. **Validate a changed job off-cycle via `workflow_dispatch` before tagging**, avoids tag-force-move churn on failure.
- **Cutting a release:** bump `project(wmr VERSION X.Y.Z)` in CMakeLists.txt + move the CHANGELOG `[Unreleased]` block under `## [X.Y.Z] - YYYY-MM-DD` (add a fresh empty `[Unreleased]`), commit `release: vX.Y.Z`, rebuild so `wmr --version` shows X.Y.Z, then `git tag vX.Y.Z && git push origin main && git push origin vX.Y.Z`. The tag push fires release.yml; `generate_release_notes: true` auto-writes the notes from commits (edit on the release page for a curated blurb). If a CoreML model was re-pinned (`kSha256*` changed in `coreml_sd_model_fetch.cpp`), ensure the new tar.gz is uploaded to HF (`hf upload froggeric/wmr <file> <path-in-repo>`) BEFORE tagging. A missed upload breaks fresh-install regen for the release (the 1.16.3-1.16.5 gap; the binary pins the new SHA but HF still serves the old bytes). Note: `generate_release_notes: true` overwrites a hand-published body when the release job runs (after the builds), so publish curated notes with `gh release create vX.Y.Z --notes-file` right after tagging and re-apply with `gh release edit vX.Y.Z --notes-file <f>` once the run completes (a watcher cron polling `gh run view --json status,conclusion` that re-applies on completion works well). Binaries attach only after the builds; Windows is the long pole. Release titles AND body H1s are `vX.Y.Z` (optionally `vX.Y.Z: subtitle`), never "wmr X.Y.Z". A failed tag run recovers as: fix on main, validate the failing shape with an `only_leg` dispatch, `gh release delete vX && git push origin :refs/tags/vX`, re-tag the fix commit, recreate the release with the curated notes, re-arm the watcher.
- **sdcpp (regen) CI gaps — RESOLVED in 1.16.2.** The v1.15.0 failures (linux `Could NOT find Vulkan (missing: Vulkan_LIBRARY glslc)`; mac x86_64 `unknown target CPU 'apple-m1'`) are fixed: linux/windows/mac-x86_64/tests build regen CPU-only (`WMR_BUILD_REGEN=ON -DWMR_REGEN_CPU_ONLY=ON`), and the CMake cross-arch guard disables ggml's GGML_NATIVE host-cpu probe on the x86_64 cross-build. mac arm64 keeps `WMR_BUILD_REGEN=ON` + `WMR_BUILD_AI_COREML_SD=ON` (Metal + CoreML). A GPU backend for faster regen on linux/windows (Vulkan) is a follow-up; the release binaries ship CPU-only there.
- **One-leg CI iteration:** `release.yml` has an `only_leg` workflow_dispatch input (all/macos-arm64/macos-x86_64/linux/windows) + a `select` job that emits a dynamic build matrix via `fromJSON(needs.select.outputs.matrix)`; a single-leg dispatch builds only that leg and skips `tests`, so a windows/linux fix can iterate without rebuilding the other platforms. Fire with `gh workflow run release.yml --ref main -f only_leg=<leg>`. A tag push / unfiltered dispatch still builds all 4 (validated: `only_leg=linux` built only linux, skipped tests + release).
- **macOS packages are Developer ID signed + notarized** (`scripts/sign_and_notarize_macos.sh`): every dylib + Mach-O is signed with the Developer ID Application identity + hardened runtime, the zip is notarized via an App Store Connect API key (`.p8`), best-effort stapled. Distribution is `.zip` (a `.tar.gz` would drop the notarization ticket's xattrs). Gated on the `MACOS_CERTIFICATE` secret; if absent the build ships ad-hoc zips (never fails the release). Validated: notary service `Accepted`, `codesign --verify --strict` clean, and the GPU (MoltenVK/NCNN) path runs under hardened runtime on Apple Silicon with NO entitlements (empty `scripts/wmr.entitlements`; `bundle_macos_vulkan.sh`'s ad-hoc re-sign is only an intermediate step before the real identity is applied). Secrets: `MACOS_CERTIFICATE` (base64 `.p12`), `MACOS_CERTIFICATE_PASSWORD`, `MACOS_TEAM_ID`, `APP_STORE_CONNECT_{KEY_ID,ISSUER_ID,API_KEY}` (base64 `.p8`). **Signing gotchas (each cost one CI round):** the `secrets` context is NOT valid in a step `if:` (gate on a workflow-level `env.HAS_CERT` boolean instead); `! pipe | grep` is unsafe under `set -euo pipefail` (use an explicit `if … exit 1`); `codesign -d` writes its diagnostic to **stderr** and at `-dv` omits the `Authority=` lines, so assert with `codesign -dvvvv … 2>&1 | grep "Authority=Developer ID Application"`; `stapler` cannot staple a loose-file zip (the online Gatekeeper check still clears it for connected users).
- `gh run view --log` returns empty until the *whole run* completes (per-job logs too). To read a finished job's log fast, `gh run cancel` the run (preserves completed jobs' logs), then read, or wait for completion. **Don't monitor a run with `gh run watch --exit-status` in a piped background command**, the pipeline masks gh's exit code (always 0) and gh can drop early on an API hiccup → a false "completed". Poll `gh run view <id> --json status,conclusion` until `status=="completed"`; `conclusion` is the source of truth. (Windows is the ~2 h long pole, every release, even mac-only, waits on it.) For a long watch, schedule a recurring `CronCreate` (~every 15 min) whose prompt does the poll and `CronDelete`s itself on completion or on any job failure, instead of blocking a turn.
- A job that already finished is readable MID-RUN (no need to cancel jobs still in flight): `gh api repos/<owner>/<repo>/actions/jobs/<job-id>/logs` returns the log as text (not a zip). Get the id and failed step with `gh run view <run> --json jobs -q '.jobs[] | select(.name|test("PATTERN")) | .databaseId'` and `… | .steps[] | select(.conclusion=="failure") | .name'`. `gh run view --job <id> --log` stays empty until the whole run completes; only the API works mid-run.
