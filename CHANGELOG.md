# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Gemini 3.6 48px watermark: cleaner still + video removal

- **Averaged still alpha.** `v2_diamond_48_still` is now the average of 10 distinct Gemini 3.6
  generations (each with the mark on a near-uniform-black patch), not a single capture,
  suppressing per-pixel capture noise ~3x. Still removal stays exact (the SynthID carrier in
  the mark region is only ~0.025/255 and content-correlated, so averaging buys a real but
  small shot-noise reduction).
- **Video uses the still alpha.** The 48px Gemini video mark is now removed with this same
  clean still alpha instead of the video-path 48px capture, which was slightly
  under-calibrated (non-black capture background + approximate correction). The still alpha
  is the single clean source for both still and video 48px removal.
- **Video diamond edge cleanup (default-on).** After the reverse-blend, the Gemini/Veo video
  path repairs the faint border/halo at the diamond's edge with `inpaint_diamond_edges` (the
  validated "U4" recipe: a TELEA inpaint confined to a thin boundary ring, residual-gated so a
  clean reversal is untouched). It only touches the edge ring; the recovered interior and the
  untouched exterior stay byte-for-byte intact. `--no-edge-cleanup` disables it for a pure
  reverse-blend. It is a safe no-op when the mark sits on a clean background (fires only when
  there is a real edge residual, e.g. busy/compressed footage).
- **Gemini 3.6 video watermark magnitude corrected.** A pristine measurement across 3 clean
  1280x720 videos (360 frames, mark on a uniform dark patch) puts the video alpha at peak
  0.304 / center 0.301, the same as the still alpha, not "stronger" as an earlier busy-fixture
  reading suggested. So the still alpha (used for video since the averaged-alpha change) is the
  correct magnitude for both. 1280x720 auto-detects cleanly (margin 96,96, 48px, score 1.00).

## [1.14.1] - 2026-07-23

### Still-image watermark auto-geometry + exact removal

First tool to auto-detect and exactly remove Gemini 3.6 Flash's visible watermark.
Supersedes the intermediate 1.13.0 / 1.14.0 releases (yanked). Cumulative change since
1.12.0:

- **Auto-detect the watermark position and size.** A polarity-invariant NCC template
  match, anchored on the predicted position then widened to the bottom-right corner,
  finds the mark and its size (36px Gemini 3.5, 48px Gemini 3.6, 96px large) with no
  manual geometry: `wmr remove` just works. `--rect`/`--geo-preset` force a position for
  faint marks; `--no-auto-geometry` opts out.
- **Exact reverse-alpha-blend by default.** `original = (watermarked − α·logo)/(1−α)`,
  the mathematical inversion, with no blur. The old Gaussian "cleanup" was inpainting
  that left a "blurred translucent area"; it is now opt-in (`--denoise`) and
  **residual-only** (it predicts clean content with cv::inpaint and blends it in only
  where the reversal left a real residual, never blurring a clean removal).
- **Dedicated Gemini 3.6 still alpha.** The 3.6 small diamond is 48px and rendered
  weaker (~0.31) than the 48px video diamond (~0.40); a capture from a real 3.6 image
  (at the mark's true top-left, no median, on a pure-black patch so decontamination
  yields the true alpha) gives an exact reversal. Templates are native per size and
  never resized (intensity is source-dependent; resizing smears the edges).
- New still flags on `remove`/`visible`/`detect`: `--rect`, `--geo-preset`,
  `--no-auto-geometry`; `--denoise {off|soft|ns|telea}` (+`ai` on AI builds), default
  `off`.

## [1.14.0] - 2026-07-23

### Changed: visible removal defaults to the exact reverse-blend (no blur)

The visible-watermark removal is now a pure reverse-alpha-blend by default, the
mathematically exact inversion. Previously a Gaussian "residual cleanup" ran after
it by default, which was inpainting (a blur-fill of the region with its surroundings)
and left a visible "blurred translucent area," especially on textured backgrounds.
Measured on a textured test image: the cleaned patch now matches the surrounding
texture (std 62.9 vs band 64.5) with zero change outside the watermark box.

- `--denoise` is now always available (not just AI builds); default is `off`.
- The opt-in cleanup (`--denoise ns|telea`, or `ai` on AI builds) is **residual-only**:
  it predicts the clean content with cv::inpaint and blends it in ONLY where the
  reverse-blend left a real residual, so a clean removal is left byte-for-byte intact
  (no whole-region blur).

### Added: dedicated Gemini 3.6 still alpha (exact reversal)

The 3.6 small diamond is removed with a dedicated 48px still alpha captured from a
real 3.6 image, instead of the 48px *video* alpha (which is stronger and over-removed
stills). On a uniform-black test image the new alpha reverses the mark near-exactly
(residual-diamond correlation 0.10 vs 0.33 with the video alpha; detection spatial
NCC 1.000). The engine applies its existing background decontamination, so the
captured alpha yields the watermark's true alpha.

## [1.13.1] - 2026-07-23

### Fixed: Gemini 3.6 small watermark is 48px, not 36px

1.13.0 shipped the still auto-geometry search but removed with the wrong mask:
Gemini 3.6 Flash's small diamond is **48px** (the same asset the video path uses for
small watermarks), not the 36px Gemini 3.5 still alpha. On a uniform background the
36px alpha under-covered the 48px mark, leaving a faint diamond ghost. Measured on
`test4`: the 48px alpha matches the mark at NCC 0.997 (vs 0.685 for 36px) and the
reverse-blend residual drops ~74%; after the fix the cleaned patch is statistically
identical to the surrounding background.

- The search now tries **both** 36px (Gemini 3.5) and 48px (Gemini 3.6) diamond
  templates and uses whichever fits, so removal runs with the matched-size alpha.
- A manual override (`--rect` / `--geo-preset`) also uses the correct size: a 48px box
  or the `gemini36-portrait` preset drives the 48px alpha (the previous default-36px
  was the bug for forced removal too).
- The calibrated preset `gemini36-portrait` is now 48px @ margin (94,96) for 896x1200.

### Fixed: `--geo-preset` now lists and validates the presets

`--geo-preset` uses an IsMember validator, so the help text shows the valid names
(`gemini36-portrait`) and an unknown name is rejected with the list, instead of being
silently ignored. `detect` reports the matched region size (e.g. 48x48), so the
resolved geometry is visible, not just the V1/V2 profile.

## [1.13.0] - 2026-07-23

### Still images: automatic watermark geometry (Gemini 3.6 Flash)

Gemini 3.6 Flash places its small 36px diamond watermark bottom-right as before,
but at a margin the position model no longer predicts. For an 896x1200 image the
model predicts margin 84; the real Gemini 3.6 margin is ~100. The ~16px gap exceeds
the old ±3px refinement, so even a clearly visible mark was reported "not detected"
(a positional bug, not a faintness one). Still images now content-detect the
position like the video path (shipped 1.12.0), adapted for the single-image reality.

- The search is **anchored** on the model-predicted position first (±40px, where the
  model is usually within ~20px), then **widened** to the bottom-right corner if the
  anchored hit is not trusted. A blind full-corner scan is unsafe on a single busy
  still (no temporal consistency to separate a faint mark from corner content).
- Regression guard: a snapped (on-table) hit is trusted at the 0.45 floor; a raw
  off-table hit must clear 0.60 before it overrides the model. V1 and V2-large keep
  today's exact model (the search runs only for V2 small).
- New flags on `remove` / `visible` / `detect`: `--rect x,y,w,h` (manual override),
  `--geo-preset <name>` (named geometry), `--no-auto-geometry` (use the model). An
  explicit `--rect`/`--geo-preset` forces removal at that position even when the
  detector's confidence is too low to confirm a faint mark. Precedence:
  `--rect` > `--geo-preset` > auto-detect > model.
- First calibrated preset: `gemini36-portrait` (896x1200, margin 100). The table
  grows as more resolutions are measured; the model stays the fallback.
- Pure OpenCV logic in a new `still_geometry` unit, unit-tested without FFmpeg.
- Note: the geometry search is polarity-invariant but the downstream NCC fusion is
  max-only, so a dark-on-bright mark it localizes can still be rejected; use
  `--rect`/`--geo-preset` then.

## [1.12.0] - 2026-07-14

### Video: automatic watermark geometry detection

The Gemini/Veo video path now detects the watermark's corner position and logo
size from the video content, so `--variant` is no longer required. The search
samples frames, runs a polarity-invariant template match against the real alpha
assets (48/96 px diamond, 68x30/99x43 Veo text) in the bottom-right corner, then
snaps the result to the known geometry table. `Auto` (the default) uses this
search and falls back to the resolution-based guess.

- `--variant {720p-1|720p-2|1080p}` still forces a geometry and disables the
  search; `--no-auto-geometry` opts out and uses the resolution guess.
- `--rect x,y,w,h` is now available on the standard Gemini/Veo path too (it was
  NotebookLM-only). Precedence: `--rect` > auto-detect > `--variant` > resolution.
- A regression guard keeps a video that already works unchanged: a snapped
  (on-table) detection is trusted, but a raw off-table detection must score at
  least 0.60 before it overrides the resolution guess.
- Detection is pure OpenCV logic in a new `geometry_detector` unit, unit-tested
  without FFmpeg (it recovers 720p-1, 720p-2 via position snap, 1080p, and is
  polarity-invariant).

## [1.11.0] - 2026-07-14

### macOS: Developer ID signing + notarization

The macOS arm64 and x86_64 packages are now **Developer ID signed + notarized**.
Gatekeeper allows them on first launch (a one-time online check) with no
`xattr -dr com.apple.quarantine` step. Distribution format changed `.tar.gz` to
`.zip` (tar.gz discards the notarization ticket's extended attributes). Run
commands in the README download table updated accordingly.

- CI imports a "Developer ID Application" cert from a GitHub secret, signs every
  dylib (libvulkan, MoltenVK) and the Mach-O executables with hardened runtime,
  then submits the zip to Apple's notary service via an App Store Connect API key.
- Signing is gated on the `MACOS_CERTIFICATE` secret: if absent, the build ships
  the old ad-hoc/unsigned artifacts in a `.zip` (never crashes the release).
- Linux and Windows are unchanged.

## [1.10.1] - 2026-07-13

### NotebookLM: MI-GAN-everywhere on Apple Silicon + `--notebooklm-method` toggle

On Apple Silicon (native arm64), MI-GAN now runs on **every** NotebookLM scene by
default: the complexity gate (NS for uniform scenes) is skipped, since CoreML on
the Neural Engine makes MI-GAN fast enough (~28 ms/frame) that the NS speed
optimization is moot there. A/B-verified (26 uniform frames, MI-GAN-vs-NS mark-ROI
delta 0.49/255 ≪ 3.0/255; no hallucinated texture on flat backgrounds). The
pipeline is encode-bound (x264 slow/CRF 14), so MI-GAN-everywhere costs only ~+20 s
on a 5-min video.

- **`--notebooklm-method {auto|ns|migan}`** (re-adds the flag 1.9.0 removed, now as
  an *override*): `auto` = the platform default (MI-GAN-everywhere on Apple Silicon,
  complexity-gated elsewhere); `ns`/`migan` force one. NS remains the fallback when
  MI-GAN is unavailable (not built, or model failed to load).
- **Unchanged elsewhere**: Apple Intel (x86_64, no Neural Engine, incl. a Rosetta-
  translated arm64 binary) + Linux/Windows (ORT, ~225 ms/frame) keep the complexity
  gate (NS for uniform, MI-GAN for intricate).
- **Behavior change** (Apple Silicon only): `--notebooklm` default shifts from
  NS+MI-GAN mix → MI-GAN-everywhere. Use `--notebooklm-method ns` to restore the
  old per-scene NS-on-uniform behavior.
- The complexity-analysis pass (a full sequential decode + Sobel per scene) is now
  skipped when it won't be consulted (forced `ns`/`migan`, or `auto` on Apple
  Silicon), a minor speedup.
- Routing lives in the pure, arch-agnostic `resolve_inpaint_method(complexity,
  threshold, has_migan, requested, platform_default)` (the arch decision is in the
  caller, so `notebooklm_gates_test` stays arch-independent).

## [1.10.0] - 2026-07-13

### NotebookLM: native CoreML MI-GAN on macOS (~11× faster)

On macOS, the MI-GAN inpainter now runs as a **native CoreML fp16 model on the
Neural Engine** — **~28 ms/frame** (vs ~225 ms on ORT-CPU; ~11× faster), A/B-
verified to match the ORT baseline within Δ1.9/255 on 12 NotebookLM scenes. This
**replaces ONNX Runtime entirely on macOS** (CoreML is a system framework — no
vendored lib; the 27 MB `.onnx` + ORT dylib are dropped from the mac packages,
replaced by a 14 MB `.mlpackage`). The same `.mlpackage` is arch-neutral, so
**Intel Macs now get MI-GAN too** (1.9.0 left them MI-GAN-free). Linux/Windows are
unchanged (ORT-CPU).

- **Why not the 1.9.0 CoreML path:** that finding ("CoreML is slower, 602 ms") was
  ONNX Runtime's *CoreML execution provider* — its graph-partitioning overhead
  (only 375/559 nodes reached CoreML). A native `coremltools` mlprogram puts the
  whole graph in one MIL program and avoids the partition — the opposite result.
- **Impl:** `src/core/migan_coreml_inpainter.mm` (Objective-C++, `WMR_AI_MIGAN_COREML`)
  shares the `MiganInpainter` interface with the ORT `.cpp`; `video_processor.cpp`'s
  dispatch is byte-identical. CMake splits `if(WMR_BUILD_AI_MIGAN)` into `if(APPLE)`
  (CoreML + `.mm` + `.mlpackage`) / `else()` (ORT).
- **macOS 14+ required** (the `.mlpackage` targets `minimum_deployment_target=macOS14`).
- **Release shape:** mac arm64 + x86_64 both ship the `.mlpackage` (x86_64 is now a
  tarball, was a bare binary); linux/windows unchanged.

## [1.9.0] - 2026-07-12

### NotebookLM: MI-GAN replaces FSR + LaMa as the default inpainter

MI-GAN (Picsart AI Research, ICCV 2023, **MIT**) replaces both FSR and LaMa as the NotebookLM intricate-scene inpainter, and it is now the **default** — there is no inpaint-method flag; the pipeline always uses **NS for uniform scenes + MI-GAN for intricate scenes** (complexity-gated via `--complexity-threshold`). MI-GAN was missed by the earlier "LaMa is the only viable" eval; a broad re-survey (MAT/FcF/ProPainter/Moebius/MxT/HINT/RETHINED/SD…) found everything else NC-licensed, no-ONNX, no-weights, diffusion-infeasible, or CUDA-only, and MI-GAN beats LaMa on every axis:

| | LaMa (removed) | MI-GAN |
|---|---|---|
| Speed (CPU) | ~1800 ms/frame | **~225 ms** (~8× faster) |
| Model size | 208 MB | **27 MB** (~7× smaller) |
| License | Apache-2.0 code + Places2 weights gray-area | **MIT (code + weights)** |
| Cartoon quality | soft (Places2-trained) | **sharp (GAN)** |

- **No method flag:** `--notebooklm-method` and `--lama-threshold` removed. Just `wmr video in.mp4 --notebooklm -o out.mp4` → NS + MI-GAN automatically.
- **MI-GAN integration:** `src/core/migan_inpainter.{hpp,cpp}` (`#ifdef WMR_AI_MIGAN`, leaked ORT singleton); feeds the whole frame (RGB **uint8** dynamic-res NCHW) + mask (0=hole/255=keep) → uint8 composited result. `--complexity-threshold` (default 15) gates NS↔MI-GAN. FSR + LaMa + the opencv_contrib xphoto dependency removed from the inpaint path (FSR was blurrier + green-artifacted on uniform scenes; LaMa was slower + soft + gray-license).
- **Detector fix:** the explainer-mode rect `(1105,660,131,16)` → `(1085,658,153,20)` — the old one started at the spiral logo's right edge, leaving ~18px of the logo unmasked (NS/FSR/LaMa hid it by blurring; MI-GAN's precise fill exposed it). Cinematic rect was already correct.
- **ONNX Runtime:** official 1.27.1 prebuilt per platform, fetched at CMake configure (SHA256-verified), `IMPORTED` target — zero CI build time (NOT the heavy vcpkg port). macOS Intel = MI-GAN-OFF (no osx-x86_64 ORT) → NS. GPU (CoreML) tested — *slower* via ORT (graph partitioning: 602 ms vs 225 ms CPU); a native coremltools build is a possible future macOS-only optimization.
- **Release shape:** macOS arm64 / Linux / Windows bundle the ORT lib + the 27 MB `migan_pipeline_v2.onnx` (Git LFS, MIT) — Linux/Windows are archives (`.tar.gz`/`.zip`), macOS arm64 a tarball, macOS Intel a single binary (MI-GAN-free). Packages are far smaller than the 1.8.0-plan's 200 MB LaMa.

## [1.7.1] - 2026-07-11

### NotebookLM: inpaint every scene (presence gate removed)

The per-scene presence gate (template-match the mark per scene, skip "absent" ones) false-negatived: on Arcade Anxiety, scenes 15 and 18 were skipped even though the watermark was visibly present, leaving it in the output. Measured with the real embedded template, a faint-but-**present** mark scores 0.34–0.43 — the **same band as a genuinely-absent scene** (0.37–0.42) — so no threshold can separate them.

- **Fix:** inpaint **every** scene. A false negative = a visible watermark (unacceptable); a false positive = inpainting an already-clean ~121×17 patch (imperceptible — FSR/NS just reconstructs the background). Verified on Arcade: 0 skipped, all 42 scenes inpainted; previously-skipped scenes 15 (clean) and 18 (soft on a complexity-125 cartoon — the documented Phase-B limit) now have the watermark removed.
- **Removed** the dead presence-gate code: `NotebookLMDetector::mark_present_in_scene`, `sample_scene_frames`, and the `notebooklm_presence_threshold` config field. The complexity gate still routes FSR vs NS.

### Consolidated release: cross-platform build fixes (first shippable 1.7.x)

1.7.0 consolidated the release into one self-contained package per platform (see below) but never shipped — the consolidated CI build needed four platform fixes, all landed here:

- **macOS x86_64:** install `nasm` (the cold `x64-osx` triplet builds `x264` from source) and build `WMR_NCNN_VULKAN=OFF` (on Apple, NCNN's simplevk does static Vulkan linkage and needs a build-time `libvulkan`, unavailable for x86_64 on the arm64 runner) → **Intel AI is CPU-only**; the arm64 build is the GPU one.
- **Windows:** the project `cmake` step now runs inside the MSVC dev env (`vswhere` + `vcvarsall.bat x64`) so it compiles with `cl.exe` to match the MSVC-built vcpkg deps (the bare step otherwise detected MinGW gcc → link failure), and `CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded` propagates `/MT` to the `ncnn` subproject to match the `x64-windows-static` deps.
- All four legs (mac arm64/x86_64, linux, windows) + the `tests` job verified green; Windows (~2 h) is the long pole, well under the 6 h cap.

## [1.7.0] - 2026-07-11

### NotebookLM FSR inpaint for intricate backgrounds (kills the NS diamond)

Phase B of the NotebookLM quality roadmap. v1.6.0's NS inpaint left a diamond/envelope PDE artifact on every non-uniform background (intensity propagates from the 4 mask edges, peaks on the diagonals). 1.7.0 routes intricate-background scenes to **FSR** (`cv::xphoto`), which blends cleanly.

- **Per-scene method routing** — `resolve_inpaint_method` (new, pure, unit-tested in `notebooklm_gates`): `auto` (new default) picks FSR for intricate scenes (complexity ≥ `--complexity-threshold`, default 15.0) and NS for uniform ones; `--notebooklm-method {ns|fsr}` forces a method. The v1.6.0 complexity gate was computed but **never wired to dispatch**; this builds the routing (per-scene resolved method threaded into the frame loop).
- **FSR, not NS, for texture** — A/B confirmed FSR removes the diamond and continues the surrounding texture; NS is local (a PDE method) and cannot. SHIFTMAP is dropped (copy-paste artifact).
- **`INPAINT_FSR_FAST`, not `_BEST`** — benchmarked on the ~131×16 mark crop, FSR_BEST is ~1.3 s/frame (endless DCT iterations, infeasible for video) vs FSR_FAST ~64 ms/frame (20× faster) with near-identical quality on a hole this small. Full Neon (9014 frames) processes in ~6 min.
- **Context-aware ROI padding** — FSR reconstructs the hole from the surrounding texture, so it gets a generous 30 px context crop (`kFsrContextPad`); NS keeps the lean `radius+4` crop (it only uses the mask edge, and runs on uniform backgrounds where the diamond is negligible).
- **`WMR_HAS_XPHOTO` guard** — `if(TARGET opencv_xphoto)` links `opencv_xphoto` when the OpenCV build includes contrib (Homebrew yes; vcpkg `contrib` feature). Without xphoto, `auto` collapses to NS for every scene — bit-identical to v1.6.0 (no regression); a one-time warning fires if FSR is explicitly requested.

#### Changed
- **Behavior change:** `--notebooklm-method` default `ns` → `auto` (FSR on intricate backgrounds). Choices `{auto|ns|fsr}` with `CLI::IsMember` validation (was silently accepting any string); SHIFTMAP/LaMa dropped from the choice set.
- `vcpkg.json`: `opencv4` += `contrib` (xphoto/FSR; monolithic but only `opencv_xphoto` links into the binary).
- 7 new `resolve_inpaint_method` unit cases (34 total).

#### Note
FSR is a **partial** fix — it kills the diamond and blends far better than NS, but the fill is still softly blurred on the most intricate backgrounds. True quality (LaMa, ~2.4 s/frame CPU → infeasible as a video default) is reserved for Phase C (optional opt-in).

### Single full-package release per platform

The 4 lean binaries + separate macOS AI tarball are replaced by **one self-contained package per (OS, arch)**, each shipping the FDnCNN AI denoise (NCNN+Vulkan) + FSR. No more lean/full split.

- **`release.yml` 4-leg matrix:** macOS arm64 (native + bundled Vulkan loader/MoltenVK → out-of-the-box GPU), macOS x86_64 (**cross-compiled** on the arm64 runner via `x64-osx` + Rosetta — the old `wmr-macos-x86_64` was arm64-mislabeled, and the only Intel runner `macos-13` was retired), Linux, Windows. `WMR_BUILD_AI_DENOISE=ON` + recursive submodules + `ai-denoise` vcpkg feature on every leg; CPU denoise smoke on all.
- **Vulkan handling:** macOS arm64 bundles the loader + MoltenVK (no system Vulkan → out-of-the-box GPU); macOS **x86_64 is built `WMR_NCNN_VULKAN=OFF` (CPU-only AI)** — on Apple, NCNN's simplevk does *static* Vulkan linkage and needs a build-time `libvulkan`, but there's no x86_64 Vulkan dylib on the arm64 cross-build runner, so Vulkan is compiled out entirely (Intel AI is CPU-only; the arm64 build is the GPU one). Linux, Windows rely on NCNN's `simplevk` runtime `dlopen` (system `vulkan-1.dll`/`libvulkan.so.1` if present, else graceful CPU fallback).
- **New `tests` CI job** (ubuntu, `WMR_BUILD_AI_DENOISE=ON + WMR_BUILD_TESTS=ON`) restores AI + routing test coverage the dev-default-OFF / release-TESTS-OFF gap previously dropped.
- `release` attaches 4 assets + `LICENSE-THIRD-PARTY.md`; the standalone `ai-denoise` job is removed (folded into the matrix + `tests`).

#### Changed
- `LICENSE-AI.md` → `LICENSE-THIRD-PARTY.md` — now accompanies **every** release binary (not just the retired AI tarball); broadened + added the OpenCV / opencv_contrib / xphoto Apache-2.0 notice. NCNN (BSD-3), volk (MIT), KAIR/FDnCNN (MIT) unchanged.

## [1.6.0] - 2026-07-11

### NotebookLM adaptive per-scene inpaint dispatch

Phase A of the NotebookLM quality roadmap. `process_notebooklm` now dispatches **per scene** instead of inpainting every frame:

- **Presence gate** — per scene, template-match the mark on sampled frames; scenes where the mark is absent are written through unmodified (no needless inpaint, no degradation of clean frames). Reuses `NotebookLMDetector::mark_present_in_scene`.
- **Complexity gate** — per scene, a Sobel gradient-energy score around the mark (`notebooklm_gates`: `background_complexity_score`) classifies the background as uniform vs intricate (logged; routing active in Phase B).
- **ROI crop** — inpaint a padded crop around the mark (not the full frame): faster, smaller visible area.
- Removal is still NS-only; clean-frame copy / `cv::xphoto` SHIFTMAP (Phase B) and LaMa (Phase C, optional) are reserved behind `InpaintMethod::ShiftMap`/`LaMa`.

#### Changed
- `InpaintMethod` enum: `ShiftMap` (+ `LaMa` behind `WMR_AI_LAMA`).
- `VideoWatermarkConfig`: `notebooklm_method`, `notebooklm_complexity_threshold`, `notebooklm_presence_threshold`.
- CLI: `--notebooklm-method {ns|shiftmap|lama}`, `--complexity-threshold`.
- New `src/video/notebooklm_gates.{hpp,cpp}` (VideoReader-free gate logic; unit-tested) + `tests/unit/notebooklm_gates_test.cpp`.
- vcpkg.json version drift fixed (was stale at 1.4.0).

#### Note
A temporal reverse-alpha "recovery" method was investigated and **ruled out** — the NotebookLM mark is adaptive (α≈0) with no mathematical inverse. Remaining quality upgrades are spatial (Phase B: clean-frame copy + SHIFTMAP; Phase C: optional LaMa ONNX).

## [1.5.0] - 2026-07-10

### NotebookLM video watermark removal

New `--notebooklm` mode removes the NotebookLM rainbow logo + "NotebookLM" wordmark from generated videos (cinematic, explainer, short-portrait exports).

- **Detection** (`src/video/notebooklm_detector.cpp`): template matching — multi-scale `|TM_CCOEFF_NORMED|` against each sampled frame, keep the best. Polarity-invariant (handles the adaptive light-on-dark / dark-on-light mark) and robust across scene cuts, where an earlier temporal-median-contrast detector failed. The detected bbox snaps to user-measured exact coordinates per known export mode.
- **Removal**: per-frame Navier-Stokes `cv::inpaint` over the mark bbox (dilated), reusing the existing inpaint primitive. Reverse alpha-blend does **not** apply — the NotebookLM mark is semi-transparent, color-adaptive, and H.264-compressed, so it is not a reversible constant-alpha overlay.
- **CLI**: `wmr video in.mp4 -o out.mp4 --notebooklm` (auto-detect); `--rect x,y,w,h` manual override for edge cases.
- **Known limitation**: on complex/textured backgrounds (e.g. explainer mode), spatial inpaint fabricates the region from its boundary — usable but imperfect. A temporal reverse-alpha method that recovers the true background is planned.

#### Changed
- `VideoProfile` gains `NotebookLM`; `VideoWatermarkConfig` gains `notebooklm_rect`.
- New embedded asset `notebooklm_mark_png` (98×14 grayscale template) in `assets/embedded_assets.hpp`.

## [1.4.0] - 2026-06-29

### Standalone macOS AI Binary

The `wmr-macos-arm64-ai` artifact (deferred in 1.3.0) now ships and runs on a clean macOS install.

#### Added
- `wmr-macos-arm64-ai.tar.gz` release artifact — bundles the Vulkan **loader** (`libvulkan.1.dylib`) + **MoltenVK** (`libMoltenVK.dylib`) + an ICD manifest next to the binary, so the Metal GPU driver travels with it. No Vulkan SDK, Homebrew, or MoltenVK needed at runtime. Layout: `wmr` (launcher) + `wmr.bin` (binary) + `lib/`. The `wmr` launcher sets `VK_ICD_FILENAMES` to the bundled manifest, then `exec`s `wmr.bin`.
- `scripts/bundle_macos_vulkan.sh` — rewrites the binary's `libvulkan` load command to `@rpath` + adds an `@loader_path/lib` rpath, writes a co-located ICD manifest (`library_path` relative to itself), ad-hoc re-signs (Apple Silicon kills stale-signed Mach-Os), and emits the launcher. Used by CI; works locally too.
- The `ai-denoise` CI job now installs `vulkan-loader` (the source of `libvulkan` — `molten-vk` ships only `libMoltenVK`), bundles the dylibs, and verifies the extracted tarball launches and runs a denoise (via the launcher, forcing CPU with `VK_ICD_FILENAMES=/nonexistent` since GitHub's paravirtualized macOS GPU can't run MoltenVK's `vkCreateInstance`; the GPU path is verified out-of-band on real Apple Silicon). The launcher respects a pre-set `VK_ICD_FILENAMES`. Re-enabled on tag pushes and re-coupled to the `release` job, so the AI tarball ships with every release.

#### Fixed
- ncnn/Vulkan teardown crash: `wmr` segfaulted (EXC_BAD_ACCESS in `ncnn::VulkanDevice::vkdevice()`) on process exit whenever the GPU path ran, because the `NcnnDenoiser` process-singleton's `~ncnn::Net` destructor ran during C++ static teardown *after* ncnn's global Vulkan device was already torn down. The singleton is now an intentionally-leaked heap instance (`WatermarkEngine::denoiser()`) so its destructor never runs at exit — the standard pattern for a process singleton that owns a GL/Vulkan context. Per-image output was always saved; the fix makes the exit code clean (was 139/SIGSEGV). Surfaced now because bundling made the GPU path reliable (the CPU fallback path never created the device).

## [1.3.0] - 2026-06-28

### FDnCNN AI Denoise (optional) — Still-Image Residual Cleanup

#### Added
- FDnCNN AI denoiser (NCNN + Vulkan, CPU fallback) as an optional residual-cleanup method after reverse alpha blending, ported from allenk/GeminiWatermarkTool (MIT). Original model from cszn/KAIR (MIT).
- `WMR_BUILD_AI_DENOISE` CMake option (OFF by default) — when ON, AI is the default still-image cleanup with automatic Gaussian fallback on init failure.
- `NcnnDenoiser` module (`src/core/ai_denoise.{hpp,cpp}`) + `src/core/ai_denoise_model.cpp` + `src/core/ncnn_shim.hpp`, all `#ifdef WMR_AI_DENOISE`-guarded.
- `--denoise {ai|soft|ns|telea|off}`, `--sigma`, `--strength`, `--radius` flags on `remove`/`visible` (AI-only; absent from the lean OFF build, which keeps `--inpaint-strength` as the sole cleanup knob).
- `WatermarkEngine::remove_watermark_detected` `InpaintConfig` overload (the `float` overload forwards for backward compatibility).
- `InpaintMethod::AiDenoise` (guarded in-enum); `InpaintConfig::sigma`.
- `[aidenoise]` ctest tag (model load, synthetic residual, masked blend, Gaussian fallback) — runs only on the ON build.
- New `LICENSE-AI.md` at repo root (NCNN BSD-3-Clause + volk MIT + KAIR/FDnCNN MIT + allenk conversion credit) shipped with the AI binary.
- New `ai-denoise` CI job producing the `wmr-macos-arm64-ai` release artifact (macOS arm64 only for the first AI release; x86_64/Linux/Windows AI builds are future work).

#### Build
- NCNN built from a git submodule at `external/ncnn/ncnn-20260113-src`; volk supplied by the vcpkg `ai-denoise` manifest feature (NOT a default dependency — lean binaries stay AI-free).
- Adds ~6.8 MB of embedded-model headers (`assets/model_core.{mem.h,id.h}`), committed so the AI CI job builds from a clean checkout without a network fetch. Provenance: headers captured from a one-time allenk/GeminiWatermarkTool `ENABLE_AI_DENOISE=ON` build (its `ncnn2mem` step); model bytes are upstream's converted weights.
- `scripts/build.sh` AI mode: `WMR_AI_DENOISE=1` (inits the submodule, adds the `vulkan-volk`/`molten-vk` Homebrew deps, passes `-DWMR_BUILD_AI_DENOISE=ON`).
- The 4-job default release matrix is untouched and AI-free (R7): no submodules, no `WMR_BUILD_AI_DENOISE`, no `ai-denoise` vcpkg feature.

#### Out of scope
- Video AI denoise — the video pipeline stays alpha-only (no inpaint, no AI).

## [1.2.0] - 2026-06-27

### Still-Image V2 (Gemini 3.5) Watermark Parity

#### Added
- `WatermarkVariant {V1, V2}` (`src/core/types.hpp`) — still-image profile generations, default V2. Separate from video `VideoVariant`.
- `get_watermark_config(W,H,variant)` + `v2_small_config_from_dims` — V2 large {192,192,96}; V2 small 36×36 with aspect-aware margin (ported verbatim from upstream allenk/GeminiWatermarkTool v0.3.1).
- V2-aware `WatermarkEngine::detect_watermark` (6-arg) / `remove_watermark` (3-arg) — default V2 with auto V2→V1 fallback on miss. The V2 alpha flows through the existing `custom_alpha` channel; `blend_modes` is unchanged.
- ±3px NCC snap refinement for V2+Small in `NccDetector::detect` (trusted iff spatial NCC ≥ 0.60), via a new trailing `enable_snap` param.
- `--legacy` / `--no-legacy` flags on `remove`, `visible`, `detect` (still subcommands) and batch. `--legacy` pins V1; `--no-legacy` pins V2 and disables fallback; both → exit 2. Video `--legacy` (Veo-text) is unchanged (CLI11-scoped per subcommand).
- New `[v2]` ctest tag: synthesized Gemini 3.5 round-trip (≤1 LSB mean), V2 small + snap (36×36 region), and V1-legacy-default tests.

#### Changed
- Still-image default profile flipped V1 → V2 (with auto fallback). Legacy 4-arg `detect_watermark` / 2-arg `remove_watermark` inline overloads keep routing to V1, so the video pipeline and existing tests are provably unchanged.
- Video pipeline, Veo, NotebookLM, SynthID — out of scope, unchanged.

#### Build — robust macOS/Homebrew workflow (survives `brew upgrade`)
- `scripts/build.sh`: self-healing build helper — verifies required Homebrew formulas, auto-wipes + reconfigures when a cached dependency path has vanished after an upgrade, configures against the stable `/opt/homebrew/opt/…` symlinks, then builds + tests.
- `cmake/FindFFMPEG.cmake`: prefer `FFMPEG_ROOT` (stable opt symlink) over pkg-config's versioned Cellar realpath, so the cached FFmpeg path survives an upgrade.
- `CMakePresets.json`: `mac-homebrew-Release`/`-Debug` presets (system libs, no vcpkg).
- `CMakeLists.txt`: synthesize the `FFTW3::fftw3f` imported target from Homebrew's variable-only fftw3 config (no-op under vcpkg) so the test target links.
- `tests/CMakeLists.txt`: exclude unused `scene_detector.cpp` from the test link (it pulled in unlinked FFmpeg via `VideoReader`).
- Test suite needs Catch2 (`brew install catch2`); `scripts/build.sh` verifies deps.

## [1.1.0] - 2026-06-02

### Phase 7: Scene Detection and Splitting — COMPLETE

#### Added
- `SceneDetector` class (`src/video/scene_detector.hpp/cpp`) — combined BGR histogram + pixel difference scene boundary detection:
  - Downsamples frames to longest-side 320px, preserves BGR color
  - Per-channel BGR Bhattacharyya distance (max across channels) + mean absolute pixel difference (MAD)
  - Combined metric: `max(per_channel_bhatt, mad)` — catches both chromatic and structural changes
  - Configurable threshold (default 0.30) and minimum scene length (15 frames)
  - Short scene merging into predecessor
  - Owns its own `VideoReader` — main reader stays pristine for processing
- `--scenes` flag on `video` subcommand — splits video into separate MP4 files at scene boundaries
- `--scene-threshold` option — scene cut sensitivity 0.0-1.0 (default 0.3)
- Scene splitting pipeline: boundary detection → single watermark detection → multi-file output with per-scene audio trimming
- `VideoWriter::copy_audio_range()` — seek-based audio range copy with PTS offset subtraction for clean scene-boundary alignment
- Production-ready split files: I-frame at start, audio trimmed to scene range, correct container duration, independently playable
- Output naming: `<stem>_<NNN>.mp4` with dynamic zero-padding; `-o` specifies directory (defaults to `<input>_scenes/`)
- CLI validation: `--scenes` rejects file output, creates directory automatically

## [1.0.0] - 2026-05-31

### Phase 6: Video Watermark Removal + CLI Polish — COMPLETE

#### Fixed
- Video watermark removal now uses pure reverse alpha blend (`remove_watermark_alpha_only`) instead of alpha blend + Gaussian inpaint — eliminates blur/diamond artifacts on both Gemini and Veo
- All frames are now processed when shot detection confirms the watermark — previously occlusion gate (NCC < 0.35) skipped frames; now falls back to shot anchor position for undetected frames
- Integrated upstream GeminiWatermarkTool V2 diamond alpha maps (36x36 and 96x96) for more accurate Gemini video removal
- Added `correct_alpha_for_background()` to recover true alpha from captures on non-black backgrounds

#### Changed
- Video pipeline: `remove_watermark_detected` → `remove_watermark_alpha_only` for lossless frame restoration
- Detection mode no longer skips frames — uses shot anchor as fallback when per-frame NCC is low

#### Added
- `VideoReader` class (`src/video/video_reader.hpp/cpp`) — FFmpeg demux+decode pipeline with seeking and frame counting
- `VideoWriter` class (`src/video/video_writer.hpp/cpp`) — FFmpeg encode pipeline (libx264, CRF 14, High profile) with audio passthrough:
  - Audio streams created before MP4 header write (valid moov atom)
  - Fresh input context for audio packet copy with timestamp rescaling
  - BGR24→YUV420P colorspace conversion via swscale
- `VideoProcessor` class (`src/video/video_processor.hpp/cpp`) — frame-by-frame watermark removal:
  - Shot-level NCC detection: samples frames across first 90% of video, takes median position/confidence
  - Per-frame occlusion gate: skips frames where watermark not reliably detected
  - Position refinement: falls back to shot anchor if detection drifts beyond tolerance
  - Frame dimension guard against corrupt seek artifacts
  - Progress output with fps and ETA
- `video` CLI subcommand: `wmr video input.mp4 -o output.mp4` with `--legacy`, `--variant`, `--crf`, `--preset`, `--codec`, `--force` options
- Codebook-free SynthID removal via `NoiseResidualSubtractor` (`src/synthid/noise_residual_subtractor.hpp/cpp`):
  - Estimates carrier from bilateral filter noise residual
  - `--codebook-free` flag as alternative to `--codebook`
- `cmake/FindFFMPEG.cmake` — cross-platform FFmpeg discovery via pkg-config / FFMPEG_ROOT (covers Homebrew and system package managers)
- CLI header with version, description, GitHub URL, and copyright
- Contextual subcommand help: shows subcommand-specific help when required args are missing (e.g. `wmr video`)
- Version output includes GitHub URL

#### Changed
- CMake links FFmpeg via imported targets (`FFMPEG::avformat` etc.) instead of variable-based linking
- `vcpkg.json` version bumped to 0.2.0
- Project version bumped to 0.2.0
- Single-image subcommands (`visible`, `synthid`, `remove`) now require `-o` explicitly — no default overwrite
- Batch processing defaults to `cleaned/` subdirectory instead of modifying originals
- Purged large codebook files (>100MB) from git history, excluded via `.gitignore`
- `.gitignore` updated to exclude HF datasets, generated codebooks, temp analysis outputs

### Phase 5: Unified CLI + Test Suite — COMPLETE

#### Added
- CLI restructured with CLI11 subcommands: `remove`, `detect`, `visible`, `synthid`, `build-codebook`
- Batch processing module (`src/cli/batch_processor.hpp/cpp`) — directory scanning with optional `--recursive`, parallel-safe
- `BatchResult` struct tracking total/succeeded/failed/skipped counts
- Catch2 v3 test suite with 17 tests, 54 assertions (all passing):
  - Unit tests: blend_modes (3), fft_context (4), spectral_codebook (4), inpaint (3)
  - Integration tests: visible_pipeline (3, 2 SKIP in build dir, pass from project root)
- Tests linked via `tests/CMakeLists.txt` with `catch_discover_tests()` and CTest integration
- Build option `WMR_BUILD_TESTS` (default ON) to toggle test building

### Phase 4: SynthID Detection + Codebook Builder — COMPLETE

#### Added
- `SynthidDetector` class (`src/detection/synthid_detector.hpp/cpp`) — 4-method Bayesian detection:
  - Noise correlation via bilateral filter denoise + NCC of noise spectrum vs profile
  - Carrier phase matching via element-wise cosine of phase differences
  - Structure ratio: energy at carrier bins vs total
  - Multi-scale consistency: phase coherence at 1x, 0.5x, 0.25x scales
  - Weighted fusion: noise_corr×0.35 + carrier_phase×0.35 + structure×0.15 + multi_scale×0.15
  - Configurable threshold (default 0.50)
- `CodebookBuilder` class (`src/synthid/codebook_builder.hpp/cpp`) — build spectral codebooks from reference images:
  - Per-channel FFT magnitude/phase accumulation
  - Automatic resolution bucketing
  - Consistency computation (std dev across samples)
  - Quality gate: warns on <3 samples per resolution

### Phase 3: SynthID Spectral Infrastructure — COMPLETE

#### Added
- `FftContext` class (`src/core/fft_context.hpp/cpp`) — FFTW3 wrapper with plan caching:
  - Forward/inverse 2D FFT with CV_32FC1/CV_32FC2 interop
  - Magnitude, phase, and polar reconstruction utilities
  - Plan caching keyed on (rows, cols, direction) with dummy arrays for FFTW_MEASURE safety
- `SpectralProfile` struct in `src/core/types.hpp` — per-resolution FFT data (magnitude, phase, consistency per BGR channel)
- `SpectralCodebook` class (`src/synthid/spectral_codebook.hpp/cpp`) — JSON-based codebook persistence:
  - Save/load with nearest-resolution fallback
  - `--codebook` CLI flag for specifying codebook path
- `CodebookSubtractor` class (`src/synthid/codebook_subtractor.hpp/cpp`) — multi-pass spectral subtraction:
  - Aggressive→moderate→gentle removal schedule
  - `--synthid-strength` CLI flag (0.0–2.0, default 1.0)
- `fftw3f` linked via vcpkg, FFTW3::fftw3f CMake target

#### Fixed
- SpectralProfile aggregate init bug: explicit field assignment prevents misaligned brace initialization
- FFTW_MEASURE input corruption: dummy arrays prevent plan creation from overwriting real data
- OpenCV `cv::cos` (nonexistent) replaced with element-wise `std::cos` loop in carrier phase matching

### Phase 2: Visible Watermark Detection + Inpainting — COMPLETE

#### Added
- `DetectionResult` struct in `src/core/types.hpp` — detection result with confidence scores, region, and per-stage scores
- `NccDetector` class (`src/detection/ncc_detector.hpp/cpp`) — three-stage NCC detection pipeline:
  - Stage 1: Spatial NCC via template matching with circuit breaker at 0.25
  - Stage 2: Gradient NCC via Sobel-filtered magnitude matching
  - Stage 3: Variance analysis comparing watermark region to reference region
  - Heuristic fusion: spatial×0.50 + gradient×0.30 + variance×0.20
  - Detection threshold: confidence ≥ 0.35
- `InpaintMethod` enum and `InpaintConfig` struct in `src/core/inpaint.hpp`
- `inpaint_residual()` function (`src/core/inpaint.hpp/cpp`) — three traditional inpainting methods:
  - Gaussian soft inpaint with gradient-weighted mask
  - TELEA inpaint (cv::inpaint INPAINT_TELEA)
  - Navier-Stokes inpaint (cv::inpaint INPAINT_NS)
  - Configurable strength, radius, and padding
- `opencv_photo` linked for cv::inpaint support
- WatermarkEngine integration: `detect_watermark()`, `remove_watermark_detected()`, `inpaint_residual()` methods
- CLI updated: default flow is now detect→remove→inpaint; `--force` skips detection; `--detect-only` reports results
- Verified: 80% detection confidence on test image, successful removal with Gaussian inpainting (9671 active pixels)

## [0.1.0] - 2026-05-28

### Phase 1: Visible Watermark Removal

#### Added
- C++20 project structure with CMake build system (CMakePresets.json, vcpkg.json)
- `WatermarkEngine` class — reverse alpha blending for visible watermark removal
- `blend_modes` module — alpha map calculation and forward/reverse alpha blending
- Embedded PNG assets (48x48 and 96x96 background captures) in `assets/embedded_assets.hpp`
- CLI via CLI11: `wmr input.png -o output.png` with `--force`, `--force-small`, `--force-large`, `-v` flags
- Auto-detection of watermark size based on image dimensions
- Support for PNG, JPEG, WebP output formats with quality preservation
