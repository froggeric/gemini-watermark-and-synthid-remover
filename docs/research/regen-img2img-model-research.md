# Research plan: alternative img2img models for the SynthID regen step

> Status: planning (2026-08-10). Research not started. This doc will hold the plan now and the findings later.

## Why

The only validated SynthID-Image scrub is low-strength img2img diffusion regen. Today that is **SDXL** (native CoreML on macOS Apple Silicon; sdcpp CPU on linux/windows/mac-Intel). It clears Google's "Verify with SynthID" verifier (9/9 varied images at strength 0.10, 50 steps). It is, however, **lossy** (regen replaces pixels) and **slow on CPU**.

Goal of this research: find whether a different img2img model clears SynthID at **equal or lower strength** (less change to the image) and/or **faster**, with a **native (non-Python) runtime** on our platform matrix.

## Constraints (do not re-derive)

- **No Python at runtime.** Python is allowed only for model conversion. Native C++ runtimes only (tenet: memory `ai-integration-tenets`).
- **HW acceleration required:** CoreML/MLX/Metal on mac; CUDA/Vulkan/DirectML on Windows/Linux.
- **The removal knee is governed by `strength`** (the schedule's starting noise), not by steps. The validated SDXL knee is s=0.10 at N=50 (5 denoise steps). Sub-0.10 needs a higher N to even get distinct steps. See `synthid-light-reconstruction-attacks.md`.
- **The only SynthID verifier is Google's manual in-app tool**: about 10 checks/day/account, noisy (documented hallucinations). Plan sweeps around that budget; do not over-read a single verdict on near-identical images. See memory `synthid-removal-reality`.
- **Lossiness is fundamental.** "Better" means same clearance at lower strength (less change), or same clearance faster, not "lossless."

## Candidates (img2img-capable, with a native C++ runtime)

Evaluated against runtime availability:
- **SDXL** (current baseline). CoreML (mac) + sdcpp (CPU).
- **SD 1.5 / 2.1.** Smaller, faster on CPU. sdcpp + CoreML.
- **SD3 / SD3.5.** sdcpp.
- **Flux.1 (schnell/dev).** High fidelity, larger. sdcpp.
- **SDXL Turbo / Lightning.** Few-step, speed candidate. sdcpp.
- Others (PixArt, Playground, etc.) only if a native backend exists.

## Evaluation criteria (per model x backend)

1. **SynthID clearance**: the lowest strength that clears Google's verifier (lower = less lossy). Reuse the validated image set.
2. **Fidelity**: diff_mean / PSNR vs the pre-regen original at the clearing strength, with and without the detail-restoration pass.
3. **Speed**: per-1024-tile time on the target backend (mac CoreML GPU; mac/linux/windows CPU).
4. **Peak memory.**
5. **Model size / download.**
6. **License / redistribution.**
7. **Native runtime availability per platform.**

## Method

1. Pick 3 to 4 candidates that have a native runtime on at least mac (CoreML or MLX) and CPU (sdcpp).
2. For each: convert (CoreML via the `apple/ml-stable-diffusion` tag-1.1.1 recipe + the mandatory fp16-fix VAE where applicable; sdcpp gguf), then run img2img at swept strengths around the SDXL knee (0.06 to 0.12) on the validated set.
3. Measure criteria 2 to 7 in-process; batch the swept outputs through the Google-verifier oracle (around the ~10/day budget) to measure criterion 1.
4. Compare to the SDXL baseline; record per-model numbers below.
5. Decide: adopt a model only if it is clearly better on clearance-strength, fidelity, or speed, with a native path on the platforms, AND output correctness is re-verified (a model change shifts numerics).

## Success criteria

- This doc holds a per-model comparison table (clearance strength, fidelity, speed, memory, size, license, platforms).
- If a winner exists: a defined integration follow-up (re-pin models, update `regenerator.cpp` backend dispatch, re-validate clearance + fidelity on colorful natural content, update README perf claims).

## Non-goals / risks

- No Python at runtime (hard rule).
- The verifier budget bounds how many model/strength cells we can clear-check; prioritize the most promising candidates.
- A faster model that needs a higher strength to clear is not a win.
- Per-model conversion effort (CoreML/MLX) can be large; timebox each.

## Findings

_(to be filled in during the research)_
