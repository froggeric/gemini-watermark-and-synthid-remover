# Research plan: alternative img2img models for the SynthID regen step

> Status: **Phase A sweep COMPLETE (2026-08-11).** SDXL-base wins by elimination; all 4 alternatives rejected on visual fidelity. Findings + verdict table below. The reusable method for evaluating future models is in **[`regen-model-evaluation-protocol.md`](./regen-model-evaluation-protocol.md)**.

## Why

The only validated SynthID-Image scrub is low-strength img2img diffusion regen. Today that is **SDXL** (native CoreML on macOS Apple Silicon; sdcpp CPU on linux/windows/mac-Intel). It clears Google's "Verify with SynthID" verifier (9/9 varied images at strength 0.10, 50 steps). It is, however, **lossy** (regen replaces pixels) and **slow on CPU**.

Goal of this research: find whether a different img2img model clears SynthID at **equal or lower strength** (less change to the image) and/or **faster**, with a **native (non-Python) runtime** on our platform matrix.

## Constraints (do not re-derive)

- **No Python at runtime is a SHIPPING rule, not a research rule.** For the selection sweep (Phase A), Python (diffusers/torch) is fine, and is in fact the fastest way to test the whole candidate space with zero conversion. A native port (Metal/CoreML/MLX/sdcpp) is Phase B work, done ONLY for a model that has already won the sweep (memory `python-ok-for-research-native-for-shipping`).
- **HW acceleration required:** CoreML/MLX/Metal on mac; CUDA/Vulkan/DirectML on Windows/Linux.
- **The removal knee is governed by `strength`** (the schedule's starting noise), not by steps. The validated SDXL knee is s=0.10 at N=50 (5 denoise steps). Sub-0.10 needs a higher N to even get distinct steps. See `synthid-light-reconstruction-attacks.md`.
- **The only SynthID verifier is Google's manual in-app tool**: about 10 checks/day/account, noisy (documented hallucinations). Plan sweeps around that budget; do not over-read a single verdict on near-identical images. See memory `synthid-removal-reality`.
- **Lossiness is fundamental.** "Better" means same clearance at lower strength (less change), or same clearance faster, not "lossless."

## Candidates

Phase A tests these in Python/diffusers (native-runtime availability only matters in Phase B):
- **SDXL** (baseline reference). Run in the same Python env for an apples-to-apples number.
- **SDXL Turbo / Lightning.** TOP candidates. Distilled from SDXL, so they retain the capacity SD-1.5 lacks, but fewer steps. Untested by anyone for SynthID clearance.
- **SD3 / SD3.5 Large.** Medium. sdcpp supports it; img2img + clearance untested.
- **Flux.1 schnell (and dev).** Lower priority. Known img2img quality problems (blurred/unrelated outputs, sdcpp #992); flow-matching strength semantics differ from SDXL's knee; bigger/slower on CPU.
- **CtrlRegen** (yepengliu/ctrlregen, ICLR 2025). The principled "regenerate from clean noise" approach vs additive-noise img2img. Worth one run as a fidelity reference, even though it is Python/diffusers only (no native runtime) so it cannot ship as-is.
- **Qwen-Image (base) — dark horse, medium priority.** Our vendored sdcpp ALREADY supports it natively (added 2025/10/12; `Qwen::QwenImageRunner`, docs/qwen_image.md), so Phase B on CPU would be cheap if it wins. It is a flow-matching DiT with strong capacity, Apache 2.0. Caveat: the documented path is txt2img; Qwen-Image-Edit is instruction-edit (`-r` + "change X to Y", Kontext-style), NOT strength-img2img. The sdcpp img2img path (init_image+strength) is model-agnostic in code, so a low-strength Qwen-Image img2img run is POSSIBLE but its SynthID-scrub quality at low strength is unverified (flow-matching low-strength img2img is finicky, same risk as Flux). Worth ONE Phase A test because of the native runtime + capacity, NOT because the community uses it for SynthID (they do not, see Findings). Qwen-Image-Edit is the wrong tool (semantic instruction-edit, not noise-injection scrub).
- **Ruled out:** **SD-1.5 / SDXS**. Empirically FAILS SynthID v2 (wiltodelta tested dreamshaper at 768px, strength 0.04/0.10 + elastic warp, all flagged positive). Lacks capacity at low strength. See Findings.
- **Evaluated + deprioritized: Microsoft Mage-Flow** (4B, MIT, flow-matching t2i + instruction-edit; `microsoft/Mage`). Small + permissive is attractive, but two blockers for THIS task: (1) wrong paradigm, it is flow-matching instruction-edit like Flux.2/Qwen-Image-Edit, NOT strength-controlled DDPM img2img, so no fine-grained minimal-perturbation knob; (2) no native runtime (Python + custom code only; not in sdcpp/CoreML/MLX; novel nr-MMDiT + Mage-VAE arch means Phase B is a from-scratch port bigger than SDXL Phase 3). Also: Microsoft pulled the official release ~2026-07 (404s; reason unconfirmed); mirrors exist (HF `mage-flow-community`). License/redistribution is the owner's call, but the retraction is a provenance flag for shipping. Optional low-priority Python probe only.
- Others from the user's list (Cosmos3, HiDream, Ideogram 4, Qwen-Image, HunyuanImage, Lumina, FIBO, Sana): pending verification, but expected to be Python-only and/or non-img2img; unlikely Phase A material.

## Evaluation criteria (per model x backend)

1. **SynthID clearance**: the lowest strength that clears Google's verifier (lower = less lossy). Reuse the validated image set.
2. **Fidelity**: diff_mean / PSNR vs the pre-regen original at the clearing strength, with and without the detail-restoration pass.
3. **Speed**: per-1024-tile time on the target backend (mac CoreML GPU; mac/linux/windows CPU).
4. **Peak memory.**
5. **Model size / download.**
6. **License / redistribution.**
7. **Native runtime availability per platform.**

## Method (two-phase)

**Phase A, research/select (Python, diffusers/torch).** The whole point of relaxing the no-Python rule here: diffusers supports every candidate natively with zero conversion, so we can sweep fast.

1. Shortlist candidates (see Candidates + Findings below) that are img2img-capable AND, per external evidence, plausibly able to clear SynthID at low strength. Already ruled out: SD-1.5-class (incl. SDXS) by wiltodelta's empirical SynthID-v2 failure.
2. For each candidate: run img2img at swept strengths (0.04 to 0.12, mapped to >= 2 effective steps by raising N where needed) on the validated set (the 9 images + the double-watermark hard case), in ONE shared Python/diffusers environment so the comparison is apples-to-apples. SDXL base is run in the same env as the baseline reference.
3. Measure fidelity (diff_mean / PSNR vs pre-regen original, +/- detail-restoration) and relative per-tile/per-image time in-process. Use the SAME prompt convention (empty/unconditional) and guidance across models for fairness.
4. Batch the swept outputs through the Google-verifier oracle (plan around the ~10/day budget; spread across days; do not over-read single verdicts) to find each model's lowest clearing strength.
5. Compare to the SDXL baseline; record per-model numbers in Findings.

**Phase B, ship (only if a Phase A winner exists).** Port the winner to a native runtime: CoreML on mac (via the `apple/ml-stable-diffusion` tag-1.1.1 recipe + mandatory fp16-fix VAE where applicable; a from-scratch conversion for non-SDXL, like Phase 3) and/or sdcpp on CPU. Re-verify output correctness (a runtime/port shift moves numerics) and re-validate clearance. Do NOT port a model that did not win the sweep.

Decide: adopt only if clearly better on clearance-strength, fidelity, or speed, AND it has a viable native path on the target platforms.

## Success criteria

- This doc holds a per-model comparison table (clearance strength, fidelity, speed, memory, size, license, platforms).
- If a winner exists: a defined integration follow-up (re-pin models, update `regenerator.cpp` backend dispatch, re-validate clearance + fidelity on colorful natural content, update README perf claims).

## Non-goals / risks

- No Python at runtime (hard rule).
- The verifier budget bounds how many model/strength cells we can clear-check; prioritize the most promising candidates.
- A faster model that needs a higher strength to clear is not a win.
- Per-model conversion effort (CoreML/MLX) can be large; timebox each.

## Findings

### Established before the sweep

**SDXL is independently validated as the right baseline.** The `wiltodelta/remove-ai-watermarks` project (Python/diffusers, MIT) arrived at the same SDXL img2img scrub independently. Their call: `stabilityai/stable-diffusion-xl-base-1.0`, `prompt=""`, `guidance_scale=7.5`, strength ~0.05, N=50, whole-image at ~1024px. Their docs claim it clears SynthID v2 on a Gemini 3 Pro output (verified via the Gemini app's "Verify with SynthID"). Same knee mechanism as ours (`effective_steps = int(N*strength)`).

**SD-1.5-class models are out (external empirical evidence).** wiltodelta shipped SD-1.5 dreamshaper first, then REMOVED it: at 768px it did NOT clear SynthID v2 at strength 0.04, 0.10, or with elastic warp a in {5,8} (all flagged positive). So SD-1.5/SDXS lack the capacity to scrub SynthID at low strength. Removes them from our candidates.

**The 0.05-vs-0.10 discrepancy is reconcilable + a fidelity lead.** Their 0.05 (2 effective steps at N=50) clears a SINGLE Gemini 3 Pro image. Our 0.10 (5 steps) is needed for our HARDEST case (double-watermarked img2img). Not contradictory: singles are easier. Action: we may run typical images at lower strength (0.05-0.08), reserving 0.10 for the hard case. Re-validate on our pipeline. Reinforces: the knee is governed by starting-noise / effective-steps, not raw step count.

**The native-runtime picture (Phase B only).** Our fast mac path is CoreML via `apple/ml-stable-diffusion` tag 1.1.1 = SDXL (and SD1.5/2.1) only. No working CoreML img2img conversion exists for SD3/SD3.5/Flux/Turbo/Lightning (community requests open, nothing shipped). apple HEAD adds SD3 Medium but fp32 CPU+GPU only (ANE-ineligible, slow). MLX runtimes (mflux, mlx-gen) that support Flux.2/Qwen are Python-over-MLX. DiffusionKit Swift is unfinished. sdcpp on Apple Silicon is Metal-broken upstream (garbage output), so sdcpp there is CPU-only (~143s/tile). Net: mac fast path stays SDXL unless a Phase A winner justifies a CoreML port. On linux/windows/mac-Intel the CPU path is sdcpp, which already supports SDXL/Turbo/Lightning/Flux (Flux via the generic init_latent+strength path; Flux docs show txt2img/instruction-edit, not strength-img2img).

**sdcpp (our vendored CPU backend) is broad.** At `master-808` (Jan 2026) it supports SDXL/SDXL-Turbo, distilled SSD-1B/Vega/SDXS, FLUX.1 dev/schnell, FLUX.2-dev/klein, FLUX.1-Kontext, plus LTX/SeFi-Image/MiniT2I denoisers. img2img = `img_gen` mode + `init_image` + `strength` (no separate img2img mode); the init_latent+strength noising is in the common sampling loop, so it is model-agnostic in principle. Our regenerator already calls exactly this one `generate_image` path, so a Phase B model swap flows through a single integration point.

### Sweep results (Phase A, in progress; 2026-08-11)

Run on a 16 GB Apple Silicon Mac via diffusers (MPS), empty-prompt img2img, fixed seed,
inputs downscaled to 1024-long-side preserving aspect (aligned %16 so every model emits
identical dims for pixel-aligned comparison). diff_mean = mean per-channel change vs the
pre-regen input (lower = less lossy). **Clearance (does it clear SynthID) is NOT measured
here** — that is the manual Google-verifier step, not yet done for any model.

| model | strength (eff steps) | diff_mean | time/gen | peak mem | fits 16 GB? |
|---|---|---|---|---|---|
| SDXL (baseline) | 0.05 (2) | ~4.0 | ~10 s | 7.5 GB | yes (clean) |
| SDXL | 0.10 (5) | ~5.4 | ~18 s | 7.5 GB | yes |
| SDXL | 0.15 (8) | ~6.2 | ~24 s | 7.5 GB | yes |
| SDXL-Turbo | 0.25 (2) | ~16.3 | ~5 s | 3.4 GB | yes |
| SDXL-Turbo | 0.50 (4) | ~26.2 | ~6 s | 3.4 GB | yes |
| SDXL-Lightning (8-step) | 0.50 (4) | ~13.5 | ~9 s | 12.4 GB | yes |
| SDXL-Lightning (8-step) | 0.88 (7) | ~39.1 | ~14 s | 12.4 GB | yes |
| Flux.1-schnell (Q5 gguf) | 0.25 (1) | ~6.3 | ~127 s | ~12 GB | marginal (model offload) |
| Z-Image-Turbo | 0.25 (2) | ~7.9 | ~230 s | ~7.7 GB | marginal (model offload) |

**Early reads:**
- **SDXL still leads on lossiness.** Its lowest perturbation (s=0.05, diff ~4.0) beats every alternative at its floor.
- **Distilled SDXL (Turbo) loses the low-perturbation regime.** Turbo's clean minimum (s=0.25, diff ~16) is ~3x more lossy than SDXL at s=0.10 (diff ~5.4); its s=0.10 point collapses (0 effective steps, pipeline error). **Lightning (8-step) is a better distilled SDXL than Turbo** (eff-4 diff ~13.5 vs Turbo eff-4 ~26.2, half the loss), but it still cannot reach SDXL's low-perturbation regime (its min tested point s=0.50 is ~2.5x SDXL's s=0.10). Distillation costs the low regime regardless of step count.
- **Low-loss ranking (min clean point):** SDXL base (4.0) < Flux.1-schnell (6.3) < Z-Image-Turbo (7.9) < Lightning (13.5) < Turbo (16.3). SDXL wins. The flow models (Flux, Z-Image) are 2nd on loss but impractically slow on 16 GB (~2-6 min/gen).

**Visual fidelity verdict (user inspection, 2026-08-11):** **Flux.1-schnell is DISQUALIFIED** - it is "awful at preserving text, then faces, then the rest." The mean-diff metric hid this (it averages over the whole image); the flow regeneration visibly mangles text and faces even at low average diff. Do NOT use Flux-schnell. This also lowers the prior on the other flow models (Z-Image, Chroma) for the scrub use case - their low diff may similarly hide localized destruction. SDXL-base remains the fidelity leader.

**SDXL-Lightning is DISQUALIFIED** too - "zero fidelity, hallucinates a lot, bad image quality." The 8-step distilled UNet at img2img strengths produces hallucinated/low-quality output (its lower mean-diff than Turbo was misleading). So BOTH distilled SDXLs (Turbo lossy, Lightning hallucinated) are out, and BOTH tested flow models (Flux mangles text/faces) are out. By elimination SDXL-base is the only candidate so far with acceptable fidelity; the remaining question is whether SDXL-base's low strengths (0.05-0.10) actually CLEAR SynthID (the verifier pass).

**SDXL-Turbo is DISQUALIFIED** - "corruption, like seeing the image through geometric stained glass" (the classic distilled 4-step grid artifact). **Z-Image-Turbo is DISQUALIFIED** - "awful at preserving text, then faces, and then the rest" (identical failure mode to Flux).

**FINAL VERDICT - all four non-base backbones rejected on visual fidelity:**

| model | paradigm | mean-diff (looked OK) | visual verdict | status |
|---|---|---|---|---|
| **SDXL-base** | full DDIM | 4.0 (s=0.05) | (acceptable - sole survivor) | **KEEP** |
| Flux.1-schnell | flow | 6.3 | destroys text, then faces | REJECT |
| Z-Image-Turbo | flow | 7.9 | destroys text, then faces | REJECT |
| SDXL-Lightning | distilled SDXL | 13.5 | zero fidelity, hallucinates | REJECT |
| SDXL-Turbo | distilled SDXL | 16.3 | geometric stained-glass | REJECT |

**CONCLUSION: SDXL-base is the winner by elimination.** Every alternative has unacceptable visual artifacts despite two having acceptable mean-diff (the metric hides localized destruction). For a SynthID scrub, where fidelity is paramount (you regenerate a user's image), SDXL-base is the only viable img2img backbone tested. Two clear failure-mode patterns: (1) **flow models (Flux, Z-Image) destroy text and faces** - a paradigm-level fidelity problem, which also lowers the prior on Chroma (Flux-family) and Qwen; (2) **distilled SDXLs (Turbo, Lightning) introduce structural artifacts** (grid / hallucination). The ONLY remaining open question is clearance: does SDXL-base at a low strength (0.05-0.10) actually clear SynthID, and what is the minimum clearing strength. The not-yet-tested candidates (Chroma, SD3.5, Qwen quantized) are long shots given these failure modes (SD3.5 is the only non-flow, non-distilled one, so it is the one remaining worth trying if SDXL clearance is insufficient).
- **Flow models (Flux, Z-Image) preserve fidelity better than Turbo at low strength** (Flux s=0.25 ~6.3, Z-Image ~7.9, near SDXL's s=0.15), but they are **slow on 16 GB** (~2-6 min/gen with model offload) because the quantized weights + T5/Qwen encoders don't fit and must offload.

**16 GB ceiling is binding (key constraint):** the big flow models (Flux Q5 ~12 GB, Z-Image, Chroma 8.9B, SD3.5 ~13 GB, Qwen-Image full ~20 GB) do not fit 16 GB cleanly. With sequential offload the GGUF-quantized ones error (accelerate offload hooks break on quantized tensors, `KeyError(None)`); with model offload they run but spike memory into heavy swap (~5 GB swap observed) and get jetsam-killed mid-run. Only the SDXL-family (SDXL/Turbo/Lightning, ~7 GB) fits without offload and runs reliably here. The bigger flow models need a >16 GB machine (or the native sdcpp/CoreML path) to test fully.

**Paradigm filter (community recs):** Flux.2-Klein-9B and Qwen-Image-Edit-2511 are instruction-edit models with NO strength-img2img pipeline in diffusers — wrong tool for a controllable minimal-perturbation scrub (the BASE Qwen-Image has img2img but is too big for 16 GB). Chroma and Z-Image-Turbo (Apache 2.0) DO have img2img pipelines and are testable.

### Methodology / diffusers gotchas (for resuming the sweep)

The harness lives at `experiments/regen-model-sweep/sweep.py` (venv at `.venv/`, outputs gitignored). Hard-won lessons:
- **Aspect: align dims to multiples of 16, round-to-nearest.** SDXL needs %8 (VAE), Flux needs %16 (patchify); %16 satisfies both so every model emits identical dims for pixel-aligned cross-model comparison. Do NOT use %64 (distorts aspect ~7%) or floor-rounding. Flux/Chroma/Z-Image img2img ALSO need explicit `height=init.height, width=init.width` (Flux otherwise defaults to 1024x1024 square).
- **GGUF transformers need `GGUFQuantizationConfig(compute_dtype=bf16)`** passed to `<Model>Transformer2DModel.from_single_file(path, quantization_config=...)`, else a tensor-shape mismatch. Then assemble the pipeline `from_pretrained(components_repo, transformer=transformer)`.
- **Quantized model + `enable_sequential_cpu_offload` = `KeyError(None)`** (accelerate offload hooks break on GGUF tensors). Use `enable_model_cpu_offload` for quantized models.
- **`from_single_file` needs a resolved local path** via `hf_hub_download(repo_id, filename)`, NOT a "repo/filename" shorthand.
- **SDXL-Lightning:** the ByteDance UNet shard is a raw state_dict, not a from_single_file checkpoint. Load the SDXL pipeline, then `pipe.unet.load_state_dict(load_file(path))`; set `EulerDiscreteScheduler(..., timestep_spacing="trailing")`.
- **Distilled models' low-strength points collapse:** `int(N*strength)` hits 0 (Turbo s=0.10 @ N=8) -> a "reshape 0 elements" pipeline error. Their usable floor is higher strength.
- **SDXL-family fp16 needs the fp16-fix VAE** (`madebyollin/sdxl-vae-fp16-fix`) or the decode darkens.
- **Write metrics.csv incrementally** (after each row) so a killed/OOM run keeps all completed rows. (16 GB jetsam-kills the big flow models mid-run.)
- **The mean-diff metric HIDES localized quality loss** (text, faces). ALWAYS pair it with human visual inspection - that is what disqualified Flux (mangled text/faces) and Lightning (hallucinated), both of which had acceptable mean-diff. Clearance + fidelity are separate from mean-diff.

### Community consensus (what other projects actually do for SynthID)

A sweep of the SynthID-removal literature/projects finds a consistent answer: **low-strength SD-family img2img**, not Qwen or Flux. Concretely:
- `mertizci/noai-watermark` (what wiltodelta wraps): SDXL img2img, low strength, "an img2img pass at low denoising strength is often enough to fool SynthID."
- wiltodelta/remove-ai-watermarks: SDXL, strength ~0.05 (removed SD-1.5 because it failed v2).
- r/comfyui SynthID-bypass thread: "re-noising the image through img2img."
- HN discussion: "Stable Diffusion img2img with 10-15% denoising strength, looping with incremental increases until detection fails."
- Allen Kuo's SynthID research report: removal forces a quality tradeoff (the encoder's embedding is the obstacle).

No source specifically uses Qwen-Image or Flux for SynthID scrubbing. Qwen-Image is popular in the general image-gen/editing ecosystem (strong Apache-2.0 model; also "Qwen" appears as Qwen3 = FLUX.2's text encoder, a separate role), but its documented paradigms are txt2img (base) and instruction-edit (Edit), neither of which is the validated noise-injection operation. It stays a Phase A dark horse only because it has a native sdcpp runtime.
