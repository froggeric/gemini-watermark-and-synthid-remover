# CoreML SDXL GPU/ANE Placement Research

## Summary

The CoreML SDXL UNet runs on the CPU (~10s per 1024 tile on M4) instead of GPU/ANE because the current SPLIT_EINSUM attention implementation likely causes CPU fallback. Re-converting with ORIGINAL attention and/or forcing GPU placement via `MLComputeUnitsCPUAndGPU` should move execution to GPU and achieve ~1-3s per tile (10x speedup). Any placement change requires re-validation on colorful content with the Google SynthID verifier.

**Status:** Research complete. Recommend proceeding with the ORIGINAL-attention re-conversion experiment (execution task).

---

## 1. Current Placement (Why CPU)

### Code Configuration
- `coreml_sd_pipeline.mm` loads models with `cfg.computeUnits = MLComputeUnitsAll` (line 160)
- `MLComputeUnitsAll` tells CoreML to prefer ANE, then GPU, then CPU
- The UNet was converted with the default **SPLIT_EINSUM** attention implementation (tag 1.1.1 of `apple/ml-stable-diffusion`)

### Why SPLIT_EINSUM Falls Back to CPU
Based on community research (madebyollin's Stable Diffusion CoreML work, 2022-2023):

1. **ANE compilation failures**: The ANE compiler is strict about reshape/permute count and tensor contiguity. Too many reshapes in the attention block cause ANE compilation to fail. madebyollin notes: "not only does ANE take forever to load because it recompiles each time - it then doesn't work!"

2. **SPLIT_EINSUM behavior**: While SPLIT_EINSUM is designed to be compatible with all compute units (CPU, GPU, ANE), in practice it often produces ops that fall back to CPU due to ANE coverage gaps.

3. **Observed outcome**: With `MLComputeUnitsAll` + SPLIT_EINSUM, the UNet likely ends up on CPU because the ANE path fails during compilation, and the GPU path may also have compatibility issues with the SPLIT_EINSUM ops.

### Empirical Evidence
- The measured perf (~10s/tile on M4) aligns with CPU-only execution
- Comparable data from M1 Pro (madebyollin, June 2023): CoreML CPU+GPU+SPLIT_EINSUM achieved 1.39 it/s (~0.72s/it) vs CoreML ALL+SPLIT_EINSUM at 1.85 it/s (~0.54s/it)
- If the current UNet were using GPU/ANE, we would expect <1s/it on M4 (a newer/faster chip)

---

## 2. How GPU/ANE Placement Would Work

### Option A: Re-convert with ORIGINAL Attention
- **Command**: Add `--attention-implementation ORIGINAL` to the `torch2coreml.py` conversion
- **Effect**: ORIGINAL attention only supports CPU and GPU (not ANE), but is GPU-optimized and avoids the SPLIT_EINSUM reshape issues that cause ANE failures
- **Trade-off**: Loses ANE, but GPU is still much faster than CPU. ORIGINAL often beats SPLIT_EINSUM on Macs despite being ANE-ineligible.

### Option B: Load-time Compute Unit Override
- **Change**: One line in `coreml_sd_pipeline.mm` line 160:
  - From: `cfg.computeUnits = MLComputeUnitsAll;`
  - To: `cfg.computeUnits = MLComputeUnitsCPUAndGPU;` or `MLComputeUnitsCPUAndNeuralEngine;`
- **Effect**: Forces CoreML to use only the specified units. This may work around SPLIT_EINSUM's ANE failures by forcing GPU-only path.
- **Risk**: May not help if SPLIT_EINSUM ops are fundamentally GPU-incompatible.

### Option C: Precision/Layout Substitution
- The converted models already use fp16 (verified in IO specs)
- Planar NCHW layout is correct (already done in pipeline)
- No obvious precision changes needed

### Converter Flags Reference (`apple/ml-stable-diffusion`)
```
--attention-implementation {ORIGINAL, SPLIT_EINSUM, SPLIT_EINSUM_V2}
```
- ORIGINAL: GPU-friendly, ANE-ineligible
- SPLIT_EINSUM: ANE-targeted, but prone to CPU fallback due to reshape issues

---

## 3. Why It Is Good

### Current Problem
- `--synthid-attack regen` on Mac is ~10s per 1024 tile (CPU-only)
- A 4K image requires ~9-12 tiles = ~90-120s per image (impractical for batches)

### Expected Improvement
Based on M1 Pro data (madebyollin, June 2023):
- CoreML CPU+GPU+ORIGINAL: ~0.72s/it (1.39 it/s)
- CoreML ALL (CPU+GPU+ANE)+SPLIT_EINSUM: ~0.54s/it (1.85 it/s)
- M4 (newer, 2024 chip) should be faster than M1 Pro

**Conservative estimate for M4:**
- GPU-only (ORIGINAL attention): ~1-2s per tile
- GPU+ANE (if achievable): ~0.5-1s per tile

This makes `--synthid-attack regen` practical for large images and batch workflows on Mac.

---

## 4. Expected Gains

### Per-Tile Speedup
- Current (CPU): ~10s per 1024 tile on M4
- Target (GPU): ~1-3s per 1024 tile on M4
- **Speedup: ~5-10x**

### Total Wall-Time Impact
- 4K image (9-12 tiles): Current ~90-120s, Target ~9-36s
- One-time model load (~50s) stays the same
- Batch workflows benefit most (amortized load cost)

### Community Benchmarks
Sources:
- [Hugging Face CoreML Diffusers Blog](https://huggingface.co/blog/diffusers-coreml): "original attention can be faster than split_einsum on some devices"
- [madebyollin gist (June 2023)](https://gist.github.com/madebyollin/86b9596ffa4ab0fa7674a16ca2aeab3d): M1 Pro saw 1.68x speedup using CoreML+ANE vs PyTorch/MPS

### Uncertainty
- No on-device M4 profiling in this research
- Actual speed depends on which ops execute on GPU vs CPU
- ANE eligibility is op-specific; some layers may still fall back

---

## 5. Risks

### A. Numerics: Output Change Must Be Re-validated
- **Risk**: A different attention implementation or compute unit can change the UNet output slightly, which could shift SynthID removal effectiveness.
- **Project rule (HARD)**: Any AI image-op change must be re-validated on COLORFUL natural content. A uniform color shift is invisible on near-gray test images and hid real bugs in Phase 2 (Metal collapse) and Phase 3 (planar layout bug).
- **Requirement**: Run the natural-image diagnostic (`WMR_COREML_SD_DIAG_IMAGE`) and compare diff_mean vs the CPU baseline. If diff_mean is small (<1.0/255), proceed to Google verifier re-check.

### B. Conversion Cost and Reproducibility
- **Time**: UNet re-conversion takes ~20-30 minutes
- **Steps**: Re-pin SHA256 in `coreml_sd_model_fetch.cpp`, re-upload to `froggeric/wmr`
- **Dependency**: The conversion venv (`~/.cache/wmr/coreml-sdxl-venv`) must exist with coremltools installed
- **Recipe**: Tag 1.1.1, coremltools `_cast` patch, `--custom-vae-version madebyollin/sdxl-vae-fp16-fix`

### C. ANE Op-Coverage Limits
- SPLIT_EINSUM is theoretically ANE-targeted, but ANE fails on too many reshapes
- ORIGINAL attention cannot use ANE by design (CPU+GPU only)
- GPU-only is still a big win over CPU, but we lose the ANE potential

### D. CI Cannot Validate This
- GitHub macOS runners use a paravirtualized Metal GPU (`AppleParavirtDevice`) and have no ANE
- Any placement change must be validated out-of-band on real Apple Silicon hardware
- CI smoke tests only prove the model loads and runs, not that it uses the expected compute unit

---

## 6. Recommendation and Execution Outline

### Recommendation
**Try Option A (ORIGINAL-attention re-conversion) first.** It has the highest chance of success based on community reports that ORIGINAL often beats SPLIT_EINSUM on Macs, even though it loses ANE access.

**Fallback**: If ORIGINAL is insufficient, try Option B (compute-unit override) with the existing SPLIT_EINSUM model.

### Execution Outline (Follow-up Task)

1. **Re-convert UNet with ORIGINAL attention** (~20-30 min)
   ```bash
   cd ~/.cache/wmr/coreml-sdxl-venv
   source bin/activate
   python -m python_coreml_stable_diffusion.torch2coreml \
     --convert-unet \
     --attention-implementation ORIGINAL \
     --model-version stabilityai/stable-diffusion-xl-base-1.0 \
     -o ~/.cache/wmr/coreml-sdxl-original-attention
   ```

2. **Time the ORIGINAL variant on M4**
   ```bash
   WMR_COREML_SD_MODELS_DIR=~/.cache/wmr/coreml-sdxl-original-attention \
   ./build/wmr synthid <colorful-image.png> \
     --synthid-attack regen --regen-backend coreml --regen-strength 0.10
   ```
   Measure per-tile time.

3. **Verify output correctness**
   - Set `WMR_COREML_SD_DIAG_IMAGE` to a colorful natural image
   - Run the CoreML smoke test with the ORIGINAL model
   - Check diff_mean vs the CPU baseline (should be <1.0/255)
   - If diff_mean is small, proceed to verifier check

4. **Re-validate removal against Google verifier**
   - Run `--synthid-attack regen` at strength 0.10 on the same test images used in Phase 3
   - Verify with Google's official SynthID detector (rate-limited, ~10/day)
   - If removal clears the mark, the numerics shift is acceptable

5. **If adopted**
   - Re-pin SHA256 in `coreml_sd_model_fetch.cpp`
   - Re-upload the ORIGINAL-attention UNet to `froggeric/wmr`
   - Update README perf claims (reflect the measured GPU speedup)
   - Tag and ship as v1.15.1 or later

### If ORIGINAL Is Insufficient
- Try `MLComputeUnitsCPUAndGPU` override with the existing SPLIT_EINSUM model
- Profile placement with `COREML_*` debug env vars or Instruments
- If GPU still falls back to CPU, the issue may be deeper than attention (possibly the VAE or scheduler ops)

---

## Sources

- [Hugging Face: Using Stable Diffusion with Core ML on Apple Silicon](https://huggingface.co/blog/diffusers-coreml)
- [Apple Research: Stable Diffusion with Core ML on Apple Silicon](https://machinelearning.apple.com/research/stable-diffusion-coreml-apple-silicon)
- [madebyollin gist: Stable Diffusion on Apple Silicon GPUs via CoreML](https://gist.github.com/madebyollin/86b9596ffa4ab0fa7674a16ca2aeab3d) (June 2023 benchmarks)
- [apple/ml-stable-diffusion GitHub](https://github.com/apple/ml-stable-diffusion)
- Project files: `src/core/coreml_sd_pipeline.mm`, `~/.claude/plans/coreml-sdxl-phase3.md`
