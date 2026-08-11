# Protocol: evaluating a new img2img model for the SynthID scrub

> Use this whenever a new image model comes out and you want to know if it could
> replace SDXL-base as the regen backbone for SynthID removal. Companion to
> `regen-img2img-model-research.md` (the record of the first evaluation round).

## The evaluation question

The SynthID scrub is **low-strength strength-controlled img2img**: encode the image,
inject noise at a `strength` fraction of the diffusion schedule, denoise. A candidate
model is "better" than SDXL-base only if it can clear SynthID at **equal or lower
strength (less lossy)**, or at the same strength **faster / lighter / with a native
runtime**, **with no visible fidelity loss**. Lossiness without clearance is worthless;
clearance without fidelity is worthless. All three matter.

The bar to beat (2026-08-11): **SDXL-base** - diff ~4.0 at s=0.05, the only tested
backbone with acceptable visual fidelity. Its clearance strength is the open number.

## Step 0 - candidate filter (before downloading anything)

Cheapest screen; skips models that cannot fit the use case. All must be true:

1. **Paradigm: must expose strength-controlled img2img**, not just txt2img or
   instruction-edit. Check `diffusers` for an `*Img2ImgPipeline` class for the model
   (e.g. `FluxImg2ImgPipeline`, `QwenImageImg2ImgPipeline`, `ChromaImg2ImgPipeline`,
   `StableDiffusion3Img2ImgPipeline`, `ZImageImg2ImgPipeline`). Instruction-edit /
   Kontext-style models (`Flux2KleinPipeline`, `QwenImageEditPipeline`, Flux.2-Klein,
   Qwen-Image-Edit, FLUX.1-Kontext) are the **wrong tool** - they change content per a
   text instruction, not a controllable noise fraction. Reject.
2. **License: permissive** (Apache-2.0, MIT, OpenRAIL) for shipping. NC/pulled/retracted
   models are the owner's call but a flag.
3. **16 GB feasibility: estimate resident size** = transformer + text encoders + VAE.
   - Fits cleanly (~<=8 GB, no offload): SDXL-family. Reliable.
   - Needs GGUF Q4/Q5 + `model_cpu_offload` (~<=13 GB): marginal, slow, jetsam risk.
   - Bigger than ~14 GB: do not attempt on the 16 GB dev Mac; needs a bigger box or sdcpp.
   The text encoder is often the binding constraint (T5-XXL ~5-9 GB, Qwen2.5-VL-7B ~7 GB);
   plan to quantize/shrink it.

## Step 1 - add the model to the harness

Edit `experiments/regen-model-sweep/sweep.py`:

- Add a `ModelSpec` to `MODELS`. Fields: `key`, `loader`, `repo`, `points`
  (list of `(strength, num_inference_steps)`), `guidance`, plus optional `filename`,
  `components_repo`, `variant`, `dtype`, `offload`.
- Reuse an existing loader if one fits (`pretrained`, `gguf_single_file`,
  `sdxl_lightning`, `qwen_img2img`, `sd3_img2img`, `zimage_img2img`, `chroma_gguf`); add
  a new `elif spec.loader == ...` branch in `load_pipeline` only if needed.
- Set `guidance` / steps / dtype from the **model card** (do not guess - wrong guidance
  or steps on a distilled/flow model gives garbage that looks like a real result). Turbo
  models are usually guidance 0; full models 3-7.5; distilled ~1.
- Pick `points` so the lowest strength gives `>= 1` effective step
  (`eff = int(steps * strength)`), and so strengths bracket the SDXL knee (0.05-0.15)
  plus a couple of higher ones. For distilled/few-step models raise `steps` to get
  distinct effective-step points (low `steps` collapses every low strength to 1 step).

Then dry-check it constructs: `.venv/bin/python -c "import sweep; print(sweep.MODELS)"`.

## Step 2 - run the lossiness sweep

```
export HF_TOKEN=$(cat ~/.cache/huggingface/token)   # some repos are gated
.venv/bin/python -u sweep.py --models <newkey>
```

- Inputs: the SynthID-bearing set in `reference-images/synthid-verified/` (+ the
  `images/*-synthid.png` hard cases). Outputs land in `outputs/<key>/`; metrics write
  **incrementally** to `outputs/metrics.csv` (a killed/OOM run keeps completed rows).
- It records `diff_mean`, `psnr`, `seconds`, `peak_mem_mb` (RSS + MPS-allocated) per run.
- **The metric is a necessary screen, not a verdict.** Mean-diff averages over the whole
  image and HIDES localized destruction (Flux/Z-Image had acceptable mean-diff yet
  destroyed text/faces). Never decide on the metric alone.

## Step 3 - visual fidelity inspection (mandatory)

Open the outputs side-by-side with their `_input_*.png` and look specifically for, in
this order: **text legibility, faces, then overall structure**. These are the failure
modes seen so far - check each new model against them:

| paradigm | observed failure | examples (2026-08-11) |
|---|---|---|
| flow (rectified-flow DiT) | destroys text, then faces, then the rest | Flux.1-schnell, Z-Image-Turbo |
| distilled SDXL (few-step) | structural artifacts (grid / "stained glass") or hallucination | SDXL-Turbo, SDXL-Lightning |

**Reject on sight** if text/faces/structure are visibly damaged, regardless of the
mean-diff number. Only models that look faithful proceed to clearance. (Log the verdict
in the research doc's findings - it is evidence about the paradigm, not just the model.)

## Step 4 - clearance check (the decisive test)

This is the only test that answers "does it actually remove SynthID", and it is manual:
Google's "Verify with SynthID" in the Gemini app (~10 checks/day/account, noisy - do not
over-read a single verdict; sweep across days).

For each model that passed Steps 2-3, take its **lowest visually-acceptable strength**
output (and 1-2 above it) and run them through the verifier. Find the **minimum clearing
strength**. Batch around the ~10/day budget; prioritize the most promising candidates.

## Decision criteria

Adopt a new model only if ALL hold:
1. **Visually faithful** at the tested strength (Step 3 passes - no text/face/structure
   damage).
2. **Clears SynthID** at strength `<=` SDXL-base's clearing strength (Step 4), i.e. less
   or equal loss. OR same clearing strength but materially faster / lighter / with a
   better native-runtime story (Step 0.3 + the `seconds`/`peak_mem_mb` columns).
3. **Ship-ready runtime exists** (CoreML on mac, sdcpp on CPU; per `ai-integration-tenets`
   - no Python at runtime). A Python-only winner triggers a Phase B port effort, not a
   ship.

Otherwise reject (with the reason recorded). The default stays SDXL-base.

## Reference: diffusers gotchas (from the first round)

See the "Methodology / diffusers gotchas" section of `regen-img2img-model-research.md`.
The recurring ones: align image dims to %16 round-to-nearest (so every model emits
identical dims); pass explicit `height`/`width` (Flux/Chroma/Z-Image default to square);
GGUF transformers need `GGUFQuantizationConfig(compute_dtype=bf16)`; quantized models
must use `model_cpu_offload` (sequential offload raises `KeyError(None)`);
`from_single_file` needs a resolved `hf_hub_download` path; SDXL fp16 needs the
fp16-fix VAE; distilled models' low-strength points collapse to 0 effective steps.

## Cadence

Re-run this protocol when a new model lands that passes the Step-0 filter AND has a
chance of beating the bar (different paradigm, or much smaller/faster, or a quantized
variant of a previously-too-big model). Do not re-test models in a known-failing paradigm
unless something changed (e.g. an MMDiT where only flow models existed before).
