# External tool review: Loyal Bear "SynthID Scrambler"

Date: 2026-08-29. Repo: `BovineOverlord/Loyal-Bear---The-SynthID-Scrambler`
(Python/diffusers, 6 commits all on 2026-08-22). Reviewed because it claims a secret SynthID removal
method. Precedent: `synthid-external-codebook-test.md` (the `aloshdenny/reverse-SynthID`
review). No source or defaults were changed. Analysis only.

## The "secret" method

There is no secret. The whole backend is a 76-line `src/pipeline.py`: a diffusers
`StableDiffusionXLImg2ImgPipeline` on **CPU at float32** with `EulerDiscreteScheduler`,
strength 0.05 ("Light", the default) or 0.10 ("Strong"), `num_inference_steps=100`
hardcoded (so 5 or 10 actual denoise steps), then a PIL re-encode that drops metadata.
No detection, no carrier analysis, no perturbation step, nothing else. Model =
`epicrealismXL_pureFix.safetensors` (a photoreal SDXL fine-tune) fetched **unpinned**
from `phuaqu/zimage111`, a personal model stash (0 downloads; other siblings are NSFW
fine-tunes). The release build Cython-compiles the same source to a `.pyd`; the
committed `__pycache__` bytecode constants match the `.py` files, so the public source
is the truth.

## Cross-check against our validation data

Their two strengths, mapped onto our step-collapse model (N is granularity, strength
governs removal):

- "Strong" 0.10 @ N=100 = 10 steps over span [0.9, 1.0]: **removal-equivalent to our
  validated default** (0.10 @ 50 = 5 steps over the same span).
- "Light" 0.05 @ N=100 = 5 steps over span [0.95, 1.0]: a true strength-0.05 attack,
  **inside the band our sweep showed to be unreliable** (0.04-0.08 cleared singles,
  missed the double-watermarked image; see `synthid-light-reconstruction-attacks.md`).
  Their default mode is, by our data, an under-removal setting, and the repo contains
  no verifier loop, fixtures, or results that would ever catch it.

Weak but real corroboration: independent arrival at deterministic Euler + 0.10 as the
working point, consistent with our published knee.

## Claims vs code

- "System that alerts me to compromise attempts": **false in source.** The only egress
  is pip installs and the HF model download. No telemetry exists.
- "Works on all OpenAI and Gemini images as of July 2026": zero test artifacts in the
  repo. Unverifiable. For metadata-based provenance the PIL re-encode is trivially
  sufficient; for a resilient watermark the only mechanism present is the same SDXL
  regen.
- "8 GB of RAM": fp32 SDXL is ~14 GB of weights alone; it will thrash on 8 GB.
- Bootstrap hardcodes `.venv/Scripts/python.exe`, so the documented Linux/macOS run
  crashes before the app starts (Windows-only in practice).
- Every input is squeezed to 1024x1024 and stretched back (aspect-ratio distortion,
  two LANCZOS resamples); the GUI's `steps` parameter is dead code; the
  `gradio>=4.44.0` floor is wrong (exactly 4.44.0 lacks `launch(theme=, css=)`, it
  only works because a fresh install resolves to a newer gradio).
- License: custom, individuals only, corporations forbidden from use **and
  examination**. Not OSI. Moot for us; we copy nothing.

## Verdict

Nothing to adopt: no validation, no detection, no native resolution, no GPU, no video,
unpinned model supply chain. The one untested idea it raises: an SDXL **photoreal
fine-tune** (epicrealismXL) vs SDXL-base as the regen backbone, which our Phase A sweep
(`regen-img2img-model-research.md`) never covered; a cheap A/B in the
`regen-model-evaluation-protocol.md` harness if backbone fidelity is ever revisited.
Caveats: content-class dependent, murky checkpoint licenses, ~6.9 GB second download.
Until then: the community-known method is the method, which is what our README already
says openly.
