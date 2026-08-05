# Light-reconstruction attacks on SynthID (VAE round-trip + upscaling)

Date: 2026-08-05. Research toward a SynthID attack lighter than full SDXL regen.
Five parallel research agents (VAE attack, upscaler candidates, native runtimes,
reconstruction-attack literature, latent-perturbation sweep). Read with
`synthid-spectral-removal-record.md` and `synthid-detection-feasibility.md`.

## The question

Full SDXL img2img regen (strength ~0.10) is the only validated SynthID-Image
removal wmr has. It is slow and lossy. Can a LIGHTER reconstruction defeat
SynthID? Two candidates: (A) a VAE/autoencoder encode-decode round-trip, (B) a
strong neural super-resolution / restoration round-trip.

## Bottom line (honest prediction)

**Both light vectors are predicted to FAIL on SynthID.** Three independent lines
of evidence converge on this:

1. **Google hardened SynthID against exactly this.** SynthID-Image paper §6.2:
   "we specifically tested and ensured robustness against off-the-shelf weak
   re-generation attack models (e.g., using variational autoencoders)." Diffusion
   is named separately (§6.1) as the threat they concede.
2. **The closest robust analogue survives far heavier attacks.** StegaStamp (a
   post-hoc deep-learning watermark, weaker than SynthID on every benchmark in the
   SynthID paper) survives a 70-step diffusion regen (CtrlRegen Table 1, TPR 0.99)
   and survives Real-ESRGAN / AdcSR upscaling (SRAttack: 22% and 36.7% BER, both
   below the 45% destruction threshold).
3. **Direct VAE-vs-SynthID evidence (video).** wiltodelta/remove-ai-watermarks
   oracle-tested a VAE + latent-noise attack on Google Veo: `noise_std=0.10` was
   still SynthID-detected; `0.15` was clean. So even VAE-plus-substantial-noise
   needs ~0.15, and pure round-trip (noise 0) is strictly weaker.

The deeper finding: **every "light" attack that approaches working is itself a
diffusion model** (AdcSR is a diffusion-SR hybrid; CtrlRegen is clean-noise
diffusion; VAE+latent-noise at sigma>0 is literally img2img at strength sigma^2).
So the real open question is not "VAE vs upscaler vs diffusion" but "what is the
lightest diffusion operating point that still removes SynthID", which is wmr's
existing `--regen-strength` knob.

## Why a pure VAE round-trip is expected to fail

A round-trip introduces ~10/255 RMS distortion (SDXL VAE, ~24.7 dB PSNR), about
400x the SynthID amplitude (~0.025/255). That sounds destructive, but the
distortion is the wrong frame: the VAE's error concentrates in high-frequency
texture the perceptual+GAN loss treats as free, while the watermark is a
content-conditional pattern optimized to live in features JPEG/resize survive
(and thus the VAE preserves). High VAE distortion does not equal watermark
removal; only collapse of the watermark direction does, and a stationary VAE
bottleneck is not adversarial to the extractor.

The moment you add latent noise to tip a round-trip into removal, you are running
img2img: the Zhao (NeurIPS 2024) regeneration attack is
`decode(encode(x) + N(0, sigma^2))`, and sigma>0 maps to img2img strength
sigma^2. There is no third regime between "pure round-trip" and "low-strength
regen".

## Why upscaling is expected to fail

SRAttack (OpenReview i05MM4h1WZ, a non-peer-reviewed preprint) directly benchmarked
Real-ESRGAN and AdcSR as watermark attacks. On StegaStamp: Real-ESRGAN 22% BER,
AdcSR 36.7% BER (sub-threshold). On older frequency watermarks (DwtDct/DwtDctSvd)
they reach ~50% (full destruction). Critical implementation detail: **upscaling
alone is a no-op** (2.15% BER on StegaStamp without pre-degradation); the attack
requires a pre-degradation step (the recipe is bicubic downscale to 0.25x, SR x4,
bicubic back to original). The pre-degradation does the damage; the SR regenerates.
And the strongest SR-shaped attack (AdcSR, CVPR 2025) is itself a diffusion model.

## The wmr scheduler quirk (verified in code)

`coreml_sd_scheduler.cpp:111`: `t_start = N - int(N * strength)`, clamped to at
least 1 step. At the default `N=20`, `int(20*s)` is 0 for s<0.05 and 1 for
0.05<=s<0.10, so **every strength below 0.10 collapses to a single Euler step at
the lowest-noise timestep** (sigma_peak ~0.041). Strengths 0.05 and 0.07 are
identical. The cliff to the validated 2-step removal (sigma_peak ~0.229) happens
abruptly at s=0.10. Implication: you cannot dial a "lighter than 0.10" regen by
lowering strength at N=20. To map the knee smoothly, vary N as well as s (e.g.
N=10 spreads the schedule so a single step lands at a higher-noise timestep).

## What is still genuinely worth testing (and why)

Both vectors are cheap to test and fill a real gap: **no one has published a
VAE-roundtrip or upscaler measurement on SynthID stills.** Google's VAE-robustness
is a one-sentence claim; wiltodelta's number is video. A null result is itself
valuable (justifies not shipping these as features, and is a publishable
negative). And there is a small chance stills differ from video.

The cheapest, highest-information test order:

1. **Pure VAE round-trip (sigma=0) on stills.** Free on mac: the fp16-fix VAE
   encoder+decoder are already separate MLModels in `coreml_sd_pipeline.mm`; a
   `vae_round_trip()` method is ~30-50 lines (encode, take `mean * 0.13025`, skip
   the UNet loop, decode). Expected null.
2. **VAE + latent-noise sweep** (sigma = 0.03, 0.06, 0.10, 0.15). Same method,
   one Gaussian add before decode. Confirms the continuum and finds the cliff
   (expected ~0.15 per wiltodelta). This is identical to img2img at strength
   sigma^2, so it also cross-checks the regen path.
3. **Real-ESRGAN with the SRAttack pre-degradation recipe** (downscale 0.25x ->
   SR x4 -> downscale back). NCNN integration mirroring `ai_denoise`
   (~2-3 days; model from xinntao/Real-ESRGAN-ncnn-vulkan, MIT, ~64 MB). Expected
   null on SynthID.
4. **img2img knee-mapping varying N** (e.g. N=10/20 at s=0.05/0.07/0.10). Uses the
   existing regen path; finds the lightest diffusion point that clears SynthID.

## Measurement protocol (no public verifier)

- **Removal (ground truth):** Google's online "Verify with SynthID" check
  (~10/day/account). Run all sweep cells from one source image in one day's quota.
- **Removal (proxy, continuous):** a tiny binary classifier trained on
  self-generated clean/watermarked Gemini pairs (the fyxme surrogate approach, AUC
  ~0.9999 on their own data). Not ground truth, but gives a continuous score
  between Google checks.
- **Fidelity:** PSNR, SSIM, LPIPS, and an eyeball check on colorful natural
  content (synthetic/gray content hid the Phase-3 VAE bug; do not skip this).
- **Report the knee** as the lowest-perturbation cell where Google returns "no
  watermark" AND LPIPS is at or below the validated s=0.10 regen's LPIPS.

## Test inputs

Use the Google-verified set already in the tree: `reference-images/synthid-verified/`
(5 SynthID-positive stills) as attack inputs; check whether each attack output
clears Google's detector. `reference-images/synthid-free/` (Google-confirmed
clean) are the negative controls.

## Forensic caveat

Regeneration attacks (light or heavy) leave a detectable footprint: attacked-image
classifiers reach AUROC 0.998-0.9999 (arXiv 2605.09203). Lighter reconstruction is
NOT stealthier (CtrlRegen, the heaviest, is the most detectable). So a lighter
removal does not buy forensic stealth; it only buys speed and fidelity.

## Native runtimes (drop-in to existing wmr backends)

- **VAE round-trip:** existing CoreML SDXL pipeline (mac); sdcpp strength=0 or a
  thin VAE-only entry (linux/windows). Zero new models on mac.
- **Real-ESRGAN:** NCNN+Volk via the `ai_denoise` pattern (linux/windows/mac-x86_64);
  CoreML `.mlpackage` via the `migan_coreml_inpainter.mm` pattern (mac). MIT.
- **NAFNet / SwinIR:** ORT via the `migan_inpainter.cpp` pattern. Apache/MIT.
- **GFPGAN / CodeFormer:** NON-COMMERCIAL license. Hard blocker for a shipped
  binary; do not implement without an explicit owner decision.

## Key sources

- SynthID-Image paper (§6.2 VAE-hardening claim): arXiv 2510.09263
- Zhao et al., regeneration attack (VAE + diffusion instances): arXiv 2306.01953,
  code github.com/XuandongZhao/WatermarkAttacker
- CtrlRegen (StegaStamp survives 70-step regen, TPR 0.99): arXiv 2410.05470
- SRAttack (Real-ESRGAN/AdcSR vs watermarks; StegaStamp sub-threshold): OpenReview i05MM4h1WZ
- wiltodelta VAE+noise vs SynthID video (sigma 0.15 floor):
  github.com/wiltodelta/remove-ai-watermarks/blob/main/docs/synthid.md
- Forensic stealth of regen attacks (AUROC 0.9999): arXiv 2605.09203
- WAVES benchmark (public-VAE watermarks removable; in-latent vs post-hoc): arXiv 2401.08573
- fyxme SynthID attack study (only diffusion worked): fyx.me/articles/...
- SDXL fp16-fix VAE fidelity (LPIPS 0.056, SSIM 0.73): huggingface.co/madebyollin/sdxl-vae-fp16-fix

---

## Empirical results (2026-08-05)

Validated against Google's official "Verify with SynthID" detector. macOS Apple Silicon,
CoreML backend, SDXL fp16-fix VAE. Test set: 9 varied Gemini-generated images (posters,
mockups, AI art), including one image-to-image generation that carries SynthID twice.
Each data point was confirmed over two rounds of detector checks. The detector is
rate-limited (~10 checks/day), so this is a small, varied sample, not large-scale.

### Removal knee (diffusion regen, N=50)

| strength | denoise steps | result |
|----------|--------------|--------|
| VAE round-trip (0 diffusion) | 0 | cleared ~2/5 images (~40%) - not reliable |
| 0.02 | 1 | detected |
| 0.04 | 2 | detected on some images (content-dependent) |
| 0.06 | 3 | "unsure" on the hardest (uniform-color) image |
| 0.08 | 4 | cleared 7/8 singly-watermarked images; failed only the double-watermarked one |
| **0.10** | **5** | **cleared 9/9, including the double-watermark case - the validated default** |

PSNR versus input at strength 0.10 ranges ~29 to 41 dB across the set (busy content
changes more; smooth content less). Fidelity is content-dependent, not a single number.

### VAE-only round-trip (0 diffusion): not reliable

The pure VAE encode-decode round-trip (no diffusion, `--regen-vae-roundtrip`) cleared
about 2 of 5 images (~40%). It sits right on the detector's decision boundary:
content-dependent, not a usable scrubber. This confirms, for stills with the SDXL
fp16-fix VAE, Google's stated section-6.2 hardening against VAE-based regeneration on
average. The images it did clear were borderline cases, not a method. The
`--regen-vae-roundtrip` flag is kept on the `synthid-vae-roundtrip` branch as a research
tool; it is not a shipping feature.

### The double-watermark edge case

One test image was an image-to-image generation (a Gemini output used as the source for
another Gemini generation), so SynthID was applied twice. It resisted lighter strengths
(failed at 4-step / 0.08) and required 0.10 (5-step) to clear. This is the reason the
default cannot safely drop below 0.10: the rare double-watermark case needs the full
strength. Normal (singly-watermarked) Gemini images clear at 0.08 (4-step) with slightly
higher fidelity, but a default must cover the worst case.

### Conclusion

- **Default strength 0.10 is validated as the minimum that clears 100% of the varied set**
  (9 images, two rounds, including the double-watermark edge case).
- Lighter strengths (0.04-0.08) clear normal singly-watermarked images but are
  content-dependent and miss the double-watermark case; not safe as a default.
- VAE-only (0 diffusion) is unreliable (~40%); not shipped.
- Output fidelity at 0.10: ~29 to 41 dB PSNR, content-dependent.

### Note on N (the step base)

SynthID removal is governed by `strength` (the starting noise level in the schedule), not
by N (the denoise granularity). The 9-image validation used N=50 (5 actual steps at
s=0.10). The shipped default starts at the same noise level (s=0.10 -> ~90% of the
schedule) regardless of N, so it is removal-equivalent; a higher N only buys marginally
finer reconstruction (slightly better fidelity) at higher compute cost.
