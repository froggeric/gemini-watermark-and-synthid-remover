# Band-split detail restore: evaluation and ship decision (2026-09-25)

Decision record for the default post-regen detail restoration shipped in 1.17.1:
`out = O - GaussianBlur(O - R, sigma)` at sigma 1.95, replacing the Wiener +
top-5% sparse restore as the default (which remains available at
`--regen-restore-band-sigma 0`).

## Origin

The formulation comes from an external investigation of two SynthID-removal
projects (both unlicensed; the idea was reimplemented, no code copied):

- **0xROOTPLS/DeSynth** (55 stars, one-weekend June 2026, no license): Qwen-Image-2512
  GGUF Q4 + lightx2v Lightning LoRA, 8 steps at denoise 0.25 (2 actual steps),
  then `blur(R, 1.95) + (O - blur(O, 1.95))`. Their magic numbers ("SynthID
  spatial cutoff ~2px", "0.25 is the minimum that defeats SynthID") are asserted,
  not derived; their evidence base is n=1-2 self-measured images.
- **00quebec/cebeuq Synthid-Bypass V2** (956 stars, active, no license; DeSynth's
  upstream): ComfyUI, the same Qwen backbone plus a DiffSynth Canny controlnet
  for structure and a Z-Image-Turbo face detailer. The global redraw runs 4
  steps at a RESOLUTION-ADAPTIVE denoise in [0.08 @ 0.3 MP, 0.15 @ 3.7 MP]
  (`SynthidBypassAdaptiveDenoise`), the face path at 0.05-0.35 adaptive.

Two facts from that investigation mattered here. First, the knee triangulates:
our validated 0.10 @ 50 steps sits inside the upstream's [0.08, 0.15] range
(DeSynth's 0.25 is the no-structure-lock outlier; their full-HF restore
re-imports more carrier, which plausibly forces the extra headroom). Second,
the band-split transplant decouples backbone fidelity from removal: the
distillation/flow damage of a few-step backbone lives in the high frequencies,
which the transplant discards, so a backbone can be chosen for
carrier-destruction-per-compute. That reopens backbone selection (Qwen-Image
2512 is Apache-2.0 and supported by our vendored sdcpp; Qwen-Image 2.1 is 3x
smaller and text-stronger but non-commercial and not runnable under 32 GB).

## The A/B

Implementation: `RestoreConfig::band_sigma` in `src/core/regen_restore.{hpp,cpp}`
(the branch runs after the luminance gate; `O - blur(D, sigma)` is algebraically
identical to DeSynth's `blur(R) + O - blur(O)`). Sweep through the real binary
(`wmr synthid`, CoreML backend, 0.10 @ 50, seed-fixed and byte-deterministic;
verified stable across e5rt cache recompiles) on the 6 bright images of the
clearance corpus, plus forced-On runs on the darkest image:

- SSIM recovery over pure regen: sigma 1.95 recovers ~2x the shipped Wiener +
  top-5% restore on every image (e.g. iau18v +0.062 vs +0.018; iu84 +0.057 vs
  +0.032). sigma 2.90 adds +0.001-0.009 more. sigma 0.95 is dominated.
- PSNR is mixed by construction (the band residual is the low-frequency
  `blur(D)`, which MSE punishes and SSIM forgives).
- Visual QC (independent vision passes on O | keep5 | band panels, 6/6):
  band-split preferred everywhere, decisive on photographic texture (keep5
  leaves waxy 95%-un-restored regions and misses low-contrast strokes below the
  top-5% magnitude cut); zero band-split artifacts (no halo, drift, ghost, or
  color shift) anywhere.
- sigma 1.95 vs 2.90 (3 textured images): perceptually tied.
- Dim forced pair (e3hcto, luma 25): band 34.97 dB / 0.860 SSIM vs keep5
  34.58 / 0.832 (both moot; see below).

Harness notes for reproduction: the pure-regen baseline must pass
`--no-regen-restore-detail` explicitly (default Auto restores on bright images);
the restore branch logs go to stdout, not stderr; artifacts and scripts live
under `dist/bandsplit/` (gitignored).

## Verifier rounds (Google "Verify with SynthID", manual)

| check | verdict |
|---|---|
| band195 iu84 (~1 MP, hardest validated case) | clear |
| band290 iu84 / 5nvcl6 / dance-summer / iau18v (~1 MP) | clear (4) |
| band290 poster-artnight (1792x2400, luma 139.8, restore applied) | clear |
| band195 volcanic (near-white) | inconclusive (content class) |
| band290 x1hsoq (dense text) | inconclusive (content class) |
| band290f gemini-pro-paid (dim luma 87.2, FORCED) | **detected** |
| band195f gemini-pro-paid (forced) | **detected** |
| band095f gemini-pro-paid (forced) | **detected** |
| band095f gemini-36-paid (dim luma 85.1, forced) | clear |
| band195f gemini-36-paid (forced) | **detected** |

Reading:

- **Bright content clears at both megapixel regimes** (up to 4.3 MP), so the
  flat 0.10 knee plus band-split holds where the upstream felt compelled to
  scale denoise with resolution.
- **Sigma direction (easy to get backwards): larger sigma restores MORE of the
  original** (sigma -> inf is the watermarked O, sigma -> 0 is pure R). A
  band290 clear therefore implies band195 safety on the same image (monotone in
  restored carrier energy).
- **The luminance gate is mandatory for the band family too**: forcing past it
  on dim content detected at sigma 0.95, 1.95 and 2.90 on one image. The cliff
  is per-image (a sibling image at the same luminance cleared at sigma 0.95 and
  detected at 1.95), so dim restore is unreliable rather than uniformly
  impossible; with no in-process verifier there is no safe way to select the
  images that tolerate it. The gate (full regen below luma 128) stays.

## Decision

Default `band_sigma = 1.95`: near-all of sigma 2.90's fidelity gain (the 195 vs
290 difference is imperceptible), strictly less original transplanted than any
config that cleared, and the value with independent external corroboration.
Ship evidence: default-path output is byte-identical to the verifier-cleared
band195 artifact; 193/193 unit and integration tests pass.

Follow-ups left open: a safe dim-content restore (per-image, oracle-paced
research), backbone re-selection under the transplant insight (Qwen family),
and a possible sigma-vs-resolution study (sigma is in pixels, so bigger images
restore proportionally more of the characterized carrier band; empirically
cleared at 4.3 MP, untested above).

## Sources

- Upstream projects: github.com/0xROOTPLS/DeSynth, github.com/00quebec/Synthid-Bypass
- Carrier characterization (r^-1.3 envelope, data band r=30-400):
  `synthid-carrier-characterization.md`
- The sparse-restore study this builds on (dilution gate, luminance cliff):
  `synthid-diff-restoration-analysis.md`
- Regen knee validation: `synthid-light-reconstruction-attacks.md`,
  `synthid-regen-validation.md`
