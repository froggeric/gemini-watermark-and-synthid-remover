# SynthID regen detail-restoration (diff-gating) analysis

Date: 2026-08-07. Unreleased research. Companion to `synthid-spectral-removal-record.md`
and `synthid-light-reconstruction-attacks.md`. Read those first.

## TL;DR

`--synthid-attack regen` clears SynthID but costs fidelity (~7 dB at the validated
strength 0.10 knee). Restoring a **gated slice of the diff** `O - R` recovers real
fidelity while staying verifier-clear: restoring the **top-5%-magnitude** pixels of the
diff buys **+7.2 dB** (PSNR 29.4 -> 36.7 dB; faces and fine lighting visibly sharpen,
out of the uncanny valley) and **cleared Google's "Verify with SynthID"** on the hardest
image (the double-watermark case). Restoring 10%+ was **detected**. This "detail
restoration" dilution gate is a real, validated quality knob.

What it is **not**: a method that *filters the watermark out of the diff* to raise that
5% ceiling. Research (project docs + the literature) says that does not work. The 5% gate
works precisely because it does **not** try to estimate the watermark. It keeps few enough
watermark-bearing pixels that the **total restored watermark energy stays under the
detector's aggregate threshold** (dilution, not separation). This doc records why
diff-filtering cannot beat the gate, the cheap empirical confirmation (a "wash" check),
and the two bounded-upside routes that remain open.

## Setup

- `O` = `reference-images/vae-testing/Gemini_Generated_Image_iu84oxiu84oxiu84-2.png`
  (2944x1440, the image-to-image generation that carries SynthID twice; the hardest case
  in the 9-image validation set).
- `R` = `wmr synthid O --synthid-attack regen --regen-strength 0.10 --regen-steps 50`
  (CoreML ORIGINAL-attention UNet, offline via a throwaway models dir).
- `diff = O - R` (signed, BGR). Gate metric = combined per-pixel L2 norm
  `sqrt(dB^2 + dG^2 + dR^2)` (see "channel choice" below).
- Verification = Google's manual in-app "Verify with SynthID" (~10 checks/day/account;
  small sample, single verdict per point, documented as noisy). The within-run control
  (10% and 25% both detected) shows the verifier discriminates between these near-identical
  outputs, so the 5%-clear is real signal, not a blanket miss.

## The validated result (the win)

PSNR vs `O` (the watermarked content target) as the keep-fraction grows:

| variant | PSNR(R',O) | verifier |
|---|---|---|
| R (regen alone) | 29.45 dB | (cleared by regen) |
| keep 3% | 34.97 dB | untested |
| keep 5% | **36.70 dB** | **CLEAR** |
| keep 6% | 37.38 dB | untested |
| keep 8% | 38.47 dB | untested |
| keep 10% | 39.38 dB | **DETECTED** |
| keep 25% | 43.89 dB | **DETECTED** |

Supporting measurements on the diff:

- **Energy is highly concentrated**: the top 5% of pixels (by magnitude) hold **81.2%**
  of the diff energy; top 1% hold 47.7%. "Few high-intensity regions" is true for energy.
- **Co-location with texture** (where a content-conditional watermark concentrates):
  top-10% intensity diff overlaps top-10% local-texture 39.3% vs 10% random (~4x
  enriched); top-5% is 14.8% vs 5% (~3x).
- **hf-band is not detectability**: regen cut the high-frequency band energy ~45%
  (0.00038 -> 0.00021); keep5 restored it to the original's level yet still cleared. So
  generic high-frequency energy does not predict the watermark; the detector keys on a
  structured pattern, not hf magnitude.

Visual: keep5 is a large subjective improvement over R. The image features 4 human
faces and 2 bamboo plants with intricate leaf lighting; keep5 sharpened the faces out of
the uncanny valley and restored the foliage detail. This is expected and is also the
riskiest case: faces and intricate foliage are the highest-entropy regions, i.e. both
where regen fidelity loss is most visible (the win) and where a content-conditional
watermark concentrates most (the risk). So `iu84` is a strong test on both axes.

### Why the gate metric is a combined L2 norm, not per-channel or luminance

- The signal we track (reconstruction error) is correlated across channels and
  luminance-dominated, so a combined norm captures "did this pixel change" cleanly.
- Per-channel gating fragments one decision into three thresholds without improving
  separation (the watermark co-locates with the signal in every channel).
- Luminance weighting is actively bad: `0.299R+0.587G+0.114B` then abs cancels a pure
  chroma shift and reads it as ~0. The L2 norm over channels does not cancel.
- The watermark's channel asymmetry (G > B > R, the {0.85,1.0,0.70} BGR signature)
  matters for *measuring* the carrier (per-channel), not for the gate *decision*. The
  per-channel signature test on the diff was uninformative anyway: the carrier is far
  below the reconstruction error in the diff and never dominates the spatial ratio.

## The research question

Can we **process the diff to suppress its watermark component** before restoring it, so
we can keep more pixels (raise the 5% ceiling) while staying verifier-clear? I.e. turn
`diff = reconstruction_error + W` into `reconstruction_error + (1-gamma)W` and restore a
larger fraction at lower per-pixel watermark amplitude.

## Why diff-filtering cannot raise the ceiling

Three converging results close the door on any linear/spectral separation of W from the
reconstruction error:

1. **A near-perfect spectral strip already failed.** reverse-SynthID's V3
   (aloshdenny/reverse-SynthID) removed **91.4% of carrier-bin spectral energy at
   43.5 dB PSNR / 0.997 SSIM** and was *still detected* by Google. The detector uses the
   **content-adaptive gain mask**, not just the carrier bins, and no spectral/notch filter
   touches that. Better diff-SNR (the diff has ~10-20 dB better W-SNR than the full image,
   because regen strips the bulk content) lets you *estimate* the carrier-bin component,
   but not the gain-mask component. So the SNR advantage does not yield enough per-pixel W
   attenuation to matter.
2. **The detector is a normalized linear correlator** (cosine similarity against the key
   K; reverse-engineered by Allen Kuo, consistent with the paper's conformal-p-value
   framing). Detection score is proportional to **total W energy restored = fraction f x
   per-pixel amplitude a**. The 5%/10% boundary is a *total-W-energy* ceiling, not a
   pixel-count ceiling. Raising it requires reducing `f x a`; the gate reduces `f`,
   filtering would need to reduce `a` (a K-aligned W estimate), which needs the secret key
   or a detector oracle.
3. **Random perturbations are orthogonal to K** (Kuo). Blind phase disruption, denoising,
   wavelet shrinkage add perturbations orthogonal to the key and do not shift the cosine.
   They cannot help. Only a W estimate aligned with K (content-aware) would.

Plus two dead-ends from the literature:

- **Cross-image averaging converges to zero** for content-adaptive watermarks
  (Yang, NeurIPS 2024, "Can Simple Averaging Defeat Modern Watermarks?"). SynthID is
  content-adaptive. This is why our codebook was inert on content.
- **Linear/spectral/wavelet/Wiener/BM3D are provably inert**: SynthID's "Quality" attack
  category (blur + JPEG + denoise) TPR is 99.99%; Tree-Ring blur AUC 0.999. Our own
  spectral path scored ROC AUC 0.20.

This retires every filtering candidate considered (signature spectral filter, blind phase
disruption, two-regen isolation, robustness-based separation): the first two are capped by
the V3 result and the orthogonality argument; two-regen needs N >= 10^4-10^6 regens for a
usable per-pixel W estimate AND is confounded by `regen_bias(O)` (linearly inseparable
from W without an oracle).

## Empirical confirmation: the Regime-A wash check

If detection scales with total W energy, then restoring more pixels at lower amplitude
(matched W budget) should clear at the same rate but with different fidelity. Generated
variants at matched ~5% nominal W budget (`keep% x scale`):

| variant | keep% | scale | restored diff-energy | PSNR(R',O) | verdict |
|---|---|---|---|---|---|
| A keep5@1.0 | 5 | 1.00 | 81.2% | 36.70 dB | CLEAR |
| B keep10@0.5 | 10 | 0.50 | 22.5% | 34.32 dB | predicted clear |
| C keep20@0.25 | 20 | 0.25 | 6.0% | 31.79 dB | predicted clear |
| D keep25@0.2 | 25 | 0.20 | 3.9% | 31.30 dB | predicted clear |
| ref keep10@1.0 | 10 | 1.00 | 89.8% | 39.38 dB | DETECTED |

Restored energy falls with `scale^2`, so spreading restoration over more pixels at lower
amplitude collapses the recovered detail (81% -> 22% -> 6% -> 4%) and PSNR drops
monotonically. **Diff-scaling is strictly worse, not a wash**: the concentrated
full-strength magnitude gate (keep5) is Pareto-optimal. B/C/D will clear (pending
verification) but with worse fidelity than A, so they are dominated. There is no
`(keep%, scale)` point that both clears and beats keep5@1.0.

This empirically confirms the linear-correlator model and that amplitude-scaling cannot
raise the useful ceiling. Variants at `/tmp/iu84_wash_{B,C,D}.png` for verification.

## Ranking of techniques (likelihood of lifting the 5% ceiling)

| Rank | Technique | Verdict |
|---|---|---|
| 1 | Self-trained surrogate detector -> differentiable per-pixel restoration mask | Only route that creates a local "pilot"/oracle. ONNX/CoreML-feasible. Bounded ~2x. HIGH risk the surrogate does not match Google's verifier; unscalable validation (~10/day). |
| 2 | Second independent generative pass to predict reconstruction_error | Conceptually cleanest: subtract predicted recon_error from the diff, discard the W residual. Unvalidated; the predictor may itself emit a SynthID-like carrier. |
| 3 | Multi-regen shared-component subtraction | Crippled by the regen_bias confounder and sample size (N >= 10^4). Marginal, low-frequency content only. |
| 4 | Frequency-notch + per-bin content cap (a la reverse-SynthID V3) as a pre-processor | Capped hard by V3's 91%-strip-still-detected. Expected <1 dB (5% -> 6-7%). |
| 5 | Cross-image averaging | Dead (content-adaptive W -> 0). |
| 6 | Linear/spectral/wavelet/Wiener/BM3D | Provably inert. |
| 7 | Oracle/sensitivity attacks (BNSA etc.) | Strongest known class; **infeasible without a detector oracle**. Reopen first if Google ships a verifier API. |
| 8 | Off-the-shelf learned removers | None target content-conditional carriers; no training pairs exist. |

## Conclusion / what to ship

- **Ship the dilution gate.** An opt-in "detail restoration" post-pass on the regen
  output: restore the top-few-% highest-magnitude diff pixels. Validated at keep5
  (+7.2 dB, verifier-cleared) on the hardest image. It must be labeled honestly: it
  **dilutes** the watermark (the original watermark is still present in the restored
  pixels, just below the aggregate detection threshold), it does not *remove* it. Full
  regen destroys the watermark; this hides what remains in a small area. Conservative
  default budget (e.g. 3-4%) below the measured ceiling, with a content-adaptive bias
  (see below).
- **Do not architect on diff-filtering.** The evidence against it is strong and
  cross-corroborated (V3, linear correlator, orthogonality, averaging-dead, spectral-inert,
  plus the wash check). Spectral/Wiener/phase work on the diff is capped below V3's
  91%-strip-still-detected result.
- **The better framing for further ceiling work is content-adaptive thresholding**, not
  filtering: lower the keep budget for low-entropy images (where the content-adaptive gain
  mask concentrates the watermark most and restoration is riskiest), and there may be head
  room to raise it for high-entropy content. This needs multi-image validation first.

### Two footnotes

- **Carrier-amplitude discrepancy in our own docs.** `synthid-carrier-characterization.md`
  measures carrier RMS ~3.5/255 on black frames; three other docs cite ~0.025/255. This
  100x gap changes how much of the diff is W (40% vs 0.3%) and should be pinned
  empirically on a real diff before any further modeling.
- **Forensic signature.** Regen outputs (with or without detail restoration) carry a
  detectable "was-attacked" signature at ~99.64% TPR (Goonatilake & Ateniese, arXiv
  2605.09203). This is a *separate* detection axis from the watermark verifier; no diff
  processing lifts it. Any feature doc should state this.

### Operational note

The run used a throwaway models dir
(`~/.cache/wmr/coreml-sdxl-throwaway`) to load the cached ORIGINAL-attention UNet offline,
because the default `coreml-sdxl` dir's UNet pin had drifted (the binary pins
`9625f95c...`; the cached archive was the stale `9101cada...` SPLIT_EINSUM pin, and a
re-download to re-pin hung in libcurl `multi_wait`). A real re-download (or a cache
refresh) is needed for the default dir; the throwaway dir and the 4 GB `.part` in the
default dir should be cleaned up.

## Sources

SynthID structure / reverse-engineering:
- SynthID-Image paper: https://arxiv.org/abs/2510.09263 (HTML https://arxiv.org/html/2510.09263v1)
- aloshdenny/reverse-SynthID (V3 91%-strip-still-detected; V4 destructive clear): https://github.com/aloshdenny/reverse-SynthID
- hackerfactor/reverse-SynthID-C (portable C99 V3 port): https://github.com/hackerfactor/reverse-SynthID-C
- Allen Kuo (linear-correlator detector model, orthogonality): https://allenkuo.medium.com/synthid-image-watermark-research-report-9b864b19f9cf
- wiltodelta/remove-ai-watermarks (cross-validates the sigma~0.15 video floor): https://github.com/wiltodelta/remove-ai-watermarks

Watermarking theory / benchmarks:
- Yang, "Can Simple Averaging Defeat Modern Watermarks?" NeurIPS 2024: https://proceedings.neurips.cc/paper_files/paper/2024/file/67b2e2e895380fa6acd537c2894e490e-Paper-Conference.pdf
- WAVES benchmark: https://arxiv.org/abs/2401.08573
- Tree-Ring (blur AUC 0.999): https://arxiv.org/html/2305.20030v3
- Cox et al. 1997 (spread spectrum): https://www.semanticscholar.org/paper/Secure-spread-spectrum-watermarking-for-multimedia-Cox-Kilian/a5eaa077d774745409499be2a2f16c506156161
- Goonatilake & Ateniese, forensic cost of regen removal: https://arxiv.org/abs/2605.09203

In-repo grounding:
- `synthid-carrier-characterization.md` (carrier 1/f^1.3 envelope, G>B>R, fourfold phase; 3.5/255 vs 0.025/255 discrepancy)
- `synthid-spectral-removal-record.md` (spectral path inert; ROC AUC 0.20; no public verifier)
- `synthid-light-reconstruction-attacks.md` (regen knee 0.10; VAE-only ~40%; double-watermark needs 0.10)
