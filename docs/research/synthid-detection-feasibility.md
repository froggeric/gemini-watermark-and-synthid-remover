# Can SynthID be reliably detected? A feasibility study

Date: 2026-08-05. Deep research across GitHub, HuggingFace, and the academic
literature (8 search angles, 14 sources deep-read, 16 claims adversarially
verified). Pairs with `synthid-detection-validation.md` (our own detector scored
against Google labels: ROC AUC 0.20). Read `synthid-investigation-summary.md`
first for broader context.

## TL;DR

**No reliable third-party detection of SynthID-Image exists, and none is
currently achievable by us.** Every public attempt falls into one of four boxes,
all of which fail:

1. Needs Google's private key or extractor (the official text detector; the
   private image verifier).
2. Spectral codebook detection: empirically dead on content images (our AUC 0.20;
   reverse-SynthID's "90%" is an unvalidated self-assessment with no control set).
3. Neural surrogate detector: the one public attempt is confounded (it detects
   "is this a Gemini/AI image", not the watermark) and its attacks do not transfer
   to Google's real detector.
4. Theory says a classifier could in principle distinguish a post-hoc watermark,
   but training it needs labeled watermarked-vs-unwatermarked pairs from the SAME
   generator, which do not exist publicly because Gemini always watermarks.

A striking meta-finding: even Google's own consumer "Verify with SynthID" tool is
unreliable (the Hacker Factor blog caught it hallucinating "not watermarked" on a
known-watermarked image, being right only 2 of 3 times, and fabricating an
analysis of a URL it never fetched). So "ground truth" itself is noisy and
rate-limited (~10 checks/day, natural-language output).

**Recommendation for this project: do not ship a SynthID detector.** It would be
misleading. The honest position stands: wmr does not detect SynthID; it suppresses
it (heuristic) or removes it (validated diffusion regen).

## What reliable detection would require

SynthID-Image is a content-conditional neural watermark (arXiv 2510.09263): an
encoder `f(image, payload) -> image` applied post-generation, detected by a
trained extractor that outputs a conformal p-value. The carrier amplitude is
~0.025/255, below the content noise floor, and the watermark is content-adaptive
by construction. Google publishes the extractor only to "trusted testers" via
partnerships; there is no open, key-bearing, third-party-runnable verifier.

A reliable detector for us would need one of:
- **Google's extractor** (private, unavailable).
- **A learned no-key classifier trained on controlled pairs**: watermarked vs
  unwatermarked images from the SAME generator, so the watermark is the only
  variable. This data does not exist publicly (Gemini always watermarks).
- **A signal that survives the content noise floor** without the extractor. The
  research confirms no such signal is reliably usable (spectral detection is dead
  on content; the only neural attempt is confounded).

## The candidate methods, ranked

### 1. Spectral codebook detection (reverse-SynthID family) — NOT viable on content

Sources: `aloshdenny/reverse-SynthID` (RobustSynthIDExtractor), the
`hackerfactor/reverse-SynthID-C` port, `zero2launch/reverse-SynthID-gpu`.

Method: extract a noise residual (wavelet/bilateral/NLM/Wiener), FFT it, and
phase-match against a codebook of "carrier" frequency bins (a hardcoded grid
reverse-engineered from black/white Gemini references) plus magnitude and
noise-structure gates. No AI, no Google extractor. This is the SAME family as
wmr's own CodebookSubtractor/SynthidDetector.

Verdict (verified): not viable on content images.
- The repo's "Detection_Rate-90%-success" badge has NO evaluation protocol behind
  it: no AUC, no confusion matrix, no held-out balanced set, no non-watermarked
  control directory. `benchmark_detection()` ingests only watermarked images and
  reports a true-positive rate with the codebook built from the same images
  (train/test leakage).
- The fixed-carrier-grid + ">99.5% cross-image phase coherence" premise
  contradicts SynthID's content-conditional neural design (arXiv 2510.09263). The
  coherence was measured on degenerate pure-black/white inputs and on the raw
  image FFT (content-dominated), then circularly (pick top bins by coherence,
  report those bins' coherence).
- The noise-structure-ratio gate (expected 1.32, gate 0.7 to 2.0) equals the
  generic Gaussian-noise baseline (sqrt(pi/2) = 1.253). Any natural image passes
  it.
- Independent confirmation: wmr's structurally-identical detector scored ROC AUC
  0.20 against 8 Google-labeled content images, with positive and negative score
  distributions fully overlapping (see `synthid-detection-validation.md`).

Applicability to a C++ tool: already implemented here (and it does not work).
zero2launch's GPU fork is removal-only; it adds nothing to detection.

### 2. Neural surrogate detector (fyxme/opensynthid-detect) — confounded, not a SynthID detector

Source: `fyxme/opensynthid-detect-0.1` on HuggingFace + the author's write-up.

Method: a ResNet-34 dual-stream classifier (6-channel input: RGB, a wavelet-denoise
residual, FFT log-magnitude, and a fixed "carrier mask"), trained by "model
extraction" with NO Google weights. Labels came from dataset provenance, not from
querying Google's detector. Reported val AUC 1.0.

Verdict (verified): refuted as a SynthID detector.
- Fatally confounded training data: positives are Nano-Banana (Gemini) generated
  images, negatives are MS-COCO real photographs. The model cannot separate "has
  SynthID" from "is a Gemini/AI-generated image." Reaching ~99% in one epoch is the
  signature of a trivial shortcut (generator fingerprint), not a subtle watermark.
  The author themselves wrote "its probably over-fit" and "99% detection after 1
  epoch doesn't seem right."
- The "carrier mask" is a literal constant (image-independent), so after the first
  BatchNorm it contributes zero per-sample information. Disabling it would not
  move accuracy.
- The decisive test: a Carlini-and-Wagner adversarial perturbation that DID fool
  the surrogate did NOT transfer to Google's real verifier. The author concedes the
  surrogate "hasn't captured enough of the internal decision boundary of the real
  model." It learned a different signal than Google's detector.
- Ground-truth-validated positives total exactly ONE image (a single manual Gemini
  check). No non-Gemini-AI control was run (n=1 on a Flux image).

Applicability to a C++ tool: mechanically portable (ResNet-34 + small CNN export
to ONNX, then CoreML on mac or ONNX Runtime/DirectML elsewhere; preprocessing is
C++-expressible, FFT already a dependency). But porting it ships a mislabeled
"is-this-Nano-Banana-generated" detector. The blocker is data, not code: an honest
detector needs watermarked Gemini vs UNWATERMARKED Gemini (same generator), which
is unavailable.

### 3. The official text detector (google-deepmind/synthid-text) — wrong variant, needs the key

Source: `google-deepmind/synthid-text` (Nature 2024, Dathathri et al.).

Method: text-only. g-values (binary, from a keyed hash tournament over n-grams)
are computed from the token sequence; three detectors score them (mean,
weighted-mean, Bayesian posterior). Confirmed text-only: no image/audio/video code
in the repo.

Verdict (verified): unusable for image detection, and unusable by third parties
even for text.
- Detection is strictly per-key. g-values are PRF(ngram, keys); under the wrong key
  they are uniform Bernoulli(0.5), so the score collapses to the 0.5 prior
  (AUC 0.5, TPR equals FPR). The README states "The Bayesian detector must be
  trained for each unique watermarking key." We do not have Google's key.
- It also requires the raw token sequence, not just the visible text.

Value to this project: conceptual only. It confirms there is no official
SynthID-Image verifier, and that official SynthID detection is per-key/per-extractor
by design.

### 4. Theoretical / no-key classifier literature — promising in principle, blocked by data

Sources: "An Undetectable Watermark for Generative Image Models" (PRC, arXiv
2410.07369, ICLR 2025); "A Transfer Attack to Image Watermarks" (arXiv 2403.15365).

PRC paper (verified):
- Theoretical: a post-processing watermark (which SynthID-Image is) cannot be made
  undetectable without extra assumptions; "one can always distinguish between a
  fixed image and any modification of it." So SOME classifier that distinguishes
  SynthID-Image likely exists.
- Empirical: a no-key ResNet18 (trained on 7,500 self-generated watermarked +
  7,500 unwatermarked pairs) detects 8 watermark schemes (DwtDct, DwtDctSvd,
  RivaGAN, StegaStamp, SSL, Stable Signature, Tree-Ring, Gaussian Shading) at
  ~100% held-out accuracy. It CANNOT learn the cryptographically-undetectable PRC
  watermark (stays at 50%). The 50% PRC result is the control proving the others
  are not trivial shortcuts.
- BUT SynthID-Image was never tested. arXiv 2510.09263 explicitly hardens against
  surrogate detectors, and the SynthID authors note paired
  (original, watermarked) data "significantly simplifies attacks, especially for
  post-hoc systems like SynthID-Image."
- Training such a classifier for SynthID needs controlled watermarked-vs-unwatermarked
  pairs from the same generator, which require Google's private encoder or the gated
  verifier.

Transfer-attack paper (verified): a no-box watermark REMOVAL attack (not a
detector), never tests SynthID, and the authors explicitly disclaim transfer to "an
entirely different" method. SynthID's p-value extractor is not a bitstring decoder
the framework maps onto.

### 5. Provenance / metadata (C2PA) — fragile, not pixel detection

SynthID now contributes to C2PA Content Credentials manifests, so an image's
metadata can SAY it carries SynthID. This is a real but fragile signal: a re-save
or any platform that strips metadata defeats it, and it asserts provenance, not
that the pixels still carry the mark. Useful as a hint, not as detection.

## The two findings that matter most

1. **The data blocker is fundamental.** Every honest neural-detection path needs
   watermarked-vs-unwatermarked pairs from the same generator. Gemini always
   watermarks, so unwatermarked Gemini images do not exist publicly. The one public
   attempt (fyxme) failed precisely because it substituted a confounded
   (Gemini vs real-photo) split. Without a Google partnership or an API path that
   can emit unwatermarked Gemini images, no third party can train a reliable
   detector. This is why "model extraction" of SynthID-Image has not succeeded.

2. **Even Google's official consumer tool is unreliable.** The Hacker Factor blog
   documented the Gemini "Verify with SynthID" check hallucinating
   "not watermarked" on a known-watermarked image, scoring only 2 of 3 on a clear
   case, and fabricating an analysis of a URL it never fetched. So the only
   external "ground truth" available is itself noisy, qualitative, and rate-limited
   (~10 checks/day). Any detection claim validated only against this tool (including
   reverse-SynthID's, and our own validation) inherits that noise.

## Recommendation for wmr

- Do not ship a SynthID detector. No public method is reliable, and shipping a
  confounded or unvalidated one would mislead users into trusting a binary
  "watermarked / not watermarked" verdict that is not justified.
- Keep `wmr detect` as visible-watermark-only (the current, correct behavior).
  The SynthID detector stays in the source behind `--codebook` (no codebook ships),
  which is honest by omission.
- The honest product position is unchanged: wmr does not detect SynthID (no public
  verifier exists; spectral detection has no discriminative power; the one neural
  attempt is confounded). It suppresses the carrier (heuristic, `--codebook-free`)
  or removes it (lossy, validated, `--synthid-attack regen`).
- If a genuinely new capability is wanted, the realistic adjacent feature is an
  "AI-generated image" heuristic (the fyxme architecture, retrained honestly and
  labeled as such). That is a different product from SynthID detection and should
  not be marketed as one.
- Revisit only if (a) Google releases a public SynthID-Image verifier, or
  (b) a controlled watermarked-vs-unwatermarked Gemini dataset becomes available.
  Neither is true today.

## Sources

- `fyxme/opensynthid-detect-0.1` (HuggingFace) + author write-up at fyx.me — the confounded neural surrogate.
- `google-deepmind/synthid-text` (GitHub) — official, text-only, per-key.
- `aloshdenny/reverse-SynthID`, `hackerfactor/reverse-SynthID-C`, `zero2launch/reverse-SynthID-gpu` (GitHub) — the spectral codebook family.
- Hacker Factor blog, "Reversing SynthID" — independent test of aloshdenny's detector + Google's consumer tool unreliability.
- arXiv 2410.07369 (PRC Watermark, ICLR 2025) — undetectability theory + no-key classifier on 8 schemes.
- arXiv 2403.15365 (Transfer Attack, Duke) — no-box removal attack, not a detector.
- arXiv 2510.09263 (SynthID-Image) — content-conditional neural design, anti-surrogate hardening.
- `wiltodelta/remove-ai-watermarks` (GitHub) — SDXL regen removal (our regen path); its "Detection" is the visible logo only.
