# Anti-Detection Pipeline Research: Defeating GenAI Image Detectors at Preserved Quality (2026-09)

Research record, 2026-09-01. Method: second 16-agent fleet (6 attack-side sweeps, each
adversarially fact-checked; completeness critic + 3 gap-fillers), ~650 source fetches.
Companion to `ai-image-detection-landscape.md` (the detector-side map). Goal: choose the
best quality-preserving way to make AI-generated images pass passive AI-image detectors,
to ship as a native C++ pipeline (standalone + chained after SynthID regen).

License facts are reported per artifact for the owner's decision (per repo convention);
nothing here is excluded on license grounds.

## 1. Verdict up front

No single method wins. The measured landscape forces a hybrid of three layers, and the
most valuable one is untested by anyone (that is our opening):

1. **Degradation attacks (JPEG/resize/blur/white noise) are dead** against 2026 SOTA.
   NTIRE 2026 winners trained on 36 distortion types (including white/ISO/shot noise,
   JPEG/JPEG2000/JPEG-AI neural compression, multi-compression chains, watermark
   insertion AND watermark-attack transformations) and still hold 0.97 robust AUC.
2. **Adversarial (L-inf bounded) perturbation is the only family with measured
   near-total evasion** against 2024-era detectors: DI-ensemble attack at eps 8/255
   drops the 4-detector average from AUC 98.0 to 4.1, survives social-media re-encoding
   (AUC 11.3 after 75%-downscale+JPEG50), and degrades Hive to 76.6 avg AUC. Quality at
   eps 8/255: PSNR 34.3 dB / LPIPS 0.033 (plain ensemble) or 31.6 dB / 0.161 (DI). At
   eps 2/255: PSNR 45.6 dB / LPIPS 0.002. Caveat: never measured against DINOv3
   committees; RAID's DINOv2 probes are the most transfer-resistant family measured
   (81.2 -> 63.8 AUC at eps 32/255 where CNN/CLIP heads go to ~0.5).
3. **Camera-statistics restoration is the flagship untested hypothesis**: cheapest
   native axis, mechanistically aimed at the newest detector families, provably OUTSIDE
   the NTIRE 2026 training distribution (which includes uncorrelated noise but NOT
   CFA/demosaic correlations, PRNU, grain texture, vignetting). Zero published
   measurement in either direction. This is a publishable A/B nobody has run.
4. **The real deployment threat has no public checkpoint**: NTIRE 2026 winners (MICV,
   Ant International), Pangram, DCCT all released no weights. We must self-train a
   DINOv2/v3 committee surrogate to have any local proxy for that class.
5. **MLLM reasoning detectors are out of scope**: no published evasion; JPEG QF50 costs
   Veritas 3 points; the semantic channel (hands, text, arithmetic coherence) is immune
   to pixel statistics. Honest boundary, same epistemic status as Google's manual
   SynthID verifier.

## 2. Attack families, ranked by evasion-per-quality (all numbers verified)

| Family | Best measured evasion | Quality at that point | vs 2026 committees | Native cost |
|---|---|---|---|---|
| DI-ensemble L-inf PGD, eps 8/255 (2410.01574v3/v4) | avg AUC 98.0 -> 4.1; Hive -> 76.6 | PSNR 31.6 / LPIPS 0.161 (plain ensemble: 34.1 / 0.092 at AUC 23.4) | untested; DINOv2 probes resist (63.8-70.3 at eps 16-32) | needs gradients (see 5) |
| RAID leave-one-out ensemble PGD, eps 16/255 (2506.03988) | Corvi 0.99 -> 0.67, Wang -> 0.62, DRCT-ConvB -> 0.56; Sightengine acc -> 0.0 | eps 16/255 class (~32-34 dB) | DINOv2 81.2 -> 69.9; unseen ViTs hold 0.73 | needs gradients |
| FPBA frequency-domain transfer (2407.20836) | AEROBLADE AP 80 -> 43, DIRE 100 -> 41; CNN victims 69.2% ASR (best config; paper-wide avg 50.5%) | PSNR 36.3 / SSIM 0.94 (best of class) | untested; CLIP victims resist (UnivFD 16.6% ASR) | surrogate + autograd |
| DuFIA dual-domain transfer (2511.15571) | UnivFD -> RINE acc 0.86 -> 0.38; survives JPEG Q90 (0.30) / Q60 (0.45) | PSNR 33.4 / LPIPS 0.062 | untested | surrogate + autograd |
| Query/score attacks (Square BSD-3, SimBA MIT) | classifiers: Square median 11-73 queries to flip; on AIGI detectors only NES measured: weakest (AUC 98 -> 79.7 at 500 queries, but best quality PSNR 49.5) | best quality of all | FBA2D (hard-label, Dec 2025) hits ASR 0.88-0.99 on AIDE/CNNSpot at 500 queries, RMSE 0.01 (~40 dB); no transfer/JPEG data | forward-only; seconds-to-minutes/image |
| Universal perturbation (offline, runtime tensor-add) | pooled AUC 7.6 no-degradation; JPEG50 -> 41 (CLIP-family recovers, CNN-family stays down) | PSNR 32.5 / LPIPS 0.156 | untested | trivial runtime |
| Camera-stats restoration | NONE published (the gap) | target ~32-40 dB (grain dose) | the one untrained cue family | pure OpenCV, ms |
| Degradation (JPEG/blur/noise alone) | collapses 2023-24 panels via "real"-bias (fake-acc -> 0, real-acc 100) | QF50-class damage | dead (inside training set) | trivial |
| Regeneration / purification | kills watermarks; re-fingerprints | strength 0.10 ~ PSNR 39 dB class | regen outputs 99.4-99.7% detectable by removal-trace classifiers | already shipped (sdcpp/CoreML) |

Key mechanism notes:
- Attack energy belongs in LOW DCT bands (FBA2D: fakes attacked in 20% lowest
  frequencies; DuFIA: random DCT masking) because JPEG preserves them. This is the
  design hint that reconciles adversarial noise with re-encode survival.
- The DI attack's value is re-encode robustness at ~2.5 dB PSNR cost over plain
  ensemble attacks; plain transfers die faster under JPEG (RAID authors state theirs
  were not built to survive post-processing).
- Universal noise is a fallback, not a strategy: it heals under JPEG on CLIP-family
  heads and a fixed tensor reused across images is itself a corpus-detectable constant.

## 3. The physics layer (camera-statistics restoration), concretely

The one family NTIRE 2026 did NOT train against: correlated camera structure. Recipe
(each ingredient is pure image ops; mitigate counter-forensics by per-image
randomization):

1. Optional bilateral pre-clean (strongest measured counter to removal-trace
   classifiers: TPR@0.1%FPR ~0.15 vs ~1.0 for everything else; 2605.09203 v2).
2. Poisson-Gaussian sensor noise in LINEAR space, BEFORE mosaicing, so it passes
   through demosaic like real sensor noise: sigma(x) = sqrt(a*x + b), camera-typical
   total ~2-6/255 (matches ISO 400-3200 phone photos; PSNR ~32-40 dB). White Gaussian
   at this dose is unmeasured; the dose that measurably collapses 2023-24 panels
   (sigma ~51/255) is gross snow. Film grain option: AFGS1 (AV1 spec, reference C in
   libaom, BSD-2 + patent license) is the best perceptually-calibrated grain model.
3. Bayer RGGB re-mosaic -> demosaic round-trip. Demosaic from a BANK of kernels
   (Malvar-He-Cutler 5x5, bilinear cross/diagonal, edge-aware), randomized per image:
   Chen/Zhao/Stamm (ICIP 2017) detect a single fixed linear re-demosaic at 99.9%,
   so a uniform kernel is corpus-detectable; randomization is the mitigation
   (untested, but the counter-detector needs a repeated signature to learn).
4. Lateral chromatic aberration (sub-pixel opposite R/B shifts ~0.5-1 px) + weak
   vignetting (1 - 0.1-0.35 r^2). Trivial, adds correlated optics structure.
5. One camera-like JPEG cycle, 4:2:0, q 88-96, optionally with a vendor-style
   quantization table (phones write q~85-95 4:2:0; no rigorous public per-model
   table exists). Completes the pipeline signature; skip when output must stay
   lossless (then the CFA+noise work still applies).

Mechanistic targeting: DCCT keys on absent CFA correlations (97-99% on GenImage/
emerging generators; its authors call CFA mimicry unaddressed); STAL keys on the
ultra-high-frequency spectral tail uplift of generated images (grain is exactly a
high-frequency additive process); LSB/bit-plane detectors key on absent sensor noise;
AEROBLADE-class keys on too-LOW VAE reconstruction error (injected noise raises it
toward the real-image side). DCCT's own blind spot is diffusion-RECONSTRUCTED images
(54% acc), which is our regen path.

Risks to measure, not assume:
- White noise is inside NTIRE training; only the correlated structure is novel.
- A defender can train an attack-specific detector on our pipeline's outputs (the
  2605.09203 play, 99%+ on six removal families; only 1/750 outputs passed removal +
  fidelity + stealth jointly). Mitigation: stochastic, per-image-randomized synthesis;
  self-audit against our own attack-specific classifier trained on our outputs.
- Chen/Zhao/Stamm-style re-demosaic counter-detectors exist as a template.

## 4. The evaluation harness (build FIRST)

Local suite (facts as verified; license per artifact, owner decides distribution):

| Detector | Family | Checkpoint | License fact | Role |
|---|---|---|---|---|
| Community-Forensics ViT-S/16@384 | trained-at-scale ViT | OwensLab/commfor-model-384, 87 MB, ready ONNX (fp32 143.8 / fp16 72 / int8 37.1 MB); CAUTION: pre-2026-07-22 HF port weights broken | MIT | what the public actually runs (101k downloads); zero-export-work |
| Corvi GRIP Grag2021_latent | forensic CNN | HF buloutian/corvi-2022-mirror, 282.5 MB | Apache-2.0 | strong CNN, full-res |
| UnivFD | CLIP + linear | head is 4 KB; official repo 404, use aimagelab/RAID_ckpt retrain | MIT (code); OpenAI CLIP-L/14 MIT | CLIP-family |
| RINE | CLIP intermediate blocks | in-repo ckpts 1-42 MB + CLIP-L/14 backbone (custom ONNX export emitting 12 CLS tokens) | Apache-2.0 | stronger CLIP-family |
| NPR | upsampling artifacts | NPR.pth in-repo (ResNet-50; preprocessing folds into graph) | NO LICENSE | best pure-CNN generalizer (91.7 avg on its benchmark) |
| DRCT-ConvB | ConvNeXt robust | ModelScope BokingChen/DRCT-2M pretrained.zip (4.2 GB) | NO LICENSE (dataset repo tagged Apache; code repo none) | the hard target: survives platform re-encoding |
| AEROBLADE-style | VAE reconstruction | no weights needed: our own SD/SDXL VAE encode-decode + LPIPS layer-2 (AlexNet, ~6 convs, hand-ported) + threshold calibrated on ~200 real + 200 generated | algorithm is a measurement; their repo unlicensed | reconstruction family; we already have the VAE natively |
| Self-trained DINOv2/v3 committee | foundation committee | facebook backbones (DINOv2 Apache-2.0 ONNX ready; DINOv3 custom license, gated) + heads we train on public NTIRE data + our own Gemini outputs | DINOv2 Apache; DINOv3 custom | the ONLY proxy for the deployment threat |

Free regression corpora: RAID 72k adversarial images + 7 retrained checkpoints (HF
aimagelab/RAID, MIT); NTIRE 2026 public train set (deepfakesMSU/NTIRE-RobustAIGenDetection,
108,750 real + 185,750 fake, 42 generators; note: NO Gemini/Imagen in the 20 open train
generators, winners added API-scraped data, so we add our own outputs); SAFE's
T2I-CoReBench 172,800 fakes from 40 generators incl. Gemini/Imagen/GPT-Image/Nano Banana.

Commercial oracles (manual, rate-limited, same discipline as the SynthID verifier):
Hive free web tool (the only commercial detector with measured adversarial numbers),
Pangram free dashboard (vendor-claims-only, no independent test; its 99.999% NTIRE
claim exceeds the actual winner's 99.78%).

Baseline fact to hold in mind: open detectors already average 44.6% on a 150k
Nano-Banana set zero-shot, so for Gemini images the marginal enemy is the
committee/commercial tier, not the open suite the literature attacks.

## 5. Native runtime architecture (no Python, no autograd at runtime)

The key unlock from the gap analysis: ONNX Runtime's inference API gives no input
gradients and TrainingSession only documents parameter gradients (offline Python
artifacts). BUT a backward graph d(loss)/d(image) can be exported OFFLINE (Python,
allowed for asset conversion) as a plain ONNX inference graph, and run at runtime as
2-3 forward-cost passes. That makes gradient-crafted per-image attacks (DI/DuFIA
class, the measured-best family) implementable natively without any autograd stack.

Recommended layering (runtime, all native):

```
wmr antidetect in.png [-o out.png]
  [--method auto|physics|adversarial|full]  [--strength 0..1]  [--lpips-budget 0.05]
  [--surrogates ...]  [--no-jpeg-cycle]  [--eval]  [--seed N]

Stage A  physics (always; pure OpenCV, ms):
    bilateral pre-clean (opt) -> sensor noise (linear, pre-mosaic)
    -> Bayer mosaic + randomized-kernel demosaic -> CA + vignette
    -> camera JPEG cycle (opt; 4:2:0 q 88-96)
Stage B  adversarial refinement (opt-in; ORT/CoreML):
    forward-only Square Attack (BSD-3 algorithm) or SimBA-DCT (MIT)
    against the mean score of the local surrogate ensemble,
    low-DCT-band-weighted, LPIPS-gated (VGG ONNX forward) early stop
    [phase 2: DI/DuFIA per-image crafting via offline-exported bprop graphs]
Stage C  report: per-detector score before/after (with --eval)
```

Chaining after `--synthid-attack regen`: regen first (base coat: kills SynthID AND
model-specific statistics, and lands in DCCT's blind spot), then physics, then
adversarial. Important honesty for docs: regen output is itself 99.4-99.7% detectable
by removal-trace classifiers (that is a different detector class than the passive
detectors we are evading; bilateral pre-clean is the measured counter).

Model distribution: mirror chosen checkpoints to huggingface.co/froggeric/wmr, exact
upstream bytes, SHA256-pinned at compile time, download-on-demand into ~/.cache/wmr/
(the regen pattern). Which checkpoints get mirrored is the owner's license call; the
facts table above feeds that decision.

## 6. Milestones

- M0 (eval harness, Python OK for research): detectors running locally + baseline
  table of wmr regen outputs and raw Gemini images vs the suite. One day of A/B per
  physics ingredient (the experiment nobody published).
- M1 (physics layer, native C++): randomized camera-stats pipeline + per-ingredient
  A/B + self-audit (train an attack-specific classifier on our own outputs to measure
  our forensic stealth).
- M2 (adversarial layer, native): Square/SimBA vs surrogate ensemble, LPIPS-gated;
  measure transfer to held-out families (DRCT-ConvB, committee surrogate, AEROBLADE)
  and JPEG Q30-90 survival. No published number exists for "query-crafted noise on a
  heterogeneous ensemble -> transfer under JPEG"; we would produce the first.
- M3 (committee surrogate): train DINOv2 committee on NTIRE data + our outputs;
  re-evaluate everything against it. First-ever attack measurement vs a committee
  (publication either way).
- M4 (integrate): `wmr antidetect` standalone + auto-chain after regen; mirror
  checkpoints; README + research writeup.

## 7. Contribution angle

Three cells in the literature are empty and we can fill them: (1) camera-statistics
injection vs current detectors (M1), (2) query-crafted ensemble noise transfer under
JPEG (M2), (3) any attack vs a DINOv3 committee (M3). A clean negative result is as
publishable as a positive one.

## 8. Goal boundary (documented, not hidden)

- Beatable: statistical/CNN/CLIP/patch/reconstruction detectors (with fresh training
  data caveat), frequency families under re-encode, open zero-shot suite.
- Borderline: DRCT-ConvB/AIDE-class, Hive (measurably degraded to ~76.6 AUC but with a
  real-image false-positive side effect operators can rethreshold around), DINOv3
  committees (untested, proxies pessimistic).
- Out of scope: MLLM reasoning detectors (Veritas/FakeVLM/GPT-4o judges; semantic
  channel), analog re-digitization (physical), and removal-trace forensics (different
  detector class; bilateral + stealth design mitigates, never guarantees).

## 9. Sources (selection beyond the detection-landscape doc)

- Attacks: 2410.01574 (DI/ensemble/universal/NES, DIMVA 2026; v1/v2 hold the white-box
  + quality-ladder tables, v3/v4 the DI/Hive/universal numbers); 2506.03988 RAID
  (+ HF aimagelab/RAID, MIT); 2407.20836 FPBA (TMM); 2511.15571 DuFIA; 2512.09264
  FBA2D; 2604.12781 Fragile Reconstruction; 2602.06530 ForgeryEraser
  (shared-public-backbone attack: AIDE 96.5 -> 14.2 at eps 8/255, survives JPEG50);
  2304.11670 StatAttack; 2004.00622 Carlini & Farid.
- Query attacks: 1912.00049 Square (BSD-3 code); 1905.07121 SimBA (MIT code);
  1804.08598 NES; ORT gradient facts: onnxruntime issue #13057 + TrainingSession docs.
- Physics: 2601.22778 DCCT; 2605.22751 STAL spectral tail; Chen/Zhao/Stamm ICIP 2017
  (re-demosaic counter-forensics); Goljan/Fridrich/Chen triangle test (PRNU copy
  detection); AFGS1 spec (aomediacodec.github.io/afgs1-spec); PurinNyova
  Image-Detection-Bypass-Utility (AGPL-3.0, zero published numbers; reimplementation
  reference only); swaylq/deai-image (MIT, unvalidated 50-80% self-claims).
- Stealth: 2605.09203 v2 (removal-trace classifiers; bilateral strongest counter;
  grain/CFA re-injection explicitly untested); 2407.10736 laundering.
- Detectors/checkpoints: see the facts table (section 4); NTIRE 2026 2604.11487;
  zero-shot benchmark 2602.07814; Community-Forensics 2411.04125.
- UnMarker (IEEE S&P 2025, 2405.08363): watermark remover reference for
  LPIPS-constrained optimization; custom non-commercial license; its SynthID 79%
  claim is README-only.
