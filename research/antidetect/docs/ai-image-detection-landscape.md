# AI Image Detection: How Detectors Work, Why, and Which Models They Cover (2026-09)

Research record, 2026-09-01. Method: 16-agent web research fleet (6 topic sweeps, each
adversarially fact-checked; completeness critic + 3 gap-fillers). ~550 source fetches.
Numbers below were verified against primary sources unless labeled otherwise
(single-source, vendor marketing, or practitioner lore).

Context for this repo: wmr strips visible watermarks (exact reverse alpha blend),
SynthID (diffusion regen), and provenance metadata (container rewrite). This research
answers the adjacent question: can AI images be identified from pixels alone, with all
metadata removed? Short answer: yes for most current generators against a current
detector, and metadata was never what detectors looked at.

## 0. The premise, corrected

"AI images can be easily automatically identified even with all metadata removed" is
true with two qualifications:

1. Passive detectors judge pixels alone; they never read metadata. Metadata removal
   only defeats provenance surfaces (C2PA manifests, "About this image", platform AI
   labels), which are the weakest signals in the entire landscape.
2. "Easily" holds for detectors trained (or calibrated) on the target's generator
   family and era. Out-of-the-box open checkpoints score 18-31% (below the 50% coin
   flip) on the newest commercial generators, but that is now well-attributed to
   detector staleness and threshold misalignment, not to the artifacts being gone:
   freshly trained detectors reach 0.95-1.00 AUC on the same generators.

## 1. Three families of "AI detection"

| Family | Mechanism | Needs mark at creation? | Survives metadata strip? | Examples |
|---|---|---|---|---|
| Provenance / metadata | Read signed C2PA manifest or IPTC labels from the container | Yes | No (screenshot/re-encode strips it; X strips it on upload) | C2PA Content Credentials, Google "About this image", platform labels |
| Watermark | Correlate against a signal embedded at generation (pixel-level or token-level) | Yes | Yes (the watermark lives in pixels) | SynthID Detector portal (Gemini/Imagen/Veo; since 2026-05 also OpenAI images), OpenAI Verify |
| Passive / forensic | Classify statistics of the pixels themselves | No | Yes (nothing to strip) | Hive, Optic AI or Not, Sensity, academic detectors, DINOv3 committees |

The rest of this document is about the third family. Key facts about the first two:

- SynthID Detector portal (announced Google I/O, 2025-05-20, waitlisted early access,
  still not general as of 2026-09): verifies only SynthID-marked content. Google claims
  10B+ pieces watermarked by 2025-05, 100B+ by I/O 2026 (self-reported, unverified).
  Detection rates/FPR unpublished. A negative result is officially "weak evidence".
  Community regen attacks (low-strength img2img, the same class as wmr's
  `--synthid-attack regen`) are reported to clear it, consistent with watermark-only design.
- Since 2026-05-19/20, every ChatGPT/Codex/API image carries SynthID in addition to
  C2PA (OpenAI announcement + PetaPixel/Mashable corroboration; the exact date is
  secondary-sourced). OpenAI's own 2024 classifier caught ~98% of its DALL-E 3 images
  at <0.5% FPR but only 5-10% of other vendors' images (The Verge). Its 2026 "Verify"
  research preview reads SynthID+C2PA for OpenAI content only.
- C2PA is growing (Leica/Canon/Nikon/Sony/Samsung S25/Pixel 10 cameras; TikTok attaches
  and joined the steering committee 2026-07; Meta reads it for "AI info" labels) but is
  null when absent: X strips it on upload, screenshots kill it, "C2PA remover" tools
  exist, and Nikon had to suspend its camera signing program 2025-09 after a key
  infrastructure vulnerability. Absence of C2PA proves nothing.

## 2. What passive detectors look for (the signal taxonomy)

### 2.1 GAN era: upsampling traces (proven, causal)

Every generator must upsample a low-dimensional latent to pixel resolution, and every
upsampling primitive leaks in the frequency domain:

- Transposed convolution ("deconvolution") with kernel size not divisible by stride
  produces uneven window overlap: checkerboard patterns visible even with random
  untrained weights, at discrete frequencies (Odena et al., Distill 2016).
- Upsampling = zero-insertion + convolution (true for BOTH transposed conv and
  nearest-neighbor): the low-res spectrum is replicated, adding spurious high-frequency
  energy. Two successive upsamplers create bright spectral blobs at 1/4 and 3/4 of each
  axis (Zhang et al., WIFS 2019, with a 1D DFT proof).
- Resize-convolution (adopted by ProGAN/BigGAN/StyleGAN) removed the visible spatial
  artifact but NOT the frequency residue. Frank et al. (ICML 2020) isolated it: StyleGAN
  variants differing only in the upsampling kernel are 98.24% (nearest) / 85.96%
  (bilinear) / 84.20% (binomial-5) separable by a linear classifier on the DCT spectrum.
  A LINEAR classifier on log-DCT hits 100.00% on FFHQ vs StyleGAN where pixels give 75.78%.

### 2.2 Diffusion era: VAE-decoder manifold + spectral shape

Diffusion images lack the clean GAN spectral peaks but carry different traces:

- VAE-decoder manifold (the strongest finding). A latent-diffusion image is BY
  CONSTRUCTION an output of its frozen VAE decoder, so that decoder reconstructs it with
  lower error than any real image, which can only project onto the nearest point of the
  decoder's manifold. AEROBLADE (CVPR 2024) exploits this training-free: mean AP 0.992
  across SD 1.1/1.5/2.1, Kandinsky 2.1, Midjourney v4/v5/v5.1. Midjourney v4-5.1's
  unpublished VAE behaves like SD2's (best reconstruction for 99.3-99.9% of MJ samples),
  which is why SD-family detectors caught MJ then. Vesnin et al. (2024) trained a
  detector ONLY on SD2.1-VAE reconstructions of real images and still detected
  Midjourney v6 at TPR 0.986: different autoencoders introduce similar artifacts.
- Spectral shape (Corvi et al., CVPRW 2023): real images decay as 1/f^alpha (alpha~2);
  diffusion images under-represent high frequencies ("too smooth") and are anisotropic
  (weaker along diagonal directions). Latent-diffusion peaks sit at multiples of 1/4 or
  1/8 of the spectrum, matching the VAE downsampling factor. StyleGAN2/3 (alias-free
  design) fit the real spectrum best: architecture, not model identity, drives the trace.
- Mid-band reconstruction error (FIRE, CVPR 2025): diffusion VAEs reconstruct the
  mid-band (FFT radius 40-120) of REAL images poorly; generated images carry less error
  there. FIRE uses only the SD-v1.5 VAE (no denoising loop) and reaches 72.0 avg AUC on
  DALL-E 3 / Kandinsky 3 / Midjourney / SDXL where DIRE gets 52.4.

### 2.3 Autoregressive era: codebook statistics (new, 2025-26)

Token-based generators (VAR, LlamaGen, Infinity, Janus Pro, RAR, Switti) leave a
discrete-distribution artifact: real images show long-tail VQ-codebook usage; AR
generation concentrates mass on peak tokens (high-frequency codebook entries activated
3-5x above real rates) because a finite codebook plus top-p/top-k sampling truncates the
tail. D3QE (ICCV 2025, ARForensics benchmark, 304K images) detects via quantization-error
latents plus usage-frequency discrepancy. Texture-era detectors trained on one AR model
fall to ~50% on architecturally distinct AR models.

Consumer MLLM generators (GPT-image-1/2, Nano Banana/2) are a hybrid: GPT-4o images read
as DIFFUSION-like to a Flux-vs-AR discriminator (GPT-ImgEval: detectability attributed
to its super-resolution pipeline amplifying upsampling interpolation artifacts; Effort
94.75%, FakeVLM 99.60% on GPT-4o images). SAP-DSP (2026-08) finds MLLM-generator traces
migrating from texture to STRUCTURE: character strokes, edge continuity, line alignment,
detectable only by structure-aware detectors.

### 2.4 Camera-side signals (the absence gives fakes away)

Real photos pass through physics generators do not simulate:

- CFA sampling + demosaicing imposes inter-channel conditional correlations. DCCT (2026,
  single-source preprint) builds detection on this and reports 88.36% accuracy from a
  one-class model trained ONLY on real photos, i.e. "looks like it came through a camera
  ISP" alone nearly separates the classes. Its separation bound is conditional on
  generators not mimicking CFA statistics (unmeasured end-state).
- PRNU (sensor pattern noise): classical, mature for camera identification; fakes have
  no consistent PRNU. Benchmarked at 64-80% for GAN discrimination (Frank et al.),
  below learned methods.
- 1/f^2 spectral falloff as the baseline every deviation is measured against.

### 2.5 Per-model fingerprints (attribution)

Each trained generator instance stamps a stable residual signature (architecture, seed,
training data; instances differing by ONE training image are distinct). PRNU-style
averaged residuals (Marra et al. 2019: 90.3% attribution across 17 sources) and learned
fingerprints (Yu et al., ICCV 2019: 99.5% instance attribution). Modern model-level
attribution runs 75-90% on GenImage-class benchmarks. This is how commercial tools
return "made with Midjourney" style labels (Hive's API enumerates ~58 generator labels;
per-label accuracy undisclosed).

### 2.6 What commercial black boxes use

Opaque trained classifiers, presumably ensembles, trained on many generator families.
Hive is the best independently measured (CCS 2024: 98.03% accuracy, 0% FPR on 280 human
artworks vs 350 AI images). New entrants lean on foundation models: Pangram (2026-07,
research preview) fine-tunes DINOv3 with "synthetic mirroring" training; all its numbers
(99.5% internal, 0.16% FPR) are vendor-published, not independently replicated.

## 3. How detection works (method families, in order of evolution)

1. Handcrafted spectral (2019-20): azimuthal FFT / DCT features + linear classifier.
   Frank et al. (ICML 2020): log-DCT + ridge regression, 100% on StyleGAN-era sets.
   Zhang's AUTO-GAN simulated artifacts to train with zero real fakes.
2. Fingerprinting / attribution (2019): correlation against per-model fingerprints.
3. Universal trained CNN (2020): Wang et al. (CVPR 2020), a plain ResNet-50 trained on
   720K ProGAN images + blur/JPEG augmentation generalizes to unseen GANs (mAP 92.6;
   StyleGAN 99.6, CycleGAN 93.5, GauGAN 89.5 under blur+JPEG(0.1)). This became the
   CNNSpot baseline every later paper beats on old generators and the thing that fails
   on new ones (0.375 mean in the 2026 zero-shot benchmark).
4. Patch / restricted receptive field (2020): Chai et al. ECCV 2020; the signal lives in
   local textures (hair, background, object boundaries), not semantics.
5. Reconstruction-based (2023-25): DIRE (DDIM invert + reconstruct + classify the error;
   99.9 in-distribution, ~70 on GenImage, chance cross-domain), SeDID, LaRE2 (latent
   space), DRCT (reconstruction-synthesized hard training samples), FIRE (VAE-only
   mid-band), AEROBLADE (training-free VAE reconstruction). The surviving signal in
   this family is the VAE decoder manifold; the diffusion-process part of DIRE's
   hypothesis did not hold up (AEROBLade found a JPEG/PNG storage bias in DIRE's
   original eval; "Revisiting DIRE" 2025 shows GAN images reconstruct WORSE than real
   ones, so reconstruction advantage is diffusion-specific).
6. Foundation-feature probes (2022-25): frozen CLIP/DINOv2 features + light heads.
   UnivFD (CLIP, CVPR 2022), RINE (intermediate CLIP blocks, ECCV 2024), AIDE (DCT
   patch selection + SRM filters + OpenCLIP semantics, ICLR 2025; GenImage mean 86.88),
   SPAI (spectral OOD via masked-frequency modeling of REAL images, CVPR 2025; avg AUC
   91.0 over 13 unseen generators incl. Flux 83.0, SD3 75.9, DALL-E 3 90.2).
7. Current frontier (2025-26): DINOv3/SigLIP2 committee classifiers with
   difficulty-aware distortion augmentation (NTIRE 2026 winners: MICV 0.9723 robust ROC
   AUC, Ant International 0.9721 with two DINOv3-7B experts), few-shot adaptation to
   new generators (Fleet, ICML 2026: Seedream 4.0 20.4% -> 73.1% with 10 shots),
   post-hoc logit calibration (AAAI 2026: much "undetectability" is threshold shift;
   CNNSpot on Midjourney 51.4% -> 67.5% from a scalar offset), and MLLM reasoning
   detectors (Veritas ICLR 2026 oral, 90.7% overall; AlignGemini, semantic + pixel
   branches, 91.8 in the wild on post-2024 generators).

## 4. Why it works, and why it sometimes does not

### Why it works

1. Generators must map low-dim latents to pixels through upsampling/decoding, which
   leaves periodic or manifold-shaped statistical traces (2.1-2.3 above), and each
   architecture family leaves a DIFFERENT trace class (GAN grids -> diffusion VAE
   manifolds -> AR codebook statistics).
2. Generators do not simulate the camera pipeline (CFA/demosaic correlations, PRNU, shot
   noise, in-camera processing). Detectors exploit this implicitly even when not
   designed to.
3. High-frequency calibration is off (GANs: excess peaks; diffusion: deficit).
4. Trained detectors key on low-level regularities shared across generators, not on
   "realism": fakeness scores are nearly uncorrelated with perceptual quality (Wang et
   al. 2020), which is why a detector can flag a beautiful image and pass an ugly photo.

### Why it fails (measured failure modes)

- Training staleness (largest measured factor). The Feb 2026 zero-shot benchmark (16
  detectors, 291 generators, 2.6M images; single-source preprint) found 43% of failures
  are training-test mismatch; mean accuracy falls from ~79% on 2020-21 generators to
  ~38% on 2024 ones. NTIRE 2026 counterpoint: with contemporaneous training data, teams
  hit 0.97 robust AUC on private held-out frontier engines (Nano Banana 2, GPT Image
  1.5, Imagen 4 Ultra, Seedream 5, Grok Imagine). OpenFake: the SAME SwinV2 architecture
  scores F1 0.992 when trained on fresh commercial-generator data and collapses when
  trained on GenImage (Midjourney v6 TPR 0.090). Detectability is mostly a property of
  training recency, not generator stealth.
- Threshold misalignment (AAAI 2026): under distribution shift the decision boundary
  drifts so detectors call everything "real"; a scalar logit offset fit on ~100 samples
  recovers 14-27 points in several cases. This is why some "below chance" numbers
  (18-24% on 2024 generators) mean miscalibration, not absence of signal.
- Compression and re-encoding. JPEG QF<=75 collapses frequency/low-level detectors to
  chance (LGrad 75.34 -> 50.00 at QF75; LNP -> 52.85 at QF50; AIDE 92.77 -> 69.60).
  Bellingcat: compressing Midjourney PNGs to 300-500 KB flipped 7/10 photorealistic
  images to "human" on AI or Not. Platform transmission (RRDataset, ICCV 2025: 2-6
  repost cycles through Telegram/WeChat/WhatsApp/X/Instagram/etc.) annihilates SAFE
  (98.29 -> 0.88) and Freq-Net (76.08 -> 4.47) while DIRE (89.72 -> 90.34) and
  DRCT-ConvB (93.52 -> 92.82) survive; best overall 89.59% (DRCT-ConvB). Commercial
  Hive survives JPEG QF15 at 91.88%.
- Adversarial perturbation. Black-box eps=8/255 "Diverse Input" attacks drop open
  detectors from ~98 to 4.1 AUC and Hive from 100.0 to 73.8 on SD-1.4 (researchers'
  measurement); the attack partially survives social-media re-encoding. White-box PGD
  takes essentially every academic detector to F1 0 (RAID); ensemble-crafted attacks
  transfer across architectures (0.99 -> 0.67 AUROC). Glaze (a style-protection tool
  repurposed) cuts Hive 98.03 -> 80.81 with FNR 3.17 -> 32.44%.
- Analog laundering (print/rescan, photo of screen) is the single most destructive
  transformation measured: DIRE 89.72 -> 1.42, most detectors below 50; AIDE most
  analog-resistant (76.04), GPT-4o zero-shot 69.23 (robustness-tuned ICL lifts it to 87.47).
- Dataset bias masquerading as detection: GenImage carries JPEG/size biases (debiasing
  gains +11pp cross-generator); DIRE's original eval had a JPEG-real/PNG-fake storage
  bias. Some published "generalization failure" was never about generator artifacts.

### The open causal question

"Newest commercial generators apply post-processing to remove spectral anomalies" is an
INFERENCE from one benchmark, hedged in its own text, and weakened by a controlled
ablation (arXiv 2510.05633): zeroing Fourier peaks on DALL-E 3/Flux/Midjourney images
leaves most modern detectors unchanged, i.e. spectral peaks are neither necessary nor
sufficient. Best-supported causes of the 2024+ zero-shot decline: training staleness
(43% failure share; 10-shot recovery; +11pp debiasing), threshold shift, and
architecture-shift artifact migration (flow-matching and AR generators leave different
traces). Grain/noise addition is confirmed only against weak commercial detectors at
heavy post-hoc noise; generator-INTEGRATED film grain or CFA mimicry has zero controlled
measurements. The "unwinnable arms race" result (ETH, NeurIPS 2025) is a Bayes-limit
argument (if generator distribution equals real distribution, no detector beats chance),
not a measurement of current generators; empirically each architectural family has so far
introduced new traces faster than it erased old ones.

## 5. Which generators are affected (per-model evidence, Sept 2026)

| Generator | Passive detectability | Evidence |
|---|---|---|
| ProGAN, StyleGAN 1-3, BigGAN, CycleGAN, StarGAN, GauGAN | Solved, even zero-shot (ProGAN 87%, StyleGAN2 82% mean across 2026 benchmark) | Wang 2020, Frank 2020, 2026 benchmark |
| SD 1.x / 2.x | Solved in-distribution (Corvi detector AP 1.000); 73% (SD 1.4) zero-shot across modern detectors | GRIP-UNINA, AEROBLADE, 2026 benchmark |
| SDXL + community fine-tunes/LoRAs | Mid. Detectable by SD-trained detectors (DRCT); community-curated sets (Chameleon) collapse all 10 tested detectors to 53-64%, most with ~0-3% fake recall | AIDE ICLR 2025 |
| Kandinsky 2.1 | Solved (AP 0.999 training-free) | AEROBLADE |
| Midjourney v4/v5/v5.1 | Solved by 2023-24 detectors (AP ~1.0; MJ's VAE ~ SD2's) | AEROBLADE, Corvi re-evals |
| Midjourney v6 | Detectable by well-trained detectors (SAFE 94.1%); killed stale GenImage-trained ones (TPR 0.090) | AIGIBench, OpenFake |
| Midjourney v6.1 | 84.0 AUC (SPAI, hardest tier for spectral OOD) | SPAI CVPR 2025 |
| Midjourney v7 | 24% zero-shot (stale checkpoints, single-source); no published fresh-trained number, but NTIRE-class training covers the tier | 2026 benchmark |
| DALL-E 2 | Detectable in-distribution; weak peaks ("spectral spread") | Corvi 2023 |
| DALL-E 3 | 31% zero-shot; collapses most detectors to ~49-55% cross-source; 90.2 AUC (SPAI) | 2026 benchmark, AIGIBench, SPAI |
| GPT-image-1 / GPT-4o images / GPT Image 2 | Surprisingly detectable in-distribution (Effort 94.75, FakeVLM 99.60 on GPT-4o; trace reads diffusion-like, attributed to the SR pipeline); GPT Image2 drops texture-era detectors 12-37 pts, traces migrate to structure | GPT-ImgEval, SAP-DSP |
| Imagen 3 / 4 | 19% zero-shot (Imagen 4, stale); 0.97 robust AUC tier when covered by fresh training (ImageGen-4 Ultra held out in NTIRE 2026 private set) | 2026 benchmark, NTIRE 2026 |
| Gemini / Nano Banana / Pro / 2 | 44.6% mean on the Nano-Banana 150K set (stale); NTIRE 2026 private set (Nano Banana Pro/2) hit at 0.97 robust AUC by top teams; SAP-DSP reports 12-37 pt drops for older detectors on Nano Banana 2 | 2026 benchmark, NTIRE 2026, SAP-DSP |
| Flux.1 dev/schnell/kontext | 21% zero-shot (stale); 83.0 AUC (SPAI); inpainting (fill) near-undetectable (AUROC 0.41-0.49) | 2026 benchmark, SPAI, 2512.16688 |
| Flux.2 | In NTIRE 2026 test sets (FLUX-2 Max); Treasure covers FLUX.2 | NTIRE 2026, Fleet |
| Adobe Firefly (v4) | 18% zero-shot (stale, single-source); absent from most academic benchmarks | 2026 benchmark |
| Seedream 4/5, Grok Imagine, Ideogram, Recraft, Kling, Doubao | Covered by Treasure/NTIRE/OpenFake (Seedream 4.0: 20.4% zero-shot -> 73.1% with 10 shots) | Fleet, NTIRE 2026, OpenFake |
| Qwen-Image, Z-Image | 99.86 (Qwen-Image, DCCT, single-source); Z-Image Turbo in NTIRE 2026 private set | DCCT, NTIRE 2026 |
| AR open models (VAR, LlamaGen, Infinity, Janus Pro, RAR, Switti) | Detectable in-family (D3QE 82 acc / 92 AP; UniGenDet 98.1 zero-shot); texture-era detectors fall to ~50% cross-AR | ARForensics, D3QE, UniGenDet |
| Partial edits (inpainting) | Near chance below ~5% edited area; Flux fill / Firefly fill AUROC 0.41-0.49; classical inpainters (LaMa, BrushNet, PowerPaint) 0.88-0.90 | 2512.16688 |

Reading guide: "zero-shot" numbers describe stale public checkpoints; they are the
floor, not the ceiling. The ceiling for any current generator is ~0.95-1.00 AUC given a
detector trained on contemporaneous, diverse data.

Commercial detector coverage: Hive's API enumerates ~58 generator labels (midjourney,
dalle, stablediffusion(+xl, inpaint), flux, adobefirefly, imagen, imagen4, 4o, grok,
ideogram, recraft, veo3, sora, ...). Label existence is not measured per-label accuracy.
Independent tests of commercial tools:

- CCS 2024 (peer-reviewed; 280 human artworks vs 350 AI images): Hive 98.03% acc / 0%
  FPR; Optic AI or Not 90.67% / 24.47% FPR (flags 1 in 4 human artworks); Illuminarty
  72.65% / 67.4% FPR. Academic baselines collapsed on the same set (DIRE 51-55%).
- NewsGuard 2026-05 (45 war-photojournalism images): on AI-EDITED images Hive caught
  only 9/15 heavily-edited (0 false positives); AI or Not caught 15/15 heavily-edited
  but flagged 87% of lightly-edited real photos; the 5 tools disagreed on 35/45 images.
- Bellingcat 2023: AI or Not 100/100 on clean Midjourney PNGs; 7/10 photorealistic
  images escaped after compressing to 300-500 KB; 6/20 award-winning real photos
  false-flagged.
- In-the-wild (Deepfake-Eval-2024): best commercial image model 0.82-0.86 acc / 0.88-0.90
  AUC vs academic sets promising 0.94-0.99.

## 6. Evasion: what actually defeats detectors (measured)

Ordered by destructiveness:

1. Adversarial perturbation (white-box PGD, eps 16/255): F1 -> 0 for essentially all
   academic detectors; even black-box ensemble transfers reach 0.99 -> 0.67 AUROC.
   Consumer tools exist (Glaze repurposed; open-source "bypass utility" suites with
   camera-pipeline simulators), most with only anecdotal evidence.
2. Analog capture (print + re-photograph, photo of screen): defeats nearly everything
   (DIRE -> 1.4%); most resistant: AIDE 76.0, DRCT-ConvB 64.3, GPT-4o 69.2 (87.5 with
   robustness-tuned in-context examples).
3. Social-platform re-encoding (repost cycles): kills frequency/low-level detectors
   (SAFE 98 -> 0.9, Freq-Net -> 4.5); DRCT-ConvB, DIRE, AIDE, and commercial Hive hold.
4. JPEG recompression: QF<=75 collapses frequency detectors to chance; semantic/hybrid
   and commercial detectors survive QF15 at >91%.
5. Regeneration (img2img): removes watermarks (SaTML 2026: PRC watermark TPR 1.0 -> 0.0
   via SDXL; bigger models are stronger removers) but RE-FINGERPRINTS the image with the
   second model's traces. Mandelli et al.: even a strength-0 VAE round-trip makes REAL
   images classify as synthetic (their two-stage detector separates laundered from
   fully-synthetic at AUC 0.994). VAE round-trips INJECT traces rather than erasing them.
6. Removal-pipeline forensics (Goonatilake & Ateniese 2026): six watermark removers
   (UnMarker, WatermarkAttacker, CtrlRegen regen, inversion, oracle inpainting) all
   leave an "implicit watermark": removal-pipeline traces detectable at 99.05-99.81%
   accuracy; blunting requires JPEG Q75 (>72% of pixels altered) or bilateral filtering.

What still catches a determined evader: multi-architecture ensembles (single-model
adversarial attacks do not transfer across dissimilar architectures), removal-pipeline
trace classifiers, expert humans on semantic cues (13 expert artists: 83% acc, beat Hive
on Glazed images; crowdworkers 59%, near chance), and VLM judges (GPT-4o 84.09% overall
on RRDataset, best of the 10 VLMs, competitive with 16 of 17 specialized detectors).

## 7. Implications for wmr

1. The provenance strip (`wmr metadata`) is orthogonal to passive detection. It defeats
   metadata/provenance surfaces, which are the weakest signals anyway. No passive
   detector ever read them.
2. The still-path reverse alpha blend does not involve a generative model, so it adds no
   generation fingerprints. It removes the visible mark; the image's SynthID carrier and
   its generation statistics remain. (The diamond-region carrier is only ~0.025/255 and
   content-correlated; the reverse blend does not and need not touch it.)
3. `--synthid-attack regen` (SDXL img2img at 0.10) clears Google's watermark but the
   output is still an AI image, now carrying SDXL's VAE-decoder traces instead of
   Google's watermark. Per the laundering and removal-forensics literature, regeneration
   traces are themselves detectable at 99%+ by suitably trained classifiers, and
   broadband high-frequency suppression is the regeneration signature. This is fine for
   the stated goal (strip SynthID), but regen must never be described as making an image
   undetectable as AI; it swaps one provenance channel for another model's fingerprints.
4. NTIRE 2026 includes invisible-watermark insertion and watermark-attack distortions in
   its training augmentation: the forensic community is explicitly hardening detectors
   against watermark-removal outputs. Expect regen-class cleaning to remain detectable.
5. If we ever want a "is this detectably AI?" diagnostic (analogous to the detection
   oracle question for SynthID), open detectors are testable locally (CNNSpot/DRCT/
   AEROBLADE checkpoints are public), unlike Google's manual verifier. A practical smoke
   test for regen outputs: run DRCT or AIDE checkpoints over wmr outputs and compare
   with the input image's score.

## 8. Primary sources (selection)

- GAN forensics: Odena et al. Distill 2016; Frank et al. ICML 2020 (arXiv 2003.08685);
  Durall & Keuper (1911.00686, 2003.01826); Zhang et al. WIFS 2019 (1907.06515); Wang
  et al. CVPR 2020 (1912.11035); Chai et al. ECCV 2020 (2008.10588); Yu et al. ICCV
  2019 (1811.08180); Marra et al. (1812.11842, 1910.01568).
- Diffusion: Corvi et al. (2211.00680 ICASSP 2023 / CVPRW 2023; 2304.06408); DIRE ICCV
  2023 (2303.09295); AEROBLADE CVPR 2024 (2401.17879); FIRE CVPR 2025 (2412.07140);
  SPAI CVPR 2025 (2411.19417); AIDE ICLR 2025 (2406.19435); RINE ECCV 2024 (2402.19091);
  GenImage NeurIPS 2023 (2306.08571); Vesnin et al. (2411.06441).
- Benchmarks 2025-26: zero-shot out-of-the-box (2602.07814, single-source preprint);
  NTIRE 2026 Robust AIGI Detection (2604.11487, CVPRW 2026); Fleet/Treasure ICML 2026
  (2606.31082); OpenFake (2509.09495); Community Forensics CVPR 2025 (2411.04125);
  AIGIBench NeurIPS 2025 (2505.12335); AAAI 2026 calibration (2602.01973); RRDataset
  ICCV 2025 (2509.09172); RAID (2506.03988); CISPA attack study (2410.01574);
  Deepfake-Eval-2024 (2503.02857); HydraFake/Veritas ICLR 2026 (2508.21048).
- AR/MLLM: D3QE ICCV 2025 (2510.05891); UniGenDet (2604.21904); SAP-DSP (2608.01258);
  GPT-ImgEval (2504.02782); AlignGemini/AIGI-Now (2512.06746); localized edits
  (2512.16688).
- Commercial: UChicago CCS 2024 (2402.03214); NewsGuard special report 2026-05-08;
  Bellingcat 2023-09-11; Hive API docs; Pangram launch post (vendor); Reuters 2026-07-10
  (Meta detector vs cropping).
- Theory/causal: ETH "Unwinnable Arms Race" NeurIPS 2025 (2509.21135); peak-removal
  ablation (2510.05633); DCCT (2601.22778, single-source); "Fake or JPEG?" (2403.17608);
  Dong et al. CVPR 2022 (spectral imprint removal).
- Watermark/provenance: SynthID Detector (blog.google, 2025-05-20); OpenAI provenance
  announcement + Verify (2026-05); SaTML 2026 regeneration attacks (2602.22197);
  Goonatilake & Ateniese, removal-pipeline forensics (2605.09203); Mandelli laundering
  (2407.10736); C2PA adoption (c2pa.org, DPReview, security.googleblog).
