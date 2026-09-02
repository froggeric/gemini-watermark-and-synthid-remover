# antidetect-eval (M0)

Measurement harness for the wmr anti-detection feature, Stage A
("camera-statistics restoration"). It baselines local AI-image detectors on
real vs AI fixtures, measures the per-ingredient effect of each Stage-A op on
detector scores (the A/B), calibrates dose defaults, and exports the Corvi
detector to ONNX.

The measured tables and recommended constants live in
`docs/research/antidetect-m0-calibration.md` (repo root) -- that doc is the
artifact the C++ constants are transcribed from. This directory holds the
machinery and the raw results.

## Setup

```bash
cd experiments/antidetect-eval
python3 -m venv .venv            # any python >= 3.12; built with 3.14
.venv/bin/pip install torch numpy opencv-python onnxruntime pillow
```

Models and third-party code (fetched once, disposable, NOT in any manifest):

```bash
mkdir -p models export third_party
# CommunityForensics ViT-S/16 ONNX -- use the FIXED repo (July 2026 weight
# regeneration); the older CommunityForensics-DeepfakeDet-ViT-ONNX repo is
# deprecated with broken weights
curl -L "https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT/resolve/main/onnx/model.onnx" -o models/commfor-vit-s16-384.onnx
curl -L "https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT/resolve/main/preprocessor_config.json" -o models/commfor-preprocessor-config.json
# Corvi GRIP Grag2021_latent checkpoint
curl -L "https://huggingface.co/buloutian/corvi-2022-mirror/resolve/main/Grag2021_latent/model_epoch_best.pth" -o models/Grag2021_latent_epoch_best.pth
# architecture + NPR (weights NPR.pth are in-repo)
git clone --depth 1 https://github.com/grip-unina/DMimageDetection.git third_party/DMimageDetection
git clone --depth 1 https://github.com/chuangchuangtan/NPR-DeepfakeDetection.git third_party/NPR-DeepfakeDetection
```

Real-photo fixtures are Wikimedia Commons "quality image" camera photographs
(downloaded once into `fixtures/real/`, attribution in
`fixtures/real/manifest.json`; re-fetching needs throttling, the CDN 429s).
AI fixtures are symlinked from the repo's untracked reference dirs by
`build_fixtures.py`.

## Files

| file | what it does |
|---|---|
| `physics_reference.py` | NumPy/OpenCV mirror of `src/core/antidetect_physics.cpp` (same op order, formulas, quantization points, dose ladder). `dose_for_strength` mirrors the CALIBRATED C++ ladder (commit c5a551a); `legacy=True` reproduces the pre-calibration ladder the historical stack-* rows were measured under. Production kernel draw = {edgeaware, bilinear}; malvar stays available for explicit-kernel runs. Run `self_check` standalone. |
| `detectors.py` | Uniform `score(img_rgb_u8) -> float` (higher = more fake, threshold 0) for commfor (ONNX, CPU), corvi and npr (torch, CPU by default -- `WMR_EVAL_DEVICE=mps` to opt in; MPS stalls on this shared machine when other processes contend the GPU). `python detectors.py` runs a direction smoke test. |
| `build_fixtures.py` | Links the curated AI fixture set, validates the real set, writes `fixtures/manifest.csv`. |
| `run_eval.py` | The A/B: 26 conditions x all fixtures x all detectors -> `results/per_image.csv`, `results/op_stats.csv`, `results/baseline.csv`, `results/ab_summary.csv`, `results/summary.md`. ~85 min on the dev Mac (CPU). Rows flush per fixture; `--resume` skips already-scored fixtures; `--legacy-stack` reproduces the pre-calibration historical grid. |
| `watchdog.sh` | Restarts `run_eval.py --resume` if `results/run.log` goes stale > 240 s (survives MPS wedges / machine contention). |
| `calibrate.py` | Turns results into the dose-ladder / flip-count tables (`results/calibration.md`). |
| `export_corvi_onnx.py` | Exports Grag2021_latent to `export/corvi-grag2021-latent.onnx` (ImageNet normalization baked in, dynamic H/W, opset 18, single file) and verifies torch-vs-ONNX parity < 1e-3 on 3 fixtures. |
| `export/UPLOAD-MANIFEST.md` | SHA256 + intended HF repo paths for every model artifact. Nothing is uploaded by this harness. |

## Run

```bash
.venv/bin/python build_fixtures.py
.venv/bin/python run_eval.py            # ~25 min; results/ populated
.venv/bin/python calibrate.py           # prints + writes results/calibration.md
.venv/bin/python export_corvi_onnx.py   # export + parity check
```

Seeds are stable (crc32 of slug+condition), so a rerun reproduces the same
condition images and scores.

## Notes and caveats

- `corvi` is res50stride1 at full resolution; inputs above 1.2 MPix are
  proportionally bicubic-capped to fit the stride-1 activation budget in
  16 GB unified memory. The cap is applied identically to baseline and every
  condition, so deltas stay comparable (flagged `corvi_capped` in
  `per_image.csv`).
- commfor's 440-px shortest-edge resize MUST be antialiased (PIL bicubic
  here; cv2 INTER_AREA is the closest C++-friendly match). Plain
  cv2 INTER_CUBIC shifts its logits by 2-4 and reads AI images as MORE fake
  (measured; see the resize-kernel table in
  docs/research/antidetect-m0-calibration.md).
- The Python mirror and the C++ TU are distribution-identical, not
  byte-identical: C++ draws from `std::mt19937_64`, the mirror from numpy
  PCG64 (same distributions, same draw order).
- The venv, `models/`, `third_party/`, and `fixtures/` are disposable local
  state; only the `.py` files, `results/`, `export/*.md`, and the two ONNX
  artifacts matter as deliverables.
