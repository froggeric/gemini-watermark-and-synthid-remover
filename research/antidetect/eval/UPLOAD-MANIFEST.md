# UPLOAD MANIFEST — antidetect model artifacts (NOT uploaded; local only)

Every model file produced by the M0 harness, with SHA256 and the intended
path in the `froggeric/wmr` HuggingFace repo. Uploading is a separate,
explicit step (the wmr runtime pins these SHA256s; see the model fetch pins).
Do NOT upload anything from this harness without the release owner's go-ahead.

| artifact | local path | sha256 | intended HF repo path |
|---|---|---|---|
| CommunityForensics ViT-S/16 @384 ONNX (fp32, fixed 2026-07-22 weights) | `experiments/antidetect-eval/models/commfor-vit-s16-384.onnx` | `a42c7d740fbb345ba9a26d469b22f301d73089ce3c6da993877ed2b6965a8ba1` | `antidetect/commfor-vit-s16-384.onnx` |
| Corvi GRIP Grag2021_latent, exported ONNX (preproc baked in, opset 18, 94,262,471 B) | `experiments/antidetect-eval/export/corvi-grag2021-latent.onnx` | `7f8a33d4d4bf89ee30251a13058b9d0c0c550d4f15f755cec77ad3fdfae0d242` | `antidetect/corvi-grag2021-latent.onnx` |
| Corvi GRIP Grag2021_latent source checkpoint (reference only, NOT for upload) | `experiments/antidetect-eval/models/Grag2021_latent_epoch_best.pth` | `0168451e84a2a35d0b141ad4f2e361a74efdd06eb8d11434087f8a0bb6e3abad` | — (mirrored from buloutian/corvi-2022-mirror) |

torch-vs-ONNX parity of the exported Corvi graph (CPU, 3 fixtures, inputs
above 1.2 MPix bicubic-capped as in scoring):

- AI 896x1200: torch +0.159855 / onnx +0.159822, |d| = 3.31e-05
- AI 2400x1792 (capped): torch -0.956319 / onnx -0.956312, |d| = 7.39e-06
- REAL 3474x2316 (capped): torch -10.146908 / onnx -10.146910, |d| = 1.91e-06

## Provenance notes

- **commfor ONNX**: downloaded from
  `buildborderless/CommunityForensics-DeepfakeDet-ViT` (`onnx/model.onnx`,
  87,442,080 bytes). Use THIS repo only: the older
  `buildborderless/CommunityForensics-DeepfakeDet-ViT-ONNX` repo is
  self-declared deprecated with broken weights (wrong intermediate_size,
  wrong classifier head); the fixed repo's CHANGELOG documents the 2026-07-22
  regeneration. The file is byte-identical to upstream (plain download), so
  re-uploading it preserves LFS dedup semantics only if the exact bytes are
  kept. License: MIT (per the repo).
- **corvi ONNX**: exported locally by `export_corvi_onnx.py` from the
  Grag2021_latent checkpoint (`buloutian/corvi-2022-mirror`). The graph bakes
  in the [0,1] -> ImageNet normalization and the spatial-mean aggregation;
  input `image` (N,3,H,W) float32 RGB in [0,1], output `logit` (N,1), logit >
  0 = AI-generated. opset 18 (the dynamo exporter's floor for this graph;
  onnxruntime >= 1.16 supports it — wmr vendors 1.27.1). Apache-2.0 (GRIP
  code) / checkpoint license per the mirror repo (research use; check before
  redistribution).
- **NPR**: not uploaded — the harness loads `NPR.pth` straight from the
  cloned `chuangchuangtan/NPR-DeepfakeDetection` repo. If NPR becomes a
  shipped surrogate later, export it the same way and add a row here.
