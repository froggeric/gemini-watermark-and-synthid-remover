#pragma once

// The local surrogate-detector manifest for the anti-detection adversarial
// stage. Deliberately ORT-free (constexpr data only) so the fetch TU, the
// tests, and any pure TU can read the pins without linking ONNX Runtime.
//
// Adding a surrogate = one entry here + the pinned file on
// huggingface.co/froggeric/wmr (exact upstream bytes, SHA256-pinned at compile
// time). License facts per model live in LICENSE-THIRD-PARTY.md; npr ships
// opt-in only (no upstream license).

#include <array>
#include <cstddef>
#include <string_view>

namespace wmr::antidetect {

// How an image becomes the model's input tensor.
enum class SurrogatePrep {
    BakedGraph,    // resize + normalize folded into the ONNX graph at export;
                   // runtime sends u8 BGR and the graph does the rest
    ResizeCropNorm // runtime: resize shortest side -> center crop -> RGB ->
                   // [0,1] -> per-channel (x-mean)/std, float NCHW
};

struct SurrogateSpec {
    const char* key;         // --surrogates name
    const char* filename;    // on-disk name in the models dir
    const char* sha256;      // compile-time pin; "" = not yet pinned (the
                             // fetch refuses and the suite runs without it)
    const char* url;         // froggeric/wmr mirror (exact upstream bytes)
    SurrogatePrep prep;
    int resize_short;        // ResizeCropNorm: shortest-side target (0 = none)
    int crop;                // ResizeCropNorm: center-crop side (0 = none)
    float mean[3];           // RGB normalization
    float stdv[3];
    bool sigmoid_output;     // output is a logit needing sigmoid -> p_fake
    float flip_threshold;    // p_fake below which the report calls it flipped
    size_t approx_bytes;     // pre-download notice
    size_t max_pixels;       // 0 = uncapped; else proportional-bicubic cap the
                             // input BEFORE the graph (the calibration
                             // contract: corvi's stride-1 ResNet-50 needs it
                             // for the activation budget, and its thresholds
                             // were measured WITH the cap applied)
};

// Order = the default --surrogates order.
inline constexpr std::array<SurrogateSpec, 3> kSurrogateManifest{{
    // Community-Forensics ViT-S/16@384 (MIT). The FIXED fp32 ONNX
    // (onnx/model.onnx) from buildborderless/CommunityForensics-DeepfakeDet-ViT
    // — weights regenerated 2026-07-22 to match the CVPR 2025 training. Do NOT
    // pin the sibling CommunityForensics-DeepfakeDet-ViT-ONNX repo: its exports
    // predate the fix (wrong classifier head, wrong MLP ratio) and output
    // near-zero uncorrelated logits (measured 2026-09-01: the fixed fp32 scores
    // +4.1/-3.4/-1.7/-4.4 on four fixtures where the stale fp16 gives
    // -0.01/-0.55/+0.36/+0.15). Preprocessing per preprocessor_config.json:
    // ANTIALIASED bicubic shortest-side 440 (cv::INTER_AREA on downscale; a
    // non-antialiased resize manufactures high-frequency energy the ViT reads
    // as generative), center crop 384, CLIP stats, single-logit head.
    {"commfor",
     "commfor-vit-s16-384.onnx",
     "a42c7d740fbb345ba9a26d469b22f301d73089ce3c6da993877ed2b6965a8ba1",
     "https://huggingface.co/froggeric/wmr/resolve/main/antidetect/commfor-vit-s16-384.onnx",
     SurrogatePrep::ResizeCropNorm, 440, 384,
     {0.48145466f, 0.4578275f, 0.40821073f},
     {0.26862954f, 0.26130258f, 0.27577711f},
     true, 0.5f, 87442080, 0},

    // Corvi GRIP Grag2021_latent (Apache-2.0), ResNet-50 stride-1 full-res.
    // One-time ONNX export with preprocessing + spatial-mean baked in
    // (experiments/antidetect-eval/export_corvi_onnx.py; opset 18, dynamic
    // H/W, float RGB [0,1] NCHW -> mean logit; torch-vs-ORT parity ~3e-5).
    // THE detector that catches Gemini 3.6 portraits (M0: 10/10 baseline,
    // +0.02..+0.36 margins, where commfor misses them). Input hard-capped at
    // 1.2 MPix proportional bicubic BEFORE the graph (stride-1 ResNet-50
    // activation budget; the calibrated thresholds were measured WITH the
    // cap; the 896x1200 portraits at 1.075 MPix stay native). Pin per
    // export/UPLOAD-MANIFEST.md.
    {"corvi",
     "corvi-grag2021-latent.onnx",
     "7f8a33d4d4bf89ee30251a13058b9d0c0c550d4f15f755cec77ad3fdfae0d242",
     "https://huggingface.co/froggeric/wmr/resolve/main/antidetect/corvi-grag2021-latent.onnx",
     SurrogatePrep::BakedGraph, 0, 0,
     {0.485f, 0.456f, 0.406f},
     {0.229f, 0.224f, 0.225f},
     false, 0.5f, 94262471ul, 1200000ul},

    // NPR (upsampling-artifact ResNet-50). Weights in-repo upstream; NO
    // upstream license — opt-in only, owner's distribution call. Measured
    // INVERTED/unusable on this fixture set (M0: AUROC 0.64; flags 9-14 of
    // 14 REAL camera photos under interpolation ops) — do not add to the
    // ensemble without retraining/revalidation; kept as a documented no-pin.
    {"npr",
     "npr-resnet50.onnx",
     "",
     "https://huggingface.co/froggeric/wmr/resolve/main/antidetect/npr-resnet50.onnx",
     SurrogatePrep::BakedGraph, 0, 0,
     {0.485f, 0.456f, 0.406f},
     {0.229f, 0.224f, 0.225f},
     false, 0.5f, 100000000ul, 0},
}};

// Returns nullptr for an unknown key.
const SurrogateSpec* find_surrogate(std::string_view key);

} // namespace wmr::antidetect
