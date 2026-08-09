#pragma once
// Detail restoration for SynthID diffusion-regen outputs.
//
// After `--synthid-attack regen` produces R from the watermarked input O, this
// optionally restores a slice of the regen-lost detail to recover fidelity on
// bright images (the carrier-to-content ratio drops with luminance, so a small
// restored slice stays under the detector threshold on bright content).
//
// Algorithm (see docs/research/synthid-diff-restoration-analysis.md):
//   D = O - R                       (signed BGR float)
//   if mode == Auto and luma(O) < 128:  return R          (full regen, safe path)
//   D_att = attenuator_B(D)         (characterized-carrier Wiener, per channel)
//   mask  = top-5% pixels by combined L2 norm |D|          (dilution gate)
//   R'    = clip(R + mask * D_att, 0, 255)
//
// MATCHES the Python reference probe /tmp/e3hcto_attenuate.py
// (attenuator_B with gamma=4, plus the max(.,0) "capped" guard from the design
// doc). Validated against Google's "Verify with SynthID" manual verifier across
// 10 varied images (6 bright cleared, 4 dim failed -> the luminance gate).
//
// The enum/struct below are ALWAYS defined (InpaintConfig carries RestoreConfig
// unconditionally so a regen-free build still compiles). The restore_detail
// symbol only exists when WMR_BUILD_REGEN is on (the TU is guarded); a
// regen-free build never calls it (regenerator.cpp is entirely guarded).
#include <opencv2/core.hpp>

namespace wmr {

// When to apply restoration. Auto is the default and uses the luminance gate.
enum class RestoreMode {
    Auto,   // luma(O) >= threshold -> restore; else full regen (return R)
    On,     // always restore (user accepts the risk on dim content)
    Off,    // never restore (guaranteed full regen; --no-regen-restore-detail)
};

struct RestoreConfig {
    RestoreMode mode = RestoreMode::Auto;

    // Auto-gate. Rec601 luminance in BGR order (0.114*B + 0.587*G + 0.299*R),
    // computed from the ORIGINAL watermarked input O. Threshold 128 (8-bit
    // midpoint; lands inside the empirical gap [107, 130] from the 10-image study).
    float luminance_threshold = 128.0f;

    // Dilution gate: keep the top `keep_fraction` of pixels by combined L2 norm
    // of the ORIGINAL D. 0.05 = top-5% (the validated knee; 10%+ was detected).
    float keep_fraction = 0.05f;

    // Attenuator B (characterized-carrier Wiener) parameters. Defaults are the
    // characterized "B3" variant (gamma=4, capped). See the design doc.
    float gamma = 4.0f;             // shrink aggressiveness; shrink = max(1 - gamma*alpha*rel, 0)
    float dc_radius = 25.0f;        // dc_ramp(r, 25) = clamp(r/25, 0, 1)
    float highfreq_cutoff = 600.0f; // rolloff(r, 600) = clamp((600-r)/300, 0, 1)
    float calib_band_low = 25.0f;   // calibration band [low, high] for the prior
    float calib_band_high = 35.0f;
    float target_r = 30.0f;         // prior calibrated so |prior(30)| ~= 50% of measured mean |F_D| near r=30
    float prior_exponent = 1.3f;    // carrier magnitude envelope r^-1.3 (power slope -2.6)

    // Channel weights, BGR order (characterized carrier relative strengths
    // G/B/R = 1.0/0.90/0.86; same spectral shape across channels, corr > 0.96).
    float channel_weight_b = 0.90f;
    float channel_weight_g = 1.00f;
    float channel_weight_r = 0.86f;
};

// Compute R' from the watermarked original O and the regen output R.
//
// Returns:
//   - R unchanged (a clone) when mode == Off, or mode == Auto and the luminance
//     gate picks full regen (luma < threshold).
//   - R' = clip(R + mask * D_att, 0, 255) otherwise (CV_8UC3, same size as R).
//
// O and R must be CV_8UC3 with identical size. Never throws; on a size/type
// mismatch it returns a clone of R (graceful no-op, like the regen fallback).
// Internals log the chosen branch via spdlog (the caller does NOT log here, so
// this stays library-safe for the unit tests).
cv::Mat restore_detail(const cv::Mat& O, const cv::Mat& R, const RestoreConfig& cfg);

// Mean Rec601 luminance of an 8-bit BGR image (0.114*B + 0.587*G + 0.299*R),
// returned in [0, 255]. Exposed for unit testing the auto-gate decision.
float mean_luminance_bgr(const cv::Mat& bgr_u8);

} // namespace wmr
