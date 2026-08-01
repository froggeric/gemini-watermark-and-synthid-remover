#pragma once

#include <opencv2/core.hpp>
#include <string>

#include "core/fft_context.hpp"
#include "synthid/spectral_codebook.hpp"

namespace wmr {

enum class RemovalStrength {
    Gentle,
    Moderate,
    Aggressive,
    Maximum
};

struct RemovalConfig {
    RemovalStrength strength = RemovalStrength::Moderate;
    float custom_strength = -1.0f;  // Override: 0.0-1.0 if >= 0
    bool phase_adaptive = false;    // Use image's own phase for uniform images
    // WS3 experiment: operate on the LAB `a` (green-red opponent) channel only,
    // keeping L and b byte-identical. Default off = today's BGR path (byte-identical).
    // Reported more detectable in `a` (vitotitto/synthid-fingerprint-analysis); this
    // flag makes the path measurable on our fixtures. Data-gated: ship only if it
    // measurably beats BGR (see docs/research/synthid-lab-a-experiment.md).
    bool lab_a = false;
    // Bypass the content-image guard (is_content_image -> num_passes=0). The guard
    // makes a codebook inert on content images (std > 0.05) because carrier
    // subtraction mainly removes image content there. This flag is for EVALUATION:
    // it lets a codebook act on content so the carrier attenuation (and any dot /
    // damage imprint) is measurable. Not a default; the guard exists for good
    // reasons on real content (see docs/research/synthid-clean-codebook-eval.md).
    bool no_content_guard = false;
};

class CodebookSubtractor {
public:
    CodebookSubtractor(FftContext& fft);

    void remove_synthid(cv::Mat& image,
                        const SpectralCodebook& codebook,
                        const RemovalConfig& config = {});

private:
    FftContext& fft_;

    static constexpr float kChannelWeights[3] = {0.85f, 1.0f, 0.70f};  // B, G, R (OpenCV order)

    struct StrengthParams {
        float removal;
        float cons_floor;
        float mag_cap;
        float dc_radius;
    };

    static StrengthParams get_strength_params(RemovalStrength strength);
    cv::Mat estimate_watermark_fft(
        const cv::Mat& image_fft,
        int channel,
        float removal_factor,
        float cons_floor,
        float mag_cap,
        float dc_radius,
        const SpectralProfile& profile,
        float image_luminance);
    cv::Mat compute_subtract_magnitude(
        const cv::Mat& image_fft,
        int channel,
        float removal_factor,
        float cons_floor,
        float mag_cap,
        float dc_radius,
        const SpectralProfile& profile);
};

} // namespace wmr
