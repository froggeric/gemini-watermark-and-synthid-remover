#pragma once

#include <opencv2/core.hpp>

#include "core/fft_context.hpp"
#include "synthid/codebook_subtractor.hpp"

namespace wmr {

class NoiseResidualSubtractor {
public:
    explicit NoiseResidualSubtractor(FftContext& fft);

    void remove_synthid(cv::Mat& image,
                        const RemovalConfig& config = {});

private:
    FftContext& fft_;

    static constexpr float kChannelWeights[3] = {0.85f, 1.0f, 0.70f};

    struct StrengthParams {
        float removal;
        float mag_cap;
        float dc_radius;
    };

    static StrengthParams get_strength_params(RemovalStrength strength);

    // WS3: LAB `a`-channel sibling path. Converts BGR -> Lab, runs the full
    // codebook-free algorithm on channel index 1 (`a`, single plane, weight 1.0),
    // keeps L and b byte-identical, merges back, Lab -> BGR. Self-contained so
    // the default BGR path (remove_synthid with config.lab_a == false) is byte-
    // for-byte unchanged.
    void remove_synthid_lab_a(cv::Mat& image, const RemovalConfig& config);

    cv::Mat estimate_carrier_from_noise(
        const cv::Mat& channel_fft,
        const cv::Mat& channel_float,
        float removal_factor,
        float mag_cap,
        float dc_radius,
        int channel_idx);

    static cv::Mat compute_dc_ramp(int rows, int cols, float radius);
};

} // namespace wmr
