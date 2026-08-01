#pragma once

#include <opencv2/core.hpp>
#include <string>

#include "core/fft_context.hpp"
#include "synthid/spectral_codebook.hpp"

namespace wmr {

struct SynthidDetectionResult {
    bool detected = false;
    float confidence = 0.0f;
    float noise_correlation = 0.0f;
    float carrier_phase_score = 0.0f;
    float structure_ratio = 0.0f;
    float multi_scale_consistency = 0.0f;
};

class SynthidDetector {
public:
    // WS3: ColorSpace sibling path so detection can be measured on the LAB `a`
    // channel too. BGR is the default (today's behavior, byte-identical).
    enum class ColorSpace { BGR, LabA };

    explicit SynthidDetector(FftContext& fft);

    SynthidDetectionResult detect(const cv::Mat& image,
                                  const SpectralCodebook& codebook,
                                  ColorSpace cs = ColorSpace::BGR) const;

    static constexpr float kDefaultThreshold = 0.55f;

private:
    FftContext& fft_;

    static constexpr float kWeightNoiseCorr = 0.35f;
    static constexpr float kWeightCarrierPhase = 0.40f;
    static constexpr float kWeightStructure = 0.20f;
    static constexpr float kWeightMultiScale = 0.05f;

    // Today's BGR path (byte-identical to the pre-WS3 detect()).
    SynthidDetectionResult detect_bgr(const cv::Mat& image,
                                      const SpectralCodebook& codebook) const;

    // WS3 sibling: convert BGR -> Lab, score on channel index 1 (`a`) against the
    // green-channel codebook profile (the closest single-channel BGR proxy for the
    // green-red opponent channel). Measurement proxy, not a calibrated `a`-space
    // codebook. See docs/research/synthid-lab-a-experiment.md.
    SynthidDetectionResult detect_lab_a(const cv::Mat& image,
                                        const SpectralCodebook& codebook) const;

    float noise_correlation(const cv::Mat& channel,
                            const cv::Mat& profile_mag) const;

    float carrier_phase_matching(const cv::Mat& channel_fft,
                                 const cv::Mat& profile_phase) const;

    float structure_ratio(const cv::Mat& channel_fft,
                          const cv::Mat& profile_mag,
                          const cv::Mat& profile_consistency) const;

    float multi_scale_consistency(const cv::Mat& image,
                                  const SpectralProfile& profile) const;
};

} // namespace wmr
