#include "detection/synthid_detector.hpp"

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

namespace wmr {

SynthidDetector::SynthidDetector(FftContext& fft)
    : fft_(fft) {}

SynthidDetectionResult SynthidDetector::detect(
    const cv::Mat& image,
    const SpectralCodebook& codebook,
    ColorSpace cs) const
{
    if (cs == ColorSpace::LabA) return detect_lab_a(image, codebook);
    return detect_bgr(image, codebook);
}

SynthidDetectionResult SynthidDetector::detect_bgr(
    const cv::Mat& image,
    const SpectralCodebook& codebook) const
{
    SynthidDetectionResult result;

    if (image.empty()) return result;

    cv::Mat work;
    if (image.channels() == 4) {
        cv::cvtColor(image, work, cv::COLOR_BGRA2BGR);
    } else if (image.channels() == 1) {
        cv::cvtColor(image, work, cv::COLOR_GRAY2BGR);
    } else {
        work = image.clone();
    }

    const int w = work.cols;
    const int h = work.rows;

    const SpectralProfile& profile = codebook.get_profile(w, h);

    // Convert to float for FFT operations
    cv::Mat work_f;
    work.convertTo(work_f, CV_32FC3, 1.0 / 255.0);

    cv::Mat channels[3];
    cv::split(work_f, channels);

    // Per-channel scores, averaged across BGR
    float avg_noise = 0.0f;
    float avg_phase = 0.0f;
    float avg_struct = 0.0f;

    for (int ch = 0; ch < 3; ++ch) {
        cv::Mat prof_mag, prof_phase, prof_cons;
        if (profile.width != w || profile.height != h) {
            cv::resize(profile.magnitude_bgr[ch], prof_mag, {w, h}, 0, 0, cv::INTER_LINEAR);
            cv::resize(profile.phase_bgr[ch], prof_phase, {w, h}, 0, 0, cv::INTER_LINEAR);
            cv::resize(profile.consistency_bgr[ch], prof_cons, {w, h}, 0, 0, cv::INTER_LINEAR);
        } else {
            prof_mag = profile.magnitude_bgr[ch];
            prof_phase = profile.phase_bgr[ch];
            prof_cons = profile.consistency_bgr[ch];
        }

        avg_noise += noise_correlation(channels[ch], prof_mag);

        cv::Mat ch_fft = fft_.forward(channels[ch]);
        avg_phase += carrier_phase_matching(ch_fft, prof_phase);
        avg_struct += structure_ratio(ch_fft, prof_mag, prof_cons);
    }

    avg_noise /= 3.0f;
    avg_phase /= 3.0f;
    avg_struct /= 3.0f;

    float ms_cons = multi_scale_consistency(work, profile);

    result.noise_correlation = avg_noise;
    result.carrier_phase_score = avg_phase;
    result.structure_ratio = avg_struct;
    result.multi_scale_consistency = ms_cons;

    result.confidence = kWeightNoiseCorr * avg_noise
                      + kWeightCarrierPhase * avg_phase
                      + kWeightStructure * avg_struct
                      + kWeightMultiScale * ms_cons;

    result.confidence = std::clamp(result.confidence, 0.0f, 1.0f);

    // Content-aware threshold: for content images (high pixel variance),
    // struct_ratio reflects natural 1/f spectral shape, not a carrier signal.
    // Raise the detection bar to avoid false positives driven by struct_ratio.
    float threshold = kDefaultThreshold;
    cv::Scalar img_mean, img_std;
    cv::meanStdDev(work_f, img_mean, img_std);
    float avg_std = static_cast<float>((img_std[0] + img_std[1] + img_std[2]) / 3.0);

    if (avg_std > 0.05f) {
        // Content image: carrier signal is <0.1% of spectral energy and
        // noise_corr/carrier_phase are at random baseline (~0.50).
        // Only detect if struct_ratio is very high (>= 0.80), which indicates
        // strong spectral similarity beyond natural 1/f patterns.
        // Use a combined gate: require struct_ratio > 0.80 for content images.
        threshold = 0.70f;
    }

    result.detected = result.confidence >= threshold;

    spdlog::debug("SynthID detect: noise={:.3f} phase={:.3f} struct={:.3f} "
                  "ms_cons={:.3f} → confidence={:.3f} ({})",
                  avg_noise, avg_phase, avg_struct, ms_cons,
                  result.confidence,
                  result.detected ? "DETECTED" : "not detected");

    // SynthID-Image uses a deep learning encoder-decoder (arXiv:2510.09263).
    // FFT correlation metrics sit at random baseline (~0.50) for both watermarked
    // and non-watermarked images, so this detector has limited discriminative power.
    // Reliable detection requires the neural decoder model (not publicly available).
    if (result.confidence < 0.60f) {
        spdlog::info("Note: FFT-based detector cannot reliably detect SynthID. "
                     "Score {:.1f}% is near random baseline.", result.confidence * 100.0f);
    }

    return result;
}

SynthidDetectionResult SynthidDetector::detect_lab_a(
    const cv::Mat& image,
    const SpectralCodebook& codebook) const
{
    // WS3 sibling path. Convert BGR -> Lab and score on channel index 1 (`a`).
    // The codebook is BGR; we use the green-channel profile (index 1) as the
    // reference, since `a` is the green-red opponent and green is the closest
    // single-channel BGR proxy (also the highest-weighted BGR channel, 1.0).
    // This is a measurement proxy for the a-vs-BGR comparison, not a calibrated
    // `a`-space codebook.
    SynthidDetectionResult result;
    if (image.empty()) return result;

    cv::Mat work;
    if (image.channels() == 4) {
        cv::cvtColor(image, work, cv::COLOR_BGRA2BGR);
    } else if (image.channels() == 1) {
        cv::cvtColor(image, work, cv::COLOR_GRAY2BGR);
    } else {
        work = image.clone();
    }

    const int w = work.cols;
    const int h = work.rows;
    const SpectralProfile& profile = codebook.get_profile(w, h);

    cv::Mat lab;
    cv::cvtColor(work, lab, cv::COLOR_BGR2Lab);
    cv::Mat lab_planes[3];
    cv::split(lab, lab_planes);
    cv::Mat a_ch;
    lab_planes[1].convertTo(a_ch, CV_32FC1, 1.0 / 255.0);

    // Reference profile: green channel (index 1), resized to match if needed.
    constexpr int kRefCh = 1;
    cv::Mat prof_mag, prof_phase, prof_cons;
    if (profile.width != w || profile.height != h) {
        cv::resize(profile.magnitude_bgr[kRefCh], prof_mag, {w, h}, 0, 0, cv::INTER_LINEAR);
        cv::resize(profile.phase_bgr[kRefCh], prof_phase, {w, h}, 0, 0, cv::INTER_LINEAR);
        cv::resize(profile.consistency_bgr[kRefCh], prof_cons, {w, h}, 0, 0, cv::INTER_LINEAR);
    } else {
        prof_mag = profile.magnitude_bgr[kRefCh];
        prof_phase = profile.phase_bgr[kRefCh];
        prof_cons = profile.consistency_bgr[kRefCh];
    }

    float noise = noise_correlation(a_ch, prof_mag);
    cv::Mat a_fft = fft_.forward(a_ch);
    float phase = carrier_phase_matching(a_fft, prof_phase);
    float struct_r = structure_ratio(a_fft, prof_mag, prof_cons);

    // multi_scale_consistency is 0.05 weight; reuse the BGR computation so the
    // a-vs-BGR confidence delta is attributable to the three chrominance metrics.
    float ms_cons = multi_scale_consistency(work, profile);

    result.noise_correlation = noise;
    result.carrier_phase_score = phase;
    result.structure_ratio = struct_r;
    result.multi_scale_consistency = ms_cons;

    result.confidence = std::clamp(
        kWeightNoiseCorr * noise
        + kWeightCarrierPhase * phase
        + kWeightStructure * struct_r
        + kWeightMultiScale * ms_cons, 0.0f, 1.0f);

    // Same content-aware threshold logic as detect_bgr, but measured on the `a`
    // plane's std (content in `a` drives the struct_ratio baseline up).
    float threshold = kDefaultThreshold;
    cv::Scalar a_mean, a_std_scalar;
    cv::meanStdDev(a_ch, a_mean, a_std_scalar);
    float a_std = static_cast<float>(a_std_scalar[0]);
    if (a_std > 0.05f) threshold = 0.70f;
    result.detected = result.confidence >= threshold;

    spdlog::debug("SynthID detect (LAB-a): noise={:.3f} phase={:.3f} struct={:.3f} "
                  "ms_cons={:.3f} -> confidence={:.3f} ({})",
                  noise, phase, struct_r, ms_cons,
                  result.confidence,
                  result.detected ? "DETECTED" : "not detected");

    return result;
}

float SynthidDetector::noise_correlation(
    const cv::Mat& channel,
    const cv::Mat& profile_mag) const
{
    // Extract noise residual via bilateral filter denoising
    cv::Mat denoised;
    cv::bilateralFilter(channel, denoised, 9, 75.0, 75.0);

    cv::Mat noise = channel - denoised;

    // FFT of noise residual
    cv::Mat noise_fft = fft_.forward(noise);
    cv::Mat noise_mag = FftContext::magnitude(noise_fft);

    // Normalized cross-correlation between noise spectrum and profile magnitude
    cv::Mat a, b;
    noise_mag.convertTo(a, CV_32FC1);
    profile_mag.convertTo(b, CV_32FC1);

    if (a.size() != b.size()) {
        cv::resize(a, a, b.size());
    }

    cv::Scalar mean_a = cv::mean(a);
    cv::Scalar mean_b = cv::mean(b);

    a -= mean_a;
    b -= mean_b;

    double dot = a.dot(b);
    double norm_a = std::sqrt(a.dot(a));
    double norm_b = std::sqrt(b.dot(b));

    if (norm_a < 1e-9 || norm_b < 1e-9) return 0.0f;

    // NCC in range [-1, 1] → map to [0, 1]
    float ncc = static_cast<float>(dot / (norm_a * norm_b));
    return std::clamp((ncc + 1.0f) * 0.5f, 0.0f, 1.0f);
}

float SynthidDetector::carrier_phase_matching(
    const cv::Mat& channel_fft,
    const cv::Mat& profile_phase) const
{
    cv::Mat img_phase = FftContext::phase(channel_fft);

    cv::Mat a, b;
    img_phase.convertTo(a, CV_32FC1);
    profile_phase.convertTo(b, CV_32FC1);

    if (a.size() != b.size()) {
        cv::resize(a, a, b.size());
    }

    // Phase coherence: cosine of phase difference
    cv::Mat phase_diff;
    cv::subtract(a, b, phase_diff);
    cv::Mat cos_vals(phase_diff.size(), CV_32FC1);
    for (int y = 0; y < phase_diff.rows; ++y) {
        const float* src = phase_diff.ptr<float>(y);
        float* dst = cos_vals.ptr<float>(y);
        for (int x = 0; x < phase_diff.cols; ++x) {
            dst[x] = std::cos(src[x]);
        }
    }

    // Average cosine similarity → 1.0 means perfect match
    float coherence = static_cast<float>(cv::mean(cos_vals)[0]);

    // Map from [-1, 1] to [0, 1]
    return std::clamp((coherence + 1.0f) * 0.5f, 0.0f, 1.0f);
}

float SynthidDetector::structure_ratio(
    const cv::Mat& channel_fft,
    const cv::Mat& profile_mag,
    const cv::Mat& profile_consistency) const
{
    cv::Mat img_mag = FftContext::magnitude(channel_fft);

    cv::Mat a, b, c;
    img_mag.convertTo(a, CV_32FC1);
    profile_mag.convertTo(b, CV_32FC1);
    profile_consistency.convertTo(c, CV_32FC1);

    if (a.size() != b.size()) {
        cv::resize(a, a, b.size());
    }
    if (a.size() != c.size()) {
        cv::resize(c, c, a.size());
    }

    // Weighted NCC: correlation between image magnitude and codebook magnitude,
    // weighted by codebook consistency (high-consistency bins contribute more).
    // This measures how well the image's spectral shape matches the carrier template.
    cv::Mat wa, wb;
    cv::multiply(a, c, wa);   // img_mag * consistency
    cv::multiply(b, c, wb);   // profile_mag * consistency

    cv::Scalar mean_wa = cv::mean(wa);
    cv::Scalar mean_wb = cv::mean(wb);

    wa -= mean_wa;
    wb -= mean_wb;

    double dot = wa.dot(wb);
    double norm_a = std::sqrt(wa.dot(wa));
    double norm_b = std::sqrt(wb.dot(wb));

    if (norm_a < 1e-9 || norm_b < 1e-9) return 0.0f;

    float ncc = static_cast<float>(dot / (norm_a * norm_b));
    return std::clamp((ncc + 1.0f) * 0.5f, 0.0f, 1.0f);
}

float SynthidDetector::multi_scale_consistency(
    const cv::Mat& image,
    const SpectralProfile& profile) const
{
    // Measure phase coherence against codebook carrier template at multiple scales.
    // SynthID carrier has stable phase — if image phase matches codebook phase
    // consistently across scales, it's a real carrier signal.

    auto phase_match_at_scale = [&](float scale) -> float {
        cv::Mat scaled;
        cv::resize(image, scaled, {}, scale, scale, cv::INTER_LINEAR);
        int sw = scaled.cols;
        int sh = scaled.rows;

        float total = 0.0f;
        for (int ch = 0; ch < 3; ++ch) {
            cv::Mat ch_float;
            cv::Mat ch_arr[3];
            cv::split(scaled, ch_arr);
            ch_arr[ch].convertTo(ch_float, CV_32FC1, 1.0 / 255.0);

            cv::Mat ch_fft = fft_.forward(ch_float);
            cv::Mat img_phase = FftContext::phase(ch_fft);

            cv::Mat prof_phase, prof_cons;
            if (profile.width != sw || profile.height != sh) {
                cv::resize(profile.phase_bgr[ch], prof_phase, {sw, sh}, 0, 0, cv::INTER_LINEAR);
                cv::resize(profile.consistency_bgr[ch], prof_cons, {sw, sh}, 0, 0, cv::INTER_LINEAR);
            } else {
                prof_phase = profile.phase_bgr[ch];
                prof_cons = profile.consistency_bgr[ch];
            }

            // Phase coherence weighted by codebook consistency:
            // cos(img_phase - codebook_phase) at high-consistency bins
            cv::Mat phase_diff;
            cv::subtract(img_phase, prof_phase, phase_diff);

            cv::Mat cos_vals(phase_diff.size(), CV_32FC1);
            for (int y = 0; y < phase_diff.rows; ++y) {
                const float* src = phase_diff.ptr<float>(y);
                float* dst = cos_vals.ptr<float>(y);
                for (int x = 0; x < phase_diff.cols; ++x) {
                    dst[x] = std::cos(src[x]);
                }
            }

            // Weight by codebook consistency — only high-consistency bins matter
            cv::Mat weighted;
            cv::multiply(cos_vals, prof_cons, weighted);

            double cons_sum = cv::sum(prof_cons)[0];
            if (cons_sum < 1e-9) continue;

            float coherence = static_cast<float>(cv::sum(weighted)[0] / cons_sum);
            total += std::clamp((coherence + 1.0f) * 0.5f, 0.0f, 1.0f);
        }
        return total / 3.0f;
    };

    float score_1x = phase_match_at_scale(1.0f);
    float score_half = phase_match_at_scale(0.5f);
    float score_quarter = phase_match_at_scale(0.25f);

    // Consistency across scales: carrier should be detected at all scales
    float mean_score = (score_1x + score_half + score_quarter) / 3.0f;
    float variance = ((score_1x - mean_score) * (score_1x - mean_score)
                    + (score_half - mean_score) * (score_half - mean_score)
                    + (score_quarter - mean_score) * (score_quarter - mean_score)) / 3.0f;

    float consistency = 1.0f - std::min(std::sqrt(variance) * 3.0f, 1.0f);
    return std::clamp(mean_score * consistency, 0.0f, 1.0f);
}

} // namespace wmr
