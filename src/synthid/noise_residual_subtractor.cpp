#include "synthid/noise_residual_subtractor.hpp"

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

namespace wmr {

NoiseResidualSubtractor::NoiseResidualSubtractor(FftContext& fft)
    : fft_(fft) {}

auto NoiseResidualSubtractor::get_strength_params(RemovalStrength strength) -> StrengthParams {
    switch (strength) {
        case RemovalStrength::Gentle:
            return {0.60f, 0.50f, 30.0f};
        case RemovalStrength::Moderate:
            return {0.80f, 0.70f, 25.0f};
        case RemovalStrength::Aggressive:
            return {0.95f, 0.90f, 20.0f};
        case RemovalStrength::Maximum:
            return {1.00f, 0.98f, 12.0f};
    }
    return {0.80f, 0.70f, 25.0f};
}

void NoiseResidualSubtractor::remove_synthid(
    cv::Mat& image,
    const RemovalConfig& config)
{
    if (image.empty()) return;

    // WS3: LAB `a`-channel sibling path. Fully separate so the default BGR path
    // (config.lab_a == false) is byte-for-byte unchanged.
    if (config.lab_a) {
        remove_synthid_lab_a(image, config);
        return;
    }

    if (image.channels() == 4) {
        cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
    } else if (image.channels() == 1) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    }

    const int h = image.rows;
    const int w = image.cols;

    RemovalStrength base_strength = config.strength;
    if (config.custom_strength >= 0.0f) {
        if (config.custom_strength <= 0.25f) base_strength = RemovalStrength::Gentle;
        else if (config.custom_strength <= 0.50f) base_strength = RemovalStrength::Moderate;
        else if (config.custom_strength <= 0.75f) base_strength = RemovalStrength::Aggressive;
        else base_strength = RemovalStrength::Maximum;
    }

    cv::Mat work;
    image.convertTo(work, CV_32FC3, 1.0 / 255.0);

    cv::Mat channels[3];
    cv::split(work, channels);

    cv::Scalar img_std;
    cv::meanStdDev(work, cv::Scalar(), img_std);
    float avg_std = static_cast<float>((img_std[0] + img_std[1] + img_std[2]) / 3.0);
    bool is_content_image = avg_std > 0.05f;

    if (is_content_image) {
        spdlog::info("Content image detected (std={:.4f}): spectral disruption (carrier <0.1% of energy).", avg_std);
    } else {
        spdlog::info("Uniform image detected (std={:.4f}): direct carrier suppression.", avg_std);
    }

    spdlog::info("Codebook-free SynthID suppression: {}x{}, strength={}, content={}",
                 w, h, static_cast<int>(base_strength), is_content_image);

    // For uniform images (std < 0.05): the carrier IS the only spectral content.
    // The most effective removal is to replace the image with its mean color
    // plus tiny natural sensor noise. This completely destroys the carrier.
    if (!is_content_image) {
        cv::Scalar img_mean = cv::mean(work);
        cv::RNG rng(std::rand());

        for (int ch = 0; ch < 3; ++ch) {
            float mean_val = static_cast<float>(img_mean[ch]);
            // Strength controls how much of the original variation to keep
            float keep_ratio = 0.0f;
            switch (base_strength) {
                case RemovalStrength::Gentle:    keep_ratio = 0.30f; break;
                case RemovalStrength::Moderate:  keep_ratio = 0.15f; break;
                case RemovalStrength::Aggressive: keep_ratio = 0.05f; break;
                case RemovalStrength::Maximum:   keep_ratio = 0.00f; break;
            }
            // Always add sensor noise — a perfectly uniform image has undefined
            // FFT phase that can correlate with codebook profiles.
            cv::Mat noise(h, w, CV_32FC1);
            rng.fill(noise, cv::RNG::NORMAL, mean_val, 0.002f);
            if (keep_ratio > 0.0f) {
                channels[ch] = channels[ch] * keep_ratio + noise * (1.0f - keep_ratio);
            } else {
                channels[ch] = noise;
            }
        }

        cv::Mat merged;
        cv::merge(channels, 3, merged);
        merged = cv::max(merged, 0.0);
        merged = cv::min(merged, 1.0);
        merged.convertTo(image, CV_8UC3, 255.0);

        spdlog::info("Uniform image: replaced with mean+noise (keep_ratio={:.2f})",
                     base_strength == RemovalStrength::Maximum ? 0.0f : 0.15f);
        return;
    }

    // Carrier frequency band where SynthID encodes data.
    // For uniform images: carrier IS the entire spectrum (1/f^1.3).
    // Replace everything except DC to fully neutralize carrier.
    // For content images: carrier occupies r=30-500 in mid frequencies.
    const float band_inner = is_content_image ? 30.0f : 3.0f;
    float max_r = std::sqrt(static_cast<float>((h/2)*(h/2) + (w/2)*(w/2)));
    const float band_outer = is_content_image ? 500.0f : max_r;
    const float ramp_width = 20.0f;

    // Build frequency band masks
    cv::Mat carrier_mask(h, w, CV_32FC1);   // 1 inside carrier band, 0 outside
    cv::Mat stop_mask(h, w, CV_32FC1);      // 0 inside carrier band, 1 outside (band-stop)
    for (int y = 0; y < h; ++y) {
        float fy = static_cast<float>(y);
        if (fy > h / 2.0f) fy -= h;
        for (int x = 0; x < w; ++x) {
            float fx = static_cast<float>(x);
            if (fx > w / 2.0f) fx -= w;
            float dist = std::sqrt(fy * fy + fx * fx);

            float lo = std::clamp((dist - band_inner) / ramp_width, 0.0f, 1.0f);
            float hi = std::clamp((band_outer - dist) / ramp_width, 0.0f, 1.0f);
            float carrier = lo * hi;
            carrier_mask.at<float>(y, x) = carrier;
            stop_mask.at<float>(y, x) = 1.0f - carrier;
        }
    }

    // Strength-dependent replacement intensity.
    // At Maximum strength, carrier band is fully replaced.
    // At lower strengths, blend original and replacement.
    float replacement_blend = 0.0f;
    switch (base_strength) {
        case RemovalStrength::Gentle:    replacement_blend = 0.40f; break;
        case RemovalStrength::Moderate:  replacement_blend = 0.65f; break;
        case RemovalStrength::Aggressive: replacement_blend = 0.85f; break;
        case RemovalStrength::Maximum:   replacement_blend = 1.00f; break;
    }

    // Phase noise sigma scales with strength
    float phase_sigma = 0.0f;
    if (is_content_image) {
        switch (base_strength) {
            case RemovalStrength::Gentle:    phase_sigma = 0.10f; break;
            case RemovalStrength::Moderate:  phase_sigma = 0.20f; break;
            case RemovalStrength::Aggressive: phase_sigma = 0.30f; break;
            case RemovalStrength::Maximum:   phase_sigma = 0.40f; break;
        }
    } else {
        // For uniform images, aggressive phase disruption in carrier band
        switch (base_strength) {
            case RemovalStrength::Gentle:    phase_sigma = 0.50f; break;
            case RemovalStrength::Moderate:  phase_sigma = 1.00f; break;
            case RemovalStrength::Aggressive: phase_sigma = 2.00f; break;
            case RemovalStrength::Maximum:   phase_sigma = 3.14f; break;
        }
    }

    cv::RNG rng(42);

    for (int ch = 0; ch < 3; ++ch) {
        cv::Mat ch_fft = fft_.forward(channels[ch]);
        cv::Mat img_mag = FftContext::magnitude(ch_fft);
        cv::Mat img_phase = FftContext::phase(ch_fft);

        cv::Mat new_mag = img_mag.clone();
        cv::Mat new_phase = img_phase.clone();

        if (!is_content_image) {
            // Spectral band replacement for uniform images:
            // The carrier IS the image content. Replace nearly the full spectrum
            // with random noise magnitude and fully random phase.
            // Only preserve the DC component (average color).
            cv::Mat replacement_mag(h, w, CV_32FC1);
            // Use very small random magnitudes — carrier-free uniform images
            // should have near-zero spectral energy outside DC.
            // DC magnitude for a black image is ~0, for gray is ~0.5.
            float dc_mag = img_mag.at<float>(h / 2, w / 2);
            for (int y = 0; y < h; ++y) {
                float fy = static_cast<float>(y);
                if (fy > h / 2.0f) fy -= h;
                for (int x = 0; x < w; ++x) {
                    float fx = static_cast<float>(x);
                    if (fx > w / 2.0f) fx -= w;
                    float dist = std::sqrt(fy * fy + fx * fx);

                    // Beyond DC: small random magnitude proportional to 1/dist
                    // to mimic natural noise floor
                    if (dist < 1.0f) {
                        replacement_mag.at<float>(y, x) = 0.0f;
                    } else {
                        // Natural noise floor: ~1/dist decay with randomness
                        float natural_floor = dc_mag * 0.001f / std::pow(dist, 0.5f);
                        replacement_mag.at<float>(y, x) = natural_floor;
                    }
                }
            }

            // Add randomness to replacement magnitude to avoid creating
            // a new detectable pattern
            cv::Mat mag_jitter(h, w, CV_32FC1);
            rng.fill(mag_jitter, cv::RNG::NORMAL, 1.0, 0.3);
            cv::max(mag_jitter, 0.2, mag_jitter);
            cv::min(mag_jitter, 3.0, mag_jitter);
            replacement_mag = replacement_mag.mul(mag_jitter);

            // Blend: keep (1-blend) of original, add blend of replacement in carrier band
            cv::Mat blended_mag = img_mag.mul(1.0f - carrier_mask) +
                (img_mag * (1.0f - replacement_blend) + replacement_mag * replacement_blend)
                    .mul(carrier_mask);

            // Preserve DC component
            blended_mag.at<float>(h / 2, w / 2) = img_mag.at<float>(h / 2, w / 2);

            new_mag = blended_mag;

            // Fully randomize phase everywhere except DC
            cv::Mat random_phase(h, w, CV_32FC1);
            rng.fill(random_phase, cv::RNG::UNIFORM, -static_cast<float>(CV_PI), static_cast<float>(CV_PI));
            new_phase = img_phase.mul(1.0f - carrier_mask) + random_phase.mul(carrier_mask);
            // Keep DC phase unchanged
            new_phase.at<float>(h / 2, w / 2) = img_phase.at<float>(h / 2, w / 2);
        } else {
            // Content images: disrupt carrier in both magnitude and phase.
            // The carrier contributes ~8% above baseline in detection scores.
            // Noise_corr and struct_ratio are magnitude-domain metrics,
            // so phase noise alone has no effect. We need magnitude perturbation
            // in the carrier band (r=30-500).
            float mag_noise_strength = 0.0f;
            switch (base_strength) {
                case RemovalStrength::Gentle:    mag_noise_strength = 0.01f; break;
                case RemovalStrength::Moderate:  mag_noise_strength = 0.03f; break;
                case RemovalStrength::Aggressive: mag_noise_strength = 0.05f; break;
                case RemovalStrength::Maximum:   mag_noise_strength = 0.10f; break;
            }

            if (mag_noise_strength > 0.0f) {
                // Multiply carrier band magnitude by random factors near 1.0
                cv::Mat mag_perturbation(h, w, CV_32FC1);
                rng.fill(mag_perturbation, cv::RNG::NORMAL, 1.0, mag_noise_strength);
                cv::max(mag_perturbation, 0.5, mag_perturbation);
                cv::min(mag_perturbation, 1.5, mag_perturbation);
                new_mag = img_mag.mul(1.0f - carrier_mask) +
                          img_mag.mul(mag_perturbation).mul(carrier_mask);
            }

            if (phase_sigma > 0.0f) {
                cv::Mat phase_noise(h, w, CV_32FC1);
                rng.fill(phase_noise, cv::RNG::NORMAL, 0.0, phase_sigma);
                new_phase = img_phase + phase_noise.mul(carrier_mask);
            }
        }

        channels[ch] = fft_.inverse(FftContext::from_polar(new_mag, new_phase));
    }

    spdlog::debug("Applied spectral band replacement: blend={:.2f}, phase_sigma={:.2f}, band=[{},{}]",
                  replacement_blend, phase_sigma, band_inner, band_outer);

    cv::Mat merged;
    cv::merge(channels, 3, merged);
    cv::GaussianBlur(merged, merged, {3, 3}, 0.4);

    merged = cv::max(merged, 0.0);
    merged = cv::min(merged, 1.0);
    merged.convertTo(image, CV_8UC3, 255.0);

    spdlog::debug("Codebook-free SynthID suppression complete");
}

void NoiseResidualSubtractor::remove_synthid_lab_a(
    cv::Mat& image, const RemovalConfig& config)
{
    // WS3 experiment: operate on the LAB `a` (green-red opponent) channel only.
    // vitotitto/synthid-fingerprint-analysis reported SynthID most detectable in
    // `a` (AUC 0.977/0.926). This path makes that measurable on our fixtures.
    // Data-gated: see docs/research/synthid-lab-a-experiment.md for the verdict.

    if (image.channels() == 4) {
        cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
    } else if (image.channels() == 1) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    }

    const int h = image.rows;
    const int w = image.cols;

    RemovalStrength base_strength = config.strength;
    if (config.custom_strength >= 0.0f) {
        if (config.custom_strength <= 0.25f) base_strength = RemovalStrength::Gentle;
        else if (config.custom_strength <= 0.50f) base_strength = RemovalStrength::Moderate;
        else if (config.custom_strength <= 0.75f) base_strength = RemovalStrength::Aggressive;
        else base_strength = RemovalStrength::Maximum;
    }

    // BGR -> Lab (8-bit). OpenCV Lab: L,a,b in [0,255], a/b centered at 128
    // (neutral). Channel index 1 is `a` (green-red opponent).
    cv::Mat lab;
    cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);
    cv::Mat lab_planes[3];
    cv::split(lab, lab_planes);

    // Operate on `a` only, as float [0,1] (single plane, weight 1.0).
    cv::Mat a_f;
    lab_planes[1].convertTo(a_f, CV_32FC1, 1.0 / 255.0);

    // Uniform-vs-content decision uses the average std across all 3 Lab planes
    // (in [0,1]), mirroring the BGR path's avg-3-channel-std threshold. Using the
    // `a` std alone would misclassify luminance-only content (a near-flat `a`
    // channel with high L structure) as uniform, wrongly replacing `a` with noise
    // instead of running the carrier-band disruption. The decision is about
    // whether there is image STRUCTURE to preserve, which is a luminance question.
    cv::Scalar lab_std_scalar;
    cv::meanStdDev(lab, cv::Scalar(), lab_std_scalar);
    float lab_std = static_cast<float>(
        (lab_std_scalar[0] + lab_std_scalar[1] + lab_std_scalar[2]) / 3.0 / 255.0);
    bool is_content_image = lab_std > 0.05f;

    cv::Scalar a_std_scalar;
    cv::meanStdDev(a_f, cv::Scalar(), a_std_scalar);
    float a_std = static_cast<float>(a_std_scalar[0]);

    if (is_content_image) {
        spdlog::info("LAB-a: content image (lab-std={:.4f}, a-std={:.4f}): spectral disruption in `a`.",
                     lab_std, a_std);
    } else {
        spdlog::info("LAB-a: uniform image (lab-std={:.4f}, a-std={:.4f}): direct carrier suppression in `a`.",
                     lab_std, a_std);
    }
    spdlog::info("LAB-a codebook-free SynthID suppression (heuristic): "
                 "{}x{}, strength={}, content={}",
                 w, h, static_cast<int>(base_strength), is_content_image);

    if (!is_content_image) {
        // Uniform `a` plane: the carrier IS the only spectral content. Replace
        // with mean+noise, mirroring the BGR uniform branch (keep_ratio schedule).
        float mean_val = static_cast<float>(cv::mean(a_f)[0]);
        float keep_ratio = 0.0f;
        switch (base_strength) {
            case RemovalStrength::Gentle:    keep_ratio = 0.30f; break;
            case RemovalStrength::Moderate:  keep_ratio = 0.15f; break;
            case RemovalStrength::Aggressive: keep_ratio = 0.05f; break;
            case RemovalStrength::Maximum:   keep_ratio = 0.00f; break;
        }
        cv::RNG rng(std::rand());
        cv::Mat noise(h, w, CV_32FC1);
        rng.fill(noise, cv::RNG::NORMAL, mean_val, 0.002f);
        cv::Mat new_a;
        if (keep_ratio > 0.0f) {
            new_a = a_f * keep_ratio + noise * (1.0f - keep_ratio);
        } else {
            new_a = noise;
        }
        new_a = cv::max(new_a, 0.0);
        new_a = cv::min(new_a, 1.0);
        new_a.convertTo(lab_planes[1], CV_8U, 255.0);
        spdlog::info("LAB-a uniform: replaced `a` with mean+noise (keep_ratio={:.2f})",
                     base_strength == RemovalStrength::Maximum ? 0.0f : keep_ratio);
    } else {
        // Content `a` plane: disrupt carrier band (r=30-500) in magnitude + phase.
        const float band_inner = 30.0f;
        float max_r = std::sqrt(static_cast<float>((h / 2) * (h / 2) + (w / 2) * (w / 2)));
        (void)max_r;
        const float band_outer = 500.0f;
        const float ramp_width = 20.0f;

        cv::Mat carrier_mask(h, w, CV_32FC1);
        for (int y = 0; y < h; ++y) {
            float fy = static_cast<float>(y);
            if (fy > h / 2.0f) fy -= h;
            for (int x = 0; x < w; ++x) {
                float fx = static_cast<float>(x);
                if (fx > w / 2.0f) fx -= w;
                float dist = std::sqrt(fy * fy + fx * fx);
                float lo = std::clamp((dist - band_inner) / ramp_width, 0.0f, 1.0f);
                float hi = std::clamp((band_outer - dist) / ramp_width, 0.0f, 1.0f);
                carrier_mask.at<float>(y, x) = lo * hi;
            }
        }

        float mag_noise_strength = 0.0f;
        switch (base_strength) {
            case RemovalStrength::Gentle:    mag_noise_strength = 0.01f; break;
            case RemovalStrength::Moderate:  mag_noise_strength = 0.03f; break;
            case RemovalStrength::Aggressive: mag_noise_strength = 0.05f; break;
            case RemovalStrength::Maximum:   mag_noise_strength = 0.10f; break;
        }
        float phase_sigma = 0.0f;
        switch (base_strength) {
            case RemovalStrength::Gentle:    phase_sigma = 0.10f; break;
            case RemovalStrength::Moderate:  phase_sigma = 0.20f; break;
            case RemovalStrength::Aggressive: phase_sigma = 0.30f; break;
            case RemovalStrength::Maximum:   phase_sigma = 0.40f; break;
        }

        cv::RNG rng(42);

        cv::Mat ch_fft = fft_.forward(a_f);
        cv::Mat img_mag = FftContext::magnitude(ch_fft);
        cv::Mat img_phase = FftContext::phase(ch_fft);
        cv::Mat new_mag = img_mag.clone();
        cv::Mat new_phase = img_phase.clone();

        if (mag_noise_strength > 0.0f) {
            cv::Mat mag_perturbation(h, w, CV_32FC1);
            rng.fill(mag_perturbation, cv::RNG::NORMAL, 1.0, mag_noise_strength);
            cv::max(mag_perturbation, 0.5, mag_perturbation);
            cv::min(mag_perturbation, 1.5, mag_perturbation);
            new_mag = img_mag.mul(1.0f - carrier_mask) +
                      img_mag.mul(mag_perturbation).mul(carrier_mask);
        }
        if (phase_sigma > 0.0f) {
            cv::Mat phase_noise(h, w, CV_32FC1);
            rng.fill(phase_noise, cv::RNG::NORMAL, 0.0, phase_sigma);
            new_phase = img_phase + phase_noise.mul(carrier_mask);
        }

        cv::Mat new_a = fft_.inverse(FftContext::from_polar(new_mag, new_phase));
        // Light 3x3 blur on the processed `a` plane only (BGR path blurs the
        // merged image; here we blur `a` before merge so L and b stay untouched).
        cv::GaussianBlur(new_a, new_a, {3, 3}, 0.4);
        new_a = cv::max(new_a, 0.0);
        new_a = cv::min(new_a, 1.0);
        new_a.convertTo(lab_planes[1], CV_8U, 255.0);
    }

    // Merge [L, a_processed, b] (L and b byte-identical to input) and Lab -> BGR.
    cv::Mat merged_lab;
    cv::merge(lab_planes, 3, merged_lab);
    cv::cvtColor(merged_lab, image, cv::COLOR_Lab2BGR);

    spdlog::debug("LAB-a codebook-free SynthID suppression complete (heuristic)");
}

cv::Mat NoiseResidualSubtractor::compute_dc_ramp(int rows, int cols, float radius) {
    cv::Mat ramp(rows, cols, CV_32FC1);
    for (int y = 0; y < rows; ++y) {
        float fy = static_cast<float>(y);
        if (fy > rows / 2.0f) fy -= rows;
        for (int x = 0; x < cols; ++x) {
            float fx = static_cast<float>(x);
            if (fx > cols / 2.0f) fx -= cols;
            float dist = std::sqrt(fy * fy + fx * fx);
            ramp.at<float>(y, x) = std::clamp(dist / radius, 0.0f, 1.0f);
        }
    }
    return ramp;
}

cv::Mat NoiseResidualSubtractor::estimate_carrier_from_noise(
    const cv::Mat& channel_fft,
    const cv::Mat& channel_float,
    float removal_factor,
    float mag_cap,
    float dc_radius,
    int channel_idx)
{
    const int rows = channel_fft.rows;
    const int cols = channel_fft.cols;
    const float ch_weight = kChannelWeights[channel_idx];

    // Extract noise residual via bilateral filter
    cv::Mat denoised;
    cv::bilateralFilter(channel_float, denoised, 9, 75.0, 75.0);
    cv::Mat noise = channel_float - denoised;

    // FFT of noise residual → carrier estimate
    cv::Mat noise_fft = fft_.forward(noise);
    cv::Mat noise_mag = FftContext::magnitude(noise_fft);
    cv::Mat noise_phase = FftContext::phase(noise_fft);

    // DC exclusion ramp
    cv::Mat dc_ramp = compute_dc_ramp(rows, cols, dc_radius);
    noise_mag = noise_mag.mul(dc_ramp);

    // Scale by removal factor and channel weight
    noise_mag *= removal_factor * ch_weight;

    // Safety cap: never subtract more than mag_cap * |image_fft|
    cv::Mat img_mag = FftContext::magnitude(channel_fft);
    cv::Mat cap;
    img_mag.copyTo(cap);
    cap *= mag_cap;
    cv::min(noise_mag, cap, noise_mag);

    // Construct complex watermark estimate using noise residual phase
    return FftContext::from_polar(noise_mag, noise_phase);
}

} // namespace wmr
