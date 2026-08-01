#include "synthid/codebook_builder.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <vector>
#include <cmath>

namespace fs = std::filesystem;

namespace wmr {

CodebookBuilder::CodebookBuilder(FftContext& fft)
    : fft_(fft) {}

BuildStats CodebookBuilder::build_from_directory(
    const std::string& dir_path,
    const std::string& output_path)
{
    BuildStats stats;

    if (!fs::exists(dir_path) || !fs::is_directory(dir_path)) {
        throw std::runtime_error("Not a valid directory: " + dir_path);
    }

    std::map<std::pair<int,int>, ProfileAccumulator> accums;

    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (!entry.is_regular_file()) continue;

        auto ext = entry.path().extension().string();
        for (auto& c : ext) c = static_cast<char>(std::tolower(c));

        if (ext != ".png" && ext != ".jpg" && ext != ".jpeg" && ext != ".webp") continue;

        cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
        if (img.empty()) {
            spdlog::warn("Skipping unreadable: {}", entry.path().filename().string());
            continue;
        }

        accumulate(img, accums);
        ++stats.total_images;
        spdlog::debug("Accumulated: {} ({}x{})",
                      entry.path().filename().string(), img.cols, img.rows);
    }

    if (stats.total_images == 0) {
        throw std::runtime_error("No images found in: " + dir_path);
    }

    SpectralCodebook codebook;

    for (auto& [key, acc] : accums) {
        if (acc.count < 3) {
            spdlog::warn("Resolution {}x{} has only {} samples (minimum 3 recommended)",
                         acc.width, acc.height, acc.count);
            ++stats.skipped_low_samples;
        }

        SpectralProfile profile = finalize(acc);
        codebook.add_profile(profile);
        ++stats.profiles_created;

        // WS2b: opt-in carrier-bin seeding, applied AFTER finalize and BEFORE
        // save. Sets consistency_bgr = phase_consistency_bgr = 1.0 at each
        // user-named bin so the subtractor's gate is fully open there. No-op
        // when --carrier-grid was not passed (carrier_bins_ is empty).
        if (!carrier_bins_.empty()) {
            seed_carrier_bins(codebook, carrier_bins_, acc.width, acc.height);
            ++stats.profiles_seeded;
        }

        spdlog::info("Profile: {}x{} ({} samples)", acc.width, acc.height, acc.count);
    }

    codebook.save(output_path);
    spdlog::info("Saved codebook: {} profiles → {} ({} seeded)",
                 stats.profiles_created, output_path, stats.profiles_seeded);

    return stats;
}

void CodebookBuilder::accumulate(
    const cv::Mat& image,
    std::map<std::pair<int,int>, ProfileAccumulator>& accums) const
{
    cv::Mat work;
    if (image.channels() == 4) {
        cv::cvtColor(image, work, cv::COLOR_BGRA2BGR);
    } else if (image.channels() == 1) {
        cv::cvtColor(image, work, cv::COLOR_GRAY2BGR);
    } else {
        work = image.clone();
    }

    int w = work.cols;
    int h = work.rows;
    auto key = std::make_pair(h, w);

    auto& acc = accums[key];
    acc.width = w;
    acc.height = h;

    cv::Mat channels[3];
    cv::split(work, channels);

    for (int ch = 0; ch < 3; ++ch) {
        cv::Mat ch_float;
        channels[ch].convertTo(ch_float, CV_32FC1, 1.0 / 255.0);

        cv::Mat ch_fft = fft_.forward(ch_float);
        cv::Mat mag = FftContext::magnitude(ch_fft);
        cv::Mat ph = FftContext::phase(ch_fft);

        if (acc.count == 0) {
            acc.mag_sum[ch] = cv::Mat::zeros(mag.size(), CV_32FC1);
            acc.cos_sum[ch] = cv::Mat::zeros(ph.size(), CV_32FC1);
            acc.sin_sum[ch] = cv::Mat::zeros(ph.size(), CV_32FC1);
        }

        // Circular phase accumulation. The previous arithmetic mean of phase was
        // wrong across the -pi/pi wraparound; storing cos/sin lets finalize compute
        // both the correct mean direction (atan2) and the phase coherence (mean
        // resultant length) used to gate the subtractor.
        cv::Mat cos_mat(ph.size(), CV_32FC1);
        cv::Mat sin_mat(ph.size(), CV_32FC1);
        for (int y = 0; y < ph.rows; ++y) {
            const float* phr = ph.ptr<float>(y);
            float* cr = cos_mat.ptr<float>(y);
            float* sr = sin_mat.ptr<float>(y);
            for (int x = 0; x < ph.cols; ++x) {
                cr[x] = std::cos(phr[x]);
                sr[x] = std::sin(phr[x]);
            }
        }

        acc.mag_sum[ch] += mag;
        acc.cos_sum[ch] += cos_mat;
        acc.sin_sum[ch] += sin_mat;
    }

    ++acc.count;
}

SpectralProfile CodebookBuilder::finalize(const ProfileAccumulator& acc) const {
    SpectralProfile profile;
    profile.width = acc.width;
    profile.height = acc.height;
    profile.sample_count = acc.count;

    float n = static_cast<float>(acc.count);

    for (int ch = 0; ch < 3; ++ch) {
        // Average magnitude
        profile.magnitude_bgr[ch] = acc.mag_sum[ch] / n;

        // Circular mean phase + phase consistency (mean resultant length in [0,1]).
        // phase = atan2(sin_mean, cos_mean); consistency = sqrt(cos_mean^2 + sin_mean^2).
        cv::Mat cos_mean = acc.cos_sum[ch] / n;
        cv::Mat sin_mean = acc.sin_sum[ch] / n;
        cv::phase(cos_mean, sin_mean, profile.phase_bgr[ch], /*angleInDegrees=*/false);
        cv::magnitude(cos_mean, sin_mean, profile.phase_consistency_bgr[ch]);

        // Carrier-selection consistency: a discriminative score that combines
        // phase coherence (PRIMARY, the cross-capture mean resultant length
        // already computed into phase_consistency_bgr) with a magnitude floor
        // (SECONDARY, so zero-magnitude bins cannot pass on numerically-trivial
        // phase alone).
        //
        // The score is  score = log1p(mean_magnitude) * phase_coherence.
        // - log1p(magnitude) weights each bin by its (soft) magnitude, so a
        //   zero-magnitude background bin (pure-black capture) scores ~0
        //   regardless of its phase. This is what excludes the bulk of the
        //   spectrum on near-identical pure-black captures.
        // - phase_coherence drops bins whose phase is incoherent across
        //   captures (random content, sensor noise). This is what excludes
        //   content-driven bins on varied captures.
        // The product is the discriminative signal the reference carrier
        // finder (reverse-SynthID SynthIDCodebookFinder.find_fourier_carriers)
        // uses; it is normalized to [0,1] so the subtractor's existing
        // floor-remap gate keeps working unchanged.
        //
        // WHY this replaces the old 1 - std/max_std metric: that metric
        // normalized the per-bin magnitude std by the GLOBAL max std. On
        // near-identical images (e.g. HF gemini_black) the per-bin std is tiny
        // EVERYWHERE, so std/max_std was ~0 everywhere and consistency
        // saturated at ~0.999 across the whole spectrum (the builder marked
        // ~100% of bins as carrier and captured the visible diamond's broadband
        // FFT as if it were the invisible carrier; subtracting it imprinted the
        // "dots" artifact). On varied content the std was noise-dominated and
        // non-discriminative in the opposite direction (the carrier bin read
        // LOWER consistency than off-grid content bins). The phase-coherence x
        // magnitude product is discriminative in both regimes.
        cv::Mat log_mag;
        cv::log(profile.magnitude_bgr[ch] + 1.0f, log_mag);
        cv::Mat score = log_mag.mul(profile.phase_consistency_bgr[ch]);

        // Normalize to [0,1] using the max score outside the DC neighborhood
        // (DC's log1p(magnitude) dwarfs every real frequency; an unmasked max
        // would shrink every other bin toward 0 the same way the old metric
        // saturated them toward 1). DC sits at [0,0] in the natural (non-
        // fftshift) layout plus 3 wraparound grid corners.
        constexpr int kDcMaskRadius = 4;
        const int rows = score.rows, cols = score.cols;
        cv::Mat dc_mask = cv::Mat::ones(score.size(), CV_8UC1);
        for (int cy : {0, rows - 1}) {
            for (int cx : {0, cols - 1}) {
                for (int dy = -kDcMaskRadius; dy <= kDcMaskRadius; ++dy) {
                    for (int dx = -kDcMaskRadius; dx <= kDcMaskRadius; ++dx) {
                        const int y = cy + dy;
                        const int x = cx + dx;
                        if (y >= 0 && y < rows && x >= 0 && x < cols) {
                            dc_mask.at<uint8_t>(y, x) = 0;
                        }
                    }
                }
            }
        }

        double max_score = 0.0;
        cv::minMaxLoc(score, nullptr, &max_score, nullptr, nullptr, dc_mask);
        if (max_score > 1e-9) {
            profile.consistency_bgr[ch] = score / static_cast<float>(max_score);
        } else {
            // Degenerate: no bin carries a non-trivial coherent signal. Mark
            // everything non-carrier (gate fully closed) rather than saturating
            // at 1.0; the subtractor then no-ops this profile.
            profile.consistency_bgr[ch] = cv::Mat::zeros(score.size(), CV_32FC1);
        }
    }

    return profile;
}

} // namespace wmr
