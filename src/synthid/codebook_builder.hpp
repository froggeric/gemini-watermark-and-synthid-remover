#pragma once

#include <string>
#include <utility>
#include <vector>
#include "synthid/spectral_codebook.hpp"
#include "core/fft_context.hpp"

namespace wmr {

struct BuildStats {
    int total_images = 0;
    int profiles_created = 0;
    int skipped_low_samples = 0;
    int profiles_seeded = 0;  // profiles that got --carrier-grid bins applied
};

class CodebookBuilder {
public:
    explicit CodebookBuilder(FftContext& fft);

    BuildStats build_from_directory(const std::string& dir_path,
                                    const std::string& output_path);

    // Opt-in carrier-bin seeding (WS2b). When non-empty, every fresh profile
    // gets seed_carrier_bins() applied to it AFTER finalize and BEFORE save,
    // setting both consistency_bgr and phase_consistency_bgr to 1.0 at each
    // bin so the subtractor's gate is fully open there. Bins are (x,y) =
    // (col,row) on the per-profile rows x cols grid.
    void set_carrier_bins(const std::vector<std::pair<int,int>>& bins) {
        carrier_bins_ = bins;
    }

private:
    FftContext& fft_;
    std::vector<std::pair<int,int>> carrier_bins_;

    struct ProfileAccumulator {
        int width = 0;
        int height = 0;
        int count = 0;
        cv::Mat mag_sum[3];
        cv::Mat cos_sum[3];     // sum of cos(phase) across captures (circular)
        cv::Mat sin_sum[3];     // sum of sin(phase) across captures (circular)
        cv::Mat mag_sq_sum[3];  // For computing std dev
    };

    void accumulate(const cv::Mat& image,
                    std::map<std::pair<int,int>, ProfileAccumulator>& accums) const;

    SpectralProfile finalize(const ProfileAccumulator& acc) const;
};

} // namespace wmr
