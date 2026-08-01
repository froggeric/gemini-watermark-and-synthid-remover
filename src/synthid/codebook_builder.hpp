#pragma once

#include <string>
#include "synthid/spectral_codebook.hpp"
#include "core/fft_context.hpp"

namespace wmr {

struct BuildStats {
    int total_images = 0;
    int profiles_created = 0;
    int skipped_low_samples = 0;
};

class CodebookBuilder {
public:
    explicit CodebookBuilder(FftContext& fft);

    BuildStats build_from_directory(const std::string& dir_path,
                                    const std::string& output_path);

private:
    FftContext& fft_;

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
