#pragma once

#include <opencv2/core.hpp>
#include <map>
#include <string>
#include <utility>

namespace wmr {

struct SpectralProfile {
    int width = 0;
    int height = 0;
    cv::Mat magnitude_bgr[3];     // CV_32FC1 per channel
    cv::Mat phase_bgr[3];         // CV_32FC1 per channel
    cv::Mat consistency_bgr[3];   // CV_32FC1 per channel [0,1]
    cv::Mat phase_consistency_bgr[3];  // CV_32FC1 per channel [0,1]: |mean(exp(i*phase))|
    int sample_count = 0;
};

class SpectralCodebook {
public:
    void load(const std::string& path);
    void save(const std::string& path) const;

    const SpectralProfile& get_profile(int width, int height) const;
    bool has_profile(int width, int height) const;
    void add_profile(const SpectralProfile& profile);

    int profile_count() const { return static_cast<int>(profiles_.size()); }

private:
    std::map<std::pair<int,int>, SpectralProfile> profiles_;
    mutable SpectralProfile fallback_;

    static constexpr const char* kMagic = "WMRCB02";        // v2: adds phase_consistency plane
    static constexpr const char* kLegacyMagic = "WMRCB01";  // v1: no phase_consistency (defaults to ones)
    static constexpr int kMagicLen = 7;
};

} // namespace wmr
