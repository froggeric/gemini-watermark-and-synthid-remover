#pragma once

#include <opencv2/core.hpp>
#include <map>
#include <string>
#include <utility>
#include <vector>

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

    // Mutable exact-match access, for external bin seeding. Returns nullptr
    // when no exact (width,height) profile exists (no nearest-resolution
    // fallback: seeding a foreign resolution would be a silent no-op).
    SpectralProfile* find_exact_profile(int width, int height);

    // Per-bin max-merge of another codebook's carrier-ACTIVATION planes and
    // magnitude into *this. For each profile key present in BOTH *this and
    // other: cv::max of consistency_bgr, phase_consistency_bgr, and
    // magnitude_bgr (so external carrier bins raise those gates / magnitude
    // without clobbering measured values that are already higher). phase_bgr
    // is INTENTIONALLY NOT MERGED: phase is a circular quantity (atan2) and
    // element-wise max of two phases is meaningless (it would replace a
    // measured phase of -1.0 with 0.0); the subtractor builds its estimate
    // via from_polar(subtract_mag, prof_phase), so a wrong phase subtracts in
    // the wrong direction and can ADD the watermark instead of removing it.
    // The left-hand codebook's phase is preserved as-is; callers wanting a
    // unified phase should rebuild the codebook rather than merge. Profile
    // keys present only in other are SKIPPED (never silently insert a foreign
    // resolution into *this). Returns the count of shared keys merged.
    int merge_from(const SpectralCodebook& other);

    int profile_count() const { return static_cast<int>(profiles_.size()); }

private:
    std::map<std::pair<int,int>, SpectralProfile> profiles_;
    mutable SpectralProfile fallback_;

    static constexpr const char* kMagic = "WMRCB02";        // v2: adds phase_consistency plane
    static constexpr const char* kLegacyMagic = "WMRCB01";  // v1: no phase_consistency (defaults to ones)
    static constexpr int kMagicLen = 7;
};

// Seed candidate SynthID carrier bins in a codebook profile. For the profile
// matching {width,height} (exact match only; if no exact match, logs a warning
// and returns without mutation), set BOTH consistency_bgr[ch].at<float>(y,x) =
// 1.0f AND phase_consistency_bgr[ch].at<float>(y,x) = 1.0f for every (x,y) in
// bins and all 3 BGR channels. CRITICAL: the subtractor gates on BOTH planes
// (consistency_bgr via the floor-remap, phase_consistency_bgr via a direct
// multiply), so seeding only consistency_bgr is a SILENT NO-OP. A fully-seeded
// bin has effective gate weight 1.0 (gate fully open: subtractor acts there).
// Bins are (col,row) = (x,y) into the profile's rows x cols = height x width
// FFT grid; out-of-range bins are clamped to the grid and logged.
void seed_carrier_bins(SpectralCodebook& cb,
                       const std::vector<std::pair<int,int>>& bins,
                       int width, int height);

} // namespace wmr
