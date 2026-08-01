#include "synthid/spectral_codebook.hpp"

#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <spdlog/spdlog.h>

namespace wmr {

void SpectralCodebook::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open codebook: " + path);
    }

    char magic[kMagicLen];
    file.read(magic, kMagicLen);
    const bool is_v2 = (std::memcmp(magic, kMagic, kMagicLen) == 0);
    const bool is_v1 = (std::memcmp(magic, kLegacyMagic, kMagicLen) == 0);
    if (!is_v2 && !is_v1) {
        throw std::runtime_error("Invalid codebook format (bad magic)");
    }

    uint32_t count = 0;
    file.read(reinterpret_cast<char*>(&count), sizeof(uint32_t));

    profiles_.clear();

    for (uint32_t i = 0; i < count; ++i) {
        SpectralProfile profile;
        uint32_t w = 0, h = 0;

        file.read(reinterpret_cast<char*>(&w), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&h), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&profile.sample_count), sizeof(int32_t));

        profile.width = static_cast<int>(w);
        profile.height = static_cast<int>(h);

        for (int ch = 0; ch < 3; ++ch) {
            uint32_t rows = 0, cols = 0;
            file.read(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&cols), sizeof(uint32_t));

            profile.magnitude_bgr[ch] = cv::Mat(rows, cols, CV_32FC1);
            file.read(reinterpret_cast<char*>(profile.magnitude_bgr[ch].data),
                      rows * cols * sizeof(float));

            profile.phase_bgr[ch] = cv::Mat(rows, cols, CV_32FC1);
            file.read(reinterpret_cast<char*>(profile.phase_bgr[ch].data),
                      rows * cols * sizeof(float));

            profile.consistency_bgr[ch] = cv::Mat(rows, cols, CV_32FC1);
            file.read(reinterpret_cast<char*>(profile.consistency_bgr[ch].data),
                      rows * cols * sizeof(float));

            if (is_v2) {
                profile.phase_consistency_bgr[ch] = cv::Mat(rows, cols, CV_32FC1);
                file.read(reinterpret_cast<char*>(profile.phase_consistency_bgr[ch].data),
                          rows * cols * sizeof(float));
            } else {
                // v1 codebooks have no phase-consistency plane; default to all-ones
                // so the subtractor's soft gate is a no-op (preserves v1 behavior).
                profile.phase_consistency_bgr[ch] = cv::Mat::ones(rows, cols, CV_32FC1);
            }
        }

        profiles_[{profile.height, profile.width}] = std::move(profile);
    }

    spdlog::debug("Loaded codebook: {} profiles from {}", count, path);
}

void SpectralCodebook::save(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create codebook: " + path);
    }

    file.write(kMagic, kMagicLen);

    uint32_t count = static_cast<uint32_t>(profiles_.size());
    file.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));

    for (const auto& [key, profile] : profiles_) {
        uint32_t w = static_cast<uint32_t>(profile.width);
        uint32_t h = static_cast<uint32_t>(profile.height);
        file.write(reinterpret_cast<const char*>(&w), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&h), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&profile.sample_count), sizeof(int32_t));

        for (int ch = 0; ch < 3; ++ch) {
            uint32_t rows = static_cast<uint32_t>(profile.magnitude_bgr[ch].rows);
            uint32_t cols = static_cast<uint32_t>(profile.magnitude_bgr[ch].cols);
            file.write(reinterpret_cast<const char*>(&rows), sizeof(uint32_t));
            file.write(reinterpret_cast<const char*>(&cols), sizeof(uint32_t));

            file.write(reinterpret_cast<const char*>(profile.magnitude_bgr[ch].data),
                       rows * cols * sizeof(float));
            file.write(reinterpret_cast<const char*>(profile.phase_bgr[ch].data),
                       rows * cols * sizeof(float));
            file.write(reinterpret_cast<const char*>(profile.consistency_bgr[ch].data),
                       rows * cols * sizeof(float));

            // phase_consistency (v2). Fall back to ones if the in-memory profile
            // didn't populate it (e.g. hand-built in a test) so serialization stays
            // well-formed; all-ones is a no-op gate.
            cv::Mat pcons = profile.phase_consistency_bgr[ch];
            if (pcons.size() != profile.magnitude_bgr[ch].size()) {
                pcons = cv::Mat::ones(profile.magnitude_bgr[ch].size(), CV_32FC1);
            }
            file.write(reinterpret_cast<const char*>(pcons.data),
                       rows * cols * sizeof(float));
        }
    }

    spdlog::debug("Saved codebook: {} profiles to {}", count, path);
}

bool SpectralCodebook::has_profile(int width, int height) const {
    return profiles_.find({height, width}) != profiles_.end();
}

const SpectralProfile& SpectralCodebook::get_profile(int width, int height) const {
    auto it = profiles_.find({height, width});
    if (it != profiles_.end()) {
        return it->second;
    }

    // Fallback: find nearest resolution by aspect ratio + pixel count
    double target_ar = static_cast<double>(height) / (width + 1e-9);
    double target_px = height * width;

    double best_dist = std::numeric_limits<double>::max();
    const SpectralProfile* best = nullptr;

    for (const auto& [key, profile] : profiles_) {
        double ar = static_cast<double>(key.first) / (key.second + 1e-9);
        double px = key.first * key.second;

        double ar_diff = std::abs(ar - target_ar);
        double px_diff = std::abs(px - target_px) / target_px;

        double dist = ar_diff * 2.0 + px_diff;
        if (dist < best_dist) {
            best_dist = dist;
            best = &profile;
        }
    }

    if (best) {
        spdlog::debug("Codebook: no exact profile for {}x{}, using nearest {}x{}",
                      width, height, best->width, best->height);
        return *best;
    }

    throw std::runtime_error("Codebook is empty — no profiles available");
}

void SpectralCodebook::add_profile(const SpectralProfile& profile) {
    profiles_[{profile.height, profile.width}] = profile;
}

} // namespace wmr
