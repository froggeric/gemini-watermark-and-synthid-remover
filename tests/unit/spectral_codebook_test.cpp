#include <catch2/catch_test_macros.hpp>

#include "synthid/spectral_codebook.hpp"
#include <opencv2/core.hpp>
#include <filesystem>
#include <fstream>

using namespace wmr;

TEST_CASE("SpectralCodebook save and load round-trip", "[codebook]") {
    // Create a codebook with one profile
    SpectralCodebook codebook;
    SpectralProfile profile;
    profile.width = 64;
    profile.height = 64;
    profile.sample_count = 5;

    for (int ch = 0; ch < 3; ++ch) {
        profile.magnitude_bgr[ch] = cv::Mat::ones(64, 64, CV_32FC1) * 0.1f;
        profile.phase_bgr[ch] = cv::Mat::zeros(64, 64, CV_32FC1);
        profile.consistency_bgr[ch] = cv::Mat::ones(64, 64, CV_32FC1) * 0.05f;
        profile.phase_consistency_bgr[ch] = cv::Mat::ones(64, 64, CV_32FC1) * 0.8f;
    }

    codebook.add_profile(profile);

    // Save
    std::string path = "/tmp/wmr_test_codebook.cb";
    REQUIRE_NOTHROW(codebook.save(path));
    REQUIRE(std::filesystem::exists(path));

    // Load
    SpectralCodebook loaded;
    REQUIRE_NOTHROW(loaded.load(path));

    REQUIRE(loaded.has_profile(64, 64));

    auto& loaded_profile = loaded.get_profile(64, 64);
    REQUIRE(loaded_profile.width == 64);
    REQUIRE(loaded_profile.height == 64);
    REQUIRE(loaded_profile.sample_count == 5);

    // Check data integrity
    cv::Mat diff;
    cv::absdiff(loaded_profile.magnitude_bgr[0], profile.magnitude_bgr[0], diff);
    double max_err = 0;
    cv::minMaxLoc(diff, nullptr, &max_err);
    REQUIRE(max_err < 1e-6);

    // phase_consistency plane (v2) round-trips too
    cv::absdiff(loaded_profile.phase_consistency_bgr[0],
                profile.phase_consistency_bgr[0], diff);
    cv::minMaxLoc(diff, nullptr, &max_err);
    REQUIRE(max_err < 1e-6);

    // Cleanup
    std::filesystem::remove(path);
}

TEST_CASE("SpectralCodebook nearest-resolution fallback", "[codebook]") {
    SpectralCodebook codebook;
    SpectralProfile profile;
    profile.width = 512;
    profile.height = 512;
    profile.sample_count = 3;

    for (int ch = 0; ch < 3; ++ch) {
        profile.magnitude_bgr[ch] = cv::Mat::ones(512, 512, CV_32FC1);
        profile.phase_bgr[ch] = cv::Mat::zeros(512, 512, CV_32FC1);
        profile.consistency_bgr[ch] = cv::Mat::ones(512, 512, CV_32FC1);
    }

    codebook.add_profile(profile);

    // Exact match
    REQUIRE(codebook.has_profile(512, 512));

    // Nearest match for 1024x1024 (same aspect ratio)
    auto& nearest = codebook.get_profile(1024, 1024);
    REQUIRE(nearest.width == 512);
    REQUIRE(nearest.height == 512);
}

TEST_CASE("SpectralCodebook empty codebook throws", "[codebook]") {
    SpectralCodebook codebook;
    REQUIRE_THROWS_AS(codebook.get_profile(100, 100), std::runtime_error);
}

TEST_CASE("SpectralCodebook invalid format throws", "[codebook]") {
    std::string path = "/tmp/wmr_test_bad.cb";
    {
        std::ofstream file(path, std::ios::binary);
        file.write("BADFORMAT", 9);
    }

    SpectralCodebook codebook;
    REQUIRE_THROWS_AS(codebook.load(path), std::runtime_error);

    std::filesystem::remove(path);
}

TEST_CASE("SpectralCodebook loads v1 (WMRCB01) and defaults phase_consistency to ones", "[codebook]") {
    // Hand-encode a v1 codebook: magic WMRCB01, 1 profile, 64x64, 3 planes/channel
    // (magnitude, phase, consistency) and NO phase_consistency plane.
    std::string path = "/tmp/wmr_test_v1.cb";
    {
        std::ofstream f(path, std::ios::binary);
        f.write("WMRCB01", 7);
        uint32_t count = 1;
        f.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));
        uint32_t w = 64, h = 64;
        int32_t sc = 3;
        f.write(reinterpret_cast<const char*>(&w), sizeof(uint32_t));
        f.write(reinterpret_cast<const char*>(&h), sizeof(uint32_t));
        f.write(reinterpret_cast<const char*>(&sc), sizeof(int32_t));
        cv::Mat plane = cv::Mat::ones(64, 64, CV_32FC1) * 0.5f;
        for (int ch = 0; ch < 3; ++ch) {
            uint32_t rows = 64, cols = 64;
            f.write(reinterpret_cast<const char*>(&rows), sizeof(uint32_t));
            f.write(reinterpret_cast<const char*>(&cols), sizeof(uint32_t));
            f.write(reinterpret_cast<const char*>(plane.data), 64 * 64 * sizeof(float));
            f.write(reinterpret_cast<const char*>(plane.data), 64 * 64 * sizeof(float));
            f.write(reinterpret_cast<const char*>(plane.data), 64 * 64 * sizeof(float));
        }
    }

    SpectralCodebook cb;
    REQUIRE_NOTHROW(cb.load(path));

    auto& p = cb.get_profile(64, 64);
    REQUIRE(p.phase_consistency_bgr[0].size() == cv::Size(64, 64));
    cv::Mat diff;
    cv::absdiff(p.phase_consistency_bgr[0], cv::Mat::ones(64, 64, CV_32FC1), diff);
    double max_err = 0;
    cv::minMaxLoc(diff, nullptr, &max_err);
    REQUIRE(max_err < 1e-6);

    std::filesystem::remove(path);
}
