// CodebookBuilder discriminative carrier-selection tests + degenerate-codebook
// OOM guard.
//
// (a) Builder fix: a codebook built from near-identical images (the dots-bug
//     scenario: pure-black captures with a fixed visible mark) must NOT mark
//     ~100% of bins as carrier. The old magnitude-variance consistency
//     (1 - std/max_std) saturated at ~0.999 everywhere on near-identical images
//     because the per-bin magnitude std was tiny relative to the global max,
//     so every bin read as "stable" and the subtractor's gate was fully open
//     at ~97% of bins. The new metric (normalized log1p(mag) * pcons) uses the
//     phase coherence as the primary discriminator and magnitude as a floor, so
//     zero-magnitude background bins score ~0 (excluded) regardless of their
//     numerically-trivial phase.
// (b) A genuine carrier bin (phase-stable across VARIED captures) must be kept:
//     a fixed additive sinusoid at a known bin is phase-coherent and carries
//     non-trivial magnitude, so the new metric scores it high while the
//     surrounding random-content bins score low.
// (c) Degenerate-codebook OOM guard: an all-zero (or all-NaN) codebook profile
//     must be rejected cleanly, not crash the subtractor with a NaN cascade.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "core/fft_context.hpp"
#include "synthid/codebook_builder.hpp"
#include "synthid/codebook_subtractor.hpp"
#include "synthid/spectral_codebook.hpp"

namespace fs = std::filesystem;

using namespace wmr;

namespace {

// Build a SpectralCodebook from a set of synthetic cv::Mat images by writing
// them into a scratch dir and calling the public builder API. Wipes the dir
// first so no stale frames leak across runs.
SpectralCodebook build_from_images(const std::vector<cv::Mat>& images,
                                   FftContext& fft,
                                   const std::string& tag) {
    fs::path scratch = fs::temp_directory_path() / ("wmr_cb_bld_" + tag);
    fs::remove_all(scratch);
    fs::create_directories(scratch);
    for (size_t i = 0; i < images.size(); ++i) {
        cv::imwrite((scratch / (std::to_string(i) + ".png")).string(), images[i]);
    }
    fs::path cb_path = scratch / "build.cb";
    CodebookBuilder builder(fft);
    builder.build_from_directory(scratch.string(), cb_path.string());
    SpectralCodebook cb;
    cb.load(cb_path.string());
    return cb;
}

// Fraction of bins in a single-channel float plane strictly above a threshold,
// excluding the DC neighborhood (radius <= dc_excl around the 4 grid corners,
// matching the builder's DC mask) so DC / its wraparound copies do not skew it.
double frac_above_excl_dc(const cv::Mat& plane, float thresh, int dc_excl = 4) {
    const int rows = plane.rows, cols = plane.cols;
    cv::Mat mask = cv::Mat::ones(plane.size(), CV_8UC1);
    for (int cy : {0, rows - 1}) {
        for (int cx : {0, cols - 1}) {
            for (int dy = -dc_excl; dy <= dc_excl; ++dy) {
                for (int dx = -dc_excl; dx <= dc_excl; ++dx) {
                    const int y = cy + dy, x = cx + dx;
                    if (y >= 0 && y < rows && x >= 0 && x < cols) {
                        mask.at<uint8_t>(y, x) = 0;
                    }
                }
            }
        }
    }
    cv::Mat above;
    cv::compare(plane, thresh, above, cv::CMP_GT);
    above &= mask;
    cv::Scalar s = cv::sum(mask);
    const double total = s[0];
    cv::Scalar s2 = cv::sum(above);
    const double hit = s2[0];
    return total > 0 ? hit / total : 0.0;
}

}  // namespace

// (a) Near-identical pure-black images with a fixed visible mark: the builder
// must NOT mark ~100% of bins as carrier. The background bins (pure black, near-
// zero FFT magnitude) must drop out of the carrier-selection gate.
TEST_CASE("CodebookBuilder: near-identical images do not saturate carrier gate",
          "[synthid][codebook-builder]") {
    // 8 near-identical 64x64 images: pure black (value 0) + a small fixed bright
    // diamond-ish mark in the top-left corner. Value 0 gives an exactly-zero FFT
    // on the background (the worst case for the old metric: std=0 everywhere ->
    // consistency = 1 - 0/max_std saturates at ~0.999). A tiny per-image dither
    // (value 0 or 1) keeps the frames "near-identical" not "byte-identical" so
    // the accumulator exercises its multi-image path.
    constexpr int W = 64, H = 64, N = 8;
    std::vector<cv::Mat> imgs;
    for (int k = 0; k < N; ++k) {
        cv::Mat m = cv::Mat::zeros(H, W, CV_8UC3);
        // Fixed 6x6 bright mark at (4,4), identical across captures (the
        // "visible diamond" analogue).
        for (int y = 4; y < 10; ++y) {
            for (int x = 4; x < 10; ++x) {
                m.at<cv::Vec3b>(y, x) = cv::Vec3b(200, 200, 200);
            }
        }
        // Trivial dither so captures are near-identical, not byte-identical.
        if (k % 2 == 0) m.at<cv::Vec3b>(0, 0) = cv::Vec3b(1, 1, 1);
        imgs.push_back(m);
    }

    FftContext fft;
    const SpectralCodebook cb = build_from_images(imgs, fft, "near_ident");
    REQUIRE(cb.has_profile(W, H));
    const auto& profile = cb.get_profile(W, H);

    // The old metric saturated at ~0.999 across ~100% of bins on this input.
    // The new metric must drop the background bins (near-zero magnitude) out of
    // the carrier-selection gate: the active-bin fraction (consistency > 0.6,
    // DC excluded) must be WELL below 100%.
    for (int ch = 0; ch < 3; ++ch) {
        const double frac = frac_above_excl_dc(profile.consistency_bgr[ch], 0.6f);
        INFO("channel " << ch << " active-bin frac (consistency>0.6, DC excl): " << frac);
        // Was ~1.0 before the fix. Require a material drop; the mark + DC area
        // is a small fraction of the 64x64 grid.
        CHECK(frac < 0.50);
    }
}

// (b) A genuine carrier bin (phase-stable across VARIED captures) is kept while
// surrounding random-content bins are dropped. The new metric's whole point:
// phase coherence is the discriminator, so a fixed additive carrier (stable
// phase + non-trivial magnitude across varied content) scores high, while
// content-driven bins (random phase) score low.
TEST_CASE("CodebookBuilder: phase-stable carrier bin kept, content bins dropped",
          "[synthid][codebook-builder]") {
    constexpr int W = 64, H = 64, N = 10;
    // Carrier bin: (cx, cy) in the non-fftshift FFT grid. A real-valued cosine
    // cos(2*pi*(cy*y + cx*x)/H) lands a positive-real peak at FFT bin (cx, cy)
    // and its conjugate mirror. Pick an off-axis bin away from DC and the axes.
    const int cy = 10, cx = 8;

    // Deterministic PRNG so the test is reproducible.
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> uval(0, 255);

    std::vector<cv::Mat> imgs;
    for (int k = 0; k < N; ++k) {
        cv::Mat m(H, W, CV_8UC3);
        // Random content per capture (varied phase -> incoherent across caps).
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                m.at<cv::Vec3b>(y, x) = cv::Vec3b(uval(rng), uval(rng), uval(rng));
            }
        }
        // Fixed additive carrier sinusoid (identical across captures -> phase
        // stable -> coherent). Amplitude large enough to clear the random
        // content's per-bin noise floor at the carrier bin.
        const float amp = 40.0f;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const float ph = 2.0f * static_cast<float>(M_PI) *
                                 (cy * y + cx * x) / static_cast<float>(H);
                const float s = amp * std::cos(ph);
                for (int ch = 0; ch < 3; ++ch) {
                    int v = m.at<cv::Vec3b>(y, x)[ch];
                    v = std::clamp(static_cast<int>(v + s), 0, 255);
                    m.at<cv::Vec3b>(y, x)[ch] = static_cast<uint8_t>(v);
                }
            }
        }
        imgs.push_back(m);
    }

    FftContext fft;
    const SpectralCodebook cb = build_from_images(imgs, fft, "carrier_keep");
    REQUIRE(cb.has_profile(W, H));
    const auto& profile = cb.get_profile(W, H);

    // Use the G channel (off-axis bin, away from DC mask).
    const auto& cons = profile.consistency_bgr[1];
    const auto& pcons = profile.phase_consistency_bgr[1];

    const float carrier_cons = cons.at<float>(cy, cx);
    const float carrier_pcons = pcons.at<float>(cy, cx);

    // Average a few off-axis, off-grid content bins for the control level.
    const int ctrl_bins[][2] = {{20, 5}, {5, 30}, {40, 50}, {15, 45}, {50, 20}};
    double ctrl_cons_sum = 0, ctrl_pcons_sum = 0;
    int n_ctrl = 0;
    for (const auto& b : ctrl_bins) {
        ctrl_cons_sum += cons.at<float>(b[0], b[1]);
        ctrl_pcons_sum += pcons.at<float>(b[0], b[1]);
        ++n_ctrl;
    }
    const double ctrl_cons = ctrl_cons_sum / n_ctrl;
    const double ctrl_pcons = ctrl_pcons_sum / n_ctrl;

    INFO("carrier bin (" << cy << "," << cx << "): cons=" << carrier_cons
         << " pcons=" << carrier_pcons);
    INFO("content control bins: mean cons=" << ctrl_cons
         << " mean pcons=" << ctrl_pcons);

    // (1) The carrier bin is kept: its phase coherence is high (stable sinusoid
    //     across captures) and its consistency is high (non-trivial magnitude).
    CHECK(carrier_pcons > 0.80f);
    CHECK(carrier_cons > 0.50f);

    // (2) Content bins are dropped: their phase is incoherent (random content)
    //     and their consistency is low.
    CHECK(ctrl_pcons < 0.50f);
    CHECK(ctrl_cons < carrier_cons * 0.6);

    // (3) The active-bin fraction across the whole spectrum is well below 100%
    //     (content bins dropped, only the carrier + its mirror + DC retained).
    const double frac = frac_above_excl_dc(cons, 0.6f);
    INFO("whole-spectrum active-bin frac (consistency>0.6, DC excl): " << frac);
    CHECK(frac < 0.30);
}

// (c) Degenerate-codebook OOM guard: an all-zero-magnitude profile (and an
// all-NaN-consistency profile) must not crash the subtractor. Before the guard
// a NaN/zero cascade could trigger a multi-exabyte cv::Mat allocation. The
// subtractor must either no-op cleanly or throw a bounded, readable error.
TEST_CASE("CodebookSubtractor: degenerate codebook does not crash (OOM guard)",
          "[synthid][codebook-builder]") {
    constexpr int W = 64, H = 64;

    auto make_image = []() {
        cv::Mat m(H, W, CV_8UC3);
        cv::randu(m, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
        return m;
    };

    FftContext fft;
    CodebookSubtractor subtractor(fft);
    RemovalConfig config;
    config.strength = RemovalStrength::Moderate;

    SECTION("all-zero magnitude and phase") {
        SpectralCodebook cb;
        SpectralProfile p;
        p.width = W; p.height = H; p.sample_count = 3;
        for (int ch = 0; ch < 3; ++ch) {
            p.magnitude_bgr[ch] = cv::Mat::zeros(H, W, CV_32FC1);
            p.phase_bgr[ch] = cv::Mat::zeros(H, W, CV_32FC1);
            p.consistency_bgr[ch] = cv::Mat::ones(H, W, CV_32FC1);
            p.phase_consistency_bgr[ch] = cv::Mat::ones(H, W, CV_32FC1);
        }
        cb.add_profile(p);

        cv::Mat img = make_image();
        cv::Mat original = img.clone();
        REQUIRE_NOTHROW(subtractor.remove_synthid(img, cb, config));
        // Output must be a finite, same-size image (no crash, no garbage dim).
        REQUIRE(img.size() == original.size());
        REQUIRE(cv::checkRange(img, true, nullptr, 0.0, 256.0));
    }

    SECTION("all-NaN consistency (0/0 from identical captures)") {
        SpectralCodebook cb;
        SpectralProfile p;
        p.width = W; p.height = H; p.sample_count = 3;
        for (int ch = 0; ch < 3; ++ch) {
            p.magnitude_bgr[ch] = cv::Mat::ones(H, W, CV_32FC1) * 10.0f;
            p.phase_bgr[ch] = cv::Mat::zeros(H, W, CV_32FC1);
            // NaN consistency is the degenerate case (0/0 from identical frames
            // without the guard). polarToCart / min / FFT with NaN must not OOM.
            p.consistency_bgr[ch] = cv::Mat(H, W, CV_32FC1, cv::Scalar(std::nanf("")));
            p.phase_consistency_bgr[ch] = cv::Mat::ones(H, W, CV_32FC1);
        }
        cb.add_profile(p);

        cv::Mat img = make_image();
        REQUIRE_NOTHROW(subtractor.remove_synthid(img, cb, config));
        REQUIRE(img.size() == cv::Size(W, H));
        REQUIRE(cv::checkRange(img, true, nullptr, 0.0, 256.0));
    }

    SECTION("NaN magnitude (minMaxLoc skips NaN, guard must still catch it)") {
        // cv::minMaxLoc silently skips NaN entries (NaN < x is false), so a
        // naive "isfinite(minMaxLoc max)" check misses a NaN magnitude. The
        // guard uses the (x != x) compare trick to detect NaN directly. A
        // NaN-magnitude codebook must be rejected, not crash.
        SpectralCodebook cb;
        SpectralProfile p;
        p.width = W; p.height = H; p.sample_count = 3;
        for (int ch = 0; ch < 3; ++ch) {
            p.magnitude_bgr[ch] = cv::Mat(H, W, CV_32FC1, cv::Scalar(std::nanf("")));
            p.phase_bgr[ch] = cv::Mat::zeros(H, W, CV_32FC1);
            p.consistency_bgr[ch] = cv::Mat::ones(H, W, CV_32FC1);
            p.phase_consistency_bgr[ch] = cv::Mat::ones(H, W, CV_32FC1);
        }
        cb.add_profile(p);

        cv::Mat img = make_image();
        REQUIRE_NOTHROW(subtractor.remove_synthid(img, cb, config));
        REQUIRE(img.size() == cv::Size(W, H));
        REQUIRE(cv::checkRange(img, true, nullptr, 0.0, 256.0));
    }
}
