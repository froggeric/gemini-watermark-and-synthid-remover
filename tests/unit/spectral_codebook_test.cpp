#include <catch2/catch_test_macros.hpp>

#include "core/fft_context.hpp"
#include "synthid/codebook_builder.hpp"
#include "synthid/spectral_codebook.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

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

TEST_CASE("seed_carrier_bins sets BOTH consistency and phase_consistency at seeded bins",
          "[codebook]") {
    // WS2b: the post-WS2a subtractor gates on BOTH consistency_bgr (floor-remap)
    // AND phase_consistency_bgr (direct multiply). Seeding only one plane is a
    // silent no-op. Verify seed_carrier_bins sets BOTH at every seeded bin and
    // leaves a non-seeded bin untouched.
    constexpr int W = 64, H = 64;
    SpectralCodebook cb;
    SpectralProfile p;
    p.width = W;
    p.height = H;
    p.sample_count = 4;
    for (int ch = 0; ch < 3; ++ch) {
        // Start both gates fully CLOSED: consistency low, phase_consistency low.
        p.consistency_bgr[ch] = cv::Mat::ones(H, W, CV_32FC1) * 0.10f;
        p.phase_consistency_bgr[ch] = cv::Mat::ones(H, W, CV_32FC1) * 0.05f;
        // Magnitude / phase planes can stay empty; seed_carrier_bins doesn't
        // touch them.
    }
    cb.add_profile(p);

    const std::vector<std::pair<int,int>> bins = {{10, 4}, {31, 22}};
    REQUIRE_NOTHROW(seed_carrier_bins(cb, bins, W, H));

    auto& prof = cb.get_profile(W, H);

    // Every seeded bin: BOTH planes must read 1.0 (post-seed effective gate
    // weight = 1.0 across all 3 BGR channels).
    for (const auto& [x, y] : bins) {
        for (int ch = 0; ch < 3; ++ch) {
            INFO("seeded bin (" << x << "," << y << ") ch=" << ch
                 << " consistency=" << prof.consistency_bgr[ch].at<float>(y, x)
                 << " phase_consistency=" << prof.phase_consistency_bgr[ch].at<float>(y, x));
            REQUIRE(prof.consistency_bgr[ch].at<float>(y, x) == 1.0f);
            REQUIRE(prof.phase_consistency_bgr[ch].at<float>(y, x) == 1.0f);
        }
    }

    // A non-seeded bin must be UNTOUCHED (still the original low value).
    const std::pair<int,int> ctrl = {5, 5};
    for (int ch = 0; ch < 3; ++ch) {
        INFO("control bin (" << ctrl.first << "," << ctrl.second << ") ch=" << ch);
        REQUIRE(prof.consistency_bgr[ch].at<float>(ctrl.second, ctrl.first) == 0.10f);
        REQUIRE(prof.phase_consistency_bgr[ch].at<float>(ctrl.second, ctrl.first) == 0.05f);
    }
}

TEST_CASE("seed_carrier_bins is a no-op when no exact profile matches", "[codebook]") {
    // Seeding a foreign resolution would be a silent no-op through the nearest-
    // resolution fallback in get_profile; seed_carrier_bins must refuse instead.
    SpectralCodebook cb;
    SpectralProfile p;
    p.width = 64;
    p.height = 64;
    p.sample_count = 1;
    for (int ch = 0; ch < 3; ++ch) {
        p.consistency_bgr[ch] = cv::Mat::zeros(64, 64, CV_32FC1);
        p.phase_consistency_bgr[ch] = cv::Mat::zeros(64, 64, CV_32FC1);
    }
    cb.add_profile(p);

    // 128x128 has no exact profile; seeding must do nothing.
    REQUIRE_NOTHROW(seed_carrier_bins(cb, {{10, 10}}, 128, 128));

    // The 64x64 profile's planes must be unchanged.
    auto& prof = cb.get_profile(64, 64);
    double cons_max = 0, pcons_max = 0;
    cv::minMaxLoc(prof.consistency_bgr[0], nullptr, &cons_max);
    cv::minMaxLoc(prof.phase_consistency_bgr[0], nullptr, &pcons_max);
    REQUIRE(cons_max == 0.0);
    REQUIRE(pcons_max == 0.0);
}

TEST_CASE("SpectralCodebook::merge_from: max on activation + magnitude, preserves dst phase",
          "[codebook]") {
    // WS2b: merge_from is the external-seed injection point. Per-bin cv::max
    // applies to the carrier-ACTIVATION planes (consistency_bgr,
    // phase_consistency_bgr) AND magnitude_bgr, so a foreign seed raises
    // those gates / magnitude without clobbering measured values that are
    // already higher. phase_bgr is INTENTIONALLY NOT MERGED: phase is a
    // circular quantity (atan2), and per-bin max of two phases is meaningless
    // (a measured phase of -1.0 would be replaced by max(-1.0, 0.5) = 0.5);
    // the subtractor builds its watermark estimate via
    // FftContext::from_polar(subtract_mag, prof_phase), so a wrong phase
    // subtracts in the wrong direction (can ADD the watermark). The left
    // codebook's phase must survive the merge. Foreign-only keys are SKIPPED
    // (never silently inserted).
    constexpr int W = 32, H = 32;

    // Destination: a measured-style profile with consistency = 0.4 everywhere
    // and phase_consistency = 0.3 everywhere, except a "measured high" bin at
    // (8, 6) where consistency = 0.95 (must NOT be clobbered by a lower seed).
    // phase_bgr is set to a non-trivial pattern (including a NEGATIVE value
    // at (2,3)) so a buggy max-merge of phase would visibly corrupt it.
    SpectralCodebook dst;
    {
        SpectralProfile p;
        p.width = W; p.height = H; p.sample_count = 5;
        for (int ch = 0; ch < 3; ++ch) {
            p.consistency_bgr[ch] = cv::Mat::ones(H, W, CV_32FC1) * 0.40f;
            p.phase_consistency_bgr[ch] = cv::Mat::ones(H, W, CV_32FC1) * 0.30f;
            p.magnitude_bgr[ch] = cv::Mat::ones(H, W, CV_32FC1) * 0.50f;
            p.phase_bgr[ch] = cv::Mat::ones(H, W, CV_32FC1) * 0.40f;
            p.consistency_bgr[ch].at<float>(6, 8) = 0.95f;   // measured-high bin
            p.phase_bgr[ch].at<float>(3, 2) = -0.7f;         // negative phase
            p.phase_bgr[ch].at<float>(6, 8) = 1.2f;          // distinct positive
        }
        dst.add_profile(p);
    }

    // Source: a seed profile (same key). At (2,3) the seed raises both gates
    // to 1.0. At (8,6) ("measured-high" in dst) the seed does NOT touch the
    // bin (leaves src's 0.0 there), so dst's 0.95 must survive the per-bin max
    // (a lower src value must not clobber a higher measured dst value).
    // src's phase_bgr is set to DIFFERENT values (most higher than dst's) so
    // a buggy max-merge of phase would visibly corrupt dst's phase. src's
    // magnitude_bgr is 0.0 everywhere EXCEPT (20,10) where it is 0.80 (> dst's
    // 0.50) so the magnitude-max path is genuinely exercised.
    SpectralCodebook src;
    {
        SpectralProfile p;
        p.width = W; p.height = H; p.sample_count = 1;
        for (int ch = 0; ch < 3; ++ch) {
            p.consistency_bgr[ch] = cv::Mat::zeros(H, W, CV_32FC1);
            p.phase_consistency_bgr[ch] = cv::Mat::zeros(H, W, CV_32FC1);
            p.magnitude_bgr[ch] = cv::Mat::zeros(H, W, CV_32FC1);
            p.phase_bgr[ch] = cv::Mat::ones(H, W, CV_32FC1) * 0.90f;  // > dst's 0.40
            // Seed raises both gates to 1.0 ONLY at (2,3). (8,6) stays at src's
            // 0.0 so dst's higher measured value wins the per-bin max there.
            p.consistency_bgr[ch].at<float>(3, 2) = 1.0f;
            p.phase_consistency_bgr[ch].at<float>(3, 2) = 1.0f;
            // phase values at the assertion bins, picked HIGHER than dst's so
            // max-merge would corrupt: (2,3) 0.5 > dst's -0.7; (8,6) 1.5 > 1.2.
            p.phase_bgr[ch].at<float>(3, 2) = 0.5f;
            p.phase_bgr[ch].at<float>(6, 8) = 1.5f;
            // magnitude at (20,10): src=0.80 > dst's 0.50, so a max-merge must
            // raise dst to 0.80 here (cv::min instead of cv::max would leave
            // dst at 0.50, failing the assertion below).
            p.magnitude_bgr[ch].at<float>(10, 20) = 0.80f;
        }
        src.add_profile(p);
    }

    // Also add a foreign-only profile to src; it must NOT appear in dst.
    {
        SpectralProfile foreign;
        foreign.width = 99; foreign.height = 99; foreign.sample_count = 1;
        for (int ch = 0; ch < 3; ++ch) {
            foreign.consistency_bgr[ch] = cv::Mat::ones(99, 99, CV_32FC1);
            foreign.phase_consistency_bgr[ch] = cv::Mat::ones(99, 99, CV_32FC1);
            foreign.magnitude_bgr[ch] = cv::Mat::ones(99, 99, CV_32FC1);
            foreign.phase_bgr[ch] = cv::Mat::zeros(99, 99, CV_32FC1);
        }
        src.add_profile(foreign);
    }

    const int merged = dst.merge_from(src);
    REQUIRE(merged == 1);  // only the 32x32 key was shared

    REQUIRE(dst.has_profile(W, H));
    REQUIRE_FALSE(dst.has_profile(99, 99));  // foreign key was skipped

    auto& prof = dst.get_profile(W, H);

    // (2, 3): src raised both gates from 0 -> 1.0; per-bin max gives 1.0.
    REQUIRE(prof.consistency_bgr[0].at<float>(3, 2) == 1.0f);
    REQUIRE(prof.phase_consistency_bgr[0].at<float>(3, 2) == 1.0f);
    // phase at (2,3) MUST be dst's original -0.7, NOT max(-0.7, 0.5) = 0.5.
    // A buggy max-merge of phase would set this to 0.5 (wrong direction).
    REQUIRE(prof.phase_bgr[0].at<float>(3, 2) == -0.7f);

    // (8, 6): dst's measured 0.95 must beat src's 0.0 (a lower src value must
    // NOT clobber a higher measured dst value at the same bin).
    REQUIRE(prof.consistency_bgr[0].at<float>(6, 8) == 0.95f);
    REQUIRE(prof.phase_consistency_bgr[0].at<float>(6, 8) == 0.30f);  // dst baseline
    // phase at (8,6) MUST be dst's original 1.2, NOT max(1.2, 1.5) = 1.5.
    REQUIRE(prof.phase_bgr[0].at<float>(6, 8) == 1.2f);

    // Magnitude per-bin max, bidirectionally:
    //   - At (20,10): src=0.80 > dst's 0.50, so the merged value MUST be 0.80
    //     (cv::min instead of cv::max would leave dst at 0.50, failing here).
    //   - At (15,15) and everywhere else: src=0.0 < dst's 0.50, so dst's
    //     measured 0.50 survives (a lower src value must not clobber a higher
    //     measured dst value).
    REQUIRE(prof.magnitude_bgr[0].at<float>(10, 20) == 0.80f);
    REQUIRE(prof.magnitude_bgr[0].at<float>(15, 15) == 0.50f);

    // A bin neither covers stays at dst's baseline values (0.40 / 0.30 / 0.40).
    REQUIRE(prof.consistency_bgr[0].at<float>(15, 15) == 0.40f);
    REQUIRE(prof.phase_consistency_bgr[0].at<float>(15, 15) == 0.30f);
    REQUIRE(prof.phase_bgr[0].at<float>(15, 15) == 0.40f);  // NOT src's 0.90
}

TEST_CASE("consistency_bgr discriminates carrier vs content bins once DC is excluded from max_std",
          "[codebook][synthid]") {
    // Phase 0.5 regression test for the consistency_bgr normalization.
    //
    // consistency_bgr = 1 - std/max_std, where max_std is the peak per-bin
    // magnitude std across captures. Pre-fix, max_std was the GLOBAL max via a
    // bare cv::minMaxLoc(std_dev). DC sits at [0,0] in the natural (non-
    // fftshift) layout and mean brightness varies more across captures than
    // any real frequency, so the global max is always at the DC neighborhood,
    // making std/max_std ~= 0 for every non-DC bin and consistency saturate
    // at ~0.99 across the whole spectrum. The magnitude-variability gate was
    // a near-no-op; only the phase_consistency gate discriminated.
    //
    // This test builds a synthetic codebook from N captures where:
    //   - mean brightness (DC) varies a lot across captures (DC std huge),
    //   - one frequency bin is a stable carrier (constant cosine amplitude),
    //   - one frequency bin is variable content (cosine amplitude varies).
    // Post-fix, max_std excludes the DC neighborhood, so the variable bin's
    // std becomes the normalization reference and consistency there drops
    // near 0 while the stable carrier bin stays near 1.
    //
    // PRE-FIX THIS ASSERTION WOULD HAVE FAILED: with the bare global max,
    // both the stable and the variable bin read consistency ~= 0.9+ (DC
    // dominated max_std), so the `variable < 0.4` check below is the
    // load-bearing regression guard. Confirmed by temporarily reverting the
    // mask in codebook_builder.cpp::finalize (both bins read > 0.85).

    constexpr int W = 64, H = 64;
    constexpr int N = 8;

    // A stable "carrier" frequency and a variable "content" frequency, both
    // off-axis and well outside the DC mask radius (radius 4 -> bins within
    // 4px of any corner are masked). (10,8) has radius ~12.8, (22,18) ~28.5.
    constexpr int stable_x = 10, stable_y = 8;
    constexpr int variable_x = 22, variable_y = 18;
    static_assert(stable_x > 5 && stable_y > 5, "stable bin must be outside DC mask");
    static_assert(variable_x > 5 && variable_y > 5, "variable bin must be outside DC mask");

    // Per-capture mean brightness (DC). Wide span so DC std dominates the
    // whole spectrum by a large margin (the pre-fix saturation condition).
    const float base_means[N] = {50, 75, 100, 125, 150, 175, 200, 225};
    // Stable carrier amplitude is CONSTANT across captures.
    const float stable_amp = 7.0f;
    // Variable content amplitude swings across captures.
    const float variable_amps[N] = {3.0f, 5.0f, 8.0f, 12.0f, 15.0f, 19.0f, 23.0f, 27.0f};

    namespace fs = std::filesystem;
    const fs::path scratch = fs::temp_directory_path() / "wmr_consistency_metric_test";
    fs::remove_all(scratch);
    fs::create_directories(scratch);

    for (int i = 0; i < N; ++i) {
        cv::Mat img(H, W, CV_8UC3);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const float ph_stable =
                    2.0f * static_cast<float>(CV_PI) *
                    (stable_x * x + stable_y * y) / static_cast<float>(W);
                const float ph_var =
                    2.0f * static_cast<float>(CV_PI) *
                    (variable_x * x + variable_y * y) / static_cast<float>(W);
                float v = base_means[i]
                          + stable_amp * std::cos(ph_stable)
                          + variable_amps[i] * std::cos(ph_var);
                // Keep within [4, 251] so there is zero clipping (clipping
                // would inject harmonics and contaminate the bins under test).
                v = std::clamp(v, 4.0f, 251.0f);
                const uint8_t u = static_cast<uint8_t>(std::lroundf(v));
                img.at<cv::Vec3b>(y, x) = cv::Vec3b(u, u, u);  // grey (B==G==R)
            }
        }
        std::ostringstream name;
        name << "synth_" << std::setw(2) << std::setfill('0') << i << ".png";
        cv::imwrite((scratch / name.str()).string(), img);
    }

    FftContext fft;
    CodebookBuilder builder(fft);
    const fs::path cb_path = scratch / "consistency.cb";
    builder.build_from_directory(scratch.string(), cb_path.string());

    SpectralCodebook cb;
    cb.load(cb_path.string());
    REQUIRE(cb.has_profile(W, H));
    const auto& prof = cb.get_profile(W, H);

    // Read consistency at both bins (channel 0; BGR are identical on grey
    // input, so any channel tells the same story).
    const float cons_stable = prof.consistency_bgr[0].at<float>(stable_y, stable_x);
    const float cons_variable = prof.consistency_bgr[0].at<float>(variable_y, variable_x);

    INFO("stable bin (" << stable_x << "," << stable_y << ") consistency=" << cons_stable
         << " | variable bin (" << variable_x << "," << variable_y << ") consistency="
         << cons_variable << " | samples=" << prof.sample_count);

    // Post-fix: the carrier bin is stable (consistency high), the content bin
    // is variable (consistency low). This is the discrimination the gate was
    // supposed to provide.
    REQUIRE(cons_stable > 0.6f);
    REQUIRE(cons_variable < 0.4f);

    fs::remove_all(scratch);
}
