#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "core/antidetect.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <string>

using namespace wmr::antidetect;

namespace {
cv::Mat synthetic(int h = 96, int w = 128) {
    cv::Mat m(h, w, CV_8UC3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const uint8_t v = static_cast<uint8_t>((x * 2 + y) % 256);
            m.at<cv::Vec3b>(y, x) = cv::Vec3b(v, 255 - v, (v / 2) + 40);
        }
    return m;
}
} // namespace

TEST_CASE("antidetect physics run populates the report and stays deterministic",
          "[antidetect][report]") {
    cv::Mat a = synthetic(), b = synthetic();
    AntidetectConfig cfg;
    cfg.method = AntidetectMethod::Physics;
    cfg.seed = 42;
    const auto ra = run_antidetect(a, cfg);
    REQUIRE(ra.ok);
    REQUIRE(ra.report.physics_ran);
    REQUIRE(ra.report.physics.jpeg_quality > 0);
    REQUIRE(!ra.report.scores.empty() == false);

    cfg.seed = 42;
    const auto rb = run_antidetect(b, cfg);
    REQUIRE(cv::sum(a != b) == cv::Scalar(0));  // same seed -> identical bytes

    // Quality band for a default-strength pass on smooth content.
    REQUIRE(ra.report.psnr > 20.0);
    REQUIRE(ra.report.psnr < 60.0);
    REQUIRE(ra.report.ssim > 0.7);
    REQUIRE(ra.report.ssim <= 1.0);
    REQUIRE(ra.report.lpips >= 0.0);
    REQUIRE(ra.report.lpips < 0.5);
}

TEST_CASE("explicit adversarial method refuses on a physics-only build",
          "[antidetect][report]") {
#ifndef WMR_ANTIDETECT_ADVERSARIAL
    cv::Mat img = synthetic();
    const cv::Mat ref = img.clone();
    AntidetectConfig cfg;
    cfg.method = AntidetectMethod::Adversarial;
    const auto r = run_antidetect(img, cfg);
    REQUIRE_FALSE(r.ok);
    REQUIRE_FALSE(r.report.physics_ran);
    REQUIRE_FALSE(r.report.note.empty());
    REQUIRE(cv::sum(img != ref) == cv::Scalar(0));  // untouched on refusal
#else
    SUCCEED("adversarial build covers this via the models-gated tests");
#endif
}

TEST_CASE("ssim_luma is 1 on identical inputs and drops with noise",
          "[antidetect][report]") {
    const cv::Mat a = synthetic();
    REQUIRE_THAT(ssim_luma(a, a), Catch::Matchers::WithinAbs(1.0, 1e-6));
    cv::Mat noisy;
    cv::Mat noise(a.size(), CV_32FC3);
    cv::randn(noise, cv::Scalar(0, 0, 0), cv::Scalar(15, 15, 15));
    cv::add(a, noise, noisy, cv::noArray(), CV_8U);  // saturating add, u8 out
    REQUIRE(ssim_luma(a, noisy) < 0.95);
    REQUIRE(ssim_luma(a, noisy) > 0.3);
}

TEST_CASE("format_antidetect_report renders the honest block", "[antidetect][report]") {
    cv::Mat img = synthetic();
    AntidetectConfig cfg;
    cfg.method = AntidetectMethod::Physics;
    cfg.seed = 1;
    const auto r = run_antidetect(img, cfg);
    const std::string plain = format_antidetect_report(r.report, false);
    REQUIRE(plain.find("Anti-detection pass:") != std::string::npos);
    REQUIRE(plain.find("physics:") != std::string::npos);
    REQUIRE(plain.find("PSNR") != std::string::npos);
    const std::string indented = format_antidetect_report(r.report, true);
    REQUIRE(indented.rfind("  Anti-detection pass:", 0) == 0);  // two-space prefix
    // No over-claim language anywhere in the report.
    for (const char* bad : {"undetectable", "guaranteed"}) {
        REQUIRE(plain.find(bad) == std::string::npos);
    }
}
