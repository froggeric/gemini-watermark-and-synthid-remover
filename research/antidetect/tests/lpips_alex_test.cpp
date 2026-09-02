#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "core/lpips_alex.hpp"
#include "lpips_alex_golden.h"

#include <opencv2/core.hpp>

#include <cmath>
#include <random>

using namespace wmr::antidetect;
using wmr::embedded::kLpipsGoldenPairCount;
using wmr::embedded::kLpipsGoldenPairs;

namespace {
cv::Mat golden_image(const uint8_t* data, int w, int h) {
    cv::Mat m(h, w, CV_8UC3);
    std::copy(data, data + static_cast<size_t>(w) * h * 3, m.data);
    return m;
}
} // namespace

TEST_CASE("layer-2 LPIPS matches the torch reference on golden pairs",
          "[antidetect][lpips]") {
    const LpipsAlex lp;
    for (size_t i = 0; i < kLpipsGoldenPairCount; ++i) {
        const auto& p = kLpipsGoldenPairs[i];
        const cv::Mat a = golden_image(p.a_bgr, p.width, p.height);
        const cv::Mat b = golden_image(p.b_bgr, p.width, p.height);
        CAPTURE(p.name, p.expected, lp.distance(a, b));
        REQUIRE_THAT(lp.distance(a, b),
                     Catch::Matchers::WithinAbs(p.expected, 2e-3));
    }
}

TEST_CASE("layer-2 LPIPS is zero on identical inputs and monotone in noise",
          "[antidetect][lpips]") {
    const LpipsAlex lp;
    cv::Mat img(96, 128, CV_8UC3, cv::Vec3b(80, 120, 160));
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x)
            img.at<cv::Vec3b>(y, x) =
                cv::Vec3b(static_cast<uint8_t>(x % 256), static_cast<uint8_t>(y % 256),
                          static_cast<uint8_t>((x + y) % 256));
    REQUIRE(lp.distance(img, img) == 0.0f);

    std::mt19937_64 rng(9);
    std::normal_distribution<float> n01(0.0f, 1.0f);
    float prev = 0.0f;
    cv::Mat noised = img.clone();
    for (const float sigma : {3.0f, 8.0f, 20.0f, 50.0f}) {
        for (int y = 0; y < noised.rows; ++y)
            for (int x = 0; x < noised.cols; ++x)
                for (int c = 0; c < 3; ++c)
                    noised.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uint8_t>(
                        noised.at<cv::Vec3b>(y, x)[c] + sigma * n01(rng) / 3.0f);
        const float d = lp.distance(img, noised);
        CAPTURE(sigma, d);
        REQUIRE(d > prev);  // more noise -> larger distance
        prev = d;
    }
    REQUIRE(prev < 1.5f);  // sanity band for layer-2 LPIPS
}

TEST_CASE("layer-2 LPIPS capped evaluation passes through small images",
          "[antidetect][lpips]") {
    const LpipsAlex lp;
    const cv::Mat a(48, 64, CV_8UC3, cv::Vec3b(30, 200, 90));
    const cv::Mat b(48, 64, CV_8UC3, cv::Vec3b(200, 30, 90));
    // max side 64 == the image size -> no resize, same result as distance().
    REQUIRE(lp.distance_capped(a, b, 64) == lp.distance(a, b));
    // A cap below the size resizes both identically; distance stays in [0, 2].
    const float capped = lp.distance_capped(a, b, 32);
    REQUIRE(capped >= 0.0f);
    REQUIRE(capped < 2.0f);
}
