#include <catch2/catch_test_macros.hpp>

#include "core/inpaint.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>

using namespace wmr;

TEST_CASE("Gaussian inpaint produces valid output", "[inpaint]") {
    cv::Mat image(256, 256, CV_8UC3, cv::Scalar(100, 120, 140));

    // Draw a small white rectangle as "residual"
    cv::rectangle(image, {100, 100, 20, 20}, cv::Scalar(255, 255, 255), cv::FILLED);

    cv::Rect region(90, 90, 40, 40);
    cv::Mat alpha(40, 40, CV_32FC1, cv::Scalar(0.5f));

    cv::Mat result = image.clone();
    InpaintConfig config;
    config.method = InpaintMethod::Gaussian;
    config.strength = 0.85f;

    REQUIRE_NOTHROW(inpaint_residual(result, region, alpha, config));
    REQUIRE(result.rows == 256);
    REQUIRE(result.cols == 256);
    REQUIRE(result.type() == CV_8UC3);
}

TEST_CASE("Telea inpaint produces valid output", "[inpaint]") {
    cv::Mat image(256, 256, CV_8UC3, cv::Scalar(80, 100, 120));
    cv::rectangle(image, {110, 110, 10, 10}, cv::Scalar(255, 255, 255), cv::FILLED);

    cv::Rect region(100, 100, 30, 30);
    cv::Mat alpha(30, 30, CV_32FC1, cv::Scalar(0.3f));

    cv::Mat result = image.clone();
    InpaintConfig config;
    config.method = InpaintMethod::Telea;
    config.strength = 0.85f;

    REQUIRE_NOTHROW(inpaint_residual(result, region, alpha, config));
    REQUIRE(result.rows == 256);
    REQUIRE(result.cols == 256);
}

TEST_CASE("Navier-Stokes inpaint produces valid output", "[inpaint]") {
    cv::Mat image(256, 256, CV_8UC3, cv::Scalar(80, 100, 120));
    cv::rectangle(image, {110, 110, 10, 10}, cv::Scalar(255, 255, 255), cv::FILLED);

    cv::Rect region(100, 100, 30, 30);
    cv::Mat alpha(30, 30, CV_32FC1, cv::Scalar(0.3f));

    cv::Mat result = image.clone();
    InpaintConfig config;
    config.method = InpaintMethod::NavierStokes;
    config.strength = 0.85f;

    REQUIRE_NOTHROW(inpaint_residual(result, region, alpha, config));
    REQUIRE(result.rows == 256);
    REQUIRE(result.cols == 256);
}

TEST_CASE("Diamond edge cleanup repairs the edge, preserves the interior", "[inpaint][video]") {
    // 48x48 diamond alpha fading to its edge (inscribed in the box).
    cv::Mat alpha(48, 48, CV_32FC1, cv::Scalar(0.0f));
    for (int y = 0; y < 48; ++y)
        for (int x = 0; x < 48; ++x) {
            float d = (std::fabs(x - 23.5f) + std::fabs(y - 23.5f)) / 23.0f;
            if (d < 1.0f) alpha.at<float>(y, x) = 0.4f * (1.0f - d);
        }

    const cv::Rect region(40, 40, 48, 48);
    // "recovered" image: uniform grey with a bright halo ring at the diamond edge
    // (the border/halo residual the cleanup targets).
    cv::Mat recovered(128, 128, CV_8UC3, cv::Scalar(90, 90, 90));
    for (int y = 0; y < 48; ++y)
        for (int x = 0; x < 48; ++x) {
            float d = (std::fabs(x - 23.5f) + std::fabs(y - 23.5f)) / 23.0f;
            if (d > 0.78f && d < 0.98f)
                recovered.at<cv::Vec3b>(region.y + y, region.x + x) = cv::Vec3b(160, 160, 160);
        }

    cv::Mat cleaned = recovered.clone();
    REQUIRE_NOTHROW(inpaint_diamond_edges(cleaned, region, alpha));
    REQUIRE(cleaned.size() == recovered.size());
    REQUIRE(cleaned.type() == CV_8UC3);

    // Interior (diamond centre) and exterior are byte-for-byte unchanged.
    REQUIRE(cleaned.at<cv::Vec3b>(64, 64) == cv::Vec3b(90, 90, 90));
    REQUIRE(cleaned.at<cv::Vec3b>(10, 10) == recovered.at<cv::Vec3b>(10, 10));

    // The halo ring is reduced: mean deviation from the background drops after cleanup.
    double before = 0.0, after = 0.0;
    int n = 0;
    for (int y = 0; y < 48; ++y)
        for (int x = 0; x < 48; ++x) {
            float d = (std::fabs(x - 23.5f) + std::fabs(y - 23.5f)) / 23.0f;
            if (d > 0.78f && d < 0.98f) {
                before += std::fabs(recovered.at<cv::Vec3b>(region.y + y, region.x + x)[0] - 90);
                after += std::fabs(cleaned.at<cv::Vec3b>(region.y + y, region.x + x)[0] - 90);
                ++n;
            }
        }
    REQUIRE(n > 50);
    REQUIRE(after < before);  // halo is fainter after the edge cleanup
}
