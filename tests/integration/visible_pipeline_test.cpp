#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "core/watermark_engine.hpp"
#include "core/types.hpp"
#include "core/blend_modes.hpp"
#include "detection/still_geometry.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>

using namespace wmr;
using Catch::Matchers::WithinAbs;

// NOTE: the real baseline file is 2400x1792-test1-gemini.png (Gemini 3.1 Pro, V1).
// An earlier revision pointed at test-images/2400x1792-gemini.png, which does not
// exist, so these two [integration] cases always SKIP. Fixed to the real filename.
static const char* kTestImage = "test-images/2400x1792-test1-gemini.png";
static const char* kAltTestImage = "../test-images/2400x1792-test1-gemini.png";

static cv::Mat load_test_image() {
    if (std::filesystem::exists(kTestImage)) {
        return cv::imread(kTestImage, cv::IMREAD_COLOR);
    }
    if (std::filesystem::exists(kAltTestImage)) {
        return cv::imread(kAltTestImage, cv::IMREAD_COLOR);
    }
    return {};
}

// Load a fixture by exact filename, probing CWD-is-project-root then tests/. Returns
// an empty Mat when absent so the caller can SKIP (the suite must not fail without it).
static cv::Mat load_fixture(const char* name) {
    std::string p1 = std::string("test-images/") + name;
    std::string p2 = std::string("../test-images/") + name;
    if (std::filesystem::exists(p1)) return cv::imread(p1, cv::IMREAD_COLOR);
    if (std::filesystem::exists(p2)) return cv::imread(p2, cv::IMREAD_COLOR);
    return {};
}

TEST_CASE("Visible watermark detection on known watermarked image", "[integration]") {
    cv::Mat image = load_test_image();
    if (image.empty()) {
        SKIP("Test image not available");
    }

    WatermarkEngine engine;
    auto result = engine.detect_watermark(image);

    REQUIRE(result.detected);
    REQUIRE(result.confidence > 0.5f);
    REQUIRE(result.region.width > 0);
    REQUIRE(result.region.height > 0);
}

TEST_CASE("Full visible pipeline: detect → remove → inpaint", "[integration]") {
    cv::Mat image = load_test_image();
    if (image.empty()) {
        SKIP("Test image not available");
    }

    cv::Mat original = image.clone();
    WatermarkEngine engine;

    auto detection = engine.detect_watermark(image);
    REQUIRE(detection.detected);

    engine.remove_watermark_detected(image, detection);

    // The bottom-right region should have changed
    cv::Rect roi = detection.region;
    cv::Mat orig_roi = original(roi);
    cv::Mat new_roi = image(roi);

    cv::Mat diff;
    cv::absdiff(orig_roi, new_roi, diff);
    double total_diff = cv::sum(diff.reshape(1))[0] + cv::sum(diff.reshape(1))[1] + cv::sum(diff.reshape(1))[2];
    REQUIRE(total_diff > 0);

    // Output should still be valid
    REQUIRE(image.rows == original.rows);
    REQUIRE(image.cols == original.cols);
    REQUIRE(image.type() == CV_8UC3);
}

TEST_CASE("Detection on clean image returns not detected", "[integration]") {
    // Create a clean gradient image (no watermark)
    cv::Mat clean(1024, 1024, CV_8UC3);
    for (int y = 0; y < 1024; ++y) {
        for (int x = 0; x < 1024; ++x) {
            clean.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uchar>(x * 255 / 1024),
                static_cast<uchar>(y * 255 / 1024),
                128
            );
        }
    }

    WatermarkEngine engine;
    auto result = engine.detect_watermark(clean);

    REQUIRE_FALSE(result.detected);
}

// ---------------------------------------------------------------------------
// V2 (Gemini 3.5) watermark geometry — pure geometry, no fixtures.
// ---------------------------------------------------------------------------
TEST_CASE("V2 watermark geometry is variant-aware", "[v2]") {
    auto eq = [](const WatermarkPosition& p, int mr, int mb, int ls) {
        return p.margin_right == mr && p.margin_bottom == mb && p.logo_size == ls;
    };

    // V1 (legacy, pre-3.5) unchanged
    REQUIRE(eq(get_watermark_config(2400, 1792, WatermarkVariant::V1), 64, 64, 96));
    REQUIRE(eq(get_watermark_config(1024, 1024, WatermarkVariant::V1), 32, 32, 48));

    // V2 large (both dims > 1024): 192px margin, 96px logo
    REQUIRE(eq(get_watermark_config(2400, 1792, WatermarkVariant::V2), 192, 192, 96));
    REQUIRE(eq(get_watermark_config(2048, 2048, WatermarkVariant::V2), 192, 192, 96));

    // V2 small (a side <= 1024): 36px logo, aspect-aware margin.
    // 1024x1024: short=1024 >= 566 -> source 2752; margin = round(192*1024/2752) = 71
    REQUIRE(eq(get_watermark_config(1024, 1024, WatermarkVariant::V2), 71, 71, 36));
    // 1280x720: short=720 >= 566 -> source 2752; margin = round(192*1280/2752) = 89
    REQUIRE(eq(get_watermark_config(1280, 720, WatermarkVariant::V2), 89, 89, 36));

    // 2-arg overload defaults to V1 (backward compatibility)
    REQUIRE(eq(get_watermark_config(1024, 1024), 32, 32, 48));
}

// ---------------------------------------------------------------------------
// V2 (Gemini 3.5) engine path — detect/remove with the V2 alpha maps.
// Synthetic round-trips (forward-blend the V2 alpha, then detect + remove).
// ---------------------------------------------------------------------------
static cv::Mat textured(int W, int H, cv::Scalar base) {
    cv::Mat img(H, W, CV_8UC3, base);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            int gx = x * 60 / W;
            int gy = y * 60 / H;
            // Moderate, varied background: a smooth gradient plus low-amplitude
            // medium-frequency variation, so the gradient/variance stages have
            // signal without drowning the spatial NCC (a pure gradient zeroes the
            // gradient stage; full-range noise drowns the spatial stage).
            int n = ((x * 5 ^ y * 3) & 0x1F) - 0x10;  // ±16
            img.at<cv::Vec3b>(y, x) = cv::Vec3b(
                cv::saturate_cast<uchar>(base[0] + gx + n),
                cv::saturate_cast<uchar>(base[1] + gy + n),
                cv::saturate_cast<uchar>(base[2] + gx));
        }
    return img;
}

static double mean_abs_diff(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat d;
    cv::absdiff(a, b, d);
    cv::Scalar s = cv::mean(d);
    return (s[0] + s[1] + s[2]) / 3.0;
}

TEST_CASE("V2 large round-trip recovers original", "[v2]") {
    cv::Mat original = textured(2048, 2048, cv::Scalar(80, 100, 120));
    cv::Mat watermarked = original.clone();

    WatermarkEngine engine;
    const auto pos_cfg = get_watermark_config(2048, 2048, WatermarkVariant::V2);  // {192,192,96}
    const cv::Point pos = pos_cfg.get_position(2048, 2048);
    const cv::Mat& alpha = engine.get_v2_diamond_alpha_large();
    REQUIRE(alpha.cols == 96);  // V2 large map decoded
    add_watermark_alpha_blend(watermarked, alpha, pos, 255.0f);

    const cv::Rect roi(pos.x, pos.y, alpha.cols, alpha.rows);
    REQUIRE(mean_abs_diff(original(roi), watermarked(roi)) > 5.0);  // watermark applied

    auto det = engine.detect_watermark(watermarked, std::nullopt, std::nullopt,
                                       nullptr, WatermarkVariant::V2, /*enable_snap=*/false);
    CAPTURE(det.spatial_score, det.gradient_score, det.variance_score, det.confidence);
    REQUIRE(det.detected);
    REQUIRE(det.confidence > 0.4f);

    engine.remove_watermark_alpha_only(watermarked, det, &alpha);
    REQUIRE(mean_abs_diff(original(roi), watermarked(roi)) < 1.5);  // near-exact recovery
}

TEST_CASE("V2 small detection uses 36x36 alpha + snap", "[v2]") {
    cv::Mat original = textured(1024, 1024, cv::Scalar(60, 90, 130));
    cv::Mat watermarked = original.clone();

    WatermarkEngine engine;
    const auto pos_cfg = get_watermark_config(1024, 1024, WatermarkVariant::V2);  // {71,71,36}
    const cv::Point pos = pos_cfg.get_position(1024, 1024);
    const cv::Mat& alpha = engine.get_v2_diamond_alpha_36();
    REQUIRE(alpha.cols == 36);
    REQUIRE(alpha.rows == 36);
    add_watermark_alpha_blend(watermarked, alpha, pos, 255.0f);

    auto det = engine.detect_watermark(watermarked, std::nullopt, std::nullopt,
                                       nullptr, WatermarkVariant::V2, /*enable_snap=*/true);
    CAPTURE(det.spatial_score, det.gradient_score, det.variance_score, det.confidence);
    REQUIRE(det.detected);
    REQUIRE(det.region.width == 36);
    REQUIRE(det.region.height == 36);
}

TEST_CASE("V2 small snap recovers the exact mark position on busy content", "[v2]") {
    // Regression for the 1px position drift that left a faint light/shadow emboss after
    // diamond removal. On busy content the raw NCC peak straddles two integer cells and
    // minMaxLoc picks the wrong one; the snap's content-suppressed re-localization must
    // recover the true cell. We forward-blend the 48px mark at a known position, start
    // the snap 1px off (as the geometry search can on busy content), and require the
    // detected region to land exactly on the true top-left.
    cv::Mat original = textured(928, 1152, cv::Scalar(50, 70, 110));
    cv::Mat watermarked = original.clone();

    WatermarkEngine engine;
    const cv::Point true_pos(784, 1008);   // 96px bottom-right margin on 928x1152
    const cv::Mat& alpha = engine.get_v2_diamond_alpha_48_still();
    REQUIRE(alpha.cols == 48);
    add_watermark_alpha_blend(watermarked, alpha, true_pos, 255.0f);

    // Start the search 1px low (margin_bottom 95 instead of 96 -> pos (784,1009)).
    const WatermarkPosition start{96, 95, 48};
    auto det = engine.detect_watermark(watermarked, WatermarkSize::Small, start, &alpha,
                                       WatermarkVariant::V2, /*enable_snap=*/true);
    CAPTURE(det.spatial_score, det.confidence, det.region.x, det.region.y);
    REQUIRE(det.detected);
    CHECK(det.region.x == true_pos.x);
    CHECK(det.region.y == true_pos.y);   // must snap up to 1008, not stay at 1009
}

TEST_CASE("V1 legacy path still works via default detect", "[v2]") {
    cv::Mat original = textured(1024, 1024, cv::Scalar(70, 80, 90));
    cv::Mat watermarked = original.clone();

    WatermarkEngine engine;
    const auto pos_cfg = get_watermark_config(1024, 1024, WatermarkVariant::V1);  // {32,32,48}
    const cv::Point pos = pos_cfg.get_position(1024, 1024);
    const cv::Mat& alpha = engine.get_alpha_map(WatermarkSize::Small);  // V1 48x48
    add_watermark_alpha_blend(watermarked, alpha, pos, 255.0f);

    // The 4-arg (default) detect resolves to V1 — legacy callers unaffected.
    auto det = engine.detect_watermark(watermarked);
    REQUIRE(det.detected);
    REQUIRE(det.region.width == 48);
}

// ---------------------------------------------------------------------------
// Still-image auto-geometry (Gemini 3.6 Flash). Gemini 3.6's small diamond is 48px
// (NOT 36px like 3.5 stills) at margin ~(94,96); the model predicts 36px@84. The
// multi-template search must recover the 48px mark on a clear image (test4), leave
// V1 untouched (test1), and let a preset force the geometry on a faint image (test3).
// ---------------------------------------------------------------------------
TEST_CASE("Gemini 3.6 (896x1200) auto-geometry finds the 48px mark", "[integration][v2]") {
    cv::Mat image = load_fixture("896x1200-test4-gemini36.png");
    if (image.empty()) SKIP("test4 fixture not available");

    WatermarkEngine engine;
    StillGeometryOverride ov;  // no override -> hybrid auto search
    auto res = engine.resolve_still_geometry(image, WatermarkVariant::V2,
                                             WatermarkSize::Small, ov);
    REQUIRE(res.pos.has_value());
    CHECK(res.pos->logo_size == 48);                  // 3.6 is 48px, not 36
    CHECK(res.pos->margin_right >= 86);               // ~94, NOT the model's 84
    CHECK(res.pos->margin_right <= 102);
    REQUIRE(res.alpha != nullptr);
    CHECK(res.alpha->cols == 48);                     // matched-size removal alpha

    auto det = engine.detect_watermark(image, WatermarkSize::Small, res.pos, res.alpha,
                                       WatermarkVariant::V2, /*enable_snap=*/true);
    CAPTURE(det.spatial_score, det.gradient_score, det.variance_score, det.confidence);
    REQUIRE(det.detected);
    CHECK(det.region.width == 48);
    // Near the measured TL (754,1056).
    CHECK(std::abs(det.region.x - 754) <= 6);
    CHECK(std::abs(det.region.y - 1056) <= 6);
}

TEST_CASE("Gemini 3.1 Pro (2400x1792) still detects as V1 (non-regression)", "[integration]") {
    cv::Mat image = load_fixture("2400x1792-test1-gemini.png");
    if (image.empty()) SKIP("test1 fixture not available");

    WatermarkEngine engine;
    // The auto search is scoped to V2 small; V1 large is untouched (returns nullopt).
    StillGeometryOverride ov;
    CHECK_FALSE(engine.resolve_still_geometry(image, WatermarkVariant::V1,
                                              WatermarkSize::Large, ov).pos.has_value());
    auto det = engine.detect_watermark(image, std::nullopt, std::nullopt, nullptr,
                                       WatermarkVariant::V1, /*enable_snap=*/false);
    REQUIRE(det.detected);
}

TEST_CASE("Gemini 3.6 faint (896x1200) geometry preset forces the 48px position",
          "[integration][v2]") {
    cv::Mat image = load_fixture("896x1200-test3-gemini36.png");
    if (image.empty()) SKIP("test3 fixture not available");

    WatermarkEngine engine;
    StillGeometryOverride ov;
    ov.preset = std::string("gemini36-portrait");   // calibrated 896x1200 -> 48px @ (96,96)
    auto res = engine.resolve_still_geometry(image, WatermarkVariant::V2,
                                             WatermarkSize::Small, ov);
    REQUIRE(res.pos.has_value());
    CHECK(res.pos->margin_right == 96);
    CHECK(res.pos->margin_bottom == 96);
    CHECK(res.pos->logo_size == 48);
    REQUIRE(res.alpha != nullptr);
    CHECK(res.alpha->cols == 48);   // the override uses the correct 48px alpha, not the default 36

    // A --rect override forces the exact measured box; logo_size follows the rect width.
    StillGeometryOverride ov2;
    ov2.rect = cv::Rect(752, 1056, 48, 48);
    auto res2 = engine.resolve_still_geometry(image, WatermarkVariant::V1,
                                              WatermarkSize::Small, ov2);
    REQUIRE(res2.pos.has_value());
    CHECK(res2.pos->margin_right == 896 - (752 + 48));
    REQUIRE(res2.alpha != nullptr);
    CHECK(res2.alpha->cols == 48);
}
