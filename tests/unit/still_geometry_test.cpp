#include <catch2/catch_test_macros.hpp>

#include "detection/still_geometry.hpp"
#include "video/geometry_detector.hpp"  // decide_auto_geometry
#include "core/watermark_engine.hpp"
#include "core/types.hpp"
#include "core/blend_modes.hpp"
#include <opencv2/imgproc.hpp>

#include <vector>

using namespace wmr;

// Same textured background helper used by the V2/geometry round-trip tests.
static cv::Mat textured(int W, int H, cv::Scalar base) {
    cv::Mat img(H, W, CV_8UC3, base);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            int gx = x * 60 / W;
            int gy = y * 60 / H;
            int n = ((x * 5 ^ y * 3) & 0x1F) - 0x10;  // +/-16
            img.at<cv::Vec3b>(y, x) = cv::Vec3b(
                cv::saturate_cast<uchar>(base[0] + gx + n),
                cv::saturate_cast<uchar>(base[1] + gy + n),
                cv::saturate_cast<uchar>(base[2] + gx));
        }
    return img;
}

namespace {
cv::Mat alpha_to_template(const cv::Mat& alpha) {
    cv::Mat t;
    alpha.convertTo(t, CV_8U, 255.0);
    return t;
}
cv::Mat to_gray(const cv::Mat& bgr) {
    cv::Mat g;
    cv::cvtColor(bgr, g, cv::COLOR_BGR2GRAY);
    return g;
}
}  // namespace

static constexpr int kW = 896, kH = 1200;

TEST_CASE("still_geometry: model under-predicts the 3.6 margin (the bug this fixes)",
          "[still_geometry]") {
    const auto p = v2_small_config_from_dims(kW, kH);  // short=896 -> source 2752
    CHECK(p.logo_size == 36);
    CHECK(p.margin_right == 84);   // round(192 * 1200/2752) = 84
    CHECK(p.margin_bottom == 84);
    // Gemini 3.6's real small mark is 48px at margin ~(96,96) -> the search recovers it.
}

TEST_CASE("still_geometry: multi-template picks the 48px Gemini 3.6 diamond",
          "[still_geometry]") {
    WatermarkEngine engine;
    const cv::Mat a36 = engine.get_v2_diamond_alpha_36();
    const cv::Mat a48 = engine.get_v2_diamond_alpha_small();   // 48px (video small) = Gemini 3.6
    REQUIRE(a48.cols == 48);
    const std::vector<cv::Mat> templates{alpha_to_template(a36), alpha_to_template(a48)};

    // Stamp the 48px diamond at the calibrated 3.6 geometry (margin 96,96).
    const int margin_r = 96, margin_b = 96;
    const cv::Point pos(kW - margin_r - 48, kH - margin_b - 48);  // (754, 1056)
    cv::Mat frame = textured(kW, kH, cv::Scalar(80, 100, 120));
    add_watermark_alpha_blend(frame, a48, pos, 255.0f);

    const cv::Point anchor = v2_small_config_from_dims(kW, kH).get_position(kW, kH);  // (776,1080)
    auto hit = locate_still_watermark_hybrid(to_gray(frame), templates, anchor, kW, kH);
    REQUIRE(hit.has_value());
    CHECK(std::abs(hit->rect.x - pos.x) <= 3);
    CHECK(std::abs(hit->rect.y - pos.y) <= 3);
    CHECK(hit->template_index == 1);   // the 48px template won
    CHECK(hit->rect.width == 48);

    // It snaps to the calibrated 48px preset.
    auto snap = snap_still_to_known(hit->rect, kW, kH);
    REQUIRE(snap.has_value());
    CHECK(snap->name == std::string("gemini36-portrait"));
    CHECK(snap->logo_size == 48);
}

TEST_CASE("still_geometry: multi-template still picks 36px (Gemini 3.5) when that fits",
          "[still_geometry]") {
    WatermarkEngine engine;
    const cv::Mat a36 = engine.get_v2_diamond_alpha_36();
    const cv::Mat a48 = engine.get_v2_diamond_alpha_small();
    const std::vector<cv::Mat> templates{alpha_to_template(a36), alpha_to_template(a48)};

    // Stamp the 36px diamond at the model position (margin 84).
    const cv::Point pos = v2_small_config_from_dims(kW, kH).get_position(kW, kH);  // (776,1080)
    cv::Mat frame = textured(kW, kH, cv::Scalar(90, 80, 110));
    add_watermark_alpha_blend(frame, a36, pos, 255.0f);

    auto hit = locate_still_watermark_hybrid(to_gray(frame), templates, pos, kW, kH);
    REQUIRE(hit.has_value());
    CHECK(hit->template_index == 0);   // 36px won
    CHECK(std::abs(hit->rect.x - pos.x) <= 3);
}

TEST_CASE("still_geometry: widen search catches a far-off-model 48px mark",
          "[still_geometry]") {
    WatermarkEngine engine;
    const cv::Mat a36 = engine.get_v2_diamond_alpha_36();
    const cv::Mat a48 = engine.get_v2_diamond_alpha_small();
    const std::vector<cv::Mat> templates{alpha_to_template(a36), alpha_to_template(a48)};
    const int margin = 200;  // well outside the +/-40 anchored window
    const cv::Point pos(kW - margin - 48, kH - margin - 48);
    cv::Mat frame = textured(kW, kH, cv::Scalar(60, 70, 90));
    add_watermark_alpha_blend(frame, a48, pos, 255.0f);

    const cv::Point anchor = v2_small_config_from_dims(kW, kH).get_position(kW, kH);
    auto hit = locate_still_watermark_hybrid(to_gray(frame), templates, anchor, kW, kH);
    CAPTURE(hit.has_value(), pos.x, pos.y);
    REQUIRE(hit.has_value());
    CHECK(std::abs(hit->rect.x - pos.x) <= 3);
    CHECK(hit->rect.y >= pos.y - 3);
}

TEST_CASE("still_geometry: snap_still_to_known respects tier + size + center L1",
          "[still_geometry]") {
    // Near the gemini36-portrait geometry (48px, short side 896 in [800,1000]).
    const cv::Rect near(kW - 96 - 48 + 10, kH - 96 - 48 + 10, 48, 48);
    auto s1 = snap_still_to_known(near, kW, kH);
    CHECK(s1.has_value());
    CHECK(s1->name == std::string("gemini36-portrait"));

    const cv::Rect far(kW - 300 - 48, kH - 300 - 48, 48, 48);
    CHECK_FALSE(snap_still_to_known(far, kW, kH).has_value());

    // Wrong resolution tier -> no snap.
    CHECK_FALSE(snap_still_to_known(cv::Rect(2048 - 96 - 48, 1408 - 96 - 48, 48, 48),
                                    2048, 1408).has_value());
    // Wrong size (36 cannot snap to the 48 slot).
    CHECK_FALSE(snap_still_to_known(cv::Rect(kW - 96 - 36, kH - 96 - 36, 36, 36),
                                    kW, kH).has_value());
}

TEST_CASE("still_geometry: rect_to_still_position keeps the rect's size", "[still_geometry]") {
    const cv::Rect rect(kW - 96 - 48, kH - 96 - 48, 48, 48);
    const auto wp = rect_to_still_position(rect, kW, kH, 48);
    CHECK(wp.margin_right == 96);
    CHECK(wp.margin_bottom == 96);
    CHECK(wp.logo_size == 48);
    CHECK(find_preset("gemini36-portrait").has_value());
    CHECK_FALSE(find_preset("nope").has_value());
}

TEST_CASE("still_geometry: regression gate boundaries (reused from video)",
          "[still_geometry]") {
    CHECK(decide_auto_geometry(true, 0.45f, kStillHighConfidence) == AutoGeometryVerdict::UseSnapped);
    CHECK(decide_auto_geometry(false, 0.59f, kStillHighConfidence) == AutoGeometryVerdict::FallBack);
    CHECK(decide_auto_geometry(false, 0.60f, kStillHighConfidence) == AutoGeometryVerdict::UseRaw);
}

TEST_CASE("still_geometry: resolve_still_geometry precedence + matched alpha size",
          "[still_geometry]") {
    WatermarkEngine engine;
    const cv::Mat a36 = engine.get_v2_diamond_alpha_36();
    const cv::Mat a48 = engine.get_v2_diamond_alpha_small();
    const std::vector<cv::Mat> templates{alpha_to_template(a36), alpha_to_template(a48)};
    const WatermarkPosition model = v2_small_config_from_dims(kW, kH);

    // --rect wins outright; logo_size follows the rect width (48 here).
    StillGeometryOverride o;
    o.rect = cv::Rect(754, 1056, 48, 48);
    cv::Mat clean = textured(kW, kH, cv::Scalar(80, 100, 120));
    auto r = resolve_still_geometry(to_gray(clean), templates, model, kW, kH, o);
    CHECK(r.source == "rect");
    CHECK(r.pos.logo_size == 48);

    // --geo-preset.
    StillGeometryOverride o2;
    o2.preset = std::string("gemini36-portrait");
    auto r2 = resolve_still_geometry(to_gray(clean), templates, model, kW, kH, o2);
    CHECK(r2.source == "preset");
    CHECK(r2.pos.margin_right == 96);
    CHECK(r2.pos.logo_size == 48);

    // --no-auto-geometry -> model fallback.
    StillGeometryOverride o3;
    o3.no_auto_geometry = true;
    auto r3 = resolve_still_geometry(to_gray(clean), templates, model, kW, kH, o3);
    CHECK(r3.source == "model");

    // Auto-detect on a 48px-stamped frame picks the 48px template.
    cv::Mat stamped = textured(kW, kH, cv::Scalar(80, 100, 120));
    add_watermark_alpha_blend(stamped, a48, cv::Point(754, 1056), 255.0f);
    StillGeometryOverride o4;
    auto r4 = resolve_still_geometry(to_gray(stamped), templates, model, kW, kH, o4);
    CHECK(r4.source != "model");
    CHECK(r4.template_index == 1);   // 48px
    CHECK(r4.pos.logo_size == 48);
}
