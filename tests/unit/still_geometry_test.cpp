#include <catch2/catch_test_macros.hpp>

#include "detection/still_geometry.hpp"
#include "video/geometry_detector.hpp"  // decide_auto_geometry
#include "core/watermark_engine.hpp"
#include "core/types.hpp"
#include "core/blend_modes.hpp"
#include <opencv2/imgproc.hpp>

using namespace wmr;

// Same textured background helper used by the V2/geometry round-trip tests: a smooth
// gradient plus low-amplitude variation so the template match has signal.
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

// The Gemini 3.6 portrait fixture dimensions and its true margin (~100, vs the
// model's 84). Used across the search/snap/precedence cases below.
static constexpr int kW = 896, kH = 1200;

TEST_CASE("still_geometry: model under-predicts the 3.6 margin (the bug this fixes)",
          "[still_geometry]") {
    const auto p = v2_small_config_from_dims(kW, kH);  // short=896 -> source 2752
    CHECK(p.logo_size == 36);
    CHECK(p.margin_right == 84);   // round(192 * 1200/2752) = 84
    CHECK(p.margin_bottom == 84);
    // The real Gemini 3.6 margin measured on test4 is ~100 -> a 16 px model error
    // the anchored search is built to absorb.
}

TEST_CASE("still_geometry: anchored search recovers an off-model mark (margin 100)",
          "[still_geometry]") {
    WatermarkEngine engine;
    const cv::Mat alpha = engine.get_v2_diamond_alpha_36();
    REQUIRE(alpha.cols == 36);

    const int margin = 100;  // the real 3.6 geometry
    const cv::Point pos(kW - margin - 36, kH - margin - 36);  // (760, 1064)
    cv::Mat frame = textured(kW, kH, cv::Scalar(80, 100, 120));
    add_watermark_alpha_blend(frame, alpha, pos, 255.0f);

    const cv::Point anchor = v2_small_config_from_dims(kW, kH).get_position(kW, kH);  // (776,1080)
    auto hit = locate_still_watermark_hybrid(to_gray(frame), alpha_to_template(alpha),
                                             anchor, kW, kH);
    REQUIRE(hit.has_value());
    CHECK(std::abs(hit->rect.x - pos.x) <= 3);
    CHECK(std::abs(hit->rect.y - pos.y) <= 3);
    CAPTURE(hit->score);
    CHECK(hit->score >= kStillMinConfidence);

    // And it snaps to the calibrated gemini36-portrait preset (short side 896 in range).
    auto snap = snap_still_to_known(hit->rect, kW, kH);
    REQUIRE(snap.has_value());
    CHECK(snap->name == std::string("gemini36-portrait"));
}

TEST_CASE("still_geometry: anchored search recovers a model-position mark (margin 84)",
          "[still_geometry]") {
    WatermarkEngine engine;
    const cv::Mat alpha = engine.get_v2_diamond_alpha_36();
    const cv::Point pos = v2_small_config_from_dims(kW, kH).get_position(kW, kH);  // (776,1080)
    cv::Mat frame = textured(kW, kH, cv::Scalar(90, 80, 110));
    add_watermark_alpha_blend(frame, alpha, pos, 255.0f);

    auto hit = locate_still_watermark_hybrid(to_gray(frame), alpha_to_template(alpha),
                                             pos, kW, kH);
    REQUIRE(hit.has_value());
    CHECK(std::abs(hit->rect.x - pos.x) <= 3);
    CHECK(std::abs(hit->rect.y - pos.y) <= 3);
}

TEST_CASE("still_geometry: widen search catches a far-off-model mark (margin 200)",
          "[still_geometry]") {
    WatermarkEngine engine;
    const cv::Mat alpha = engine.get_v2_diamond_alpha_36();
    const int margin = 200;  // well outside the +/-40 anchored window
    const cv::Point pos(kW - margin - 36, kH - margin - 36);  // (660, 964)
    cv::Mat frame = textured(kW, kH, cv::Scalar(60, 70, 90));
    add_watermark_alpha_blend(frame, alpha, pos, 255.0f);

    const cv::Point anchor = v2_small_config_from_dims(kW, kH).get_position(kW, kH);
    // Default thresholds: a raw off-table hit must clear 0.60. A clear stamp does.
    auto hit = locate_still_watermark_hybrid(to_gray(frame), alpha_to_template(alpha),
                                             anchor, kW, kH);
    CAPTURE(hit.has_value(), pos.x, pos.y);
    REQUIRE(hit.has_value());
    CHECK(std::abs(hit->rect.x - pos.x) <= 3);
    CHECK(std::abs(hit->rect.y - pos.y) <= 3);
}

TEST_CASE("still_geometry: snap_still_to_known respects resolution tier + center L1",
          "[still_geometry]") {
    // Near the gemini36-portrait geometry (short side 896 in [800,1000]).
    const cv::Rect near(kW - 100 - 36 + 10, kH - 100 - 36 + 10, 36, 36);
    auto s1 = snap_still_to_known(near, kW, kH);
    CHECK(s1.has_value());
    CHECK(s1->name == std::string("gemini36-portrait"));

    // Far from any preset -> no snap.
    const cv::Rect far(kW - 300 - 36, kH - 300 - 36, 36, 36);
    CHECK_FALSE(snap_still_to_known(far, kW, kH).has_value());

    // Wrong resolution tier (short side 1400, outside [800,1000]) -> no snap even if
    // the margins would otherwise match.
    CHECK_FALSE(snap_still_to_known(cv::Rect(2048 - 100 - 36, 1408 - 100 - 36, 36, 36),
                                    2048, 1408).has_value());

    // Wrong size (a 96-wide hit cannot snap to a 36 slot).
    CHECK_FALSE(snap_still_to_known(cv::Rect(kW - 100 - 36, kH - 100 - 36, 96, 96),
                                    kW, kH).has_value());
}

TEST_CASE("still_geometry: rect_to_still_position + find_preset", "[still_geometry]") {
    const cv::Rect rect(kW - 100 - 36, kH - 100 - 36, 36, 36);  // margin 100,100
    const auto wp = rect_to_still_position(rect, kW, kH);
    CHECK(wp.margin_right == 100);
    CHECK(wp.margin_bottom == 100);
    CHECK(wp.logo_size == 36);

    CHECK(find_preset("gemini36-portrait").has_value());
    CHECK_FALSE(find_preset("nope").has_value());
}

TEST_CASE("still_geometry: regression gate boundaries (reused from video)",
          "[still_geometry]") {
    // A snapped hit is trusted at the min confidence; a raw hit must clear 0.60.
    CHECK(decide_auto_geometry(true, 0.45f, kStillHighConfidence) == AutoGeometryVerdict::UseSnapped);
    CHECK(decide_auto_geometry(false, 0.59f, kStillHighConfidence) == AutoGeometryVerdict::FallBack);
    CHECK(decide_auto_geometry(false, 0.60f, kStillHighConfidence) == AutoGeometryVerdict::UseRaw);
}

TEST_CASE("still_geometry: resolve_still_geometry precedence (rect > preset > auto > model)",
          "[still_geometry]") {
    WatermarkEngine engine;
    const cv::Mat alpha = engine.get_v2_diamond_alpha_36();
    const cv::Mat tmpl = alpha_to_template(alpha);
    const WatermarkPosition model = v2_small_config_from_dims(kW, kH);

    // --rect wins outright, even on a clean frame.
    StillGeometryOverride o;
    o.rect = cv::Rect(760, 1063, 36, 36);
    cv::Mat clean = textured(kW, kH, cv::Scalar(80, 100, 120));
    auto r = resolve_still_geometry(to_gray(clean), tmpl, model, kW, kH, o);
    CHECK(r.source == "rect");
    CHECK(r.pos.margin_right == kW - (760 + 36));
    CHECK(r.pos.margin_bottom == kH - (1063 + 36));

    // --geo-preset.
    StillGeometryOverride o2;
    o2.preset = std::string("gemini36-portrait");
    auto r2 = resolve_still_geometry(to_gray(clean), tmpl, model, kW, kH, o2);
    CHECK(r2.source == "preset");
    CHECK(r2.pos.margin_right == 100);

    // --no-auto-geometry -> model fallback.
    StillGeometryOverride o3;
    o3.no_auto_geometry = true;
    auto r3 = resolve_still_geometry(to_gray(clean), tmpl, model, kW, kH, o3);
    CHECK(r3.source == "model");
    CHECK(r3.pos.margin_right == 84);

    // Auto-detect on a stamped frame at the 3.6 geometry.
    cv::Mat stamped = textured(kW, kH, cv::Scalar(80, 100, 120));
    add_watermark_alpha_blend(stamped, alpha, cv::Point(760, 1064), 255.0f);
    StillGeometryOverride o4;
    auto r4 = resolve_still_geometry(to_gray(stamped), tmpl, model, kW, kH, o4);
    CHECK(r4.source != "model");  // anchored + preset snap accepts it
    CHECK(r4.pos.margin_right >= 90);  // near the true ~100, not the model's 84

    // Unknown preset name falls through to auto/model.
    StillGeometryOverride o5;
    o5.preset = std::string("does-not-exist");
    auto r5 = resolve_still_geometry(to_gray(clean), tmpl, model, kW, kH, o5);
    CHECK(r5.source == "model");  // clean frame -> no confident auto hit -> model
}
