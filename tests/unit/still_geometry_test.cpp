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
    const cv::Mat a48 = engine.get_v2_diamond_alpha_48_still();   // 48px still capture = Gemini 3.6
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
    const cv::Mat a48 = engine.get_v2_diamond_alpha_48_still();
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
    const cv::Mat a48 = engine.get_v2_diamond_alpha_48_still();
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

// ---------------------------------------------------------------------------
// Saturated-content collision (Gemini 3.8 regression). The mark is a WHITE
// overlay, so white content through the footprint carries zero mark signal
// (0.3*255 + 0.7*255 == 255) but huge NCC noise. On a real Gemini 3.8 image
// (mark across white poster text) the raw search scored the true spot 0.39 and
// let a text artifact win; the saturated-content suppression in
// locate_still_watermark_hybrid recovers it (0.78 there).
// ---------------------------------------------------------------------------
TEST_CASE("still_geometry: white text through the mark does not defeat the search",
          "[still_geometry]") {
    WatermarkEngine engine;
    const cv::Mat a36 = engine.get_v2_diamond_alpha_36();
    const cv::Mat a48 = engine.get_v2_diamond_alpha_48_still();
    const std::vector<cv::Mat> templates{alpha_to_template(a36), alpha_to_template(a48)};

    // Two heavy white text lines straight through the calibrated mark footprint.
    // Parameters verified in python: without suppression the 36px template wins on
    // a text artifact (0.49 at the WRONG position); with it the 48px mark wins at
    // the true spot (0.60). The mark itself is never suppressed: its pixels are
    // below the 200 saturation bar on this dark base.
    const cv::Point pos(kW - 96 - 48, kH - 96 - 48);   // (752,1056)
    cv::Mat frame = textured(kW, kH, cv::Scalar(45, 55, 80));
    cv::putText(frame, "SUBMIT YOUR ENTRY NOW!", cv::Point(kW - 360, 1064),
                cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255, 255, 255), 3);
    cv::putText(frame, "vichealth.club/entry", cv::Point(kW - 360, 1096),
                cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255, 255, 255), 3);
    add_watermark_alpha_blend(frame, a48, pos, 255.0f);

    const cv::Point anchor = v2_small_config_from_dims(kW, kH).get_position(kW, kH);
    auto hit = locate_still_watermark_hybrid(to_gray(frame), templates, anchor, kW, kH);
    CAPTURE(hit.has_value());
    REQUIRE(hit.has_value());
    CHECK(hit->template_index == 1);                    // the 48px mark, not a text artifact
    CHECK(std::abs(hit->rect.x - pos.x) <= 3);
    CHECK(std::abs(hit->rect.y - pos.y) <= 3);
}

TEST_CASE("still_geometry: a snapped hit resolves to the PINNED preset, not the raw peak",
          "[still_geometry]") {
    // On busy content the raw NCC peak can sit tens of px from the calibrated mark
    // (a real 896x1200 poster drew it 35 px left on the correct row). Once a hit is
    // RECOGNIZED as a preset geometry (center L1 within tol), the resolved position
    // must be the preset's measured margins, not the raw peak. Here the mark is
    // stamped 30 px left of the preset slot: the peak lands at the stamp, the snap
    // fires, and the resolved margins are the preset's (96,96), not the stamp's
    // (126,96).
    WatermarkEngine engine;
    const cv::Mat a36 = engine.get_v2_diamond_alpha_36();
    const cv::Mat a48 = engine.get_v2_diamond_alpha_48_still();
    const std::vector<cv::Mat> templates{alpha_to_template(a36), alpha_to_template(a48)};
    const WatermarkPosition model = v2_small_config_from_dims(kW, kH);

    const cv::Point true_pos(kW - 96 - 48, kH - 96 - 48);      // (752,1056)
    cv::Mat frame = textured(kW, kH, cv::Scalar(50, 60, 90));
    add_watermark_alpha_blend(frame, a48, cv::Point(true_pos.x - 30, true_pos.y), 255.0f);

    StillGeometryOverride o;
    auto r = resolve_still_geometry(to_gray(frame), templates, model, kW, kH, o);
    CHECK(r.source == "auto/snapped");
    CHECK(r.pos.margin_right == 96);       // pinned to the preset, not the raw 126
    CHECK(r.pos.margin_bottom == 96);
    CHECK(r.pos.logo_size == 48);
    CHECK(r.template_index == 1);
}

// Gemini 3.6 stamps a 48px diamond at margin (96,96) even on large (>1024px) outputs.
// The size heuristic calls these "Large" (96px model); the content search must still
// recover the real 48px mark. kLgW/kLgH match the paintings-wm fixtures (2400x1792).
static constexpr int kLgW = 2400, kLgH = 1792;

TEST_CASE("still_geometry: Large Gemini 3.6 (48px @ 96,96) resolves via the search",
          "[still_geometry]") {
    WatermarkEngine engine;
    const cv::Mat a48 = engine.get_v2_diamond_alpha_48_still();
    REQUIRE(a48.cols == 48);

    // Stamp the 48px mark at the real Gemini 3.6 large geometry: margin (96,96).
    const cv::Point pos(kLgW - 96 - 48, kLgH - 96 - 48);  // (2256, 1648)
    cv::Mat frame = textured(kLgW, kLgH, cv::Scalar(70, 90, 110));
    add_watermark_alpha_blend(frame, a48, pos, 255.0f);

    // Size heuristic says Large; the member search must run anyway and find the 48px.
    StillGeometryOverride o;
    auto r = engine.resolve_still_geometry(frame, WatermarkVariant::V2,
                                           WatermarkSize::Large, o);
    REQUIRE(r.pos.has_value());
    CHECK(r.pos->margin_right == 96);
    CHECK(r.pos->margin_bottom == 96);
    CHECK(r.pos->logo_size == 48);
    REQUIRE(r.alpha != nullptr);
    CHECK(r.alpha->cols == 48);
}

TEST_CASE("still_geometry: Large 96px mark at (192,192) resolves via the 2K preset",
          "[still_geometry]") {
    // Was "falls back to model, no 48px false hit" before the 96px template joined
    // the search: a genuine 96px diamond at margin (192,192) (Gemini 3.5 legacy-large
    // and Gemini 3.8 2K, 1696x2528/1728x2462 measured) now snaps to the
    // gemini38-2k-portrait preset (short side 1792 is in [1600,1800]). The resolved
    // position equals the V2-large model position, so removal is unchanged on clean
    // content; the snap's value is TRUST (a content-collided fusion gate can be
    // bypassed). The 48px template still must not win on a 96px mark.
    WatermarkEngine engine;
    const cv::Mat a96 = engine.get_v2_diamond_alpha_large();
    REQUIRE(a96.cols == 96);

    // Stamp the real 96px mark at margin (192,192).
    const cv::Point pos(kLgW - 192 - 96, kLgH - 192 - 96);  // (2112, 1504)
    cv::Mat frame = textured(kLgW, kLgH, cv::Scalar(70, 90, 110));
    add_watermark_alpha_blend(frame, a96, pos, 255.0f);

    StillGeometryOverride o;
    auto r = engine.resolve_still_geometry(frame, WatermarkVariant::V2,
                                           WatermarkSize::Large, o);
    REQUIRE(r.pos.has_value());
    CHECK(r.pos->margin_right == 192);
    CHECK(r.pos->margin_bottom == 192);
    CHECK(r.pos->logo_size == 96);
    CHECK(r.trusted);
    CHECK(r.source == std::string("auto/snapped"));
    REQUIRE(r.alpha != nullptr);
    CHECK(r.alpha->cols == 96);   // routed to the V2 large alpha, not the 48px still
}

TEST_CASE("still_geometry: a raw (unsnapped) 96px hit is discarded — the V1 mark's slot",
          "[still_geometry]") {
    // The 96px template also matches the legacy V1 mark (96px @ margin 64,64; 0.99
    // NCC on the Gemini 3.1 Pro fixtures). An auto/raw 96px hit must NOT become a V2
    // override (wrong alpha for a V1 mark): the engine discards it, the image falls
    // back to the model and the V1 path handles it exactly as before.
    WatermarkEngine engine;
    const cv::Mat a96 = engine.get_v2_diamond_alpha_large();

    // Stamp a 96px mark at the V1-large geometry: margin (64,64), short side 1792
    // inside the 2K tier so ONLY the missing snap distinguishes it from the 3.8 case.
    const cv::Point pos(kLgW - 64 - 96, kLgH - 64 - 96);  // (2240, 1632)
    cv::Mat frame = textured(kLgW, kLgH, cv::Scalar(70, 90, 110));
    add_watermark_alpha_blend(frame, a96, pos, 255.0f);

    StillGeometryOverride o;
    auto r = engine.resolve_still_geometry(frame, WatermarkVariant::V2,
                                           WatermarkSize::Large, o);
    CAPTURE(r.source, r.score);
    CHECK_FALSE(r.pos.has_value());   // discarded -> model fallback
    CHECK_FALSE(r.trusted);
    CHECK(r.alpha == nullptr);
}

TEST_CASE("still_geometry: resolve_still_geometry precedence + matched alpha size",
          "[still_geometry]") {
    WatermarkEngine engine;
    const cv::Mat a36 = engine.get_v2_diamond_alpha_36();
    const cv::Mat a48 = engine.get_v2_diamond_alpha_48_still();
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
