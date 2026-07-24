#include <catch2/catch_test_macros.hpp>

#include "video/geometry_detector.hpp"
#include "core/watermark_engine.hpp"
#include "core/types.hpp"
#include "core/blend_modes.hpp"
#include <opencv2/imgproc.hpp>

#include <vector>

using namespace wmr;

// Same textured background helper used by the V2 round-trip tests: a smooth
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
// CV_32FC1 [0,1] alpha -> CV_8UC1 template (the dtype match_mark expects).
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

// Gemini corner window (mirrors the production constant): a 320x320 box anchored
// at the bottom-right, large enough to hold a 1080p mark (margin 192 + size 96).
static cv::Rect gem_corner(int W, int H) {
    const int x0 = std::max(0, W - 320);
    const int y0 = std::max(0, H - 320);
    return cv::Rect(x0, y0, W - x0, H - y0);
}

static constexpr float kThresh = 0.45f;

TEST_CASE("geometry: recovers 720p-1 (48 @ margin 72,72)", "[geometry]") {
    WatermarkEngine engine;
    const cv::Mat a48 = engine.get_v2_diamond_alpha_48_still();
    const cv::Mat a96 = engine.get_v2_diamond_alpha_large();
    const std::vector<cv::Mat> templates{alpha_to_template(a48), alpha_to_template(a96)};

    const int W = 1280, H = 720;
    const cv::Point pos(W - 72 - 48, H - 72 - 48);  // (1160, 600)
    cv::Mat frame = textured(W, H, cv::Scalar(80, 100, 120));
    add_watermark_alpha_blend(frame, a48, pos, 255.0f);

    auto hit = detect_geometry_in_frames({to_gray(frame)}, templates, gem_corner(W, H), kThresh);
    REQUIRE(hit.has_value());
    CHECK(hit->template_index == 0);
    CHECK(std::abs(hit->rect.x - pos.x) <= 3);
    CHECK(std::abs(hit->rect.y - pos.y) <= 3);
    CHECK(hit->score >= kThresh);

    auto snap = snap_geometry_to_known(hit->rect, W, H, VideoProfile::GeminiDiamond);
    CHECK(snap.snapped);
    CHECK(snap.label == "720p-1");
    CHECK(snap.geometry.margin_right == 72);
    CHECK(snap.geometry.margin_bottom == 72);
    CHECK(snap.geometry.logo_size == 48);
}

TEST_CASE("geometry: recovers 720p-2 (48 @ margin 29,40) via position snap", "[geometry]") {
    WatermarkEngine engine;
    const cv::Mat a48 = engine.get_v2_diamond_alpha_48_still();
    const cv::Mat a96 = engine.get_v2_diamond_alpha_large();
    const std::vector<cv::Mat> templates{alpha_to_template(a48), alpha_to_template(a96)};

    const int W = 1280, H = 720;
    const cv::Point pos(W - 29 - 48, H - 40 - 48);  // (1203, 632)
    cv::Mat frame = textured(W, H, cv::Scalar(90, 70, 110));
    add_watermark_alpha_blend(frame, a48, pos, 255.0f);

    auto hit = detect_geometry_in_frames({to_gray(frame)}, templates, gem_corner(W, H), kThresh);
    REQUIRE(hit.has_value());
    CHECK(hit->template_index == 0);  // size cannot tell 720p-1 from 720p-2
    CHECK(std::abs(hit->rect.x - pos.x) <= 3);

    auto snap = snap_geometry_to_known(hit->rect, W, H, VideoProfile::GeminiDiamond);
    CHECK(snap.snapped);
    CHECK(snap.label == "720p-2");
    CHECK(snap.geometry.margin_right == 29);
    CHECK(snap.geometry.margin_bottom == 40);
    CHECK(snap.geometry.logo_size == 44);  // table value (alpha still 48 via the gate)
}

TEST_CASE("geometry: recovers 1080p (96 @ margin 192,192)", "[geometry]") {
    WatermarkEngine engine;
    const cv::Mat a48 = engine.get_v2_diamond_alpha_48_still();
    const cv::Mat a96 = engine.get_v2_diamond_alpha_large();
    const std::vector<cv::Mat> templates{alpha_to_template(a48), alpha_to_template(a96)};

    const int W = 1920, H = 1080;
    const cv::Point pos(W - 192 - 96, H - 192 - 96);  // (1632, 792)
    cv::Mat frame = textured(W, H, cv::Scalar(70, 90, 130));
    add_watermark_alpha_blend(frame, a96, pos, 255.0f);

    auto hit = detect_geometry_in_frames({to_gray(frame)}, templates, gem_corner(W, H), kThresh);
    REQUIRE(hit.has_value());
    CHECK(hit->template_index == 1);
    CHECK(std::abs(hit->rect.x - pos.x) <= 3);

    auto snap = snap_geometry_to_known(hit->rect, W, H, VideoProfile::GeminiDiamond);
    CHECK(snap.snapped);
    CHECK(snap.label == "1080p");
    CHECK(snap.geometry.margin_right == 192);
    CHECK(snap.geometry.margin_bottom == 192);
    CHECK(snap.geometry.logo_size == 96);
}

TEST_CASE("geometry: snap tolerates a 10px near-miss, rejects an 80px miss", "[geometry]") {
    const int W = 1280, H = 720;
    // True 720p-1 mark center is (1184, 624).
    const cv::Rect near_miss(1160 + 10, 600 + 10, 48, 48);  // center (1194,634), 20 L1
    auto s1 = snap_geometry_to_known(near_miss, W, H, VideoProfile::GeminiDiamond);
    CHECK(s1.snapped);
    CHECK(s1.label == "720p-1");

    const cv::Rect far_miss(1160 - 80, 600, 48, 48);  // center (1104,624), 80 L1 from 720p-1
    auto s2 = snap_geometry_to_known(far_miss, W, H, VideoProfile::GeminiDiamond);
    CHECK_FALSE(s2.snapped);
    CHECK(s2.label == "auto");
    // Margins derived from the raw rect's far corner.
    CHECK(s2.geometry.margin_right == W - (far_miss.x + far_miss.width));
}

TEST_CASE("geometry: polarity-invariant (bright-on-dark and dark-on-bright)", "[geometry]") {
    WatermarkEngine engine;
    const cv::Mat a48 = engine.get_v2_diamond_alpha_48_still();
    const cv::Mat a96 = engine.get_v2_diamond_alpha_large();
    const std::vector<cv::Mat> templates{alpha_to_template(a48), alpha_to_template(a96)};
    const int W = 1280, H = 720;
    const cv::Point pos(W - 72 - 48, H - 72 - 48);

    cv::Mat bright = textured(W, H, cv::Scalar(25, 25, 25));      // dark bg, bright mark
    add_watermark_alpha_blend(bright, a48, pos, 255.0f);
    auto h1 = detect_geometry_in_frames({to_gray(bright)}, templates, gem_corner(W, H), kThresh);
    REQUIRE(h1.has_value());
    CHECK(h1->template_index == 0);
    CHECK(h1->score >= kThresh);

    cv::Mat dark = textured(W, H, cv::Scalar(210, 210, 210));     // bright bg, dark mark
    add_watermark_alpha_blend(dark, a48, pos, 0.0f);
    auto h2 = detect_geometry_in_frames({to_gray(dark)}, templates, gem_corner(W, H), kThresh);
    REQUIRE(h2.has_value());
    CHECK(h2->template_index == 0);
    CHECK(h2->score >= kThresh);
}

TEST_CASE("geometry: clean frame yields no detection", "[geometry]") {
    WatermarkEngine engine;
    const cv::Mat a48 = engine.get_v2_diamond_alpha_48_still();
    const cv::Mat a96 = engine.get_v2_diamond_alpha_large();
    const std::vector<cv::Mat> templates{alpha_to_template(a48), alpha_to_template(a96)};
    const int W = 1280, H = 720;
    cv::Mat frame = textured(W, H, cv::Scalar(80, 100, 120));  // no watermark

    auto hit = detect_geometry_in_frames({to_gray(frame)}, templates, gem_corner(W, H), kThresh);
    CHECK_FALSE(hit.has_value());
}

TEST_CASE("geometry: rect_to_watermark_position + effective_alpha_size", "[geometry]") {
    // effective_alpha_size mirrors the >48/>68 gate (P720_2's 44 routes to 48).
    CHECK(effective_alpha_size(VideoProfile::GeminiDiamond, 48) == cv::Size(48, 48));
    CHECK(effective_alpha_size(VideoProfile::GeminiDiamond, 96) == cv::Size(96, 96));
    CHECK(effective_alpha_size(VideoProfile::GeminiDiamond, 44) == cv::Size(48, 48));
    CHECK(effective_alpha_size(VideoProfile::VeoLegacy, 68) == cv::Size(68, 30));
    CHECK(effective_alpha_size(VideoProfile::VeoLegacy, 99) == cv::Size(99, 43));
    CHECK(effective_alpha_size(VideoProfile::VeoLegacy, 70) == cv::Size(99, 43));  // 70 > 68

    // Round-trip a manual rect: 48-wide diamond at margin 72,72.
    const int W = 1280, H = 720;
    const cv::Rect rect(W - 72 - 48, H - 72 - 48, 48, 48);
    auto wp = rect_to_watermark_position(rect, W, H, VideoProfile::GeminiDiamond);
    CHECK(wp.margin_right == 72);
    CHECK(wp.margin_bottom == 72);
    CHECK(wp.logo_size == 48);

    // A slightly-off width (50) snaps to the nearest asset width (48), not 96.
    auto wp2 = rect_to_watermark_position(cv::Rect(100, 100, 50, 50), W, H, VideoProfile::GeminiDiamond);
    CHECK(wp2.logo_size == 48);
}

TEST_CASE("geometry: regression gate decides when auto overrides the guess", "[geometry]") {
    constexpr float kRawBar = 0.60f;  // matches kAutoOverrideRawScore in video_processor.cpp
    // A snapped (on-table) detection is trusted at the detection min score.
    CHECK(decide_auto_geometry(true, 0.45f, kRawBar) == AutoGeometryVerdict::UseSnapped);
    CHECK(decide_auto_geometry(true, 0.99f, kRawBar) == AutoGeometryVerdict::UseSnapped);
    // A raw (off-table) detection must clear the bar, else fall back.
    CHECK(decide_auto_geometry(false, 0.55f, kRawBar) == AutoGeometryVerdict::FallBack);
    CHECK(decide_auto_geometry(false, 0.60f, kRawBar) == AutoGeometryVerdict::UseRaw);  // at the bar
    CHECK(decide_auto_geometry(false, 0.80f, kRawBar) == AutoGeometryVerdict::UseRaw);  // above
}

