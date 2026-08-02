#include <catch2/catch_test_macros.hpp>
#include <opencv2/core.hpp>
#include "core/watermark_engine.hpp"
#include "core/inpaint.hpp"
using namespace wmr;

TEST_CASE("regen dispatch graceful fallback", "[regen][dispatch]") {
    WatermarkEngine engine;
    cv::Mat img = cv::Mat_<cv::Vec3b>(64, 64, cv::Vec3b(50, 60, 70));
    cv::Mat before = img.clone();
    InpaintConfig cfg;
#ifdef WMR_BUILD_REGEN
    cfg.method = InpaintMethod::DiffusionRegen;
    cfg.regen_allow_download = false;   // force "no model" -> regen fails -> graceful
#endif
    DetectionResult dr{}; dr.confidence = 0.0f; dr.detected = false;  // no visible mark; regen runs on whole image regardless
    REQUIRE_NOTHROW(engine.remove_watermark_detected(img, dr, cfg));
    REQUIRE(img.size() == before.size());   // survived, sane size
}
