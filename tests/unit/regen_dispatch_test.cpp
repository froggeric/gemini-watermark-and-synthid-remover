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
    // Point at a guaranteed-missing file so initialize ALWAYS takes the
    // "model absent + download disabled -> return false" path, WITHOUT creating
    // an sd_ctx / Metal device (which would abort the test-exe at exit via the
    // ggml-metal static teardown). Cache-independent: passes whether or not the
    // real SDXL model is cached in ~/.cache/wmr/.
    cfg.regen_model_path = "/nonexistent/wmr-regen-no-model-test.safetensors";
#endif
    DetectionResult dr{}; dr.confidence = 0.0f; dr.detected = false;  // no visible mark; regen runs on whole image regardless
    // The bool return surfaces a failed/no-op regen to the exit code: a failing regen
    // (no model + allow_download=false here) MUST return false. Keep the NOTHROW check
    // too (graceful = no throw). Needs no model — the failure is the point (SKIP-free).
    bool ok = true;
    REQUIRE_NOTHROW(ok = engine.remove_watermark_detected(img, dr, cfg));
#ifdef WMR_BUILD_REGEN
    REQUIRE_FALSE(ok);                     // regen no-op'd -> false
#else
    REQUIRE(ok);                            // non-regen path: alpha-blend no-op'd on no-mark, returns true
#endif
    REQUIRE(img.size() == before.size());   // survived, sane size
}
