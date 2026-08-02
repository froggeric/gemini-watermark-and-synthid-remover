#include <catch2/catch_test_macros.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdlib>
#include <filesystem>
#include "core/regenerator.hpp"
using namespace wmr;

TEST_CASE("regen smoke (needs model + vae)", "[regen][smoke]") {
    // Running this test creates + destroys an sd_ctx, which aborts the Catch2
    // test-exe at process exit (ggml-metal static device-registry teardown).
    // So it is a STANDALONE dev probe, not a normal unit test: SKIP by default,
    // even when the model is cached. Opt in with WMR_REGEN_SMOKE_RUN=1.
    if (!std::getenv("WMR_REGEN_SMOKE_RUN")) {
        WARN("skipping regen smoke: set WMR_REGEN_SMOKE_RUN=1 to run (needs cached model; "
             "standalone only - the test exe aborts at exit due to ggml-metal teardown)");
        return;
    }

    namespace fs = std::filesystem;
    const char* env_m  = std::getenv("WMR_REGEN_MODEL");
    const char* env_v  = std::getenv("WMR_REGEN_VAE");
    std::string home   = std::getenv("HOME") ? std::getenv("HOME") : ".";
    fs::path cache_m   = fs::path(home) / ".cache" / "wmr" / "sd_xl_base_1.0.safetensors";
    fs::path cache_v   = fs::path(home) / ".cache" / "wmr" / "sdxl_vae-fp16-fix.safetensors";
    bool have_model = (env_m && env_m[0]) || fs::exists(cache_m);
    bool have_vae   = (env_v && env_v[0]) || fs::exists(cache_v);
    if (!have_model || !have_vae) {
        WARN("skipping regen smoke: need SDXL model + fp16-fix VAE "
             "(set $WMR_REGEN_MODEL + $WMR_REGEN_VAE, or cache both in ~/.cache/wmr/)");
        return;
    }

    Regenerator r;
    RegenConfig cfg; cfg.strength = 0.05f; cfg.steps = 8; cfg.allow_download = false;
    REQUIRE(r.initialize(cfg));
    REQUIRE(r.is_ready());
    cv::Mat img = cv::imread("test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-black-gemini.png");
    if (img.empty()) { WARN("skipping: fixture absent"); return; }
    cv::resize(img, img, {1024, 1024});
    cv::Mat out = img.clone();
    REQUIRE(r.regen(out, cfg));
    // Validity check: regen is a REGENERATION (SDXL img2img) that legitimately
    // changes pixels, so input/output mean-closeness is NOT the criterion.
    // Instead assert a structurally valid, finite, non-saturated image.
    REQUIRE(out.size() == cv::Size(1024, 1024));
    REQUIRE(out.type() == CV_8UC3);
    REQUIRE(cv::checkRange(out, true));  // no NaN / inf
    cv::Scalar m1 = cv::mean(out);
    for (int c = 0; c < 3; ++c) {
        INFO("channel " << c << " mean=" << m1[c]);
        REQUIRE(m1[c] >= 0.5);
        REQUIRE(m1[c] <= 254.5);
    }
}

