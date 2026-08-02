#include <catch2/catch_test_macros.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdlib>
#include <filesystem>
#include "core/regenerator.hpp"
using namespace wmr;

TEST_CASE("regen smoke (needs model + vae)", "[regen][smoke]") {
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
    cv::Scalar m0 = cv::mean(img);
    cv::Mat out = img.clone();
    REQUIRE(r.regen(out, cfg));
    REQUIRE(out.size() == cv::Size(1024, 1024));
    REQUIRE(out.type() == CV_8UC3);
    cv::Scalar m1 = cv::mean(out);
    for (int c = 0; c < 3; ++c) REQUIRE(std::abs(m1[c] - m0[c]) <= 25.0);
}
