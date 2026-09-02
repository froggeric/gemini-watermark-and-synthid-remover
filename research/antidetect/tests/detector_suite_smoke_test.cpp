// Real-model smoke for the ORT surrogate suite. SKIPs unless
// WMR_TEST_ANTIDETECT_MODELS points at a dir containing the pinned surrogate
// files (the ai_denoise_test real-model precedent: developers / an opt-in CI
// job run it; the default suite stays network-free).
#include <catch2/catch_test_macros.hpp>

#ifdef WMR_ANTIDETECT_ADVERSARIAL
#include "core/detector_suite_ort.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>

using namespace wmr::antidetect;

TEST_CASE("detector suite scores are sane and deterministic", "[antidetect][suite]") {
    const char* dir = std::getenv("WMR_TEST_ANTIDETECT_MODELS");
    if (dir == nullptr || *dir == '\0') {
        SKIP("WMR_TEST_ANTIDETECT_MODELS not set; surrogate models not fetched");
    }

    DetectorSuite suite;
    std::vector<const SurrogateSpec*> specs;
    for (const auto& s : kSurrogateManifest)
        if (std::filesystem::exists(std::filesystem::path(dir) / s.filename))
            specs.push_back(&s);
    if (specs.empty()) SKIP("no surrogate model files present in the models dir");
    REQUIRE(suite.initialize(specs, dir));

    // A flat neutral image: any sane detector gives a stable score.
    cv::Mat flat(384, 384, CV_8UC3, cv::Scalar(128, 128, 128));
    const auto s1 = suite.score(flat);
    const auto s2 = suite.score(flat);
    REQUIRE(s1.size() == specs.size());
    for (size_t i = 0; i < s1.size(); ++i) {
        REQUIRE(s1[i] >= 0.0f);
        REQUIRE(s1[i] <= 1.0f);
        REQUIRE(s1[i] == s2[i]);  // deterministic scoring
    }
}
#endif
