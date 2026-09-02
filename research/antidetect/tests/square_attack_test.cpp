// Square Attack unit tests: a MOCK scorer (no ORT), the real embedded LPIPS.
#include <catch2/catch_test_macros.hpp>

#include "core/square_attack.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <random>

using namespace wmr::antidetect;

namespace {

// Deterministic base engineered to sit just under a decision boundary the
// eps-bounded attack can cross.
cv::Mat near_boundary_image(int h, int w) {
    return cv::Mat(h, w, CV_8UC3, cv::Vec3b(90, 90, 127));  // mean R = 127
}

// Ramp "detector": p_fake falls steeply with the mean red level, 0.99 at 127
// to 0.01 at 128.5. Continuous (so the greedy loop gets a gradient from the
// first accepted square) and steep enough that partial eps-coverage of the
// image already crosses the flip threshold — a mock with a reachable target
// inside the L_inf box.
std::vector<float> step_scorer(const cv::Mat& m) {
    cv::Mat ch[3];
    cv::split(m, ch);
    const double mean_r = cv::mean(ch[2])[0];
    const float p = static_cast<float>(0.99 - 1.9 * (mean_r - 127.0));
    return {std::min(0.99f, std::max(0.01f, p))};
}

cv::Mat textured_image(int h, int w, uint64_t seed) {
    cv::Mat m(h, w, CV_8UC3);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> u(0, 255);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            m.at<cv::Vec3b>(y, x) = cv::Vec3b(u(rng), u(rng), u(rng));
    return m;
}

} // namespace

TEST_CASE("dct_band_weight kills high-frequency energy", "[antidetect][square]") {
    cv::Mat low(64, 64, CV_32F);
    for (int y = 0; y < 64; ++y)
        for (int x = 0; x < 64; ++x)
            low.at<float>(y, x) = std::sin(x / 10.0f) + std::cos(y / 12.0f);
    cv::Mat high(64, 64, CV_32F);
    for (int y = 0; y < 64; ++y)
        for (int x = 0; x < 64; ++x)
            high.at<float>(y, x) = std::sin(x * 0.9f) * std::cos(y * 0.8f);

    const cv::Mat low_kept = dct_band_weight(low, 0.2f);
    const cv::Mat high_kept = dct_band_weight(high, 0.2f);
    cv::Mat d;
    cv::absdiff(low, low_kept, d);
    const double low_mean = cv::mean(d)[0];
    cv::absdiff(high, high_kept, d);
    const double high_mean = cv::mean(d)[0];
    REQUIRE(low_mean < 0.05);         // low band passes through (reflect-pad
                                      // smears a little boundary energy)
    REQUIRE(high_mean > 0.1);         // high band is destroyed
}

TEST_CASE("square attack flips a boundary-crossing detector within its bounds",
          "[antidetect][square]") {
    const cv::Mat base = near_boundary_image(96, 128);
    const LpipsAlex lpips;
    SquareAttackConfig cfg;
    cfg.seed = 42;
    cfg.max_queries = 600;
    cfg.lpips_budget = 0.15f;  // generous: the mock must be winnable
    cfg.eps = 8.0f / 255.0f;   // real energy: with the OLD rendering bug the
                               // flip came free from a +0.5 rounding artifact
                               // on odd base pixels (127 -> 128); the fixed
                               // attack must EARN it with perturbation mass

    const SquareAttackResult r =
        run_square_attack(base, step_scorer, {0.5f}, lpips, cfg, nullptr);

    REQUIRE(r.base_mean_pfake > 0.9f);       // the mock starts "fake"
    REQUIRE(r.best_mean_pfake < 0.5f);       // ...and is driven across the flip
                                             // threshold. (The margin objective
                                             // stops at the crossing: margin 0
                                             // leaves nothing to minimize, so
                                             // the old "< 0.05" expectation
                                             // belonged to the mean-objective
                                             // era and to the rendering bug.)
    REQUIRE(r.all_flipped);
    REQUIRE(r.queries_used <= cfg.max_queries);

    // Eps box + u8 quantization slack: every pixel within eps + 1 of base.
    cv::Mat d;
    cv::absdiff(base, r.out, d);
    double maxd = 0.0;
    cv::minMaxIdx(d, nullptr, &maxd);
    REQUIRE(maxd <= 255.0 * cfg.eps + 1.0);

    // Determinism: same seed, same bytes.
    const SquareAttackResult r2 =
        run_square_attack(base, step_scorer, {0.5f}, lpips, cfg, nullptr);
    REQUIRE(cv::sum(r.out != r2.out) == cv::Scalar(0));
}

TEST_CASE("square attack respects the LPIPS budget", "[antidetect][square]") {
    // NOTE: on a FLAT base the layer-2 LPIPS is exactly invariant to uniform
    // shifts (per-channel spatial unit normalization cancels the scale of a
    // constant map), so the gate is only meaningful on textured content.
    const cv::Mat base = textured_image(64, 64, 11);
    const LpipsAlex lpips;
    SquareAttackConfig tight;
    tight.seed = 1;
    tight.max_queries = 300;
    tight.lpips_budget = 0.0f;  // only a bit-identical candidate could pass
    const SquareAttackResult r =
        run_square_attack(base, step_scorer, {0.5f}, lpips, tight, nullptr);
    REQUIRE(r.best_mean_pfake == r.base_mean_pfake);  // no candidate passed the gate
    REQUIRE(cv::sum(r.out != base) == cv::Scalar(0));
}
