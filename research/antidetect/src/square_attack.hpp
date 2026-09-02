#pragma once

#include "core/lpips_alex.hpp"

#include <opencv2/core.hpp>

#include <functional>
#include <vector>

namespace wmr::antidetect {

// Stage B: a score-based Square Attack (Andriushchenko et al., ECCV 2020 —
// algorithm reimplemented from the paper, the reference code is BSD-3) against
// the LOCAL surrogate ensemble. One ensemble evaluation per iteration; greedy
// accept on mean p_fake; perturbation energy confined to the lowest DCT band
// (the JPEG-surviving frequencies, per FBA2D/DuFIA); LPIPS perceptual gate.
// Pure OpenCV + the embedded LPIPS: the scorer is injected, so the loop is
// unit-testable with a mock and without ORT.

// Returns one p_fake per ensemble member (order fixed by the caller).
using EnsembleScorer = std::function<std::vector<float>(const cv::Mat& bgr_u8)>;

struct SquareAttackConfig {
    float eps = 2.0f / 255.0f;       // hard-capped at 4/255 by the facade
    int max_queries = 500;           // budget = 200 + int(800 * strength), cap 1000
    float lpips_budget = 0.05f;
    float dct_keep_frac = 0.2f;      // perturbation energy kept in the lowest band
    int lpips_eval_side = 512;       // LPIPS gate evaluated on a downscaled copy
    uint64_t seed = 0;
    bool disable_early_stop = false; // transfer experiments only: spend the full
                                     // budget even when every surrogate has
                                     // flipped (the local-vs-commercial oracle
                                     // mismatch is the thing under test)
};

struct SquareAttackResult {
    cv::Mat out;                     // best candidate (the input when nothing improved)
    float base_mean_pfake = 1.0f;
    float best_mean_pfake = 1.0f;
    int queries_used = 0;
    double lpips_final = 0.0;
    bool all_flipped = false;        // every member below its threshold at best
};

// Runs the attack. `flip_thresholds` (one per ensemble member, from the
// manifest) drive the early stop. `progress` (optional) is called with
// (queries_used, max_queries) after each iteration.
SquareAttackResult run_square_attack(const cv::Mat& base_bgr_u8,
                                     const EnsembleScorer& scorer,
                                     const std::vector<float>& flip_thresholds,
                                     const LpipsAlex& lpips,
                                     const SquareAttackConfig& cfg,
                                     const std::function<void(int, int)>& progress);

// Zero every DCT coefficient outside the lowest keep_frac x keep_frac corner
// block (per channel). Exposed for tests. Input CV_32F single channel.
cv::Mat dct_band_weight(const cv::Mat& channel_f32, float keep_frac);

} // namespace wmr::antidetect
