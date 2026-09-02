#pragma once

#include "core/antidetect_physics.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace wmr::antidetect {

// Facade for the anti-detection pipeline: stage A (physics, always available),
// stage B (adversarial, only when built WITH the surrogate suite and the models
// are present), and the stage-C report. See
// docs/research/anti-detection-pipeline-research.md for the honest scope:
// validated against the LOCAL surrogate suite and open detector families only.
enum class AntidetectMethod {
    Physics,     // stage A only, zero downloads, works everywhere
    Adversarial, // stage B only; refuses (ok=false) when unavailable
    Full,        // stage A then stage B (default)
    Auto,        // full, silently degrading to physics when B is unavailable
};

struct AntidetectConfig {
    AntidetectMethod method = AntidetectMethod::Full;
    float strength = 0.5f;         // scales every physics dose + the eps/query budget
    float lpips_budget = 0.05f;    // perceptual gate for the adversarial stage
    bool jpeg_cycle = true;
    long long seed = -1;           // >= 0 -> deterministic output
    std::string surrogates = "commfor,corvi";
    bool allow_download = true;
};

struct SurrogateScore {
    std::string name;
    float before = 0.0f;  // mean p_fake before the pass
    float after = 0.0f;
};

struct AntidetectReport {
    bool physics_ran = false;
    bool adversarial_ran = false;
    bool adversarial_available = true;  // false on physics-only builds
    std::string note;                   // degradation / refusal reason
    PhysicsStats physics;
    double psnr = 0.0;
    double ssim = 0.0;
    double lpips = 0.0;
    std::vector<SurrogateScore> scores;  // populated when the suite could run
};

struct AntidetectResult {
    bool ok = false;  // false only for a refused explicit `adversarial` request
    AntidetectReport report;
};

#ifdef WMR_ANTIDETECT_ADVERSARIAL
// Stage B entry point, implemented in antidetect_adversarial.cpp. Declared
// here (external linkage) so the facade compiles without ORT headers.
struct AdversarialStageResult {
    bool ran = false;
    std::string note;
    std::vector<SurrogateScore> scores;
};
AdversarialStageResult run_adversarial_stage(cv::Mat& bgr_u8,
                                             const AntidetectConfig& cfg,
                                             double& lpips_out);
#endif

// Runs in place. Degradation policy (single place; the CLI only maps it):
//   physics                          -> never touches the network.
//   full/auto + models absent        -> physics-only; note set (auto stays
//                                       silent-ish, full warns via the note).
//   adversarial + models absent      -> refused: ok=false, note says why.
AntidetectResult run_antidetect(cv::Mat& bgr_u8, const AntidetectConfig& cfg);

// Gaussian-window SSIM (Wang et al. 2004) on BT.601 luma, inputs in [0,1] float
// internally; 1.0 for identical inputs. opencv_quality is not linked, hence the
// hand-rolled implementation.
double ssim_luma(const cv::Mat& a_u8, const cv::Mat& b_u8);

// Human-readable report block (spdlog-friendly one string per line, joined
// with '\n'; indented adds a two-space prefix for the chained/batch flow).
std::string format_antidetect_report(const AntidetectReport& r, bool indented);

const char* antidetect_method_name(AntidetectMethod m);

} // namespace wmr::antidetect
