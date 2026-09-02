// Stage B wiring: fetch surrogate models, run the Square Attack against the
// local ensemble, and report before/after scores. Only compiled when the
// adversarial feature pulled in ORT (WMR_ANTIDETECT_ADVERSARIAL).
#ifdef WMR_ANTIDETECT_ADVERSARIAL
#include "core/antidetect.hpp"

#include "core/antidetect_models_fetch.hpp"
#include "core/detector_suite_ort.hpp"
#include "core/square_attack.hpp"
#include "cli/progress.hpp"

#include <spdlog/spdlog.h>
#include <fmt/format.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

namespace wmr::antidetect {

AdversarialStageResult run_adversarial_stage(cv::Mat& bgr_u8,
                                             const AntidetectConfig& cfg,
                                             double& lpips_out) {
    AdversarialStageResult out;

    // 1. Ensure the surrogate models (never fatal here; the facade decides
    //    whether a missing stage is a refusal or a degradation).
    const SurrogateFetchResult fetched =
        ensure_surrogate_models(cfg.surrogates, cfg.allow_download);
    if (fetched.available.empty()) {
        out.note = "surrogate models unavailable (" + fetched.note + ")";
        return out;
    }

    // 2. Process-lifetime suite (deliberately leaked singleton: destroying an
    //    Ort::Session during static teardown races ORT's own teardown — the
    //    NcnnDenoiser rationale, applied to ORT).
    static DetectorSuite* suite = new DetectorSuite();
    static std::vector<const SurrogateSpec*> loaded;
    if (!suite->is_ready() || loaded != fetched.available) {
        if (!suite->initialize(fetched.available, antidetect_models_dir())) {
            out.note = "no surrogate detector could be loaded";
            return out;
        }
        loaded = fetched.available;
    }

    // 3. Before-scores.
    const std::vector<float> before = suite->score(bgr_u8);

    // 4. Attack. eps maps from strength (<= 4/255 hard cap, also clamped in
    //    the loop); the query budget scales with strength.
    SquareAttackConfig sac;
    sac.eps = (1.0f + 3.0f * cfg.strength) / 255.0f;
    sac.max_queries = std::min(1000, 200 + static_cast<int>(800.0f * cfg.strength));
    sac.lpips_budget = cfg.lpips_budget;
    sac.seed = cfg.seed >= 0 ? static_cast<uint64_t>(cfg.seed)
                             : (static_cast<uint64_t>(std::random_device()()) << 32) ^
                                   std::random_device()();

    // Transfer-experiment escapes (debug, env-gated; the shipped CLI can never
    // exceed eps 4/255 or skip the early stop):
    //   WMR_AD_EPS_255=<1..16>  override eps (the literature's measured
    //                           commercial-detector drops live at 8/255).
    //   WMR_AD_NO_EARLY_STOP=1  spend the full query budget even when every
    //                           local surrogate already reads "real" (a
    //                           local-vs-commercial oracle mismatch is
    //                           exactly what the transfer test exposes).
    if (const char* eps_env = std::getenv("WMR_AD_EPS_255")) {
        const float e = std::clamp(static_cast<float>(std::atof(eps_env)), 0.0f, 16.0f);
        if (e > 0.0f) {
            sac.eps = e / 255.0f;
            spdlog::warn("Antidetect: WMR_AD_EPS_255 override -> eps {:.0f}/255 "
                         "(EXPERIMENTAL transfer test, visible quality)",
                         e);
        }
    }
    if (std::getenv("WMR_AD_NO_EARLY_STOP")) {
        sac.disable_early_stop = true;
        spdlog::warn("Antidetect: WMR_AD_NO_EARLY_STOP set -> running the full "
                     "query budget (transfer test)");
    }
    if (const char* lp_env = std::getenv("WMR_AD_LPIPS_BUDGET")) {
        const float b = static_cast<float>(std::atof(lp_env));
        if (b > 0.0f) {
            sac.lpips_budget = b;
            spdlog::warn("Antidetect: WMR_AD_LPIPS_BUDGET override -> {:.2f} "
                         "(EXPERIMENTAL transfer test)", b);
        }
    }

    const LpipsAlex lpips;
    std::vector<float> thresholds;
    for (const SurrogateSpec* s : suite->specs()) thresholds.push_back(s->flip_threshold);

    // Captureless: `suite` has static storage, so the lambda names it directly.
    const auto scorer = [](const cv::Mat& m) { return suite->score(m); };
    // Query progress on stderr (the report block itself prints via spdlog on
    // stdout after the loop). No Stage header: the facade's report names the
    // stage, and whether physics ran before it is not known here.
    auto rep = std::make_unique<ProgressReporter>("  adversarial", sac.max_queries, "q");
    const auto progress = [&rep](int done, int total) {
        rep->update(done);
    };

    const SquareAttackResult res =
        run_square_attack(bgr_u8, scorer, thresholds, lpips, sac, progress);
    // The summary must say what actually ran: ProgressReporter::finish() always
    // renders the bar at 100% (by design, for callers that complete fewer
    // units than declared), so "600/600" alone would read as "600 queries
    // spent" even when the attack early-stopped at 0 because every surrogate
    // was already below its flip threshold.
    std::string summary;
    if (res.queries_used == 0) {
        summary = fmt::format("all surrogates already below threshold "
                              "(mean p_fake {:.3f}); 0 queries needed",
                              res.best_mean_pfake);
    } else if (res.queries_used < sac.max_queries) {
        summary = fmt::format("mean p_fake {:.3f} -> {:.3f}, early stop at {}/{} queries",
                              res.base_mean_pfake, res.best_mean_pfake,
                              res.queries_used, sac.max_queries);
    } else {
        summary = fmt::format("mean p_fake {:.3f} -> {:.3f}", res.base_mean_pfake,
                              res.best_mean_pfake);
    }
    rep->finish(summary);
    bgr_u8 = res.out;
    lpips_out = res.lpips_final;

    // 5. After-scores + report rows.
    const std::vector<float> after = suite->score(bgr_u8);
    for (size_t i = 0; i < suite->specs().size(); ++i) {
        SurrogateScore sc;
        sc.name = suite->specs()[i]->key;
        sc.before = i < before.size() ? before[i] : -1.0f;
        sc.after = i < after.size() ? after[i] : -1.0f;
        out.scores.push_back(std::move(sc));
    }
    out.ran = true;
    return out;
}

} // namespace wmr::antidetect
#endif
