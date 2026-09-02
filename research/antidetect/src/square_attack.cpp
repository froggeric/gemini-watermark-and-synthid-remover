#include "core/square_attack.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <random>

namespace wmr::antidetect {
namespace {

float mean_of(const std::vector<float>& v) {
    if (v.empty()) return 1.0f;
    double s = 0.0;
    for (const float x : v) s += x;
    return static_cast<float>(s / v.size());
}

bool all_below(const std::vector<float>& v, const std::vector<float>& thr) {
    if (v.size() != thr.size() || v.empty()) return false;
    for (size_t i = 0; i < v.size(); ++i)
        if (v[i] >= thr[i]) return false;
    return true;
}

// Elementwise clamp of a float plane to [-lim, lim].
void clamp_abs(cv::Mat& m, float lim) {
    cv::threshold(m, m, lim, lim, cv::THRESH_TRUNC);
    cv::multiply(m, -1.0f, m);
    cv::threshold(m, m, lim, lim, cv::THRESH_TRUNC);
    cv::multiply(m, -1.0f, m);
}

} // namespace

cv::Mat dct_band_weight(const cv::Mat& channel_f32, float keep_frac) {
    // cv::dct needs even dims; pad (reflect), transform, mask the corner block,
    // invert, crop back. The DC term is always kept.
    const int h = channel_f32.rows, w = channel_f32.cols;
    const int hp = h + (h % 2), wp = w + (w % 2);
    cv::Mat padded;
    cv::copyMakeBorder(channel_f32, padded, 0, hp - h, 0, wp - w,
                       cv::BORDER_REFLECT);
    cv::Mat freq;
    cv::dct(padded, freq);
    const int kh = std::max(1, static_cast<int>(hp * keep_frac));
    const int kw = std::max(1, static_cast<int>(wp * keep_frac));
    cv::Mat mask = cv::Mat::zeros(hp, wp, CV_32F);
    mask(cv::Rect(0, 0, kw, kh)).setTo(1.0f);
    freq = freq.mul(mask);
    cv::Mat spatial;
    cv::idct(freq, spatial);
    return spatial(cv::Rect(0, 0, w, h)).clone();
}

SquareAttackResult run_square_attack(const cv::Mat& base_bgr_u8,
                                     const EnsembleScorer& scorer,
                                     const std::vector<float>& flip_thresholds,
                                     const LpipsAlex& lpips,
                                     const SquareAttackConfig& cfg,
                                     const std::function<void(int, int)>& progress) {
    SquareAttackResult res;
    res.out = base_bgr_u8.clone();
    if (!scorer) return res;

    // Hard cap 16/255 (experimental headroom): the CLI strength mapping never
    // exceeds 4/255, so nothing user-facing reaches past that; the
    // WMR_AD_EPS_255 debug override in antidetect_adversarial.cpp is the only
    // caller that can, and it exists for transfer experiments (the literature's
    // measured commercial-detector drops live at eps 8/255).
    const float eps = std::clamp(cfg.eps, 0.0f, 16.0f / 255.0f);
    const int h = base_bgr_u8.rows, w = base_bgr_u8.cols;
    const int min_side = std::min(h, w);
    std::mt19937_64 rng(cfg.seed);

    const std::vector<float> base_scores = scorer(base_bgr_u8);
    res.base_mean_pfake = mean_of(base_scores);
    res.best_mean_pfake = res.base_mean_pfake;
    res.all_flipped = all_below(base_scores, flip_thresholds);
    if ((res.all_flipped && !cfg.disable_early_stop) || eps <= 0.0f)
        return res;  // nothing to do

    // Margin objective: sum of each score's distance ABOVE its flip threshold
    // (0 for already-flipped detectors). Minimizing the raw MEAN has a
    // measured failure mode: a detector that saturates near 0 vetoes every
    // candidate (any move nudges it up while the hard detector does not
    // immediately improve), freezing the attack with a near-empty delta
    // (observed 2026-09-02: commfor 0.990 -> 0.996 over 1000 queries while
    // corvi sat at 0). Margins make the ensemble's already-won members
    // invisible to the acceptance rule so the budget goes where it's needed.
    const auto margin_sum = [&flip_thresholds](const std::vector<float>& s) {
        float m = 0.0f;
        for (size_t i = 0; i < s.size() && i < flip_thresholds.size(); ++i)
            m += std::max(0.0f, s[i] - flip_thresholds[i]);
        return m;
    };
    float best_margin = margin_sum(base_scores);

    // Fixed planes: the base per-channel float (u8-rounded candidate = base +
    // projected delta, so what is scored is exactly what is returned).
    std::vector<cv::Mat> base_planes;
    {
        cv::Mat f;
        base_bgr_u8.convertTo(f, CV_32F);
        cv::split(f, base_planes);
    }
    std::vector<cv::Mat> delta_ch = {
        cv::Mat::zeros(h, w, CV_32F), cv::Mat::zeros(h, w, CV_32F),
        cv::Mat::zeros(h, w, CV_32F)};

    // Square-size schedule: the fraction p of the min side starts large and is
    // halved on a geometric schedule (the paper's shrinking search).
    float p = 0.25f;
    const int shrink_every = std::max(8, cfg.max_queries / 8);
    cv::Mat best_u8 = base_bgr_u8.clone();

    for (int q = 1; q <= cfg.max_queries; ++q) {
        if ((q % shrink_every) == 0) p = std::max(0.02f, p * 0.5f);

        const int side = std::max(2, static_cast<int>(std::lround(p * min_side)));
        std::uniform_int_distribution<int> xo(0, w - side);
        std::uniform_int_distribution<int> yo(0, h - side);
        const cv::Rect sq(xo(rng), yo(rng), side, side);
        // cfg.eps is normalized ([0,1] convention, e.g. 8/255); the delta
        // planes live in the 0-255 domain, so the increments and the box
        // clamp must scale by 255. The original version used the normalized
        // eps directly - a 255x-too-small perturbation that (with the
        // rounding bug below) froze the whole attack: zero accepts over
        // 1000 queries, PSNR 99 dB vs the base.
        const float eps255 = eps * 255.0f;
        cv::Vec3f inc;
        {
            std::uniform_real_distribution<float> u(-eps255, eps255);
            for (int c = 0; c < 3; ++c) inc[c] = u(rng);
        }

        // Tentative update on copies; committed only when accepted.
        std::vector<cv::Mat> trial = delta_ch;
        for (int c = 0; c < 3; ++c) trial[c](sq) += inc[c];

        // Project to the lowest DCT band, THEN clamp to the eps box (the band
        // projection redistributes energy and can push coefficients past the
        // box; the clamp has to be the last linear step so the u8 candidate is
        // guaranteed within eps of the base), then render. Rounding is plain
        // nearest (saturate_cast/cvRound) with NO +0.5 beta: a beta of 0.5
        // under round-half-to-even bumps every ODD-valued base pixel +1 even
        // at zero band, so every candidate carried a systematic dither the
        // base itself never had - the margin comparison then compared
        // different images and no candidate could ever win.
        cv::Mat candidate(h, w, CV_8UC3);
        std::vector<cv::Mat> planes(3);
        for (int c = 0; c < 3; ++c) {
            cv::Mat band = dct_band_weight(trial[c], cfg.dct_keep_frac);
            clamp_abs(band, eps255);
            cv::Mat sum;
            cv::add(base_planes[c], band, sum);
            sum.convertTo(planes[c], CV_8U, 1.0, 0.0);
        }
        cv::merge(planes, candidate);

        const std::vector<float> scores = scorer(candidate);
        const float m = margin_sum(scores);
        if (m < best_margin) {
            const double lp =
                lpips.distance_capped(base_bgr_u8, candidate, cfg.lpips_eval_side);
            if (lp <= cfg.lpips_budget) {
                delta_ch = std::move(trial);
                best_margin = m;
                res.best_mean_pfake = mean_of(scores);
                res.lpips_final = lp;
                best_u8 = candidate.clone();
                res.all_flipped = all_below(scores, flip_thresholds);
                if (res.all_flipped && !cfg.disable_early_stop) {
                    res.queries_used = q;
                    res.out = best_u8;
                    if (progress) progress(q, cfg.max_queries);
                    return res;  // early stop: everyone flipped
                }
            }
        }
        res.queries_used = q;
        if (progress && (q % 25 == 0 || q == cfg.max_queries))
            progress(q, cfg.max_queries);
    }
    res.out = best_u8;
    return res;
}

} // namespace wmr::antidetect
