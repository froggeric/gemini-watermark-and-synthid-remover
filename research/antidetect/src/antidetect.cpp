#include "core/antidetect.hpp"

#include "core/lpips_alex.hpp"

#include <opencv2/imgproc.hpp>

#include <fmt/format.h>

#include <cmath>
#include <random>

namespace wmr::antidetect {
namespace {

std::mt19937_64 make_rng(long long seed) {
    if (seed >= 0) return std::mt19937_64(static_cast<uint64_t>(seed));
    std::random_device rd;
    return std::mt19937_64((static_cast<uint64_t>(rd()) << 32) ^ rd());
}

} // namespace

const char* antidetect_method_name(AntidetectMethod m) {
    switch (m) {
        case AntidetectMethod::Physics: return "physics";
        case AntidetectMethod::Adversarial: return "adversarial";
        case AntidetectMethod::Full: return "full";
        case AntidetectMethod::Auto: return "auto";
    }
    return "?";
}

double ssim_luma(const cv::Mat& a_u8, const cv::Mat& b_u8) {
    cv::Mat ga, gb;
    cv::cvtColor(a_u8, ga, cv::COLOR_BGR2GRAY);
    cv::cvtColor(b_u8, gb, cv::COLOR_BGR2GRAY);
    ga.convertTo(ga, CV_32F, 1.0 / 255.0);
    gb.convertTo(gb, CV_32F, 1.0 / 255.0);

    // 11x11 Gaussian window, sigma 1.5 (the reference SSIM configuration).
    const cv::Mat k = cv::getGaussianKernel(11, 1.5, CV_32F);
    const cv::Mat k2 = k * k.t();
    auto blur = [&](const cv::Mat& m) {
        cv::Mat out;
        cv::filter2D(m, out, CV_32F, k2, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
        return out;
    };

    const cv::Mat mu_a = blur(ga), mu_b = blur(gb);
    cv::Mat aa, bb, ab;
    cv::multiply(ga, ga, aa);
    cv::multiply(gb, gb, bb);
    cv::multiply(ga, gb, ab);
    const cv::Mat var_a = blur(aa) - mu_a.mul(mu_a);
    const cv::Mat var_b = blur(bb) - mu_b.mul(mu_b);
    const cv::Mat cov = blur(ab) - mu_a.mul(mu_b);

    // SSIM = [(2*mu_a*mu_b + C1) * (2*cov + C2)] /
    //        [(mu_a^2 + mu_b^2 + C1) * (var_a + var_b + C2)]  -- MULTIPLICATIVE
    // luminance-contrast coupling (an additive variant reads ~1 for anything).
    const double c1 = 0.01 * 0.01, c2 = 0.03 * 0.03;
    cv::Mat mu_a2, mu_b2, mu_ab, l_num, l_den, c_num, c_den;
    cv::multiply(mu_a, mu_a, mu_a2);
    cv::multiply(mu_b, mu_b, mu_b2);
    cv::multiply(mu_a, mu_b, mu_ab);
    l_num = 2.0 * mu_ab + c1;
    l_den = mu_a2 + mu_b2 + c1;
    c_num = 2.0 * cov + c2;
    c_den = var_a + var_b + c2;
    cv::Mat num, den, ssim_map;
    cv::multiply(l_num, c_num, num);
    cv::multiply(l_den, c_den, den);
    cv::divide(num, den, ssim_map);
    return cv::mean(ssim_map)[0];
}

AntidetectResult run_antidetect(cv::Mat& bgr_u8, const AntidetectConfig& cfg) {
    AntidetectResult res;
    AntidetectReport& rep = res.report;
    res.ok = true;

    const cv::Mat original = bgr_u8.clone();

#if !defined(WMR_ANTIDETECT_ADVERSARIAL)
    rep.adversarial_available = false;
    if (cfg.method == AntidetectMethod::Adversarial) {
        res.ok = false;
        rep.note = "adversarial stage unavailable in this build (physics-only)";
        return res;  // image untouched on refusal
    }
    if (cfg.method == AntidetectMethod::Full)
        rep.note = "adversarial stage unavailable in this build; ran physics only";
#endif

    const bool wants_adversarial = cfg.method == AntidetectMethod::Adversarial ||
                                   cfg.method == AntidetectMethod::Full ||
                                   cfg.method == AntidetectMethod::Auto;

    // Stage A first (full = physics then adversarial: the attack optimizes the
    // final image, so it runs on the physics output, not under it).
    if (cfg.method != AntidetectMethod::Adversarial) {
        PhysicsConfig pc;
        pc.strength = cfg.strength;
        pc.jpeg_cycle = cfg.jpeg_cycle;
        std::mt19937_64 rng = make_rng(cfg.seed);
        rep.physics = apply_physics(bgr_u8, pc, rng);
        rep.physics_ran = true;
    }

#if defined(WMR_ANTIDETECT_ADVERSARIAL)
    if (wants_adversarial) {
        AdversarialStageResult ad = run_adversarial_stage(bgr_u8, cfg, rep.lpips);
        rep.adversarial_ran = ad.ran;
        rep.note = ad.note;
        rep.scores = std::move(ad.scores);
        if (!ad.ran && cfg.method == AntidetectMethod::Adversarial) {
            res.ok = false;  // explicit request refused (models absent etc.)
        }
    }
#endif

    if (!res.ok) return res;  // adversarial-only refusal: image untouched

    // Stage C metrics vs the pre-pass original (identical inputs -> PSNR
    // clamped to 99 dB).
    cv::Mat f32a, f32b, diff;
    bgr_u8.convertTo(f32a, CV_32F);
    original.convertTo(f32b, CV_32F);
    cv::absdiff(f32a, f32b, diff);
    cv::multiply(diff, diff, diff);
    const double mse = cv::mean(diff)[0];  // mean squared error, u8 units
    rep.psnr = (mse <= 1e-9) ? 99.0 : 10.0 * std::log10(255.0 * 255.0 / mse);
    rep.ssim = ssim_luma(original, bgr_u8);
    if (rep.lpips == 0.0) {
        static const LpipsAlex kLpips;  // weights are compile-time constants
        rep.lpips = kLpips.distance_capped(original, bgr_u8, 512);
    }
    return res;
}

std::string format_antidetect_report(const AntidetectReport& r, bool indented) {
    const std::string pad = indented ? "  " : "";
    std::string out;
    const auto line = [&](const std::string& s) { out += pad + s + "\n"; };

    line("Anti-detection pass:");
    if (r.physics_ran) {
        line(fmt::format(
            "  physics: kernel={} noise={:.1f}/255 ca={:.2f}px vignette={:.2f}{}",
            r.physics.kernel == DemosaicKernel::Malvar5x5 ? "malvar"
                : r.physics.kernel == DemosaicKernel::Bilinear ? "bilinear"
                                                               : "edge-aware",
            r.physics.noise_sigma_midtone_255, r.physics.ca_shift_px,
            r.physics.vignette_k,
            r.physics.jpeg_quality > 0
                ? fmt::format(" jpeg_q={}", r.physics.jpeg_quality)
                : std::string(" jpeg=off")));
    }
    if (r.adversarial_ran) {
        line(fmt::format("  adversarial: {} surrogate(s)", r.scores.size()));
        for (const auto& s : r.scores)
            line(fmt::format("    {:<10} {:.3f} -> {:.3f}", s.name, s.before, s.after));
    } else if (!r.adversarial_available) {
        line("  adversarial: unavailable in this build (physics-only)");
    }
    if (!r.note.empty()) line(fmt::format("  note: {}", r.note));
    // Quality metrics only exist when a stage actually ran (a refused pass
    // leaves the image untouched and the zeros would be misleading).
    if (r.physics_ran || r.adversarial_ran) {
        line(fmt::format("  quality vs input: PSNR {:.1f} dB, SSIM {:.3f}, LPIPS {:.3f}",
                         r.psnr, r.ssim, r.lpips));
    }
    return out;
}

} // namespace wmr::antidetect
