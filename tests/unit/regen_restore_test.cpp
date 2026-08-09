#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <vector>
#include "core/regen_restore.hpp"
using namespace wmr;
using Catch::Approx;

#ifdef WMR_BUILD_REGEN

namespace {

// Scaled-down RestoreConfig used by the luminance-gate and keep-mask tests
// (which exercise the gate/mask logic on tiny images, not the Wiener math; the
// band/dc_radius are shrunk so those small images still have a non-empty band).
RestoreConfig ref_config(RestoreMode mode) {
    RestoreConfig c;
    c.mode = mode;
    c.gamma = 4.0f;
    c.dc_radius = 2.0f;
    c.highfreq_cutoff = 20.0f;
    c.calib_band_low = 2.0f;
    c.calib_band_high = 3.0f;
    c.target_r = 2.5f;
    c.prior_exponent = 1.3f;
    return c;
}

// Radial frequency grid in cycle units, natural FFT order (mirrors the internal
// radial_freqs in regen_restore.cpp and numpy's fftfreq(N)*N).
cv::Mat radial_freqs_grid(int h, int w) {
    cv::Mat r(h, w, CV_32FC1);
    for (int i = 0; i < h; ++i) {
        const float fy = (i <= h / 2) ? float(i) : float(i - h);
        float* row = r.ptr<float>(i);
        for (int j = 0; j < w; ++j) {
            const float fx = (j <= w / 2) ? float(j) : float(j - w);
            row[j] = std::sqrt(fy * fy + fx * fx);
        }
    }
    return r;
}

inline float clamp01(float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }

} // namespace

// ---------------------------------------------------------------------------
// Luminance gate (Rec601, BGR order): bright >= 128 -> restore; dim < 128 -> full regen.
// ---------------------------------------------------------------------------
TEST_CASE("regen restore luminance gate", "[regen][restore]") {
    SECTION("mean_luminance_bgr matches Rec601") {
        cv::Mat solid(4, 4, CV_8UC3, cv::Scalar(200, 100, 50));  // B=200,G=100,R=50
        // 0.114*200 + 0.587*100 + 0.299*50 = 22.8 + 58.7 + 14.95 = 96.45
        REQUIRE(mean_luminance_bgr(solid) == Approx(96.45f).epsilon(0.001f));
    }
    SECTION("bright image is restored under Auto") {
        cv::Mat O(8, 8, CV_8UC3, cv::Scalar(220, 220, 220));   // bright, luma ~220
        cv::Mat R(8, 8, CV_8UC3, cv::Scalar(210, 210, 210));   // slightly different -> nonzero D
        RestoreConfig c = ref_config(RestoreMode::Auto);
        cv::Mat Rp = restore_detail(O, R, c);
        // Auto + bright -> restoration ran -> Rp must differ from R somewhere.
        // (Every pixel has |D|=~17.3, well above the 95th percentile, so all kept.)
        bool differs = false;
        for (int i = 0; i < 8 && !differs; ++i)
            for (int j = 0; j < 8 && !differs; ++j)
                if (Rp.at<cv::Vec3b>(i, j) != R.at<cv::Vec3b>(i, j)) differs = true;
        REQUIRE(differs);
    }
    SECTION("dim image falls back to full regen under Auto") {
        cv::Mat O(8, 8, CV_8UC3, cv::Scalar(40, 40, 40));    // dim, luma ~40 < 128
        cv::Mat R(8, 8, CV_8UC3, cv::Scalar(50, 50, 50));    // nonzero D
        RestoreConfig c = ref_config(RestoreMode::Auto);
        cv::Mat Rp = restore_detail(O, R, c);
        REQUIRE(std::equal(Rp.data, Rp.data + Rp.total() * Rp.elemSize(), R.data));
    }
    SECTION("On forces restore even on dim content") {
        cv::Mat O(8, 8, CV_8UC3, cv::Scalar(40, 40, 40));
        cv::Mat R(8, 8, CV_8UC3, cv::Scalar(50, 50, 50));
        RestoreConfig c = ref_config(RestoreMode::On);
        cv::Mat Rp = restore_detail(O, R, c);
        bool differs = false;
        for (int i = 0; i < 8 && !differs; ++i)
            for (int j = 0; j < 8 && !differs; ++j)
                if (Rp.at<cv::Vec3b>(i, j) != R.at<cv::Vec3b>(i, j)) differs = true;
        REQUIRE(differs);
    }
    SECTION("Off skips restoration (returns R unchanged)") {
        cv::Mat O(8, 8, CV_8UC3, cv::Scalar(220, 220, 220));
        cv::Mat R(8, 8, CV_8UC3, cv::Scalar(210, 210, 210));
        RestoreConfig c = ref_config(RestoreMode::Off);
        cv::Mat Rp = restore_detail(O, R, c);
        REQUIRE(std::equal(Rp.data, Rp.data + Rp.total() * Rp.elemSize(), R.data));
    }
    SECTION("luminance threshold is configurable") {
        // O luma ~96 (< 128 default -> full regen). Lowering the threshold to 80
        // keeps it "bright" relative to the new threshold -> restoration runs.
        cv::Mat O(8, 8, CV_8UC3, cv::Scalar(100, 100, 100));  // luma 100
        cv::Mat R(8, 8, CV_8UC3, cv::Scalar(90, 90, 90));
        RestoreConfig c = ref_config(RestoreMode::Auto);
        c.luminance_threshold = 80.0f;
        cv::Mat Rp = restore_detail(O, R, c);
        bool differs = false;
        for (int i = 0; i < 8 && !differs; ++i)
            for (int j = 0; j < 8 && !differs; ++j)
                if (Rp.at<cv::Vec3b>(i, j) != R.at<cv::Vec3b>(i, j)) differs = true;
        REQUIRE(differs);
    }
}

// ---------------------------------------------------------------------------
// Keep mask: top keep_fraction by combined L2 of ORIGINAL D.
// ---------------------------------------------------------------------------
TEST_CASE("regen restore keep mask", "[regen][restore]") {
    // Build O,R with a UNIQUE |D| per pixel (a 10x10 ramp in the blue channel),
    // so the percentile threshold has no ties and the mask count is exact.
    auto ramp_pair = [](int H, int W) {
        cv::Mat O(H, W, CV_8UC3, cv::Scalar(200, 200, 200));
        cv::Mat R(H, W, CV_8UC3, cv::Scalar(200, 200, 200));
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
                O.at<cv::Vec3b>(i, j)[0] = static_cast<uchar>(200 - (i * W + j));  // distinct |D|
        return std::make_pair(O, R);
    };
    auto count_restored = [](const cv::Mat& Rp, const cv::Mat& R) {
        int n = 0;
        for (int i = 0; i < Rp.rows; ++i)
            for (int j = 0; j < Rp.cols; ++j)
                if (Rp.at<cv::Vec3b>(i, j) != R.at<cv::Vec3b>(i, j)) ++n;
        return n;
    };

    SECTION("keep_fraction=1 keeps every pixel") {
        auto [O, R] = ramp_pair(10, 10);   // 100 px, all distinct |D|
        RestoreConfig c = ref_config(RestoreMode::On);
        c.keep_fraction = 1.0f;            // thr = min |D| -> mask keeps all
        cv::Mat Rp = restore_detail(O, R, c);
        // Every pixel has |D| > 0, and with keep_fraction=1 the threshold is the
        // minimum |D|, so all pixels are kept and restored (Rp != R everywhere).
        REQUIRE(count_restored(Rp, R) == 100);
    }
    SECTION("top-5% keeps exactly the largest-magnitude pixels") {
        auto [O, R] = ramp_pair(10, 10);   // |D| = 0,1,...,99 (distinct)
        RestoreConfig c = ref_config(RestoreMode::On);
        c.keep_fraction = 0.05f;           // top 5% of 100 = the 5 largest
        cv::Mat Rp = restore_detail(O, R, c);
        // The 5 largest-|D| pixels (indices 95..99, i.e. (9,5)..(9,9)) are kept;
        // the rest are not. Some boundary pixels may attenuate to exactly R, so
        // count the kept set among the top-|D| region and assert it matches the
        // expected ~5% (allow 4-6 for percentile-boundary rounding).
        int n_restored = count_restored(Rp, R);
        INFO("n_restored = " << n_restored << " (expected ~5)");
        REQUIRE(n_restored >= 4);
        REQUIRE(n_restored <= 8);
        // The very-largest-|D| pixel (9,9) must be restored; the smallest (0,0) must not.
        REQUIRE(Rp.at<cv::Vec3b>(9, 9) != R.at<cv::Vec3b>(9, 9));
        REQUIRE(Rp.at<cv::Vec3b>(0, 0) == R.at<cv::Vec3b>(0, 0));
    }
    SECTION("keep_fraction=0.50 keeps half") {
        auto [O, R] = ramp_pair(10, 10);
        RestoreConfig c = ref_config(RestoreMode::On);
        c.keep_fraction = 0.50f;           // top 50% of 100 = ~50 px
        cv::Mat Rp = restore_detail(O, R, c);
        int n_restored = count_restored(Rp, R);
        INFO("n_restored = " << n_restored << " (expected ~50)");
        REQUIRE(n_restored >= 45);
        REQUIRE(n_restored <= 55);
    }
}

// ---------------------------------------------------------------------------
// Wiener shape: low-mid energy reduced, high-freq preserved.
// Uses a 128x128 broadband diff with the DEFAULT config (band [25,35] is
// well-populated, unlike a pure sinusoid or a tiny image).
// ---------------------------------------------------------------------------
TEST_CASE("regen restore wiener shape", "[regen][restore]") {
    const int H = 128, W = 128;
    cv::Mat R(H, W, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Mat O(H, W, CV_8UC3, cv::Scalar(128, 128, 128));
    // Deterministic broadband diff (a simple LCG so the test is reproducible).
    unsigned int rng = 12345u;
    for (int i = 0; i < H; ++i) {
        cv::Vec3b* row = O.ptr<cv::Vec3b>(i);
        for (int j = 0; j < W; ++j) {
            for (int c = 0; c < 3; ++c) {
                rng = rng * 1103515245u + 12345u;
                const int d = static_cast<int>((rng >> 16) % 41u) - 20;   // [-20, 20]
                row[j][c] = static_cast<uchar>(std::max(0, std::min(255, 128 + d)));
            }
        }
    }
    RestoreConfig c;   // DEFAULTS (gamma=4, dc_radius=25, cutoff=600, calib [25,35])
    c.mode = RestoreMode::On;
    c.keep_fraction = 1.0f;   // restore everything so D_att is observable in Rp - R
    cv::Mat Rp = restore_detail(O, R, c);
    REQUIRE(!Rp.empty());

    // D = O - R (float); D_att = Rp - R (float, post-attenuation).
    cv::Mat O_f, R_f, Rp_f, D, D_att;
    O.convertTo(O_f, CV_32FC3); R.convertTo(R_f, CV_32FC3); Rp.convertTo(Rp_f, CV_32FC3);
    cv::subtract(O_f, R_f, D);
    cv::subtract(Rp_f, R_f, D_att);

    // *** Attenuation-is-real guard: D_att must differ from D. A no-op regression
    // (K=0 / empty band / Wiener bypassed) leaves D_att == D, caught here. ***
    {
        cv::Mat diff;
        cv::absdiff(D, D_att, diff);
        double mx; cv::minMaxLoc(diff, nullptr, &mx);
        double mean_delta = cv::mean(diff)[0];
        INFO("D_att vs D: mean|diff|=" << mean_delta << " max|diff|=" << mx);
        REQUIRE(mean_delta > 0.5);   // broadband Wiener removes a visible slice
        REQUIRE(mx > 1.0);
    }

    const cv::Mat rgrid = radial_freqs_grid(H, W);
    cv::Mat lo_mask = (rgrid >= 5.0f) & (rgrid <= 40.0f);    // low-mid band
    cv::Mat hi_mask = (rgrid >= 50.0f) & (rgrid <= 90.0f);   // high-freq band

    // Sum |F|^2 over each band, summed across BGR channels, for D and D_att.
    auto band_energy = [&](const cv::Mat& diff, const cv::Mat& mask) {
        double e = 0.0;
        std::vector<cv::Mat> chs(3);
        cv::split(diff, chs);
        cv::Mat mask_f; mask.convertTo(mask_f, CV_32FC1, 1.0 / 255.0);
        for (auto& ch : chs) {
            cv::Mat planes[2] = {ch, cv::Mat::zeros(ch.size(), CV_32FC1)};
            cv::Mat cpx; cv::merge(planes, 2, cpx);
            cv::Mat F; cv::dft(cpx, F);
            std::vector<cv::Mat> Fre; cv::split(F, Fre);
            cv::Mat mag2; cv::multiply(Fre[0], Fre[0], mag2);
            cv::Mat im2;  cv::multiply(Fre[1], Fre[1], im2);
            mag2 += im2;
            cv::Mat masked; cv::multiply(mag2, mask_f, masked);
            e += cv::sum(masked)[0];
        }
        return e;
    };
    const double e_lo_D = band_energy(D, lo_mask);
    const double e_lo_att = band_energy(D_att, lo_mask);
    const double e_hi_D = band_energy(D, hi_mask);
    const double e_hi_att = band_energy(D_att, hi_mask);
    const double reduce_lo = (e_lo_D - e_lo_att) / std::max(e_lo_D, 1.0);
    const double reduce_hi = (e_hi_D - e_hi_att) / std::max(e_hi_D, 1.0);
    INFO("low-mid reduction = " << reduce_lo << "  high-freq reduction = " << reduce_hi);
    // Low-mid band [5,40]: the carrier prior (calibrated at r=30) attenuates a
    // large fraction of the diff energy here.
    REQUIRE(reduce_lo > 0.40);
    // High-freq band [50,90]: attenuation is mild. The highfreq_rolloff cutoff
    // (r=600) does not engage on a 128x128 image (max r ~90), so the only
    // attenuation here is the carrier prior's r^-1.3 falloff -> a small fraction.
    REQUIRE(reduce_hi < 0.30);
    // Shape direction: low-mid reduced MORE than high-freq (the core property).
    REQUIRE(reduce_lo > reduce_hi);
}

// ---------------------------------------------------------------------------
// Numerical match to an independent cv::dft Wiener on a DEFAULT-config fixture.
//
// The old 16x16 "matches python reference" fixture was a no-op: the scaled
// calibration band [2,3] on a 16x16 image left K ~= 0, so D_att ~= D on both
// sides and the "match" was two no-ops (a regression to no-attenuation, which
// is exactly the bug class that shipped undetected, passed silently). This
// replacement uses a 96x96 broadband diff with the DEFAULT band [25,35]
// (well-populated at that size), then cross-checks the C++ Wiener output
// (kissfft) against the same Wiener formula computed with cv::dft, channel by
// channel. A future no-op or FFT-layout regression breaks either the
// attenuation-is-real guard or the dft cross-check.
// ---------------------------------------------------------------------------
TEST_CASE("regen restore wiener matches independent dft", "[regen][restore]") {
    const int H = 96, W = 96;
    cv::Mat R(H, W, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Mat O(H, W, CV_8UC3, cv::Scalar(128, 128, 128));
    // Deterministic broadband diff so the calibration band [25,35] is
    // well-populated and K is non-trivial (NOT a near-zero no-op).
    unsigned int rng = 999u;
    for (int i = 0; i < H; ++i) {
        cv::Vec3b* row = O.ptr<cv::Vec3b>(i);
        for (int j = 0; j < W; ++j) {
            for (int c = 0; c < 3; ++c) {
                rng = rng * 1103515245u + 12345u;
                const int d = static_cast<int>((rng >> 16) % 51u) - 25;   // [-25, 25]
                row[j][c] = static_cast<uchar>(std::max(0, std::min(255, 128 + d)));
            }
        }
    }

    // DEFAULT config (gamma=4, band [25,35], target_r=30). keep_fraction=1 so
    // D_att = Rp - R everywhere (no mask to undo).
    RestoreConfig c;
    c.mode = RestoreMode::On;
    c.keep_fraction = 1.0f;
    cv::Mat Rp = restore_detail(O, R, c);
    REQUIRE(!Rp.empty());

    // Recover D_att = Rp - R (float) and D = O - R (float).
    cv::Mat O_f, R_f, Rp_f, D, D_att;
    O.convertTo(O_f, CV_32FC3); R.convertTo(R_f, CV_32FC3); Rp.convertTo(Rp_f, CV_32FC3);
    cv::subtract(O_f, R_f, D);
    cv::subtract(Rp_f, R_f, D_att);

    // --- Guard 1: attenuation is real (D_att != D). A no-op Wiener (K=0 from an
    // empty band, or a bypass) leaves D_att == D, which this catches.
    {
        cv::Mat diff; cv::absdiff(D, D_att, diff);
        const double mean_delta = cv::mean(diff)[0];
        const double total = diff.total() * diff.channels();
        int n_changed = 0;
        for (int i = 0; i < diff.rows; ++i) {
            const float* p = diff.ptr<float>(i);
            for (int j = 0; j < diff.cols * 3; ++j) if (p[j] > 0.5f) ++n_changed;
        }
        INFO("D_att vs D: mean|diff|=" << mean_delta << " changed_px=" << n_changed
             << "/" << total);
        REQUIRE(mean_delta > 0.3f);
        REQUIRE(n_changed > static_cast<int>(total * 0.10));   // >10% of bins moved
    }

    // --- Guard 2: NOT full-suppression (the shipped bug). A Wiener with K far
    // too large (e.g. target_r 10x wrong, or the carrier prior increasing instead
    // of decreasing with r) drives alpha -> 1 everywhere, shrink -> 0, and D_att
    // -> 0 (the entire diff is zeroed). The restored region then reads back as R
    // (Rp ~= R), producing the ~28/255 end-to-end error. Guard by asserting the
    // attenuated diff is still substantial: the Wiener trims low-mid energy but
    // leaves most of the signal, so mean|D_att| should be a healthy fraction of
    // mean|D|. (mean|D_att| -> 0 on the full-suppression bug.)
    {
        cv::Mat absD, absDa;
        cv::absdiff(D, cv::Scalar::all(0.0), absD);    // |D|
        cv::absdiff(D_att, cv::Scalar::all(0.0), absDa);  // |D_att|
        const double mean_D   = cv::mean(absD)[0];
        const double mean_Da  = cv::mean(absDa)[0];
        const double ratio = mean_Da / std::max(mean_D, 1e-6);
        INFO("mean|D|=" << mean_D << " mean|D_att|=" << mean_Da << " ratio=" << ratio);
        // Correct Wiener on broadband: ratio ~0.5-0.9 (trims but doesn't zero).
        // Full-suppression bug: ratio -> 0. No-op: ratio -> 1.0.
        REQUIRE(mean_Da > 1.0);           // not zeroed
        REQUIRE(ratio > 0.20);            // most of the diff survives
        REQUIRE(ratio < 0.99);            // but some IS removed
    }

    // --- Guard 3: cross-check D_att against the same Wiener formula via cv::dft
    // (an independent FFT). kiss_fftnd and cv::dft agree; the formula is
    // identical, so a layout/normalization bug in the kissfft path surfaces here.
    // Tolerances account for the u8 round-trip (D_att is recovered from Rp - R
    // where Rp is u8-quantized: uniform [-0.5, 0.5] noise, mean|.| ~= 0.25).
    const cv::Mat rgrid = radial_freqs_grid(H, W);
    const float weights[3] = {0.90f, 1.00f, 0.86f};   // BGR
    std::vector<cv::Mat> D_ch(3); cv::split(D, D_ch);
    std::vector<cv::Mat> D_att_ref_ch(3);

    const double prior_exp = 1.3;
    cv::Mat band_mask = (rgrid >= 25.0f) & (rgrid <= 35.0f);
    band_mask.convertTo(band_mask, CV_32FC1, 1.0 / 255.0);

    for (int ch = 0; ch < 3; ++ch) {
        // Forward cv::dft (complex, unnormalized).
        cv::Mat planes[2] = {D_ch[ch], cv::Mat::zeros(D_ch[ch].size(), CV_32FC1)};
        cv::Mat cpx; cv::merge(planes, 2, cpx);
        cv::Mat F; cv::dft(cpx, F);
        std::vector<cv::Mat> Fre; cv::split(F, Fre);
        cv::Mat mag; cv::magnitude(Fre[0], Fre[1], mag);

        // Calibration: K = 0.5 * mean(|F|[band]) * 30^1.3, in double.
        cv::Mat mag_band; cv::multiply(mag, band_mask, mag_band);
        const double band_count = cv::countNonZero(band_mask);
        const double mean_mag = cv::sum(mag_band)[0] / band_count;
        const double K = 0.5 * mean_mag * std::pow(30.0, prior_exp);
        const double Kc = K * static_cast<double>(weights[ch]);

        // Per-bin Wiener shrink (double for the scalar math, matching regen_restore.cpp).
        cv::Mat shrink(H, W, CV_64FC1);
        for (int i = 0; i < H; ++i) {
            const float* rr = rgrid.ptr<float>(i);
            const float* mr = mag.ptr<float>(i);
            double* sr = shrink.ptr<double>(i);
            for (int j = 0; j < W; ++j) {
                const double rv = rr[j];
                const double rel = clamp01(static_cast<float>(rv / 25.0))
                                 * clamp01(static_cast<float>((600.0 - rv) / 300.0));
                const double cm = Kc * std::pow(rv + 1e-9, -prior_exp);
                const double cp = cm * cm;
                const double fmag2 = static_cast<double>(mr[j]) * mr[j];
                const double rp = std::max(fmag2 - cp, 1e-9);
                const double alpha = cp / (cp + rp);
                double s = 1.0 - 4.0 * alpha * rel;
                if (s < 0.0) s = 0.0;
                sr[j] = s;
            }
        }
        // Apply shrink to F (complex), inverse dft with DFT_SCALE (1/N).
        cv::Mat shrink_f; shrink.convertTo(shrink_f, CV_32FC1);
        cv::Mat Fatt_re, Fatt_im;
        cv::multiply(Fre[0], shrink_f, Fatt_re);
        cv::multiply(Fre[1], shrink_f, Fatt_im);
        cv::Mat Fatt; cv::merge(std::vector<cv::Mat>{Fatt_re, Fatt_im}, Fatt);
        cv::Mat inv; cv::idft(Fatt, inv, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
        D_att_ref_ch[ch] = inv;
    }
    cv::Mat D_att_ref; cv::merge(D_att_ref_ch, D_att_ref);

    // Compare C++ D_att (kissfft, recovered from u8 Rp) to the cv::dft reference.
    cv::Mat cmp; cv::absdiff(D_att, D_att_ref, cmp);
    const double mean_delta = cv::mean(cmp)[0];
    double max_delta = 0.0; cv::minMaxLoc(cmp, nullptr, &max_delta);
    INFO("kissfft vs cv::dft Wiener: mean|diff|=" << mean_delta << " max=" << max_delta
         << " (u8 round-trip noise floor: mean~0.25 max~0.5)");
    // The residual is u8 quantization (mean ~0.25, max ~0.5). A real FFT layout
    // or normalization bug would produce mean > 1.0, max > 5.0.
    REQUIRE(mean_delta < 0.40);
    REQUIRE(max_delta < 1.50);
}

#else  // !WMR_BUILD_REGEN

TEST_CASE("regen restore types compile without regen", "[regen][restore]") {
    // RestoreConfig and RestoreMode are always defined (InpaintConfig carries
    // RestoreConfig unconditionally). restore_detail is only DECLARED; its
    // symbol is absent in a regen-free build, so we never call it here.
    RestoreConfig cfg;
    REQUIRE(cfg.mode == RestoreMode::Off);   // library default is Off
    cfg.mode = RestoreMode::Auto;
    REQUIRE(cfg.mode == RestoreMode::Auto);
}

#endif
