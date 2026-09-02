#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "core/antidetect_physics.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <random>

using namespace wmr::antidetect;

namespace {

double mean_abs_diff(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat d;
    cv::absdiff(a, b, d);
    return cv::mean(d)[0];
}

cv::Mat solid(const cv::Vec3b& color, int h = 64, int w = 64) {
    return cv::Mat(h, w, CV_8UC3, color);
}

// Horizontal neutral gradient (all channels equal): locally correlated content,
// which is exactly what gradient-corrected demosaicing assumes — every kernel is
// exact on it by construction.
cv::Mat neutral_gradient(int h = 64, int w = 64) {
    cv::Mat m(h, w, CV_8UC3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const uint8_t v = static_cast<uint8_t>(x * 255 / (w - 1));
            m.at<cv::Vec3b>(y, x) = cv::Vec3b(v, v, v);
        }
    return m;
}

} // namespace

TEST_CASE("bayer mosaic places channels on the RGGB lattice", "[antidetect]") {
    const cv::Mat red = solid(cv::Vec3b(0, 0, 255));  // BGR: pure red
    const cv::Mat bayer = bayer_mosaic_rggb(red);
    REQUIRE(bayer.type() == CV_8UC1);
    REQUIRE(bayer.rows == red.rows);
    // (0,0)=R=255, (0,1)=G=0, (1,0)=G=0, (1,1)=B=0 locks the RGGB convention.
    REQUIRE(bayer.at<uint8_t>(0, 0) == 255);
    REQUIRE(bayer.at<uint8_t>(0, 1) == 0);
    REQUIRE(bayer.at<uint8_t>(1, 0) == 0);
    REQUIRE(bayer.at<uint8_t>(1, 1) == 0);
}

TEST_CASE("demosaic round-trips a neutral solid exactly for every kernel",
          "[antidetect]") {
    const cv::Mat gray = solid(cv::Vec3b(128, 128, 128));
    const cv::Mat bayer = bayer_mosaic_rggb(gray);
    for (int k = 0; k < static_cast<int>(DemosaicKernel::Count); ++k) {
        const cv::Mat back = demosaic(bayer, static_cast<DemosaicKernel>(k));
        CAPTURE(k);
        // All kernels have DC gain 1, so a neutral solid is reproduced exactly.
        REQUIRE(back.type() == CV_8UC3);
        REQUIRE(mean_abs_diff(back, gray) < 1.0);
    }
}

TEST_CASE("demosaic preserves chromatic solids for the averaging kernels",
          "[antidetect]") {
    // MHC deliberately trades DC gain for gradient correction using
    // cross-channel terms, so it overshoots on SYNTHETIC chromatic solids
    // (natural content has correlated channels). Bilinear and the edge-aware
    // variant only average same-channel sites and are exact on any solid.
    const cv::Mat red = solid(cv::Vec3b(0, 0, 255));
    const cv::Mat bayer = bayer_mosaic_rggb(red);
    for (const auto k : {DemosaicKernel::Bilinear, DemosaicKernel::EdgeAware}) {
        const cv::Mat back = demosaic(bayer, k);
        CAPTURE(static_cast<int>(k));
        const auto mean = cv::mean(back);
        REQUIRE(mean[2] > 250.0);  // R survives
        REQUIRE(mean[1] < 8.0);    // G clean
        REQUIRE(mean[0] < 8.0);    // B clean
    }
}

TEST_CASE("demosaic interpolates a smooth correlated gradient closely for every kernel",
          "[antidetect]") {
    const cv::Mat grad = neutral_gradient();
    const cv::Mat bayer = bayer_mosaic_rggb(grad);
    for (int k = 0; k < static_cast<int>(DemosaicKernel::Count); ++k) {
        const cv::Mat back = demosaic(bayer, static_cast<DemosaicKernel>(k));
        CAPTURE(k);
        const double tol =
            (k == static_cast<int>(DemosaicKernel::EdgeAware)) ? 6.0 : 2.0;
        REQUIRE(mean_abs_diff(back, grad) < tol);
    }
}

TEST_CASE("demosaic round-trips smooth COLORFUL content closely for every kernel",
          "[antidetect]") {
    // Regression: the Malvar blue plane once passed (vertical, horizontal)
    // kernel order on top of assemble_plane's phase mirror — a double swap
    // that mis-oriented blue on the G phases. Invisible on the neutral tests
    // above (every kernel estimates the same value when R==G==B) but a ~10 dB
    // PSNR crater on colorful content (the "validate on colorful content"
    // rule). Smooth uncorrelated per-channel gradients: colorful enough to
    // expose a channel/orientation error, smooth enough that a correct
    // gradient-corrected demosaic reconstructs it closely.
    cv::Mat img(96, 96, CV_8UC3);
    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            const float fx = x / 96.0f, fy = y / 96.0f;
            img.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uint8_t>(110 + 90 * std::sin(2.0f * fy + 0.7f)),
                static_cast<uint8_t>(120 + 80 * std::cos(1.7f * fx)),
                static_cast<uint8_t>(100 + 95 * std::sin(1.3f * fx + 2.1f * fy)));
        }
    }
    const cv::Mat bayer = bayer_mosaic_rggb(img);
    for (int k = 0; k < static_cast<int>(DemosaicKernel::Count); ++k) {
        const cv::Mat back = demosaic(bayer, static_cast<DemosaicKernel>(k));
        CAPTURE(k);
        REQUIRE(mean_abs_diff(back, img) < 6.0);  // Malvar should be ~2; the
        // pre-fix double-swap measured ~40
    }
}

TEST_CASE("dose_for_strength matches the measured calibration", "[antidetect]") {
    const PhysicsDose lo = dose_for_strength(0.0f);
    const PhysicsDose mid = dose_for_strength(0.5f);
    const PhysicsDose hi = dose_for_strength(1.0f);
    REQUIRE(lo.noise_a == 0.0f);
    REQUIRE(lo.noise_b == 0.0f);
    REQUIRE(lo.bilateral_d == 0.0f);
    REQUIRE(lo.ca_px == 0.0f);
    REQUIRE(lo.vignette_k == 0.0f);
    // The M0 A/B measured vignette and CA as pure quality cost (no detector
    // effect at any dose), so the calibrated ladder keeps them at zero.
    REQUIRE(mid.ca_px == 0.0f);
    REQUIRE(mid.vignette_k == 0.0f);
    REQUIRE(hi.ca_px == 0.0f);
    REQUIRE(hi.vignette_k == 0.0f);
    // Sensor noise only ramps in at the top of the range (>= 0.75), where the
    // full dose measurably helps commfor; the default strengths stay clean.
    REQUIRE(mid.noise_a == 0.0f);
    REQUIRE(dose_for_strength(0.7f).noise_a == 0.0f);
    REQUIRE(dose_for_strength(0.8f).noise_a > 0.0f);
    REQUIRE(mid.noise_a < hi.noise_a);
    REQUIRE(mid.bilateral_d < hi.bilateral_d);
    // Midtone total sigma at strength 1.0 lands in the ISO 400-3200 band
    // (~6/255); the ladder is calibrated in the M0 doc.
    const double sigma_hi =
        std::sqrt(0.25 * hi.noise_a + hi.noise_b) * 255.0;
    REQUIRE(sigma_hi > 4.0);
    REQUIRE(sigma_hi < 8.0);
    REQUIRE(hi.jpeg_q_lo == 88);
    REQUIRE(hi.jpeg_q_hi == 96);
}

TEST_CASE("sensor noise measured sigma matches the dose on a flat patch",
          "[antidetect]") {
    cv::Mat flat = solid(cv::Vec3b(128, 128, 128), 128, 128);
    const cv::Mat ref = flat.clone();
    std::mt19937_64 rng(7);
    const float a = 2.1e-3f, b = 2.0e-5f;
    add_poisson_gaussian_linear(flat, a, b, rng);
    cv::Mat diff;
    cv::absdiff(flat, ref, diff);
    const double measured = cv::mean(cv::mean(diff))[0];
    // sigma_u8 at midtone ~ 6/255 (the sRGB slope ~1.07 at linear 0.216 nearly
    // cancels the x=0.25-vs-0.216 approximation), so mean|diff| ~ 6*0.8 = 4.8.
    REQUIRE(measured > 0.5);
    REQUIRE(measured < 6.5);
}

TEST_CASE("vignette leaves the center and dims the corners", "[antidetect]") {
    cv::Mat img = solid(cv::Vec3b(255, 255, 255), 100, 100);
    const cv::Mat ref = img.clone();
    apply_vignette(img, 0.3f);
    const auto c = img.at<cv::Vec3b>(50, 50);
    const auto corner = img.at<cv::Vec3b>(0, 0);
    REQUIRE(std::abs(c[0] - 255) <= 2);          // center untouched (r ~ 0.005)
    REQUIRE(std::abs(corner[0] - 178) <= 1);     // 255 * (1 - 0.3), r_norm = 1
    REQUIRE(img.size() == ref.size());
}

TEST_CASE("chromatic aberration shifts R and B in opposite directions", "[antidetect]") {
    // A vertical black/white edge: shifting R right and B left must move both
    // planes' edges while G stays put.
    cv::Mat img(64, 64, CV_8UC3, cv::Vec3b(0, 0, 0));
    img(cv::Rect(32, 0, 32, 64)).setTo(cv::Vec3b(255, 255, 255));
    const cv::Mat ref = img.clone();
    lateral_chromatic_aberration(img, 1.0f, 0.0f);
    REQUIRE(img.size() == ref.size());
    std::vector<cv::Mat> now, before;
    cv::split(img, now);
    cv::split(ref, before);
    cv::Mat dr, db, dg;
    cv::absdiff(now[2], before[2], dr);
    cv::absdiff(now[0], before[0], db);
    cv::absdiff(now[1], before[1], dg);
    REQUIRE(cv::mean(dr)[0] > 0.5);
    REQUIRE(cv::mean(db)[0] > 0.5);
    REQUIRE(cv::mean(dg)[0] == 0.0);
}

TEST_CASE("camera jpeg cycle is deterministic and mild at camera quality",
          "[antidetect]") {
    cv::Mat img = neutral_gradient(96, 96);
    const cv::Mat ref = img.clone();
    cv::Mat a = img.clone(), b = img.clone();
    camera_jpeg_cycle(a, 90);
    camera_jpeg_cycle(b, 90);
    REQUIRE(cv::sum(a != b) == cv::Scalar(0));  // fixed quality -> identical bytes
    REQUIRE(a.size() == ref.size());
    REQUIRE(mean_abs_diff(a, ref) < 8.0);       // q90 4:2:0 is visually clean
}

TEST_CASE("apply_physics is deterministic per seed and crops odd sizes",
          "[antidetect]") {
    const cv::Mat img = neutral_gradient(65, 97);  // deliberately odd
    cv::Mat a = img.clone(), b = img.clone(), c = img.clone();
    PhysicsConfig cfg;
    cfg.strength = 0.5f;
    std::mt19937_64 rng_a(42), rng_b(42), rng_c(43);
    const PhysicsStats sa = apply_physics(a, cfg, rng_a);
    const PhysicsStats sb = apply_physics(b, cfg, rng_b);
    const PhysicsStats sc = apply_physics(c, cfg, rng_c);
    REQUIRE(cv::sum(a != b) == cv::Scalar(0));       // same seed -> identical
    REQUIRE(cv::sum(a == c) != cv::Scalar(0, 0, 0)); // different seed -> different
    REQUIRE(a.cols == 96);                           // odd column cropped
    REQUIRE(a.rows == 64);                           // odd row cropped
    REQUIRE(sa.kernel == sb.kernel);
    REQUIRE(sa.jpeg_quality == sb.jpeg_quality);
    REQUIRE(sa.jpeg_quality >= 88);
    REQUIRE(sa.jpeg_quality <= 96);
    // The calibrated ladder keeps the default (0.5) noise-clean; noise only
    // ramps in at the top of the range.
    REQUIRE(sa.noise_sigma_midtone_255 == 0.0f);
    {
        cv::Mat d = img.clone();
        PhysicsConfig hot;
        hot.strength = 1.0f;
        std::mt19937_64 rng_d(42);
        const PhysicsStats sd = apply_physics(d, hot, rng_d);
        REQUIRE(sd.noise_sigma_midtone_255 > 0.0f);
    }
    // A strength-0 run is the identity apart from the always-on (not
    // dose-scaled) mosaic/demosaic round-trip, which is exact on neutral
    // content for every kernel in the bank.
    cv::Mat z = solid(cv::Vec3b(128, 128, 128), 32, 32);
    const cv::Mat zref = z.clone();
    std::mt19937_64 rng_z(1);
    PhysicsConfig zero;
    zero.strength = 0.0f;
    zero.jpeg_cycle = false;
    apply_physics(z, zero, rng_z);
    REQUIRE(mean_abs_diff(z, zref) < 1.0);
}
