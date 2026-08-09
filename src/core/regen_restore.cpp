#ifdef WMR_BUILD_REGEN
#include "core/regen_restore.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

// kissfft (vendored, BSD-3-Clause). Floating-point build: the inverse is NOT
// scaled by 1/N internally (C_FIXDIV is a no-op without FIXED_POINT), so we
// apply 1/N ourselves to match numpy.fft.ifft2. Validated bit-for-bit against
// numpy.fft.fft2/ifft2 on a 4x3 fixture (forward identical, inverse-with-1/N
// identical, round-trip exact to 1e-6). Row-major layout: dims = {H, W}.
#include "kiss_fft.h"
#include "kiss_fftnd.h"

namespace wmr {
namespace {

// ---------------------------------------------------------------------------
// KissFFT scope guards (C allocations, no exceptions on the hot path).
// ---------------------------------------------------------------------------
struct KissFFTndDeleter {
    void operator()(void* p) const noexcept { if (p) KISS_FFT_FREE(p); }
};
using KissFFTndHandle = std::unique_ptr<void, KissFFTndDeleter>;

KissFFTndHandle make_fftnd_cfg(const int dims[2], int inverse) {
    return KissFFTndHandle(kiss_fftnd_alloc(dims, 2, inverse, nullptr, nullptr));
}

// ---------------------------------------------------------------------------
// Radial frequency grid in cycle units, natural FFT order.
//   fy[i] = i if i <= h/2 else i - h        (numpy fftfreq(h)*h)
//   fx[j] = j if j <= w/2 else j - w
// Mirrors the Python `radial_freqs(h, w)`. Stored row-major (H x W) to match
// the kiss_fftnd layout.
// ---------------------------------------------------------------------------
cv::Mat radial_freqs(int h, int w) {
    cv::Mat r(h, w, CV_32FC1);
    for (int i = 0; i < h; ++i) {
        const float fy = (i <= h / 2) ? static_cast<float>(i)
                                       : static_cast<float>(i - h);
        float* row = r.ptr<float>(i);
        for (int j = 0; j < w; ++j) {
            const float fx = (j <= w / 2) ? static_cast<float>(j)
                                           : static_cast<float>(j - w);
            row[j] = std::sqrt(fy * fy + fx * fx);
        }
    }
    return r;
}

inline float clamp01(float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }

// Mirrors the Python dc_ramp / highfreq_rolloff exactly.
float dc_ramp(float r, float dc_radius) {
    return clamp01(r / dc_radius);
}
float highfreq_rolloff(float r, float cutoff) {
    // Python: clip((cutoff - r) / (cutoff * 0.5), 0, 1)
    return clamp01((cutoff - r) / (cutoff * 0.5f));
}

// ---------------------------------------------------------------------------
// Per-channel Wiener attenuation (Attenuator B). Mirrors the Python
// `wiener_capped(D, gamma=4.0)` (the validated "B3" variant, restore_study.py).
//
// The calibration (band mean, K, K_c) and the per-bin Wiener shrink math
// (carrier_mag/power, residual_power, alpha, shrink) are computed in double to
// match the Python reference's effective float64 behavior: in numpy, `0.5 *
// mag_D[band].mean()` promotes K to float64, which cascades through every
// downstream per-bin quantity. The 2D FFT itself stays float32 (kissfft);
// double there would buy <1e-6 and cost ~2x. Keeping the cheap scalar stages in
// double prevents float32 rounding in the calibration from biasing K (the
// prior is squared into carrier_power, so a small K error is amplified), which
// on large real diffs (band magnitudes ~1e5 summed over ~2000 bins) can move
// the mean attenuation by several/255 if accumulated in float.
//
// channel_float: CV_32FC1, the signed diff for one BGR channel.
// r:             CV_32FC1 radial freqs (cycle units, natural FFT order).
// Returns:       CV_32FC1 attenuated channel (real part of the inverse FFT).
// ---------------------------------------------------------------------------
cv::Mat attenuate_channel(const cv::Mat& channel_float, const cv::Mat& r,
                          const RestoreConfig& cfg, float channel_weight) {
    const int H = channel_float.rows;
    const int W = channel_float.cols;
    const int N = H * W;

    const int dims[2] = {H, W};
    auto cfg_fwd = make_fftnd_cfg(dims, /*inverse=*/0);
    auto cfg_inv = make_fftnd_cfg(dims, /*inverse=*/1);
    if (!cfg_fwd || !cfg_inv) {
        // Allocation failure: return the channel unchanged (graceful no-op).
        return channel_float.clone();
    }

    // Pack into kiss_fft_cpx (real input, imaginary = 0), row-major.
    std::vector<kiss_fft_cpx> in(N), F(N), att(N);
    for (int i = 0; i < H; ++i) {
        const float* row = channel_float.ptr<float>(i);
        kiss_fft_cpx* dst = in.data() + i * W;
        for (int j = 0; j < W; ++j) { dst[j].r = row[j]; dst[j].i = 0.0f; }
    }

    kiss_fftnd(reinterpret_cast<kiss_fftnd_cfg>(cfg_fwd.get()), in.data(), F.data());

    // Calibrate K so the carrier magnitude prior at r=target_r ~= 50% of the
    // measured mean |F_D| over r in [calib_low, calib_high]. Mirrors Python:
    //   mean_mag_D_r30 = mag_D[band].mean()
    //   K = 0.5 * mean_mag_D_r30 * (target_r ** 1.3)
    //   K_c = K * weight[c]
    // In double: the band sum runs over ~2000 bins of magnitude ~1e5, which in
    // float32 would accumulate multi-ULP error in the running sum.
    double band_sum = 0.0;
    long band_count = 0;
    for (int i = 0; i < H; ++i) {
        const float* rrow = r.ptr<float>(i);
        const kiss_fft_cpx* frow = F.data() + i * W;
        for (int j = 0; j < W; ++j) {
            if (rrow[j] >= cfg.calib_band_low && rrow[j] <= cfg.calib_band_high) {
                band_sum += std::sqrt(double(frow[j].r) * frow[j].r
                                       + double(frow[j].i) * frow[j].i);
                ++band_count;
            }
        }
    }
    const double mean_mag_band = (band_count > 0)
        ? (band_sum / static_cast<double>(band_count)) : 0.0;
    const double K = 0.5 * mean_mag_band * std::pow(static_cast<double>(cfg.target_r),
                                                     static_cast<double>(cfg.prior_exponent));
    const double K_c = K * static_cast<double>(channel_weight);

    // Apply the Wiener shrink in-place on F -> att. The per-bin math runs in
    // double to match the Python reference (K is float64 in numpy, which
    // promotes the whole shrink chain). The FFT bins (frow) stay float32
    // (kissfft output); we widen per-bin for the scalar math only.
    const double gamma = static_cast<double>(cfg.gamma);
    const double dc_radius = static_cast<double>(cfg.dc_radius);
    const double cutoff = static_cast<double>(cfg.highfreq_cutoff);
    const double prior_exp = static_cast<double>(cfg.prior_exponent);
    for (int i = 0; i < H; ++i) {
        const float* rrow = r.ptr<float>(i);
        const kiss_fft_cpx* frow = F.data() + i * W;
        kiss_fft_cpx* arow = att.data() + i * W;
        for (int j = 0; j < W; ++j) {
            // reliability (rel) stays float: the Python reference computes it
            // from the float32 radial grid, so float here matches numpy. The
            // downstream alpha/shrink chain is double (K promotes in numpy).
            const float rv = rrow[j];
            const float reliability = dc_ramp(rv, static_cast<float>(dc_radius))
                                    * highfreq_rolloff(rv, static_cast<float>(cutoff));
            const double carrier_mag = K_c * std::pow(static_cast<double>(rv) + 1e-9, -prior_exp);
            const double carrier_power = carrier_mag * carrier_mag;
            const double fmag2 = static_cast<double>(frow[j].r) * frow[j].r
                                 + static_cast<double>(frow[j].i) * frow[j].i;
            const double residual_power = std::max(fmag2 - carrier_power, 1e-9);
            const double alpha = carrier_power / (carrier_power + residual_power);
            // shrink = max(1 - gamma*alpha*reliability, 0)  (capped; never inverts)
            double shrink = 1.0 - gamma * alpha * static_cast<double>(reliability);
            if (shrink < 0.0) shrink = 0.0;
            const float sf = static_cast<float>(shrink);
            arow[j].r = frow[j].r * sf;
            arow[j].i = frow[j].i * sf;
        }
    }

    // Inverse + numpy-matching 1/N scaling.
    std::vector<kiss_fft_cpx> out(N);
    kiss_fftnd(reinterpret_cast<kiss_fftnd_cfg>(cfg_inv.get()), att.data(), out.data());
    const float inv_N = 1.0f / static_cast<float>(N);

    cv::Mat result(H, W, CV_32FC1);
    for (int i = 0; i < H; ++i) {
        float* row = result.ptr<float>(i);
        const kiss_fft_cpx* src = out.data() + i * W;
        for (int j = 0; j < W; ++j) { row[j] = src[j].r * inv_N; }
    }
    return result;
}

// Replicate numpy.percentile(a, 100*frac) with the default (linear) interpolation.
//   rank = frac * (N - 1);  lo = floor(rank);  hi = ceil(rank);  frac_between = rank - lo;
//   value = sorted[lo] + frac_between * (sorted[hi] - sorted[lo])
// `vals` is copied and sorted (the caller does not need the original order).
float percentile_linear(std::vector<float> vals, float frac) {
    if (vals.empty()) return 0.0f;
    std::sort(vals.begin(), vals.end());
    const float rank = frac * static_cast<float>(vals.size() - 1);
    const int lo = static_cast<int>(std::floor(rank));
    const int hi = static_cast<int>(std::ceil(rank));
    const float t = rank - static_cast<float>(lo);
    return vals[lo] + t * (vals[hi] - vals[lo]);
}

} // namespace

float mean_luminance_bgr(const cv::Mat& bgr_u8) {
    if (bgr_u8.empty()) return 0.0f;
    // Rec601 in BGR order: 0.114*B + 0.587*G + 0.299*R. Compute the per-pixel
    // luminance then average. cv::Mat::forEach over CV_8UC3 gives a Vec3b.
    double acc = 0.0;
    const int H = bgr_u8.rows, W = bgr_u8.cols;
    for (int i = 0; i < H; ++i) {
        const cv::Vec3b* p = bgr_u8.ptr<cv::Vec3b>(i);
        for (int j = 0; j < W; ++j) {
            const float b = p[j][0], g = p[j][1], r = p[j][2];
            acc += 0.114f * b + 0.587f * g + 0.299f * r;
        }
    }
    return static_cast<float>(acc / static_cast<double>(H * W));
}

cv::Mat restore_detail(const cv::Mat& O, const cv::Mat& R, const RestoreConfig& cfg) {
    // Graceful no-op on a size/type mismatch (regen's own fallback contract).
    if (O.empty() || R.empty() || O.size() != R.size()
        || O.type() != CV_8UC3 || R.type() != CV_8UC3) {
        return R.clone();
    }

    // Forced-Off: full regen, return R unchanged. This branch covers both the
    // CLI --no-regen-restore-detail override AND the library default (RestoreConfig
    // defaults to Off so a plain RegenConfig with no restore set is a clean no-op).
    if (cfg.mode == RestoreMode::Off) {
        spdlog::info("regen: detail-restoration off, full regen");
        return R.clone();
    }

    // Auto-gate: measure luminance of O; dim -> full regen (safe path).
    if (cfg.mode == RestoreMode::Auto) {
        const float luma = mean_luminance_bgr(O);
        if (luma < cfg.luminance_threshold) {
            spdlog::info("regen: detail-restoration skipped, full regen (luminance={:.1f} < {:.0f})",
                         luma, cfg.luminance_threshold);
            return R.clone();
        }
        // (the "applied" log is emitted below, after we know it ran)
    }

    // D = O - R  (signed BGR float, CV_32FC3).
    cv::Mat O_f, R_f, D;
    O.convertTo(O_f, CV_32FC3);
    R.convertTo(R_f, CV_32FC3);
    cv::subtract(O_f, R_f, D);

    // Keep mask: top keep_fraction of pixels by the combined L2 norm of the
    // ORIGINAL D (sqrt(dB^2 + dG^2 + dR^2)). Matches the Python:
    //   mag = sqrt((D**2).sum(axis=2));  thr = percentile(mag, 95);  mask = mag >= thr
    // i.e. the threshold is the (1 - keep_fraction) quantile (95th for top-5%),
    // and the mask keeps pixels whose |D| is >= that threshold.
    const int H = D.rows, W = D.cols;
    std::vector<float> mag_vec;
    mag_vec.reserve(static_cast<size_t>(H) * W);
    cv::Mat mag(H, W, CV_32FC1);
    for (int i = 0; i < H; ++i) {
        const cv::Vec3f* drow = D.ptr<cv::Vec3f>(i);
        float* mrow = mag.ptr<float>(i);
        for (int j = 0; j < W; ++j) {
            const float b = drow[j][0], g = drow[j][1], r = drow[j][2];
            const float m = std::sqrt(b * b + g * g + r * r);
            mrow[j] = m;
            mag_vec.push_back(m);
        }
    }
    const float thr = percentile_linear(std::move(mag_vec), 1.0f - cfg.keep_fraction);
    cv::Mat mask = (mag >= thr);   // CV_8UC1, 255 where kept, 0 elsewhere

    // Attenuator B: per-channel characterized-carrier Wiener on D.
    const cv::Mat r = radial_freqs(H, W);
    std::vector<cv::Mat> D_ch(3);
    cv::split(D, D_ch);
    const float weights[3] = {cfg.channel_weight_b,   // B (idx 0)
                              cfg.channel_weight_g,   // G (idx 1)
                              cfg.channel_weight_r};  // R (idx 2)
    std::vector<cv::Mat> D_att_ch(3);
    for (int c = 0; c < 3; ++c) {
        D_att_ch[c] = attenuate_channel(D_ch[c], r, cfg, weights[c]);
    }
    cv::Mat D_att;
    cv::merge(D_att_ch, D_att);   // CV_32FC3

    // R' = clip(R + mask * D_att, 0, 255). mask is {0,255}; rescale to {0,1}.
    cv::Mat mask_f;
    mask.convertTo(mask_f, CV_32FC1, 1.0 / 255.0);
    cv::Mat mask3;
    cv::merge(std::vector<cv::Mat>{mask_f, mask_f, mask_f}, mask3);

    cv::Mat Rp;
    cv::Mat contrib;
    cv::multiply(mask3, D_att, contrib);
    cv::add(R_f, contrib, Rp);
    cv::max(Rp, 0.0, Rp);
    cv::min(Rp, 255.0, Rp);

    cv::Mat out;
    Rp.convertTo(out, CV_8UC3);   // round-to-nearest via saturate_cast in convertTo

    // Transparency log. Names the override when forced; Auto reports the measured luma.
    const float luma = mean_luminance_bgr(O);
    const int keep_pct = static_cast<int>(std::lround(cfg.keep_fraction * 100.0f));
    if (cfg.mode == RestoreMode::On) {
        spdlog::info("regen: detail-restoration applied (--regen-restore-detail forced; "
                     "luminance={:.1f}, keep={}%)",
                     luma, keep_pct);
    } else {
        spdlog::info("regen: detail-restoration applied (luminance={:.1f}, keep={}%)",
                     luma, keep_pct);
    }
    return out;
}

} // namespace wmr
#endif
