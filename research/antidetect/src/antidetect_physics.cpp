#include "core/antidetect_physics.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>

namespace wmr::antidetect {
namespace {

// sRGB <-> linear LUTs (u8 domain). The noise model lives in the linear domain
// because photon shot noise scales with the sensor value, not the gamma-encoded
// one.
const std::array<float, 256>& srgb_to_linear_lut() {
    static const std::array<float, 256> lut = [] {
        std::array<float, 256> l{};
        for (int v = 0; v < 256; ++v) {
            const float s = static_cast<float>(v) / 255.0f;
            l[v] = (s <= 0.04045f) ? s / 12.92f
                                   : std::pow((s + 0.055f) / 1.055f, 2.4f);
        }
        return l;
    }();
    return lut;
}

uint8_t linear_to_srgb_u8(float x) {
    x = std::clamp(x, 0.0f, 1.0f);
    const float s = (x <= 0.0031308f) ? 12.92f * x
                                       : 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
    return static_cast<uint8_t>(std::lround(std::clamp(s, 0.0f, 1.0f) * 255.0f));
}

// Odd (>=5) bilateral diameter, as cv::bilateralFilter expects.
int odd_diameter(float d) {
    int di = static_cast<int>(std::lround(d));
    if (di < 5) di = 5;
    if (di % 2 == 0) ++di;
    return di;
}

// Per-site-class masks for the RGGB lattice: 255 where the site class sits.
// Built by tiling a 2x2 unit with cv::repeat (cyclic tiling), so there is no
// per-pixel python-style loop and no strided cv::Range (which does not exist).
struct SiteMasks {
    cv::Mat r;   // (even row, even col)
    cv::Mat gr;  // (even row, odd col)
    cv::Mat gb;  // (odd row, even col)
    cv::Mat b;   // (odd row, odd col)
};

SiteMasks site_masks(int h, int w) {
    auto tile = [](int r0, int c0) {
        cv::Mat t = cv::Mat::zeros(2, 2, CV_8UC1);
        t.at<uint8_t>(r0, c0) = 255;
        return t;
    };
    SiteMasks m;
    cv::repeat(tile(0, 0), h / 2, w / 2, m.r);
    cv::repeat(tile(0, 1), h / 2, w / 2, m.gr);
    cv::repeat(tile(1, 0), h / 2, w / 2, m.gb);
    cv::repeat(tile(1, 1), h / 2, w / 2, m.b);
    return m;
}

// 1x3 / 3x1 / 3x3 bilinear interpolation kernels (phase-aware: which one applies
// depends on the site class, not the image). Values are the plain neighbor means.
cv::Mat kernel_horiz() { return (cv::Mat_<float>(1, 3) << 0.5f, 0.0f, 0.5f); }
cv::Mat kernel_vert()  { return (cv::Mat_<float>(3, 1) << 0.5f, 0.0f, 0.5f); }
cv::Mat kernel_diag() {
    return (cv::Mat_<float>(3, 3) << 0.25f, 0.0f, 0.25f,
                                     0.0f,  0.0f, 0.0f,
                                     0.25f, 0.0f, 0.25f);
}
cv::Mat kernel_cross() {  // 4-neighborhood mean (G at a chroma site)
    return (cv::Mat_<float>(3, 3) << 0.0f, 0.25f, 0.0f,
                                     0.25f, 0.0f, 0.25f,
                                     0.0f, 0.25f, 0.0f);
}

// Malvar-He-Cutler 5x5 gradient-corrected kernels (ICASSP 2004), applied to the
// raw mosaic (all sites mixed); each linear combination estimates one channel at
// one site class. All divided by 8 (DC gain 1).
cv::Mat mhc_g_at_chroma() {
    return (cv::Mat_<float>(5, 5) << 0,  0, -1,  0,  0,
                                     0,  0,  2,  0,  0,
                                    -1,  2,  4,  2, -1,
                                     0,  0,  2,  0,  0,
                                     0,  0, -1,  0,  0) / 8.0f;
}
cv::Mat mhc_chroma_at_g_row() {  // e.g. R at a G site inside a red row (R horizontal)
    return (cv::Mat_<float>(5, 5) << 0,    0,  0.5f, 0,  0,
                                     0,   -1,  0,   -1,  0,
                                    -1,    4,  5,    4, -1,
                                     0,   -1,  0,   -1,  0,
                                     0,    0,  0.5f, 0,  0) / 8.0f;
}
cv::Mat mhc_chroma_at_g_col() {  // transpose: chroma vertical at the other G phase
    cv::Mat k = mhc_chroma_at_g_row();
    cv::transpose(k, k);
    return k;
}
cv::Mat mhc_chroma_at_chroma() {  // e.g. R at a B site (pure diagonal estimation)
    return (cv::Mat_<float>(5, 5) << 0,     0, -1.5f, 0,  0,
                                     0,     2,  0,    2,  0,
                                    -1.5f,  0,  6,    0, -1.5f,
                                     0,     2,  0,    2,  0,
                                     0,     0, -1.5f, 0,  0) / 8.0f;
}

// Assemble one output channel plane from the mosaic: identity at the channel's
// own sites, a filtered estimate everywhere else. Phases: a red-site chroma
// (B) sees its chroma partner diagonally; a G site sees red horizontally in a
// red row (gr) and vertically in a blue row (gb) — mirrored for B.
cv::Mat assemble_plane(const cv::Mat& mosaic32, const cv::Mat& k_row, const cv::Mat& k_col,
                       const cv::Mat& k_diag, const SiteMasks& m, bool chroma_is_red) {
    cv::Mat plane = mosaic32.clone();  // identity; other sites overwritten below
    cv::Mat f_row, f_col, f_diag;
    cv::filter2D(mosaic32, f_row, CV_32F, k_row, cv::Point(-1, -1), 0,
                 cv::BORDER_REFLECT);
    cv::filter2D(mosaic32, f_col, CV_32F, k_col, cv::Point(-1, -1), 0,
                 cv::BORDER_REFLECT);
    cv::filter2D(mosaic32, f_diag, CV_32F, k_diag, cv::Point(-1, -1), 0,
                 cv::BORDER_REFLECT);
    if (chroma_is_red) {
        f_row.copyTo(plane, m.gr);
        f_col.copyTo(plane, m.gb);
        f_diag.copyTo(plane, m.b);
    } else {
        f_col.copyTo(plane, m.gr);
        f_row.copyTo(plane, m.gb);
        f_diag.copyTo(plane, m.r);
    }
    return plane;
}

} // namespace

PhysicsDose dose_for_strength(float strength) {
    // Calibrated 2026-09-01 against the M0 per-ingredient A/B (25 AI fixtures
    // x 3 local detectors; docs/research/antidetect-m0-calibration.md). The
    // JPEG cycle does almost all the detector work (corvi: all 12 flagged
    // images cleared by q96 ALONE at 0.7/255; commfor: 3/5 by q88 at
    // 1.3/255), the cheap mosaic kernels add corvi coverage (~1.9/255), and
    // the physically-heavy ops are measured dead weight: vignette clears
    // nothing at any dose while costing up to 14/255, CA nothing at 2-3/255,
    // low-dose noise makes commfor MORE suspicious (mean p_fake 0.22 -> 0.46
    // at the 0.25 dose), and the malvar kernel costs 2.8x edgeaware while
    // being the one kernel whose output corvi still flags. The default ladder
    // is therefore bilateral + randomized {edgeaware,bilinear} mosaic + JPEG
    // cycle (~2.2/255 ~= 41 dB PSNR on detailed content); sensor noise ramps
    // in only at the top of the range (>= 0.75), where the full dose
    // measurably helps commfor (3/5 cleared at 4.6/255). Vignette/CA stay at
    // 0 in the ladder (the ops remain available for experiments via
    // PhysicsConfig-style direct calls).
    const float s = std::clamp(strength, 0.0f, 1.0f);
    const bool active = s > 0.05f;  // strength 0 is a no-op apart from the
                                    // structural CFA round-trip (exact on
                                    // neutral content for every kernel).
    PhysicsDose d;
    d.bilateral_d = active ? 5.0f + 6.0f * s : 0.0f;
    const float noise_t = std::clamp((s - 0.75f) / 0.25f, 0.0f, 1.0f);
    d.noise_a = 2.1e-3f * noise_t;   // shot gain
    d.noise_b = 2.0e-5f * noise_t;   // read floor
    d.ca_px = 0.0f;
    d.vignette_k = 0.0f;
    d.jpeg_q_lo = 88;
    d.jpeg_q_hi = 96;
    return d;
}

void bilateral_preclean(cv::Mat& bgr_u8, float diameter) {
    if (diameter <= 0.0f) return;
    const int d = odd_diameter(diameter);
    cv::Mat out;
    // Mild settings: preserves edges, only flattens the finest correlations
    // (the removal-trace signal) without the cartoon look of a strong pass.
    cv::bilateralFilter(bgr_u8, out, d, 3.0f * static_cast<float>(d),
                        static_cast<float>(d) * 0.5f);
    bgr_u8 = out;
}

void add_poisson_gaussian_linear(cv::Mat& bgr_u8, float a, float b,
                                 std::mt19937_64& rng) {
    if (a <= 0.0f && b <= 0.0f) return;
    const auto& lut = srgb_to_linear_lut();
    std::normal_distribution<float> n01(0.0f, 1.0f);

    cv::Mat out(bgr_u8.rows, bgr_u8.cols, CV_8UC3);
    for (int y = 0; y < bgr_u8.rows; ++y) {
        const auto* src = bgr_u8.ptr<cv::Vec3b>(y);
        auto* dst = out.ptr<cv::Vec3b>(y);
        for (int x = 0; x < bgr_u8.cols; ++x) {
            for (int c = 0; c < 3; ++c) {
                const float lin = lut[src[x][c]];
                const float sigma = std::sqrt(a * lin + b);
                dst[x][c] = linear_to_srgb_u8(lin + sigma * n01(rng));
            }
        }
    }
    bgr_u8 = out;
}

cv::Mat bayer_mosaic_rggb(const cv::Mat& bgr_u8) {
    const int h = bgr_u8.rows, w = bgr_u8.cols;
    const SiteMasks m = site_masks(h, w);
    std::vector<cv::Mat> ch;
    cv::split(bgr_u8, ch);  // 0=B, 1=G, 2=R
    cv::Mat bayer = cv::Mat::zeros(h, w, CV_8UC1), tmp;
    // Masked ops leave non-mask entries of a PRE-EXISTING dst untouched, so tmp
    // must be zeroed between writes or stale channel values accumulate (the
    // u8 add then saturates and the mosaic is garbage).
    const auto masked = [&tmp](const cv::Mat& channel, const cv::Mat& mask) {
        tmp = 0;
        cv::bitwise_and(channel, channel, tmp, mask);
        return tmp;
    };
    bayer += masked(ch[2], m.r);   // R at (even row, even col)
    bayer += masked(ch[1], m.gr);  // G
    bayer += masked(ch[1], m.gb);  // G
    bayer += masked(ch[0], m.b);   // B at (odd, odd)
    return bayer;
}

cv::Mat demosaic(const cv::Mat& bayer_u8, DemosaicKernel k) {
    const int h = bayer_u8.rows, w = bayer_u8.cols;

    if (k == DemosaicKernel::EdgeAware) {
        // OpenCV's edge-aware demosaicing. Our mosaic has R at (0,0) -> B at
        // (1,1) and G at (1,2), which is OpenCV's "BG" naming (the code letters
        // describe the pixels of the SECOND row, second and third columns). The
        // solid-color unit test locks this convention.
        cv::Mat out;
        cv::cvtColor(bayer_u8, out, cv::COLOR_BayerBG2BGR_EA);
        return out;
    }

    const SiteMasks m = site_masks(h, w);
    cv::Mat mosaic32;
    bayer_u8.convertTo(mosaic32, CV_32F);

    cv::Mat r_plane, g_plane, b_plane;
    if (k == DemosaicKernel::Malvar5x5) {
        // Both planes pass (horizontal, vertical) kernel order; the phase
        // mirror lives in assemble_plane's chroma_is_red=false branch (blue at
        // a gr site is vertical, at a gb site horizontal). Swapping the kernel
        // args here TOO would cancel that mirror and mis-orient blue on the G
        // phases — invisible on neutral content (every kernel estimates the
        // same value), a ~10 dB PSNR crater on colorful content.
        r_plane = assemble_plane(mosaic32, mhc_chroma_at_g_row(),
                                 mhc_chroma_at_g_col(), mhc_chroma_at_chroma(), m,
                                 true);
        b_plane = assemble_plane(mosaic32, mhc_chroma_at_g_row(),
                                 mhc_chroma_at_g_col(), mhc_chroma_at_chroma(), m,
                                 false);
        g_plane = mosaic32.clone();
        cv::Mat fg;
        cv::filter2D(mosaic32, fg, CV_32F, mhc_g_at_chroma(), cv::Point(-1, -1), 0,
                     cv::BORDER_REFLECT);
        fg.copyTo(g_plane, m.r);
        fg.copyTo(g_plane, m.b);
    } else {  // Bilinear
        r_plane = assemble_plane(mosaic32, kernel_horiz(), kernel_vert(),
                                 kernel_diag(), m, true);
        b_plane = assemble_plane(mosaic32, kernel_horiz(), kernel_vert(),
                                 kernel_diag(), m, false);
        g_plane = mosaic32.clone();
        cv::Mat fg;
        cv::filter2D(mosaic32, fg, CV_32F, kernel_cross(), cv::Point(-1, -1), 0,
                     cv::BORDER_REFLECT);
        fg.copyTo(g_plane, m.r);
        fg.copyTo(g_plane, m.b);
    }

    cv::Mat out(h, w, CV_8UC3);
    for (int y = 0; y < h; ++y) {
        const float* pr = r_plane.ptr<float>(y);
        const float* pg = g_plane.ptr<float>(y);
        const float* pb = b_plane.ptr<float>(y);
        auto* dst = out.ptr<cv::Vec3b>(y);
        for (int x = 0; x < w; ++x) {
            dst[x] = cv::Vec3b(cv::saturate_cast<uint8_t>(pb[x]),
                               cv::saturate_cast<uint8_t>(pg[x]),
                               cv::saturate_cast<uint8_t>(pr[x]));
        }
    }
    return out;
}

void lateral_chromatic_aberration(cv::Mat& bgr_u8, float dx, float dy) {
    if (dx == 0.0f && dy == 0.0f) return;  // identity warps are pure cost
    // R and B shift in opposite sub-pixel directions (a fixed per-image lens
    // property). warpAffine inverse-maps, so +t moves content by +t.
    std::vector<cv::Mat> ch;
    cv::split(bgr_u8, ch);
    const cv::Mat m_r = (cv::Mat_<float>(2, 3) << 1, 0, dx, 0, 1, dy);
    const cv::Mat m_b = (cv::Mat_<float>(2, 3) << 1, 0, -dx, 0, 1, -dy);
    cv::Mat r_shift, b_shift;
    cv::warpAffine(ch[2], r_shift, m_r, ch[2].size(), cv::INTER_LINEAR,
                   cv::BORDER_REFLECT);
    cv::warpAffine(ch[0], b_shift, m_b, ch[0].size(), cv::INTER_LINEAR,
                   cv::BORDER_REFLECT);
    r_shift.copyTo(ch[2]);
    b_shift.copyTo(ch[0]);
    cv::merge(ch, bgr_u8);
}

void apply_vignette(cv::Mat& bgr_u8, float k) {
    if (k <= 0.0f) return;
    const int h = bgr_u8.rows, w = bgr_u8.cols;
    const float cx = (w - 1) * 0.5f, cy = (h - 1) * 0.5f;
    const float rmax = std::sqrt(cx * cx + cy * cy);
    cv::Mat out(bgr_u8.rows, bgr_u8.cols, CV_8UC3);
    for (int y = 0; y < h; ++y) {
        const auto* src = bgr_u8.ptr<cv::Vec3b>(y);
        auto* dst = out.ptr<cv::Vec3b>(y);
        for (int x = 0; x < w; ++x) {
            const float rx = (x - cx) / rmax, ry = (y - cy) / rmax;
            const float f = std::max(0.0f, 1.0f - k * (rx * rx + ry * ry));
            for (int c = 0; c < 3; ++c)
                dst[x][c] = static_cast<uint8_t>(std::lround(src[x][c] * f));
        }
    }
    bgr_u8 = out;
}

void camera_jpeg_cycle(cv::Mat& bgr_u8, int quality) {
    std::vector<uint8_t> buf;
    const std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, quality};
    if (!cv::imencode(".jpg", bgr_u8, buf, params)) return;
    cv::Mat back = cv::imdecode(buf, cv::IMREAD_COLOR);
    if (!back.empty()) bgr_u8 = back;
}

PhysicsStats apply_physics(cv::Mat& bgr_u8, const PhysicsConfig& cfg,
                           std::mt19937_64& rng) {
    const PhysicsDose dose = dose_for_strength(cfg.strength);
    PhysicsStats st{};

    // Fixed RNG-consumption order (kernel, CA, JPEG quality, then noise) so a
    // seed reproduces the output exactly. The randomized draw is restricted to
    // the measured-cheap kernels {edgeaware, bilinear}: malvar costs 2.8x
    // their perturbation and is the one kernel whose output corvi still flags
    // (M0 A/B; the enum + kernels stay available for direct/experimental use).
    std::uniform_int_distribution<int> which_kernel(0, 1);
    st.kernel = which_kernel(rng) == 0 ? DemosaicKernel::EdgeAware
                                       : DemosaicKernel::Bilinear;

    std::normal_distribution<float> ca_component(0.0f, dose.ca_px);
    const float dx = std::clamp(ca_component(rng), -2.0f, 2.0f);
    const float dy = std::clamp(ca_component(rng), -2.0f, 2.0f);
    st.ca_shift_px = std::sqrt(dx * dx + dy * dy);

    st.jpeg_quality = 0;
    if (cfg.jpeg_cycle) {
        std::uniform_int_distribution<int> q(dose.jpeg_q_lo, dose.jpeg_q_hi);
        st.jpeg_quality = q(rng);
    }

    st.bilateral_d = dose.bilateral_d;
    st.noise_a = dose.noise_a;
    st.noise_b = dose.noise_b;
    st.noise_sigma_midtone_255 =
        std::sqrt(0.25f * dose.noise_a + dose.noise_b) * 255.0f;
    st.vignette_k = dose.vignette_k;

    if (dose.bilateral_d > 0.0f) bilateral_preclean(bgr_u8, dose.bilateral_d);
    add_poisson_gaussian_linear(bgr_u8, dose.noise_a, dose.noise_b, rng);

    // Bayer needs even dims; a 1px crop is invisible at these sizes.
    if (bgr_u8.rows % 2 != 0 || bgr_u8.cols % 2 != 0)
        bgr_u8 = bgr_u8(cv::Rect(0, 0, bgr_u8.cols & ~1, bgr_u8.rows & ~1)).clone();

    const cv::Mat mosaic = bayer_mosaic_rggb(bgr_u8);
    bgr_u8 = demosaic(mosaic, st.kernel);

    lateral_chromatic_aberration(bgr_u8, dx, dy);
    apply_vignette(bgr_u8, dose.vignette_k);
    if (st.jpeg_quality > 0) camera_jpeg_cycle(bgr_u8, st.jpeg_quality);
    return st;
}

} // namespace wmr::antidetect
