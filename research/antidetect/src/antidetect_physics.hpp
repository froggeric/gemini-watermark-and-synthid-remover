#pragma once

#include <opencv2/core.hpp>
#include <cstdint>
#include <random>

namespace wmr::antidetect {

// Stage A of the anti-detection pipeline: camera-statistics restoration.
//
// Passive AI-image detectors key on the ABSENCE of camera-pipeline statistics
// (CFA/demosaic inter-channel correlations, sensor noise, optics). This TU
// re-injects them with pure OpenCV ops, no models, no network:
//
//   bilateral pre-clean -> sRGB-to-linear -> Poisson-Gaussian sensor noise on
//   the linear values (pre-mosaic, so it passes through demosaicing exactly
//   like real sensor noise) -> Bayer RGGB mosaic -> demosaic with a per-image
//   RANDOMIZED kernel -> lateral chromatic aberration -> weak vignette ->
//   one camera-like JPEG cycle (4:2:0, q 88-96).
//
// The kernel (and CA direction, JPEG quality) are drawn per image from the
// caller's RNG: a single fixed demosaic kernel applied uniformly is 99.9%
// detectable by CFA-aware counter-forensics (Chen/Zhao/Stamm, ICIP 2017), so
// randomization is a design requirement, not a nicety.
//
// Pure TU (OpenCV only, no FFmpeg / no feature gates) so it links into the
// test binary directly (the notebooklm_gates rule). All randomness flows
// through the explicitly-passed std::mt19937_64 so --seed reproduces outputs
// byte-for-byte; cv::theRNG()/cv::randn are deliberately NOT used.

enum class DemosaicKernel {
    Malvar5x5,  // Malvar-He-Cutler gradient-corrected 5x5 linear interpolation
    Bilinear,   // phase-aware bilinear (3x3 horizontal/vertical/diagonal averaging)
    EdgeAware,  // OpenCV's edge-aware variational demosaicing
    // Count sentinel
    Count
};

// Dose ladder for a --strength in [0,1]. Single source of truth for every
// stage-A magnitude; the values are calibrated in
// docs/research/antidetect-m0-calibration.md (the M0 A/B harness) and
// hand-transcribed here. Literature anchors: total sensor noise 2-6/255 for
// an ISO 400-3200 phone photo, lateral CA 0.5-1 px, vignette 1-0.10..0.35*r^2,
// in-camera JPEG q~88-96 4:2:0.
struct PhysicsDose {
    float bilateral_d = 9.0f;   // bilateral pre-clean diameter in px (0 = skip)
    float noise_a = 0.0f;       // shot-noise gain:  sigma^2(x) = a*x + b, x in [0,1] linear
    float noise_b = 0.0f;       // read-noise floor: (sigma at midtone ~ sqrt(0.25a + b))
    float ca_px = 0.8f;         // lateral chromatic aberration shift magnitude (px)
    float vignette_k = 0.22f;   // corner attenuation: pixel *= 1 - k * r_norm^2
    int jpeg_q_lo = 88;         // camera JPEG quality drawn uniformly from [lo, hi]
    int jpeg_q_hi = 96;
};

PhysicsDose dose_for_strength(float strength);

struct PhysicsConfig {
    float strength = 0.5f;      // 0..1, scales every dose above
    bool jpeg_cycle = true;     // final camera-like JPEG re-encode (the default;
                                // --no-jpeg-cycle keeps the output lossless-domain)
};

// What was actually applied (the random draws), echoed by the stage-C report.
struct PhysicsStats {
    float bilateral_d = 0.0f;
    float noise_a = 0.0f;
    float noise_b = 0.0f;
    float noise_sigma_midtone_255 = 0.0f;   // derived, for the report
    DemosaicKernel kernel = DemosaicKernel::Malvar5x5;
    float ca_shift_px = 0.0f;               // signed magnitude actually drawn
    float vignette_k = 0.0f;
    int jpeg_quality = 0;                   // 0 = cycle skipped
};

// Apply the full physics pipeline in place. The RNG is consumed in a fixed
// order (kernel, CA direction/magnitude, JPEG quality, then the noise stream)
// so the same seed reproduces the same output. Odd-sized images are cropped by
// one row/column before mosaicing (Bayer needs even dims).
PhysicsStats apply_physics(cv::Mat& bgr_u8, const PhysicsConfig& cfg, std::mt19937_64& rng);

// --- Individual ops, exposed for unit tests and the M0 A/B harness mirror ---

// Mild edge-preserving smooth. The measured counter to removal-trace forensic
// classifiers (arXiv 2605.09203: bilateral is the strongest single blunter,
// TPR@0.1%FPR ~0.15 vs ~1.0 for everything else). diameter 0 = no-op.
void bilateral_preclean(cv::Mat& bgr_u8, float diameter);

// Physics-shaped sensor noise: Gaussian with per-pixel sigma(x) = sqrt(a*x + b)
// on the sRGB-LINEAR values (a Gaussian with lambda-dependent variance is the
// standard large-lambda approximation of Poisson photon shot noise, so this IS
// the Poisson-Gaussian model without the O(lambda) sampling cost). Applied
// pre-mosaic by apply_physics so the noise is interpolated by the demosaicer
// exactly like sensor noise would be.
void add_poisson_gaussian_linear(cv::Mat& bgr_u8, float a, float b, std::mt19937_64& rng);

// Sample onto a Bayer RGGB lattice: one channel per site, returned as a
// single-channel u8 Mat of the same size ((0,0)=R, (0,1)/(1,0)=G, (1,1)=B).
cv::Mat bayer_mosaic_rggb(const cv::Mat& bgr_u8);

// Reconstruct RGB from a Bayer RGGB mosaic with the chosen algorithm.
cv::Mat demosaic(const cv::Mat& bayer_u8, DemosaicKernel k);

// Sub-pixel lateral chromatic aberration: R and B planes shifted in opposite
// directions (a fixed per-image lens property, not per-pixel).
void lateral_chromatic_aberration(cv::Mat& bgr_u8, float r_dx, float r_dy);

// Radial falloff: multiply by (1 - k * r_norm^2), r_norm = 1 at the corners.
void apply_vignette(cv::Mat& bgr_u8, float k);

// One camera-like JPEG round-trip at the given quality (OpenCV writes 4:2:0
// chroma subsampling by default, matching phone/camera output).
void camera_jpeg_cycle(cv::Mat& bgr_u8, int quality);

} // namespace wmr::antidetect
