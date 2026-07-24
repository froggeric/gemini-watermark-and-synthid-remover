#pragma once

#include <opencv2/core.hpp>

namespace wmr {

enum class InpaintMethod {
    Gaussian,
    Telea,
    NavierStokes,
    ShiftMap              // opencv_contrib xphoto INPAINT_SHIFTMAP (Phase B; usable only when WMR_HAS_XPHOTO)
#ifdef WMR_AI_DENOISE
    , AiDenoise  // FDnCNN NCNN/Vulkan AI denoise (dispatched by WatermarkEngine)
#endif
};

struct InpaintConfig {
    float strength = 0.85f;
    InpaintMethod method =
#ifdef WMR_AI_DENOISE
        InpaintMethod::AiDenoise;  // AI denoise is the default cleanup when built
#else
        InpaintMethod::Gaussian;
#endif
    int radius = 10;
    int padding = 32;
    bool full_mask = false;  // use full alpha region as inpaint mask (vs gradient edges)
    float sigma = 50.0f;     // FDnCNN sigma (1-150); unused by Gaussian/Telea/NS
};

void inpaint_residual(
    cv::Mat& image,
    const cv::Rect& region,
    const cv::Mat& alpha_map,
    const InpaintConfig& config = InpaintConfig{});

// Edge-only cleanup for the Gemini/Veo video diamond. The reverse-blend leaves a
// faint border/halo concentrated on the diamond's boundary (under-removal + H.264
// ringing). This repairs ONLY a thin ring around the footprint edge, leaving the
// recovered interior and the untouched exterior byte-for-byte intact (unlike
// inpaint_residual, which can touch the whole footprint). Tuned to the validated
// "U4" recipe: TELEA, residual-gated, binary blend at the defect pixels.
struct EdgeCleanupConfig {
    int ring_radius = 3;          // band = dilate(footprint, 2r+1) - erode(footprint, 2r+1)
    double residual_thresh = 14.0; // gate: repair edge px where max-ch |cur-ref| > this (0..255)
    int inpaint_radius = 3;       // cv::inpaint neighbourhood
    float strength = 1.0f;        // blend weight at gated pixels (1.0 = full replace)
};
void inpaint_diamond_edges(
    cv::Mat& image,
    const cv::Rect& region,
    const cv::Mat& alpha_map,
    const EdgeCleanupConfig& config = EdgeCleanupConfig{});

} // namespace wmr
