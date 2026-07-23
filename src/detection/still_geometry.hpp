#pragma once

// Content-based watermark geometry detection for the STILL-image path (the Gemini
// 3.5+/3.6 small 36px diamond). This is the still analog of video/geometry_detector,
// adapted for the single-image reality:
//
//   video uses a blind multi-frame corner scan, which works because it aggregates
//   ~12 frames for temporal consistency (the static mark wins over transient content).
//   A single busy still has no such advantage: a faint mark (NCC ~0.48) can be beaten
//   by corner content (~0.51-0.54) in a blind scan. So the still search is ANCHORED
//   on the model-predicted position first (the model is usually within ~20 px), then
//   widened to the corner only if the anchored hit is not trusted.
//
// Pure logic only: depends on OpenCV + core/types.hpp + video/geometry_detector.hpp
// (itself pure OpenCV). It must NOT pull FFmpeg, so it links into the test exe.
// Callers (WatermarkEngine) supply the alpha template + the grayscale frame.

#include <opencv2/core.hpp>

#include <optional>
#include <string>

#include "core/types.hpp"
#include "video/geometry_detector.hpp"  // detect_geometry_in_frames, decide_auto_geometry

namespace wmr {

// Min |NCC| for a candidate geometry to count as a detection (same as video/NotebookLM).
inline constexpr float kStillMinConfidence = 0.45f;
// A raw (off-table) detection must clear this to override the model, so a busy-corner
// false positive cannot regress an image that already works. Same value as video's
// kAutoOverrideRawScore and NccDetector's snap gate (0.60).
inline constexpr float kStillHighConfidence = 0.60f;
// Half-width of the anchored search window around the model-predicted top-left.
// Covers the observed model error (~16-20 px for Gemini 3.6 at 896x1200) with margin.
inline constexpr int kStillAnchorPad = 40;

struct StillGeometryHit {
    cv::Rect rect;        // detected bbox in full-frame coords (size == template size)
    float score = 0.0f;   // polarity-invariant |TM_CCOEFF_NORMED| in [0,1]
};

// A calibrated known watermark geometry for a resolution tier. The continuous model
// (v2_small_config_from_dims) is the fallback for uncalibrated resolutions; these
// entries pin exact measured geometries, and let a faint mark be trusted via the
// snap gate once its position is recognized as a known geometry.
struct StillPreset {
    const char* name;
    int min_short;        // short-side pixel range this preset covers
    int max_short;
    int margin_right;
    int margin_bottom;
    int logo_size;
};

// The calibrated table. First entry: Gemini 3.6 Flash portrait ~1k (896x1200),
// measured at margin ~100 (the model predicts 84 here). Grow as more fixtures arrive.
inline constexpr StillPreset kStillPresets[] = {
    { "gemini36-portrait", 800, 1000, 100, 100, 36 },
};

struct StillGeometryOverride {
    std::optional<cv::Rect> rect;       // --rect x,y,w,h
    std::optional<std::string> preset;  // --geo-preset <name>
    bool no_auto_geometry = false;      // --no-auto-geometry
};

struct StillResolvedGeometry {
    WatermarkPosition pos;
    std::string source;   // "rect" | "preset" | "auto/snapped" | "auto/raw" | "model"
    float score = 0.0f;
};

// Look up a named preset (exact name match). For --geo-preset.
std::optional<StillPreset> find_preset(const std::string& name);

// Snap a detected rect to the nearest calibrated preset whose short-side range
// matches the image and whose logo size matches the detected size, by center L1
// within tol_px. Returns nullopt when no preset matches.
std::optional<StillPreset> snap_still_to_known(const cv::Rect& detected, int W, int H,
                                               int tol_px = 40);

// Build a WatermarkPosition from a manual/detected rect. logo_size is fixed at 36 for
// the V2 small path (no nearest-width gate needed, unlike the video version).
WatermarkPosition rect_to_still_position(const cv::Rect& rect, int W, int H,
                                         int logo_size = 36);

// The hybrid search. (1) Anchored +/-kStillAnchorPad around model_anchor — the common
// case where the model is approximately right. If that hit is not trusted (neither a
// preset snap nor a high enough raw score), (2) widen to the bottom-right corner
// window (max(0,W-320) x max(0,H-320)) and gate the raw result. Returns nullopt when
// nothing clears the bar (caller falls back to the model). `alpha_template_8u` is the
// single 36px V2 diamond, CV_8UC1.
//
// NOTE: this match is polarity-invariant (max(|mx|,|mn|)), but NccDetector's stage-1
// spatial NCC is MAX-only, so a dark-on-bright mark localized here can still be
// rejected by the downstream fusion. Bright marks (the common case) confirm fine.
std::optional<StillGeometryHit> locate_still_watermark_hybrid(
    const cv::Mat& gray_frame,
    const cv::Mat& alpha_template_8u,
    cv::Point model_anchor,
    int W, int H,
    float min_confidence = kStillMinConfidence,
    float high_confidence = kStillHighConfidence);

// Pure precedence chokepoint: --rect > --geo-preset > hybrid auto-detect > model.
// Takes the already-loaded alpha template + the model position; the engine wraps it
// to supply the alpha + grayscale + model anchor.
StillResolvedGeometry resolve_still_geometry(
    const cv::Mat& gray_frame,
    const cv::Mat& alpha_template_8u,
    const WatermarkPosition& model_pos,
    int W, int H,
    const StillGeometryOverride& override);

} // namespace wmr
