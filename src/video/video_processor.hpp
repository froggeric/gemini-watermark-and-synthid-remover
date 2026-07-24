#pragma once

#include <string>
#include <optional>
#include <cstdint>
#include <chrono>
#include <opencv2/core.hpp>

#include "core/types.hpp"
#include "video/video_writer.hpp"
#include "video/notebooklm_detector.hpp"
#include "video/geometry_detector.hpp"

namespace wmr {

struct VideoWatermarkConfig {
    VideoProfile profile = VideoProfile::GeminiDiamond;
    VideoVariant variant = VideoVariant::Auto;
    bool force = false;
    float inpaint_strength = 0.85f;
    bool scenes = false;
    // Gemini/Veo diamond: after the reverse-blend, repair the faint border/halo on the
    // diamond's edge ring only (TELEA, residual-gated; interior/exterior untouched). ON by
    // default; --no-edge-cleanup disables for a pure reverse-blend. Skipped for VeoLegacy.
    bool edge_cleanup = true;
    std::optional<cv::Rect> rect;             // --rect x,y,w,h override (Gemini/Veo and NotebookLM)
    bool no_auto_geometry = false;            // --no-auto-geometry: skip content-based geometry search
    double scene_threshold = 0.4;
    // NotebookLM adaptive dispatch: uniform scenes -> NS, intricate scenes ->
    // MI-GAN (when WMR_AI_MIGAN; else NS). On Apple Silicon the default is
    // MI-GAN-everywhere (the ANE makes it fast); elsewhere the complexity gate
    // applies. --notebooklm-method {auto|ns|migan} overrides ("auto" = platform
    // default). NS is always the MI-GAN-unavailable fallback.
    double notebooklm_complexity_threshold = 15.0; // --complexity-threshold (intricate -> MI-GAN if score >= this)
    std::string notebooklm_method = "auto";        // --notebooklm-method {auto|ns|migan}
};

struct VideoResult {
    int64_t frames_processed = 0;
    int64_t frames_skipped = 0;
    float detection_confidence = 0.0f;
    double elapsed_seconds = 0.0;
    bool success = false;
    std::string message;
};

class VideoProcessor {
public:
    VideoResult process(const std::string& input_path,
                        const std::string& output_path,
                        const VideoWatermarkConfig& config,
                        const EncodeOptions& encode_opts = {});

private:
    static constexpr int kShotSampleCount = 12;
    static constexpr double kShotCoverage = 0.9;       // sample first 90% of video
    static constexpr int kPositionTolerancePx = 4;      // ±4px from shot anchor
    static constexpr float kOcclusionGateNcc = 0.35f;   // skip frames below this NCC
    static constexpr int kProgressIntervalFrames = 100;

    struct ShotDetection {
        bool found = false;
        cv::Point position;
        WatermarkSize size = WatermarkSize::Small;
        float confidence = 0.0f;
        cv::Rect region;
    };

    ShotDetection detect_in_shot(class VideoReader& reader,
                                 class WatermarkEngine& engine,
                                 const VideoWatermarkConfig& config,
                                 WatermarkPosition geo,
                                 int64_t range_start = -1,
                                 int64_t range_end = -1,
                                 int max_samples = kShotSampleCount);

    WatermarkPosition resolve_geometry(
        const VideoWatermarkConfig& config, int width, int height) const;

    WatermarkSize geometry_to_size(const WatermarkPosition& geo) const;

    // Alpha map + default anchor (top-left + bbox) for a resolved geometry. The
    // alpha is chosen through effective_alpha_size (the single >48/>68 gate); the
    // position/region use the alpha's real dims (handles non-square Veo), falling
    // back to the square logo_size when the asset is unavailable. The one helper
    // every video path calls instead of re-spelling the pick + position math.
    struct VideoAlphaAnchor {
        const cv::Mat* alpha = nullptr;   // always assigned by select_video_alpha
        cv::Point position{};             // always overwritten (def-init is defensive)
        cv::Rect region{};
    };
    VideoAlphaAnchor select_video_alpha(const class WatermarkEngine& engine,
                                        VideoProfile profile,
                                        const WatermarkPosition& geo,
                                        int width, int height) const;

    // Effective geometry honoring precedence:
    //   --rect > auto-detect > --variant > resolution guess.
    // Auto-detect samples frames via the reader, so it only runs for
    // VideoVariant::Auto when not --force / --no-auto-geometry. A snapped
    // (on-table) detection is always trusted; a raw off-table detection must
    // clear kAutoOverrideRawScore to override the resolution guess, so a
    // false positive cannot regress a video that works today.
    struct ResolvedGeometry {
        WatermarkPosition geo;
        std::string source;   // "rect" | "auto/snapped" | "auto/raw" | "force" | "variant" | "resolution"
        float score = 0.0f;   // auto-detect confidence (0 otherwise)
    };
    ResolvedGeometry resolve_effective_geometry(
        const VideoWatermarkConfig& config, int width, int height,
        class VideoReader& reader, class WatermarkEngine& engine);

    // Content-based geometry search: sample ~12 frames over the first 90%,
    // multi-template match (real alpha assets at native size) in the
    // bottom-right corner, then snap-to-known. Returns nullopt when fewer
    // than ~2 frames sample or no candidate clears the min confidence.
    // `score` receives the raw detection score on success.
    std::optional<SnappedGeometry> auto_detect_geometry(
        class VideoReader& reader, class WatermarkEngine& engine,
        const VideoWatermarkConfig& config, float& score);

    // NotebookLM path: per-scene adaptive dispatch (presence + complexity gates)
    VideoResult process_notebooklm(const std::string& input_path,
                                    const std::string& output_path,
                                    const VideoWatermarkConfig& config,
                                    const EncodeOptions& encode_opts);
};

} // namespace wmr
