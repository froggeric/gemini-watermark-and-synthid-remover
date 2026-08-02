#pragma once

#include <opencv2/core.hpp>
#include <optional>
#include <memory>

#include "core/types.hpp"
#include "core/inpaint.hpp"
#include "detection/still_geometry.hpp"  // StillGeometryOverride + resolve_still_geometry
#include "embedded_assets.hpp"
#ifdef WMR_AI_DENOISE
#include "core/ai_denoise.hpp"
#endif
#ifdef WMR_BUILD_REGEN
#include "core/regenerator.hpp"
#endif

namespace wmr {

class NccDetector;

class WatermarkEngine {
public:
    WatermarkEngine();
    ~WatermarkEngine();

    void remove_watermark(cv::Mat& image,
                          std::optional<WatermarkSize> force_size,
                          std::optional<WatermarkVariant> force_variant);

    // Legacy 2-arg path: variant resolved by the 3-arg overload
    // (has_v2_ ? V2 : V1). Kept for CLI --force backward compatibility.
    void remove_watermark(cv::Mat& image,
                          std::optional<WatermarkSize> force_size = std::nullopt) {
        remove_watermark(image, force_size, std::nullopt);
    }

    // Reverse alpha blend + residual cleanup using a full InpaintConfig (method,
    // strength, sigma, radius, padding). The CLI/batch path uses this overload
    // to select the cleanup method (AI / Gaussian / NS / Telea / off).
    void remove_watermark_detected(cv::Mat& image,
                                   const DetectionResult& detection,
                                   const InpaintConfig& cfg,
                                   const cv::Mat* custom_alpha = nullptr);

    // Convenience overload: builds an InpaintConfig from a strength fraction and
    // forwards (preserves existing callers — video path, tests). The cleanup
    // method defaults to AiDenoise when built (engine config default), else Gaussian.
    void remove_watermark_detected(cv::Mat& image,
                                   const DetectionResult& detection,
                                   float inpaint_strength = 0.85f,
                                   const cv::Mat* custom_alpha = nullptr) {
        InpaintConfig cfg;
        cfg.strength = inpaint_strength;
        remove_watermark_detected(image, detection, cfg, custom_alpha);
    }

    // Alpha blend only — no inpaint. Caller handles residual cleanup.
    void remove_watermark_alpha_only(cv::Mat& image,
                                     const DetectionResult& detection,
                                     const cv::Mat* custom_alpha = nullptr);

    // Variant-aware still-image detection. The 6-arg form resolves the profile
    // (default V2 when V2 assets are loaded, else V1). The inline 4-arg overload
    // preserves legacy V1 behavior for existing callers (video path, tests).
    DetectionResult detect_watermark(
        const cv::Mat& image,
        std::optional<WatermarkSize> force_size,
        std::optional<WatermarkPosition> force_position,
        const cv::Mat* custom_alpha,
        std::optional<WatermarkVariant> force_variant,
        bool enable_snap) const;

    DetectionResult detect_watermark(
        const cv::Mat& image,
        std::optional<WatermarkSize> force_size = std::nullopt,
        std::optional<WatermarkPosition> force_position = std::nullopt,
        const cv::Mat* custom_alpha = nullptr) const
    {
        // Legacy 4-arg path: V1, no snap. Used by video + existing tests.
        return detect_watermark(image, force_size, force_position, custom_alpha,
                                WatermarkVariant::V1, /*enable_snap=*/false);
    }

    // Result of the still-image geometry resolution: the position to force (nullopt =
    // use the model) and the matched-size removal alpha (36 or 48 px; nullptr = let the
    // detector pick the default). The alpha follows the resolved logo_size for ALL
    // paths (auto, --rect, --geo-preset), so a manual override removes with the correct
    // size, not the default 36px.
    struct StillResolveResult {
        std::optional<WatermarkPosition> pos;
        const cv::Mat* alpha = nullptr;
    };

    // Resolve the still-image watermark position via the hybrid auto-geometry search
    // (Gemini 3.5+/3.6 small diamond), honoring --rect / --geo-preset overrides. Runs
    // the content search only for V2 small; other variants return {nullopt, nullptr}
    // (use the model) unless an explicit --rect override is given. Returns pos=nullopt
    // when the result is the model position, so detect_watermark derives it itself (and
    // the V2->V1 fallback path is unaffected). Call this ONCE per image, before the
    // variant loop, and feed pos as detect_watermark's force_position + alpha as its
    // custom_alpha.
    StillResolveResult resolve_still_geometry(
        const cv::Mat& image,
        WatermarkVariant variant,
        WatermarkSize size,
        const StillGeometryOverride& override) const;

    void inpaint_residual(cv::Mat& image,
                          const cv::Rect& region,
                          const InpaintConfig& config = {},
                          const cv::Mat* custom_alpha = nullptr) const;

    void add_watermark(cv::Mat& image,
                       std::optional<WatermarkSize> force_size = std::nullopt);

    const cv::Mat& get_alpha_map(WatermarkSize size) const;
    const cv::Mat& get_veo_text_alpha() const { return alpha_map_veo_text_; }
    const cv::Mat& get_v2_diamond_alpha_36() const { return alpha_map_v2_diamond_36_; }
    const cv::Mat& get_v2_diamond_alpha_48_still() const { return alpha_map_v2_diamond_48_still_; }
    const cv::Mat& get_v2_diamond_alpha_large() const { return alpha_map_v2_diamond_large_; }
    const cv::Mat& get_veo_text_alpha_small() const { return alpha_map_veo_text_small_; }
    const cv::Mat& get_veo_text_alpha_large() const { return alpha_map_veo_text_large_; }

    // Resolve the still-image alpha for a variant: the V2 map when available,
    // else the V1 map. Public so the CLI detect→remove flow can forward it as
    // custom_alpha to the removers.
    const cv::Mat& get_still_alpha(WatermarkSize size, WatermarkVariant v) const {
        return (v == WatermarkVariant::V2 && has_v2_) ? get_v2_alpha(size)
                                                       : get_alpha_map(size);
    }

private:
    cv::Mat alpha_map_small_;
    cv::Mat alpha_map_large_;
    cv::Mat alpha_map_veo_text_;
    cv::Mat alpha_map_v2_diamond_36_;
    cv::Mat alpha_map_v2_diamond_48_still_;   // Gemini 3.6 48px (averaged still capture; still + video)
    cv::Mat alpha_map_v2_diamond_large_;
    cv::Mat alpha_map_veo_text_small_;
    cv::Mat alpha_map_veo_text_large_;
    float logo_value_ = 255.0f;
    bool has_v2_ = false;  // V2 (Gemini 3.5) alpha maps decoded

    std::unique_ptr<NccDetector> detector_;

    const cv::Mat& get_v2_alpha(WatermarkSize size) const;
    cv::Mat create_interpolated_alpha(int width, int height, WatermarkSize size) const;
#ifdef WMR_AI_DENOISE
    static NcnnDenoiser& denoiser();  // process-wide lazy singleton (model loads once)
#endif
#ifdef WMR_BUILD_REGEN
    static Regenerator& regenerator();  // intentionally-leaked process singleton
#endif
    void init_alpha_maps();
};

} // namespace wmr
