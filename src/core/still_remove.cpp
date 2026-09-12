#include "core/still_remove.hpp"

#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>
#include <cctype>
#include <sstream>
#include <vector>

#include "metadata/provenance.hpp"

namespace wmr {

std::optional<cv::Rect> parse_rect(const std::string& s) {
    // MOVED VERBATIM from cli_app.cpp:83-94 (format + non-negativity check;
    // per-image bounds are checked by callers that know the image size).
    if (s.empty()) return std::nullopt;
    int x, y, w, h;
    char sep1, sep2, sep3;
    std::istringstream ss(s);
    if (ss >> x >> sep1 >> y >> sep2 >> w >> sep3 >> h &&
        sep1 == ',' && sep2 == ',' && sep3 == ',' &&
        x >= 0 && y >= 0 && w > 0 && h > 0) {
        return cv::Rect(x, y, w, h);
    }
    return std::nullopt;
}

bool resolve_inpaint_config(const StillRemoveOptions& o, InpaintConfig& out) {
    // MOVED from cli_app.cpp:63-79, CliOptions -> StillRemoveOptions.
    out.strength = o.denoise_strength_pct / 100.0f;
    out.radius = o.denoise_radius;
    out.sigma = o.denoise_sigma;
    out.padding = 32;
    const std::string& m = o.denoise_method;
    if (m == "off")  return false;
    if (m == "soft") { out.method = InpaintMethod::Gaussian;      return true; }
    if (m == "ns")   { out.method = InpaintMethod::NavierStokes; return true; }
    if (m == "telea"){ out.method = InpaintMethod::Telea;         return true; }
#ifdef WMR_AI_DENOISE
    if (m == "ai")   { out.method = InpaintMethod::AiDenoise;     return true; }
#endif
    out.method = InpaintMethod::Gaussian;  // unreachable (validated input)
    return true;
}

bool write_still_output(const std::filesystem::path& path, const cv::Mat& image,
                        bool keep_provenance) {
    // MOVED from the identical tails cli_app.cpp:415-442 / batch_processor.cpp:189-217.
    // (Behavior note: the CLI single-image tail previously matched extensions
    // case-sensitively; an uppercase-extension -o now gets the explicit codec
    // params - batch parity.)
    auto parent = path.parent_path();
    if (!parent.empty() && !std::filesystem::exists(parent))
        std::filesystem::create_directories(parent);
    std::vector<int> params;
    std::string ext = path.extension().string();
    for (auto& c : ext) c = static_cast<char>(std::tolower(c));
    if (ext == ".jpg" || ext == ".jpeg")      params = {cv::IMWRITE_JPEG_QUALITY, 100};
    else if (ext == ".png")                   params = {cv::IMWRITE_PNG_COMPRESSION, 6};
    else if (ext == ".webp")                  params = {cv::IMWRITE_WEBP_QUALITY, 101};
    if (!cv::imwrite(path.string(), image, params)) return false;
    if (!keep_provenance) {
        auto pr = wmr::provenance::post_write_provenance_strip(path.string(),
                                                               /*keep_standard=*/true);
        if (pr.rewritten && pr.items_removed > 0)
            spdlog::info("Stripped {} provenance item(s) from output", pr.items_removed);
    }
    return true;
}

StillRemoveOutcome remove_still(WatermarkEngine& engine, cv::Mat& image,
                                const StillRemoveOptions& o) {
    // MOVED from cli_app.cpp:194-309 (the AutoRemove visible block), with the
    // bool-returning try_remove changed to return the successful DetectionResult
    // so the outcome can carry bbox/score. The dead `return 2` branch
    // (cli_app.cpp:302-307, always-true else-if) is NOT carried over.
    StillRemoveOutcome out;
    try {
        if (image.empty()) {
            // The engine is silent on an empty Mat (no throw, nothing detected);
            // the shared policy reports it as a failure, not "no watermark".
            out.outcome = StillOutcome::Failed;
            out.error = "empty image";
            return out;
        }
        if (o.force) {
            WatermarkVariant active = o.force_variant.value_or(WatermarkVariant::V2);
            spdlog::info("Force mode: removing visible watermark ({})",
                         active == WatermarkVariant::V1 ? "V1" : "V2");
            engine.remove_watermark(image, o.force_size, o.force_variant);
            out.outcome = StillOutcome::Removed;
            out.forced = true;
            out.variant = active == WatermarkVariant::V1 ? "V1" : "V2";
            return out;
        }

        const bool explicit_override =
            o.geometry.rect.has_value() || o.geometry.preset.has_value();
        const WatermarkSize sz =
            o.force_size.value_or(get_watermark_size(image.cols, image.rows));

        WatermarkEngine::StillResolveResult resolved;  // defaults: pos=nullopt, source="model"
        resolved = engine.resolve_still_geometry(image, WatermarkVariant::V2, sz,
                                                 o.geometry);

        auto try_remove = [&](WatermarkVariant v,
                              std::optional<WatermarkPosition> force_pos,
                              const cv::Mat* alpha_override)
            -> std::optional<DetectionResult> {
            const bool snap = !explicit_override && (
                force_pos.has_value() ||
                (v == WatermarkVariant::V2 && sz == WatermarkSize::Small));
            auto detection = engine.detect_watermark(image, o.force_size, force_pos,
                                                     alpha_override, v, snap);
            if (!detection.detected) {
                const bool vouched = force_pos.has_value() &&
                    (explicit_override ||
                     (resolved.trusted && detection.spatial_score >= 0.0f));
                if (!vouched) return std::nullopt;
                if (explicit_override) {
                    spdlog::info("Removing at overridden position "
                                 "(confidence {:.1f}% below the detection gate)",
                                 detection.confidence * 100.0f);
                } else {
                    spdlog::info("Removing at auto-detected position "
                                 "(fusion confidence {:.1f}% below the gate; "
                                 "geometry search: {}, NCC {:.2f})",
                                 detection.confidence * 100.0f,
                                 resolved.source, resolved.score);
                }
            } else {
                spdlog::info("Visible watermark detected ({:.1f}%, {}) at ({},{}) size={}, removing...",
                             detection.confidence * 100.0f,
                             v == WatermarkVariant::V1 ? "V1" : "V2",
                             detection.region.x, detection.region.y, detection.region.width);
            }
            const cv::Mat& alpha = alpha_override ? *alpha_override
                                                  : engine.get_still_alpha(detection.size, v);
            InpaintConfig icfg;
            if (resolve_inpaint_config(o, icfg)) {
                engine.remove_watermark_detected(image, detection, icfg, &alpha);
            } else {
                engine.remove_watermark_alpha_only(image, detection, &alpha);
            }
            out.variant = v == WatermarkVariant::V1 ? "V1" : "V2";
            return detection;
        };

        WatermarkVariant primary = o.force_variant.value_or(WatermarkVariant::V2);
        const bool is_v2 = (primary == WatermarkVariant::V2);
        auto hit = try_remove(primary, is_v2 ? resolved.pos : std::nullopt,
                              is_v2 ? resolved.alpha : nullptr);
        if (!hit && o.try_v1_fallback && primary == WatermarkVariant::V2) {
            spdlog::info("V2 profile not detected — retrying with legacy V1");
            hit = try_remove(WatermarkVariant::V1, std::nullopt, nullptr);
        }
        if (!hit) {
            spdlog::debug("No visible watermark detected");
            return out;  // NoWatermark
        }
        out.outcome = StillOutcome::Removed;
        out.bbox = hit->region;
        out.score = hit->confidence;
        out.geometry_source = resolved.source;
        return out;
    } catch (const std::exception& e) {
        out = StillRemoveOutcome{};
        out.outcome = StillOutcome::Failed;
        out.error = e.what();
        return out;
    } catch (...) {
        out = StillRemoveOutcome{};
        out.outcome = StillOutcome::Failed;
        out.error = "unknown error";
        return out;
    }
}

} // namespace wmr
