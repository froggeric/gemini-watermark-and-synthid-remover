#include "detection/still_geometry.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace wmr {

std::optional<StillPreset> find_preset(const std::string& name) {
    for (const StillPreset& p : kStillPresets) {
        if (name == p.name) return p;
    }
    return std::nullopt;
}

WatermarkPosition rect_to_still_position(const cv::Rect& rect, int W, int H, int logo_size) {
    return {W - (rect.x + rect.width),
            H - (rect.y + rect.height),
            logo_size};
}

std::optional<StillPreset> snap_still_to_known(const cv::Rect& detected, int W, int H,
                                               int tol_px) {
    const int short_side = std::min(W, H);
    const cv::Point dc(detected.x + detected.width / 2,
                       detected.y + detected.height / 2);

    int best_dist = tol_px + 1;
    std::optional<StillPreset> best;
    for (const StillPreset& p : kStillPresets) {
        // Only presets calibrated for this resolution tier, and whose logo size
        // matches the detected size (so a future 96px hit cannot snap to a 36 slot).
        if (short_side < p.min_short || short_side > p.max_short) continue;
        if (p.logo_size != detected.width || p.logo_size != detected.height) continue;
        const cv::Point tl(W - p.margin_right - p.logo_size,
                           H - p.margin_bottom - p.logo_size);
        const cv::Point center(tl.x + p.logo_size / 2, tl.y + p.logo_size / 2);
        const int dist = std::abs(dc.x - center.x) + std::abs(dc.y - center.y);
        if (dist < best_dist) {
            best_dist = dist;
            best = p;
        }
    }
    return best;
}

namespace {
// Run detect_geometry_in_frames (pure, polarity-invariant) over a single window.
std::optional<StillGeometryHit> search_window(const cv::Mat& gray, const cv::Mat& alpha8u,
                                              const cv::Rect& window, float min_confidence) {
    const std::vector<cv::Mat> frames{gray};
    const std::vector<cv::Mat> templates{alpha8u};
    auto hit = detect_geometry_in_frames(frames, templates, window, min_confidence);
    if (!hit) return std::nullopt;
    return StillGeometryHit{hit->rect, hit->score};
}

// Trust a hit if it snaps to a known preset (trusted at the min confidence) or clears
// the high-confidence bar (a raw off-table hit). Returns true to accept, false to
// keep looking / fall back.
bool hit_is_trusted(const cv::Rect& rect, float score, int W, int H, float high_confidence) {
    const bool snapped = snap_still_to_known(rect, W, H).has_value();
    const auto verdict = decide_auto_geometry(snapped, score, high_confidence);
    return verdict != AutoGeometryVerdict::FallBack;
}
}  // namespace

std::optional<StillGeometryHit> locate_still_watermark_hybrid(
    const cv::Mat& gray_frame, const cv::Mat& alpha_template_8u,
    cv::Point model_anchor, int W, int H,
    float min_confidence, float high_confidence)
{
    if (gray_frame.empty() || alpha_template_8u.empty()) return std::nullopt;
    const int tw = alpha_template_8u.cols;
    const int th = alpha_template_8u.rows;
    const int pad = kStillAnchorPad;

    // (1) Anchored window around the model prediction.
    const cv::Rect anchored(std::max(0, model_anchor.x - pad),
                            std::max(0, model_anchor.y - pad),
                            tw + 2 * pad, th + 2 * pad);
    if (auto ah = search_window(gray_frame, alpha_template_8u, anchored, min_confidence)) {
        if (hit_is_trusted(ah->rect, ah->score, W, H, high_confidence)) return ah;
    }

    // (2) Widen to the bottom-right corner window (same as Gemini video).
    const int x0 = std::max(0, W - 320);
    const int y0 = std::max(0, H - 320);
    const cv::Rect corner(x0, y0, W - x0, H - y0);
    if (auto wh = search_window(gray_frame, alpha_template_8u, corner, min_confidence)) {
        if (hit_is_trusted(wh->rect, wh->score, W, H, high_confidence)) return wh;
    }

    return std::nullopt;
}

StillResolvedGeometry resolve_still_geometry(
    const cv::Mat& gray_frame, const cv::Mat& alpha_template_8u,
    const WatermarkPosition& model_pos, int W, int H,
    const StillGeometryOverride& override)
{
    // (1) Manual --rect wins outright.
    if (override.rect) {
        return {rect_to_still_position(*override.rect, W, H), "rect", 0.0f};
    }
    // (2) Named --geo-preset.
    if (override.preset) {
        if (auto p = find_preset(*override.preset)) {
            return {WatermarkPosition{p->margin_right, p->margin_bottom, p->logo_size},
                    "preset", 0.0f};
        }
        // Unknown name: fall through to auto/model (caller logs the miss).
    }
    // (3) Hybrid auto-detect.
    if (!override.no_auto_geometry) {
        const cv::Point anchor = model_pos.get_position(W, H);
        if (auto hit = locate_still_watermark_hybrid(gray_frame, alpha_template_8u,
                                                     anchor, W, H)) {
            const bool snapped = snap_still_to_known(hit->rect, W, H).has_value();
            const std::string src = snapped ? "auto/snapped" : "auto/raw";
            return {rect_to_still_position(hit->rect, W, H), src, hit->score};
        }
    }
    // (4) Model fallback.
    return {model_pos, "model", 0.0f};
}

} // namespace wmr
