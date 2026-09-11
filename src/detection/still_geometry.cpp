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
// Replace saturated bright content (white text / poster highlights) with the local
// median background before the template search. The Gemini diamond is a white
// overlay, so where the underlying content is already near-white the blend is a
// no-op (0.3*255 + 0.7*255 == 255): those pixels carry zero mark signal but huge
// NCC noise. Measured on a Gemini 3.8 image whose mark sits across white poster
// text: the true position scores 0.39 raw (beaten by a text artifact at 0.45) but
// 0.78 after suppression, with the artifact down at 0.20.
//
// The two-condition gate can never erase the mark itself: a mark pixel only
// exceeds 200 when the background is above ~177 (alpha 0.30 white overlay), and
// there its deviation from background is 0.3*(255-bg) < 23, far below the 60
// outlier bar. Dark-polarity marks (the |min| arm) are untouched: only bright
// outliers are replaced. No-op on frames with no near-white content.
cv::Mat suppress_saturated_content(const cv::Mat& gray) {
    if (gray.empty() || gray.type() != CV_8UC1) return gray;
    cv::Mat med;
    cv::medianBlur(gray, med, 21);
    cv::Mat diff;
    cv::subtract(gray, med, diff, cv::noArray(), CV_16S);
    cv::Mat mask = (gray > 200) & (diff > 60);
    cv::Mat clean = gray.clone();
    med.copyTo(clean, mask);
    return clean;
}

// Run detect_geometry_in_frames (pure, polarity-invariant, multi-template) over one
// window. Returns the winning rect + score + which template matched.
std::optional<StillGeometryHit> search_window(const cv::Mat& gray,
                                              const std::vector<cv::Mat>& templates8u,
                                              const cv::Rect& window, float min_confidence) {
    const std::vector<cv::Mat> frames{gray};
    auto hit = detect_geometry_in_frames(frames, templates8u, window, min_confidence);
    if (!hit) return std::nullopt;
    return StillGeometryHit{hit->rect, hit->score, hit->template_index};
}

// Trust a hit if it snaps to a known preset (trusted at the min confidence) or clears
// the high-confidence bar (a raw off-table hit). Returns true to accept, false to
// keep looking / fall back.
bool hit_is_trusted(const cv::Rect& rect, float score, int W, int H, float high_confidence) {
    const bool snapped = snap_still_to_known(rect, W, H).has_value();
    const auto verdict = decide_auto_geometry(snapped, score, high_confidence);
    return verdict != AutoGeometryVerdict::FallBack;
}

// ---------------------------------------------------------------------------
// Pass 2: the visibility-weighted (carry-weighted) NCC.
//
// The mark is a white overlay: dev = alpha * (255 - bg), so on near-white content
// the bump is ~0 (a white overlay on white leaves nothing) and on dark content it
// is up to ~76/255. When MOST of the footprint sits on white content (posters with
// white emblems/text through the corner), the plain NCC is blind: measured 0.11-0.41
// at the true position on 4 Gemini 3.8 2K references. Weighting each pixel by its
// carry w = (255 - bg)/255 — correlating the template only against the background
// dark enough to show the mark — recovers them at 0.76-0.88, at the exact position.
//
// The score is the weighted, mean-centered cosine between the template and the
// saturated-content-suppressed gray (NOT a median-subtracted deviation map: a
// kernel comparable to the mark size eats the mark itself, measured 0.89 -> 0.44).
// All terms are cross-correlations (cheap): seven matchTemplate(CCORR) calls per
// template, three shared across templates.
//
// Rejected variants (measured on the same set, do not re-try):
//   - sigma-whitening (devide by local noise): the noise here is STRUCTURED
//     content, not per-pixel additive; whitening down-weights the wrong pixels
//     and loses the working cases (0.08-0.32 at the true position).
//   - binary carry mask: degenerate peaks (1.00) on tiny dark patches.
//   - bright-excursion capping min(gray, med+T): no effect on the buried set.
// ---------------------------------------------------------------------------
cv::Mat weighted_ncc_map(const cv::Mat& gray8u, const cv::Mat& tmpl32f) {
    CV_Assert(gray8u.type() == CV_8UC1 && tmpl32f.type() == CV_32F);

    cv::Mat med8u;
    cv::medianBlur(gray8u, med8u, kStillWeightedKernel);

    // Saturated-content suppression, same rule as pass 1 (against the k41 median).
    cv::Mat clean8u = gray8u.clone();
    {
        cv::Mat diff;
        cv::subtract(gray8u, med8u, diff, cv::noArray(), CV_16S);
        cv::Mat mask = (gray8u > 200) & (diff > 60);
        med8u.copyTo(clean8u, mask);
    }

    cv::Mat medf, cleanf;
    med8u.convertTo(medf, CV_32F);
    clean8u.convertTo(cleanf, CV_32F);

    // Carry weights: where the background is dark enough to show a white overlay.
    cv::Mat carry = 255.0f - medf;
    cv::max(carry, 0.0f, carry);
    carry *= 1.0f / 255.0f;

    const cv::Mat ones = cv::Mat::ones(tmpl32f.size(), CV_32F);
    cv::Mat tmpl2 = tmpl32f.mul(tmpl32f);

    cv::Mat w_clean = carry.mul(cleanf);
    cv::Mat w_clean2 = w_clean.mul(cleanf);

    cv::Mat Sw, Swx, Swx2, Swa, Swa2, Swax;
    cv::matchTemplate(carry, ones, Sw, cv::TM_CCORR);
    cv::matchTemplate(w_clean, ones, Swx, cv::TM_CCORR);
    cv::matchTemplate(w_clean2, ones, Swx2, cv::TM_CCORR);
    cv::matchTemplate(carry, tmpl32f, Swa, cv::TM_CCORR);
    cv::matchTemplate(carry, tmpl2, Swa2, cv::TM_CCORR);
    cv::matchTemplate(w_clean, tmpl32f, Swax, cv::TM_CCORR);

    // Degeneracy guard: too little carry mass (window on near-white content) is
    // not a valid measurement. Floor = fraction of the full footprint count.
    const float mass_floor = kStillWeightedMinMass *
                             static_cast<float>(tmpl32f.cols * tmpl32f.rows);
    const cv::Mat mass_ok = Sw >= mass_floor;  // 8U mask

    cv::max(Sw, 1e-6f, Sw);
    cv::Mat num = Swax - Swa.mul(1.0f / Sw).mul(Swx);

    cv::Mat den_a = Swa2 - Swa.mul(Swa).mul(1.0f / Sw);
    cv::Mat den_x = Swx2 - Swx.mul(Swx).mul(1.0f / Sw);
    cv::max(den_a, 1e-6f, den_a);
    cv::max(den_x, 1e-6f, den_x);
    cv::Mat den = den_a.mul(den_x);
    cv::sqrt(den, den);

    cv::Mat out;
    cv::divide(num, den, out);
    out.setTo(0.0f, ~mass_ok);  // invalidate degenerate windows
    return out;
}

// Verify a snapped hit AT THE PINNED POSITION before trusting it. The raw search
// peak can wander tens of px from the true mark on busy content; a content look-
// alike that lands inside the snap tolerance would then be pinned to the preset and
// trusted at the low snapped bar. Re-score at the exact preset top-left with both
// scorers (pass-1 suppressed NCC and the pass-2 weighted NCC) and require at least
// one to clear the min confidence. Measured discriminator: real marks score
// 0.478-0.88 (the buried ones only on the weighted score); a sparkle-content
// wanderer on a clean painting scored 0.055 at the pinned spot it was pinned to.
bool snap_position_verified(const cv::Mat& gray, const std::vector<cv::Mat>& templates8u,
                            int template_index, const cv::Point& tl, float min_confidence)
{
    if (template_index < 0 || template_index >= static_cast<int>(templates8u.size())) return false;
    const cv::Mat& t8u = templates8u[template_index];
    if (t8u.empty()) return false;
    cv::Mat tpl;
    t8u.convertTo(tpl, CV_32F, 1.0 / 255.0);
    const int S = t8u.cols;
    constexpr int kCtx = 30;  // context for the median/weight kernels
    const cv::Rect win = cv::Rect(tl.x - kCtx, tl.y - kCtx, S + 2 * kCtx, S + 2 * kCtx) &
                         cv::Rect(0, 0, gray.cols, gray.rows);
    if (win.width < S || win.height < S) return false;
    const cv::Mat region(gray, win);
    const cv::Point off(tl.x - win.x, tl.y - win.y);

    // Pass-1 score: suppressed NCC at the exact pinned offset.
    {
        cv::Mat med, clean = region.clone();
        cv::medianBlur(region, med, 21);
        cv::Mat diff;
        cv::subtract(region, med, diff, cv::noArray(), CV_16S);
        cv::Mat mask = (region > 200) & (diff > 60);
        med.copyTo(clean, mask);
        cv::Mat cleanf;
        clean.convertTo(cleanf, CV_32F);
        cv::Mat r;
        cv::matchTemplate(cleanf, tpl, r, cv::TM_CCOEFF_NORMED);
        if (r.at<float>(off) >= min_confidence) return true;
    }
    // Pass-2 score: carry-weighted NCC at the same offset.
    const cv::Mat w = weighted_ncc_map(region, tpl);
    return w.at<float>(off) >= min_confidence;
}

// Pass-2 search over one window: best (polarity-invariant) hit per template, then
// the first candidate (in score order) that clears the bar AND snaps to a preset.
// Snap is REQUIRED (the weighting amplifies content; only a calibrated position
// makes a weighted hit trustworthy). Returns nullopt when nothing qualifies.
std::optional<StillGeometryHit> search_window_weighted(
    const cv::Mat& gray, const std::vector<cv::Mat>& templates8u,
    const cv::Rect& window, int W, int H)
{
    // Clamp to the frame (a Mat view of an out-of-bounds rect throws).
    const cv::Rect win = window & cv::Rect(0, 0, gray.cols, gray.rows);
    if (win.width <= 0 || win.height <= 0) return std::nullopt;
    const cv::Mat region(gray, win);    struct Candidate { float score; cv::Rect rect; int ti; };
    std::vector<Candidate> cands;
    for (std::size_t ti = 0; ti < templates8u.size(); ++ti) {
        const cv::Mat& t8u = templates8u[ti];
        if (t8u.empty() || t8u.cols > region.cols || t8u.rows > region.rows) continue;
        cv::Mat tpl;
        t8u.convertTo(tpl, CV_32F, 1.0 / 255.0);
        const cv::Mat r = weighted_ncc_map(region, tpl);
        double mn, mx;
        cv::Point loc_mn, loc_mx;
        cv::minMaxLoc(r, &mn, &mx, &loc_mn, &loc_mx);
        const float score = static_cast<float>(std::max(std::fabs(mx), std::fabs(mn)));
        const cv::Point loc = (std::fabs(mx) >= std::fabs(mn)) ? loc_mx : loc_mn;
        cands.push_back({score, cv::Rect(window.x + loc.x, window.y + loc.y,
                                         t8u.cols, t8u.rows), static_cast<int>(ti)});
    }
    std::sort(cands.begin(), cands.end(),
              [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
    for (const Candidate& c : cands) {
        if (c.score < kStillWeightedMinConfidence) break;
        if (snap_still_to_known(c.rect, W, H).has_value()) {
            return StillGeometryHit{c.rect, c.score, c.ti};
        }
    }
    return std::nullopt;
}
}  // namespace

std::optional<StillGeometryHit> locate_still_watermark_hybrid(
    const cv::Mat& gray_frame, const std::vector<cv::Mat>& alpha_templates_8u,
    cv::Point model_anchor, int W, int H,
    float min_confidence, float high_confidence)
{
    if (gray_frame.empty() || alpha_templates_8u.empty()) return std::nullopt;
    int maxw = 0, maxh = 0;
    for (const cv::Mat& t : alpha_templates_8u) {
        if (t.empty()) continue;
        maxw = std::max(maxw, t.cols);
        maxh = std::max(maxh, t.rows);
    }
    if (maxw == 0 || maxh == 0) return std::nullopt;
    const int pad = kStillAnchorPad;

    // Search on the saturated-content-suppressed frame (see suppress_saturated_content).
    // The returned rect is in unchanged coordinates: suppression is per-pixel.
    const cv::Mat search_frame = suppress_saturated_content(gray_frame);

    // (1) Anchored window around the model prediction (sized for the largest template).
    const cv::Rect anchored(std::max(0, model_anchor.x - pad),
                            std::max(0, model_anchor.y - pad),
                            maxw + 2 * pad, maxh + 2 * pad);
    if (auto ah = search_window(search_frame, alpha_templates_8u, anchored, min_confidence)) {
        if (hit_is_trusted(ah->rect, ah->score, W, H, high_confidence)) return ah;
    }

    // (2) Widen to the bottom-right corner window (same as Gemini video).
    const int x0 = std::max(0, W - 320);
    const int y0 = std::max(0, H - 320);
    const cv::Rect corner(x0, y0, W - x0, H - y0);
    if (auto wh = search_window(search_frame, alpha_templates_8u, corner, min_confidence)) {
        if (hit_is_trusted(wh->rect, wh->score, W, H, high_confidence)) return wh;
    }

    // (3) Visibility-weighted pass (ONLY when nothing trusted so far): recovers
    // marks buried under near-white content, where the plain NCC is blind because
    // a white overlay leaves no signal there. Accepts only a preset-snapped hit
    // above kStillWeightedMinConfidence (see weighted_ncc_map for the reasoning
    // and the rejected alternatives).
    if (auto ah = search_window_weighted(gray_frame, alpha_templates_8u, anchored, W, H)) {
        return ah;
    }
    if (auto wh = search_window_weighted(gray_frame, alpha_templates_8u, corner, W, H)) {
        return wh;
    }

    return std::nullopt;
}

StillResolvedGeometry resolve_still_geometry(
    const cv::Mat& gray_frame, const std::vector<cv::Mat>& alpha_templates_8u,
    const WatermarkPosition& model_pos, int W, int H,
    const StillGeometryOverride& override)
{
    // (1) Manual --rect wins outright. logo_size = the rect's own width so the caller
    // picks the matching removal alpha (a 48px box -> 48px alpha).
    if (override.rect) {
        return {rect_to_still_position(*override.rect, W, H, override.rect->width),
                "rect", 0.0f, -1};
    }
    // (2) Named --geo-preset.
    if (override.preset) {
        if (auto p = find_preset(*override.preset)) {
            return {WatermarkPosition{p->margin_right, p->margin_bottom, p->logo_size},
                    "preset", 0.0f, -1};
        }
        // Unknown name: fall through to auto/model (caller logs the miss).
    }
    // (3) Hybrid auto-detect.
    if (!override.no_auto_geometry) {
        const cv::Point anchor = model_pos.get_position(W, H);
        if (auto hit = locate_still_watermark_hybrid(gray_frame, alpha_templates_8u,
                                                     anchor, W, H)) {
            // A snap does two things: trusts the hit at the min confidence AND pins
            // the returned position to the calibrated preset geometry. The raw NCC
            // peak can sit tens of px off on busy content (measured: a poster drew
            // the peak 35 px left of the calibrated mark on the same y-row), and the
            // preset margins are the measured true edges — so once the hit is
            // RECOGNIZED as a known geometry, the known geometry is the better
            // position (same semantics as the video path's snap_geometry_to_known).
            // The snap is only TRUSTED after re-scoring at the pinned position:
            // a content look-alike can wander inside the snap tolerance from real
            // content (measured: sparkle art on a clean painting, raw 0.54, pinned
            // spot scores 0.055). Verification failure falls back to the model.
            if (auto p = snap_still_to_known(hit->rect, W, H)) {
                const cv::Point tl(W - p->margin_right - p->logo_size,
                                   H - p->margin_bottom - p->logo_size);
                if (snap_position_verified(gray_frame, alpha_templates_8u,
                                           hit->template_index, tl,
                                           kStillMinConfidence)) {
                    return {WatermarkPosition{p->margin_right, p->margin_bottom, p->logo_size},
                            "auto/snapped", hit->score, hit->template_index};
                }
                // Snap rejected: the preset position holds no mark. The hit may
                // still stand at ITS OWN position if it is strong enough for the
                // raw bar (a genuine mark sitting off-preset, or a strong wander);
                // a weak wanderer (0.45-0.75) dies here.
            }
            // Off-table raw hit (>= high confidence): the detected rect IS the
            // position. logo_size = the matched template's width (36 or 48).
            const int logo = (hit->template_index >= 0 &&
                              hit->template_index < static_cast<int>(alpha_templates_8u.size()))
                                 ? alpha_templates_8u[hit->template_index].cols
                                 : model_pos.logo_size;
            return {rect_to_still_position(hit->rect, W, H, logo), "auto/raw",
                    hit->score, hit->template_index};
        }
    }
    // (4) Model fallback.
    return {model_pos, "model", 0.0f, -1};
}

} // namespace wmr
