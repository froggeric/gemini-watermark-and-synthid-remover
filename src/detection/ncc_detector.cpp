#include "detection/ncc_detector.hpp"

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>

namespace wmr {

NccDetector::NccDetector(const cv::Mat& alpha_small, const cv::Mat& alpha_large,
                         const cv::Mat& alpha_veo_text)
    : alpha_map_small_(alpha_small.clone()),
      alpha_map_large_(alpha_large.clone()),
      alpha_map_veo_text_(alpha_veo_text.empty() ? cv::Mat() : alpha_veo_text.clone()) {}

const cv::Mat& NccDetector::get_alpha_map(WatermarkSize size) const {
    return (size == WatermarkSize::Small) ? alpha_map_small_ : alpha_map_large_;
}

DetectionResult NccDetector::detect(
    const cv::Mat& image,
    std::optional<WatermarkSize> force_size,
    std::optional<WatermarkPosition> force_position,
    const cv::Mat* custom_alpha,
    bool enable_snap) const
{
    DetectionResult result;

    if (image.empty()) return result;

    const WatermarkSize size = force_size.value_or(
        get_watermark_size(image.cols, image.rows));

    // Use forced position (for video) or fall back to image defaults
    WatermarkPosition config = force_position.value_or(
        get_watermark_config(image.cols, image.rows));

    // Select alpha map: custom (e.g., Veo text) or standard square
    const cv::Mat& alpha_map = custom_alpha ? *custom_alpha : get_alpha_map(size);

    // Compute position — for non-square custom alpha, use actual dimensions
    cv::Point pos;
    if (custom_alpha) {
        pos = {image.cols - config.margin_right - custom_alpha->cols,
               image.rows - config.margin_bottom - custom_alpha->rows};
    } else {
        pos = config.get_position(image.cols, image.rows);
    }

    result.size = size;
    result.region = cv::Rect(pos.x, pos.y, alpha_map.cols, alpha_map.rows);

    // Snap mode (V2 small): widen the ROI by snap_pad so the full alpha template
    // can slide and absorb the ~1-3 px error in the aspect-aware position.
    const int snap_pad = enable_snap ? 3 : 0;

    // ROI clamped to image bounds (expanded by snap_pad when snapping)
    const int x1 = std::max(0, pos.x - snap_pad);
    const int y1 = std::max(0, pos.y - snap_pad);
    const int x2 = std::min(image.cols, pos.x + alpha_map.cols + snap_pad);
    const int y2 = std::min(image.rows, pos.y + alpha_map.rows + snap_pad);

    if (x1 >= x2 || y1 >= y2) {
        spdlog::debug("Detection: ROI out of bounds");
        return result;
    }

    // Extract region, convert to grayscale float [0,1]
    const cv::Rect image_roi(x1, y1, x2 - x1, y2 - y1);
    const cv::Mat region = image(image_roi);

    cv::Mat gray_region;
    if (region.channels() >= 3) {
        cv::cvtColor(region, gray_region, cv::COLOR_BGR2GRAY);
    } else {
        gray_region = region.clone();
    }

    cv::Mat gray_f;
    gray_region.convertTo(gray_f, CV_32F, 1.0 / 255.0);

    // Alpha region: when snapping, the FULL alpha template slides across the
    // wider gray region; otherwise a clamped sub-rect aligned to pos.
    cv::Mat alpha_region;
    if (enable_snap) {
        alpha_region = alpha_map;
    } else {
        const cv::Rect alpha_roi(x1 - pos.x, y1 - pos.y, x2 - x1, y2 - y1);
        alpha_region = alpha_map(alpha_roi);
    }

    // Stage 1: Spatial NCC
    cv::Mat spatial_match;
    cv::matchTemplate(gray_f, alpha_region, spatial_match, cv::TM_CCOEFF_NORMED);

    double spatial_score;
    cv::Point match_loc;
    cv::minMaxLoc(spatial_match, nullptr, &spatial_score, nullptr, &match_loc);
    result.spatial_score = static_cast<float>(spatial_score);

    // Trust the snapped offset only when correlation is strong; busy backgrounds
    // can otherwise pull the match toward content artefacts.
    if (enable_snap && spatial_score >= 0.60) {
        // Content-suppressed refinement. The raw NCC peak (match_loc) can straddle two
        // integer cells on busy content, and minMaxLoc then picks the wrong one: a 1px
        // error that a hard-edged diamond (a near step edge at ~0.30 alpha) turns into a
        // visible light/shadow emboss. Suppress slow content with a median filter and
        // re-localize the mark on the prominence (integer cell). Gemini places the mark
        // at an integer margin, so integer localization is exact here. A sub-pixel alpha
        // shift was tried and rejected: on busy content the averaged alpha template is
        // not an exact per-instance match, which biases the parabolic peak (~0.4px) and
        // the shift then reintroduces the emboss. Validated: exact on a clean black-bg
        // capture; correct integer on busy content.
        cv::Point snap_loc = match_loc;
        constexpr int kMedianKernel = 9;        // odd; radius = kernel/2 of median context
        const int mr = kMedianKernel / 2;
        const int mx1 = std::max(0, x1 - mr);
        const int my1 = std::max(0, y1 - mr);
        const int mx2 = std::min(image.cols, x2 + mr);
        const int my2 = std::min(image.rows, y2 + mr);
        if (mx2 - mx1 >= alpha_map.cols + 2 && my2 - my1 >= alpha_map.rows + 2) {
            cv::Mat med_region = image(cv::Rect(mx1, my1, mx2 - mx1, my2 - my1));
            cv::Mat med_gray;
            if (med_region.channels() >= 3)
                cv::cvtColor(med_region, med_gray, cv::COLOR_BGR2GRAY);
            else
                med_gray = med_region.clone();
            cv::Mat med_bg;
            cv::medianBlur(med_gray, med_bg, kMedianKernel);
            cv::Mat med_gray_f, med_bg_f;
            med_gray.convertTo(med_gray_f, CV_32F);
            med_bg.convertTo(med_bg_f, CV_32F);
            cv::Mat prominence = med_gray_f - med_bg_f;   // suppresses slow content

            // Search restricted to the original +/-snap_pad window (in med coords).
            const cv::Mat prom_search = prominence(cv::Rect(x1 - mx1, y1 - my1,
                                                            x2 - x1, y2 - y1));
            if (prom_search.cols >= alpha_map.cols && prom_search.rows >= alpha_map.rows) {
                cv::Mat hp_match;
                cv::matchTemplate(prom_search, alpha_map, hp_match, cv::TM_CCOEFF_NORMED);
                if (!hp_match.empty()) {
                    cv::Point hp_loc;
                    cv::minMaxLoc(hp_match, nullptr, nullptr, nullptr, &hp_loc);
                    snap_loc = cv::Point(x1 + hp_loc.x, y1 + hp_loc.y);
                }
            }
        }
        pos.x = snap_loc.x;
        pos.y = snap_loc.y;
        result.region = cv::Rect(pos.x, pos.y, alpha_map.cols, alpha_map.rows);
    }

    // Circuit breaker
    constexpr double kSpatialThreshold = 0.25;
    if (spatial_score < kSpatialThreshold) {
        spdlog::debug("Detection: spatial={:.3f} < {:.2f}, rejected",
                      spatial_score, kSpatialThreshold);
        result.confidence = static_cast<float>(spatial_score * 0.5);
        return result;
    }

    // Stage 2: Gradient NCC
    cv::Mat img_gx, img_gy, img_gmag;
    cv::Sobel(gray_f, img_gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray_f, img_gy, CV_32F, 0, 1, 3);
    cv::magnitude(img_gx, img_gy, img_gmag);

    cv::Mat alpha_gx, alpha_gy, alpha_gmag;
    cv::Sobel(alpha_region, alpha_gx, CV_32F, 1, 0, 3);
    cv::Sobel(alpha_region, alpha_gy, CV_32F, 0, 1, 3);
    cv::magnitude(alpha_gx, alpha_gy, alpha_gmag);

    cv::Mat grad_match;
    cv::matchTemplate(img_gmag, alpha_gmag, grad_match, cv::TM_CCOEFF_NORMED);

    double grad_score;
    cv::minMaxLoc(grad_match, nullptr, &grad_score);
    result.gradient_score = static_cast<float>(grad_score);

    // Stage 3: Variance analysis
    double var_score = 0.0;
    const int ref_h = std::min(y1, config.logo_size);

    if (ref_h > 8) {
        const cv::Rect ref_roi(x1, y1 - ref_h, x2 - x1, ref_h);
        const cv::Mat ref_region = image(ref_roi);
        cv::Mat gray_ref;
        if (ref_region.channels() >= 3) {
            cv::cvtColor(ref_region, gray_ref, cv::COLOR_BGR2GRAY);
        } else {
            gray_ref = ref_region;
        }

        cv::Scalar s_wm, s_ref;
        cv::meanStdDev(gray_region, cv::noArray(), s_wm);
        cv::meanStdDev(gray_ref, cv::noArray(), s_ref);

        if (s_ref[0] > 5.0) {
            var_score = std::clamp(1.0 - (s_wm[0] / s_ref[0]), 0.0, 1.0);
        }
    }
    result.variance_score = static_cast<float>(var_score);

    // Heuristic fusion
    const double confidence =
        (spatial_score * 0.50) +
        (grad_score * 0.30) +
        (var_score * 0.20);

    result.confidence = static_cast<float>(std::clamp(confidence, 0.0, 1.0));
    result.detected = (result.confidence >= 0.35f);

    spdlog::debug("Detection: spatial={:.3f}, grad={:.3f}, var={:.3f} "
                  "-> conf={:.3f} ({})",
                  spatial_score, grad_score, var_score, result.confidence,
                  result.detected ? "DETECTED" : "not detected");

    return result;
}

} // namespace wmr
