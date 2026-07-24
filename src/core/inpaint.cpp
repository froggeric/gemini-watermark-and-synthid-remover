#include "core/inpaint.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>

namespace wmr {

// Residual-only cleanup. The image arriving here is ALREADY reverse-alpha-blended
// (the mathematically exact reversal). This step only repairs pixels where that
// reversal left a visible artifact (a faint diamond ghost from an imperfect alpha or
// compression), instead of blurring the whole watermark footprint.
//
// Method: predict the clean content with cv::inpaint (Navier-Stokes or Telea) seeded
// from the surrounding pixels, compare it to the reverse-blended result, and blend
// the prediction in ONLY where the two differ beyond a threshold (the residual). Good
// reversals (reverse-blended ~= inpainted prediction) are left byte-for-byte intact,
// so this never blurs a clean removal.
void inpaint_residual(
    cv::Mat& image,
    const cv::Rect& region,
    const cv::Mat& alpha_map,
    const InpaintConfig& config)
{
    if (image.empty() || region.width < 4 || region.height < 4) return;

    const float strength = std::clamp(config.strength, 0.0f, 1.0f);
    if (strength < 0.001f) return;

    // Padded region for inpaint context.
    cv::Rect padded(
        region.x - config.padding,
        region.y - config.padding,
        region.width + config.padding * 2,
        region.height + config.padding * 2);
    padded &= cv::Rect(0, 0, image.cols, image.rows);
    if (padded.width < 8 || padded.height < 8) return;

    cv::Rect inner(
        region.x - padded.x,
        region.y - padded.y,
        region.width,
        region.height);
    inner &= cv::Rect(0, 0, padded.width, padded.height);

    // Watermark footprint mask (where the reverse-blend operated), dilated to cover
    // compression spread.
    cv::Mat alpha_resized;
    const int interp = (region.width > alpha_map.cols) ? cv::INTER_LINEAR : cv::INTER_AREA;
    cv::resize(alpha_map, alpha_resized, cv::Size(region.width, region.height), 0, 0, interp);

    cv::Mat footprint;
    cv::threshold(alpha_resized, footprint, 0.05f, 255.0f, cv::THRESH_BINARY);
    footprint.convertTo(footprint, CV_8U);
    if (cv::countNonZero(footprint) == 0) return;
    cv::Mat dk = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::dilate(footprint, footprint, dk);

    cv::Mat mask = cv::Mat::zeros(padded.size(), CV_8UC1);
    footprint.copyTo(mask(inner));

    // Predicted clean content: inpaint the footprint from the surrounding pixels.
    cv::Mat padded_area = image(padded).clone();
    const int cv_method = (config.method == InpaintMethod::Telea)
        ? cv::INPAINT_TELEA : cv::INPAINT_NS;  // Gaussian falls back to NS here
    cv::Mat reference;
    cv::inpaint(padded_area, mask, reference, config.radius, cv_method);

    // Residual = how far the reverse-blended image sits from the prediction, within
    // the footprint only.
    cv::Mat cur_f, ref_f;
    image(padded).convertTo(cur_f, CV_32FC3);
    reference.convertTo(ref_f, CV_32FC3);
    cv::Mat diff;
    cv::absdiff(cur_f, ref_f, diff);
    std::vector<cv::Mat> ch;
    cv::split(diff, ch);
    cv::Mat residual;
    cv::max(ch[0], ch[1], residual);
    cv::max(residual, ch[2], residual);  // max-channel residual, 0..255

    // Fix only pixels that deviate beyond the residual threshold (and are in the
    // footprint). The threshold is a small multiple of the local noise floor, so a
    // clean reversal (residual ~= inpaint error) is not touched.
    constexpr double kResidualThreshold = 16.0;  // 0..255
    cv::Mat fix_mask;
    cv::threshold(residual, fix_mask, kResidualThreshold, 1.0, cv::THRESH_BINARY);
    cv::Mat fix_mask_u8;
    fix_mask.convertTo(fix_mask_u8, CV_8U, 255.0);
    fix_mask_u8 &= mask;  // restrict to the footprint
    cv::Mat dk2 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::dilate(fix_mask_u8, fix_mask_u8, dk2);

    const int active = cv::countNonZero(fix_mask_u8);
    if (active == 0) {
        spdlog::debug("inpaint: residual-only, no residual pixels found (clean reversal, untouched)");
        return;
    }

    // Blend the prediction into the residual pixels (strength-weighted).
    cv::Mat weight;
    fix_mask_u8.convertTo(weight, CV_32F, 1.0 / 255.0);
    weight *= strength;
    cv::Mat weight_3ch;
    cv::merge(std::vector<cv::Mat>{weight, weight, weight}, weight_3ch);

    cv::Mat one_minus_w = cv::Scalar(1.0, 1.0, 1.0) - weight_3ch;
    cv::multiply(cur_f, one_minus_w, cur_f);
    cv::multiply(ref_f, weight_3ch, ref_f);
    cv::Mat result_f = cur_f + ref_f;

    cv::Mat dst = image(padded);
    result_f.convertTo(dst, CV_8UC3);

    const char* name = (config.method == InpaintMethod::Telea) ? "TELEA" : "NS";
    spdlog::debug("inpaint: residual-only {}, {} residual pixels repaired at {:.0f}%",
                  name, active, strength * 100.0);
}

// Edge-only repair for the video diamond. See inpaint.hpp for the rationale. This is
// the validated "U4" recipe: a TELEA inpaint confined to a thin boundary ring, gated to
// only the edge pixels where the reverse-blend left a real residual, then blended in.
// The recovered interior and the untouched exterior are never modified.
void inpaint_diamond_edges(
    cv::Mat& image,
    const cv::Rect& region,
    const cv::Mat& alpha_map,
    const EdgeCleanupConfig& config)
{
    if (image.empty() || region.width < 4 || region.height < 4) return;

    const int pad = 16;
    cv::Rect padded(region.x - pad, region.y - pad,
                    region.width + pad * 2, region.height + pad * 2);
    padded &= cv::Rect(0, 0, image.cols, image.rows);
    if (padded.width < 8 || padded.height < 8) return;

    cv::Rect inner(region.x - padded.x, region.y - padded.y,
                   region.width, region.height);
    inner &= cv::Rect(0, 0, padded.width, padded.height);
    if (inner.width < 4 || inner.height < 4) return;

    // Footprint (alpha > 0.05) resized to the region.
    cv::Mat alpha_resized;
    const int interp = (region.width > alpha_map.cols) ? cv::INTER_LINEAR : cv::INTER_AREA;
    cv::resize(alpha_map, alpha_resized, cv::Size(region.width, region.height), 0, 0, interp);
    cv::Mat foot;
    cv::threshold(alpha_resized, foot, 0.05f, 255.0f, cv::THRESH_BINARY);
    foot.convertTo(foot, CV_8U);
    if (cv::countNonZero(foot) == 0) return;

    // Edge band = dilate(footprint) - erode(footprint): a ring on the diamond boundary.
    const int ks = std::max(3, config.ring_radius * 2 + 1);
    cv::Mat elem = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ks, ks));
    cv::Mat dil, ero, band;
    cv::dilate(foot, dil, elem);
    cv::erode(foot, ero, elem);
    cv::subtract(dil, ero, band);

    cv::Mat mask = cv::Mat::zeros(padded.size(), CV_8UC1);
    band.copyTo(mask(inner));
    if (cv::countNonZero(mask) == 0) return;

    // Predict clean edge content with TELEA (propagates isophotes -> keeps shape/colour).
    cv::Mat padded_area = image(padded).clone();
    cv::Mat ref0;
    cv::inpaint(padded_area, mask, ref0, config.inpaint_radius, cv::INPAINT_TELEA);

    // Gate: keep only edge pixels where the reverse-blended image deviates from the
    // prediction (the residual / halo). Clean edges are left untouched.
    cv::Mat cur_f, ref_f;
    image(padded).convertTo(cur_f, CV_32FC3);
    ref0.convertTo(ref_f, CV_32FC3);
    cv::Mat diff;
    cv::absdiff(cur_f, ref_f, diff);
    std::vector<cv::Mat> ch;
    cv::split(diff, ch);
    cv::Mat resid;
    cv::max(ch[0], ch[1], resid);
    cv::max(resid, ch[2], resid);

    cv::Mat gate;
    cv::threshold(resid, gate, config.residual_thresh, 255.0, cv::THRESH_BINARY);
    gate.convertTo(gate, CV_8U);
    gate &= mask;
    const int active = cv::countNonZero(gate);
    if (active == 0) {
        spdlog::debug("inpaint: diamond-edges, no residual at the edge (untouched)");
        return;
    }

    // Re-inpaint the (smaller) gated mask for the final fill, then blend.
    cv::Mat ref;
    cv::inpaint(padded_area, gate, ref, config.inpaint_radius, cv::INPAINT_TELEA);
    ref.convertTo(ref_f, CV_32FC3);

    cv::Mat w;
    gate.convertTo(w, CV_32F, static_cast<double>(config.strength) / 255.0);
    cv::Mat w3;
    cv::merge(std::vector<cv::Mat>{w, w, w}, w3);
    cv::Mat one_minus_w = cv::Scalar(1.0, 1.0, 1.0) - w3;
    cv::multiply(cur_f, one_minus_w, cur_f);
    cv::multiply(ref_f, w3, ref_f);
    cv::Mat result_f = cur_f + ref_f;

    cv::Mat dst = image(padded);
    result_f.convertTo(dst, CV_8UC3);

    spdlog::debug("inpaint: diamond-edges, {} edge px repaired (TELEA, ring r={}, gated>{:.0f})",
                  active, config.ring_radius, config.residual_thresh);
}

} // namespace wmr
