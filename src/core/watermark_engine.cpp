#include "core/watermark_engine.hpp"
#include "core/blend_modes.hpp"
#include "core/inpaint.hpp"
#include "detection/ncc_detector.hpp"
#ifdef WMR_AI_DENOISE
#include "core/ai_denoise.hpp"
#endif

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace wmr {

// Correct an alpha map extracted from a non-black background capture.
// Raw alpha = max(R,G,B)/255 includes background brightness contamination:
//   alpha_raw = alpha_true + (bg/255) * (1 - alpha_true)
// True alpha is recovered by: alpha_true = (alpha_raw - bg/255) / (1 - bg/255)
// Background is estimated from border pixels of the capture (likely non-watermark).
static cv::Mat correct_alpha_for_background(const cv::Mat& alpha_raw, const cv::Mat& bg_capture) {
    if (alpha_raw.empty() || bg_capture.empty()) return alpha_raw;

    // Estimate background from border pixels of the capture
    cv::Mat gray;
    if (bg_capture.channels() >= 3) {
        std::vector<cv::Mat> channels;
        cv::split(bg_capture, channels);
        cv::max(channels[0], channels[1], gray);
        cv::max(gray, channels[2], gray);
    } else {
        gray = bg_capture.clone();
    }

    // Collect border pixel values (top/bottom rows, left/right cols)
    std::vector<float> border_vals;
    int h = gray.rows, w = gray.cols;
    for (int c = 0; c < w; ++c) {
        border_vals.push_back(gray.at<uchar>(0, c) / 255.0f);
        border_vals.push_back(gray.at<uchar>(h - 1, c) / 255.0f);
    }
    for (int r = 1; r < h - 1; ++r) {
        border_vals.push_back(gray.at<uchar>(r, 0) / 255.0f);
        border_vals.push_back(gray.at<uchar>(r, w - 1) / 255.0f);
    }

    // Use 25th percentile as background estimate (robust to watermark at edges)
    std::sort(border_vals.begin(), border_vals.end());
    float bg_norm = border_vals[border_vals.size() / 4];

    if (bg_norm < 0.01f) return alpha_raw.clone();  // dark enough, no correction needed

    float denom = 1.0f - bg_norm;
    if (denom < 0.01f) return alpha_raw.clone();

    cv::Mat corrected = (alpha_raw - bg_norm) / denom;
    cv::max(corrected, 0.0f, corrected);  // clamp negatives to 0

    spdlog::debug("Alpha correction: bg={:.3f}, raw max={:.3f}, corrected max={:.3f}",
                  bg_norm, alpha_raw.ptr<float>(0)[0] != 0 ? *std::max_element(alpha_raw.begin<float>(), alpha_raw.end<float>()) : 0,
                  *std::max_element(corrected.begin<float>(), corrected.end<float>()));

    return corrected;
}

WatermarkEngine::WatermarkEngine() {
    init_alpha_maps();
    detector_ = std::make_unique<NccDetector>(alpha_map_small_, alpha_map_large_, alpha_map_veo_text_);
}

WatermarkEngine::~WatermarkEngine() = default;

void WatermarkEngine::init_alpha_maps() {
    std::vector<unsigned char> buf_small(
        embedded::bg_48_png, embedded::bg_48_png + embedded::bg_48_png_size);
    std::vector<unsigned char> buf_large(
        embedded::bg_96_png, embedded::bg_96_png + embedded::bg_96_png_size);

    cv::Mat bg_small = cv::imdecode(buf_small, cv::IMREAD_COLOR);
    if (bg_small.empty()) {
        throw std::runtime_error("Failed to decode embedded 48x48 background capture");
    }

    cv::Mat bg_large = cv::imdecode(buf_large, cv::IMREAD_COLOR);
    if (bg_large.empty()) {
        throw std::runtime_error("Failed to decode embedded 96x96 background capture");
    }

    if (bg_small.cols != 48 || bg_small.rows != 48) {
        cv::resize(bg_small, bg_small, cv::Size(48, 48), 0, 0, cv::INTER_AREA);
    }
    if (bg_large.cols != 96 || bg_large.rows != 96) {
        cv::resize(bg_large, bg_large, cv::Size(96, 96), 0, 0, cv::INTER_AREA);
    }

    alpha_map_small_ = calculate_alpha_map(bg_small);
    alpha_map_large_ = calculate_alpha_map(bg_large);

    // Load Veo legacy text alpha map (30x22 "Veo" text)
    if (embedded::veo_text_png_size > 0) {
        std::vector<unsigned char> buf_veo(
            embedded::veo_text_png, embedded::veo_text_png + embedded::veo_text_png_size);
        cv::Mat bg_veo = cv::imdecode(buf_veo, cv::IMREAD_COLOR);
        if (!bg_veo.empty()) {
            alpha_map_veo_text_ = calculate_alpha_map(bg_veo);
        }
    }

    // Load V2 diamond alpha maps — upstream Gemini 3.5+ captures
    if (embedded::v2_diamond_36_png_size > 0) {
        std::vector<unsigned char> buf(
            embedded::v2_diamond_36_png, embedded::v2_diamond_36_png + embedded::v2_diamond_36_png_size);
        cv::Mat bg = cv::imdecode(buf, cv::IMREAD_COLOR);
        if (!bg.empty()) {
            alpha_map_v2_diamond_36_ = correct_alpha_for_background(calculate_alpha_map(bg), bg);
        }
    }
    if (embedded::v2_diamond_48_still_png_size > 0) {
        std::vector<unsigned char> buf(
            embedded::v2_diamond_48_still_png,
            embedded::v2_diamond_48_still_png + embedded::v2_diamond_48_still_png_size);
        cv::Mat bg = cv::imdecode(buf, cv::IMREAD_COLOR);
        if (!bg.empty()) {
            alpha_map_v2_diamond_48_still_ = correct_alpha_for_background(calculate_alpha_map(bg), bg);
        }
    }
    if (embedded::v2_diamond_96_png_size > 0) {
        std::vector<unsigned char> buf(
            embedded::v2_diamond_96_png, embedded::v2_diamond_96_png + embedded::v2_diamond_96_png_size);
        cv::Mat bg = cv::imdecode(buf, cv::IMREAD_COLOR);
        if (!bg.empty()) {
            alpha_map_v2_diamond_large_ = correct_alpha_for_background(calculate_alpha_map(bg), bg);
        }
    }

    has_v2_ = !alpha_map_v2_diamond_36_.empty() && !alpha_map_v2_diamond_large_.empty();

    // Load reference Veo text alpha maps (68x30 and 99x43) — with background correction
    if (embedded::veo_text_small_png_size > 0) {
        std::vector<unsigned char> buf(
            embedded::veo_text_small_png, embedded::veo_text_small_png + embedded::veo_text_small_png_size);
        cv::Mat bg = cv::imdecode(buf, cv::IMREAD_COLOR);
        if (!bg.empty()) {
            alpha_map_veo_text_small_ = correct_alpha_for_background(calculate_alpha_map(bg), bg);
        }
    }
    if (embedded::veo_text_large_png_size > 0) {
        std::vector<unsigned char> buf(
            embedded::veo_text_large_png, embedded::veo_text_large_png + embedded::veo_text_large_png_size);
        cv::Mat bg = cv::imdecode(buf, cv::IMREAD_COLOR);
        if (!bg.empty()) {
            alpha_map_veo_text_large_ = correct_alpha_for_background(calculate_alpha_map(bg), bg);
        }
    }

    spdlog::debug("Alpha maps initialized: small {}x{}, large {}x{}, veo_text {}x{}, "
                  "v2_diamond_36 {}x{}, v2_diamond_48_still {}x{}, v2_diamond_large {}x{}, "
                  "veo_text_ref_small {}x{}, veo_text_ref_large {}x{}",
                  alpha_map_small_.cols, alpha_map_small_.rows,
                  alpha_map_large_.cols, alpha_map_large_.rows,
                  alpha_map_veo_text_.cols, alpha_map_veo_text_.rows,
                  alpha_map_v2_diamond_36_.cols, alpha_map_v2_diamond_36_.rows,
                  alpha_map_v2_diamond_48_still_.cols, alpha_map_v2_diamond_48_still_.rows,
                  alpha_map_v2_diamond_large_.cols, alpha_map_v2_diamond_large_.rows,
                  alpha_map_veo_text_small_.cols, alpha_map_veo_text_small_.rows,
                  alpha_map_veo_text_large_.cols, alpha_map_veo_text_large_.rows);
}

DetectionResult WatermarkEngine::detect_watermark(
    const cv::Mat& image,
    std::optional<WatermarkSize> force_size,
    std::optional<WatermarkPosition> force_position,
    const cv::Mat* custom_alpha,
    std::optional<WatermarkVariant> force_variant,
    bool enable_snap) const
{
    const WatermarkVariant variant = force_variant.value_or(
        has_v2_ ? WatermarkVariant::V2 : WatermarkVariant::V1);
    const WatermarkSize size = force_size.value_or(
        get_watermark_size(image.cols, image.rows));

    // Resolve position from the variant unless the caller forced one (video).
    WatermarkPosition pos_cfg = force_position.value_or(
        get_watermark_config(image.cols, image.rows, variant));

    // Select alpha: caller-supplied custom_alpha wins; else V1 or V2 map.
    const cv::Mat* alpha = custom_alpha;
    cv::Mat v2_buf;  // owning holder when we resolve a V2 map
    if (!alpha) {
        if (variant == WatermarkVariant::V2 && has_v2_) {
            v2_buf = get_v2_alpha(size);
            alpha = &v2_buf;
        } else {
            alpha = &get_alpha_map(size);
        }
    }

    return detector_->detect(image, size, pos_cfg, alpha, enable_snap);
}

WatermarkEngine::StillResolveResult WatermarkEngine::resolve_still_geometry(
    const cv::Mat& image, WatermarkVariant variant, WatermarkSize size,
    const StillGeometryOverride& override) const
{
    const int W = image.cols, H = image.rows;

    // Pick the removal alpha for a resolved logo_size: 48px (Gemini 3.6) uses the still
    // capture; 36px (Gemini 3.5) uses the 36 capture. No legacy fallback: these are
    // constexpr-embedded PNGs that always decode in a correct build, so an empty Mat would
    // signal a build bug, not a runtime condition.
    auto alpha_for_logo = [&](int logo_size) -> const cv::Mat* {
        if (logo_size > 40) {
            return alpha_map_v2_diamond_48_still_.empty() ? nullptr
                                                          : &alpha_map_v2_diamond_48_still_;
        }
        return alpha_map_v2_diamond_36_.empty() ? nullptr : &alpha_map_v2_diamond_36_;
    };

    // (1) Manual --rect wins outright (any variant). logo_size = the rect's own width,
    // so the removal uses the matching-size alpha (a 48px box -> 48px alpha).
    if (override.rect) {
        WatermarkPosition p = rect_to_still_position(*override.rect, W, H, override.rect->width);
        return {p, alpha_for_logo(p.logo_size)};
    }
    // The content search only covers the V2 small path (the variant with a known model
    // error and content-matchable diamond). V1/V2-large keep the model.
    if (variant != WatermarkVariant::V2 || size != WatermarkSize::Small || !has_v2_) {
        return {std::nullopt, nullptr};
    }

    const WatermarkPosition model_pos = get_watermark_config(W, H, variant);

    // Candidate templates: BOTH small-diamond sizes (36 = Gemini 3.5, 48 = Gemini 3.6
    // still). The search reports which matched so removal uses the right-size alpha.
    cv::Mat gray;
    if (image.channels() >= 3) cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else                        gray = image.clone();
    std::vector<cv::Mat> templates;
    if (!alpha_map_v2_diamond_36_.empty()) {
        cv::Mat t; alpha_map_v2_diamond_36_.convertTo(t, CV_8U, 255.0); templates.push_back(t);
    }
    const cv::Mat& a48 = alpha_map_v2_diamond_48_still_;
    if (!a48.empty()) {
        cv::Mat t; a48.convertTo(t, CV_8U, 255.0); templates.push_back(t);
    }
    if (templates.empty()) return {std::nullopt, nullptr};

    // wmr::-qualified: resolves to the free function in still_geometry.cpp (this member
    // is in class scope; qualified lookup does not find it).
    const StillResolvedGeometry r = wmr::resolve_still_geometry(
        gray, templates, model_pos, W, H, override);
    spdlog::debug("Still geometry: source={}, score={:.2f}, margin=({},{}) logo_size={}",
                  r.source, r.score, r.pos.margin_right, r.pos.margin_bottom, r.pos.logo_size);
    if (r.source == "model") return {std::nullopt, nullptr};
    return {r.pos, alpha_for_logo(r.pos.logo_size)};
}

void WatermarkEngine::remove_watermark(cv::Mat& image,
                                        std::optional<WatermarkSize> force_size,
                                        std::optional<WatermarkVariant> force_variant) {
    if (image.empty()) {
        throw std::runtime_error("Empty image provided");
    }

    if (image.channels() == 4) {
        cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
    } else if (image.channels() == 1) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    }

    const WatermarkVariant variant = force_variant.value_or(
        has_v2_ ? WatermarkVariant::V2 : WatermarkVariant::V1);
    const WatermarkSize size = force_size.value_or(
        get_watermark_size(image.cols, image.rows));

    // Detect first so the V2-small snap refinement locks the exact pixels the
    // detector matched; removal then operates on the same region.
    const bool do_snap = (variant == WatermarkVariant::V2 && size == WatermarkSize::Small);
    const DetectionResult det = detect_watermark(
        image, size, std::nullopt, /*custom_alpha=*/nullptr, variant, do_snap);

    const cv::Mat& alpha = (variant == WatermarkVariant::V2 && has_v2_)
                               ? get_v2_alpha(size) : get_alpha_map(size);
    const cv::Point pos(det.region.x, det.region.y);

    spdlog::debug("Removing watermark at ({},{}) {}x{} (variant {})",
                  pos.x, pos.y, alpha.cols, alpha.rows,
                  variant == WatermarkVariant::V1 ? "V1" : "V2");

    remove_watermark_alpha_blend(image, alpha, pos, logo_value_);
}

bool WatermarkEngine::remove_watermark_detected(
    cv::Mat& image,
    const DetectionResult& detection,
    const InpaintConfig& cfg,
    const cv::Mat* custom_alpha)
{
    if (image.empty()) return false;

    if (image.channels() == 4) {
        cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
    } else if (image.channels() == 1) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    }

#ifdef WMR_BUILD_REGEN
    // SynthID diffusion-regen: SDXL img2img over the WHOLE image (regen replaces
    // content, so the visible-mark alpha-blend + residual cleanup below never run).
    // On any failure (no model/backend, regen error) this is an honest no-op: the
    // image is returned byte-for-byte unchanged and the CLI surfaces the failure to
    // a non-zero exit code. (The spectral path was removed in 1.16.0; regen is the
    // only SynthID operation.)
    if (cfg.method == InpaintMethod::DiffusionRegen) {
        RegenConfig rc;
        rc.strength = cfg.regen_strength;
        rc.steps = cfg.regen_steps;
        rc.tile = cfg.regen_tile;
        rc.allow_download = cfg.regen_allow_download;
        rc.model_path = cfg.regen_model_path;
        rc.vae_path   = cfg.regen_vae_path;   // empty -> resolve + download fp16-fix VAE
        rc.backend    = cfg.regen_backend;    // auto|cpu|metal|vulkan (auto->CPU on Apple Silicon)
        rc.restore    = cfg.regen_restore;    // post-regen detail restoration (Off/Auto/On)
        Regenerator& reg = regenerator();
        if (!reg.is_ready() && !reg.initialize(rc)) {
            spdlog::warn("SynthID regen unavailable (no model/backend); image unchanged.");
            return false;  // graceful: image left unchanged, CLI surfaces non-zero exit
        }
        if (!reg.regen(image, rc)) {
            spdlog::warn("SynthID regen failed on this image; image left unchanged.");
            return false;
        }
        return true;  // regen replaced the whole image; skip alpha-blend + residual cleanup
    }
#endif

    const cv::Mat& alpha_map = custom_alpha ? *custom_alpha : get_alpha_map(detection.size);
    cv::Point pos(detection.region.x, detection.region.y);

    spdlog::debug("Removing detected watermark at ({}, {}) conf={:.2f}",
                  pos.x, pos.y, detection.confidence);

    remove_watermark_alpha_blend(image, alpha_map, pos, logo_value_);

    // Residual cleanup. "--denoise off" is represented as the (non-AI) caller
    // skipping this call entirely; the engine never sees an "off" method here.
#ifdef WMR_AI_DENOISE
    if (cfg.method == InpaintMethod::AiDenoise) {
        NcnnDenoiser& d = denoiser();  // lazy process-wide singleton
        if (!d.is_ready()) d.initialize();
        if (d.is_ready()) {
            d.denoise(image, detection.region, alpha_map,
                      cfg.sigma, cfg.strength, cfg.padding);
            return true;  // AI did the cleanup; skip software inpaint
        }
        spdlog::warn("AI denoise unavailable - falling back to Gaussian");
        InpaintConfig fallback = cfg;
        fallback.method = InpaintMethod::Gaussian;
        inpaint_residual(image, detection.region, fallback, custom_alpha);
        return true;
    }
#endif
    inpaint_residual(image, detection.region, cfg, custom_alpha);
    return true;
}

void WatermarkEngine::remove_watermark_alpha_only(
    cv::Mat& image,
    const DetectionResult& detection,
    const cv::Mat* custom_alpha)
{
    if (image.empty()) return;

    if (image.channels() == 4) {
        cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
    } else if (image.channels() == 1) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    }

    const cv::Mat& alpha_map = custom_alpha ? *custom_alpha : get_alpha_map(detection.size);
    cv::Point pos(detection.region.x, detection.region.y);

    remove_watermark_alpha_blend(image, alpha_map, pos, logo_value_);
}

void WatermarkEngine::inpaint_residual(
    cv::Mat& image,
    const cv::Rect& region,
    const InpaintConfig& config,
    const cv::Mat* custom_alpha) const
{
    const cv::Mat& alpha_map = custom_alpha ? *custom_alpha : get_alpha_map(
        (region.width <= 48 && region.height <= 48)
            ? WatermarkSize::Small : WatermarkSize::Large);

    wmr::inpaint_residual(image, region, alpha_map, config);
}

void WatermarkEngine::add_watermark(cv::Mat& image,
                                     std::optional<WatermarkSize> force_size) {
    if (image.empty()) {
        throw std::runtime_error("Empty image provided");
    }

    if (image.channels() == 4) {
        cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
    } else if (image.channels() == 1) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    }

    WatermarkSize size = force_size.value_or(
        get_watermark_size(image.cols, image.rows));

    auto config = get_watermark_config(image.cols, image.rows);
    if (force_size.has_value()) {
        if (*force_size == WatermarkSize::Small) {
            config = {32, 32, 48};
        } else {
            config = {64, 64, 96};
        }
    }

    cv::Point pos = config.get_position(image.cols, image.rows);
    const cv::Mat& alpha_map = get_alpha_map(size);

    add_watermark_alpha_blend(image, alpha_map, pos, logo_value_);
}

const cv::Mat& WatermarkEngine::get_alpha_map(WatermarkSize size) const {
    return (size == WatermarkSize::Small) ? alpha_map_small_ : alpha_map_large_;
}

const cv::Mat& WatermarkEngine::get_v2_alpha(WatermarkSize size) const {
    return (size == WatermarkSize::Small) ? alpha_map_v2_diamond_36_
                                          : alpha_map_v2_diamond_large_;
}

cv::Mat WatermarkEngine::create_interpolated_alpha(int width, int height,
                                                     WatermarkSize size) const {
    const cv::Mat& source = get_alpha_map(size);

    if (width == source.cols && height == source.rows) {
        return source.clone();
    }

    cv::Mat interpolated;
    int method = (width > source.cols || height > source.rows)
                     ? cv::INTER_LINEAR
                     : cv::INTER_AREA;

    cv::resize(source, interpolated, cv::Size(width, height), 0, 0, method);
    return interpolated;
}

#ifdef WMR_AI_DENOISE
NcnnDenoiser& WatermarkEngine::denoiser() {
    // Intentionally leaked (never destroyed). ncnn's Vulkan weight allocators
    // reference the process-global GPU instance, which ncnn tears down at exit
    // in an order we don't control. Destroying the embedded ncnn::Net during
    // C++ static teardown can then dereference an already-freed Vulkan device
    // (EXC_BAD_ACCESS in VulkanDevice::vkdevice() at exit). The model +
    // ~few MB of runtime state are reclaimed by the OS at process exit, and the
    // actual image work is already complete by then — so we skip the destructor
    // rather than race the global GPU-instance teardown. Standard pattern for a
    // process singleton that owns a GL/Vulkan context.
    static NcnnDenoiser* instance = new NcnnDenoiser();  // process-wide; model loads on first use
    return *instance;
}
#endif

#ifdef WMR_BUILD_REGEN
Regenerator& WatermarkEngine::regenerator() {
    // Intentionally leaked: stable-diffusion's GGML backend races global device
    // teardown during static destruction (same hazard as ncnn/Vulkan). Never delete.
    // Magic-statics init (same pattern as denoiser() above): function-scope `static`
    // init is thread-safe in C++11+, closing the read-write race the prior
    // "if (!s) s = new ..." form had under concurrent first-use. regen() itself
    // remains NOT thread-safe (documented in the header).
    static Regenerator* s = new Regenerator();   // process-wide; model loads on first use
    return *s;
}
#endif

} // namespace wmr
