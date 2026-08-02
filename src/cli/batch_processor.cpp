#include "cli/batch_processor.hpp"
#include "core/watermark_engine.hpp"
#include "core/types.hpp"
#include "core/fft_context.hpp"
#include "synthid/spectral_codebook.hpp"
#include "synthid/codebook_subtractor.hpp"
#include "synthid/noise_residual_subtractor.hpp"

#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;

namespace wmr {

static bool is_image_file(const fs::path& p) {
    auto ext = p.extension().string();
    for (auto& c : ext) c = static_cast<char>(std::tolower(c));
    return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".webp";
}

static std::vector<fs::path> collect_files(const std::string& dir, bool recursive) {
    std::vector<fs::path> files;

    if (recursive) {
        for (const auto& entry : fs::recursive_directory_iterator(dir)) {
            if (entry.is_regular_file() && is_image_file(entry.path())) {
                files.push_back(entry.path());
            }
        }
    } else {
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.is_regular_file() && is_image_file(entry.path())) {
                files.push_back(entry.path());
            }
        }
    }

    std::sort(files.begin(), files.end());
    return files;
}

static int process_single(const fs::path& input, const CliOptions& opts) {
    cv::Mat image = cv::imread(input.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
        spdlog::error("Failed to load: {}", input.filename().string());
        return 1;
    }

    WatermarkEngine engine;

    // Visible watermark processing
    if (opts.mode == CliMode::AutoRemove || opts.mode == CliMode::VisibleOnly) {
        std::optional<WatermarkSize> force_size;
        if (opts.force_small) force_size = WatermarkSize::Small;
        else if (opts.force_large) force_size = WatermarkSize::Large;

        auto [force_variant, try_fallback] = resolve_still_variant(opts);

        // Resolve the still geometry overrides + hybrid auto-search ONCE per image
        // (V2 small only). Mirrors process_single_image so batch and single-image
        // modes handle Gemini 3.6 identically (incl. the matched 48px alpha).
        WatermarkEngine::StillResolveResult resolved;
        if (!opts.force) {
            StillGeometryOverride gov;
            resolve_still_geometry_override(opts, gov);  // already validated in batch_process
            const WatermarkSize sz =
                force_size.value_or(get_watermark_size(image.cols, image.rows));
            resolved = engine.resolve_still_geometry(
                image, WatermarkVariant::V2, sz, gov);
        }

        // An explicit --rect/--geo-preset forces removal at that position even when
        // the detector's confidence is too low to confirm (faint mark). Mirrors the
        // single-image path.
        const bool explicit_override =
            !opts.still_rect_str.empty() || !opts.still_geo_preset.empty();

        auto try_remove = [&](WatermarkVariant v,
                              std::optional<WatermarkPosition> force_pos,
                              const cv::Mat* alpha_override) -> bool {
            bool snap = force_pos.has_value() ||
                (v == WatermarkVariant::V2 &&
                 force_size.value_or(get_watermark_size(image.cols, image.rows))
                     == WatermarkSize::Small);
            auto detection = engine.detect_watermark(image, force_size, force_pos,
                                                     alpha_override, v, snap);
            if (!detection.detected && !(explicit_override && force_pos.has_value())) {
                return false;
            }
            const cv::Mat& alpha = alpha_override ? *alpha_override
                                                  : engine.get_still_alpha(detection.size, v);
            InpaintConfig icfg;
            bool do_cleanup = resolve_inpaint_config(opts, icfg);
            if (do_cleanup) {
                engine.remove_watermark_detected(image, detection, icfg, &alpha);
            } else {
                engine.remove_watermark_alpha_only(image, detection, &alpha);
            }
            return true;
        };

        if (opts.force) {
            engine.remove_watermark(image, force_size, force_variant);
        } else {
            WatermarkVariant primary = force_variant.value_or(WatermarkVariant::V2);
            const bool is_v2 = (primary == WatermarkVariant::V2);
            if (!try_remove(primary, is_v2 ? resolved.pos : std::nullopt,
                            is_v2 ? resolved.alpha : nullptr) &&
                try_fallback && primary == WatermarkVariant::V2) {
                try_remove(WatermarkVariant::V1, std::nullopt, nullptr);
            }
        }
    }

    // SynthID processing. Mirrors process_single_image's synthid branch: regen runs
    // a whole-image SDXL scrub (skipping the spectral path), off skips entirely, and
    // spectral (default) is today's codebook/codebook-free path (byte-identical).
    if ((opts.mode == CliMode::AutoRemove && opts.synthid) || opts.mode == CliMode::SynthidOnly) {
        if (opts.synthid_attack == "regen") {
#ifdef WMR_BUILD_REGEN
            // regen = whole-image SDXL img2img. No codebook required. On failure the
            // engine returns false (logs + leaves the image unchanged). Mirror the
            // single-image path: surface the failure so the batch loop counts this
            // image as failed (not silently as success) and skip the misleading save.
            InpaintConfig ic;
            ic.method = InpaintMethod::DiffusionRegen;
            ic.regen_strength       = opts.regen_strength;
            ic.regen_steps          = opts.regen_steps;
            ic.regen_tile           = !opts.regen_no_tile;
            ic.regen_allow_download = !opts.regen_no_download;
            ic.regen_model_path     = opts.regen_model_path;
            ic.regen_vae_path       = opts.regen_vae_path;
            DetectionResult dr{};  // regen ignores visible-mark detection
            bool ok = engine.remove_watermark_detected(image, dr, ic);
            if (!ok) {
                spdlog::warn("  SynthID regen did not complete (no model/backend, or it failed); output unchanged.");
                return 1;
            }
#else
            spdlog::error("--synthid-attack regen: this wmr build is regen-free (WMR_BUILD_REGEN off). "
                          "Rebuild with WMR_BUILD_REGEN=1, or use --synthid-attack spectral.");
            return 1;
#endif
        } else if (opts.synthid_attack == "off") {
            spdlog::info("SynthID attack is 'off'; skipping the synthid pass");
        } else {
            RemovalConfig config;
            config.custom_strength = opts.synthid_strength;
            config.phase_adaptive = opts.phase_adaptive;
            config.lab_a = opts.lab_a;

            if (!opts.codebook_path.empty()) {
                FftContext fft;
                SpectralCodebook codebook;
                codebook.load(opts.codebook_path);
                CodebookSubtractor subtractor(fft);
                subtractor.remove_synthid(image, codebook, config);
            } else if (opts.codebook_free) {
                FftContext fft;
                NoiseResidualSubtractor subtractor(fft);
                subtractor.remove_synthid(image, config);
            }
        }
    }

    // Determine output path
    fs::path out_path;
    if (!opts.output_path.empty()) {
        fs::path out_dir(opts.output_path);
        auto rel = fs::relative(input, opts.input_path);
        out_path = out_dir / rel;
        if (!out_path.has_extension()) {
            out_path /= input.filename();
        }
    } else {
        auto rel = fs::relative(input, opts.input_path);
        out_path = fs::path(opts.input_path) / "cleaned" / rel;
    }

    if (!out_path.parent_path().empty() && !fs::exists(out_path.parent_path())) {
        fs::create_directories(out_path.parent_path());
    }

    std::vector<int> params;
    std::string ext = out_path.extension().string();
    for (auto& c : ext) c = static_cast<char>(std::tolower(c));
    if (ext == ".jpg" || ext == ".jpeg") {
        params = {cv::IMWRITE_JPEG_QUALITY, 100};
    } else if (ext == ".png") {
        params = {cv::IMWRITE_PNG_COMPRESSION, 6};
    } else if (ext == ".webp") {
        params = {cv::IMWRITE_WEBP_QUALITY, 101};
    }

    if (!cv::imwrite(out_path.string(), image, params)) {
        spdlog::error("Failed to save: {}", out_path.string());
        return 1;
    }

    spdlog::info("  → {}", out_path.string());
    return 0;
}

BatchResult batch_process(const CliOptions& opts, const ProgressCallback& progress) {
    BatchResult result;

    auto files = collect_files(opts.input_path, opts.recursive);
    result.total = static_cast<int>(files.size());

    if (files.empty()) {
        spdlog::warn("No image files found in: {}", opts.input_path);
        return result;
    }

    // Validate the still geometry override once (--rect), so a malformed value fails
    // the whole batch with a single error rather than one per file.
    if (opts.mode == CliMode::AutoRemove || opts.mode == CliMode::VisibleOnly) {
        StillGeometryOverride gov;
        if (!resolve_still_geometry_override(opts, gov)) {
            return result;  // error already logged
        }
    }

    spdlog::info("Processing {} images...", result.total);

    for (int i = 0; i < result.total; ++i) {
        const auto& file = files[i];
        spdlog::info("[{}/{}] {}", i + 1, result.total, file.filename().string());

        if (progress) {
            progress(i + 1, result.total, file.filename().string());
        }

        try {
            int rc = process_single(file, opts);
            if (rc == 0) {
                ++result.succeeded;
            } else {
                ++result.failed;
            }
        } catch (const std::exception& e) {
            spdlog::error("  Error: {}", e.what());
            ++result.failed;
        }
    }

    spdlog::info("Batch complete: {} ok, {} failed, {} skipped of {} total",
                 result.succeeded, result.failed, result.skipped, result.total);

    return result;
}

} // namespace wmr
