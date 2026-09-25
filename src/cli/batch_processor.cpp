#include "cli/batch_processor.hpp"
#include "cli/progress.hpp"
#include "core/watermark_engine.hpp"
#include "core/types.hpp"
#include "core/still_remove.hpp"

#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <fmt/format.h>

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
        return 2;  // load failure -> batch counts as skipped (distinct from a processing failure)
    }

    WatermarkEngine engine;

    // Visible watermark removal: the ONE shared policy (core/still_remove), the
    // same copy the single-image CLI path runs. Removed and NoWatermark both
    // fall through to the save (batch re-encodes no-watermark files today);
    // Failed counts as a per-image failure.
    if (opts.mode == CliMode::AutoRemove) {
        StillRemoveOptions sro;
        if (!build_still_remove_options(opts, sro)) return 1;  // pre-validated below
        StillRemoveOutcome ro = remove_still(engine, image, sro);
        if (ro.outcome == StillOutcome::Failed) {
            spdlog::error("  Removal failed: {}", ro.error);
            return 1;
        }
    }

    // SynthID scrub. Spectral detection/suppression was removed in 1.16.0; the only
    // SynthID operation is `--synthid-attack regen` (lossy SDXL img2img). Mirrors
    // process_single_image: on AutoRemove (batch from `remove`) regen runs only when
    // the user explicitly passed --synthid-attack; on SynthidOnly (batch from
    // `synthid`) it always runs. This is the method-extension seam: a future method
    // adds one IsMember value + one branch below.
    if ((opts.mode == CliMode::AutoRemove && opts.synthid_attack_requested) ||
        opts.mode == CliMode::SynthidOnly) {
        if (opts.synthid_attack == "regen") {
#ifdef WMR_BUILD_REGEN
            // regen = whole-image SDXL img2img. On failure the engine returns false
            // (logs + leaves the image unchanged). Mirror the single-image path:
            // surface the failure so the batch loop counts this image as failed
            // (not silently as success) and skip the misleading save.
            InpaintConfig ic;
            ic.method = InpaintMethod::DiffusionRegen;
            ic.regen_strength       = opts.regen_strength;
            ic.regen_steps          = opts.regen_steps;
            ic.regen_tile           = !opts.regen_no_tile;
            ic.regen_allow_download = !opts.regen_no_download;
            ic.regen_model_path     = opts.regen_model_path;
            ic.regen_vae_path       = opts.regen_vae_path;
            ic.regen_backend        = opts.regen_backend;
            ic.regen_restore.mode   = opts.regen_restore_detail   ? RestoreMode::On
                                      : opts.no_regen_restore_detail ? RestoreMode::Off
                                                                     : RestoreMode::Auto;
            ic.regen_restore.band_sigma = opts.regen_restore_band_sigma;
            DetectionResult dr{};  // regen ignores visible-mark detection
            bool ok = engine.remove_watermark_detected(image, dr, ic);
            if (!ok) {
                spdlog::warn("  SynthID regen did not complete (no model/backend, or it failed); output unchanged.");
                return 1;
            }
#else
            spdlog::error("--synthid-attack regen: this wmr build is regen-free (WMR_BUILD_REGEN off). "
                          "Rebuild with WMR_BUILD_REGEN=1.");
            return 1;
#endif
        }
        // Future SynthID methods: add the IsMember value + a branch here.
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

    if (!write_still_output(out_path, image, opts.keep_provenance)) {
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

    // Validate the shared options once (--rect), so a malformed value fails the
    // whole batch with a single error rather than one per file. rc parity with
    // the single-image path: a malformed --rect exits 1 (nothing processed).
    if (opts.mode == CliMode::AutoRemove) {
        StillRemoveOptions sro;
        if (!build_still_remove_options(opts, sro)) {
            result.failed = 1;  // run_cli maps failed > 0 -> rc 1
            return result;      // error already logged
        }
    }

    spdlog::info("Processing {} images...", result.total);

    // B1: progress-stream header naming the output dir, gated on --no-progress so
    // it only appears with the rest of the progress UX. (The reporter itself is a
    // no-op when progress is disabled, so the header must be gated to match.)
    if (progress_enabled()) {
        std::string out_dir = opts.output_path.empty()
            ? (opts.input_path + "/cleaned")
            : opts.output_path;
        std::fprintf(stderr, "Batch: %d image%s -> %s\n",
                     result.total, result.total == 1 ? "" : "s", out_dir.c_str());
        std::fflush(stderr);
    }

    // Live batch progress (stderr). The per-image index line that used to go to
    // stdout is replaced by the reporter's extra (filename + per-item time); the
    // per-image "saved -> path" line from process_single still goes to stdout.
    // B1: unit "img" -> rate renders as "img/s" (was "image/s").
    ProgressReporter batch_rep("  batch", result.total, "img");

    for (int i = 0; i < result.total; ++i) {
        const auto& file = files[i];

        if (progress) {
            progress(i + 1, result.total, file.filename().string());
        }

        int rc = 1;
        auto t_item_start = std::chrono::steady_clock::now();
        try {
            rc = process_single(file, opts);
        } catch (const std::exception& e) {
            spdlog::error("  Error: {}", e.what());
            rc = 1;
        }
        double item_sec = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_item_start).count();

        // B2: rc==2 = load failure (skipped), rc==0 = ok, else failed.
        if (rc == 0) ++result.succeeded;
        else if (rc == 2) ++result.skipped;
        else ++result.failed;

        // B1: per-item time in the extra field, e.g. "a.png (1.2s)". C3 moves
        // extra before the bar so the filename leads the line.
        batch_rep.update(i + 1,
                         fmt::format("{} ({:.1f}s)", file.filename().string(), item_sec));
    }
    batch_rep.finish();

    // B2: only mention "skipped" when it is non-zero (the field is otherwise
    // inert now that load failures are the only thing that increments it).
    if (result.skipped > 0) {
        spdlog::info("Batch complete: {} ok, {} failed, {} skipped of {} total",
                     result.succeeded, result.failed, result.skipped, result.total);
    } else {
        spdlog::info("Batch complete: {} ok, {} failed of {} total",
                     result.succeeded, result.failed, result.total);
    }

    return result;
}

} // namespace wmr
