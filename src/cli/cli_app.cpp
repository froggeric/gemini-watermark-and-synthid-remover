#include "cli/cli_app.hpp"
#include "cli/batch_processor.hpp"
#include "cli/progress.hpp"
#include "core/watermark_engine.hpp"
#include "core/types.hpp"
#include "core/update_check.hpp"
#include "metadata/provenance.hpp"
#ifdef WMR_BUILD_REGEN
#include "core/regenerator.hpp"  // last_regen_run_stats() for the SynthID recap
#endif
#if defined(WMR_BUILD_AI_COREML_SD) && defined(__APPLE__)
#include "core/coreml_cache.hpp"  // clear_coreml_execution_cache (the cache subcommand)
#endif
#include "video/video_processor.hpp"

#include <opencv2/imgcodecs.hpp>
#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <sstream>
#include <vector>

#ifndef APP_VERSION
#define APP_VERSION "1.1.0"
#endif

#ifndef APP_NAME
#define APP_NAME "wmr"
#endif

namespace {
void print_header(std::ostream& os) {
    os << wmr::kHeaderRule << "\n"
        << "  wmr v" APP_VERSION " — watermark remover\n"
        << "  Remove Gemini/Veo visible watermarks.\n"
        << "  Remove SynthID invisible watermarks via lossy regen (no detection; validated vs Google's SynthID verifier).\n"
        << "  --synthid-attack regen is lossy (SDXL img2img; ~6.5 GB model + ~335 MB VAE download on first use).\n"
        << "  https://github.com/froggeric/gemini-watermark-and-synthid-remover\n"
        << "  Copyright 2026 Frédéric Guigand\n"
        << wmr::kHeaderRule << "\n\n";
}
} // namespace

namespace wmr {

std::pair<std::optional<WatermarkVariant>, bool>
resolve_still_variant(const CliOptions& opts) {
    if (opts.still_legacy)    return {WatermarkVariant::V1, false};
    if (opts.still_no_legacy) return {WatermarkVariant::V2, false};
    return {std::nullopt, true};  // default: V2 with auto V2→V1 fallback
}

// Resolve the residual-cleanup InpaintConfig from CLI opts.
// Returns false when the user chose "--denoise off" (skip cleanup entirely).
// method mapping: soft→Gaussian, ns→NavierStrokes, telea→Telea, ai→AiDenoise,
// off→(caller skips). strength is stored as percent (0-300) and divided here.
bool resolve_inpaint_config(const CliOptions& opts, InpaintConfig& out) {
    out.strength = opts.denoise_strength_pct / 100.0f;
    out.radius = opts.denoise_radius;
    out.sigma = opts.denoise_sigma;
    out.padding = 32;
    const std::string& m = opts.denoise_method;
    if (m == "off")  return false;  // skip cleanup entirely (reverse-blend only)
    if (m == "soft") { out.method = InpaintMethod::Gaussian;      return true; }
    if (m == "ns")   { out.method = InpaintMethod::NavierStokes; return true; }
    if (m == "telea"){ out.method = InpaintMethod::Telea;         return true; }
#ifdef WMR_AI_DENOISE
    if (m == "ai")   { out.method = InpaintMethod::AiDenoise;     return true; }
#endif
    // Unknown / unreachable (CLI11 IsMember validates choices) → Gaussian.
    out.method = InpaintMethod::Gaussian;
    return true;
}

// Parse a "x,y,w,h" rect string. Returns nullopt for an empty OR malformed string;
// the caller distinguishes the two (empty = flag absent, malformed = user error).
std::optional<cv::Rect> parse_rect(const std::string& s) {
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

// Build the still-image geometry override from CLI opts. Returns false (with a logged
// error) when --rect was given but malformed.
bool resolve_still_geometry_override(const CliOptions& opts, StillGeometryOverride& out) {
    if (!opts.still_rect_str.empty()) {
        out.rect = parse_rect(opts.still_rect_str);
        if (!out.rect) {
            spdlog::error("Invalid --rect format. Expected: x,y,w,h (e.g. --rect 760,1063,36,36)");
            return false;
        }
    }
    if (!opts.still_geo_preset.empty()) out.preset = opts.still_geo_preset;
    out.no_auto_geometry = opts.still_no_auto_geometry;
    return true;
}

static int process_detect(const CliOptions& opts) {
    cv::Mat image = cv::imread(opts.input_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        spdlog::error("Failed to load image: {}", opts.input_path);
        return 1;
    }

    spdlog::info("Image: {}x{}", image.cols, image.rows);

    // Visible watermark detection — report the requested profile(s).
    // Default (no flag): report both V2 (current) and V1 (legacy).
    {
        WatermarkEngine engine;
        const WatermarkSize sz = get_watermark_size(image.cols, image.rows);

        StillGeometryOverride gov;
        if (!resolve_still_geometry_override(opts, gov)) return 1;
        // Resolve the hybrid auto-geometry ONCE for the V2 profile (the search only
        // runs for V2 small). The V1 report uses no forced position / override alpha.
        WatermarkEngine::StillResolveResult resolved;
        if (!opts.still_legacy) {
            resolved = engine.resolve_still_geometry(image, WatermarkVariant::V2, sz, gov);
        }

        auto report = [&](WatermarkVariant v) {
            const bool is_v2 = (v == WatermarkVariant::V2);
            const std::optional<WatermarkPosition> fp = is_v2 ? resolved.pos : std::nullopt;
            const cv::Mat* alpha_ov = is_v2 ? resolved.alpha : nullptr;
            // An explicit --rect/--geo-preset forces the reported position; do not
            // let the snap refinement override it (consistent with the remove path).
            const bool explicit_ov = gov.rect.has_value() || gov.preset.has_value();
            const bool snap = !explicit_ov && (
                fp.has_value() || (is_v2 && sz == WatermarkSize::Small));
            auto result = engine.detect_watermark(image, std::nullopt, fp,
                                                  alpha_ov, v, snap);
            const char* tag = (v == WatermarkVariant::V1) ? "[VISIBLE V1]" : "[VISIBLE V2]";
            if (result.detected) {
                spdlog::info("{} DETECTED (confidence: {:.1f}%)", tag,
                             result.confidence * 100.0f);
                spdlog::info("  Region: ({}, {}) {}x{}", result.region.x, result.region.y,
                             result.region.width, result.region.height);
            } else {
                spdlog::info("{} not detected ({:.1f}%)", tag, result.confidence * 100.0f);
            }
        };
        if (opts.still_legacy)         report(WatermarkVariant::V1);
        else if (opts.still_no_legacy) report(WatermarkVariant::V2);
        else { report(WatermarkVariant::V2); report(WatermarkVariant::V1); }
    }

    // SynthID detection was removed in 1.16.0: the spectral detector had no
    // discriminative power (ROC AUC 0.20 on Google-verifier-labeled images) and
    // the only SynthID verifier is Google's manual in-app "Verify with SynthID" tool (no API; not automatable). `wmr detect` is visible-only now.
    // See docs/research/synthid-spectral-removal-record.md.
    return 0;
}

static int process_single_image(const CliOptions& opts) {
    cv::Mat image = cv::imread(opts.input_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        spdlog::error("Failed to load image: {}", opts.input_path);
        return 1;
    }

    spdlog::info("Loading: {}", opts.input_path);
    spdlog::info("Image: {}x{}", image.cols, image.rows);

    bool did_work = false;

    // Visible watermark removal
    if (opts.mode == CliMode::AutoRemove) {
        WatermarkEngine engine;
        std::optional<WatermarkSize> force_size;
        if (opts.force_small) force_size = WatermarkSize::Small;
        else if (opts.force_large) force_size = WatermarkSize::Large;

        auto [force_variant, try_fallback] = resolve_still_variant(opts);

        // Resolve the still geometry overrides + hybrid auto-search ONCE, for the V2
        // profile (the search only runs for V2 small). The V2 primary attempt uses the
        // resolved position + the matched-size alpha (36 or 48); the V1 fallback uses
        // neither. --force skips detection entirely, so it is excluded here.
        WatermarkEngine::StillResolveResult resolved;
        if (!opts.force) {
            StillGeometryOverride gov;
            if (!resolve_still_geometry_override(opts, gov)) return 1;
            const WatermarkSize sz =
                force_size.value_or(get_watermark_size(image.cols, image.rows));
            resolved = engine.resolve_still_geometry(
                image, WatermarkVariant::V2, sz, gov);
        }

        // An explicit --rect/--geo-preset means "remove at this position" even when
        // the detector's confidence is too low to confirm (a faint mark the search
        // localized but could not pass the gate). Without an override, require a real
        // detection.
        const bool explicit_override =
            !opts.still_rect_str.empty() || !opts.still_geo_preset.empty();

        // Detect→remove for one variant; true if a watermark was found and removed.
        auto try_remove = [&](WatermarkVariant v,
                              std::optional<WatermarkPosition> force_pos,
                              const cv::Mat* alpha_override) -> bool {
            // An explicit --rect/--geo-preset forces removal at exactly that position;
            // do not let the snap refinement override it. Auto-geometry (and the
            // V2-small default) still snaps to refine its approximate position.
            const bool snap = !explicit_override && (
                force_pos.has_value() ||
                (v == WatermarkVariant::V2 &&
                 force_size.value_or(get_watermark_size(image.cols, image.rows))
                     == WatermarkSize::Small));
            auto detection = engine.detect_watermark(image, force_size, force_pos,
                                                     alpha_override, v, snap);
            if (!detection.detected) {
                if (!(explicit_override && force_pos.has_value())) return false;
                spdlog::info("Removing at overridden position "
                             "(confidence {:.1f}% below the detection gate)",
                             detection.confidence * 100.0f);
            } else {
                // S2: surface auto-geometry's decision (position + matched size)
                // so the user can see what was resolved.
                spdlog::info("Visible watermark detected ({:.1f}%, {}) at ({},{}) size={}, removing...",
                             detection.confidence * 100.0f,
                             v == WatermarkVariant::V1 ? "V1" : "V2",
                             detection.region.x, detection.region.y, detection.region.width);
            }
            const cv::Mat& alpha = alpha_override ? *alpha_override
                                                  : engine.get_still_alpha(detection.size, v);
            InpaintConfig icfg;
            bool do_cleanup = resolve_inpaint_config(opts, icfg);
            if (do_cleanup) {
                engine.remove_watermark_detected(image, detection, icfg, &alpha);
            } else {
                // --denoise off: reverse-blend only, no residual cleanup.
                engine.remove_watermark_alpha_only(image, detection, &alpha);
            }
            return true;
        };

        if (opts.force) {
            WatermarkVariant active = force_variant.value_or(WatermarkVariant::V2);
            spdlog::info("Force mode: removing visible watermark ({})",
                         active == WatermarkVariant::V1 ? "V1" : "V2");
            engine.remove_watermark(image, force_size, force_variant);
            did_work = true;
        } else {
            WatermarkVariant primary = force_variant.value_or(WatermarkVariant::V2);
            const bool is_v2 = (primary == WatermarkVariant::V2);
            bool removed = try_remove(primary, is_v2 ? resolved.pos : std::nullopt,
                                      is_v2 ? resolved.alpha : nullptr);
            if (!removed && try_fallback && primary == WatermarkVariant::V2) {
                spdlog::info("V2 profile not detected — retrying with legacy V1");
                removed = try_remove(WatermarkVariant::V1, std::nullopt, nullptr);
            }
            if (removed) {
                did_work = true;
            } else if (opts.mode == CliMode::AutoRemove) {
                spdlog::debug("No visible watermark detected");
            } else {
                spdlog::warn("No visible watermark detected. Use --force to remove anyway.");
                return 2;
            }
        }
    }

    // SynthID scrub. The spectral detector + suppressor was removed in 1.16.0
    // (it did not work; see docs/research/synthid-spectral-removal-record.md), so
    // the only SynthID operation is `--synthid-attack regen` (lossy SDXL img2img).
    //
    // When it runs:
    //   - SynthidOnly mode (the `synthid` subcommand): always. Its whole purpose.
    //   - AutoRemove mode (the `remove` subcommand): only when the user EXPLICITLY
    //     passed --synthid-attack (regen is lossy + downloads ~6.5 GB, so opt-in).
    //     Because "regen" is the default string value, the request is detected via
    //     CLI11's option count, not the string alone (synthid_attack_requested).
    //
    // This block is the method-extension seam: a future SynthID method is one new
    // IsMember value on --synthid-attack + one new branch below. Keep the per-method
    // dispatch intact; do not collapse it to a hardcoded "regen is the only path".
    const bool synthid_attack_requested = opts.synthid_attack_requested;
    if (opts.mode == CliMode::SynthidOnly ||
        (opts.mode == CliMode::AutoRemove && synthid_attack_requested)) {

        if (opts.synthid_attack == "regen") {
#ifdef WMR_BUILD_REGEN
            // regen = SDXL img2img over the WHOLE image (a SynthID-only scrub). It does
            // NOT remove the visible Gemini diamond (strength ~0.10 cannot); in AutoRemove
            // mode the visible pass above already ran, in SynthidOnly mode the input is the
            // user's responsibility (visible-clean first if needed). No codebook required.
            WatermarkEngine engine;             // reaches the leaked regenerator singleton
            InpaintConfig ic;
            ic.method = InpaintMethod::DiffusionRegen;
            ic.regen_strength       = opts.regen_strength;
            ic.regen_steps          = opts.regen_steps;
            ic.regen_tile           = !opts.regen_no_tile;
            ic.regen_allow_download = !opts.regen_no_download;
            ic.regen_model_path     = opts.regen_model_path;
            ic.regen_vae_path       = opts.regen_vae_path;
            ic.regen_backend        = opts.regen_backend;
            // Resolve the detail-restoration mode (default Auto: bright >=128 -> restore,
            // dim -> full regen). --regen-restore-detail forces On; --no- forces Off;
            // neither is Auto. Both at once is rejected below (the mutex check).
            ic.regen_restore.mode = opts.regen_restore_detail   ? RestoreMode::On
                                  : opts.no_regen_restore_detail ? RestoreMode::Off
                                                                  : RestoreMode::Auto;
            DetectionResult dr{};  // regen ignores visible-mark detection (whole-image scrub)
            dr.detected = false;
            dr.confidence = 0.0f;
            // engine returns false when regen no-op'd (no model/backend, or regen failed);
            // the image is byte-for-byte unchanged in that case. Surface it to the exit
            // code so a scripted caller can tell regen did not run, and skip the misleading
            // "complete" line + the save of an unchanged image.
            bool ok = engine.remove_watermark_detected(image, dr, ic);
            if (!ok) {
                spdlog::warn("SynthID regen did not complete (no model/backend, or it failed); output is unchanged.");
                return 1;
            }
            // Recap line: elapsed + tile count + backend. The stats come from the
            // regenerator singleton (written at the end of regen()); the honesty
            // caveat is unchanged.
            const auto stats = wmr::last_regen_run_stats();
            // Clean duration string for the recap (seconds / m+s / h+m), never the
            // "<10s"/"--" progress-ETA forms.
            auto recap_dur = [](double s) -> std::string {
                if (s <= 0.0) return "";
                int total = static_cast<int>(s + 0.5);
                if (total < 60) return fmt::format("{}s", total);
                if (total < 3600) return fmt::format("{}m{}s", total / 60, total % 60);
                return fmt::format("{}h{}m", total / 3600, (total % 3600) / 60);
            };
            std::string recap = "SynthID regen complete";
            std::string paren;
            if (stats.tiles > 0) {
                paren = fmt::format("{} tile{}, {}", stats.tiles,
                                    stats.tiles == 1 ? "" : "s", stats.backend);
            } else if (!stats.backend.empty()) {
                paren = stats.backend;
            }
            std::string dur = recap_dur(stats.elapsed_seconds);
            if (!dur.empty() && !paren.empty()) {
                recap += fmt::format(" in {} ({})", dur, paren);
            } else if (!dur.empty()) {
                recap += fmt::format(" in {}", dur);
            } else if (!paren.empty()) {
                recap += fmt::format(" ({})", paren);
            }
            spdlog::info("{} (lossy; verify via Google's SynthID tool if you need to confirm)",
                         recap);
            did_work = true;
#else
            spdlog::error("--synthid-attack regen: this wmr build is regen-free (WMR_BUILD_REGEN off). "
                          "Rebuild with WMR_BUILD_REGEN=1.");
            return 1;
#endif
        }
        // Future SynthID methods: add the IsMember value + a branch here.
    }

    if (!did_work) {
        spdlog::info("No watermarks found or removed");
        return 0;
    }

    // Save output
    if (opts.output_path.empty()) {
        spdlog::error("No output path specified. Use -o <path> to set the output file.");
        return 1;
    }
    std::string output = opts.output_path;
    std::filesystem::path out_path(output);
    if (!out_path.parent_path().empty() && !std::filesystem::exists(out_path.parent_path())) {
        std::filesystem::create_directories(out_path.parent_path());
    }

    std::vector<int> params;
    std::string ext = out_path.extension().string();
    if (ext == ".jpg" || ext == ".jpeg") {
        params = {cv::IMWRITE_JPEG_QUALITY, 100};
    } else if (ext == ".png") {
        params = {cv::IMWRITE_PNG_COMPRESSION, 6};
    } else if (ext == ".webp") {
        params = {cv::IMWRITE_WEBP_QUALITY, 101};
    }

    if (!cv::imwrite(output, image, params)) {
        spdlog::error("Failed to save: {}", output);
        return 1;
    }

    // Guaranteed provenance-free output (DECISION A). Defensive on today's OpenCV
    // output (which already strips all metadata on write): the scan finds nothing
    // and the file is not rewritten. The guarantee holds if the encoder changes.
    if (!opts.keep_provenance) {
        auto pr = wmr::provenance::post_write_provenance_strip(output, /*keep_standard=*/true);
        if (pr.rewritten && pr.items_removed > 0)
            spdlog::info("Stripped {} provenance item(s) from output", pr.items_removed);
    }

    spdlog::info("Saved: {}", output);
    return 0;
}

static int process_video(const CliOptions& opts) {
    VideoWatermarkConfig config;

    // Resolve video profile
    if (opts.notebooklm_profile) {
        config.profile = VideoProfile::NotebookLM;
    } else {
        config.profile = opts.legacy_profile ? VideoProfile::VeoLegacy
                                              : VideoProfile::GeminiDiamond;
    }

    // Parse --rect x,y,w,h manual override (any video profile)
    if (!opts.notebooklm_rect_str.empty()) {
        config.rect = parse_rect(opts.notebooklm_rect_str);
        if (!config.rect) {
            spdlog::error("Invalid --rect format. Expected: x,y,w,h (e.g. --rect 1145,689,121,17)");
            return 1;
        }
    }

    // Parse variant string
    if (opts.video_variant_str == "720p-1") {
        config.variant = VideoVariant::P720_1;
    } else if (opts.video_variant_str == "720p-2") {
        config.variant = VideoVariant::P720_2;
    } else if (opts.video_variant_str == "1080p") {
        config.variant = VideoVariant::P1080p;
    } else {
        config.variant = VideoVariant::Auto;
    }

    config.force = opts.force;
    config.inpaint_strength = opts.inpaint_strength;
    config.scenes = opts.scenes;
    config.scene_threshold = opts.scene_threshold;
    config.no_auto_geometry = opts.no_auto_geometry;
    config.edge_cleanup = !opts.no_edge_cleanup;
    config.notebooklm_complexity_threshold = opts.notebooklm_complexity_threshold;
    config.notebooklm_method = opts.notebooklm_method;
    if (opts.notebooklm_method != "auto" && config.profile != VideoProfile::NotebookLM) {
        spdlog::warn("--notebooklm-method ignored (only valid with --notebooklm)");
    }

    EncodeOptions encode;
    encode.codec = opts.video_codec;
    encode.crf = opts.video_crf;
    encode.preset = opts.video_preset;

    // Resolve output path (file or directory depending on --scenes)
    std::string output = opts.output_path;
    if (config.scenes) {
        if (!output.empty()) {
            std::filesystem::path p(output);
            if (std::filesystem::exists(p) && std::filesystem::is_regular_file(p)) {
                spdlog::error("--scenes requires a directory output, not a file. Got: {}", output);
                return 1;
            }
        } else {
            std::filesystem::path p(opts.input_path);
            output = (p.parent_path() / (p.stem().string() + "_scenes")).string();
        }
        std::filesystem::create_directories(output);
    } else {
        if (output.empty()) {
            std::filesystem::path p(opts.input_path);
            output = (p.parent_path() / (p.stem().string() + "_clean" + p.extension().string())).string();
        }
    }

    VideoProcessor processor;
    auto result = processor.process(opts.input_path, output, config, encode);

    return result.success ? 0 : 1;
}

// `wmr cache` — manage wmr's local caches. Today only `--clear-coreml` (macOS):
// remove_all the app-scoped e5bundlecache so CoreML recompiles fresh on the next
// regen. The shared ~/Library/Caches/CoreML is NEVER touched. On Linux/Windows
// (or a CoreML-SD-OFF build) the flag is accepted but logs a no-op message and
// exits 0, so help text stays cross-platform consistent.
static int process_cache(const CliOptions& opts) {
    if (!opts.cache_clear_coreml) {
        // No flag: surface the available cache ops. (Today only one.)
        spdlog::info("wmr cache manages wmr's local caches. "
                     "Available: --clear-coreml (macOS).");
        return 0;
    }
#if defined(WMR_BUILD_AI_COREML_SD) && defined(__APPLE__)
    std::filesystem::path cache_dir = wmr::coreml_app_cache_dir();
    std::filesystem::path e5 = cache_dir / "com.apple.e5rt.e5bundlecache";
    bool ok = wmr::clear_coreml_execution_cache(cache_dir);
    if (ok) {
        spdlog::info("Cleared the CoreML execution cache ({}).", e5.string());
        spdlog::info("CoreML will recompile on the next regen (one-time, ~30s).");
        return 0;
    }
    // A clear failure is non-fatal (warns were already emitted by the removal
    // helper). Do not block the user; surface the partial result.
    spdlog::warn("CoreML execution cache clear was partial (see warnings above).");
    return 0;
#else
    spdlog::info("No CoreML execution cache on this platform (macOS only).");
    return 0;
#endif
}

namespace {

// Read a whole file into a byte buffer. Returns nullopt on any IO error.
// Used by `wmr metadata`, which reads raw container bytes itself (it does not
// go through cv::imread; the metadata path is lossless byte IO, never a pixel
// decode/re-encode).
std::optional<std::vector<std::byte>> read_whole_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return std::nullopt;
    const std::streamoff sz = f.tellg();
    if (sz < 0) return std::nullopt;
    f.seekg(0, std::ios::beg);
    std::vector<std::byte> buf(static_cast<std::size_t>(sz));
    if (sz > 0) {
        f.read(reinterpret_cast<char*>(buf.data()), sz);
        if (!f) return std::nullopt;
    }
    return buf;
}

bool write_whole_file(const std::filesystem::path& path,
                      const std::byte* data, std::size_t n) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    if (n > 0)
        f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n));
    return static_cast<bool>(f);
}

const char* provenance_format_name(provenance::ContainerFormat f) {
    switch (f) {
        case provenance::ContainerFormat::Png: return "PNG";
        case provenance::ContainerFormat::Jpeg: return "JPEG";
        case provenance::ContainerFormat::WebPRiff: return "WebP";
        case provenance::ContainerFormat::IsobmffImage:
            return "ISOBMFF image (AVIF/HEIF/JPEG-XL)";
        case provenance::ContainerFormat::IsobmffVideo:
            return "ISOBMFF video (MP4/MOV)";
        default: return "unknown";
    }
}

void print_provenance_report(const std::string& label,
                             const provenance::MetadataReport& rep) {
    spdlog::info("{}: {} ({})", label, provenance_format_name(rep.format),
                 rep.supported ? "supported" : "unsupported, passed through");
    if (rep.has_c2pa) {
        spdlog::info("  C2PA manifest detected{}",
                     rep.c2pa_note ? ("; possible issuer/tool: " + *rep.c2pa_note)
                                   : "");
    }
    for (const auto& f : rep.findings)
        spdlog::info("  {}: {}", f.where, f.detail);
    if (rep.supported && rep.findings.empty())
        spdlog::info("  no provenance metadata found");
}

} // namespace

// `wmr metadata`: report and losslessly strip C2PA / AI-provenance metadata.
// Operates on raw container bytes only (never decodes pixels). This is a
// SEPARATE lightweight byte-IO loop, NOT batch_process (which loads into a
// cv::Mat and re-encodes via OpenCV, which would be lossy on pixels and would
// itself drop all metadata).
static int process_metadata(const CliOptions& opts) {
    namespace fs = std::filesystem;

    // Core: read one file, report, and (unless --dry-run) write stripped output.
    // out_path is nullopt when not writing (single-file --dry-run, or directory
    // --dry-run). Returns 0 on success, 1 on error.
    auto handle_one = [&](const fs::path& in_path,
                          const std::optional<fs::path>& out_path) -> int {
        auto data = read_whole_file(in_path.string());
        if (!data) {
            spdlog::warn("{}: read error, skipped", in_path.string());
            return 1;
        }
        const auto rep = provenance::report_provenance(*data);
        print_provenance_report(in_path.string(), rep);

        if (opts.metadata_dry_run || !out_path) return 0;

        provenance::StripOptions sopts;
        sopts.keep_standard = !opts.metadata_strip_all;
        sopts.dry_run = false;
        const auto sr = provenance::strip_provenance(*data, sopts);

        const fs::path& op = *out_path;
        std::error_code ec;
        if (!op.parent_path().empty())
            fs::create_directories(op.parent_path(), ec);

        // Fail-safe: when supported and something was stripped, write `out`.
        // Otherwise (clean / unsupported / malformed) copy the input through
        // UNCHANGED so the user still gets an output file.
        bool wrote = false;
        if (sr.ok && sr.supported && sr.items_removed > 0 && !sr.out.empty()) {
            wrote = write_whole_file(op, sr.out.data(), sr.out.size());
        } else {
            wrote = write_whole_file(op, data->data(), data->size());
        }
        if (!wrote) {
            spdlog::error("{}: failed to write output", op.string());
            return 1;
        }
        if (sr.ok && sr.supported && sr.items_removed > 0)
            spdlog::info("  stripped {} item(s) -> {}", sr.items_removed, op.string());
        else
            spdlog::info("  copied through -> {}", op.string());
        return 0;
    };

    if (fs::is_directory(opts.input_path)) {
        const fs::path input_dir(opts.input_path);
        fs::path out_dir;
        if (!opts.output_path.empty()) {
            out_dir = fs::path(opts.output_path);
        } else {
            out_dir = input_dir.parent_path() /
                      (input_dir.filename().string() + "_clean");
        }
        std::error_code ec;
        fs::create_directories(out_dir, ec);

        int total = 0, failed = 0;
        auto visit = [&](const fs::path& in_file) {
            ec.clear();
            if (!fs::is_regular_file(in_file, ec)) return;
            ++total;
            const auto rel = fs::relative(in_file, input_dir, ec);
            if (ec) {
                spdlog::warn("{}: relative-path error, skipped", in_file.string());
                ++failed;
                return;
            }
            std::optional<fs::path> out_file;
            if (!opts.metadata_dry_run)
                out_file = out_dir / rel;
            if (handle_one(in_file, out_file) != 0) ++failed;
        };
        if (opts.recursive) {
            for (auto it = fs::recursive_directory_iterator(input_dir, ec);
                 it != fs::recursive_directory_iterator(); it.increment(ec)) {
                if (ec) break;
                visit(it->path());
            }
        } else {
            for (auto it = fs::directory_iterator(input_dir, ec);
                 it != fs::directory_iterator(); it.increment(ec)) {
                if (ec) break;
                visit(it->path());
            }
        }
        spdlog::info("Processed {} file(s){}; {} failed",
                     total,
                     opts.metadata_dry_run ? " (dry run)" : "",
                     failed);
        return failed > 0 ? 1 : 0;
    }

    // Single file.
    const fs::path input(opts.input_path);
    std::optional<fs::path> out_path;
    if (!opts.metadata_dry_run) {
        if (!opts.output_path.empty()) {
            out_path = fs::path(opts.output_path);
        } else {
            out_path = input.parent_path() /
                       (input.stem().string() + "_clean" + input.extension().string());
        }
    }
    return handle_one(input, out_path);
}

int run_cli(int argc, char* argv[]) {
    CLI::App app{"", APP_NAME};
    app.set_version_flag("-V,--version",
        std::string(APP_VERSION) + "\nhttps://github.com/froggeric/gemini-watermark-and-synthid-remover");
    app.fallthrough();

    // Show help when called with no arguments
    if (argc <= 1) {
        print_header(std::cout);
        std::cout << app.help() << std::endl;
        return 0;
    }

    CliOptions opts;

    // Common options shared across subcommands
    auto add_common = [&](CLI::App* cmd) {
        cmd->add_flag("-v,--verbose", opts.verbose, "Verbose output");
        cmd->add_flag("--no-progress", opts.no_progress,
                      "Suppress progress output (errors + final summary only)");
        cmd->add_flag("--no-update-check", opts.no_update_check,
                      "Skip the startup update check (also WMR_NO_UPDATE_CHECK=1)");
    };

    // Still-image geometry flags shared by remove / visible / detect.
    const std::vector<std::string> preset_names(
        std::begin(kStillPresetNames), std::end(kStillPresetNames));
    auto add_still_geometry = [&](CLI::App* cmd) {
        cmd->add_option("--rect", opts.still_rect_str,
                        "Manual watermark rect x,y,w,h (overrides auto-detect)");
        cmd->add_option("--geo-preset", opts.still_geo_preset,
                        "Named geometry preset (calibrated Gemini 3.6 sizes)")
            ->check(CLI::IsMember(preset_names));
        cmd->add_flag("--no-auto-geometry", opts.still_no_auto_geometry,
                      "Disable the content-based geometry search; use the model position");
    };

    // Residual cleanup flags shared by remove / visible. The DEFAULT is "off" (pure
    // mathematical reverse-alpha-blend, no blur). opt in to a residual-only cleanup
    // with --denoise soft|ns|telea (or ai on AI builds).
    std::vector<std::string> denoise_choices{"off", "soft", "ns", "telea"};
#ifdef WMR_AI_DENOISE
    denoise_choices.insert(denoise_choices.begin(), "ai");
#endif
    auto add_denoise = [&](CLI::App* cmd) {
        cmd->add_option("--denoise", opts.denoise_method,
                        "Residual cleanup after reverse-blend: off (default, exact "
                        "reversal) | soft (Gaussian) | ns (Navier-Stokes) | telea")
            ->check(CLI::IsMember(denoise_choices));
        cmd->add_option("--strength", opts.denoise_strength_pct,
                        "Cleanup strength 0-300 percent (default 120)")
            ->check(CLI::Range(0.0f, 300.0f));
        cmd->add_option("--radius", opts.denoise_radius,
                        "Inpaint radius 1-25 (default 10)")
            ->check(CLI::Range(1, 25));
#ifdef WMR_AI_DENOISE
        cmd->add_option("--sigma", opts.denoise_sigma,
                        "AI denoise noise level 1-150 (default 50)")
            ->check(CLI::Range(1.0f, 150.0f));
#endif
    };

    // SynthID-attack selection + regen knobs, shared by remove + synthid. The
    // --synthid-attack description is the exported honesty-lock string (the wording
    // test asserts on it). The regen flags are always bound (even on a regen-free
    // lean build) so `--synthid-attack regen` parses, then the dispatch path
    // rejects it with a clear error rather than failing at CLI parse time.
    // Returns the --synthid-attack option pointer so the caller (remove) can tell
    // "explicitly requested" from "default" via ->count() (regen is opt-in on remove).
    auto add_synthid_attack = [&](CLI::App* cmd) -> CLI::Option* {
        CLI::Option* atk = cmd->add_option("--synthid-attack", opts.synthid_attack,
                        synthid_attack_help_text())
            ->capture_default_str()
            ->check(CLI::IsMember({"regen"}));
        cmd->add_option("--regen-strength", opts.regen_strength,
                        "regen img2img denoising strength 0.02-0.15 "
                        "(lower = closer to original, weaker scrub)")
            ->capture_default_str()
            ->check(CLI::Range(0.02f, 0.15f));
        cmd->add_option("--regen-steps", opts.regen_steps,
                        "regen sample steps (fewer is faster; 50 is the validated default)")
            ->capture_default_str()
            ->check(CLI::Range(1, 100));
        cmd->add_flag("--regen-no-download", opts.regen_no_download,
                      "regen: never touch the network; error if the model is absent "
                      "(offline/air-gapped use)");
        cmd->add_flag("--regen-no-tile", opts.regen_no_tile,
                      "regen: disable tiled img2img; aspect-fit to <=1024 then resize "
                      "back (faster, lower fidelity)");
        cmd->add_option("--regen-model-path", opts.regen_model_path,
                        "regen: path to an SDXL .safetensors; overrides the model "
                        "download/cache resolution")
            ->check(CLI::ExistingPath);
        cmd->add_option("--regen-vae-path", opts.regen_vae_path,
                        "regen: path to an SDXL VAE .safetensors; overrides the "
                        "fp16-fix VAE download/cache. Point at the embedded VAE only "
                        "for experiments (it NaNs in fp16)")
            ->check(CLI::ExistingPath);
        cmd->add_option("--regen-backend", opts.regen_backend,
                        "regen runtime backend: auto (default; CPU on Apple Silicon "
                        "where Metal is broken, GPU elsewhere), cpu, metal (Apple GPU; "
                        "currently produces garbage on Apple Silicon), vulkan, or coreml "
                        "(Apple Silicon native CoreML SDXL; opt-in only).")
            ->capture_default_str()
            ->check(CLI::IsMember({"auto", "cpu", "metal", "vulkan", "coreml"}));
        cmd->add_flag("--regen-restore-detail", opts.regen_restore_detail,
                      "regen: restore detail lost to regen (top-5% diff slice + carrier "
                      "Wiener attenuation) to recover fidelity on bright images. FORCES "
                      "restoration on dim content too (user accepts the risk; verify via "
                      "Google's SynthID tool). Mutually exclusive with --no-regen-restore-detail.");
        cmd->add_flag("--no-regen-restore-detail", opts.no_regen_restore_detail,
                      "regen: never restore detail; guaranteed full SynthID removal "
                      "(the safe path on dim content). Mutually exclusive with "
                      "--regen-restore-detail.");
        return atk;
    };

    // --- remove (default) ---
    auto* remove_cmd = app.add_subcommand("remove", "Auto-detect and remove watermarks");
    remove_cmd->add_option("input", opts.input_path, "Input image or directory")
        ->required()
        ->check(CLI::ExistingPath);
    remove_cmd->add_flag("-f,--force", opts.force, "Skip detection");
    remove_cmd->add_flag("--force-small", opts.force_small, "Force 48x48 watermark");
    remove_cmd->add_flag("--force-large", opts.force_large, "Force 96x96 watermark");
    remove_cmd->add_flag("--legacy", opts.still_legacy,
                         "Use legacy Gemini (pre-3.5) V1 watermark profile");
    remove_cmd->add_flag("--no-legacy", opts.still_no_legacy,
                         "Pin current (Gemini 3.5+) V2 profile; disable auto fallback");
    remove_cmd->add_option("--inpaint-strength", opts.inpaint_strength,
                           "Inpaint strength 0.0-1.0")
        ->check(CLI::Range(0.0f, 1.0f));
    add_denoise(remove_cmd);
    // regen is opt-in on remove: only runs when --synthid-attack is explicitly passed.
    CLI::Option* remove_attack_opt = add_synthid_attack(remove_cmd);
    remove_cmd->add_flag("-r,--recursive", opts.recursive, "Process directories recursively");
    remove_cmd->add_option("-o,--output", opts.output_path, "Output path (required for files; batch defaults to cleaned/)");
    remove_cmd->add_flag("--keep-provenance", opts.keep_provenance,
        "Keep C2PA/AI-provenance metadata in the output (default: strip it). "
        "v1 strip covers PNG and JPEG and is lossless on pixels; the default "
        "post-write pass is a no-op on today's OpenCV output (which already "
        "drops all metadata). See `wmr metadata`.");
    add_still_geometry(remove_cmd);
    add_common(remove_cmd);

    // --- detect ---
    auto* detect_cmd = app.add_subcommand("detect", "Detect watermarks without modifying");
    detect_cmd->add_option("input", opts.input_path, "Input image")
        ->required()
        ->check(CLI::ExistingFile);
    detect_cmd->add_flag("--legacy", opts.still_legacy,
                         "Report only the legacy Gemini (pre-3.5) V1 profile");
    detect_cmd->add_flag("--no-legacy", opts.still_no_legacy,
                         "Report only the current (Gemini 3.5+) V2 profile");
    add_still_geometry(detect_cmd);
    add_common(detect_cmd);

    // --- synthid ---
    // The synthid subcommand's whole purpose is the SynthID scrub, so regen runs
    // by default (no need to pass --synthid-attack). The flag is still accepted
    // for symmetry and as the future method-extension seam.
    auto* synthid_cmd = app.add_subcommand("synthid",
        "Remove SynthID invisible watermark via lossy SDXL regen "
        "(no detection; lossy, validated against Google's SynthID verifier)");
    synthid_cmd->add_option("input", opts.input_path, "Input image")
        ->required()
        ->check(CLI::ExistingFile);
    synthid_cmd->add_flag("-f,--force", opts.force, "Skip detection");
    synthid_cmd->add_option("-o,--output", opts.output_path, "Output path (required)");
    synthid_cmd->add_flag("--keep-provenance", opts.keep_provenance,
        "Keep C2PA/AI-provenance metadata in the output (default: strip it). "
        "v1 strip covers PNG and JPEG and is lossless on pixels; the default "
        "post-write pass is a no-op on today's OpenCV output (which already "
        "drops all metadata). See `wmr metadata`.");
    add_synthid_attack(synthid_cmd);
    add_common(synthid_cmd);

    // --- video ---
    auto* video_cmd = app.add_subcommand("video", "Remove watermark from video");
    video_cmd->add_option("input", opts.input_path, "Input video path")
        ->required()
        ->check(CLI::ExistingFile);
    video_cmd->add_option("-o,--output", opts.output_path, "Output path (default: <input>_clean.mp4)");
    video_cmd->add_flag("--legacy", opts.legacy_profile,
                         "Use Veo legacy text profile");
    video_cmd->add_flag("--notebooklm", opts.notebooklm_profile,
                         "Remove NotebookLM watermark (per-scene adaptive MI-GAN/NS inpaint)");
    video_cmd->add_option("--rect", opts.notebooklm_rect_str,
                           "Manual watermark rect x,y,w,h (overrides auto-detect; "
                           "Gemini/Veo and NotebookLM profiles)");
    video_cmd->add_option("--complexity-threshold", opts.notebooklm_complexity_threshold,
                           "Background-complexity floor above which MI-GAN is used "
                           "(below it, NS); default 15.0");
    video_cmd->add_option("--notebooklm-method", opts.notebooklm_method,
                           "NotebookLM inpaint method: auto (platform default) | ns | migan")
        ->check(CLI::IsMember({"auto", "ns", "migan"}));
    video_cmd->add_option("--variant", opts.video_variant_str,
                           "Force geometry: 720p-1, 720p-2, 1080p");
    video_cmd->add_flag("--no-auto-geometry", opts.no_auto_geometry,
                           "Disable content-based geometry search; use the resolution guess");
    video_cmd->add_flag("--no-edge-cleanup", opts.no_edge_cleanup,
                           "Pure reverse-blend; skip the diamond edge cleanup");
    video_cmd->add_flag("-f,--force", opts.force, "Skip detection");
    video_cmd->add_option("--crf", opts.video_crf, "Encode CRF")
        ->check(CLI::Range(0, 51));
    video_cmd->add_option("--preset", opts.video_preset, "Encode preset");
    video_cmd->add_option("--codec", opts.video_codec, "Video codec");
    video_cmd->add_flag("--scenes", opts.scenes,
                         "Enable scene detection for multi-scene videos");
    video_cmd->add_option("--scene-threshold", opts.scene_threshold,
                           "Scene cut sensitivity 0.0-1.0 (default: 0.3)")
        ->check(CLI::Range(0.0, 1.0));
    video_cmd->add_option("--inpaint-strength", opts.inpaint_strength,
                           "Inpaint strength 0.0-1.0")
        ->check(CLI::Range(0.0f, 1.0f));
    video_cmd->add_flag("--keep-provenance", opts.keep_provenance,
        "Accepted for CLI symmetry. wmr video re-encodes via FFmpeg (a fresh "
        "container), so input C2PA boxes are not carried into the output by "
        "construction; an explicit provenance strip is a later phase (v1).");
    add_common(video_cmd);

    // --- cache ---
    // Manages wmr's local caches. Today only --clear-coreml (macOS): clears the
    // app-scoped e5bundlecache so CoreML recompiles fresh on the next regen.
    auto* cache_cmd = app.add_subcommand("cache", "Manage wmr's local caches");
    cache_cmd->add_flag("--clear-coreml", opts.cache_clear_coreml,
                        "Clear wmr's app-scoped CoreML execution cache "
                        "(~/Library/Caches/wmr/com.apple.e5rt.e5bundlecache/ on macOS). "
                        "CoreML recompiles on the next regen (one-time, ~30s). "
                        "Does NOT touch ~/Library/Caches/CoreML or the model .mlpackage files. "
                        "No-op on Linux/Windows.");
    add_common(cache_cmd);

    // --- metadata ---
    // Report and losslessly strip C2PA / AI-provenance metadata from a PNG or
    // JPEG. Operates on container bytes only; never decodes pixels. v1 covers
    // PNG and JPEG; WebP/AVIF/HEIF/JPEG-XL/MP4 are reported and passed through.
    auto* metadata_cmd = app.add_subcommand(
        "metadata", "Report and strip C2PA / AI-provenance metadata (lossless on pixels)");
    metadata_cmd->add_option("input", opts.input_path, "Input image or directory")
        ->required()
        ->check(CLI::ExistingPath);
    metadata_cmd->add_option("-o,--output", opts.output_path,
        "Output path (file; for a directory input defaults to <input>_clean/)");
    metadata_cmd->add_flag("--dry-run", opts.metadata_dry_run,
        "Report findings only; do not write");
    metadata_cmd->add_flag("--strip-all", opts.metadata_strip_all,
        "Drop all non-essential metadata, not just AI markers");
    metadata_cmd->add_flag("-r,--recursive", opts.recursive,
        "Process directories recursively");
    add_common(metadata_cmd);

    // Default subcommand: if no subcommand given, treat as positional for backward compat
    app.require_subcommand(0, 1);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        // If a subcommand was parsed but failed on missing required args,
        // show the subcommand help instead of a bare error.
        auto subs = app.get_subcommands();
        if (!subs.empty()) {
            print_header(std::cout);
            std::cout << subs.front()->help() << std::endl;
            return 1;
        }
        return app.exit(e);
    }

    if (opts.verbose) {
        spdlog::set_level(spdlog::level::debug);
    } else {
        spdlog::set_level(spdlog::level::info);
    }

    // Master switch for the progress UX. Applied once, before any subcommand
    // dispatch, so every reporter constructed below (download / load / tile /
    // frame / batch) honors it. Errors + final summary always print regardless.
    wmr::set_progress_enabled(!opts.no_progress);

    if (opts.force_small && opts.force_large) {
        spdlog::error("Cannot use both --force-small and --force-large");
        return 1;
    }
    if (opts.still_legacy && opts.still_no_legacy) {
        spdlog::error("Cannot use both --legacy and --no-legacy");
        return 2;
    }
    if (opts.regen_restore_detail && opts.no_regen_restore_detail) {
        spdlog::error("Cannot use both --regen-restore-detail and --no-regen-restore-detail");
        return 2;
    }

    // On the remove subcommand, SynthID regen is opt-in: it runs only when the
    // user explicitly passed --synthid-attack. The synthid subcommand runs regen
    // unconditionally (its whole purpose), so the request flag is irrelevant there.
    opts.synthid_attack_requested = (remove_attack_opt->count() > 0);

    // Determine mode from subcommand
    if (detect_cmd->parsed()) {
        opts.mode = CliMode::Detect;
    } else if (synthid_cmd->parsed()) {
        opts.mode = CliMode::SynthidOnly;
    } else if (video_cmd->parsed()) {
        opts.mode = CliMode::Video;
    } else if (cache_cmd->parsed()) {
        opts.mode = CliMode::Cache;
    } else if (metadata_cmd->parsed()) {
        opts.mode = CliMode::Metadata;
    } else {
        opts.mode = CliMode::AutoRemove;
    }

    int rc = 0;
    try {
        switch (opts.mode) {
            case CliMode::Detect:
                rc = process_detect(opts);
                break;

            case CliMode::Video:
                rc = process_video(opts);
                break;

            case CliMode::Cache:
                rc = process_cache(opts);
                break;

            case CliMode::Metadata:
                rc = process_metadata(opts);
                break;

            case CliMode::AutoRemove:
            case CliMode::SynthidOnly: {
                // Check if input is a directory → batch mode
                if (std::filesystem::is_directory(opts.input_path)) {
                    auto result = batch_process(opts);
                    rc = result.failed > 0 ? 1 : 0;
                } else {
                    rc = process_single_image(opts);
                }
                break;
            }
        }
    } catch (const std::exception& e) {
        spdlog::error("Error: {}", e.what());
        rc = 1;
    }

#ifdef WMR_UPDATE_CHECK
    // Reached ONLY after a dispatched subcommand (the no-args path returned
    // before parse; --version/--help/parse errors returned from the
    // CLI::ParseError catch above and never reach here). Never throws, never
    // changes rc, writes only to stderr.
    wmr::maybe_check_for_update(opts.no_update_check);
#endif
    return rc;
}

} // namespace wmr
