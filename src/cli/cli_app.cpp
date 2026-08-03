#include "cli/cli_app.hpp"
#include "cli/batch_processor.hpp"
#include "core/watermark_engine.hpp"
#include "core/types.hpp"
#include "core/fft_context.hpp"
#include "synthid/spectral_codebook.hpp"
#include "synthid/codebook_subtractor.hpp"
#include "synthid/noise_residual_subtractor.hpp"
#include "synthid/codebook_builder.hpp"
#include "detection/synthid_detector.hpp"
#include "video/video_processor.hpp"

#include <opencv2/imgcodecs.hpp>
#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <filesystem>
#include <iostream>
#include <iterator>
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
    os << "--------------------------------------------\n"
        << "  wmr v" APP_VERSION " — watermark remover\n"
        << "  Remove Gemini/Veo visible watermarks\n"
        << "  and suppress SynthID invisible watermarks (heuristic; not a verifiable removal).\n"
        << "  --synthid-attack regen is lossy (SDXL img2img; ~6.5 GB model + ~335 MB VAE download on first use).\n"
        << "  https://github.com/froggeric/gemini-watermark-and-synthid-remover\n"
        << "  Copyright 2026 Frederic Guigand\n"
        << "--------------------------------------------\n\n";
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

// Parse a "x1,y1;x2,y2;..." carrier-bin list (FFT-bin coords on the per-profile
// rows x cols grid). Returns nullopt for an empty OR malformed string; the
// caller distinguishes the two. Empty entries (e.g. a trailing ';') are tolerated.
// Each bin must parse as two non-negative integers separated by ',' (e.g.
// "84,0;168,0"). Whitespace tolerance: std::istringstream::operator>> skips
// leading whitespace within a token, so " 84 , 0" parses the same as "84,0".
// Whitespace AROUND a token (after a ';') is also fine. A token with no ','
// (e.g. "84") or non-numeric chars (e.g. "not_a_bin") is rejected as malformed.
std::optional<std::vector<std::pair<int,int>>> parse_bin_list(const std::string& s) {
    if (s.empty()) return std::nullopt;
    std::vector<std::pair<int,int>> bins;

    // Split on ';' manually. Whitespace inside a token is handled by
    // istringstream's >> above (it skips leading ws within the token); the
    // separator between x and y must still be a literal ','.
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(';', i);
        if (j == std::string::npos) j = s.size();
        if (j > i) {  // skip empty entries (consecutive ';' or trailing)
            const std::string token = s.substr(i, j - i);
            int x, y;
            char sep;
            std::istringstream ss(token);
            if (ss >> x >> sep >> y && sep == ',' && x >= 0 && y >= 0) {
                bins.emplace_back(x, y);
            } else {
                return std::nullopt;  // malformed entry → user error
            }
        }
        i = (j == s.size()) ? j : j + 1;
    }

    if (bins.empty()) return std::nullopt;  // only ';' entries → treat as malformed
    return bins;
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
            const bool snap = fp.has_value() || (is_v2 && sz == WatermarkSize::Small);
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

    // SynthID detection
    if (!opts.codebook_path.empty()) {
        FftContext fft;
        SpectralCodebook codebook;
        try {
            codebook.load(opts.codebook_path);
        } catch (const std::exception& e) {
            spdlog::error("Cannot load codebook: {}", e.what());
            return 1;
        }

        SynthidDetector detector(fft);
        auto result = detector.detect(image, codebook);
        if (result.detected) {
            spdlog::info("[SYNTHID] DETECTED (confidence: {:.1f}%)",
                         result.confidence * 100.0f);
            spdlog::info("  noise_corr={:.3f} carrier_phase={:.3f} "
                         "struct_ratio={:.3f} ms_consistency={:.3f}",
                         result.noise_correlation, result.carrier_phase_score,
                         result.structure_ratio, result.multi_scale_consistency);
        } else {
            spdlog::info("[SYNTHID] not detected ({:.1f}%)",
                         result.confidence * 100.0f);
        }
    }

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
    if (opts.mode == CliMode::AutoRemove || opts.mode == CliMode::VisibleOnly) {
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
            bool snap = force_pos.has_value() ||
                (v == WatermarkVariant::V2 &&
                 force_size.value_or(get_watermark_size(image.cols, image.rows))
                     == WatermarkSize::Small);
            auto detection = engine.detect_watermark(image, force_size, force_pos,
                                                     alpha_override, v, snap);
            if (!detection.detected) {
                if (!(explicit_override && force_pos.has_value())) return false;
                spdlog::info("Removing at overridden position "
                             "(confidence {:.1f}% below the detection gate)",
                             detection.confidence * 100.0f);
            } else {
                spdlog::info("Visible watermark detected ({:.1f}%, {}), removing...",
                             detection.confidence * 100.0f,
                             v == WatermarkVariant::V1 ? "V1" : "V2");
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

    // SynthID suppression. Branches on opts.synthid_attack:
    //   "regen"    = SDXL img2img over the WHOLE image (lossy; the only *validated*
    //                SynthID scrub). Skips the spectral codebook path entirely.
    //   "off"      = skip the synthid pass even with --synthid set.
    //   "spectral" = today's frequency-domain heuristic (DEFAULT, byte-identical to
    //                pre-regen behavior when no --synthid-attack flag is passed).
    if (opts.mode == CliMode::SynthidOnly ||
        (opts.mode == CliMode::AutoRemove && opts.synthid)) {

        if (opts.synthid_attack == "regen") {
#ifdef WMR_BUILD_REGEN
            // regen = SDXL img2img over the WHOLE image (a SynthID-only scrub). It does
            // NOT remove the visible Gemini diamond (strength ~0.05 cannot); in AutoRemove
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
            spdlog::info("SynthID regen complete (lossy; not a verifiable removal)");
            did_work = true;
#else
            spdlog::error("--synthid-attack regen: this wmr build is regen-free (WMR_BUILD_REGEN off). "
                          "Rebuild with WMR_BUILD_REGEN=1, or use --synthid-attack spectral.");
            return 1;
#endif
        } else if (opts.synthid_attack == "off") {
            spdlog::info("SynthID attack is 'off'; skipping the synthid pass");
        } else {
            // ---- DEFAULT "spectral": today's frequency-domain path (codebook/codebook-free) ----
            if (opts.codebook_path.empty() && !opts.codebook_free) {
                spdlog::error("SynthID suppression requires --codebook <path> or --codebook-free");
                return 1;
            }

            FftContext fft;
            RemovalConfig config;
            config.custom_strength = opts.synthid_strength;
            config.phase_adaptive = opts.phase_adaptive;
            config.lab_a = opts.lab_a;
            config.no_content_guard = opts.no_content_guard;

            if (!opts.codebook_path.empty()) {
                SpectralCodebook codebook;
                codebook.load(opts.codebook_path);

                if (!opts.force) {
                    SynthidDetector detector(fft);
                    auto det = detector.detect(image, codebook,
                        opts.lab_a ? SynthidDetector::ColorSpace::LabA
                                   : SynthidDetector::ColorSpace::BGR);
                    if (!det.detected) {
                        spdlog::debug("No SynthID detected ({:.1f}%)", det.confidence * 100.0f);
                        if (opts.mode == CliMode::SynthidOnly) {
                            spdlog::warn("No SynthID detected. Use --force to suppress anyway.");
                            return 2;
                        }
                    } else {
                        spdlog::info("SynthID detected ({:.1f}%), suppressing carrier...",
                                     det.confidence * 100.0f);
                    }
                }

                CodebookSubtractor subtractor(fft);
                subtractor.remove_synthid(image, codebook, config);
            } else {
                spdlog::info("Using codebook-free suppression (noise residual estimation)");
                NoiseResidualSubtractor subtractor(fft);
                subtractor.remove_synthid(image, config);
            }

            spdlog::info("SynthID suppression complete (frequency-domain heuristic; not a verifiable removal)");
            did_work = true;
        }
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

    spdlog::info("Saved: {}", output);
    return 0;
}

static int process_build_codebook(const CliOptions& opts) {
    FftContext fft;
    CodebookBuilder builder(fft);

    // Parse --carrier-grid (opt-in carrier-bin seeding). Empty = no seeding
    // (today's behavior). Malformed = user error.
    if (!opts.carrier_grid_str.empty()) {
        auto bins = parse_bin_list(opts.carrier_grid_str);
        if (!bins) {
            spdlog::error("Invalid --carrier-grid format. Expected: x1,y1;x2,y2;... "
                         "(e.g. --carrier-grid 84,0;168,0)");
            return 1;
        }
        builder.set_carrier_bins(*bins);
        spdlog::info("Carrier-bin seeding: {} bin(s) will be applied to every fresh profile",
                     bins->size());
    }

    auto stats = builder.build_from_directory(opts.input_path, opts.output_path);

    spdlog::info("Build complete: {} images → {} profiles ({} low-sample)",
                 stats.total_images, stats.profiles_created,
                 stats.skipped_low_samples);

    return stats.profiles_created > 0 ? 0 : 1;
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
    auto add_synthid_attack = [&](CLI::App* cmd) {
        cmd->add_option("--synthid-attack", opts.synthid_attack,
                        synthid_attack_help_text())
            ->capture_default_str()
            ->check(CLI::IsMember({"off", "spectral", "regen"}));
        cmd->add_option("--regen-strength", opts.regen_strength,
                        "regen img2img denoising strength 0.02-0.15 "
                        "(lower = closer to original, weaker scrub)")
            ->capture_default_str()
            ->check(CLI::Range(0.02f, 0.15f));
        cmd->add_option("--regen-steps", opts.regen_steps,
                        "regen sample steps (fewer is faster; ~8-30)")
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
    remove_cmd->add_flag("--synthid", opts.synthid, "Also suppress SynthID carrier");
    remove_cmd->add_option("--codebook", opts.codebook_path, "Spectral codebook path (.wcb)");
    remove_cmd->add_flag("--codebook-free", opts.codebook_free,
                          "Estimate carrier from noise residual (no codebook needed)");
    remove_cmd->add_flag("--lab-a", opts.lab_a,
                         "WS3 experiment: suppress in the LAB `a` channel only (with --codebook-free/--codebook)");
    remove_cmd->add_flag("--no-content-guard", opts.no_content_guard,
                         "Bypass the content-image guard so the codebook acts on content (evaluation)");
    remove_cmd->add_option("--synthid-strength", opts.synthid_strength,
                           "SynthID suppression strength 0.0-2.0")
        ->check(CLI::Range(0.0f, 2.0f));
    remove_cmd->add_option("--inpaint-strength", opts.inpaint_strength,
                           "Inpaint strength 0.0-1.0")
        ->check(CLI::Range(0.0f, 1.0f));
    add_denoise(remove_cmd);
    add_synthid_attack(remove_cmd);
    remove_cmd->add_flag("-r,--recursive", opts.recursive, "Process directories recursively");
    remove_cmd->add_option("-o,--output", opts.output_path, "Output path (required for files; batch defaults to cleaned/)");
    add_still_geometry(remove_cmd);
    add_common(remove_cmd);

    // --- detect ---
    auto* detect_cmd = app.add_subcommand("detect", "Detect watermarks without modifying");
    detect_cmd->add_option("input", opts.input_path, "Input image")
        ->required()
        ->check(CLI::ExistingFile);
    detect_cmd->add_option("--codebook", opts.codebook_path, "Spectral codebook for SynthID detection");
    detect_cmd->add_flag("--legacy", opts.still_legacy,
                         "Report only the legacy Gemini (pre-3.5) V1 profile");
    detect_cmd->add_flag("--no-legacy", opts.still_no_legacy,
                         "Report only the current (Gemini 3.5+) V2 profile");
    add_still_geometry(detect_cmd);
    add_common(detect_cmd);

    // --- visible ---
    auto* visible_cmd = app.add_subcommand("visible", "Remove visible watermark only");
    visible_cmd->add_option("input", opts.input_path, "Input image")
        ->required()
        ->check(CLI::ExistingFile);
    visible_cmd->add_flag("-f,--force", opts.force, "Skip detection");
    visible_cmd->add_flag("--force-small", opts.force_small, "Force 48x48");
    visible_cmd->add_flag("--force-large", opts.force_large, "Force 96x96");
    visible_cmd->add_flag("--legacy", opts.still_legacy,
                          "Use legacy Gemini (pre-3.5) V1 watermark profile");
    visible_cmd->add_flag("--no-legacy", opts.still_no_legacy,
                          "Pin current (Gemini 3.5+) V2 profile; disable auto fallback");
    visible_cmd->add_option("--inpaint-strength", opts.inpaint_strength,
                            "Inpaint strength 0.0-1.0")
        ->check(CLI::Range(0.0f, 1.0f));
    add_denoise(visible_cmd);
    visible_cmd->add_option("-o,--output", opts.output_path, "Output path (required)");
    add_still_geometry(visible_cmd);
    add_common(visible_cmd);

    // --- synthid ---
    auto* synthid_cmd = app.add_subcommand("synthid", "Suppress SynthID invisible watermark (frequency-domain heuristic; not a verifiable removal)");
    synthid_cmd->add_option("input", opts.input_path, "Input image")
        ->required()
        ->check(CLI::ExistingFile);
    synthid_cmd->add_flag("-f,--force", opts.force, "Skip detection");
    synthid_cmd->add_option("--codebook", opts.codebook_path, "Spectral codebook path (.wcb)");
    synthid_cmd->add_flag("--codebook-free", opts.codebook_free,
                          "Estimate carrier from noise residual (no codebook needed)");
    synthid_cmd->add_flag("--phase-adaptive", opts.phase_adaptive,
                          "Use image's own phase for uniform images (conjugate subtraction)");
    synthid_cmd->add_flag("--lab-a", opts.lab_a,
                          "WS3 experiment: suppress in the LAB `a` channel only");
    synthid_cmd->add_flag("--no-content-guard", opts.no_content_guard,
                          "Bypass the content-image guard so the codebook acts on content (evaluation)");
    synthid_cmd->add_option("--synthid-strength", opts.synthid_strength,
                            "SynthID suppression strength 0.0-2.0")
        ->check(CLI::Range(0.0f, 2.0f));
    synthid_cmd->add_option("-o,--output", opts.output_path, "Output path (required)");
    add_synthid_attack(synthid_cmd);
    add_common(synthid_cmd);

    // --- build-codebook ---
    auto* build_cmd = app.add_subcommand("build-codebook", "Build SynthID codebook from reference images");
    build_cmd->add_option("input", opts.input_path, "Directory of reference images")
        ->required()
        ->check(CLI::ExistingDirectory);
    build_cmd->add_option("-o,--output", opts.output_path, "Output codebook path")
        ->required();
    build_cmd->add_option("--carrier-grid", opts.carrier_grid_str,
                           "Seed candidate SynthID carrier bins in every fresh profile. "
                           "Format: \"x1,y1;x2,y2;...\" (FFT-bin coords on the per-profile "
                           "rows x cols grid). Sets consistency + phase_consistency = 1.0 "
                           "at each bin so the subtractor acts there. Opt-in; absent = no "
                           "seeding (default).");
    add_common(build_cmd);

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
    add_common(video_cmd);

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

    if (opts.force_small && opts.force_large) {
        spdlog::error("Cannot use both --force-small and --force-large");
        return 1;
    }
    if (opts.still_legacy && opts.still_no_legacy) {
        spdlog::error("Cannot use both --legacy and --no-legacy");
        return 2;
    }

    // Determine mode from subcommand
    if (detect_cmd->parsed()) {
        opts.mode = CliMode::Detect;
    } else if (visible_cmd->parsed()) {
        opts.mode = CliMode::VisibleOnly;
    } else if (synthid_cmd->parsed()) {
        opts.mode = CliMode::SynthidOnly;
    } else if (build_cmd->parsed()) {
        opts.mode = CliMode::BuildCodebook;
    } else if (video_cmd->parsed()) {
        opts.mode = CliMode::Video;
    } else {
        opts.mode = CliMode::AutoRemove;
    }

    try {
        switch (opts.mode) {
            case CliMode::Detect:
                return process_detect(opts);

            case CliMode::BuildCodebook:
                return process_build_codebook(opts);

            case CliMode::Video:
                return process_video(opts);

            case CliMode::AutoRemove:
            case CliMode::VisibleOnly:
            case CliMode::SynthidOnly: {
                // Check if input is a directory → batch mode
                if (std::filesystem::is_directory(opts.input_path)) {
                    auto result = batch_process(opts);
                    return result.failed > 0 ? 1 : 0;
                }
                return process_single_image(opts);
            }
        }
    } catch (const std::exception& e) {
        spdlog::error("Error: {}", e.what());
        return 1;
    }

    return 0;
}

} // namespace wmr
