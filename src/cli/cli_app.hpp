#pragma once

#include <string>
#include <optional>
#include <utility>
#include <vector>

#include "core/types.hpp"
#include "core/inpaint.hpp"
#include "detection/still_geometry.hpp"  // StillGeometryOverride

namespace wmr {

enum class CliMode {
    AutoRemove,
    Detect,
    VisibleOnly,
    SynthidOnly,
    BuildCodebook,
    Video,
};

struct CliOptions {
    CliMode mode = CliMode::AutoRemove;
    std::string input_path;
    std::string output_path;
    bool force = false;
    bool force_small = false;
    bool force_large = false;
    bool verbose = false;
    bool detect_only = false;
    float inpaint_strength = 0.85f;
    bool synthid = false;
    std::string codebook_path;
    float synthid_strength = 0.50f;
    bool recursive = false;
    bool codebook_free = false;
    bool phase_adaptive = false;
    bool lab_a = false;               // WS3 experiment: operate on LAB `a` channel only (synthid path)
    bool no_content_guard = false;     // synthid: bypass the content-image skip so a codebook acts on content (evaluation)
    bool legacy_profile = false;       // video: Veo legacy text profile
    bool notebooklm_profile = false;    // video: NotebookLM profile
    std::string notebooklm_rect_str;     // video: manual rect override "x,y,w,h"
    double notebooklm_complexity_threshold = 15.0; // video: --complexity-threshold (NS vs MI-GAN gate)
    std::string notebooklm_method = "auto"; // video: --notebooklm-method {auto|ns|migan}
    bool still_legacy = false;         // still images: pin legacy V1 (Gemini pre-3.5)
    bool still_no_legacy = false;      // still images: pin current V2, disable V2→V1 fallback
    std::string still_rect_str;        // still images: --rect "x,y,w,h" manual override
    std::string still_geo_preset;      // still images: --geo-preset <name>
    bool still_no_auto_geometry = false; // still images: --no-auto-geometry
    std::string video_variant_str;
    bool no_auto_geometry = false;   // video: --no-auto-geometry (skip content-based geometry search)
    bool no_edge_cleanup = false;    // video: --no-edge-cleanup (pure reverse-blend, no edge cleanup)
    bool scenes = false;
    double scene_threshold = 0.30;
    int video_crf = 14;
    std::string video_preset = "slow";
    std::string video_codec = "libx264";
    // Residual-cleanup method. Default "off" = pure reverse-alpha-blend (the exact
    // mathematical reversal, no blur). Opt in to a residual-only cleanup with
    // soft|ns|telea (or ai on AI builds). Always bound to --denoise now.
    std::string denoise_method = "off";
    float denoise_sigma = 50.0f;        // --sigma 1-150
    float denoise_strength_pct = 120.0f; // --strength 0-300 (percent; /100 internally)
    int denoise_radius = 10;            // --radius 1-25

    // build-codebook: opt-in carrier-bin seeding. Parsed from --carrier-grid
    // "x1,y1;x2,y2;..." (FFT-bin coords on the per-profile rows x cols grid).
    // Empty (flag absent) = no seeding = today's behavior.
    std::string carrier_grid_str;
};

// Resolve the still-image profile variant from CLI flags.
// Returns {force_variant, try_v1_fallback}:
//   --legacy      → {V1, false}
//   --no-legacy   → {V2, false}
//   (neither)     → {nullopt, true}  (default V2 with auto V2→V1 fallback)
std::pair<std::optional<WatermarkVariant>, bool>
resolve_still_variant(const CliOptions& opts);

// Resolve the residual-cleanup InpaintConfig from CLI opts (shared by the single-
// image and batch paths). Returns false when the user chose "--denoise off"
// (the caller then skips cleanup, reverse-blending only).
bool resolve_inpaint_config(const CliOptions& opts, InpaintConfig& out);

// Parse a "x,y,w,h" rect string. Returns nullopt for an empty OR malformed string;
// the caller distinguishes the two (empty = no flag, malformed = error).
std::optional<cv::Rect> parse_rect(const std::string& s);

// Parse a "x1,y1;x2,y2;..." carrier-bin list (FFT-bin coords on the per-profile
// rows x cols grid). Returns nullopt for an empty OR malformed string; the
// caller distinguishes the two (empty = no flag, malformed = user error). Empty
// entries (e.g. trailing ';') are tolerated.
std::optional<std::vector<std::pair<int,int>>> parse_bin_list(const std::string& s);

// Build the still-image geometry override from CLI opts. Returns false (with a
// logged error) when --rect was given but malformed.
bool resolve_still_geometry_override(const CliOptions& opts, StillGeometryOverride& out);

int run_cli(int argc, char* argv[]);

} // namespace wmr
