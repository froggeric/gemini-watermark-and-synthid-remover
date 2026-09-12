#pragma once

// The ONE copy of the still-image detect->remove policy, shared by the CLI
// single-image path, batch, and the GUI. Extracted from cli_app.cpp:194-309 and
// batch_processor.cpp:57-131 (previously duplicate copies; this is where the
// historical snap-override / fall-through leaks lived).
//
// Engine-level TU: no CliOptions, no CLI11, no cli/*.hpp, so it links into
// wmr_tests (the paths.cpp precedent). The CLI maps CliOptions ->
// StillRemoveOptions with a thin glue function (cli_app.cpp); the GUI builds
// StillRemoveOptions directly from its validated JSON options.

#include <opencv2/core.hpp>
#include <filesystem>
#include <optional>
#include <string>

#include "core/types.hpp"
#include "core/inpaint.hpp"
#include "core/watermark_engine.hpp"
#include "detection/still_geometry.hpp"

namespace wmr {

// Options for the shared still-image remove policy. Defaults mirror the CLI
// (the resolve_inpaint_config mapping): denoise off, sigma 50, strength 120%,
// radius 10. The GUI never overrides them in v1.
struct StillRemoveOptions {
    bool force = false;                             // --force
    std::optional<WatermarkSize> force_size;        // --force-small / --force-large
    std::optional<WatermarkVariant> force_variant;  // --legacy / --no-legacy
    bool try_v1_fallback = true;                    // neither legacy flag: V2 then V1
    StillGeometryOverride geometry;                 // rect / preset / no_auto_geometry
    std::string denoise_method = "off";             // off|soft|ns|telea|ai
    float denoise_sigma = 50.0f;
    float denoise_strength_pct = 120.0f;
    int denoise_radius = 10;
};

enum class StillOutcome { Removed, NoWatermark, Failed };

struct StillRemoveOutcome {
    StillOutcome outcome = StillOutcome::NoWatermark;
    bool forced = false;                  // removal ran because force was set
    std::optional<cv::Rect> bbox;         // detection region (nullopt when nothing detected)
    float score = 0.0f;                   // detection confidence in [0,1]
    std::string geometry_source;          // rect|preset|auto/snapped|auto/raw|model; "" when force
    std::string variant;                  // "V1" | "V2"; "" when nothing ran
    std::string error;                    // non-empty iff Failed
};

// Detect + remove IN PLACE (the engine convention: cv::Mat& like every
// remover). The caller clones the original first if it needs the pre-removal
// pixels. Never throws: any exception maps to Failed with the message.
StillRemoveOutcome remove_still(WatermarkEngine& engine, cv::Mat& image,
                                const StillRemoveOptions& opts);

// The shared write tail: codec params by extension (JPEG 100 / PNG 6 / WebP
// 101), then the post-write provenance strip unless keep_provenance. Returns
// false on write failure (callers map that to a failure). Callers of this TU
// must not call cv::imwrite directly.
bool write_still_output(const std::filesystem::path& path, const cv::Mat& image,
                        bool keep_provenance);

// Moved option helpers (were in cli_app.{hpp,cpp}).
// Parse a "x,y,w,h" rect string; nullopt on malformed (an empty string is
// "no rect", not an error).
std::optional<cv::Rect> parse_rect(const std::string& s);
// Resolve the residual-cleanup InpaintConfig. Returns false for "off"
// (skip cleanup entirely; reverse-blend only).
bool resolve_inpaint_config(const StillRemoveOptions& opts, InpaintConfig& out);

} // namespace wmr
