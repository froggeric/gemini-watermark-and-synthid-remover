#pragma once
// The GUI API options matrix (Task 5), pure and unit-testable: no httplib, no
// I/O. The POST /api/jobs handler only hands it the parsed "options" part; the
// spec's stable error codes come back as strings so this TU stays HTTP-free.
// Spec: docs/superpowers/specs/2026-09-12-gui-webui-design.md ("Options and
// validation", "Invalid combinations").
#include <string>

#include "gui/vendored.hpp"  // nlohmann::json (single include point)

#include "gui/jobs.hpp"  // JobOptions (the output type)

namespace wmr::gui {

// Build-feature surface the API reports/validates against (GET /api/version
// echoes it; denoise "ai" is rejected when denoise_ai is false).
struct ApiFeatures {
    bool denoise_ai = false;
};

// The feature set of THIS binary, from the build gates. Built once at launch
// and threaded through register_routes; the pure validator takes it as data
// so tests can cover both sides of each gate.
inline ApiFeatures build_api_features() {
#if defined(WMR_AI_DENOISE) && WMR_AI_DENOISE
    return ApiFeatures{true};
#else
    return ApiFeatures{false};
#endif
}

// Validate a PARSED options object (must be a JSON object; the text entry
// below parses). Known keys only: denoise, legacy, geoPreset, rect,
// keepProvenance, forceRemove. Returns true and fills `out` on success; on
// failure returns false with error_code "invalid_option" or
// "invalid_combination" and a human message for the error envelope.
bool validate_job_options(const nlohmann::json& opts, const ApiFeatures& features,
                          JobOptions& out, std::string& error_code, std::string& message);

// The raw multipart "options" part: the 64 KB cap is checked BEFORE parsing
// (spec: "The options part is rejected 400 above 64 KB"), then JSON parse,
// then validate_job_options. An empty text (no options part) is valid and
// yields the defaults.
bool validate_options_part(const std::string& text, const ApiFeatures& features,
                           JobOptions& out, std::string& error_code, std::string& message);

}  // namespace wmr::gui
