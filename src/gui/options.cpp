#include "gui/options.hpp"

#include <cstdint>
#include <optional>
#include <utility>

#include "detection/still_geometry.hpp"  // kStillPresetNames (single source)

namespace wmr::gui {
namespace {

// Spec: "The options part is rejected 400 above 64 KB."
constexpr std::size_t kMaxOptionsBytes = 64 * 1024;

const char* kValidDenoise[] = {"off", "soft", "ns", "telea", "ai"};

// The spec pins this message verbatim for the legacy combinations.
const char* kLegacyComboMessage =
    "rect/geoPreset apply to the V2 profile only; legacy (V1) uses fixed positions";

bool fail(std::string& code, std::string& message, const char* c, std::string msg) {
    code = c;
    message = std::move(msg);
    return false;
}

// JSON integer -> int without silent wraparound (nlohmann stores wide
// unsigned/integer values; cv::Rect fields are 32-bit ints).
bool json_int(const nlohmann::json& v, int& out) {
    if (v.is_number_unsigned()) {
        const std::uint64_t u = v.get<std::uint64_t>();
        if (u > static_cast<std::uint64_t>(INT32_MAX)) return false;
        out = static_cast<int>(u);
        return true;
    }
    if (v.is_number_integer()) {  // signed (unsigned handled above)
        const std::int64_t n = v.get<std::int64_t>();
        if (n < INT32_MIN || n > INT32_MAX) return false;
        out = static_cast<int>(n);
        return true;
    }
    return false;
}

bool preset_known(const std::string& name) {
    // Validated against the calibrated table's name array, never a local list.
    for (const char* known : kStillPresetNames)
        if (name == known) return true;
    return false;
}

}  // namespace

bool validate_job_options(const nlohmann::json& opts, const ApiFeatures& features,
                          JobOptions& out, std::string& error_code, std::string& message) {
    out = JobOptions{};  // defaults; only rewritten on success below

    if (!opts.is_object())
        return fail(error_code, message, "invalid_option", "options must be a JSON object");

    bool legacy = false, force = false;
    std::optional<cv::Rect> rect;
    std::optional<std::string> preset;

    for (auto it = opts.begin(); it != opts.end(); ++it) {
        const std::string& key = it.key();
        const nlohmann::json& v = it.value();
        if (key == "denoise") {
            if (!v.is_string())
                return fail(error_code, message, "invalid_option", "denoise must be a string");
            const std::string d = v.get<std::string>();
            bool known = false;
            for (const char* k : kValidDenoise)
                if (d == k) known = true;
            if (!known)
                return fail(error_code, message, "invalid_option",
                            "denoise must be one of off, soft, ns, telea, ai");
            if (d == "ai" && !features.denoise_ai)
                return fail(error_code, message, "invalid_option",
                            "denoise \"ai\" is not available in this build");
            out.denoise = d;
        } else if (key == "legacy") {
            if (!v.is_boolean())
                return fail(error_code, message, "invalid_option", "legacy must be a boolean");
            legacy = v.get<bool>();
            out.legacy = legacy;
        } else if (key == "geoPreset") {
            if (!v.is_string())
                return fail(error_code, message, "invalid_option", "geoPreset must be a string");
            const std::string p = v.get<std::string>();
            if (!preset_known(p))
                return fail(error_code, message, "invalid_option",
                            "geoPreset must be one of the calibrated presets (GET "
                            "/api/version lists them)");
            preset = p;
            out.geo_preset = p;
        } else if (key == "rect") {
            if (!v.is_array() || v.size() != 4)
                return fail(error_code, message, "invalid_option",
                            "rect must be [x, y, w, h] with exactly four integers");
            int vals[4] = {0, 0, 0, 0};
            for (int i = 0; i < 4; ++i) {
                if (!json_int(v[i], vals[i]))
                    return fail(error_code, message, "invalid_option",
                                "rect must be [x, y, w, h] with exactly four integers");
            }
            if (vals[0] < 0 || vals[1] < 0)
                return fail(error_code, message, "invalid_option", "rect x and y must be >= 0");
            if (vals[2] < 8 || vals[3] < 8)
                return fail(error_code, message, "invalid_option", "rect w and h must be >= 8");
            rect = cv::Rect(vals[0], vals[1], vals[2], vals[3]);
            out.rect = *rect;
        } else if (key == "keepProvenance") {
            if (!v.is_boolean())
                return fail(error_code, message, "invalid_option",
                            "keepProvenance must be a boolean");
            out.keep_provenance = v.get<bool>();
        } else if (key == "forceRemove") {
            if (!v.is_boolean())
                return fail(error_code, message, "invalid_option",
                            "forceRemove must be a boolean");
            force = v.get<bool>();
            out.force = force;
        } else {
            return fail(error_code, message, "invalid_option",
                        "unknown option key: " + key);
        }
    }

    // The engine silently ignores these pairs (CLI parity); the API must not.
    if (legacy && (rect || preset))
        return fail(error_code, message, "invalid_combination", kLegacyComboMessage);
    if (force && (rect || preset))
        return fail(error_code, message, "invalid_combination",
                    "rect/geoPreset cannot be combined with forceRemove (force removes at "
                    "the model position)");
    if (rect && preset)
        return fail(error_code, message, "invalid_combination",
                    "rect and geoPreset are mutually exclusive; provide one");
    // legacy + forceRemove is valid (the documented --force --legacy combo).

    return true;
}

bool validate_options_part(const std::string& text, const ApiFeatures& features,
                           JobOptions& out, std::string& error_code, std::string& message) {
    out = JobOptions{};
    if (text.empty()) return true;  // no options part: defaults

    // The cap first: a hostile options part must never reach the parser.
    if (text.size() > kMaxOptionsBytes)
        return fail(error_code, message, "invalid_option", "options part exceeds 64 KB");

    const nlohmann::json j = nlohmann::json::parse(text, nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded())
        return fail(error_code, message, "invalid_option", "options is not valid JSON");
    return validate_job_options(j, features, out, error_code, message);
}

}  // namespace wmr::gui
