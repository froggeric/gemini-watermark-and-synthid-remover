#pragma once

#ifdef WMR_UPDATE_CHECK

#include <array>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

namespace wmr {
namespace fs = std::filesystem;

// (Helpers are added in later tasks. This stub only declares the entry point
//  so the build harness compiles.)

struct FetchResult {
    bool ok = false;
    int http_code = 0;
    std::string body;
    std::string etag;
    std::string error;
};
using FetchFn = std::function<FetchResult(const std::string& etag)>;

std::optional<std::array<unsigned, 3>> parse_version(std::string_view s);
// -1 if current<latest, 0 if equal, +1 if current>latest. A malformed current
// yields +1 ("never newer"). A malformed latest yields +1 (cannot confirm newer).
int compare_versions(std::string_view current, std::string_view latest);
std::string parse_tag(std::string_view tag_name);  // strip one leading v/V
std::optional<std::string> parse_release_json(std::string_view body);

bool should_show(bool no_update_flag, bool env_no_update, bool env_ci,
                 bool env_do_not_track, bool is_tty);
bool should_fetch(long long age_s, long long interval_s);
// Pure color policy: the same logic as color_enabled(), but with the TTY result
// and the two env values passed in, so it is fully unit-testable with no
// isatty/getenv dependency (a non-TTY CI runner would otherwise force
// color_enabled() false regardless of NO_COLOR/TERM, making AC-9 untestable).
// no_color/term may be nullptr (getenv's "unset" return).
bool color_enabled_for(bool is_tty, const char* no_color, const char* term);
bool color_enabled();

// Notify-only update check. Never throws, never writes to stdout, never changes
// the process exit code. `no_update_check` is the --no-update-check flag value.
void maybe_check_for_update(bool no_update_check,
                            FetchFn fetch = {});

}  // namespace wmr

#endif  // WMR_UPDATE_CHECK
