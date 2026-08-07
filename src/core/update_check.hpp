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

// Persisted cache of the last check result. read_cache returns an empty instance
// for any missing/truncated/wrong-type file (never throws).
struct CacheData {
    long long last_check_epoch = 0;  // unix seconds of last fetch attempt
    std::string latest_version;      // empty == unknown
    std::string etag;                // may be empty
};
CacheData read_cache(const fs::path& p);
bool write_cache(const fs::path& p, const CacheData& d);  // atomic; false on failure

// Render the three-line notice. current/latest are the version strings to show.
std::string format_notice(std::string_view current, std::string_view latest, bool color);

// HTTPS GET of /releases/latest. Returns the response body + the ETag header
// (empty if none). On any failure, ok=false with an error string. Zero payload:
// versionless UA, no query string, no body.
FetchResult fetch_latest_release(const std::string& etag);

// Gate-free cache + fetch + notice core. Reads cache_path, shows from cache if a
// known newer version exists, fetches (via fetch) only if the cache is stale,
// writes the cache, and shows if newly-found-newer. Takes an explicit cache path
// and interval so it is deterministic and unit-testable with no TTY/CI/env
// dependency. Never throws; writes only to stderr.
void run_update_check(const fs::path& cache_path, long long interval_s,
                      bool color, FetchFn fetch);

// The gated entry point run_cli calls. Evaluates the opt-out / CI / TTY gate
// (should_show); if eligible, resolves the cache path + interval + color and
// delegates to run_update_check. Flushes stderr before returning. Never throws.
void maybe_check_for_update(bool no_update_check,
                            FetchFn fetch = fetch_latest_release);

}  // namespace wmr

#endif  // WMR_UPDATE_CHECK
