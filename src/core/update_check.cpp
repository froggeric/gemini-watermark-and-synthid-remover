#ifdef WMR_UPDATE_CHECK
#include "core/update_check.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>

#ifdef _WIN32
#include <io.h>       // _isatty, _fileno
#include <cstdio>     // stderr (for _fileno)
#include <process.h>  // _getpid (used by write_cache's PID-suffixed temp)
#else
#include <unistd.h>   // getpid, STDERR_FILENO
#endif

namespace wmr {

namespace {
bool env_nonempty(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0';
}
bool env_equals(const char* name, const char* want) {
    const char* v = std::getenv(name);
    return v != nullptr && std::strcmp(v, want) == 0;
}
bool stderr_is_tty() {
#ifdef _WIN32
    return _isatty(_fileno(stderr)) != 0;
#else
    return isatty(STDERR_FILENO) != 0;
#endif
}
}  // namespace

std::optional<std::array<unsigned, 3>> parse_version(std::string_view s) {
    if (!s.empty() && (s[0] == 'v' || s[0] == 'V')) s.remove_prefix(1);
    if (s.empty()) return std::nullopt;
    std::array<unsigned, 3> v{0, 0, 0};
    size_t idx = 0, i = 0;
    while (idx < 3 && i < s.size()) {
        std::string tok;
        while (i < s.size() && s[i] != '.') tok.push_back(s[i++]);
        // Reject empty tokens ("1..2" -> empty between the dots) and tokens that do
        // not start with a digit. The digit check also defeats strtoul's documented
        // sign-skip: strtoul("-1") returns ULONG_MAX (it negates the magnitude) with
        // NO ERANGE on LP64, so without this guard "-1" would parse as a huge value.
        // It likewise rejects "+1", leading whitespace, and letters ("1.a.2" -> "a").
        if (tok.empty() || tok[0] < '0' || tok[0] > '9') return std::nullopt;
        char* end = nullptr;
        errno = 0;
        unsigned long n = std::strtoul(tok.c_str(), &end, 10);
        // Overflow handling must cover BOTH widths:
        //   - On LP64 (mac/linux 64-bit), unsigned long is 64-bit, so the test value
        //     99999999999 fits without ERANGE. The n > UINT_MAX check catches it
        //     (otherwise the static_cast<unsigned> below would silently truncate it
        //     to 1215751683 and the overflow test would fail).
        //   - On 32-bit / LLP64 where unsigned long is 32-bit, 99999999999 trips
        //     ERANGE (strtoul saturates at ULONG_MAX); the errno branch catches it.
        if (errno == ERANGE || n > static_cast<unsigned long>(std::numeric_limits<unsigned>::max()))
            return std::nullopt;
        // strtoul parses the numeric PREFIX and stops at the first non-digit, which is
        // how a pre-release suffix is dropped ("2-rc1" -> 2). That is intended; we do
        // NOT require *end == '\0'. tok[0] being a digit already guarantees >= 1 digit
        // was consumed (end != tok.c_str()).
        v[idx++] = static_cast<unsigned>(n);
        if (i < s.size() && s[i] == '.') ++i;
    }
    return v;
}

int compare_versions(std::string_view current, std::string_view latest) {
    auto c = parse_version(current);
    auto l = parse_version(latest);
    if (!c || !l) return 1;  // malformed => never report "newer"
    if (*c < *l) return -1;
    if (*c > *l) return 1;
    return 0;
}

std::string parse_tag(std::string_view tag_name) {
    if (!tag_name.empty() && (tag_name[0] == 'v' || tag_name[0] == 'V'))
        return std::string(tag_name.substr(1));
    return std::string(tag_name);
}

bool should_show(bool no_update_flag, bool env_no_update, bool env_ci,
                 bool env_do_not_track, bool is_tty) {
    return !no_update_flag && !env_no_update && !env_ci && !env_do_not_track && is_tty;
}

bool should_fetch(long long age_s, long long interval_s) {
    if (age_s < 0) age_s = 0;
    return age_s >= interval_s;
}

bool color_enabled_for(bool is_tty, const char* no_color, const char* term) {
    if (!is_tty) return false;
    if (no_color != nullptr && no_color[0] != '\0') return false;  // any non-empty disables
    if (term != nullptr && std::strcmp(term, "dumb") == 0) return false;
    return true;
}

bool color_enabled() {
    return color_enabled_for(stderr_is_tty(),
                             std::getenv("NO_COLOR"),
                             std::getenv("TERM"));
}

namespace {
// Find the string value of the top-level-ish "tag_name" field. Returns nullopt
// if not found or not a JSON string. Tolerant: does not need a full parser.
std::optional<std::string> extract_string_field(std::string_view body, std::string_view key) {
    // search for "key"
    for (size_t i = 0; i + key.size() + 2 < body.size(); ++i) {
        if (body[i] != '"') continue;
        if (body.substr(i + 1, key.size()) != key) continue;
        if (body[i + 1 + key.size()] != '"') continue;
        // found "key"; scan forward to the next ':' then the next '"'
        size_t j = i + 1 + key.size() + 1;
        while (j < body.size() && body[j] != ':') ++j;
        if (j == body.size()) return std::nullopt;
        ++j;  // past ':'
        while (j < body.size() && (body[j] == ' ' || body[j] == '\t' || body[j] == '\n' || body[j] == '\r')) ++j;
        if (j >= body.size() || body[j] != '"') return std::nullopt;  // not a string
        ++j;
        std::string out;
        while (j < body.size() && body[j] != '"') {
            if (body[j] == '\\' && j + 1 < body.size()) { out.push_back(body[j + 1]); j += 2; }
            else { out.push_back(body[j++]); }
        }
        if (j >= body.size()) return std::nullopt;  // unterminated
        return out;
    }
    return std::nullopt;
}
}  // namespace

std::optional<std::string> parse_release_json(std::string_view body) {
    return extract_string_field(body, "tag_name");
}

// Filled in across later tasks. The stub fetch + no-op orchestrator let the
// build harness link before any real logic exists.
FetchResult fetch_latest_release(const std::string& /*etag*/) {
    return {/*ok=*/false, /*http_code=*/0, /*body=*/"", /*etag=*/"", /*error=*/"unimplemented"};
}

// Resolves the default fetch when the caller omits it. Defined out-of-line so
// the header's default argument `FetchFn fetch = {}` can be replaced by a real
// default at the call site in run_cli (which passes fetch_latest_release).
void maybe_check_for_update(bool /*no_update_check*/, FetchFn /*fetch*/) {
    // no-op for now
}

}  // namespace wmr
#endif  // WMR_UPDATE_CHECK
