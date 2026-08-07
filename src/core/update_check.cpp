#ifdef WMR_UPDATE_CHECK
#include "core/update_check.hpp"

#include "cli/cli_app.hpp"  // wmr::kHeaderRule
#include "core/paths.hpp"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <system_error>

#include <curl/curl.h>

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

// libcurl write/header buffers for fetch_latest_release. Mirrors the
// model_downloader.cpp idiom (separate body + ETag capture).
struct CurlBuf {
    std::string body;
    std::string etag;
};
size_t uc_write_cb(char* ptr, size_t sz, size_t n, void* ud) {
    auto* b = static_cast<CurlBuf*>(ud);
    b->body.append(ptr, sz * n);
    return sz * n;
}
size_t uc_header_cb(char* buf, size_t sz, size_t n, void* ud) {
    auto* b = static_cast<CurlBuf*>(ud);
    size_t len = sz * n;
    std::string h(buf, len);
    // Match "ETag:" (case-insensitive on the key prefix).
    if (len > 5 && (h[0] == 'E' || h[0] == 'e') &&
        (h[1] == 'T' || h[1] == 't') && (h[2] == 'a' || h[2] == 'A') &&
        (h[3] == 'g' || h[3] == 'G') && h[4] == ':') {
        std::string v = h.substr(5);
        // trim whitespace + trailing CR/LF
        while (!v.empty() && (v.front() == ' ' || v.front() == '\t')) v.erase(0, 1);
        while (!v.empty() && (v.back() == '\r' || v.back() == '\n' || v.back() == ' ')) v.pop_back();
        b->etag = v;
    }
    return len;
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

namespace {
// Tiny JSON string escaper for writing the cache.
std::string esc(std::string_view s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') { out.push_back('\\'); out.push_back(c); }
        else if (c == '\n') out += "\\n";
        else out.push_back(c);
    }
    return out;
}
}  // namespace

CacheData read_cache(const fs::path& p) {
    CacheData d;
    std::ifstream f(p);
    if (!f) return d;
    std::stringstream ss; ss << f.rdbuf();
    std::string body = ss.str();
    // Reuse the string-field extractor for the two string fields; epoch is numeric.
    if (auto v = extract_string_field(body, "latest_version")) d.latest_version = *v;
    if (auto v = extract_string_field(body, "etag")) d.etag = *v;
    // last_check_epoch: scan for the numeric value after "last_check_epoch".
    auto pos = body.find("\"last_check_epoch\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            d.last_check_epoch = std::strtoll(body.c_str() + pos + 1, nullptr, 10);
        }
    }
    return d;
}

bool write_cache(const fs::path& p, const CacheData& d) {
    std::error_code ec;
    fs::create_directories(p.parent_path(), ec);
    std::ostringstream ss;
    ss << "{\"last_check_epoch\":" << d.last_check_epoch
       << ",\"latest_version\":\"" << esc(d.latest_version) << "\""
       << ",\"etag\":\"" << esc(d.etag) << "\"}";
    std::string data = ss.str();
    // PID-suffixed temp in the SAME directory => rename is atomic (same fs).
    // Use the global getpid() from <unistd.h> (POSIX), NOT std::getpid (no such
    // name in std; <cstdlib> does not declare it). <unistd.h> is included via the
    // Task 3 non-Windows branch; on Windows <process.h> provides _getpid().
    std::string pid;
#ifdef _WIN32
    pid = std::to_string(_getpid());
#else
    pid = std::to_string(getpid());
#endif
    std::string tmpname = p.filename().string() + ".tmp." + pid;
    fs::path tmp = p.parent_path() / tmpname;
    {
        std::ofstream of(tmp, std::ios::binary);
        if (!of) return false;
        of << data;
        of.flush();
        if (!of) { std::error_code rm; fs::remove(tmp, rm); return false; }
    }
    fs::rename(tmp, p, ec);  // atomic replace on POSIX; MOVEFILE_REPLACE_EXISTING on Windows
    if (ec) { std::error_code rm; fs::remove(tmp, rm); return false; }
    return true;
}

std::string format_notice(std::string_view current, std::string_view latest, bool color) {
    std::ostringstream os;
    const char* B = color ? "\033[1m\033[33m" : "";  // bold + yellow
    const char* R = color ? "\033[0m" : "";
    os << B << wmr::kHeaderRule << R << "\n"
       << B << "  A new release of wmr is available: " << current << " -> " << latest << R << "\n"
       << B << "  Download: https://github.com/froggeric/gemini-watermark-and-synthid-remover/releases/latest" << R << "\n"
       << B << "  Disable with WMR_NO_UPDATE_CHECK=1 or --no-update-check." << R << "\n"
       << B << wmr::kHeaderRule << R << "\n";
    return os.str();
}

// HTTPS GET of /releases/latest. Zero payload: versionless UA, no query string,
// no body, no version/OS/arch/id header. CURLOPT_ACCEPT_ENCODING "" is REQUIRED:
// api.github.com returns gzip by default and without it the body arrives as raw
// gzip bytes (parse_release_json then silently fails to find tag_name).
FetchResult fetch_latest_release(const std::string& etag) {
    FetchResult r;
    CURL* curl = curl_easy_init();
    if (!curl) { r.error = "curl_easy_init failed"; return r; }
    CurlBuf buf;
    constexpr const char* kUrl =
        "https://api.github.com/repos/froggeric/gemini-watermark-and-synthid-remover/releases/latest";
    struct curl_slist* hdrs = nullptr;
    hdrs = curl_slist_append(hdrs, "Accept: application/vnd.github+json");
    std::string inm;
    if (!etag.empty()) { inm = "If-None-Match: " + etag; hdrs = curl_slist_append(hdrs, inm.c_str()); }

    curl_easy_setopt(curl, CURLOPT_URL, kUrl);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdrs);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "wmr");           // versionless
    curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "");         // auto-decompress gzip (REQUIRED)
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, 1500L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 3000L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, uc_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, uc_header_cb);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, &buf);

    CURLcode rc = curl_easy_perform(curl);
    long code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
    curl_slist_free_all(hdrs);
    curl_easy_cleanup(curl);

    if (rc != CURLE_OK) { r.error = std::string("curl: ") + curl_easy_strerror(rc); return r; }
    r.ok = true;
    r.http_code = static_cast<int>(code);
    r.body = std::move(buf.body);
    r.etag = std::move(buf.etag);
    return r;
}

namespace {
long long now_epoch() {
    using namespace std::chrono;
    return duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
}
long long parse_interval_env() {
    const char* v = std::getenv("WMR_UPDATE_CHECK_INTERVAL");
    if (!v || !*v) return 86400;
    char* end = nullptr;
    errno = 0;
    // strtoll (not strtoul) so the return type matches `long long` with no
    // implementation-defined cast: a value > LLONG_MAX trips ERANGE -> default
    // (instead of wrapping to a negative and making should_fetch always true),
    // and a leading '-' parses negative -> rejected below (spec: huge = never
    // fetch; negative is not a valid throttle).
    long long n = std::strtoll(v, &end, 10);
    if (end == v || errno == ERANGE || n < 0) return 86400;  // empty/garbage/overflow/negative
    return n;
}
#ifndef APP_VERSION
#define APP_VERSION "0.0.0"
#endif
}  // namespace

// Gate-free core. Deterministic + unit-testable (takes the cache path and interval
// explicitly, reads no TTY/CI env here). Writes only to stderr; never throws.
void run_update_check(const fs::path& cache_path, long long interval_s,
                      bool color, FetchFn fetch) {
    CacheData cd = read_cache(cache_path);
    bool printed = false;

    // 1) Show from cache if a known newer version exists (no network).
    if (!cd.latest_version.empty() &&
        compare_versions(APP_VERSION, cd.latest_version) < 0) {
        std::fputs(format_notice(APP_VERSION, cd.latest_version, color).c_str(), stderr);
        printed = true;
    }

    // 2) Fetch only if the cache is stale.
    long long now = now_epoch();
    long long age = now - cd.last_check_epoch;
    if (should_fetch(age, interval_s) && fetch) {
        FetchResult fr = fetch(cd.etag);
        if (fr.ok && fr.http_code == 200) {
            // parse_tag strips the leading v/V so the stored value AND the notice
            // show "1.16.4" (matching the spec cache schema + notice format), not
            // "v1.16.4". compare_versions is v-tolerant either way, but display is not.
            if (auto tag = parse_release_json(fr.body)) cd.latest_version = parse_tag(*tag);
            // else: malformed body -> keep previous latest_version
        }  // 304 or any failure: keep previous latest_version
        if (!fr.etag.empty()) cd.etag = fr.etag;
        cd.last_check_epoch = now;
        write_cache(cache_path, cd);

        if (!printed && !cd.latest_version.empty() &&
            compare_versions(APP_VERSION, cd.latest_version) < 0) {
            std::fputs(format_notice(APP_VERSION, cd.latest_version, color).c_str(), stderr);
        }
    }
}

// The gated entry point run_cli calls. Evaluates the opt-out / CI / TTY gate
// (should_show); if eligible, resolves the cache path + interval + color and
// delegates to run_update_check. Flushes stderr before returning. Never throws.
void maybe_check_for_update(bool no_update_check, FetchFn fetch) {
    try {
        if (!should_show(no_update_check,
                         env_nonempty("WMR_NO_UPDATE_CHECK"),
                         env_nonempty("CI"),
                         env_equals("DO_NOT_TRACK", "1"),
                         stderr_is_tty())) {
            return;  // not eligible: no network, no notice
        }
        fs::path cache_path = user_cache_dir() / "update-check.json";
        run_update_check(cache_path, parse_interval_env(), color_enabled(), fetch);
    } catch (...) {
        // never propagate past the boundary
    }
    std::fflush(stderr);
}

}  // namespace wmr
#endif  // WMR_UPDATE_CHECK
