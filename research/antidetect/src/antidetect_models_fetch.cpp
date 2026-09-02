#ifdef WMR_BUILD_ANTIDETECT
#include "core/antidetect_models_fetch.hpp"

#include "core/model_downloader.hpp"
#include "core/paths.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#else
#include <unistd.h>
#endif

namespace wmr::antidetect {
namespace fs = std::filesystem;

namespace {

// The regenerator.cpp exe_dir (shared semantics; duplicated here so the
// ungated fetch TU does not link against regenerator.cpp).
fs::path exe_dir() {
#if defined(_WIN32)
    char buf[MAX_PATH];
    HMODULE m = nullptr;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, (LPCSTR)&exe_dir, &m);
    GetModuleFileNameA(m, buf, MAX_PATH);
    return fs::path(buf).parent_path();
#else
    char buf[4096];
# if defined(__APPLE__)
    uint32_t n = sizeof(buf);
    if (_NSGetExecutablePath(buf, &n) == 0) return fs::path(buf).parent_path();
# else
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = 0;
        return fs::path(buf).parent_path();
    }
# endif
    return fs::current_path();
#endif
}

// One model file: present + verified (sidecar fast path) or downloaded.
// Returns true when usable. Mirrors coreml_sd_model_fetch.cpp's fetch_one.
bool ensure_one(const SurrogateSpec& spec, const fs::path& dir, bool allow_download,
                std::string& note) {
    const fs::path dest = dir / spec.filename;

    if (spec.sha256 == nullptr || spec.sha256[0] == '\0') {
        if (!fs::exists(dest)) {
            note += spec.key;
            note += " not yet pinned; ";
            return false;
        }
        return true;  // unpinned but user-provided (env dir): trust it
    }
    const std::string pin(spec.sha256);

    // Warm path: file + sidecar matching the current pin (O(1), no re-hash).
    std::error_code ec;
    const std::string sidecar_path = dest.string() + ".sha256.ok";
    if (fs::exists(dest, ec) && fs::exists(sidecar_path, ec)) {
        std::ifstream f(sidecar_path);
        std::string stamped((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
        while (!stamped.empty() && (stamped.back() == '\n' || stamped.back() == '\r'))
            stamped.pop_back();
        if (stamped == pin) return true;
    }

    const auto r = download_pinned_file(spec.url, dest, pin, allow_download,
                                        /*allow_empty_hash=*/false,
                                        make_byte_progress(spec.filename));
    if (!r.ok) {
        note += spec.key;
        note += " unavailable (";
        note += r.error;
        note += "); ";
        return false;
    }
    std::ofstream(sidecar_path) << pin << "\n";
    return true;
}

std::vector<std::string> split_keys(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (const char c : s) {
        if (c == ',' || c == ' ') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

} // namespace

fs::path antidetect_models_dir() {
    // Resolution order mirrors the CoreML models dir; shared by the presence
    // check and the fetch so they can never disagree.
    if (const char* env = std::getenv("WMR_ANTIDETECT_MODELS_DIR");
        env && *env) {
        return fs::path(env);
    }
    std::error_code ec;
    fs::path p = exe_dir() / "antidetect";
    if (fs::exists(p, ec)) return p;
    p = exe_dir() / ".." / "share" / "wmr" / "antidetect";
    if (fs::exists(p, ec)) return p;
    return wmr::user_cache_dir() / "antidetect";
}

SurrogateFetchResult ensure_surrogate_models(const std::string& requested_keys,
                                             bool allow_download) {
    SurrogateFetchResult res;
    const fs::path dir = antidetect_models_dir();
    std::error_code ec;
    fs::create_directories(dir, ec);

    for (const std::string& key : split_keys(requested_keys)) {
        const SurrogateSpec* spec = find_surrogate(key);
        if (!spec) {
            res.note += "unknown surrogate '" + key + "'; ";
            continue;
        }
        if (ensure_one(*spec, dir, allow_download, res.note))
            res.available.push_back(spec);
    }
    while (!res.note.empty() &&
           (res.note.back() == ' ' || res.note.back() == ';'))
        res.note.pop_back();
    return res;
}

uint64_t antidetect_cache_bytes() {
    const fs::path cache = wmr::user_cache_dir() / "antidetect";
    uint64_t bytes = 0;
    std::error_code ec;
    if (!fs::exists(cache, ec)) return 0;
    for (const auto& e : fs::directory_iterator(cache, ec)) {
        if (e.is_regular_file(ec)) bytes += static_cast<uint64_t>(e.file_size(ec));
    }
    return bytes;
}

int clear_antidetect_models() {
    const fs::path cache = wmr::user_cache_dir() / "antidetect";
    std::error_code ec;
    if (!fs::exists(cache, ec)) return 0;
    int removed = 0;
    for (const auto& e : fs::directory_iterator(cache, ec)) {
        if (fs::remove(e.path(), ec)) ++removed;
    }
    return removed;
}

const SurrogateSpec* find_surrogate(std::string_view key) {
    for (const auto& s : kSurrogateManifest)
        if (key == s.key) return &s;
    return nullptr;
}

} // namespace wmr::antidetect
#endif  // WMR_BUILD_ANTIDETECT
