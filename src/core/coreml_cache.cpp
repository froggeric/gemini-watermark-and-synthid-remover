// Auto-management of wmr's app-scoped CoreML execution cache on macOS.
// See coreml_cache.hpp for the design and scope.
#if defined(WMR_BUILD_AI_COREML_SD) && defined(__APPLE__)

#include "core/coreml_cache.hpp"

#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <sys/utsname.h>  // uname(3)

#include <cstdlib>       // std::getenv
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>

namespace fs = std::filesystem;
namespace wmr {

namespace {

// The e5bundlecache directory name CoreML uses for the app-scoped compiled
// Metal graphs. Stable across macOS versions (Apple-internal cache layout).
constexpr const char* kE5BundleCacheDirname = "com.apple.e5rt.e5bundlecache";
// The wmr-owned staleness sidecar (one tiny text file holding the last
// model_key the cache was cleared/rewritten for). The .wmr_meta suffix avoids
// colliding with anything CoreML writes into the same parent directory.
constexpr const char* kSidecarFilename = "com.apple.e5rt.e5bundlecache.wmr_meta";

fs::path e5_dir(const fs::path& wmr_cache_dir) { return wmr_cache_dir / kE5BundleCacheDirname; }
fs::path sidecar_path(const fs::path& wmr_cache_dir) { return wmr_cache_dir / kSidecarFilename; }

}  // namespace

fs::path coreml_app_cache_dir() {
    // CoreML writes its app-scoped compile cache under the macOS-conventional
    // per-app Caches directory: <HOME>/Library/Caches/<executable-name>/. For
    // the wmr binary the executable-name suffix is "wmr" (the binary name set
    // by project(wmr ...)). NSSearchPathForDirectoriesInDomains(NSCachesDirectory,
    // NSUserDomainMask, YES)[0] would be the Apple-clean API, but on macOS HOME
    // is set by the system to the same location NSSearchPath reads from, so
    // getenv("HOME") + "/Library/Caches/wmr" is equivalent without dragging in
    // Foundation/ObjC. If HOME is unset (rare; e.g. a launchd context), return
    // an empty path and let the caller no-op.
    const char* home = std::getenv("HOME");
    if (!home || home[0] == '\0') return fs::path{};
    return fs::path(home) / "Library" / "Caches" / "wmr";
}

namespace {

// Read the sidecar (absent or unreadable -> empty string, treated as STALE so
// the next run clears + rewrites it).
std::string read_sidecar(const fs::path& p) {
    std::error_code ec;
    if (!fs::exists(p, ec)) return std::string{};
    std::ifstream f(p);
    if (!f) return std::string{};
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// Write the sidecar (best-effort; warns on failure). The parent dir is
// user_cache_dir() which is created upstream, but create_directories is a
// no-op if it already exists, so be defensive.
void write_sidecar(const fs::path& p, std::string_view key) {
    try {
        std::error_code ec;
        if (!p.parent_path().empty() && !fs::exists(p.parent_path(), ec)) {
            fs::create_directories(p.parent_path(), ec);
        }
        std::ofstream f(p, std::ios::trunc);
        if (f) {
            f << key;
        } else {
            spdlog::warn("regen: could not open the CoreML cache sidecar for write ({})",
                         p.string());
        }
    } catch (const std::exception& e) {
        spdlog::warn("regen: failed to write the CoreML cache sidecar ({}): {}",
                     p.string(), e.what());
    }
}

// remove_all with a warn-on-failure wrapper. Returns true on success (including
// "did not exist"). NEVER throws.
bool try_remove_all(const fs::path& p) {
    std::error_code ec;
    if (!fs::exists(p, ec)) return true;
    fs::remove_all(p, ec);
    if (ec) {
        spdlog::warn("regen: failed to remove the CoreML cache directory ({}): {}",
                     p.string(), ec.message());
        return false;
    }
    // remove_all returns 0 if the path vanished between the exists() check and
    // the call; treat a non-existent post-state as success.
    bool still = fs::exists(p, ec);
    return !still;
}

}  // namespace

std::string compose_model_key(std::string_view wmr_version,
                              std::string_view sha_unet,
                              std::string_view sha_vae_encoder,
                              std::string_view sha_vae_decoder,
                              std::string_view macos_version) {
    return fmt::format("{}|{}|{}|{}|{}", wmr_version, sha_unet, sha_vae_encoder,
                       sha_vae_decoder, macos_version);
}

bool should_clear_for_staleness(std::string_view prev_key, std::string_view cur_key) {
    // Absent sidecar (empty prev) -> STALE -> clear (re-stamp the sidecar).
    if (prev_key.empty()) return true;
    return prev_key != cur_key;
}

bool should_clear_for_size(uintmax_t total_bytes, uintmax_t threshold_bytes) {
    return total_bytes > threshold_bytes;
}

std::string macos_version() {
    utsname u;
    if (uname(&u) != 0) return std::string{};
    // u.release is e.g. "25.5.0" on Darwin. The Darwin major.minor maps 1:1 to
    // a macOS major.minor in practice (Darwin 24.x = macOS 15.x; 25.x = macOS
    // 26.x), and CoreML keys its compile cache off the OS build, so any change
    // here invalidates the cache. Keep major.minor only — patch versions do
    // not change the SDK headers CoreML compiles against.
    std::string rel = u.release;
    auto first_dot = rel.find('.');
    if (first_dot == std::string::npos) return rel;
    auto second_dot = rel.find('.', first_dot + 1);
    return second_dot == std::string::npos ? rel : rel.substr(0, second_dot);
}

uintmax_t directory_size_capped(const fs::path& dir, uintmax_t cap) {
    std::error_code ec;
    if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) return 0;
    uintmax_t total = 0;
    // Construct with an error_code so a construction failure (e.g. permission
    // denied at the root) does not throw; on failure the iterator equals end
    // and the loop is skipped. skip_permission_denied avoids the common throw
    // on a protected subdirectory.
    fs::recursive_directory_iterator it(
        dir, fs::directory_options::skip_permission_denied, ec);
    fs::recursive_directory_iterator end;
    for (; it != end; it.increment(ec)) {
        if (ec) {
            // Entry-level error (broken symlink, permission, etc.): skip THIS
            // entry, keep going. Clear so the next increment is meaningful.
            ec.clear();
            continue;
        }
        const auto& entry = *it;
        std::error_code rec;
        if (entry.is_regular_file(rec)) {
            std::error_code sec;
            auto sz = entry.file_size(sec);
            if (!sec) total += sz;
            if (total > cap) return total;  // EARLY EXIT at the cap
        }
    }
    return total;
}

void manage_coreml_execution_cache(const fs::path& wmr_cache_dir, std::string_view model_key) {
    if (wmr_cache_dir.empty()) {
        // HOME unset (rare; launchd context). No app cache to manage: log + skip
        // so we never operate on a relative path resolved against the cwd.
        spdlog::debug("regen: CoreML cache dir unresolved (HOME unset); skipping cache management");
        return;
    }
    fs::path e5 = e5_dir(wmr_cache_dir);
    fs::path sidecar = sidecar_path(wmr_cache_dir);

    bool cleared = false;
    const char* reason = "";

    // 1) Staleness check (O(1) sidecar read).
    std::string prev = read_sidecar(sidecar);
    if (should_clear_for_staleness(prev, model_key)) {
        cleared = try_remove_all(e5);
        reason = "stale";
    } else {
        // 2) Size-guard safety net (with EARLY-EXIT at the cap so a bloated
        //    cache does not pay for a full walk). Skipped when the staleness
        //    branch already ran (the dir is gone / about to be repopulated).
        uintmax_t total = directory_size_capped(e5, kCoreMLCacheSizeGuardBytes);
        if (should_clear_for_size(total, kCoreMLCacheSizeGuardBytes)) {
            cleared = try_remove_all(e5);
            reason = ">= 6 GB";
        }
    }

    if (cleared) {
        spdlog::info("regen: cleared the CoreML execution cache ({}; "
                     "one-time recompile this run)", reason);
    }

    // 3) ALWAYS rewrite the sidecar with the current model_key. The cache is
    //    now fresh for this key: either it was just cleared (and CoreML will
    //    recompile into the same key) or it already matched. On a clear-FAILURE
    //    we still rewrite: CoreML will use whatever survives, and the next
    //    real staleness event (a future wmr/model/macOS change) WILL clear it
    //    again. The manual `wmr cache --clear-coreml` is the recovery path for
    //    a stubborn clear-failure.
    write_sidecar(sidecar, model_key);
}

bool clear_coreml_execution_cache(const fs::path& wmr_cache_dir) {
    if (wmr_cache_dir.empty()) {
        spdlog::warn("regen: CoreML cache dir unresolved (HOME unset); nothing to clear");
        return true;  // nothing to clear -> vacuously successful
    }
    return try_remove_all(e5_dir(wmr_cache_dir));
}

}  // namespace wmr

#endif
