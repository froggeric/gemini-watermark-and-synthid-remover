#include "gui/run_dir.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>

#include <fmt/chrono.h>
#include <fmt/format.h>

#include "core/paths.hpp"

#ifdef _WIN32
#include <process.h>
#include <windows.h>
#define getpid _getpid
#else
#include <errno.h>
#include <signal.h>
#include <unistd.h>
#endif

namespace wmr::gui {
namespace {

namespace fs = std::filesystem;

fs::path gui_root(const fs::path& root_arg) {
    return root_arg.empty() ? (wmr::user_cache_dir() / "gui") : root_arg;
}

// Reads a decimal pid from <dir>/pid; false when absent or not an integer
// (a dir without a readable sidecar is not ours: never touch it).
bool read_pid_file(const fs::path& dir, long& pid_out) {
    std::ifstream in(dir / "pid");
    if (!in) return false;
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (text.empty()) return false;
    try {
        size_t pos = 0;
        const long pid = std::stol(text, &pos);
        while (pos < text.size() &&
               (text[pos] == ' ' || text[pos] == '\n' || text[pos] == '\r' || text[pos] == '\t'))
            ++pos;
        if (pos != text.size() || pid <= 0) return false;
        pid_out = pid;
        return true;
    } catch (...) {
        return false;
    }
}

bool pid_alive(long pid) {
#ifdef _WIN32
    // NOTE: Windows reuses pids aggressively; OpenProcess succeeding may be a
    // DIFFERENT process. The caller pairs this with the 7-day mtime age rule.
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, static_cast<DWORD>(pid));
    if (!h) return false;
    CloseHandle(h);
    return true;
#else
    if (kill(static_cast<pid_t>(pid), 0) == 0) return true;  // signalable: alive
    return errno != ESRCH;  // EPERM etc. = alive under another user; only ESRCH is dead
#endif
}

}  // namespace

fs::path create_run_dir(const fs::path& root_arg) {
    const fs::path root = gui_root(root_arg);
    std::error_code ec;
    fs::create_directories(root, ec);
    const std::string run_id = fmt::format("{:%Y%m%d%H%M%S}-{}", std::chrono::system_clock::now(),
                                           getpid());
    const fs::path dir = root / run_id;
    fs::create_directories(dir, ec);  // same-second same-pid reuse is tolerated
    // 0700 even when the dir pre-existed with looser bits (cache hygiene:
    // uploads land here before any auth beyond the token).
    fs::permissions(dir, fs::perms::owner_all, fs::perm_options::replace, ec);
    {
        std::ofstream out(dir / "pid", std::ios::trunc);
        out << getpid();
    }
    return dir;
}

void cleanup_dead_runs(const fs::path& root_arg) {
    const fs::path root = gui_root(root_arg);
    std::error_code ec;
    if (!fs::is_directory(root, ec)) return;
    for (fs::directory_iterator it(root, ec), end; it != end; it.increment(ec)) {
        if (ec) break;
        std::error_code dirc;
        if (!it->is_directory(dirc)) continue;
        long pid = 0;
        if (!read_pid_file(it->path(), pid)) continue;  // not a run dir: never touch it
        bool dead;
#ifdef _WIN32
        if (pid_alive(pid)) {
            // The pid may have been recycled onto another process: keep the dir
            // only while it is young, else age it out (7 days by mtime).
            std::error_code tc;
            const auto mtime = fs::last_write_time(it->path(), tc);
            dead = !tc && (fs::file_time_type::clock::now() - mtime) > std::chrono::hours(24 * 7);
        } else {
            dead = true;
        }
#else
        dead = !pid_alive(pid);
#endif
        std::error_code rm;
        if (dead) fs::remove_all(it->path(), rm);  // best-effort; next start retries
    }
}

void remove_run_dir(const fs::path& dir) {
    std::error_code ec;
    fs::remove_all(dir, ec);  // graceful shutdown only; best-effort
}

std::uintmax_t stored_bytes_under(const fs::path& root) {
    std::uintmax_t total = 0;
    std::error_code ec;
    if (!fs::is_directory(root, ec)) return 0;
    for (fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied,
                                             ec),
                end;
         it != end; it.increment(ec)) {
        if (ec) break;
        std::error_code fc;
        if (it->is_regular_file(fc) && !it->is_symlink(fc)) total += it->file_size(fc);
    }
    return total;
}

}  // namespace wmr::gui
