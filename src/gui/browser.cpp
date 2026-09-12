#include "gui/browser.hpp"

#include <cstdint>
#include <cstdlib>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#include <shellapi.h>
#endif

namespace wmr::gui {
namespace {

#if !defined(_WIN32)
// POSIX sh single-quoting. The URL is self-built (host + hex token) so it
// contains no shell metacharacters, but quote anyway.
std::string shell_quote(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out.push_back(c);
        }
    }
    out += "'";
    return out;
}

bool run_sh(const std::string& cmd) { return std::system(cmd.c_str()) == 0; }

std::string replace_first(const std::string& hay, const std::string& needle,
                          const std::string& with) {
    const size_t pos = hay.find(needle);
    if (pos == std::string::npos) return hay;
    return hay.substr(0, pos) + with + hay.substr(pos + needle.size());
}
#endif

}  // namespace

bool open_browser(const std::string& url) {
#if defined(_WIN32)
    // UTF-8 -> UTF-16, then the shell's "open" verb (default browser).
    const int wlen = MultiByteToWideChar(CP_UTF8, 0, url.c_str(), -1, nullptr, 0);
    if (wlen <= 0) return false;
    std::wstring wide(static_cast<size_t>(wlen), L'\0');
    if (MultiByteToWideChar(CP_UTF8, 0, url.c_str(), -1, wide.data(), wlen) <= 0) {
        return false;
    }
    const auto ret = reinterpret_cast<uintptr_t>(
        ShellExecuteW(nullptr, L"open", wide.c_str(), nullptr, nullptr, SW_SHOWNORMAL));
    return ret > 32;  // >32 = success (Win32 ShellExecute convention)
#else
    // 1. $BROWSER: an optional %s placeholder takes the URL, else append it.
    if (const char* browser = std::getenv("BROWSER")) {
        const std::string b = browser;
        if (!b.empty()) {
            const std::string cmd = b.find("%s") != std::string::npos
                ? replace_first(b, "%s", shell_quote(url))
                : b + " " + shell_quote(url);
            return run_sh(cmd);
        }
    }
#if defined(__APPLE__)
    return run_sh("open " + shell_quote(url));
#else
    // Linux family. WSL first: without WSLg neither DISPLAY nor
    // WAYLAND_DISPLAY is set, yet wslview / cmd.exe still work.
    if (std::getenv("WSL_DISTRO_NAME")) {
        if (run_sh("wslview " + shell_quote(url))) return true;
        // start's first quoted arg is the window title, hence the "".
        return run_sh("cmd.exe /c start \"\" \"" + url + "\"");
    }
    if (!std::getenv("DISPLAY") && !std::getenv("WAYLAND_DISPLAY")) return false;
    return run_sh("xdg-open " + shell_quote(url));
#endif
#endif
}

}
