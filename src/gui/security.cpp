#include "gui/security.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>

#if defined(_WIN32)
#include <windows.h>
#include <bcrypt.h>
#else
// <sys/random.h> on BOTH POSIX platforms: getentropy (macOS) and
// getrandom (Linux). The brief's "<unistd.h> on macOS" is wrong here.
#include <sys/random.h>
#endif

namespace wmr::gui {
namespace {

// Fills buf[0..n) from the platform CSPRNG; false only when the OS entropy
// source is broken. Both getentropy and getrandom cap a single call at 256
// bytes, so fill in chunks.
bool fill_random(unsigned char* buf, size_t n) {
#if defined(_WIN32)
    return BCryptGenRandom(nullptr, buf, (ULONG)n, BCRYPT_USE_SYSTEM_PREFERRED_RNG)
           == STATUS_SUCCESS;
#elif defined(__linux__)
    size_t done = 0;
    while (done < n) {
        const size_t chunk = std::min(n - done, size_t(256));
        const ssize_t got = getrandom(buf + done, chunk, 0);
        if (got <= 0) return false;
        done += static_cast<size_t>(got);
    }
    return true;
#else
    size_t done = 0;
    while (done < n) {
        const size_t chunk = std::min(n - done, size_t(256));
        if (getentropy(buf + done, chunk) != 0) return false;
        done += chunk;
    }
    return true;
#endif
}

}  // namespace

std::string generate_token() {
    unsigned char raw[16];
    bool ok = false;
    for (int attempt = 0; attempt < 4 && !ok; ++attempt) {
        ok = fill_random(raw, sizeof raw);
    }
    if (!ok) {
        std::fprintf(stderr, "wmr gui: system entropy source failed; refusing to start\n");
        std::abort();
    }

    static const char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(32);
    for (unsigned char b : raw) {
        out.push_back(kHex[b >> 4]);
        out.push_back(kHex[b & 0x0F]);
    }
    return out;
}

bool host_allowed(const std::string& host_header, int port) {
    if (host_header.empty()) return false;
    // Split on the LAST ':' so an (bracketed) IPv6 literal keeps its colons.
    const size_t colon = host_header.rfind(':');
    if (colon == std::string::npos) return false;  // no port

    std::string host = host_header.substr(0, colon);
    const std::string port_str = host_header.substr(colon + 1);

    if (host.size() >= 2 && host.front() == '[' && host.back() == ']') {
        host = host.substr(1, host.size() - 2);  // strip [::1] brackets
    }
    std::transform(host.begin(), host.end(), host.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (!host.empty() && host.back() == '.') host.pop_back();  // one trailing dot

    if (port_str != std::to_string(port)) return false;
    return host == "127.0.0.1" || host == "localhost" || host == "::1";
}

}
