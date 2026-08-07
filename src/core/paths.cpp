#include "core/paths.hpp"

#include <cstdlib>

namespace wmr {

fs::path user_cache_dir() {
    const char* home = std::getenv("HOME");
#ifdef _WIN32
    if (!home || home[0] == '\0') {
        home = std::getenv("USERPROFILE");
    }
#endif
    fs::path cache = (home && home[0] != '\0')
                         ? fs::path(home) / ".cache" / "wmr"
                         : fs::current_path() / "wmr-cache";
    std::error_code ec;
    fs::create_directories(cache, ec);  // best-effort; ignore failure
    return cache;
}

}  // namespace wmr
