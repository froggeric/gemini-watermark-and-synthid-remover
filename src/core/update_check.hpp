#pragma once

#ifdef WMR_UPDATE_CHECK

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

// Notify-only update check. Never throws, never writes to stdout, never changes
// the process exit code. `no_update_check` is the --no-update-check flag value.
void maybe_check_for_update(bool no_update_check,
                            FetchFn fetch = {});

}  // namespace wmr

#endif  // WMR_UPDATE_CHECK
