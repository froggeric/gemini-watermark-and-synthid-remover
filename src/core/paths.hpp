#pragma once

#include <filesystem>

namespace wmr {
namespace fs = std::filesystem;

// wmr's cache directory: $HOME/.cache/wmr on POSIX,
// %USERPROFILE%\.cache\wmr on Windows, <cwd>/wmr-cache when neither is set.
// Created on first call. Single source of truth shared by the regen model
// cache and the update check.
fs::path user_cache_dir();

}  // namespace wmr
