#pragma once
// Per-run cache directories for the GUI server (Task 4). Each server run owns
// ~/.cache/wmr/gui/<run_id>/ (run_id = timestamp-pid, mode 0700, pid sidecar).
// Concurrent instances coexist: startup cleanup removes only sibling run dirs
// whose recorded pid is dead; the shared root is never wiped. The optional
// root parameter (default: user_cache_dir()/"gui") exists so the dead-pid
// cleanup is unit-testable against a temp dir.
#include <cstdint>
#include <filesystem>

namespace wmr::gui {

// Returns the new run dir. Throws std::runtime_error (path + OS message) when
// the root or run dir cannot be created, chmod'd 0700, or the pid sidecar
// cannot be written: startup fails loudly rather than serving with a broken
// cache whose every upload would fail with no diagnostic.
std::filesystem::path create_run_dir(const std::filesystem::path& root = {});
void cleanup_dead_runs(const std::filesystem::path& root = {});   // once at startup, before serving
void remove_run_dir(const std::filesystem::path&);                // graceful shutdown only
// Sum of regular-file sizes under root (recursive) - the 4 GiB aggregate gate.
std::uintmax_t stored_bytes_under(const std::filesystem::path& root);

}  // namespace wmr::gui
