#pragma once
// Auto-management of wmr's app-scoped CoreML execution cache on macOS.
//
// CoreML stores its compiled Metal graphs (MPSGraphExecutable bundles) in an
// app-scoped directory <wmr_cache>/com.apple.e5rt.e5bundlecache/ keyed by model
// + macOS build + GPU family. CoreML provides NO eviction, so across wmr binary
// upgrades, CoreML model re-pins, and macOS upgrades the directory accumulates
// stale compile artifacts (observed ~138 GB on the dev machine; also produces
// Apple-framework warnings about a stale "manifest.plist"). This TU clears it
// when stale (per a sidecar) or when it has clearly ballooned past a size
// threshold, then lets CoreML recompile (one-time ~30s on the next regen).
//
// Scope is intentionally narrow:
//   * ONLY <wmr_cache>/com.apple.e5rt.e5bundlecache is touched.
//   * NEVER ~/Library/Caches/CoreML (shared across all CoreML apps).
//   * NEVER the model .mlpackage files under <wmr_cache>/coreml-sdxl/.
//   * macOS only (WMR_BUILD_AI_COREML_SD + __APPLE__). Linux/Windows unaffected.
//
// All operations are best-effort: a filesystem error is warned and the regen
// flow continues. The CLI is single-process; no concurrency lock (caveat: two
// `wmr synthid` runs pointed at the same cache_dir in parallel could race
// CoreML's own writers — rare in practice and outside the CLI's scope).
#if defined(WMR_BUILD_AI_COREML_SD) && defined(__APPLE__)

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace wmr {
namespace fs = std::filesystem;

// The ~6 GB size-guard threshold. A normal CoreML SDXL compile is ~2-4 GB; the
// guard catches accumulation the staleness sidecar misses (orphaned compile
// artifacts within a stable config). Walked with an EARLY-EXIT: the moment the
// running total exceeds this, the walk stops (bounded by the threshold, not by
// the cache size). 6 GB comfortably covers one full SDXL compile (UNet ~4 GB +
// VAE enc/dec) while catching real bloat.
inline constexpr uintmax_t kCoreMLCacheSizeGuardBytes =
    uintmax_t(6) * 1024 * 1024 * 1024;

// --- Pure helpers (no filesystem, no OS calls) for unit testing. ---

// Compose the deterministic model_key:
//   "<wmr_version>|<sha_unet>|<sha_vae_encoder>|<sha_vae_decoder>|<macos_version>"
// Each input changes the compile output, so any change invalidates the cache.
std::string compose_model_key(std::string_view wmr_version,
                              std::string_view sha_unet,
                              std::string_view sha_vae_encoder,
                              std::string_view sha_vae_decoder,
                              std::string_view macos_version);

// True when the previous sidecar key differs from the current key, OR the
// sidecar was absent (prev is empty). True => the e5bundlecache is stale.
bool should_clear_for_staleness(std::string_view prev_key,
                                std::string_view cur_key);

// True when the cache size strictly exceeds the threshold (bloat safety net).
bool should_clear_for_size(uintmax_t total_bytes, uintmax_t threshold_bytes);

// --- Runtime + filesystem helpers (mac-only). ---

// Resolve the directory CoreML writes its app-scoped compile cache under for
// the wmr process: <HOME>/Library/Caches/wmr/ on macOS (CoreML uses the macOS-
// conventional per-app Caches location, NOT the XDG-style ~/.cache/wmr/ that
// wmr::user_cache_dir() returns for the model fetch + update check). Returns
// an empty path if HOME is unset (the cache cannot be managed in that case;
// callers no-op gracefully). Best-effort: does NOT create the directory (the
// sidecar writer creates its parent as needed).
fs::path coreml_app_cache_dir();

// Resolve the macOS "major.minor" (e.g. "25.5" from Darwin uname -r). Empty on
// failure (the key then varies only on wmr/model pins, which is still correct).
// Uses uname(3) (no Foundation/ObjC dependency).
std::string macos_version();

// Recursively sum file sizes under `dir`, stopping early once the running total
// STRICTLY exceeds `cap` (returns the overflowed value). Best-effort: per-entry
// filesystem errors are skipped (counted as 0); a missing/non-dir `dir` is 0.
uintmax_t directory_size_capped(const fs::path& dir, uintmax_t cap);

// Auto-manage the cache. Called once in regenerator.cpp's CoreML init, AFTER the
// models are ensured present (so a model re-pin that triggered a re-download
// also invalidates the compile cache). Reads the sidecar, clears on staleness
// or size-bloat (early-exit walk), and ALWAYS rewrites the sidecar with the
// current model_key (the cache is now fresh for this key). Graceful throughout.
void manage_coreml_execution_cache(const fs::path& wmr_cache_dir,
                                   std::string_view model_key);

// Manual clear (the `wmr cache --clear-coreml` subcommand). remove_all the
// e5bundlecache; returns true on success (including "did not exist"). The sidecar
// is intentionally NOT touched: the cleared cache will be recompiled by CoreML
// into the same key (no wmr/model/macOS change), so the existing sidecar stays
// accurate and the next regen run does NOT re-clear.
bool clear_coreml_execution_cache(const fs::path& wmr_cache_dir);

}  // namespace wmr

#endif
