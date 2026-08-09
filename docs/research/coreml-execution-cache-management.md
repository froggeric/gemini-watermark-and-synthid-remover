# CoreML execution-cache management (macOS) — spec

Status: SPEC (owner to approve before implementation). Patch release.

## Problem

CoreML's app-scoped execution cache — `<wmr_cache>/com.apple.e5rt.e5bundlecache/`
(defaults to `~/Library/Caches/wmr/com.apple.e5rt.e5bundlecache/`) — holds the compiled
Metal graphs (`MPSGraphExecutable` bundles) for the SDXL models, keyed by model + macOS
build + GPU family. CoreML provides **no eviction**, so it accumulates across model
re-pins, wmr binary upgrades, and macOS upgrades. Observed on the dev machine: **138 GB**.
A stale/partial entry also produces Apple-framework warnings (`The file "manifest.plist"
couldn't be opened`, `Unable to load MPSGraphExecutable ... @ GetMPSGraphExecutable`) —
non-fatal (CoreML recovers), but noisy. Today wmr does not manage this cache; the user can
only reclaim disk with a manual `rm -rf`.

## Goal

wmr auto-manages its own (app-scoped) CoreML execution cache on macOS: clears it when stale,
so it cannot balloon, at the cost of a one-time recompile.

## Why it's safe

- The e5bundlecache is a PERFORMANCE cache. The `.mlpackage` is the source of truth. Clearing
  forces a one-time CoreML recompile (~30s for the UNet) on the next run; correctness is unaffected.
- wmr clears ONLY the app-scoped `<wmr_cache>/com.apple.e5rt.e5bundlecache`. It NEVER touches
  `~/Library/Caches/CoreML` (shared across all CoreML apps).
- macOS-only (`WMR_BUILD_AI_COREML_SD` + `__APPLE__`). Linux/Windows (CPU sdcpp regen) unaffected.
- The model-fetch cache (`<wmr_cache>/coreml-sdxl/`, the `.mlpackage` + `.sha256.ok` sidecars) is
  unchanged — this is about the COMPILED execution cache, not the model files.

## Design

**Staleness sidecar (primary, O(1) per run).** A tiny file
`<wmr_cache>/com.apple.e5rt.e5bundlecache.wmr_meta` holds a key derived from:
- the wmr `APP_VERSION` (binary upgrade),
- the current CoreML UNet pin `kSha256Unet` (model re-pin; the VAEs compile too — include the
  VAE pins or a combined model-set hash),
- the macOS version (major.minor, via `NSProcessInfo.operatingSystemVersion` or `uname -r`;
  macOS upgrade).

On CoreML init, wmr computes the current key; if it differs from the sidecar (or the sidecar is
absent) → stale → clear the e5bundlecache + write the new sidecar. Reading a tiny file + one cheap
OS-version call is negligible overhead. This clears on the three real staleness events: wmr upgrade,
model re-pin, macOS upgrade — which is what produced the 138 GB (many model variants across dev).

**Bloat safety net (optional — owner decision).** Even when the sidecar matches, if the e5bundlecache
exceeds a size threshold (e.g. 8 GB, comfortably above one model's ~2-4 GB compile), clear it. Catches
accumulation the sidecar misses (CoreML leaving orphans within a stable config). Cost: a
`std::filesystem` recursive size sum, run only AFTER the sidecar check (skipped when the sidecar just
cleared). On a normal small cache this is fast; on a one-time huge cache it's the clear itself. If the
owner prefers zero per-run overhead, drop this and rely on the sidecar + the manual clear command.

**Clear.** `std::filesystem::remove_all(<wmr_cache>/com.apple.e5rt.e5bundlecache)`. Log:
`regen: cleared the CoreML execution cache (stale: <reason>; one-time recompile this run)`. Graceful
on failure (permissions): `spdlog::warn` + continue (CoreML uses the existing cache; no crash). No
concurrency lock — the CLI is single-process; note as a caveat.

**Code location.** New `src/core/coreml_cache.{hpp,cpp}` (guarded `#if defined(WMR_BUILD_AI_COREML_SD) && defined(__APPLE__)`):
`void manage_coreml_execution_cache(const std::filesystem::path& wmr_cache_dir, std::string_view model_key)`.
Called once in `regenerator.cpp` `Impl::initialize` (the CoreML branch), passing the resolved wmr
cache dir (`user_cache_dir()`, already in `src/core/paths.cpp`) and the model key (the UNet pin / a
combined model-set hash from `coreml_sd_model_fetch.cpp`).

**Manual override.** A `wmr cache --clear-coreml` subcommand (or `--clear-coreml-cache` flag) that
clears the e5bundlecache and exits — for users who want to reclaim disk or clear after an event the
sidecar missed.

## Testing

- Unit (pure): the staleness-key computation; the clear decision (sidecar matches → no clear;
  differs/absent → clear; size-guard threshold → clear). Put the pure logic in a testable helper.
- Integration (mac, manual): run regen → populates cache; edit the sidecar's version → next run
  clears + recompiles; verify the e5bundlecache is removed then repopulated; verify no crash on a
  permission-denied simulate; verify `~/Library/Caches/CoreML` is untouched.

## Version + CHANGELOG

Patch increment. Since 1.16.6 is unreleased, this can fold into 1.16.6 or be 1.16.7 (owner call).
CHANGELOG (under the SynthID/CoreML regen section): "wmr now auto-manages its CoreML execution cache
on macOS — clears it on a wmr upgrade, a model re-pin, or a macOS upgrade — so it can no longer
balloon (observed 138 GB un-evicted). The only cost is a one-time ~30s recompile after a clear. Linux
and Windows are unaffected."

## Open decisions (owner)

1. **Size-guard safety net:** include it (catch-all, small per-run cost on a small cache) or rely on
   the staleness sidecar + the manual clear command (zero per-run overhead)?
2. **Manual clear surface:** a `wmr cache --clear-coreml` subcommand, or a global `--clear-coreml-cache` flag?
3. **Version:** fold into the unreleased 1.16.6, or cut 1.16.7?
