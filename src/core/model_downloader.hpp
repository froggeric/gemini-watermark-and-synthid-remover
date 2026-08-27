#pragma once
#ifdef WMR_BUILD_REGEN
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>
#include "cli/progress.hpp"
namespace wmr {
namespace fs = std::filesystem;

struct DownloadResult {
    bool ok = false;
    std::string path;
    std::string error;
};

// Verify `file`'s SHA256 (lowercase hex) equals `expected`. Empty `expected` = skip
// (returns true) ONLY from `verify_sha256` directly. The downloader refuses an empty
// pin unless `allow_empty_hash` is set (tests only). Backed by OpenSSL (the regen
// build links OpenSSL::Crypto via Homebrew openssl@3); a hand-rolled hash is NOT
// acceptable for a 6.5 GB model integrity gate.
bool verify_sha256(const fs::path& file, const std::string& expected_sha256_lower);

// Progress callback type (bytes downloaded, total). May be null. Returning false
// aborts the transfer.
using DownloadProgressFn = std::function<bool(uint64_t downloaded, uint64_t total)>;

// Download `url` to `dest`, verifying against `expected_sha256_lower`.
//
// Resumable (TRUE HTTP Range resume, not a truncate-and-restart): a `.part` side file
// is kept across attempts; on each attempt the already-downloaded byte count is sent
// as an HTTP Range request, the response code is inspected (206 = append, 200 = server
// ignored Range -> truncate and restart, 416 = already complete -> finalize), and the
// full file is SHA256-verified after the last byte. Atomic rename to `dest` on success.
//
// `allow_download=false` refuses the network and returns ok=false when `dest` is absent
// (an already-present + verified `dest` is ok=true without network). `progress` (if set)
// is called with byte milestones and may return false to abort. A bad/partial file is
// deleted and retried once. `allow_empty_hash=true` permits an empty pin (TESTS ONLY:
// a local file:// fixture deriving a hash); the real model download MUST pass the real
// pin, else this returns ok=false with error "refusing unpinned download".
DownloadResult download_pinned_file(const std::string& url,
                                    const fs::path& dest,
                                    const std::string& expected_sha256_lower,
                                    bool allow_download,
                                    bool allow_empty_hash = false,
                                    DownloadProgressFn progress = nullptr);

// A DownloadProgressFn rendering a ByteProgress line for the file being fetched
// (bytes / percent / rate / ETA + bar on a TTY, timestamped milestone lines when
// piped). Shared by the sdcpp model/VAE fetch and the CoreML model fetch so
// every multi-GB download reports the same way.
DownloadProgressFn make_byte_progress(const std::string& filename);

// The sdcpp (CPU-regen) models that earlier versions auto-downloaded into the
// user cache on a mac FIRST RUN, before CoreML became the auto default (the
// pre-1.16.11 order fetched them before resolving the backend). Once the CoreML
// pipeline is in use they are dead weight (~7.2 GB); they re-download on demand
// if --regen-backend cpu is ever passed. Only files physically inside the user
// cache dir are considered, never user-specified (WMR_REGEN_MODEL) or
// exe-dir-bundled copies.
struct LeftoverCpuModels {
    std::vector<fs::path> paths;
    uint64_t bytes = 0;
};
LeftoverCpuModels find_leftover_cpu_models();
// Removes exactly the listed paths (best-effort). Returns the count removed.
int remove_leftover_cpu_models(const LeftoverCpuModels& leftovers);

}  // namespace wmr
#endif
