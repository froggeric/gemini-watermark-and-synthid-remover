#pragma once
// Surrogate-model fetch for the anti-detection adversarial stage (the
// coreml_sd_model_fetch pattern: SHA256-pinned files, .sha256.ok sidecar fast
// path, resumable download with progress, --antidetect-no-download refusal).
#ifdef WMR_BUILD_ANTIDETECT

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "core/detector_suite.hpp"

namespace wmr::antidetect {

struct SurrogateFetchResult {
    std::vector<const SurrogateSpec*> available;  // present + verified
    std::string note;  // why anything is missing (unpinned/skipped/failed)
};

// Ensure the comma-separated `requested_keys` ("commfor,corvi") are present and
// verified in the models dir, downloading pinned ones when allowed. Never
// fatal: missing entries are skipped and named in `note` (the caller degrades
// per the facade policy). Resolution order for the models dir (the CoreML
// pattern, shared by the presence check and the fetch so they cannot disagree):
//   $WMR_ANTIDETECT_MODELS_DIR > <exe>/../share/wmr/antidetect >
//   <exe>/antidetect > ~/.cache/wmr/antidetect
SurrogateFetchResult ensure_surrogate_models(const std::string& requested_keys,
                                             bool allow_download);

std::filesystem::path antidetect_models_dir();

// `wmr cache --clear-antidetect-models` support. Only files physically inside
// the USER CACHE dir are counted/cleared, never env-var or exe-dir copies (the
// leftover-cpu-models rule).
uint64_t antidetect_cache_bytes();
int clear_antidetect_models();

} // namespace wmr::antidetect
#endif
