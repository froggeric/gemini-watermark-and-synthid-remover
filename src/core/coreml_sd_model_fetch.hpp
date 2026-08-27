#pragma once
#ifdef WMR_BUILD_REGEN
#ifdef WMR_BUILD_AI_COREML_SD
#include <filesystem>
#include <vector>

namespace wmr {
namespace fs = std::filesystem;

// Ensure CoreML SDXL models are present in `models_dir`.
// If any required file is missing and `allow_download` is true, downloads it
// from the HuggingFace repo (froggeric/wmr), verifies SHA256, and extracts the
// .tar.gz archives. On any failure (network, SHA mismatch, extract), logs and
// returns false (caller fails closed; the user should re-run with --regen-backend
// cpu). The WARM path (every file present) returns true WITHOUT the network ONLY
// after SHA-verifying each archive against the current pin: a sidecar
// <archive>.sha256.ok holds the last-verified pin so the warm path is O(1) per
// pin instead of re-hashing the 4.8 GB UNet every run. A re-pin invalidates the
// sidecar (the pin string differs), so adoption and rollback reach already-cached
// users.
//
// Required files:
//   - coreml-sdxl-unet.mlpackage.tar.gz (extracted to .mlpackage dir)
//   - coreml-sdxl-vae-encoder.mlpackage.tar.gz (extracted to .mlpackage dir)
//   - coreml-sdxl-vae-decoder.mlpackage.tar.gz (extracted to .mlpackage dir)
//   - empty_prompt_embeds.bin (copied as-is)
//
bool ensure_coreml_models(const fs::path& models_dir, bool allow_download,
                          bool show_progress = false);

// One CoreML model artifact to ensure-present. Decouples the cache/SHA/extract
// logic from the pinned constants so it is unit-testable against tiny file://
// fixtures instead of the real 4.8 GB archive.
struct CoreMLModelFile {
    const char* filename;        // e.g. "coreml-sdxl-unet.mlpackage.tar.gz"
    const char* sha256;          // lowercase hex pin (the CURRENT pin)
    bool is_archive;             // true: .tar.gz extracted to mlpackage_dir; false: raw file
    const char* mlpackage_dir;   // extracted dir name when is_archive, else nullptr
};

// Testable core of ensure_coreml_models. For each file: SHA-verify the on-disk
// bytes against the current pin (re-download from base_url+"/"+filename on
// mismatch or absence), then ensure the archive is extracted. A sidecar
// <filename>.sha256.ok holds the last-verified pin so the warm path is O(1) per
// pin instead of re-hashing the 4.8 GB UNet (~10 s) every run. allow_download
// gates only the network; a present+verified cache returns true offline. Returns
// true iff every file is present+verified and every archive extracted.
bool ensure_coreml_model_files(const std::filesystem::path& models_dir,
                               const std::vector<CoreMLModelFile>& files,
                               bool allow_download,
                               const std::string& base_url,
                               bool show_progress = false);

// Accessors for the current model SHA256 pins. Exposed so other TUs (the
// execution-cache key in coreml_cache.cpp) can re-derive the SAME pin set that
// drives the model-fetch cache — a re-pin then invalidates BOTH caches in
// lockstep (the e5bundlecache compile + the .sha256.ok sidecar).
const char* coreml_unet_sha256();
const char* coreml_vae_encoder_sha256();
const char* coreml_vae_decoder_sha256();

}  // namespace wmr
#endif
#endif
