#pragma once
#ifdef WMR_BUILD_REGEN
#ifdef WMR_BUILD_AI_COREML_SD
#include <filesystem>

namespace wmr {
namespace fs = std::filesystem;

// Ensure CoreML SDXL models are present in `models_dir`.
// If any required files are missing and `allow_download` is true, downloads them
// from the HuggingFace repo (froggeric/wmr), verifies SHA256, and extracts the
// .tar.gz archives. On any failure (network, SHA mismatch, extract), logs and
// returns false (caller should fall back to CPU/spectral). If all files are already
// present and verified, returns true without touching the network.
//
// Required files:
//   - coreml-sdxl-unet.mlpackage.tar.gz (extracted to .mlpackage dir)
//   - coreml-sdxl-vae-encoder.mlpackage.tar.gz (extracted to .mlpackage dir)
//   - coreml-sdxl-vae-decoder.mlpackage.tar.gz (extracted to .mlpackage dir)
//   - empty_prompt_embeds.bin (copied as-is)
//
bool ensure_coreml_models(const fs::path& models_dir, bool allow_download);

}  // namespace wmr
#endif
#endif
