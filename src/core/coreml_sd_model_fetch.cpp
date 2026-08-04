#ifdef WMR_BUILD_REGEN
#ifdef WMR_BUILD_AI_COREML_SD
#include "core/coreml_sd_model_fetch.hpp"
#include "core/model_downloader.hpp"

#include <spdlog/spdlog.h>
#include <filesystem>
#include <string>
#include <cstdio>

#ifdef _WIN32
#include <windows.h>
#else
#include <cstdlib>
#endif

namespace fs = std::filesystem;
namespace wmr {

namespace {
constexpr const char* kHfRepoUrl = "https://huggingface.co/froggeric/wmr/resolve/main";

// SHA256 pins for the CoreML SDXL models (computed from the archives)
constexpr const char* kSha256Unet =
    "9101cadaefb5f98dad645ed81704ebe775ca173d926a61a4ed27a41d7e743f7a";
constexpr const char* kSha256VaeEncoder =
    "370232dd23330abe34c17b19d8b7c08f55c5938887ad1ef608b2bc0b4e000036";
constexpr const char* kSha256VaeDecoder =
    "f477d3ca98a19143d81c199c1d02d3ed1737d4638487042c71775016fe399424";
constexpr const char* kSha256Embeds =
    "e27ab49bda70deda842a83afa00e67488533f881f5a359bf31b514c00a8038fe";

constexpr const char* kFilenameUnet = "coreml-sdxl-unet.mlpackage.tar.gz";
constexpr const char* kFilenameVaeEncoder = "coreml-sdxl-vae-encoder.mlpackage.tar.gz";
constexpr const char* kFilenameVaeDecoder = "coreml-sdxl-vae-decoder.mlpackage.tar.gz";
constexpr const char* kFilenameEmbeds = "empty_prompt_embeds.bin";

// Extract a .tar.gz archive to the target directory using tar.
// Returns true on success, false on failure.
bool extract_tar_gz(const fs::path& archive, const fs::path& target_dir) {
    fs::create_directories(target_dir);

    // Build tar command: tar -xzf <archive> -C <target_dir>
    std::string cmd = "tar -xzf \"" + archive.string() + "\" -C \"" + target_dir.string() + "\"";

    spdlog::info("regen: extracting {} to {}", archive.filename().string(), target_dir.string());

#ifdef _WIN32
    // Use Windows-specific command
    cmd = "tar -xzf \"" + archive.string() + "\" -C \"" + target_dir.string() + "\"";
#else
    // Unix-like: tar is always present on macOS
#endif

    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        spdlog::error("regen: tar extraction failed with code {}", ret);
        return false;
    }
    return true;
}

// Download a single file from HF, verify SHA256, and extract if it's a .tar.gz.
// Returns true if the file is present and verified (either pre-existing or downloaded).
// Returns false on any failure (network, SHA mismatch, extract failure).
bool fetch_model_file(const std::string& filename,
                      const std::string& sha256,
                      const fs::path& models_dir,
                      bool allow_download) {
    fs::path local_path = models_dir / filename;

    // Check if already present and verified
    if (fs::exists(local_path)) {
        if (verify_sha256(local_path, sha256)) {
            spdlog::debug("regen: {} already present and verified", filename);
            return true;
        }
        spdlog::warn("regen: {} exists but SHA mismatch, re-downloading", filename);
        fs::remove(local_path);
    }

    if (!allow_download) {
        spdlog::warn("regen: {} absent and download disabled", filename);
        return false;
    }

    // Build download URL
    std::string url = std::string(kHfRepoUrl) + "/" + filename;

    // Download with SHA256 verification
    auto result = download_pinned_file(url, local_path, sha256, true, false, nullptr);
    if (!result.ok) {
        spdlog::error("regen: failed to download {}: {}", filename, result.error);
        return false;
    }

    spdlog::info("regen: downloaded {} successfully", filename);
    return true;
}

}  // namespace

bool ensure_coreml_models(const fs::path& models_dir, bool allow_download) {
    fs::create_directories(models_dir);

    // Check if all required files are present
    bool all_present = true;
    if (!fs::exists(models_dir / kFilenameEmbeds)) all_present = false;
    if (!fs::exists(models_dir / kFilenameUnet)) all_present = false;
    if (!fs::exists(models_dir / kFilenameVaeEncoder)) all_present = false;
    if (!fs::exists(models_dir / kFilenameVaeDecoder)) all_present = false;

    // Also check for extracted .mlpackage directories
    bool extracted_present =
        fs::exists(models_dir / "Stable_Diffusion_version_stabilityai_stable-diffusion-xl-base-1.0_unet.mlpackage") &&
        fs::exists(models_dir / "Stable_Diffusion_version_stabilityai_stable-diffusion-xl-base-1.0_vae_encoder.mlpackage") &&
        fs::exists(models_dir / "Stable_Diffusion_version_stabilityai_stable-diffusion-xl-base-1.0_vae_decoder.mlpackage");

    if (all_present && extracted_present) {
        spdlog::debug("regen: CoreML models already present and extracted");
        return true;
    }

    if (!allow_download) {
        spdlog::warn("regen: CoreML models incomplete and download disabled");
        return false;
    }

    spdlog::info("regen: fetching CoreML SDXL models from {}", kHfRepoUrl);

    // Download the 4 files
    if (!fetch_model_file(kFilenameEmbeds, kSha256Embeds, models_dir, allow_download))
        return false;
    if (!fetch_model_file(kFilenameUnet, kSha256Unet, models_dir, allow_download))
        return false;
    if (!fetch_model_file(kFilenameVaeEncoder, kSha256VaeEncoder, models_dir, allow_download))
        return false;
    if (!fetch_model_file(kFilenameVaeDecoder, kSha256VaeDecoder, models_dir, allow_download))
        return false;

    // Extract the 3 .tar.gz archives
    if (!extract_tar_gz(models_dir / kFilenameUnet, models_dir)) return false;
    if (!extract_tar_gz(models_dir / kFilenameVaeEncoder, models_dir)) return false;
    if (!extract_tar_gz(models_dir / kFilenameVaeDecoder, models_dir)) return false;

    spdlog::info("regen: CoreML SDXL models downloaded and extracted successfully");
    return true;
}

}  // namespace wmr
#endif
#endif
