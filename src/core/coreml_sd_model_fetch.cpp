#ifdef WMR_BUILD_REGEN
#ifdef WMR_BUILD_AI_COREML_SD
#include "core/coreml_sd_model_fetch.hpp"
#include "core/model_downloader.hpp"

#include <spdlog/spdlog.h>
#include <filesystem>
#include <fstream>
#include <iterator>
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
// UNet: ORIGINAL attention (since 1.16.3; was SPLIT_EINSUM before, pin 9101cada...).
constexpr const char* kSha256Unet =
    "9625f95c9da0fe7a46e8ac0d5cc2b112be42c7c7ed32487966be5a5838fb430c";
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

    spdlog::debug("regen: extracting {} to {}", archive.filename().string(), target_dir.string());

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

struct FetchResult { bool ok = false; bool refetched = false; };

// Fetch one file: present+sidecar-pin-match -> no-op (O(1) warm path);
// present+SHA-verified -> refresh sidecar, no-op; present+SHA-mismatch -> remove +
// re-download; absent -> download. allow_download gates the network only;
// show_progress renders a ByteProgress line during the (multi-GB) download.
// base_url+"/"+filename is the source URL. Reuses download_pinned_file (OpenSSL
// SHA, true HTTP Range resume, atomic rename).
FetchResult fetch_one(const std::string& filename, const std::string& sha,
                      const fs::path& models_dir, bool allow_download,
                      const std::string& base_url, bool show_progress) {
    fs::path local_path = models_dir / filename;
    fs::path ok_sidecar = models_dir / (filename + ".sha256.ok");

    // O(1) warm path: a sidecar holding the CURRENT pin means we already
    // verified these exact bytes on a previous run. A re-pin invalidates it
    // because the pin string differs.
    if (fs::exists(local_path) && fs::exists(ok_sidecar)) {
        std::ifstream okf(ok_sidecar);
        std::string prev((std::istreambuf_iterator<char>(okf)), {});
        if (prev == sha) {
            spdlog::debug("regen: {} verified via sidecar (pin match)", filename);
            return {true, false};
        }
    }

    // Verify (or re-verify) the on-disk bytes.
    if (fs::exists(local_path)) {
        if (verify_sha256(local_path, sha)) {
            { std::ofstream o(ok_sidecar); o << sha; }  // stamp/refresh sidecar
            spdlog::debug("regen: {} present and verified", filename);
            return {true, false};
        }
        spdlog::warn("regen: {} exists but SHA mismatch, re-downloading", filename);
        fs::remove(local_path);
        fs::remove(ok_sidecar);
    }

    if (!allow_download) {
        spdlog::warn("regen: {} absent and download disabled", filename);
        return {false, false};
    }

    std::string url = base_url + "/" + filename;
    auto result = download_pinned_file(url, local_path, sha, true, false,
                                       show_progress ? make_byte_progress(filename) : nullptr);
    if (!result.ok) {
        spdlog::error("regen: failed to download {}: {}", filename, result.error);
        return {false, false};
    }
    { std::ofstream o(ok_sidecar); o << sha; }  // download_pinned_file already SHA-verified
    spdlog::info("regen: downloaded {} successfully", filename);
    return {true, true};
}

// Re-extract an archive when its extracted dir is missing OR the archive was
// just (re)downloaded this call (stale-free: remove the old dir first).
bool ensure_extracted(const fs::path& models_dir, const CoreMLModelFile& f, bool was_refetched) {
    if (!f.is_archive) return true;
    fs::path dir = models_dir / f.mlpackage_dir;
    if (fs::exists(dir) && !was_refetched) return true;
    if (fs::exists(dir)) fs::remove_all(dir);
    return extract_tar_gz(models_dir / f.filename, models_dir);
}

}  // namespace

bool ensure_coreml_model_files(const fs::path& models_dir,
                               const std::vector<CoreMLModelFile>& files,
                               bool allow_download,
                               const std::string& base_url,
                               bool show_progress) {
    fs::create_directories(models_dir);
    for (const auto& f : files) {
        auto fr = fetch_one(f.filename, f.sha256, models_dir, allow_download, base_url,
                            show_progress);
        if (!fr.ok) return false;
        if (!ensure_extracted(models_dir, f, fr.refetched)) return false;
    }
    return true;
}

bool ensure_coreml_models(const fs::path& models_dir, bool allow_download,
                          bool show_progress) {
    static const std::vector<CoreMLModelFile> kFiles = {
        {kFilenameEmbeds,     kSha256Embeds,     false, nullptr},
        {kFilenameUnet,       kSha256Unet,       true,
         "Stable_Diffusion_version_stabilityai_stable-diffusion-xl-base-1.0_unet.mlpackage"},
        {kFilenameVaeEncoder, kSha256VaeEncoder, true,
         "Stable_Diffusion_version_stabilityai_stable-diffusion-xl-base-1.0_vae_encoder.mlpackage"},
        {kFilenameVaeDecoder, kSha256VaeDecoder, true,
         "Stable_Diffusion_version_stabilityai_stable-diffusion-xl-base-1.0_vae_decoder.mlpackage"},
    };
    return ensure_coreml_model_files(models_dir, kFiles, allow_download, kHfRepoUrl,
                                     show_progress);
}

// Thin accessors over the file-local pins so other TUs can re-derive the same
// key set without the constants needing external linkage.
const char* coreml_unet_sha256() { return kSha256Unet; }
const char* coreml_vae_encoder_sha256() { return kSha256VaeEncoder; }
const char* coreml_vae_decoder_sha256() { return kSha256VaeDecoder; }

}  // namespace wmr
#endif
#endif
