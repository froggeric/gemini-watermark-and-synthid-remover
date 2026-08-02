#pragma once
#ifdef WMR_BUILD_REGEN
#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include "core/regen_backend.hpp"
namespace wmr {

struct RegenConfig {
    float strength = 0.05f;       // img2img denoising strength (valid 0.02..0.15)
    int   steps = 20;             // sample steps
    int   tile_size = 1024;       // SDXL-native tile edge (img2img runs at this res)
    int   overlap = 128;          // feather overlap between tiles
    int   seed = 42;              // deterministic (SynthID scrubbing must be reproducible)
    bool  tile = true;            // false = aspect-fit fallback (resize to <=1024, no tiling)
    bool  allow_download = true;  // --regen-no-download flips this (offline/air-gapped)
    std::string model_path;       // override; empty = resolve+download
    std::string vae_path;         // override; empty = resolve + download the pinned fp16-fix VAE
                                  // (REQUIRED for SDXL fp16; the embedded VAE produces black/NaN output).
    std::string prompt;           // empty by default (no conditioning bias)
};

// SDXL img2img regenerator over leejet/stable-diffusion.cpp. Holds one sd_ctx_t*
// created in initialize() after resolving+downloading the SHA256-pinned model + fp16-fix VAE.
//
// LIFETIME: the process-wide singleton in WatermarkEngine::regenerator() is intentionally
// LEAKED (mirror WatermarkEngine::denoiser() + MiganInpainter). Destroying an sd_ctx during
// C++ static teardown races the GGML backend's global device teardown (the same Vulkan/CUDA/
// Metal hazard as ncnn). The leak is specifically about static-destruction ORDER: a
// FUNCTION-SCOPE (stack) Regenerator is safe to destruct normally, because user code runs
// before static teardown. Only the raw-new'd singleton must never be deleted.
//
// THREAD-SAFETY: NOT thread-safe (one sd_ctx_t, single internal RNG/stream state), mirroring
// MiganInpainter. wmr's batch path is sequential, so this is fine; do NOT call regen()
// concurrently from multiple threads on the same singleton.
class Regenerator {
public:
    Regenerator();
    ~Regenerator();
    Regenerator(const Regenerator&) = delete;
    Regenerator& operator=(const Regenerator&) = delete;

    bool initialize(const RegenConfig& cfg);   // resolve/download model, create sd_ctx
    [[nodiscard]] bool is_ready() const;

    // Run SDXL img2img over the whole image at cfg.strength. For images larger than
    // tile_size on either axis (default), runs MultiDiffusion-style tiled img2img so
    // the native resolution is preserved; else a single pass at the image size.
    // Writes the result back into `image` (BGR u8, same size). Returns false on a
    // backend/model error (caller falls back to the spectral path; never throws).
    bool regen(cv::Mat& image, const RegenConfig& cfg);

private:
    struct Impl;
    struct ImplDeleter { void operator()(Impl*) const; };  // defined out-of-line
    std::unique_ptr<Impl, ImplDeleter> m_impl;
};

} // namespace wmr
#endif
