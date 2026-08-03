#ifdef WMR_BUILD_REGEN
#include "core/regenerator.hpp"
#include "core/model_downloader.hpp"
#include "core/regen_tiling.hpp"
#include "core/regen_backend.hpp"
#include <spdlog/spdlog.h>
#include <stable-diffusion.h>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <atomic>
#include <filesystem>
#include <memory>
#include <thread>
#include <cstdlib>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#else
#include <unistd.h>
#endif
namespace fs = std::filesystem;
namespace wmr {

namespace {
// Set true once new_sd_ctx succeeds + supports image generation. Read by
// regenerator_was_used() so main() can std::_Exit before static destruction
// (the ggml-metal static-teardown GGML_ASSERT fires regardless of whether
// our Regenerator is leaked or freed, because it's in ggml's OWN static).
std::atomic<bool> g_ctx_alive{false};

fs::path exe_dir() {
#ifdef _WIN32
    char buf[MAX_PATH]; HMODULE m=nullptr;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, (LPCSTR)&exe_dir, &m);
    GetModuleFileNameA(m, buf, MAX_PATH);
    return fs::path(buf).parent_path();
#else
    char buf[4096];
# ifdef __APPLE__
    uint32_t n = sizeof(buf);
    if (_NSGetExecutablePath(buf, &n) == 0) return fs::path(buf).parent_path();
# else
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) { buf[n] = 0; return fs::path(buf).parent_path(); }
# endif
#endif
    return fs::current_path();
}
fs::path cache_dir() {
    const char* home = std::getenv("HOME");
#ifdef _WIN32
    if (!home) home = std::getenv("USERPROFILE");
#endif
    fs::path cache = (home && home[0]) ? fs::path(home) / ".cache" / "wmr"
                                       : fs::current_path() / "wmr-cache";
    fs::create_directories(cache);
    return cache;
}
fs::path resolve_model(const RegenConfig& cfg) {
    if (!cfg.model_path.empty()) return cfg.model_path;
    const char* env = std::getenv("WMR_REGEN_MODEL");
    if (env && env[0]) return env;
    fs::path name = "sd_xl_base_1.0.safetensors";
    fs::path p = exe_dir() / name;                              if (fs::exists(p)) return p;
    p = exe_dir() / ".." / "share" / "wmr" / name;              if (fs::exists(p)) return p;
    return cache_dir() / name;
}
// Resolve the fp16-fix VAE (REQUIRED for SDXL fp16; see "Verified facts"). An explicit
// cfg.vae_path wins (including a user pointing at the embedded VAE for experiments); an
// empty cfg.vae_path means "download + use the pinned fp16-fix VAE", NOT "embedded VAE".
fs::path resolve_vae(const RegenConfig& cfg) {
    if (!cfg.vae_path.empty()) return cfg.vae_path;
    const char* env = std::getenv("WMR_REGEN_VAE");
    if (env && env[0]) return env;
    fs::path name = "sdxl_vae-fp16-fix.safetensors";
    fs::path p = exe_dir() / name;                              if (fs::exists(p)) return p;
    p = exe_dir() / ".." / "share" / "wmr" / name;              if (fs::exists(p)) return p;
    return cache_dir() / name;
}

// Pinned sources. SHA256 == the HuggingFace LFS content oid (the HF tree API exposes it
// as `lfs.sha256`); the VAE oid was cross-verified by downloading the 335 MB file and
// running `shasum -a 256` (byte-for-byte match), so the model oid is trusted the same way
// without downloading the 6.5 GB checkpoint. The fp16-fix VAE file is named
// `sdxl_vae.safetensors` on HF (NOT `sdxl_vae-fp16-fix.safetensors`; that path 404s).
constexpr const char* kRegenModelUrl =
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors";
constexpr const char* kRegenModelSha256 =
    "31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b";
constexpr const char* kRegenVaeUrl =
    "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors";
constexpr const char* kRegenVaeSha256 =
    "235745af8d86bf4a4c1b5b4f529868b37019a10f7c0b2e79ad0abca3a22bc6e1";

// Print byte milestones (~5%) so a 6.5 GB first-run fetch is not silent. Returns true.
DownloadProgressFn make_progress_logger(const std::string& what) {
    auto last = std::make_shared<int>(-1);
    return [what, last](uint64_t done, uint64_t total) -> bool {
        if (total == 0) return true;
        int pct = (int)(done * 100 / total);
        int step = (pct / 5) * 5;
        if (step != *last) { *last = step; spdlog::info("regen: {} {}%", what, pct); }
        return true;
    };
}

int round_down_8(int n) { return std::max(8, (n / 8) * 8); }
} // namespace

struct Regenerator::Impl {
    sd_ctx_t* ctx = nullptr;
    RegenConfig cfg{};
    fs::path vae_path_resolved;
    bool ready = false;
    ~Impl() { /* intentionally NOT freeing ctx here; see header + ImplDeleter */ }
    void destroy() { if (ctx) { free_sd_ctx(ctx); ctx = nullptr; } }

    bool img2img_tile(const cv::Mat& bgr_tile, cv::Mat& out_tile, const RegenConfig& c) {
        // Caller (regen()) guarantees bgr_tile is CV_8UC3.
        cv::Mat rgb; cv::cvtColor(bgr_tile, rgb, cv::COLOR_BGR2RGB);

        // SDXL latent = H/8 x W/8: round the GEN dims down to a multiple of 8. If that
        // differs from the tile, resize the init image to the gen dims; the caller resizes
        // the output back to the tile rect.
        int gw = round_down_8(rgb.cols), gh = round_down_8(rgb.rows);
        if (gw != rgb.cols || gh != rgb.rows)
            cv::resize(rgb, rgb, cv::Size(gw, gh), 0, 0, cv::INTER_AREA);

        sd_image_t init{ (uint32_t)rgb.cols, (uint32_t)rgb.rows, 3, rgb.data };
        sd_img_gen_params_t p; sd_img_gen_params_init(&p);
        p.init_image = init;
        p.width = init.width; p.height = init.height;
        p.strength = c.strength;
        p.seed = c.seed;
        p.prompt = c.prompt.empty() ? "" : c.prompt.c_str();
        p.negative_prompt = "";
        p.sample_params.sample_steps = c.steps;
        p.sample_params.sample_method = EULER_A_SAMPLE_METHOD;
        // SCHEDULER_COUNT is a SENTINEL, not a real value (verified fact). generate_image
        // does not auto-resolve it; resolve it here the way the sdcpp CLI does.
        p.sample_params.scheduler = sd_get_default_scheduler(ctx, p.sample_params.sample_method);
        p.batch_count = 1;
        p.vae_tiling_params.enabled = true;
        p.vae_tiling_params.tile_size_x = 512;
        p.vae_tiling_params.tile_size_y = 512;
        sd_image_t* out = nullptr; int n = 0;
        if (!generate_image(ctx, &p, &out, &n) || n < 1 || out == nullptr) {
            if (out) free_sd_images(out, n);
            return false;
        }
        cv::Mat tmp((int)out[0].height, (int)out[0].width, CV_8UC3, out[0].data);
        cv::cvtColor(tmp, out_tile, cv::COLOR_RGB2BGR);
        free_sd_images(out, n);
        return true;
    }
};

Regenerator::Regenerator() : m_impl(new Impl{}, ImplDeleter{}) {}
Regenerator::~Regenerator() = default;
void Regenerator::ImplDeleter::operator()(Impl* p) const {
    // For short-lived (non-singleton) instances, free the ctx normally. The leaked
    // process singleton in WatermarkEngine::regenerator() is raw-new'd and NEVER reaches
    // here, so the GGML/device teardown never runs during static destruction (the same
    // rationale as the NCNN denoiser). The leak is specifically about static-teardown
    // ORDER, not normal destruction: a function-scope (stack) Regenerator is safe.
    if (p) { p->destroy(); delete p; }
}

bool Regenerator::initialize(const RegenConfig& cfg) {
    m_impl->cfg = cfg;

    // 1) Resolve + download the model (base SDXL fp16 checkpoint).
    fs::path model = resolve_model(cfg);
    if (!fs::exists(model) || fs::file_size(model) < (1ull<<30)) {  // <1GB => missing/incomplete
        if (!cfg.allow_download) { spdlog::warn("regen: model absent and download disabled"); return false; }
        auto r = download_pinned_file(kRegenModelUrl, model, kRegenModelSha256, true,
                                      /*allow_empty_hash=*/false, make_progress_logger("model"));
        if (!r.ok) { spdlog::warn("regen: model download failed: {}", r.error); return false; }
    }
    // 2) Resolve + download the fp16-fix VAE (REQUIRED; the embedded fp16 VAE NaNs).
    fs::path vae = resolve_vae(cfg);
    m_impl->vae_path_resolved = vae;
    if (vae.empty() || !fs::exists(vae) || fs::file_size(vae) < (1ull<<24)) {  // <16MB => missing
        if (!cfg.allow_download) { spdlog::warn("regen: fp16-fix VAE absent and download disabled"); return false; }
        auto r = download_pinned_file(kRegenVaeUrl, vae, kRegenVaeSha256, true,
                                      /*allow_empty_hash=*/false, make_progress_logger("vae"));
        if (!r.ok) { spdlog::warn("regen: fp16-fix VAE download failed: {}", r.error); return false; }
    }

    // 3) Backend: resolve cfg.backend ("auto"/"cpu"/"metal"/"vulkan"). On Apple Silicon
    //    Auto is forced to CPU, because the Metal backend produces garbage output for SDXL
    //    img2img (upstream leejet/stable-diffusion.cpp / ggml bug; CPU produces correct
    //    output). Override with --regen-backend. Per upstream docs/backend.md, an empty
    //    backend (Auto, non-Apple) makes stable-diffusion.cpp auto-select GPU -> integrated
    //    GPU -> CPU, so it never fails for lack of a device and needs no manual
    //    sd_list_devices probe. The resolved string is a const char* literal (static
    //    storage), safe to assign directly to cp.backend.
    RegenBackend b = regen_backend_from_string(cfg.backend);  // ""/auto->Auto, cpu/metal/vulkan
    if (b == RegenBackend::Auto) {
#if defined(__APPLE__)
        // Metal produces garbage output for SDXL img2img on Apple Silicon (upstream
        // sdcpp/ggml bug; the CPU backend produces correct output). Default to CPU until
        // a Vulkan/MoltenVK path is verified. Override with --regen-backend.
        b = RegenBackend::Cpu;
        spdlog::warn("regen: Metal backend is broken on Apple Silicon (upstream sdcpp/ggml bug); "
                     "using CPU (~3x slower, impractical for 4K tiled). Override with "
                     "--regen-backend {metal,vulkan} at your own risk.");
#endif
    }
    const char* backend = regen_backend_string(b);  // "" (non-mac auto), "cpu", "metal", "vulkan0"

    // 4) Create the ctx. SDXL: clip_l/clip_g are embedded in the base checkpoint, so they
    //    stay nullptr. wtype left at the init default (auto-detect from the file).
    //    model_path/vae_path MUST point to storage that outlives the new_sd_ctx call:
    //    fs::path::string() returns a TEMPORARY, so .c_str() on it would dangle after the
    //    statement. Hold them in locals.
    std::string model_s = model.string();
    std::string vae_s   = vae.string();
    sd_ctx_params_t cp; sd_ctx_params_init(&cp);
    cp.model_path  = model_s.c_str();
    cp.vae_path    = vae_s.c_str();    // never null: the fp16-fix VAE
    cp.clip_l_path = nullptr; cp.clip_g_path = nullptr; cp.t5xxl_path = nullptr;
    cp.n_threads = std::max(1, (int)std::thread::hardware_concurrency());
    cp.backend    = backend;           // "" = auto (GPU -> CPU); const char* literal, never null
    cp.flash_attn = true;              // safe on every backend (auto-fallback where unsupported)
    m_impl->ctx = new_sd_ctx(&cp);
    if (!m_impl->ctx || !sd_ctx_supports_image_generation(m_impl->ctx)) {
        spdlog::warn("regen: sd_ctx creation failed (backend='{}', model='{}')",
                     backend[0] ? backend : "auto", model_s);
        m_impl->ready = false; return false;
    }
    m_impl->ready = true;
    g_ctx_alive.store(true);   // the ggml backend is now alive; its static teardown will abort
    spdlog::info("regen: sd_ctx ready, backend='{}', model='{}', vae='{}'",
                 backend[0] ? backend : "auto", model_s, vae_s);
    return true;
}
bool Regenerator::is_ready() const { return m_impl && m_impl->ready; }

// Normalize any input to BGR 8UC3 (grayscale -> BGR, RGBA -> BGR, non-u8 depth -> u8).
// cv::imread already returns BGR 8UC3 by default; this is a safety net for programmatic callers.
static cv::Mat normalize_to_bgr8u3(const cv::Mat& image) {
    cv::Mat bgr;
    if (image.channels() == 1)      cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
    else if (image.channels() == 4) cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
    else                            bgr = image;
    if (bgr.depth() != CV_8U) bgr.convertTo(bgr, CV_8U);
    return bgr;
}

bool Regenerator::regen(cv::Mat& image, const RegenConfig& cfg) {
    if (!is_ready()) return false;
    cv::Mat bgr = normalize_to_bgr8u3(image);   // operate on a normalized copy
    const bool need_tile = cfg.tile && (bgr.cols > cfg.tile_size || bgr.rows > cfg.tile_size);
    if (!need_tile) {
        cv::Mat out;
        if (!m_impl->img2img_tile(bgr, out, cfg)) return false;
        if (out.size() != image.size())
            cv::resize(out, image, image.size(), 0, 0, cv::INTER_LANCZOS4);
        else image = out;
        return true;
    }
    auto tiles = build_regen_tiles(bgr.cols, bgr.rows, cfg.tile_size, cfg.overlap);
    if (tiles.empty()) return false;
    cv::Mat accum = cv::Mat::zeros(bgr.size(), CV_32FC3);
    cv::Mat wsum  = cv::Mat::zeros(bgr.size(), CV_32FC1);
    for (const auto& t : tiles) {
        cv::Mat out_tile;
        if (!m_impl->img2img_tile(bgr(t.rect), out_tile, cfg)) {
            spdlog::warn("regen: tile {}x{} at ({},{}) failed", t.rect.width, t.rect.height, t.rect.x, t.rect.y);
            return false;
        }
        if (out_tile.size() != t.rect.size())
            cv::resize(out_tile, out_tile, t.rect.size(), 0, 0, cv::INTER_LANCZOS4);
        cv::Mat out_f; out_tile.convertTo(out_f, CV_32FC3);
        cv::Mat w3; cv::merge(std::vector<cv::Mat>{t.weight, t.weight, t.weight}, w3);
        cv::Mat contrib; cv::multiply(out_f, w3, contrib);
        cv::Mat acc_roi = accum(t.rect), ws_roi = wsum(t.rect);
        cv::add(acc_roi, contrib, acc_roi);
        cv::add(ws_roi, t.weight, ws_roi);
    }
    cv::Mat w3full; cv::merge(std::vector<cv::Mat>{wsum, wsum, wsum}, w3full);
    cv::Mat eps; cv::add(w3full, 1e-6, eps);
    cv::Mat blended; cv::divide(accum, eps, blended);
    blended.convertTo(image, CV_8UC3);   // output is always CV_8UC3, same w/h as input
    return true;
}

bool regenerator_was_used() { return g_ctx_alive.load(); }

} // namespace wmr
#endif
