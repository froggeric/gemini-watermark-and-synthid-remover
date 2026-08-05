#ifdef WMR_BUILD_REGEN
#include "core/regenerator.hpp"
#include "core/model_downloader.hpp"
#include "core/regen_tiling.hpp"
#include "core/regen_backend.hpp"
#ifdef WMR_BUILD_AI_COREML_SD
#include "core/coreml_sd_pipeline.hpp"
#include "core/coreml_sd_model_fetch.hpp"
#endif
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

// Pinned sources, mirrored to the project's own HuggingFace repo (froggeric/wmr) so the
// CPU regen path depends only on our own infra (not the upstream stabilityai / madebyollin
// repos). SHA256 == the HuggingFace LFS content oid (the HF tree API exposes it as
// `lfs.sha256`), which is content-addressed: mirroring the exact upstream bytes to
// froggeric/wmr leaves both oids unchanged. The VAE oid was cross-verified by downloading
// the 335 MB file and running `shasum -a 256` (byte-for-byte match); the model oid is
// trusted the same way without downloading the 6.5 GB checkpoint. The fp16-fix VAE file is
// named `sdxl_vae.safetensors` (NOT `sdxl_vae-fp16-fix.safetensors`; that path 404s).
constexpr const char* kRegenModelUrl =
    "https://huggingface.co/froggeric/wmr/resolve/main/sd_xl_base_1.0.safetensors";
constexpr const char* kRegenModelSha256 =
    "31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b";
constexpr const char* kRegenVaeUrl =
    "https://huggingface.co/froggeric/wmr/resolve/main/sdxl_vae.safetensors";
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

// Check if CoreML SDXL models are present at the resolved models directory.
// Returns true only if BOTH at least one .mlpackage (unet/VAE/encoder) AND empty_prompt_embeds.bin exist.
// This allows Auto to fall back to CPU on mac when the 6.5 GB models are absent.
bool coreml_models_present() {
    fs::path models_dir;
    const char* env_dir = std::getenv("WMR_COREML_SD_MODELS_DIR");
    if (env_dir && env_dir[0]) {
        models_dir = fs::path(env_dir);
    } else {
        fs::path exe = exe_dir();
        fs::path share = exe / ".." / "share" / "wmr" / "coreml-sdxl";
        if (fs::exists(share)) {
            models_dir = share;
        } else if (fs::exists(exe / "coreml-sdxl")) {
            models_dir = exe / "coreml-sdxl";
        } else {
            models_dir = cache_dir() / "coreml-sdxl";
        }
    }
    // Check for the embeds binary first (quick check)
    fs::path embeds_bin = models_dir / "empty_prompt_embeds.bin";
    if (!fs::exists(embeds_bin)) return false;
    // Check for at least one .mlpackage (unet, VAE, or text_encoder)
    // The .mlpackage directories have the full HuggingFace-prefixed names
    for (const auto& entry : fs::directory_iterator(models_dir)) {
        if (entry.is_directory() && entry.path().extension() == ".mlpackage") {
            return true;
        }
    }
    return false;
}

} // namespace

struct Regenerator::Impl {
    sd_ctx_t* ctx = nullptr;
    RegenConfig cfg{};
    fs::path vae_path_resolved;
    bool ready = false;
#ifdef WMR_BUILD_AI_COREML_SD
    std::unique_ptr<CoreMLSDPipeline> coreml_pipeline;
#endif
    ~Impl() { /* intentionally NOT freeing ctx here; see header + ImplDeleter */ }
    void destroy() { if (ctx) { free_sd_ctx(ctx); ctx = nullptr; } }

    bool img2img_tile(const cv::Mat& bgr_tile, cv::Mat& out_tile, const RegenConfig& c) {
#ifdef WMR_BUILD_AI_COREML_SD
        if (coreml_pipeline && coreml_pipeline->is_ready()) {
            // CoreML path: run one 1024x1024 tile. The caller's tiled dispatch (regen())
            // already handles >1024 images by calling img2img_tile per tile.
            // Resize input to 1024x1024 (CoreML requirement), then resize output back.
            cv::Mat resized_1024;
            cv::Size original_size = bgr_tile.size();
            if (bgr_tile.size() != cv::Size(1024, 1024)) {
                cv::resize(bgr_tile, resized_1024, cv::Size(1024, 1024), 0, 0, cv::INTER_AREA);
            } else {
                resized_1024 = bgr_tile;
            }
            uint64_t seed = 42;  // TODO: make configurable via --regen-seed
            cv::Mat result = coreml_pipeline->img2img(resized_1024, c.strength, c.steps, seed);
            if (result.empty()) return false;
            // Resize output back to original tile size
            if (result.size() != original_size) {
                cv::resize(result, out_tile, original_size, 0, 0, cv::INTER_LANCZOS4);
            } else {
                out_tile = result;
            }
            return true;
        }
#endif
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
        // Deterministic Euler (NOT Euler-A). The validated removal knee (CoreML,
        // strength 0.10 @ N=50 = int(50*0.10) = 5 denoise steps) uses
        // CoreMLSDEulerScheduler: a deterministic Euler on the standard discrete
        // sigma schedule. sdcpp's EULER_A is ancestral (re-injects noise each step)
        // and over-denoises at the same 0.10 strength: more lossy, NOT a lower
        // removal threshold. EULER (deterministic) + DISCRETE_SCHEDULER (the SDXL
        // default sd_get_default_scheduler returns) matches the CoreML schedule, so
        // the 0.10 knee applies uniformly across backends. Do NOT lower the strength
        // on CPU instead: at N=50, strength 0.05 collapses to int(2.5) = 2 steps and
        // under-removes. Keep 0.10.
        p.sample_params.sample_method = EULER_SAMPLE_METHOD;
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

    // 3) Backend: resolve cfg.backend ("auto"/"cpu"/"metal"/"vulkan"/"coreml").
    //    On Apple Silicon, Auto prefers CoreML (fast, correct) if the 6.5 GB .mlpackage
    //    models are present, else falls back to CPU (so a user without models still gets
    //    working CPU regen, not a silent skip). Metal is broken (upstream sdcpp/ggml bug).
    //    Override with --regen-backend. Per upstream docs/backend.md, an empty backend
    //    (Auto, non-Apple) makes stable-diffusion.cpp auto-select GPU -> integrated GPU ->
    //    CPU, so it never fails for lack of a device and needs no manual sd_list_devices
    //    probe. The resolved string is a const char* literal (static storage), safe to
    //    assign directly to cp.backend.
    RegenBackend b = regen_backend_from_string(cfg.backend);  // ""/auto->Auto, cpu/metal/vulkan/coreml
    if (b == RegenBackend::Auto) {
#if defined(__APPLE__) && defined(WMR_BUILD_AI_COREML_SD)
        // On macOS with CoreML build: Auto prefers CoreML if models are present,
        // else falls back to CPU (slow but correct).
        if (coreml_models_present()) {
            b = RegenBackend::CoreML;
            spdlog::info("regen: CoreML models detected, using CoreML backend (fast, correct). "
                         "Override with --regen-backend cpu.");
        } else {
            b = RegenBackend::Cpu;
            spdlog::info("regen: CoreML models not found, using CPU backend (slow, ~3 min for 4K). "
                         "For fast regen, download the .mlpackage models to ~/.cache/wmr/coreml-sdxl "
                         "or set WMR_COREML_SD_MODELS_DIR. Override with --regen-backend.");
        }
#elif defined(__APPLE__)
        // macOS without CoreML build: CPU only (Metal is broken).
        b = RegenBackend::Cpu;
        spdlog::warn("regen: Metal backend is broken on Apple Silicon (upstream sdcpp/ggml bug); "
                     "using CPU (~3x slower, impractical for 4K tiled). Override with "
                     "--regen-backend {metal,vulkan} at your own risk.");
#endif
    }

#ifdef WMR_BUILD_AI_COREML_SD
    // CoreML backend: use the native CoreML pipeline instead of sdcpp.
    // Auto routes to CoreML on mac when models are present.
    if (b == RegenBackend::CoreML) {
        // Resolve models directory: env var, exe-dir/../share/wmr/coreml-sdxl,
        // exe-dir/coreml-sdxl, or ~/.cache/wmr/coreml-sdxl.
        fs::path models_dir;
        const char* env_dir = std::getenv("WMR_COREML_SD_MODELS_DIR");
        if (env_dir && env_dir[0]) {
            models_dir = fs::path(env_dir);
        } else {
            fs::path exe = exe_dir();
            fs::path share = exe / ".." / "share" / "wmr" / "coreml-sdxl";
            if (fs::exists(share)) {
                models_dir = share;
            } else if (fs::exists(exe / "coreml-sdxl")) {
                models_dir = exe / "coreml-sdxl";
            } else {
                models_dir = cache_dir() / "coreml-sdxl";
            }
        }
        // Fetch models from HuggingFace if missing (unless --regen-no-download)
        if (!ensure_coreml_models(models_dir, cfg.allow_download)) {
            spdlog::warn("regen: CoreML models not available (models_dir='{}'). "
                         "Regen did not run; image unchanged. Use --regen-backend cpu for CPU regen, "
                         "or allow downloads by omitting --regen-no-download.", models_dir.string());
            return false;
        }
        m_impl->coreml_pipeline = std::make_unique<CoreMLSDPipeline>();
        fs::path embeds_bin = models_dir / "empty_prompt_embeds.bin";
        if (!m_impl->coreml_pipeline->initialize(models_dir.string(), embeds_bin.string())) {
            spdlog::warn("regen: CoreML pipeline initialization failed (models_dir='{}'). "
                         "Regen did not run; image unchanged.", models_dir.string());
            return false;
        }
        if (!m_impl->coreml_pipeline->is_ready()) {
            spdlog::warn("regen: CoreML pipeline not ready after initialization. Regen did not run; image unchanged.");
            return false;
        }
        m_impl->ready = true;
        spdlog::info("regen: CoreML SDXL pipeline ready, models_dir='{}'", models_dir.string());
        return true;
    }
#endif
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
