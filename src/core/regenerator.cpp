#ifdef WMR_BUILD_REGEN
#include "core/regenerator.hpp"
#include "core/model_downloader.hpp"
#include "core/regen_tiling.hpp"
#include "core/regen_backend.hpp"
#include "core/regen_restore.hpp"
#include "core/paths.hpp"
#include "cli/progress.hpp"
#ifdef WMR_BUILD_AI_COREML_SD
#include "core/coreml_sd_pipeline.hpp"
#include "core/coreml_sd_model_fetch.hpp"
#include "core/coreml_cache.hpp"  // manage_coreml_execution_cache
#include "cli/cli_app.hpp"        // wmr::kHeaderRule (the cleanup notice frame)
#include <cstdio>
#ifdef __APPLE__
#include <unistd.h>  // isatty (the interactive y/N offer)
#endif
#endif
#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <stable-diffusion.h>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
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
    return wmr::user_cache_dir();
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

// Last-run stats (tile count + elapsed + resolved backend label), read by the
// CLI for the SynthID-complete recap. Mutex-guarded for the (single-threaded
// in practice) write/read pair; regen is not called concurrently.
std::mutex g_stats_mu;
RegenRunStats g_last_stats{};

int round_down_8(int n) { return std::max(8, (n / 8) * 8); }

// Resolve the CoreML models directory: env var, exe-dir/../share/wmr/coreml-sdxl,
// exe-dir/coreml-sdxl, or ~/.cache/wmr/coreml-sdxl. Shared by the presence check
// and the fetch so they can never disagree.
fs::path resolve_coreml_models_dir() {
    const char* env_dir = std::getenv("WMR_COREML_SD_MODELS_DIR");
    if (env_dir && env_dir[0]) {
        return fs::path(env_dir);
    }
    fs::path exe = exe_dir();
    fs::path share = exe / ".." / "share" / "wmr" / "coreml-sdxl";
    if (fs::exists(share)) {
        return share;
    }
    if (fs::exists(exe / "coreml-sdxl")) {
        return exe / "coreml-sdxl";
    }
    return cache_dir() / "coreml-sdxl";
}

// Check if CoreML SDXL models are present at the models directory.
// Returns true only if BOTH at least one .mlpackage (unet/VAE/encoder) AND
// empty_prompt_embeds.bin exist. Used to decide whether Auto needs to fetch them.
bool coreml_models_present(const fs::path& models_dir) {
    // Check for the embeds binary first (quick check)
    fs::path embeds_bin = models_dir / "empty_prompt_embeds.bin";
    if (!fs::exists(embeds_bin)) return false;
    // Check for at least one .mlpackage (unet, VAE, or text_encoder)
    // The .mlpackage directories have the full HuggingFace-prefixed names
    std::error_code ec;
    if (!fs::exists(models_dir, ec)) return false;
    for (const auto& entry : fs::directory_iterator(models_dir)) {
        if (entry.is_directory() && entry.path().extension() == ".mlpackage") {
            return true;
        }
    }
    return false;
}

#if defined(__APPLE__) && defined(WMR_BUILD_AI_COREML_SD)
// After the CoreML pipeline is ready, the sdcpp CPU models that a pre-1.16.11
// first run auto-downloaded are dead weight (~7.2 GB): Auto routes to CoreML,
// and --regen-backend cpu re-downloads them on demand. Offer the reclaim with a
// VISIBLE framed stderr notice (this fixes the users already affected by the
// old CPU-default order); prompt y/N only when stdin AND stderr are TTYs, so
// scripted runs never block and just see the notice + the manual command.
// Runs once per process (initialize runs once per process, incl. batch mode).
void offer_leftover_cpu_models_cleanup() {
    LeftoverCpuModels leftovers = find_leftover_cpu_models();
    if (leftovers.paths.empty()) return;
    const double gb = static_cast<double>(leftovers.bytes) / (1024.0 * 1024.0 * 1024.0);
    char size[32];
    if (gb >= 1.0) std::snprintf(size, sizeof(size), "~%.1f GB", gb);
    else std::snprintf(size, sizeof(size), "~%.0f MB", gb * 1024.0);
    const std::string rule(kHeaderRule);
    std::fputs("\n", stderr);
    std::fputs(rule.c_str(), stderr);
    std::fputc('\n', stderr);
    std::fprintf(stderr, "wmr note: the CPU regen models are still in the cache (%s):\n", size);
    for (const auto& p : leftovers.paths) {
        std::fprintf(stderr, "  %s\n", p.string().c_str());
    }
    std::fputs("They are no longer needed now that the CoreML (GPU) backend is in use; an\n"
               "older wmr downloaded them on first run. Deleting is safe: they re-download\n"
               "on demand if you ever pass --regen-backend cpu.\n", stderr);
    std::fputs("Reclaim any time with: wmr cache --clear-cpu-models\n", stderr);
    std::fputs(rule.c_str(), stderr);
    std::fputc('\n', stderr);
    std::fflush(stderr);
    if (!isatty(STDIN_FILENO) || !isatty(STDERR_FILENO)) return;  // scripted: notice only
    std::fputs("Delete them now? [y/N] ", stderr);
    std::fflush(stderr);
    char buf[16] = {0};
    if (std::fgets(buf, sizeof(buf), stdin) == nullptr) return;  // EOF -> default No
    if (buf[0] != 'y' && buf[0] != 'Y') {
        std::fputs("Kept.\n", stderr);
        return;
    }
    const int n = remove_leftover_cpu_models(leftovers);
    if (n > 0) {
        std::fprintf(stderr, "Removed %d file(s), reclaimed %s.\n", n, size);
    } else {
        std::fputs("Could not remove them; try: wmr cache --clear-cpu-models\n", stderr);
    }
}
#endif

} // namespace

struct Regenerator::Impl {
    sd_ctx_t* ctx = nullptr;
    RegenConfig cfg{};
    fs::path vae_path_resolved;
    bool ready = false;
    // Backend label for the tile-line extra + the SynthID-complete recap.
    // Plain compute-unit names: "GPU"/"CPU" (CoreML), "CPU"/"Metal"/"Vulkan"/"CUDA"
    // (sdcpp). For CoreML it is refined after the first predict.
    std::string backend_label;
    // Adaptive stage numbering: 4 stages when a first-run download happens
    // (download -> load -> regen -> restore), 3 stages when models are cached
    // (load -> regen -> restore). Computed once in initialize(); regen() reads it.
    int stage_total = 4;
    int load_stage_idx = 2;
    int regen_stage_idx = 3;
    int restore_stage_idx = 4;
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

    // Adaptive stage numbering: the download stage only counts when it actually
    // runs. Cached -> 3 stages (load/regen/restore -> [1/3]/[2/3]/[3/3]); a
    // first-run download -> 4 stages ([1/4]..[4/4]). This avoids showing "[2/4]"
    // when the user never saw a "[1/4]". Computed per the path that actually
    // runs below (CoreML vs sdcpp); regen() reads the Impl fields.
    auto set_stages_no_dl = [&] {
        m_impl->stage_total = 3;
        m_impl->load_stage_idx = 1;
        m_impl->regen_stage_idx = 2;
        m_impl->restore_stage_idx = 3;
    };
    auto set_stages_with_dl = [&] {
        m_impl->stage_total = 4;
        m_impl->load_stage_idx = 2;
        m_impl->regen_stage_idx = 3;
        m_impl->restore_stage_idx = 4;
    };
    bool dl_stage_shown = false;  // a [1/4] download stage was already printed

    // 1) Resolve the backend FIRST, before any model bytes hit the disk. On a
    //    CoreML-capable mac, Auto prefers the native CoreML pipeline and
    //    BOOTSTRAPS its models (downloads them on first use, ~6 GB); the sdcpp
    //    checkpoint (7.2 GB) is only fetched when the sdcpp path actually runs
    //    (an explicit cpu/metal/vulkan backend, or a CoreML bootstrap failure
    //    falling back). Before 1.16.11 the sdcpp models were downloaded
    //    unconditionally BEFORE the backend was chosen, so every fresh Mac
    //    burned 7.2 GB and then ran the slow CPU path (CoreML was never fetched
    //    by Auto). Metal is broken (upstream sdcpp/ggml bug). Per upstream
    //    docs/backend.md, an empty backend string (Auto, non-Apple) makes
    //    stable-diffusion.cpp auto-select GPU -> integrated GPU -> CPU, so it
    //    never fails for lack of a device and needs no manual sd_list_devices
    //    probe. The resolved string is a const char* literal (static storage),
    //    safe to assign directly to cp.backend.
    RegenBackend b = regen_backend_from_string(cfg.backend);  // ""/auto->Auto, cpu/metal/vulkan/coreml
#if defined(__APPLE__) && defined(WMR_BUILD_AI_COREML_SD)
    if (b == RegenBackend::Auto || b == RegenBackend::CoreML) {
        const bool explicit_coreml = (b == RegenBackend::CoreML);
        fs::path models_dir = resolve_coreml_models_dir();
        const bool present = coreml_models_present(models_dir);
        if (!present && !cfg.allow_download) {
            if (explicit_coreml) {
                spdlog::warn("regen: CoreML models not available (models_dir='{}'). "
                             "Regen did not run; image unchanged. Use --regen-backend cpu for CPU regen, "
                             "or allow downloads by omitting --regen-no-download.", models_dir.string());
                return false;
            }
            spdlog::info("regen: CoreML models not cached and download disabled "
                         "(--regen-no-download); using the CPU backend.");
            b = RegenBackend::Cpu;
        } else {
            if (present) {
                set_stages_no_dl();
                if (!explicit_coreml) {
                    spdlog::info("regen: using the CoreML backend. "
                                 "Override with --regen-backend cpu.");
                }
            } else {
                set_stages_with_dl();
                dl_stage_shown = true;
                spdlog::info("regen: CoreML (GPU) models not cached; downloading them now "
                             "(one-time first run). Override with --regen-backend cpu to use "
                             "the CPU backend instead.");
                Stage dl_stage(1, m_impl->stage_total, "Downloading CoreML models",
                               "one-time first run");
            }
            bool ok = ensure_coreml_models(models_dir, cfg.allow_download,
                                           /*show_progress=*/true);
            if (ok) {
                // Manage the app-scoped CoreML execution cache BEFORE the pipeline
                // initializes (the first pipeline load triggers the CoreML compile that
                // POPULATES the e5bundlecache). The key encodes everything that changes
                // the compile output: wmr version, the three model SHA pins (a re-pin
                // is the common dev case that ballooned the cache), and the macOS
                // version. Best-effort; warns on failure and never throws. See
                // coreml_cache.hpp for the full design + scope.
                //
                // NOTE: CoreML writes its compile cache under <HOME>/Library/Caches/wmr/
                // (the macOS-conventional per-app Caches location), NOT under
                // user_cache_dir() (~/.cache/wmr/, used for the model fetch). The
                // coreml_app_cache_dir() helper resolves the right root.
                {
                    std::string model_key = compose_model_key(
                        APP_VERSION,
                        coreml_unet_sha256(),
                        coreml_vae_encoder_sha256(),
                        coreml_vae_decoder_sha256(),
                        macos_version());
                    manage_coreml_execution_cache(coreml_app_cache_dir(), model_key);
                }
                m_impl->coreml_pipeline = std::make_unique<CoreMLSDPipeline>();
                fs::path embeds_bin = models_dir / "empty_prompt_embeds.bin";
                {
                    // First-load triggers a one-time CoreML compile (cached by CoreML afterwards).
                    Stage load_stage(m_impl->load_stage_idx, m_impl->stage_total,
                                     "Loading CoreML pipeline", "one-time first-run compile, ~30s");
                    if (!m_impl->coreml_pipeline->initialize(models_dir.string(), embeds_bin.string())) {
                        spdlog::warn("regen: CoreML pipeline initialization failed (models_dir='{}').",
                                     models_dir.string());
                        ok = false;
                    }
                }
                if (ok && !m_impl->coreml_pipeline->is_ready()) {
                    spdlog::warn("regen: CoreML pipeline not ready after initialization.");
                    ok = false;
                }
                if (ok) {
                    // Placement (GPU vs CPU) is resolved at the first predict; default
                    // label is "GPU" until then (CoreML on mac is GPU-bound under
                    // MLComputeUnitsAll).
                    m_impl->backend_label = "GPU";
                    m_impl->ready = true;
                    spdlog::debug("regen: CoreML SDXL pipeline ready, models_dir='{}'",
                                  models_dir.string());
                    // Visible only when there is actually something to reclaim.
                    offer_leftover_cpu_models_cleanup();
                    return true;
                }
            }
            if (explicit_coreml) {
                spdlog::warn("regen: CoreML models not available (models_dir='{}'). "
                             "Regen did not run; image unchanged. Use --regen-backend cpu for CPU regen, "
                             "or allow downloads by omitting --regen-no-download.", models_dir.string());
                return false;
            }
            spdlog::warn("regen: CoreML backend unavailable; falling back to the CPU backend.");
            b = RegenBackend::Cpu;
        }
    }
#elif defined(__APPLE__)
    if (b == RegenBackend::Auto) {
        // macOS without CoreML build: CPU only (Metal is broken).
        b = RegenBackend::Cpu;
        spdlog::warn("regen: Metal backend is broken on Apple Silicon (upstream sdcpp/ggml bug); "
                     "using CPU (~3x slower, impractical for 4K tiled). Override with "
                     "--regen-backend {metal,vulkan} at your own risk.");
    }
#endif

    // 2) sdcpp path (explicit cpu/metal/vulkan, non-Apple auto, or the CPU
    //    fallback from a failed CoreML attempt): resolve + download the model
    //    (base SDXL fp16 checkpoint) and the fp16-fix VAE. The download is a
    //    first-run-only cost; on a warm cache neither file is needed.
    fs::path model = resolve_model(cfg);
    const bool need_model_dl = !fs::exists(model) || fs::file_size(model) < (1ull<<30);
    fs::path vae = resolve_vae(cfg);
    m_impl->vae_path_resolved = vae;
    const bool need_vae_dl = vae.empty() || !fs::exists(vae) || fs::file_size(vae) < (1ull<<24);
    const bool any_download = (need_model_dl || need_vae_dl) && cfg.allow_download;
    if ((need_model_dl || need_vae_dl) && !cfg.allow_download) {
        spdlog::warn("regen: model/VAE absent and download disabled");
        return false;
    }
    if (any_download) {
        set_stages_with_dl();
        Stage dl_stage(1, m_impl->stage_total, "Downloading model", "one-time first run");
    } else if (!dl_stage_shown) {
        set_stages_no_dl();
    }
    if (need_model_dl) {
        auto r = download_pinned_file(kRegenModelUrl, model, kRegenModelSha256, true,
                                      /*allow_empty_hash=*/false,
                                      make_byte_progress(model.filename().string()));
        if (!r.ok) { spdlog::warn("regen: model download failed: {}", r.error); return false; }
    }
    if (need_vae_dl) {
        auto r = download_pinned_file(kRegenVaeUrl, vae, kRegenVaeSha256, true,
                                      /*allow_empty_hash=*/false,
                                      make_byte_progress(vae.filename().string()));
        if (!r.ok) { spdlog::warn("regen: fp16-fix VAE download failed: {}", r.error); return false; }
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
    {
        Stage load_stage(m_impl->load_stage_idx, m_impl->stage_total,
                         "Loading model", "one-time first run");
        m_impl->ctx = new_sd_ctx(&cp);
    }
    // Resolve the backend label for the tile-line extra + recap. b is the resolved
    // (non-Auto) backend here; plain compute-unit names for the user-facing line.
    switch (b) {
        case RegenBackend::Cpu:    m_impl->backend_label = "CPU"; break;
        case RegenBackend::Metal:  m_impl->backend_label = "Metal"; break;
        case RegenBackend::Vulkan: m_impl->backend_label = "Vulkan"; break;
        case RegenBackend::Cuda:   m_impl->backend_label = "CUDA"; break;
        default: m_impl->backend_label = backend[0] ? backend : "auto"; break;
    }
    if (!m_impl->ctx || !sd_ctx_supports_image_generation(m_impl->ctx)) {
        spdlog::warn("regen: sd_ctx creation failed (backend='{}', model='{}')",
                     backend[0] ? backend : "auto", model_s);
        m_impl->ready = false; return false;
    }
    m_impl->ready = true;
    g_ctx_alive.store(true);   // the ggml backend is now alive; its static teardown will abort
    spdlog::debug("regen: sd_ctx ready, backend='{}', model='{}', vae='{}'",
                  backend[0] ? backend : "auto", model_s, vae_s);
    return true;
}
bool Regenerator::is_ready() const { return m_impl && m_impl->ready; }

// Normalize any input to BGR 8UC3 (grayscale -> BGR, RGBA -> BGR, non-u8 depth -> u8).
// cv::imread already returns BGR 8UC3 by default; this is a safety net for programmatic callers.
// MUST return a deep copy: regen() keeps `bgr` as the pre-regen original O and later writes
// the regen output R back into `image` (blended.convertTo(image) in the tiled path, or
// cv::resize into image). A shallow `bgr = image` shares the buffer, so that write
// silently corrupts O -> R, making D = O - R = 0 in restore_detail and zeroing D_att
// (the Wiener becomes a no-op, R' = R, ~28/255 end-to-end error on the restored region).
static cv::Mat normalize_to_bgr8u3(const cv::Mat& image) {
    cv::Mat bgr;
    if (image.channels() == 1)      cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
    else if (image.channels() == 4) cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
    else                            bgr = image.clone();   // deep copy (regen writes into `image`)
    if (bgr.depth() != CV_8U) bgr.convertTo(bgr, CV_8U);
    return bgr;
}

bool Regenerator::regen(cv::Mat& image, const RegenConfig& cfg) {
    if (!is_ready()) return false;
    // Reset per-run stats (the recap reads these on success).
    { std::lock_guard<std::mutex> lk(g_stats_mu); g_last_stats = RegenRunStats{}; }
    auto t_regen_start = std::chrono::steady_clock::now();
    cv::Mat bgr = normalize_to_bgr8u3(image);   // operate on a normalized copy
    const bool need_tile = cfg.tile && (bgr.cols > cfg.tile_size || bgr.rows > cfg.tile_size);

    // Helper: per-tile timing + the resolved backend label for the reporter extra.
    // Plain "<GPU|CPU|Metal|...>, <X.X>s/tile" — the per-unit seconds carry the
    // speed; the rate field on the line handles the rest. (The first-predict ms
    // is a dev-only debug log now, not crammed into the tile line.)
    auto backend_extra = [&](double tile_s) -> std::string {
#ifdef WMR_BUILD_AI_COREML_SD
        if (m_impl->coreml_pipeline && m_impl->coreml_pipeline->is_ready()) {
            // The first predict ran inside img2img_tile above and resolved the
            // placement ("GPU"/"CPU"); fold it into the backend label.
            std::string label = m_impl->coreml_pipeline->placement_label();
            if (!label.empty()) m_impl->backend_label = label;
        }
#endif
        return fmt::format("{}, {:.1f}s/tile", m_impl->backend_label, tile_s);
    };
    const std::string regen_label =
        fmt::format("[{}/{}] Regenerating", m_impl->regen_stage_idx, m_impl->stage_total);

    if (!need_tile) {
        ProgressReporter rep(regen_label, 1, "tile");
        auto t_tile0 = std::chrono::steady_clock::now();
        cv::Mat out;
        if (!m_impl->img2img_tile(bgr, out, cfg)) { rep.finish(); return false; }
        double tile_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_tile0).count();
        rep.update(1, backend_extra(tile_s));
        rep.finish();
        if (out.size() != image.size())
            cv::resize(out, image, image.size(), 0, 0, cv::INTER_LANCZOS4);
        else image = out;
        // Detail restoration (post-regen, on the full-res R). `bgr` is O (the
        // pre-regen original); `image` is R. Always called so the chosen branch
        // (Auto gate, forced On, or Off) is logged; Off/Auto-skip return R unchanged.
        Stage restore_stage(m_impl->restore_stage_idx, m_impl->stage_total, "Detail restoration");
        image = restore_detail(bgr, image, cfg.restore);
        auto t_end = std::chrono::steady_clock::now();
        { std::lock_guard<std::mutex> lk(g_stats_mu);
          g_last_stats = {std::chrono::duration<double>(t_end - t_regen_start).count(),
                          1, m_impl->backend_label}; }
        return true;
    }
    auto tiles = build_regen_tiles(bgr.cols, bgr.rows, cfg.tile_size, cfg.overlap);
    if (tiles.empty()) return false;
    cv::Mat accum = cv::Mat::zeros(bgr.size(), CV_32FC3);
    cv::Mat wsum  = cv::Mat::zeros(bgr.size(), CV_32FC1);
    ProgressReporter rep(regen_label, static_cast<int>(tiles.size()), "tile");
    for (size_t i = 0; i < tiles.size(); ++i) {
        const auto& t = tiles[i];
        auto t_tile0 = std::chrono::steady_clock::now();
        cv::Mat out_tile;
        if (!m_impl->img2img_tile(bgr(t.rect), out_tile, cfg)) {
            spdlog::warn("regen: tile {}x{} at ({},{}) failed", t.rect.width, t.rect.height, t.rect.x, t.rect.y);
            rep.finish();
            return false;
        }
        double tile_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_tile0).count();
        rep.update(static_cast<int>(i + 1), backend_extra(tile_s));
        if (out_tile.size() != t.rect.size())
            cv::resize(out_tile, out_tile, t.rect.size(), 0, 0, cv::INTER_LANCZOS4);
        cv::Mat out_f; out_tile.convertTo(out_f, CV_32FC3);
        cv::Mat w3; cv::merge(std::vector<cv::Mat>{t.weight, t.weight, t.weight}, w3);
        cv::Mat contrib; cv::multiply(out_f, w3, contrib);
        cv::Mat acc_roi = accum(t.rect), ws_roi = wsum(t.rect);
        cv::add(acc_roi, contrib, acc_roi);
        cv::add(ws_roi, t.weight, ws_roi);
    }
    rep.finish();
    cv::Mat w3full; cv::merge(std::vector<cv::Mat>{wsum, wsum, wsum}, w3full);
    cv::Mat eps; cv::add(w3full, 1e-6, eps);
    cv::Mat blended; cv::divide(accum, eps, blended);
    blended.convertTo(image, CV_8UC3);   // output is always CV_8UC3, same w/h as input
    // Detail restoration on the tiled-then-reassembled full-res R (same as the
    // single-tile path). Always called so the branch is logged.
    Stage restore_stage(m_impl->restore_stage_idx, m_impl->stage_total, "Detail restoration");
    image = restore_detail(bgr, image, cfg.restore);
    auto t_end = std::chrono::steady_clock::now();
    { std::lock_guard<std::mutex> lk(g_stats_mu);
      g_last_stats = {std::chrono::duration<double>(t_end - t_regen_start).count(),
                      static_cast<int>(tiles.size()), m_impl->backend_label}; }
    return true;
}

bool regenerator_was_used() { return g_ctx_alive.load(); }

RegenRunStats last_regen_run_stats() {
    std::lock_guard<std::mutex> lk(g_stats_mu);
    return g_last_stats;
}

} // namespace wmr
#endif
