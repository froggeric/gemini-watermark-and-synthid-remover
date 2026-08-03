/**
 * @file    coreml_sd_pipeline.mm
 * @brief   CoreML SDXL img2img pipeline for SynthID regen (Apple Silicon)
 * @license Apache-2.0 (project)
 *
 * @details
 * ObjC++ impl of `wmr::CoreMLSDPipeline`. Runs Stable Diffusion XL base-1.0
 * via native CoreML on the ANE. Empty prompt, CFG guidance=1.0 (unconditional).
 * Models are converted .mlpackage (Task 1); scheduler is CoreMLSDEulerScheduler
 * (Task 2); text embeddings are pre-baked.
 *
 * Pipeline: BGR tile -> RGB [-1,1] -> VAE encode -> latent distribution ->
 * sample + add_noise (scheduler) -> Euler denoise loop (UNet + step) ->
 * VAE decode -> RGB [-1,1] -> BGR.
 *
 * IO verified (Task 1 load spike):
 * - unet: sample (2,4,128,128) fp16, timestep (2) fp16, encoder_hidden_states
 *   (2,2048,1,77) fp16 -> noise_pred (2,4,128,128) fp32
 * - vae_encoder: x (1,3,1024,1024) fp16 -> latent (1,8,128,128) fp32
 *   (mean+logvar concatenated)
 * - vae_decoder: z (1,4,128,128) fp16 -> image (1,3,1024,1024) fp32 [-1,1]
 *
 * VAE scaling factor: 0.13025 (SDXL vae/config.json).
 *
 * Build: Objective-C++ (.mm), linked against CoreML+Foundation+Accelerate.
 * `WMR_BUILD_AI_COREML_SD` gates this TU.
 */

#ifdef WMR_BUILD_AI_COREML_SD

#include "core/coreml_sd_pipeline.hpp"
#include "core/coreml_sd_scheduler.hpp"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <Accelerate/Accelerate.h>
#include <opencv2/imgproc.hpp>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mach-o/dyld.h>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace wmr {
namespace {

// Directory containing the running executable (macOS).
std::filesystem::path exe_dir() {
    char buf[4096];
    uint32_t sz = sizeof(buf);
    if (_NSGetExecutablePath(buf, &sz) == 0)
        return std::filesystem::weakly_canonical(buf).parent_path();
    return std::filesystem::current_path();
}

// SDXL VAE scaling factor (from vae/config.json)
constexpr float kVAEScalingFactor = 0.13025f;

// Latent and tile dimensions (SDXL base-1.0)
constexpr int kLatentSize = 128;
constexpr int kTileSize = 1024;
constexpr int kEmbedBatch = 2;  // CFG batch size (uncond duplicated)

// Box-Muller transform for standard normal distribution
float box_muller(uint32_t& state) {
    // Simple LCG for state
    state = state * 1664525u + 1013904223u;
    float u1 = (state & 0xFFFFFFu) / 16777216.0f;
    state = state * 1664525u + 1013904223u;
    float u2 = (state & 0xFFFFFFu) / 16777216.0f;
    const float pi = 3.14159265358979323846f;
    // Box-Muller: r = sqrt(-2 * ln(u1)), theta = 2 * pi * u2
    float r = std::sqrt(-2.0f * std::log(std::max(1e-6f, u1)));
    float theta = 2.0f * pi * u2;
    return r * std::cos(theta);
}

// Fill buffer with standard normal noise (Box-Muller, seeded)
void fill_std_normal(float* buf, size_t n, uint32_t& state) {
    for (size_t i = 0; i < n; i += 2) {
        float u1 = (state & 0xFFFFFFu) / 16777216.0f;
        state = state * 1664525u + 1013904223u;
        float u2 = (state & 0xFFFFFFu) / 16777216.0f;
        state = state * 1664525u + 1013904223u;
        const float pi = 3.14159265358979323846f;
        // Box-Muller: r = sqrt(-2 * ln(u1)), theta = 2 * pi * u2
        float r = std::sqrt(-2.0f * std::log(std::max(1e-6f, u1)));
        float theta = 2.0f * pi * u2;
        buf[i] = r * std::cos(theta);
        if (i + 1 < n) buf[i + 1] = r * std::sin(theta);
    }
}

} // namespace

struct CoreMLSDPipeline::Impl {
    MLModel* unet = nil;
    MLModel* vae_encoder = nil;
    MLModel* vae_decoder = nil;
    std::vector<__fp16> embeds;  // (2,2048,1,77) fp16
    CoreMLSDEulerScheduler scheduler;
    bool ready = false;
};

CoreMLSDPipeline::CoreMLSDPipeline() : m_impl(std::make_unique<Impl>()) {}

CoreMLSDPipeline::~CoreMLSDPipeline() {
    // Leaked singleton in practice (CoreML teardown races static destruction)
#if !__has_feature(objc_arc)
    if (m_impl) {
        if (m_impl->unet) { [m_impl->unet release]; m_impl->unet = nil; }
        if (m_impl->vae_encoder) { [m_impl->vae_encoder release]; m_impl->vae_encoder = nil; }
        if (m_impl->vae_decoder) { [m_impl->vae_decoder release]; m_impl->vae_decoder = nil; }
    }
#endif
}

bool CoreMLSDPipeline::initialize(const std::string& models_dir, const std::string& embeds_bin) {
    if (m_impl->ready) return true;

    std::filesystem::path models_path(models_dir);
    if (!std::filesystem::exists(models_path)) {
        spdlog::warn("CoreML SDXL models directory not found: {}", models_dir);
        return false;
    }

    // Load baked embeddings
    std::ifstream embeds_file(embeds_bin, std::ios::binary);
    if (!embeds_file) {
        spdlog::warn("CoreML SDXL embeddings not found: {}", embeds_bin);
        return false;
    }

    constexpr size_t kEmbedSize = kEmbedBatch * 2048 * 1 * 77;  // (2,2048,1,77)
    m_impl->embeds.resize(kEmbedSize);
    embeds_file.read(reinterpret_cast<char*>(m_impl->embeds.data()), kEmbedSize * sizeof(__fp16));
    if (embeds_file.gcount() != static_cast<std::streamsize>(kEmbedSize * sizeof(__fp16))) {
        spdlog::warn("CoreML SDXL embeddings size mismatch (expected {}, got {})",
                     kEmbedSize * sizeof(__fp16), embeds_file.gcount());
        return false;
    }

    // Embeds loaded successfully
    spdlog::debug("CoreML SDXL embeddings loaded: {} elements", kEmbedSize);

    @try {
        NSError* err = nil;
        MLModelConfiguration* cfg = [[MLModelConfiguration alloc] init];
        cfg.computeUnits = MLComputeUnitsAll;  // ANE preferred, GPU/CPU fallback

        // Load UNet
        std::filesystem::path unet_path = models_path / "Stable_Diffusion_version_stabilityai_stable-diffusion-xl-base-1.0_unet.mlpackage";
        NSURL* unet_url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:unet_path.c_str()]];
        NSURL* unet_compiled = [MLModel compileModelAtURL:unet_url error:&err];
        if (!unet_compiled) {
            spdlog::warn("CoreML SDXL UNet compile failed: {}",
                         err ? [[err localizedDescription] UTF8String] : "unknown");
            return false;
        }
        MLModel* unet = [MLModel modelWithContentsOfURL:unet_compiled configuration:cfg error:&err];
        if (!unet) {
            spdlog::warn("CoreML SDXL UNet load failed: {}",
                         err ? [[err localizedDescription] UTF8String] : "unknown");
            return false;
        }
#if !__has_feature(objc_arc)
        m_impl->unet = [unet retain];
#else
        m_impl->unet = unet;
#endif

        // Load VAE encoder
        std::filesystem::path vae_enc_path = models_path / "Stable_Diffusion_version_stabilityai_stable-diffusion-xl-base-1.0_vae_encoder.mlpackage";
        NSURL* vae_enc_url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:vae_enc_path.c_str()]];
        NSURL* vae_enc_compiled = [MLModel compileModelAtURL:vae_enc_url error:&err];
        if (!vae_enc_compiled) {
            spdlog::warn("CoreML SDXL VAE encoder compile failed: {}",
                         err ? [[err localizedDescription] UTF8String] : "unknown");
            return false;
        }
        MLModel* vae_encoder = [MLModel modelWithContentsOfURL:vae_enc_compiled configuration:cfg error:&err];
        if (!vae_encoder) {
            spdlog::warn("CoreML SDXL VAE encoder load failed: {}",
                         err ? [[err localizedDescription] UTF8String] : "unknown");
            return false;
        }
#if !__has_feature(objc_arc)
        m_impl->vae_encoder = [vae_encoder retain];
#else
        m_impl->vae_encoder = vae_encoder;
#endif

        // Load VAE decoder
        std::filesystem::path vae_dec_path = models_path / "Stable_Diffusion_version_stabilityai_stable-diffusion-xl-base-1.0_vae_decoder.mlpackage";
        NSURL* vae_dec_url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:vae_dec_path.c_str()]];
        NSURL* vae_dec_compiled = [MLModel compileModelAtURL:vae_dec_url error:&err];
        if (!vae_dec_compiled) {
            spdlog::warn("CoreML SDXL VAE decoder compile failed: {}",
                         err ? [[err localizedDescription] UTF8String] : "unknown");
            return false;
        }
        MLModel* vae_decoder = [MLModel modelWithContentsOfURL:vae_dec_compiled configuration:cfg error:&err];
        if (!vae_decoder) {
            spdlog::warn("CoreML SDXL VAE decoder load failed: {}",
                         err ? [[err localizedDescription] UTF8String] : "unknown");
            return false;
        }
#if !__has_feature(objc_arc)
        m_impl->vae_decoder = [vae_decoder retain];
#else
        m_impl->vae_decoder = vae_decoder;
#endif

        m_impl->ready = true;
        spdlog::info("CoreML SDXL pipeline ready (models={}, embeds={})",
                     models_dir, embeds_bin);
    } @catch (NSException* ex) {
        spdlog::warn("CoreML SDXL init exception: {}", [[ex reason] UTF8String]);
        m_impl->ready = false;
        return false;
    }
    return true;
}

bool CoreMLSDPipeline::is_ready() const { return m_impl->ready; }

cv::Mat CoreMLSDPipeline::img2img(const cv::Mat& tile_bgr, float strength, int steps, uint64_t seed) {
    if (!m_impl->ready) return {};

    // Validate input
    if (tile_bgr.rows != kTileSize || tile_bgr.cols != kTileSize || tile_bgr.type() != CV_8UC3) {
        spdlog::warn("CoreML SDXL img2img: input must be 1024x1024 BGR uint8");
        return {};
    }

    @autoreleasepool {
        // --- Step 1: Preprocess BGR -> RGB [-1,1] (1,3,1024,1024) fp16 ---
        cv::Mat rgb;
        cv::cvtColor(tile_bgr, rgb, cv::COLOR_BGR2RGB);
        cv::Mat rgb_float;
        rgb.convertTo(rgb_float, CV_32F, 1.0 / 127.5, -1.0);  // [0,255] -> [-1,1]

        // --- Step 2: VAE encode -> latent distribution ---
        // Build input: (1,3,1024,1024) fp16 PLANAR. rgb_float is interleaved HWC
        // (CV_32FC3); de-interleave so each channel plane is contiguous (matches the
        // MLMultiArray NCHW layout the encoder expects). A plain element-wise copy
        // would shuffle spatial + channel data and garble the encode.
        std::vector<__fp16> vae_enc_input(3 * kTileSize * kTileSize);
        const int H = kTileSize, W = kTileSize;
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < H; ++h) {
                const cv::Vec3f* row = rgb_float.ptr<cv::Vec3f>(h);
                __fp16* dst_plane = vae_enc_input.data() + static_cast<size_t>(c) * H * W
                                 + static_cast<size_t>(h) * W;
                for (int w = 0; w < W; ++w) dst_plane[w] = static_cast<__fp16>(row[w][c]);
            }
        }

        NSError* err = nil;
        NSArray<NSNumber*>* shape3 = @[@1, @3, @(kTileSize), @(kTileSize)];
        NSArray<NSNumber*>* strides3 = @[@(3 * kTileSize * kTileSize), @(kTileSize * kTileSize), @(kTileSize), @1];
        MLMultiArray* vae_enc_ma = [[MLMultiArray alloc]
            initWithDataPointer:reinterpret_cast<void*>(vae_enc_input.data())
                          shape:shape3
                       dataType:MLMultiArrayDataTypeFloat16
                        strides:strides3
                     deallocator:nil
                          error:&err];
        if (!vae_enc_ma) {
            spdlog::warn("CoreML SDXL VAE encoder input array failed");
            return {};
        }

        MLDictionaryFeatureProvider* vae_enc_fp = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{@"x": vae_enc_ma} error:&err];
        if (!vae_enc_fp) {
            spdlog::warn("CoreML SDXL VAE encoder feature provider failed");
            return {};
        }

        id<MLFeatureProvider> vae_enc_out = [m_impl->vae_encoder predictionFromFeatures:vae_enc_fp
                                                                                 options:[[MLPredictionOptions alloc] init]
                                                                                   error:&err];
        if (!vae_enc_out) {
            spdlog::warn("CoreML SDXL VAE encoder prediction failed: {}",
                         err ? [[err localizedDescription] UTF8String] : "unknown");
            return {};
        }

        MLMultiArray* latent_ma = [[vae_enc_out featureValueForName:@"latent"] multiArrayValue];
        if (!latent_ma) {
            spdlog::warn("CoreML SDXL VAE encoder: latent output missing");
            return {};
        }

        // latent is (1,8,128,128) fp32 = concatenated mean (channels 0-4) and logvar (channels 4-8)
        std::vector<float> latent_full(kLatentSize * kLatentSize * 8);
        if (latent_ma.dataType == MLMultiArrayDataTypeFloat32) {
            std::memcpy(latent_full.data(), latent_ma.dataPointer, latent_full.size() * sizeof(float));
        } else if (latent_ma.dataType == MLMultiArrayDataTypeFloat16) {
            const __fp16* p = static_cast<const __fp16*>(latent_ma.dataPointer);
            for (size_t i = 0; i < latent_full.size(); ++i) latent_full[i] = static_cast<float>(p[i]);
        } else {
            spdlog::warn("CoreML SDXL VAE encoder: unsupported output dtype {}", (int)latent_ma.dataType);
            return {};
        }

        // Split mean (channels 0-3) and logvar (channels 4-7) from the planar
        // (1,8,128,128) encoder output, storing each as planar (4,128,128). The
        // elementwise latent ops + UNet (planar (2,4,128,128)) + VAE decoder then
        // share one consistent layout. Indexing the planar source as interleaved
        // (i*8+c) would shuffle channels and garble the latent.
        constexpr size_t kLatentPlane = static_cast<size_t>(kLatentSize) * kLatentSize;  // 128*128
        std::vector<float> mean(4 * kLatentPlane);
        std::vector<float> logvar(4 * kLatentPlane);
        const float* src_latent = latent_full.data();
        float* dst_mean = mean.data();
        float* dst_logvar = logvar.data();
        for (int c = 0; c < 4; ++c) {
            for (size_t i = 0; i < kLatentPlane; ++i) {
                dst_mean[c * kLatentPlane + i]   = src_latent[c * kLatentPlane + i];
                dst_logvar[c * kLatentPlane + i] = src_latent[(4 + c) * kLatentPlane + i];
            }
        }

        // Sample latent: mean + std * noise, where std = exp(0.5 * clamp(logvar, -30, 20))
        std::vector<float> latent(kLatentSize * kLatentSize * 4);
        uint32_t rng_state = static_cast<uint32_t>(seed);
        fill_std_normal(latent.data(), latent.size(), rng_state);
        for (size_t i = 0; i < latent.size(); ++i) {
            float logvar_clamped = std::clamp(dst_logvar[i], -30.0f, 20.0f);
            float std = std::exp(0.5f * logvar_clamped);
            latent[i] = dst_mean[i] + std * latent[i];
        }

        // Apply VAE scaling factor
        vDSP_vsmul(latent.data(), 1, &kVAEScalingFactor, latent.data(), 1, latent.size());

        // --- Step 3: Euler scheduler setup ---
        // Use set_timesteps_img2img to set up the truncated schedule directly
        m_impl->scheduler.set_timesteps_img2img(steps, strength);
        const auto& timesteps = m_impl->scheduler.timesteps();
        if (timesteps.empty()) {
            spdlog::warn("CoreML SDXL img2img: no timesteps for strength={}, steps={}", strength, steps);
            return {};
        }

        // --- Step 4: Add initial noise (img2img) ---
        std::vector<float> noise(kLatentSize * kLatentSize * 4);
        rng_state = static_cast<uint32_t>(seed);
        fill_std_normal(noise.data(), noise.size(), rng_state);

        std::vector<float> noisy_latent(kLatentSize * kLatentSize * 4);
        // Use the first timestep from the truncated schedule
        m_impl->scheduler.add_noise(latent.data(), noise.data(), timesteps.front(),
                                    noisy_latent.data(), latent.size());

        // --- Step 5: Denoise loop ---
        std::vector<float> scaled_latent(kLatentSize * kLatentSize * 4);
        std::vector<float> latent_out(kLatentSize * kLatentSize * 4);

        for (size_t step_idx = 0; step_idx < timesteps.size(); ++step_idx) {
            // Scale model input
            m_impl->scheduler.scale_model_input(noisy_latent.data(), static_cast<int>(step_idx),
                                                scaled_latent.data(), latent.size());

            // Build UNet input: sample = cat(scaled, scaled) along batch -> (2,4,128,128) fp16
            std::vector<__fp16> unet_sample(kEmbedBatch * 4 * kLatentSize * kLatentSize);
            for (size_t i = 0; i < 4 * kLatentSize * kLatentSize; ++i) {
                unet_sample[i] = static_cast<__fp16>(scaled_latent[i]);
                unet_sample[4 * kLatentSize * kLatentSize + i] = static_cast<__fp16>(scaled_latent[i]);
            }

            // timestep = (2,) fp16, filled with current timestep
            std::vector<__fp16> unet_timestep(kEmbedBatch);
            for (int i = 0; i < kEmbedBatch; ++i) {
                unet_timestep[i] = static_cast<__fp16>(timesteps[step_idx]);
            }

            // Build UNet inputs
            NSArray<NSNumber*>* sample_shape = @[@(kEmbedBatch), @4, @(kLatentSize), @(kLatentSize)];
            NSArray<NSNumber*>* sample_strides = @[@(4 * kLatentSize * kLatentSize), @(kLatentSize * kLatentSize), @(kLatentSize), @1];
            MLMultiArray* sample_ma = [[MLMultiArray alloc]
                initWithDataPointer:reinterpret_cast<void*>(unet_sample.data())
                              shape:sample_shape
                           dataType:MLMultiArrayDataTypeFloat16
                            strides:sample_strides
                         deallocator:nil
                              error:&err];
            if (!sample_ma) {
                spdlog::warn("CoreML SDXL UNet sample array failed");
                return {};
            }

            NSArray<NSNumber*>* timestep_shape = @[@(kEmbedBatch)];
            NSArray<NSNumber*>* timestep_strides = @[@1];
            MLMultiArray* timestep_ma = [[MLMultiArray alloc]
                initWithDataPointer:reinterpret_cast<void*>(unet_timestep.data())
                              shape:timestep_shape
                           dataType:MLMultiArrayDataTypeFloat16
                            strides:timestep_strides
                         deallocator:nil
                              error:&err];
            if (!timestep_ma) {
                spdlog::warn("CoreML SDXL UNet timestep array failed");
                return {};
            }

            NSArray<NSNumber*>* embed_shape = @[@(kEmbedBatch), @2048, @1, @77];
            NSArray<NSNumber*>* embed_strides = @[@(2048 * 77), @77, @77, @1];
            MLMultiArray* embed_ma = [[MLMultiArray alloc]
                initWithDataPointer:reinterpret_cast<void*>(m_impl->embeds.data())
                              shape:embed_shape
                           dataType:MLMultiArrayDataTypeFloat16
                            strides:embed_strides
                         deallocator:nil
                              error:&err];
            if (!embed_ma) {
                spdlog::warn("CoreML SDXL UNet embed array failed");
                return {};
            }

            MLDictionaryFeatureProvider* unet_fp = [[MLDictionaryFeatureProvider alloc]
                initWithDictionary:@{@"sample": sample_ma, @"timestep": timestep_ma,
                                   @"encoder_hidden_states": embed_ma} error:&err];
            if (!unet_fp) {
                spdlog::warn("CoreML SDXL UNet feature provider failed");
                return {};
            }

            id<MLFeatureProvider> unet_out = [m_impl->unet predictionFromFeatures:unet_fp
                                                                         options:[[MLPredictionOptions alloc] init]
                                                                           error:&err];
            if (!unet_out) {
                spdlog::warn("CoreML SDXL UNet prediction failed: {}",
                             err ? [[err localizedDescription] UTF8String] : "unknown");
                return {};
            }

            MLMultiArray* noise_pred_ma = [[unet_out featureValueForName:@"noise_pred"] multiArrayValue];
            if (!noise_pred_ma) {
                spdlog::warn("CoreML SDXL UNet: noise_pred output missing");
                return {};
            }

            // Read noise_pred (2,4,128,128) fp32, take batch 0 (CFG with guidance 1.0)
            std::vector<float> noise_pred(kLatentSize * kLatentSize * 4);
            if (noise_pred_ma.dataType == MLMultiArrayDataTypeFloat32) {
                const float* p = static_cast<const float*>(noise_pred_ma.dataPointer);
                std::memcpy(noise_pred.data(), p, noise_pred.size() * sizeof(float));
            } else if (noise_pred_ma.dataType == MLMultiArrayDataTypeFloat16) {
                const __fp16* p = static_cast<const __fp16*>(noise_pred_ma.dataPointer);
                for (size_t i = 0; i < noise_pred.size(); ++i) noise_pred[i] = static_cast<float>(p[i]);
            } else {
                spdlog::warn("CoreML SDXL UNet: unsupported noise_pred dtype {}", (int)noise_pred_ma.dataType);
                return {};
            }

            // Euler step
            m_impl->scheduler.step(noise_pred.data(), static_cast<int>(step_idx),
                                   noisy_latent.data(), latent_out.data(), latent.size());
            noisy_latent = latent_out;
        }

        // --- Step 6: VAE decode ---
        // Divide by scaling factor
        vDSP_vsdiv(noisy_latent.data(), 1, &kVAEScalingFactor, noisy_latent.data(), 1, noisy_latent.size());

        // Convert to fp16 and decode
        std::vector<__fp16> vae_dec_input(4 * kLatentSize * kLatentSize);
        for (size_t i = 0; i < 4 * kLatentSize * kLatentSize; ++i) {
            vae_dec_input[i] = static_cast<__fp16>(noisy_latent[i]);
        }

        NSArray<NSNumber*>* dec_shape = @[@1, @4, @(kLatentSize), @(kLatentSize)];
        NSArray<NSNumber*>* dec_strides = @[@(4 * kLatentSize * kLatentSize), @(kLatentSize * kLatentSize), @(kLatentSize), @1];
        MLMultiArray* vae_dec_ma = [[MLMultiArray alloc]
            initWithDataPointer:reinterpret_cast<void*>(vae_dec_input.data())
                          shape:dec_shape
                       dataType:MLMultiArrayDataTypeFloat16
                        strides:dec_strides
                     deallocator:nil
                          error:&err];
        if (!vae_dec_ma) {
            spdlog::warn("CoreML SDXL VAE decoder input array failed");
            return {};
        }

        MLDictionaryFeatureProvider* vae_dec_fp = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{@"z": vae_dec_ma} error:&err];
        if (!vae_dec_fp) {
            spdlog::warn("CoreML SDXL VAE decoder feature provider failed");
            return {};
        }

        id<MLFeatureProvider> vae_dec_out = [m_impl->vae_decoder predictionFromFeatures:vae_dec_fp
                                                                                 options:[[MLPredictionOptions alloc] init]
                                                                                   error:&err];
        if (!vae_dec_out) {
            spdlog::warn("CoreML SDXL VAE decoder prediction failed: {}",
                         err ? [[err localizedDescription] UTF8String] : "unknown");
            return {};
        }

        MLMultiArray* image_ma = [[vae_dec_out featureValueForName:@"image"] multiArrayValue];
        if (!image_ma) {
            spdlog::warn("CoreML SDXL VAE decoder: image output missing");
            return {};
        }

        // Read image (1,3,1024,1024) fp32 in [-1,1], denormalize to [0,255], RGB->BGR
        std::vector<float> image_float(3 * kTileSize * kTileSize);
        if (image_ma.dataType == MLMultiArrayDataTypeFloat32) {
            std::memcpy(image_float.data(), image_ma.dataPointer, image_float.size() * sizeof(float));
        } else if (image_ma.dataType == MLMultiArrayDataTypeFloat16) {
            const __fp16* p = static_cast<const __fp16*>(image_ma.dataPointer);
            for (size_t i = 0; i < image_float.size(); ++i) image_float[i] = static_cast<float>(p[i]);
        } else {
            spdlog::warn("CoreML SDXL VAE decoder: unsupported image dtype {}", (int)image_ma.dataType);
            return {};
        }

        // VAE output is (1,3,1024,1024) in planar format (channels-first).
        // Convert to interleaved RGB (R,G,B, R,G,B, ...) for OpenCV.
        std::vector<float> image_interleaved(3 * kTileSize * kTileSize);
        const size_t plane_size = kTileSize * kTileSize;
        for (size_t i = 0; i < plane_size; ++i) {
            image_interleaved[i * 3 + 0] = image_float[0 * plane_size + i];  // R
            image_interleaved[i * 3 + 1] = image_float[1 * plane_size + i];  // G
            image_interleaved[i * 3 + 2] = image_float[2 * plane_size + i];  // B
        }

        // Denormalize [-1,1] -> [0,255]
        cv::Mat result_rgb(kTileSize, kTileSize, CV_32FC3, image_interleaved.data());
        cv::Mat result_uint8;
        result_rgb.convertTo(result_uint8, CV_8UC3, 127.5, 127.5);  // (x+1)*127.5
        cv::Mat result_bgr;
        cv::cvtColor(result_uint8, result_bgr, cv::COLOR_RGB2BGR);

        return result_bgr.clone();
    }
}

} // namespace wmr

#endif // WMR_BUILD_AI_COREML_SD
