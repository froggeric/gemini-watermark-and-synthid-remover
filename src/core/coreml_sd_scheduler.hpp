// Copyright 2025 wmr contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#ifdef WMR_BUILD_AI_COREML_SD

#include <cstddef>
#include <cstdint>
#include <vector>

namespace wmr {

/**
 * EulerDiscreteScheduler for SDXL img2img denoising.
 *
 * C++ port of diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler
 * with SDXL defaults. Uses Apple Accelerate framework for vector math.
 *
 * Config (from SDXL scheduler_config.json):
 * - num_train_timesteps = 1000
 * - beta_start = 0.00085, beta_end = 0.012
 * - beta_schedule = "scaled_linear"
 * - prediction_type = "epsilon"
 * - timestep_spacing = "leading", steps_offset = 1
 * - interpolation_type = "linear"
 */
class CoreMLSDEulerScheduler {
public:
    // SDXL default config
    static constexpr int kNumTrainTimesteps = 1000;
    static constexpr float kBetaStart = 0.00085f;
    static constexpr float kBetaEnd = 0.012f;

    CoreMLSDEulerScheduler();
    ~CoreMLSDEulerScheduler();

    // Initialize the train schedule (betas, alphas, alphas_cumprod, sigmas)
    // Called once during construction.
    void init();

    // Set inference timesteps for denoising loop.
    // Computes inference sigmas (length num_steps+1, last = 0) and timesteps.
    void set_timesteps(int num_inference_steps);

    // Set img2img timesteps for denoising loop.
    // Computes truncated timesteps based on strength and sets sigmas_/timesteps_
    // to the truncated schedule. Use this for img2img so the denoise loop can
    // iterate step_index 0..(k-1) over the correct sigmas.
    void set_timesteps_img2img(int num_inference_steps, float strength);

    // Get img2img truncated timesteps for a given strength.
    // diffusers: t_start = num_inference_steps - int(num_inference_steps * strength)
    // Returns timesteps[t_start:]
    std::vector<int> img2img_timesteps(int num_inference_steps, float strength) const;

    // Add noise to sample at a given timestep.
    // noisy_sample = sample + noise * sigma
    void add_noise(const float* sample, const float* noise, int timestep,
                   float* out, size_t n) const;

    // Scale model input by 1/sqrt(sigma^2 + 1).
    // Called before model forward pass.
    void scale_model_input(const float* sample, int step_index,
                           float* out, size_t n) const;

    // Euler step: compute prev_sample from model_output and sample.
    // For epsilon prediction:
    //   pred_original_sample = sample - sigma * model_output
    //   derivative = (sample - pred_original_sample) / sigma
    //   dt = sigma_next - sigma
    //   prev_sample = sample + derivative * dt
    // Increments internal step counter.
    void step(const float* model_output, int step_index,
              const float* sample, float* out, size_t n);

    // Accessors
    const std::vector<float>& sigmas() const { return sigmas_; }
    const std::vector<int>& timesteps() const { return timesteps_; }
    const std::vector<float>& alphas_cumprod() const { return alphas_cumprod_; }
    int step_index() const { return step_index_; }

private:
    // Train schedule (computed in init)
    std::vector<float> betas_;
    std::vector<float> alphas_;
    std::vector<float> alphas_cumprod_;
    std::vector<float> train_sigmas_;  // Length kNumTrainTimesteps + 1

    // Inference schedule (computed in set_timesteps)
    std::vector<float> sigmas_;       // Length num_inference_steps + 1
    std::vector<int> timesteps_;       // Length num_inference_steps
    int num_inference_steps_;

    // Step state
    int step_index_;
};

} // namespace wmr

#endif // WMR_BUILD_AI_COREML_SD
