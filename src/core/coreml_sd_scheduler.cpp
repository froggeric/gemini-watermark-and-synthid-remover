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

#ifdef WMR_BUILD_AI_COREML_SD

#include "core/coreml_sd_scheduler.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include <Accelerate/Accelerate.h>

namespace wmr {

CoreMLSDEulerScheduler::CoreMLSDEulerScheduler()
    : num_inference_steps_(0), step_index_(0) {
    init();
}

CoreMLSDEulerScheduler::~CoreMLSDEulerScheduler() = default;

void CoreMLSDEulerScheduler::init() {
    // Compute betas: scaled_linear schedule
    // betas = linspace(sqrt(beta_start), sqrt(beta_end), num_train_timesteps)^2
    betas_.resize(kNumTrainTimesteps);
    float sqrt_start = std::sqrt(kBetaStart);
    float sqrt_end = std::sqrt(kBetaEnd);
    for (int i = 0; i < kNumTrainTimesteps; ++i) {
        float t = static_cast<float>(i) / (kNumTrainTimesteps - 1);
        float sqrt_beta = sqrt_start + t * (sqrt_end - sqrt_start);
        betas_[i] = sqrt_beta * sqrt_beta;
    }

    // Compute alphas = 1 - betas
    alphas_.resize(kNumTrainTimesteps);
    for (int i = 0; i < kNumTrainTimesteps; ++i) {
        alphas_[i] = 1.0f - betas_[i];
    }

    // Compute alphas_cumprod = cumprod(alphas)
    alphas_cumprod_.resize(kNumTrainTimesteps);
    std::partial_sum(alphas_.begin(), alphas_.end(), alphas_cumprod_.begin(),
                    std::multiplies<float>());

    // Compute train sigmas: sqrt((1 - alphas_cumprod) / alphas_cumprod), flipped
    // Then append 0 at the end (so length is num_train_timesteps + 1)
    train_sigmas_.resize(kNumTrainTimesteps + 1);
    for (int i = 0; i < kNumTrainTimesteps; ++i) {
        float alpha_cumprod = alphas_cumprod_[i];
        train_sigmas_[kNumTrainTimesteps - 1 - i] =
            std::sqrt((1.0f - alpha_cumprod) / alpha_cumprod);
    }
    train_sigmas_[kNumTrainTimesteps] = 0.0f;
}

void CoreMLSDEulerScheduler::set_timesteps(int num_inference_steps) {
    num_inference_steps_ = num_inference_steps;
    step_index_ = 0;

    // Leading timestep spacing (SDXL default)
    // step_ratio = num_train_timesteps // num_inference_steps
    // timesteps = (arange(0, num_inference_steps) * step_ratio).round()[::-1] + steps_offset
    int step_ratio = kNumTrainTimesteps / num_inference_steps;
    timesteps_.resize(num_inference_steps);
    for (int i = 0; i < num_inference_steps; ++i) {
        // i from 0 to num_inference_steps-1 maps to reversed order
        int reversed_idx = num_inference_steps - 1 - i;
        timesteps_[i] = static_cast<int>(std::round(reversed_idx * step_ratio)) + 1;
    }

    // Interpolate sigmas at the timesteps (linear interpolation)
    // timesteps are in [0, num_train_timesteps), need to map to sigmas
    sigmas_.resize(num_inference_steps + 1);

    // For each timestep, find the corresponding sigma via linear interpolation
    // The train_sigmas correspond to timesteps [999, 998, ..., 0, terminal]
    // So train_sigmas[i] corresponds to timestep = num_train_timesteps - 1 - i
    for (int i = 0; i < num_inference_steps; ++i) {
        int t = timesteps_[i];
        // Find the index in train_sigmas that corresponds to this timestep
        // train_sigmas[idx] corresponds to timestep = num_train_timesteps - 1 - idx
        // So idx = num_train_timesteps - 1 - t
        int idx = kNumTrainTimesteps - 1 - t;
        if (idx >= 0 && idx < kNumTrainTimesteps) {
            sigmas_[i] = train_sigmas_[idx];
        } else {
            // Should not happen with valid timesteps
            sigmas_[i] = train_sigmas_[std::max(0, std::min(idx, kNumTrainTimesteps))];
        }
    }
    // Final sigma is 0
    sigmas_[num_inference_steps] = 0.0f;
}

std::vector<int> CoreMLSDEulerScheduler::img2img_timesteps(
    int num_inference_steps, float strength) const {
    // diffusers: t_start = num_inference_steps - int(num_inference_steps * strength)
    int t_start = num_inference_steps - static_cast<int>(num_inference_steps * strength);

    // Need to compute the timesteps for the full inference first
    CoreMLSDEulerScheduler temp;
    temp.set_timesteps(num_inference_steps);

    // Return truncated timesteps from t_start to end
    std::vector<int> result;
    for (int i = t_start; i < num_inference_steps; ++i) {
        result.push_back(temp.timesteps_[i]);
    }
    return result;
}

void CoreMLSDEulerScheduler::add_noise(const float* sample, const float* noise,
                                      int timestep, float* out, size_t n) const {
    // Find sigma for this timestep
    // Matches diffusers index_for_timestep logic: find first match in timesteps
    int sigma_idx = -1;
    for (size_t i = 0; i < timesteps_.size(); ++i) {
        if (timesteps_[i] == timestep) {
            sigma_idx = static_cast<int>(i);
            break;
        }
    }

    // If not found in inference timesteps, fall back to train sigmas
    float sigma;
    if (sigma_idx >= 0 && sigma_idx < static_cast<int>(sigmas_.size())) {
        sigma = sigmas_[sigma_idx];
    } else {
        // Map timestep to train_sigmas index directly
        int idx = kNumTrainTimesteps - 1 - timestep;
        sigma = (idx >= 0 && idx < kNumTrainTimesteps) ? train_sigmas_[idx] : 0.0f;
    }

    // noisy_sample = sample + noise * sigma
    // Use Accelerate for vector operations
    // out = sample + sigma * noise
    // cblas_saxpy(n, alpha, x, 1, y, 1) computes y = alpha * x + y
    std::copy(sample, sample + n, out);
    cblas_saxpy(static_cast<int>(n), sigma, noise, 1, out, 1);
}

void CoreMLSDEulerScheduler::scale_model_input(const float* sample, int step_index,
                                               float* out, size_t n) const {
    // scale = 1 / sqrt(sigma^2 + 1)
    if (step_index < 0 || step_index >= static_cast<int>(sigmas_.size())) {
        // Invalid step index, should not happen
        std::copy(sample, sample + n, out);
        return;
    }

    float sigma = sigmas_[step_index];
    float scale = 1.0f / std::sqrt(sigma * sigma + 1.0f);

    // out = sample * scale
    vDSP_vsmul(sample, 1, &scale, out, 1, n);
}

void CoreMLSDEulerScheduler::step(const float* model_output, int step_index,
                                  const float* sample, float* out, size_t n) {
    // Euler step with epsilon prediction
    if (step_index < 0 || step_index + 1 >= static_cast<int>(sigmas_.size())) {
        // Invalid step index, should not happen
        std::copy(sample, sample + n, out);
        return;
    }

    float sigma = sigmas_[step_index];
    float sigma_next = sigmas_[step_index + 1];

    // For epsilon prediction:
    // pred_original_sample = sample - sigma * model_output
    std::vector<float> pred_original(n);
    std::copy(sample, sample + n, pred_original.data());
    cblas_saxpy(static_cast<int>(n), -sigma, model_output, 1, pred_original.data(), 1);

    // derivative = (sample - pred_original_sample) / sigma
    // But for epsilon: sample - (sample - sigma * model_output) = sigma * model_output
    // So derivative = model_output
    // This is the same as diffusers' implementation for epsilon prediction
    std::vector<float> derivative(n);
    vDSP_vsub(pred_original.data(), 1, sample, 1, derivative.data(), 1, n);
    vDSP_vsdiv(derivative.data(), 1, &sigma, derivative.data(), 1, n);

    // dt = sigma_next - sigma
    float dt = sigma_next - sigma;

    // prev_sample = sample + derivative * dt
    std::copy(sample, sample + n, out);
    cblas_saxpy(static_cast<int>(n), dt, derivative.data(), 1, out, 1);

    // Increment step index
    step_index_++;
}

} // namespace wmr

#endif // WMR_BUILD_AI_COREML_SD
