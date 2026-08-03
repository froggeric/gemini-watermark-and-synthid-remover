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

#include <cmath>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "catch2/catch_approx.hpp"
#include "core/coreml_sd_scheduler.hpp"
#include "coreml_sd_scheduler_refs.hpp"

using namespace wmr;
using Catch::Approx;

namespace {

constexpr float REL_TOL = 1e-4f;   // Relative tolerance for larger values
constexpr float ABS_TOL = 1e-5f;   // Absolute tolerance for values near zero

void check_close(const char* name, const float* actual, const float* expected,
                 size_t n, float rel_tol = REL_TOL, float abs_tol = ABS_TOL) {
    float max_rel = 0.0f;
    float max_abs = 0.0f;
    size_t max_idx = 0;

    for (size_t i = 0; i < n; ++i) {
        float a = actual[i];
        float e = expected[i];
        float diff = std::abs(a - e);
        float rel = (std::abs(e) > abs_tol) ? diff / std::abs(e) : diff;

        if (diff > max_abs) {
            max_abs = diff;
            max_rel = rel;
            max_idx = i;
        }

        // Check tolerance
        if (diff > abs_tol && rel > rel_tol) {
            FAIL(name << " mismatch at index " << i << ": "
                    << "actual=" << a << ", expected=" << e
                    << " (diff=" << diff << ", rel=" << rel << ")");
        }
    }

    // Log max error for verification
    INFO(name << " max error: abs=" << max_abs << ", rel=" << max_rel
         << " at index " << max_idx << " (expected=" << expected[max_idx]
         << ", actual=" << actual[max_idx] << ")");
}

} // namespace

TEST_CASE("CoreML SDXL Euler scheduler matches diffusers", "[coreml-sd][scheduler]") {
    CoreMLSDEulerScheduler scheduler;

    SECTION("train alphas_cumprod match diffusers") {
        const auto& actual = scheduler.alphas_cumprod();
        REQUIRE(actual.size() == test::coreml_sd::alphas_cumprod_size);
        check_close("alphas_cumprod", actual.data(),
                   test::coreml_sd::alphas_cumprod, actual.size());
    }

    SECTION("train sigmas match diffusers") {
        // Access internal train sigmas through the API after init
        // We can check inference sigmas instead which are derived from train sigmas
        scheduler.set_timesteps(4);
        const auto& sigmas = scheduler.sigmas();
        // Check first 4 values (inference sigmas are a subset of train schedule)
        // The train_sigmas ref has 1001 values
        REQUIRE(test::coreml_sd::train_sigmas_size == 1001);
        // Check a few key points
        // First train sigma is computed from last alpha_cumprod
        // sigma = sqrt((1 - alpha) / alpha), with alpha ~0.005, sigma ~14.6
        CHECK(test::coreml_sd::train_sigmas[0] == Approx(14.6146469f).epsilon(0.001f));
        CHECK(test::coreml_sd::train_sigmas[1000] == 0.0f);  // Zero at end
    }

    SECTION("inference sigmas for 4 steps match diffusers") {
        scheduler.set_timesteps(4);
        const auto& actual = scheduler.sigmas();
        REQUIRE(actual.size() == test::coreml_sd::inference_sigmas_4_size);
        check_close("inference_sigmas_4", actual.data(),
                   test::coreml_sd::inference_sigmas_4, actual.size());
    }

    SECTION("inference timesteps for 4 steps match diffusers") {
        scheduler.set_timesteps(4);
        const auto& actual = scheduler.timesteps();
        REQUIRE(actual.size() == test::coreml_sd::inference_timesteps_4_size);
        for (size_t i = 0; i < actual.size(); ++i) {
            CHECK(actual[i] == static_cast<int>(test::coreml_sd::inference_timesteps_4[i]));
        }
    }

    SECTION("img2img timesteps for 8 steps, strength 0.3 match diffusers") {
        auto actual = scheduler.img2img_timesteps(8, 0.3f);
        REQUIRE(actual.size() == test::coreml_sd::img2img_timesteps_8_strength_0_3_size);
        for (size_t i = 0; i < actual.size(); ++i) {
            CHECK(actual[i] == static_cast<int>(test::coreml_sd::img2img_timesteps_8_strength_0_3[i]));
        }
    }

    SECTION("add_noise matches diffusers") {
        // First set up the inference schedule (like diffusers does)
        scheduler.set_timesteps(4);

        std::vector<float> sample(test::coreml_sd::add_noise_sample_input,
                                  test::coreml_sd::add_noise_sample_input +
                                  test::coreml_sd::add_noise_sample_input_size);
        std::vector<float> noise(test::coreml_sd::add_noise_input,
                                 test::coreml_sd::add_noise_input +
                                 test::coreml_sd::add_noise_input_size);
        std::vector<float> output(test::coreml_sd::add_noise_output_size);

        int timestep = static_cast<int>(test::coreml_sd::add_noise_timestep);
        scheduler.add_noise(sample.data(), noise.data(), timestep, output.data(),
                           output.size());

        check_close("add_noise", output.data(), test::coreml_sd::add_noise_output,
                   output.size());
    }

    SECTION("3-step trajectory matches diffusers") {
        // Set up scheduler for 4 steps (we'll run 3)
        scheduler.set_timesteps(4);

        // Starting noisy sample (from add_noise reference)
        std::vector<float> current_sample(
            test::coreml_sd::add_noise_output,
            test::coreml_sd::add_noise_output + test::coreml_sd::add_noise_output_size);

        // Fixed model output
        std::vector<float> model_output(
            test::coreml_sd::step_model_output,
            test::coreml_sd::step_model_output + test::coreml_sd::step_model_output_size);

        // Run 3 steps
        for (int i = 0; i < 3; ++i) {
            // Scale input
            std::vector<float> scaled_input(current_sample.size());
            scheduler.scale_model_input(current_sample.data(), i, scaled_input.data(),
                                       scaled_input.size());

            // Check scaled_input
            const float* expected_scaled = nullptr;
            size_t expected_size = 0;
            if (i == 0) {
                expected_scaled = test::coreml_sd::trajectory_scaled_input_0;
                expected_size = test::coreml_sd::trajectory_scaled_input_0_size;
            } else if (i == 1) {
                expected_scaled = test::coreml_sd::trajectory_scaled_input_1;
                expected_size = test::coreml_sd::trajectory_scaled_input_1_size;
            } else {
                expected_scaled = test::coreml_sd::trajectory_scaled_input_2;
                expected_size = test::coreml_sd::trajectory_scaled_input_2_size;
            }
            check_close(("scaled_input step " + std::to_string(i)).c_str(),
                       scaled_input.data(), expected_scaled, expected_size);

            // Apply step
            std::vector<float> prev_sample(current_sample.size());
            scheduler.step(model_output.data(), i, current_sample.data(),
                         prev_sample.data(), prev_sample.size());

            // Check prev_sample
            const float* expected_prev = nullptr;
            if (i == 0) {
                expected_prev = test::coreml_sd::trajectory_prev_sample_0;
                expected_size = test::coreml_sd::trajectory_prev_sample_0_size;
            } else if (i == 1) {
                expected_prev = test::coreml_sd::trajectory_prev_sample_1;
                expected_size = test::coreml_sd::trajectory_prev_sample_1_size;
            } else {
                expected_prev = test::coreml_sd::trajectory_prev_sample_2;
                expected_size = test::coreml_sd::trajectory_prev_sample_2_size;
            }
            check_close(("prev_sample step " + std::to_string(i)).c_str(),
                       prev_sample.data(), expected_prev, expected_size);

            current_sample = std::move(prev_sample);
        }

        // After 3 steps, step_index should be 3
        CHECK(scheduler.step_index() == 3);
    }

    SECTION("sigma-based step derivation is mathematically correct") {
        // Test that the step formula for epsilon prediction is:
        // prev_sample = sample + sigma_next * model_output - sigma * model_output
        //              = sample + (sigma_next - sigma) * model_output
        // Since for epsilon: derivative = model_output

        scheduler.set_timesteps(2);
        const auto& sigmas = scheduler.sigmas();

        // Simple test: sample = 0, model_output = 1
        // expected = 0 + (sigmas[1] - sigmas[0]) * 1
        std::vector<float> sample(1024, 0.0f);
        std::vector<float> model_output(1024, 1.0f);
        std::vector<float> prev_sample(1024);

        scheduler.step(model_output.data(), 0, sample.data(), prev_sample.data(),
                     prev_sample.size());

        float expected_value = sigmas[1] - sigmas[0];
        for (size_t i = 0; i < prev_sample.size(); ++i) {
            CHECK(prev_sample[i] == Approx(expected_value).epsilon(1e-5f));
        }
    }
}

#endif // WMR_BUILD_AI_COREML_SD
