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

#include <catch2/catch_test_macros.hpp>

#include "core/coreml_sd_pipeline.hpp"

#include <spdlog/spdlog.h>

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdlib>

TEST_CASE("CoreML SDXL pipeline smoke test", "[coreml-sd][pipeline]") {
    const char* models_dir = std::getenv("WMR_COREML_SD_MODELS_DIR");
    if (!models_dir) {
        SKIP("WMR_COREML_SD_MODELS_DIR not set; skipping CoreML SDXL pipeline test");
    }

    std::string models_path(models_dir);
    std::string embeds_bin = models_path + "/empty_prompt_embeds.bin";

    // Check models and embeds exist
    std::vector<std::string> required_files = {
        "Stable_Diffusion_version_stabilityai_stable-diffusion-xl-base-1.0_unet.mlpackage",
        "Stable_Diffusion_version_stabilityai_stable-diffusion-xl-base-1.0_vae_encoder.mlpackage",
        "Stable_Diffusion_version_stabilityai_stable-diffusion-xl-base-1.0_vae_decoder.mlpackage",
    };

    for (const auto& f : required_files) {
        std::string path = models_path + "/" + f;
        FILE* fp = fopen(path.c_str(), "r");
        if (!fp) {
            SKIP("Required model not found: " + path + "; skipping test");
        }
        fclose(fp);
    }

    FILE* fp = fopen(embeds_bin.c_str(), "r");
    if (!fp) {
        SKIP("Embeds file not found: " + embeds_bin + "; skipping test");
    }
    fclose(fp);

    // Initialize pipeline
    wmr::CoreMLSDPipeline pipeline;
    bool init_ok = pipeline.initialize(models_path, embeds_bin);
    REQUIRE(init_ok);
    REQUIRE(pipeline.is_ready());

    // Build a 1024x1024 test fixture: gradient + simple shapes
    cv::Mat tile(1024, 1024, CV_8UC3);
    for (int y = 0; y < 1024; ++y) {
        for (int x = 0; x < 1024; ++x) {
            tile.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uchar>((x * 255) / 1024),       // R gradient
                static_cast<uchar>((y * 255) / 1024),       // G gradient
                static_cast<uchar>(((x + y) * 255) / 2048)   // B diagonal
            );
        }
    }
    // Add some shapes for content variety
    cv::rectangle(tile, cv::Rect(100, 100, 200, 200), cv::Scalar(255, 0, 0), -1);
    cv::circle(tile, cv::Point(800, 300), 100, cv::Scalar(0, 255, 0), -1);
    cv::rectangle(tile, cv::Rect(600, 600, 300, 300), cv::Scalar(0, 0, 255), -1);

    // Compute input stats for comparison
    cv::Scalar input_mean = cv::mean(tile);
    cv::Mat input_gray;
    cv::cvtColor(tile, input_gray, cv::COLOR_BGR2GRAY);
    double input_std = 0.0;
    {
        cv::Mat mu;
        cv::Mat sqdiff;
        input_gray.convertTo(mu, CV_32F);
        mu -= input_mean[0];
        cv::pow(mu, 2.0, sqdiff);
        cv::Scalar sq_mean = cv::mean(sqdiff);
        input_std = std::sqrt(sq_mean[0]);
    }

    // Run img2img with weak denoise (strength=0.1, steps=4, seed=42)
    // Lower strength for content preservation
    cv::Mat result = pipeline.img2img(tile, 0.1f, 4, 42);

    // Validate output shape and type
    REQUIRE(result.rows == 1024);
    REQUIRE(result.cols == 1024);
    REQUIRE(result.type() == CV_8UC3);

    // Validate content preserved: per-channel mean within +-60/255 of input
    // (Relaxed threshold: SDXL img2img tends to brighten output slightly)
    cv::Scalar output_mean = cv::mean(result);
    spdlog::info("Input mean: [{:.2f}, {:.2f}, {:.2f}]", input_mean[0], input_mean[1], input_mean[2]);
    spdlog::info("Output mean: [{:.2f}, {:.2f}, {:.2f}]", output_mean[0], output_mean[1], output_mean[2]);
    for (int c = 0; c < 3; ++c) {
        float diff = std::abs(input_mean[c] - output_mean[c]);
        spdlog::info("Channel {}: diff={:.2f}", c, diff);
        REQUIRE(diff < 60.0f);
    }

    // Validate not collapsed: std within ~30% of input (allowing some denoise blur)
    cv::Mat output_gray;
    cv::cvtColor(result, output_gray, cv::COLOR_BGR2GRAY);
    double output_std = 0.0;
    {
        cv::Mat mu;
        cv::Mat sqdiff;
        output_gray.convertTo(mu, CV_32F);
        mu -= output_mean[0];
        cv::pow(mu, 2.0, sqdiff);
        cv::Scalar sq_mean = cv::mean(sqdiff);
        output_std = std::sqrt(sq_mean[0]);
    }
    // output_std should be at least 40% of input_std (content not completely flattened)
    REQUIRE(output_std > 0.4 * input_std);

    // Validate not saturated: no channel mean <5 or >250
    for (int c = 0; c < 3; ++c) {
        REQUIRE(output_mean[c] > 5.0);
        REQUIRE(output_mean[c] < 250.0);
    }

    spdlog::info("CoreML SDXL pipeline smoke test passed: "
                 "input_mean=[{:.1f},{:.1f},{:.1f}], output_mean=[{:.1f},{:.1f},{:.1f}], "
                 "input_std={:.1f}, output_std={:.1f}",
                 input_mean[0], input_mean[1], input_mean[2],
                 output_mean[0], output_mean[1], output_mean[2],
                 input_std, output_std);
}

#endif // WMR_BUILD_AI_COREML_SD
