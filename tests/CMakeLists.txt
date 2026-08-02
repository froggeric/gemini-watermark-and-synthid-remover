cmake_minimum_required(VERSION 3.21)

find_package(Catch2 REQUIRED)

# Test sources
set(TEST_SOURCES
    unit/blend_modes_test.cpp
    unit/fft_context_test.cpp
    unit/spectral_codebook_test.cpp
    unit/inpaint_test.cpp
    unit/notebooklm_gates_test.cpp
    unit/geometry_detector_test.cpp
    unit/still_geometry_test.cpp
    unit/synthid_wording_test.cpp
    unit/synthid_attack_cli_test.cpp
    unit/codebook_subtractor_test.cpp
    unit/codebook_builder_test.cpp
    unit/lab_a_experiment_test.cpp
    integration/visible_pipeline_test.cpp
)

# Library sources needed by tests
set(LIB_SOURCES
    ${CMAKE_SOURCE_DIR}/src/core/blend_modes.cpp
    ${CMAKE_SOURCE_DIR}/src/core/watermark_engine.cpp
    ${CMAKE_SOURCE_DIR}/src/core/inpaint.cpp
    ${CMAKE_SOURCE_DIR}/src/core/fft_context.cpp
    ${CMAKE_SOURCE_DIR}/src/detection/ncc_detector.cpp
    ${CMAKE_SOURCE_DIR}/src/detection/synthid_detector.cpp
    ${CMAKE_SOURCE_DIR}/src/synthid/spectral_codebook.cpp
    ${CMAKE_SOURCE_DIR}/src/synthid/codebook_subtractor.cpp
    ${CMAKE_SOURCE_DIR}/src/synthid/codebook_builder.cpp
    ${CMAKE_SOURCE_DIR}/src/synthid/noise_residual_subtractor.cpp
    ${CMAKE_SOURCE_DIR}/src/video/notebooklm_gates.cpp  # VideoReader-free (pure OpenCV); safe for the test link
    ${CMAKE_SOURCE_DIR}/src/video/geometry_detector.cpp  # VideoReader-free (pure OpenCV); safe for the test link
    ${CMAKE_SOURCE_DIR}/src/detection/still_geometry.cpp  # VideoReader-free (pure OpenCV); safe for the test link
    # NOTE: src/video/scene_detector.cpp is intentionally excluded — it pulls in
    # VideoReader (FFmpeg), which the test target does not link. No test
    # references SceneDetector today; re-add it (with the FFMPEG link) if a
    # scene-detection test is added.
)

add_executable(wmr_tests ${TEST_SOURCES} ${LIB_SOURCES})

target_include_directories(wmr_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/assets
)

target_link_libraries(wmr_tests PRIVATE
    Catch2::Catch2WithMain
    opencv_core
    opencv_imgcodecs
    opencv_imgproc
    opencv_photo
    fmt::fmt
    spdlog::spdlog
    FFTW3::fftw3f
)

# Register tests with CTest. Run from the source root so the integration tests find
# test-images/ relative to CWD (catch_discover_tests otherwise runs from
# build/tests/, where neither test-images/ nor ../test-images/ resolves and every
# fixture-backed case silently SKIPs).
include(CTest)
include(Catch)
catch_discover_tests(wmr_tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# AI denoise (FDnCNN via NCNN/Vulkan) — mirrors the main CMakeLists block.
# target_sources is used because wmr_tests was already created above; mutating
# the LIB_SOURCES/TEST_SOURCES lists at this point would have no effect.
if(WMR_BUILD_AI_DENOISE)
    target_sources(wmr_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/src/core/ai_denoise.cpp
        ${CMAKE_SOURCE_DIR}/src/core/ai_denoise_model.cpp
        unit/ai_denoise_test.cpp
    )

    if(TARGET volk::volk)
        set(WMR_VOLK_TARGET volk::volk)
    else()
        set(WMR_VOLK_TARGET volk)
    endif()

    target_link_libraries(wmr_tests PRIVATE ncnn ${WMR_VOLK_TARGET})
    target_compile_definitions(wmr_tests PRIVATE WMR_AI_DENOISE=1)
endif()

# SynthID diffusion-regen (SDXL img2img via stable-diffusion.cpp) — mirrors the main
# CMakeLists block. Like WMR_BUILD_AI_DENOISE above, target_sources mutates the
# already-created wmr_tests target. Backend resolver (Task 2) is the first source;
# later tasks append their own lines.
if(WMR_BUILD_REGEN)
    target_sources(wmr_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/src/core/regen_backend.cpp
        ${CMAKE_SOURCE_DIR}/src/core/regen_tiling.cpp
        ${CMAKE_SOURCE_DIR}/src/core/model_downloader.cpp
        ${CMAKE_SOURCE_DIR}/src/core/regenerator.cpp
        unit/regen_backend_test.cpp
        unit/regen_tiling_test.cpp
        unit/regen_downloader_test.cpp
        unit/regen_smoke_test.cpp
        unit/regen_dispatch_test.cpp
    )
    target_link_libraries(wmr_tests PRIVATE stable-diffusion CURL::libcurl OpenSSL::Crypto)
    target_compile_definitions(wmr_tests PRIVATE WMR_BUILD_REGEN=1)
endif()
