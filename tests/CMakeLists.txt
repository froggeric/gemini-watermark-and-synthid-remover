cmake_minimum_required(VERSION 3.21)

find_package(Catch2 REQUIRED)

# Test sources
set(TEST_SOURCES
    unit/blend_modes_test.cpp
    unit/inpaint_test.cpp
    unit/notebooklm_gates_test.cpp
    unit/geometry_detector_test.cpp
    unit/still_geometry_test.cpp
    unit/synthid_attack_cli_test.cpp
    unit/progress_test.cpp
    unit/provenance_test.cpp
    unit/png_chunks_test.cpp
    unit/jpeg_markers_test.cpp
    integration/visible_pipeline_test.cpp
)

# Library sources needed by tests
set(LIB_SOURCES
    ${CMAKE_SOURCE_DIR}/src/core/blend_modes.cpp
    ${CMAKE_SOURCE_DIR}/src/core/watermark_engine.cpp
    ${CMAKE_SOURCE_DIR}/src/core/inpaint.cpp
    ${CMAKE_SOURCE_DIR}/src/detection/ncc_detector.cpp
    ${CMAKE_SOURCE_DIR}/src/video/notebooklm_gates.cpp  # VideoReader-free (pure OpenCV); safe for the test link
    ${CMAKE_SOURCE_DIR}/src/video/geometry_detector.cpp  # VideoReader-free (pure OpenCV); safe for the test link
    ${CMAKE_SOURCE_DIR}/src/detection/still_geometry.cpp  # VideoReader-free (pure OpenCV); safe for the test link
    ${CMAKE_SOURCE_DIR}/src/core/paths.cpp  # ungated; regenerator.cpp (regen mirror block) calls user_cache_dir()
    ${CMAKE_SOURCE_DIR}/src/cli/progress.cpp  # ungated; shared by video/remove/batch (always built)
    ${CMAKE_SOURCE_DIR}/src/metadata/provenance.cpp  # ungated; provenance strip (no OpenCV/FFmpeg)
    ${CMAKE_SOURCE_DIR}/src/metadata/png_chunks.cpp   # ungated; pure TU, paths.cpp precedent
    ${CMAKE_SOURCE_DIR}/src/metadata/jpeg_markers.cpp # ungated; pure TU, paths.cpp precedent
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
    # Mirror of the main CMakeLists kissfft static lib (regen_restore uses it).
    add_library(kissfft_tests STATIC
        ${CMAKE_SOURCE_DIR}/external/kissfft/kiss_fft.c
        ${CMAKE_SOURCE_DIR}/external/kissfft/kiss_fftnd.c
    )
    target_include_directories(kissfft_tests PUBLIC ${CMAKE_SOURCE_DIR}/external/kissfft)
    if(MSVC)
        target_compile_options(kissfft_tests PRIVATE /W1)
    else()
        target_compile_options(kissfft_tests PRIVATE -w)
    endif()

    target_sources(wmr_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/src/core/regen_backend.cpp
        ${CMAKE_SOURCE_DIR}/src/core/regen_tiling.cpp
        ${CMAKE_SOURCE_DIR}/src/core/model_downloader.cpp
        ${CMAKE_SOURCE_DIR}/src/core/regenerator.cpp
        ${CMAKE_SOURCE_DIR}/src/core/regen_restore.cpp
        unit/regen_backend_test.cpp
        unit/regen_tiling_test.cpp
        unit/regen_downloader_test.cpp
        unit/regen_smoke_test.cpp
        unit/regen_dispatch_test.cpp
        unit/regen_restore_test.cpp
    )
    target_link_libraries(wmr_tests PRIVATE stable-diffusion CURL::libcurl OpenSSL::Crypto kissfft_tests)
    target_compile_definitions(wmr_tests PRIVATE WMR_BUILD_REGEN=1)
endif()

# CoreML SDXL img2img (Euler scheduler + pipeline) — mirrors the main CMakeLists block.
if(WMR_BUILD_AI_COREML_SD)
    if(NOT APPLE)
        message(FATAL_ERROR "WMR_BUILD_AI_COREML_SD is macOS-only (requires Accelerate framework)")
    endif()

    find_library(ACCELERATE_LIB Accelerate REQUIRED)
    find_library(COREML_LIB CoreML REQUIRED)
    find_library(FOUNDATION_LIB Foundation REQUIRED)

    enable_language(OBJCXX)
    target_sources(wmr_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/src/core/coreml_sd_scheduler.cpp
        ${CMAKE_SOURCE_DIR}/src/core/coreml_sd_pipeline.mm
        ${CMAKE_SOURCE_DIR}/src/core/coreml_sd_model_fetch.cpp
        ${CMAKE_SOURCE_DIR}/src/core/coreml_cache.cpp
        unit/coreml_sd_scheduler_test.cpp
        unit/coreml_sd_pipeline_smoke_test.cpp
        unit/coreml_sd_compute_units_test.mm
        unit/coreml_sd_model_cache_test.cpp
        unit/coreml_cache_test.cpp
    )
    target_link_libraries(wmr_tests PRIVATE ${ACCELERATE_LIB} ${COREML_LIB} ${FOUNDATION_LIB})
    # regenerator.cpp (compiled by the WMR_BUILD_REGEN mirror block above)
    # references APP_VERSION inside its CoreML branch. Define it here so the
    # combination WMR_BUILD_REGEN=ON + WMR_BUILD_AI_COREML_SD=ON compiles. The
    # WMR_UPDATE_CHECK mirror block also defines APP_VERSION="0.0.0"; identical
    # redefinition is a no-op for the compiler.
    target_compile_definitions(wmr_tests PRIVATE WMR_BUILD_AI_COREML_SD=1 APP_VERSION="0.0.0")
endif()

# Update check (notify-only): mirrors the main CMakeLists block.
# NOTE: src/core/paths.cpp is intentionally NOT listed here; Task 1 added it to the
# unconditional LIB_SOURCES list above (regenerator.cpp, compiled by the regen mirror
# block, also needs it). Listing it here would be a duplicate-source error when
# WMR_BUILD_REGEN is ON too.
if(WMR_UPDATE_CHECK)
    target_sources(wmr_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/src/core/update_check.cpp
        ${CMAKE_SOURCE_DIR}/tests/unit/update_check_test.cpp
        ${CMAKE_SOURCE_DIR}/tests/unit/update_check_smoke_test.cpp)
    target_link_libraries(wmr_tests PRIVATE CURL::libcurl)
    target_compile_definitions(wmr_tests PRIVATE WMR_UPDATE_CHECK=1 APP_VERSION="0.0.0")
endif()
