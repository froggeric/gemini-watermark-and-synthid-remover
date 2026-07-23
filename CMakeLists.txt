cmake_minimum_required(VERSION 3.21)
project(wmr VERSION 1.14.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Sources
set(SOURCES
    src/main.cpp
    src/core/blend_modes.cpp
    src/core/watermark_engine.cpp
    src/core/inpaint.cpp
    src/core/fft_context.cpp
    src/detection/ncc_detector.cpp
    src/detection/synthid_detector.cpp
    src/synthid/spectral_codebook.cpp
    src/synthid/codebook_subtractor.cpp
    src/synthid/noise_residual_subtractor.cpp
    src/synthid/codebook_builder.cpp
    src/cli/cli_app.cpp
    src/cli/batch_processor.cpp
    src/video/video_reader.cpp
    src/video/video_writer.cpp
    src/video/video_processor.cpp
    src/video/scene_detector.cpp
    src/video/notebooklm_detector.cpp
    src/video/notebooklm_gates.cpp
    src/video/geometry_detector.cpp
    src/detection/still_geometry.cpp
)

# Executable
add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/assets
)

# Compile definitions
target_compile_definitions(${PROJECT_NAME} PRIVATE
    APP_VERSION="${PROJECT_VERSION}"
    APP_NAME="${PROJECT_NAME}"
)

# Find dependencies via vcpkg
find_package(OpenCV REQUIRED)

# opencv_contrib xphoto (FSR inpaint for NotebookLM intricate backgrounds). The
# opencv_xphoto imported target exists only when the OpenCV build includes
# contrib xphoto (Homebrew opencv yes; vcpkg yes once the opencv4 `contrib`
# feature is enabled). Guarded so a build without xphoto compiles cleanly and
# falls back to NS at runtime.
if(TARGET opencv_xphoto)
    target_link_libraries(${PROJECT_NAME} PRIVATE opencv_xphoto)
    target_compile_definitions(${PROJECT_NAME} PRIVATE WMR_HAS_XPHOTO=1)
endif()

find_package(fmt REQUIRED)
find_package(CLI11 REQUIRED)
find_package(spdlog REQUIRED)
find_package(FFTW3f REQUIRED)
find_package(FFMPEG REQUIRED)

# Homebrew's fftw3 ships a variable-only CMake config (FFTW3f_LIBRARIES is the
# bare library name) with no imported target, whereas vcpkg provides FFTW3::fftw3f.
# Synthesize the target from the variables when it is missing so the test
# executable (which links FFTW3::fftw3f) builds under both providers. No-op when
# the provider already supplies the target (e.g. vcpkg).
if(FFTW3f_FOUND AND NOT TARGET FFTW3::fftw3f)
    add_library(FFTW3::fftw3f INTERFACE IMPORTED)
    set_target_properties(FFTW3::fftw3f PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${FFTW3f_INCLUDE_DIRS}"
        INTERFACE_LINK_DIRECTORIES "${FFTW3f_LIBRARY_DIRS}"
        INTERFACE_LINK_LIBRARIES "${FFTW3f_LIBRARIES}")
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE
    opencv_core
    opencv_imgcodecs
    opencv_imgproc
    opencv_photo
    fmt::fmt
    CLI11::CLI11
    spdlog::spdlog
    ${FFTW3f_LIBRARIES}
    FFMPEG::avformat
    FFMPEG::avcodec
    FFMPEG::avutil
    FFMPEG::swscale
    FFMPEG::swresample
)
target_include_directories(${PROJECT_NAME} PRIVATE ${FFTW3f_INCLUDE_DIRS})
target_link_directories(${PROJECT_NAME} PRIVATE ${FFTW3f_LIBRARY_DIRS})

# macOS system frameworks required by FFmpeg
if(APPLE)
    find_library(COREMEDIA_LIB CoreMedia)
    find_library(COREVIDEO_LIB CoreVideo)
    find_library(COREAUDIO_LIB CoreAudio)
    find_library(VIDEOTOOLBOX_LIB VideoToolbox)
    find_library(COREFOUNDATION_LIB CoreFoundation)
    find_library(AUDIOTOOLBOX_LIB AudioToolbox)
    find_library(ACCELERATE_LIB Accelerate)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        ${COREMEDIA_LIB}
        ${COREVIDEO_LIB}
        ${COREAUDIO_LIB}
        ${VIDEOTOOLBOX_LIB}
        ${COREFOUNDATION_LIB}
        ${AUDIOTOOLBOX_LIB}
        ${ACCELERATE_LIB}
    )
endif()

# Windows system libraries required by FFmpeg
if(WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        ws2_32 crypt32 bcrypt secur32 ncrypt mfplat mfuuid strmiids
    )
endif()

# x264 is a transitive dependency of FFmpeg's libx264 codec (static linking)
find_library(X264_LIBRARY NAMES x264 PATHS ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib NO_DEFAULT_PATH)
if(NOT X264_LIBRARY)
    find_library(X264_LIBRARY NAMES x264)
endif()
if(X264_LIBRARY)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${X264_LIBRARY})
endif()

# Platform-specific flags
if(APPLE)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
elseif(UNIX)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
    target_link_options(${PROJECT_NAME} PRIVATE -static-libgcc -static-libstdc++)
elseif(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4 /utf-8)
    target_compile_definitions(${PROJECT_NAME} PRIVATE _CRT_SECURE_NO_WARNINGS)
    if(VCPKG_TARGET_TRIPLET MATCHES "static")
        # Static CRT (/MT) for the project AND subprojects (NCNN + glslang via
        # add_subdirectory), so the whole link matches the x64-windows-static
        # vcpkg deps (which are /MT). The VARIABLE form sets the default for
        # every target created afterward (incl. ncnn), while the target
        # property pins wmr itself. Without this ncnn builds /MD -> LNK2005
        # (msvcprt vs libcpmt) + LNK2019 (__imp_nearbyintf/erff/fesetround,
        # dynamic-ucrt import stubs the static-ucrt link can't resolve).
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
        set_property(TARGET ${PROJECT_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
    endif()
endif()

# Release optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(NOT MSVC)
        target_compile_options(${PROJECT_NAME} PRIVATE -O3)
    else()
        target_compile_options(${PROJECT_NAME} PRIVATE /O2)
    endif()
endif()

# AI denoise (FDnCNN via NCNN/Vulkan) — opt-in. OFF by default keeps the lean
# default build volk/NCNN-free. Runs BEFORE the tests subdirectory so the volk
# imported target exists when tests/CMakeLists.txt's mirror block executes.
option(WMR_BUILD_AI_DENOISE "Build FDnCNN AI denoise (NCNN/Vulkan)" OFF)
if(WMR_BUILD_AI_DENOISE)
    # The NCNN submodule must be present.
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/external/ncnn/ncnn-20260113-src/CMakeLists.txt")
        message(FATAL_ERROR "NCNN submodule missing. Run: git submodule update --init --recursive")
    endif()

    # volk provides the Vulkan loader bindings used by the ncnn_shim.
    find_package(volk CONFIG REQUIRED)

    # Diagnostic-only: surfaces where CMake's FindVulkan resolved the loader.
    find_package(Vulkan QUIET)

    # Configure the in-tree NCNN as a static lib. Vulkan GPU support is ON by
    # default; the macOS x86_64 cross-build sets WMR_NCNN_VULKAN=OFF (CPU-only
    # AI) because on APPLE NCNN's simplevk does STATIC Vulkan linkage and needs
    # a build-time libvulkan — but there is no x86_64 Vulkan dylib on the arm64
    # runner to link against. (Linux/Windows use simplevk's runtime dlopen
    # fallback, so they keep Vulkan with no build-time loader.) Cache knobs are
    # set BEFORE add_subdirectory so NCNN picks them up.
    option(WMR_NCNN_VULKAN "Build NCNN with Vulkan GPU support (OFF = CPU-only AI denoise)" ON)
    set(NCNN_VULKAN ${WMR_NCNN_VULKAN} CACHE BOOL "" FORCE)
    set(NCNN_SHARED_LIB OFF CACHE BOOL "" FORCE)
    set(NCNN_OPENMP OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(NCNN_ENABLE_LTO OFF CACHE BOOL "" FORCE)
    add_subdirectory(${CMAKE_SOURCE_DIR}/external/ncnn/ncnn-20260113-src ncnn)

    target_sources(${PROJECT_NAME} PRIVATE
        src/core/ai_denoise.cpp
        src/core/ai_denoise_model.cpp
    )

    if(TARGET volk::volk)
        set(WMR_VOLK_TARGET volk::volk)
    else()
        set(WMR_VOLK_TARGET volk)
    endif()

    target_link_libraries(${PROJECT_NAME} PRIVATE ncnn ${WMR_VOLK_TARGET})
    target_compile_definitions(${PROJECT_NAME} PRIVATE WMR_AI_DENOISE=1)
endif()

# MI-GAN inpainter (ONNX Runtime, CPU) for NotebookLM intricate backgrounds — the
# DEFAULT intricate-scene method (no user flag; uniform scenes use NS). OFF by
# default keeps the lean dev build ORT-free. Rather than the vcpkg onnxruntime
# port (heavy source build -> threatens the Windows 6 h CI cap), the OFFICIAL
# ONNX Runtime prebuilt for the host platform is fetched into the build tree at
# configure time (file(DOWNLOAD) + SHA256) and exposed as an IMPORTED shared-lib
# target. This keeps ORT out of the CI build path entirely (no build-time hit)
# and pins a known-good version (1.27.1). NOTE: ORT dropped osx-x86_64 prebuilts
# after v1.23.0, so the macOS Intel cross-build sets WMR_BUILD_AI_MIGAN=OFF
# (MI-GAN unavailable there; the complexity gate falls back to NS). The model
# (assets/migan_pipeline_v2.onnx) is resolved by MiganInpainter relative to the
# executable at runtime.
option(WMR_BUILD_AI_MIGAN "Build MI-GAN inpainter for NotebookLM intricate backgrounds (CoreML on macOS, ONNX Runtime on Linux/Windows)" OFF)
if(WMR_BUILD_AI_MIGAN)
  if(APPLE)
    # ---- macOS: native CoreML (replaces ORT). CoreML is a system framework (no
    # vendored lib) and the .mlpackage is arch-neutral, so this enables BOTH
    # arm64 and x86_64 — Intel Macs (previously MI-GAN-OFF) gain MI-GAN. ~28 ms
    # /frame on the ANE (vs ORT-CPU ~225 ms; A/B-verified Δ1.9/255). ----
    enable_language(OBJCXX)
    find_library(COREML_LIB CoreML REQUIRED)
    find_library(FOUNDATION_LIB Foundation REQUIRED)
    target_sources(${PROJECT_NAME} PRIVATE src/core/migan_coreml_inpainter.mm)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${COREML_LIB} ${FOUNDATION_LIB})
    target_compile_definitions(${PROJECT_NAME} PRIVATE WMR_AI_MIGAN=1 WMR_AI_MIGAN_COREML=1)

    # Ship the .mlpackage DIRECTORY next to the binary (copy_directory — the model
    # is a multi-file bundle, not a single file; copy_if_different won't work).
    set(MIGAN_MLPACKAGE "${CMAKE_SOURCE_DIR}/assets/migan_512_places2_fp16.mlpackage")
    if(EXISTS "${MIGAN_MLPACKAGE}")
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${MIGAN_MLPACKAGE}"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>/migan_512_places2_fp16.mlpackage")
    endif()
  else()
    # ---- Linux/Windows: ONNX Runtime CPU (unchanged). ----
    set(ORT_VERSION "1.27.1")
    set(ORT_HASH_linux-x64 "25b1ef1fea1acd210d63f8f24dc870ad6e077795ce1f54876252c6d3803c15af")
    set(ORT_HASH_win-x64   "2e00414a63fdef0914cd5a5ede6c707844878e0c08e1b6693842f0451b2df2a1")

    if(WIN32)
        set(ORT_PLATFORM "win-x64")
        set(ORT_ARCHIVE "onnxruntime-${ORT_PLATFORM}-${ORT_VERSION}.zip")
        set(ORT_LIBFILE "lib/onnxruntime.dll")
        set(ORT_IMPLIB "lib/onnxruntime.lib")
    else()
        set(ORT_PLATFORM "linux-x64")
        set(ORT_ARCHIVE "onnxruntime-${ORT_PLATFORM}-${ORT_VERSION}.tgz")
        set(ORT_LIBFILE "lib/libonnxruntime.so")      # symlink -> .so.1 -> .so.1.27.1
        set(ORT_IMPLIB "")
    endif()

    set(ORT_ROOT "${CMAKE_BINARY_DIR}/onnxruntime-${ORT_PLATFORM}-${ORT_VERSION}")
    set(ORT_STAMP "${ORT_ROOT}/.ort.stamp")
    if(NOT EXISTS "${ORT_STAMP}")
        set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_ARCHIVE}")
        message(STATUS "Fetching ONNX Runtime ${ORT_VERSION} (${ORT_PLATFORM})...")
        file(DOWNLOAD "${ORT_URL}" "${CMAKE_BINARY_DIR}/${ORT_ARCHIVE}"
             STATUS ort_dl_status SHOW_PROGRESS
             EXPECTED_HASH "SHA256=${ORT_HASH_${ORT_PLATFORM}}")
        list(GET ort_dl_status 0 ort_dl_rc)
        if(ort_dl_rc)
            list(GET ort_dl_status 1 ort_dl_msg)
            message(FATAL_ERROR "ONNX Runtime download failed: ${ort_dl_msg}")
        endif()
        file(ARCHIVE_EXTRACT INPUT "${CMAKE_BINARY_DIR}/${ORT_ARCHIVE}" DESTINATION "${ORT_ROOT}.extract")
        file(REMOVE_RECURSE "${ORT_ROOT}")
        file(RENAME "${ORT_ROOT}.extract/onnxruntime-${ORT_PLATFORM}-${ORT_VERSION}" "${ORT_ROOT}")
        file(REMOVE_RECURSE "${ORT_ROOT}.extract")
        file(WRITE "${ORT_STAMP}" "${ORT_VERSION}\n")
    endif()

    add_library(wmr_ort SHARED IMPORTED)
    set_target_properties(wmr_ort PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ORT_ROOT}/include"
        IMPORTED_LOCATION "${ORT_ROOT}/${ORT_LIBFILE}")
    if(ORT_IMPLIB)
        set_property(TARGET wmr_ort APPEND PROPERTY IMPORTED_IMPLIB "${ORT_ROOT}/${ORT_IMPLIB}")
    endif()

    target_sources(${PROJECT_NAME} PRIVATE src/core/migan_inpainter.cpp)
    target_link_libraries(${PROJECT_NAME} PRIVATE wmr_ort)
    # Let a local dev build find the ORT shared lib from the build tree at runtime.
    set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY BUILD_RPATH "${ORT_ROOT}/lib")
    if(WIN32)
        # Windows resolves the DLL from the exe dir; co-locate it post-build so dev
        # runs + the CI smoke (build/wmr.exe) load onnxruntime.dll. (Linux finds it
        # via BUILD_RPATH.)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ORT_ROOT}/lib/onnxruntime.dll" "$<TARGET_FILE_DIR:${PROJECT_NAME}>/onnxruntime.dll")
    endif()
    target_compile_definitions(${PROJECT_NAME} PRIVATE WMR_AI_MIGAN=1)

    # Ship the MI-GAN model next to the binary when it is present in assets/ so a
    # dev build runs without the WMR_MIGAN_MODEL env override.
    if(EXISTS "${CMAKE_SOURCE_DIR}/assets/migan_pipeline_v2.onnx")
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_SOURCE_DIR}/assets/migan_pipeline_v2.onnx"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>/migan_pipeline_v2.onnx")
    endif()
  endif()
endif()


# Tests
option(WMR_BUILD_TESTS "Build test suite" ON)
if(WMR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
