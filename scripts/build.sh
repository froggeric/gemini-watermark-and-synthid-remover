#!/usr/bin/env bash
#
# Robust macOS/Homebrew build helper for wmr.
#
# A plain cached CMake build breaks after `brew upgrade`: CMake caches absolute
# versioned Cellar paths (e.g. /opt/homebrew/Cellar/ffmpeg/8.1.1/...) that get
# deleted when Homebrew updates the formula, so the next reconfigure fails.
# This script:
#   1. verifies the required Homebrew formulas are installed,
#   2. detects a cache left stale by an upgrade and reconfigures cleanly,
#   3. configures against the STABLE Homebrew opt symlinks (brew --prefix),
#   4. builds and (optionably) runs the test suite.
#
# Usage:
#   scripts/build.sh                 # Release build + tests
#   BUILD_TYPE=Debug scripts/build.sh
#   RUN_TESTS=0 scripts/build.sh     # skip tests
#   BUILD_DIR=build-alt scripts/build.sh
#   WMR_AI_DENOISE=1 scripts/build.sh  # FDnCNN AI denoise (NCNN/Vulkan)
#   WMR_AI_MIGAN=1 scripts/build.sh    # MI-GAN NotebookLM inpainter (ONNX Runtime)
#   WMR_BUILD_REGEN=1 scripts/build.sh # SynthID diffusion-regen (stable-diffusion.cpp)
set -euo pipefail

BUILD_TYPE="${BUILD_TYPE:-Release}"
RUN_TESTS="${RUN_TESTS:-1}"
BUILD_DIR="${BUILD_DIR:-build}"
# When WMR_AI_DENOISE=1, build the NCNN/Vulkan FDnCNN denoiser. Adds the
# vulkan/volk Homebrew deps, inits the NCNN submodule, and enables
# -DWMR_BUILD_AI_DENOISE=ON. Unset/0 keeps the lean default build unchanged.
AI_DENOISE="${WMR_AI_DENOISE:-0}"
# When WMR_AI_MIGAN=1, build the MI-GAN NotebookLM inpainter (ONNX Runtime).
# No extra deps — ONNX Runtime is fetched as an official prebuilt at configure
# time. Unset/0 keeps MI-GAN out of the build (NS-only NotebookLM path).
AI_MIGAN="${WMR_AI_MIGAN:-0}"
# When WMR_BUILD_REGEN=1, build SynthID diffusion-regen (leejet/stable-diffusion.cpp
# SDXL img2img). Adds Homebrew openssl@3 to the prefix path so find_package(OpenSSL)
# resolves the keg (libcurl comes from the system on macOS). Inits the
# stable-diffusion.cpp submodule (+ its ggml submodule). Unset/0 keeps the lean
# build stable-diffusion/ggml/curl-free.
REGEN="${WMR_BUILD_REGEN:-0}"
# When WMR_BUILD_AI_COREML_SD=1, build CoreML SDXL img2img pipeline (macOS only).
# Requires Accelerate framework (system). Unset/0 keeps the lean build without CoreML.
COREML_SD="${WMR_BUILD_AI_COREML_SD:-0}"

DEPS=(opencv ffmpeg catch2 fmt spdlog cli11)

# AI-denoise mode: pull in the Vulkan toolchain + init the NCNN submodule.
if [ "${AI_DENOISE}" = "1" ]; then
  DEPS+=(vulkan-volk vulkan-loader vulkan-headers molten-vk)
  echo ">> AI denoise mode: initialising NCNN submodule"
  git submodule update --init --recursive
fi

# Regen mode: stable-diffusion.cpp links OpenSSL (keg-only, needs the explicit
# root so find_package(OpenSSL) resolves it) and libcurl. macOS ships libcurl
# system-wide (/usr/lib/libcurl.dylib), so find_package(CURL) resolves it
# without a Homebrew prefix; the Homebrew `curl` formula is keg-only and NOT
# required. The submodule (+ its ggml submodule) must be present.
if [ "${REGEN}" = "1" ]; then
  DEPS+=(openssl@3)
  echo ">> Regen mode: initialising stable-diffusion.cpp submodule"
  git submodule update --init --recursive external/stable-diffusion.cpp
fi

# 1. Verify Homebrew and required formulas.
if ! command -v brew >/dev/null 2>&1; then
  echo "error: Homebrew is required (https://brew.sh)" >&2
  exit 1
fi
# `brew list` can exit non-zero (warnings) even when formulas are present, so
# capture its output and match with grep instead of relying on the pipe status
# (which `pipefail` would turn into a false "missing").
brew_formulas="$(brew list --formula 2>/dev/null || true)"
missing=()
for f in "${DEPS[@]}"; do
  if ! grep -qx "$f" <<<"$brew_formulas"; then
    missing+=("$f")
  fi
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "error: missing Homebrew formulas: ${missing[*]}" >&2
  echo "       fix: brew install ${missing[*]}" >&2
  exit 1
fi

# 2. Detect a stale cache: any cached Cellar path that no longer exists means a
#    formula was upgraded since the last configure — wipe and start fresh.
if [ -f "${BUILD_DIR}/CMakeCache.txt" ]; then
  stale=0
  while IFS= read -r p; do
    if [ ! -e "$p" ]; then stale=1; break; fi
  done < <(grep -oE '(/opt/homebrew|/usr/local)/Cellar/[A-Za-z0-9._/-]+' \
           "${BUILD_DIR}/CMakeCache.txt" | sort -u || true)
  if [ "$stale" -eq 1 ]; then
    echo ">> Stale build cache (Homebrew upgraded a dependency) — reconfiguring ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
  fi
fi

# 3. Configure against stable Homebrew opt prefixes.
PREFIXES=""
for f in "${DEPS[@]}"; do
  p="$(brew --prefix "$f")"
  PREFIXES="${PREFIXES:+${PREFIXES};}${p}"
done

cmake -S . -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_PREFIX_PATH="${PREFIXES}" \
  -DOpenCV_DIR="$(brew --prefix opencv)/lib/cmake/opencv4" \
  -DFFMPEG_ROOT="$(brew --prefix ffmpeg)" \
  -DWMR_BUILD_TESTS=ON \
  $([ "${AI_DENOISE}" = "1" ] && echo "-DWMR_BUILD_AI_DENOISE=ON") \
  $([ "${AI_MIGAN}" = "1" ] && echo "-DWMR_BUILD_AI_MIGAN=ON") \
  $([ "${REGEN}" = "1" ] && echo "-DWMR_BUILD_REGEN=ON -DOPENSSL_ROOT_DIR=$(brew --prefix openssl@3)") \
  $([ "${COREML_SD}" = "1" ] && echo "-DWMR_BUILD_AI_COREML_SD=ON -DCMAKE_OBJCXX_COMPILE_OBJECT='\''<CMAKE_OBJCXX_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>'\''")

# 4. Build.
cmake --build "${BUILD_DIR}" --parallel

# 5. Tests.
if [ "${RUN_TESTS}" = "1" ]; then
  ctest --test-dir "${BUILD_DIR}" --output-on-failure
fi

echo ">> Done. Binary: ${BUILD_DIR}/wmr   Tests: ${BUILD_DIR}/tests/wmr_tests"
