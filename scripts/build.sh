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
#   WMR_AI_DENOISE=0 scripts/build.sh   # skip FDnCNN AI denoise (NCNN/Vulkan)
#   WMR_AI_MIGAN=0 scripts/build.sh     # skip MI-GAN NotebookLM inpainter
#   WMR_BUILD_REGEN=0 scripts/build.sh  # skip SynthID diffusion-regen (sdcpp)
set -euo pipefail

BUILD_TYPE="${BUILD_TYPE:-Release}"
RUN_TESTS="${RUN_TESTS:-1}"
BUILD_DIR="${BUILD_DIR:-build}"
# The build is FULL by default: every AI feature that builds on this platform is ON.
# There is no separate "lean" build. The flags below are platform-aware defaults;
# set the env vars above to 0 to force a feature off.
# FDnCNN AI denoise (NCNN/Vulkan) + MI-GAN inpainter (CoreML on mac, ORT elsewhere)
# are cross-platform, so they are on everywhere.
AI_DENOISE="${WMR_AI_DENOISE:-1}"
AI_MIGAN="${WMR_AI_MIGAN:-1}"
# SynthID diffusion-regen (stable-diffusion.cpp + CoreML SDXL) builds on macOS; it has
# build gaps on linux/windows/mac-Intel (Vulkan/glslc, MSVC, cross-compile), so it is
# off by default there. Override WMR_BUILD_REGEN=1 on those platforms at your own risk.
if [ "$(uname)" = "Darwin" ]; then
    REGEN="${WMR_BUILD_REGEN:-1}"
    COREML_SD="${WMR_BUILD_AI_COREML_SD:-1}"   # CoreML SDXL is mac-only
else
    REGEN="${WMR_BUILD_REGEN:-0}"
    COREML_SD="${WMR_BUILD_AI_COREML_SD:-0}"
fi

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
  $([ "${COREML_SD}" = "1" ] && echo "-DWMR_BUILD_AI_COREML_SD=ON")

# 4. Build.
cmake --build "${BUILD_DIR}" --parallel

# 5. Tests.
if [ "${RUN_TESTS}" = "1" ]; then
  ctest --test-dir "${BUILD_DIR}" --output-on-failure
fi

echo ">> Done. Binary: ${BUILD_DIR}/wmr   Tests: ${BUILD_DIR}/tests/wmr_tests"
