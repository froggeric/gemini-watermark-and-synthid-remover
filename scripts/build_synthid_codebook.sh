#!/usr/bin/env bash
#
# Build a clean multi-resolution SynthID codebook from the HF reverse-SynthID
# dataset + local Gemini 3.6 fixtures, using the fixed discriminative builder.
#
# The builder's carrier-selection gate is  normalize(log1p(mean_mag) * pcons)
# where pcons is the cross-capture phase coherence (mean resultant length).
# This drops the broadband saturation of the old 1-std/max_std metric: on
# near-identical pure-black captures the active-bin fraction falls from ~100%
# to the diamond footprint (~1%), and on content captures it falls to ~0%
# (the carrier is not resolvable above the content noise floor).
#
# Inputs (defaults match the task spec; override via env):
#   $BLACK_DIR   - HF gemini_black  (near-uniform, visible diamond; 1024x1024)
#   $RANDOM_DIR  - HF gemini_random (real content, visible diamond; 2816x1536)
#   $WHITE_DIR   - HF gemini_white  (optional; near-uniform white)
#   $FIXTURE_DIR - local Gemini 3.6 content fixtures (896x1200, optional)
# Output:
#   $OUT         - multi-resolution .wcb (default /tmp/clean_multi.wcb)
#
# Usage:
#   ./scripts/build_synthid_codebook.sh
#   BLACK_DIR=/data/gemini_white RANDOM_DIR=/data/gemini_random OUT=/tmp/cb.wcb \
#     ./scripts/build_synthid_codebook.sh
#
# The codebook is NOT shipped as a default yet: on content images the carrier
# is not resolvable above the noise floor (pcons stays at the ~1/sqrt(N) random
# floor), so the codebook is effectively inert there. See
# docs/research/synthid-clean-codebook-eval.md for the evaluation.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BLACK_DIR="${BLACK_DIR:-/tmp/rsid/gemini_black}"
RANDOM_DIR="${RANDOM_DIR:-/tmp/rsid/gemini_random}"
WHITE_DIR="${WHITE_DIR:-}"
FIXTURE_DIR="${FIXTURE_DIR:-$ROOT/reference-images/896x1200-gemini36}"
OUT="${OUT:-/tmp/clean_multi.wcb}"

WMR_BIN="${WMR_BIN:-$ROOT/build/wmr}"
if [[ ! -x "$WMR_BIN" ]]; then
    echo "error: $WMR_BIN not found. Run scripts/build.sh first." >&2
    exit 1
fi

# Stage all sources into one flat scratch dir (the builder groups by resolution
# automatically, so mixed sizes are fine). Symlinks are followed by cv::imread.
SCRATCH="$(mktemp -d /tmp/wmr_cb_build.XXXXXX)"
trap 'rm -rf "$SCRATCH"' EXIT

add_dir() {
    local dir="$1" label="$2"
    if [[ -z "$dir" || ! -d "$dir" ]]; then
        echo "  skip $label (not a directory: '$dir')"
        return
    fi
    local n=0
    while IFS= read -r -d '' f; do
        ln -sf "$f" "$SCRATCH/$(basename "$f")"
        n=$((n + 1))
    done < <(find "$dir" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' \) -print0)
    echo "  staged $n image(s) from $label ($dir)"
}

echo "Staging codebook sources:"
add_dir "$BLACK_DIR"   "gemini_black"
add_dir "$RANDOM_DIR"  "gemini_random"
add_dir "$WHITE_DIR"   "gemini_white (optional)"
add_dir "$FIXTURE_DIR" "local fixtures"

echo "Building codebook -> $OUT"
SPDLOG_LEVEL=info "$WMR_BIN" build-codebook "$SCRATCH" -o "$OUT"

echo "Done. Inspect with: python3 docs/research/synthid_content_probe.py $OUT"
