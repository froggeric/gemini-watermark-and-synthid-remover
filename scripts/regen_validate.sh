#!/usr/bin/env bash
#
# Dev-only measurement harness for `wmr synthid --synthid-attack regen`.
#
# *** NOT RUNTIME CODE ***
# This script is a developer-side measurement tool. It shells out to python3
# (cv2 + numpy) to compute per-channel mean validity and a relative spectral
# carrier-band energy drop. python3 is NEVER linked into the wmr binary; this
# harness lives under scripts/ alongside the other dev measurement tools
# (verify_removal.py, visualize_spectral.py, build_synthid_codebook.sh) and is
# not packaged into any release. It does NOT violate the no-runtime-Python
# tenet (that tenet covers what the shipped binary executes at runtime).
#
# What it measures:
#   1. wall time of `wmr synthid <img> --synthid-attack regen`
#   2. per-channel mean (BGR) of input vs output   -> validity (mean-delta)
#   3. FFT magnitude energy in the radius 30-500 band of the grayscale image,
#      input vs output                              -> relative carrier-band drop %
#
# IMPORTANT: the carrier-band drop is a RELATIVE heuristic, not a removal
# proof. There is no public SynthID-Image verifier (see
# docs/research/synthid-investigation-summary.md and the MEMORY note
# "SynthID removal reality"). The only external signal is the Gemini in-app
# check (~10/day per account). Record the numbers here as a reproducibility
# anchor, not as evidence of removal.
#
# Usage:
#   scripts/regen_validate.sh <image>            # default: build/wmr, /tmp/wmr_regen_out.png
#   WMR=./build/wmr OUT=/tmp/o.png scripts/regen_validate.sh <image>
#   scripts/regen_validate.sh test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-black-gemini.png
#
# Requirements (NOT downloaded by this script; `wmr` itself fetches them on
# first regen run to ~/.cache/wmr/):
#   - a wmr binary built with WMR_BUILD_REGEN=ON (Task 1)
#   - SDXL base fp16 checkpoint (~6.5 GB, SHA256-pinned by wmr)
#   - SDXL fp16-fix VAE (~250 MB, SHA256-pinned by wmr)
#   - python3 with cv2 + numpy (developer machine; pip install opencv-python numpy)
#
# Output: human-readable lines, including a final SUMMARY line whose three
# numbers map 1:1 to the columns of the results table in
# docs/research/synthid-regen-validation.md (wall_s, validity mean-delta,
# carrier-band drop %).
set -euo pipefail

img="${1:?image required (e.g. test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-black-gemini.png)}"
wmr="${WMR:-build/wmr}"
out="${OUT:-/tmp/wmr_regen_out.png}"

if [[ ! -x "$wmr" ]]; then
    echo "error: wmr binary not found at '$wmr' (set WMR=... or build with WMR_BUILD_REGEN=ON)" >&2
    exit 1
fi
if [[ ! -f "$img" ]]; then
    echo "error: input image not found: $img" >&2
    exit 1
fi

echo "== timing regen on $img =="
echo "== wmr: $wmr ; out: $out =="
start=$(date +%s)
# Tail the last 20 lines so the per-step progress + final status are visible
# without flooding the terminal (sdcpp logs one line per step per tile).
"$wmr" synthid "$img" --synthid-attack regen -o "$out" 2>&1 | tail -20
end=$(date +%s)
wall=$((end - start))
echo "wall_seconds=$wall"

# --- Validity: per-channel mean (BGR) in vs out ---
python3 - "$img" "$out" <<'PY'
import sys, cv2, numpy as np
a = cv2.imread(sys.argv[1])
b = cv2.imread(sys.argv[2])
if a is None or b is None:
    print("error: could not read one of the images (NaN/black output = a scale/VAE bug)", file=sys.stderr)
    sys.exit(1)
ma = a.reshape(-1, 3).mean(0)
mb = b.reshape(-1, 3).mean(0)
delta = np.abs(ma - mb)
print(f"mean_in  B={ma[0]:.2f} G={ma[1]:.2f} R={ma[2]:.2f}")
print(f"mean_out B={mb[0]:.2f} G={mb[1]:.2f} R={mb[2]:.2f}")
print(f"mean_delta_per_channel B={delta[0]:.2f} G={delta[1]:.2f} R={delta[2]:.2f} (max {delta.max():.2f} /255)")
PY

# --- Carrier-band energy probe (relative heuristic, not a removal proof) ---
# FFT magnitude energy in the radius 30-500 band of the grayscale image.
# On a uniform SynthID-bearing fixture, a real scrub should depress this band;
# on busy content the content energy dominates and the drop is not interpretable
# (use the uniform pure-color fixtures for this metric, see the doc).
python3 - "$img" "$out" <<'PY'
import sys, cv2, numpy as np
def band_e(path):
    g = cv2.imread(path, 0).astype(np.float32)
    F = np.fft.fftshift(np.abs(np.fft.fft2(g)))
    cy, cx = np.array(F.shape) // 2
    Y, X = np.ogrid[:F.shape[0], :F.shape[1]]
    r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    m = (r >= 30) & (r <= 500)
    return float(F[m].sum())
e0 = band_e(sys.argv[1])
e1 = band_e(sys.argv[2])
drop = 100.0 * (1.0 - e1 / e0) if e0 > 0 else float("nan")
print(f"carrier_band_in={e0:.3e} carrier_band_out={e1:.3e} drop={drop:.1f}%")
PY

echo "SUMMARY wall_s=$wall  (see docs/research/synthid-regen-validation.md table)"
