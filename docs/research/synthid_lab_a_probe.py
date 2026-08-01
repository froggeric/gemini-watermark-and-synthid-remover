#!/usr/bin/env python3
"""WS3 LAB-`a` channel experiment: measure a-vs-BGR post-suppression residual.

Runs the real wmr binary (codebook-free path) in two modes on every fixture:
  BGR   : ./build/wmr synthid <img> --codebook-free            -o <out>
  LAB-a : ./build/wmr synthid <img> --codebook-free --lab-a    -o <out>

Metric (LOWER = better suppression): carrier-band residual energy
  sum |FFT|^2 over radius r in [3, 400], measured on the OUTPUT image.
Reported per BGR channel (B,G,R), the BGR total, AND on the `a` channel of the
output (convert output -> Lab, take channel index 1). The `a`-channel number is
the direct target of the lab-a path; the BGR total is what an observer sees.

Cross-validation: the 10 896x1200 content fixtures are split into disjoint halves
5a / 5b; a real effect reproduces on both, a fluke does not.

Also samples the 2400x1792 pure-black fixtures (uniform images, std<0.05) where
both paths replace the image with mean+noise.

Usage: python3 docs/research/synthid_lab_a_probe.py
"""

import cv2
import numpy as np
import subprocess
import sys
import glob
import os

WMR = "./build/wmr"
TMP_BGR = "/tmp/wmr_ws3_bgr.png"
TMP_LABA = "/tmp/wmr_ws3_laba.png"

BAND_INNER = 3
BAND_OUTER = 400


def run_wmr(img, lab_a, out_path):
    cmd = [WMR, "synthid", img, "--codebook-free", "-o", out_path]
    if lab_a:
        cmd.append("--lab-a")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("FAIL:", " ".join(cmd), file=sys.stderr)
        print(r.stderr, file=sys.stderr)
        sys.exit(1)


def carrier_band_energy(img_bgr):
    """sum |FFT|^2 over r in [BAND_INNER, BAND_OUTER], per BGR channel + a channel.

    Returns dict with b,g,r,total (on BGR planes) and a_ch (on Lab `a` plane).
    All planes are normalized to [0,1] before FFT (same convention as the C++
    codebook-free path), so magnitudes are comparable across paths.
    """
    h, w = img_bgr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yy = np.where(yy > h / 2.0, yy - h, yy)
    xx = np.where(xx > w / 2.0, xx - w, xx)
    dist = np.sqrt(yy ** 2 + xx ** 2)
    band = ((dist >= BAND_INNER) & (dist <= BAND_OUTER)).astype(np.float32)
    # FFT shift so DC is centered (match the dist grid above)
    def band_energy(plane_u8):
        f = plane_u8.astype(np.float32) / 255.0
        F = np.fft.fft2(f)
        Fs = np.fft.fftshift(F)
        mag2 = np.abs(Fs) ** 2
        return float((mag2 * band).sum())

    b, g, r = cv2.split(img_bgr)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    a_ch = lab[:, :, 1]
    return {
        "b": band_energy(b),
        "g": band_energy(g),
        "r": band_energy(r),
        "total": band_energy(b) + band_energy(g) + band_energy(r),
        "a": band_energy(a_ch),
    }


def measure_fixture(img_path):
    """Return (bgr_residual, laba_residual, orig_residual) dicts."""
    src = cv2.imread(img_path)
    if src is None:
        return None
    orig = carrier_band_energy(src)
    run_wmr(img_path, False, TMP_BGR)
    run_wmr(img_path, True, TMP_LABA)
    bgr = carrier_band_energy(cv2.imread(TMP_BGR))
    laba = carrier_band_energy(cv2.imread(TMP_LABA))
    return orig, bgr, laba


def aggregate(results):
    """Mean total + a-channel energy for orig/bgr/laba across fixtures.

    results entries are (name, orig_dict, bgr_dict, laba_dict).
    """
    if not results:
        return None
    n = len(results)
    return {
        "orig_total": sum(x[1]["total"] for x in results) / n,
        "bgr_total": sum(x[2]["total"] for x in results) / n,
        "laba_total": sum(x[3]["total"] for x in results) / n,
        "bgr_a": sum(x[2]["a"] for x in results) / n,
        "laba_a": sum(x[3]["a"] for x in results) / n,
    }


def print_table(title, results):
    if not results:
        print(f"\n{title}: (no fixtures)\n")
        return
    print(f"\n{title}")
    print(f"{'fixture':<52s} {'orig_total':>12s} {'bgr_total':>12s} "
          f"{'laba_total':>12s} {'bgr_a':>12s} {'laba_a':>12s} "
          f"{'lab/bgr%':>9s}")
    for name, orig, bgr, laba in results:
        ratio = 100.0 * laba["total"] / max(bgr["total"], 1e-12)
        short = os.path.basename(name)[:50]
        print(f"{short:<52s} {orig['total']:12.1f} {bgr['total']:12.1f} "
              f"{laba['total']:12.1f} {bgr['a']:12.4f} {laba['a']:12.4f} "
              f"{ratio:8.1f}%")


def main():
    if not os.path.exists(WMR):
        print(f"wmr binary not found at {WMR}", file=sys.stderr)
        sys.exit(1)

    content_dir = "reference-images/896x1200-gemini36"
    black_dir = "test-images/gemini-3.1-pro/2400x1792/pure-black"

    content = sorted(glob.glob(os.path.join(content_dir, "*.png")))
    black = sorted(glob.glob(os.path.join(black_dir, "*.png")))

    # 5a / 5b disjoint split of the content set (matches prior research docs).
    split_a = content[:5]
    split_b = content[5:]

    print("=" * 110)
    print("WS3 LAB-a vs BGR: carrier-band residual (sum |FFT|^2 in r=3..400), LOWER = better")
    print(f"content fixtures: {len(content)}  pure-black fixtures: {len(black)}")

    # ---- full content set ----
    content_res = []
    for f in content:
        r = measure_fixture(f)
        if r:
            content_res.append((f, *r))
    print_table("CONTENT SET (896x1200-gemini36, all 10)", content_res)

    # ---- split cross-validation ----
    res_a = [(f, *measure_fixture(f)) for f in split_a] if split_a else []
    res_b = [(f, *measure_fixture(f)) for f in split_b] if split_b else []
    print_table("SPLIT 5a", res_a)
    print_table("SPLIT 5b", res_b)

    # ---- aggregate ----
    print("\n" + "=" * 110)
    print("AGGREGATE (mean across fixtures)")
    for label, res in [("content-all", content_res), ("split-5a", res_a), ("split-5b", res_b)]:
        if not res:
            continue
        a = aggregate(res)
        ratio = 100.0 * a["laba_total"] / max(a["bgr_total"], 1e-12)
        bgr_suppr = 100.0 * (1.0 - a["bgr_total"] / max(a["orig_total"], 1e-12))
        laba_suppr = 100.0 * (1.0 - a["laba_total"] / max(a["orig_total"], 1e-12))
        print(f"{label:<14s}: orig_total={a['orig_total']:12.1f}  bgr_total={a['bgr_total']:12.1f}  "
              f"laba_total={a['laba_total']:12.1f}  lab/bgr={ratio:6.1f}%  | "
              f"bgr_a={a['bgr_a']:.1f}  laba_a={a['laba_a']:.1f}  | "
              f"bgr_suppr={bgr_suppr:+.1f}%  laba_suppr={laba_suppr:+.1f}%")

    # ---- pure-black sample (first 10) ----
    black_sample = black[:10]
    black_res = []
    for f in black_sample:
        r = measure_fixture(f)
        if r:
            black_res.append((f, *r))
    print_table("PURE-BLACK SAMPLE (first 10 of 2400x1792/pure-black)", black_res)

    print("\n" + "=" * 110)
    print("DONE")


if __name__ == "__main__":
    main()
