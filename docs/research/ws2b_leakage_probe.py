#!/usr/bin/env python3
# WS2b leakage probe (canonical version; the report at
# docs/research/synthid-48-96-leakage-check.md cites this file).
#
# Question: are the (48,96)-style candidate SynthID carrier bins actually
# spectral LEAKAGE from the visible Gemini diamond, or are they a real
# invisible carrier? Test: if removing the visible mark (exact reverse-alpha
# blend) collapses the energy at those bins, they were visible leakage.
#
# Procedure:
#   1. E_before = sum |FFT(ch)|^2 in a 3x3 window around each candidate bin
#      per BGR channel, on the raw fixture image.
#   2. ./build/wmr remove <fixture> -o <clean.png>  (reverse-alpha blend).
#   3. E_after = same measurement on <clean.png>.
#   4. Verdict: ratio < 0.5 (in all 3 channels) -> the bin collapsed on
#      visible-mark removal. Counts as LEAKAGE. Otherwise persistent.
#
# Bins scale from the published 512x512 grid proportionally to the fixture
# resolution. Two controls are added: the DC bin (must NOT collapse) and an
# off-grid bin at similar radius to the candidates but with no arithmetic
# coincidence with 48/96/88. If the off-grid control ALSO collapses, the
# fixture is non-discriminative (the visible mark dominates every bin);
# see the report for why this matters on the near-uniform test4 fixture.
#
# Usage:
#   python3 ws2b_leakage_probe.py [<raw.png> <clean.png>]
# Defaults: test-images/896x1200-test4-gemini36.png and /tmp/clean_test4.png

import sys
import cv2
import numpy as np

RAW = sys.argv[1] if len(sys.argv) > 1 else "test-images/896x1200-test4-gemini36.png"
CLEAN = sys.argv[2] if len(sys.argv) > 2 else "/tmp/clean_test4.png"

# 512x512 published grid -> fixture resolution (col_x, row_y)
W_512, H_512 = 512, 512
GRID_512 = [(48, 0), (96, 0), (0, 88), (48, 88), (96, 88)]


def fft_mag_sq(img_bgr):
    out = []
    for ch in range(3):
        f = img_bgr[:, :, ch].astype(np.float32) / 255.0
        # numpy.fft.fft2 layout matches FFTW: DC at [0,0], fx = x (or x-W),
        # fy = y (or y-H). Shape is (H, W).
        F = np.fft.fft2(f)
        out.append(np.abs(F) ** 2)
    return out


def window_energy(mag2, bx, by, r=1):
    H, W = mag2.shape
    idxs_y = [(by + dy) % H for dy in range(-r, r + 1)]
    idxs_x = [(bx + dx) % W for dx in range(-r, r + 1)]
    return float(mag2[np.ix_(idxs_y, idxs_x)].sum())


def measure(mag2_per_ch, bins, label):
    print(f"\n=== {label} ===")
    e = {}
    for (bx, by) in bins:
        per_ch = [window_energy(mag2_per_ch[ch], bx, by) for ch in range(3)]
        e[(bx, by)] = per_ch
        print(f"  bin ({bx:4d},{by:4d}): "
              f"B={per_ch[0]:.4e}  G={per_ch[1]:.4e}  R={per_ch[2]:.4e}")
    return e


def main():
    raw = cv2.imread(RAW, cv2.IMREAD_COLOR)
    clean = cv2.imread(CLEAN, cv2.IMREAD_COLOR)
    if raw is None:
        print(f"MISSING: {RAW}", file=sys.stderr); sys.exit(2)
    if clean is None:
        print(f"MISSING: {CLEAN} (run ./build/wmr remove first)", file=sys.stderr); sys.exit(2)

    H, W = raw.shape[:2]
    print(f"Raw image: {W}x{H}")
    print(f"Clean image: {clean.shape[1]}x{clean.shape[0]}")
    assert clean.shape[:2] == raw.shape[:2], "raw/clean shape mismatch"

    bins = []
    for (x512, y512) in GRID_512:
        x = int(round(x512 * W / W_512))
        y = int(round(y512 * H / H_512))
        bins.append((x, y))
    bins.append((0, 0))       # DC control
    bins.append((50, 200))    # off-grid control, radius ~206

    print(f"\nCandidate bins (x=col, y=row): {bins}")
    print(f"  scaled from 512x512 grid: {GRID_512} + 2 controls (DC, off-grid)")

    mag2_raw = fft_mag_sq(raw)
    mag2_clean = fft_mag_sq(clean)

    e_before = measure(mag2_raw, bins, "E_before (raw fixture)")
    e_after = measure(mag2_clean, bins, "E_after  (visible diamond removed)")

    print("\n=== Ratio E_after / E_before (per BGR channel) ===")
    print(f"  {'bin':>15}  {'B':>10}  {'G':>10}  {'R':>10}  {'verdict':>15}")
    print(f"  {'-'*15}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*15}")
    leakage_count = 0
    real_count = 0
    grid_bins = bins[: len(GRID_512)]
    for (bx, by) in bins:
        eb = e_before[(bx, by)]
        ea = e_after[(bx, by)]
        ratios = [ea[ch] / max(eb[ch], 1e-30) for ch in range(3)]
        is_grid = (bx, by) in grid_bins
        if is_grid:
            collapsed = all(r < 0.5 for r in ratios)
            verdict = "LEAKAGE" if collapsed else "real/persistent"
            if collapsed:
                leakage_count += 1
            else:
                real_count += 1
        else:
            verdict = "(control)"
        print(f"  ({bx:4d},{by:4d})     "
              f"{ratios[0]:>10.4f}  {ratios[1]:>10.4f}  {ratios[2]:>10.4f}  {verdict:>15}")

    print()
    print(f"Grid bins collapsed (LEAKAGE): {leakage_count}/{len(grid_bins)}")
    print(f"Grid bins persistent (real):   {real_count}/{len(grid_bins)}")
    print()
    print("NOTE: if the off-grid control (50, 200) ALSO collapsed, the fixture")
    print("is non-discriminative (the visible diamond dominates every bin on a")
    print("near-uniform image). See docs/research/synthid-48-96-leakage-check.md.")


if __name__ == "__main__":
    main()
