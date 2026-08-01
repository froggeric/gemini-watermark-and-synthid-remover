#!/usr/bin/env python3
# Rigorous discriminator for Q1: are the (48,96,88) grid bins SPECIAL, or do they
# just sit on a broadband phase-stable baseline (visible-diamond leakage)?
#
# Three tests on the averaged content codebook profile:
#   T1. Global phase_consistency histogram: what fraction of ALL spectrum bins
#       clear the >0.6 / >0.9 bar? If most do, the bar is meaningless here
#       (a fixed-position component dominates phase everywhere).
#   T2. Radial pcons profile: median pcons in radius buckets, with the 5 unique
#       positive grid bins overlaid. Grid bins that sit ON the curve are not
#       special; ones notably ABOVE it are candidate carriers.
#   T3. Radius-matched ring: for each grid bin, 24 off-grid bins at the SAME
#       radius; compare the grid bin's pcons and magnitude to the ring's median
#       and 90th pct. A real carrier must beat its radius-matched peers.
#
# Excludes DC (0,0) and the DC row/col (row=0, col=0) from baseline stats, since
# those carry image-wide DC / column-sum energy that swamps everything.
#
# Usage: python3 synthid_content_ring_probe.py <codebook.wcb> [label]

import struct, sys, math
import numpy as np

GRID_512 = [(48, 0), (96, 0), (0, 88), (48, 88), (96, 88)]  # 5 unique positive
CHAN = ["B", "G", "R"]


def load_codebook(path):
    with open(path, "rb") as f:
        magic = f.read(7)
        is_v2 = (magic == b"WMRCB02")
        assert is_v2, f"expected v2 codebook, got {magic!r}"
        (count,) = struct.unpack("<I", f.read(4))
        prof = None
        for _ in range(count):
            w, h, sc = struct.unpack("<IIi", f.read(12))
            p = {"width": w, "height": h, "sample_count": sc,
                 "mag": [], "phase": [], "cons": [], "pcons": []}
            for ch in range(3):
                rows, cols = struct.unpack("<II", f.read(8))
                n = rows * cols
                mag = np.frombuffer(f.read(n*4), "<f4").reshape(rows, cols).copy()
                phase = np.frombuffer(f.read(n*4), "<f4").reshape(rows, cols).copy()
                cons = np.frombuffer(f.read(n*4), "<f4").reshape(rows, cols).copy()
                pcons = np.frombuffer(f.read(n*4), "<f4").reshape(rows, cols).copy()
                p["mag"].append(mag); p["phase"].append(phase)
                p["cons"].append(cons); p["pcons"].append(pcons)
            prof = p
    return prof


def wrapped_radius(x, y, W, H):
    dx = min(x, W - x)
    dy = min(y, H - y)
    return math.hypot(dx, dy)


def main():
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else path
    p = load_codebook(path)
    W, H, N = p["width"], p["height"], p["sample_count"]
    print(f"\n######## {label}  (W={W} H={H} N={N}; random pcons floor ~{1/math.sqrt(N):.3f}) ########")

    # Unique positive grid bins in image coords.
    grid = []
    for (c512, r512) in GRID_512:
        x = int(round(c512 * W / 512)) % W
        y = int(round(r512 * H / 512)) % H
        grid.append((x, y, f"({c512},{r512})->({x},{y})"))

    # ---- T1: global pcons histogram (exclude DC + DC row/col) ----
    print("\n[T1] Global phase_consistency distribution (excluding DC + row=0 + col=0):")
    for ch in range(3):
        pc = p["pcons"][ch].copy()
        mask = np.ones_like(pc, dtype=bool)
        mask[0, :] = False  # row 0 (horizontal-DC axis)
        mask[:, 0] = False  # col 0 (vertical-DC axis)
        vals = pc[mask]
        n = vals.size
        for thr in (0.6, 0.9, 0.95, 0.99):
            frac = 100.0 * int(np.sum(vals >= thr)) / n
            print(f"   {CHAN[ch]}: pcons>={thr:.2f}: {frac:6.2f}%   "
                  f"(median={np.median(vals):.4f}, mean={np.mean(vals):.4f}, "
                  f"p10={np.percentile(vals,10):.4f}, p90={np.percentile(vals,90):.4f})")

    # ---- T2: radial pcons + magnitude profile, grid bins overlaid ----
    print("\n[T2] Radial profile (median over off-axis bins in radius buckets) "
          "vs grid-bin values:")
    # Build radius map once.
    ys, xs = np.indices((H, W))
    rdx = np.minimum(xs, W - xs)
    rdy = np.minimum(ys, H - ys)
    R = np.hypot(rdx, rdy)
    offaxis = (xs != 0) & (ys != 0)
    buckets = [(0, 50), (50, 100), (100, 150), (150, 200),
               (200, 250), (250, 300), (300, 350), (350, 400), (400, 500)]
    print(f"   {'r-range':>10}  {'med_pcons_G':>11} {'med_mag_G':>11}  | "
          f"{'grid bin':>22}  {'r':>6}  {'pcons_G':>8} {'mag_G':>9}")
    # Precompute per-bin radius for grid bins.
    def grid_vals(x, y, ch):
        return float(p["pcons"][ch][y, x]), float(p["mag"][ch][y, x])
    for (lo, hi) in buckets:
        m = offaxis & (R >= lo) & (R < hi)
        med_pc = float(np.median(p["pcons"][1][m])) if m.any() else float("nan")
        med_mg = float(np.median(p["mag"][1][m])) if m.any() else float("nan")
        line = f"   [{lo:3d},{hi:3d})   {med_pc:11.4f} {med_mg:11.4f}  |"
        # find grid bins in this bucket
        any_g = False
        for (x, y, tag) in grid:
            rg = wrapped_radius(x, y, W, H)
            if lo <= rg < hi:
                gpc, gmg = grid_vals(x, y, 1)
                line += f"  {tag:>22}  {rg:6.1f}  {gpc:8.4f} {gmg:9.4f}"
                any_g = True
        if not any_g:
            line += "  (no grid bin in bucket)"
        print(line)

    # ---- T3: radius-matched ring around each grid bin ----
    print("\n[T3] Radius-matched ring (24 off-grid, off-axis samples at same radius):")
    print(f"   {'grid bin':>22}  {'ch':>3}  {'grid_pcons':>10} "
          f"{'ring_med':>9} {'ring_p90':>9}  |  {'grid_mag':>9} "
          f"{'ring_med':>9} {'ring_p90':>9}  |  {'verdict':>10}")
    for (x, y, tag) in grid:
        rg = wrapped_radius(x, y, W, H)
        # sample 24 angles, skip grid positions and axes
        angs = [2 * math.pi * k / 48 for k in range(48)]  # 48 samples, thin to 24
        ring_pcons = [[], [], []]
        ring_mag = [[], [], []]
        for a in angs:
            rx = int(round(rg * math.cos(a)))
            ry = int(round(rg * math.sin(a)))
            # wrap to [0,W), [0,H) via the FFT's negative-frequency mirror
            rx = rx % W
            ry = ry % H
            # skip DC axes (would skew baseline) and skip the grid bin itself
            if rx == 0 or ry == 0:
                continue
            # skip if this lands on another grid bin
            if any((rx == gx and ry == gy) for (gx, gy, _) in grid):
                continue
            # skip near-duplicates of the candidate (within 3px)
            if abs(rx - x) + abs(ry - y) < 3:
                continue
            for ch in range(3):
                ring_pcons[ch].append(float(p["pcons"][ch][ry, rx]))
                ring_mag[ch].append(float(p["mag"][ch][ry, rx]))
        for ch in range(3):
            rp = ring_pcons[ch]
            rm = ring_mag[ch]
            if not rp:
                continue
            gpc = float(p["pcons"][ch][y, x])
            gmg = float(p["mag"][ch][y, x])
            med_rp = float(np.median(rp)); p90_rp = float(np.percentile(rp, 90))
            med_rm = float(np.median(rm)); p90_rm = float(np.percentile(rm, 90))
            # verdict: grid beats ring p90 on BOTH pcons and magnitude
            special = (gpc > p90_rp) and (gmg > p90_rm)
            verdict = "SPECIAL" if special else "not special"
            print(f"   {tag:>22}  {CHAN[ch]:>3}  {gpc:10.4f} "
                  f"{med_rp:9.4f} {p90_rp:9.4f}  |  {gmg:9.4f} "
                  f"{med_rm:9.4f} {p90_rm:9.4f}  |  {verdict:>10}")


if __name__ == "__main__":
    main()
