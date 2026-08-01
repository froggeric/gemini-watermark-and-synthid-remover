#!/usr/bin/env python3
# SynthID content-fixture analysis probe (Q1 + Q2).
#
# Reads a .wcb codebook built by `wmr build-codebook` from the 10 real-content
# Gemini 3.6 896x1200 fixtures, and measures three planes at each candidate
# SynthID carrier bin and at off-grid controls:
#   - magnitude_bgr        (mean |FFT| across images)
#   - consistency_bgr      (1 - std/max_std, in [0,1])
#   - phase_consistency_bgr (mean resultant length |mean(exp(i*phase))|, in [0,1])
#
# Question Q1: is the published (48,96,88) carrier grid a REAL cross-image-stable
# carrier, or noise / visible-diamond leakage? A REAL carrier bin has HIGH
# phase_consistency (stable phase across images) AND magnitude above the local
# median. The discriminator is cross-image stability, NOT before/after removal.
#
# Scaling note: the fixtures are portrait W=896, H=1200 (cv2 shape (1200,896,*)).
# The published grid is on 512x512. We scale each axis to the image axis:
#   col_img = round(col_512 * W/512) = round(col_512 * 896/512)
#   row_img = round(row_512 * H/512) = round(row_512 * 1200/512)
# (The task prompt's formula "row*896/512, col*1200/512" has the axes swapped
# relative to the actual portrait geometry; this probe uses the geometry-correct
# mapping, which also matches docs/research/ws2b_leakage_probe.py.)
#
# Usage:
#   python3 synthid_content_probe.py <codebook.wcb> [label]

import struct
import sys
import numpy as np

# Published 512x512 SynthID reverse-engineering grid (col_512, row_512).
# (±48,0),(±96,0),(0,±88),(±48,±88),(±96,±88).
GRID_512 = [(48, 0), (96, 0), (0, 88), (48, 88), (96, 88),
            (-48, 0), (-96, 0), (0, -88), (-48, -88), (-96, -88)]

# Off-grid control bins (col_512, row_512) at comparable radii, chosen with no
# arithmetic coincidence with 48/96/88.
CONTROLS_512 = [(60, 100), (30, 50), (110, 40), (70, 130),
                (20, 88), (48, 50), (130, 150), (10, 200)]

CHAN = ["B", "G", "R"]


def load_codebook(path):
    with open(path, "rb") as f:
        magic = f.read(7)
        if magic not in (b"WMRCB02", b"WMRCB01"):
            raise ValueError(f"bad magic: {magic!r}")
        is_v2 = (magic == b"WMRCB02")
        (count,) = struct.unpack("<I", f.read(4))
        profiles = {}
        for _ in range(count):
            w, h, sc = struct.unpack("<IIi", f.read(12))
            prof = {"width": w, "height": h, "sample_count": sc,
                    "mag": [], "phase": [], "cons": [], "pcons": []}
            for ch in range(3):
                (rows, cols) = struct.unpack("<II", f.read(8))
                n = rows * cols
                mag = np.frombuffer(f.read(n * 4), dtype="<f4").reshape(rows, cols).copy()
                phase = np.frombuffer(f.read(n * 4), dtype="<f4").reshape(rows, cols).copy()
                cons = np.frombuffer(f.read(n * 4), dtype="<f4").reshape(rows, cols).copy()
                if is_v2:
                    pcons = np.frombuffer(f.read(n * 4), dtype="<f4").reshape(rows, cols).copy()
                else:
                    pcons = np.ones((rows, cols), dtype="<f4")
                prof["mag"].append(mag)
                prof["phase"].append(phase)
                prof["cons"].append(cons)
                prof["pcons"].append(pcons)
            profiles[(h, w)] = prof
    return profiles


def scale_bin(col_512, row_512, W, H):
    # Map published 512-grid bin to image FFT bin. Negative indices wrap (FFT is
    # periodic); the stored plane index is the raw frequency index in [0, W) x [0, H).
    c = int(round(col_512 * W / 512)) % W
    r = int(round(row_512 * H / 512)) % H
    return c, r


def local_median(plane, x, y, radius=4):
    """Median of the magnitude plane in an NxN window (excluding the center),
    rolled for FFT periodicity. Returns the local baseline."""
    H, W = plane.shape
    vals = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            vals.append(plane[(y + dy) % H, (x + dx) % W])
    return float(np.median(vals))


def fmt(v):
    return f"{v:9.4f}"


def measure_codebook(path, label):
    profiles = load_codebook(path)
    assert profiles, "no profiles in codebook"
    # one profile expected: key (height, width)
    (h, w), prof = next(iter(profiles.items()))
    N = prof["sample_count"]
    print(f"\n############ {label} ############")
    print(f"codebook: {path}")
    print(f"profile: width={w} height={h} samples={N}  "
          f"(random-phase pcons floor ~ 1/sqrt(N) = {1.0/np.sqrt(N):.3f})")
    assert w == 896 and h == 1200, f"unexpected dims w={w} h={h}"

    rows = []
    print(f"\n  {'bin (512 -> img)':>28}  "
          f"{'mag_B':>9} {'mag_G':>9} {'mag_R':>9}  "
          f"{'cons_B':>9} {'cons_G':>9} {'cons_R':>9}  "
          f"{'pcons_B':>9} {'pcons_G':>9} {'pcons_R':>9}  "
          f"{'mag/locmed_B/G/R':>22}")
    print("  " + "-" * 150)

    def do_bin(col_512, row_512, kind):
        x, y = scale_bin(col_512, row_512, w, h)
        mags, conss, pconss, ratios = [], [], [], []
        for ch in range(3):
            mag = float(prof["mag"][ch][y, x])
            cons = float(prof["cons"][ch][y, x])
            pcons = float(prof["pcons"][ch][y, x])
            locmed = local_median(prof["mag"][ch], x, y, radius=4)
            mags.append(mag); conss.append(cons); pconss.append(pcons)
            ratios.append(mag / locmed if locmed > 1e-30 else float("nan"))
        tag = f"({col_512:>4},{row_512:>4})->({x:>4},{y:>4}) {kind}"
        print(f"  {tag:>28}  "
              f"{fmt(mags[0])} {fmt(mags[1])} {fmt(mags[2])}  "
              f"{fmt(conss[0])} {fmt(conss[1])} {fmt(conss[2])}  "
              f"{fmt(pconss[0])} {fmt(pconss[1])} {fmt(pconss[2])}  "
              f"{fmt(ratios[0])} {fmt(ratios[1])} {fmt(ratios[2])}")
        return {"bin512": (col_512, row_512), "img": (x, y), "kind": kind,
                "mag": mags, "cons": conss, "pcons": pconss, "mag_over_locmed": ratios}

    for (cx, cy) in GRID_512:
        rows.append(do_bin(cx, cy, "grid"))
    for (cx, cy) in CONTROLS_512:
        rows.append(do_bin(cx, cy, "ctrl"))

    # Summary: fraction of grid vs ctrl bins crossing thresholds.
    def frac_above(rows_subset, key, thr):
        tot = 0; hit = 0
        for r in rows_subset:
            ok = all(r[key][ch] >= thr for ch in range(3))
            tot += 1; hit += int(ok)
        return hit, tot

    grid = [r for r in rows if r["kind"] == "grid"]
    ctrl = [r for r in rows if r["kind"] == "ctrl"]
    for key, thr in [("pcons", 0.6), ("cons", 0.6)]:
        gh, gt = frac_above(grid, key, thr)
        ch_, ct = frac_above(ctrl, key, thr)
        print(f"  {key} >= {thr}:  grid {gh}/{gt}   ctrl {ch_}/{ct}")

    return {"label": label, "N": N, "w": w, "h": h, "rows": rows}


def per_channel_split(path, label):
    """Q2: per-channel carrier energy split, restricted to the candidate grid
    bins, two aggregation methods."""
    profiles = load_codebook(path)
    (h, w), prof = next(iter(profiles.items()))
    N = prof["sample_count"]
    print(f"\n============ {label} : per-channel split (Q2) ============")
    # Collect the 10 grid bins (image coords). Use the mean magnitude plane.
    bins_img = [scale_bin(c, r, w, h) for (c, r) in GRID_512]

    # Method A: sum |FFT|^2 (power) over the grid bins, per channel.
    power = [0.0, 0.0, 0.0]
    for (x, y) in bins_img:
        for ch in range(3):
            m = float(prof["mag"][ch][y, x])
            power[ch] += m * m
    tot = sum(power)
    pct_pow = [100.0 * p / tot for p in power]

    # Method B: sum |FFT| (magnitude) over grid bins within band r=3..400 from DC.
    # radius measured on the image FFT grid (sqrt(x^2+y^2) with wrap to nearest).
    def radius(x, y):
        dx = min(x, w - x)
        dy = min(y, h - y)
        return (dx * dx + dy * dy) ** 0.5
    magband = [0.0, 0.0, 0.0]
    for (x, y) in bins_img:
        if 3.0 <= radius(x, y) <= 400.0:
            for ch in range(3):
                magband[ch] += float(prof["mag"][ch][y, x])
    totb = sum(magband)
    pct_mag = [100.0 * m / totb for m in magband]

    print(f"  method A (sum |FFT|^2 over 10 grid bins):  "
          f"B={power[0]:.4e} G={power[1]:.4e} R={power[2]:.4e}")
    print(f"     pct: B={pct_pow[0]:.2f}%  G={pct_pow[1]:.2f}%  R={pct_pow[2]:.2f}%")
    print(f"  method B (sum |FFT|, band r=3..400, grid bins only):")
    print(f"     sum: B={magband[0]:.4e} G={magband[1]:.4e} R={magband[2]:.4e}")
    print(f"     pct: B={pct_mag[0]:.2f}%  G={pct_mag[1]:.2f}%  R={pct_mag[2]:.2f}%")
    print(f"  reference weights {{B:0.85, G:1.0, R:0.70}} -> "
          f"pct of sum(0.85+1.0+0.70)=2.55: B=33.3% G=39.2% R=27.5%")
    print(f"  doc ordering G>B>R: {'YES' if pct_mag[1]>pct_mag[0]>pct_mag[2] else 'NO'} "
          f"(method B), {'YES' if pct_pow[1]>pct_pow[0]>pct_pow[2] else 'NO'} (method A)")
    return {"label": label, "pct_pow": pct_pow, "pct_mag": pct_mag}


def main():
    cbs = sys.argv[1:]
    if not cbs:
        cbs = [("/tmp/content10.wcb", "FULL-10"),
               ("/tmp/content5a.wcb", "SUBSET-5a"),
               ("/tmp/content5b.wcb", "SUBSET-5b")]
    if isinstance(cbs[0], tuple):
        items = cbs
    else:
        # pair paths with labels derived from order
        labels = ["FULL-10", "SUBSET-5a", "SUBSET-5b", "CB-3", "CB-4"]
        items = list(zip(cbs, labels[:len(cbs)]))
    results = []
    for path, label in items:
        results.append(measure_codebook(path, label))
    for path, label in items:
        per_channel_split(path, label)


if __name__ == "__main__":
    main()
