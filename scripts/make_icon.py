#!/usr/bin/env python3
"""Regenerate the wmr icon set from the REAL watermark mask.

Dev-time only (outputs are committed). The icon is the watermark itself:
the calibrated 96px V2 diamond mask (assets/watermark-masks/) rendered in
amber on the teal tile, centered, dissolving from a quarter of the way in
along the down-right diagonal with a shaped S-curve (smooth in, fast
middle, soft landing) until only the top-left quarter and the north/west
points remain (the removal story). True-transparent rounded corners.

Run after changing the mask or the design constants:
    python3 scripts/make_icon.py
"""
import pathlib
import struct
import zlib

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
MASK = ROOT / "assets/watermark-masks/gemini-diamond-96px.png"
OUT = ROOT / "assets/gui"

TEAL = (12, 107, 117)      # #0C6B75 tile
MCX = MCY = 0.0            # mask centroid / radial table (set in main())
MEXT = MANG = None
MR_MAX = 1.0               # the table's max extent (feather scale)
AMBER = (255, 193, 69)     # #FFC145 the mark
TILE_RX = 116              # rounded-corner radius, 512-viewBox units
MARK_SPAN = 0.58           # the mark's width as a fraction of the canvas
                          # (clear teal margin around the tips: they must not
                          # stretch toward the tile edges)
# The removal: HALF the diamond remains, via a SMOOTH straight-banded wipe.
# The gradient runs along the down-right diagonal with flat iso-lines (no
# per-direction normalization: shaping the fade to the star's own outline
# replicates its spiky contour at every radius and reads as rays/implosion).
# Normalization is one scalar - the mark's radius - so the ramp is solid at
# the midline and reaches zero where the material along the diagonal ends
# (for a diamond, x+y beyond one radius is empty box).
FADE_START = -0.50         # ramp start in radii relative to the midline:
                          # -0.5 = a quarter of the way into the diamond
                          # (the material spans -1..+1 along the diagonal)
FADE_END = 1.00            # ramp end: zero at the shape's boundary
FADE_SHAPE = 0.6           # The dissolve curve: smoothstep(t)^SHAPE. The
                          # plain root (t^0.5) has a vertical tangent at the
                          # onset - a hard crease where solid meets fade. The
                          # powered smoothstep keeps zero slope at BOTH ends
                          # (smooth in, smooth out) while pulling the steep
                          # middle earlier than a plain S-curve, preserving
                          # the fast-rise-then-ease character the eye liked.
# The real mask's alpha is faint at the tips (~15% of plateau) and full at
# the body: as a direct opacity it renders the whole icon as a ~40% wash.
# For the icon, normalize into a NEAR-SOLID silhouette that keeps the tips'
# taper: below 15% of plateau -> gone, above 55% -> full amber, between ->
# the smooth ramp (edge AA rides the same ramp).
SOLID_LO, SOLID_HI = 0.15, 0.55


def mask_alpha():
    m = cv2.imread(str(MASK), cv2.IMREAD_UNCHANGED)
    a = m if m.ndim == 2 else (m[:, :, 3] if m.shape[2] == 4 else m[:, :, 0])
    a = a.astype(np.float64)
    ref = np.percentile(a, 98)          # the plateau, robust to stray peaks
    norm = np.clip(a / ref, 0.0, 1.0)
    return np.clip((norm - SOLID_LO) / (SOLID_HI - SOLID_LO), 0.0, 1.0)


def shape_radial_table(mask, bins=720):
    """Per-direction extent of the mark's own silhouette (in mask pixels),
    from its alpha-weighted centroid. Normalizing each pixel's radius by
    this makes 'fraction of the shape' direction-correct: the concave
    diagonals of the diamond get short extents, the points long ones."""
    solid = mask > 0.5
    ys, xs = np.where(solid)
    w = mask[solid]
    cx = (xs * w).sum() / w.sum()
    cy = (ys * w).sum() / w.sum()
    ang = np.arctan2(ys - cy, xs - cx)
    r = np.hypot(xs - cx, ys - cy)
    extent = np.zeros(bins)
    edges = np.linspace(-np.pi, np.pi, bins + 1)
    idx = np.clip(np.digitize(ang, edges) - 1, 0, bins - 1)
    np.maximum.at(extent, idx, r)
    # fill any empty bins (thin mask coverage) with the nearest non-zero
    nz = np.where(extent > 0)[0]
    if len(nz):
        pos = np.arange(bins)
        extent = np.interp(pos, nz, extent[nz])
    return cx, cy, extent, np.linspace(-np.pi + np.pi / bins, np.pi, bins)


def render(size, mask):
    """Supersampled composite: teal rounded tile + centered amber mark whose
    bottom-right outer portion dissolves toward the corner."""
    ss = 3                                # supersampling factor
    n = size * ss
    ys, xs = np.mgrid[0:n, 0:n]
    x = (xs + 0.5) / n * 512.0
    y = (ys + 0.5) / n * 512.0

    # Tile: rounded square [0,512]^2 with corner radius TILE_RX.
    r = TILE_RX
    cx = np.clip(x, r, 512 - r)
    cy = np.clip(y, r, 512 - r)
    tile = ((x - cx) ** 2 + (y - cy) ** 2) <= r * r

    # Mark: sample the mask (bilinear) over its centered MARK_SPAN box.
    # Outside the box the mark simply does not exist - the clipped bilinear
    # coords must be zeroed, else they replicate the bright tip row/column
    # across the margins as a spurious smear.
    mh, mw = mask.shape
    bw = 512.0 * MARK_SPAN
    x0 = (512 - bw) / 2
    in_box = ((x >= x0) & (x <= x0 + bw) & (y >= x0) & (y <= x0 + bw))
    u = np.clip((x - x0) / bw * (mw - 1), 0, mw - 1)
    v = np.clip((y - x0) / bw * (mh - 1), 0, mh - 1)
    ui, vi = u.astype(int), v.astype(int)
    u2 = np.minimum(ui + 1, mw - 1)
    v2 = np.minimum(vi + 1, mh - 1)
    fu, fv = u - ui, v - vi
    a = (mask[vi, ui] * (1 - fu) * (1 - fv) + mask[vi, u2] * fu * (1 - fv)
         + mask[v2, ui] * (1 - fu) * fv + mask[v2, u2] * fu * fv)
    a = np.where(in_box, a, 0.0)

    # The removal: one smooth diagonal wipe, starting a quarter of the way
    # into the mark (FADE_START) and shaped by smoothstep^FADE_SHAPE: zero
    # slope at both ends (no crease at the onset, soft landing at the
    # boundary) with the steep middle pulled early - the perceptual
    # "accelerating away" the eye wants (a plain t^2 fade reads as fading
    # too late; a plain root has a brutal onset).
    pxm, pym = u - MCX, v - MCY
    s = (pxm + pym) / MR_MAX
    t = np.clip((s - FADE_START) / (FADE_END - FADE_START), 0.0, 1.0)
    curve = t * t * (3.0 - 2.0 * t)                 # smoothstep: zero slope
    dissolve = curve ** FADE_SHAPE                   # at both ends; shape pulls
    a = a * (1.0 - dissolve)                         # the steep middle earlier

    out = np.zeros((n, n, 4), np.uint8)
    for c, val in enumerate(TEAL):
        out[..., c] = np.where(tile, val, 0)
    out[..., 3] = np.where(tile, 255, 0)
    for c, val in enumerate(AMBER):
        layer = np.where(tile & (a > 0), val, 0)
        out[..., c] = (out[..., c].astype(np.float64) * (1 - a)
                       + layer * a).round().astype(np.uint8)
    out[..., 3] = np.where(tile, (out[..., 3].astype(np.float64) * (1 - a)
                                  + 255 * a).round().astype(np.uint8), 0)

    # Downsample (box filter) to the final size.
    img = out.reshape(size, ss, size, ss, 4).mean(axis=(1, 3)).round().astype(np.uint8)
    return img


def write_png(path, img):
    h, w = img.shape[:2]
    raw = b"".join(b"\x00" + row.tobytes() for row in img)

    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    path.write_bytes(data)
    return len(data)


def main():
    global MCX, MCY, MEXT, MANG
    mask = mask_alpha()
    MCX, MCY, MEXT, MANG = shape_radial_table(mask)
    globals()['MR_MAX'] = float(MEXT.max())
    for name, size in [("icon-512.png", 512), ("favicon-96.png", 96),
                       ("favicon-32.png", 32), ("apple-touch-icon.png", 180)]:
        n = write_png(OUT / name, render(size, mask))
        print(f"wrote {OUT/name} ({n} bytes, {size}x{size})")


if __name__ == "__main__":
    main()
