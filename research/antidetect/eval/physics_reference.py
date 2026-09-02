#!/usr/bin/env python3
"""Reference NumPy/OpenCV mirror of wmr "antidetect" Stage A (camera-statistics
restoration), for the M0 A/B calibration.

This file mirrors src/core/antidetect_physics.{hpp,cpp} 1:1 in op order,
formulas, quantization points, kernel bank, and dose ladder, so the measured
tables in docs/research/antidetect-m0-calibration.md can be transcribed into
the C++ constants directly. Differences from C++ (unavoidable / documented):

  * Channel order: C++ works on BGR Mats (OpenCV native); this mirror works on
    RGB arrays. The mosaic phase math is channel-agnostic; the edge-aware path
    uses COLOR_BayerBG2RGB_EA where C++ uses COLOR_BayerBG2BGR_EA (the same
    pattern mapping, RGB-ordered output).
  * RNG: C++ draws from std::mt19937_64 in a fixed order (kernel, CA dx/dy,
    JPEG quality, then the noise stream). This mirror draws the same
    distributions from a numpy PCG64 generator in the same order, so outputs
    are distribution-identical but NOT byte-identical to C++.
  * cv2.BORDER_REFLECT (OpenCV "gfedcb|..." border), matching the C++
    BORDER_REFLECT everywhere.

Pipeline (EXACT order, from antidetect_physics.cpp apply_physics):

    1. bilateral pre-clean     d = odd(5 + 6s) for s > 0.05 else skip;
                               sigma_color = 3d, sigma_space = d/2
    2. sensor noise, u8 domain: per channel, x_u8 -> linear LUT, add
       N(0, sqrt(a*x + b)), back to u8. a = 2.1e-3 s, b = 2e-5 s.
    3. crop to even dims (Bayer needs 2x2 tiling)
    4. Bayer RGGB mosaic (R at even/even, G at even/odd + odd/even,
       B at odd/odd), u8
    5. demosaic with a kernel from the bank
       {malvar5x5, bilinear, edgeaware} (random per image in production)
    6. lateral chromatic aberration: dx,dy ~ N(0, ca_px) clamped to +-2,
       R plane shifted by +t, B by -t (bilinear warp, BORDER_REFLECT)
    7. vignette: *= max(0, 1 - k*r^2), r normalized so the frame corner is 1
    8. one camera JPEG cycle, q uniform in [88, 96] (OpenCV default 4:2:0)

Dose ladder dose_for_strength(s) mirrors the C++ CALIBRATED ladder (commit
c5a551a, from the M0 A/B in docs/research/antidetect-m0-calibration.md):

    bilateral_d = 0 (s<=0.05) else 5 + 6s
    noise_a = 2.1e-3 * t,  noise_b = 2e-5 * t,  t = clamp((s-0.75)/0.25)
    ca_px = 0 (off)           vignette_k = 0 (off)
    jpeg_q ~ U[88, 96]
    production kernel draw ~ U{edgeaware, bilinear}   (malvar excluded)

legacy=True reproduces the pre-calibration literature-anchor ladder
(noise_a = 2.1e-3 s, ca_px = 0.5+0.7s, vignette 0.10+0.25s, kernel draw over
all three) under which the historical stack-* rows of results/per_image.csv
were measured.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Dose ladder (mirrors C++ PhysicsDose / dose_for_strength)


@dataclass
class PhysicsDose:
    bilateral_d: float = 9.0
    noise_a: float = 0.0
    noise_b: float = 0.0
    ca_px: float = 0.8
    vignette_k: float = 0.22
    jpeg_q_lo: int = 88
    jpeg_q_hi: int = 96


def dose_for_strength(strength: float, legacy: bool = False) -> PhysicsDose:
    """The CALIBRATED ladder (mirrors C++ dose_for_strength after commit
    c5a551a, 2026-09-01): bilateral + randomized {edgeaware, bilinear}
    mosaic + JPEG q88-96; sensor noise ramps in only at strength >= 0.75;
    CA and vignette are 0 at every strength (measured dead weight: they
    clear no detector fully while costing 2-14/255 of visible perturbation).

    legacy=True reproduces the PRE-calibration literature-anchor ladder that
    the stack-* rows of results/per_image.csv were measured under (kept so
    the historical A/B grid stays reproducible; see
    docs/research/antidetect-m0-calibration.md).
    """
    s = float(np.clip(strength, 0.0, 1.0))
    d = PhysicsDose()
    if legacy:
        d.bilateral_d = 0.0 if s <= 0.05 else 5.0 + 6.0 * s
        d.noise_a = 2.1e-3 * s
        d.noise_b = 2.0e-5 * s
        d.ca_px = 0.5 + 0.7 * s
        d.vignette_k = 0.10 + 0.25 * s
        d.jpeg_q_lo = 88
        d.jpeg_q_hi = 96
        return d
    active = s > 0.05
    noise_t = float(np.clip((s - 0.75) / 0.25, 0.0, 1.0))
    d.bilateral_d = 5.0 + 6.0 * s if active else 0.0
    d.noise_a = 2.1e-3 * noise_t
    d.noise_b = 2.0e-5 * noise_t
    d.ca_px = 0.0
    d.vignette_k = 0.0
    d.jpeg_q_lo = 88
    d.jpeg_q_hi = 96
    return d


# Full set (per-ingredient experiments + explicit-kernel tests) vs the
# production draw: C++ apply_physics draws uniformly over {edgeaware,
# bilinear} only (malvar measured both costlier ~3x and the one kernel whose
# output corvi still flags 11/12).
KERNEL_BANK = ("malvar5x5", "bilinear", "edgeaware")
PRODUCTION_KERNELS = ("edgeaware", "bilinear")

# ---------------------------------------------------------------------------
# sRGB <-> linear, u8 domain LUTs (mirrors the C++ LUTs)


def _build_luts() -> tuple[np.ndarray, np.ndarray]:
    to_lin = np.empty(256, dtype=np.float64)
    for v in range(256):
        s = v / 255.0
        to_lin[v] = s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4
    # inverse: piecewise, applied to floats then rounded to u8
    return to_lin, None


SRGB_TO_LINEAR_LUT = _build_luts()[0]


def linear_to_srgb_u8(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    s = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)
    return np.rint(np.clip(s, 0.0, 1.0) * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Individual ops (names mirror the C++ functions)


def odd_diameter(d: float) -> int:
    di = int(np.rint(d))
    if di < 5:
        di = 5
    if di % 2 == 0:
        di += 1
    return di


def bilateral_preclean(img_u8: np.ndarray, diameter: float) -> np.ndarray:
    """Mild edge-preserving smooth (u8 in, u8 out). diameter <= 0 = no-op."""
    if diameter <= 0:
        return img_u8
    d = odd_diameter(diameter)
    out = cv2.bilateralFilter(img_u8, d, 3.0 * d, d * 0.5)
    return out


def add_poisson_gaussian_linear(
    img_u8: np.ndarray, a: float, b: float, rng: np.random.Generator
) -> np.ndarray:
    """sigma(x) = sqrt(a*x + b) on the sRGB-LINEAR values, back to u8.

    The Gaussian with signal-dependent variance is the standard large-lambda
    approximation of Poisson photon shot noise (same argument as the C++
    comment). Per-pixel, per-channel independent draws.
    """
    if a <= 0 and b <= 0:
        return img_u8
    lin = SRGB_TO_LINEAR_LUT[img_u8]
    sigma = np.sqrt(a * lin + b)
    noise = rng.standard_normal(img_u8.shape) * sigma
    return linear_to_srgb_u8(lin + noise)


# --- Bayer mosaic / demosaic -------------------------------------------------

# site masks for the RGGB lattice (mirrors C++ SiteMasks)


def site_masks(h: int, w: int):
    yy, xx = np.mgrid[0:h, 0:w]
    m_r = (yy % 2 == 0) & (xx % 2 == 0)
    m_gr = (yy % 2 == 0) & (xx % 2 == 1)
    m_gb = (yy % 2 == 1) & (xx % 2 == 0)
    m_b = (yy % 2 == 1) & (xx % 2 == 1)
    return m_r, m_gr, m_gb, m_b


def bayer_mosaic_rggb(img_u8: np.ndarray) -> np.ndarray:
    """One channel per site, single-plane u8 mosaic (R at even/even)."""
    m_r, m_gr, m_gb, m_b = site_masks(*img_u8.shape[:2])
    r, g, b = img_u8[..., 0], img_u8[..., 1], img_u8[..., 2]
    mosaic = np.zeros(img_u8.shape[:2], dtype=np.uint8)
    mosaic[m_r] = r[m_r]
    mosaic[m_gr] = g[m_gr]
    mosaic[m_gb] = g[m_gb]
    mosaic[m_b] = b[m_b]
    return mosaic


# 3x3 bilinear kernels (phase-aware; which applies depends on site class)
_K_HORIZ = np.array([[0.5], [0.0], [0.5]], dtype=np.float32).T  # 1x3
_K_VERT = np.array([[0.5], [0.0], [0.5]], dtype=np.float32)     # 3x1
_K_DIAG = np.array(
    [[0.25, 0, 0.25], [0, 0, 0], [0.25, 0, 0.25]], dtype=np.float32)
_K_CROSS = np.array(
    [[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]], dtype=np.float32)

# Malvar-He-Cutler 5x5 kernels (ICASSP 2004), applied to the RAW mosaic
_MHC_G_AT_CHROMA = np.array([
    [0, 0, -1, 0, 0],
    [0, 0, 2, 0, 0],
    [-1, 2, 4, 2, -1],
    [0, 0, 2, 0, 0],
    [0, 0, -1, 0, 0],
], dtype=np.float32) / 8.0
_MHC_CHROMA_AT_G_ROW = np.array([
    [0, 0, 0.5, 0, 0],
    [0, -1, 0, -1, 0],
    [-1, 4, 5, 4, -1],
    [0, -1, 0, -1, 0],
    [0, 0, 0.5, 0, 0],
], dtype=np.float32) / 8.0
_MHC_CHROMA_AT_G_COL = _MHC_CHROMA_AT_G_ROW.T.copy()
_MHC_CHROMA_AT_CHROMA = np.array([
    [0, 0, -1.5, 0, 0],
    [0, 2, 0, 2, 0],
    [-1.5, 0, 6, 0, -1.5],
    [0, 2, 0, 2, 0],
    [0, 0, -1.5, 0, 0],
], dtype=np.float32) / 8.0


def _f2d(plane: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return cv2.filter2D(plane, -1, kernel, borderType=cv2.BORDER_REFLECT)


def _assemble_plane(
    mosaic32: np.ndarray, k_row: np.ndarray, k_col: np.ndarray,
    k_diag: np.ndarray, masks, chroma_is_red: bool,
) -> np.ndarray:
    """Identity at own sites, filtered estimate elsewhere (mirrors C++).

    The kernels are applied to the RAW mosaic (all site classes mixed); each
    kernel's taps land on the right site class by construction. For red
    chroma: row-kernel at gr sites (R horizontal), col-kernel at gb sites,
    diag-kernel at b sites. Mirrored for blue.
    """
    m_r, m_gr, m_gb, m_b = masks
    plane = mosaic32.copy()
    f_row = _f2d(mosaic32, k_row)
    f_col = _f2d(mosaic32, k_col)
    f_diag = _f2d(mosaic32, k_diag)
    if chroma_is_red:
        plane[m_gr] = f_row[m_gr]
        plane[m_gb] = f_col[m_gb]
        plane[m_b] = f_diag[m_b]
    else:
        plane[m_gr] = f_col[m_gr]
        plane[m_gb] = f_row[m_gb]
        plane[m_r] = f_diag[m_r]
    return plane


def demosaic(bayer_u8: np.ndarray, kernel: str) -> np.ndarray:
    """Reconstruct RGB (u8) from an RGGB mosaic. Kernels mirror the C++ bank."""
    h, w = bayer_u8.shape
    masks = site_masks(h, w)
    m_r, m_gr, m_gb, m_b = masks

    if kernel == "edgeaware":
        # OpenCV's edge-aware demosaicing; our R-at-(0,0) mosaic is OpenCV's
        # "BG" naming (the code letters describe the second row, cols 2-3).
        # C++ uses BG2BGR_EA; we use the RGB variant of the same mapping.
        rgb = cv2.demosaicing(bayer_u8, cv2.COLOR_BayerBG2RGB_EA)
        return rgb

    mosaic32 = bayer_u8.astype(np.float32)

    if kernel == "malvar5x5":
        r_plane = _assemble_plane(
            mosaic32, _MHC_CHROMA_AT_G_ROW, _MHC_CHROMA_AT_G_COL,
            _MHC_CHROMA_AT_CHROMA, masks, True)
        b_plane = _assemble_plane(
            mosaic32, _MHC_CHROMA_AT_G_COL, _MHC_CHROMA_AT_G_ROW,
            _MHC_CHROMA_AT_CHROMA, masks, False)
        g_plane = mosaic32.copy()
        fg = _f2d(mosaic32, _MHC_G_AT_CHROMA)
        g_plane[m_r] = fg[m_r]
        g_plane[m_b] = fg[m_b]
    elif kernel == "bilinear":
        r_plane = _assemble_plane(
            mosaic32, _K_HORIZ, _K_VERT, _K_DIAG, masks, True)
        b_plane = _assemble_plane(
            mosaic32, _K_HORIZ, _K_VERT, _K_DIAG, masks, False)
        g_plane = mosaic32.copy()
        fg = _f2d(mosaic32, _K_CROSS)
        g_plane[m_r] = fg[m_r]
        g_plane[m_b] = fg[m_b]
    else:
        raise ValueError(f"unknown kernel {kernel}")

    out = np.stack([
        np.clip(np.rint(r_plane), 0, 255),
        np.clip(np.rint(g_plane), 0, 255),
        np.clip(np.rint(b_plane), 0, 255),
    ], axis=-1).astype(np.uint8)
    return out


def lateral_chromatic_aberration(
    img_u8: np.ndarray, dx: float, dy: float
) -> np.ndarray:
    """R and B planes shifted by +-t (bilinear warp, BORDER_REFLECT)."""
    h, w = img_u8.shape[:2]
    m_r = np.float32([[1, 0, dx], [0, 1, dy]])
    m_b = np.float32([[1, 0, -dx], [0, 1, -dy]])
    r = cv2.warpAffine(img_u8[..., 0], m_r, (w, h), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)
    b = cv2.warpAffine(img_u8[..., 2], m_b, (w, h), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)
    out = img_u8.copy()
    out[..., 0] = r
    out[..., 2] = b
    return out


def apply_vignette(img_u8: np.ndarray, k: float) -> np.ndarray:
    """Multiply by max(0, 1 - k*r^2), r = 1 at the frame corner (u8 out)."""
    if k <= 0:
        return img_u8
    h, w = img_u8.shape[:2]
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    rmax = np.sqrt(cx * cx + cy * cy)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rx = (xx - cx) / rmax
    ry = (yy - cy) / rmax
    f = np.maximum(0.0, 1.0 - k * (rx * rx + ry * ry))
    return np.rint(img_u8 * f[..., None]).astype(np.uint8)


def camera_jpeg_cycle(img_u8: np.ndarray, quality: int) -> np.ndarray:
    """One camera-like JPEG round trip (OpenCV default 4:2:0 subsampling)."""
    ok, enc = cv2.imencode(
        ".jpg", cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    assert ok
    dec = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return dec


# ---------------------------------------------------------------------------
# Full Stage A


@dataclass
class StageAConfig:
    """Per-op enables (for per-ingredient A/B). Doses come from
    dose_for_strength(strength, legacy=ladder == "legacy"); explicit
    overrides for single-op runs.

    ladder="legacy" reproduces the PRE-calibration grid that the stack-*
    rows of results/per_image.csv were measured under.
    """

    strength: float = 0.5
    jpeg_cycle: bool = True
    ladder: str = "calibrated"      # "calibrated" | "legacy"
    # per-op enables
    bilateral: bool = True
    noise: bool = True
    mosaic: bool = True
    ca: bool = True
    vignette: bool = True
    # optional explicit dose overrides
    kernel: str | None = None      # None = production draw {edgeaware, bilinear}
    ca_px_override: float | None = None
    jpeg_q_override: int | None = None
    noise_a_override: float | None = None
    noise_b_override: float | None = None


def apply_physics(
    img_u8: np.ndarray, cfg: StageAConfig, seed: int
) -> tuple[np.ndarray, dict]:
    """Apply Stage A to an RGB u8 image. Returns (image, stats).

    Deterministic: seed controls the numpy PCG64 stream; the RNG consumption
    order mirrors C++ (kernel, CA dx/dy, JPEG quality, then the noise stream).
    """
    rng = np.random.default_rng(seed)
    dose = dose_for_strength(cfg.strength, legacy=cfg.ladder == "legacy")

    # draws in C++ order; the production kernel draw is uniform over
    # {edgeaware, bilinear} (PRODUCTION_KERNELS), mirroring the C++
    # uniform_int(0, 1) after c5a551a. Explicit kernels (per-ingredient
    # conditions, tests) may still name any KERNEL_BANK entry.
    bank = PRODUCTION_KERNELS if cfg.kernel is None else (cfg.kernel,)
    kernel = bank[int(rng.integers(0, len(bank)))]
    ca_px = dose.ca_px if cfg.ca_px_override is None else cfg.ca_px_override
    dx = float(np.clip(rng.normal(0.0, ca_px), -2.0, 2.0))
    dy = float(np.clip(rng.normal(0.0, ca_px), -2.0, 2.0))
    jpeg_q = 0
    if cfg.jpeg_cycle:
        jpeg_q = cfg.jpeg_q_override if cfg.jpeg_q_override is not None else int(
            rng.integers(dose.jpeg_q_lo, dose.jpeg_q_hi + 1))

    img = img_u8
    if cfg.bilateral and dose.bilateral_d > 0:
        img = bilateral_preclean(img, dose.bilateral_d)
    if cfg.noise:
        a = dose.noise_a if cfg.noise_a_override is None else cfg.noise_a_override
        b = dose.noise_b if cfg.noise_b_override is None else cfg.noise_b_override
        img = add_poisson_gaussian_linear(img, a, b, rng)

    # Bayer needs even dims; a 1px crop is invisible at these sizes
    h, w = img.shape[:2]
    if h % 2 or w % 2:
        img = img[: h & ~1, : w & ~1]

    if cfg.mosaic:
        img = demosaic(bayer_mosaic_rggb(img), kernel)
    if cfg.ca:
        img = lateral_chromatic_aberration(img, dx, dy)
    if cfg.vignette:
        img = apply_vignette(img, dose.vignette_k)
    if jpeg_q > 0:
        img = camera_jpeg_cycle(img, jpeg_q)

    stats = {
        "kernel": kernel, "ca_dx": dx, "ca_dy": dy, "jpeg_q": jpeg_q,
        "bilateral_d": dose.bilateral_d, "noise_a": dose.noise_a,
        "noise_b": dose.noise_b, "vignette_k": dose.vignette_k,
        "noise_sigma_midtone_255": float(
            np.sqrt(0.25 * dose.noise_a + dose.noise_b) * 255.0),
    }
    return img, stats


def measure_perturbation(img_u8: np.ndarray, out_u8: np.ndarray) -> float:
    """Mean |out - in| in uint8 levels 0..255 (crude perturbation meter)."""
    h = min(img_u8.shape[0], out_u8.shape[0])
    w = min(img_u8.shape[1], out_u8.shape[1])
    return float(np.mean(
        np.abs(out_u8[:h, :w].astype(np.float32)
               - img_u8[:h, :w].astype(np.float32))))


# ---------------------------------------------------------------------------
# Self-checks: phase wiring, kernel sanity, determinism.


def self_check() -> None:
    # Bayer naming probe: R at (even,even) -> OpenCV "BG", interior sites only
    probe = np.full((6, 6), 100, dtype=np.uint8)
    probe[2, 2] = 200  # R site
    probe[3, 3] = 255  # B site
    rgb = cv2.demosaicing(probe, cv2.COLOR_BayerBG2RGB)
    assert rgb[2, 2, 0] == 200, "BayerBG2RGB does not put R at (even,even)"
    assert rgb[3, 3, 2] == 255, "BayerBG2RGB does not put B at (odd,odd)"

    # mosaic round-trips site values exactly
    img = np.zeros((6, 6, 3), dtype=np.uint8)
    img[...] = 60
    img[2, 2] = (200, 90, 40)   # R site keeps R
    img[3, 3] = (10, 90, 255)   # B site keeps B
    img[2, 3] = (10, 170, 40)   # gr keeps G
    img[3, 2] = (10, 130, 40)   # gb keeps G
    mos = bayer_mosaic_rggb(img)
    assert mos[2, 2] == 200 and mos[3, 3] == 255
    assert mos[2, 3] == 170 and mos[3, 2] == 130

    # all kernels reconstruct a smooth image to reasonable error
    yy, xx = np.mgrid[0:64, 0:96].astype(np.float32)
    smooth = np.stack([
        40 + 40 * np.sin(yy / 9.0),
        90 + 40 * np.cos(xx / 13.0),
        60 + 40 * np.sin((xx + yy) / 17.0),
    ], axis=-1).astype(np.uint8)
    for k in KERNEL_BANK:
        rec = demosaic(bayer_mosaic_rggb(smooth), k)
        err = np.abs(rec[3:-3, 3:-3].astype(np.float32)
                     - smooth[3:-3, 3:-3].astype(np.float32))
        assert err.mean() < 12.0, f"kernel {k} reconstruction error {err.mean():.2f}/255"

    # CA + vignette + JPEG smoke
    out = lateral_chromatic_aberration(smooth, 0.7, -0.3)
    assert out.shape == smooth.shape
    out = apply_vignette(smooth, 0.3)
    assert out[0, 0].astype(np.float32).mean() < smooth[0, 0].mean()  # corner darker
    out = camera_jpeg_cycle(smooth, 92)
    assert out.shape == smooth.shape

    # full pipeline determinism
    rng_img = np.random.default_rng(0).integers(0, 256, (32, 48, 3)).astype(np.uint8)
    a = apply_physics(rng_img, StageAConfig(strength=0.5), 42)[0]
    b = apply_physics(rng_img, StageAConfig(strength=0.5), 42)[0]
    assert (a == b).all(), "apply_physics not deterministic for fixed seed"

    # dose ladders: calibrated (post c5a551a) + legacy (historical grid)
    cal = [dose_for_strength(s) for s in (0.0, 0.5, 0.74, 0.75, 0.9, 1.0)]
    assert cal[0].bilateral_d == 0.0 and cal[0].noise_a == 0.0
    assert cal[1].bilateral_d == 8.0 and cal[1].noise_a == 0.0
    assert cal[2].noise_a == 0.0            # gate: no noise below 0.75
    assert cal[3].noise_a == 0.0            # t = (0.75-0.75)/0.25 = 0 exactly
    assert cal[4].noise_a > 0.0             # ramp is live above 0.75
    assert cal[5].noise_a == 2.1e-3         # full dose at s = 1.0
    assert all(d.ca_px == 0.0 and d.vignette_k == 0.0 for d in cal)
    legacy = [dose_for_strength(s, legacy=True) for s in (0.25, 0.5, 1.0)]
    m = [np.sqrt(0.25 * d.noise_a + d.noise_b) * 255 for d in legacy]
    assert m[0] < m[1] < m[2] and legacy[2].vignette_k == 0.35

    # production kernel draw is restricted to {edgeaware, bilinear}
    seen = {apply_physics(rng_img, StageAConfig(strength=0.5), s)[1]["kernel"]
            for s in range(40)}
    assert seen <= set(PRODUCTION_KERNELS) and len(seen) == 2, seen

    print("physics_reference self-check OK "
          f"(legacy midtone sigma/255 @ s=0.25/0.5/1.0: {m[0]:.2f}/{m[1]:.2f}/{m[2]:.2f})")


if __name__ == "__main__":
    self_check()
