# SynthID (48, 96) carrier-bin leakage check (WS2b)

**Date:** 2026-08-01
**Task:** WS2b
**Question:** A reverse-SynthID repo published a carrier-bin grid
`(±48, 0), (±96, 0), (0, ±88), (±48, ±88), (±96, ±88)` at 512x512. Those
numbers suspiciously coincide with the visible Gemini diamond's pixel sizes
(48 px small / 96 px large). Are those bins:

  (a) spectral LEAKAGE from the VISIBLE diamond (i.e. the published grid is
      an artifact of measuring the visible mark, not the invisible carrier), or
  (b) a REAL invisible SynthID carrier that happens to land at those bins?

**Decision rule (binary, from the WS2b spec):** measure `E_before` and
`E_after` (sum `|FFT(channel)|^2` in a 3x3 window around each candidate bin,
per BGR channel) on a fixture, before and after removing the visible diamond
with the exact reverse-alpha-blend. If `E_after / E_before < 0.5` at the
sampled bins, the bins are leakage (helpers ship inert, opt-in only). If
`>= 0.5`, the bins are a real carrier (default-seed wires in).

---

## 1. Canonical fixture: `test-images/896x1200-test4-gemini36.png`

Near-uniform black with a clear 48 px visible diamond at margin (96, 96).
The `v2_diamond_48_still` alpha was captured from this image.

### Bins at 896x1200 (scaled proportionally from the 512x512 grid)

col (x) scale = `W / 512 = 896 / 512 = 1.750`
row (y) scale = `H / 512 = 1200 / 512 = 2.34375`

| 512x512 bin (x, y) | 896x1200 bin (x, y) | radius |
|---|---|---|
| (48, 0)    | (84, 0)     | 84  |
| (96, 0)    | (168, 0)    | 168 |
| (0, 88)    | (0, 206)    | 206 |
| (48, 88)   | (84, 206)   | 222 |
| (96, 88)   | (168, 206)  | 266 |

Two controls are added: the DC bin `(0, 0)` (must NOT collapse) and an
off-grid bin `(50, 200)` at radius similar to the candidates but with no
arithmetic coincidence with 48/96/88.

### Measurement (per BGR channel, 3x3 window)

Visible removal command: `./build/wmr remove test-images/896x1200-test4-gemini36.png -o /tmp/clean_test4.png`

| bin (x, y) | E_before B / G / R | E_after B / G / R | ratio B / G / R | verdict |
|---|---|---|---|---|
| (84, 0)    | 2.42e3 / 2.16e3 / 2.46e3 | 3.90e2 / 4.27e2 / 5.69e2 | **0.161 / 0.198 / 0.232** | LEAKAGE |
| (168, 0)   | 8.33e2 / 6.94e2 / 8.61e2 | 8.73e2 / 7.15e2 / 7.59e2 | 1.048 / 1.030 / 0.881    | persistent |
| (0, 206)   | 5.61e6 / 5.56e6 / 5.54e6 | 5.59e6 / 5.55e6 / 5.53e6 | 0.998 / 0.998 / 0.998    | persistent |
| (84, 206)  | 4.43e2 / 4.57e2 / 4.69e2 | 4.99e0 / 7.28e0 / 3.72e0 | **0.011 / 0.016 / 0.008** | LEAKAGE |
| (168, 206) | 1.18e2 / 1.34e2 / 1.12e2 | 1.30e1 / 2.41e1 / 1.10e1 | **0.110 / 0.180 / 0.098** | LEAKAGE |
| (0, 0) DC control    | 5.23e11 / 5.23e11 / 5.23e11 | 5.23e11 / 5.23e11 / 5.23e11 | 0.9999 / 0.9999 / 0.9999 | (control, unchanged) |
| (50, 200) off-grid   | 2.47e1 / 2.23e1 / 2.92e1 | 9.62e0 / 3.04e0 / 3.79e0 | **0.389 / 0.137 / 0.130** | (control, COLLAPSED) |

### Naive verdict on this fixture alone

3/5 candidate bins collapse (`ratio < 0.5` in all three channels) -> LEAKAGE.
Per the binary decision rule, this would trigger "ship inert."

### Why this fixture is non-discriminative (rigor caveat)

**The off-grid control bin `(50, 200)` ALSO collapses** on this fixture
(ratio 0.13 - 0.39). That collapse is not a leakage signature of the
candidate bins specifically; it is a property of the FIXTURE. The test4
image is near-uniform black except for the visible diamond, so the diamond
is the dominant spectral content at EVERY non-DC, non-axis bin. Removing
the diamond collapses the entire mid-frequency noise floor, not just the
candidate bins. The collapse at `(84, 0)`, `(84, 206)`, `(168, 206)` is
therefore consistent with "the diamond dominates every bin on this clean
fixture," not with "these specific bins are visible leakage while others
are not."

The two persistent candidate bins confirm this read:
- `(168, 0)` is on the x-axis, near a high-order sinc null of the 48 px
  diamond (sinc argument `pi * 168 * 48 / 896 = 9*pi`, near a zero). The
  diamond's contribution there is ~0, so the bin is noise-floor-dominated
  and removing the diamond does not dent it.
- `(0, 206)` is on the y-axis, with an outlier energy of 5.6 M (3-4 orders
  of magnitude above the other candidates), which is a row-correlated
  sensor/encoder artifact, not the diamond. Removing the diamond leaves it
  intact.

So the test4 result reveals the diamond's spectral footprint is broad on a
near-uniform fixture, but it does NOT isolate the candidate bins as
leakage-specific.

---

## 2. Cross-validation: `test-images/896x1200-test3-gemini36.png`

A busy poster background with a faint 48 px diamond. The diamond is faint
(NCC ~0.48, below the detection gate), so removal is forced via
`--geo-preset gemini36-portrait`. With content providing competing spectral
energy, the candidate bins can be checked for a leakage-specific collapse
vs the off-grid control.

Visible removal command: `./build/wmr remove test-images/896x1200-test3-gemini36.png -o /tmp/clean_test3.png --geo-preset gemini36-portrait`

### Measurement (per BGR channel, 3x3 window)

| bin (x, y) | ratio B / G / R | verdict |
|---|---|---|
| (84, 0)    | 1.016 / 1.016 / 1.015 | persistent |
| (168, 0)   | 0.999 / 0.999 / 0.999 | persistent |
| (0, 206)   | 0.999 / 0.999 / 0.999 | persistent |
| (84, 206)  | 0.981 / 0.981 / 0.981 | persistent |
| (168, 206) | 1.006 / 1.005 / 1.004 | persistent |
| (0, 0) DC control    | 0.9999 / 0.9999 / 0.9999 | (control) |
| (50, 200) off-grid   | 0.993 / 0.993 / 0.993 | (control, also unchanged) |

### Verdict on the cross-validation fixture

0/5 candidate bins collapse. All candidate bins and the off-grid control
behave identically (ratio `~ 1.00`). On a busy fixture, removing the faint
visible diamond does NOT change the energy at any of the candidate bins
(content dominates). This is also NOT evidence the candidate bins are a
real carrier; it is evidence that on a content-dominated fixture, the
candidate bins are indistinguishable from any other mid-frequency bin.

---

## 3. Net verdict and decision

The two fixtures give contradictory naive verdicts, and neither is
discriminative:

- **test4 (canonical, near-uniform):** candidate bins collapse, but so does
  the off-grid control. The collapse is a fixture property (the diamond
  dominates every bin on a near-uniform image), not a leakage signature.
- **test3 (busy content):** nothing collapses, candidate bins are
  indistinguishable from controls. There is no positive evidence of a real
  carrier at these bins either.

The measurement therefore CANNOT confirm the published grid is a real
SynthID carrier. Combined with the suspicious arithmetic coincidence
(`48`, `96` are exactly the visible diamond's pixel sizes), the safe call
is to treat the grid as NOT validated and ship the helpers inert.

**VERDICT:** INCONCLUSIVE_LEAKAGE_SUSPECTED. The (48, 96) grid cannot be
confirmed as a real invisible carrier on the available fixtures, and the
arithmetic coincidence with the visible-diamond pixel sizes remains
suspicious.

**DECISION:** Ship the WS2b helpers INERT.

  - `seed_carrier_bins` and `SpectralCodebook::merge_from` ship as utilities.
  - `--carrier-grid "x,y;..."` is OPT-IN on `build-codebook`. No default
    grid is wired in for any resolution.
  - Default behavior (flag absent) is byte-identical to pre-WS2b.

The burden of proof is on confirming the carrier exists. Inert helpers do
no harm: a future task with a confirmed SynthID-positive, visible-mark-
negative fixture (or a verified carrier-bin reading from a published paper
that includes the transform details) can flip the default on for a measured
resolution; that is a one-line change in `CodebookBuilder::build_from_directory`.
Seeding by default today would silently activate the subtractor at those
bins on every codebook, which would over-subtract image content if the
bins are not in fact a carrier.

### Why not call this a clean LEAKAGE verdict

A clean LEAKAGE verdict would require the off-grid control to NOT collapse
while the candidate bins DO collapse (i.e. the candidate bins are
specifically tied to the visible mark while ordinary bins are not). On
test4 the off-grid control also collapses, so the leakage label is not
specifically earned. The conservative treatment (ship inert) is the same
as the leakage branch's action, but the reached-via path is "unverifiable,
not confirmed real" rather than "demonstrated to be visible leakage."

### Reproduction

```sh
# Canonical fixture
./build/wmr remove test-images/896x1200-test4-gemini36.png -o /tmp/clean_test4.png

# Cross-validation (faint mark, busy background)
./build/wmr remove test-images/896x1200-test3-gemini36.png -o /tmp/clean_test3.png \
    --geo-preset gemini36-portrait

# Probe: measures E_before / E_after at the scaled candidate bins + 2 controls.
python3 docs/research/ws2b_leakage_probe.py   # RAW / CLEAN paths are consts at top
```

The probe source used for the measurements above lives at
`docs/research/ws2b_leakage_probe.py` (a copy was used during the investigation;
the canonical version is committed alongside this doc).
