# SynthID content-fixture analysis (Q1 carrier reality + Q2 per-channel split)

**Date:** 2026-08-01
**Task:** Resolve two Phase-0-inconclusive questions directly, with raw numbers
and cross-validation, using the 10 real-content Gemini 3.6 fixtures at
`reference-images/896x1200-gemini36/`.
**Probes:** `docs/research/synthid_content_probe.py` (per-bin raw reads +
per-channel split), `docs/research/synthid_content_ring_probe.py` (global /
radial / radius-matched-ring discriminators).
**Prior art:** extends `docs/research/synthid-48-96-leakage-check.md` (WS2b,
verdict `INCONCLUSIVE_LEAKAGE_SUSPECTED`, helpers shipped inert).

---

## 1. Methodology

### 1.1 Fixtures and codebooks

Three codebooks built with `./build/wmr build-codebook <dir> -o <out>` (no
`--carrier-grid`, so the raw measured profiles are untouched):

| codebook | source | N (images) | random pcons floor `1/sqrt(N)` |
|---|---|---|---|
| `content10.wcb` | all 10 content images | 10 | 0.316 |
| `content5a.wcb` | disjoint subset {5ibg2e, 9zmi4w, d306uc, gawws5g, h57euo} | 5 | 0.447 |
| `content5b.wcb` | disjoint subset {lopo0h, ptrgzj, r3hwh1, ryenhd, yp0j31} | 5 | 0.447 |

`5a` and `5b` partition the 10 images, so they are independent samples: a real
signal reproduces on both halves, a fluke does not (per the CLAUDE.md rule:
never score a measurement on the same frames you would act on).

### 1.2 The three codebook planes (from `src/synthid/codebook_builder.cpp`)

For each BGR channel, per FFT bin, after accumulating across the N images:

- `magnitude_bgr[ch]` = mean `|FFT|` (the averaged magnitude).
- `consistency_bgr[ch]` = `1 - std/max_std`, in [0,1], where `std` is the
  per-bin magnitude std across images and `max_std` is the **global** max
  across the whole spectrum for that channel.
- `phase_consistency_bgr[ch]` = mean resultant length
  `|mean_k(exp(i*phase_k))|`, in [0,1]. Independent of any global max; an
  **absolute** coherence measure. For pure random content phase this is
  `~1/sqrt(N)` (the floor column above); a stable phase (real carrier OR any
  fixed-position component) drives it toward 1.

**Headline methodological finding:** `consistency_bgr` is normalized against
the global max std, which sits at DC / the DC column and dwarfs everything.
For every non-DC bin `std/max_std` is tiny, so `consistency` saturates at
`~0.99` across the whole spectrum (see the tables). It is non-discriminative
by construction. `phase_consistency` is the sharp instrument and is used as
the primary signal below.

### 1.3 Bin scaling (portrait geometry correction)

The fixtures are portrait: cv2 shape `(1200, 896, 4)` = `H=1200, W=896`
(verified directly). The task prompt's formula
`row = round(bin_row * 896/512), col = round(bin_col * 1200/512)` has the two
axes swapped relative to the actual geometry (it assumes `H=896, W=1200`).
This probe uses the geometry-correct mapping, which is also the one used by
the prior WS2b doc (`synthid-48-96-leakage-check.md` section 1) and by
`ws2b_leakage_probe.py`:

```
col_img = round(col_512 * W / 512) = round(col_512 * 896/512)   # x, horizontal
row_img = round(row_512 * H / 512) = round(row_512 * 1200/512)  # y, vertical
```

Profile planes are stored `rows x cols = H x W`; a bin `(x, y)` is read at
`plane[y, x]`. Negative published indices wrap via FFT periodicity
(`-48 -> 464 -> img 812`, etc.). By the FFT conjugate symmetry of real
signals, each positive bin and its negative mirror read identically; this was
verified in the raw output (e.g. `(84,0)` and `(812,0)` both read
`22.2453 / 20.5481 / 24.7475` for B/G/R on `content10`), confirming the
indexing.

### 1.4 The 5 unique positive grid bins at 896x1200

Published 512x512 SynthID reverse-engineering grid:
`(±48,0), (±96,0), (0,±88), (±48,±88), (±96,±88)`.

| 512 bin (x, y) | img bin (x, y) | wrapped radius | note |
|---|---|---|---|
| (48, 0)  | (84, 0)   | 84.0  | on row=0 (horizontal-freq axis) |
| (96, 0)  | (168, 0)  | 168.0 | on row=0 |
| (0, 88)  | (0, 206)  | 206.0 | on col=0 (vertical-freq / DC-column axis) |
| (48, 88) | (84, 206) | 222.5 | off-axis |
| (96, 88) | (168, 206)| 265.8 | off-axis |

The 5 negative mirrors `(812,0), (728,0), (0,994), (812,994), (728,994)` read
identically to their positive counterparts and are omitted from the tables.

Off-grid controls (all scaled the same way, none on the 48/96/88 grid):
`(60,100)->(105,234)`, `(30,50)->(52,117)`, `(110,40)->(192,94)`,
`(70,130)->(122,305)`, `(20,88)->(35,206)`, `(48,50)->(84,117)`,
`(130,150)->(228,352)`, `(10,200)->(18,469)`.

---

## 2. Q1 raw per-bin measurements

Format per bin: `mag_B / mag_G / mag_R`, then `cons_B / cons_G / cons_R`, then
`pcons_B / pcons_G / pcons_R`. Read straight from the codebook planes.

### 2.1 FULL-10 codebook

| bin (x, y) | kind | magnitude B/G/R | consistency B/G/R | phase_consistency B/G/R |
|---|---|---|---|---|
| (84, 0)   | grid | 22.25 / 20.55 / 24.75 | 0.9975 / 0.9973 / 0.9967 | 0.9925 / 0.9912 / 0.9904 |
| (168, 0)  | grid | 19.62 / 17.41 / 21.10 | 0.9896 / 0.9903 / 0.9913 | 0.9827 / 0.9834 / 0.9827 |
| (0, 206)  | grid | 200.6 / 197.5 / 204.9 | 0.2099 / 0.2245 / 0.3266 | 0.9596 / 0.9530 / 0.9431 |
| (84, 206) | grid | 7.578 / 7.371 / 7.492 | 0.9988 / 0.9989 / 0.9988 | 0.9978 / 0.9981 / 0.9982 |
| (168, 206)| grid | 1.278 / 1.262 / 2.097 | 0.9990 / 0.9986 / 0.9989 | 0.8295 / 0.6939 / 0.9428 |
| (105, 234) | ctrl | 6.218 / 6.304 / 6.172 | 0.9984 / 0.9986 / 0.9987 | 0.9969 / 0.9953 / 0.9954 |
| (52, 117)  | ctrl | 9.709 / 9.961 / 10.03 | 0.9986 / 0.9987 / 0.9988 | 0.9987 / 0.9979 / 0.9992 |
| (192, 94)  | ctrl | 2.424 / 2.256 / 2.270 | 0.9990 / 0.9993 / 0.9991 | 0.9704 / 0.9611 / 0.9355 |
| (122, 305) | ctrl | 4.072 / 3.959 / 4.062 | 0.9984 / 0.9985 / 0.9987 | 0.9898 / 0.9895 / 0.9918 |
| (35, 206)  | ctrl | 3.518 / 3.619 / 3.755 | 0.9987 / 0.9991 / 0.9993 | 0.9865 / 0.9981 / 0.9948 |
| (84, 117)  | ctrl | 4.775 / 4.612 / 4.779 | 0.9991 / 0.9991 / 0.9993 | 0.9961 / 0.9961 / 0.9951 |
| (228, 352) | ctrl | 1.639 / 1.687 / 1.674 | 0.9986 / 0.9986 / 0.9988 | 0.9864 / 0.9805 / 0.9847 |
| (18, 469)  | ctrl | 0.776 / 0.690 / 0.722 | 0.9991 / 0.9990 / 0.9991 | 0.7167 / 0.5414 / 0.4512 |

Naive pass-count on the task's bar (`consistency > 0.6 AND pcons > 0.6, all 3
ch`): grid 4/5 (only `(0,206)` fails on consistency), controls 7/8. Not
discriminative.

### 2.2 Disjoint 5a codebook (cross-validation)

| bin (x, y) | kind | magnitude B/G/R | consistency B/G/R | phase_consistency B/G/R |
|---|---|---|---|---|
| (84, 0)   | grid | 22.45 / 20.76 / 24.31 | 0.9949 / 0.9880 / 0.9951 | 0.9999 / 0.9996 / 0.9997 |
| (168, 0)  | grid | 20.41 / 18.16 / 21.71 | 0.9486 / 0.9593 / 0.9675 | 0.9935 / 0.9900 / 0.9847 |
| (0, 206)  | grid | 95.16 / 92.02 / 100.3 | 0.9024 / 0.9115 / 0.9445 | 0.9977 / 0.9969 / 0.9966 |
| (84, 206) | grid | 7.353 / 7.178 / 7.403 | 0.9965 / 0.9965 / 0.9975 | 0.9966 / 0.9974 / 0.9968 |
| (168, 206)| grid | 1.159 / 1.095 / 1.832 | 0.9952 / 0.9936 / 0.9970 | 0.7514 / 0.6451 / 0.9099 |
| (105, 234) | ctrl | 6.178 / 6.422 / 6.070 | 0.9950 / 0.9957 / 0.9970 | 0.9995 / 0.9973 / 0.9978 |
| (52, 117)  | ctrl | 9.602 / 9.880 / 10.15 | 0.9969 / 0.9969 / 0.9992 | 0.9988 / 0.9988 / 0.9993 |
| (18, 469)  | ctrl | 0.880 / 0.769 / 0.867 | 0.9960 / 0.9954 / 0.9970 | 0.9740 / 0.9621 / 0.7586 |

(All 8 controls behave the same as on FULL-10; three shown for brevity, full
table in the probe stdout.)

### 2.3 Disjoint 5b codebook (cross-validation)

| bin (x, y) | kind | magnitude B/G/R | consistency B/G/R | phase_consistency B/G/R |
|---|---|---|---|---|
| (84, 0)   | grid | 22.04 / 20.34 / 25.18 | 0.9976 / 0.9989 / 0.9965 | 0.9871 / 0.9845 / 0.9841 |
| (168, 0)  | grid | 18.82 / 16.65 / 20.49 | 0.9949 / 0.9954 / 0.9950 | 0.9722 / 0.9772 / 0.9813 |
| (0, 206)  | grid | 306.1 / 303.0 / 309.6 | 0.2100 / 0.3273 / 0.2286 | 0.9296 / 0.9185 / 0.8998 |
| (84, 206) | grid | 7.803 / 7.564 / 7.582 | 0.9991 / 0.9993 / 0.9987 | 0.9997 / 0.9990 / 0.9998 |
| (168, 206)| grid | 1.398 / 1.428 / 2.363 | 0.9994 / 0.9998 / 0.9992 | 0.9139 / 0.7876 / 0.9773 |
| (105, 234) | ctrl | 6.257 / 6.187 / 6.274 | 0.9986 / 0.9991 / 0.9987 | 0.9957 / 0.9954 / 0.9948 |
| (52, 117)  | ctrl | 9.817 / 10.04 / 9.910 | 0.9986 / 0.9990 / 0.9986 | 0.9986 / 0.9979 / 0.9990 |
| (18, 469)  | ctrl | 0.672 / 0.611 / 0.576 | 0.9995 / 0.9996 / 0.9995 | 0.5008 / 0.5077 / 0.4381 |

The magnitude at `(84,206)` is stable across all three codebooks (`7.37 - 7.80`
on the G channel); the pcons at `(168,206)` and `(0,206)` swings between
subsets (`(0,206)` pcons G is 0.9530 on FULL-10, 0.9969 on 5a, 0.9185 on 5b),
i.e. it is content-driven, not a fixed carrier.

---

## 3. Q1 discriminator tests (is the grid SPECIAL?)

### 3.1 T1 - global phase_consistency distribution

Fraction of ALL spectrum bins (excluding DC and the row=0 / col=0 axes) whose
`phase_consistency` clears each threshold, FULL-10 (G channel; B and R are
within 0.1%):

| threshold | % of spectrum above it |
|---|---|
| >= 0.60 | 68.44% |
| >= 0.90 | 40.36% |
| >= 0.95 | 29.50% |
| >= 0.99 | 11.24% |

median `0.8199`, mean `0.7220`, p10 `0.2947`, p90 `0.9916`. Random-content
floor for N=10 is `~0.316`, which matches the p10 (the genuinely
content-dominated bins sit at the floor). The bulk sits far above the floor.

**Interpretation:** 68% of the spectrum already clears the `>0.6 pcons` bar,
and 11% clears `>0.99`. The bar is not specific to any grid; a broadband
phase-stable component dominates most of the spectrum. The only physically
plausible source is the fixed-position visible diamond (same (x,y), same shape,
same alpha across all 10 images): a fixed-position additive component pins the
FFT phase wherever its magnitude exceeds the content's. This is the
visible-diamond leakage signature, broadband.

### 3.2 T2 - radial pcons / magnitude profile vs grid-bin values (FULL-10, G)

| radius bucket | median pcons (off-axis) | median mag | grid bin in bucket | grid pcons | grid mag |
|---|---|---|---|---|---|
| [50, 100)  | 0.9990 | 13.99 | (84,0)   r=84   | 0.9912 | 20.55 |
| [150, 200) | 0.9866 | 3.53  | (168,0)  r=168  | 0.9834 | 17.41 |
| [200, 250) | 0.9687 | 2.28  | (0,206)  r=206  | 0.9530 | 197.5 (DC-col artifact) |
| [200, 250) | 0.9687 | 2.28  | (84,206) r=222  | 0.9981 | 7.37 |
| [250, 300) | 0.9392 | 1.69  | (168,206) r=266 | 0.6939 | 1.26 |

The grid bins' `pcons` sit on or below the radial median in 4 of 5 cases. The
two grid bins on the row=0 axis (`(84,0)`, `(168,0)`) do carry elevated
magnitude vs the bucket median, but they sit on the horizontal-frequency axis,
which is inherently elevated for natural images (the axis carries the
row-average profile); that is an axis property, not grid-specific. `(0,206)` is
on the col=0 axis (the DC column: `F(0,y) = sum_x image`, 60x the local
median) and is a content / column-sum artifact, not a carrier candidate.

### 3.3 T3 - radius-matched ring (the decisive test)

For each grid bin, 24 off-grid, off-axis samples at the SAME radius. "SPECIAL"
= the grid bin beats the ring's 90th percentile on BOTH pcons AND magnitude
(all 3 channels).

FULL-10 verdict:

| grid bin | B | G | R |
|---|---|---|---|
| (84, 0)    | not special | not special | not special |
| (168, 0)   | not special | not special | not special |
| (0, 206)   | not special | not special | not special |
| (84, 206)  | SPECIAL | SPECIAL | SPECIAL |
| (168, 206) | not special | not special | not special |

`5a` flags only `G` at `(84,206)` (and a couple of axis-adjacent bins that the
FULL-10 ring rejects). `5b` flags `(84,206)` on all three. So `(84,206)` is the
single grid bin with a marginal, mildly-reproducible elevation, and even there
the beat is marginal (`pcons 0.9981` vs ring `p90 0.9969`; `mag 7.37` vs ring
`p90 6.02` on G, i.e. top-~10% of its ring, not an outlier). A real carrier
grid would have all (or most) bins special and reproducing cleanly on both
halves. Only 1/5 qualifies and marginally.

---

## 4. Q1 verdict: NOISE / LEAKAGE (inert default stands)

The published `(48, 96, 88)` grid is NOT identifiable as a real invisible
carrier on these content fixtures. Evidence:

1. The task's own bar (`consistency > 0.6 AND pcons > 0.6 AND magnitude above
   local median`) is cleared by 68% of the entire spectrum, because
   `consistency` is saturated at `~0.99` everywhere (DC-normalized,
   non-discriminative) and `pcons > 0.6` is the broadband baseline. The bar
   does not separate the grid from off-grid controls (4/5 grid bins pass;
   7/8 controls pass).
2. The radius-matched ring test (T3) flags only 1 of 5 unique grid bins as
   marginally special on FULL-10, and the result does not reproduce cleanly
   across the disjoint 5a/5b subsets. 4 of 5 grid bins are indistinguishable
   from same-radius off-grid peers on both pcons and magnitude.
3. The grid bins' `pcons` sits on or below the radial median curve (T2); the
   broadband phase stability decreases monotonically with radius, exactly as
   a fixed-position visible diamond's spectral envelope would.
4. Cross-image stability cannot, on its own, attribute the stability to an
   invisible carrier versus the fixed-position visible diamond: both produce
   stable phase. The discriminating question is whether the stability is
   CONCENTRATED at the grid, and it is not.

This RESOLVES the prior WS2b `INCONCLUSIVE_LEAKAGE_SUSPECTED` toward the
"not-a-resolvable-carrier" branch, with the broadband diamond-plus-content
baseline now directly measured. Combined with the arithmetic coincidence
(`48`, `96` are exactly the visible diamond's pixel sizes) this is consistent
with the published grid being an artifact of measuring the visible mark.

**Recommendation (for the controller, not applied here):** keep the WS2b
default-seeding path inert for 896x1200. Do not flip it on. The only scenario
that would justify revisiting is a fixture that is SynthID-positive AND
visible-mark-negative (or a verified carrier-bin reading from a published
transform), neither of which exists in this set.

Caveat (honest limitation): the broadband diamond dominance on these fixtures
can mask a weak carrier at the same bins. The cross-image-stability test
establishes "no grid-concentrated signal above the diamond baseline," not
"carrier definitely absent." It cannot, on content fixtures alone, prove the
negative. It does prove the grid is not the clear, strong carrier the default
flip would require.

---

## 5. Q2 per-channel (B/G/R) split

Two aggregation methods, summed over the 10 candidate grid bins (positive +
negative), reported as % of the B+G+R total.

| codebook | method A: sum `|FFT|^2` (% B / G / R) | method B: sum `|FFT|`, band r=3..400 (% B / G / R) |
|---|---|---|
| FULL-10 | 33.19 / 32.06 / 34.75 | 33.25 / 32.30 / 34.45 |
| 5a      | 32.91 / 30.44 / 36.65 | 33.21 / 31.55 / 35.24 |
| 5b      | 33.29 / 32.58 / 34.13 | 33.27 / 32.60 / 34.12 |

The reference weights `{B:0.85, G:1.0, R:0.70}` are `B=33.3% / G=39.2% /
R=27.5%` of their sum (2.55); the doc's stated ordering is `G > B > R`.

**The split is stable on content (no clipping):** both methods and all three
codebooks agree on the ordering `R > B > G`, with `R` strongest and `G`
weakest, and the percentages are nearly flat (each channel within a few
percent of 1/3). The ordering is the opposite of the documented `G > B > R`
and contradicts the weights, which predict `G` dominant and `R` weakest.

### Q2 verdict

A stable per-channel ordering DOES resolve on content (no clipping): `R > B >
G`, reproducible across FULL-10 and both disjoint 5-subsets, by both the
power and the magnitude aggregation. It does NOT match the current weights
`{B:0.85, G:1.0, R:0.70}` nor the doc's `G > B > R`; it points the other way
(R strongest, G weakest).

Entanglement caveat (do not re-derive weights from this without reading Q1):
Q1 established that these grid bins are not clearly carrier bins; their
energy is dominated by the visible-diamond leakage plus content. The measured
`R > B > G` therefore most plausibly reflects the diamond's BGR spectral
signature (the visible mark is bright, rendered near-neutral with a slight red
tilt) and the shared content envelope, not the invisible SynthID carrier. So
the split resolves cleanly as a measurement, but it is not a clean
measurement OF the carrier. If a confirmed SynthID-positive / visible-mark-
negative fixture ever becomes available, re-derive the weights there; on these
content fixtures the data does not justify changing `{B:0.85, G:1.0, R:0.70}`.

---

## 6. Reproduction

```sh
# Build the three codebooks (no --carrier-grid: raw measured profiles).
mkdir -p /tmp/sub10 /tmp/sub5_a /tmp/sub5_b
# (symlink the 10 fixtures into /tmp/sub10, and the two disjoint 5-sets into
#  /tmp/sub5_a, /tmp/sub5_b)
./build/wmr build-codebook /tmp/sub10  -o /tmp/content10.wcb
./build/wmr build-codebook /tmp/sub5_a -o /tmp/content5a.wcb
./build/wmr build-codebook /tmp/sub5_b -o /tmp/content5b.wcb

# Raw per-bin reads + per-channel split (Q1 + Q2 tables).
python3 docs/research/synthid_content_probe.py \
    /tmp/content10.wcb /tmp/content5a.wcb /tmp/content5b.wcb

# Discriminators (global histogram, radial profile, radius-matched ring).
for cb in /tmp/content10.wcb /tmp/content5a.wcb /tmp/content5b.wcb; do
  python3 docs/research/synthid_content_ring_probe.py "$cb" "$(basename $cb .wcb)"
done
```

The fixture set is `reference-images/896x1200-gemini36/` (10 PNGs, all
`1200x896` portrait, SynthID-bearing, with the visible 48 px diamond at
margin `(96, 96)`).
