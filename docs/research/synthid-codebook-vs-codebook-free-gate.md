# SynthID codebook vs codebook-free value gate

**Date:** 2026-08-01
**Branch:** `synthid-correctness-pack` (`wmr` v1.15.0, 58/58 tests green)
**Question:** does a dataset-derived codebook BEAT the already-shipped codebook-free
path on SynthID suppression? Should we build and ship a dataset codebook as the
default?
**Verdict:** **NOT WORTH IT.** The codebook loses on the matched solid-color
domain (on the raw metric) and is statistically identical to codebook-free on
content (the realistic domain), where it is architecturally inert by design.
Full numbers and reasoning below.

---

## 1. Summary

| domain | path | carrier-band residual (sum |FFT|^2, r=3..400) | % suppression | residual ratio cb/cf |
|---|---|---|---|---|
| solid-color (matched, n=20) | original | 3.9903e+13 | 0 | |
| | codebook | 7.6243e+12 | +80.89% | 5.90 (cb leaves MORE) |
| | codebook-free | 1.2917e+12 | +96.76% | |
| content (realistic, n=88) | original | 1.9608e+17 | 0 | |
| | codebook | 1.9341e+17 | +1.36% | 0.999 (tie) |
| | codebook-free | 1.9351e+17 | +1.31% | |

Headline: codebook = `cb_main.wcb` built from HF `gemini_black` train split +
local fixtures (3 profiles: 1024x1024, 896x1200, 2400x1792). Default strength
(Moderate). Held-out test set strictly disjoint from all training images.

- On the **matched solid-color** domain the codebook leaves **5.9x more**
  residual than codebook-free. Codebook-free drives the residual to the noise
  floor by replacing the image with mean+noise (its uniform-image branch), which
  the raw residual metric rewards. The codebook is the more surgical path
  (mean |out-orig| = 0.133 vs 0.512 on a 0-255 scale) but does not win on
  suppression.
- On the **content** domain the codebook and codebook-free are **identical
  within noise** (ratio 0.999, suppression 1.36% vs 1.31%). This is by
  construction: on content images (`avg_std > 0.05`) `CodebookSubtractor` sets
  `num_passes = 0` and skips carrier subtraction entirely, so the loaded
  codebook is never used in the per-pass loop. No codebook, however well
  trained, can beat codebook-free on content under the current code.
- Reproducible across two disjoint training splits (cb_A, cb_B) and stable under
  Maximum strength.

---

## 2. Methodology

### 2.1 Data

Downloaded from `huggingface.co/datasets/aoxo/reverse-synthid` via `hf download`:

| subset | N | resolution (WxH) | pixel std / 255 | path class (`avg_std > 0.05`) |
|---|---|---|---|---|
| `gemini_black` | 100 PNG | 1024x1024 | 0.0139 | uniform (solid-color) |
| `gemini_random` | 88 PNG | 2816x1536 | 0.2826 | content |

(101 files in `gemini_black` incl. one stray `black.jpg`; the 100 PNGs are used.)

### 2.2 Training set (codebook source) = HF-black train split UNION local fixtures

Per the scope update, the gate codebook is built from the union so it is
genuinely multi-resolution. Training images (all go into the codebook, never
tested):

| source | count | resolution |
|---|---|---|
| HF `gemini_black` indices [0:80] | 80 | 1024x1024 (solid black) |
| `reference-images/896x1200-gemini36/*.png` | 10 | 896x1200 (content) |
| `test-images/gemini-3.1-pro/2400x1792/pure-black/*.png` | 30 | 2400x1792 (solid black) |
| `test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-*-gemini.png` | 9 | 2400x1792 (pure color) |
| `test-images/2400x1792-test1-gemini.png`, `test2-gemini.png` | 2 | 2400x1792 (content) |
| `test-images/896x1200-test3-gemini36.png` | 1 | 896x1200 (content) |
| **total** | **132** | 3 profiles |

`wmr build-codebook` output (auto-groups by resolution):

```
Profile: 1024x1024 (80 samples)
Profile: 896x1200  (11 samples)
Profile: 2400x1792 (41 samples)
Saved codebook: 3 profiles -> /tmp/cb_main.wcb (0 seeded)
```

### 2.3 Held-out test set (strictly disjoint from ALL training)

Per the "never score on frames you derived from" rule:

| set | source | count | resolution | domain |
|---|---|---|---|---|
| solid | HF `gemini_black` [80:100] | 20 | 1024x1024 | matched (exact profile) |
| content | HF `gemini_random` (all) | 88 | 2816x1536 | realistic (no exact profile; resized) |

The held-out solid set has an exact 1024x1024 profile in the codebook (best case
for the codebook). The content set is held out entirely; it is not in any
training codebook in this gate (the production-build scope note in section 8
covers what changes, and what does not, if `gemini_random` is later added).

### 2.4 Cross-validation codebooks (disjoint HF-black splits)

| codebook | HF-black split | local fixtures | 1024x1024 samples |
|---|---|---|---|
| `cb_main` | [0:80] | all (shared) | 80 |
| `cb_A` | [0:40] | all (shared) | 40 |
| `cb_B` | [40:80] | all (shared) | 40 |

`cb_A` and `cb_B` partition the HF-black train half. The local fixtures are a
constant scaffold shared by all three (the variable is the HF-black half). A
real signal reproduces on both A and B; a fluke does not.

### 2.5 Metric: carrier-band residual

Per image and per BGR channel, `sum |FFT|^2` over the annulus `r = 3..400`,
where `r` is the distance from DC. Reported as B, G, R, and TOTAL = B+G+R.

- Frequency layout matches the C++ subtractor exactly (natural FFT, DC at
  `[0,0]`; index `> N/2` wraps negative). numpy `fft2` (default norm) matches raw
  FFTW forward, so absolute values are comparable within one resolution and %
  comparisons are exact.
- `r = 3..400` is valid for all three resolutions (max radius: 1024 -> 724,
  896x1200 -> 752, 2816x1536 -> 1603).
- **Caveat (load-bearing for the verdict):** on content images the r=3..400 band
  is 99.9% image content, not carrier (carrier is ~0.025/255 and
  content-conditional per CLAUDE.md; <0.1% of spectral energy per the carrier
  characterization doc). So a lower number on content can mean "more content
  destroyed", not "more carrier removed". The metric is clean on solid-color
  (carrier IS the dominant spectral content there). Section 7 explains why the
  codebook is inert on content regardless.

Suppression runs: `wmr synthid <img> -o <out> -f --codebook <cb>` (codebook path)
and `wmr synthid <img> -o <out> -f --codebook-free` (codebook-free path), default
strength unless noted. The default SynthID path is codebook-free.

---

## 3. Headline result (cb_main, default Moderate strength)

### 3.1 Solid-color domain (held-out HF-black, n=20, 1024x1024, uniform)

| | B | G | R | TOTAL |
|---|---|---|---|---|
| original | 1.3288e+13 | 1.3319e+13 | 1.3296e+13 | 3.9903e+13 |
| codebook | 2.4419e+12 | 2.0300e+12 | 3.1524e+12 | 7.6243e+12 |
| codebook-free | 4.2967e+11 | 4.3201e+11 | 4.2998e+11 | 1.2917e+12 |
| **% sup codebook** | -81.6% | -84.8% | -76.3% | **-80.9%** |
| **% sup codebook-free** | -96.8% | -96.8% | -96.8% | **-96.8%** |

- Codebook-free suppresses the band by **96.76%** vs the codebook's **80.89%**.
  Codebook residual is **5.90x higher** than codebook-free.
- Per channel, the codebook subtracts most on G (weight 1.0, highest consistency)
  and least on R (weight 0.70): residual G=2.03e12 < B=2.44e12 < R=3.15e12,
  matching the `{B:0.85, G:1.0, R:0.70}` weights. Codebook-free equalizes all
  three to ~4.3e11 (its uniform branch replaces the whole image with mean+noise,
  so the channel weighting is moot).
- Fidelity (mean |out-orig|, 0-255): codebook **0.133**, codebook-free **0.512**.
  Both are near-imperceptible on solid black, but the codebook is the more
  surgical edit. Output pixel std: codebook 1.623, codebook-free 0.733.

### 3.2 Content domain (held-out HF-random, n=88, 2816x1536, content)

| | B | G | R | TOTAL |
|---|---|---|---|---|
| original | 5.9934e+16 | 5.9797e+16 | 7.6348e+16 | 1.9608e+17 |
| codebook | 5.9106e+16 | 5.9041e+16 | 7.5259e+16 | 1.9341e+17 |
| codebook-free | 5.9129e+16 | 5.9078e+16 | 7.5300e+16 | 1.9351e+17 |
| **% sup codebook** | -1.38% | -1.26% | -1.43% | **-1.36%** |
| **% sup codebook-free** | -1.34% | -1.20% | -1.37% | **-1.31%** |

- Codebook and codebook-free are identical within noise: residual ratio
  **0.999**, suppression 1.36% vs 1.31%.
- The ~1.3% "suppression" on content is NOT carrier removal. Both paths leave the
  FFT magnitude essentially intact on content (see section 7); the small drop is
  the shared final 3x3 GaussianBlur (`codebook_subtractor.cpp:261`,
  `noise_residual_subtractor.cpp:277`) attenuating high frequencies. The carrier
  itself is below the content noise floor and is untouched by either path.
- Fidelity: codebook 3.860, codebook-free 3.803 (both ~1.5% of full scale).

---

## 4. Cross-validation (disjoint HF-black splits)

| domain | run | n | % sup codebook | % sup codebook-free | ratio cb/cf |
|---|---|---|---|---|---|
| solid | cb_main (black 0:80) | 20 | +80.89% | +96.76% | 5.903 |
| solid | cb_A (black 0:40) | 20 | +80.75% | +96.76% | 5.946 |
| solid | cb_B (black 40:80) | 20 | +81.07% | +96.76% | 5.848 |
| content | cb_main | 88 | +1.36% | +1.31% | 0.999 |
| content | cb_A (subset 30) | 30 | +1.49% | +1.45% | 1.000 |
| content | cb_B (subset 30) | 30 | +1.49% | +1.45% | 1.000 |

- Solid: the codebook result reproduces within **0.3 percentage points** across
  the two disjoint HF-black halves (80.75 / 81.07). Not a fluke. Codebook-free is
  deterministic (96.76% on all three).
- Content: cb_A and cb_B are byte-for-byte identical to each other (1.49 / 1.45)
  because the codebook is not consulted on content, so the split is irrelevant
  there.

---

## 5. Maximum-strength stress

| domain | n | % sup codebook | % sup codebook-free | ratio cb/cf | fid cb | fid cf |
|---|---|---|---|---|---|---|
| solid | 20 | +97.79% | +98.85% | 1.932 | 0.456 | 0.611 |
| content | 30 | +3.96% | +3.71% | 0.997 | 7.735 | 7.776 |

At Maximum strength the codebook's 4-pass subtraction nearly catches up on
solid (ratio drops from 5.90 to 1.93) but codebook-free still wins on raw
residual. On content the tie holds (0.997), and both paths become more
destructive (fidelity ~7.7/255). Maximum strength does not change the verdict.

---

## 6. Secondary cross-check: repo SynthID detector (weak signal)

`wmr detect <img> --codebook cb_main.wcb --legacy` (n=10 per domain). The repo's
own code flags this FFT-correlation detector as near random baseline for
SynthID-Image (`synthid_detector.cpp:118-125`), so this is a sanity delta only,
not a verdict input.

| domain | original conf | codebook conf | codebook-free conf |
|---|---|---|---|
| solid | 80.0% | 77.7% | 79.9% |
| content | 56.8% | 56.9% | 56.8% |

- Solid: the detector's `struct_ratio` saturates at 0.999 (it tracks the 1/f
  spectral shape, which the mean+noise replace preserves). So the detector does
  NOT validate codebook-free's lower raw residual; if anything it slightly
  favors the codebook (-2.3 pts vs -0.1 pts). This reinforces that the raw
  residual metric on solid rewards "image replaced by noise", not carrier
  removal a detector would recognize.
- Content: all three sit at the ~57% random baseline. No movement. Neither path
  changes what the (weak) detector sees.

---

## 7. Why the codebook is inert on content (code path)

This is the load-bearing finding. Both subtractors branch on
`avg_std > 0.05` (solid black reads 0.0139, content reads 0.28):

- `CodebookSubtractor::remove_synthid` (`codebook_subtractor.cpp:109-125`):
  on a content image it logs `"Content image detected ... Skipping carrier
  subtraction, applying spectral disruption only."` and sets `num_passes = 0`.
  The per-pass loop that consults the codebook profile (`magnitude_bgr`,
  `consistency_bgr`, `phase_consistency_bgr`, `phase_bgr`) never runs. The
  remaining phase-noise step (lines 218-256) uses a `phase_sigma` fixed by the
  strength enum and reads NO codebook field. So on content, the loaded codebook
  is dead weight: cb_main, cb_A, cb_B, or no codebook at all produce the same
  output. The cross-validation table confirms this (content cb_A == cb_B to the
  digit).
- `NoiseResidualSubtractor::remove_synthid` (codebook-free): on content it
  perturbs magnitude by ~3% (Moderate) and adds phase noise in the r=30..500
  band. It takes no codebook by definition.

This is a deliberate design choice, and the data backs it: on content the
carrier is <0.1% of spectral energy, so magnitude subtraction "primarily removes
image content" (the code comment). The 1.36% band drop on content is the final
3x3 blur, not carrier suppression. So the realistic-use-case question ("does a
codebook help on real images?") answers itself: no, because applying it would
damage content, and the code intentionally does not apply it.

---

## 8. Verdict: NOT WORTH IT

1. **Realistic content domain: the codebook cannot beat codebook-free.** It is
   identical within noise (0.999 ratio) and that is by construction, not by bad
   training data. The code path makes it inert on content. Reproducible across
   two disjoint training splits and at Maximum strength.
2. **Matched solid-color domain: the codebook loses on the raw metric.**
   Codebook-free suppresses 96.76% vs the codebook's 80.89% (5.9x lower
   residual) because its uniform-image branch replaces the image with mean+noise
   and drives the band to the noise floor. The codebook is more faithful to the
   original (fidelity 0.133 vs 0.512) but "more faithful to a solid-black
   carrier image" is not a meaningful product property. Solid-color is not a
   real-world use case; users process real images.
3. **The raw metric rewards destruction on solid, so codebook-free's win there
   is partly a metric artifact.** The repo's detector (acknowledged
   near-baseline) does not validate it. There is no metric on which the codebook
   is the clear, decisive winner.
4. **Cost.** The codebook is a ~300 MB file (`/tmp/cb_main.wcb` = 308 MB for 3
   profiles). Codebook-free needs no download and is the already-shipped default.
   A 300 MB asset that changes nothing on content and loses on solid is not
   worth shipping as the default.

**Recommendation (for the controller, not applied here):** do NOT ship a
dataset-derived codebook as the default. Keep `--codebook` as the opt-in path it
already is. The codebook only becomes worth revisiting if (a) the code path is
changed to let carrier subtraction run on content (today it deliberately does
not, because it would damage content), AND (b) a content-safe subtraction is
demonstrated to actually reduce a verified SynthID signal (which requires the
neural decoder the repo notes is not public).

### 8.1 Production-build scope note (does adding more data change this?)

The final production codebook would additionally include HF `gemini_white` (253),
HF `gemini_random` (88), and other verified-SynthID still-image dirs. **That does
not change the verdict**, because the gate is architectural, not data-limited:

- Adding `gemini_random` to the codebook would give the content branch a matching
  2816x1536 profile, but the content branch sets `num_passes = 0` and never reads
  it. The output would still be identical to codebook-free on content.
- More solid-color profiles (`gemini_white`, etc.) would only help on
  solid-color inputs, where codebook-free already wins on the raw metric and the
  detector does not validate the codebook.
- The one thing that WOULD change the picture is a content-image carrier-
  subtraction mode that does not exist today. Per the `CodebookSubtractor`
  comment, it was tried and rejected ("carrier subtraction primarily removes
  image content"). Re-opening that requires a content-safe operator (a real
  detector or a learned mask), not more codebook data.

### 8.2 Honest limitations of this gate

- The carrier-band residual is a proxy, not a SynthID verdict. Verified SynthID
  removal needs the neural decoder (arXiv:2510.09263), which is not public. The
  repo's FFT detector is near random baseline. So "lower residual" is a
  frequency-band reading, not proof of removal, and on content it is dominated
  by image content.
- The HF `gemini_random` content set is 2816x1536, which has no exact codebook
  profile and is resized. This does not matter here (content branch ignores the
  profile), but it would matter for any future content-subtraction mode.
- `gemini_black` is solid black at one resolution (1024x1024). Other solid
  colors and resolutions are covered by the local fixtures in the codebook but
  not all present in the held-out test set; the solid-domain verdict rests on
  the 20 held-out 1024x1024 black images plus the prior carrier characterization
  doc.

---

## 9. Reproduction

```sh
# 0. download HF subsets (gemini_black: 100 PNG, gemini_random: 88 PNG)
hf download aoxo/reverse-synthid --repo-type dataset \
  --include "gemini_black/*" --include "gemini_random/*" --local-dir /tmp/rsid

# 1. assemble strictly-disjoint splits (Python; symlinks by index)
#    cb_train_main = HF-black[0:80]  + local fixtures (3 res)
#    cb_train_A    = HF-black[0:40]  + local fixtures
#    cb_train_B    = HF-black[40:80] + local fixtures
#    test_black    = HF-black[80:100]                 (solid held-out)
#    test_random   = HF gemini_random (all)           (content held-out)
# (see the harness's split section; local fixtures are the 4 paths in section 2.2)

# 2. build the three codebooks
./build/wmr build-codebook /tmp/cb_train_main -o /tmp/cb_main.wcb  # 3 profiles
./build/wmr build-codebook /tmp/cb_train_A    -o /tmp/cb_A.wcb
./build/wmr build-codebook /tmp/cb_train_B    -o /tmp/cb_B.wcb

# 3. A/B suppress + measure residual (default strength)
python3 /tmp/measure_gate.py /tmp/cb_main.wcb default main
GATE_CONTENT=/tmp/test_random_30 python3 /tmp/measure_gate.py /tmp/cb_A.wcb default A
GATE_CONTENT=/tmp/test_random_30 python3 /tmp/measure_gate.py /tmp/cb_B.wcb default B
# Maximum strength stress:
GATE_CONTENT=/tmp/test_random_30 python3 /tmp/measure_gate.py /tmp/cb_main.wcb 1.0 max

# 4. secondary detector cross-check (weak signal, near baseline)
python3 /tmp/detector_check.py /tmp/cb_main.wcb main
```

Harness: `/tmp/measure_gate.py` (residual + fidelity), `/tmp/detector_check.py`
(repo detector delta). Raw per-image JSON: `/tmp/gate_result_{main,A,B,max}.json`.
Downloaded dataset left at `/tmp/rsid` for the follow-up build per the task spec.

### Files touched by this gate

- Doc (this file): `docs/research/synthid-codebook-vs-codebook-free-gate.md`.
- No source changes, no default changes. Codebooks and harness live under
  `/tmp` and are not committed.
