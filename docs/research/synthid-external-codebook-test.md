# reverse-SynthID expert codebook through our subtractor

Date: 2026-08-01. Branch: `synthid-correctness-pack`. Follows `synthid-codebook-not-viable.md`.

Goal: settle whether an expert-built spectral codebook, run through OUR
`CodebookSubtractor`, (a) produces the scattered-dot artifact and (b) suppresses
the SynthID carrier. This is the "one last try" for the fixed-codebook approach,
using reverse-SynthID's prebuilt `rsid_spectral_codebook_v4.npz` (built with
cross-color consensus from solid-color reference images, so the visible watermark
is excluded by construction).

No source or defaults were changed. Measurement only.

## 1. Format conversion (their npz -> our WMRCB02 .wcb)

Source field semantics (read from `src/extraction/synthid_bypass_v4.py` in
`aloshdenny/reverse-SynthID`):

- `mag`   uint8 + per-profile `mag__scale`: stored LOG2, so
  `mag_real = 2^(uint8 * mag__scale) - 1`. It is the mean across solid-color
  references of `mean|FFT|`, and the FFT is taken on `[0,255]` images.
- `phase` int8 + `phase__scale` (= pi/127): `phase_real = int8 * phase__scale`,
  the angle of the cross-color mean unit-phase.
- `cons`  uint8 (no scale, so /255): cross-color phase coherence
  `|mean_colors(e^{i*phi})|` in [0,1]. Bins with `cons < 0.55` are zeroed at
  save, so the stored `cons` is already a sparse carrier mask. Semantically this
  matches our `phase_consistency_bgr` (mean resultant length), NOT our
  `consistency_bgr` (magnitude-variability gate).
- `cw`    uint8 /255: per-bin carrier weight `cons^2 * (0.5 + 0.5*inv_agreement)`.
- channel axis is RGB.

Conversions applied to fit our subtractor (BGR, FFT on `[0,1]` images, full
spectrum, fftw natural layout):

- `mag`: log2-decode, then divide by 255 to put it into `[0,1]`-image FFT units
  (our subtractor FFTs the `/255` image). Verified two ways: the DC bin decodes
  to 442600 in `[0,1]` units (plausible for the mean DC of mixed solid colors),
  and the inverse-FFT spatial carrier has std 0.0015, matching genuine SynthID
  strength (sub-LSB). Without the `/255` the spatial std is 0.37, which would
  destroy the image; the scale is required.
- `phase`: `int8 * phase__scale`, no rescale.
- half-spectrum `(1024, 513)` -> full `(1024, 1024)` with row-negation:
  `F[y, x] = conj(half[(-y) mod H, W-x])`. Verified: the inverse-FFT imaginary
  part is 5e-17 of the real part and the result matches `np.fft.irfft2` exactly.
  (The naive mirror without the row-negation leaves a 0.43 imaginary/real ratio,
  a trap.)
- RGB -> BGR by reversing the channel axis.

Three codebooks written (`/tmp/convert_rsid.py`):

- `cb_rsid.wcb` (PRIMARY): `consistency_bgr = ones`, `phase_consistency_bgr =
  ones`. Disables both of our gates so the raw `magnitude * phase` subtraction
  runs. This is the fair test of whether their carrier signal suppresses.
- `cb_rsid_faithful.wcb`: `consistency_bgr = ones`, `phase_consistency_bgr =
  their cons`. The semantically-aligned mapping (their phase coherence gates our
  phase-consistency gate).
- `cb_rsid_cwmag.wcb`: PRIMARY, but magnitude premultiplied by their per-bin
  `cw` carrier weight.

Channel-weight note: their per-channel weights are `{R:0.85, G:1.0, B:0.70}`
(their comment: "G strongest for SynthID"). Ours are `{B:0.85, G:1.0, R:0.70}`
(BGR). The numeric values coincide but B and R are swapped relative to the data.
Running their codebook through our subtractor therefore weights B more and R less
than their own pipeline would. This is part of the honest "through our subtractor"
result, not corrected for.

Load verification: all three files parse as `WMRCB02`, count=1, `1024x1024`,
sample_count=8 (their `n_content_refs`). `wmr synthid <img> --codebook <cb>
--force` produces valid outputs (per-channel means 0.43 to 0.79 on a near-black
input of mean ~0.48/255, no saturation).

## 2. Dot artifact

Test image: a held-out `gemini_black` 1024x1024 image (near-pure-black, pixel
mean ~0.48/255, std 3.55/255, classified uniform so the carrier-subtraction
passes run). `--synthid-strength 1.0 --force`.

Findings:

- The PRIMARY `cb_rsid.wcb` does NOT reproduce the contaminated-codebook
  behavior. The contaminated `cb_blackonly.wcb` NUKES the carrier band (see
  section 3) and produces a near-flat amplified diff (it over-smooths). The rsid
  codebook produces a moderate speckle that is comparable to the near-inert
  `cb_blanked.wcb` baseline. The image tool, on an equal-scale three-panel
  montage of the amplified residuals, reads the contaminated panel as "extremely
  sparse, 1-3 specks", and the rsid and blanked panels as comparable moderate
  speckle fields (rsid slightly denser than contaminated, markedly less dense
  than a codebook-free path).
- The rsid-specific carrier subtraction (isolated by differencing the rsid and
  blanked outputs, which cancels the common phase-noise step) has max amplitude
  31/255 and mean 0.46/255. The contaminated-specific subtraction (same
  isolation) has max 138/255. The rsid effect is roughly 4x weaker and does not
  show the bright, structured imprint the contaminated codebook leaves.
- High-frequency spatial energy (Laplacian variance, outside the diamond) of
  every output is BELOW the input (ratios 0.47 to 0.69). No variant ADDS visible
  high-frequency dots to the output; all of them smooth the image.

Dot verdict: their expert codebook does NOT produce the diamond-contamination
artifact. The cross-color consensus construction excludes the visible watermark
from the codebook, so the subtractor never imprints it back. What residual
speckle the rsid output shows is dominated by the phase-noise step (which runs
for every variant at sigma 0.70 for uniform images at Maximum strength) plus a
small carrier-shaped component from the genuine carrier being subtracted.

## 3. Carrier-band suppression (phase-noise-invariant)

The phase-noise step changes phase only, so it preserves `|FFT|`. Carrier-band
energy `sum |FFT|^2` over `r = 3..400` therefore isolates the magnitude
(carrier) subtraction cleanly. Per-channel average, three held-out images,
strength 1.0:

| variant                        | img1   | img2   | img3   | notes                         |
|--------------------------------|--------|--------|--------|-------------------------------|
| Gaussian-blur-only floor       | 0.85%  |        |        | the final 3x3 blur alone      |
| `cb_rsid` (expert, gates off)  | 5.18%  | 5.93%  | 4.93%  | genuine carrier               |
| `cb_rsid_faithful` (cons gate) | 6.43%  |        |        | their cons -> our pcons gate  |
| `cb_rsid_cwmag` (mag * cw)     | 7.13%  |        |        | adds their per-bin weight     |
| `cb_blanked` (diamond zeroed)  | 5.85%  | 6.56%  | 5.57%  | our cleaned codebook          |
| `cb_blackonly` (contaminated)  | 98.55% | 97.28% | 97.26% | nukes the band                |
| `--codebook-free`              | 98.84% |        |        | also nukes                    |

Suppression verdict: the expert codebook suppresses the carrier band by about
5-7%, which is modest, genuine, and non-damaging. It sits between the blur-only
floor (0.85%) and the damaging nuke (98%). It is in the same band as our own
diamond-cleaned `cb_blanked` codebook. The magnitude cap is NOT binding (their
carrier magnitude ~4 in `[0,1]` FFT units vs image `|FFT| ~20` at carrier bins,
cap `0.98 * |img_fft| ~19.6`), so this is their full carrier magnitude being
subtracted, not a cap artifact.

Why only ~5%: their codebook has nonzero magnitude at 16.9% of bins (the
cross-color carrier bins; everywhere else `cons < 0.55` was zeroed at build).
At those bins their magnitude is genuine SynthID strength. A coherent carrier
present at 17% of bins, at sub-LSB spatial amplitude, is only a small fraction
of the carrier-band energy of even a near-black image, so subtracting all of it
lifts only ~5% of the band energy.

## 4. Why our contaminated codebook nukes and theirs does not

Active-bin analysis of the three codebooks (ch0):

| codebook      | bins with mag > 0 | gate open (cons > 0.30) | effective | mag max |
|---------------|-------------------|-------------------------|-----------|---------|
| `cb_rsid`     | 16.9%             | 100% (gates forced ones)| 16.9%     | 442601 (DC, excluded by ramp) |
| `cb_blackonly`| 100%              | 100%                    | 100%      | 1974    |
| `cb_blanked`  | 100%              | 100%                    | 100%      | 1651    |

Our `cb_blackonly` was built from 80 near-identical black images, so
`consistency = 1 - std/max_std` is ~0.997 at EVERY bin (the per-bin std across
near-identical captures is tiny everywhere). The visible diamond's broadband FFT
(it has sharp edges, so its energy spreads across many frequencies) is captured
as "carrier" at every bin, with a peak magnitude of 1974. With the gate wide
open everywhere, the subtractor removes the diamond's spectrum from the whole
image, which is the 98% band nuke.

Their cross-color consensus (black, white, red, green, gray, ...) flags only
bins whose phase is coherent across ALL backgrounds. The visible diamond, being
absent from their reference set's content baseline by construction, does not
leak in; only the genuine cross-color carrier survives, at 17% of bins and
genuine magnitude.

## 5. Verdict

(a) Dots: the expert codebook does NOT produce the dot/nuke artifact. The
contamination in our codebook is a BUILD flaw (building from near-identical
images that contain the visible diamond sets consistency to ~1 everywhere and
captures the diamond's broadband FFT as carrier). Their cross-color consensus
build avoids it entirely. The dot artifact is not a `CodebookSubtractor` flaw:
fed a clean codebook, our subtractor behaves well (targeted, non-damaging).

(b) Suppression: the expert codebook suppresses the carrier meaningfully but
modestly, about 5-7%, which is the genuine SynthID carrier footprint. It does
not approach the damaging 98% over-removal, and it is only a few percent above
the blur-only floor.

Is a fixed spectral codebook viable? For modest, targeted, non-damaging
suppression on uniform images, yes, the expert codebook proves it works. For
meaningful suppression, no: the genuine SynthID-Image carrier is too weak
(~0.025/255, below the sensor-noise floor) for any fixed spectral template to
capture more than its own small consistent component, however expertly built.
This empirically confirms the hypothesis in `synthid-codebook-not-viable.md`
(that reverse-SynthID's shipped codebook is affected by the same ceiling):
through our subtractor it is near-inert-grade, suppressing only the genuine
carrier, and it avoids the dots only because the expert build excludes the
visible watermark rather than because it captures more invisible carrier.

Two honest caveats:

- The rsid codebook is profiled for `gemini-3.1-flash-image-preview`; the
  `gemini_black` test images are of unknown model. If the carriers differ, the
  rsid codebook would under-match. Its spatial carrier amplitude (std 0.0015)
  matches known SynthID strength, so this is probably not the limiting factor
  here, but it is not controlled for.
- Through THEIR subtractor the same codebook may suppress more: their
  `blog_pure` path uses `mag_cap * |img_fft|` as the amplitude and takes phase
  from the codebook, and they multi-pass with PSNR-floor rollback and apply the
  per-bin `cw`. This test runs their codebook through OUR subtractor only, which
  is what was asked.

## Artifacts

Conversion script and the three generated codebooks are under `/tmp`
(`/tmp/convert_rsid.py`, `/tmp/cb_rsid.wcb`, `/tmp/cb_rsid_faithful.wcb`,
`/tmp/cb_rsid_cwmag.wcb`). Not committed (regenerable, large).
