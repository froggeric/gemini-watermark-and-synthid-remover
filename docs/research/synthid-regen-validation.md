# SynthID regen validation: real-hardware results (procedure + scaffold)

Out-of-band measurement plan and results table for `wmr synthid --synthid-attack regen` (Phase 2 of the SynthID work; see `~/.claude/plans/synthid-regen-phase2.md` Task 9, and `docs/research/synthid-investigation-summary.md` for the broader SynthID context). This doc is the procedure the owner runs on real hardware and the table the numbers go into. It is NOT a CI artifact: the `build` matrix in `release.yml` has no NVIDIA GPU on Windows and only a paravirtualized Metal device on mac, so GPU validation cannot happen in CI. The owner runs it locally and pastes the numbers below.

## Honesty caveat (read first)

Google's "Verify with SynthID" IS an official verifier, but it is manual (in-app, account-gated, ~10 image checks per 24h, natural-language output only; no API). So regen's per-output "removal" **cannot be auto-confirmed by us in-process** (the knee below WAS validated against that tool). The carrier-band energy metric below is a **relative heuristic**: it measures how much the FFT magnitude in a fixed mid-frequency band drops between input and output, on a uniform fixture where the content noise floor cannot mask the signal. A large drop is consistent with carrier suppression; it is not a proof of removal. Record the numbers here as a reproducibility anchor (seed is fixed at 42 so a scrub is bit-stable given the same model + backend), and treat the Gemini in-app check as the only external signal.

This is consistent with the honesty reframe shipped for the spectral path: the docs say "suppress (heuristic; not a verifiable removal)", never "remove". The regen path is the only technique with published evidence of defeating a SynthID detector (reverse-SynthID's Round 06), but the evidence is empirical, not a guarantee.

## The harness

`scripts/regen_validate.sh <image>` (dev-only, NOT runtime code). It:

1. Times `wmr synthid <image> --synthid-attack regen -o /tmp/wmr_regen_out.png`.
2. Prints per-channel (BGR) mean of input vs output, plus the per-channel mean-delta (the "validity" check: a scrub should not shift the global color far).
3. Prints FFT magnitude energy in the radius 30-500 band of the grayscale image, input vs output, and the relative drop as a percentage.

The two measurement steps use `python3 + cv2 + numpy`. That is acceptable because the harness is a developer-side tool, never linked into the `wmr` binary; it sits under `scripts/` next to `verify_removal.py`, `visualize_spectral.py`, and `build_synthid_codebook.sh`, and is not packaged into any release. The no-runtime-Python tenet covers what the shipped binary executes, and this script is not that.

Run it as (defaults `WMR=build/wmr`, `OUT=/tmp/wmr_regen_out.png`):

```bash
./scripts/regen_validate.sh test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-black-gemini.png
# or against a release binary + custom output path:
WMR=./wmr OUT=/tmp/o.png ./scripts/regen_validate.sh test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-black-gemini.png
```

Requirements (the harness does NOT download these; `wmr` itself fetches them on first regen run, SHA256-pinned, to `~/.cache/wmr/`):

- a `wmr` binary built with `WMR_BUILD_REGEN=ON` (Task 1)
- SDXL base fp16 checkpoint (~6.5 GB)
- SDXL fp16-fix VAE (~250 MB, REQUIRED to avoid the well-known fp16-VAE NaN/black-output bug)
- `python3` with `cv2` and `numpy` on the dev machine (`pip install opencv-python numpy`)

## Fixtures

The fixture path in the original plan (`test-images/gemini-3.1-pro/2400x1792/pure-*-gemini.png`) is wrong: the real files are flat-named inside the `2400x1792/` subdir, prefixed with the dimensions. Use the actual paths below.

### Uniform pure-color fixtures (for the carrier-band probe)

These are 2400x1792 (call it "4K" for the table) near-uniform pure-color fields with the Gemini mark. Use these for the carrier-band drop metric: on a uniform field the content noise floor is minimal, so a carrier-band change is attributable. Black and white are the cleanest (no chroma content at all).

```
test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-black-gemini.png   (primary)
test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-white-gemini.png   (primary)
test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-grey-gemini.png
test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-red-gemini.png
test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-green-gemini.png
test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-blue-gemini.png
test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-cyan-gemini.png
test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-magenta-gemini.png
test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-yellow-gemini.png
```

For a statistical sample (a single capture's carrier is weakly content-correlated), the `pure-black/` subdir holds 30 distinct real Gemini 3.1 generations on near-uniform black at the same 2400x1792 size:

```
test-images/gemini-3.1-pro/2400x1792/pure-black/Gemini_Generated_Image_*.png   (30 files)
```

Loop them and average the carrier-band drop if a single-image number looks noisy.

### Content fixture (for the lossy / fidelity comparison)

regen is a low-strength SDXL img2img pass: it is lossy by design (it redraws the image). The carrier-band drop is not interpretable on busy content (content energy dominates the band), so on content we measure fidelity instead (validity mean-delta, and eyeball the structural fidelity). Use a real Gemini generation with real content:

```
test-images/2400x1792-test1-gemini.png     (4K content)
test-images/2400x1792-test2-gemini.png     (4K content)
```

### Smaller fixture (for the CPU 1024 row)

For the CPU row the interesting number is wall time on a single 1024-class image (tiled regen at 4K on CPU is impractical, so the CPU target is the smaller path). Use a `--regen-no-tile` run on a smaller fixture, or accept that 4K CPU is the "do not attempt" cell:

```
test-images/896x1200-test3-gemini36.png
test-images/896x1200-test4-gemini36.png
```

(These are Gemini 3.6 fixtures; whether they carry SynthID is not separately verified, but the wall-time measurement is the point of the CPU row.)

## Procedure

For each row of the table below, on the target hardware:

1. Build `wmr` with `WMR_BUILD_REGEN=ON` on that platform (Metal on mac arm64, CUDA on linux/windows with an NVIDIA GPU, CPU otherwise). Confirm the backend by checking the `wmr synthid ... regen` log line for the device it picked (Task 2 resolver; `sd_list_devices` falls back to CPU if the named backend cannot init).
2. Run the harness on the primary black fixture (first regen run downloads ~6.5 GB + 250 MB; subsequent runs are warm):
   ```bash
   ./scripts/regen_validate.sh test-images/gemini-3.1-pro/2400x1792/2400x1792-pure-black-gemini.png
   ```
3. Record `wall_seconds`, the per-channel `mean_delta` (max of B/G/R), and `carrier_band drop %` into the table.
4. Spot-check fidelity on a content fixture:
   ```bash
   ./scripts/regen_validate.sh test-images/2400x1792-test1-gemini.png
   ```
   Record the validity mean-delta. Open input vs output side by side and note any tile seams or detail loss (raise `--regen-steps` and overlap if seams appear).
5. (Optional, external signal) Run a few outputs through the Gemini in-app SynthID check (~10/day per account). This is the only external signal; record the qualitative result, not as proof.

`seed=42` is fixed in `RegenConfig`, so a scrub is reproducible across runs given the same model + backend.

## Results

| Hardware + backend                | Fixture / size         | wall_s | validity mean-delta vs input (max channel, /255) | carrier-band drop % | notes |
|-----------------------------------|------------------------|--------|--------------------------------------------------|---------------------|-------|
| M4 Pro, CoreML (Phase 3)         | 896x1200 Gemini 3.6 (single-tile) | ~68 total (~50 load + ~18 inference) | 0.38 (G) | TBD | ~3.4x faster than CPU on same fixture; mean diff vs input 0.36/255 |
| M4 Pro, CPU (sdcpp, Phase 2)      | 896x1200 Gemini 3.6 (single-tile) | ~231 total (inference only) | 1.01 (G/B) | TBD | Mean diff vs input 0.89/255; CoreML vs CPU inter-backend diff 1.23/255 |
| M4 Pro, CoreML (Phase 3)          | 2400x1792 black (4K)   | TBD    | TBD                                              | TBD                 | |
| M4 Pro, CoreML (Phase 3)          | 2400x1792 test1 (4K content) | TBD | TBD                                           | n/a (content)       | fidelity spot-check |
| M2 Pro, Metal                     | 2400x1792 black (4K)   | TBD    | TBD                                              | TBD                 | |
| M2 Pro, Metal                     | 2400x1792 test1 (4K content) | TBD | TBD                                           | n/a (content)       | fidelity spot-check |
| RTX-class GPU, CUDA               | 2400x1792 black (4K)   | TBD    | TBD                                              | TBD                 | |
| RTX-class GPU, CUDA               | 2400x1792 test1 (4K content) | TBD | TBD                                           | n/a (content)       | fidelity spot-check |
| CPU (no GPU)                      | 896x1200 (1024-class, `--regen-no-tile`) | ~231 | 1.01                                  | TBD                 | 4K tiled on CPU is not a target |

### CoreML A/B Validation (2026-08-04)

Test image: `reference-images/896x1200-gemini36/Gemini_Generated_Image_gawws5gawws5gaww.png` (896x1200)

**CoreML backend (`--regen-backend coreml`)**
- Wall time: ~68s total (~50s one-time model load + ~18s inference)
- Mean diff vs input: 0.36/255
- Per-channel (B,G,R): (0.35, 0.38, 0.35)

**CPU backend (`--regen-backend cpu`, sdcpp)**
- Wall time: ~231s total (inference only)
- Mean diff vs input: 0.89/255
- Per-channel (B,G,R): (1.01, 1.01, 0.65)

**Inter-backend comparison**
- CoreML vs CPU diff: 1.23/255
- Per-channel (B,G,R): (1.35, 1.36, 0.97)
- Speedup: ~3.4x faster (CoreML)

Both backends produce content-preserving results with low inter-backend diff, validating the CoreML implementation.

### Phase 4 ORIGINAL-attention GPU placement A/B (2026-08-06)

Task #43: the macOS CoreML UNet was re-converted with `--attention-implementation
ORIGINAL` so it routes to the Apple GPU instead of collapsing to CPU (the
SPLIT_EINSUM variant fails the ANE and falls back). This A/B isolates the
compute-unit variable on the SAME ORIGINAL model.

Setup: ORIGINAL-attention UNet loaded via the pipeline directly (bypassing the
regenerator bootstrap, which SHA-pins the SPLIT_EINSUM UNet), single 1024x1024
tile of `test-images/poster-artnight.png` (resized, so no tiling confound),
strength 0.10, N=50 (5 denoise steps), seed 42. The new `WMR_COREML_SD_COMPUTE_UNITS`
knob flips the unit; the first-UNet-predict timing log (the only runtime placement
signal, since CoreML exposes no chosen-unit API) records where each run landed.

| Run | Compute units | UNet predict #1 | input->output diff_mean |
|-----|---------------|-----------------|-------------------------|
| A   | `all` (GPU)   | 4836 ms         | 8.17/255                |
| B   | `cpu`         | 7967 ms         | 7.93/255                |

GPU-vs-CPU output diff (the gate): overall **1.92/255**, per-channel RGB
(2.17, 1.68, 1.92); per-channel means within ~1/255 (no saturation); max abs
pixel diff 255 on a handful of outlier pixels (mean stays tiny). Gate (< 2.0/255):
**PASS**.

Conclusions:

- ORIGINAL attention routes to the GPU. Run A's predict #1 (4836 ms) is well under
  run B's forced-CPU 7967 ms; if `all` were also CPU, both would sit at ~8000 ms.
  (predict #1 includes one-time GPU context warmup, so it is a placement detector,
  not a steady-state per-tile number; Phase 3.2 measures steady-state.)
- GPU and CPU produce near-identical output (1.92/255 apart = fp16 GPU-vs-CPU
  rounding over 5 Euler steps + the VAE round trip; no systematic shift, no
  collapse). They are removal-equivalent at s=0.10/N=50, so the validated CPU
  removal knee carries over to the GPU output.
- This validates #43's core hypothesis (ORIGINAL runs on the GPU and is correct)
  at the cheap-pre-filter gate. The two remaining gates are the user's: the Google
  "Verify with SynthID" re-clear on the 9-image set (Phase 2.2, the adoption gate)
  and the Instruments placement confirmation + steady-state timing (Phase 3.x).

Caveat: total img2img wall on this first run was GPU ~19s vs CPU ~31s for the one
tile, both dominated by the first-run CoreML compile/warmup. Steady-state per-tile
(Phase 3.2) is needed before claiming the ~1-3 s/tile target; do not quote predict
#1 as the per-tile cost.

### Phase 4 ORIGINAL/GPU SynthID verifier re-clear (2026-08-06) - ADOPTION GATE PASSED

The 9-image Phase 3 validation set (`reference-images/vae-testing/`, including the
double-watermarked img2img image `Gemini_Generated_Image_iu84oxiu84oxiu84-2.png`)
was regenerated with the ORIGINAL-attention UNet on the GPU (CoreML, compute
units=all, default strength 0.10 / steps 50, real tiled img2img; outputs in
`reference-images/vae-testing-regen-gpu/`). Each output was then checked against
Google's official "Verify with SynthID" verifier.

Result: **9/9 cleared (no SynthID detected), including the double-watermarked
image.** This is the adoption gate for #43: ORIGINAL-on-GPU preserves the
validated removal knee, so the CPU/SPLIT_EINSUM 9/9 from 2026-08-05 carries over
to the GPU. No re-checks were needed (verifier noise did not trigger).

This clears #43 to proceed to distribution (Phase 4), pending the Instruments
placement confirmation + steady-state timing (Phase 3.x) and the standard
release-cut steps.

## Acceptance targets (owner verifies)

These are the targets Task 9's acceptance criteria check against. Each must hold on at least one GPU backend:

- **4K regen wall time < 10 minutes** on a GPU backend (Metal M2 Pro, and/or RTX-class CUDA).
- **Carrier-band drop >= 40%** on a uniform SynthID-bearing fixture (primary: `2400x1792-pure-black-gemini.png`).
- **Validity within +/-25/255** (max-channel per-channel mean-delta between input and output) on the same uniform fixture.

If Metal misses the wall-time target, the documented escape is the Phase-3 CoreML fallback (NOT in this plan; see risks in `synthid-regen-phase2.md`). If the carrier-band drop is under target on uniform black, raise `--regen-strength` (default 0.05; range 0.02 to 0.15) and re-measure; the trade-off is more fidelity loss. If validity drifts past +/-25/255, lower `--regen-strength`.

## Known limitations + measurement gotchas

- **Metal SDXL may be slow** on large matrices (flagged by the upstream `stable-diffusion.cpp` build doc). The M2 Pro row is the one most likely to miss the 10-minute target.
- **Windows CUDA** cannot run in CI (windows-latest has no NVIDIA GPU); the CUDA row is owner-run only.
- **Carrier-band drop on content is not interpretable.** The metric only answers a question on a uniform fixture. Do not quote a content-fixture drop number as evidence of anything; record validity + fidelity there instead.
- **Single-image carrier measurement is noisy.** The SynthID carrier is ~0.025/255, sub-LSB, and weakly content-correlated; a single black capture's drop can swing with capture noise. Average across the `pure-black/` subdir (30 distinct generations) for a stable number.
- **VAE NaN/black output** is the well-known fp16-VAE failure mode. The harness's validity block errors out if either image fails to decode; if you see it, confirm the fp16-fix VAE is being passed (`--regen-vae-path` override exists, but the default resolution already pins it).
- **Tile seams** at higher strength: low strength (0.04 to 0.07) minimizes drift; raise `--regen-steps` and overlap if seams appear on the content-fixture row.
- **First-run download is large** (~6.5 GB model + ~250 MB VAE). The downloader is resumable (HTTP Range via `.part`); `--regen-no-download` plus a manually-placed model + VAE covers air-gapped use.
- **No public verifier API.** Google's "Verify with SynthID" is manual/in-app. The numbers above are a reproducibility anchor; the knee was confirmed via that manual tool.
