# SynthID regen validation: real-hardware results (procedure + scaffold)

Out-of-band measurement plan and results table for `wmr synthid --synthid-attack regen` (Phase 2 of the SynthID work; see `~/.claude/plans/synthid-regen-phase2.md` Task 9, and `docs/research/synthid-investigation-summary.md` for the broader SynthID context). This doc is the procedure the owner runs on real hardware and the table the numbers go into. It is NOT a CI artifact: the `build` matrix in `release.yml` has no NVIDIA GPU on Windows and only a paravirtualized Metal device on mac, so GPU validation cannot happen in CI. The owner runs it locally and pastes the numbers below.

## Honesty caveat (read first)

There is **no public SynthID-Image verifier** (Gemini's in-app check is account-gated, ~10 image checks per 24h, natural-language output only). So regen's "removal" **cannot be objectively proven by us**. The carrier-band energy metric below is a **relative heuristic**: it measures how much the FFT magnitude in a fixed mid-frequency band drops between input and output, on a uniform fixture where the content noise floor cannot mask the signal. A large drop is consistent with carrier suppression; it is not a proof of removal. Record the numbers here as a reproducibility anchor (seed is fixed at 42 so a scrub is bit-stable given the same model + backend), and treat the Gemini in-app check as the only external signal.

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

TBD (owner run). Replace each cell with the measured value; mark "n/a" where the run was not done and "DNF" if it did not finish.

| Hardware + backend                | Fixture / size         | wall_s | validity mean-delta vs input (max channel, /255) | carrier-band drop % | notes |
|-----------------------------------|------------------------|--------|--------------------------------------------------|---------------------|-------|
| M2 Pro, Metal                     | 2400x1792 black (4K)   | TBD    | TBD                                              | TBD                 | |
| M2 Pro, Metal                     | 2400x1792 test1 (4K content) | TBD | TBD                                           | n/a (content)       | fidelity spot-check |
| RTX-class GPU, CUDA               | 2400x1792 black (4K)   | TBD    | TBD                                              | TBD                 | |
| RTX-class GPU, CUDA               | 2400x1792 test1 (4K content) | TBD | TBD                                           | n/a (content)       | fidelity spot-check |
| CPU (no GPU)                      | 896x1200 (1024-class, `--regen-no-tile`) | TBD | TBD                                  | TBD                 | 4K tiled on CPU is not a target |

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
- **No public verifier.** Stated twice on purpose. The numbers above are a reproducibility anchor, not a removal proof.
