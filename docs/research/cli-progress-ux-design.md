# CLI progress UX design

Status: APPROVED by the owner (all eight decisions below locked). Implementation pending.
Companion to the operation docs; this governs what `wmr` prints while working.

## Why

Reduce *perceived* wait time and the "is it stuck?" anxiety, not just add chrome.
Grounded in: Myers (CHI '85, percent-done feedback reduces perceived duration);
Maister (1985, uncertain/unoccupied waits feel longer, so stage labels + honest ETAs
help); Harrison et al. (AVI '06, pauses near the end feel worst, so do not let the bar
sit at 99%); Tversky/Kahneman framing (one-time costs are tolerated when labeled as
one-time). The single biggest win is breaking the **silent regen tile loop**.

## Current-state gaps (surveyed)

- Regen tile loop (`src/core/regenerator.cpp` ~402-416) is silent for minutes on a 4K regen.
- Download milestones are bare "5%" with no bytes/rate/ETA (`regenerator.cpp` ~100-109).
- One-time costs (first download, first CoreML compile) are not framed as one-time, so they read as stalls.
- No resource/placement signal (the UNet-on-GPU finding is buried in a one-shot diagnostic).
- Single point-estimate ETAs in video that are wild early.
- No TTY/non-TTY split: append-only timestamped lines everywhere (laggy on a terminal, fine for CI).

## Per-operation design

### `synthid` / `--synthid-attack regen` (the design driver)

TTY (interactive), refreshing single line per stage:
```
[1/4] Downloading model (one-time first run)
  sd_xl_base_1.0.safetensors   3.41 GiB / 6.46 GiB  52%  18.2 MiB/s  ETA 2m47s  [####################----------------] 52%
[2/4] Loading CoreML pipeline (one-time first-run compile, ~30s)
[3/4] Regenerating  tile 5/12  coreml-gpu  4.1s/tile  0.24 tile/s  ETA 29s  [##############------] 42%
[4/4] Detail restoration  applied (luminance=132, keep=5%)
SynthID regen complete in 3m51s (lossy; verify with Google's SynthID tool to confirm)
```
Non-TTY (piped/CI): same stages + content, append-only, timestamp prefix, throttled to
milestones (every ~25% of download; every tile; on percent-change). This keeps CI logs
linear and greppable.

### `video`
Single refreshing line, scene-aware, ETA hidden until >=5% of frames:
```
  frame 1240/4521  27%  143.2 fps  ETA 23s  scene 1/3  [######--------------]
Audio: copied
Done: 4498 processed, 23 skipped in 35m47s (detection conf=0.91)
```

### `remove` / `visible` / `detect`
Stay terse (sub-second). Optional polish: add resolved geometry to the "removing..." line.

### Batch
```
Batch: 24 images -> cleaned/
  [3/24] gemini36-portrait.png  done (0.4s)  0.9 img/s  ETA 9s
Batch complete: 24 ok, 0 failed in 11.2s
```

## Abstraction: `src/cli/progress.{hpp,cpp}`

A small TTY-aware reporter (pure helpers unit-tested; rendering verified manually):

```cpp
namespace wmr {
enum class ProgressStream { Stderr, Stdout };
bool progress_is_tty(ProgressStream s);                 // isatty + NO_COLOR/TERM=dumb/CI gate
std::string format_bytes(uint64_t b);                   // "3.41 GiB"
std::string format_eta(double seconds);                 // "2m47s", "<10s", "ETA --"
std::string format_bar(double frac, int width = 24);    // ASCII "[####---------]"
class RateEstimator {                                   // EWMA; ETA hidden until >=3 samples + >=5% done
    void sample(double unit_seconds);
    double rate_per_sec() const;
    double eta_seconds(int remaining) const;            // 0 == unknown
};
class ProgressReporter {                                // tiles / frames / batch items
    ProgressReporter(std::string label, int total, std::string unit, ProgressStream s = Stderr);
    void update(int done, const std::string& extra);    // throttled refresh
    void finish(const std::string& summary);
};
class ByteProgress {                                    // downloads (curl xferinfo-driven)
    ByteProgress(std::string label, uint64_t total_bytes, ProgressStream s = Stderr);
    bool update(uint64_t done);                         // false == cancel (curl contract)
    void finish();
};
class Stage {                                           // "[k/N] <name>" frame
    Stage(int index, int total, std::string name, std::string note = "");
};
}
```

- TTY: refresh with `\r`, ANSI bold labels, ASCII bar; throttle >=66 ms (frame loops),
  >=500 ms (downloads). Non-TTY: append-only milestone lines, only on throttle/percent-change.
- Color gated on TTY + `NO_COLOR` unset + `TERM != "dumb"` + `CI` unset; never color-only.
- `progress.cpp` in top-level `SOURCES` AND `tests` `LIB_SOURCES` (shared, not feature-gated).
- Regen-specific calls stay under `#ifdef WMR_BUILD_REGEN`.

## Call sites to replace/augment

- `regenerator.cpp` download milestones -> `ByteProgress` (bytes/rate/ETA/bar) driven by the existing curl `xferinfo`; strip the full URL (show filename).
- `regenerator.cpp` CoreML init -> `Stage(2,4,...)` with "(one-time first-run compile, ~Xs)".
- `regenerator.cpp` tile loop -> `ProgressReporter("Regenerating", tiles, "tile")`; per-tile `update(i, "coreml-gpu, 4.1s/tile")`; time each tile.
- `coreml_sd_pipeline.mm` first-predict diagnostic -> route the placement verdict into tile-1 `extra` (`coreml-gpu (first predict 4787 ms => GPU)`); label `coreml-gpu` vs `coreml-cpu` by the ms band.
- `regen_restore.cpp` -> `Stage(4,4,"Detail restoration")`.
- `cli_app.cpp` SynthID-complete line -> add elapsed + tile count + backend recap.
- `video_processor.cpp` frame loops -> `ProgressReporter` with scene index + rolling ETA (hidden early); add an `Audio: copied` stage label.
- `batch_processor.cpp` -> `ProgressReporter("batch", total)`.

## Locked decisions (owner-approved)

1. Progress -> **stderr**; spdlog info stays stdout. (`wmr … > /dev/null` still shows progress; stdout clean.)
2. Default = the new UX; `-v` = debug; add **`--no-progress`** to suppress.
3. No tri-state `--progress`; auto-detect (TTY on, non-TTY milestone) + `--no-progress`.
4. Color: bold labels + ASCII bar, gated TTY + `NO_COLOR`/`TERM=dumb`/`CI`, never color-only.
5. One-time framing on first download + first CoreML compile.
6. Backend string on the tile line (`coreml-gpu`/`cpu`/`metal`).
7. ASCII bar `[####---------]`.
8. ETA hidden until >=3 samples + >=5% done; rolling EWMA; range when uncertain.
