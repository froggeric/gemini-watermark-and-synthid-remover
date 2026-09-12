# wmr GUI: embedded local web UI (design spec)

- Date: 2026-09-12
- Status: design approved by the owner (all three sections), pre-implementation
- Target release: 1.17.0

## Decision

wmr gets a GUI as an embedded local web UI inside the existing single binary. Running
`wmr` with no arguments always launches it (terminal or pipe); an explicit `wmr gui`
subcommand does the same. Opt-outs: `WMR_NO_GUI=1` env or `CI` set in the environment
restore the current help-printing behavior; `wmr --help` always prints help.

The owner's stated preference order was: native feel per platform, then macOS-only
native, then local web UI. A 4-agent research sweep (2026-09-12: repo integration
surface, cross-platform framework landscape, web-UI prior art, macOS-native path)
found:

- Cross-platform native C++ toolkits do not fit this project. Qt has the best
  widgets but multi-hour vcpkg builds per CI leg (a real risk to the Windows ~6 h
  cap) plus LGPL static-linking obligations. wxWidgets is the only technical fit
  (real native controls, minutes of CI, license permits static linking) but reads
  dated on macOS and lacks Windows dark mode until 3.4.0. Slint (Fluent styling
  everywhere since 1.16, plus a Rust toolchain in CI), Flutter Desktop (a Dart app
  wrapped around the engine, no Rosetta cross-compile story), GTK4 (vcpkg port is
  shared-only, gtkmm has no port), and Dear ImGui (presents as a dev tool) are all
  rejected.
- The macOS-native pattern is a SwiftUI app spawning the embedded CLI as a
  subprocess, about 11 to 15 working days, mac-only, plus a new Xcode CI leg and a
  Swift codebase to maintain. Direct Swift-C++ interop is a poor fit for this
  codebase (cv::Exception cannot be caught in Swift; spdlog/fmt templates defeat
  the clang importer; dual CMake+Xcode builds forever). Ollama, HandBrake, and
  whisper.cpp all use process or C-API boundaries instead.
- The embedded local web UI reaches every platform in the same binary at near-zero
  CI cost, and the exact reference implementation is already vendored in-tree:
  `external/stable-diffusion.cpp/examples/server/` is a cpp-httplib server with an
  embedded web UI and an async job manager (queued/running/done/failed/cancelled,
  cancel endpoint, polling). llama.cpp's built-in web UI (same pattern, same HTTP
  library) is well received in practice.

The owner chose the web UI, with no commitment to a later native shell. The local
API is nevertheless designed as a clean contract so a WKWebView or SwiftUI wrapper
could be added later without redesign (the Ollama architecture), but that is not on
any roadmap.

## Scope (v1)

The core still-image flow, for mainly non-technical users:

- Drag/drop or browse to pick 1..N images (multiple files form one batch job).
- Per-file preview of the detected watermark bbox, score, and geometry source.
- Remove with CLI-equivalent defaults; advanced overrides behind a disclosure.
- Before/after comparison view (slider).
- Per-file download of the cleaned image.

Out of scope for v1 (stay CLI-only): video, SynthID regen, metadata inspect/strip,
cache management. Also deferred: SSE live progress, ZIP download-all, thumbnails,
per-job log capture, remote bind (`--gui-host`), TLS.

## Architecture

One process, three thread roles:

1. cpp-httplib's worker pool serves HTTP.
2. Exactly one job worker thread runs all engine work, serialized. This is forced
   by the codebase: NcnnDenoiser, Regenerator, and MiganInpainter are documented
   not-thread-safe singletons, and the CoreML e5rt cache has no concurrency lock.
   A batch of N images is one job that loops files sequentially, the same shape as
   `batch_processor` today.
3. The worker updates a mutex-guarded job-state struct that HTTP threads read
   (polling). No SSE/WebSocket in v1: still-image removal is seconds-fast and
   polling at ~500 ms is plenty; polling also survives page reload because the
   server is the source of truth.

### Entry points

- `wmr gui`: starts the server. Flags: `--gui-port N` (default: ephemeral),
  `--no-browser`. Binds 127.0.0.1 only, always.
- No-args (`argc <= 1`, currently the help-print early return at
  `src/cli/cli_app.cpp:765`): always launches the GUI, interactive or not.
  `WMR_NO_GUI=1` or `CI` present in the environment prints help instead (the `CI`
  gate mirrors the update-check convention). `wmr --help`, `wmr --version`, and
  parse-error paths are unchanged.
- Known trade-off: under CLI11 subcommand resolution a file literally named `gui`
  must be passed as `wmr remove gui`. Acceptable; documented.
- Server lifetime: until SIGINT. The console prints a banner (version, URL,
  `Ctrl-C to stop`, `wmr --help for CLI usage`), opens the default browser only
  after `listen()` succeeds (avoids the Syncthing connection-refused race), and
  treats browser-launch failure as non-fatal (the printed URL is the fallback).
  Browser launch matrix: macOS `open`; Linux `xdg-open` gated on
  `DISPLAY`/`WAYLAND_DISPLAY`; Windows `ShellExecuteW`; WSL `wslview` then
  `cmd.exe /c start`; `BROWSER` env override honored.

### Security (all mandatory, all cheap)

- Bind 127.0.0.1 only, ephemeral port by default (bind port 0), `--gui-port` to pin.
- Per-run 128-bit token from the system CSPRNG, embedded in the URL, enforced on
  every request including static assets; miss returns 404, never 401 (probes learn
  nothing).
- Host header allowlist: `127.0.0.1:port`, `localhost:port`, `[::1]:port`. This
  defeats DNS rebinding (qBittorrent's mitigation).
- Reject `Sec-Fetch-Site: cross-site` when present.
- Never emit CORS headers (cross-origin reads stay browser-blocked).
- Caps: ~200 MB per uploaded file, ~100 files per job, at most 8 pending jobs
  (beyond that, reject with 429); reject oversize uploads early with 413.
- Magic-byte image sniff before decode; reject non-images.
- No TLS on loopback.
- Privacy invariant: the served page makes zero external requests (no CDN, no
  fonts, no analytics). The only network activity is the local server. Stated in
  the README.

### Job model (the sd.cpp AsyncJobManager pattern)

- `POST` a job, receive `{job_id}`; states `queued | running | done | failed |
  cancelled`.
- Per-file results appear in the state as they complete: detection bbox, score,
  geometry source, variant, outcome (`removed | passthrough | no-watermark |
  failed`), error text.
- Cancel is cooperative, checked between files (single-image removal is too fast
  to interrupt mid-file).
- Results live under `~/.cache/wmr/gui/<job_id>/`. Lifecycle: the whole
  `~/.cache/wmr/gui/` directory is wiped on server start; while the server runs,
  finished results stay available for re-download until the server exits.
- CLI exit-code semantics are preserved as outcomes: CLI rc 2 ("no watermark
  detected without --force") is the per-file `no-watermark` outcome;
  `forceRemove` maps to `--force`.

## Engine integration

### Policy extraction (no third copy)

The still-image detect-to-remove policy (V2-to-V1 variant fallback, explicit
override vs trusted-geometry gate vs negative-spatial-NCC abort) currently exists
twice: `process_single_image` (`src/cli/cli_app.cpp:194-309`) and
`batch_processor.cpp:57-131`. The GUI must not become a third copy; this is where
the repo's subtle historical bugs lived (the snap-override leak, the fall-through
leak).

New engine-level unit `src/core/still_remove.{hpp,cpp}`:

```cpp
struct StillRemoveOutcome {
    // detection: bbox, score, geometry source, variant
    // outcome: removed | passthrough | no-watermark | failed
    // error text when failed
};
StillRemoveOutcome remove_still(WatermarkEngine&, const cv::Mat&,
                                const StillRemoveOptions&);
```

CLI, batch, and GUI all call it. The extraction is mechanical (move code, no
behavior redesign), ungated in top-level `SOURCES` and tests `LIB_SOURCES` so it
unit-tests like `still_geometry`. Verified byte-identical on the manual regression
battery: test1/test2 (V1 anchors), test4 (clean 48px), test3 (busy 48px), the
864x1231 3.8 fixture (byte-identity vs its `--geo-preset` output), poster-artnight
plus paintings/ (noWM: zero removal), reference-images/gemini38-images/ (7 of 8
auto-remove), using the git-stash old-binary md5 protocol.

### Progress plumbing: unchanged in v1

Still-image jobs are fast; per-file status in the job state is all the UI needs.
The existing ProgressReporter keeps printing milestone lines to the server's
terminal (the llama-server model). The callback-sink refactor is deferred until a
long-job feature (video, regen) joins the GUI.

### Deferred hazards (documented, not built in v1)

- `offer_leftover_cpu_models_cleanup` blocks on stdin when stdin and stderr are
  TTYs; only reachable via CoreML regen init, so unreachable in v1. Must be
  addressed before regen joins the GUI.
- `main()` calls `std::_Exit` after any run that created an sd_ctx (ggml teardown
  abort). A long-lived GUI server that ever runs regen in-process must respect
  this on shutdown. Not reachable in v1.
- `av_log_set_level(AV_LOG_QUIET)` is process-global in the video path; video is
  not in the v1 GUI.
- The update check sits at the end of `run_cli` and is structurally skipped on the
  GUI path (no-args returns into the server loop before the chokepoint). Fine for
  v1; the GUI does not surface update notices.

## API contract

All paths are token-prefixed; bodies are JSON unless noted.

| Endpoint | Purpose |
|---|---|
| `GET /` | The embedded UI (single page, all assets embedded). |
| `GET /api/version` | `{version, features: {denoise_ai: bool, ...}}` so the UI hides absent options. |
| `POST /api/jobs` | Multipart: image files + an options object. Returns `{job_id}`. |
| `GET /api/jobs/{id}` | Job status + per-file results (bbox, score, source, outcome, error). |
| `POST /api/jobs/{id}/cancel` | Cooperative cancel between files. |
| `GET /api/jobs/{id}/files/{i}/image?kind=cleaned\|original` | `cleaned` downloads with `Content-Disposition: attachment`; `original` serves the upload for the before/after view. |

Options accepted on `POST /api/jobs`, validated with the same rules the CLI uses:

- `denoise`: `off | soft | ns | telea | ai` (default `off`; `ai` rejected unless
  the build reports it)
- `legacy`: bool (default false)
- `geoPreset`: `gemini36-portrait | gemini36-large | gemini38-2k-portrait | null`
- `rect`: `[x, y, w, h] | null`
- `keepProvenance`: bool (default false, i.e. strip, matching the CLI default)
- `forceRemove`: bool (default false; maps to `--force`)

## Frontend

- One page, hand-written vanilla HTML/CSS/JS. No framework, no build step, no CDN,
  no web fonts; anything the page needs ships inside the binary.
- Sources committed at `assets/gui/{index.html, app.js, style.css}`; a dev-time
  script regenerates `assets/embedded_gui_assets.hpp` (constexpr byte arrays, the
  existing embedded-assets pattern); the generated header is committed so CI needs
  no new tooling.
- Layout:
  - Full-window drop zone plus a Browse button; multiple files form one batch.
  - Minimal options row; an Advanced disclosure holds: legacy toggle, geo-preset
    dropdown, rect fields (x, y, w, h), denoise select, keep-provenance checkbox.
  - Results: one row per file with thumbnail, status badge (removed / no watermark
    found / failed), the detected bbox drawn as a client-side overlay, click to
    open a before/after slider, per-file Download button, Cancel while running.
- Styling: `prefers-color-scheme` dark/light, system font stack. This spec fixes
  structure only; the visual pass happens at implementation time.
- Downloads arrive as `<stem>_clean.<ext>` via the browser download manager.

## Build, dependencies, CI

- `WMR_BUILD_GUI`, default ON, following the existing feature-block shape
  (`option` + `target_sources(wmr PRIVATE src/gui/*.cpp)` +
  `target_compile_definitions(wmr PRIVATE WMR_GUI=1)`), placed after the
  `WMR_UPDATE_CHECK` block and before `add_subdirectory(tests)`.
- Vendored header-only deps, plain files under `external/`:
  - `external/httplib/htplib.h`: cpp-httplib, MIT, current release. An independent
    copy, not the sdcpp submodule's 0.28, so the GUI builds with regen OFF.
  - `external/json/json.hpp`: nlohmann/json, MIT, single header.
  - MSVC `/W4` noise handled with a warning-suppressing include wrapper (the
    kissfft `/W1` precedent) if needed.
- `vcpkg.json` unchanged; header-only deps add seconds of compile on every leg.
  An OFF build: no `gui` subcommand, no-args prints help (today's behavior), zero
  GUI symbols (`nm` clean).
- `LICENSE-THIRD-PARTY.md` gains cpp-httplib and nlohmann/json (both MIT).
- Tests: a `target_sources(wmr_tests PRIVATE ...)` mirror block under
  `if(WMR_BUILD_GUI)`. GUI TUs must not pull FFmpeg into `wmr_tests` (the test
  link boundary); still-image-only GUI code links cleanly.

## Testing

- Unit (`[gui]` tag in wmr_tests): job state machine including queue caps and
  cancel; token/Host/`Sec-Fetch-Site` validation; options validation parity with
  the CLI; `still_remove` outcomes on forward-blended fixtures (the
  `add_watermark_alpha_blend` round-trip pattern).
- Integration: the real server on an ephemeral port, exercised with cpp-httplib's
  client from the test binary (same vendored header, no new dependency). Covers
  upload to poll to download, cancel, 404 on bad token, Host rejection.
- Manual (repo convention: `cli_app.cpp` is not ctest-covered): no-args launch
  (TTY and piped), `WMR_NO_GUI=1`, `CI` set, `wmr gui --no-browser`, a full
  browser pass, and the regression battery after the policy extraction.

## Rollout

- Version 1.17.0.
- README gains a "Graphical mode" section: the no-args behavior change,
  `WMR_NO_GUI`/`CI` opt-outs, `wmr gui` flags, and the zero-external-requests
  privacy note.
- CHANGELOG entry states the no-args behavior change prominently.
- Standard release cut protocol (tag, 4-leg build, curated notes).
