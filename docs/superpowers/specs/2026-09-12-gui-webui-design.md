# wmr GUI: embedded local web UI (design spec)

- Date: 2026-09-12
- Status: adversarially reviewed and finalized (v2). All findings from a 4-critic
  review (clarity/feasibility, architecture/data, failure handling,
  security/scalability) plus an adjudication pass are folded in; see the Review
  record at the end.
- Target release: 1.17.0

## Decision

wmr gets a GUI as an embedded local web UI inside the existing single binary.
Running `wmr` with no arguments always launches it (terminal or pipe); an
explicit `wmr gui` subcommand does the same. Opt-outs: `WMR_NO_GUI=1` env or
`CI` set in the environment restore the current help-printing behavior for the
no-args launch only; `wmr --help` always prints help.

The owner's stated preference order was: native feel per platform, then
macOS-only native, then local web UI. A 4-agent research sweep (2026-09-12: repo
integration surface, cross-platform framework landscape, web-UI prior art,
macOS-native path) found:

- Cross-platform native C++ toolkits do not fit this project. Qt has the best
  widgets but multi-hour vcpkg builds per CI leg (a real risk to the Windows
  ~6 h cap) plus LGPL static-linking obligations. wxWidgets is the only
  technical fit (real native controls, minutes of CI, license permits static
  linking) but reads dated on macOS and lacks Windows dark mode until 3.4.0.
  Slint (Fluent styling everywhere since 1.16, plus a Rust toolchain in CI),
  Flutter Desktop (a Dart app wrapped around the engine, no Rosetta
  cross-compile story), GTK4 (vcpkg port is shared-only, gtkmm has no port),
  and Dear ImGui (presents as a dev tool) are all rejected.
- The macOS-native pattern is a SwiftUI app spawning the embedded CLI as a
  subprocess, about 11 to 15 working days, mac-only, plus a new Xcode CI leg
  and a Swift codebase to maintain. Direct Swift-C++ interop is a poor fit for
  this codebase (cv::Exception cannot be caught in Swift; spdlog/fmt templates
  defeat the clang importer; dual CMake+Xcode builds forever). Ollama,
  HandBrake, and whisper.cpp all use process or C-API boundaries instead.
- The embedded local web UI reaches every platform in the same binary at
  near-zero CI cost, and the exact reference implementation is already vendored
  in-tree: `external/stable-diffusion.cpp/examples/server/` is a cpp-httplib
  server with an embedded web UI and an async job manager. llama.cpp's built-in
  web UI (same pattern, same HTTP library) is well received in practice.

The owner chose the web UI, with no commitment to a later native shell. The
local API is nevertheless designed as a clean contract so a WKWebView or
SwiftUI wrapper could be added later without redesign (the Ollama
architecture), but that is not on any roadmap.

## Scope (v1)

The core still-image flow, for mainly non-technical users:

- Drag/drop or browse to pick 1..N images (multiple files form one batch job);
  removal starts automatically on submit.
- The results view shows, per file, the detected watermark bbox (a client-side
  overlay on the before/after view), score, geometry source, and outcome after
  processing.
- Before/after comparison view (slider).
- Per-file download of the cleaned image.

Two deliberate simplifications, recorded so they are not "fixed" by accident:

- **No pre-removal detection step in v1.** The bbox overlay is a post-hoc
  result annotation, not a preview-then-confirm flow. This is safe because a
  file with no watermark is never modified. A detect-only job mode is a natural
  later addition, not v1.
- **No per-row thumbnails.** Result rows show filename + status badge +
  actions; clicking a row opens the before/after view. Server-side thumbnail
  generation stays deferred, and the UI must NOT hot-load originals as row
  previews (a 100-file job would decode the whole batch in the tab).

Out of scope for v1 (stay CLI-only): video, SynthID regen, metadata
inspect/strip, cache management. Also deferred: SSE live progress, ZIP
download-all, thumbnails, per-job log capture, remote bind (`--gui-host`),
TLS, streaming multipart parsing.

## Architecture

One process, three thread roles:

1. cpp-httplib's worker pool serves HTTP.
2. Exactly one job worker thread runs all engine work, serialized. This is
   forced by the codebase: NcnnDenoiser, Regenerator, and MiganInpainter are
   documented not-thread-safe singletons, and the CoreML e5rt cache has no
   concurrency lock. A batch of N images is one job that loops files
   sequentially, the same shape as `batch_processor` today.
3. The worker updates a mutex-guarded job-state struct that HTTP threads read
   (polling at ~500 ms). No SSE/WebSocket in v1. The page recovers its state
   after a reload through the job-list endpoint, not client persistence.

### Entry points

- `wmr gui`: starts the server. Flags: `--gui-port N` (default: ephemeral),
  `--no-browser`. Binds 127.0.0.1 only, always.
- No-args (`argc <= 1`, currently the help-print early return at
  `src/cli/cli_app.cpp:765`): always launches the GUI, interactive or not.
  `WMR_NO_GUI=1` or `CI` present in the environment prints help instead (the
  `CI` gate mirrors the update-check convention). These gates apply ONLY to
  the no-args launch; an explicit `wmr gui` always launches.
- `WMR_BUILD_GUI=OFF` builds: `wmr gui` prints "wmr gui: this build is
  GUI-free (WMR_BUILD_GUI off). Rebuild with WMR_BUILD_GUI=1." and exits 2
  (the regen-free precedent, `batch_processor.cpp:166-169`); no-args prints
  help.
- `wmr --help`, `wmr --version`, and parse-error paths are unchanged.
- Known trade-off: under CLI11 subcommand resolution a file literally named
  `gui` must be passed as `wmr remove gui`. Acceptable; documented.
- **Bind failure** on a pinned `--gui-port`: print to stderr "port N in use;
  choose another or omit --gui-port for an ephemeral one", exit 1, no browser
  launch. An ephemeral bind failure is not retried (same error path).
- **Shutdown.** SIGINT/SIGTERM trigger a graceful stop: the server stops
  accepting, all running jobs get their cancel flag set, the main thread joins
  the worker with a 5 s bound (still-image files finish in seconds), then
  removes the run directory and exits 0. A second Ctrl-C, or the 5 s timeout,
  exits immediately without cleanup; leftovers are removed by the next start's
  dead-run rule. The banner states "Ctrl-C to stop (a running job finishes its
  current file first)".
- **Browser launch** happens only after `listen()` succeeds (avoids the
  Syncthing connection-refused race) and is non-fatal on failure (the printed
  URL is the fallback). Matrix: macOS `open`; Linux `xdg-open` gated on
  `DISPLAY`/`WAYLAND_DISPLAY`; Windows `ShellExecuteW` (link `shell32`); WSL
  `wslview` then `cmd.exe /c start`; `BROWSER` env override honored.
- **Console banner** (stdout): version, the tokenized URL, the Ctrl-C hint,
  and `wmr --help for CLI usage`.

### Routing and security

Threat model: another website in the user's browser issuing requests at the
local port (CSRF, DNS rebinding), and oversized/malformed uploads. All rules
below are mandatory.

**Token grammar.** The server answers ONLY paths matching
`^/<token>(/.*)?$`; every other path, including `/`, `/favicon.ico`, and
un-prefixed `/api/...`, returns 404 with an empty body. The token is a
per-run 128-bit value from the system CSPRNG, rendered as 32 hex chars,
regenerated on every server start. The banner prints
`http://127.0.0.1:<port>/<token>/`. The UI is served at `GET /<token>/`
(plus `/<token>/app.js` and `/<token>/style.css`) and issues same-origin
relative fetches to `/<token>/api/...`. The token never appears in a query
string and is never written to any log.

**Authentication before buffering.** The token, Host, and Sec-Fetch-Site
checks are installed via `set_pre_routing_handler`, which cpp-httplib runs
before it reads (and RAM-buffers) any request body. An unauthenticated or
malformed request is refused before a byte of its body is read.

- Host validation: reject any request whose Host header is absent, empty, or
  not exactly one of `{127.0.0.1, localhost, [::1]}` (after lowercasing and
  stripping one trailing dot) concatenated with the actual bound port. This is
  the primary DNS-rebinding defense.
- Reject `Sec-Fetch-Site: cross-site` when present (403).
- Never emit CORS headers (cross-origin reads stay browser-blocked).

**Upload and resource caps.** cpp-httplib buffers the entire request body in
memory (req.body plus parsed copies, ~2x transient RSS), so the request cap is
a RAM budget:

- `set_payload_max_length(1 GiB)`: a Content-Length above it is answered 413
  before any body byte is read; chunked bodies abort mid-stream at the same
  bound.
- Per-file cap exactly 200 MiB (209,715,200 bytes), enforced at multipart
  parse; violation returns 413 naming the offending files.
- Per job: at most 100 files AND at most 1 GiB total. The UI splits a larger
  drop into multiple jobs (one POST each, shown as separate jobs).
- At most 8 pending jobs; further POSTs get 429 (checked under the job-map
  mutex).
- Aggregate bound: total bytes stored under `~/.cache/wmr/gui` across all
  live runs must stay <= 4 GB; a POST that would exceed it is rejected 413.
  Finished jobs are not evicted; the bound gates new uploads only.

**Decode gate (decompression bombs).** The magic-byte sniff also extracts
dimensions (PNG IHDR at fixed offset; JPEG SOFn marker walk; WebP
VP8/VP8L/VP8X) and fails the file with a per-file `failed` outcome, error
"image too large (WxH)", when `width*height > 120,000,000` px or either side
exceeds 16,384 px. The check runs before any `cv::imdecode` allocation. This
bounds the decode itself, the full-image median-21 in the still-geometry
search path, and the browser-side before/after decode of two full bitmaps.
(The NCC detector needs no bound: it crops a mark-sized ROI before its float
work.)

**Accepted formats.** Decided by magic sniff only, never by extension: PNG,
JPEG, WebP (parity with `batch_processor.cpp:20-24`). Anything else (HEIC,
PDF, 0-byte entries, dropped-folder entries) becomes a per-file `failed`
outcome with error "unsupported format: <filename> (use PNG, JPEG, or WebP)";
one bad file does not sink the POST. A POST with zero sniff-valid files
returns 400 "no supported images" and creates no job.

**Storage naming (client filenames are untrusted).** `job_id` is
server-generated, 12 chars `[a-z0-9]` from the CSPRNG, never derived from
user input. Stored artifacts use index-based names:
`<run_dir>/<job_id>/orig/<i>.<ext>` and `<run_dir>/<job_id>/out/<i>_clean.<ext>`,
with `<ext>` from the server-side sniff. The client filename is metadata
only: echoed in JSON and in `Content-Disposition`, sanitized (basename only;
control chars, quotes, and `\/:?*"<>|` stripped; Windows reserved stems
suffixed; length capped at 120 chars; fallback `image<N>`), encoded as
`Content-Disposition: attachment; filename*=UTF-8''<percent-encoded-
stem>_clean.<ext>` (RFC 5987). A same-name collision inside one job
disambiguates as `<stem>_2_clean.<ext>`. Path parameters `{id}` and `{i}`
resolve only through the in-memory job map (`{i}` parsed as an integer, 404
out of range); no string from a request ever builds a filesystem path.

**Served content discipline.** Every response carries
`X-Content-Type-Options: nosniff`. Image responses get a Content-Type derived
only from the server-side magic sniff, never from the client multipart
Content-Type or filename. The HTML response carries
`Content-Security-Policy: default-src 'none'; img-src 'self'; script-src
'self'; style-src 'self'`, which technically enforces the zero-external-
requests invariant. The three UI sources are served as three separate
resources (no build-time inlining into one document) precisely so this CSP
stays enforceable. The frontend inserts all server-supplied strings
(filenames, error text) via `textContent`, never `innerHTML`.

**Cache directory hygiene.** GUI run directories are created with mode 0700
(chmod'd 0700 if they pre-exist with looser bits) before any upload is
written. See the job model for lifecycle.

### Job model

**Per-run directories (concurrent instances coexist).** Each server run owns
`~/.cache/wmr/gui/<run_id>/` (run_id = timestamp-pid) containing a sidecar
pid file. On start, a run removes only sibling run directories whose recorded
PID is no longer alive (`kill(pid, 0)`/ESRCH on POSIX, `OpenProcess` with
`PROCESS_QUERY_LIMITED_INFORMATION` on Windows), never a live server's. Its
own directory is removed on graceful shutdown (see Entry points). The shared
`gui/` root is never wiped wholesale. There is no single-instance lock; with
ephemeral ports, two instances coexist harmlessly, each self-contained.

**Job and file states.** Job states: `queued | running | done | failed |
cancelled`. Job-level semantics: `done` when the worker finishes the loop
regardless of per-file failures; `failed` only for job-level errors (storage,
queue); per-file failures never fail the job. Per-file outcomes:

| Outcome | Meaning | Artifacts | UI |
|---|---|---|---|
| `removed` | Mark found and removed, or `forceRemove` ran | orig + cleaned | Download enabled; badge shows "removed (forced)" when forced |
| `no-watermark` | No detection after all variant attempts, not forced | orig only | Download disabled; before/after hidden; `kind=cleaned` returns 404 |
| `failed` | Decode error, unsupported format, oversize, rect out of bounds, engine or write failure | orig only | Error text shown |
| `not-run` | File never reached because the job was cancelled | none | Terminal rows of a cancelled job |

`passthrough` does not exist in v1 (no CLI condition produces it; the earlier
draft's rc-2 anchor was dead code, see Review record). `bbox`, `score`, and
`geometry_source` are null for `no-watermark`, `failed`, and `not-run` files.

**Worker isolation.** The worker wraps each file in try/catch(...): any throw
(cv::Exception, bad_alloc) or a false write sets that file's outcome to
`failed` with the exception text and the batch continues (mirroring
`batch_processor.cpp:272-277` and `:204-207`). The job loop itself is
additionally wrapped in try/catch(...): an escape marks the whole job
`failed` and the worker survives for the next job.

**Cancel transition table.**

| Job state | On `POST .../cancel` |
|---|---|
| `queued` | Leaves the queue; state becomes `cancelled` immediately; no files run |
| `running` | Sets the cancel flag under the manager mutex; the worker checks between files, stops, sets `cancelled`; already-completed files keep their outcomes and stay downloadable |
| `done`/`failed`/`cancelled` | 409 Conflict with the current state in the error body |
| unknown id | 404, same as any unknown id |

Terminal-state writes (by the worker or by cancel) are mutually exclusive
under the same mutex: first terminal state wins, so a cancel arriving as the
last file completes may legitimately return `done`.

**Upload persistence.** Accepted files are written to
`<job_dir>/orig/<i>.<ext>` before the job is queued; `kind=original|cleaned`
always streams from disk, so originals are never held resident for the server
lifetime.

**Job discovery.** `GET /api/jobs` lists this run's jobs; the UI calls it on
page load and re-attaches to running/recent jobs (this is the reload-recovery
mechanism). No client-side persistence in v1.

## Engine integration

### Policy extraction (no third copy)

The still-image detect-to-remove policy (V2-to-V1 variant fallback, explicit
override vs trusted-geometry gate vs negative-spatial-NCC abort) currently
exists twice: `process_single_image` (`src/cli/cli_app.cpp:194-309`) and
`batch_processor.cpp:57-131`. The GUI must not become a third copy; this is
where the repo's subtle historical bugs lived (the snap-override leak, the
fall-through leak).

New engine-level unit `src/core/still_remove.{hpp,cpp}`:

```cpp
struct StillRemoveOutcome {
    // detection: bbox, score, geometry_source, variant (null when no detection ran)
    // outcome: removed | no-watermark | failed
    // error text when failed
};
StillRemoveOutcome remove_still(WatermarkEngine&, cv::Mat& image,
                                const StillRemoveOptions&);
bool write_still_output(const std::filesystem::path&, const cv::Mat&,
                        bool keep_provenance);
```

- **In-place signature, engine convention.** Removal mutates `image` in place
  like every engine API (`watermark_engine.hpp:27`, `:48`). The caller clones
  the original before calling; that clone is what `kind=original` serves and
  what the before/after view renders. `StillRemoveOutcome` never carries
  pixels.
- **The four option-resolution helpers move with the policy.**
  `parse_rect`, `resolve_still_variant`, `resolve_inpaint_config`, and
  `resolve_still_geometry_override` (currently `cli_app.hpp:115-179`,
  defined in `cli_app.cpp`) are rewritten to take `StillRemoveOptions`
  instead of `CliOptions` and live in `still_remove`. `cli_app.cpp` and
  `batch_processor.cpp` keep only CLI11 glue that maps `CliOptions` to
  `StillRemoveOptions`. Without this move the extracted TU cannot link in
  `wmr_tests` (the test exe compiles neither `cli_app.cpp` nor
  `batch_processor.cpp`).
- **The write tail moves too.** `write_still_output` implements the shared
  write policy: codec parameters by extension (JPEG quality 100, PNG
  compression 6, WebP quality 101), then `post_write_provenance_strip`
  unless `keep_provenance` (the write tail currently exists twice:
  `cli_app.cpp:420-442`, `batch_processor.cpp:193-217`; the GUI worker would
  be a third). CLI, batch, and the GUI worker all call it; no caller writes
  with `cv::imwrite` directly. The output extension always equals the input's
  (as sniffed by the server).
- **Defaults are the CLI's, verbatim.** `StillRemoveOptions` defaults mirror
  the CLI: denoise off; strength 120, radius 10 for the cleanup methods (the
  `resolve_inpaint_config` mapping, `cli_app.cpp:63-79`). The GUI never
  overrides them in v1.
- **Decode contract.** `remove_still` requires an 8-bit BGR Mat; the GUI
  decodes persisted uploads with `cv::imdecode(bytes, cv::IMREAD_COLOR)`,
  behavior-identical to the CLI's `imread` at `cli_app.cpp:182` (16-bit,
  CMYK, and alpha inputs convert the same way). Parity is covered by AC7.
- Placement: ungated in top-level `SOURCES` and tests `LIB_SOURCES` (the
  `paths.cpp` precedent), so it unit-tests like `still_geometry`.
- The extraction is verified byte-identical on the manual regression battery:
  test1/test2 (V1 anchors), test4 (clean 48px), test3 (busy 48px), the
  864x1231 3.8 fixture (byte-identity vs its `--geo-preset` output),
  poster-artnight plus paintings/ (noWM: zero removal),
  reference-images/gemini38-images/ (7 of 8 auto-remove), using the git-stash
  old-binary md5 protocol.

### Progress plumbing: unchanged in v1

Still-image jobs are fast; per-file status in the job state is all the UI
needs. The existing ProgressReporter keeps printing milestone lines to the
server's terminal (the llama-server model). The callback-sink refactor is
deferred until a long-job feature (video, regen) joins the GUI.

### Deferred hazards (documented, not built in v1)

- `offer_leftover_cpu_models_cleanup` blocks on stdin when stdin and stderr
  are TTYs; only reachable via CoreML regen init, so unreachable in v1. Must
  be addressed before regen joins the GUI.
- `main()` calls `std::_Exit` after any run that created an sd_ctx (ggml
  teardown abort). A long-lived GUI server that ever runs regen in-process
  must respect this on shutdown. Not reachable in v1.
- `av_log_set_level(AV_LOG_QUIET)` is process-global in the video path; video
  is not in the v1 GUI.
- **Update check:** both GUI entry points (no-args and the `gui` subcommand)
  return from `run_cli` BEFORE the `WMR_UPDATE_CHECK` chokepoint: the no-args
  branch returns before CLI11 parse, and the `gui` dispatch returns its rc
  directly rather than falling through. No network fetch ever runs at server
  shutdown; the GUI does not surface update notices.

## API contract

All paths are token-prefixed per the token grammar (`/<token>/api/...`);
bodies are JSON unless noted.

| Endpoint | Purpose |
|---|---|
| `GET /<token>/` | The embedded UI (plus `/app.js`, `/style.css`; three separate resources). |
| `GET /<token>/api/version` | `{version, features: {denoise_ai: bool}, presets: [...]}`; `presets` is generated from `kStillPresetNames` (single source); an absent feature key means false; unknown keys are ignored. |
| `GET /<token>/api/jobs` | This run's jobs: `[{job_id, status, created, file_count}]`, newest first. The UI calls it on load and re-attaches. |
| `POST /<token>/api/jobs` | Multipart: image files + one `options` part (JSON). Returns `{job_id}`. |
| `GET /<token>/api/jobs/{id}` | Job status + per-file results (below). |
| `POST /<token>/api/jobs/{id}/cancel` | Per the cancel transition table. |
| `GET /<token>/api/jobs/{id}/files/{i}/image?kind=cleaned\|original` | `cleaned` downloads with `Content-Disposition: attachment`; `original` serves the persisted upload. Both stream from disk. |

**Response shapes.** Every non-2xx JSON response uses one envelope:
`{"error": {"code": "<stable_snake_case>", "message": "<human text>"}}` with
codes `invalid_option`, `invalid_combination`, `payload_too_large`,
`too_many_pending`, `unknown_job`, `job_already_finished`.
`GET /api/jobs/{id}` returns `{job_id, status, created_at, queue_position,
current_file_index, files: [{index, name, outcome, bbox, score,
geometry_source, variant, error}]}` where `bbox`, `score`, and
`geometry_source` are null until that file has run. `geometry_source` is one
of `rect | preset | auto/snapped | auto/raw | model` (the engine's own
values, `still_geometry.hpp:104`, `watermark_engine.hpp:107`), null when no
detection ran.

**Status codes.** 400 invalid options/combos; 404 unknown id, bad token, any
un-prefixed path; 403 cross-site fetch; 409 cancel on a terminal job; 413
oversize (request, file, or aggregate); 415 wrong Content-Type; 429 queue
cap.

**Options and validation.** Options part: `denoise` (`off | soft | ns |
telea | ai`; `ai` is 400 `invalid_option` when `/api/version` reports
`features.denoise_ai == false`), `legacy` (bool), `geoPreset` (must be in
`kStillPresetNames`, validated server-side against that table, never a
string list in the handler), `rect` (`[x, y, w, h]`, four integers, `x, y >=
0`, `w, h >= 8`), `keepProvenance` (bool, default false = strip), and
`forceRemove` (bool, default false). The options part is rejected 400 above
64 KB.

**Invalid combinations (400 `invalid_combination`).** The engine silently
ignores some pairs (verified: geometry resolution is skipped under force,
`cli_app.cpp:207`, and the V1 attempt receives no override, `:294-295`;
CLAUDE.md documents `--legacy --rect` as a silent no-op). The CLI stays
silent for backward compatibility; the API must not:

- `legacy` + `rect` or `geoPreset`: rejected, message "rect/geoPreset apply
  to the V2 profile only; legacy (V1) uses fixed positions".
- `forceRemove` + `rect` or `geoPreset`: rejected.
- `rect` + `geoPreset`: rejected (the CLI's `rect > preset` precedence stays
  CLI-only; the UI exposes one at a time).
- `legacy` + `forceRemove` is valid (the documented `--force --legacy`
  combo).

The UI disables the rect and geo-preset inputs while legacy or forceRemove
is checked. **Per-file bounds check** (in the worker, since files in one
batch differ in size): a rect is usable only if `x+w <= W` and `y+h <= H`
for that image; otherwise the file's outcome is `failed` with error "rect
WxH at (x,y) exceeds image WxH". Deliberately stricter than CLI parity.

**Content types.** `POST /api/jobs` requires `multipart/form-data`; every
other body-bearing endpoint requires `application/json`; violations return
415.

## Frontend

- One page, hand-written vanilla HTML/CSS/JS. No framework, no build step,
  no CDN, no web fonts; anything the page needs ships inside the binary.
- Sources committed at `assets/gui/{index.html, app.js, style.css}`,
  embedded as three byte arrays served at `/<token>/`, `/<token>/app.js`,
  `/<token>/style.css` (no build-time inlining; keeps the CSP enforceable).
  A deterministic dev-time script regenerates
  `assets/embedded_gui_assets.hpp` (the existing constexpr-array pattern);
  the generated header is committed so CI needs no new tooling. A `[gui]`
  unit test asserts each committed embedded byte array equals the
  corresponding `assets/gui` file on disk, failing the suite on drift.
- Layout:
  - Full-window drop zone plus a Browse button; multiple files form one
    batch (larger than the caps: the UI splits into multiple jobs and shows
    them separately).
  - Minimal options row; an Advanced disclosure holds: legacy toggle,
    geo-preset dropdown (populated from `/api/version` `presets`), rect
    fields (x, y, w, h), denoise select (`ai` only offered when available),
    keep-provenance checkbox. Rect/preset inputs disable while legacy or
    forceRemove is checked.
  - Results: one row per file with filename, status badge, and actions; NO
    thumbnails; click opens the before/after slider view with the bbox
    overlay; per-file Download button; Cancel while running.
- **Offline rule (covers both Ctrl-C and restart).** Any failed fetch, or
  any 404 on a token-prefixed `/api/*` call while the page holds job state,
  shows a persistent banner: "wmr has stopped or was restarted; this page
  can no longer reach it and previous results are gone. Restart wmr and
  open the new URL it prints." Cancel and Download are disabled; the banner
  never auto-clears and never auto-reloads (the stale token also 404s on
  `/api/version`, so auto-recovery is impossible). Results already in hand
  stay viewable; in-flight jobs do not.
- **Reload recovery.** On load, the UI calls `GET /api/jobs` and re-attaches
  to running/recent jobs; an unknown job id renders the empty drop zone.
- Styling: `prefers-color-scheme` dark/light, system font stack. Structure
  is fixed by this spec; the visual pass happens at implementation time.
- Downloads arrive as `<stem>_clean.<ext>` via the browser download manager.

## Build, dependencies, CI

- `WMR_BUILD_GUI`, default ON, following the existing feature-block shape
  (`option` + `target_sources(wmr PRIVATE src/gui/*.cpp)` +
  `target_compile_definitions(wmr PRIVATE WMR_GUI=1)`), placed after the
  `WMR_UPDATE_CHECK` block and before `add_subdirectory(tests)`. On WIN32
  the block links `shell32` (ShellExecuteW).
- Vendored header-only deps, plain files under `external/`:
  - `external/httplib/httplib.h`: cpp-httplib, MIT, current release. An
    independent copy, not the sdcpp submodule's 0.28, so the GUI builds with
    regen OFF.
  - `external/json/json.hpp`: nlohmann/json, MIT, single header.
  - MSVC `/W4` noise handled with a warning-suppressing include wrapper (the
    kissfft `/W1` precedent) if needed.
- `vcpkg.json` unchanged; header-only deps add seconds of compile on every
  leg. An OFF build: `wmr gui` prints the GUI-free message and exits 2,
  no-args prints help, zero GUI symbols (`nm` clean).
- Include discipline: `src/gui` TUs include `still_remove.hpp`, engine and
  standard headers only, never `cli_app.hpp` (keeps `cli_app.cpp` out of the
  test link); the still-image-only GUI links without FFmpeg.
- `LICENSE-THIRD-PARTY.md` gains cpp-httplib and nlohmann/json (both MIT).
- Tests: a `target_sources(wmr_tests PRIVATE ...)` mirror block under
  `if(WMR_BUILD_GUI)`.

## Testing

- Unit (`[gui]` tag in wmr_tests): job state machine including queue caps,
  cancel transitions, and the first-terminal-wins rule; token/Host/
  `Sec-Fetch-Site` validation; options and combination validation parity
  with this spec; the dimension gate (a crafted tiny PNG with a huge IHDR);
  the embedded-asset drift guard; `still_remove` outcomes on
  forward-blended fixtures (the `add_watermark_alpha_blend` round-trip
  pattern); `write_still_output` codec/strip behavior.
- Integration (live server on an ephemeral port, exercised with
  cpp-httplib's client from the test binary; no new dependency): upload to
  poll to download; per-file incremental results; the outcome table's
  artifact rules (`kind=cleaned` 404 for no-watermark); the cancel table;
  404 on bad token and un-prefixed paths; Host rejection; 413 before body
  read; 429 at the queue cap; 400 invalid combinations.
- Manual (repo convention: `cli_app.cpp` is not ctest-covered): the AC1/AC2/
  AC9 launch matrix, a full browser pass, and the regression battery after
  the policy extraction.

## Acceptance criteria

Each AC names its verification method.

- **AC1** (manual): `wmr` with no args starts the server, prints the
  tokenized URL banner, and opens the browser at it.
- **AC2** (manual, byte-diff): `WMR_NO_GUI=1 wmr` and `CI=1 wmr` print
  today's header + help and exit 0, byte-identical to the pre-GUI build.
- **AC3** (integration): un-prefixed paths, bad tokens, `/`, and
  `/favicon.ico` return 404 with an empty body; a bad Host header returns
  404; `Sec-Fetch-Site: cross-site` returns 403; no response ever carries a
  CORS header.
- **AC4** (integration): a Content-Length above the request cap gets 413
  before any body byte is read; a 9th pending job gets 429.
- **AC5** (integration): a multi-file job surfaces per-file results
  incrementally; every outcome row follows the outcome table; cleaned
  downloads decode and arrive as `<stem>_clean.<ext>` with
  `Content-Disposition: attachment`.
- **AC6** (integration): cancel behaves per the transition table (queued:
  immediate cancelled; running: stops between files, completed files keep
  outcomes and downloads; terminal: 409; unknown id: 404).
- **AC7** (integration): the same input + options through the API and
  through `wmr remove` produce pixel-identical cleaned outputs; invalid
  values and invalid combinations return 400.
- **AC8** (manual protocol): the `still_remove` extraction passes the
  regression battery byte-identity check against the pre-extraction binary.
- **AC9** (manual): an OFF build prints help on no-args, shows zero GUI
  symbols in `nm`, leaves `vcpkg.json` untouched, and `wmr gui` exits 2
  with the GUI-free message.
- **AC10** (integration + manual): a page reload recovers job state from
  `GET /api/jobs`; an unknown job id shows the empty drop zone; the offline
  banner appears when the server is stopped and never auto-clears.

## Rollout

- Version 1.17.0.
- README gains a "Graphical mode" section: the no-args behavior change,
  `WMR_NO_GUI`/`CI` opt-outs, `wmr gui` flags, the zero-external-requests
  privacy note, and where session files live (`~/.cache/wmr/gui`, 0700,
  removed at next start).
- CHANGELOG entry states the no-args behavior change prominently.
- Standard release cut protocol (tag, 4-leg build, curated notes).

## Review record

This spec was adversarially reviewed on 2026-09-12 by four critic agents
(clarity/feasibility/completeness, architecture/data design, failure
handling, security/scalability) with a fifth adjudicator verifying every
finding against the codebase before adoption: 45 raw findings, 36 adopted as
proposed, 9 adopted with corrected minimal resolutions, 1 withdrawn, plus 5
gaps the critics all missed. Notable corrections folded in: the original
draft's rc-2 anchoring was dead code (live-verified: no-watermark returns
rc 0 and writes nothing in single-image mode); the draft's caps permitted a
~20 GB in-RAM request body (cpp-httplib buffers whole bodies; now bounded at
1 GiB with pre-routing auth); the draft's wipe-on-start destroyed concurrent
instances' live jobs (now per-run directories with dead-pid cleanup); the
token grammar contradicted the endpoint table (now one exact rule); the
draft would have imported the CLI's silent `--legacy --rect` no-op into the
API (now 400); and the extraction as originally scoped could not link in
`wmr_tests` (the four option helpers now move with the policy).
