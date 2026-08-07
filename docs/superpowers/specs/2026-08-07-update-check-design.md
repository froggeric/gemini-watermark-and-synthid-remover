# Update Check (notify-only): Design Spec

**Date:** 2026-08-07
**Status:** Approved (brainstormed 2026-08-07); revised after principal-engineer
review; ready for implementation planning.
**Owner:** Frédéric Guigand

## Context

wmr ships as GitHub Release binaries for macOS (arm64, x86_64), Linux, and
Windows, plus a from-source Homebrew path. Most users install by downloading the
release archive, so they have no package manager pushing updates at them. A user
can run a stale `wmr` for a long time without knowing a newer release exists.

This spec adds an **automatic update check**: on a real subcommand, wmr compares
its compiled-in version against the latest GitHub release and prints a short,
styled notice to stderr when a newer release is available. It is **notify-only**:
wmr never downloads a binary and never runs fetched code. Telling the user is the
entire feature.

The central tension is privacy. wmr's whole value proposition is removing tracking
watermarks, so any "phone home" is sensitive. A visible-watermark-only user
currently makes **zero** network calls; this check would be the first. The
research (gh CLI's April 2026 default-on-telemetry backlash; the Console Do Not
Track debate; what ripgrep / fd / Bitwarden CLI do) sets a higher bar for a
privacy-marked tool: the check must be **zero-payload**, **documented**, **trivial
to disable**, and **suppressed in scripts and CI**. The design below holds that
line.

### Research basis

Two research sweeps (industry update UX patterns; security, privacy, and
infrastructure) grounded every decision. Key sources are listed at the end. The
decisions track the 2024-2026 consensus: default-on update check with a 24-hour
cache and a one-line opt-out; default-OFF telemetry. The bright line between a
version-only check and payload telemetry is held strictly.

## Scope

**In scope:**

- Compare `APP_VERSION` to the latest release tag on startup; print a styled,
  three-line notice to stderr when newer.
- Default-on, with a 24-hour throttle cache and trivial opt-out (flag, env,
  auto-suppression in CI / non-TTY / pipes).
- A dedicated `WMR_UPDATE_CHECK` CMake option (default ON), curl-backed, fully
  `#ifdef`-guarded so an OFF build has zero update-check symbols.

**Out of scope (explicit non-goals):**

- **Self-update.** No `wmr update`, no binary download, no in-place swap. (Most
  users installed from an archive with an unknown location; in-place replacement
  has real platform pitfalls: Windows locks the running exe, macOS loses its
  notarization staple. And notify-only is strictly safer per supply-chain best
  practice: xz-utils, event-stream, the `curl | bash` timing attack.)
- **Signature verification of the manifest.** The MVP reads a version string over
  HTTPS; no binary is downloaded and no code runs. SHA256 / minisign / cosign /
  GitHub Artifact Attestations become mandatory only when a future self-update
  downloads a binary. For reading a version number they are out of scope.
- **Telemetry of any kind.** No version, OS, arch, unique id, or usage data is
  ever transmitted.
- **A static manifest on HuggingFace / GitHub Pages.** The GitHub Releases API is
  used; a static `latest.json` is the fallback only if rate limits ever bite.
- **A background / async fetch.** The check is synchronous. The cache-plus-
  background split (npm / Deno "lag by one invocation") is a future refinement if
  the once-a-day cost ever matters.
- **An on-demand `wmr version --check`.** The `version` subcommand stays
  read-only (gh / Deno / pip convention).

## Locked decisions

| Decision | Choice | Why |
|---|---|---|
| Scope | Notify-only | Safest; smallest surface; no binary-swap pitfalls. |
| Default stance | Default-on, opt-out | Matches 2024-2026 consensus; holds the version-only line. |
| Execution | Synchronous, throttled, GitHub Releases API | Simplest (no threads, no `_Exit` hazard, no new infra, no release-time sync). pip and Homebrew do this. |
| Placement | End of `run_cli`, after the subcommand's output, to stderr | The user's real result is the priority; convention (gh, npm, Deno, pip all print last); the once-a-day cost lands after the result is delivered. |
| Wording | Three lines (news, download link, disable hint) | gh-style news plus the one action plus the trust/opt-out line. |
| Styling | `----` dash rules (matching `print_header`) plus bold plus yellow, gated on terminal support | Consistent with the existing header idiom; three independent visibility layers (rules, bold, color), each degrading gracefully. |
| Build | `option(WMR_UPDATE_CHECK ... ON)`, links `CURL::libcurl`, `#ifdef`-guarded | Decoupled from regen; rides every release build (curl already present there); a dev can turn it off. |

## Architecture

### Where it hooks (precise control-flow placement)

**Verified against `src/cli/cli_app.cpp`.** `run_cli()` does **not** have a single
common tail today: every subcommand case `return`s directly from the `switch` at
the bottom of `run_cli` (Detect / Video / AutoRemove / VisibleOnly / SynthidOnly),
and the trailing `return 0;` after the switch is unreachable. In addition, the
`--version`, `--help`, subcommand-`--help`, and any CLI parse error are **not**
clean early returns: CLI11 throws a `CLI::ParseError` (or subclass) which is
caught by the `catch (const CLI::ParseError& e)` block, which then returns. Only
the no-args path (`argc <= 1`) returns cleanly before parse.

Therefore the implementation **must restructure the bottom of `run_cli`** so there
is exactly one post-dispatch chokepoint, instead of adding a call to every return
site. Concretely:

1. Capture the dispatched subcommand's return code into a local `int rc`
   (currently each `case` returns directly). The `switch` sets `rc` instead of
   returning; the surrounding `try { ... } catch (const std::exception& e) { ... }`
   sets `rc = 1` instead of returning 1.
2. After the try/catch, on the **dispatched path only**, call the update check:
   ```cpp
   #ifdef WMR_UPDATE_CHECK
       wmr::maybe_check_for_update(opts);  // void; never throws; stderr-only
   #endif
       return rc;
   ```
3. The chokepoint is reached **only when a subcommand was parsed and dispatched**.
   It is **not** reached on the no-args path (early `return 0;` before parse), nor
   on the `CLI::ParseError` path (the `catch (const CLI::ParseError&)` block
   returns directly and must not fall through to the check). So `--version`,
   `--help`, subcommand `--help`, missing-required-arg, and bad-flag invocations
   all skip the check by construction (a help/discovery or error flow does no
   work, so it must not phone home).

This both satisfies the locked "end of `run_cli`, after the subcommand's output"
decision and closes the "does it run on `--help` / parse errors?" ambiguity:
no, it does not. It does run after a subcommand that itself failed (returned
non-zero or threw `std::exception`); gh and npm behave the same way and the
result was already printed, so the once-a-day notice is acceptable there.

Because the notice prints inside `run_cli` (before it returns), it is complete
and flushed before `main()` can take the regen `std::_Exit` path
(`src/main.cpp`). `maybe_check_for_update` ends with an explicit
`std::fflush(stderr)` (or `std::cerr.flush()`) so the bytes are out of the
process buffer before any `_Exit`; the regen branch's own `std::fflush(nullptr)`
in `main` is regen-only and is not relied upon. Verified: the `_Exit` path fires
only when `regenerator_was_used()` is true, and the update check does not touch
the regenerator, so the gate stays accurate.

### The gate (when the check runs at all)

`should_show(...)` is a **pure** predicate (truth-table unit-tested, in the
spirit of `notebooklm_gates` / `decide_auto_geometry`). The notice subsystem
engages only when **all** hold:

- `WMR_UPDATE_CHECK` is ON at compile time.
- `opts.no_update_check` is false (the `--no-update-check` flag, added to
  `CliOptions` and bound via the shared `add_common` lambda so it is present on
  every subcommand).
- `WMR_NO_UPDATE_CHECK` env is unset or empty.
- `CI` env is unset or empty (good-citizen CI suppression).
- `DO_NOT_TRACK` env is not the string `"1"` (good-citizen; honors the Console
  Do Not Track convention).
- stderr is a TTY (`isatty(STDERR_FILENO)`; on Windows use `_isatty(_fileno(stderr))`).

If any condition fails, `maybe_check_for_update` returns immediately and no
network is touched.

### The cache-age gate (when the network fetch runs)

On an eligible run:

1. Read the cache. If it records a `latest_version` strictly newer than
   `APP_VERSION`, **print the notice from the cache with no network** (gh shows
   the banner whenever a newer version is known, not only on the run that fetched
   it). Set a `printed` flag.
2. Only if the cache is absent or its age is at least the interval (default
   86400 s; overridable via `WMR_UPDATE_CHECK_INTERVAL` for testing), perform the
   network fetch (below), update the cache, then print if newer **and not already
   printed** in step 1.

So the common path (cache fresh) is a cache read plus compare plus print, with
zero network. The fetch happens at most once per interval per machine (default
once per day), per user profile (see "Cache location").

### The fetch, compare, cache

- `GET https://api.github.com/repos/froggeric/gemini-watermark-and-synthid-remover/releases/latest`
  (the endpoint returns the latest **non-prerelease, non-draft** release per
  GitHub's docs; it returns **404** if no such release exists, handled as a 4xx).
- Headers (set via `CURLOPT_HTTPHEADER`):
  - `Accept: application/vnd.github+json`
  - `User-Agent: wmr` (**versionless**; the only things the server learns are the
    IP, the time, and that it is wmr, which the repo path already implies). Note
    this differs from the regen downloader's `wmr-regen/1.0` UA; the two calls are
    independent and each keeps its own constant.
- `If-None-Match: <cached etag>` when the cache holds one (conditional request).
  The ETag is GitHub-originated and content-derived (the same value is handed to
  every client that fetched the same release state), so resending it is not a
  user-identifying disclosure; it buys a cheap `304` (see Privacy).
- **Decompression (REQUIRED):** set `CURLOPT_ACCEPT_ENCODING` to the empty string
  `""`. api.github.com returns `Content-Encoding: gzip` by default; with the empty
  string, libcurl advertises all the encodings it can decode and auto-
  decompresses the body, so the JSON parser sees plain text. Omitting this is the
  single most likely silent-failure (the body arrives gzipped, JSON parse fails,
  the check swallows it and silently never works).
- **Redirects:** `CURLOPT_FOLLOWLOCATION = 1` (cap with the libcurl default; this
  endpoint rarely redirects, but a regional move must not break the check).
- **Timeouts:** `CURLOPT_CONNECTTIMEOUT_MS = 1500`, `CURLOPT_TIMEOUT_MS = 3000`
  (the whole transfer, connect included, is capped at 3 s).
- **HTTP/HTTPS only:** the URL is `https://`; no downgrade is ever attempted.

Parse and compare:

- `parse_release_json(body)` (pure, unit-tested) extracts `tag_name` and the
  response `ETag`. Missing or non-string `tag_name` returns `nullopt`.
- `parse_tag(tag_name)` strips one leading `v` or `V` and returns the rest.
- `parse_version(s)` (pure) strips one leading `v`/`V`, splits on `.`, parses up
  to three components as non-negative integers (via `strtoul` with full-consume
  check; missing trailing components count as 0; a fourth or later component is
  ignored). Any non-numeric, empty, or overflowing component makes the whole
  version `nullopt` (malformed). A pre-release suffix (e.g. `-rc1`) is dropped by
  parsing only the numeric prefix of each component; wmr does not ship
  prereleases, and `/releases/latest` never returns one anyway.
- `compare_versions(current, latest)` returns -1/0/+1 over the three numeric
  tuples. A malformed **current** (`APP_VERSION`, which is normally `X.Y.Z` but is
  guarded anyway) means "never newer": no notice. A malformed **latest** means the
  fetch is treated as a failure (see Error handling): write `last_check_epoch =
  now`, keep the previous `latest_version`, print nothing new.

Cache file and schema (see "Cache location" for the path):

```json
{
  "last_check_epoch": 1723000000,
  "latest_version": "1.16.4",
  "etag": "\"abc123\""
}
```

- `last_check_epoch`: unix seconds of the last fetch **attempt** (success OR
  failure; written on every fetch so an offline stretch does not retry-storm).
- `latest_version`: last known latest (empty string or absent means "unknown").
- `etag`: GitHub-issued ETag to resend (may be empty).

**Atomic write:** write to `update-check.json.tmp.<pid>` (PID-suffix so two
concurrent wmr processes do not clobber each other's temp file) **in the same
directory** as the target (same filesystem, so the rename is atomic), then
`std::filesystem::rename(tmp, target)`. C++17 `std::filesystem::rename` replaces
an existing destination atomically on POSIX and via `MOVEFILE_REPLACE_EXISTING`
on Windows. A reader that opens the target mid-rename sees either the old file in
full or the new file in full, never a partial. The temp file is removed on any
write failure.

On **any** fetch failure (DNS, TLS, timeout, 4xx / 5xx, `403` rate-limit, `404`
no-release, bad JSON, missing `tag_name`, malformed tag), write
`last_check_epoch = now` so the next retry is not for a full interval (no retry
storm when offline), and keep the previous `latest_version` and `etag` so a
previously-known notice still shows.

### Cache location (reuse the existing wmr cache dir)

**Verified correction.** The spec previously said "reuse the model downloader's
cache-path resolution." That is wrong: `src/core/model_downloader.{hpp,cpp}` takes
a fully-resolved `dest` path as an argument and does **not** resolve the cache
directory. The resolution lives in `src/core/regenerator.cpp`'s file-local
`cache_dir()`:

```cpp
fs::path cache_dir() {
    const char* home = std::getenv("HOME");
#ifdef _WIN32
    if (!home) home = std::getenv("USERPROFILE");
#endif
    fs::path cache = (home && home[0]) ? fs::path(home) / ".cache" / "wmr"
                                       : fs::current_path() / "wmr-cache";
    fs::create_directories(cache);
    return cache;
}
```

Two problems with reusing it as-is:

1. It is in an anonymous namespace inside `regenerator.cpp`, so it is not
   reachable from `update_check.cpp`.
2. The whole TU is `#ifdef WMR_BUILD_REGEN`-gated, but `WMR_UPDATE_CHECK` is
   independent of regen and can be ON while regen is OFF.

**Resolution (do not reverse the locked decision):** extract `cache_dir()` into a
tiny shared, **ungated** helper, `src/core/paths.{hpp,cpp}` (e.g.
`wmr::user_cache_dir()`), have `regenerator.cpp` call it (delete its private
copy), and have `update_check.cpp` use `<user_cache_dir()> / "update-check.json"`.
This keeps a single source of truth for `~/.cache/wmr/`, keeps the update check
working when regen is OFF, and does not change any path regen already uses. The
cache file is therefore `~/.cache/wmr/update-check.json` (or
`%USERPROFILE%\.cache\wmr\update-check.json` on Windows; or `<cwd>/wmr-cache/...`
when neither `HOME` nor `USERPROFILE` is set, matching the existing fallback). It
does **not** honor `XDG_CACHE_HOME`, deliberately, so it stays co-located with the
regen model cache (which also ignores XDG today).

### The notice (exact output)

Printed to stderr, after the subcommand output. The content is the three approved
lines, indented two spaces, framed by the same `----` rule as `print_header`:

```
--------------------------------------------
  A new release of wmr is available: 1.16.3 -> 1.16.4
  Download: https://github.com/froggeric/gemini-watermark-and-synthid-remover/releases/latest
  Disable with WMR_NO_UPDATE_CHECK=1 or --no-update-check.
--------------------------------------------
```

- The rule **must be the exact literal `print_header` emits** (the 44-dash string
  in `cli_app.cpp`). To prevent drift, extract that literal into one shared
  constant (e.g. `wmr::kHeaderRule`) that both `print_header` and
  `format_notice` reference; do not hardcode a second copy.
- The versions are `APP_VERSION` and the parsed `tag_name`; the arrow is ASCII
  `->`.
- With color enabled, the whole block (text and both rules) is wrapped in
  **bold + yellow**. With color disabled, the dashes remain and the block is still
  framed. The URL line spills past the rule, exactly as `print_header`'s URL line
  already does.

### Styling helper (color decision)

`color_enabled()` returns true iff **all** hold:

- `isatty(STDERR_FILENO)` (already guaranteed by the gate; on Windows
  `_isatty(_fileno(stderr))`).
- `getenv("NO_COLOR")` is null **or** points to an empty string. (Rule, stated
  exactly to kill the empty-vs-unset ambiguity: an empty `NO_COLOR=` is treated
  as unset, so color stays ON. Any non-empty value disables. This is the strict
  no-color.org reading.)
- `getenv("TERM")` is null or is not the string `"dumb"`.
- **Windows only:** virtual-terminal processing was successfully enabled on the
  stderr console handle (a one-time `GetStdHandle(STD_ERROR_HANDLE)` +
  `GetConsoleMode` + `SetConsoleMode(... | ENABLE_VIRTUAL_TERMINAL_PROCESSING)`,
  cached in a static bool). If enabling fails (classic `conhost` on an old
  Windows), force color OFF so the user does not see raw `ESC[1m` escapes. On
  non-Windows this condition is always true. Because this notice is the **first**
  use of ANSI in wmr (`print_header` is plain text), there is no existing Windows
  ANSI handling to mirror; this gate is the entire Windows color story.

Emit ANSI (`\033[1m` bold, `\033[33m` yellow, `\033[0m` reset) only when
`color_enabled()` is true. No terminfo / `tput` probing. Bold (SGR 1) and the
basic yellow (SGR 33) are supported on every terminal that is not `TERM=dumb`.

## Components / files

- **`src/core/paths.{hpp,cpp}`** (new, ungated): `fs::path user_cache_dir();`,
  extracted from `regenerator.cpp::cache_dir()`. `regenerator.cpp` is changed
  only to call it and drop its private copy; no path changes.
- **`src/core/update_check.hpp`** plus **`src/core/update_check.cpp`** (new,
  `#ifdef WMR_UPDATE_CHECK`-guarded): the entry point and the helpers.
  - `parse_version(s) -> std::optional<std::array<unsigned,3>>` (pure).
  - `compare_versions(current, latest) -> int` (pure; malformed current => never
    newer).
  - `parse_tag(tag_name) -> std::string` (strip one leading `v`/`V`; pure).
  - `parse_release_json(body) -> std::optional<ReleaseInfo>` where
    `ReleaseInfo { std::string tag; std::string etag; }` (pure; missing/non-string
    `tag_name` => `nullopt`).
  - `should_show(bool no_update_flag, bool env_no_update, bool env_ci,
    bool env_do_not_track, bool is_tty) -> bool` (pure truth table).
  - `should_fetch(cache_age_s, interval_s) -> bool` (pure; `age >= interval`).
  - `read_cache(path) -> CacheData` (corrupt/missing/wrong-type fields =>
    empty `CacheData`).
  - `write_cache(path, CacheData)` (PID-suffixed temp plus `fs::rename`; temp in
    same dir; temp removed on failure).
  - `fetch_latest_release(etag) -> FetchResult` (the thin libcurl HTTP call; the
    only piece not covered by unit tests in CI).
  - `format_notice(current, latest, color_enabled) -> std::string` (exact styled
    string; references the shared `kHeaderRule`).
  - `color_enabled() -> bool` (the styling predicate above).
  - `maybe_check_for_update(const CliOptions&)` (orchestrates the above; the one
    call site from `run_cli`; `void`, never throws, ends by flushing stderr).
- **`src/cli/cli_app.hpp`**: add `bool no_update_check = false;` to `CliOptions`.
- **`src/cli/cli_app.cpp`**:
  - In `add_common`, bind `--no-update-check`:
    `cmd->add_flag("--no-update-check", opts.no_update_check, "Skip the startup update check");`
    (so it appears on every subcommand).
  - Restructure the bottom of `run_cli` to the single chokepoint described above
    and add the one guarded call.
  - Extract `print_header`'s dash literal into `wmr::kHeaderRule` (shared with
    `format_notice`).
- **`CMakeLists.txt`** near the other `option(...)` lines, **before
  `add_subdirectory(tests)`** (so the test mirror block resolves the cached
  option, and so the main exe's `find_package(CURL)` runs once):
  ```cmake
  option(WMR_UPDATE_CHECK "Check for a newer wmr release on startup (notify-only)" ON)
  if(WMR_UPDATE_CHECK)
      find_package(CURL REQUIRED)
      # paths.cpp lives in the top-level SOURCES list (ungated; regen also uses
      # it), so it is NOT listed here again (would be a duplicate-source error).
      target_sources(${PROJECT_NAME} PRIVATE src/core/update_check.cpp)
      target_link_libraries(${PROJECT_NAME} PRIVATE CURL::libcurl)
      target_compile_definitions(${PROJECT_NAME} PRIVATE WMR_UPDATE_CHECK=1)
  endif()
  ```
  (`paths.cpp` is ungated so `regenerator.cpp` can always call `user_cache_dir()`;
  listing it here is harmless when regen is OFF, and required when update-check is
  ON. If `WMR_UPDATE_CHECK` is OFF, `paths.cpp` is still needed by regen, so also
  add `src/core/paths.cpp` to the unconditional `SOURCES` list at the top, OR have
  regen's block add it; pick one and keep a single owner. Recommended: put
  `src/core/paths.cpp` in the top-level `SOURCES` list since it is tiny and
  always-safe.)
- **`tests/CMakeLists.txt`**: mirror the `WMR_AI_DENOISE` block (do **not** edit
  the `TEST_SOURCES` list; `wmr_tests` is already created by then). Add after the
  existing mirror blocks:
  ```cmake
  if(WMR_UPDATE_CHECK)
      target_sources(wmr_tests PRIVATE
          ${CMAKE_SOURCE_DIR}/src/core/paths.cpp
          ${CMAKE_SOURCE_DIR}/src/core/update_check.cpp
          unit/update_check_test.cpp)
      target_link_libraries(wmr_tests PRIVATE CURL::libcurl)
      target_compile_definitions(wmr_tests PRIVATE WMR_UPDATE_CHECK=1)
  endif()
  ```
  Because `update_check.cpp` contains both the pure helpers and
  `fetch_latest_release` (curl), the test target links `CURL::libcurl`; the pure
  helpers do not call it, so the network-free unit tests stay offline. The
  network smoke test is opt-in (see Testing).
- **`vcpkg.json`**: add `curl` to the **base** `dependencies` list (not only the
  `regen` feature). `WMR_UPDATE_CHECK` defaults ON, so a default vcpkg configure
  (`WMR_UPDATE_CHECK` on, `regen` off) must find curl; today curl is only pulled
  by the `regen` feature, so the new `find_package(CURL REQUIRED)` would fail in
  that configuration. curl is a fast, small build relative to ffmpeg/opencv and
  is already built for `regen`, so moving it to base keeps the default-ON build
  honest. (Alternative rejected: a separate `update-check` feature, because the
  locked decision is default-ON and a feature would make the default config
  fail to configure.)
- **`README.md`**: document the check, the opt-out (`--no-update-check`,
  `WMR_NO_UPDATE_CHECK`, `CI`, `DO_NOT_TRACK=1`), the zero-payload privacy stance,
  and the cache file location.

The pure helpers (`parse_version`, `compare_versions`, `parse_tag`,
`parse_release_json`, `should_show`, `should_fetch`, `format_notice`,
`color_enabled`) take their inputs as arguments and do no I/O, so they are fully
testable offline. Only `fetch_latest_release` touches the network and is covered
by the opt-in smoke test.

## Data flow (an eligible run)

1. Subcommand runs and prints its result (image saved, report, video progress).
   `run_cli` captures its return code into `rc` (or `rc = 1` on `std::exception`).
2. On the post-dispatch chokepoint, `run_cli` calls `maybe_check_for_update(opts)`
   (only if `WMR_UPDATE_CHECK` compiled in).
3. Evaluate `should_show(...)`. False -> return immediately (no network).
4. Read the cache (absent / corrupt -> empty `CacheData`).
5. If cached `latest_version` parses and is strictly newer than `APP_VERSION`,
   format and print the notice now (no network). Set `printed = true`.
6. Compute `age = max(0, now - last_check_epoch)`. If
   `should_fetch(age, interval)`, call `fetch_latest_release(etag)` with the 3 s
   timeout.
   - On 200: `parse_release_json` -> `{tag, etag}`; set `latest_version = tag`,
     `etag` from the response.
   - On 304: keep `latest_version`; update `etag` if the response echoes one.
   - On any failure (network / 4xx / 5xx / 404 / 403 / parse error / malformed
     tag): keep `latest_version` and `etag` unchanged.
   - Always set `last_check_epoch = now`. Write the cache atomically.
7. If `latest_version` is strictly newer than `APP_VERSION` **and** `printed` is
   still false, format and print the notice now.
8. Flush stderr. `run_cli` returns `rc`.

Every step after 2 is wrapped so no exception, non-zero exit, or stdout output
can result. The check is a leaf concern with zero ability to affect the command's
exit code (`rc` is returned untouched) or stdout.

## CLI / env surface (exact semantics)

| Knob | Disables when | Notes |
|---|---|---|
| `--no-update-check` (flag, on every subcommand via `add_common`) | the flag is present | Stored in `CliOptions::no_update_check`. |
| `WMR_NO_UPDATE_CHECK` | env is present **and** non-empty | Empty `WMR_NO_UPDATE_CHECK=` is treated as unset (symmetric with `NO_COLOR`). The notice hint shows `=1`. |
| `CI` | env is present **and** non-empty | Good-citizen CI suppression. |
| `DO_NOT_TRACK` | env equals exactly `"1"` | Console Do Not Track convention. |
| `WMR_UPDATE_CHECK_INTERVAL` | (does not disable; tunes throttle) | Parsed as `unsigned long` seconds. Default 86400. On empty / unparseable / negative (impossible for unsigned, but a leading `-` fails parse) -> fall back to 86400. `0` -> fetch on every eligible run (no throttle; useful for tests). Huge values -> effectively never fetch (but the cache-read notice from step 5 still works). |

Naming follows the near-universal `<TOOL>_NO_UPDATE_CHECK` convention
(`GH_NO_UPDATE_NOTIFIER`, `FLY_NO_UPDATE_CHECK`, `DENO_NO_UPDATE_CHECK`).

## Error handling and edge cases

| Failure / case | Behavior |
|---|---|
| DNS / TLS / connect failure | Swallow; write `last_check_epoch = now`; keep old `latest_version`; print nothing new. |
| Timeout (3 s total, 1.5 s connect) | Same. |
| HTTP 4xx / 5xx (incl. 404 no-release) | Same. `/releases/latest` returns 404 when no non-prerelease, non-draft release exists. |
| HTTP 403 (rate limit; UA is always sent) | Same. At one request per interval per IP, the 60/hour unauthenticated limit is a non-issue even behind NAT for a tool of this size. Optionally read `X-RateLimit-Reset` and write `last_check_epoch = max(now, reset)` to back off to the server's window; the simple behavior (back off one interval) is the MVP. |
| 304 Not Modified | Refresh `last_check_epoch`; keep `latest_version`; update `etag` if echoed; no print unless already known-newer. |
| Gzip-encoded body | Handled: `CURLOPT_ACCEPT_ENCODING ""` makes libcurl auto-decompress. Never reaches the parser as raw gzip. |
| HTTP redirect (3xx) | Handled: `CURLOPT_FOLLOWLOCATION = 1`. |
| Malformed JSON / missing `tag_name` / non-string `tag_name` | `parse_release_json` returns `nullopt`; treat as fetch failure (write `last_check_epoch = now`, keep old `latest_version`, print nothing new). |
| Malformed `latest_version` string (in cache or response) | `parse_version` returns `nullopt`; treated as "not newer"; not printed; on the fetch path it counts as a failure so the cache `latest_version` is left unchanged. |
| Cache read failure / missing file / wrong-type fields | `read_cache` returns empty `CacheData`; proceed to fetch. |
| Cache write failure / cache dir not writable | Swallow; the next run re-fetches. No crash, no partial file (the temp is removed). If the dir cannot be created, no notice is ever shown and no error is surfaced (safe direction). |
| Concurrent wmr processes racing the cache | PID-suffixed temp plus atomic `rename`; the last writer wins, readers see either old or new in full, never partial. Two simultaneous fetches waste one request (acceptable). |
| Clock skew (system clock moved back) | `age = max(0, now - last_check_epoch)`; a skewed-back clock makes the cache look fresh, so the check just happens less often. A previously-known newer `latest_version` still prints from cache. Safe direction; never a spurious notice. |
| Running binary NEWER than the latest release (dev past last tag) | `compare_versions` returns not-newer; no notice. Correct. |
| Dev build with `APP_VERSION == "1.1.0"` (the `#ifndef APP_VERSION` fallback) | `1.1.0 < 1.16.x`, so the notice shows on every eligible run. This is correct (the build genuinely reports an old/blank version) and not special-cased; devs can `WMR_NO_UPDATE_CHECK=1`. |
| TTY that lies / pseudo-terminal that cannot render ANSI | Worst case: raw ANSI escapes are shown, but the dashes still frame the notice. On Windows the VT-enable step fails closed (color off). |
| `NO_COLOR` empty vs unset | Both => color ON (empty is treated as unset). Any non-empty value => color OFF. |
| `WMR_UPDATE_CHECK_INTERVAL` = 0 | Fetch on every eligible run (no throttle). |
| `WMR_UPDATE_CHECK_INTERVAL` = garbage / negative | Fall back to 86400. |
| Prerelease / draft release on GitHub | Not returned by `/releases/latest`; irrelevant. A `tag_name` with a `-rc1` suffix parses to its numeric prefix. |
| Opt-out / non-TTY / CI / DNT | No network at all (`should_show` false before any I/O). |
| Extremely long-lived process | n/a; wmr is a run-and-exit CLI. A shell that calls wmr repeatedly is throttled by the per-machine cache. |

No path returns non-zero, throws (uncaught), or writes to stdout.

## Privacy and security posture

- **Zero payload.** Versionless `User-Agent: wmr`, no query string, no body. The
  server learns: the requester IP, the time, and that the client is wmr (already
  implied by the repo path in the URL). This is strictly smaller than the
  multi-GB model fetches regen users already make from HuggingFace. The only
  header beyond the fixed set is `If-None-Match`, which echoes a GitHub-originated
  content ETag (the same for every client fetching the same release state), so it
  is not a user-identifying token.
- **No code execution.** No binary is downloaded in the MVP. The supply-chain
  concern (an update channel as attack vector) does not apply to reading a version
  string.
- **HTTPS only.** No plaintext, no downgrade.
- **Trivial, documented opt-out.** Flag, env, and the disable hint is printed in
  the notice itself.
- **Script / CI safe.** Suppressed when stderr is not a TTY, when `CI` is set, or
  when `DO_NOT_TRACK=1`.
- **First-network-call honesty.** For a visible-only user this is the first
  network behavior. It is documented in the README and the notice, and it is
  opt-out by default (matching consensus), with the payload held strictly to zero.

Signature verification (minisign / cosign / GitHub Artifact Attestations, plus
GitHub's native release-asset SHA256 digests) is **deferred to any future
self-update**, where downloading and running a binary makes it mandatory.

## Acceptance criteria

Each criterion is independently testable (an engineer can write a failing test
from it). Numbers are referenced from the Tests section.

**AC-1 `parse_version` (pure).** `parse_version("1.16.4")` == `{1,16,4}`;
`parse_version("v1.16.4")` == `{1,16,4}`; `parse_version("V1.16.4")` ==
`{1,16,4}`; `parse_version("2")` == `{2,0,0}`; `parse_version("1.2.3.4")` ==
`{1,2,3}` (fourth ignored); `parse_version("1.2-rc1")` parses `{1,2}` (suffix
dropped); `parse_version("")`, `parse_version("v")`, `parse_version("1.a.2")`,
`parse_version("1..2")`, and an overflow component each return `nullopt`.

**AC-2 `compare_versions` (pure).** `compare("1.16.3","1.16.4") < 0`;
`compare("1.16.4","1.16.4") == 0`; `compare("1.17.0","1.16.9") > 0`;
`compare("2.0.0","1.99.99") > 0` (major wins); `compare("1.16","1.16.0") == 0`
(missing == 0); a malformed **current** returns "not newer" regardless of latest.

**AC-3 `parse_tag` (pure).** `parse_tag("v1.16.4")` == `"1.16.4"`;
`parse_tag("V1.16.4")` == `"1.16.4"`; `parse_tag("1.16.4")` == `"1.16.4"`.

**AC-4 `parse_release_json` (pure).** A real-shaped body with `tag_name` and
`ETag` yields `{tag, etag}`; a body missing `tag_name`, with a non-string
`tag_name`, or with unparseable JSON yields `nullopt`.

**AC-5 `should_show` (pure truth table).** Returns true iff all inputs are the
"enable" value (flag false, env_no_update false, env_ci false, env_do_not_track
false, is_tty true). Flipping any one to its "disable" value makes it false. All
five are independently decisive (no input is ignored).

**AC-6 `should_fetch` (pure).** `should_fetch(age=86399, 86400)` == false;
`should_fetch(86400, 86400)` == true (>= boundary fetches);
`should_fetch(0, 0)` == true (interval 0 => always); a negative age is clamped to
0 (so `should_fetch(-100, 86400)` == false).

**AC-7 `read_cache` / `write_cache` (round-trip + resilience).** A written cache
reads back identical. A truncated file, an empty file, a JSON with wrong-type
fields, and a missing file each read back as the empty `CacheData` (never throw).
`write_cache` to a temp plus rename leaves no `*.tmp.*` file on success; on a
simulated mid-write failure the temp is removed and the target is unchanged.

**AC-8 `format_notice` (exact string).** With color off, the output equals the
spec's three-line block byte-for-byte (same 44-dash rule as `print_header`,
two-space indent, ASCII `->`, the exact URL, the exact disable line). With color
on, the same text is wrapped once in the bold+yellow SGR sequence and ends with
reset. `format_notice` writes nothing to stdout.

**AC-9 `color_enabled` (predicate).** Default (no env) => true; `NO_COLOR=""`
=> true; `NO_COLOR=1` (or any non-empty) => false; `TERM=dumb` => false;
non-TTY => false; on Windows when VT-enable fails => false (tested via an
injected enable-result where feasible; otherwise verified manually).

**AC-10 End-to-end gate (integration).** With an injected fake "latest" newer
than `APP_VERSION` (the pure helpers make this trivial: drive
`maybe_check_for_update` with a stubbed `fetch_latest_release`), the notice lands
on stderr, stdout is byte-for-byte empty, ANSI codes appear only when
`color_enabled`, and the process exit code is the subcommand's own (the check
never changes it).

**AC-11 Parse-error / help paths skip the check.** `wmr --version`,
`wmr --help`, `wmr remove --help`, `wmr` (no args), and a bad-flag invocation
make zero network calls and never reach the chokepoint (verified by asserting
`run_cli` does not call `maybe_check_for_update` on those paths; a parse-error
test can use a `maybe_check_for_update` mock/spy).

**AC-12 `_Exit` flush.** When the regen path ran (so `main` takes the
`std::_Exit` branch), the notice (when one was due) is fully present in the
captured stderr. The orchestrator flushes stderr before returning.

**AC-13 Build toggle.** `WMR_UPDATE_CHECK=OFF` compiles with zero update-check
symbols (no `update_check.cpp` in the link, no curl dependency introduced by this
feature). `WMR_UPDATE_CHECK=ON` with `WMR_BUILD_REGEN=OFF` configures and links
(curl resolved via Homebrew or via the base vcpkg dependency).

**AC-14 Zero payload (verified by inspection).** The fetch sets exactly
`CURLOPT_URL` (the fixed HTTPS URL), `CURLOPT_USERAGENT "wmr"`, the fixed
`Accept` header, the optional `If-None-Match`, and the transport options
(timeouts, encoding, follow redirects). No `CURLOPT_POSTFIELDS`, no query string,
no extra header carries version, OS, arch, count, or id.

## Testing

**Unit (`tests/unit/update_check_test.cpp`, no network):** AC-1 through AC-9,
plus the `read_cache`/`write_cache` atomic-write property. Feed
`parse_release_json` captured (real) response bodies as fixtures.

**Smoke (`[update-check][smoke]`, SKIP unless `WMR_NETWORK_TESTS=1`):** AC for
the network path. Real fetch against the live API; assert a parseable tag `>=
APP_VERSION` (a release at or past the current version, since the local build's
tag may equal the latest); no throw; completes under the 3 s timeout; the gzip
path works (body is decoded, not raw gzip). Mirrors the AI-model smoke-test
pattern.

**Integration:** AC-10 (end-to-end with a stubbed fetch), AC-11 (skip on
parse-error / help), AC-12 (`_Exit` flush). Use a spy for `maybe_check_for_update`
to assert reachability without the network.

**Build verification (off-CI or a workflow_dispatch leg):** AC-13 (OFF build,
and ON-without-regen configure), AC-14 (zero-payload by code inspection plus an
assertion that no version/OS/id string is referenced in `update_check.cpp`).

## Risks and gotchas

- **api.github.com returns gzip by default.** Mitigated: `CURLOPT_ACCEPT_ENCODING
  ""`. This is the highest-risk silent failure if forgotten; the smoke test
  asserts a decoded body.
- **GitHub 403** (rate limit, or a missing User-Agent). Mitigated: always send
  `User-Agent: wmr`; on 403, write `last_check_epoch = now` and back off.
- **The regen `_Exit` path.** Mitigated: the notice prints and flushes inside
  `run_cli`, before `main()` can `std::_Exit`.
- **`--version` / `--help` / parse errors must not trigger it.** Mitigated: they
  return from the `CLI::ParseError` catch (or the no-args early return), which is
  upstream of the post-dispatch chokepoint.
- **Color on odd terminals / old Windows conhost.** Mitigated: gated on
  `isatty && !NO_COLOR && TERM != dumb`, plus Windows VT-enable that fails closed.
- **Over-suppression** if a user exports `CI` outside CI. Safe direction: they
  get no check, never a spurious one.
- **Cache-path extraction is a small refactor.** `regenerator.cpp::cache_dir()`
  moves to `src/core/paths.cpp` (ungated). Behavior is identical; the regen
  model-cache path does not change. Verify with the existing regen downloader
  tests (no path change expected).
- **curl availability without regen.** Every release leg builds regen ON
  (verified in `.github/workflows/release.yml`: all four legs pass
  `WMR_BUILD_REGEN=ON`), so every shipped binary has curl. A from-source build
  with `WMR_UPDATE_CHECK=ON` and `regen` OFF needs curl: Homebrew provides it;
  vcpkg gets it via the new base dependency. The dev Homebrew build (`scripts/
  build.sh`) already has curl (regen ON by default there too).
- **CMake configure-time model-copy gotcha does not apply** (no model assets
  involved).

## Build notes

- The `WMR_UPDATE_CHECK` block runs **before `add_subdirectory(tests)`** so the
  test target can mirror it against `WMR_UPDATE_CHECK=1` (mirror the
  `WMR_BUILD_AI_DENOISE` block in `tests/CMakeLists.txt`, not the smoke-only
  pattern, and use `target_sources` since `wmr_tests` already exists).
- The release workflow needs no change: all four legs already build with regen
  ON, so curl is present and the check ships everywhere. Asset names are
  unchanged (`wmr-macos-arm64.zip`, `wmr-macos-x86_64.zip`,
  `wmr-linux-x86_64.tar.gz`, `wmr-windows-x86_64.zip`). No new release artifact.
- macOS signing / notarization needs no change: nothing new is signed (no binary
  download, no dylib).
- `src/core/paths.cpp` is listed in the top-level `SOURCES` (always compiled,
  ungated) so both `regenerator.cpp` (regen build) and `update_check.cpp`
  (update-check build) link it without duplicate-ownership ambiguity.

## Future / out of scope

- **Self-update** (`wmr update`: pick the platform asset, verify SHA256 plus a
  minisign / cosign signature or a GitHub Artifact Attestation, swap the binary
  atomically, re-verify with `codesign --verify --strict` on macOS). This is
  where supply-chain hardening becomes mandatory.
- **Static `latest.json` on `huggingface.co/froggeric/wmr`** (the host wmr
  already trusts) if GitHub API rate limits ever bite.
- **Background / async fetch** (cache-read sync, refresh on a detached thread)
  for zero added latency, if the once-a-day cost ever matters.
- **`wmr version --check`** for an explicit on-demand check.
- **Rule width adaptation** to terminal width (`TIOCGWINSZ`).
- **Honor `X-RateLimit-Reset`** on 403 for a server-aligned backoff.
- **CLICOLOR / CLICOLOR_FORCE** support, if color-control parity with other
  tools is requested.

## Research references

- gh CLI update check: `cli/cli` `internal/ghcmd/cmd.go`, `internal/update/update.go`.
- npm `update-notifier`: `yeoman/update-notifier` (`update-notifier.js`, `check.js`).
- Deno upgrade / check: `denoland/deno` `cli/tools/upgrade.rs`; `DENO_NO_UPDATE_CHECK`.
- pip self-check: `pypa/pip` `src/pip/_internal/self_outdated_check.py`.
- Homebrew auto-update: `Homebrew/brew` `Library/Homebrew/brew.sh` (60 s -> 300 s ->
  86400 s history, issue #6382).
- Console Do Not Track: `donottrack.sh`.
- GitHub CLI telemetry backlash (April 2026): `cli/cli` changelog plus The Register.
- GitHub REST rate limits; release-asset SHA256 digests (June 2025); Artifact
  Attestations; the `/releases/latest` endpoint returns the latest non-prerelease,
  non-draft release (404 otherwise).
- api.github.com default `Content-Encoding: gzip` (libcurl `CURLOPT_ACCEPT_ENCODING`
  auto-decode).
- Supply-chain: xz-utils CVE-2024-3094; event-stream post-mortem; the
  `curl | bash` timing attack.
- macOS notarization / Gatekeeper / `com.apple.quarantine`; `codesign --verify`.
- rustup channel manifest layout (static-file approach) for the static-manifest
  fallback.
