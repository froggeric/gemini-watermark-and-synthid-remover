# Update Check (notify-only) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a notify-only update check to `wmr`: on a real subcommand, compare `APP_VERSION` to the latest GitHub release tag and print a styled three-line notice to stderr when newer.

**Architecture:** A new `WMR_UPDATE_CHECK` CMake feature (default ON) compiles `src/core/update_check.cpp` (pure helpers + one libcurl fetch + an orchestrator) and hooks the orchestrator at a single chokepoint at the end of `run_cli`. A tiny shared `src/core/paths.cpp` (`user_cache_dir()`) is extracted from `regenerator.cpp` so the cache path resolves even when regen is OFF. Zero payload, zero code execution, fully `#ifdef`-guarded.

**Tech Stack:** C++20, libcurl (already used by the regen model downloader), Catch2 tests, CMake `option()` + `target_sources` mirror blocks, CLI11 (adds `--no-update-check`).

**User decisions (already made):** notify-only (no self-update); default-on with opt-out; synchronous, 24h-throttled, GitHub Releases API; notice at end of `run_cli` to stderr; three-line wording; `----` rules matching `print_header` plus bold yellow (gated on `isatty && !NO_COLOR && TERM!=dumb`); `WMR_UPDATE_CHECK` CMake option default ON; `curl` added to base vcpkg deps (agreed). Spec: `docs/superpowers/specs/2026-08-07-update-check-design.md`.

**Minor plan-level refinements over the spec (correct, not reversing any locked decision):**
- `parse_release_json(body)` returns `std::optional<std::string>` (just `tag_name`). The `ETag` is an HTTP response header captured by the curl header callback in `fetch_latest_release`, not a JSON field, so it is not parsed from the body.
- `maybe_check_for_update(bool no_update_check, FetchFn fetch = fetch_latest_release)` takes the one bool it needs instead of the whole `CliOptions`, so `update_check.hpp` does not drag in `cli/cli_app.hpp`. The `run_cli` call site passes `opts.no_update_check`.
- The test target compiles `update_check.cpp` with `APP_VERSION="0.0.0"` so the orchestrator is testable without the real version macro.

---

## File Structure

- **`src/core/paths.hpp` / `src/core/paths.cpp`** (new, UNGATED): `wmr::user_cache_dir()`. Single source of truth for `~/.cache/wmr/`. `regenerator.cpp` delegates to it. Listed in BOTH the top-level `SOURCES` list and the test `LIB_SOURCES` list so the main exe, the update-check build, AND the test link (which compiles `regenerator.cpp` via the regen mirror block) all resolve it.
- **`src/core/update_check.hpp` / `src/core/update_check.cpp`** (new, `#ifdef WMR_UPDATE_CHECK`): pure helpers (`parse_version`, `parse_tag`, `compare_versions`, `parse_release_json`, `should_show`, `should_fetch`, `color_enabled`, `read_cache`, `write_cache`, `format_notice`), the curl fetch (`fetch_latest_release`), the gate-free core (`run_update_check`), and the gated entry point (`maybe_check_for_update`, which `run_cli` calls).
- **`src/cli/cli_app.hpp`**: add `inline constexpr std::string_view kHeaderRule = "...";` (the shared dash literal) and `bool no_update_check = false;` to `CliOptions`.
- **`src/cli/cli_app.cpp`**: `print_header` uses `kHeaderRule`; `add_common` binds `--no-update-check`; the bottom of `run_cli` is restructured to one chokepoint that calls the check.
- **`CMakeLists.txt`**: `option(WMR_UPDATE_CHECK ... ON)` block before `add_subdirectory(tests)` (`:443`); `src/core/paths.cpp` added to `SOURCES` (`:17`).
- **`tests/CMakeLists.txt`**: `src/core/paths.cpp` added to `LIB_SOURCES` (`:17`, Task 1); a `WMR_UPDATE_CHECK` mirror block (uses `target_sources`, mirrors the `WMR_BUILD_AI_DENOISE` block at `:59`) that does NOT list `paths.cpp` (it is in `LIB_SOURCES`).
- **`tests/unit/update_check_test.cpp`** (new): unit tests AC-1..9, the end-to-end-with-stub test AC-10, and the parse-error/help skip test AC-11.
- **`tests/unit/update_check_smoke_test.cpp`** (new): opt-in network smoke test.
- **`vcpkg.json`**: `curl` added to base `dependencies`.
- **`README.md`**: documents the check, opt-out, and zero-payload stance.

---

## Task 1: Extract `user_cache_dir()` into `src/core/paths.{hpp,cpp}`

**Goal:** Move the cache-directory resolution out of `regenerator.cpp` into a shared, ungated helper so both regen and the update check use one source of truth, with zero behavior change.

**Files:**
- Create: `src/core/paths.hpp`, `src/core/paths.cpp`
- Modify: `src/core/regenerator.cpp` (its file-local `cache_dir()` delegates to the new helper), `CMakeLists.txt` (add `src/core/paths.cpp` to `SOURCES` at `:17`), `tests/CMakeLists.txt` (add `src/core/paths.cpp` to the test `LIB_SOURCES` at `:17`)

**Acceptance Criteria:**
- [ ] `wmr::user_cache_dir()` exists in `src/core/paths.hpp`, is NOT `#ifdef`-gated, and returns a path whose final element is `wmr` and whose parent is `.cache` (or the `<cwd>/wmr-cache` fallback when `HOME`/`USERPROFILE` are unset).
- [ ] `regenerator.cpp` no longer defines its own `cache_dir()` body; it calls `wmr::user_cache_dir()` (minimal diff: its local `cache_dir()` becomes `return wmr::user_cache_dir();`, plus `#include "core/paths.hpp"`). All existing `cache_dir()` call sites in `regenerator.cpp` are unchanged.
- [ ] `src/core/paths.cpp` is in BOTH the top-level `SOURCES` list (main exe) AND the test `LIB_SOURCES` list (`tests/CMakeLists.txt:17`) so every configuration links it. The test list is the critical one: the `WMR_BUILD_REGEN` mirror block compiles `regenerator.cpp` into `wmr_tests`, and after this task `regenerator.cpp` calls `wmr::user_cache_dir()`, so `paths.cpp` must be in the test link or the regen/tests CI leg fails with an undefined reference. It is pure and FFmpeg-free (only `<cstdlib>` + `<filesystem>`), so it is safe alongside the other pure helpers already in `LIB_SOURCES`.
- [ ] A full build with `WMR_BUILD_REGEN=ON` still links and the regen downloader tests pass (no path change).

**Verify:** `scripts/build.sh` succeeds; `./build/tests/wmr_tests "[regen]"` passes (or SKIPs if no model); `./build/wmr --version` still prints the version.

**Steps:**

- [ ] **Step 1: Create `src/core/paths.hpp`**

```cpp
#pragma once

#include <filesystem>

namespace wmr {
namespace fs = std::filesystem;

// wmr's cache directory: $HOME/.cache/wmr on POSIX,
// %USERPROFILE%\.cache\wmr on Windows, <cwd>/wmr-cache when neither is set.
// Created on first call. Single source of truth shared by the regen model
// cache and the update check.
fs::path user_cache_dir();

}  // namespace wmr
```

- [ ] **Step 2: Create `src/core/paths.cpp`**

```cpp
#include "core/paths.hpp"

#include <cstdlib>

namespace wmr {

fs::path user_cache_dir() {
    const char* home = std::getenv("HOME");
#ifdef _WIN32
    if (!home || home[0] == '\0') {
        home = std::getenv("USERPROFILE");
    }
#endif
    fs::path cache = (home && home[0] != '\0')
                         ? fs::path(home) / ".cache" / "wmr"
                         : fs::current_path() / "wmr-cache";
    std::error_code ec;
    fs::create_directories(cache, ec);  // best-effort; ignore failure
    return cache;
}

}  // namespace wmr
```

- [ ] **Step 3: Wire `regenerator.cpp` to delegate**

In `src/core/regenerator.cpp`, add the include near the other `core/` includes:

```cpp
#include "core/paths.hpp"
```

Replace the body of its anonymous-namespace `cache_dir()` (around `regenerator.cpp:55`) with a delegation so all call sites keep working:

```cpp
fs::path cache_dir() {
    return wmr::user_cache_dir();
}
```

(Delete the old `HOME`/`USERPROFILE`/`create_directories` logic from `cache_dir()`; it now lives in `paths.cpp`.)

- [ ] **Step 4: Add `paths.cpp` to the top-level `SOURCES`**

In `CMakeLists.txt`, add one line to the `set(SOURCES ...)` block at `:17`:

```cmake
    src/core/paths.cpp
```

- [ ] **Step 4b: Add `paths.cpp` to the test `LIB_SOURCES`**

In `tests/CMakeLists.txt`, add one line to the `set(LIB_SOURCES ...)` block at `:17` (next to the other pure, FFmpeg-free helpers like `still_geometry.cpp`):

```cmake
    ${CMAKE_SOURCE_DIR}/src/core/paths.cpp  # ungated; regenerator.cpp (regen mirror block) calls user_cache_dir()
```

This is required, not optional: the `WMR_BUILD_REGEN` mirror block (`tests/CMakeLists.txt:80`) compiles `regenerator.cpp` into `wmr_tests`, and after this task that TU calls `wmr::user_cache_dir()`. Without `paths.cpp` in the test link the regen/tests CI leg fails to link. Because it is pure and ungated, listing it unconditionally in `LIB_SOURCES` is correct for every feature combination (the `WMR_UPDATE_CHECK` test mirror block in Task 2 must therefore NOT list `paths.cpp` again, or `WMR_BUILD_REGEN=ON + WMR_UPDATE_CHECK=ON` produces a duplicate-source error).

- [ ] **Step 5: Build and verify**

```bash
scripts/build.sh
```
Expected: build succeeds (main exe links `paths.cpp` via `SOURCES`). Then confirm the test link is also satisfied:
```bash
./build/tests/wmr_tests "[regen]" 2>/dev/null; ./build/wmr --version
```
Expected: regen tests pass or SKIP (the test exe links `paths.cpp` via `LIB_SOURCES`, resolving `wmr::user_cache_dir()`); `wmr --version` prints the version.

- [ ] **Step 6: Commit**

```bash
git add src/core/paths.hpp src/core/paths.cpp src/core/regenerator.cpp CMakeLists.txt tests/CMakeLists.txt
git commit -m "refactor: extract user_cache_dir() to src/core/paths (shared by regen + update check)"
```

---

## Task 2: Build wiring for `WMR_UPDATE_CHECK` (stub + CMake + vcpkg)

**Goal:** Stand up the compile/test harness: a stub `update_check.{hpp,cpp}`, the CMake option + curl link, curl as a base vcpkg dep, and the tests mirror block. No behavior yet; this proves the ON build configures/links and the OFF build stays clean.

**Files:**
- Create: `src/core/update_check.hpp`, `src/core/update_check.cpp`, `tests/unit/update_check_test.cpp`
- Modify: `CMakeLists.txt` (add the `WMR_UPDATE_CHECK` option block before `:443`), `tests/CMakeLists.txt` (add the mirror block), `vcpkg.json` (add `curl` to base deps)

**Acceptance Criteria:**
- [ ] `option(WMR_UPDATE_CHECK ... ON)` exists in `CMakeLists.txt` before `add_subdirectory(tests)`.
- [ ] When ON: `find_package(CURL REQUIRED)` runs, `src/core/update_check.cpp` is a target source, `CURL::libcurl` is linked, `WMR_UPDATE_CHECK=1` is defined on the main exe.
- [ ] When OFF (`-DWMR_UPDATE_CHECK=OFF`): the build configures and links with zero update-check symbols and no curl dependency introduced by this feature.
- [ ] `tests/CMakeLists.txt` has a `WMR_UPDATE_CHECK` mirror block that adds `update_check.cpp` and `unit/update_check_test.cpp`, links `CURL::libcurl`, and defines `WMR_UPDATE_CHECK=1 APP_VERSION="0.0.0"`. It does NOT list `paths.cpp` (that lives in the unconditional `LIB_SOURCES` list from Task 1, so listing it here would be a duplicate-source error when `WMR_BUILD_REGEN=ON` as well).
- [ ] `vcpkg.json` base `dependencies` includes `curl`.
- [ ] The stub test `tests/unit/update_check_test.cpp` has one trivial `[update-check]` test that compiles and passes (e.g., asserts `parse_tag` is declared; filled in next task).

**Verify:** `scripts/build.sh` (ON by default) succeeds and `./build/tests/wmr_tests "[update-check]"` runs the stub test green. A separate `cmake -B build-off -S . -DWMR_UPDATE_CHECK=OFF -GNinja && cmake --build build-off` configures and builds.

**Steps:**

- [ ] **Step 1: Create the stub `src/core/update_check.hpp`**

```cpp
#pragma once

#ifdef WMR_UPDATE_CHECK

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

namespace wmr {
namespace fs = std::filesystem;

// (Helpers are added in later tasks. This stub only declares the entry point
//  so the build harness compiles.)

struct FetchResult {
    bool ok = false;
    int http_code = 0;
    std::string body;
    std::string etag;
    std::string error;
};
using FetchFn = std::function<FetchResult(const std::string& etag)>;

// Notify-only update check. Never throws, never writes to stdout, never changes
// the process exit code. `no_update_check` is the --no-update-check flag value.
void maybe_check_for_update(bool no_update_check,
                            FetchFn fetch = {});

}  // namespace wmr

#endif  // WMR_UPDATE_CHECK
```

- [ ] **Step 2: Create the stub `src/core/update_check.cpp`**

```cpp
#ifdef WMR_UPDATE_CHECK
#include "core/update_check.hpp"

namespace wmr {

// Filled in across later tasks. The stub fetch + no-op orchestrator let the
// build harness link before any real logic exists.
FetchResult fetch_latest_release(const std::string& /*etag*/) {
    return {/*ok=*/false, /*http_code=*/0, /*body=*/"", /*etag=*/"", /*error=*/"unimplemented"};
}

// Resolves the default fetch when the caller omits it. Defined out-of-line so
// the header's default argument `FetchFn fetch = {}` can be replaced by a real
// default at the call site in run_cli (which passes fetch_latest_release).
void maybe_check_for_update(bool /*no_update_check*/, FetchFn /*fetch*/) {
    // no-op for now
}

}  // namespace wmr
#endif  // WMR_UPDATE_CHECK
```

- [ ] **Step 3: Add the CMake option block**

In `CMakeLists.txt`, immediately before `add_subdirectory(tests)` (`:443`), add:

```cmake
# Update check (notify-only). Runs BEFORE add_subdirectory(tests) so the tests
# mirror block below sees the cached option.
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

- [ ] **Step 4: Add curl to base vcpkg deps**

In `vcpkg.json`, add `"curl"` to the base `dependencies` array (alongside `fmt`, `cli11`, etc.):

```json
    "dependencies": [
      { "name": "opencv4", "default-features": false, "features": ["jpeg", "png", "webp", "contrib"] },
      "fmt",
      "cli11",
      "spdlog",
      "catch2",
      "curl",
      { "name": "ffmpeg", "features": ["x264"] }
    ],
```

- [ ] **Step 5: Add the tests mirror block**

In `tests/CMakeLists.txt`, after the `WMR_BUILD_AI_COREML_SD` block (after `:118`), add:

```cmake
# Update check (notify-only): mirrors the main CMakeLists block.
# NOTE: src/core/paths.cpp is intentionally NOT listed here; Task 1 added it to the
# unconditional LIB_SOURCES list above (regenerator.cpp, compiled by the regen mirror
# block, also needs it). Listing it here would be a duplicate-source error when
# WMR_BUILD_REGEN is ON too.
if(WMR_UPDATE_CHECK)
    target_sources(wmr_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/src/core/update_check.cpp
        ${CMAKE_SOURCE_DIR}/tests/unit/update_check_test.cpp)
    target_link_libraries(wmr_tests PRIVATE CURL::libcurl)
    target_compile_definitions(wmr_tests PRIVATE WMR_UPDATE_CHECK=1 APP_VERSION="0.0.0")
endif()
```

- [ ] **Step 6: Create the stub test `tests/unit/update_check_test.cpp`**

```cpp
#include <catch2/catch_test_macros.hpp>

#ifdef WMR_UPDATE_CHECK
#include "core/update_check.hpp"
using namespace wmr;
#endif

TEST_CASE("update_check harness compiles", "[update-check]") {
    REQUIRE(true);
}
```

- [ ] **Step 7: Build ON (default) and run the stub test**

```bash
scripts/build.sh && ./build/tests/wmr_tests "[update-check]"
```
Expected: builds; one test passes.

- [ ] **Step 8: Verify the OFF build stays clean**

```bash
cmake -B build-off -S . -GNinja -DWMR_UPDATE_CHECK=OFF && cmake --build build-off
```
Expected: configures and builds with no update-check symbols.

- [ ] **Step 9: Commit**

```bash
git add src/core/update_check.hpp src/core/update_check.cpp tests/unit/update_check_test.cpp \
        CMakeLists.txt tests/CMakeLists.txt vcpkg.json
git commit -m "build: add WMR_UPDATE_CHECK option (curl base dep, stub update_check, tests mirror)"
```

---

## Task 3: Pure parse / compare / gate / color helpers

**Goal:** Implement the pure, side-effect-free helpers (version parsing, JSON tag extraction, the show/fetch gates, the color predicate) with full unit-test coverage. These need no network and no I/O.

**Files:**
- Modify: `src/core/update_check.hpp` (add declarations), `src/core/update_check.cpp` (add implementations), `tests/unit/update_check_test.cpp` (add AC-1..6, AC-9 tests)

**Acceptance Criteria (spec AC-1..6, AC-9):**
- [ ] **AC-1** `parse_version("1.16.4")=={1,16,4}`; `"v1.16.4"` and `"V1.16.4"` parse the same; `"2"=={2,0,0}`; `"1.2.3.4"=={1,2,3}` (fourth ignored); `"1.2-rc1"` parses `{1,2}` (suffix dropped); `""`, `"v"`, `"1.a.2"`, `"1..2"`, and an overflowing component each return `nullopt`.
- [ ] **AC-2** `compare_versions("1.16.3","1.16.4")<0`; equal `==0`; `"1.17.0">"1.16.9"` `>0`; major wins `"2.0.0">"1.99.99"`; missing `=="1.16"=="1.16.0"`; a malformed current returns `>0` (never newer) regardless of latest.
- [ ] **AC-3** `parse_tag("v1.16.4")=="1.16.4"`; `"V1.16.4"` and `"1.16.4"` both yield `"1.16.4"`.
- [ ] **AC-4** `parse_release_json` extracts `tag_name` from a real-shaped body; returns `nullopt` when `tag_name` is missing, non-string, or the JSON is unparseable.
- [ ] **AC-5** `should_show` returns true iff all five inputs are the enable value; flipping any one disables; all five are independently decisive.
- [ ] **AC-6** `should_fetch(86399,86400)==false`; `should_fetch(86400,86400)==true`; `should_fetch(0,0)==true`; a negative age is clamped to 0 so `should_fetch(-100,86400)==false`.
- [ ] **AC-9** `color_enabled_for` (the pure policy): default (no env, TTY) true; `NO_COLOR=""` (empty) true; `NO_COLOR="1"` (any non-empty) false; `TERM="dumb"` false; non-TTY false. Each input is independently decisive. (`color_enabled()` is the public predicate that delegates to `color_enabled_for(stderr_is_tty(), getenv("NO_COLOR"), getenv("TERM"))`; its end-to-end behavior is also exercised by the Task 6 piped-output check.)

**Verify:** `./build/tests/wmr_tests "[update-check]"` (all added sections pass).

**Steps:**

- [ ] **Step 1: Add declarations to `src/core/update_check.hpp`**

First add `<array>` to the header's existing include block at the top (alongside `<filesystem>`, `<functional>`, `<optional>`, `<string>`, `<string_view>`). Do NOT put the `#include` inside `namespace wmr {`: a standard-library `#include` inside a namespace declares the names into `wmr::std::...` (the header's own `namespace std { ... }` becomes nested), so `std::array` used below would fail to resolve. Includes always go at file scope before any `namespace`:

```cpp
#include <array>
```

Then, inside `namespace wmr {` (still under `#ifdef WMR_UPDATE_CHECK`), add the declarations:

```cpp
std::optional<std::array<unsigned, 3>> parse_version(std::string_view s);
// -1 if current<latest, 0 if equal, +1 if current>latest. A malformed current
// yields +1 ("never newer"). A malformed latest yields +1 (cannot confirm newer).
int compare_versions(std::string_view current, std::string_view latest);
std::string parse_tag(std::string_view tag_name);  // strip one leading v/V
std::optional<std::string> parse_release_json(std::string_view body);

bool should_show(bool no_update_flag, bool env_no_update, bool env_ci,
                 bool env_do_not_track, bool is_tty);
bool should_fetch(long long age_s, long long interval_s);
// Pure color policy: the same logic as color_enabled(), but with the TTY result
// and the two env values passed in, so it is fully unit-testable with no
// isatty/getenv dependency (a non-TTY CI runner would otherwise force
// color_enabled() false regardless of NO_COLOR/TERM, making AC-9 untestable).
// no_color/term may be nullptr (getenv's "unset" return).
bool color_enabled_for(bool is_tty, const char* no_color, const char* term);
bool color_enabled();
```

- [ ] **Step 2: Write the failing tests (AC-1..6, AC-9) in `tests/unit/update_check_test.cpp`**

Replace the stub test body with the includes + the cases below. There is exactly ONE `#ifdef WMR_UPDATE_CHECK ... #endif` block in this file; it wraps ALL update-check test cases. Tasks 4 and 5 APPEND new cases inside the same block (before the single trailing `#endif`), they do not open a second block. Keep the shared top-of-block includes and `using namespace wmr;`:

```cpp
#include <catch2/catch_test_macros.hpp>

#ifdef WMR_UPDATE_CHECK
#include "core/update_check.hpp"
using namespace wmr;

TEST_CASE("parse_version", "[update-check]") {
    REQUIRE(parse_version("1.16.4").value() == std::array<unsigned,3>{1,16,4});
    REQUIRE(parse_version("v1.16.4").value() == std::array<unsigned,3>{1,16,4});
    REQUIRE(parse_version("V1.16.4").value() == std::array<unsigned,3>{1,16,4});
    REQUIRE(parse_version("2").value() == std::array<unsigned,3>{2,0,0});
    REQUIRE(parse_version("1.2.3.4").value() == std::array<unsigned,3>{1,2,3});
    REQUIRE(parse_version("1.2-rc1").value() == std::array<unsigned,3>{1,2,0});
    REQUIRE_FALSE(parse_version("").has_value());
    REQUIRE_FALSE(parse_version("v").has_value());
    REQUIRE_FALSE(parse_version("1.a.2").has_value());
    REQUIRE_FALSE(parse_version("1..2").has_value());
    REQUIRE_FALSE(parse_version("99999999999").has_value());  // overflow
}

TEST_CASE("compare_versions", "[update-check]") {
    REQUIRE(compare_versions("1.16.3","1.16.4") < 0);
    REQUIRE(compare_versions("1.16.4","1.16.4") == 0);
    REQUIRE(compare_versions("1.17.0","1.16.9") > 0);
    REQUIRE(compare_versions("2.0.0","1.99.99") > 0);
    REQUIRE(compare_versions("1.16","1.16.0") == 0);
    REQUIRE(compare_versions("notaver","1.16.4") > 0);  // malformed current never newer
}

TEST_CASE("parse_tag", "[update-check]") {
    REQUIRE(parse_tag("v1.16.4") == "1.16.4");
    REQUIRE(parse_tag("V1.16.4") == "1.16.4");
    REQUIRE(parse_tag("1.16.4") == "1.16.4");
}

TEST_CASE("parse_release_json", "[update-check]") {
    const char* ok = R"({"tag_name":"v1.16.4","assets":[]})";
    REQUIRE(parse_release_json(ok).value() == "v1.16.4");
    REQUIRE_FALSE(parse_release_json(R"({"assets":[]})").has_value());      // missing
    REQUIRE_FALSE(parse_release_json(R"({"tag_name":123})").has_value());   // non-string
    REQUIRE_FALSE(parse_release_json("not json{").has_value());             // unparseable
}

TEST_CASE("should_show truth table", "[update-check]") {
    REQUIRE(should_show(false,false,false,false,true) == true);
    REQUIRE(should_show(true, false,false,false,true) == false);
    REQUIRE(should_show(false,true, false,false,true) == false);
    REQUIRE(should_show(false,false,true, false,true) == false);
    REQUIRE(should_show(false,false,false,true, true) == false);
    REQUIRE(should_show(false,false,false,false,false) == false);  // non-TTY
}

TEST_CASE("should_fetch", "[update-check]") {
    REQUIRE(should_fetch(86399, 86400) == false);
    REQUIRE(should_fetch(86400, 86400) == true);
    REQUIRE(should_fetch(0, 0) == true);
    REQUIRE(should_fetch(-100, 86400) == false);
}

TEST_CASE("color_enabled_for (pure policy)", "[update-check]") {
    REQUIRE(color_enabled_for(true,  nullptr, nullptr) == true);   // default: TTY, no NO_COLOR, no TERM
    REQUIRE(color_enabled_for(true,  "",      nullptr) == true);   // NO_COLOR empty == unset == ON
    REQUIRE(color_enabled_for(true,  "1",     nullptr) == false);  // NO_COLOR non-empty == OFF
    REQUIRE(color_enabled_for(true,  "any",   nullptr) == false);  // any non-empty value disables
    REQUIRE(color_enabled_for(true,  nullptr, "xterm") == true);   // TERM != dumb == ON
    REQUIRE(color_enabled_for(true,  nullptr, "dumb") == false);   // TERM == dumb == OFF
    REQUIRE(color_enabled_for(false, nullptr, nullptr) == false);  // non-TTY always OFF (independent)
}
#endif
```

- [ ] **Step 3: Run tests to confirm they fail (functions undefined)**

```bash
cmake --build build && ./build/tests/wmr_tests "[update-check]"
```
Expected: link failure or failures (helpers not yet defined).

- [ ] **Step 4: Implement the helpers in `src/core/update_check.cpp`**

Add (after the includes at top):

```cpp
#include "core/update_check.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>

#ifdef _WIN32
#include <io.h>       // _isatty, _fileno
#include <cstdio>     // stderr (for _fileno)
#include <process.h>  // _getpid (used by write_cache's PID-suffixed temp)
#else
#include <unistd.h>   // getpid, STDERR_FILENO
#endif

namespace wmr {

namespace {
bool env_nonempty(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0';
}
bool env_equals(const char* name, const char* want) {
    const char* v = std::getenv(name);
    return v != nullptr && std::strcmp(v, want) == 0;
}
bool stderr_is_tty() {
#ifdef _WIN32
    return _isatty(_fileno(stderr)) != 0;
#else
    return isatty(STDERR_FILENO) != 0;
#endif
}
}  // namespace

std::optional<std::array<unsigned, 3>> parse_version(std::string_view s) {
    if (!s.empty() && (s[0] == 'v' || s[0] == 'V')) s.remove_prefix(1);
    if (s.empty()) return std::nullopt;
    std::array<unsigned, 3> v{0, 0, 0};
    size_t idx = 0, i = 0;
    while (idx < 3 && i < s.size()) {
        std::string tok;
        while (i < s.size() && s[i] != '.') tok.push_back(s[i++]);
        // Reject empty tokens ("1..2" -> empty between the dots) and tokens that do
        // not start with a digit. The digit check also defeats strtoul's documented
        // sign-skip: strtoul("-1") returns ULONG_MAX (it negates the magnitude) with
        // NO ERANGE on LP64, so without this guard "-1" would parse as a huge value.
        // It likewise rejects "+1", leading whitespace, and letters ("1.a.2" -> "a").
        if (tok.empty() || tok[0] < '0' || tok[0] > '9') return std::nullopt;
        char* end = nullptr;
        errno = 0;
        unsigned long n = std::strtoul(tok.c_str(), &end, 10);
        // Overflow handling must cover BOTH widths:
        //   - On LP64 (mac/linux 64-bit), unsigned long is 64-bit, so the test value
        //     99999999999 fits without ERANGE. The n > UINT_MAX check catches it
        //     (otherwise the static_cast<unsigned> below would silently truncate it
        //     to 1215751683 and the overflow test would fail).
        //   - On 32-bit / LLP64 where unsigned long is 32-bit, 99999999999 trips
        //     ERANGE (strtoul saturates at ULONG_MAX); the errno branch catches it.
        if (errno == ERANGE || n > static_cast<unsigned long>(std::numeric_limits<unsigned>::max()))
            return std::nullopt;
        // strtoul parses the numeric PREFIX and stops at the first non-digit, which is
        // how a pre-release suffix is dropped ("2-rc1" -> 2). That is intended; we do
        // NOT require *end == '\0'. tok[0] being a digit already guarantees >= 1 digit
        // was consumed (end != tok.c_str()).
        v[idx++] = static_cast<unsigned>(n);
        if (i < s.size() && s[i] == '.') ++i;
    }
    return v;
}

int compare_versions(std::string_view current, std::string_view latest) {
    auto c = parse_version(current);
    auto l = parse_version(latest);
    if (!c || !l) return 1;  // malformed => never report "newer"
    if (*c < *l) return -1;
    if (*c > *l) return 1;
    return 0;
}

std::string parse_tag(std::string_view tag_name) {
    if (!tag_name.empty() && (tag_name[0] == 'v' || tag_name[0] == 'V'))
        return std::string(tag_name.substr(1));
    return std::string(tag_name);
}

bool should_show(bool no_update_flag, bool env_no_update, bool env_ci,
                 bool env_do_not_track, bool is_tty) {
    return !no_update_flag && !env_no_update && !env_ci && !env_do_not_track && is_tty;
}

bool should_fetch(long long age_s, long long interval_s) {
    if (age_s < 0) age_s = 0;
    return age_s >= interval_s;
}

bool color_enabled_for(bool is_tty, const char* no_color, const char* term) {
    if (!is_tty) return false;
    if (no_color != nullptr && no_color[0] != '\0') return false;  // any non-empty disables
    if (term != nullptr && std::strcmp(term, "dumb") == 0) return false;
    return true;
}

bool color_enabled() {
    return color_enabled_for(stderr_is_tty(),
                             std::getenv("NO_COLOR"),
                             std::getenv("TERM"));
}
```

- [ ] **Step 5: Add the JSON parser (`parse_release_json`)**

Append to `update_check.cpp`. Use a minimal hand-rolled scan (avoids a JSON dependency). It looks for `"tag_name"` then a string value:

```cpp
namespace {
// Find the string value of the top-level-ish "tag_name" field. Returns nullopt
// if not found or not a JSON string. Tolerant: does not need a full parser.
std::optional<std::string> extract_string_field(std::string_view body, std::string_view key) {
    // search for "key"
    for (size_t i = 0; i + key.size() + 2 < body.size(); ++i) {
        if (body[i] != '"') continue;
        if (body.substr(i + 1, key.size()) != key) continue;
        if (body[i + 1 + key.size()] != '"') continue;
        // found "key"; scan forward to the next ':' then the next '"'
        size_t j = i + 1 + key.size() + 1;
        while (j < body.size() && body[j] != ':') ++j;
        if (j == body.size()) return std::nullopt;
        ++j;  // past ':'
        while (j < body.size() && (body[j] == ' ' || body[j] == '\t' || body[j] == '\n' || body[j] == '\r')) ++j;
        if (j >= body.size() || body[j] != '"') return std::nullopt;  // not a string
        ++j;
        std::string out;
        while (j < body.size() && body[j] != '"') {
            if (body[j] == '\\' && j + 1 < body.size()) { out.push_back(body[j + 1]); j += 2; }
            else { out.push_back(body[j++]); }
        }
        if (j >= body.size()) return std::nullopt;  // unterminated
        return out;
    }
    return std::nullopt;
}
}  // namespace

std::optional<std::string> parse_release_json(std::string_view body) {
    return extract_string_field(body, "tag_name");
}
```

- [ ] **Step 6: Run tests to confirm they pass**

```bash
cmake --build build && ./build/tests/wmr_tests "[update-check]"
```
Expected: all sections pass.

- [ ] **Step 7: Commit**

```bash
git add src/core/update_check.hpp src/core/update_check.cpp tests/unit/update_check_test.cpp
git commit -m "feat(update-check): pure parse/compare/gate/color helpers with tests"
```

---

## Task 4: Cache I/O and notice formatting

**Goal:** Implement cache persistence (atomic, race-safe) and notice rendering (the exact three-line block with the shared `kHeaderRule`).

**Files:**
- Modify: `src/core/update_check.hpp`, `src/core/update_check.cpp`, `src/cli/cli_app.hpp` (add `kHeaderRule`), `src/cli/cli_app.cpp` (`print_header` uses `kHeaderRule`), `tests/unit/update_check_test.cpp` (AC-7, AC-8)

**Acceptance Criteria (spec AC-7, AC-8):**
- [ ] **AC-7** A written cache reads back identical. A truncated file, an empty file, a JSON with wrong-type fields, and a missing file each read back as the empty `CacheData` (never throw). `write_cache` leaves no `*.tmp.*` on success; on a simulated mid-write failure the temp is removed and the target is unchanged.
- [ ] **AC-8** With color off, `format_notice` output equals the spec's three-line block byte-for-byte (same dash rule as `print_header`, two-space indent, ASCII `->`, the exact URL, the exact disable line). With color on, the same text is wrapped once in bold+yellow SGR and ends with reset. Writes nothing to stdout.
- [ ] `kHeaderRule` is defined once (in `cli_app.hpp`); both `print_header` and `format_notice` reference it (no second copy).

**Verify:** `./build/tests/wmr_tests "[update-check]"` passes the new AC-7/AC-8 sections; `./build/wmr` (no args) still prints the header unchanged.

**Steps:**

- [ ] **Step 1: Add `kHeaderRule` to `src/cli/cli_app.hpp`**

Near the top of the file (ungated, before `CliOptions`), add:

```cpp
#include <string_view>

namespace wmr {
// The dash rule used by print_header and the update-check notice. One literal,
// shared, so the two framed blocks cannot drift apart.
inline constexpr std::string_view kHeaderRule =
    "--------------------------------------------";  // 44 dashes
}  // namespace wmr
```

(Count the dashes to exactly match the literal in `print_header` at `cli_app.cpp:27`.)

- [ ] **Step 2: Make `print_header` use it**

In `src/cli/cli_app.cpp`, change the rule lines in `print_header` (around `:27` and the closing rule) to stream `wmr::kHeaderRule`:

```cpp
void print_header(std::ostream& os) {
    os << wmr::kHeaderRule << "\n"
       << "  wmr v" APP_VERSION " — watermark remover\n"
       << "  Remove Gemini/Veo visible watermarks.\n"
       << "  Remove SynthID invisible watermarks via lossy regen (no detection; validated vs Google's SynthID verifier).\n"
       << "  --synthid-attack regen is lossy (SDXL img2img; ~6.5 GB model + ~335 MB VAE download on first use).\n"
       << "  https://github.com/froggeric/gemini-watermark-and-synthid-remover\n"
       << "  Copyright 2026 Frédéric Guigand\n"
       << wmr::kHeaderRule << "\n\n";
}
```

- [ ] **Step 3: Add cache + format declarations to `src/core/update_check.hpp`**

```cpp
struct CacheData {
    long long last_check_epoch = 0;  // unix seconds of last fetch attempt
    std::string latest_version;      // empty == unknown
    std::string etag;                // may be empty
};
CacheData read_cache(const fs::path& p);
bool write_cache(const fs::path& p, const CacheData& d);  // atomic; false on failure

// Render the three-line notice. current/latest are the version strings to show.
std::string format_notice(std::string_view current, std::string_view latest, bool color);
```

(Add `#include "cli/cli_app.hpp"` is NOT needed; `format_notice` references `kHeaderRule` via that header, so add the include at the top of `update_check.cpp` instead to keep the .hpp light. See Step 5.)

- [ ] **Step 4: Write the failing tests (AC-7, AC-8)**

Append to `tests/unit/update_check_test.cpp` (inside the `#ifdef WMR_UPDATE_CHECK` block, after the existing tests):

```cpp
#include "cli/cli_app.hpp"  // wmr::kHeaderRule
#include <cstdio>
#include <fstream>
#include <filesystem>

namespace tfs = std::filesystem;

TEST_CASE("cache round-trip and resilience", "[update-check]") {
    auto tmp = tfs::temp_directory_path() / ("wmr_uc_test_" + std::to_string(getpid()) + ".json");
    CacheData in{1723000000, "1.16.4", "\"abc\""};
    REQUIRE(write_cache(tmp, in));
    CacheData out = read_cache(tmp);
    REQUIRE(out.last_check_epoch == 1723000000);
    REQUIRE(out.latest_version == "1.16.4");
    REQUIRE(out.etag == "\"abc\"");
    tfs::remove(tmp);

    // missing file => empty
    REQUIRE(read_cache(tmp).latest_version.empty());

    // empty file => empty
    std::ofstream(tmp).put(' ');
    REQUIRE(read_cache(tmp).latest_version.empty());
    tfs::remove(tmp);

    // garbage => empty
    { std::ofstream f(tmp); f << "{ this is not json "; }
    REQUIRE(read_cache(tmp).latest_version.empty());
    tfs::remove(tmp);
}

TEST_CASE("format_notice exact string", "[update-check]") {
    std::string s = format_notice("1.16.3", "1.16.4", /*color=*/false);
    REQUIRE(s.find("A new release of wmr is available: 1.16.3 -> 1.16.4") != std::string::npos);
    REQUIRE(s.find("Download: https://github.com/froggeric/gemini-watermark-and-synthid-remover/releases/latest") != std::string::npos);
    REQUIRE(s.find("Disable with WMR_NO_UPDATE_CHECK=1 or --no-update-check.") != std::string::npos);
    REQUIRE(s.find("  ") != std::string::npos);              // two-space indent present
    REQUIRE(s.find(wmr::kHeaderRule) != std::string::npos);  // shares the header rule
    REQUIRE(s.find("\033[") == std::string::npos);           // no ANSI when color off

    std::string c = format_notice("1.16.3", "1.16.4", /*color=*/true);
    REQUIRE(c.find("\033[1m") != std::string::npos);         // bold
    REQUIRE(c.find("\033[33m") != std::string::npos);        // yellow
    REQUIRE(c.find("\033[0m") != std::string::npos);         // reset
}
```

Also add a `getpid` include: `#include <unistd.h>` (POSIX) at the top of the test file, and use the unqualified global `getpid()`, NOT `std::getpid()` (POSIX `getpid` is declared by `<unistd.h>` in the GLOBAL namespace; it is not a `std::` function and `<cstdlib>` does not provide `std::getpid`, so `std::getpid` is ill-formed). The test target runs on mac/linux CI only (the `tests` job is ubuntu; Windows is not a test leg), so the POSIX `getpid`/`setenv`/`unsetenv` calls are safe here.

- [ ] **Step 5: Implement cache + format in `src/core/update_check.cpp`**

At the top of `update_check.cpp`, add includes:

```cpp
#include "cli/cli_app.hpp"  // wmr::kHeaderRule
#include "core/paths.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <system_error>
```

Then:

```cpp
namespace {
// Tiny JSON string escaper for writing the cache.
std::string esc(std::string_view s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') { out.push_back('\\'); out.push_back(c); }
        else if (c == '\n') out += "\\n";
        else out.push_back(c);
    }
    return out;
}
}  // namespace

CacheData read_cache(const fs::path& p) {
    CacheData d;
    std::ifstream f(p);
    if (!f) return d;
    std::stringstream ss; ss << f.rdbuf();
    std::string body = ss.str();
    // Reuse the string-field extractor for last_check_epoch + the two strings.
    // (extract_string_field is defined in Task 3; epoch is numeric, parsed manually.)
    // latest_version
    if (auto v = extract_string_field(body, "latest_version")) d.latest_version = *v;
    if (auto v = extract_string_field(body, "etag")) d.etag = *v;
    // last_check_epoch: scan for the numeric value after "last_check_epoch"
    auto pos = body.find("\"last_check_epoch\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            d.last_check_epoch = std::strtoll(body.c_str() + pos + 1, nullptr, 10);
        }
    }
    return d;
}

bool write_cache(const fs::path& p, const CacheData& d) {
    std::error_code ec;
    fs::create_directories(p.parent_path(), ec);
    std::ostringstream ss;
    ss << "{\"last_check_epoch\":" << d.last_check_epoch
       << ",\"latest_version\":\"" << esc(d.latest_version) << "\""
       << ",\"etag\":\"" << esc(d.etag) << "\"}";
    std::string data = ss.str();
    // PID-suffixed temp in the SAME directory => rename is atomic (same fs).
    // Use the global getpid() from <unistd.h> (POSIX), NOT std::getpid (no such
    // name in std; <cstdlib> does not declare it). <unistd.h> is included via the
    // Task 3 non-Windows branch; on Windows <process.h> provides _getpid().
    std::string pid;
#ifdef _WIN32
    pid = std::to_string(_getpid());
#else
    pid = std::to_string(getpid());
#endif
    std::string tmpname = p.filename().string() + ".tmp." + pid;
    fs::path tmp = p.parent_path() / tmpname;
    {
        std::ofstream of(tmp, std::ios::binary);
        if (!of) return false;
        of << data;
        of.flush();
        if (!of) { std::error_code rm; fs::remove(tmp, rm); return false; }
    }
    fs::rename(tmp, p, ec);  // atomic replace on POSIX; MOVEFILE_REPLACE_EXISTING on Windows
    if (ec) { std::error_code rm; fs::remove(tmp, rm); return false; }
    return true;
}

std::string format_notice(std::string_view current, std::string_view latest, bool color) {
    std::ostringstream os;
    const char* B = color ? "\033[1m\033[33m" : "";  // bold + yellow
    const char* R = color ? "\033[0m" : "";
    os << B << wmr::kHeaderRule << R << "\n"
       << B << "  A new release of wmr is available: " << current << " -> " << latest << R << "\n"
       << B << "  Download: https://github.com/froggeric/gemini-watermark-and-synthid-remover/releases/latest" << R << "\n"
       << B << "  Disable with WMR_NO_UPDATE_CHECK=1 or --no-update-check." << R << "\n"
       << B << wmr::kHeaderRule << R << "\n";
    return os.str();
}
```

Note: `extract_string_field` lives in an anonymous namespace in the same TU (from Task 3, Step 5), so `read_cache` can call it. Keep that helper accessible (do not re-declare it).

- [ ] **Step 6: Build and run tests**

```bash
cmake --build build && ./build/tests/wmr_tests "[update-check]"
```
Expected: AC-7 and AC-8 sections pass; header still prints correctly:
```bash
./build/wmr 2>&1 | head -2
```

- [ ] **Step 7: Commit**

```bash
git add src/core/update_check.hpp src/core/update_check.cpp src/cli/cli_app.hpp src/cli/cli_app.cpp \
        tests/unit/update_check_test.cpp
git commit -m "feat(update-check): atomic cache I/O + notice formatting (shared kHeaderRule)"
```

---

## Task 5: libcurl fetch and the orchestrator

**Goal:** Implement `fetch_latest_release` (HTTPS GET to the GitHub Releases API with the gzip fix, redirects, timeouts, versionless UA, conditional ETag) and the orchestrator. The orchestrator is split into TWO functions so it is unit-testable in CI: `run_update_check` (the cache + fetch + notice I/O, with NO TTY/CI/opt-out gate) and `maybe_check_for_update` (the gated entry point `run_cli` calls, which evaluates `should_show` and then delegates to `run_update_check`).

**Why the split (correctness fix over the original single-function design):** the original AC-10 test drove `maybe_check_for_update` directly with a stubbed fetch and a `HOME`-isolated temp dir. That cannot pass in CI: `maybe_check_for_update`'s gate calls `should_show(..., stderr_is_tty())`, and in CI stderr is piped (not a TTY) AND `CI` is set, so the gate returns false and the function bails BEFORE any fetch or cache write. The cache-write assertion would then fail on every CI run (it only passes interactively, where stderr is a TTY and `CI` is unset). Extracting the gate-free core lets AC-10 assert the fetch/cache/notice logic deterministically with a temp cache path and no environment dependency. The gate itself stays covered by AC-5 (`should_show` truth table, which makes each input independently decisive).

**Files:**
- Modify: `src/core/update_check.hpp` (declare `fetch_latest_release` + `run_update_check` + the gated `maybe_check_for_update`), `src/core/update_check.cpp` (implement fetch + core + gate), `tests/unit/update_check_test.cpp` (AC-10 drives `run_update_check`)

**Acceptance Criteria (spec AC-10, AC-14):**
- [ ] **AC-10** With a stubbed `fetch` returning a newer tag, `run_update_check(temp_cache, 0, false, stub)` writes `latest_version` to the cache (read back as the v-stripped tag), does not throw, and leaves stdout empty. With a stub returning an older tag, the cache is updated but the no-newer-notice path is taken (the exact notice string is already pinned by the AC-8 `format_notice` test; here we assert the core consumes the fetch result). The test needs no `HOME`/`CI`/TTY manipulation and runs identically in CI and locally.
- [ ] **AC-14 (zero payload, by inspection)** `fetch_latest_release` sets exactly: `CURLOPT_URL` (fixed HTTPS URL), `CURLOPT_USERAGENT "wmr"`, the fixed `Accept` header, the optional `If-None-Match`, `CURLOPT_ACCEPT_ENCODING ""`, `CURLOPT_FOLLOWLOCATION 1`, `CURLOPT_CONNECTTIMEOUT_MS 1500`, `CURLOPT_TIMEOUT_MS 3000`. No `CURLOPT_POSTFIELDS`, no query string, no header carrying version/OS/arch/id.
- [ ] `maybe_check_for_update` (the gate) flushes stderr before returning; on an ineligible run it touches no network and writes nothing.
- [ ] The cached `latest_version` is stored WITHOUT the leading `v` (the tag is run through `parse_tag` before storage), matching the spec's cache schema (`"latest_version": "1.16.4"`) and the notice format (`1.16.3 -> 1.16.4`).

**Verify:** `./build/tests/wmr_tests "[update-check]"` passes the AC-10 section in CI (no TTY/`CI` dependency). (The real network path is the separate smoke test in Task 7.)

**Steps:**

- [ ] **Step 1: Declare `fetch_latest_release`, `run_update_check`, and the gated entry point in `src/core/update_check.hpp`**

```cpp
// HTTPS GET of /releases/latest. Returns the response body + the ETag header
// (empty if none). On any failure, ok=false with an error string. Zero payload:
// versionless UA, no query string, no body.
FetchResult fetch_latest_release(const std::string& etag);

// Gate-free cache + fetch + notice core. Reads cache_path, shows from cache if a
// known newer version exists, fetches (via fetch) only if the cache is stale,
// writes the cache, and shows if newly-found-newer. Takes an explicit cache path
// and interval so it is deterministic and unit-testable with no TTY/CI/env
// dependency. Never throws; writes only to stderr.
void run_update_check(const fs::path& cache_path, long long interval_s,
                      bool color, FetchFn fetch);

// The gated entry point run_cli calls. Evaluates the opt-out / CI / TTY gate
// (should_show); if eligible, resolves the cache path + interval + color and
// delegates to run_update_check. Flushes stderr before returning. Never throws.
void maybe_check_for_update(bool no_update_check,
                            FetchFn fetch = fetch_latest_release);
```

(`fetch_latest_release` is declared before `maybe_check_for_update`'s default argument references it, so the default resolves.)

- [ ] **Step 2: Write the failing AC-10 test**

Append to `tests/unit/update_check_test.cpp` (inside the single `#ifdef WMR_UPDATE_CHECK` block, before the trailing `#endif`). This drives `run_update_check` directly with a temp cache path and a stubbed fetch, so it needs NO `HOME`/`CI`/TTY manipulation and passes identically in CI and locally. (`APP_VERSION` is `"0.0.0"` in the test target, so a `v9.9.9` tag is newer and the cache-write path is exercised. The exact notice string is pinned by AC-8; this case asserts the core consumes the fetch result and stores the v-stripped tag.)

The `<unistd.h>` include (for `getpid`) was already added in Task 4; no new top-of-file includes are needed here (the temp path uses `std::filesystem::temp_directory_path()` from `<filesystem>`, already included).

```cpp
TEST_CASE("run_update_check: newer fetch updates cache (v stripped); no throw", "[update-check]") {
    fs::path cache = tfs::temp_directory_path() / ("wmr_uc_core_" + std::to_string(getpid()) + ".json");
    tfs::remove(cache);

    FetchResult newer; newer.ok = true; newer.http_code = 200;
    newer.body = R"({"tag_name":"v9.9.9","assets":[]})";
    CHECK_NOTHROW(run_update_check(cache, /*interval_s=*/0, /*color=*/false,
                                   [&](const std::string&){ return newer; }));
    REQUIRE(read_cache(cache).latest_version == "9.9.9");  // parse_tag stripped the leading v

    // last_check_epoch was written (the fetch ran, interval=0 => always).
    REQUIRE(read_cache(cache).last_check_epoch > 0);
    tfs::remove(cache);

    // Older tag: still consumed into the cache (no throw), but the notice path is
    // the not-newer branch (the exact notice bytes are pinned by the AC-8 test).
    FetchResult older; older.ok = true; older.http_code = 200;
    older.body = R"({"tag_name":"v0.0.1"})";
    CHECK_NOTHROW(run_update_check(cache, 0, false,
                                   [&](const std::string&){ return older; }));
    REQUIRE(read_cache(cache).latest_version == "0.0.1");
    tfs::remove(cache);

    // 304 / fetch failure: previous latest_version is preserved, epoch advances.
    {
        std::ofstream f(cache); f << R"({"last_check_epoch":1,"latest_version":"1.0.0","etag":""})";
    }
    FetchResult notmod; notmod.ok = true; notmod.http_code = 304;  // 304 -> keep previous
    CHECK_NOTHROW(run_update_check(cache, 0, false,
                                   [&](const std::string&){ return notmod; }));
    REQUIRE(read_cache(cache).latest_version == "1.0.0");  // unchanged
    REQUIRE(read_cache(cache).last_check_epoch > 1);        // epoch advanced
    tfs::remove(cache);
}
```

(Requires `#include <fstream>` for the manual cache seed in the 304 case; add it to the test file's top includes if not already present. `opt-out skips fetch` is NOT asserted here, because it is a property of the GATE (`maybe_check_for_update`), and the gate is fully covered by AC-5's `should_show` truth table: `no_update_flag=true -> false`, independently of every other input. Re-asserting it through the gate in CI would require faking a TTY and unsetting `CI`, which is exactly the fragility the core extraction removes.)

- [ ] **Step 3: Run tests to confirm failure**

```bash
cmake --build build && ./build/tests/wmr_tests "[update-check]"
```
Expected: failures (orchestrator is still the stub).

- [ ] **Step 4: Implement `fetch_latest_release` in `src/core/update_check.cpp`**

Add at top of file:

```cpp
#include <curl/curl.h>
#include <chrono>
```

Mirror the codebase's curl idiom (`src/core/model_downloader.cpp`):

```cpp
namespace {
struct CurlBuf {
    std::string body;
    std::string etag;
};
size_t uc_write_cb(char* ptr, size_t sz, size_t n, void* ud) {
    auto* b = static_cast<CurlBuf*>(ud);
    b->body.append(ptr, sz * n);
    return sz * n;
}
size_t uc_header_cb(char* buf, size_t sz, size_t n, void* ud) {
    auto* b = static_cast<CurlBuf*>(ud);
    size_t len = sz * n;
    std::string h(buf, len);
    // Match "ETag:" (case-insensitive on the key prefix).
    if (len > 5 && (h[0] == 'E' || h[0] == 'e') &&
        (h[1] == 'T' || h[1] == 't') && (h[2] == 'a' || h[2] == 'A') &&
        (h[3] == 'g' || h[3] == 'G') && h[4] == ':') {
        std::string v = h.substr(5);
        // trim whitespace + trailing CR/LF
        while (!v.empty() && (v.front() == ' ' || v.front() == '\t')) v.erase(0, 1);
        while (!v.empty() && (v.back() == '\r' || v.back() == '\n' || v.back() == ' ')) v.pop_back();
        b->etag = v;
    }
    return len;
}
}  // namespace

FetchResult fetch_latest_release(const std::string& etag) {
    FetchResult r;
    CURL* curl = curl_easy_init();
    if (!curl) { r.error = "curl_easy_init failed"; return r; }
    CurlBuf buf;
    constexpr const char* kUrl =
        "https://api.github.com/repos/froggeric/gemini-watermark-and-synthid-remover/releases/latest";
    struct curl_slist* hdrs = nullptr;
    hdrs = curl_slist_append(hdrs, "Accept: application/vnd.github+json");
    std::string inm;
    if (!etag.empty()) { inm = "If-None-Match: " + etag; hdrs = curl_slist_append(hdrs, inm.c_str()); }

    curl_easy_setopt(curl, CURLOPT_URL, kUrl);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdrs);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "wmr");          // versionless
    curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "");        // auto-decompress gzip (REQUIRED)
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, 1500L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 3000L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, uc_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, uc_header_cb);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, &buf);

    CURLcode rc = curl_easy_perform(curl);
    long code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
    curl_slist_free_all(hdrs);
    curl_easy_cleanup(curl);

    if (rc != CURLE_OK) { r.error = std::string("curl: ") + curl_easy_strerror(rc); return r; }
    r.ok = true;
    r.http_code = static_cast<int>(code);
    r.body = std::move(buf.body);
    r.etag = std::move(buf.etag);
    return r;
}
```

- [ ] **Step 5: Implement `run_update_check` (core) and `maybe_check_for_update` (gate)**

```cpp
namespace {
long long now_epoch() {
    using namespace std::chrono;
    return duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
}
long long parse_interval_env() {
    const char* v = std::getenv("WMR_UPDATE_CHECK_INTERVAL");
    if (!v || !*v) return 86400;
    char* end = nullptr;
    errno = 0;
    // strtoll (not strtoul) so the return type matches `long long` with no
    // implementation-defined cast: a value > LLONG_MAX trips ERANGE -> default
    // (instead of wrapping to a negative and making should_fetch always true),
    // and a leading '-' parses negative -> rejected below (spec: huge = never
    // fetch; negative is not a valid throttle).
    long long n = std::strtoll(v, &end, 10);
    if (end == v || errno == ERANGE || n < 0) return 86400;  // empty/garbage/overflow/negative
    return n;
}
#ifndef APP_VERSION
#define APP_VERSION "0.0.0"
#endif
}  // namespace

// Gate-free core. Deterministic + unit-testable (takes the cache path and interval
// explicitly, reads no TTY/CI env here). Writes only to stderr; never throws.
void run_update_check(const fs::path& cache_path, long long interval_s,
                      bool color, FetchFn fetch) {
    CacheData cd = read_cache(cache_path);
    bool printed = false;

    // 1) Show from cache if a known newer version exists (no network).
    if (!cd.latest_version.empty() &&
        compare_versions(APP_VERSION, cd.latest_version) < 0) {
        std::fputs(format_notice(APP_VERSION, cd.latest_version, color).c_str(), stderr);
        printed = true;
    }

    // 2) Fetch only if the cache is stale.
    long long now = now_epoch();
    long long age = now - cd.last_check_epoch;
    if (should_fetch(age, interval_s) && fetch) {
        FetchResult fr = fetch(cd.etag);
        if (fr.ok && fr.http_code == 200) {
            // parse_tag strips the leading v/V so the stored value AND the notice
            // show "1.16.4" (matching the spec cache schema + notice format), not
            // "v1.16.4". compare_versions is v-tolerant either way, but display is not.
            if (auto tag = parse_release_json(fr.body)) cd.latest_version = parse_tag(*tag);
            // else: malformed body -> keep previous latest_version
        }  // 304 or any failure: keep previous latest_version
        if (!fr.etag.empty()) cd.etag = fr.etag;
        cd.last_check_epoch = now;
        write_cache(cache_path, cd);

        if (!printed && !cd.latest_version.empty() &&
            compare_versions(APP_VERSION, cd.latest_version) < 0) {
            std::fputs(format_notice(APP_VERSION, cd.latest_version, color).c_str(), stderr);
        }
    }
}

void maybe_check_for_update(bool no_update_check, FetchFn fetch) {
    try {
        if (!should_show(no_update_check,
                         env_nonempty("WMR_NO_UPDATE_CHECK"),
                         env_nonempty("CI"),
                         env_equals("DO_NOT_TRACK", "1"),
                         stderr_is_tty())) {
            return;  // not eligible: no network, no notice
        }
        fs::path cache_path = user_cache_dir() / "update-check.json";
        run_update_check(cache_path, parse_interval_env(), color_enabled(), fetch);
    } catch (...) {
        // never propagate past the boundary
    }
    std::fflush(stderr);
}
```

(Delete the Task 2 stub `maybe_check_for_update` and `fetch_latest_release`; keep `now_epoch`, `parse_interval_env`, and the `APP_VERSION` fallback in the anonymous namespace. `parse_interval_env` parses with `strtoll` into a `long long` (matching `should_fetch`'s and the core's `long long` types with no cast), and rejects empty/no-digits (`end == v`), overflow (`errno == ERANGE`), and negative inputs (`n < 0`); any of those falls back to 86400.)

- [ ] **Step 6: Build and run tests**

```bash
cmake --build build && ./build/tests/wmr_tests "[update-check]"
```
Expected: AC-10 passes (no-throw; newer/older/304 cases; v-stripped storage). The opt-out-skip behavior is covered by AC-5's `should_show` truth table plus the gate structure, not by this network-free case.

- [ ] **Step 7: Commit**

```bash
git add src/core/update_check.hpp src/core/update_check.cpp tests/unit/update_check_test.cpp
git commit -m "feat(update-check): curl fetch (gzip-safe) + orchestrator with injectable fetch"
```

---

## Task 6: Wire the check into `run_cli`

**Goal:** Add the `--no-update-check` flag and call the orchestrator at one chokepoint at the end of `run_cli`, reached only after a dispatched subcommand (not on `--version`/`--help`/parse errors/no-args).

**Files:**
- Modify: `src/cli/cli_app.hpp` (`CliOptions::no_update_check`), `src/cli/cli_app.cpp` (`add_common` binds the flag; restructure the bottom of `run_cli` into one chokepoint). No change to `tests/unit/update_check_test.cpp`: AC-11 is structural + behavioral, not a spy unit test (see Step 4).

**Acceptance Criteria (spec AC-11, AC-12):**
- [ ] **AC-11** `wmr --version`, `wmr --help`, `wmr remove --help`, `wmr` (no args), and a bad-flag invocation never reach `maybe_check_for_update` and make zero network calls. Verified structurally (the `maybe_check_for_update` call sits in the single post-dispatch chokepoint, AFTER the no-args early return and the `CLI::ParseError` catch, so those paths cannot reach it) and behaviorally (`wmr --version` and a piped `wmr remove` show no banner). No runtime unit test: `run_cli` is not compiled into `wmr_tests` (`cli_app.cpp` is not in `LIB_SOURCES`), so a spy is not reachable from the test exe.
- [ ] **AC-12** When the regen path ran (so `main` takes `std::_Exit`), a due notice is fully present in captured stderr (the orchestrator flushes stderr before `run_cli` returns).
- [ ] The check runs after a real subcommand whether it succeeded or failed (the captured `rc` is returned untouched).

**Verify:** `cmake --build build`; `./build/wmr --version` (no banner); `./build/wmr remove <fixture>` still works and prints nothing extra when piped (non-TTY suppresses).

**Steps:**

- [ ] **Step 1: Add the flag to `CliOptions` (`src/cli/cli_app.hpp`)**

In `struct CliOptions` (around `:29`), add:

```cpp
    bool no_update_check = false;
```

- [ ] **Step 2: Bind `--no-update-check` in `add_common` (`src/cli/cli_app.cpp`)**

In the `add_common` lambda (around `:436`), add after the `--verbose` line:

```cpp
        cmd->add_flag("--no-update-check", opts.no_update_check,
                      "Skip the startup update check (also WMR_NO_UPDATE_CHECK=1)");
```

- [ ] **Step 3: Restructure the bottom of `run_cli` into one chokepoint**

Replace the dispatch `try { switch (...) { case ...: return ...; } } catch (const std::exception& e) { ... return 1; } return 0;` block (around `cli_app.cpp:684-708`) so each case sets `rc` instead of returning, the catch sets `rc = 1`, and a single chokepoint after it runs the check and returns `rc`:

```cpp
    int rc = 0;
    try {
        switch (opts.mode) {
            case CliMode::Detect:  rc = process_detect(opts); break;
            case CliMode::Video:   rc = process_video(opts);  break;
            case CliMode::AutoRemove:
            case CliMode::VisibleOnly:
            case CliMode::SynthidOnly: {
                if (std::filesystem::is_directory(opts.input_path)) {
                    auto result = batch_process(opts);
                    rc = result.failed > 0 ? 1 : 0;
                } else {
                    rc = process_single_image(opts);
                }
                break;
            }
        }
    } catch (const std::exception& e) {
        spdlog::error("Error: {}", e.what());
        rc = 1;
    }

#ifdef WMR_UPDATE_CHECK
    // Reached ONLY after a dispatched subcommand (the no-args path returned
    // before parse; --version/--help/parse errors returned from the
    // CLI::ParseError catch above and never reach here). Never throws, never
    // changes rc, writes only to stderr.
    wmr::maybe_check_for_update(opts.no_update_check);
#endif
    return rc;
}
```

Add `#include "core/update_check.hpp"` near the top of `cli_app.cpp` (ungated include is fine; the header self-guards).

- [ ] **Step 4: Verify AC-11 (parse-error/help/no-args skip the check): structural**

`run_cli` is not compiled into the `wmr_tests` target (`cli_app.cpp` is not in `tests/CMakeLists.txt`'s `LIB_SOURCES`), so AC-11 cannot be a runtime unit test against `run_cli`. It is a structural guarantee, verified two ways:

1. **By inspection of the restructured `run_cli`:** the no-args path returns before `app.parse` (`cli_app.cpp:430`); `--version`/`--help`/subcommand-`--help`/bad-flag throw `CLI::ParseError` caught at `:639` and return from that catch; the `maybe_check_for_update` call sits in the single post-dispatch chokepoint AFTER both, so those paths cannot reach it.
2. **By behavior (Step 5 below):** `wmr --version` and a piped (non-TTY) `wmr remove` produce no update banner.

No runtime unit test is added for AC-11 (a placeholder `REQUIRE(true)` would assert nothing). If `cli_app.cpp` is later compiled into the test target, add a spy asserting `maybe_check_for_update` is not called on `--version`/`--help`.

- [ ] **Step 5: Build and verify behavior**

```bash
cmake --build build
./build/wmr --version
./build/wmr remove test-images/896x1200-test4-gemini36.png -o /tmp/wmr_uc_out.png 2>/dev/null
```
Expected: `--version` prints version, no update banner; the piped `remove` (non-TTY stderr) prints no banner.

- [ ] **Step 6: Commit**

```bash
git add src/cli/cli_app.hpp src/cli/cli_app.cpp
git commit -m "feat(update-check): wire --no-update-check + single end-of-run_cli chokepoint"
```

---

## Task 7: Opt-in network smoke test

**Goal:** Add a real-network smoke test that proves the live fetch returns a parseable tag and a gzip-decoded body. SKIPs unless `WMR_NETWORK_TESTS=1`.

**Files:**
- Create: `tests/unit/update_check_smoke_test.cpp`
- Modify: `tests/CMakeLists.txt` (add the smoke source to the `WMR_UPDATE_CHECK` mirror block)

**Acceptance Criteria:**
- [ ] The smoke test `[update-check][smoke]` SKIPs when `WMR_NETWORK_TESTS` is unset; runs when set.
- [ ] When run: `fetch_latest_release("")` returns `ok=true`, `http_code==200`, a body containing `"tag_name"`, and `parse_release_json(body)` returns a non-empty tag.
- [ ] The tag, parsed via `parse_version`, is a valid `{maj,min,patch}` (sanity: it is a release tag).

**Verify:** `WMR_NETWORK_TESTS=1 ./build/tests/wmr_tests "[update-check][smoke]"` passes; without the env var it SKIPs.

**Steps:**

- [ ] **Step 1: Create `tests/unit/update_check_smoke_test.cpp`**

```cpp
#include <catch2/catch_test_macros.hpp>

#ifdef WMR_UPDATE_CHECK
#include "core/update_check.hpp"
#include <cstdlib>
using namespace wmr;

TEST_CASE("live GitHub releases/latest fetch decodes", "[update-check][smoke]") {
    if (!std::getenv("WMR_NETWORK_TESTS")) { WARN("set WMR_NETWORK_TESTS=1 to run"); return; }

    FetchResult fr = fetch_latest_release("");
    REQUIRE(fr.ok);
    REQUIRE(fr.http_code == 200);
    REQUIRE(fr.body.find("tag_name") != std::string::npos);  // decoded, not raw gzip
    auto tag = parse_release_json(fr.body);
    REQUIRE(tag.has_value());
    REQUIRE_FALSE(tag->empty());
    REQUIRE(parse_version(*tag).has_value());
}
#endif
```

- [ ] **Step 2: Add it to the tests mirror block**

In `tests/CMakeLists.txt`, add the smoke source to the `WMR_UPDATE_CHECK` block's `target_sources`:

```cmake
        ${CMAKE_SOURCE_DIR}/tests/unit/update_check_smoke_test.cpp
```

- [ ] **Step 3: Run (SKIP by default; live when env set)**

```bash
./build/tests/wmr_tests "[update-check][smoke]"             # SKIP
WMR_NETWORK_TESTS=1 ./build/tests/wmr_tests "[update-check][smoke]"  # live, passes
```

- [ ] **Step 4: Commit**

```bash
git add tests/unit/update_check_smoke_test.cpp tests/CMakeLists.txt
git commit -m "test(update-check): opt-in network smoke test (gzip-decoded fetch)"
```

---

## Task 8: README docs + final build verification

**Goal:** Document the feature for users (the check, the opt-out, the zero-payload stance, the cache location) and verify the full build matrix (ON, OFF, ON-without-regen) and the zero-payload claim.

**Files:**
- Modify: `README.md`

**Acceptance Criteria (spec AC-13, AC-14):**
- [ ] **AC-13** `WMR_UPDATE_CHECK=OFF` builds with zero update-check symbols and no curl dependency introduced by this feature. `WMR_UPDATE_CHECK=ON` with `WMR_BUILD_REGEN=OFF` configures and links (curl resolved via the base vcpkg dep or Homebrew).
- [ ] **AC-14** README documents: the check exists and is default-on; the opt-out (`--no-update-check`, `WMR_NO_UPDATE_CHECK`, `CI`, `DO_NOT_TRACK=1`); that it is zero-payload (versionless UA, no version/OS/id sent); the cache file location (`~/.cache/wmr/update-check.json`); the 24h throttle and `WMR_UPDATE_CHECK_INTERVAL`.
- [ ] `ctest --test-dir build --output-on-failure` is green (all `[update-check]` tests pass; smoke SKIPs without the env var).

**Verify:**
```bash
cmake -B build-off -S . -GNinja -DWMR_UPDATE_CHECK=OFF && cmake --build build-off   # clean
cmake -B build-noregen -S . -GNinja -DWMR_UPDATE_CHECK=ON -DWMR_BUILD_REGEN=OFF && cmake --build build-noregen
scripts/build.sh && ctest --test-dir build --output-on-failure
```

**Steps:**

- [ ] **Step 1: Document in `README.md`**

Add a short section (near the install/usage area). Example text:

````markdown
## Update check

wmr checks once per day whether a newer release is available and prints a short
notice to stderr after your command finishes. It is on by default and is
**notify-only**: it never downloads or replaces the binary.

- **Zero payload.** The check is a single HTTPS GET of the latest release tag
  with a versionless `User-Agent: wmr`. No version, OS, arch, or id is sent.
- **Opt out:** `--no-update-check`, `WMR_NO_UPDATE_CHECK=1`, or it auto-disables
  when stderr is not a terminal, when `CI` is set, or when `DO_NOT_TRACK=1`.
- **Throttle:** at most once per 24 h; override with
  `WMR_UPDATE_CHECK_INTERVAL=<seconds>` (use `0` for tests).
- **Cache:** `~/.cache/wmr/update-check.json`.

Builds without it: `cmake -DWMR_UPDATE_CHECK=OFF ...`.
````

- [ ] **Step 2: Run the three build verifications** (the commands in **Verify** above).

- [ ] **Step 3: Run the full test suite**

```bash
ctest --test-dir build --output-on-failure
```
Expected: all green (smoke SKIPs without `WMR_NETWORK_TESTS`).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(update-check): document the default-on notify-only check + opt-out"
```

---

## Dependency order

1 → 2 (independent of 1) → 3 → 4 → 5 (needs 1, 4) → 6 (needs 4, 5) → 7 (needs 5) → 8 (needs 6, 7). Tasks 1 and 2 both edit `tests/CMakeLists.txt`, so they cannot be parallel writers, but either order links: Task 2's STUB `update_check.cpp` is a no-op that does NOT reference `user_cache_dir()`, so it links without `paths.cpp`. The hard link dependency is Task 5 → Task 1: Task 5's real `maybe_check_for_update` calls `wmr::user_cache_dir()` (in `paths.cpp`), so the test target needs `paths.cpp` in `LIB_SOURCES` (Task 1's `Step 4b`) before Task 5's tests run.

Tasks 3, 4, 5, 6 each modify `src/core/update_check.{hpp,cpp}` and/or `src/cli/cli_app.{hpp,cpp}`, so they must run **sequentially** (never parallel writers on the same file).
