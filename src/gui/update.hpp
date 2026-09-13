#pragma once
// Server-side update check for the GUI page: the CLI's notify-only check
// (same 24 h cache at ~/.cache/wmr/update-check.json, same env opt-outs:
// WMR_NO_UPDATE_CHECK, CI, DO_NOT_TRACK), run ONCE in the background at
// server start and surfaced through /api/version so the page can show a
// small "newer version available" line. Zero payload, exactly like the CLI.
// Compiled out entirely on WMR_UPDATE_CHECK=OFF builds (stubbed here).
#include <string>

namespace wmr::gui {

struct UpdateInfo {
    bool enabled = false;   // compiled in, not opted out, check started
    bool known = false;     // a result is available (fresh cache or fetch done)
    bool newer = false;
    std::string current, latest;
    // The release page (constant; the API body's html_url is the same target).
    static constexpr const char* url =
        "https://github.com/froggeric/gemini-watermark-and-synthid-remover/releases/latest";
};

// Kick off the once-per-run background check; returns immediately (a stale
// cache is resolved synchronously, so the common case touches no network).
// No-op when the feature is compiled out or opted out via the environment.
void start_update_check_async();

// Snapshot for /api/version.
UpdateInfo update_info();

// Join the background check (graceful-shutdown tail; the _Exit paths die
// with the process instead). No-op when nothing is running.
void stop_update_check();

}  // namespace wmr::gui
