#pragma once
// The GUI launch entry point (Task 6): one call for both ways in (`wmr gui`
// and bare `wmr` when neither WMR_NO_GUI nor CI is set). Owns the whole
// server lifetime: run dir, dead-run cleanup, JobManager, GuiServer, signal
// handlers, and the graceful-shutdown tail. Binary-only wiring (cli_app.cpp
// calls this); not compiled into the test exe.

namespace wmr::gui {

// Creates the run dir (0700, pid sidecar), cleans dead sibling runs, starts
// the JobManager + GuiServer, installs SIGINT/SIGTERM handlers, blocks until
// stop, then cancels + joins the worker (<= 5 s), removes the run dir, and
// returns 0 (1 on bind failure or an unusable cache root).
int run_gui(int port, bool no_browser);

}  // namespace wmr::gui
