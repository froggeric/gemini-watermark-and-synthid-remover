#include "gui/launch.hpp"

// The server lifetime in one place (Task 6): run dir + dead-run cleanup,
// JobManager, GuiServer, SIGINT/SIGTERM, and the graceful-shutdown tail.
// Spec: docs/superpowers/specs/2026-09-12-gui-webui-design.md ("Launch and
// shutdown").

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <string>

#include "gui/api.hpp"
#include "gui/http_server.hpp"
#include "gui/jobs.hpp"
#include "gui/options.hpp"
#include "gui/run_dir.hpp"

namespace wmr::gui {
namespace {

// Shutdown wiring, deliberately this simple: an async-safe flag plus the
// server pointer. g_server is set after the server object exists and BEFORE
// the handlers are installed (the handler dereferences it). v0.56's
// Server::stop() is noexcept (an atomic listen-socket exchange +
// shutdown/close), so calling it from the signal handler is safe; it closes
// the socket and unblocks listen_after_bind.
GuiServer* g_server = nullptr;
std::sig_atomic_t g_shutdown = 0;

extern "C" void handle_shutdown(int) {
    if (g_shutdown) {
        std::_Exit(0);  // second Ctrl-C in the grace window: immediate exit
    }
    g_shutdown = 1;
    if (g_server != nullptr) {
        g_server->stop();
    }
}

}  // namespace

int run_gui(int port, bool no_browser) {
    std::filesystem::path run_dir;
    try {
        // Dead sibling runs first (only dirs whose recorded pid is dead; the
        // shared root is never wiped), then this run's own 0700 dir.
        cleanup_dead_runs();
        run_dir = create_run_dir();
    } catch (const std::exception& e) {
        // create_run_dir fails loudly (path + OS message): serving with a
        // broken cache would fail every upload with no diagnostic.
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }

    // Declared BEFORE the server: the route closures borrow the manager, so
    // it must outlive the server (and its request threads).
    JobManager jobs(run_dir);

    const ApiFeatures features = build_api_features();
    const EmbeddedUi ui;  // Task 7 fills the views; empty views 404 for now

    GuiServer server;
    g_server = &server;
    std::signal(SIGINT, handle_shutdown);
    std::signal(SIGTERM, handle_shutdown);

    GuiServerConfig cfg;
    cfg.port = port;
    cfg.open_browser = !no_browser;
    cfg.banner_tail =
        "Press Ctrl-C to stop (a running job finishes its current file first).\n"
        "wmr --help for CLI usage";

    // Blocks in listen_after_bind until stop(); 1 on bind failure.
    const int rc = server.run(cfg, [features, ui, &jobs, &server](httplib::Server& svr,
                                                                  const std::string& token) {
        // The page's Quit button runs the same graceful path as Ctrl-C: set
        // the flag (so a later real signal is treated as the "second" one and
        // _Exit's) and close the accept socket; the tail below does the rest.
        register_routes(svr, token, jobs, features, ui, [&server] {
            g_shutdown = 1;
            server.stop();
        });
    });

    // Graceful tail. Bounded: on expiry the manager _Exit(0)s itself (static
    // teardown skipped; the leftover run dir dies at the next start's
    // dead-run cleanup), so only the clean-join path reaches the removal. A
    // second Ctrl-C during the window exits in the handler above.
    jobs.cancel_all_and_join(std::chrono::seconds(5));
    remove_run_dir(run_dir);

    // Back to the default disposition for whatever teardown remains.
    std::signal(SIGINT, SIG_DFL);
    std::signal(SIGTERM, SIG_DFL);
    g_server = nullptr;
    return rc;
}

}  // namespace wmr::gui
