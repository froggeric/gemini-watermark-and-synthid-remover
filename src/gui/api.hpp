#pragma once
// The GUI REST layer (Task 5): every token-prefixed route (cpp-httplib
// patterns are full-path; the hex token is regex-safe), the one JSON error
// envelope, and the download-path helpers. Route registration matches
// GuiServer::RouteRegistrar's shape: launch captures the JobManager/features/
// UI and calls this with (svr, token).
#include <string>

#include "gui/vendored.hpp"  // httplib (single include point for the vendored headers)

#include "gui/embedded_ui.hpp"
#include "gui/jobs.hpp"
#include "gui/options.hpp"

namespace wmr::gui {

// Registers: GET / (+/app.js, /style.css), GET /api/version, GET|POST
// /api/jobs, GET /api/jobs/{id}, POST /api/jobs/{id}/cancel,
// GET /api/jobs/{id}/files/{i}/image?kind=cleaned|original - all under
// "/" + token. `jobs` is borrowed and must outlive the server; `features`
// and `ui` are copied into the route closures (callers may pass temporaries).
void register_routes(httplib::Server& svr, const std::string& token,
                     JobManager& jobs, const ApiFeatures& features,
                     const EmbeddedUi& ui);

// --- Pure download-path helpers (unit-tested in gui_options_test) ---

// Sanitized download/display name from an untrusted client filename: basename
// only; control chars, quotes, and \/:?*"<>| stripped; Windows reserved stems
// suffixed "_"; capped at 120 bytes on a UTF-8 boundary; falls back to
// "image<N>" when nothing survives.
std::string sanitize_filename(const std::string& name, int fallback_index);

// RFC 5987 filename* encoding: keep [A-Za-z0-9._-], %XX-encode every other
// byte (the input is UTF-8 already).
std::string pct_encode(const std::string& s);

// Content-Type for a sniffed extension ("png"|"jpg"|"webp"; anything else is
// a generic binary stream). Never derived from client-supplied types.
std::string mime_for(const std::string& ext);

}  // namespace wmr::gui
