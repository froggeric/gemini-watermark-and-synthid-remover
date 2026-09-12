#include "gui/vendored.hpp"

namespace wmr::gui {
// Reported by GET /api/version for diagnostics (kept from the Task-2 build check).
// The gui_build prefix keeps the mangled symbol greppable: the OFF-build check is
// `nm | grep -c gui_build` == 0, and a plain httplib_version mangling
// (_ZN3wmr3gui15httplib_versionEv) does not contain the substring.
const char* gui_build_httplib_version() { return CPPHTTPLIB_VERSION; }
}
