#include "gui/vendored.hpp"

#include <string_view>

namespace wmr::gui {
// Reported by GET /api/version for diagnostics (kept from the Task-2 build check).
// The gui_build prefix keeps the mangled symbol greppable: the OFF-build check is
// `nm | grep -c gui_build` == 0, and a plain httplib_version mangling
// (_ZN3wmr3gui15httplib_versionEv) does not contain the substring.
const char* gui_build_httplib_version() { return CPPHTTPLIB_VERSION; }

// Compile-time pin on WHICH vendored httplib.h we actually included. sdcpp
// vendors its own (0.28.0) under external/stable-diffusion.cpp; if include
// paths ever let that one shadow ours, this fires instead of a silent
// behavior/ABI drift.
static_assert(std::string_view(CPPHTTPLIB_VERSION) == "0.56.0",
              "vendored cpp-httplib version drifted (sdcpp 0.28 shadowing?)");
}
