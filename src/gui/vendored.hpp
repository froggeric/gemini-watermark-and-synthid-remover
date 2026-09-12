#pragma once
// Single include point for the vendored GUI headers (kissfft precedent:
// third-party warnings are silenced here, not at the CMake level).
#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#include "httplib.h"   // cpp-httplib v0.56.0 (MIT), external/httplib
#include "json.hpp"    // nlohmann/json v3.12.0 (MIT), external/json
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
