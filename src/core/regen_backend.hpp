#pragma once

#include <string_view>

namespace wmr {

// SDXL regen backend, mapped to stable-diffusion.cpp's sd_ctx_params_t.backend
// string. Per upstream docs/backend.md the accepted device names are cpu / metal /
// cuda0 / vulkan0, and an EMPTY string ("") selects the default = GPU -> integrated
// GPU -> CPU (auto, never fails for lack of a device). Auto is our runtime default;
// the others are for an explicit --regen-backend override (Task 7).
enum class RegenBackend { Auto, Cpu, Cuda, Vulkan, Metal };

// Runtime default: Auto (empty backend string -> sdcpp picks GPU then CPU). Pure.
RegenBackend default_regen_backend();

// "" / "auto" / unknown -> Auto; otherwise the matching backend.
RegenBackend regen_backend_from_string(std::string_view s);

// The sd_ctx_params_t.backend device name (Auto -> "" = sdcpp auto).
const char* regen_backend_string(RegenBackend b);

} // namespace wmr
