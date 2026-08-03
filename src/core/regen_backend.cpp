#include "core/regen_backend.hpp"

namespace wmr {

RegenBackend default_regen_backend() { return RegenBackend::Auto; }

RegenBackend regen_backend_from_string(std::string_view s) {
    if (s.empty())      return RegenBackend::Auto;
    if (s == "auto")    return RegenBackend::Auto;
    if (s == "cpu")     return RegenBackend::Cpu;
    if (s == "metal")   return RegenBackend::Metal;
    if (s == "cuda0" || s == "cuda")   return RegenBackend::Cuda;
    // Accept both the CLI-facing "vulkan" and the sdcpp device string "vulkan0".
    // The CLI --regen-backend IsMember set is {auto,cpu,metal,vulkan}; without
    // accepting "vulkan" here it fell through to Auto -> Cpu on Apple Silicon,
    // silently never using the Vulkan backend.
    if (s == "vulkan0" || s == "vulkan") return RegenBackend::Vulkan;
    if (s == "coreml") return RegenBackend::CoreML;
    return RegenBackend::Auto;  // unknown -> auto default
}

const char* regen_backend_string(RegenBackend b) {
    switch (b) {
        case RegenBackend::Auto:   return "";        // empty = sdcpp auto (GPU -> CPU)
        case RegenBackend::Cpu:    return "cpu";
        case RegenBackend::Metal:  return "metal";
        case RegenBackend::Cuda:   return "cuda0";
        case RegenBackend::Vulkan: return "vulkan0";
        case RegenBackend::CoreML: return "coreml";
    }
    return "";
}

} // namespace wmr
