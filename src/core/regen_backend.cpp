#include "core/regen_backend.hpp"

namespace wmr {

RegenBackend default_regen_backend() { return RegenBackend::Auto; }

RegenBackend regen_backend_from_string(std::string_view s) {
    if (s.empty())      return RegenBackend::Auto;
    if (s == "auto")    return RegenBackend::Auto;
    if (s == "cpu")     return RegenBackend::Cpu;
    if (s == "metal")   return RegenBackend::Metal;
    if (s == "cuda0")   return RegenBackend::Cuda;
    if (s == "vulkan0") return RegenBackend::Vulkan;
    return RegenBackend::Auto;  // unknown -> auto default
}

const char* regen_backend_string(RegenBackend b) {
    switch (b) {
        case RegenBackend::Auto:   return "";        // empty = sdcpp auto (GPU -> CPU)
        case RegenBackend::Cpu:    return "cpu";
        case RegenBackend::Metal:  return "metal";
        case RegenBackend::Cuda:   return "cuda0";
        case RegenBackend::Vulkan: return "vulkan0";
    }
    return "";
}

} // namespace wmr
