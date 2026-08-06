#pragma once
// Parse WMR_COREML_SD_COMPUTE_UNITS into an MLComputeUnits enum.
#ifdef WMR_BUILD_AI_COREML_SD

#import <CoreML/CoreML.h>   // ObjC++ header; both consumers (coreml_sd_pipeline.mm + the
                            // .mm test) compile as ObjC++, so #import is correct here

#include <string_view>

namespace wmr {

// Map the WMR_COREML_SD_COMPUTE_UNITS env value to an MLComputeUnits enum.
//
// Recognized tokens (case-sensitive): "all" -> MLComputeUnitsAll (also the
// default for empty/unset/unknown), "cpu_gpu" or "cpu-gpu" ->
// MLComputeUnitsCPUAndGPU, "cpu_ane" or "cpu-ane" ->
// MLComputeUnitsCPUAndNeuralEngine, "cpu" -> MLComputeUnitsCPUOnly.
//
// Factored out of coreml_sd_pipeline.mm so the mapping is unit-testable without
// constructing a CoreML model. The ORIGINAL-attention UNet routes to the GPU
// under .all (ANE declines the large attention matmuls); cpu_gpu forces GPU if
// .all ever misplaces a future model variant.
MLComputeUnits parse_coreml_compute_units(std::string_view s) noexcept;

}  // namespace wmr

#endif // WMR_BUILD_AI_COREML_SD
