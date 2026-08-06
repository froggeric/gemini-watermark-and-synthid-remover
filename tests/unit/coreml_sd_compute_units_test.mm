// Copyright 2025 wmr contributors
// Licensed under the Apache License, Version 2.0 (see LICENSE in repo root).
#ifdef WMR_BUILD_AI_COREML_SD

#include <catch2/catch_test_macros.hpp>
#import <CoreML/CoreML.h>   // this test is a .mm TU; CoreML.h is Objective-C and will not
                             // compile as C++ (.cpp) regardless of #import vs #include

#include "core/coreml_sd_compute_units.hpp"

using namespace wmr;

TEST_CASE("coreml compute-unit env parse", "[coreml-sd][compute-units]") {
    REQUIRE(parse_coreml_compute_units("all")     == MLComputeUnitsAll);
    REQUIRE(parse_coreml_compute_units("cpu_gpu") == MLComputeUnitsCPUAndGPU);
    REQUIRE(parse_coreml_compute_units("cpu-gpu") == MLComputeUnitsCPUAndGPU);
    REQUIRE(parse_coreml_compute_units("cpu_ane") == MLComputeUnitsCPUAndNeuralEngine);
    REQUIRE(parse_coreml_compute_units("cpu-ane") == MLComputeUnitsCPUAndNeuralEngine);
    REQUIRE(parse_coreml_compute_units("cpu")     == MLComputeUnitsCPUOnly);
    // Unknown tokens and the empty/unset case default to All.
    REQUIRE(parse_coreml_compute_units("")        == MLComputeUnitsAll);
    REQUIRE(parse_coreml_compute_units("gpu")     == MLComputeUnitsAll);
    REQUIRE(parse_coreml_compute_units("GPU")     == MLComputeUnitsAll);  // case-sensitive
    REQUIRE(parse_coreml_compute_units("ane")     == MLComputeUnitsAll);
    REQUIRE(parse_coreml_compute_units("random")  == MLComputeUnitsAll);
}

#endif // WMR_BUILD_AI_COREML_SD
