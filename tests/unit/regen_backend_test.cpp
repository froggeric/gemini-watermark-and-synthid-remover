#include <string>
#include <catch2/catch_test_macros.hpp>
#include "core/regen_backend.hpp"

using namespace wmr;

TEST_CASE("regen backend resolver", "[regen][backend]") {
    SECTION("string mapping uses upstream device names") {
        REQUIRE(regen_backend_string(RegenBackend::Auto)   == "");
        REQUIRE(regen_backend_string(RegenBackend::Cpu)    == "cpu");
        REQUIRE(regen_backend_string(RegenBackend::Metal)  == "metal");
        REQUIRE(regen_backend_string(RegenBackend::Cuda)   == "cuda0");
        REQUIRE(regen_backend_string(RegenBackend::Vulkan) == "vulkan0");
    }
    SECTION("parse") {
        REQUIRE(regen_backend_from_string("")        == RegenBackend::Auto);
        REQUIRE(regen_backend_from_string("auto")    == RegenBackend::Auto);
        REQUIRE(regen_backend_from_string("garbage") == RegenBackend::Auto);
        REQUIRE(regen_backend_from_string("cpu")     == RegenBackend::Cpu);
        REQUIRE(regen_backend_from_string("metal")   == RegenBackend::Metal);
        REQUIRE(regen_backend_from_string("cuda0")   == RegenBackend::Cuda);
        REQUIRE(regen_backend_from_string("vulkan0") == RegenBackend::Vulkan);
        REQUIRE(regen_backend_from_string("cuda")    == RegenBackend::Cuda);    // CLI --regen-backend string
        REQUIRE(regen_backend_from_string("vulkan")  == RegenBackend::Vulkan);  // CLI --regen-backend string
    }
    SECTION("default is Auto (empty = sdcpp auto GPU->CPU)") {
        REQUIRE(default_regen_backend() == RegenBackend::Auto);
        REQUIRE(std::string(regen_backend_string(default_regen_backend())) == "");
    }
}
