#include <catch2/catch_test_macros.hpp>

#ifdef WMR_UPDATE_CHECK
#include "core/update_check.hpp"
using namespace wmr;
#endif

TEST_CASE("update_check harness compiles", "[update-check]") {
    REQUIRE(true);
}
