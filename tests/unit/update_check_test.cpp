#include <catch2/catch_test_macros.hpp>

#ifdef WMR_UPDATE_CHECK
#include "core/update_check.hpp"
using namespace wmr;

TEST_CASE("parse_version", "[update-check]") {
    REQUIRE(parse_version("1.16.4").value() == std::array<unsigned,3>{1,16,4});
    REQUIRE(parse_version("v1.16.4").value() == std::array<unsigned,3>{1,16,4});
    REQUIRE(parse_version("V1.16.4").value() == std::array<unsigned,3>{1,16,4});
    REQUIRE(parse_version("2").value() == std::array<unsigned,3>{2,0,0});
    REQUIRE(parse_version("1.2.3.4").value() == std::array<unsigned,3>{1,2,3});
    REQUIRE(parse_version("1.2-rc1").value() == std::array<unsigned,3>{1,2,0});
    REQUIRE_FALSE(parse_version("").has_value());
    REQUIRE_FALSE(parse_version("v").has_value());
    REQUIRE_FALSE(parse_version("1.a.2").has_value());
    REQUIRE_FALSE(parse_version("1..2").has_value());
    REQUIRE_FALSE(parse_version("99999999999").has_value());  // overflow
}

TEST_CASE("compare_versions", "[update-check]") {
    REQUIRE(compare_versions("1.16.3","1.16.4") < 0);
    REQUIRE(compare_versions("1.16.4","1.16.4") == 0);
    REQUIRE(compare_versions("1.17.0","1.16.9") > 0);
    REQUIRE(compare_versions("2.0.0","1.99.99") > 0);
    REQUIRE(compare_versions("1.16","1.16.0") == 0);
    REQUIRE(compare_versions("notaver","1.16.4") > 0);  // malformed current never newer
}

TEST_CASE("parse_tag", "[update-check]") {
    REQUIRE(parse_tag("v1.16.4") == "1.16.4");
    REQUIRE(parse_tag("V1.16.4") == "1.16.4");
    REQUIRE(parse_tag("1.16.4") == "1.16.4");
}

TEST_CASE("parse_release_json", "[update-check]") {
    const char* ok = R"({"tag_name":"v1.16.4","assets":[]})";
    REQUIRE(parse_release_json(ok).value() == "v1.16.4");
    REQUIRE_FALSE(parse_release_json(R"({"assets":[]})").has_value());      // missing
    REQUIRE_FALSE(parse_release_json(R"({"tag_name":123})").has_value());   // non-string
    REQUIRE_FALSE(parse_release_json("not json{").has_value());             // unparseable
}

TEST_CASE("should_show truth table", "[update-check]") {
    REQUIRE(should_show(false,false,false,false,true) == true);
    REQUIRE(should_show(true, false,false,false,true) == false);
    REQUIRE(should_show(false,true, false,false,true) == false);
    REQUIRE(should_show(false,false,true, false,true) == false);
    REQUIRE(should_show(false,false,false,true, true) == false);
    REQUIRE(should_show(false,false,false,false,false) == false);  // non-TTY
}

TEST_CASE("should_fetch", "[update-check]") {
    REQUIRE(should_fetch(86399, 86400) == false);
    REQUIRE(should_fetch(86400, 86400) == true);
    REQUIRE(should_fetch(0, 0) == true);
    REQUIRE(should_fetch(-100, 86400) == false);
}

TEST_CASE("color_enabled_for (pure policy)", "[update-check]") {
    REQUIRE(color_enabled_for(true,  nullptr, nullptr) == true);   // default: TTY, no NO_COLOR, no TERM
    REQUIRE(color_enabled_for(true,  "",      nullptr) == true);   // NO_COLOR empty == unset == ON
    REQUIRE(color_enabled_for(true,  "1",     nullptr) == false);  // NO_COLOR non-empty == OFF
    REQUIRE(color_enabled_for(true,  "any",   nullptr) == false);  // any non-empty value disables
    REQUIRE(color_enabled_for(true,  nullptr, "xterm") == true);   // TERM != dumb == ON
    REQUIRE(color_enabled_for(true,  nullptr, "dumb") == false);   // TERM == dumb == OFF
    REQUIRE(color_enabled_for(false, nullptr, nullptr) == false);  // non-TTY always OFF (independent)
}
#endif
