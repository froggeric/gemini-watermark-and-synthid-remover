#include <catch2/catch_test_macros.hpp>
#include "gui/security.hpp"
#include "gui/browser.hpp"

using namespace wmr::gui;

TEST_CASE("token is 32 lowercase hex", "[gui]") {
    auto t = generate_token();
    REQUIRE(t.size() == 32);
    for (char c : t) REQUIRE((isdigit(c) || ('a' <= c && c <= 'f')));
    REQUIRE(generate_token() != t);  // per-run randomness
}

TEST_CASE("host allowlist", "[gui]") {
    REQUIRE(host_allowed("127.0.0.1:54321", 54321));
    REQUIRE(host_allowed("localhost:54321", 54321));
    REQUIRE(host_allowed("[::1]:54321", 54321));
    REQUIRE(host_allowed("LOCALHOST.:54321", 54321));    // case + one trailing dot
    REQUIRE_FALSE(host_allowed("127.0.0.1:54322", 54321));  // wrong port
    REQUIRE_FALSE(host_allowed("evil.com:54321", 54321));
    REQUIRE_FALSE(host_allowed("", 54321));
    REQUIRE_FALSE(host_allowed("127.0.0.1", 54321));       // no port
}
