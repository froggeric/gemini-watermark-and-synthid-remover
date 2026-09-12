#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include "gui/image_sniff.hpp"

using namespace wmr::gui;

TEST_CASE("sniff classifies formats by magic", "[gui]") {
    const unsigned char png[] = {0x89,'P','N','G','\r','\n',0x1a,'\n',
        0,0,0,13,'I','H','D','R', 0,0,0,100, 0,0,0,50};
    auto r = sniff_image(png, sizeof png);
    REQUIRE(r.format == SniffFormat::Png);
    REQUIRE(r.ext == "png");
    REQUIRE(r.width == 100); REQUIRE(r.height == 50);
    REQUIRE_FALSE(r.too_large);

    const unsigned char jpeg[] = {(unsigned char)0xFF,(unsigned char)0xD8,(unsigned char)0xFF,
        (unsigned char)0xE0, 0,16,'J','F','I','F',0,1,1,0,0,1,0,1,0,0,
        (unsigned char)0xFF,(unsigned char)0xC0,0,17,8, 0,200, 0,100, 3};
    auto j = sniff_image(jpeg, sizeof jpeg);
    REQUIRE(j.format == SniffFormat::Jpeg);
    REQUIRE(j.width == 100); REQUIRE(j.height == 200);   // SOF0 stores height then width (fixture H=0x00C8=200, W=0x0064=100)

    const unsigned char ft[] = {0,0,0,24,'f','t','y','p','h','e','i','c'};
    REQUIRE(sniff_image(ft, sizeof ft).format == SniffFormat::Unsupported);
    REQUIRE(sniff_image(nullptr, 0).empty);

    // Truncated magic prefix: attacker-controlled multipart bytes must never
    // read out of bounds; classify as Unsupported, do not crash.
    const unsigned char trunc[] = {0x89,'P','N','G','\r','\n',0x1a,'\n',0,0};
    REQUIRE(sniff_image(trunc, sizeof trunc).format == SniffFormat::Unsupported);
}

TEST_CASE("dimension gate rejects decompression bombs", "[gui]") {
    // A tiny PNG header declaring 65535x65535.
    unsigned char png[] = {0x89,'P','N','G','\r','\n',0x1a,'\n',
        0,0,0,13,'I','H','D','R', 0,0,0xFF,0xFF, 0,0,0xFF,0xFF};
    auto r = sniff_image(png, sizeof png);
    REQUIRE(r.format == SniffFormat::Png);
    REQUIRE(r.too_large);
}
