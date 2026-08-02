#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <opencv2/core.hpp>
#include "core/regen_tiling.hpp"
using namespace wmr;
using Catch::Approx;

static bool fully_covered(int w, int h, int tile, int ov) {
    auto tiles = build_regen_tiles(w, h, tile, ov);
    cv::Mat cover = cv::Mat::zeros(h, w, CV_32FC1);
    for (const auto& t : tiles) {
        REQUIRE(t.rect.x >= 0); REQUIRE(t.rect.y >= 0);
        REQUIRE(t.rect.x + t.rect.width  <= w);
        REQUIRE(t.rect.y + t.rect.height <= h);
        REQUIRE(t.weight.size() == t.rect.size());
        REQUIRE(t.weight.type() == CV_32FC1);
        double wmn, wmx; cv::minMaxLoc(t.weight, &wmn, &wmx);
        REQUIRE(wmn >= 0.0f); REQUIRE(wmx <= 1.0f + 1e-4f);  // per-tile weight in [0,1]
        cover(t.rect) += t.weight;
    }
    // Every pixel covered by >0 weight (no holes). Sum need not be 1 (caller normalizes).
    double mn, mx; cv::minMaxLoc(cover, &mn, &mx);
    (void)mx;
    return mn > 1e-6;
}

TEST_CASE("regen tiling", "[regen][tiling]") {
    SECTION("full coverage, no OOB, per-tile weights in [0,1]") {
        REQUIRE(fully_covered(4096, 3072, 1024, 128));
        REQUIRE(fully_covered(2048, 2048, 1024, 128));
        REQUIRE(fully_covered(1500, 900, 1024, 128));
        REQUIRE(fully_covered(1024, 1024, 1024, 128));
        REQUIRE(fully_covered(333, 257, 1024, 128));   // both dims < tile
    }
    SECTION("single tile when image fits tile, all-ones weight") {
        auto tiles = build_regen_tiles(800, 600, 1024, 128);
        REQUIRE(tiles.size() == 1);
        REQUIRE(tiles[0].rect == cv::Rect(0, 0, 800, 600));
        double wmn, wmx; cv::minMaxLoc(tiles[0].weight, &wmn, &wmx);
        REQUIRE(wmn == Approx(1.0f).margin(1e-5f));
        REQUIRE(wmx == Approx(1.0f).margin(1e-5f));
    }
    SECTION("4K produces >=4 full 1024x1024 tiles") {
        auto tiles = build_regen_tiles(4096, 4096, 1024, 128);
        REQUIRE(tiles.size() >= 4);
        int full = 0;
        for (const auto& t : tiles)
            if (t.rect.width == 1024 && t.rect.height == 1024) ++full;
        REQUIRE(full >= 4);
    }
}
